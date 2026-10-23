"""RiskGuard (PLAN A2 ``risk.py``; DESK-1.2): pre-trade checks for every order in every mode.

CONTRACTS fixes only: the trading states ACTIVE|REDUCING|HALTED, "RiskGuard denies SELL beyond the held base
balance on spot accounts", the limit list (precision/lot rounding, min notional, max order notional, max position,
max orders per minute, daily loss limit) and ``RiskLimits.unlimited()``. The rest (signatures, reason strings,
defaults, transitions) are the minimum choices listed in the NT-223 report. Pure logic, standard library only.

``check`` is called by the ExecutionEngine before ``Broker.submit`` and by the manual-order path; a ``size_frac``
order is checked twice (at submit on the last close, at the fill on the resolved quantity), so a rate-limit slot is
taken only by ``check(..., record=True)`` (the default); pass ``record=False`` for the estimate. Orders that only
reduce a position are exempt from the notional and rate limits (see ``RiskGuard.check``).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Callable, Mapping, Optional

from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.costs import RESEARCH_ZERO, CostProfile
from neural_trade.desk.core.model import Instrument, Liquidity, Order, OrderType, Side
from neural_trade.desk.core.portfolio import DUST, Portfolio

DAY_MS = 86_400_000
WINDOW_MS = 60_000


class TradingState(str, Enum):
    ACTIVE = "ACTIVE"
    REDUCING = "REDUCING"  # only orders that reduce an open position
    HALTED = "HALTED"  # nothing passes


# reason strings (stable: tests, journal and UI key on them)
HALTED = "halted"
REDUCING_ONLY = "reducing_only"
REDUCE_ONLY_VIOLATION = "reduce_only_violation"
INVALID_ORDER = "invalid_order"
WRONG_INSTRUMENT = "wrong_instrument"
QTY_ROUNDS_TO_ZERO = "qty_rounds_to_zero"
PRICE_ROUNDS_TO_ZERO = "price_rounds_to_zero"
MIN_QTY = "min_qty"
MAX_QTY = "max_qty"
MIN_NOTIONAL = "min_notional"
MAX_ORDER_NOTIONAL = "max_order_notional"
MAX_POSITION = "max_position"
MAX_ORDERS_PER_MINUTE = "max_orders_per_minute"
DAILY_LOSS_LIMIT = "daily_loss_limit"
MISSING_MARKS = "missing_marks"
INSUFFICIENT_CASH = "insufficient_cash"
SPOT_SELL_EXCEEDS_BALANCE = "spot_sell_exceeds_balance"
SPOT_SHORT_NO_OWN_LONG = "spot_short_no_own_long"
CLIPPED_TO_POSITION = "clipped_to_position"
REDUCE_EXCEEDS_POSITION = "reduce_exceeds_position"  # the position is already covered by resting reduce orders


@dataclass(frozen=True)
class RiskLimits:
    """``None`` disables a limit. Notionals and the daily loss are in quote currency. Defaults are conservative."""

    max_order_notional: Optional[float] = 1_000.0
    max_position_notional: Optional[float] = 5_000.0  # |position after the order| x price, per position_id
    max_orders_per_minute: Optional[int] = 30  # sliding 60 s window on the Clock
    daily_loss_limit: Optional[float] = 500.0  # equity fall since the UTC day's first mark; reached at >=

    @classmethod
    def unlimited(cls) -> "RiskLimits":
        return cls(None, None, None, None)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""  # the denial reason; for an allowed clipped order CLIPPED_TO_POSITION
    qty: Optional[float] = None  # the quantity to submit (rounded, possibly clipped); None when denied
    price: Optional[float] = None  # the rounded limit price, if the order has one

    def __bool__(self) -> bool:
        return self.allowed


def _deny(reason: str) -> RiskDecision:
    return RiskDecision(False, reason)


def round_down(x: float, inc: float) -> float:
    """Floor ``x`` to a multiple of ``inc`` (0 disables), exact in decimal."""
    if not inc:
        return x
    d = Decimal(str(inc))
    return float((Decimal(str(x)) / d).to_integral_value(ROUND_DOWN) * d)


def round_nearest(x: float, inc: float) -> float:
    """Round ``x`` to the nearest multiple of ``inc``, half-way up (0 disables), exact in decimal."""
    if not inc:
        return x
    d = Decimal(str(inc))
    return float((Decimal(str(x)) / d).to_integral_value(ROUND_HALF_UP) * d)


class RiskGuard:
    def __init__(self, limits: RiskLimits, instrument: Instrument, portfolio: Portfolio, clock: Clock, *,
                 costs: CostProfile = RESEARCH_ZERO,
                 on_state_change: Optional[Callable[[TradingState, TradingState, str], None]] = None) -> None:
        self.limits, self.instrument, self.portfolio, self._clock = limits, instrument, portfolio, clock
        self._on_state_change = on_state_change
        self.costs = costs  # the account's cost profile: its fee enters the spot cash check
        self.state = TradingState.ACTIVE
        self.state_reason = ""
        self._sent: deque[int] = deque()  # ts_ms of recorded orders
        self._day: Optional[int] = None
        self._day_start_equity = 0.0

    # ------------------------------------------------------------------ state
    def _set(self, new: TradingState, reason: str) -> None:
        old = self.state
        if new is old:
            return
        self.state, self.state_reason = new, reason
        if self._on_state_change:
            self._on_state_change(old, new, reason)

    def halt(self, reason: str) -> None:
        """Any state -> HALTED. Idempotent: a second call changes nothing and notifies nobody."""
        self._set(TradingState.HALTED, reason)

    def reduce_only(self, reason: str = "") -> None:
        """ACTIVE -> REDUCING. Ignored while HALTED (a halt is not softened; use ``resume``)."""
        if self.state is TradingState.ACTIVE:
            self._set(TradingState.REDUCING, reason)

    def resume(self, reason: str = "") -> None:
        """REDUCING|HALTED -> ACTIVE (explicit operator action)."""
        self._set(TradingState.ACTIVE, reason)

    # ------------------------------------------------------------------ daily loss
    def update(self, marks: Mapping[str, float], *, day_start_equity: Optional[float] = None) -> None:
        """Roll the UTC day. THE DAY'S BASELINE IS THE FIRST OBSERVATION OF THE DAY (the equity at the first
        ``update`` or ``check`` after UTC midnight), not midnight itself: the engine MUST call ``update(marks)`` on
        every bar, and after a restart pass ``day_start_equity`` (the stored opening equity of the current day) so
        the loss already taken today is not forgotten. ``marks`` must cover every held instrument."""
        now = self._clock.now_ms()
        day = now // DAY_MS
        if day != self._day:
            self._day = day
            self._day_start_equity = (float(day_start_equity) if day_start_equity is not None
                                      else self.portfolio.account_state(marks, now).equity)

    def day_loss(self, marks: Mapping[str, float]) -> float:
        """Equity fall since the day's start (positive = loss)."""
        self.update(marks)
        return self._day_start_equity - self.portfolio.account_state(marks, self._clock.now_ms()).equity

    # ------------------------------------------------------------------ the check
    def check(self, order: Order, *, ref_price: float, marks: Optional[Mapping[str, float]] = None,
              record: bool = True, day_start_equity: Optional[float] = None,
              committed_reduce_qty: float = 0.0) -> RiskDecision:
        """Decide on ``order``. ``ref_price`` is the price a market order would trade at (last close at submit, the
        resolved open at the fill). ``marks`` defaults to ``{instrument.id: ref_price}``; when the portfolio holds
        other instruments the caller must pass marks for all of them (else ``missing_marks``). An allowed decision
        carries the quantity and price to submit.

        Valuation price of an order (notional, max position): MARKET ``ref_price``; LIMIT its rounded price;
        STOP_MARKET / STOP_LIMIT buys ``max(trigger, limit price, ref_price)`` (the worst case), sells the trigger;
        for an opening order also the highest bracket level (a non-offset leg; an offset leg is ``entry + offset``).
        Bracket legs are validated as absolute levels (an offset leg is ``entry + offset``): finite, > 0, take-profit
        above and stop-loss below the entry for a BUY, reversed for a SELL (so a BUY's tp offset must be > 0, its
        sl offset < 0; CONTRACTS ``tp_is_offset``).

        ``committed_reduce_qty`` is the quantity already committed to resting reduce orders of this order's position
        (the ExecutionEngine passes it). A reducing order is limited to ``|position| - committed``: clipped to it
        (``clipped_to_position``), or denied ``reduce_exceeds_position`` when nothing is left. A flipping order
        in ACTIVE is not a reduction and keeps its meaning (it is clipped only under reduce_only, REDUCING or the
        loss limit, like before, now to the same capacity).

        Closing: when ``reduce_only`` or the order asks for at least the whole position, and flooring to the size
        step would leave a remainder below ``min_qty`` (an unsellable dust), the whole position is submitted,
        unrounded. Opening orders still floor.

        Risk reduction stays possible: an order that only reduces its position (after clipping to it) is exempt from
        ``max_order_notional``, the position and daily-loss limits and the per-minute rate limit and takes no rate
        slot; it is still subject to HALTED, ``invalid_order``, rounding, min/max quantity and min notional.
        Opening orders keep every limit. While the daily loss limit is reached, a flipping order is clipped to the
        position like in REDUCING. On spot a BUY above available cash
        plus its fee (the cost profile's taker fee for market and stop orders, maker for LIMIT, at the order's
        price) is denied ``insufficient_cash``."""
        ins, lim = self.instrument, self.limits
        if self.state is TradingState.HALTED:
            return _deny(HALTED)
        if order.instrument_id != ins.id:
            return _deny(WRONG_INSTRUMENT)
        if not _pos(ref_price):
            return _deny(INVALID_ORDER)
        marks = dict(marks) if marks is not None else {ins.id: ref_price}
        if any(p.qty > 0 and p.instrument_id not in marks for p in self.portfolio.positions.values()):
            return _deny(MISSING_MARKS)
        self.update(marks, day_start_equity=day_start_equity)
        sign = 1 if order.side is Side.BUY else -1
        committed = committed_reduce_qty
        if not (isinstance(committed, (int, float)) and math.isfinite(committed) and committed >= 0):
            return _deny(INVALID_ORDER)

        # price: rounded to the tick; a market order trades at ref_price
        px: Optional[float] = None
        if order.price is not None:
            if not _pos(order.price):
                return _deny(INVALID_ORDER)
            px = round_nearest(order.price, ins.price_increment)
            if px <= 0:
                return _deny(PRICE_ROUNDS_TO_ZERO)
        trig: Optional[float] = None
        if order.trigger_price is not None:
            if not _pos(order.trigger_price):
                return _deny(INVALID_ORDER)
            trig = round_nearest(order.trigger_price, ins.price_increment)
            if trig <= 0:
                return _deny(PRICE_ROUNDS_TO_ZERO)
        if trig is None:
            price = px if px is not None else ref_price
        elif sign > 0:
            price = max(trig, px or 0.0, ref_price)
        else:
            price = trig
        legs = self._bracket_levels(order, sign, price)
        if legs is None:
            return _deny(INVALID_ORDER)
        val_price = max([price, *legs])

        # quantity: a size_frac order resolves on REALISED equity (A5.1/A5.9: cash + signed qty x avg price)
        if order.qty is not None:
            qty = order.qty
        else:
            frac = order.size_frac
            if frac is None or not math.isfinite(frac):
                return _deny(INVALID_ORDER)
            qty = min(max(frac, 0.0), 1.0) * self.portfolio.realised_equity() / price
        if not (math.isfinite(qty) and qty > 0):
            return _deny(INVALID_ORDER)
        raw = qty
        qty = round_down(qty, ins.size_increment)
        if qty <= 0:
            return _deny(QTY_ROUNDS_TO_ZERO)

        pos = self.portfolio.position(ins.id, order.strategy_id)
        held = self.portfolio.signed_qty(pos)
        reducing = held * sign < 0
        loss_hit = lim.daily_loss_limit is not None and self.day_loss(marks) >= lim.daily_loss_limit
        clipped = False

        limited = order.reduce_only or self.state is TradingState.REDUCING or loss_hit
        if limited and not reducing:
            if order.reduce_only:
                return _deny(REDUCE_ONLY_VIOLATION)
            return _deny(REDUCING_ONLY if self.state is TradingState.REDUCING else DAILY_LOSS_LIMIT)
        if reducing and (limited or qty <= abs(held) + DUST):  # a reduction, or a flip that must be clipped
            cap = abs(held) - committed  # what no resting reduce order has claimed yet
            if cap <= DUST:
                return _deny(REDUCE_EXCEEDS_POSITION)
            if qty > cap + DUST:
                qty, clipped = round_down(cap, ins.size_increment), True
                if qty <= 0:
                    return _deny(QTY_ROUNDS_TO_ZERO)
            if (order.reduce_only or raw >= cap - DUST) and DUST < cap - qty < ins.min_qty:
                qty = cap  # closing: do not leave an unsellable remainder
        pure_reduce = reducing and qty <= abs(held) + DUST

        if ins.is_spot and sign < 0:
            balance = self.portfolio.base_balance(ins.id)
            if qty > balance + DUST:
                return _deny(SPOT_SELL_EXCEEDS_BALANCE)
            if qty > held + DUST:  # covered by another strategy's long, but this one has no long of its own
                return _deny(SPOT_SHORT_NO_OWN_LONG)

        if qty < ins.min_qty:
            return _deny(MIN_QTY)
        if ins.max_qty is not None and qty > ins.max_qty:
            return _deny(MAX_QTY)
        notional = qty * (price if pure_reduce else val_price)
        if notional < ins.min_notional:
            return _deny(MIN_NOTIONAL)
        if not pure_reduce:
            if ins.is_spot and sign > 0:
                liq = Liquidity.MAKER if order.type is OrderType.LIMIT else Liquidity.TAKER
                if qty * price * (1 + self.costs.fee_rate(liq)) > self.portfolio.cash:
                    return _deny(INSUFFICIENT_CASH)
            if lim.max_order_notional is not None and notional > lim.max_order_notional:
                return _deny(MAX_ORDER_NOTIONAL)
            after = held + sign * qty
            if (lim.max_position_notional is not None and abs(after) > abs(held)
                    and abs(after) * val_price > lim.max_position_notional):
                return _deny(MAX_POSITION)
        now = self._clock.now_ms()
        if lim.max_orders_per_minute is not None and not pure_reduce:
            while self._sent and self._sent[0] <= now - WINDOW_MS:  # 60 s old or older leaves the window
                self._sent.popleft()
            if len(self._sent) >= lim.max_orders_per_minute:
                return _deny(MAX_ORDERS_PER_MINUTE)
        if record and not pure_reduce:
            self._sent.append(now)
        return RiskDecision(True, CLIPPED_TO_POSITION if clipped else "", qty, px)

    def _bracket_levels(self, order: Order, sign: int, entry: float) -> Optional[list[float]]:
        """Absolute bracket levels of an order (for valuation), or None when a leg is invalid."""
        br = order.bracket
        if not br:
            return []
        offset = bool(br.get("tp_is_offset"))
        out: list[float] = []
        for key, up in (("take_profit", sign > 0), ("stop_loss", sign < 0)):
            v = br.get(key)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                return None
            lv = entry + v if offset else v
            if not math.isfinite(lv) or lv <= 0 or (lv <= entry if up else lv >= entry):
                return None
            out.append(lv)
        return out


def _pos(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


__all__ = ["RiskGuard", "RiskLimits", "RiskDecision", "TradingState", "round_down", "round_nearest"]

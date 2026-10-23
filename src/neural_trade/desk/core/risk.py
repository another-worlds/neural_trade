"""RiskGuard (PLAN A2 ``risk.py``; DESK-1.2): pre-trade checks for every order in every mode.

CONTRACTS fixes only: the trading states ACTIVE|REDUCING|HALTED, "RiskGuard denies SELL beyond the held base
balance on spot accounts", the limit list (precision/lot rounding, min notional, max order notional, max position,
max orders per minute, daily loss limit) and ``RiskLimits.unlimited()``. The rest (signatures, reason strings,
defaults, transitions) are the minimum choices listed in the NT-223 report. Pure logic, standard library only.

``check`` is called by the ExecutionEngine before ``Broker.submit`` and by the manual-order path; a ``size_frac``
order is checked twice (at submit on the last close, at the fill on the resolved quantity), so a rate-limit slot is
taken only by ``check(..., record=True)`` (the default); pass ``record=False`` for the estimate.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Callable, Mapping, Optional

from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.model import Instrument, Order, Side
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
SPOT_SELL_EXCEEDS_BALANCE = "spot_sell_exceeds_balance"
SPOT_SHORT_NO_OWN_LONG = "spot_short_no_own_long"
CLIPPED_TO_POSITION = "clipped_to_position"


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
                 on_state_change: Optional[Callable[[TradingState, TradingState, str], None]] = None) -> None:
        self.limits, self.instrument, self.portfolio, self._clock = limits, instrument, portfolio, clock
        self._on_state_change = on_state_change
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
    def update(self, marks: Mapping[str, float]) -> None:
        """Roll the UTC day: the first call of a day records its starting equity. The engine calls it on every bar."""
        now = self._clock.now_ms()
        day = now // DAY_MS
        if day != self._day:
            self._day, self._day_start_equity = day, self.portfolio.account_state(marks, now).equity

    def day_loss(self, marks: Mapping[str, float]) -> float:
        """Equity fall since the day's start (positive = loss)."""
        self.update(marks)
        return self._day_start_equity - self.portfolio.account_state(marks, self._clock.now_ms()).equity

    # ------------------------------------------------------------------ the check
    def check(self, order: Order, *, ref_price: float, marks: Optional[Mapping[str, float]] = None,
              record: bool = True) -> RiskDecision:
        """Decide on ``order``. ``ref_price`` is the price a market order would trade at (last close at submit, the
        resolved open at the fill); a priced order uses its own (rounded) price. ``marks`` defaults to
        ``{instrument.id: ref_price}``. An allowed decision carries the quantity and price to submit."""
        ins, lim = self.instrument, self.limits
        if self.state is TradingState.HALTED:
            return _deny(HALTED)
        if order.instrument_id != ins.id:
            return _deny(WRONG_INSTRUMENT)
        if not (math.isfinite(ref_price) and ref_price > 0):
            return _deny(INVALID_ORDER)
        marks = dict(marks) if marks is not None else {ins.id: ref_price}
        self.update(marks)

        # price: rounded to the tick; a market order trades at ref_price
        px: Optional[float] = None
        if order.price is not None:
            if not (math.isfinite(order.price) and order.price > 0):
                return _deny(INVALID_ORDER)
            px = round_nearest(order.price, ins.price_increment)
            if px <= 0:
                return _deny(PRICE_ROUNDS_TO_ZERO)
        price = px if px is not None else ref_price

        # quantity: a size_frac order resolves on realised equity (A5.1), then floors to the lot size
        if order.qty is not None:
            qty = order.qty
        else:
            frac = order.size_frac
            if frac is None or not math.isfinite(frac):
                return _deny(INVALID_ORDER)
            equity = self.portfolio.account_state(marks, self._clock.now_ms()).equity
            qty = min(max(frac, 0.0), 1.0) * equity / price
        if not (math.isfinite(qty) and qty > 0):
            return _deny(INVALID_ORDER)
        qty = round_down(qty, ins.size_increment)
        if qty <= 0:
            return _deny(QTY_ROUNDS_TO_ZERO)

        sign = 1 if order.side is Side.BUY else -1
        pos = self.portfolio.position(ins.id, order.strategy_id)
        held = self.portfolio.signed_qty(pos)
        reducing = held * sign < 0
        clipped = False

        if order.reduce_only or self.state is TradingState.REDUCING:
            if not reducing:
                return _deny(REDUCE_ONLY_VIOLATION if order.reduce_only else REDUCING_ONLY)
            if qty > abs(held) + DUST:  # would flip: clip to the position
                qty, clipped = round_down(abs(held), ins.size_increment), True
                if qty <= 0:
                    return _deny(QTY_ROUNDS_TO_ZERO)

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
        notional = qty * price
        if notional < ins.min_notional:
            return _deny(MIN_NOTIONAL)
        if lim.max_order_notional is not None and notional > lim.max_order_notional:
            return _deny(MAX_ORDER_NOTIONAL)
        after = held + sign * qty
        if (lim.max_position_notional is not None and abs(after) > abs(held)
                and abs(after) * price > lim.max_position_notional):
            return _deny(MAX_POSITION)
        if (lim.daily_loss_limit is not None and not reducing
                and self.day_loss(marks) >= lim.daily_loss_limit):
            return _deny(DAILY_LOSS_LIMIT)
        now = self._clock.now_ms()
        if lim.max_orders_per_minute is not None:
            while self._sent and self._sent[0] <= now - WINDOW_MS:  # 60 s old or older leaves the window
                self._sent.popleft()
            if len(self._sent) >= lim.max_orders_per_minute:
                return _deny(MAX_ORDERS_PER_MINUTE)
        if record:
            self._sent.append(now)
        return RiskDecision(True, CLIPPED_TO_POSITION if clipped else "", qty, px)


__all__ = ["RiskGuard", "RiskLimits", "RiskDecision", "TradingState", "round_down", "round_nearest"]

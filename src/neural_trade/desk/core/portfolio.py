"""Portfolio (PLAN A2, A5; DESK-1.1): NETTING positions, cash, costs, realised and unrealised P&L, equity mark.

One ``Portfolio`` is one account. Positions are netted per ``position_id`` (``{instrument}-{strategy}``). Pure
logic, standard library only. Accounting follows ``strategy/backtest.py`` operand for operand:

* cash: ``cash -= side x qty x Fill.price + Fill.fee`` (``Fill.price`` is the slipped price);
* a position's ``avg_price`` and every P&L are on MIDS (``Fill.mid``); the slip cost of a fill is
  ``side x (price - mid) x qty`` and the fee is ``Fill.fee``, so a closed trade's ``costs`` = slip + fees of its
  legs and ``net_pnl = gross_pnl - costs`` (the oracle's ``Trade``);
* ``Position.realised_pnl`` is gross on mids and ``Position.fees`` the fees only, both for the current
  open-to-flat lifecycle (reset when a flat position reopens); the account's ``realised_pnl`` is net: gross
  realised minus every cost paid, i.e. the oracle's ``equity_cash - initial``;
* equity = cash + signed quantity x mark = initial + realised + unrealised (an identity the tests pin).

Gaps CONTRACTS leaves open (minimum choices, listed in the NT-222 report): the Portfolio API itself, the
closed-trade record, partial-reduce trade records, ignoring a repeated non-empty ``trade_id``, and the spot
backstop raising ``OverfillError`` for a sell beyond the held base balance, ``Portfolio.restore`` (positions and
cash from the database after a restart) and the bracket storage API (``set_bracket`` / ``get_bracket``; the stored
value is the ``Order.bracket`` dict, CONTRACTS A5.3; the Portfolio only stores it, the engine decides when it is
replaced).

Quantities within ``DUST`` (1e-12) of a position are treated as that position: a fill that overshoots a close by no
more than ``DUST`` closes it and opens nothing, on both sides.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.model import (AccountKind, AccountState, Fill, OverfillError, Position, PositionEvent,
                                          PositionSide, Side)


DUST = 1e-12  # quantity tolerance for closing a position and for the spot backstop


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class ClosedTrade:
    """A reduction of a position (a full close in the oracle's terms): gross on mids, costs = slip + fees."""

    position_id: str
    side: PositionSide  # the side of the position that was reduced
    qty: float
    entry_price: float  # the position's average mid
    exit_price: float  # the reducing fill's mid
    ts_opened: int
    ts_closed: int
    gross_pnl: float
    costs: float

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs


class Portfolio:
    def __init__(self, account_id: str, *, kind: AccountKind = AccountKind.BACKTEST, quote_currency: str = "USDT",
                 initial_cash: float = 10_000.0, allow_short: bool = True, clock: Optional[Clock] = None) -> None:
        """``allow_short=False`` is the spot backstop: a sell may not take the NETTED base balance of an instrument
        below zero. It is a per-instrument rule, not a per-strategy one: a strategy with no position of its own may
        go SHORT while another strategy's long keeps the netted balance >= 0 (open question for RiskGuard, DESK-1.2).
        It is not derived from ``Instrument.is_spot``; the caller (ExecutionEngine) sets it."""
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {initial_cash}")
        self.account_id, self.kind, self.quote_currency = account_id, AccountKind(kind), quote_currency
        self.initial_cash, self.cash, self.allow_short = float(initial_cash), float(initial_cash), allow_short
        self._clock = clock
        self.positions: dict[str, Position] = {}
        self.trades: list[ClosedTrade] = []
        self.equity_curve: list[tuple[int, float]] = []  # (ts, equity) of every mark
        self.fees_paid = self.costs_paid = self.gross_realised = 0.0
        self.peak_equity = self.initial_cash
        self.max_drawdown = 0.0  # largest peak-to-trough fall as a fraction of the peak (positive, as the oracle)
        self.drawdown = 0.0  # the same at the last mark
        self._entry_costs: dict[str, float] = {}  # costs attached to a position's open quantity
        self._seen: set[str] = set()
        self._brackets: dict[str, dict[str, Any]] = {}  # active bracket per open position (A5.3)

    # ------------------------------------------------------------------ restart
    def restore(self, positions: Iterable[Position], cash: float, *, entry_costs: Optional[Mapping[str, float]] = None,
                brackets: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
        """Load open positions and cash stored by a previous process into this FRESH portfolio, so that they close
        with the right P&L. ``entry_costs`` maps position_id to the costs (slip + fees) attached to its open quantity
        (default 0); ``brackets`` maps position_id to its stored bracket. Flat positions are skipped. The equity curve,
        trades and cost totals are not restored (they are history, not state)."""
        if self.positions or self.trades or self._seen:
            raise ValueError("restore needs a fresh portfolio")
        _finite("cash", cash)
        loaded: dict[str, Position] = {}
        for p in positions:
            if p.account_id != self.account_id:
                raise ValueError(f"position {p.position_id} belongs to account {p.account_id}")
            for name in ("qty", "avg_price", "realised_pnl", "fees"):
                _finite(f"{p.position_id}.{name}", getattr(p, name))
            if p.side is PositionSide.FLAT or p.qty <= 0:
                continue
            if p.position_id in loaded:
                raise ValueError(f"duplicate position {p.position_id}")
            loaded[p.position_id] = dataclasses.replace(p)
        costs = {k: _finite(f"entry_costs[{k}]", float(v)) for k, v in (entry_costs or {}).items()}
        for k in list(costs) + list(brackets or {}):
            if k not in loaded:
                raise ValueError(f"entry cost or bracket for unknown or flat position {k}")
        self.positions.update(loaded)
        self._entry_costs.update({k: costs.get(k, 0.0) for k in loaded})
        self._brackets.update({k: dict(v) for k, v in (brackets or {}).items()})
        self.cash = float(cash)

    # ------------------------------------------------------------------ brackets
    def set_bracket(self, position_id: str, bracket: Optional[Mapping[str, Any]]) -> None:
        """Store the active bracket of an open position (``None`` clears it). Cleared automatically on close."""
        pos = self.positions.get(position_id)
        if pos is None or pos.qty <= 0:
            raise ValueError(f"no open position {position_id}")
        if bracket is None:
            self._brackets.pop(position_id, None)
        else:
            self._brackets[position_id] = dict(bracket)

    def get_bracket(self, position_id: str) -> Optional[dict[str, Any]]:
        b = self._brackets.get(position_id)
        return None if b is None else dict(b)

    # ------------------------------------------------------------------ reads
    def position(self, instrument_id: str, strategy_id: Optional[str] = None) -> Position:
        pid = f"{instrument_id}-{strategy_id}"
        return self.positions.get(pid) or Position(instrument_id=instrument_id, account_id=self.account_id,
                                                   strategy_id=strategy_id)

    def signed_qty(self, position: Position) -> float:
        return position.qty * (1 if position.side is PositionSide.LONG else -1 if position.side is PositionSide.SHORT
                               else 0)

    def base_balance(self, instrument_id: str) -> float:
        """Signed quantity of an instrument over all its positions (the spot base balance)."""
        return sum(self.signed_qty(p) for p in self.positions.values() if p.instrument_id == instrument_id)

    # ------------------------------------------------------------------ fills
    def apply_fill(self, fill: Fill, *, strategy_id: Optional[str] = None) -> list[PositionEvent]:
        """Apply a fill to cash and to the NETTING position. Returns one ``PositionEvent`` (position snapshot) per
        position change; a fill that crosses zero returns two: the close, then the opening of the other side.
        A repeated non-empty ``trade_id`` is ignored (returns [])."""
        if not (math.isfinite(fill.qty) and fill.qty > 0):
            raise ValueError(f"fill qty must be finite and > 0, got {fill.qty}")
        for name in ("price", "mid", "fee"):
            _finite(f"fill {name}", getattr(fill, name))
        if fill.trade_id and fill.trade_id in self._seen:
            return []
        sign = 1 if fill.side is Side.BUY else -1
        pos = self.positions.get(f"{fill.instrument_id}-{strategy_id}")
        held = self.signed_qty(pos) if pos else 0.0
        if not self.allow_short and sign < 0 and self.base_balance(fill.instrument_id) - fill.qty < -DUST:
            raise OverfillError(f"sell of {fill.qty} exceeds the held base balance "
                                f"{self.base_balance(fill.instrument_id)} on {fill.instrument_id}")
        if fill.trade_id:
            self._seen.add(fill.trade_id)
        if pos is None:
            pos = self.positions[f"{fill.instrument_id}-{strategy_id}"] = Position(
                instrument_id=fill.instrument_id, account_id=self.account_id, strategy_id=strategy_id)
        slip = sign * (fill.price - fill.mid) * fill.qty
        costs = slip + fill.fee
        self.cash -= sign * fill.qty * fill.price + fill.fee
        self.fees_paid += fill.fee
        self.costs_paid += costs

        events: list[PositionEvent] = []
        close_qty = min(fill.qty, abs(held)) if held * sign < 0 else 0.0
        open_qty = fill.qty - close_qty
        if close_qty > 0 and open_qty <= DUST:  # an overshoot within the tolerance closes, it does not flip
            open_qty = 0.0
        if close_qty > 0:
            share = close_qty / fill.qty
            entry_part = self._entry_costs[pos.position_id] * close_qty / pos.qty
            gross = (1 if held > 0 else -1) * close_qty * (fill.mid - pos.avg_price)
            self._entry_costs[pos.position_id] -= entry_part
            self.gross_realised += gross
            self.trades.append(ClosedTrade(pos.position_id, pos.side, close_qty, pos.avg_price, fill.mid,
                                           pos.ts_opened, fill.ts, gross, entry_part + costs * share))
            pos.realised_pnl += gross
            pos.fees += fill.fee * share
            pos.qty -= close_qty
            if pos.qty <= DUST:
                pos.qty, pos.side, pos.ts_closed = 0.0, PositionSide.FLAT, fill.ts
                self._entry_costs[pos.position_id] = 0.0
                self._brackets.pop(pos.position_id, None)
            events.append(self._event(pos, fill))
            if open_qty > 0:  # the next position starts from zero: fresh lifecycle
                pos.avg_price, pos.realised_pnl, pos.fees = 0.0, 0.0, 0.0
        if open_qty > 0:
            share = open_qty / fill.qty
            if pos.qty == 0.0:
                pos.side = PositionSide.LONG if sign > 0 else PositionSide.SHORT
                pos.avg_price, pos.ts_opened, pos.ts_closed = 0.0, fill.ts, None
                if close_qty == 0:
                    pos.realised_pnl, pos.fees = 0.0, 0.0
                self._entry_costs[pos.position_id] = 0.0
            pos.avg_price = (pos.qty * pos.avg_price + open_qty * fill.mid) / (pos.qty + open_qty)
            pos.qty += open_qty
            pos.fees += fill.fee * share
            self._entry_costs[pos.position_id] += costs * share
            events.append(self._event(pos, fill))
        return events

    def _event(self, pos: Position, fill: Fill) -> PositionEvent:
        ts_init = self._clock.now_ms() if self._clock else fill.ts
        return PositionEvent(ts_event=fill.ts, ts_init=ts_init, position=dataclasses.replace(pos))

    # ------------------------------------------------------------------ marks
    def unrealised_pnl(self, marks: Mapping[str, float]) -> float:
        return sum(self.signed_qty(p) * (marks[p.instrument_id] - p.avg_price)
                   for p in self.positions.values() if p.qty > 0)

    def equity(self, marks: Mapping[str, float]) -> float:
        """cash + signed quantity x mark, net of costs paid (A5.8). ``marks`` maps instrument id to price."""
        return self.cash + sum(self.signed_qty(p) * marks[p.instrument_id] for p in self.positions.values()
                               if p.qty > 0)

    def _check_marks(self, marks: Mapping[str, float]) -> None:
        for p in self.positions.values():
            if p.qty > 0:
                _finite(f"mark {p.instrument_id}", marks[p.instrument_id])

    def account_state(self, marks: Mapping[str, float], ts: int) -> AccountState:
        """The account at ``marks`` WITHOUT touching the equity curve, peak or drawdown (RiskGuard, strategy context)."""
        self._check_marks(marks)
        return self._state(self.equity(marks), marks, ts)

    def mark(self, marks: Mapping[str, float], ts: int) -> AccountState:
        """Mark equity at ``marks`` (bar closes), update the drawdown, record the point on ``equity_curve``.
        A non-finite mark raises ``ValueError`` before anything changes."""
        self._check_marks(marks)
        eq = self.equity(marks)
        self.equity_curve.append((int(ts), eq))
        self.peak_equity = max(self.peak_equity, eq)
        self.drawdown = (self.peak_equity - eq) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, self.drawdown)
        return self._state(eq, marks, ts)

    def _state(self, eq: float, marks: Mapping[str, float], ts: int) -> AccountState:
        return AccountState(account_id=self.account_id, kind=self.kind, quote_currency=self.quote_currency,
                            cash=self.cash, equity=eq, locked=0.0,
                            realised_pnl=self.gross_realised - self.costs_paid,
                            unrealised_pnl=self.unrealised_pnl(marks), ts=int(ts))

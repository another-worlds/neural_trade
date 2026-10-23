"""ExecutionEngine (PLAN A2 ``execution.py``, A4, A5; DESK-1.3): commands -> RiskGuard -> Broker, events -> state
machine -> Portfolio -> bus.

One engine is one account (one ``Portfolio``, one ``RiskGuard``, one ``Broker``). Pure logic, standard library only:
the database is not imported; the journal is an injected sink, any callable ``(kind, payload) -> None`` (the
``Journal.append`` of C1 fits).

Order status changes happen ONLY in ``apply`` (``submit`` and ``cancel`` build their own events and apply them).
The legal edges are ``model.STATUS_EDGES`` plus the ``late_fill`` exception (A4).

Per fill, in this order, after the Portfolio is updated: ``OrderEvent`` (the status change), ``FillEvent``,
``PositionEvent`` (one or two), ``AccountEvent``; the journal gets the same records in the same order.

Gaps CONTRACTS leaves open (minimum choices, listed in the NT-224 report):

* bus topics ``"order"``, ``"fill"``, ``"position"``, ``"account"``; journal kinds ``order_<status>`` (for example
  ``order_denied``), ``late_fill``, ``fill``, ``position``, ``account``, ``bracket``, ``halt``; the journal payload
  carries ``ts_event`` and ``ts_init`` itself (``Journal.append`` takes no ts_event);
* ``OrderEvent.kind`` from a broker is the lower-case status it reports: accepted, rejected, triggered, canceled,
  expired; the engine emits submitted, denied, partially_filled, filled and late_fill itself;
* a ``size_frac`` order has no known total quantity (the broker resolves it at the fill): its first Fill is terminal
  (FILLED) and the by-quantity overfill check does not apply to it; a later Fill with a new trade_id raises
  ``OverfillError`` through the FILLED rule;
* the Broker protocol has no modify, so ``ModifyOrder`` raises ``NotImplementedError``; a user's explicit close goes
  through RiskGuard like any order (the force_reduce path of NT-272 does not exist yet);
* an offset bracket (``tp_is_offset``) is stored verbatim; resolving it against the fill bar's open is the Engine's /
  SimBroker's job (they hold the bar).
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Union

from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.model import (
    AccountEvent, CancelAll, CancelOrder, ClosePosition, DeskError, Fill, FillEvent, Halt, ModifyOrder, Order,
    OrderEvent, OrderStatus, OrderType, OverfillError, Side, SubmitOrder, check_transition,
)
from neural_trade.desk.core.portfolio import DUST, Portfolio
from neural_trade.desk.core.risk import MISSING_MARKS, RiskGuard

log = logging.getLogger(__name__)

JournalSink = Callable[[str, dict], None]
Command = Union[SubmitOrder, ModifyOrder, CancelOrder, CancelAll, ClosePosition, Halt]

_S = OrderStatus
RESTING = frozenset({_S.ACCEPTED, _S.TRIGGERED, _S.PARTIALLY_FILLED})  # can be cancelled
_LIVE = frozenset({_S.SUBMITTED, *RESTING})  # sent to the broker and not closed
_KIND_STATUS = {"accepted": _S.ACCEPTED, "rejected": _S.REJECTED, "triggered": _S.TRIGGERED,
                "canceled": _S.CANCELED, "expired": _S.EXPIRED}


class UnknownOrder(DeskError):
    pass


class Broker(Protocol):
    """C1 ``desk/broker/base.py`` (the real module arrives with DESK-1.4); only what the engine calls."""

    def submit(self, order: Order) -> OrderEvent: ...
    def cancel(self, client_id: str) -> OrderEvent: ...


class ExecutionEngine:
    def __init__(self, broker: Broker, portfolio: Portfolio, risk: RiskGuard, bus: EventBus, clock: Clock,
                 journal: JournalSink) -> None:
        self.broker, self.portfolio, self.risk, self.bus, self.clock = broker, portfolio, risk, bus, clock
        self._journal = journal
        self.marks: dict[str, float] = {}  # last price per instrument; the Engine sets it at every bar close
        self._orders: dict[str, Order] = {}
        self._reserved: dict[str, float] = {}  # quantity the risk check approved per order (reduce-order accounting)
        self._trades: set[str] = set()
        self._seq = 0

    # ------------------------------------------------------------------ reads
    def order(self, client_id: str) -> Optional[Order]:
        return self._orders.get(client_id)

    @property
    def orders(self) -> Mapping[str, Order]:
        return dict(self._orders)

    def open_orders(self, instrument_id: Optional[str] = None) -> list[Order]:
        return [o for o in self._orders.values() if o.status in _LIVE
                and (instrument_id is None or o.instrument_id == instrument_id)]

    def set_marks(self, marks: Mapping[str, float]) -> None:
        self.marks.update({k: float(v) for k, v in marks.items()})

    # ------------------------------------------------------------------ commands
    def handle(self, cmd: Command) -> Any:
        if isinstance(cmd, SubmitOrder):
            return self.submit(cmd.order)
        if isinstance(cmd, CancelOrder):
            return self.cancel(cmd.client_id)
        if isinstance(cmd, CancelAll):
            return self.cancel_all(cmd.account_id, cmd.instrument_id)
        if isinstance(cmd, ClosePosition):
            return self.close_position(cmd.position_id, cmd.tag)
        if isinstance(cmd, Halt):
            return self.halt(cmd.reason)
        if isinstance(cmd, ModifyOrder):
            raise NotImplementedError("the Broker protocol (CONTRACTS C1) has no modify; cancel and resubmit")
        raise TypeError(f"unknown command {type(cmd).__name__}")

    def submit(self, order: Order) -> Order:
        """RiskGuard first; a denial is a DENIED order (reason + journal record) and never reaches the broker.
        Idempotent on ``client_id``: the existing order comes back and nothing else happens."""
        existing = self._orders.get(order.client_id)
        if existing is not None:
            return existing
        if order.account_id != self.portfolio.account_id:
            raise ValueError(f"order {order.client_id} is for account {order.account_id}, "
                             f"this engine trades {self.portfolio.account_id}")
        now = self.clock.now_ms()
        o = dataclasses.replace(order, tags=dict(order.tags), status=_S.INITIALIZED,
                                ts_created=order.ts_created or now, ts_updated=now)
        self._orders[o.client_id] = o
        ref = self._ref_price(o)
        if ref is None:
            decision_reason, qty, price = MISSING_MARKS, None, None
        else:
            marks = {**self.marks}
            marks.setdefault(o.instrument_id, ref)
            d = self.risk.check(o, ref_price=ref, marks=marks, committed_reduce_qty=self._committed_reduce(o))
            decision_reason, qty, price = ("" if d.allowed else d.reason), d.qty, d.price
        if decision_reason:
            self.apply(OrderEvent(now, now, "denied", self._with_reason(o, decision_reason)))
            return o
        if o.qty is not None:
            o.qty = qty  # rounded / clipped by the risk check; a size_frac order keeps its fraction
        if price is not None:
            o.price = price
        self._reserved[o.client_id] = qty
        self.apply(OrderEvent(now, now, "submitted", o))
        self.apply(self.broker.submit(dataclasses.replace(o, tags=dict(o.tags))))
        return o

    def cancel(self, client_id: str) -> Order:
        o = self._orders.get(client_id)
        if o is None:
            raise UnknownOrder(client_id)
        if o.is_closed:
            return o
        if o.status not in RESTING:  # SUBMITTED has no cancel edge (A4)
            check_transition(o.status, _S.CANCELED)
        self.apply(self.broker.cancel(client_id))
        return o

    def cancel_all(self, account_id: Optional[str] = None, instrument_id: Optional[str] = None) -> list[Order]:
        """Cancel every resting order of the account (and instrument), in submission order."""
        if account_id is not None and account_id != self.portfolio.account_id:
            return []
        return [self.cancel(o.client_id) for o in list(self._orders.values())
                if o.status in RESTING and (instrument_id is None or o.instrument_id == instrument_id)]

    def close_position(self, position_id: str, tag: str = "") -> Optional[Order]:
        """A reduce-only market order for the whole position, through RiskGuard like any order (so HALTED denies it)."""
        pos = self.portfolio.positions.get(position_id)
        if pos is None or pos.qty <= DUST:
            return None
        self._seq += 1
        side = Side.SELL if self.portfolio.signed_qty(pos) > 0 else Side.BUY
        return self.submit(Order(client_id=f"close:{position_id}:{self._seq}", instrument_id=pos.instrument_id,
                                 account_id=self.portfolio.account_id, strategy_id=pos.strategy_id, side=side,
                                 type=OrderType.MARKET, qty=pos.qty, reduce_only=True,
                                 tags={"tag": tag, "reason": "CLOSE"}))

    def halt(self, reason: str) -> None:
        """RiskGuard to HALTED (denies every new order), then cancel what rests. A failing cancel is logged and
        journalled; it never stops the others."""
        self.risk.halt(reason)
        self._write("halt", {"reason": reason}, self.clock.now_ms())
        for o in [x for x in self._orders.values() if x.status in RESTING]:
            try:
                self.cancel(o.client_id)
            except Exception as exc:  # noqa: BLE001 - a halt must reach every order
                log.exception("halt: cancel of %s failed", o.client_id)
                self._write("halt_cancel_failed", {"client_id": o.client_id, "error": repr(exc)}, self.clock.now_ms())

    # ------------------------------------------------------------------ events: the only place status changes
    def apply_all(self, events: Iterable[Union[Fill, FillEvent, OrderEvent]]) -> None:
        for ev in list(events):
            self.apply(ev)

    def apply(self, event: Union[Fill, FillEvent, OrderEvent]) -> None:
        if isinstance(event, Fill):
            event = FillEvent(event.ts, self.clock.now_ms(), event)
        if isinstance(event, FillEvent):
            self._apply_fill(event)
        elif isinstance(event, OrderEvent):
            self._apply_order_event(event)
        else:
            raise TypeError(f"cannot apply {type(event).__name__}")

    def _apply_order_event(self, ev: OrderEvent) -> None:
        o = self._get(ev.order.client_id)
        kind = ev.kind
        new = _S(kind.upper()) if kind in ("denied", "submitted") else _KIND_STATUS.get(kind)
        if new is None:
            raise ValueError(f"unsupported order event kind {kind!r}")
        check_transition(o.status, new)
        if ev.order.venue_id:
            o.venue_id = ev.order.venue_id
        self._commit(o, new, kind, ev.ts_event, reason=ev.order.reason)

    def _apply_fill(self, ev: FillEvent) -> None:
        f = ev.fill
        o = self._get(f.client_id)
        if f.trade_id in self._trades:  # a known trade is never applied twice
            log.debug("duplicate fill %s ignored", f.trade_id)
            return
        if f.instrument_id != o.instrument_id or f.side is not o.side:
            raise ValueError(f"fill {f.trade_id} does not match order {o.client_id}")
        late = o.status is _S.CANCELED
        if o.status is _S.FILLED:
            raise OverfillError(f"fill {f.trade_id} for the already filled order {o.client_id}")
        total = o.filled_qty + f.qty
        if o.qty is not None and total > o.qty + DUST:
            raise OverfillError(f"fills of {o.client_id} sum to {total}, above its quantity {o.qty}")
        done = o.qty is None or total >= o.qty - DUST
        new = _S.FILLED if done else _S.PARTIALLY_FILLED
        check_transition(o.status, new, late_fill=late)  # before anything changes
        before = self.portfolio.signed_qty(self.portfolio.position(o.instrument_id, o.strategy_id))
        pos_events = self.portfolio.apply_fill(f, strategy_id=o.strategy_id)  # may raise: nothing is changed yet
        self._trades.add(f.trade_id)
        o.avg_fill_price = (o.avg_fill_price * o.filled_qty + f.price * f.qty) / total
        o.filled_qty = total
        self._commit(o, new, "late_fill" if late else new.value.lower(), f.ts, late=late, extra={"trade_id": f.trade_id})
        self.bus.publish("fill", ev)
        self._write("fill", {"fill": f.to_dict()}, f.ts)
        for pe in pos_events:
            self.bus.publish("position", pe)
            self._write("position", {"position": pe.position.to_dict()}, pe.ts_event)
        pos = self.portfolio.position(o.instrument_id, o.strategy_id)
        self._replace_bracket(o, pos, before, f)
        marks = {**self.marks}
        marks.setdefault(f.instrument_id, f.mid)
        acct = AccountEvent(f.ts, self.clock.now_ms(), self.portfolio.account_state(marks, f.ts))
        self.bus.publish("account", acct)
        self._write("account", {"account": acct.account.to_dict()}, f.ts)

    # ------------------------------------------------------------------ internals
    def _get(self, client_id: str) -> Order:
        o = self._orders.get(client_id)
        if o is None:
            raise UnknownOrder(client_id)
        return o

    def _ref_price(self, o: Order) -> Optional[float]:
        for p in (self.marks.get(o.instrument_id), o.price, o.trigger_price):
            if p is not None and p > 0:
                return p
        return None

    @staticmethod
    def _with_reason(o: Order, reason: str) -> Order:
        return dataclasses.replace(o, reason=reason, tags=dict(o.tags))

    def _committed_reduce(self, new: Order) -> float:
        """Quantity still claimed by the resting reduce orders of the new order's position (A5, RiskGuard)."""
        held = self.portfolio.signed_qty(self.portfolio.position(new.instrument_id, new.strategy_id))
        if held == 0:
            return 0.0
        opposite = Side.SELL if held > 0 else Side.BUY
        return sum(max(self._reserved.get(o.client_id, 0.0) - o.filled_qty, 0.0) for o in self._orders.values()
                   if o.status in _LIVE and o.side is opposite and o.instrument_id == new.instrument_id
                   and o.strategy_id == new.strategy_id)

    def _commit(self, o: Order, new: OrderStatus, kind: str, ts_event: int, *, late: bool = False,
                reason: Optional[str] = None, extra: Optional[dict] = None) -> None:
        old = o.status
        o.status, o.ts_updated = new, ts_event
        if reason:
            o.reason = reason
        snap = dataclasses.replace(o, tags=dict(o.tags))
        ev = OrderEvent(ts_event, self.clock.now_ms(), kind, snap)
        self.bus.publish("order", ev)
        self._write(kind if late else f"order_{kind}", {"from": old.value, "to": new.value, "reason": o.reason,
                                                       "order": snap.to_dict(), **(extra or {})}, ts_event)

    def _replace_bracket(self, o: Order, pos: Any, before: float, f: Fill) -> None:
        """A5.3: a fill that opens or enlarges the position replaces its bracket with the order's (last fill wins; no
        bracket keeps the old one; a reduction keeps it on the remainder). Every replacement is journalled."""
        sign = 1 if f.side is Side.BUY else -1
        after = self.portfolio.signed_qty(pos)
        if not o.bracket or not (sign * after > DUST and sign * after > sign * before + DUST):
            return
        old = self.portfolio.get_bracket(pos.position_id)
        self.portfolio.set_bracket(pos.position_id, o.bracket)
        self._write("bracket", {"position_id": pos.position_id, "old": old, "new": dict(o.bracket),
                                "client_id": o.client_id}, f.ts)

    def _write(self, kind: str, payload: dict, ts_event: int) -> None:
        self._journal(kind, {"ts_event": ts_event, "ts_init": self.clock.now_ms(), **payload})

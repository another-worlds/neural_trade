"""ExecutionEngine (PLAN A2 ``execution.py``, A4, A5; DESK-1.3): commands -> RiskGuard -> Broker, events -> state
machine -> Portfolio -> bus.

One engine is one account (one ``Portfolio``, one ``RiskGuard``, one ``Broker``). Pure logic, standard library only:
the database is not imported; the journal is an injected sink, any callable ``(kind, payload) -> None`` (the
``Journal.append`` of C1 fits).

Order status changes happen ONLY in ``apply`` (``submit`` and ``cancel`` build their own events and apply them).
The legal edges are ``model.STATUS_EDGES`` plus the ``late_fill`` exception (A4).

**Journal first, then the bus.** Every record is written to the journal before its event is published. A subscriber
that raises never propagates and never loses a later record: the bus isolates it (``bus.errors``, ``on_error``; NT-294);
a ``strict`` bus re-raises, and the engine then logs it and appends it to ``handler_errors`` (the last 100). A journal
sink that raises does not undo the state change (the venue fact is true): the failure is counted in
``journal_failures`` and recorded in ``handler_errors``, the record stays in a pending queue (``JournalWriter``) and is
written, in order, before the next record. Nothing is ever dropped. Above ``JOURNAL_PENDING_LIMIT`` (1000) pending
records the engine calls ``RiskGuard.halt("journal_unavailable")`` once per outage (new orders are denied; exits go
through the force_reduce path of NT-272 when it exists) and logs it at ERROR; sink failures are logged on the first
and then every 1000th failure.

**Dead orders.** An order whose cancel the venue acknowledged (``cancel_acked``) is dead for the engine's bookkeeping
whatever its status after a late fill (A4 ``late_fill``: CANCELED -> PARTIALLY_FILLED): it is not in
``open_orders()``, claims no reduce quantity, holds no reservation, is never cancelled again and cannot be modified.
Its late fills are still applied to the Portfolio (a venue fact), journalled and published, and a later fill may take
it to FILLED.

Per fill, in this order, after the Portfolio is updated: the order record (status change), ``fill``, ``position``
(one or two), ``account``.

**Thread model.** The engine is single-owner: one event-loop task or one thread owns it and every mutation comes from
there; other threads hand work over (``loop.call_soon_threadsafe``). The first mutation records the owner thread id;
a mutation from another thread raises ``RuntimeError`` (``check_owner=False`` switches the guard off). Reads are not
locked: ``open_orders()`` and ``orders`` return snapshots (a new list / dict built from the current values), so a read
from another thread never fails with "dictionary changed size during iteration"; the ``Order`` objects in them are the
live ones and may change under the reader.

Gaps CONTRACTS leaves open (minimum choices, listed in the NT-224 report):

* bus topics ``"order"``, ``"fill"``, ``"position"``, ``"account"``; journal kinds ``order_<status>`` (for example
  ``order_denied``), ``order_modified``, ``order_modify_denied``, ``late_fill``, ``fill``, ``position``, ``account``,
  ``bracket``, ``halt``, ``cancel_failed``, ``halt_cancel_failed``, ``order_modify_failed``; the journal payload carries
  ``ts_event`` and ``ts_init`` itself;
* ``OrderEvent.kind`` from a broker: accepted, rejected, triggered, canceled, expired, modified; the engine emits
  submitted, denied, partially_filled, filled and late_fill itself. A broker that raises in ``submit`` gives a REJECTED
  order with reason ``broker_error:<ExceptionName>``; one that raises in ``cancel`` during ``cancel_all`` / ``halt``
  is journalled per order and the loop goes on (``last_cancel_failures``); one that raises in ``modify`` is journalled
  ``order_modify_failed`` and re-raised as ``BrokerError``;
* a ``size_frac`` order has no known total quantity: its first Fill is terminal (FILLED) and the by-quantity overfill
  check does not apply, unless the broker's ``accepted`` event carries ``OrderEvent.resolved_qty`` (DESK-6.1's live
  broker should send it): then partial fills are allowed up to it (``order.qty`` stays None);
* ``modify`` is allowed on ACCEPTED and PARTIALLY_FILLED orders that are not dead (A4 has no modify edge; TRIGGERED is
  already live at the venue). The modified order is re-checked by RiskGuard (``record=False``: a modify takes no rate
  slot; committed reduce quantity excludes the order itself); price and trigger are rounded to the tick; a denial
  leaves the order unchanged and is journalled; a broker ``modified`` event below the filled quantity is refused;
* a user's explicit close goes through RiskGuard like any order (the force_reduce path of NT-272 does not exist yet);
* an offset bracket (``tp_is_offset``) is stored verbatim; resolving it against the fill bar's open is the Engine's /
  SimBroker's job (they hold the bar).
"""
from __future__ import annotations

import dataclasses
import logging
import math
import threading
from collections import deque
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Union

from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.model import (
    CLOSED_STATUSES, AccountEvent, CancelAll, CancelOrder, ClosePosition, DeskError, Fill, FillEvent, Halt, ModifyOrder, Order,
    OrderEvent, OrderStatus, OrderType, OverfillError, Side, SubmitOrder, check_transition,
)
from neural_trade.desk.core.portfolio import DUST, Portfolio
from neural_trade.desk.core.risk import MISSING_MARKS, RiskGuard, round_nearest

log = logging.getLogger(__name__)

JOURNAL_PENDING_LIMIT = 1000  # pending journal records above which new orders are halted
JOURNAL_LOG_EVERY = 1000  # sink failures are logged on the first and then every 1000th
JOURNAL_HALT_REASON = "journal_unavailable"

JournalSink = Callable[[str, dict], None]
Command = Union[SubmitOrder, ModifyOrder, CancelOrder, CancelAll, ClosePosition, Halt]

_S = OrderStatus
RESTING = frozenset({_S.ACCEPTED, _S.TRIGGERED, _S.PARTIALLY_FILLED})  # can be cancelled
MODIFIABLE = frozenset({_S.ACCEPTED, _S.PARTIALLY_FILLED})
_LIVE = frozenset({_S.SUBMITTED, *RESTING})  # sent to the broker and not closed
_KIND_STATUS = {"accepted": _S.ACCEPTED, "rejected": _S.REJECTED, "triggered": _S.TRIGGERED,
                "canceled": _S.CANCELED, "expired": _S.EXPIRED}
_IDENTITY = ("instrument_id", "strategy_id", "side", "type", "qty", "size_frac", "price", "trigger_price",
             "reduce_only", "bracket")


class UnknownOrder(DeskError):
    pass


class UnknownAccount(DeskError):
    pass


class DuplicateClientId(ValueError):
    pass


class BrokerError(RuntimeError):
    """A broker call failed; ``__cause__`` is the broker's exception."""


class JournalWriter:
    """Writes journal records in order through a sink that may fail. A failed record stays at the head of the queue
    and is retried before the next one; nothing is dropped. Above ``limit`` pending records ``on_overflow`` is called
    once per outage (the queue drained resets it)."""

    def __init__(self, sink: "JournalSink", *, on_error: Callable[[BaseException], None],
                 on_overflow: Callable[[], None], limit: int = JOURNAL_PENDING_LIMIT) -> None:
        self._sink, self._on_error, self._on_overflow, self.limit = sink, on_error, on_overflow, limit
        self._pending: deque[tuple[str, dict]] = deque()
        self.failures = 0
        self.overflowed = False

    @property
    def pending(self) -> int:
        return len(self._pending)

    def write(self, kind: str, record: dict) -> None:
        self._pending.append((kind, record))
        while self._pending:  # earlier failed records go first, in order
            try:
                self._sink(*self._pending[0])
            except Exception as exc:  # noqa: BLE001 - the state change stands; the record is retried
                self.failures += 1
                if self.failures % JOURNAL_LOG_EVERY == 1:
                    log.error("journal sink failed on %s (failure %d, %d pending)", self._pending[0][0],
                              self.failures, len(self._pending), exc_info=exc)
                self._on_error(exc)
                if len(self._pending) > self.limit and not self.overflowed:
                    self.overflowed = True
                    log.error("journal has %d pending records (limit %d): halting new orders",
                              len(self._pending), self.limit)
                    self._on_overflow()
                return
            self._pending.popleft()
        self.overflowed = False


class Broker(Protocol):
    """C1 ``desk/broker/base.py`` (the real module arrives with DESK-1.4); only what the engine calls."""

    def submit(self, order: Order) -> OrderEvent: ...
    def cancel(self, client_id: str) -> OrderEvent: ...
    def modify(self, client_id: str, *, price: Optional[float] = None, trigger_price: Optional[float] = None,
               qty: Optional[float] = None) -> OrderEvent: ...  # returns kind "modified" with the new values


class ExecutionEngine:
    def __init__(self, broker: Broker, portfolio: Portfolio, risk: RiskGuard, bus: EventBus, clock: Clock,
                 journal: JournalSink, *, check_owner: bool = True) -> None:
        self.broker, self.portfolio, self.risk, self.bus, self.clock = broker, portfolio, risk, bus, clock
        self._journal = JournalWriter(journal, on_error=self._record_error,
                                      on_overflow=lambda: self.risk.halt(JOURNAL_HALT_REASON))
        self.marks: dict[str, float] = {}  # last price per instrument; the Engine sets it at every bar close
        self.handler_errors: list[BaseException] = []  # the last 100 subscriber / journal / cancel failures
        self.cancel_acked: set[str] = set()  # orders the venue already cancelled once: dead (module docstring)
        self.last_cancel_failures: list[tuple[str, BaseException]] = []  # of the last cancel_all / halt
        self._check_owner, self._owner = check_owner, None
        self._orders: dict[str, Order] = {}
        self._originals: dict[str, Order] = {}  # as submitted, for the duplicate-client-id comparison
        self._reserved: dict[str, float] = {}  # quantity the risk check approved per order (reduce-order accounting)
        self._resolved: dict[str, float] = {}  # size_frac orders whose accept event carried resolved_qty
        self._trades: set[str] = set()
        self._seq = 0

    def assert_owner(self) -> None:
        if not self._check_owner:
            return
        me = threading.get_ident()
        if self._owner is None:
            self._owner = me
        elif self._owner != me:
            raise RuntimeError("ExecutionEngine is single-owner: mutate it from its owning thread/task only")

    # ------------------------------------------------------------------ reads
    @property
    def journal_failures(self) -> int:
        return self._journal.failures

    @property
    def journal_pending(self) -> int:
        """Journal records not written yet (the sink is failing)."""
        return self._journal.pending

    def order(self, client_id: str) -> Optional[Order]:
        return self._orders.get(client_id)

    @property
    def orders(self) -> Mapping[str, Order]:
        return dict(self._orders)

    def is_live(self, o: Order) -> bool:
        """Sent to the broker, not closed and not dead (a cancel the venue acknowledged)."""
        return o.status in _LIVE and o.client_id not in self.cancel_acked

    def open_orders(self, instrument_id: Optional[str] = None) -> list[Order]:
        """A snapshot list of the live orders (safe to read from another thread)."""
        return [o for o in list(self._orders.values()) if self.is_live(o)
                and (instrument_id is None or o.instrument_id == instrument_id)]

    def set_marks(self, marks: Mapping[str, float]) -> None:
        self.assert_owner()
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
            return self.modify(cmd.client_id, price=cmd.price, trigger_price=cmd.trigger_price, qty=cmd.qty)
        raise TypeError(f"unknown command {type(cmd).__name__}")

    def submit(self, order: Order) -> Order:
        """RiskGuard first; a denial is a DENIED order (reason + journal record) and never reaches the broker.
        Idempotent on ``client_id``: the same content returns the existing order, different content raises
        ``DuplicateClientId`` naming the differing fields."""
        self.assert_owner()
        existing = self._orders.get(order.client_id)
        if existing is not None:
            diff = [f for f in _IDENTITY if getattr(order, f) != getattr(self._originals[order.client_id], f)]
            if diff:
                raise DuplicateClientId(f"client_id {order.client_id} was used with different {', '.join(diff)}")
            return existing
        if order.account_id != self.portfolio.account_id:
            raise ValueError(f"order {order.client_id} is for account {order.account_id}, "
                             f"this engine trades {self.portfolio.account_id}")
        now = self.clock.now_ms()
        self._originals[order.client_id] = dataclasses.replace(order, tags=dict(order.tags))
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
        try:
            ev = self.broker.submit(dataclasses.replace(o, tags=dict(o.tags)))
        except Exception as exc:  # noqa: BLE001 - the venue never saw it: a rejection, not a crash
            log.exception("broker.submit failed for %s", o.client_id)
            self._record_error(exc)
            self._reserved.pop(o.client_id, None)
            t = self.clock.now_ms()
            ev = OrderEvent(t, t, "rejected", self._with_reason(o, f"broker_error:{type(exc).__name__}"))
        self.apply(ev)
        return o

    def cancel(self, client_id: str) -> Order:
        self.assert_owner()
        o = self._get(client_id)
        if o.is_closed or o.client_id in self.cancel_acked:  # a dead order is never cancelled twice
            return o
        if o.status not in RESTING:  # SUBMITTED has no cancel edge (A4)
            check_transition(o.status, _S.CANCELED)
        self.apply(self.broker.cancel(client_id))
        return o

    def cancel_all(self, account_id: Optional[str] = None, instrument_id: Optional[str] = None) -> list[Order]:
        """Cancel every resting order of the account (and instrument), in submission order; returns the cancelled
        ones. An order the venue already cancelled once (and that only came back through a late fill) is not
        cancelled again. A cancel that raises is journalled ``cancel_failed``, recorded in ``handler_errors`` and
        ``last_cancel_failures`` (client_id, exception), and the loop goes on."""
        self.assert_owner()
        if account_id is not None and account_id != self.portfolio.account_id:
            raise UnknownAccount(account_id)
        return self._cancel_each(self._resting(instrument_id), "cancel_failed")

    def _cancel_each(self, orders: list[Order], fail_kind: str) -> list[Order]:
        done: list[Order] = []
        self.last_cancel_failures = []
        for o in orders:
            try:
                done.append(self.cancel(o.client_id))
            except Exception as exc:  # noqa: BLE001 - one failing cancel must not stop the others
                log.error("cancel of %s failed: %r", o.client_id, exc, exc_info=exc)
                self._record_error(exc)
                self.last_cancel_failures.append((o.client_id, exc))
                self._write(fail_kind, {"client_id": o.client_id, "reason": f"broker_error:{type(exc).__name__}",
                                        "error": repr(exc)}, self.clock.now_ms())
        return done

    def _resting(self, instrument_id: Optional[str] = None) -> list[Order]:
        return [o for o in list(self._orders.values()) if o.status in RESTING and o.client_id not in self.cancel_acked
                and (instrument_id is None or o.instrument_id == instrument_id)]

    def modify(self, client_id: str, *, price: Optional[float] = None, trigger_price: Optional[float] = None,
               qty: Optional[float] = None) -> Order:
        """Change price, trigger or total quantity of a resting order. RiskGuard re-checks it; a denial leaves the
        order unchanged and is journalled as ``order_modify_denied``. Price and trigger are rounded to the tick. A
        broker exception is journalled ``order_modify_failed``, leaves the order unchanged and raises
        ``BrokerError``."""
        self.assert_owner()
        o = self._get(client_id)
        if o.status not in MODIFIABLE or o.client_id in self.cancel_acked:
            raise ValueError(f"order {client_id} is {o.status.value}"
                             f"{' (cancel acknowledged)' if o.client_id in self.cancel_acked else ''}; "
                             "only live ACCEPTED or PARTIALLY_FILLED orders can be modified")
        if qty is not None and (o.qty is None or qty <= o.filled_qty):
            raise ValueError(f"qty must exceed the filled {o.filled_qty}; a size_frac order has no qty to change")
        cand = dataclasses.replace(
            o, tags=dict(o.tags), price=o.price if price is None else price,
            trigger_price=o.trigger_price if trigger_price is None else trigger_price,
            qty=None if o.qty is None else (o.qty if qty is None else qty) - o.filled_qty)
        ref = self._ref_price(cand)
        reason, d = MISSING_MARKS, None
        if ref is not None:
            marks = {**self.marks}
            marks.setdefault(o.instrument_id, ref)
            d = self.risk.check(cand, ref_price=ref, marks=marks, record=False,
                                committed_reduce_qty=self._committed_reduce(cand, exclude=client_id))
            reason = "" if d.allowed else d.reason
        if reason:
            self._write("order_modify_denied", {"client_id": client_id, "reason": reason,
                                                "order": dataclasses.replace(o, tags=dict(o.tags)).to_dict()},
                        self.clock.now_ms())
            return o
        new_qty = None if o.qty is None else d.qty + o.filled_qty
        trig = None if cand.trigger_price is None else round_nearest(cand.trigger_price,
                                                                     self.risk.instrument.price_increment)
        try:
            ev = self.broker.modify(client_id, price=d.price if cand.price is not None else None,
                                    trigger_price=trig, qty=new_qty)
        except Exception as exc:  # noqa: BLE001 - journalled, then raised to the caller
            log.error("broker.modify failed for %s: %r", client_id, exc, exc_info=exc)
            self._record_error(exc)
            self._write("order_modify_failed", {"client_id": client_id, "reason": f"broker_error:{type(exc).__name__}",
                                                "order": dataclasses.replace(o, tags=dict(o.tags)).to_dict()},
                        self.clock.now_ms())
            raise BrokerError(f"broker.modify failed for {client_id}: {exc!r}") from exc
        self.apply(ev)
        return o

    def close_position(self, position_id: str, tag: str = "") -> Optional[Order]:
        """A reduce-only market order for the whole position, through RiskGuard like any order (so HALTED denies it)."""
        self.assert_owner()
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
        self.assert_owner()
        self.risk.halt(reason)
        self._write("halt", {"reason": reason}, self.clock.now_ms())
        self._cancel_each(self._resting(), "halt_cancel_failed")

    # ------------------------------------------------------------------ events: the only place status changes
    def apply_all(self, events: Iterable[Union[Fill, FillEvent, OrderEvent]]) -> None:
        for ev in list(events):
            self.apply(ev)

    def apply(self, event: Union[Fill, FillEvent, OrderEvent]) -> None:
        self.assert_owner()
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
        if kind == "modified":
            if o.status not in MODIFIABLE or o.client_id in self.cancel_acked:
                raise ValueError(f"order {o.client_id} is {o.status.value} or dead; it cannot be modified")
            new_qty = ev.order.qty if o.qty is not None else None
            if new_qty is not None and not (math.isfinite(new_qty) and new_qty >= o.filled_qty - DUST):
                raise ValueError(f"modified qty {new_qty} of {o.client_id} is below its filled {o.filled_qty}")
            o.price, o.trigger_price = ev.order.price, ev.order.trigger_price
            if new_qty is not None:
                o.qty = self._reserved[o.client_id] = new_qty
            self._commit(o, o.status, kind, ev.ts_event)
            return
        new = _S(kind.upper()) if kind in ("denied", "submitted") else _KIND_STATUS.get(kind)
        if new is None:
            raise ValueError(f"unsupported order event kind {kind!r}")
        resolved = ev.resolved_qty if new is _S.ACCEPTED and o.qty is None else None
        if resolved is not None and not (math.isfinite(resolved) and resolved > 0):
            raise ValueError(f"resolved_qty {resolved} of {o.client_id} must be finite and > 0")
        check_transition(o.status, new)
        if ev.order.venue_id:
            o.venue_id = ev.order.venue_id
        if resolved is not None:
            self._resolved[o.client_id] = self._reserved[o.client_id] = float(resolved)
        if new is _S.CANCELED:
            self.cancel_acked.add(o.client_id)
        if new in CLOSED_STATUSES:
            self._reserved.pop(o.client_id, None)  # a dead or closed order holds no reservation
        self._commit(o, new, kind, ev.ts_event, reason=ev.order.reason)

    def _apply_fill(self, ev: FillEvent) -> None:
        f = ev.fill
        if not f.trade_id:
            raise ValueError("a fill needs a non-empty trade_id (it is the dedup key)")
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
        cap = o.qty if o.qty is not None else self._resolved.get(o.client_id)  # None: a size_frac nobody resolved
        if cap is not None and total > cap + DUST:
            raise OverfillError(f"fills of {o.client_id} sum to {total}, above its quantity {cap}")
        done = cap is None or total >= cap - DUST
        new = _S.FILLED if done else _S.PARTIALLY_FILLED
        check_transition(o.status, new, late_fill=late)  # before anything changes
        before = self.portfolio.signed_qty(self.portfolio.position(o.instrument_id, o.strategy_id))
        pos_events = self.portfolio.apply_fill(f, strategy_id=o.strategy_id)  # may raise: nothing is changed yet
        self._trades.add(f.trade_id)
        o.avg_fill_price = (o.avg_fill_price * o.filled_qty + f.price * f.qty) / total
        o.filled_qty = total
        if new is _S.FILLED:
            self._reserved.pop(o.client_id, None)
        self._commit(o, new, "late_fill" if late else new.value.lower(), f.ts, late=late, extra={"trade_id": f.trade_id})
        self._emit("fill", ev, "fill", {"fill": f.to_dict()}, f.ts)
        for pe in pos_events:
            self._emit("position", pe, "position", {"position": pe.position.to_dict()}, pe.ts_event)
        self._replace_bracket(o, self.portfolio.position(o.instrument_id, o.strategy_id), before, f)
        marks = {**self.marks}
        marks.setdefault(f.instrument_id, f.mid)
        acct = AccountEvent(f.ts, self.clock.now_ms(), self.portfolio.account_state(marks, f.ts))
        self._emit("account", acct, "account", {"account": acct.account.to_dict()}, f.ts)

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

    def _committed_reduce(self, new: Order, exclude: Optional[str] = None) -> float:
        """Quantity of the new order's position claimed by resting reduce orders (A5, RiskGuard): of each live (not
        dead) opposite-side order of the same instrument and strategy only the part that can reduce the position counts,
        min(its remaining quantity, the position), and the sum is capped at the position."""
        held = self.portfolio.signed_qty(self.portfolio.position(new.instrument_id, new.strategy_id))
        cap = abs(held)
        if cap == 0:
            return 0.0
        opposite = Side.SELL if held > 0 else Side.BUY
        total = sum(min(max(self._reserved.get(o.client_id, 0.0) - o.filled_qty, 0.0), cap)
                    for o in self._orders.values()
                    if self.is_live(o) and o.side is opposite and o.instrument_id == new.instrument_id
                    and o.strategy_id == new.strategy_id and o.client_id != exclude)
        return min(total, cap)

    def _commit(self, o: Order, new: OrderStatus, kind: str, ts_event: int, *, late: bool = False,
                reason: Optional[str] = None, extra: Optional[dict] = None) -> None:
        old = o.status
        o.status, o.ts_updated = new, ts_event
        if reason:
            o.reason = reason
        snap = dataclasses.replace(o, tags=dict(o.tags))
        self._emit("order", OrderEvent(ts_event, self.clock.now_ms(), kind, snap),
                   kind if late else f"order_{kind}",
                   {"from": old.value, "to": new.value, "reason": o.reason, "order": snap.to_dict(), **(extra or {})},
                   ts_event)

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

    def _emit(self, topic: str, event: Any, kind: str, payload: dict, ts_event: int) -> None:
        self._write(kind, payload, ts_event)  # journal first: a crash in a handler loses no record
        try:
            self.bus.publish(topic, event)
        except Exception as exc:  # noqa: BLE001 - a subscriber must not break the engine
            log.exception("subscriber of %r raised", topic)
            self._record_error(exc)

    def _write(self, kind: str, payload: dict, ts_event: int) -> None:
        self._journal.write(kind, {"ts_event": ts_event, "ts_init": self.clock.now_ms(), **payload})

    def _record_error(self, exc: BaseException) -> None:
        self.handler_errors.append(exc)
        del self.handler_errors[:-100]

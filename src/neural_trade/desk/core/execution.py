"""ExecutionEngine (PLAN A2 ``execution.py``, A4, A5; DESK-1.3): commands -> RiskGuard -> Broker, events -> state
machine -> Portfolio -> bus.

One engine is one account (one ``Portfolio``, one ``RiskGuard``, one ``Broker``). Pure logic, standard library only:
the database is not imported; the journal is an injected sink, any callable ``(kind, payload) -> None`` (the
``Journal.append`` of C1 fits).

Order status changes happen ONLY in ``apply`` (``submit`` and ``cancel`` build their own events and apply them).
The legal edges are ``model.STATUS_EDGES`` plus two exceptions: ``late_fill`` (A4) and the refused fill below.

**Commitments (NT-295).** ``submit`` and ``modify`` hand RiskGuard what the engine's live OPENING orders already commit:
``committed_open_notional`` (account total of notional x (1 + fee + slip), maker fee for LIMIT, taker for the rest; cash
and gross cap) and ``committed_position_notional`` (plain notional of the same strategy and instrument; max position).
The part of a resting order that can still reduce the held position is not committed; dead orders and the order itself
(modify) count for nothing.

**The binding check at the fill (CONTRACTS A5.1).** A broker calls ``precheck_fill(client_id, mid)`` before it reports a
fill; ``apply`` runs the same check on every fill. A fill that only reduces the held position (opposite side, not larger
than it; ``reduce_only`` alone is not enough) is never re-checked or refused. Every other fill is re-checked by RiskGuard
as a MARKET order at ``mid`` (``record=False``, no rate slot, own commitments excluded, the order's earlier fills added to
``max_order_notional``) when the order's cumulative reference price is above the price it was valued at
(``FILL_PRICE_TOLERANCE`` relative), and always for a ``size_frac`` or ``reduce_only`` order, an order that was opposite
to its position at submit (RiskGuard valued part of it as a reduction), a late fill, or while the account is HALTED or
REDUCING. A ``size_frac`` fill (NT-299, A5.1) is sized at the fill on ``Portfolio.realised_equity()`` (equity x frac /
mid), clipped by cash and the re-check; the quantity reserved at submit serves the commitments only; a fill above that
sizing (``SIZE_FRAC_TOLERANCE`` relative) has its excess refused. An absolute bracket level a gap put on the wrong side
of the fill price does not refuse the entry (it was validated at submit; the gap rule exits at the open). Denied: a fill
opposite to the position keeps its reducing part (clipped); otherwise ``fill_refused`` is journalled, the order ends
REJECTED (CANCELED if part of it was filled) with reason ``fill_price_check_failed`` and the broker must not fill it.
Clipped (PAPER / BACKTEST): only the allowed quantity is applied (``precheck_fill`` returns it and journals the refused
excess once per order), the fee is scaled to the applied part, ``fill_refused`` records the quantities and the order ends
FILLED or CANCELED with what was applied. A simulated fill the Portfolio cannot hold (spot balance) is refused the same
way (check ``books_refused``). On a LIVE account the venue fact stands: the fill is applied in full, journalled
``fill_refused`` (``applied`` true) and RiskGuard is halted; a venue fill for an order the engine marked REJECTED is
booked as ``venue_fill_on_rejected`` (the order stays REJECTED) and halts too; a LIVE fill the engine cannot book at all
(no spot balance, a FILLED / EXPIRED / SUBMITTED / DENIED order, an unknown client_id, ...) is journalled
``venue_fill_unbooked`` with the fill and an order snapshot, RiskGuard is halted, and only then is the error raised. A
LIVE ``size_frac`` accept must carry ``resolved_qty``: without it the order is REJECTED (``resolved_qty_missing``),
RiskGuard halted and the venue asked to cancel it (a broker error is journalled ``cancel_failed``). A fill above a
resolved ``size_frac`` order's ``resolved_qty`` has its excess refused on PAPER / BACKTEST (``fill_refused``, check
``resolved_qty_exceeded``, the applied part booked) and is booked in full on LIVE (``fill_refused`` applied, halt); a
LIVE fill on a REJECTED order is booked only up to the order's own (or resolved) quantity, else ``venue_fill_unbooked``.

**Handlers (re-entrancy).** A bus handler that calls a mutating method (``submit``, ``apply``, ``cancel``, ...) while the
engine publishes does not run at once: the call is queued, returns ``None`` and runs, in order, when the outermost
mutating call returns, so the journal order of a fill is never interleaved with another's. Its errors go to
``handler_errors``. ``precheck_fill`` is refused inside a handler (it returns a decision). At most
``HANDLER_QUEUE_LIMIT`` (1000) calls may wait, and one drain runs at most that many: beyond either the queue is dropped,
``handler_queue_overflow`` is journalled and RiskGuard halted.

**Journal first, then the bus.** Every record is written to the journal before its event is published. A subscriber
that raises never propagates and never loses a later record: the bus isolates it (``bus.errors``, ``on_error``; NT-294);
a ``strict`` bus re-raises, and the engine then logs it and appends it to ``handler_errors`` (the last 100). A journal
sink that raises does not undo the state change (the venue fact is true): the failure is counted in
``journal_failures`` and recorded in ``handler_errors``, the record stays in a pending queue (``JournalWriter``) and is
written, in order, before the next record. Nothing is ever dropped. Above ``JOURNAL_PENDING_LIMIT`` (1000) pending
records the engine calls ``RiskGuard.halt("journal_unavailable")`` (new orders are denied; exits go through the
force_reduce path of NT-272 when it exists), journals that halt (queued behind the others), and re-asserts it on every
failed write while the queue is above the limit; a raising halt hook never interrupts the write. The ERROR line is
once per outage; sink failures are logged on the first and then every 1000th failure.

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
  ``bracket``, ``halt``, ``cancel_failed``, ``halt_cancel_failed``, ``order_modify_failed``, ``fill_refused``,
  ``fill_after_refusal``, ``venue_fill_on_rejected``, ``venue_fill_unbooked``, ``handler_queue_overflow``,
  ``late_fill_refused``, ``late_fill_flip``; the journal payload carries ``ts_event`` and ``ts_init`` itself. Every fill
  of a dead order is ``late_fill``. A late fill the books cannot hold (spot balance) is journalled
  ``late_fill_refused`` and raises ``OverfillError``; one that flips a position is applied, journalled
  ``late_fill_flip``; both halt RiskGuard;
* ``OrderEvent.kind`` from a broker: accepted, rejected, triggered, canceled, expired, modified; the engine emits
  submitted, denied, partially_filled, filled and late_fill itself. A broker that raises in ``submit`` gives a REJECTED
  order with reason ``broker_error:<ExceptionName>``; one that raises in ``cancel`` during ``cancel_all`` / ``halt``
  is journalled per order and the loop goes on (``last_cancel_failures``); one that raises in a single ``cancel`` is
  journalled ``cancel_failed`` and re-raised as ``BrokerError``, in ``modify`` ``order_modify_failed`` and the same;
* a ``size_frac`` order has no known total quantity: its first Fill is terminal (FILLED) and the by-quantity overfill
  check does not apply, unless the broker's ``accepted`` event carries ``OrderEvent.resolved_qty`` (mandatory on LIVE,
  NT-299): then partial fills are allowed up to it (``order.qty`` stays None);
* ``modify`` is allowed on ACCEPTED and PARTIALLY_FILLED orders that are not dead (A4 has no modify edge; TRIGGERED is
  already live at the venue). The modified order is re-checked by RiskGuard (``record=False``: a modify takes no rate
  slot; committed reduce quantity excludes the order itself); price and trigger are rounded to the tick; a denial
  leaves the order unchanged and is journalled; a broker ``modified`` event below the filled quantity is refused;
* a user's explicit close goes through RiskGuard like any order (the force_reduce path of NT-272 does not exist yet);
* an offset bracket (``tp_is_offset``) is stored verbatim; resolving it against the fill bar's open is the Engine's /
  SimBroker's job (they hold the bar);
* not handled here (NT-299 notes): several strategies' ``size_frac`` entries filled at the same open are each sized on
  the realised equity before any of them, while the engine bounds each on the equity after the earlier ones were applied
  (with costs the later ones are clipped by the fee): the Engine's A5.9 loop decides the order (DESK-1.7); a LIVE broker
  should never send a ``size_frac`` order to the venue but resolve it to a quantity first (DESK-6.1), which makes the
  ``resolved_qty_missing`` rejection a backstop.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import math
import threading
from collections import deque
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Union

from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.model import (
    CLOSED_STATUSES, AccountEvent, AccountKind, CancelAll, CancelOrder, ClosePosition, DeskError, Fill, FillEvent, Halt,
    Liquidity, ModifyOrder, Order, OrderEvent, OrderStatus, OrderType, OverfillError, Side, SubmitOrder, check_transition,
)
from neural_trade.desk.core.portfolio import DUST, Portfolio
from neural_trade.desk.core.risk import (
    CLIPPED_TO_POSITION, INVALID_ORDER, MISSING_MARKS, RiskDecision, RiskGuard, TradingState, round_nearest,
)

log = logging.getLogger(__name__)

JOURNAL_PENDING_LIMIT = 1000  # pending journal records above which new orders are halted
JOURNAL_LOG_EVERY = 1000  # sink failures are logged on the first and then every 1000th
JOURNAL_HALT_REASON = "journal_unavailable"
FILL_REFUSED = "fill_price_check_failed"  # reason of an order whose fill failed the binding check (CONTRACTS A5.1)
HANDLER_QUEUE_LIMIT = 1000  # calls a bus handler may queue (and the calls one drain may run)
FILL_PRICE_TOLERANCE = 1e-6  # relative: a fill reference price above the checked price by more than this is re-checked
SIZE_FRAC_TOLERANCE = 1e-9  # relative: a size_frac fill above its A5.1 equity sizing by more than this is clipped
RESOLVED_QTY_MISSING = "resolved_qty_missing"  # reason of a LIVE size_frac accept without resolved_qty (rejected)
RESOLVED_QTY_EXCEEDED = "resolved_qty_exceeded"  # fill_refused check: a fill above a size_frac order's resolved_qty
VENUE_FILL_UNBOOKED = "venue_fill_unbooked"  # journal kind and halt reason of a LIVE fill the engine cannot book

JournalSink = Callable[[str, dict], None]
Command = Union[SubmitOrder, ModifyOrder, CancelOrder, CancelAll, ClosePosition, Halt]

_S = OrderStatus
RESTING = frozenset({_S.ACCEPTED, _S.TRIGGERED, _S.PARTIALLY_FILLED})  # can be cancelled
MODIFIABLE = frozenset({_S.ACCEPTED, _S.PARTIALLY_FILLED})
_LIVE = frozenset({_S.SUBMITTED, *RESTING})  # sent to the broker and not closed
_KIND_STATUS = {"accepted": _S.ACCEPTED, "rejected": _S.REJECTED, "triggered": _S.TRIGGERED,
                "canceled": _S.CANCELED, "expired": _S.EXPIRED}
_IDENTITY = ("instrument_id", "strategy_id", "side", "type", "qty", "size_frac", "price", "trigger_price",
             "reduce_only", "bracket", "tif", "bot_id", "account_id")


class UnknownOrder(DeskError):
    pass


class UnknownAccount(DeskError):
    pass


class DuplicateClientId(ValueError):
    pass


class BrokerError(RuntimeError):
    """A broker call failed; ``__cause__`` is the broker's exception."""


def _mutating(fn: Callable) -> Callable:
    """A public mutating method of the engine. Called from a bus handler (while the engine is publishing) it is queued
    and runs, in call order, after the outermost mutating call returns, so a handler cannot reorder the journal
    (fill b written between the records of fill a). The queued call returns ``None``; its exceptions are logged and
    kept in ``handler_errors`` (the caller is long gone)."""

    @functools.wraps(fn)
    def wrapper(self: "ExecutionEngine", *args: Any, **kwargs: Any) -> Any:
        self.assert_owner()
        q = self._handlers
        if q.publishing:
            q.defer(fn, args, kwargs)
            return None
        return q.run(self, fn, args, kwargs)

    return wrapper


class HandlerQueue:
    """The calls bus handlers make while the engine publishes (module docstring "Handlers"). ``publishing`` is the depth
    of ``bus.publish`` calls and ``ops`` the depth of the engine's public mutating calls. A call made while publishing is
    queued (``defer``); ``drain`` runs the queue, in call order, when the outermost mutating call returns. At most
    ``limit`` calls may wait and one drain runs at most ``limit``: beyond either the queue is dropped and closed until
    the drain ends and ``on_overflow(name)`` runs (the engine journals ``handler_queue_overflow`` and halts); a refused
    ``defer`` raises ``RuntimeError`` in the handler. A queued call's exception goes to ``on_error``."""

    def __init__(self, *, on_overflow: Callable[[str], None], on_error: Callable[[BaseException], None],
                 limit: int = HANDLER_QUEUE_LIMIT) -> None:
        self._on_overflow, self._on_error, self.limit = on_overflow, on_error, limit
        self.calls: deque[tuple[Callable, tuple, dict]] = deque()
        self.publishing = 0  # depth of bus.publish calls: a mutation requested then is queued (``_mutating``)
        self.ops = 0  # depth of public mutating calls
        self.draining = False
        self.closed = False  # a drain overflowed: handlers may queue nothing more until it ends

    def defer(self, fn: Callable, args: tuple, kwargs: dict) -> None:
        if self.closed or len(self.calls) >= self.limit:
            self.overflow(fn.__name__)
            raise RuntimeError("handler call queue overflow")
        self.calls.append((fn, args, kwargs))

    def run(self, target: Any, fn: Callable, args: tuple, kwargs: dict) -> Any:
        """Run ``fn`` as a mutating call now; the queue drains when the outermost one returns."""
        self.ops += 1
        try:
            return fn(target, *args, **kwargs)
        finally:
            self.ops -= 1
            if self.ops == 0:
                self.drain(target)

    def overflow(self, name: str) -> None:
        if self.closed:
            return
        self.closed = True
        self.calls.clear()
        self._on_overflow(name)

    def drain(self, target: Any) -> None:
        """Run the queued calls, in order, once the outermost mutating call has returned."""
        if self.draining:
            return
        self.draining = True
        try:
            n = 0
            while self.calls:
                n += 1
                if n > self.limit and not self.closed:  # handlers keep feeding the queue
                    self.overflow("drain")
                    continue
                fn, args, kwargs = self.calls.popleft()
                self.ops += 1
                try:
                    fn(target, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - the caller returned long ago
                    log.error("queued %s failed: %r", fn.__name__, exc, exc_info=exc)
                    self._on_error(exc)
                finally:
                    self.ops -= 1
        finally:
            self.draining = False
            self.closed = False


class JournalWriter:
    """Writes journal records in order through a sink that may fail. A failed record stays at the head of the queue
    and is retried before the next one; nothing is dropped. Above ``limit`` pending records ``on_overflow`` is called
    on every failed write (the halt is re-asserted); the ERROR log line is once per outage (the queue drained resets
    it)."""

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
                if len(self._pending) > self.limit:
                    if not self.overflowed:
                        self.overflowed = True
                        log.error("journal has %d pending records (limit %d): halting new orders",
                                  len(self._pending), self.limit)
                    try:  # re-asserted on every failure above the limit (an operator resume during the outage is undone)
                        self._on_overflow()
                    except Exception as halt_exc:  # noqa: BLE001 - a fill is never left half-written by the halt hook
                        log.error("overflow halt hook failed: %r", halt_exc, exc_info=halt_exc)
                        self._on_error(halt_exc)
                return
            self._pending.popleft()
        self.overflowed = False

    def enqueue(self, kind: str, record: dict) -> None:
        """Append a record without trying the sink (used from ``on_overflow``, which runs inside a failing write)."""
        self._pending.append((kind, record))


class Broker(Protocol):
    """C1 ``desk/broker/base.py`` (the real module arrives with DESK-1.4); only what the engine calls."""

    def submit(self, order: Order) -> OrderEvent: ...
    def cancel(self, client_id: str) -> OrderEvent: ...
    def modify(self, client_id: str, *, price: Optional[float] = None, trigger_price: Optional[float] = None,
               qty: Optional[float] = None) -> OrderEvent: ...  # returns kind "modified" with the new values


class FillGate:
    """The binding check at the fill (module docstring; CONTRACTS A5.1) and the commitments RiskGuard sees, for one
    ``ExecutionEngine``. It owns the per-order state the check needs (the checked price, the refused orders, a
    ``size_frac`` order's sized quantity, the fill limit a clipped precheck set, the cumulative fill notional) and reads
    the engine's orders, Portfolio, RiskGuard and marks; it writes only through the engine's journal and order-event
    path."""

    def __init__(self, engine: "ExecutionEngine") -> None:
        self.e = engine
        self.checked: dict[str, float] = {}  # the price RiskGuard valued each order at (submit / last modify)
        self.refused: set[str] = set()  # orders whose fill the binding check refused
        self.sized: dict[str, float] = {}  # size_frac orders: the approved / resolved quantity (kept when closed)
        self.limit: dict[str, float] = {}  # filled quantity a clipped precheck allows an order to reach
        self.notional: dict[str, float] = {}  # sum of qty x mid of each order's applied fills (cumulative check)
        self.opposite: set[str] = set()  # orders opposite to their position at submit / modify: always re-checked

    # ------------------------------------------------------------------ reducing and commitments
    def held(self, o: Order) -> float:
        p = self.e.portfolio
        return p.signed_qty(p.position(o.instrument_id, o.strategy_id))

    def reducing(self, o: Order, qty: float) -> bool:
        """True when filling ``qty`` of ``o`` can only reduce its position (opposite to the holding and not larger than
        it): such a fill opens no exposure, so the binding check and a halt never refuse it. ``reduce_only`` alone is
        not enough (B): a reduce-only order whose position is gone, or a fill beyond the holding, is checked like an
        opening fill."""
        held = self.held(o)
        return held * (1 if o.side is Side.BUY else -1) < 0 and qty <= abs(held) + DUST

    def note_opposite(self, o: Order) -> None:
        """Remember an order opposite to its position when RiskGuard valued it (submit, modify): RiskGuard counted part
        of it as a reduction, so a later fill that no longer reduces (the position went away another way) is always
        re-checked, the price shortcut of ACTIVE included (NT-299: probes C1a, C2)."""
        if self.held(o) * (1 if o.side is Side.BUY else -1) < 0:
            self.opposite.add(o.client_id)

    def reducing_part(self, o: Order, qty: float) -> float:
        """The part of a fill of ``qty`` of ``o`` that reduces the held position: ``min(qty, |held|)`` when ``o`` is
        opposite to the holding, else 0."""
        held = self.held(o)
        return min(qty, abs(held)) if held * (1 if o.side is Side.BUY else -1) < 0 else 0.0

    @staticmethod
    def valuation_price(o: Order, ref: float) -> float:
        """The price RiskGuard values the order at (its notional rule): a market order ``ref``, a limit its price,
        a stop buy ``max(trigger, limit, ref)``, a stop sell its trigger. Brackets are not included."""
        px, trig = o.price, o.trigger_price
        if trig is None:
            return px if px is not None else ref
        return max(trig, px or 0.0, ref) if o.side is Side.BUY else trig

    def committed_reduce(self, new: Order, exclude: Optional[str] = None) -> float:
        """Quantity of the new order's position claimed by resting reduce orders (A5, RiskGuard): of each live (not
        dead) opposite-side order of the same instrument and strategy only the part that can reduce the position counts,
        min(its remaining quantity, the position), and the sum is capped at the position."""
        e = self.e
        held = self.held(new)
        cap = abs(held)
        if cap == 0:
            return 0.0
        opposite = Side.SELL if held > 0 else Side.BUY
        total = sum(min(max(e._reserved.get(o.client_id, 0.0) - o.filled_qty, 0.0), cap)
                    for o in e._live_orders()
                    if e.is_live(o) and o.side is opposite and o.instrument_id == new.instrument_id
                    and o.strategy_id == new.strategy_id and o.client_id != exclude)
        return min(total, cap)

    def committed_open(self, new: Order, exclude: Optional[str] = None) -> tuple[float, float]:
        """The notional of the live OPENING orders for the next RiskGuard check, as ``(account, position)``:

        * account: every live opening order's quote notional x ``(1 + fee + slip)`` with the account's cost profile
          (the maker fee for LIMIT, the taker fee for market and stop orders), so the fees and slip of resting orders
          are reserved too (``committed_open_notional``: cash and gross cap);
        * position: the plain notional of those of the same strategy and instrument as ``new``
          (``committed_position_notional``: max position).

        Opening part of an order = its unfilled reserved quantity minus what can still reduce the held position
        (orders in submission order take the position's quantity first), valued at ``valuation_price`` with the current
        mark. Dead orders (cancel acknowledged), closed orders, ``new`` itself and ``exclude`` count for nothing."""
        e = self.e
        costs = e.risk.costs
        left: dict[tuple[str, Optional[str]], float] = {}
        acct = posn = 0.0
        for o in e._live_orders():
            if o is new or o.client_id == exclude or o.reduce_only or not e.is_live(o):
                continue
            rem = max(e._reserved.get(o.client_id, 0.0) - o.filled_qty, 0.0)
            key = (o.instrument_id, o.strategy_id)
            held = e.portfolio.signed_qty(e.portfolio.position(*key))
            if key not in left:
                left[key] = abs(held)
            if (held > 0 and o.side is Side.SELL) or (held < 0 and o.side is Side.BUY):
                take = min(rem, left[key])
                left[key] -= take
                rem -= take
            ref = e._ref_price(o)
            if rem <= DUST or ref is None:
                continue
            notional = rem * self.valuation_price(o, ref)
            liq = Liquidity.MAKER if o.type is OrderType.LIMIT else Liquidity.TAKER
            acct += notional * (1 + costs.fee_rate(liq) + costs.slip_rate)
            if key == (new.instrument_id, new.strategy_id):
                posn += notional
        return acct, posn

    # ------------------------------------------------------------------ the check
    def precheck(self, o: Order, fill_price: float, qty: Optional[float]) -> RiskDecision:
        """``ExecutionEngine.precheck_fill`` after its owner / handler / LIVE guards."""
        e, cid = self.e, o.client_id
        if cid in self.refused or not e.is_live(o):
            return RiskDecision(False, "order_not_live")
        d = self.decision(o, fill_price, qty)
        want = qty if qty is not None else max((o.qty or 0.0) - o.filled_qty, 0.0)
        clipped = d.allowed and o.size_frac is None and d.qty is not None and d.qty < want - DUST
        old = self.limit.get(cid)
        new = o.filled_qty + d.qty if clipped else None
        if clipped and old is not None and new >= old - DUST:
            return d  # the excess beyond this limit was journalled by an earlier precheck: once per order (NT-299)
        if (not d.allowed and d.reason != "order_not_live") or clipped:
            q = e._handlers
            q.ops += 1
            try:
                if clipped:  # (A) the excess is refused now; the order ends once the allowed part is filled
                    self.limit[cid] = new  # the filled quantity counts: the limit is on the order's total
                    asked = want if old is None else old - o.filled_qty  # a tighter clip refuses only the new part
                    self.write_refused(o, fill_price, d.reason or CLIPPED_TO_POSITION, self.checked.get(cid, 0.0),
                                       applied=False, dead=False, qty=(asked, d.qty))
                else:
                    self.refuse(o, fill_price, d.reason, self.checked.get(cid, 0.0))
            finally:
                q.ops -= 1
                if q.ops == 0:
                    q.drain(e)
        return d

    def equity_sized(self, o: Order, mid: float) -> float:
        """A5.1: the quantity of a ``size_frac`` fill at the reference price ``mid``, clip(size_frac, 0, 1) x
        ``Portfolio.realised_equity()`` / mid (the SimBroker's and RiskGuard's arithmetic, so equal in float)."""
        return min(max(o.size_frac or 0.0, 0.0), 1.0) * self.e.portfolio.realised_equity() / mid

    def decision(self, o: Order, mid: float, qty: Optional[float], *, always: bool = False) -> RiskDecision:
        """RiskGuard on the fill itself (see ``precheck_fill``). ``qty`` None: the unfilled quantity (an explicit
        order), the resolved one (a ``size_frac`` order whose accept carried ``resolved_qty``) or, for any other
        ``size_frac`` order, the A5.1 quantity sized at ``mid`` on realised equity (NT-299: never the quantity reserved
        at submit, which serves the commitments only). A ``size_frac`` fill above that equity sizing (relative tolerance
        ``SIZE_FRAC_TOLERANCE``) or above its resolved quantity, and a fill above the limit a clipped precheck set, are
        capped first (the decision then carries less than ``qty``). Reducing fills pass. The check runs when the order's
        cumulative reference price (earlier fills and this one) is above the checked price by more than the tolerance;
        always for a ``size_frac`` order, a ``reduce_only`` order (a reduce-only fill that does not reduce must be
        refused), an order that was opposite to its position at submit (part of it was valued as a reduction, NT-299),
        while the account is not ACTIVE (C), or when ``always``. A denied fill that is opposite to the held position
        keeps its reducing part (a clipped decision): reductions are never refused."""
        e, cid = self.e, o.client_id
        sf = o.size_frac is not None
        resolved = e._resolved.get(cid)
        by_equity = sf and resolved is None  # sized at the fill on realised equity (A5.1)
        if by_equity:
            rem = self.equity_sized(o, mid)
            cap = rem * (1 + SIZE_FRAC_TOLERANCE)
        else:
            rem = max((resolved if sf else (o.qty or 0.0)) - o.filled_qty, 0.0)
            cap = rem if sf else math.inf  # a resolved size_frac order fills at most what the venue resolved
        limit = self.limit.get(cid)
        if limit is not None:
            cap = min(cap, max(limit - o.filled_qty, 0.0))
            rem = min(rem, cap)
        q = rem if qty is None or qty > cap else qty  # above the cap: at most the remainder (rem <= cap)
        if q <= DUST:
            return RiskDecision(False, "order_not_live")
        if self.reducing(o, q):
            return RiskDecision(True, "", q)
        checked = self.checked.get(cid)
        prior = self.notional.get(cid, 0.0)
        above = checked is not None and prior + q * mid > (o.filled_qty + q) * checked * (1 + FILL_PRICE_TOLERANCE)
        # ``o.reduce_only`` is implied by ``opposite`` today (RiskGuard denies a reduce-only order that does not reduce,
        # so every accepted one was opposite at submit: QA mutant Q13 is equivalent); it stays for the broker-made
        # reduce-only exits NT-296 adopts without a submit
        if not (always or sf or o.reduce_only or above or cid in self.opposite
                or e.risk.state is not TradingState.ACTIVE):
            return RiskDecision(True, "", q)
        # a MARKET order at the fill price, the order's own commitments excluded, no rate slot; its earlier fills
        # count against max_order_notional (E). An equity-sized fill is checked as the size_frac order RiskGuard sizes
        # at ``mid`` (cash clip included) and the fill gets at most that; a smaller fill denied that way (a limit the
        # full sizing breaks) is checked on its own quantity.
        d = self.check_at(o, mid, None if by_equity else q, prior)
        if by_equity:
            if d.allowed:  # RiskGuard's full sizing trusts the fill (up to the tolerance); a cash clip binds
                d = dataclasses.replace(d, qty=q if d.qty >= rem * (1 - SIZE_FRAC_TOLERANCE) else min(d.qty, q))
            elif q < rem * (1 - SIZE_FRAC_TOLERANCE):
                d = self.check_at(o, mid, q, prior)
        if not d.allowed:
            part = self.reducing_part(o, q)
            if part > DUST:
                d = RiskDecision(True, CLIPPED_TO_POSITION, part, d.price)
        return d

    def check_at(self, o: Order, mid: float, qty: Optional[float], prior: float) -> RiskDecision:
        """RiskGuard.check of ``o`` as a MARKET order at ``mid``: ``qty`` None checks it as its ``size_frac``."""
        e, cid = self.e, o.client_id
        cand = dataclasses.replace(o, tags=dict(o.tags), type=OrderType.MARKET, price=None, trigger_price=None,
                                   qty=qty, size_frac=o.size_frac if qty is None else None)
        marks = {**e.marks}
        marks[o.instrument_id] = mid
        acct, posn = self.committed_open(cand, exclude=cid)
        kw = dict(ref_price=mid, marks=marks, record=False, fill_recheck=True,
                  committed_reduce_qty=self.committed_reduce(cand, exclude=cid),
                  committed_open_notional=acct, committed_position_notional=posn, prior_order_notional=prior)
        d = e.risk.check(cand, **kw)
        if not d.allowed and d.reason == INVALID_ORDER and cand.bracket:
            # a gap through an absolute bracket level puts it on the wrong side of the fill price: the bracket was
            # validated at submit and the gap rule exits at the open (A5.2, A5.3, the oracle); the fill is judged
            # without that leg only, the valid legs keep valuing it (NT-299: nt225_parity seeds 0, 2, 3; QA B5)
            # (retrying on any other denial changes nothing: a wrong-side leg is the only bracket-dependent denial,
            # and the denials RiskGuard returns before it (halted, instrument, marks) do not depend on the legs; QA
            # mutant K9 is equivalent)
            kept = self.right_side_legs(cand, mid)
            if kept != cand.bracket:
                d = e.risk.check(dataclasses.replace(cand, bracket=kept), **kw)
        return d

    @staticmethod
    def right_side_legs(o: Order, mid: float) -> dict:
        """``o.bracket`` without the absolute legs on the wrong side of ``mid`` (RiskGuard's side rule: a BUY's take
        profit above the entry and its stop below, reversed for a SELL). Offset legs and malformed legs are kept (an
        invalid one still denies the order)."""
        br = dict(o.bracket or {})
        if br.get("tp_is_offset"):
            return br
        buy = o.side is Side.BUY
        for key, up in (("take_profit", buy), ("stop_loss", not buy)):
            v = br.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 \
                    and (v <= mid if up else v >= mid):
                br[key] = None
        return br

    # ------------------------------------------------------------------ refusals
    def write_refused(self, o: Order, mid: float, check_reason: str, checked: float, *, applied: bool, dead: bool,
                      qty: Optional[tuple[float, float]]) -> None:
        rec = {"client_id": o.client_id, "reason": FILL_REFUSED, "check": check_reason, "fill_price": mid,
               "checked_price": checked, "applied": applied, "dead": dead, "filled_qty": o.filled_qty}
        if qty is not None:  # a clipped fill: the quantity asked, the part allowed, the part refused
            rec.update(fill_qty=qty[0], allowed_qty=qty[1], refused_qty=qty[0] - qty[1])
        self.e._write("fill_refused", rec, self.e.clock.now_ms())

    def refuse(self, o: Order, mid: float, check_reason: str, checked: float, *, applied: bool = False,
               dead: bool = False, qty: Optional[tuple[float, float]] = None) -> None:
        """Journal ``fill_refused``; unless ``applied`` (a LIVE venue fact) the order is refused for good: ended
        REJECTED (nothing filled) or CANCELED (part filled), with reason ``fill_price_check_failed``. ``qty`` is
        (asked, allowed) for a clipped fill; a FILLED or dead order only records it."""
        self.write_refused(o, mid, check_reason, checked, applied=applied, dead=dead, qty=qty)
        if applied:
            return
        self.refused.add(o.client_id)
        if dead or o.status is _S.FILLED:  # a late fill of a cancelled order, or nothing left: the order is over
            return
        self.end_refused(o)

    def end_refused(self, o: Order) -> None:
        e = self.e
        now = e.clock.now_ms()
        self.refused.add(o.client_id)
        self.limit.pop(o.client_id, None)
        kind = "rejected" if o.filled_qty <= DUST else "canceled"  # PARTIALLY_FILLED -> REJECTED is no edge
        e._apply_order_event(OrderEvent(now, now, kind, e._with_reason(o, FILL_REFUSED)), fill_refused=True)


class ExecutionEngine:
    def __init__(self, broker: Broker, portfolio: Portfolio, risk: RiskGuard, bus: EventBus, clock: Clock,
                 journal: JournalSink, *, check_owner: bool = True) -> None:
        self.broker, self.portfolio, self.risk, self.bus, self.clock = broker, portfolio, risk, bus, clock
        self._journal = JournalWriter(journal, on_error=self._record_error, on_overflow=self._journal_overflow)
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
        self._handlers = HandlerQueue(on_overflow=self._on_queue_overflow, on_error=self._record_error)
        self._gate = FillGate(self)
        # the gate's state under the engine's names (the same objects; never reassigned)
        self._checked, self._refused, self._sized = self._gate.checked, self._gate.refused, self._gate.sized
        self._fill_limit, self._fill_notional = self._gate.limit, self._gate.notional
        self._live: dict[str, None] = {}  # ids of the live orders (``is_live``), in submission order
        self._cancel_queued = False
        self._unbooked_journalled = False  # the current LIVE fill's failure is already journalled (late_fill_refused)

    # the handler queue's state under the engine's names (tests and probes read them)
    @property
    def _deferred(self) -> deque:
        return self._handlers.calls

    @property
    def _publishing(self) -> int:
        return self._handlers.publishing

    @property
    def _draining(self) -> bool:
        return self._handlers.draining

    @property
    def _queue_closed(self) -> bool:
        return self._handlers.closed

    @property
    def _ops(self) -> int:
        return self._handlers.ops

    @_ops.setter
    def _ops(self, value: int) -> None:
        self._handlers.ops = value

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
        return [o for o in self._live_orders() if self.is_live(o)
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

    @_mutating
    def submit(self, order: Order) -> Optional[Order]:
        """RiskGuard first; a denial is a DENIED order (reason + journal record) and never reaches the broker.
        Idempotent on ``client_id``: the same content returns the existing order, different content raises
        ``DuplicateClientId`` naming the differing fields. RiskGuard sees the notional the engine's live opening orders
        already commit (``_committed_open``). Called from a bus handler it is queued and returns ``None``."""
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
            acct, posn = self._committed_open(o)
            d = self.risk.check(o, ref_price=ref, marks=marks, committed_reduce_qty=self._committed_reduce(o),
                                committed_open_notional=acct, committed_position_notional=posn)
            decision_reason, qty, price = ("" if d.allowed else d.reason), d.qty, d.price
        if decision_reason:
            self.apply(OrderEvent(now, now, "denied", self._with_reason(o, decision_reason)))
            return o
        if o.qty is not None:
            o.qty = qty  # rounded / clipped by the risk check; a size_frac order keeps its fraction
        if price is not None:
            o.price = price
        if o.trigger_price is not None:  # the tick, as modify does (the venue would reject or round it otherwise)
            o.trigger_price = round_nearest(o.trigger_price, self.risk.instrument.price_increment)
        self._reserved[o.client_id] = qty
        if o.size_frac is not None:
            self._sized[o.client_id] = qty
        self._checked[o.client_id] = self._gate.valuation_price(o, ref)
        self._gate.note_opposite(o)
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

    @_mutating
    def cancel(self, client_id: str) -> Order:
        """Cancel one order. A broker exception is journalled ``cancel_failed``, leaves the order unchanged and raises
        ``BrokerError`` (``__cause__`` is the broker's exception), like ``modify``."""
        return self._cancel(client_id, "cancel_failed")

    def _cancel(self, client_id: str, fail_kind: str) -> Order:
        o = self._get(client_id)
        if o.is_closed or o.client_id in self.cancel_acked:  # a dead order is never cancelled twice
            return o
        if o.status not in RESTING:  # SUBMITTED has no cancel edge (A4)
            check_transition(o.status, _S.CANCELED)
        try:
            ev = self.broker.cancel(client_id)
        except Exception as exc:  # noqa: BLE001 - journalled, then raised to the caller
            log.error("broker.cancel failed for %s: %r", client_id, exc, exc_info=exc)
            self._record_error(exc)
            self._write(fail_kind, {"client_id": client_id, "reason": f"broker_error:{type(exc).__name__}",
                                    "error": repr(exc)}, self.clock.now_ms())
            raise BrokerError(f"broker.cancel failed for {client_id}: {exc!r}") from exc
        self.apply(ev)
        return o

    @_mutating
    def cancel_all(self, account_id: Optional[str] = None, instrument_id: Optional[str] = None) -> list[Order]:
        """Cancel every resting order of the account (and instrument), in submission order; returns the cancelled
        ones. An order the venue already cancelled once (and that only came back through a late fill) is not
        cancelled again. A cancel that raises is journalled ``cancel_failed``, recorded in ``handler_errors`` and
        ``last_cancel_failures`` (client_id, exception), and the loop goes on."""
        if account_id is not None and account_id != self.portfolio.account_id:
            raise UnknownAccount(account_id)
        return self._cancel_each(self._resting(instrument_id), "cancel_failed")

    def _cancel_each(self, orders: list[Order], fail_kind: str, *, reset: bool = True) -> list[Order]:
        done: list[Order] = []
        if reset:
            self.last_cancel_failures = []
        for o in orders:
            try:
                done.append(self._cancel(o.client_id, fail_kind))  # a broker failure is journalled by _cancel
            except BrokerError as exc:
                self.last_cancel_failures.append((o.client_id, exc.__cause__ or exc))
            except Exception as exc:  # noqa: BLE001 - one failing cancel must not stop the others
                log.error("cancel of %s failed: %r", o.client_id, exc, exc_info=exc)
                self._record_error(exc)
                self.last_cancel_failures.append((o.client_id, exc))
                self._write(fail_kind, {"client_id": o.client_id, "reason": f"broker_error:{type(exc).__name__}",
                                        "error": repr(exc)}, self.clock.now_ms())
        return done

    def _resting(self, instrument_id: Optional[str] = None) -> list[Order]:
        return [o for o in self._live_orders() if o.status in RESTING and o.client_id not in self.cancel_acked
                and (instrument_id is None or o.instrument_id == instrument_id)]

    @_mutating
    def modify(self, client_id: str, *, price: Optional[float] = None, trigger_price: Optional[float] = None,
               qty: Optional[float] = None) -> Order:
        """Change price, trigger or total quantity of a resting order. RiskGuard re-checks it; a denial leaves the
        order unchanged and is journalled as ``order_modify_denied``. Price and trigger are rounded to the tick. A
        broker exception is journalled ``order_modify_failed``, leaves the order unchanged and raises
        ``BrokerError``."""
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
            acct, posn = self._committed_open(cand, exclude=client_id)
            d = self.risk.check(cand, ref_price=ref, marks=marks, record=False,
                                committed_reduce_qty=self._committed_reduce(cand, exclude=client_id),
                                committed_open_notional=acct, committed_position_notional=posn,
                                prior_order_notional=self._fill_notional.get(client_id, 0.0))
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
        self._gate.note_opposite(o)
        self.apply(ev)
        return o

    @_mutating
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

    @_mutating
    def halt(self, reason: str) -> None:
        """RiskGuard to HALTED (denies every new order), then cancel what rests. A failing cancel is logged and
        journalled; it never stops the others."""
        self.risk.halt(reason)
        self._write("halt", {"reason": reason}, self.clock.now_ms())
        self._cancel_each(self._resting(), "halt_cancel_failed")

    # ------------------------------------------------------------------ events: the only place status changes
    @_mutating
    def apply_all(self, events: Iterable[Union[Fill, FillEvent, OrderEvent]]) -> None:
        for ev in list(events):
            self.apply(ev)

    @_mutating
    def apply(self, event: Union[Fill, FillEvent, OrderEvent]) -> None:
        if isinstance(event, Fill):
            event = FillEvent(event.ts, self.clock.now_ms(), event)
        if isinstance(event, FillEvent):
            self._apply_fill(event)
        elif isinstance(event, OrderEvent):
            self._apply_order_event(event)
        else:
            raise TypeError(f"cannot apply {type(event).__name__}")

    def _apply_order_event(self, ev: OrderEvent, *, fill_refused: bool = False) -> None:
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
            ref = self._ref_price(o)
            if ref is not None:
                self._checked[o.client_id] = self._gate.valuation_price(o, ref)
            self._commit(o, o.status, kind, ev.ts_event)
            return
        new = _S(kind.upper()) if kind in ("denied", "submitted") else _KIND_STATUS.get(kind)
        if new is None:
            raise ValueError(f"unsupported order event kind {kind!r}")
        resolved = ev.resolved_qty if new is _S.ACCEPTED and o.qty is None else None
        if resolved is not None and not (math.isfinite(resolved) and resolved > 0):
            raise ValueError(f"resolved_qty {resolved} of {o.client_id} must be finite and > 0")
        unresolved = (new is _S.ACCEPTED and o.qty is None and resolved is None
                      and self.portfolio.kind is AccountKind.LIVE)
        if unresolved:  # NT-299: a LIVE size_frac accept must carry resolved_qty; the engine cannot bound its fills
            check_transition(o.status, _S.REJECTED)
            if ev.order.venue_id:
                o.venue_id = ev.order.venue_id
            self._reserved.pop(o.client_id, None)
            self._commit(o, _S.REJECTED, "rejected", ev.ts_event, reason=RESOLVED_QTY_MISSING)
            self._halt_risk(RESOLVED_QTY_MISSING)
            try:  # the venue accepted it: ask it to cancel (the order is REJECTED here, the reply changes nothing)
                self.broker.cancel(o.client_id)
            except Exception as exc:  # noqa: BLE001 - journalled; fills that still arrive go to venue_fill_on_rejected
                log.error("venue cancel of %s (resolved_qty missing) failed: %r", o.client_id, exc, exc_info=exc)
                self._record_error(exc)
                self._write("cancel_failed", {"client_id": o.client_id, "reason": f"broker_error:{type(exc).__name__}",
                                              "error": repr(exc)}, self.clock.now_ms())
            return
        if not (fill_refused and new is _S.REJECTED and o.status in (_S.ACCEPTED, _S.TRIGGERED)):
            check_transition(o.status, new)  # the one exception edge besides late_fill: a refused fill (A5.1)
        if ev.order.venue_id:
            o.venue_id = ev.order.venue_id
        if resolved is not None:
            self._resolved[o.client_id] = self._reserved[o.client_id] = self._sized[o.client_id] = float(resolved)
        if new is _S.CANCELED:
            self.cancel_acked.add(o.client_id)
        if new in CLOSED_STATUSES:
            self._reserved.pop(o.client_id, None)  # a dead or closed order holds no reservation
        self._commit(o, new, kind, ev.ts_event, reason=ev.order.reason)

    def _apply_fill(self, ev: FillEvent) -> None:
        """Simulated accounts: ``_book_fill``. LIVE (NT-299): a venue fact is never lost; a fill the engine cannot book
        (no spot balance, a fill after the order is FILLED or EXPIRED, before its accept, beyond its quantity, for an
        unknown client_id, ...) is journalled ``venue_fill_unbooked`` with the fill and an order snapshot and RiskGuard
        is halted; only then does the error propagate."""
        if self.portfolio.kind is not AccountKind.LIVE:
            self._book_fill(ev)
            return
        f = ev.fill
        self._unbooked_journalled = False
        try:
            self._book_fill(ev)
        except Exception as exc:
            if not self._unbooked_journalled and not (f.trade_id and f.trade_id in self._trades):
                self._venue_fill_unbooked(f, exc)
            raise

    def _venue_fill_unbooked(self, f: Fill, exc: BaseException) -> None:
        o = self._orders.get(f.client_id)
        snap = {"unknown": f.client_id} if o is None else dataclasses.replace(o, tags=dict(o.tags)).to_dict()
        try:
            fill = f.to_dict()
        except Exception:  # noqa: BLE001 - a malformed venue report is still recorded
            fill = {"repr": repr(f)}
        log.error("LIVE fill %s of %s could not be booked: %r", f.trade_id, f.client_id, exc)
        self._write(VENUE_FILL_UNBOOKED, {"client_id": f.client_id, "trade_id": f.trade_id, "fill": fill,
                                          "order": snap, "error": f"{type(exc).__name__}: {exc}"}, f.ts)
        self._halt_risk(VENUE_FILL_UNBOOKED)

    def _book_fill(self, ev: FillEvent) -> None:
        f = ev.fill
        if not f.trade_id:
            raise ValueError("a fill needs a non-empty trade_id (it is the dedup key)")
        o = self._get(f.client_id)
        if f.trade_id in self._trades:  # a known trade is never applied twice
            log.debug("duplicate fill %s ignored", f.trade_id)
            return
        if f.instrument_id != o.instrument_id or f.side is not o.side:
            raise ValueError(f"fill {f.trade_id} does not match order {o.client_id}")
        live_acct = self.portfolio.kind is AccountKind.LIVE
        if live_acct and o.status is _S.REJECTED:  # the venue filled an order the engine thought rejected (D)
            self._venue_fill_on_rejected(o, ev)
            return
        if not live_acct and (o.client_id in self._refused or o.status is _S.REJECTED):
            # a simulated fill for an order that was refused or never lived: nothing is applied (the SimBroker must
            # drop the order when precheck_fill or the engine refuses it)
            self._write("fill_after_refusal", {"client_id": o.client_id, "trade_id": f.trade_id, "qty": f.qty,
                                               "price": f.price, "status": o.status.value}, f.ts)
            return
        late = o.client_id in self.cancel_acked  # dead: every fill after the venue's cancel ack is a late fill (g)
        if o.status is _S.FILLED:
            raise OverfillError(f"fill {f.trade_id} for the already filled order {o.client_id}")
        cap = o.qty if o.qty is not None else self._resolved.get(o.client_id)  # None: a size_frac nobody resolved
        over = cap is not None and o.filled_qty + f.qty > cap + DUST
        # NT-299 (lead, repair 1): a fill above a resolved size_frac order's resolved_qty has its excess refused on
        # PAPER / BACKTEST (the applied part is booked) and is booked in full on LIVE with fill_refused (applied) + halt
        resolved_left = max(cap - o.filled_qty, 0.0) if over and o.qty is None else None
        if over and resolved_left is None:
            raise OverfillError(f"fills of {o.client_id} sum to {o.filled_qty + f.qty}, above its quantity {cap}")
        full = cap is None or o.filled_qty + f.qty >= cap - DUST
        check_transition(o.status, _S.FILLED if full else _S.PARTIALLY_FILLED, late_fill=late)  # before any change
        # the binding check failed for (part of) the fill: (check reason, checked price, fill qty, allowed qty)
        unwanted: Optional[tuple[str, float, float, float]] = None
        checked_px = self._checked.get(o.client_id, 0.0)
        if late and live_acct and resolved_left is not None:  # a LIVE late fill gets no re-check, only this record
            unwanted = (RESOLVED_QTY_EXCEEDED, checked_px, f.qty, resolved_left)
        if not (late and live_acct):  # a LIVE late fill is a venue fact: only the flip / spot rules below apply
            d = self._gate.decision(o, f.mid, f.qty, always=late)
            allowed = d.qty if d.allowed and d.qty is not None else 0.0
            if allowed < f.qty - DUST:  # denied, or clipped (A): only the allowed quantity may be applied
                why = d.reason or CLIPPED_TO_POSITION
                if resolved_left is not None and allowed >= resolved_left - DUST:
                    why = RESOLVED_QTY_EXCEEDED  # only the quantity beyond resolved_qty is refused
                unwanted = (why, checked_px, f.qty, allowed)
                if not live_acct:  # simulated: apply the allowed part only; nothing at all when it is 0
                    if allowed <= DUST:
                        self._gate.refuse(o, f.mid, unwanted[0], unwanted[1], dead=late)
                        return
                    f = dataclasses.replace(f, qty=allowed, fee=f.fee * allowed / f.qty)
                    ev = FillEvent(ev.ts_event, ev.ts_init, f)
        total = o.filled_qty + f.qty
        done = cap is None or total >= cap - DUST
        new = _S.FILLED if done else _S.PARTIALLY_FILLED
        check_transition(o.status, new, late_fill=late)  # before anything changes
        before = self.portfolio.signed_qty(self.portfolio.position(o.instrument_id, o.strategy_id))
        try:
            pos_events = self.portfolio.apply_fill(f, strategy_id=o.strategy_id)  # may raise: nothing is changed yet
        except OverfillError as exc:
            if late:  # a venue fact the books cannot hold (spot balance): record it and stop trading (f)
                self._late_fill_event(o, f, "late_fill_refused", {"error": repr(exc)})
                self._halt_risk("late_fill_refused")
                self._unbooked_journalled = True  # LIVE: this record is the venue_fill_unbooked of NT-299
            elif not live_acct:  # a simulated fill the books cannot hold is refused like the binding check (NT-299)
                self._gate.refuse(o, f.mid, "books_refused", self._checked.get(o.client_id, 0.0))
                return
            raise
        self._trades.add(f.trade_id)
        o.avg_fill_price = (o.avg_fill_price * o.filled_qty + f.price * f.qty) / total
        o.filled_qty = total
        self._fill_notional[o.client_id] = self._fill_notional.get(o.client_id, 0.0) + f.qty * f.mid
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
        if unwanted is not None:
            reason, checked, asked, allowed = unwanted
            if live_acct:  # the venue fact was applied in full: journal what the check refused, halt
                self._gate.refuse(o, f.mid, reason, checked, applied=True,
                                  qty=(asked, allowed) if allowed > DUST else None)
                self._halt_risk(FILL_REFUSED)
            else:  # the allowed part is applied: the excess is refused and the order ends (A)
                self._gate.refuse(o, f.mid, reason, checked, dead=late, qty=(asked, allowed))
        elif not late and o.client_id in self._fill_limit and o.filled_qty >= self._fill_limit[o.client_id] - DUST:
            self._gate.end_refused(o)  # a clipped precheck already journalled the refused excess
        sign = 1 if f.side is Side.BUY else -1
        after = self.portfolio.signed_qty(self.portfolio.position(o.instrument_id, o.strategy_id))
        if late and sign * before < 0 and sign * after > DUST:  # the late fill flipped the position (margin)
            self._late_fill_event(o, f, "late_fill_flip", {"before": before, "after": after})
            self._halt_risk("late_fill_flip")

    # ------------------------------------------------------------------ the binding check at the fill (A5.1)
    def precheck_fill(self, client_id: str, fill_price: float, *, qty: Optional[float] = None) -> RiskDecision:
        """The binding check of CONTRACTS A5.1, for a broker about to report a fill of ``client_id`` at the reference
        price ``fill_price`` (the fill's ``mid``: before spread and slippage, which RiskGuard adds). Allowed: the fill
        may go ahead with the decision's quantity; a ``size_frac`` order's quantity is sized at ``fill_price`` on
        realised equity (A5.1; at most that, or its resolved quantity, also when ``qty`` asks for more; never the
        quantity reserved at submit). Clipped (an explicit-quantity order whose decision carries less than asked, reason
        ``clipped_to_position``): the broker may fill only ``d.qty``; the excess is refused now, journalled once per
        order (``fill_refused`` with ``fill_qty`` / ``allowed_qty`` / ``refused_qty``) and the order
        ends (CANCELED, reason ``fill_price_check_failed``) once that quantity is filled. Denied: the engine has already
        journalled ``fill_refused`` and ended the order (REJECTED with reason ``fill_price_check_failed``, CANCELED with
        the same reason when part of it was filled); the broker must not fill it. When the check runs (and what is
        never refused) is ``FillGate.decision``'s rule, the module docstring's "binding check". ``qty`` defaults to the
        unfilled quantity. Not allowed from a bus handler (it would reorder the journal).

        The per-minute rate limit is ignored (``fill_recheck``). On a LIVE account this is a no-op (the venue fills by
        itself; ``apply`` journals and halts). The SimBroker must drop an order this method (or ``apply``) refused: a
        later simulated fill for it is journalled ``fill_after_refusal`` and ignored."""
        self.assert_owner()
        if self._handlers.publishing:
            raise RuntimeError("precheck_fill cannot run inside a bus handler")
        o = self._get(client_id)
        if self.portfolio.kind is AccountKind.LIVE:
            return RiskDecision(True, "", qty)
        return self._gate.precheck(o, fill_price, qty)

    def _venue_fill_on_rejected(self, o: Order, ev: FillEvent) -> None:
        """(D) LIVE: the venue filled an order the engine marked REJECTED (a submit that timed out but reached it). A
        real fill is never lost: it is applied to the Portfolio, journalled ``venue_fill_on_rejected`` (on the fill
        topic) with the order, the position and account follow, and RiskGuard is halted. The order stays REJECTED (A4
        has no edge out of it); its filled quantity and average price record what the venue did."""
        f = ev.fill
        total = o.filled_qty + f.qty
        cap = o.qty if o.qty is not None else self._resolved.get(o.client_id)
        if cap is None or total > cap + DUST:  # beyond the order, or no known quantity: venue_fill_unbooked (NT-299)
            raise OverfillError(f"fills of {o.client_id} sum to {total}, above its quantity {cap} (rejected order)")
        pos_events = self.portfolio.apply_fill(f, strategy_id=o.strategy_id)  # may raise: nothing is changed yet
        self._trades.add(f.trade_id)
        o.avg_fill_price = (o.avg_fill_price * o.filled_qty + f.price * f.qty) / total
        o.filled_qty = total
        self._fill_notional[o.client_id] = self._fill_notional.get(o.client_id, 0.0) + f.qty * f.mid
        snap = dataclasses.replace(o, tags=dict(o.tags))
        self._emit("fill", ev, "venue_fill_on_rejected",
                   {"client_id": o.client_id, "trade_id": f.trade_id, "fill": f.to_dict(), "order": snap.to_dict()},
                   f.ts)
        for pe in pos_events:
            self._emit("position", pe, "position", {"position": pe.position.to_dict()}, pe.ts_event)
        marks = {**self.marks}
        marks.setdefault(f.instrument_id, f.mid)
        acct = AccountEvent(f.ts, self.clock.now_ms(), self.portfolio.account_state(marks, f.ts))
        self._emit("account", acct, "account", {"account": acct.account.to_dict()}, f.ts)
        self._halt_risk("venue_fill_on_rejected")

    def _late_fill_event(self, o: Order, f: Fill, kind: str, extra: dict) -> None:
        snap = dataclasses.replace(o, tags=dict(o.tags))
        self._emit("order", OrderEvent(f.ts, self.clock.now_ms(), kind, snap), kind,
                   {"client_id": o.client_id, "trade_id": f.trade_id, "order": snap.to_dict(), "fill": f.to_dict(),
                    **extra}, f.ts)

    def _halt_risk(self, reason: str) -> None:
        """RiskGuard to HALTED because of a venue fact; journalled once per state change, and the resting OPENING
        orders are cancelled once the current operation ends (protective reduce orders stay)."""
        was = self.risk.state is TradingState.HALTED
        try:
            self.risk.halt(reason)
        except Exception as exc:  # noqa: BLE001 - the state is set before the callback runs
            log.error("RiskGuard.halt(%s) callback failed: %r", reason, exc, exc_info=exc)
            self._record_error(exc)
        if not was:
            self._write("halt", {"reason": reason}, self.clock.now_ms())
            self._queue_cancel_opening()

    def _queue_cancel_opening(self) -> None:
        if not self._cancel_queued:
            self._cancel_queued = True
            self._deferred.append((ExecutionEngine._cancel_opening, (), {}))

    def _cancel_opening(self) -> None:
        self._cancel_queued = False
        left = lambda o: max(self._reserved.get(o.client_id, 0.0) - o.filled_qty, 0.0)  # noqa: E731
        self._cancel_each([o for o in self._resting() if not self._gate.reducing(o, left(o))],
                          "halt_cancel_failed", reset=False)  # adds to the failures of the call that halted

    def _on_queue_overflow(self, name: str) -> None:
        """The handler queue overflowed (``HandlerQueue``; it already dropped the queue): journal it and halt."""
        log.error("handler call queue overflow at %s: halting", name)
        self._write("handler_queue_overflow", {"method": name, "limit": HANDLER_QUEUE_LIMIT}, self.clock.now_ms())
        self._halt_risk("handler_queue_overflow")

    def _journal_overflow(self) -> None:
        was = self.risk.state is TradingState.HALTED
        try:
            self.risk.halt(JOURNAL_HALT_REASON)
        finally:  # the halt itself is journalled (queued: the sink is down), once per state change
            if not was and self.risk.state is TradingState.HALTED:
                now = self.clock.now_ms()
                self._journal.enqueue("halt", {"ts_event": now, "ts_init": now, "reason": JOURNAL_HALT_REASON})
                self._queue_cancel_opening()

    def _drain(self) -> None:
        """Run the calls queued by bus handlers (``HandlerQueue.drain``)."""
        self._handlers.drain(self)

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
        return self._gate.committed_reduce(new, exclude)

    def _live_orders(self) -> list[Order]:
        """The live orders in submission order, from the index (cost: the live orders, not every order ever sent)."""
        return [self._orders[cid] for cid in list(self._live)]

    def _committed_open(self, new: Order, exclude: Optional[str] = None) -> tuple[float, float]:
        return self._gate.committed_open(new, exclude)

    def _commit(self, o: Order, new: OrderStatus, kind: str, ts_event: int, *, late: bool = False,
                reason: Optional[str] = None, extra: Optional[dict] = None) -> None:
        old = o.status
        o.status, o.ts_updated = new, ts_event
        if self.is_live(o):
            self._live[o.client_id] = None
        else:
            self._live.pop(o.client_id, None)
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
        self._handlers.publishing += 1  # mutations asked for by handlers wait for the outermost call (``_mutating``)
        try:
            self.bus.publish(topic, event)
        except Exception as exc:  # noqa: BLE001 - a subscriber must not break the engine
            log.exception("subscriber of %r raised", topic)
            self._record_error(exc)
        finally:
            self._handlers.publishing -= 1

    def _write(self, kind: str, payload: dict, ts_event: int) -> None:
        self._journal.write(kind, {"ts_event": ts_event, "ts_init": self.clock.now_ms(), **payload})

    def _record_error(self, exc: BaseException) -> None:
        self.handler_errors.append(exc)
        del self.handler_errors[:-100]

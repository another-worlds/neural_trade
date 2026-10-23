"""SimBroker (PLAN A2 ``broker/sim.py``, A5; DESK-1.4): fills from CLOSED bars, for backtest, replay and paper.

The broker only REPORTS (``Fill`` / ``OrderEvent`` lists); the ExecutionEngine journals and applies them. It sees a bar
only when one of the ``fill_*`` / ``on_bar_intrabar`` methods is called with it, so it cannot look ahead. Per closed
bar the caller runs A5.9 in this order, applying every returned list before the next call:

1. ``fill_exits(bar)``: pending reduce-only MARKET orders (TIME, strategy exits, ``close_position``) at the OPEN;
2. ``fill_entries(bar, cash)``: the other pending MARKET orders at the open; a ``size_frac`` order is sized on ``cash``,
   which the caller passes as ``Portfolio.realised_equity()`` after step 1 (A5.1; the parameter keeps C1's name);
3. ``on_bar_intrabar(bar)``: brackets of the positions in the Portfolio (this bar's high/low, fill bar included),
   then resting LIMIT / STOP_MARKET / STOP_LIMIT orders;
4. ``fill_at_close(bar)``: MARKET orders tagged ``tags["fill_at"] == "close"`` (the Engine's EOW exits) at the CLOSE.

An order is eligible for a bar only if the bar opens after the last bar the broker had seen when the order was
submitted, and not before ``order.ts_created`` (A5.6): a fill never precedes the order's bar. ``cancel`` takes effect at
once, so a cancel sent before a bar is fed beats that bar's fill.

Gaps CONTRACTS leaves open (the NT-225 report lists them):

* ``SimBroker(instrument, cost, portfolio, clock, tiebreak=...)``: the broker READS the Portfolio (positions, stored
  brackets, ``realised_equity`` for intrabar ``size_frac``, the reduce-only quantity); it never writes it;
* a bracket hit is reported as ``[OrderEvent("accepted", exit_order), Fill]`` for a synthetic reduce-only MARKET order
  ``bracket:<position_id>:<TP|SL>:<ts_open>`` (status SUBMITTED, ``tags["reason"]`` TP / SL) that the engine does not
  know: the Engine must register it before applying the events. The exit closes the whole position at the level
  (``fills.bracket_exit``: a gap fills at the open) with taker costs on that mid. An offset bracket resolves against the
  open of the bar of the entry fill that carried it, remembered here; an offset bracket this broker never saw filled
  is skipped with a warning. A position opened intrabar (limit / stop) is first checked on the NEXT bar;
* exits: a reduce-only order is clipped to the position it reduces; with none it EXPIRES (``reduce_only_no_position``),
  as the oracle drops a pending exit whose position is gone. An order whose size resolves to nothing, or below the
  instrument minimum, EXPIRES (``zero_size`` / ``below_minimum``) with ``resolved_qty`` 0. Quantities are floored to
  ``size_increment``, except the close of a whole position;
* ``resolved_qty``: a ``size_frac`` order is resolved at the fill, after its "accepted" event, so the resolved
  quantity is ``Fill.qty`` and ``SimBroker.resolved[client_id]``; only an expiry carries ``OrderEvent.resolved_qty``;
* STOP_LIMIT: the stop triggers on bar b (event "triggered"); its limit is evaluated from bar b + 1 (the order of a
  bar's high and low is unknown). STOP_MARKET reports "triggered" then the Fill. A touch fills a limit (A5.2);
* ``tags["oco"]``: orders sharing a group id cancel each other when one fills (event "canceled",
  ``oco_sibling_filled``); a same-bar tie takes the stop first under ``sl_first``, the limit first under ``tp_first``;
* TIF: IOC / FOK (LIMIT only; no partial fills at bar resolution) expire after their first eligible bar; DAY expires
  on the first bar of a later UTC day; GTC rests. IOC / FOK on a stop is rejected (``tif_not_supported``);
* ``cancel`` / ``modify`` of an order the broker does not hold raise ``UnknownBrokerOrder``; ``sync`` reports nothing;
* ``tp_sl_on='close'`` of the oracle is not supported (brackets use high/low). Pure logic, standard library only.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Optional

from neural_trade.desk.broker.base import UnknownBrokerOrder
from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.costs import CostProfile
from neural_trade.desk.core.fills import bracket_exit, limit_fill, market_fill, stop_fill
from neural_trade.desk.core.model import (Bar, Fill, Instrument, Order, OrderEvent, OrderStatus, OrderType,
                                          PositionSide, Side, TimeInForce)
from neural_trade.desk.core.portfolio import DUST, Portfolio
from neural_trade.desk.core.risk import round_down

log = logging.getLogger(__name__)

MS_DAY = 86_400_000
FILL_AT = "fill_at"  # Order.tags key
CLOSE = "close"
OCO = "oco"  # Order.tags key
_STOPS = (OrderType.STOP_MARKET, OrderType.STOP_LIMIT)


@dataclass
class _Rest:
    order: Order
    after: int  # ts_open of the last bar seen at submit: eligible only for bars opening after it
    seq: int
    day0: int
    trigger_ts: Optional[int] = None  # STOP_LIMIT: the bar that triggered it (the limit starts the bar after)


def _sign(o: Order) -> int:
    return 1 if o.side is Side.BUY else -1


class SimBroker:
    def __init__(self, instrument: Instrument, cost: CostProfile, portfolio: Portfolio, clock: Clock, *,
                 tiebreak: str = "sl_first") -> None:
        if tiebreak not in ("sl_first", "tp_first"):
            raise ValueError(f"tiebreak must be 'sl_first' or 'tp_first', got {tiebreak!r}")
        self.ins, self.cost, self.pf, self.clock, self.tiebreak = instrument, cost, portfolio, clock, tiebreak
        self.resolved: dict[str, float] = {}  # client_id -> quantity a size_frac / reduce order was filled at
        self._rests: dict[str, _Rest] = {}
        self._closed: set[str] = set()
        self._bases: dict[tuple, float] = {}  # (position_id, bracket items) -> open of the entry fill bar
        self._used: dict[str, float] = {}  # position_id -> quantity already reported as reduced in this call
        self._seen = -1
        self._seq = self._trades = 0

    @property
    def pending(self) -> list[Order]:
        return [dataclasses.replace(r.order, tags=dict(r.order.tags)) for r in self._rests.values()]

    # ------------------------------------------------------------------ commands
    def submit(self, order: Order) -> OrderEvent:
        now = self.clock.now_ms()
        why = "duplicate_client_id" if order.client_id in self._rests or order.client_id in self._closed \
            else self._invalid(order)
        if why:
            return self._ev("rejected", order, now, why)
        self._seq += 1
        o = dataclasses.replace(order, tags=dict(order.tags), venue_id=f"sim-{order.client_id}")
        self._rests[o.client_id] = _Rest(o, self._seen, self._seq, (order.ts_created or now) // MS_DAY)
        return self._ev("accepted", o, now)

    def cancel(self, client_id: str) -> OrderEvent:
        r = self._rests.get(client_id)
        if r is None:
            raise UnknownBrokerOrder(f"{client_id}: not held by the broker"
                                     f"{' (already closed)' if client_id in self._closed else ''}")
        return self._drop(r, "canceled", "", self.clock.now_ms())

    def modify(self, client_id: str, *, price: Optional[float] = None, trigger_price: Optional[float] = None,
               qty: Optional[float] = None) -> OrderEvent:
        r = self._rests.get(client_id)
        if r is None:
            raise UnknownBrokerOrder(f"{client_id}: not held by the broker")
        if qty is not None and r.order.qty is None:
            raise ValueError(f"{client_id} is a size_frac order: it has no quantity to change")
        new = dataclasses.replace(r.order, tags=dict(r.order.tags), qty=r.order.qty if qty is None else qty,
                                  price=r.order.price if price is None else price,
                                  trigger_price=r.order.trigger_price if trigger_price is None else trigger_price)
        why = self._invalid(new)
        if why:
            raise ValueError(f"modify of {client_id} refused: {why}")
        r.order = new
        now = self.clock.now_ms()
        return self._ev("modified", new, now)

    def sync(self) -> list[Fill | OrderEvent]:
        return []

    def _invalid(self, o: Order) -> str:
        pos_num = lambda x: x is not None and math.isfinite(x) and x > 0  # noqa: E731
        if o.instrument_id != self.ins.id:
            return "wrong_instrument"
        if o.qty is not None and not pos_num(o.qty) or o.size_frac is not None and not pos_num(o.size_frac):
            return "invalid_quantity"
        if o.type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and not pos_num(o.price):
            return "invalid_price"
        if o.type in _STOPS and not pos_num(o.trigger_price):
            return "invalid_trigger"
        if o.type in _STOPS and o.tif in (TimeInForce.IOC, TimeInForce.FOK):
            return "tif_not_supported"
        return ""

    # ------------------------------------------------------------------ bar steps (A5.9)
    def fill_exits(self, bar: Bar) -> list[Fill | OrderEvent]:
        self._begin(bar)
        return self._markets(bar, lambda o: o.reduce_only and o.tags.get(FILL_AT) != CLOSE, None, at_close=False)

    def fill_entries(self, bar: Bar, cash: float) -> list[Fill | OrderEvent]:
        self._begin(bar)
        return self._markets(bar, lambda o: not o.reduce_only and o.tags.get(FILL_AT) != CLOSE, cash, at_close=False)

    def fill_at_close(self, bar: Bar) -> list[Fill | OrderEvent]:
        self._begin(bar)
        return self._markets(bar, lambda o: o.tags.get(FILL_AT) == CLOSE, None, at_close=True)

    def on_bar_intrabar(self, bar: Bar) -> list[Fill | OrderEvent]:
        self._begin(bar)
        out: list[Fill | OrderEvent] = []
        self._brackets(bar, out)
        self._resting(bar, out)
        return out

    def _begin(self, bar: Bar) -> None:
        if not bar.is_closed:
            raise ValueError("the broker fills on CLOSED bars only")
        if bar.instrument_id != self.ins.id:
            raise ValueError(f"bar of {bar.instrument_id}, this broker trades {self.ins.id}")
        if bar.ts_open_ms < self._seen:
            raise ValueError(f"bars must be fed in time order: {bar.ts_open_ms} < {self._seen}")
        self._seen = bar.ts_open_ms
        self._used = {}

    def _eligible(self, r: _Rest, bar: Bar, at_close: bool = False) -> bool:
        if at_close:  # created at or before this bar's close: it may fill at this close
            return r.after <= bar.ts_open_ms and r.order.ts_created <= bar.ts_close_ms
        return r.after < bar.ts_open_ms and r.order.ts_created <= bar.ts_open_ms

    def _markets(self, bar, pick, equity, *, at_close) -> list[Fill | OrderEvent]:
        out: list[Fill | OrderEvent] = []
        mid, ts = (bar.close, bar.ts_close_ms) if at_close else (bar.open, bar.ts_open_ms)
        for r in sorted(self._rests.values(), key=lambda x: x.seq):
            o = r.order
            if o.type is not OrderType.MARKET or not pick(o) or not self._eligible(r, bar, at_close):
                continue
            qty, why = self._resolve(o, mid, self.pf.realised_equity() if equity is None else equity)
            if qty is None:
                out.append(self._drop(r, "expired", why, ts, resolved=0.0))
            else:
                self._fill(r, bar, market_fill(mid, _sign(o), self.cost, qty), ts, out)
        return out

    def _brackets(self, bar: Bar, out: list) -> None:
        for pid, pos in list(self.pf.positions.items()):
            raw = self.pf.get_bracket(pid) if pos.instrument_id == self.ins.id and pos.qty > DUST else None
            if not raw:
                continue
            tp, sl = raw.get("take_profit"), raw.get("stop_loss")
            if raw.get("tp_is_offset"):
                base = self._bases.get(self._key(pid, raw))
                if base is None:
                    log.warning("offset bracket of %s was never seen filled by this broker: skipped", pid)
                    continue
                tp, sl = (None if v is None else float(base + v) for v in (tp, sl))
            sign = 1 if pos.side is PositionSide.LONG else -1
            hit = bracket_exit(bar, sign, tp, sl, self.tiebreak)
            if hit is None:
                continue
            reason, px = hit
            now = self.clock.now_ms()
            o = Order(client_id=f"bracket:{pid}:{reason}:{bar.ts_open_ms}", instrument_id=self.ins.id,
                      account_id=self.pf.account_id, strategy_id=pos.strategy_id,
                      side=Side.SELL if sign > 0 else Side.BUY, type=OrderType.MARKET, qty=pos.qty, reduce_only=True,
                      status=OrderStatus.SUBMITTED, ts_created=bar.ts_open_ms, ts_updated=bar.ts_open_ms,
                      tags={"reason": reason, "bracket": True})
            out.append(OrderEvent(bar.ts_open_ms, now, "accepted", o))
            out.append(self._mkfill(o, market_fill(px, -sign, self.cost, pos.qty), bar.ts_open_ms))
            self._used[pid] = self._used.get(pid, 0.0) + pos.qty

    def _resting(self, bar: Bar, out: list) -> None:
        sl_first = self.tiebreak == "sl_first"

        def rank(r: _Rest) -> tuple:  # an OCO tie: stop first under sl_first, limit first under tp_first
            grouped = r.order.tags.get(OCO) is not None
            return (0 if not grouped or (r.order.type in _STOPS) == sl_first else 1, r.seq)

        for r in sorted((r for r in self._rests.values() if r.order.type is not OrderType.MARKET
                         and self._eligible(r, bar)), key=rank):
            o = r.order
            if o.client_id not in self._rests:  # cancelled by an OCO sibling earlier in this call
                continue
            ts, sgn = bar.ts_open_ms, _sign(o)
            if o.tif is TimeInForce.DAY and bar.ts_open_ms // MS_DAY > r.day0:
                out.append(self._drop(r, "expired", "day_expired", ts))
                continue
            limit_stage = o.type is OrderType.LIMIT or (o.type is OrderType.STOP_LIMIT and r.trigger_ts is not None
                                                        and r.trigger_ts < bar.ts_open_ms)
            fn, level = (limit_fill, o.price) if limit_stage else (stop_fill, o.trigger_price)
            probe = fn(bar, sgn, level, self.cost, 1.0)
            if limit_stage:
                if probe is None:
                    if o.tif in (TimeInForce.IOC, TimeInForce.FOK):
                        out.append(self._drop(r, "expired", "ioc_unfilled", ts))
                    continue
            else:
                if probe is None:
                    continue
                out.append(self._ev("triggered", o, ts))
                if o.type is OrderType.STOP_LIMIT:
                    r.trigger_ts = ts
                    continue
            qty, why = self._resolve(o, probe.mid, self.pf.realised_equity())
            if qty is None:
                out.append(self._drop(r, "expired", why, ts, resolved=0.0))
            else:
                self._fill(r, bar, fn(bar, sgn, level, self.cost, qty), ts, out)

    # ------------------------------------------------------------------ helpers
    def _avail(self, o: Order) -> float:
        """Quantity of the position this reduce-only order may still reduce (0 when none, or the wrong side)."""
        pos = self.pf.positions.get(f"{o.instrument_id}-{o.strategy_id}")
        if pos is None or pos.qty <= DUST or (pos.side is PositionSide.LONG) != (o.side is Side.SELL):
            return 0.0
        return pos.qty - self._used.get(pos.position_id, 0.0)

    def _resolve(self, o: Order, mid: float, equity: float) -> tuple[Optional[float], str]:
        """(quantity at this fill, "") or (None, why the order cannot fill). A5.1: size_frac is
        clip(size_frac, 0, 1) x equity / mid."""
        ins = self.ins
        req = o.qty if o.qty is not None else min(max(o.size_frac, 0.0), 1.0) * equity / mid
        whole = False
        if o.reduce_only:
            avail = self._avail(o)
            if avail <= DUST:
                return None, "reduce_only_no_position"
            whole = req >= avail - DUST
            req = min(req, avail)
        qty = req if whole else round_down(req, ins.size_increment)
        if ins.max_qty is not None:
            qty = min(qty, ins.max_qty)
        if not qty > DUST:
            return None, "zero_size"
        if not whole and (qty < ins.min_qty or qty * mid < ins.min_notional):
            return None, "below_minimum"
        return qty, ""

    def _mkfill(self, o: Order, f: Fill, ts: int) -> Fill:
        self._trades += 1
        return dataclasses.replace(f, trade_id=f"sim-{self._trades}", client_id=o.client_id,
                                   instrument_id=self.ins.id, fee_currency=self.ins.quote, ts=ts)

    def _fill(self, r: _Rest, bar: Bar, f: Fill, ts: int, out: list) -> None:
        o = r.order
        f = self._mkfill(o, f, ts)
        self.resolved[o.client_id] = f.qty
        pid = f"{o.instrument_id}-{o.strategy_id}"
        if o.reduce_only:
            self._used[pid] = self._used.get(pid, 0.0) + f.qty
        elif o.bracket:
            self._bases[self._key(pid, o.bracket)] = bar.open
        del self._rests[o.client_id]
        self._closed.add(o.client_id)
        out.append(f)
        group = o.tags.get(OCO)
        if group is not None:
            for sib in list(self._rests.values()):
                if sib.order.tags.get(OCO) == group:
                    out.append(self._drop(sib, "canceled", "oco_sibling_filled", ts))

    @staticmethod
    def _key(pid: str, bracket: dict) -> tuple:
        return pid, tuple(sorted(bracket.items()))

    def _drop(self, r: _Rest, kind: str, reason: str, ts: int, resolved: Optional[float] = None) -> OrderEvent:
        del self._rests[r.order.client_id]
        self._closed.add(r.order.client_id)
        return self._ev(kind, r.order, ts, reason, resolved)

    def _ev(self, kind: str, o: Order, ts: int, reason: str = "", resolved: Optional[float] = None) -> OrderEvent:
        snap = dataclasses.replace(o, tags=dict(o.tags), reason=reason or o.reason)
        return OrderEvent(ts, self.clock.now_ms(), kind, snap, resolved)

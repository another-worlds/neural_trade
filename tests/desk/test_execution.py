"""NT-224 (DESK-1.3): ExecutionEngine pinned by scripted broker events with literal expected states.

The broker is a scripted ``FakeBroker`` and the journal a list; nothing here needs a database or a bar.
"""
from __future__ import annotations

import dataclasses
import itertools
import logging
import threading
from collections import deque

import pytest

from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import SimClock
from neural_trade.desk.core.costs import RESEARCH_ZERO, CostProfile
from neural_trade.desk.core.execution import DuplicateClientId, ExecutionEngine, UnknownAccount, UnknownOrder
from neural_trade.desk.core.model import (
    AccountKind, CancelAll, CancelOrder, ClosePosition, Fill, FillEvent, Halt, IllegalTransition, Instrument,
    ModifyOrder, Order, OrderEvent, OverfillError,
)
from neural_trade.desk.core.portfolio import Portfolio
from neural_trade.desk.core.risk import RiskGuard, RiskLimits, TradingState

X = "X"
_ids = itertools.count(1)


class FakeBroker:
    """Scripted: ``submit_kind`` is what the next submit reports; every call is recorded."""

    def __init__(self) -> None:
        self.submitted: list[Order] = []
        self.canceled: list[str] = []
        self.submit_kind = "accepted"
        self.submit_reason = ""
        self.cancel_kind = "canceled"
        self.venue = {}
        self.raise_on_submit = None
        self.modified: list[dict] = []
        self.accept_tags: dict = {}
        self.resolved_qty = None
        self.raise_on_cancel: dict = {}
        self.raise_on_modify = None
        self.modified_qty_override = None

    def modify(self, client_id, *, price=None, trigger_price=None, qty=None):
        if self.raise_on_modify:
            raise self.raise_on_modify
        self.modified.append({"price": price, "trigger_price": trigger_price, "qty": qty})
        v = self.venue[client_id]
        if self.modified_qty_override is not None:
            qty = self.modified_qty_override
        return OrderEvent(1000, 1000, "modified", dataclasses.replace(v, price=price, trigger_price=trigger_price,
                                                                      qty=qty if v.qty is not None else None))

    def submit(self, order):
        self.submitted.append(order)
        self.venue.setdefault(order.client_id, order)  # engine-made orders (close_position) too
        if self.raise_on_submit:
            raise self.raise_on_submit
        return OrderEvent(1000, 1000, self.submit_kind,
                          dataclasses.replace(order, reason=self.submit_reason, venue_id=f"v-{order.client_id}",
                                              tags={**order.tags, **self.accept_tags}),
                          resolved_qty=self.resolved_qty)

    def cancel(self, client_id):
        if self.raise_on_cancel and client_id in self.raise_on_cancel:
            raise self.raise_on_cancel[client_id]
        self.canceled.append(client_id)
        return OrderEvent(1000, 1000, self.cancel_kind, dataclasses.replace(self.venue[client_id]))


class Rig:
    def __init__(self, limits=None, *, is_spot=False, cash=10_000.0, strict_bus=False, costs=RESEARCH_ZERO,
                 kind=AccountKind.PAPER, on_state_change=None):
        ins = Instrument(id=X, symbol="X/USDT", base="X", quote="USDT", price_increment=0.0, size_increment=0.0,
                         min_qty=0.0, min_notional=0.0, is_spot=is_spot)
        self.clock = SimClock(1000)
        self.pf = Portfolio("acc", kind=kind, initial_cash=cash, allow_short=not is_spot, clock=self.clock)
        self.risk = RiskGuard(limits or RiskLimits.unlimited(), ins, self.pf, self.clock, costs=costs,
                              on_state_change=on_state_change)
        self.broker, self.bus, self.journal, self.seen = FakeBroker(), EventBus(strict=strict_bus), [], []
        for topic in ("order", "fill", "position", "account"):
            self.bus.subscribe(topic, lambda e, t=topic: self.seen.append((t, e)))
        self.down, self.fail_once = False, None
        self.eng = ExecutionEngine(self.broker, self.pf, self.risk, self.bus, self.clock, self._sink)
        self.eng.set_marks({X: 100.0})
        self.checks = []
        real = self.risk.check

        def spy(order, **kw):  # records every call and its decision
            d = real(order, **kw)
            self.checks.append((order.client_id, kw, d))
            return d

        self.risk.check = spy

    def _sink(self, kind, payload):
        if self.down or self.fail_once == kind:
            self.fail_once = None if self.fail_once == kind else self.fail_once
            raise OSError("journal unavailable")
        self.journal.append((kind, payload))

    def submit(self, side="BUY", qty=2.0, type="LIMIT", price=100.0, **kw):
        cid = kw.pop("client_id", f"c{next(_ids)}")
        o = Order(client_id=cid, instrument_id=X, account_id="acc", strategy_id=kw.pop("strategy", "s"), side=side, type=type,
                  qty=qty, price=price if type != "MARKET" else None, **kw)
        out = self.eng.submit(o)
        self.broker.venue[cid] = out
        return out

    def fill(self, o, qty, *, tid=None, price=100.0, ts=1500):
        return Fill(trade_id=tid or f"t{next(_ids)}", client_id=o.client_id, instrument_id=X, side=o.side, qty=qty,
                    price=price, mid=price, fee=0.0, fee_currency="USDT", liquidity="TAKER", ts=ts)

    def kinds(self):
        return [k for k, _ in self.journal]


# ------------------------------------------------------------------ A4 machine on scripted sequences
def test_accept_partial_fill_states_and_journal_for_a_full_life():
    r = Rig()
    o = r.submit(qty=2.0)
    assert o.status.value == "ACCEPTED" and o.venue_id == f"v-{o.client_id}"
    r.clock.set(2000)
    r.eng.apply(r.fill(o, 0.5, price=100.0, ts=1500))
    assert o.status.value == "PARTIALLY_FILLED" and o.filled_qty == 0.5 and o.avg_fill_price == 100.0
    r.eng.apply(r.fill(o, 1.5, price=102.0, ts=1600))
    assert o.status.value == "FILLED" and o.filled_qty == 2.0 and o.avg_fill_price == pytest.approx(101.5)
    assert r.pf.position(X, "s").qty == 2.0 and o.is_closed
    assert r.kinds() == ["order_submitted", "order_accepted", "order_partially_filled", "fill", "position", "account",
                         "order_filled", "fill", "position", "account"]
    for kind, p in r.journal:  # every record carries both timestamps
        assert isinstance(p["ts_event"], int) and p["ts_init"] in (1000, 2000), kind
    life = [(k, p["from"], p["to"], p["ts_event"], p["ts_init"]) for k, p in r.journal if k.startswith("order_")]
    assert life == [("order_submitted", "INITIALIZED", "SUBMITTED", 1000, 1000),
                    ("order_accepted", "SUBMITTED", "ACCEPTED", 1000, 1000),
                    ("order_partially_filled", "ACCEPTED", "PARTIALLY_FILLED", 1500, 2000),
                    ("order_filled", "PARTIALLY_FILLED", "FILLED", 1600, 2000)]


def test_reject_ends_rejected_with_the_venue_reason():
    r = Rig()
    r.broker.submit_kind, r.broker.submit_reason = "rejected", "insufficient balance"
    o = r.submit()
    assert o.status.value == "REJECTED" and o.reason == "insufficient balance" and o.is_closed
    assert r.kinds() == ["order_submitted", "order_rejected"]
    n = len(r.journal)  # a simulated fill for a rejected order is journalled and ignored (NT-295)
    r.eng.apply(r.fill(o, 1.0, tid="stray"))
    assert r.kinds()[n:] == ["fill_after_refusal"] and r.journal[n][1]["trade_id"] == "stray"
    assert r.pf.positions == {} and o.status.value == "REJECTED"
    live = Rig(kind=AccountKind.LIVE)  # LIVE: the venue fact is booked (NT-295 repair 2, point D; see test_295_d2_*)
    live.broker.submit_kind = "rejected"
    lo = live.submit()
    live.eng.apply(live.fill(lo, 1.0))
    assert lo.status.value == "REJECTED" and live.pf.position(X, "s").qty == 1.0 and "venue_fill_on_rejected" in live.kinds()


def test_cancel_from_accepted_and_from_partially_filled():
    r = Rig()
    a = r.submit()
    assert r.eng.cancel(a.client_id).status.value == "CANCELED" and r.broker.canceled == [a.client_id]
    b = r.submit()
    r.eng.apply(r.fill(b, 1.0))
    assert b.status.value == "PARTIALLY_FILLED"
    r.eng.cancel(b.client_id)
    assert b.status.value == "CANCELED" and b.filled_qty == 1.0
    assert r.eng.cancel(b.client_id) is b and r.broker.canceled == [a.client_id, b.client_id]  # closed: no second call


def test_triggered_expired_and_the_unlisted_edges():
    r = Rig()
    o = r.submit()
    r.eng.apply(OrderEvent(1100, 1100, "triggered", o))
    assert o.status.value == "TRIGGERED"
    r.eng.apply(r.fill(o, 0.5))
    assert o.status.value == "PARTIALLY_FILLED"
    r.eng.apply(OrderEvent(1200, 1200, "expired", o))
    assert o.status.value == "EXPIRED" and o.is_closed
    p = r.submit()
    with pytest.raises(IllegalTransition):  # ACCEPTED -> ACCEPTED is not an edge
        r.eng.apply(OrderEvent(1100, 1100, "accepted", p))
    assert p.status.value == "ACCEPTED"
    with pytest.raises(IllegalTransition):  # EXPIRED -> CANCELED is not an edge
        r.eng.apply(OrderEvent(1300, 1300, "canceled", o))
    assert o.status.value == "EXPIRED"
    r.broker.submit_kind = "triggered"
    with pytest.raises(IllegalTransition):  # SUBMITTED -> TRIGGERED is not an edge
        r.submit()
    with pytest.raises(ValueError):
        r.eng.apply(OrderEvent(1, 1, "weird", p))


def test_late_fill_after_cancel_is_applied_to_the_portfolio():
    r = Rig()
    o = r.submit(qty=2.0)
    r.eng.cancel(o.client_id)
    assert o.status.value == "CANCELED"
    n = len(r.journal)
    r.eng.apply(r.fill(o, 0.5, ts=1700))
    assert o.status.value == "PARTIALLY_FILLED" and o.filled_qty == 0.5 and r.pf.position(X, "s").qty == 0.5
    assert [k for k, _ in r.journal[n:]] == ["late_fill", "fill", "position", "account"]
    assert r.journal[n][1]["from"] == "CANCELED" and r.journal[n][1]["to"] == "PARTIALLY_FILLED"
    n2 = len(r.journal)
    r.eng.apply(r.fill(o, 1.5, ts=1800))
    assert o.status.value == "FILLED" and r.pf.position(X, "s").qty == 2.0
    assert [k for k, _ in r.journal[n2:]] == ["late_fill", "fill", "position", "account"]  # (g) not order_filled

    q = r.submit(qty=1.0)
    r.eng.cancel(q.client_id)
    r.eng.apply(r.fill(q, 1.0))
    assert q.status.value == "FILLED"  # CANCELED -> FILLED straight away
    # (g) every fill of a dead order is a late fill: o's 0.5 and 1.5, q's 1.0
    assert r.kinds().count("late_fill") == 3 and "order_filled" not in r.kinds()[n:]


def test_late_fill_beyond_the_quantity_is_an_overfill():
    r = Rig()
    o = r.submit(qty=1.0)
    r.eng.cancel(o.client_id)
    with pytest.raises(OverfillError):
        r.eng.apply(r.fill(o, 1.5))
    assert o.status.value == "CANCELED" and r.pf.positions == {}


# ------------------------------------------------------------------ duplicates and overfill
def test_duplicate_trade_id_is_ignored_entirely():
    r = Rig()
    o = r.submit(qty=2.0)
    r.eng.apply(r.fill(o, 1.0, tid="T1"))
    cash, nj, ns = r.pf.cash, len(r.journal), len(r.seen)
    r.eng.apply(r.fill(o, 1.0, tid="T1"))
    assert (r.pf.cash, len(r.journal), len(r.seen)) == (cash, nj, ns)
    assert o.filled_qty == 1.0 and o.status.value == "PARTIALLY_FILLED" and r.pf.position(X, "s").qty == 1.0
    r.eng.apply(r.fill(o, 1.0, tid="T2"))
    assert o.status.value == "FILLED"
    n = len(r.journal)
    r.eng.apply(r.fill(o, 1.0, tid="T2"))  # a known trade for a FILLED order: ignored
    assert len(r.journal) == n and r.pf.position(X, "s").qty == 2.0
    with pytest.raises(OverfillError):  # an unknown one: overfill
        r.eng.apply(r.fill(o, 1.0, tid="T3"))


def test_overfill_raises_and_leaves_everything_unchanged():
    r = Rig()
    o = r.submit(qty=2.0)
    r.eng.apply(r.fill(o, 1.5, tid="A"))
    snap = (o.status, o.filled_qty, o.avg_fill_price, r.pf.cash, len(r.journal), len(r.seen))
    with pytest.raises(OverfillError):
        r.eng.apply(r.fill(o, 1.0, tid="B"))
    assert snap == (o.status, o.filled_qty, o.avg_fill_price, r.pf.cash, len(r.journal), len(r.seen))
    r.eng.apply(r.fill(o, 0.5, tid="B"))  # the failed trade id was not recorded as seen
    assert o.status.value == "FILLED"


def test_fill_for_an_unknown_order_or_a_mismatched_one():
    r = Rig()
    o = r.submit()
    ghost = dataclasses.replace(o, client_id="nope")
    with pytest.raises(UnknownOrder):
        r.eng.apply(r.fill(ghost, 1.0))
    with pytest.raises(ValueError):
        r.eng.apply(dataclasses.replace(r.fill(o, 1.0), side="SELL"))


# ------------------------------------------------------------------ commands -> RiskGuard -> Broker
def test_denial_is_a_rejected_order_never_a_broker_call():
    r = Rig(RiskLimits(max_order_notional=150.0, max_position_notional=None, max_orders_per_minute=None,
                       daily_loss_limit=None))
    o = r.submit(qty=2.0, price=100.0)
    assert o.status.value == "DENIED" and o.reason == "max_order_notional" and o.is_closed
    assert r.broker.submitted == []
    assert r.kinds() == ["order_denied"]
    p = r.journal[0][1]
    assert (p["from"], p["to"], p["reason"], p["ts_event"], p["ts_init"]) == ("INITIALIZED", "DENIED",
                                                                              "max_order_notional", 1000, 1000)
    assert p["order"]["reason"] == "max_order_notional"
    assert [t for t, _ in r.seen] == ["order"] and r.seen[0][1].kind == "denied"
    ok = r.submit(qty=1.0, price=100.0)
    assert ok.status.value == "ACCEPTED" and len(r.broker.submitted) == 1


def test_no_reference_price_is_a_denial():
    r = Rig()
    r.eng.marks.clear()
    o = r.submit(type="MARKET")
    assert o.status.value == "DENIED" and o.reason == "missing_marks" and r.broker.submitted == []


def test_resubmitting_a_client_id_returns_the_existing_order_or_names_the_difference():
    r = Rig()
    o = r.submit(client_id="same")
    n = len(r.journal)
    again = r.eng.submit(dataclasses.replace(o, status="INITIALIZED"))
    assert again is o and len(r.broker.submitted) == 1 and len(r.journal) == n
    for field, value in (("qty", 9.0), ("price", 99.0), ("side", "SELL"), ("strategy_id", "t"), ("type", "MARKET")):
        with pytest.raises(DuplicateClientId, match=field):
            r.eng.submit(dataclasses.replace(o, **{field: value}))
    assert len(r.broker.submitted) == 1 and len(r.journal) == n and issubclass(DuplicateClientId, ValueError)
    r.eng.marks.clear()
    d = r.submit(client_id="den", type="MARKET")
    assert d.status.value == "DENIED" and r.eng.submit(dataclasses.replace(d, status="INITIALIZED")) is d


def test_the_broker_gets_a_snapshot_with_the_risk_rounded_quantity():
    r = Rig()
    r.risk.instrument = dataclasses.replace(r.risk.instrument, size_increment=0.1, price_increment=0.5)
    o = r.submit(qty=1.234, price=100.26)
    sent = r.broker.submitted[0]
    assert (sent.qty, sent.price, sent.status.value) == (1.2, 100.5, "SUBMITTED") and sent is not o
    assert (o.qty, o.price) == (1.2, 100.5)


def test_halt_calls_riskguard_cancels_resting_orders_and_denies_new_ones():
    r = Rig()
    a, b, c = r.submit(), r.submit(), r.submit()
    r.eng.apply(r.fill(b, 1.0))  # PARTIALLY_FILLED rests too
    r.eng.cancel(c.client_id)
    r.eng.handle(Halt("kill switch"))
    assert r.risk.state is TradingState.HALTED and r.risk.state_reason == "kill switch"
    assert (a.status.value, b.status.value, c.status.value) == ("CANCELED",) * 3
    assert r.broker.canceled == [c.client_id, a.client_id, b.client_id]
    assert ("halt", 1000) == (r.journal[[k for k in r.kinds()].index("halt")][0],
                              r.journal[[k for k in r.kinds()].index("halt")][1]["ts_event"])
    n = len(r.broker.submitted)
    d = r.submit()
    assert d.status.value == "DENIED" and d.reason == "halted" and len(r.broker.submitted) == n


def test_halt_keeps_cancelling_when_one_cancel_fails():
    r = Rig()
    a, b = r.submit(), r.submit()
    del r.broker.venue[a.client_id]  # the broker blows up on a
    r.eng.halt("x")
    assert a.status.value == "ACCEPTED" and b.status.value == "CANCELED"
    assert "halt_cancel_failed" in r.kinds()


def test_cancel_all_by_account_and_instrument():
    r = Rig()
    a, b = r.submit(), r.submit()
    with pytest.raises(UnknownAccount):
        r.eng.cancel_all("other")
    assert a.status.value == "ACCEPTED"
    assert r.eng.cancel_all("acc", "OTHER-INSTRUMENT") == []
    out = r.eng.handle(CancelAll("acc"))
    assert [o.client_id for o in out] == [a.client_id, b.client_id]
    assert (a.status.value, b.status.value) == ("CANCELED", "CANCELED")
    assert r.eng.cancel_all("acc") == []
    with pytest.raises(UnknownOrder):
        r.eng.handle(CancelOrder("nope"))


def test_cancel_of_a_submitted_order_has_no_edge_and_makes_no_broker_call():
    r = Rig()
    r.broker.submit_kind = "accepted"
    o = r.submit()
    o.status = type(o.status)("SUBMITTED")  # only the engine moves it; forced here to reach the guard
    with pytest.raises(IllegalTransition):
        r.eng.cancel(o.client_id)
    assert r.broker.canceled == []


def test_unknown_command_and_foreign_account():
    r = Rig()
    with pytest.raises(TypeError):
        r.eng.handle("nonsense")
    with pytest.raises(ValueError):
        r.eng.submit(Order(client_id="z", instrument_id=X, account_id="elsewhere", side="BUY", type="MARKET", qty=1))


def test_committed_reduce_qty_comes_from_the_engines_resting_reduce_orders():
    r = Rig()
    long = r.submit(side="BUY", qty=10.0)
    r.eng.apply(r.fill(long, 10.0))
    s1 = r.submit(side="SELL", qty=6.0, price=110.0)
    assert r.checks[-1][1]["committed_reduce_qty"] == 0.0 and s1.qty == 6.0
    s2 = r.submit(side="SELL", qty=6.0, price=111.0)  # only 4 of the position is left to reduce
    assert r.checks[-1][1]["committed_reduce_qty"] == 6.0
    assert s2.qty == 4.0 and r.broker.submitted[-1].qty == 4.0 and r.checks[-1][2].reason == "clipped_to_position"
    s3 = r.submit(side="SELL", qty=1.0, price=112.0)
    assert r.checks[-1][1]["committed_reduce_qty"] == 10.0
    assert s3.status.value == "DENIED" and s3.reason == "reduce_exceeds_position"
    r.eng.cancel(s1.client_id)  # a cancelled order releases its claim
    s4 = r.submit(side="SELL", qty=3.0, price=113.0)
    assert r.checks[-1][1]["committed_reduce_qty"] == 4.0 and s4.status.value == "ACCEPTED"
    r.eng.apply(r.fill(s2, 1.0))  # a part-filled order claims only what is left of it
    r.submit(side="SELL", qty=1.0, price=114.0)
    assert r.checks[-1][1]["committed_reduce_qty"] == pytest.approx(3.0 + 3.0)


def test_size_frac_is_sized_by_riskguard_on_realised_equity_not_cash():
    r = Rig()
    o = r.submit(qty=10.0, type="MARKET")  # a long of 10 at 100: cash 9000, realised equity 10000
    r.eng.apply(r.fill(o, 10.0))
    r.eng.set_marks({X: 110.0})
    r.submit(side="BUY", qty=None, size_frac=0.5, type="MARKET")
    cid, kw, d = r.checks[-1]
    assert kw["ref_price"] == 110.0 and d.qty == pytest.approx(0.5 * 10_000.0 / 110.0)  # cash alone: 40.9
    sent = r.broker.submitted[-1]
    assert sent.qty is None and sent.size_frac == 0.5 and r.eng.order(cid).status.value == "ACCEPTED"


def test_close_position_goes_through_riskguard_as_a_reduce_only_market_order():
    r = Rig()
    o = r.submit(qty=3.0)
    assert r.eng.handle(ClosePosition("X-s")) is None  # flat
    r.eng.apply(r.fill(o, 3.0))
    c = r.eng.handle(ClosePosition("X-s", "manual"))
    sent = r.broker.submitted[-1]
    assert (sent.side.value, sent.type.value, sent.qty, sent.reduce_only) == ("SELL", "MARKET", 3.0, True)
    assert c.tags == {"tag": "manual", "reason": "CLOSE"} and c.status.value == "ACCEPTED"
    r.eng.halt("stop")
    d = r.eng.close_position("X-s")  # HALTED denies everything, the engine's own exits included
    assert d.status.value == "DENIED" and d.reason == "halted"


# ------------------------------------------------------------------ bus order and portfolio
def test_a_fill_emits_order_fill_position_account_in_that_order_after_the_portfolio_moved():
    r = Rig()
    o = r.submit(qty=3.0)
    r.seen.clear()
    r.bus.subscribe("fill", lambda e: r.seen.append(("cash_at_fill", r.pf.cash)))
    r.eng.apply(r.fill(o, 1.0))
    assert [t for t, _ in r.seen] == ["order", "fill", "cash_at_fill", "position", "account"]
    assert r.seen[2][1] == 9_900.0
    assert isinstance(r.seen[1][1], FillEvent) and r.seen[0][1].kind == "partially_filled"
    acct = r.seen[-1][1].account
    assert (acct.cash, acct.equity, acct.account_id) == (9_900.0, 10_000.0, "acc")
    assert r.seen[-1][1].ts_init == 1000 and r.seen[-1][1].ts_event == 1500
    s = r.submit(side="SELL", qty=3.0, price=100.0)  # margin-like instrument: 3 against 1 held flips to short 2
    r.seen.clear()
    r.eng.apply(r.fill(s, 3.0))
    assert [t for t, _ in r.seen if t != "cash_at_fill"] == ["order", "fill", "position", "position", "account"]  # close, then open short


def test_brackets_follow_a5_3():
    r = Rig()
    b1 = {"take_profit": 110.0, "stop_loss": 95.0, "tp_is_offset": False, "max_hold": None}
    b2 = {"take_profit": 120.0, "stop_loss": 99.0, "tp_is_offset": False, "max_hold": 5}
    a = r.submit(qty=2.0, bracket=b1)
    r.eng.apply(r.fill(a, 2.0))
    assert r.pf.get_bracket("X-s") == b1
    plain = r.submit(qty=1.0)
    r.eng.apply(r.fill(plain, 1.0))
    assert r.pf.get_bracket("X-s") == b1  # no bracket on the adding order keeps the old one
    c = r.submit(qty=1.0, bracket=b2)
    r.eng.apply(r.fill(c, 1.0))
    assert r.pf.get_bracket("X-s") == b2  # last fill wins
    rep = [p for k, p in r.journal if k == "bracket"]
    assert len(rep) == 2 and rep[1]["old"] == b1 and rep[1]["new"] == b2 and rep[0]["old"] is None
    d = r.submit(side="SELL", qty=1.0, bracket={"take_profit": 90.0, "stop_loss": 105.0, "tp_is_offset": False})
    r.eng.apply(r.fill(d, 1.0))
    assert r.pf.get_bracket("X-s") == b2 and len([k for k in r.kinds() if k == "bracket"]) == 2  # reduce keeps it
    e = r.submit(side="SELL", qty=3.0)
    r.eng.apply(r.fill(e, 3.0))
    assert r.pf.get_bracket("X-s") is None  # closed: the portfolio cleared it


def test_apply_all_and_the_spot_backstop_leaves_the_order_untouched():
    r = Rig(is_spot=True)
    o = r.submit(side="BUY", qty=1.0)
    s = r.submit(side="SELL", qty=1.0, price=101.0)  # the guard denies a spot sell with nothing held
    assert s.status.value == "DENIED" and s.reason == "spot_sell_exceeds_balance"
    r.eng.apply_all([r.fill(o, 1.0)])
    assert o.status.value == "FILLED" and r.pf.base_balance(X) == 1.0


# ------------------------------------------------------------------ crash ordering (journal first, handlers isolated)
def test_every_record_is_journalled_before_its_event_is_published():
    r = Rig()
    o = r.submit(qty=2.0)
    n, at = len(r.journal), {}
    for topic, kind in (("order", "order_filled"), ("fill", "fill"), ("position", "position"), ("account", "account")):
        r.bus.subscribe(topic, lambda e, k=kind: at.setdefault(k, k in r.kinds()[n:]))
    r.eng.apply(r.fill(o, 2.0))
    assert at == {"order_filled": True, "fill": True, "position": True, "account": True}


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("topic", ["order", "fill", "position", "account"])
def test_a_raising_subscriber_loses_no_record_and_is_recorded(topic, strict):
    r = Rig(strict_bus=strict)
    o = r.submit(qty=2.0)
    n = len(r.journal)

    def boom(e):
        raise RuntimeError("boom")

    r.bus.subscribe(topic, boom)
    r.eng.apply(r.fill(o, 2.0))
    assert o.status.value == "FILLED" and r.pf.position(X, "s").qty == 2.0
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account"]
    if strict:  # a strict bus re-raises: the engine records it
        assert [type(e) for e in r.eng.handler_errors] == [RuntimeError] and not r.bus.errors
    else:  # the bus isolates it (NT-294)
        assert r.eng.handler_errors == [] and [e.topic for e in r.bus.errors] == [topic]
    for _ in range(150):
        r.eng._record_error(RuntimeError())
    assert len(r.eng.handler_errors) == 100


def test_a_journal_sink_that_raises_once_keeps_the_state_and_retries_in_order():
    r = Rig()
    o = r.submit(qty=2.0)
    n = len(r.journal)
    r.fail_once = "fill"
    r.eng.apply(r.fill(o, 2.0, tid="T"))
    assert o.status.value == "FILLED" and r.pf.position(X, "s").qty == 2.0
    assert r.eng.journal_failures == 1 and isinstance(r.eng.handler_errors[0], OSError)
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account"]  # retried before the next record
    r.eng.apply(r.fill(o, 2.0, tid="T"))  # re-delivery stays ignored
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account"] and r.pf.position(X, "s").qty == 2.0


def test_a_journal_that_is_down_queues_records_and_flushes_them_first_on_recovery():
    r = Rig()
    o = r.submit(qty=2.0)
    n = len(r.journal)
    r.down = True
    r.eng.apply(r.fill(o, 2.0))
    assert o.status.value == "FILLED" and r.kinds()[n:] == [] and r.eng.journal_failures >= 4
    r.down = False
    p = r.submit(qty=1.0)
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account", "order_submitted", "order_accepted"]
    assert p.status.value == "ACCEPTED"


# ------------------------------------------------------------------ broker failure, duplicates, reservations
def test_a_broker_exception_is_a_rejection_that_is_idempotent():
    r = Rig()
    r.broker.raise_on_submit = ConnectionError("down")
    o = r.submit(client_id="b1")
    assert o.status.value == "REJECTED" and o.reason == "broker_error:ConnectionError" and o.is_closed
    assert r.kinds() == ["order_submitted", "order_rejected"] and "b1" not in r.eng._reserved
    assert [e.kind for t, e in r.seen if t == "order"] == ["submitted", "rejected"]
    r.broker.raise_on_submit = None
    assert r.eng.submit(dataclasses.replace(o, status="INITIALIZED", reason="")) is o and len(r.broker.submitted) == 1
    assert r.eng.cancel("b1") is o and r.broker.canceled == []
    assert r.submit(client_id="b2").status.value == "ACCEPTED"


def test_committed_reduce_counts_only_the_part_that_reduces_the_position_per_strategy():
    r = Rig()
    a = r.submit(side="BUY", qty=2.0)
    r.eng.apply(r.fill(a, 2.0))
    r.submit(side="SELL", qty=6.0, price=110.0)  # a flip order: only 2 of it can reduce the long of 2
    s = r.submit(side="SELL", qty=1.0, price=111.0)
    assert r.checks[-1][1]["committed_reduce_qty"] == 2.0
    assert s.status.value == "DENIED" and s.reason == "reduce_exceeds_position"
    t = r.submit(side="BUY", qty=10.0, strategy="t")
    r.eng.apply(r.fill(t, 10.0))
    u = r.submit(side="SELL", qty=6.0, price=113.0, strategy="t")  # s's resting sells are not t's
    assert r.checks[-1][1]["committed_reduce_qty"] == 0.0 and u.qty == 6.0 and u.status.value == "ACCEPTED"


def test_cancel_all_and_halt_do_not_cancel_twice_after_a_late_fill():
    r = Rig()
    o = r.submit(qty=2.0)
    r.eng.cancel(o.client_id)
    r.eng.apply(r.fill(o, 0.5))
    assert o.status.value == "PARTIALLY_FILLED" and o.client_id in r.eng.cancel_acked
    assert r.eng.cancel_all("acc") == [] and r.broker.canceled == [o.client_id]
    r.eng.halt("x")
    assert r.broker.canceled == [o.client_id]


# ------------------------------------------------------------------ modify
def test_modify_rechecks_calls_the_broker_journals_and_publishes():
    r = Rig()
    o = r.submit(qty=2.0)
    r.seen.clear()
    assert r.eng.handle(ModifyOrder(o.client_id, price=99.5, qty=3.0)) is o
    assert r.broker.modified == [{"price": 99.5, "trigger_price": None, "qty": 3.0}]
    assert (o.price, o.qty, o.status.value) == (99.5, 3.0, "ACCEPTED") and r.eng._reserved[o.client_id] == 3.0
    k, p = r.journal[-1]
    assert (k, p["from"], p["to"], p["order"]["price"], p["ts_init"]) == ("order_modified", "ACCEPTED", "ACCEPTED", 99.5, 1000)
    assert [(t, e.kind) for t, e in r.seen] == [("order", "modified")]
    r.eng.apply(r.fill(o, 3.0))
    assert o.status.value == "FILLED"  # fills are checked against the new quantity


def test_modify_denied_leaves_the_order_unchanged_and_journals_the_denial():
    r = Rig(RiskLimits(max_order_notional=150.0, max_position_notional=None, max_orders_per_minute=None,
                       daily_loss_limit=None))
    o = r.submit(qty=1.0)
    n = len(r.journal)
    assert r.eng.modify(o.client_id, qty=2.0) is o
    assert (o.qty, o.price) == (1.0, 100.0) and r.broker.modified == []
    assert r.kinds()[n:] == ["order_modify_denied"] and r.journal[-1][1]["reason"] == "max_order_notional"


def test_modify_does_not_count_the_order_itself_as_committed_and_checks_status_and_qty():
    r = Rig()
    a = r.submit(side="BUY", qty=10.0)
    r.eng.apply(r.fill(a, 10.0))
    s = r.submit(side="SELL", qty=10.0, price=110.0)
    r.eng.modify(s.client_id, price=111.0)
    assert r.checks[-1][1]["committed_reduce_qty"] == 0.0 and s.price == 111.0 and r.broker.modified
    r.eng.apply(r.fill(s, 4.0))
    with pytest.raises(ValueError):
        r.eng.modify(s.client_id, qty=4.0)  # not above what is filled
    r.eng.modify(s.client_id, qty=8.0)
    assert s.qty == 8.0
    with pytest.raises(ValueError):
        r.eng.modify(a.client_id, price=1.0)  # FILLED
    m = r.submit(side="BUY", qty=None, size_frac=0.1, type="MARKET")
    with pytest.raises(ValueError):
        r.eng.modify(m.client_id, qty=1.0)


# ------------------------------------------------------------------ size_frac, owner thread, small things
def test_size_frac_first_fill_is_terminal_unless_the_broker_resolves_the_quantity():
    r = Rig()
    o = r.submit(qty=None, size_frac=0.5, type="MARKET")
    r.eng.apply(r.fill(o, 1.0))
    assert o.status.value == "FILLED"
    with pytest.raises(OverfillError):
        r.eng.apply(r.fill(o, 1.0))
    r.broker.accept_tags = {"resolved_qty": 9.0}  # the old string tag is no longer read
    t = r.submit(qty=None, size_frac=0.5, type="MARKET")
    assert t.client_id not in r.eng._resolved
    r.broker.accept_tags, r.broker.resolved_qty = {}, 4.0
    q = r.submit(qty=None, size_frac=0.5, type="MARKET")
    assert q.qty is None and q.size_frac == 0.5 and r.eng._resolved[q.client_id] == 4.0
    r.eng.apply(r.fill(q, 1.0))
    assert q.status.value == "PARTIALLY_FILLED"
    r.eng.apply(r.fill(q, 3.0))
    assert q.status.value == "FILLED" and q.qty is None  # the engine sets nothing else on the Order
    for bad in (0.0, -1.0, float("nan"), float("inf")):  # refused before anything changes
        r.broker.resolved_qty = bad
        with pytest.raises(ValueError, match="resolved_qty"):
            r.submit(qty=None, size_frac=0.5, type="MARKET")
    r.broker.resolved_qty = None
    p = r.submit(qty=None, size_frac=0.5, type="MARKET")
    r.eng.apply(r.fill(p, 3.0))
    with pytest.raises(OverfillError):
        r.eng.apply(r.fill(p, 1.5))


def test_the_engine_has_one_owner_thread():
    r = Rig()
    errs = []

    def other(eng):
        for call in (lambda: eng.set_marks({X: 1.0}), lambda: eng.halt("x"), lambda: eng.cancel_all()):
            try:
                call()
            except RuntimeError as e:
                errs.append(e)

    t = threading.Thread(target=other, args=(r.eng,))
    t.start()
    t.join()
    assert len(errs) == 3 and r.eng.marks[X] == 100.0 and r.risk.state is TradingState.ACTIVE
    free = ExecutionEngine(r.broker, r.pf, r.risk, r.bus, r.clock, r._sink, check_owner=False)
    free.set_marks({X: 5.0})
    t = threading.Thread(target=lambda: free.set_marks({X: 6.0}))
    t.start()
    t.join()
    assert free.marks[X] == 6.0


def test_an_empty_trade_id_is_refused():
    r = Rig()
    o = r.submit(qty=2.0)
    n = len(r.journal)
    with pytest.raises(ValueError):
        r.eng.apply(dataclasses.replace(r.fill(o, 1.0), trade_id=""))
    assert o.filled_qty == 0.0 and len(r.journal) == n and r.pf.positions == {}


# ------------------------------------------------------------------ repair round 2: dead orders after a late fill
def test_a_late_partial_fill_after_a_cancel_ack_does_not_block_the_close():
    """QA repro (nt224b_late.py): long 2, resting SELL 4, cancel acked, late fill 1, then close the remaining 1."""
    r = Rig()
    b = r.submit(side="BUY", qty=2.0, type="MARKET")
    r.eng.apply(r.fill(b, 2.0))
    s = r.submit(side="SELL", qty=4.0, price=100.0)
    assert s.qty == 4.0 and s.status.value == "ACCEPTED"
    r.eng.cancel(s.client_id)
    r.eng.apply(r.fill(s, 1.0, ts=1700))  # the venue fact: one more was sold before the cancel took
    assert s.status.value == "PARTIALLY_FILLED" and s.filled_qty == 1.0
    assert r.pf.signed_qty(r.pf.position(X, "s")) == 1.0
    assert r.eng.open_orders() == [] and not r.eng.is_live(s) and s.client_id not in r.eng._reserved
    assert r.eng.cancel(s.client_id) is s and r.broker.canceled == [s.client_id]  # dead: never cancelled again
    with pytest.raises(ValueError, match="cancel acknowledged"):
        r.eng.modify(s.client_id, price=101.0)
    c = r.eng.close_position("X-s")
    assert r.checks[-1][1]["committed_reduce_qty"] == 0.0
    assert (c.status.value, c.reason, c.qty, c.side.value) == ("ACCEPTED", "", 1.0, "SELL")
    r.eng.apply(r.fill(s, 3.0, ts=1800))  # a later late fill takes the dead order to FILLED (venue fact)
    assert s.status.value == "FILLED" and r.pf.signed_qty(r.pf.position(X, "s")) == -2.0


def test_committed_reduce_ignores_same_side_orders():
    r = Rig()
    a = r.submit(side="BUY", qty=4.0)
    r.eng.apply(r.fill(a, 4.0))
    r.submit(side="BUY", qty=3.0, price=99.0)  # resting on the position's own side: it adds, it does not reduce
    s = r.submit(side="SELL", qty=4.0, price=101.0)
    assert r.checks[-1][1]["committed_reduce_qty"] == 0.0 and s.qty == 4.0 and s.status.value == "ACCEPTED"


def test_open_orders_is_a_snapshot_list():
    r = Rig()
    a = r.submit()
    snap = r.eng.open_orders()
    b = r.submit()
    assert snap == [a] and r.eng.open_orders() == [a, b] and r.eng.open_orders() is not r.eng.open_orders()


# ------------------------------------------------------------------ journal bound
class _ErrLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def errlog():
    import neural_trade.desk.core.execution as ex
    h = _ErrLog()
    ex.log.addHandler(h)
    yield h
    ex.log.removeHandler(h)


def test_a_journal_outage_halts_new_orders_once_above_the_limit_and_drops_nothing(errlog):
    caplog = errlog
    r = Rig()
    r.eng._journal.limit = 5
    r.down = True
    orders = [r.submit(qty=1.0) for _ in range(3)]  # 2 records each
    # 6 records + the halt itself + the cancels of the 3 resting opening orders (order_canceled each)
    assert r.eng.journal_pending == 10 and r.risk.state is TradingState.HALTED
    assert r.risk.state_reason == "journal_unavailable" and r.eng.open_orders() == []
    assert r.eng.journal_failures == 9
    d = r.submit(qty=1.0)  # an 11th pending record: still one halt record, one log line
    assert d.status.value == "DENIED" and d.reason == "halted" and r.eng.journal_pending == 11
    assert sum("halting new orders" in m for m in caplog.messages) == 1
    assert sum("journal sink failed" in m for m in caplog.messages) == 1  # first failure only (then every 1000th)
    r.down = False
    r.eng.apply(r.fill(orders[0], 1.0))
    assert r.eng.journal_pending == 0
    assert r.kinds()[:11] == ["order_submitted", "order_accepted"] * 3 + ["halt"] + ["order_canceled"] * 3 + ["order_denied"]
    assert r.journal[6][1]["reason"] == "journal_unavailable"
    # the cancelled order's late fill opens exposure while HALTED: refused by the binding check (journal only)
    assert r.kinds()[-1] == "fill_refused" and not r.eng._journal.overflowed


def test_journal_failure_logging_is_rate_limited(errlog):
    import neural_trade.desk.core.execution as ex
    caplog = errlog

    def sink(kind, payload):
        raise OSError("x")

    fails = []
    w = ex.JournalWriter(sink, on_error=fails.append, on_overflow=lambda: None, limit=10**9)
    for i in range(2001):
        w.write("k", {"i": i})
    assert w.failures == 2001 and w.pending == 2001 and len(fails) == 2001
    assert sum("journal sink failed" in m for m in caplog.messages) == 3  # failures 1, 1001, 2001


# ------------------------------------------------------------------ broker failures in cancel_all / modify
def test_cancel_all_isolates_a_failing_cancel_and_reports_it():
    r = Rig()
    a, b, c = r.submit(), r.submit(), r.submit()
    r.broker.raise_on_cancel = {b.client_id: ConnectionError("down")}
    out = r.eng.cancel_all("acc")
    assert [o.client_id for o in out] == [a.client_id, c.client_id]
    assert (a.status.value, b.status.value, c.status.value) == ("CANCELED", "ACCEPTED", "CANCELED")
    assert [(cid, type(e)) for cid, e in r.eng.last_cancel_failures] == [(b.client_id, ConnectionError)]
    assert isinstance(r.eng.handler_errors[-1], ConnectionError)
    rec = [p for k, p in r.journal if k == "cancel_failed"]
    assert len(rec) == 1 and rec[0]["client_id"] == b.client_id and rec[0]["reason"] == "broker_error:ConnectionError"
    r.broker.raise_on_cancel = {}
    assert [o.client_id for o in r.eng.cancel_all()] == [b.client_id] and r.eng.last_cancel_failures == []


def test_a_broker_exception_in_modify_is_journalled_and_raised_as_broker_error():
    from neural_trade.desk.core.execution import BrokerError
    r = Rig()
    o = r.submit(qty=2.0)
    cause = TimeoutError("slow")
    r.broker.raise_on_modify = cause
    with pytest.raises(BrokerError) as ei:
        r.eng.modify(o.client_id, price=99.0, qty=3.0)
    assert ei.value.__cause__ is cause and issubclass(BrokerError, RuntimeError)
    assert (o.price, o.qty, o.status.value) == (100.0, 2.0, "ACCEPTED") and r.eng._reserved[o.client_id] == 2.0
    k, p = r.journal[-1]
    assert (k, p["client_id"], p["reason"]) == ("order_modify_failed", o.client_id, "broker_error:TimeoutError")


# ------------------------------------------------------------------ identity, modify rounding and venue checks
def test_duplicate_client_id_names_reduce_only_and_bracket():
    r = Rig()
    o = r.submit(client_id="dup")
    with pytest.raises(DuplicateClientId, match="reduce_only"):
        r.eng.submit(dataclasses.replace(o, reduce_only=True))
    with pytest.raises(DuplicateClientId, match="bracket"):
        r.eng.submit(dataclasses.replace(o, bracket={"take_profit": 110.0, "stop_loss": 90.0}))


def test_modify_rounds_the_trigger_and_refuses_a_venue_qty_below_the_filled():
    r = Rig()
    r.risk.instrument = dataclasses.replace(r.risk.instrument, price_increment=0.5)
    o = r.submit(qty=2.0, type="STOP_LIMIT", price=100.0, trigger_price=100.0)
    r.eng.modify(o.client_id, price=101.26, trigger_price=100.74)
    assert r.broker.modified[-1] == {"price": 101.5, "trigger_price": 100.5, "qty": 2.0}
    assert (o.price, o.trigger_price) == (101.5, 100.5)
    r.eng.apply(r.fill(o, 1.5))
    r.broker.modified_qty_override = 1.0  # the venue reports a total below what is already filled
    n = len(r.journal)
    with pytest.raises(ValueError, match="below its filled"):
        r.eng.modify(o.client_id, price=102.0)
    assert (o.price, o.qty, o.status.value, len(r.journal)) == (101.5, 2.0, "PARTIALLY_FILLED", n)


# ------------------------------------------------------------------ fuzz: no close is denied without a live opposite order
@pytest.mark.parametrize("seed", range(20))
def test_fuzz_a_close_is_denied_only_while_a_live_opposite_order_exists(seed):
    import random
    rng = random.Random(seed)
    r = Rig(cash=1e6)
    errors = (IllegalTransition, OverfillError, UnknownOrder, ValueError)
    for step in range(200):
        orders = list(r.eng.orders.values())
        act = rng.choice(["submit", "submit", "oev", "fill", "fill", "cancel", "close"]) if orders else "submit"
        try:
            if act == "submit":
                r.submit(side=rng.choice(["BUY", "SELL"]), qty=rng.choice([1.0, 2.0, 3.0]),
                         strategy=rng.choice(["s1", "s2"]), reduce_only=rng.random() < 0.15)
            elif act == "oev":
                o = rng.choice(orders)
                r.eng.apply(OrderEvent(1, 1, rng.choice(["accepted", "triggered", "canceled", "expired"]), o))
            elif act == "fill":
                o = rng.choice(orders)
                rem = (o.qty or 1.0) - o.filled_qty
                r.eng.apply(r.fill(o, max(rng.choice([rem, rem / 2]), 0.01)))
            elif act == "cancel":
                r.eng.cancel(rng.choice(orders).client_id)
        except errors:
            pass
        if act != "close":
            continue
        for p in list(r.pf.positions.values()):
            if p.qty <= 1e-9:
                continue
            held = r.pf.signed_qty(p)
            c = r.eng.close_position(p.position_id)
            if c.reason == "reduce_exceeds_position":
                live = [o for o in r.eng.open_orders() if o.strategy_id == p.strategy_id
                        and o.side.value == ("SELL" if held > 0 else "BUY")]
                assert live, f"seed {seed} step {step}: close of {p.position_id} denied with no live opposite order"
        assert not any(o.client_id in r.eng.cancel_acked for o in r.eng.open_orders())


# ------------------------------------------------------------------ NT-295: commitments, the binding check, qa-deep items
PROF = CostProfile(name="p", fee_bps_taker=10.0, fee_bps_maker=5.0, half_spread_bps=4.0, slippage_bps=1.0)  # slip 5e-4


def _commit_kw(r):
    kw = r.checks[-1][1]
    return kw["committed_open_notional"], kw["committed_position_notional"]


def test_295_p1_committed_open_notional_is_the_live_opening_notional_with_fees_and_slip():
    r = Rig(costs=PROF)
    a = r.submit(side="BUY", qty=10.0, type="LIMIT", price=100.0)
    assert _commit_kw(r) == (0.0, 0.0)
    r.submit(side="BUY", qty=4.0, type="STOP_MARKET", price=None, trigger_price=110.0)
    assert _commit_kw(r) == (pytest.approx(1001.0), pytest.approx(1000.0))  # LIMIT: 1000 x (1 + 5bp maker + 5bp slip)
    r.submit(side="BUY", qty=1.0, type="MARKET", strategy="t")
    # + the stop at max(110, mark 100) x 4 = 440 x (1 + 10bp taker + 5bp slip) = 440.66; strategy t has no order yet
    assert _commit_kw(r) == (pytest.approx(1441.66), 0.0)
    r.submit(side="BUY", qty=1.0, type="LIMIT", price=100.0)
    # + the market order of t: 100 x 1.0015 = 100.15 (account only); the position of s is a + the stop = 1440
    assert _commit_kw(r) == (pytest.approx(1441.66 + 100.15), pytest.approx(1440.0))
    r.eng.cancel(a.client_id)  # a dead order commits nothing
    r.submit(side="BUY", qty=1.0, type="LIMIT", price=100.0)
    assert _commit_kw(r) == (pytest.approx(440.66 + 100.15 + 100.1), pytest.approx(440.0 + 100.0))


def test_295_p1_resting_reduce_orders_commit_only_what_exceeds_the_position():
    r = Rig(costs=PROF)
    b = r.submit(side="BUY", qty=10.0, type="MARKET")
    r.eng.apply(r.fill(b, 10.0))
    r.submit(side="SELL", qty=4.0, type="LIMIT", price=100.0)  # reduces 4 of the 10 held
    r.submit(side="SELL", qty=1.0, type="LIMIT", price=100.0)
    assert _commit_kw(r) == (0.0, 0.0)
    r.submit(side="SELL", qty=12.0, type="LIMIT", price=100.0)  # a flip: 5 left to reduce, 7 open a short
    assert _commit_kw(r) == (0.0, 0.0)
    r.submit(side="SELL", qty=1.0, type="LIMIT", price=100.0)
    assert _commit_kw(r) == (pytest.approx(700.0 * 1.001), pytest.approx(700.0))


def test_295_p1_the_costs_of_resting_orders_are_reserved_in_the_cash_check():
    r = Rig(is_spot=True, costs=PROF)
    r.submit(side="BUY", qty=99.0, type="LIMIT", price=100.0)  # 9900 x 1.001 = 9909.9 of the 10000 cash
    assert r.submit(side="BUY", qty=0.91, type="LIMIT", price=100.0).reason == "insufficient_cash"  # + 91.091 > 10000
    o = r.submit(side="BUY", qty=0.9, type="LIMIT", price=100.0)  # + 90.09 = 9999.99
    assert o.status.value == "ACCEPTED"


def test_295_p2_max_position_counts_only_the_orders_of_the_same_position():
    lim = dataclasses.replace(RiskLimits.unlimited(), max_position_notional=5_000.0)
    r = Rig(lim)
    assert r.submit(strategy="a", qty=30.0).status.value == "ACCEPTED"  # 3000
    assert r.submit(strategy="b", qty=40.0).status.value == "ACCEPTED"  # 4000: not 3000 + 4000 against 5000
    d = r.submit(strategy="a", qty=21.0)  # 2100 + 3000 > 5000
    assert (d.status.value, d.reason) == ("DENIED", "max_position")
    assert r.submit(strategy="a", qty=20.0).status.value == "ACCEPTED"  # exactly 5000
    assert _commit_kw(r) == (pytest.approx(3000.0 + 4000.0), pytest.approx(3000.0))


# ---- point 3: the binding check at the fill
def _stop_rig(kind=AccountKind.PAPER, **kw):
    r = Rig(is_spot=True, kind=kind, **kw)
    o = r.submit(side="BUY", qty=90.0, type="STOP_MARKET", price=None, trigger_price=100.0)  # valued at 100: 9000
    assert o.status.value == "ACCEPTED" and r.eng._checked[o.client_id] == 100.0
    return r, o


def test_295_p3_precheck_fill_allows_inside_the_tolerance_and_when_the_gap_still_fits():
    r, o = _stop_rig()
    n, nj = len(r.checks), len(r.journal)
    d = r.eng.precheck_fill(o.client_id, 100.00005)  # within 1e-6 relative: RiskGuard is not called
    assert (d.allowed, d.qty) == (True, 90.0) and len(r.checks) == n
    d = r.eng.precheck_fill(o.client_id, 105.0)  # 90 x 105 = 9450 <= 10000
    assert (d.allowed, d.qty) == (True, 90.0) and len(r.checks) == n + 1 and r.checks[-1][1]["record"] is False
    assert o.status.value == "ACCEPTED" and len(r.journal) == nj
    assert r.eng.precheck_fill(o.client_id, 90.0).allowed  # a fill below the checked price is never re-checked


def test_295_p3_a_stop_gap_that_breaks_the_check_refuses_the_fill_and_rejects_the_order():
    r, o = _stop_rig()
    n = len(r.journal)
    d = r.eng.precheck_fill(o.client_id, 120.0)  # 90 x 120 = 10800 > 10000 cash
    assert (d.allowed, d.reason) == (False, "insufficient_cash")
    assert (o.status.value, o.reason) == ("REJECTED", "fill_price_check_failed") and not r.eng.is_live(o)
    assert r.kinds()[n:] == ["fill_refused", "order_rejected"]
    assert r.journal[n][1] == {"ts_event": 1000, "ts_init": 1000, "client_id": o.client_id,
                               "reason": "fill_price_check_failed", "check": "insufficient_cash", "fill_price": 120.0,
                               "checked_price": 100.0, "applied": False, "dead": False, "filled_qty": 0.0}
    assert (r.journal[n + 1][1]["from"], r.journal[n + 1][1]["to"]) == ("ACCEPTED", "REJECTED")
    assert r.seen[-1][0] == "order" and r.seen[-1][1].kind == "rejected" and r.pf.positions == {}
    assert r.eng.precheck_fill(o.client_id, 120.0).reason == "order_not_live"


def test_295_p3_a_size_frac_order_is_sized_on_the_fill_price():
    r = Rig(is_spot=True)
    o = r.submit(side="BUY", qty=None, size_frac=1.0, type="STOP_MARKET", price=None, trigger_price=100.0)
    assert o.status.value == "ACCEPTED" and r.eng._reserved[o.client_id] == 100.0  # the estimate: 10000 / 100
    d = r.eng.precheck_fill(o.client_id, 120.0)
    assert d.allowed and d.qty == pytest.approx(10_000.0 / 120.0, rel=1e-12) and o.status.value == "ACCEPTED"
    # always re-checked, never qty None: at 100.00005 (inside the tolerance) the cash affords 10000 / 100.00005
    assert r.eng.precheck_fill(o.client_id, 100.00005).qty == pytest.approx(10_000.0 / 100.00005, rel=1e-12)
    # below the checked price: still sized at the fill on equity (NT-299 reverses decision F: not the reserved 100)
    assert r.eng.precheck_fill(o.client_id, 90.0).qty == pytest.approx(10_000.0 / 90.0, rel=1e-12)


def test_295_p3_an_unwanted_simulated_fill_is_refused_without_touching_the_books():
    r, o = _stop_rig()
    n = len(r.journal)
    r.eng.apply(r.fill(o, 90.0, price=120.0))
    assert (o.status.value, o.reason, o.filled_qty) == ("REJECTED", "fill_price_check_failed", 0.0)
    assert r.pf.positions == {} and r.pf.cash == 10_000.0 and r.risk.state is TradingState.ACTIVE
    assert r.kinds()[n:] == ["fill_refused", "order_rejected"]
    assert r.journal[n][1]["applied"] is False and r.journal[n][1]["check"] == "insufficient_cash"


def test_295_p3_a_partly_filled_order_whose_rest_is_refused_ends_canceled():
    r, o = _stop_rig()
    r.eng.apply(r.fill(o, 30.0, price=100.0))
    assert o.status.value == "PARTIALLY_FILLED"
    d = r.eng.precheck_fill(o.client_id, 120.0)  # 60 x 120 = 7200 > the 7000 cash left
    assert (d.allowed, d.reason) == (False, "insufficient_cash")
    assert (o.status.value, o.reason, o.filled_qty) == ("CANCELED", "fill_price_check_failed", 30.0)
    assert r.pf.position(X, "s").qty == 30.0


def test_295_p3_a_live_fill_that_fails_the_check_is_applied_journalled_and_halts():
    r, o = _stop_rig(kind=AccountKind.LIVE)
    n = len(r.journal)
    r.eng.apply(r.fill(o, 90.0, price=120.0))  # the venue filled it: the books follow
    assert o.status.value == "FILLED" and r.pf.position(X, "s").qty == 90.0 and r.pf.cash == -800.0
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account", "fill_refused", "halt"]
    rec = r.journal[n + 4][1]
    assert (rec["applied"], rec["reason"], rec["check"], rec["fill_price"]) == (True, "fill_price_check_failed",
                                                                                "insufficient_cash", 120.0)
    assert r.journal[n + 5][1]["reason"] == "fill_price_check_failed"
    assert r.risk.state is TradingState.HALTED and r.risk.state_reason == "fill_price_check_failed"


def test_295_p3_an_ordinary_fill_is_not_rechecked_and_precheck_is_refused_inside_a_handler():
    r, o = _stop_rig()
    n = len(r.checks)
    r.eng.apply(r.fill(o, 90.0, price=100.0))
    assert o.status.value == "FILLED" and len(r.checks) == n and r.risk.state is TradingState.ACTIVE
    p = r.submit(qty=1.0, price=100.0)
    r.bus.subscribe("order", lambda e: r.eng.precheck_fill(p.client_id, 1.0))
    r.eng.apply(r.fill(p, 1.0))  # the handler's RuntimeError is isolated by the bus
    assert [type(e.exc) for e in r.bus.errors] == [RuntimeError] and "bus handler" in str(r.bus.errors[0].exc)


# ---- the qa-deep items of NT-224
def test_295_a_submit_sends_a_trigger_rounded_to_the_tick():
    r = Rig()
    r.risk.instrument = dataclasses.replace(r.risk.instrument, price_increment=0.01)
    o = r.submit(qty=2.0, type="STOP_LIMIT", price=100.0, trigger_price=100.006, client_id="stp")
    sent = r.broker.submitted[0]
    assert (sent.price, sent.trigger_price, o.trigger_price) == (100.0, 100.01, 100.01)
    assert r.eng.submit(Order(client_id="stp", instrument_id=X, account_id="acc", strategy_id="s", side="BUY",
                              type="STOP_LIMIT", qty=2.0, price=100.0, trigger_price=100.006)) is o  # as submitted


def test_295_b_a_failing_halt_hook_never_leaves_a_fill_half_written():
    def boom(old, new, reason):
        raise RuntimeError("hook")

    r = Rig(on_state_change=boom)
    o = r.submit(qty=2.0)
    r.eng._journal.limit = 1
    r.down = True
    r.eng.apply(r.fill(o, 2.0))  # the 2nd pending record (fill) overflows while the hook raises
    assert o.status.value == "FILLED" and r.pf.position(X, "s").qty == 2.0 and r.risk.state is TradingState.HALTED
    assert any(str(e) == "hook" for e in r.eng.handler_errors)
    r.down = False
    r.submit(qty=1.0)  # denied (halted): flushes the queue
    assert r.kinds() == ["order_submitted", "order_accepted", "order_filled", "fill", "halt", "position", "account",
                         "order_denied"]
    assert r.journal[4][1]["reason"] == "journal_unavailable"


def test_295_b_the_journal_halt_is_reasserted_after_a_resume_during_the_outage():
    r = Rig()
    r.eng._journal.limit = 1
    r.down = True
    r.submit(qty=1.0)
    assert r.risk.state is TradingState.HALTED
    r.risk.resume("operator")  # the sink is still down
    assert r.risk.state is TradingState.ACTIVE
    r.submit(qty=1.0)
    assert r.risk.state is TradingState.HALTED and r.risk.state_reason == "journal_unavailable"
    r.down = False
    r.submit(qty=1.0)
    assert r.kinds().count("halt") == 2  # each state change is journalled


def test_295_c_a_handler_that_applies_a_fill_is_queued_behind_the_current_one():
    r = Rig()
    a, b = r.submit(qty=1.0), r.submit(qty=1.0)
    fb, got = r.fill(b, 1.0), []

    def on_order(e):
        if e.kind == "filled" and e.order.client_id == a.client_id:
            got.append(r.eng.apply(fb))  # returns None: queued

    r.bus.subscribe("order", on_order)
    n = len(r.journal)
    r.eng.apply(r.fill(a, 1.0))

    def tag(k, p):
        if k == "order_filled":
            return f"{k}:{p['order']['client_id']}"
        return f"{k}:{p['fill']['client_id']}" if k == "fill" else k

    assert [tag(k, p) for k, p in r.journal[n:]] == [
        f"order_filled:{a.client_id}", f"fill:{a.client_id}", "position", "account",
        f"order_filled:{b.client_id}", f"fill:{b.client_id}", "position", "account"]
    assert got == [None] and b.status.value == "FILLED" and r.pf.position(X, "s").qty == 2.0
    assert r.eng.handler_errors == [] and r.eng._deferred == deque()


def test_295_c_queued_commands_run_in_call_order_and_their_errors_are_recorded():
    r = Rig()
    a = r.submit(qty=1.0)

    def on_order(e):
        if e.kind == "filled" and e.order.client_id == a.client_id:
            r.eng.submit(Order(client_id="q1", instrument_id=X, account_id="acc", strategy_id="s", side="BUY",
                               type="LIMIT", qty=1.0, price=100.0))
            r.eng.submit(Order(client_id="q1", instrument_id=X, account_id="acc", strategy_id="s", side="BUY",
                               type="LIMIT", qty=3.0, price=100.0))  # same id, other content: fails when it runs
            r.eng.cancel("q1")

    r.bus.subscribe("order", on_order)
    r.eng.apply(r.fill(a, 1.0))
    assert [type(e) for e in r.eng.handler_errors] == [DuplicateClientId]
    assert r.eng.order("q1").status.value == "CANCELED" and r.broker.canceled == ["q1"]


def test_295_d_identity_includes_tif_bot_and_account():
    r = Rig()
    o = r.submit(client_id="dup", bot_id="bot1")
    for field, value in (("tif", "IOC"), ("bot_id", "bot2"), ("account_id", "other")):
        with pytest.raises(DuplicateClientId, match=field):
            r.eng.submit(dataclasses.replace(o, **{field: value}))
    assert r.eng.submit(dataclasses.replace(o)) is o


def test_295_e_a_single_cancel_failure_is_journalled_and_raised_as_broker_error():
    from neural_trade.desk.core.execution import BrokerError
    r = Rig()
    o = r.submit()
    cause = ConnectionError("down")
    r.broker.raise_on_cancel = {o.client_id: cause}
    with pytest.raises(BrokerError) as ei:
        r.eng.cancel(o.client_id)
    assert ei.value.__cause__ is cause and o.status.value == "ACCEPTED" and r.eng.handler_errors == [cause]
    k, p = r.journal[-1]
    assert (k, p["client_id"], p["reason"]) == ("cancel_failed", o.client_id, "broker_error:ConnectionError")
    n = len(r.journal)
    r.eng.halt("x")  # one record per failure, under the halt's own kind
    assert [k for k, _ in r.journal[n:]] == ["halt", "halt_cancel_failed"]
    assert r.eng.last_cancel_failures == [(o.client_id, cause)]


def test_295_f_a_late_fill_that_breaks_the_spot_balance_is_journalled_and_halts():
    r = Rig(is_spot=True, kind=AccountKind.LIVE)
    b = r.submit(side="BUY", qty=2.0, type="MARKET")
    r.eng.apply(r.fill(b, 2.0))
    s = r.submit(side="SELL", qty=2.0, type="LIMIT", price=100.0)
    r.eng.cancel(s.client_id)
    c = r.eng.close_position("X-s")
    r.eng.apply(r.fill(c, 2.0))  # the position went flat another way
    n = len(r.journal)
    with pytest.raises(OverfillError):
        r.eng.apply(r.fill(s, 2.0, tid="late"))
    assert s.status.value == "CANCELED" and r.pf.base_balance(X) == 0.0 and r.risk.state is TradingState.HALTED
    assert r.risk.state_reason == "late_fill_refused"
    assert r.kinds()[n:] == ["late_fill_refused", "halt"]
    assert r.journal[n][1]["trade_id"] == "late" and "exceeds the held base balance" in r.journal[n][1]["error"]
    assert r.seen[-1][1].kind == "late_fill_refused" and "late" not in r.eng._trades


def test_295_f_a_late_fill_that_flips_a_margin_position_is_journalled_and_halts():
    r = Rig()
    b = r.submit(side="BUY", qty=2.0, type="MARKET")
    r.eng.apply(r.fill(b, 2.0))
    s = r.submit(side="SELL", qty=3.0, type="LIMIT", price=100.0)  # 2 close, 1 opens a short
    r.eng.cancel(s.client_id)
    n = len(r.journal)
    r.eng.apply(r.fill(s, 3.0))
    assert s.status.value == "FILLED" and r.pf.signed_qty(r.pf.position(X, "s")) == -1.0
    assert r.kinds()[n:] == ["late_fill", "fill", "position", "position", "account", "late_fill_flip", "halt"]
    assert (r.journal[n + 5][1]["before"], r.journal[n + 5][1]["after"]) == (2.0, -1.0)
    assert r.risk.state is TradingState.HALTED and r.risk.state_reason == "late_fill_flip"
    # an ordinary flip is the strategy's business: no halt
    r2 = Rig()
    b2 = r2.submit(side="BUY", qty=2.0, type="MARKET")
    r2.eng.apply(r2.fill(b2, 2.0))
    s2 = r2.submit(side="SELL", qty=3.0, type="LIMIT", price=100.0)
    r2.eng.apply(r2.fill(s2, 3.0))
    assert r2.risk.state is TradingState.ACTIVE and "late_fill_flip" not in r2.kinds()


def test_295_p1_a_dead_order_after_a_late_fill_commits_nothing():
    r = Rig()
    a = r.submit(side="BUY", qty=10.0, type="LIMIT", price=100.0)
    r.eng.cancel(a.client_id)
    r.eng.apply(r.fill(a, 4.0))  # late: PARTIALLY_FILLED but dead, 6 x 100 still "unfilled" on paper
    assert a.status.value == "PARTIALLY_FILLED" and not r.eng.is_live(a)
    r.submit(side="BUY", qty=1.0, type="LIMIT", price=100.0)
    assert _commit_kw(r) == (0.0, 0.0)


def test_295_p1_modify_is_checked_against_the_other_orders_but_not_against_itself():
    r = Rig(is_spot=True)
    r.submit(side="BUY", qty=50.0, type="LIMIT", price=100.0)  # 5000
    b = r.submit(side="BUY", qty=50.0, type="LIMIT", price=100.0)  # 5000: all the cash
    assert b.status.value == "ACCEPTED"
    r.eng.modify(b.client_id, price=99.5)  # itself is excluded: 50 x 99.5 + 5000 <= 10000
    assert (b.price, _commit_kw(r)) == (99.5, (5000.0, 5000.0))
    n = len(r.journal)
    r.eng.modify(b.client_id, qty=51.0)  # 51 x 99.5 = 5074.5 + 5000 > 10000
    assert b.qty == 50.0 and r.kinds()[n:] == ["order_modify_denied"]
    assert r.journal[n][1]["reason"] == "insufficient_cash"


# ------------------------------------------------------------------ NT-295 repair round 1
def test_295_r1_reducing_fills_are_never_refused_not_even_while_halted():
    r = Rig()  # margin, PAPER, cash 10000
    s = r.submit(side="SELL", qty=10.0, type="MARKET")
    r.eng.apply(r.fill(s, 10.0))  # short 10 at 100
    cover = r.submit(side="BUY", qty=10.0, type="STOP_MARKET", price=None, trigger_price=105.0, reduce_only=True)
    assert cover.status.value == "ACCEPTED" and r.eng._checked[cover.client_id] == 105.0
    r.risk.halt("kill switch")
    d = r.eng.precheck_fill(cover.client_id, 108.0)  # QA: refused `halted` before
    assert (d.allowed, d.qty) == (True, 10.0) and cover.status.value == "ACCEPTED"
    r.eng.apply(r.fill(cover, 10.0, price=108.0))
    assert cover.status.value == "FILLED" and r.pf.cash == 10_000.0 + 1_000.0 - 1_080.0 and "fill_refused" not in r.kinds()
    # a take-profit limit sell above its level, and a plain (not reduce_only) sell that only reduces
    r2 = Rig(is_spot=True)
    b = r2.submit(side="BUY", qty=10.0, type="MARKET")
    r2.eng.apply(r2.fill(b, 10.0))
    tp = r2.submit(side="SELL", qty=4.0, type="LIMIT", price=110.0, reduce_only=True)
    plain = r2.submit(side="SELL", qty=5.0, type="LIMIT", price=120.0)
    r2.risk.halt("kill switch")
    assert r2.eng.precheck_fill(tp.client_id, 115.0).allowed and r2.eng.precheck_fill(plain.client_id, 125.0).allowed
    r2.eng.apply(r2.fill(tp, 4.0, price=115.0))
    r2.eng.apply(r2.fill(plain, 5.0, price=125.0))
    assert (tp.status.value, plain.status.value) == ("FILLED", "FILLED") and r2.pf.position(X, "s").qty == 1.0
    # an opening fill is still refused while halted
    r3, o = _stop_rig()
    r3.risk.halt("kill switch")
    assert r3.eng.precheck_fill(o.client_id, 110.0).reason == "halted" and o.status.value == "REJECTED"


def test_295_r1_a_venue_fact_halt_cancels_the_opening_orders_and_keeps_the_protective_ones():
    r = Rig(is_spot=True)
    b = r.submit(side="BUY", qty=10.0, type="MARKET")
    r.eng.apply(r.fill(b, 10.0))
    protect = r.submit(side="SELL", qty=7.0, type="LIMIT", price=120.0)  # reduces 7 of the 10 held
    opening = r.submit(side="BUY", qty=1.0, type="LIMIT", price=90.0)
    stop = r.submit(side="SELL", qty=3.0, type="STOP_MARKET", price=None, trigger_price=80.0, reduce_only=True)
    n = len(r.journal)
    r.eng._ops += 1
    r.eng._halt_risk("late_fill_flip")  # inside an operation, as in apply
    r.eng._ops -= 1
    r.eng._drain()
    assert r.risk.state is TradingState.HALTED and r.broker.canceled == [opening.client_id]
    assert (opening.status.value, protect.status.value, stop.status.value) == ("CANCELED", "ACCEPTED", "ACCEPTED")
    assert r.kinds()[n:] == ["halt", "order_canceled"]


def test_295_r2_the_fill_recheck_ignores_the_order_rate_limit():
    lim = dataclasses.replace(RiskLimits.unlimited(), max_orders_per_minute=1)
    r = Rig(lim, is_spot=True)
    o = r.submit(side="BUY", qty=90.0, type="STOP_MARKET", price=None, trigger_price=100.0)  # takes the only slot
    assert o.status.value == "ACCEPTED" and r.submit(qty=1.0).reason == "max_orders_per_minute"
    d = r.eng.precheck_fill(o.client_id, 105.0)  # QA: refused max_orders_per_minute before
    assert (d.allowed, d.qty) == (True, 90.0) and r.checks[-1][1]["fill_recheck"] is True
    assert len(r.risk._sent) == 1  # and it took no slot


def test_295_r3_a_size_frac_fill_is_always_rechecked_against_the_cash_left():
    r = Rig(is_spot=True)
    a = r.submit(side="BUY", qty=89.0, type="LIMIT", price=100.0)  # commits 8900
    s = r.submit(side="BUY", qty=None, size_frac=1.0, type="MARKET")  # clipped to the 1100 left: 11
    assert r.eng._reserved[s.client_id] == 11.0
    r.eng.apply(dataclasses.replace(r.fill(a, 89.0), fee=50.0))  # the fee leaves 10000 - 8900 - 50 = 1050 of cash
    d = r.eng.precheck_fill(s.client_id, 99.0)  # below the checked price: QA's probe returned qty None here
    assert d.allowed and d.qty == pytest.approx(1050.0 / 99.0, rel=1e-12) and d.qty < 11.0
    r.eng.apply(r.fill(s, d.qty, price=99.0))
    assert r.pf.cash >= -1e-9 and s.status.value == "FILLED"


def test_295_r4_live_precheck_is_a_noop_and_simulated_fills_after_a_refusal_are_ignored():
    r, o = _stop_rig(kind=AccountKind.LIVE)
    n = len(r.journal)
    d = r.eng.precheck_fill(o.client_id, 120.0)
    assert (d.allowed, d.qty) == (True, None) and o.status.value == "ACCEPTED" and len(r.journal) == n
    r, o = _stop_rig()
    r.eng.apply(r.fill(o, 30.0, price=100.0))
    assert not r.eng.precheck_fill(o.client_id, 120.0).allowed and o.status.value == "CANCELED"
    n = len(r.journal)
    r.eng.apply(r.fill(o, 60.0, price=120.0, tid="after"))  # the simulator reports it anyway
    assert r.kinds()[n:] == ["fill_after_refusal"] and r.journal[n][1]["trade_id"] == "after"
    assert r.pf.position(X, "s").qty == 30.0 and o.filled_qty == 30.0


def test_295_r4_a_late_fill_of_a_cancelled_order_is_rechecked_on_paper_but_not_on_live():
    for kind, applied in ((AccountKind.PAPER, False), (AccountKind.LIVE, True)):
        r = Rig(is_spot=True, kind=kind)
        a = r.submit(side="BUY", qty=50.0, type="LIMIT", price=100.0)
        r.eng.cancel(a.client_id)  # the user cancelled it: not a refusal
        b = r.submit(side="BUY", qty=60.0, type="MARKET")
        r.eng.apply(r.fill(b, 60.0))  # cash 4000
        n = len(r.journal)
        r.eng.apply(r.fill(a, 50.0, tid="late"))  # 5000 > 4000
        if applied:  # the venue filled it: the books follow
            assert r.kinds()[n:] == ["late_fill", "fill", "position", "account"] and r.pf.cash == -1_000.0
        else:
            assert r.kinds()[n:] == ["fill_refused"] and r.journal[n][1]["check"] == "insufficient_cash"
            assert a.status.value == "CANCELED" and r.pf.cash == 4_000.0 and r.pf.position(X, "s").qty == 60.0
            r.eng.apply(r.fill(a, 50.0, tid="late2"))
            assert r.kinds()[n + 1:] == ["fill_after_refusal"]


def test_295_r6_a_handler_that_always_submits_hits_the_queue_cap():
    r = Rig(cash=1e12)
    count = itertools.count()

    def spin(e):
        if len(r.eng.orders) < 2000:
            r.eng.submit(Order(client_id=f"spin{next(count)}", instrument_id=X, account_id="acc", strategy_id="s",
                               side="BUY", type="LIMIT", qty=1.0, price=100.0))

    r.bus.subscribe("order", spin)
    r.submit(qty=1.0)
    assert len(r.eng.orders) < 2000  # QA c2: without the cap the handler ran to its own limit
    assert r.risk.state is TradingState.HALTED and r.risk.state_reason == "handler_queue_overflow"
    assert r.kinds().count("handler_queue_overflow") == 1
    assert any(isinstance(e.exc, RuntimeError) and "queue overflow" in str(e.exc) for e in r.bus.errors)
    assert not r.eng._deferred and not r.eng._queue_closed and r.eng.open_orders() == []  # opening orders cancelled


def test_295_r7_commitments_scan_only_the_live_orders():
    r = Rig(cash=1e12)
    for i in range(5000):
        o = r.submit(qty=1.0, price=100.0, client_id=f"old{i}")
        r.eng.cancel(o.client_id)
    assert r.eng.open_orders() == []

    class Counting(dict):
        scans = 0

        def values(self):
            Counting.scans += 1
            return super().values()

        def items(self):
            Counting.scans += 1
            return super().items()

        def __iter__(self):
            Counting.scans += 1
            return super().__iter__()

    r.eng._orders = Counting(r.eng._orders)
    live = [r.submit(qty=1.0, price=100.0) for _ in range(3)]
    r.eng.modify(live[0].client_id, price=99.0)
    assert [o.client_id for o in r.eng.open_orders()] == [o.client_id for o in live]
    r.eng.cancel_all("acc")
    assert Counting.scans == 0


def test_295_r8_a_venue_fact_halt_journals_the_halt_once():
    r = Rig()
    r.eng._ops += 1
    r.eng._halt_risk("late_fill_flip")
    r.eng._halt_risk("fill_price_check_failed")  # already HALTED: no second record, the first reason stays
    r.eng._ops -= 1
    assert r.risk.state_reason == "late_fill_flip" and r.kinds() == ["halt"]
    r2, o = _stop_rig(kind=AccountKind.LIVE)
    r2.eng.apply(r2.fill(o, 90.0, price=120.0))
    assert r2.kinds().count("halt") == 1 and r2.kinds().count("fill_refused") == 1


# ------------------------------------------------------------------ NT-295 repair round 2 (lead decisions A-F, M8, M14)
def _long_one(**kw):
    r = Rig(**kw)
    b = r.submit(side="BUY", qty=1.0, type="MARKET")
    r.eng.apply(r.fill(b, 1.0))
    return r


def _held(r, sid="s"):
    return r.pf.signed_qty(r.pf.position(X, sid))


def test_295_r2a_a_clipped_fill_applies_only_the_allowed_quantity():
    # QA probe R1e: margin, long 1, SELL 3 MARKET, account REDUCING, fill at 101: the position became -2
    r = _long_one()
    s = r.submit(side="SELL", qty=3.0, type="MARKET")
    r.risk.reduce_only("qa")
    n = len(r.journal)
    r.eng.apply(r.fill(s, 3.0, price=101.0))
    assert (s.status.value, s.filled_qty, s.reason, _held(r)) == ("CANCELED", 1.0, "fill_price_check_failed", 0.0)
    assert r.kinds()[n:] == ["order_partially_filled", "fill", "position", "account", "fill_refused",
                             "order_canceled"]
    assert r.journal[n + 1][1]["fill"]["qty"] == 1.0
    rec = r.journal[n + 4][1]
    assert (rec["check"], rec["applied"], rec["fill_qty"], rec["allowed_qty"], rec["refused_qty"]) == (
        "clipped_to_position", False, 3.0, 1.0, 2.0)
    r.eng.apply(r.fill(s, 2.0, price=101.0, tid="rest"))  # the simulator must have dropped it
    assert r.kinds()[-1] == "fill_after_refusal" and _held(r) == 0.0
    # precheck_fill returns the clipped quantity, refuses the excess at once, and the order ends with the allowed fill
    r = _long_one()
    s = r.submit(side="SELL", qty=3.0, type="MARKET")
    r.risk.reduce_only("qa")
    n = len(r.journal)
    d = r.eng.precheck_fill(s.client_id, 101.0)
    assert (d.allowed, d.qty, d.reason) == (True, 1.0, "clipped_to_position") and s.status.value == "ACCEPTED"
    assert r.kinds()[n:] == ["fill_refused"] and r.journal[n][1]["refused_qty"] == 2.0
    r.eng.apply(r.fill(s, 1.0, price=101.0))
    assert (s.status.value, s.filled_qty, _held(r)) == ("CANCELED", 1.0, 0.0)
    assert r.kinds()[n + 1:] == ["order_partially_filled", "fill", "position", "account", "order_canceled"]
    # LIVE: the venue fact is applied in full, the excess journalled (applied) and RiskGuard halted
    r = _long_one(kind=AccountKind.LIVE)
    s = r.submit(side="SELL", qty=3.0, type="MARKET")
    r.risk.reduce_only("qa")
    r.eng.apply(r.fill(s, 3.0, price=101.0))
    assert (s.status.value, _held(r), r.risk.state) == ("FILLED", -2.0, TradingState.HALTED)
    assert r.kinds()[-2:] == ["fill_refused", "halt"]
    rec = r.journal[-2][1]
    assert (rec["applied"], rec["refused_qty"], rec["check"]) == (True, 2.0, "clipped_to_position")


def _ro_outlives_its_position(**kw):
    """Long 1; a cancelled SELL and a resting reduce-only SELL; the cancelled one fills late and closes the long."""
    r = _long_one(**kw)
    old = r.submit(side="SELL", qty=1.0, type="LIMIT", price=120.0)
    r.eng.cancel(old.client_id)
    ro = r.submit(side="SELL", qty=1.0, type="LIMIT", price=110.0, reduce_only=True)
    r.eng.apply(r.fill(old, 1.0, price=120.0))
    assert _held(r) == 0.0 and ro.status.value == "ACCEPTED"
    return r, ro


def test_295_r2b_reduce_only_reduces_only_against_the_held_position():
    for spot in (False, True):  # QA probe nt295b_probe3: HALTED; margin opened -1, spot raised OverfillError
        r, ro = _ro_outlives_its_position(is_spot=spot)
        r.risk.halt("qa")
        n = len(r.journal)
        r.eng.apply(r.fill(ro, 1.0, price=110.0))
        assert (ro.status.value, ro.reason, _held(r)) == ("REJECTED", "fill_price_check_failed", 0.0)
        assert r.kinds()[n:] == ["fill_refused", "order_rejected"] and r.journal[n][1]["check"] == "halted"
    # ACTIVE, at a price below the checked one: a reduce-only fill that does not reduce is still re-checked
    r, ro = _ro_outlives_its_position()
    r.eng.apply(r.fill(ro, 1.0, price=100.0))
    assert (ro.status.value, _held(r)) == ("REJECTED", 0.0)
    assert r.journal[-2][1]["check"] == "reduce_only_violation"
    # the part of a reduce-only fill beyond the holding is the excess of point A
    r = _long_one()
    ro = r.submit(side="SELL", qty=3.0, type="LIMIT", price=100.0, reduce_only=True)
    assert ro.qty == 1.0  # RiskGuard clipped it at submit; a venue may still report more
    ro.qty = 3.0
    r.eng.apply(r.fill(ro, 3.0, price=100.0))
    assert (ro.filled_qty, _held(r), r.journal[-2][1]["refused_qty"]) == (1.0, 0.0, 2.0)
    # LIVE: the venue fact is applied, journalled and halts
    r, ro = _ro_outlives_its_position(kind=AccountKind.LIVE)
    r.eng.apply(r.fill(ro, 1.0, price=100.0))
    assert (_held(r), r.risk.state, r.journal[-2][1]["applied"]) == (-1.0, TradingState.HALTED, True)


def test_295_r2c_opening_fills_are_rechecked_while_halted_or_reducing():
    # QA probe R1f: HALTED with the cancels failing, fills at the checked price
    r = _long_one()
    flip = r.submit(side="SELL", qty=3.0, type="LIMIT", price=100.0)
    buy = r.submit(side="BUY", qty=3.0, type="LIMIT", price=100.0)
    r.broker.raise_on_cancel = {flip.client_id: ConnectionError("down"), buy.client_id: ConnectionError("down")}
    r.eng.halt("qa")
    assert (flip.status.value, buy.status.value) == ("ACCEPTED", "ACCEPTED")
    r.eng.apply(r.fill(flip, 3.0, price=100.0))  # only the reducing part: the long is closed, no short opens
    assert (flip.status.value, flip.filled_qty, _held(r)) == ("CANCELED", 1.0, 0.0)
    assert r.journal[-2][1]["check"] == "clipped_to_position" and r.journal[-2][1]["refused_qty"] == 2.0
    r.eng.apply(r.fill(buy, 3.0, price=99.0))
    assert (buy.status.value, _held(r)) == ("REJECTED", 0.0) and r.journal[-2][1]["check"] == "halted"
    # REDUCING: an opening fill below the checked price is refused
    r = Rig()
    o = r.submit(side="BUY", qty=2.0, type="LIMIT", price=100.0)
    r.risk.reduce_only("qa")
    r.eng.apply(r.fill(o, 2.0, price=99.0))
    assert (o.status.value, r.pf.positions) == ("REJECTED", {}) and r.journal[-2][1]["check"] == "reducing_only"
    # ACTIVE keeps the price shortcut: no RiskGuard call at or below the checked price
    r = Rig()
    o = r.submit(side="BUY", qty=2.0, type="LIMIT", price=100.0)
    n = len(r.checks)
    r.eng.apply(r.fill(o, 2.0, price=99.0))
    assert o.status.value == "FILLED" and len(r.checks) == n


def test_295_r2d_a_live_fill_on_a_rejected_order_is_booked_and_halts():
    r = Rig(kind=AccountKind.LIVE)
    r.broker.raise_on_submit = TimeoutError("submit timed out")  # but it reached the venue
    o = r.submit(qty=1.0)
    assert (o.status.value, o.reason) == ("REJECTED", "broker_error:TimeoutError")
    n, ns = len(r.journal), len(r.seen)
    r.eng.apply(r.fill(o, 1.0, tid="venue"))
    assert (o.status.value, o.filled_qty, r.pf.position(X, "s").qty) == ("REJECTED", 1.0, 1.0)
    assert r.kinds()[n:] == ["venue_fill_on_rejected", "position", "account", "halt"]
    rec = r.journal[n][1]
    assert (rec["trade_id"], rec["fill"]["qty"], rec["order"]["status"], rec["order"]["filled_qty"]) == (
        "venue", 1.0, "REJECTED", 1.0)
    assert [t for t, _ in r.seen[ns:]] == ["fill", "position", "account"]
    assert r.risk.state is TradingState.HALTED and r.risk.state_reason == "venue_fill_on_rejected"
    n = len(r.journal)
    r.eng.apply(r.fill(o, 1.0, tid="venue"))  # the dedup key holds
    assert len(r.journal) == n and r.pf.position(X, "s").qty == 1.0
    # PAPER / BACKTEST: a simulated fill for a rejected order is still ignored
    for kind in (AccountKind.PAPER, AccountKind.BACKTEST):
        p = Rig(kind=kind)
        p.broker.raise_on_submit = TimeoutError("x")
        po = p.submit(qty=1.0)
        p.eng.apply(p.fill(po, 1.0))
        assert p.kinds()[-1] == "fill_after_refusal" and p.pf.positions == {}


def test_295_r2e_partial_fills_are_checked_on_the_cumulative_notional():
    lim = dataclasses.replace(RiskLimits.unlimited(), max_order_notional=1000.0)
    for second in (104.0, 100.5):  # QA probe R4a'': 5 x 103 + 4.9 x 104 = 1024.6; at 100.5 still 1007.45
        r = Rig(lim, cash=1e5)
        o = r.submit(side="BUY", qty=9.9, type="STOP_MARKET", price=None, trigger_price=101.0)  # 9.9 x 101 = 999.9
        assert o.status.value == "ACCEPTED"
        r.eng.apply(r.fill(o, 5.0, price=103.0))  # 515: passes
        assert o.status.value == "PARTIALLY_FILLED"
        r.eng.apply(r.fill(o, 4.9, price=second))
        assert (o.status.value, o.filled_qty, r.pf.position(X, "s").qty) == ("CANCELED", 5.0, 5.0)
        assert r.journal[-2][1]["check"] == "max_order_notional"
        assert r.checks[-1][1]["prior_order_notional"] == 515.0
    r = Rig(lim, cash=1e5, kind=AccountKind.LIVE)  # LIVE: applied, journalled, halted
    o = r.submit(side="BUY", qty=9.9, type="STOP_MARKET", price=None, trigger_price=101.0)
    r.eng.apply(r.fill(o, 5.0, price=103.0))
    assert r.risk.state is TradingState.ACTIVE
    r.eng.apply(r.fill(o, 4.9, price=104.0))
    assert (o.status.value, r.risk.state, r.journal[-2][1]["check"]) == ("FILLED", TradingState.HALTED,
                                                                         "max_order_notional")


def test_295_r2f_size_frac_fills_are_bounded_by_the_equity_sizing_at_the_fill():
    # NT-299: the bound is 0.1 x 10000 / 100 = 10 sized at the fill (equal to the reservation here only by construction)
    r = Rig(is_spot=True)
    s = r.submit(side="BUY", qty=None, size_frac=0.1, type="MARKET")
    assert r.eng._reserved[s.client_id] == 10.0
    n = len(r.journal)
    d = r.eng.precheck_fill(s.client_id, 100.0, qty=50.0)  # QA probe R3a returned 50
    assert (d.allowed, d.qty, s.status.value, len(r.journal)) == (True, 10.0, "ACCEPTED", n)
    r.eng.apply(r.fill(s, 50.0, price=100.0))  # QA probe R3b applied 50
    assert (s.status.value, s.filled_qty, r.pf.cash, r.pf.position(X, "s").qty) == ("FILLED", 10.0, 9000.0, 10.0)
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account", "fill_refused"]
    assert (r.journal[-1][1]["allowed_qty"], r.journal[-1][1]["refused_qty"]) == (10.0, 40.0)


def _limit_order(cid):
    return Order(client_id=cid, instrument_id=X, account_id="acc", strategy_id="s", side="BUY", type="LIMIT",
                 qty=1.0, price=100.0)


def test_295_r2_m8_one_publish_cannot_queue_more_than_the_cap():
    from neural_trade.desk.core.execution import HANDLER_QUEUE_LIMIT
    r = Rig(cash=1e12)
    refused_at = []

    def burst(e):
        if e.kind == "submitted" and e.order.client_id == "first":
            for i in range(HANDLER_QUEUE_LIMIT + 5):
                try:
                    r.eng.submit(_limit_order(f"b{i}"))
                except RuntimeError:
                    refused_at.append(i)
                    break

    r.bus.subscribe("order", burst)
    r.submit(qty=1.0, client_id="first")
    assert refused_at == [HANDLER_QUEUE_LIMIT]  # the call beyond the cap is refused at once
    assert list(r.eng.orders) == ["first"]  # and the dropped queue ran nothing
    assert r.kinds().count("handler_queue_overflow") == 1 and r.risk.state_reason == "handler_queue_overflow"


def test_295_r2_m14_one_drain_runs_at_most_the_cap():
    from neural_trade.desk.core.execution import HANDLER_QUEUE_LIMIT
    r = Rig(cash=1e12)
    count = itertools.count()

    def chain(e):  # one queued call per drained call: the queue never grows, the drain never ends by itself
        if e.kind == "submitted":
            i = next(count)
            if i < HANDLER_QUEUE_LIMIT + 200:
                r.eng.submit(_limit_order(f"ch{i}"))

    r.bus.subscribe("order", chain)
    r.submit(qty=1.0, client_id="first")
    assert r.kinds().count("handler_queue_overflow") == 1 and r.risk.state_reason == "handler_queue_overflow"
    assert len(r.eng.orders) == HANDLER_QUEUE_LIMIT + 1


def test_295_r2_a_reducing_fill_makes_no_riskguard_call_in_any_state():
    for state in ("ACTIVE", "REDUCING", "HALTED"):
        r = _long_one()
        tp = r.submit(side="SELL", qty=1.0, type="LIMIT", price=101.0)
        if state == "REDUCING":
            r.risk.reduce_only("qa")
        elif state == "HALTED":
            r.risk.halt("qa")
        n = len(r.checks)
        assert r.eng.precheck_fill(tp.client_id, 130.0).allowed
        r.eng.apply(r.fill(tp, 1.0, price=130.0))
        assert (tp.status.value, _held(r), len(r.checks)) == ("FILLED", 0.0, n), state


def test_295_r2_the_live_index_holds_only_the_live_orders():
    r = Rig()
    a, b, c = r.submit(qty=1.0), r.submit(qty=1.0), r.submit(qty=1.0)
    r.eng.cancel(a.client_id)
    r.eng.apply(r.fill(b, 1.0))
    r.broker.submit_kind = "rejected"
    r.submit(qty=1.0)
    assert list(r.eng._live) == [c.client_id] == [o.client_id for o in r.eng.open_orders()]


# ------------------------------------------------------------------ NT-299 (NT-295 continuation, corrected decision F)
def _lim(**kw):
    return dataclasses.replace(RiskLimits.unlimited(), **kw)


def test_299_p1_a_size_frac_fill_is_sized_at_the_fill_open_on_realised_equity():
    """QA probe F1 (decision F reversed): close 100 -> next open 95; A5.1 sizes 0.5 x 10000 / 95, not the 50 reserved."""
    r = Rig()  # margin, PAPER, cash 10000
    o = r.submit(side="BUY", qty=None, size_frac=0.5, type="MARKET")
    assert r.eng._reserved[o.client_id] == 50.0  # the estimate at the last close: commitments only
    d = r.eng.precheck_fill(o.client_id, 95.0)
    assert d.allowed and d.qty == 5_000.0 / 95.0
    n = len(r.journal)
    r.eng.apply(r.fill(o, 5_000.0 / 95.0, price=95.0))
    assert (o.status.value, o.filled_qty, r.pf.position(X, "s").qty) == ("FILLED", 5_000.0 / 95.0, 5_000.0 / 95.0)
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account"]  # nothing refused
    s = r.submit(qty=None, size_frac=0.5, type="MARKET")
    eq = r.pf.realised_equity()
    assert r.eng.precheck_fill(s.client_id, 105.0).qty == 0.5 * eq / 105.0  # opening higher sizes smaller (probe S1)


def test_299_p1_a_size_frac_fill_above_the_equity_sizing_is_clipped_to_it():
    r = Rig()
    o = r.submit(side="BUY", qty=None, size_frac=0.5, type="MARKET")
    q = 5_000.0 / 95.0
    n = len(r.journal)
    r.eng.apply(r.fill(o, 5 * q, price=95.0))  # a 5x too large simulated fill
    assert (o.status.value, o.filled_qty, r.pf.position(X, "s").qty) == ("FILLED", q, q)
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account", "fill_refused"]
    rec = r.journal[-1][1]
    assert (rec["fill_qty"], rec["allowed_qty"], rec["refused_qty"]) == (5 * q, q, 4 * q)
    s = r.submit(qty=None, size_frac=0.5, type="MARKET")
    assert r.eng.precheck_fill(s.client_id, 95.0, qty=5 * q).qty == 0.5 * r.pf.realised_equity() / 95.0
    # the relative tolerance is 1e-9: inside it the fill is trusted, beyond it the excess is refused
    for factor, refused in ((1 + 5e-10, False), (1 + 1e-8, True)):
        r = Rig()
        o = r.submit(side="BUY", qty=None, size_frac=0.5, type="MARKET")
        r.eng.apply(r.fill(o, q * factor, price=95.0))
        assert ("fill_refused" in r.kinds()) is refused, factor
        assert o.filled_qty == (q if refused else q * factor)


def test_299_p1_a_spot_size_frac_fill_is_clipped_to_the_cash_at_the_fill():
    r = Rig(is_spot=True, costs=PROF)  # taker 10 bp + slip 5 bp
    o = r.submit(side="BUY", qty=None, size_frac=1.0, type="MARKET")
    d = r.eng.precheck_fill(o.client_id, 95.0)
    unit = 95.0 * (1 + 0.0010 + 0.0005)
    assert d.allowed and d.reason == "clipped_to_cash" and d.qty == pytest.approx(10_000.0 / unit, rel=1e-12)
    assert d.qty * unit <= 10_000.0 < 10_000.0 / 95.0 * unit


def test_299_p1_parity_with_run_backtest_when_entries_open_below_the_previous_close(monkeypatch):
    """QA nt295c_parity.py: OP[1] = 99.5 < C[0] = 100 and OP[6] = 104.5 < C[5] = 105 (23.6% of the reference CSV's bars
    open below the previous close); the capped sizing of NT-295 drifted by up to 2.34 of equity at zero costs."""
    import numpy as np

    from tests.desk import test_sim_broker as sim
    op, lo = list(sim.OP), list(sim.LO)
    for k, v in ((1, 99.5), (6, 104.5)):
        op[k], lo[k] = v, min(lo[k], v)
    monkeypatch.setattr(sim, "OP", op)
    monkeypatch.setattr(sim, "LO", lo)
    for cost, final in ((RESEARCH_ZERO, 10_227.8137), (sim.PRICED, 10_108.6867)):
        res = sim.oracle(cost, None)
        pf, log = sim.desk_run(cost, None)
        assert len(sim.trades_of(log)) == len(res.trades) == 5
        assert np.allclose([e for _, e in pf.equity_curve], res.equity[1:], rtol=0, atol=1e-9)
        assert pf.equity_curve[-1][1] == pytest.approx(final, abs=1e-4)
        assert sim.trades_of(log)[0][3] == pytest.approx(res.trades[0].entry_price, rel=1e-12)


def test_299_p1_a_gap_through_an_absolute_bracket_level_does_not_refuse_the_entry():
    """nt225_parity: a LONG with stop 98 decided at the close 100 opens at 97; the oracle fills it and the gap rule
    exits at the open. The re-check at the fill judges the entry without the bracket it cannot validate any more."""
    r = Rig()
    o = r.submit(side="BUY", qty=None, size_frac=1.0, type="MARKET",
                 bracket={"take_profit": 104.0, "stop_loss": 98.0, "tp_is_offset": False})
    assert o.status.value == "ACCEPTED"
    r.eng.apply(r.fill(o, 10_000.0 / 97.0, price=97.0))
    assert (o.status.value, r.pf.position(X, "s").qty) == ("FILLED", 10_000.0 / 97.0)
    assert r.pf.get_bracket("X-s")["stop_loss"] == 98.0 and "fill_refused" not in r.kinds()


# ---- point 2: the max_position hole
def _long_two(**kw):
    r = Rig(**kw)
    b = r.submit(side="BUY", qty=2.0, type="MARKET")
    r.eng.apply(r.fill(b, 2.0))
    return r


def test_299_p2_an_order_beyond_the_committed_reductions_is_denied_at_submit():
    r = _long_two(limits=_lim(max_position_notional=400.0))  # QA probe C1a / C1b
    a = r.submit(side="SELL", qty=2.0, type="MARKET")
    c = r.submit(side="SELL", qty=5.0, type="MARKET")  # after SELL 2 fills: short 5 = 500 > 400
    assert (a.status.value, c.status.value, c.reason) == ("ACCEPTED", "DENIED", "max_position")
    assert r.checks[-1][1]["committed_reduce_qty"] == 2.0
    assert r.submit(side="SELL", qty=4.0, type="MARKET").status.value == "ACCEPTED"  # short 4 = 400


def test_299_p2_a_fill_that_no_longer_reduces_is_rechecked_even_in_active():
    # QA probe C2: the SELL 3 was valued as 2 reducing + 1 opening; a late fill closed the long, so it opens short 3
    r = _long_two(limits=_lim(max_position_notional=250.0))
    dead = r.submit(side="SELL", qty=2.0, type="LIMIT", price=110.0)
    r.eng.cancel(dead.client_id)
    nw = r.submit(side="SELL", qty=3.0, type="MARKET")
    assert nw.status.value == "ACCEPTED" and nw.client_id in r.eng._gate.opposite
    r.eng.apply(r.fill(dead, 2.0, price=110.0))  # late: flat now
    n = len(r.checks)
    d = r.eng.precheck_fill(nw.client_id, 100.0)  # ACTIVE, at the checked price: no price shortcut for it
    assert (d.allowed, d.reason, len(r.checks)) == (False, "max_position", n + 1)
    assert (nw.status.value, nw.reason, _held(r)) == ("REJECTED", "fill_price_check_failed", 0.0)
    # without the precheck, apply refuses the fill the same way
    r = _long_two(limits=_lim(max_position_notional=250.0))
    dead = r.submit(side="SELL", qty=2.0, type="LIMIT", price=110.0)
    r.eng.cancel(dead.client_id)
    nw = r.submit(side="SELL", qty=3.0, type="MARKET")
    r.eng.apply(r.fill(dead, 2.0, price=110.0))
    r.eng.apply(r.fill(nw, 3.0))
    assert (nw.status.value, _held(r)) == ("REJECTED", 0.0) and r.journal[-2][1]["check"] == "max_position"


def test_299_p2_spot_a_sell_with_no_balance_left_is_refused_and_journalled_never_raised():
    # QA probe C3: spot PAPER; the late fill of a cancelled SELL sold the holding; the next SELL fill has nothing to sell
    for precheck in (True, False):
        r = _long_two(is_spot=True)
        dead = r.submit(side="SELL", qty=2.0, type="LIMIT", price=110.0)
        r.eng.cancel(dead.client_id)
        nw = r.submit(side="SELL", qty=2.0, type="MARKET")
        r.eng.apply(r.fill(dead, 2.0, price=110.0))
        n = len(r.journal)
        if precheck:
            assert r.eng.precheck_fill(nw.client_id, 100.0).reason == "spot_sell_exceeds_balance"
        r.eng.apply(r.fill(nw, 2.0))  # no OverfillError
        assert (nw.status.value, r.pf.base_balance(X)) == ("REJECTED", 0.0)
        assert r.kinds()[n:] == ["fill_refused", "order_rejected"] + (["fill_after_refusal"] if precheck else [])
        assert r.journal[n][1]["check"] == "spot_sell_exceeds_balance"


def test_299_p2_a_simulated_fill_the_books_cannot_hold_is_refused_not_raised():
    from neural_trade.desk.core.risk import RiskDecision
    r = _long_two(is_spot=True)
    dead = r.submit(side="SELL", qty=2.0, type="LIMIT", price=110.0)
    r.eng.cancel(dead.client_id)
    s = r.submit(side="SELL", qty=2.0, type="LIMIT", price=100.0)
    r.eng.apply(r.fill(dead, 2.0, price=110.0))  # late: flat
    r.eng._gate.decision = lambda o, mid, qty, always=False: RiskDecision(True, "", qty)  # a check that let it pass
    n = len(r.journal)
    r.eng.apply(r.fill(s, 2.0))
    assert (s.status.value, s.reason, r.pf.base_balance(X)) == ("REJECTED", "fill_price_check_failed", 0.0)
    assert r.kinds()[n:] == ["fill_refused", "order_rejected"] and r.journal[n][1]["check"] == "books_refused"


# ---- point 3: modify counts the filled notional
def test_299_p3_modify_counts_the_filled_notional_against_max_order_notional():
    r = Rig(_lim(max_order_notional=1_000.0), cash=1e5)  # QA probe E1
    o = r.submit(side="BUY", qty=10.0, type="LIMIT", price=100.0)
    r.eng.apply(r.fill(o, 9.0))
    n = len(r.journal)
    r.eng.modify(o.client_id, qty=18.0)  # 900 filled + 900 more > 1000
    assert (o.qty, r.kinds()[n:], r.journal[n][1]["reason"]) == (10.0, ["order_modify_denied"], "max_order_notional")
    assert r.checks[-1][1]["prior_order_notional"] == 900.0
    r.eng.modify(o.client_id, qty=11.0)  # 900 + 200 > 1000 too
    assert o.qty == 10.0
    r.eng.modify(o.client_id, price=99.0)  # 900 + 99 <= 1000
    assert (o.price, o.qty) == (99.0, 10.0)


# ---- point 4: precheck journals once; clipped fees, limits and late fills
def _flip_halted(**kw):
    r = _long_one(**kw)
    s = r.submit(side="SELL", qty=3.0, type="MARKET")
    r.risk.halt("qa")
    return r, s


def test_299_p4_a_clipped_precheck_journals_the_refused_excess_once():
    r, s = _flip_halted()  # QA probe A2: four records before
    n = len(r.journal)
    for qty in (None, None, 3.0):
        d = r.eng.precheck_fill(s.client_id, 101.0, qty=qty)
        assert (d.allowed, d.qty) == (True, 1.0)
    r.eng.apply(r.fill(s, 0.5, price=101.0))
    assert r.eng.precheck_fill(s.client_id, 101.0).qty == 0.5
    r.eng.apply(r.fill(s, 0.5, price=101.0))
    assert (s.status.value, s.filled_qty, s.reason, _held(r)) == ("CANCELED", 1.0, "fill_price_check_failed", 0.0)
    recs = [p for k, p in r.journal[n:] if k == "fill_refused"]
    assert [(p["fill_qty"], p["allowed_qty"], p["refused_qty"]) for p in recs] == [(3.0, 1.0, 2.0)]


def test_299_p4_a_clipped_fill_pays_the_fee_of_the_applied_part_only():
    r, s = _flip_halted()  # QA probe A1 / mutant Q2
    cash = r.pf.cash
    r.eng.apply(dataclasses.replace(r.fill(s, 3.0, price=101.0), price=100.9, fee=0.3))
    fill = [p["fill"] for k, p in r.journal if k == "fill"][-1]
    assert (fill["qty"], fill["fee"]) == (1.0, pytest.approx(0.1, rel=1e-12))
    assert r.pf.cash - cash == pytest.approx(100.9 - 0.1, rel=1e-12)


def test_299_p4_the_precheck_limit_counts_what_is_already_filled():
    r = Rig()  # mutant Q3: the limit set to the clipped quantity alone
    b = r.submit(side="BUY", qty=3.0, type="MARKET")
    r.eng.apply(r.fill(b, 3.0))
    s = r.submit(side="SELL", qty=5.0, type="MARKET")
    r.eng.apply(r.fill(s, 1.0))  # reduces: long 2 left
    r.risk.halt("qa")
    n = len(r.journal)
    d = r.eng.precheck_fill(s.client_id, 100.0)
    assert (d.allowed, d.qty, r.eng._fill_limit[s.client_id]) == (True, 2.0, 3.0)
    r.eng.apply(r.fill(s, 2.0))
    assert (s.status.value, s.filled_qty, _held(r)) == ("CANCELED", 3.0, 0.0)
    assert r.kinds()[n:].count("fill_refused") == 1


def test_299_p4_a_clipped_late_fill_adds_no_order_canceled():
    r = _long_one()  # mutant Q11: the late (dead) refusal passed dead=False
    s = r.submit(side="SELL", qty=3.0, type="LIMIT", price=100.0)
    r.eng.cancel(s.client_id)
    r.risk.halt("qa")
    n = len(r.journal)
    r.eng.apply(r.fill(s, 3.0))  # late: only the reducing 1 is applied
    assert (s.status.value, s.filled_qty, _held(r)) == ("PARTIALLY_FILLED", 1.0, 0.0)
    assert r.kinds()[n:] == ["late_fill", "fill", "position", "account", "fill_refused"]
    assert (r.journal[-1][1]["dead"], r.journal[-1][1]["refused_qty"]) == (True, 2.0)


# ---- point 5: LIVE never loses a venue fact
def _unbooked(r, n):
    recs = [p for k, p in r.journal[n:] if k == "venue_fill_unbooked"]
    assert len(recs) == 1 and r.risk.state is TradingState.HALTED
    return recs[0]


def test_299_p5_a_live_sell_with_no_spot_balance_is_journalled_unbooked_and_halts():
    r = _long_two(is_spot=True, kind=AccountKind.LIVE)  # QA probe L1
    dead = r.submit(side="SELL", qty=2.0, type="LIMIT", price=110.0)
    r.eng.cancel(dead.client_id)
    nw = r.submit(side="SELL", qty=2.0, type="MARKET")
    r.eng.apply(r.fill(dead, 2.0, price=110.0))  # late, applied: flat
    n = len(r.journal)
    with pytest.raises(OverfillError):
        r.eng.apply(r.fill(nw, 2.0, tid="venue"))
    rec = _unbooked(r, n)
    assert r.kinds()[n:n + 2] == ["venue_fill_unbooked", "halt"] and r.risk.state_reason == "venue_fill_unbooked"
    assert (rec["trade_id"], rec["fill"]["qty"], rec["order"]["client_id"], rec["order"]["status"]) == (
        "venue", 2.0, nw.client_id, "ACCEPTED")
    assert "OverfillError" in rec["error"] and r.pf.base_balance(X) == 0.0 and "venue" not in r.eng._trades


def test_299_p5_live_fills_on_filled_expired_unaccepted_or_unknown_orders_are_journalled():
    # a second fill of a FILLED size_frac order (resolved 4)
    r = Rig(kind=AccountKind.LIVE)
    r.broker.resolved_qty = 4.0
    o = r.submit(qty=None, size_frac=0.5, type="MARKET")
    r.eng.apply(r.fill(o, 4.0))
    n = len(r.journal)
    with pytest.raises(OverfillError):
        r.eng.apply(r.fill(o, 1.0))
    assert _unbooked(r, n)["order"]["status"] == "FILLED"
    # a fill on an EXPIRED order (QA probe L3)
    r = Rig(kind=AccountKind.LIVE)
    o = r.submit(qty=1.0)
    r.eng.apply(OrderEvent(1100, 1100, "expired", o))
    n = len(r.journal)
    with pytest.raises(IllegalTransition):
        r.eng.apply(r.fill(o, 1.0))
    assert _unbooked(r, n)["order"]["status"] == "EXPIRED" and r.pf.positions == {}
    # an unknown client_id
    r = Rig(kind=AccountKind.LIVE)
    ghost = Order(client_id="ghost", instrument_id=X, account_id="acc", strategy_id="s", side="BUY", type="MARKET",
                  qty=1.0)
    n = len(r.journal)
    with pytest.raises(UnknownOrder):
        r.eng.apply(r.fill(ghost, 1.0))
    assert _unbooked(r, n)["order"] == {"unknown": "ghost"}
    # a fill reported while the order is still SUBMITTED (the venue's fill arrives inside broker.submit)
    r = Rig(kind=AccountKind.LIVE)
    caught = []
    real = r.broker.submit

    def submit_and_fill(order):
        try:
            r.eng.apply(r.fill(order, 1.0, tid="early"))
        except IllegalTransition as exc:
            caught.append(exc)
        return real(order)

    r.broker.submit = submit_and_fill
    n = len(r.journal)
    r.submit(qty=1.0)
    assert len(caught) == 1 and _unbooked(r, n)["order"]["status"] == "SUBMITTED"
    # PAPER keeps raising without a record (a simulator bug, not a venue fact)
    p = Rig()
    po = p.submit(qty=1.0)
    p.eng.apply(OrderEvent(1100, 1100, "expired", po))
    n = len(p.journal)
    with pytest.raises(IllegalTransition):
        p.eng.apply(p.fill(po, 1.0))
    assert len(p.journal) == n and p.risk.state is TradingState.ACTIVE


def test_299_p5_a_live_size_frac_accept_without_resolved_qty_is_rejected_and_halts():
    r = Rig(kind=AccountKind.LIVE)  # QA probe L2
    o = r.submit(qty=None, size_frac=0.5, type="MARKET")
    assert (o.status.value, o.reason) == ("REJECTED", "resolved_qty_missing")
    assert r.kinds() == ["order_submitted", "order_rejected", "halt"]
    assert r.risk.state is TradingState.HALTED and r.risk.state_reason == "resolved_qty_missing"
    r.eng.apply(r.fill(o, 20.0, tid="v1"))  # the venue still fills it: booked as venue_fill_on_rejected
    r.eng.apply(r.fill(o, 30.0, tid="v2"))
    assert r.pf.position(X, "s").qty == 50.0 and r.kinds().count("venue_fill_on_rejected") == 2
    p = Rig()  # PAPER: the SimBroker resolves at the fill; no resolved_qty needed
    assert p.submit(qty=None, size_frac=0.5, type="MARKET").status.value == "ACCEPTED"


def test_299_p1_a_smaller_size_frac_fill_is_judged_on_its_own_when_the_full_sizing_breaks_a_limit():
    r = Rig(_lim(max_order_notional=600.0))
    b = r.submit(side="BUY", qty=6.0, type="MARKET")
    r.eng.apply(r.fill(b, 6.0))
    o = r.submit(side="BUY", qty=None, size_frac=0.05, type="MARKET")  # 0.05 x 10000 / 100 = 5: 500 <= 600
    assert o.status.value == "ACCEPTED"
    s = r.submit(side="SELL", qty=6.0, type="MARKET")
    r.eng.apply(r.fill(s, 6.0, price=500.0))  # +2400 realised: the full sizing is now 6.2 (620 > 600)
    r.eng.apply(r.fill(o, 5.0))  # a broker that sized it smaller: 500 passes on its own
    assert (o.status.value, o.filled_qty, _held(r)) == ("FILLED", 5.0, 5.0)

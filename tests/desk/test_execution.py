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
    with pytest.raises(IllegalTransition):  # a closed order takes no fill and nothing changes
        r.eng.apply(r.fill(o, 1.0))
    assert r.pf.positions == {} and o.status.value == "REJECTED"


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
    assert r.eng.journal_pending == 7 and r.risk.state is TradingState.HALTED  # 6 records + the halt itself
    assert r.risk.state_reason == "journal_unavailable"
    assert r.eng.journal_failures == 6
    d = r.submit(qty=1.0)  # an 8th pending record: still one halt record, one log line
    assert d.status.value == "DENIED" and d.reason == "halted" and r.eng.journal_pending == 8
    assert sum("halting new orders" in m for m in caplog.messages) == 1
    assert sum("journal sink failed" in m for m in caplog.messages) == 1  # first failure only (then every 1000th)
    r.down = False
    r.eng.apply(r.fill(orders[0], 1.0))
    assert r.eng.journal_pending == 0
    assert r.kinds()[:8] == ["order_submitted", "order_accepted"] * 3 + ["halt", "order_denied"]
    assert r.journal[6][1]["reason"] == "journal_unavailable"
    assert r.kinds()[-4:] == ["order_filled", "fill", "position", "account"] and not r.eng._journal.overflowed


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
                               "checked_price": 100.0, "applied": False, "filled_qty": 0.0}
    assert (r.journal[n + 1][1]["from"], r.journal[n + 1][1]["to"]) == ("ACCEPTED", "REJECTED")
    assert r.seen[-1][0] == "order" and r.seen[-1][1].kind == "rejected" and r.pf.positions == {}
    assert r.eng.precheck_fill(o.client_id, 120.0).reason == "order_not_live"


def test_295_p3_a_size_frac_order_is_sized_on_the_fill_price():
    r = Rig(is_spot=True)
    o = r.submit(side="BUY", qty=None, size_frac=1.0, type="STOP_MARKET", price=None, trigger_price=100.0)
    assert o.status.value == "ACCEPTED" and r.eng._reserved[o.client_id] == 100.0  # the estimate: 10000 / 100
    d = r.eng.precheck_fill(o.client_id, 120.0)
    assert d.allowed and d.qty == pytest.approx(10_000.0 / 120.0, rel=1e-12) and o.status.value == "ACCEPTED"
    assert r.eng.precheck_fill(o.client_id, 100.00005).qty is None  # inside the tolerance: the broker resolves it


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
    r = Rig(is_spot=True)
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

"""NT-224 (DESK-1.3): ExecutionEngine pinned by scripted broker events with literal expected states.

The broker is a scripted ``FakeBroker`` and the journal a list; nothing here needs a database or a bar.
"""
from __future__ import annotations

import dataclasses
import itertools
import threading

import pytest

from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import SimClock
from neural_trade.desk.core.costs import RESEARCH_ZERO
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

    def modify(self, client_id, *, price=None, trigger_price=None, qty=None):
        self.modified.append({"price": price, "trigger_price": trigger_price, "qty": qty})
        v = self.venue[client_id]
        return OrderEvent(1000, 1000, "modified", dataclasses.replace(v, price=price, trigger_price=trigger_price,
                                                                      qty=qty if v.qty is not None else None))

    def submit(self, order):
        self.submitted.append(order)
        if self.raise_on_submit:
            raise self.raise_on_submit
        return OrderEvent(1000, 1000, self.submit_kind,
                          dataclasses.replace(order, reason=self.submit_reason, venue_id=f"v-{order.client_id}",
                                              tags={**order.tags, **self.accept_tags}))

    def cancel(self, client_id):
        self.canceled.append(client_id)
        return OrderEvent(1000, 1000, self.cancel_kind, dataclasses.replace(self.venue[client_id]))


class Rig:
    def __init__(self, limits=None, *, is_spot=False, cash=10_000.0):
        ins = Instrument(id=X, symbol="X/USDT", base="X", quote="USDT", price_increment=0.0, size_increment=0.0,
                         min_qty=0.0, min_notional=0.0, is_spot=is_spot)
        self.clock = SimClock(1000)
        self.pf = Portfolio("acc", kind=AccountKind.PAPER, initial_cash=cash, allow_short=not is_spot,
                            clock=self.clock)
        self.risk = RiskGuard(limits or RiskLimits.unlimited(), ins, self.pf, self.clock, costs=RESEARCH_ZERO)
        self.broker, self.bus, self.journal, self.seen = FakeBroker(), EventBus(), [], []
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
    r.eng.apply(r.fill(o, 1.5, ts=1800))
    assert o.status.value == "FILLED" and r.pf.position(X, "s").qty == 2.0

    q = r.submit(qty=1.0)
    r.eng.cancel(q.client_id)
    r.eng.apply(r.fill(q, 1.0))
    assert q.status.value == "FILLED"  # CANCELED -> FILLED straight away
    assert r.kinds().count("late_fill") == 2


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


@pytest.mark.parametrize("topic", ["order", "fill", "position", "account"])
def test_a_raising_subscriber_loses_no_record_and_is_recorded(topic):
    r = Rig()
    o = r.submit(qty=2.0)
    n = len(r.journal)

    def boom(e):
        raise RuntimeError("boom")

    r.bus.subscribe(topic, boom)
    r.eng.apply(r.fill(o, 2.0))
    assert o.status.value == "FILLED" and r.pf.position(X, "s").qty == 2.0
    assert r.kinds()[n:] == ["order_filled", "fill", "position", "account"]
    assert [type(e) for e in r.eng.handler_errors] == [RuntimeError]
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
    r.broker.accept_tags = {"resolved_qty": 4.0}
    q = r.submit(qty=None, size_frac=0.5, type="MARKET")
    assert q.qty is None and q.size_frac == 0.5 and r.eng._resolved[q.client_id] == 4.0
    r.eng.apply(r.fill(q, 1.0))
    assert q.status.value == "PARTIALLY_FILLED"
    r.eng.apply(r.fill(q, 3.0))
    assert q.status.value == "FILLED"
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

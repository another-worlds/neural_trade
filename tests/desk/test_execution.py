"""NT-224 (DESK-1.3): ExecutionEngine pinned by scripted broker events with literal expected states.

The broker is a scripted ``FakeBroker`` and the journal a list; nothing here needs a database or a bar.
"""
from __future__ import annotations

import dataclasses
import itertools

import pytest

from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import SimClock
from neural_trade.desk.core.execution import ExecutionEngine, UnknownOrder
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

    def submit(self, order):
        self.submitted.append(order)
        return OrderEvent(1000, 1000, self.submit_kind,
                          dataclasses.replace(order, reason=self.submit_reason, venue_id=f"v-{order.client_id}"))

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
        self.risk = RiskGuard(limits or RiskLimits.unlimited(), ins, self.pf, self.clock)
        self.broker, self.bus, self.journal, self.seen = FakeBroker(), EventBus(), [], []
        for topic in ("order", "fill", "position", "account"):
            self.bus.subscribe(topic, lambda e, t=topic: self.seen.append((t, e)))
        self.eng = ExecutionEngine(self.broker, self.pf, self.risk, self.bus, self.clock,
                                   lambda k, p: self.journal.append((k, p)))
        self.eng.set_marks({X: 100.0})
        self.checks = []
        real = self.risk.check

        def spy(order, **kw):  # records every call and its decision
            d = real(order, **kw)
            self.checks.append((order.client_id, kw, d))
            return d

        self.risk.check = spy

    def submit(self, side="BUY", qty=2.0, type="LIMIT", price=100.0, **kw):
        cid = kw.pop("client_id", f"c{next(_ids)}")
        o = Order(client_id=cid, instrument_id=X, account_id="acc", strategy_id="s", side=side, type=type,
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


def test_resubmitting_a_client_id_returns_the_existing_order():
    r = Rig()
    o = r.submit(client_id="same")
    n = len(r.journal)
    again = r.eng.submit(dataclasses.replace(o, qty=9.0, status="INITIALIZED"))
    assert again is o and o.qty == 2.0 and len(r.broker.submitted) == 1 and len(r.journal) == n
    r.broker.venue["den"] = None
    r.eng.marks.clear()
    d = r.submit(client_id="den", type="MARKET")
    assert r.eng.submit(d) is d and r.broker.submitted[-1].client_id == "same"


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
    assert r.eng.cancel_all("other") == [] and a.status.value == "ACCEPTED"
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


def test_modify_is_not_in_the_broker_contract():
    r = Rig()
    with pytest.raises(NotImplementedError):
        r.eng.handle(ModifyOrder("c", price=1.0))
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

"""NT-225 (DESK-1.4): SimBroker pinned rule by rule (CONTRACTS A5) with hand-computed literals, the oracle's order of
operations, a parity check against ``strategy/backtest.run_backtest`` and a causality property test.

Bars are ``bar(i, open, high, low, close)`` of 1 minute: bar i opens at ``i * MIN`` and closes at ``(i + 1) * MIN``.
The rig drives the broker like the Engine will (A5.9): exits, entries, intrabar, then it sets the clock to the close.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neural_trade.desk.broker.base import UnknownBrokerOrder
from neural_trade.desk.broker.sim import SimBroker
from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import SimClock
from neural_trade.desk.core.costs import RESEARCH_ZERO, CostProfile
from neural_trade.desk.core.execution import ExecutionEngine
from neural_trade.desk.core.model import (AccountKind, Bar, Fill, Instrument, Order, OrderEvent, OrderStatus,
                                          OrderType, Side)
from neural_trade.desk.core.portfolio import Portfolio
from neural_trade.desk.core.risk import RiskGuard, RiskLimits
from neural_trade.strategy.backtest import BacktestConfig, Bars, run_backtest
from neural_trade.strategy.strategies import Strategy
from neural_trade.strategy.trades import Order as OracleOrder

MIN = 60_000
X = "X"
INS = Instrument(id=X, symbol="X/USDT", base="X", quote="USDT", price_increment=0.0, size_increment=0.0,
                 min_qty=0.0, min_notional=0.0, is_spot=False)
PRICED = CostProfile(name="priced", fee_bps_taker=10.0, fee_bps_maker=5.0, half_spread_bps=1.0, slippage_bps=2.0)
B, S = Side.BUY, Side.SELL
M, L_, SM, SL_ = OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_MARKET, OrderType.STOP_LIMIT


def bar(i, o, h, lo, c):
    return Bar(X, 1, i * MIN, o, h, lo, c, 1.0, True)


def fills(evs):
    return [e for e in evs if isinstance(e, Fill)]


def kinds(evs):
    return [e.kind if isinstance(e, OrderEvent) else "fill" for e in evs]


class Rig:
    def __init__(self, cost=RESEARCH_ZERO, tiebreak="sl_first", cash=10_000.0, ins=INS):
        self.clock = SimClock(0)
        self.pf = Portfolio("acc", kind=AccountKind.BACKTEST, initial_cash=cash, clock=self.clock)
        self.b = SimBroker(ins, cost, self.pf, self.clock, tiebreak=tiebreak)
        self.orders: dict[str, Order] = {}

    def order(self, cid, side, typ, **kw):
        o = Order(client_id=cid, instrument_id=X, account_id="acc", side=side, type=typ,
                  ts_created=self.clock.now_ms(), **kw)
        self.orders[cid] = o
        return o

    def submit(self, cid, side, typ, **kw):
        if "qty" not in kw and "size_frac" not in kw:
            kw["qty"] = 1.0
        return self.b.submit(self.order(cid, side, typ, **kw))

    def apply(self, evs):
        for e in evs:
            if isinstance(e, Fill):
                self.pf.apply_fill(e)
                o = self.orders.get(e.client_id)
                if o is not None and o.bracket and not o.reduce_only:
                    self.pf.set_bracket(f"{X}-None", o.bracket)

    def feed(self, b):
        self.clock.set(b.ts_open_ms)
        out = []
        for step in (lambda: self.b.fill_exits(b), lambda: self.b.fill_entries(b, self.pf.realised_equity()),
                     lambda: self.b.on_bar_intrabar(b)):
            evs = step()
            self.apply(evs)
            out += evs
        self.clock.set(b.ts_close_ms)
        return out

    def hold(self, side, qty, price, strategy_id=None):
        """Put a position in the Portfolio without going through the broker."""
        f = Fill("seed", "seed", X, side, qty, price, price, 0.0, "USDT", "TAKER", 0)
        self.pf.apply_fill(f, strategy_id=strategy_id)


def first(evs):
    (f,) = fills(evs)
    return f


# ------------------------------------------------------------------ A5.1 market at the next open
def test_market_fills_at_the_next_bars_open_not_the_submit_bar():
    r = Rig()
    assert kinds(r.feed(bar(0, 100, 101, 99, 100))) == []
    ev = r.submit("m1", B, M, qty=2.0)  # decided at bar 0's close
    assert ev.kind == "accepted" and ev.order.venue_id == "sim-m1"
    f = first(r.feed(bar(1, 101, 102, 100, 101.5)))
    assert (f.price, f.mid, f.qty, f.fee, f.ts, f.client_id, f.side) == (101.0, 101.0, 2.0, 0.0, MIN, "m1", B)
    assert f.liquidity.value == "TAKER" and f.trade_id and f.fee_currency == "USDT"
    assert kinds(r.feed(bar(2, 102, 103, 101, 102))) == []  # filled once


def test_market_costs_are_the_oracle_arithmetic():
    r = Rig(PRICED)
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("m1", B, M, qty=2.0)
    f = first(r.feed(bar(1, 101, 102, 100, 101.5)))
    assert f.mid == 101.0 and f.price == pytest.approx(101.0 * (1 + 3e-4), rel=1e-15)  # slip on the price
    assert f.fee == pytest.approx(10 / 1e4 * 2.0 * 101.0, rel=1e-15)  # taker fee on the MID notional


def test_size_frac_is_sized_at_the_fill_on_the_passed_equity_and_reported():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("m1", B, M, size_frac=0.5)
    f = first(r.feed(bar(1, 200, 201, 199, 200)))  # 0.5 x 10000 / 200
    assert f.qty == 25.0 and r.b.resolved["m1"] == 25.0


def test_size_frac_above_one_is_clipped_and_equity_argument_is_used():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("m1", B, M, size_frac=3.0)
    b1 = bar(1, 125, 126, 124, 125)
    r.clock.set(b1.ts_open_ms)
    assert r.b.fill_exits(b1) == []
    f = first(r.b.fill_entries(b1, 5_000.0))  # the caller's equity, not the portfolio's 10000
    assert f.qty == 5_000.0 / 125


def test_size_frac_uses_realised_equity_not_cash_when_another_position_is_open():
    r = Rig()
    r.hold(B, 10.0, 100.0, strategy_id="s1")  # cash 9000, realised equity 9000 + 10 x 100 = 10000
    assert r.pf.cash == 9_000.0 and r.pf.realised_equity() == 10_000.0
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("m1", B, M, size_frac=1.0, strategy_id="s2")
    f = first(r.feed(bar(1, 100, 101, 99, 100)))
    assert f.qty == 100.0  # 10000 / 100; sizing on cash would give 90


def test_size_frac_that_resolves_to_nothing_expires_with_resolved_qty_zero():
    ins = dataclasses.replace(INS, size_increment=0.01, min_qty=0.5)
    r = Rig(ins=ins)
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("m1", B, M, size_frac=0.0001)  # 0.0001 x 10000 / 100 = 0.01, under min_qty 0.5
    (e,) = r.feed(bar(1, 100, 101, 99, 100))
    assert (e.kind, e.order.reason, e.resolved_qty) == ("expired", "below_minimum", 0.0)
    r.submit("m2", B, M, size_frac=0.123456)  # 0.123456 x 10000 / 100 = 12.3456, floored to the 0.01 step
    assert first(r.feed(bar(2, 100, 101, 99, 100))).qty == 12.34


def test_an_order_submitted_during_a_bars_cycle_waits_for_the_next_bar():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    b1 = bar(1, 100, 101, 99, 100)
    r.clock.set(b1.ts_open_ms)  # ts_created == the bar's open, so only the broker's own bar tracking can hold it back
    assert r.b.fill_exits(b1) == []
    r.submit("m", B, M)
    r.submit("l", B, L_, price=100.0)
    assert r.b.fill_entries(b1, 10_000.0) == [] and r.b.on_bar_intrabar(b1) == []
    assert sorted(f.client_id for f in fills(r.feed(bar(2, 100, 101, 99, 100)))) == ["l", "m"]


# ------------------------------------------------------------------ A5.2 limit
def test_limit_buy_touch_gap_and_resting():
    r = Rig(PRICED)
    r.feed(bar(0, 100, 101, 95, 100))  # bar 0's low is below the limit, but the order is born at its close
    assert r.submit("l1", B, L_, qty=3.0, price=99.0).kind == "accepted"
    assert kinds(r.feed(bar(1, 100, 101, 99.5, 100))) == []  # low 99.5 > 99: no touch, order rests
    f = first(r.feed(bar(2, 100, 101, 99.0, 100)))  # touch at exactly 99 counts, fills at min(99, open 100)
    assert (f.mid, f.ts, f.liquidity.value) == (99.0, 2 * MIN, "MAKER")
    assert f.fee == pytest.approx(5 / 1e4 * 3.0 * 99.0, rel=1e-15)  # the maker fee
    r.submit("l2", B, L_, qty=1.0, price=99.0)
    f2 = first(r.feed(bar(3, 97, 98, 96, 97.5)))  # gaps through the limit: fills at the open 97
    assert (f2.mid, f2.ts) == (97.0, 3 * MIN)


def test_limit_sell_touch_and_gap():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("l1", S, L_, qty=1.0, price=103.0)
    assert kinds(r.feed(bar(1, 100, 102.5, 99, 101))) == []
    assert first(r.feed(bar(2, 101, 103.0, 100, 102))).mid == 103.0
    r.submit("l2", S, L_, qty=1.0, price=103.0)
    assert first(r.feed(bar(3, 105, 106, 104, 105))).mid == 105.0  # max(103, open 105)


# ------------------------------------------------------------------ A5.2 stop and stop-limit
def test_stop_buy_trigger_and_gap_reports_the_open_as_the_fill_price():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("s1", B, SM, qty=1.0, trigger_price=105.0)
    assert kinds(r.feed(bar(1, 100, 104.9, 99, 104))) == []  # high below the trigger
    evs = r.feed(bar(2, 104, 105.0, 103, 104.5))  # high == trigger: triggered, fills at max(105, 104)
    assert kinds(evs) == ["triggered", "fill"] and first(evs).mid == 105.0 and first(evs).ts == 2 * MIN
    r.submit("s2", B, SM, qty=2.0, trigger_price=105.0)
    f = first(r.feed(bar(3, 107, 108, 106, 107.5)))  # gaps above the trigger: fills at the open
    assert (f.mid, f.price, f.qty) == (107.0, 107.0, 2.0)


def test_stop_sell_trigger_and_gap_down():
    r = Rig(PRICED)
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("s1", S, SM, qty=1.0, trigger_price=95.0)
    f = first(r.feed(bar(1, 100, 101, 95.0, 96)))
    assert f.mid == 95.0 and f.price == pytest.approx(95.0 * (1 - 3e-4), rel=1e-15)
    r.submit("s2", S, SM, qty=1.0, trigger_price=95.0)
    assert first(r.feed(bar(2, 93, 94, 92, 93))).mid == 93.0  # min(95, open 93)


def test_stop_limit_has_two_stages_and_the_limit_starts_after_the_trigger_bar():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("sl", B, SL_, qty=1.0, trigger_price=105.0, price=106.0)
    evs = r.feed(bar(1, 100, 106, 99, 105))  # trigger touched; this bar's low is under the limit but stage two waits
    assert kinds(evs) == ["triggered"]
    assert kinds(r.feed(bar(2, 107, 108, 106.5, 107))) == []  # triggered, limit 106 not reached (low 106.5)
    f = first(r.feed(bar(3, 106.5, 107, 105.5, 106)))  # touch: min(106, open 106.5)
    assert (f.mid, f.ts, f.liquidity.value) == (106.0, 3 * MIN, "MAKER")


# ------------------------------------------------------------------ A5.3 brackets
def entry_with_bracket(r, side=B, tp=105.0, sl=95.0, offset=False, qty=1.0):
    r.feed(bar(0, 100, 100, 100, 100))
    r.submit("e1", side, M, qty=qty, bracket={"take_profit": tp, "stop_loss": sl, "tp_is_offset": offset,
                                              "max_hold": None})


@pytest.mark.parametrize("tiebreak, reason, price", [("sl_first", "SL", 95.0), ("tp_first", "TP", 105.0)])
def test_long_bracket_both_hit_in_the_fill_bar_follows_the_tiebreak(tiebreak, reason, price):
    r = Rig(tiebreak=tiebreak)
    entry_with_bracket(r)
    evs = r.feed(bar(1, 100, 106, 94, 100))  # entry at the open AND the same bar's high/low checked (fill bar included)
    assert kinds(evs) == ["fill", "accepted", "fill"]
    entry, exit_ = fills(evs)
    assert (entry.mid, entry.side, entry.ts) == (100.0, B, MIN)
    assert (exit_.mid, exit_.side, exit_.qty, exit_.ts) == (price, S, 1.0, MIN)
    assert evs[1].order.tags["reason"] == reason and evs[1].order.client_id.startswith("bracket:X-None:" + reason)
    assert r.pf.positions["X-None"].qty == 0.0  # the engine's job, here the rig: position closed


@pytest.mark.parametrize("tiebreak, reason, price", [("sl_first", "SL", 105.0), ("tp_first", "TP", 95.0)])
def test_short_bracket_tiebreak(tiebreak, reason, price):
    r = Rig(tiebreak=tiebreak)
    entry_with_bracket(r, side=S, tp=95.0, sl=105.0)
    evs = r.feed(bar(1, 100, 106, 94, 100))
    assert (fills(evs)[1].mid, fills(evs)[1].side, evs[1].order.tags["reason"]) == (price, B, reason)


def test_bracket_gap_through_the_stop_and_through_the_take_profit_fills_at_the_open():
    r = Rig()
    entry_with_bracket(r)
    r.feed(bar(1, 100, 101, 99, 100))
    f = fills(r.feed(bar(2, 90, 92, 89, 91)))[0]  # sl 95, open 90
    assert (f.mid, f.ts) == (90.0, 2 * MIN)
    r2 = Rig()
    entry_with_bracket(r2)
    r2.feed(bar(1, 100, 101, 99, 100))
    assert fills(r2.feed(bar(2, 108, 109, 107, 108)))[0].mid == 108.0  # tp 105, open 108


def test_bracket_exit_costs_are_taker_costs_on_the_level():
    r = Rig(PRICED)
    entry_with_bracket(r, qty=2.0)
    evs = r.feed(bar(1, 100, 105.5, 99, 105))  # TP only
    exit_ = fills(evs)[1]
    assert exit_.mid == 105.0 and exit_.price == pytest.approx(105.0 * (1 - 3e-4), rel=1e-15)
    assert exit_.fee == pytest.approx(10 / 1e4 * 2.0 * 105.0, rel=1e-15) and exit_.liquidity.value == "TAKER"


def test_offset_bracket_resolves_against_the_entry_bars_open_not_a_later_one():
    r = Rig()
    entry_with_bracket(r, tp=5.0, sl=-4.0, offset=True)  # fills at open 100: tp 105, sl 96
    r.feed(bar(1, 100, 101, 99, 100.5))
    evs = r.feed(bar(2, 101, 105.5, 100, 105))  # a later open of 101 would make tp 106
    assert fills(evs)[0].mid == 105.0 and evs[0].order.tags["reason"] == "TP"


def test_no_bracket_check_on_a_bar_before_the_entry_fill():
    r = Rig()
    r.feed(bar(0, 100, 100, 100, 100))
    r.submit("e1", B, M, qty=1.0, bracket={"take_profit": 99.0, "stop_loss": 95.0, "tp_is_offset": False,
                                           "max_hold": None})
    evs = r.feed(bar(1, 100, 100.0, 100.0, 100.0))  # tp 99 is below the entry: hit at once on the fill bar (gap rule)
    assert [e.order.tags.get("reason") for e in evs if isinstance(e, OrderEvent)] == ["TP"]


# ------------------------------------------------------------------ OCO
def oco_pair(r, tiebreak_bar):
    r.hold(B, 1.0, 100.0)
    r.feed(bar(0, 100, 100, 100, 100))
    r.submit("tp", S, L_, qty=1.0, price=105.0, reduce_only=True, tags={"oco": "g"})
    r.submit("sl", S, SM, qty=1.0, trigger_price=95.0, reduce_only=True, tags={"oco": "g"})
    return r.feed(tiebreak_bar)


@pytest.mark.parametrize("tiebreak, winner, price", [("sl_first", "sl", 95.0), ("tp_first", "tp", 105.0)])
def test_oco_one_fills_the_sibling_is_cancelled_and_a_same_bar_tie_follows_the_tiebreak(tiebreak, winner, price):
    r = Rig(tiebreak=tiebreak)
    evs = oco_pair(r, bar(1, 100, 106, 94, 100))  # both touched
    (f,) = fills(evs)
    assert (f.client_id, f.mid) == (winner, price)
    (c,) = [e for e in evs if isinstance(e, OrderEvent) and e.kind == "canceled"]
    assert c.order.client_id == ({"tp", "sl"} - {winner}).pop() and c.order.reason == "oco_sibling_filled"
    with pytest.raises(UnknownBrokerOrder):
        r.b.cancel(c.order.client_id)  # gone
    assert kinds(r.feed(bar(2, 100, 120, 80, 100))) == []  # nothing left to fill


def test_oco_take_profit_alone_cancels_the_stop():
    r = Rig()
    evs = oco_pair(r, bar(1, 100, 105, 99, 104))
    assert [(e.kind if isinstance(e, OrderEvent) else "fill") for e in evs] == ["fill", "canceled"]
    assert first(evs).client_id == "tp" and r.b.pending == []


# ------------------------------------------------------------------ TIME, EOW, reduce_only
def test_time_exit_is_a_reduce_only_market_filled_at_the_next_open():
    r = Rig()
    r.hold(B, 1.0, 100.0)
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("time", S, M, qty=1.0, reduce_only=True, tags={"reason": "TIME"})  # queued at the close of bar 0
    f = first(r.feed(bar(1, 103, 104, 102, 103.5)))
    assert (f.mid, f.ts, f.side) == (103.0, MIN, S)


def test_exit_fills_before_entry_and_entry_is_sized_after_it():
    r = Rig()
    r.hold(B, 10.0, 100.0)  # cash 9000, realised 10000
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("x", S, M, qty=10.0, reduce_only=True)
    r.submit("e", S, M, size_frac=1.0)  # a reversal entry
    b1 = bar(1, 110, 111, 109, 110)
    r.clock.set(b1.ts_open_ms)
    (fx,) = fills(r.b.fill_exits(b1))
    r.apply([fx])  # cash 9000 + 1100 = 10100 = realised equity of a flat account
    (fe,) = fills(r.b.fill_entries(b1, r.pf.realised_equity()))
    assert (fx.client_id, fx.mid, fe.client_id, fe.qty) == ("x", 110.0, "e", 10_100.0 / 110)


def test_eow_exit_on_a_data_break_fills_at_the_close_and_not_at_the_next_open():
    r = Rig()
    r.hold(B, 1.0, 100.0)
    r.feed(bar(0, 100, 101, 99, 100))
    b1 = bar(1, 100, 103, 99, 102)  # a hole follows bar 1 (breaks[1])
    r.feed(b1)
    r.submit("eow", S, M, qty=1.0, reduce_only=True, tags={"fill_at": "close", "reason": "EOW"})
    assert r.b.fill_exits(bar(1, 100, 103, 99, 102)) == []  # not an open fill
    (f,) = fills(r.b.fill_at_close(b1))
    assert (f.mid, f.ts, f.client_id) == (102.0, 2 * MIN, "eow")
    assert r.b.pending == []


def test_reduce_only_is_clipped_to_the_position_and_expires_without_one():
    r = Rig()
    r.hold(B, 2.0, 100.0)
    r.feed(bar(0, 100, 100, 100, 100))
    r.submit("a", S, M, qty=5.0, reduce_only=True)
    r.submit("b", S, M, qty=1.0, reduce_only=True)  # the position is spent by "a" in the same call
    r.submit("c", B, M, qty=1.0, reduce_only=True)  # the wrong side
    evs = r.feed(bar(1, 100, 101, 99, 100))
    assert [(e.client_id, e.qty) for e in fills(evs)] == [("a", 2.0)]
    assert [(e.order.client_id, e.kind, e.order.reason) for e in evs if isinstance(e, OrderEvent)] == [
        ("b", "expired", "reduce_only_no_position"), ("c", "expired", "reduce_only_no_position")]


def test_same_bar_two_reduce_only_limits_cannot_oversell():
    r = Rig()
    r.hold(B, 1.0, 100.0)
    r.feed(bar(0, 100, 100, 100, 100))
    r.submit("a", S, L_, qty=1.0, price=101.0, reduce_only=True)
    r.submit("b", S, L_, qty=1.0, price=102.0, reduce_only=True)
    evs = r.feed(bar(1, 100, 103, 99, 100))
    assert [f.client_id for f in fills(evs)] == ["a"]
    assert [(e.order.client_id, e.kind) for e in evs if isinstance(e, OrderEvent)] == [("b", "expired")]


# ------------------------------------------------------------------ cancel, modify, tif, rejections
def test_cancel_before_the_bar_wins_and_a_filled_order_cannot_be_cancelled():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("l1", B, L_, price=99.0)
    r.submit("l2", B, L_, price=99.0)
    ev = r.b.cancel("l1")
    assert (ev.kind, ev.order.client_id) == ("canceled", "l1")
    evs = r.feed(bar(1, 100, 101, 98, 100))  # would have touched both
    assert [f.client_id for f in fills(evs)] == ["l2"]
    for cid in ("l1", "l2", "nope"):
        with pytest.raises(UnknownBrokerOrder):
            r.b.cancel(cid)


def test_modify_changes_price_and_quantity_of_a_resting_order():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("l1", B, L_, qty=1.0, price=95.0)
    ev = r.b.modify("l1", price=99.0, qty=3.0)
    assert (ev.kind, ev.order.price, ev.order.qty) == ("modified", 99.0, 3.0)
    f = first(r.feed(bar(1, 100, 101, 99, 100)))
    assert (f.mid, f.qty) == (99.0, 3.0)
    with pytest.raises(UnknownBrokerOrder):
        r.b.modify("l1", price=1.0)
    r.submit("sf", B, L_, size_frac=0.5, price=90.0)
    with pytest.raises(ValueError, match="size_frac"):
        r.b.modify("sf", qty=2.0)
    with pytest.raises(ValueError, match="invalid_price"):
        r.b.modify("sf", price=-1.0)


def test_tif_ioc_expires_after_its_first_bar_and_day_expires_on_the_next_utc_day():
    r = Rig()
    r.feed(bar(0, 100, 101, 99, 100))
    r.submit("ioc", B, L_, price=90.0, tif="IOC")
    r.submit("gtc", B, L_, price=90.0)
    evs = r.feed(bar(1, 100, 101, 99, 100))
    assert [(e.order.client_id, e.kind, e.order.reason) for e in evs] == [("ioc", "expired", "ioc_unfilled")]
    assert [o.client_id for o in r.b.pending] == ["gtc"]
    d = Rig()
    d.clock.set(1_000)
    d.submit("day", B, L_, price=90.0, tif="DAY")
    day = 86_400_000 // MIN
    assert d.feed(bar(day - 1, 100, 101, 99, 100)) == []  # still the same UTC day
    (e,) = d.feed(bar(day, 100, 101, 99, 100))
    assert (e.kind, e.order.reason) == ("expired", "day_expired")


@pytest.mark.parametrize("kw, reason", [
    (dict(typ=L_), "invalid_price"), (dict(typ=SM), "invalid_trigger"), (dict(typ=SL_, trigger_price=5.0), "invalid_price"),
    (dict(typ=M, qty=-1.0), "invalid_quantity"), (dict(typ=SM, trigger_price=5.0, tif="FOK"), "tif_not_supported"),
    (dict(typ=M, instrument_id="Y"), "wrong_instrument")])
def test_invalid_orders_are_rejected_with_a_reason(kw, reason):
    r = Rig()
    typ = kw.pop("typ")
    iid = kw.pop("instrument_id", X)
    o = dataclasses.replace(r.order("bad", B, typ, **{"qty": 1.0, **kw}), instrument_id=iid)
    ev = r.b.submit(o)
    assert (ev.kind, ev.order.reason) == ("rejected", reason)


def test_duplicate_client_id_is_rejected():
    r = Rig()
    assert r.submit("d", B, M).kind == "accepted"
    assert r.submit("d", B, M).order.reason == "duplicate_client_id"


# ------------------------------------------------------------------ feeding rules
def test_open_bars_and_out_of_order_bars_are_refused():
    r = Rig()
    with pytest.raises(ValueError, match="CLOSED"):
        r.b.fill_exits(dataclasses.replace(bar(0, 1, 1, 1, 1), is_closed=False))
    r.feed(bar(5, 100, 101, 99, 100))
    with pytest.raises(ValueError, match="time order"):
        r.b.fill_exits(bar(4, 100, 101, 99, 100))


# ------------------------------------------------------------------ parity with the oracle (small harness)
class _Sig:
    def __len__(self):
        return 12


class _Scripted(Strategy):
    def __init__(self, orders):
        self.orders = orders

    def decide(self, s, t):
        return self.orders.get(t)


OP = [100, 100, 101, 102, 104, 104, 105, 101, 102, 102, 103, 103]
H = [101, 102, 103, 105, 105, 106.5, 106, 102, 103, 103, 104, 104]
LO = [99, 99.5, 100.5, 101.5, 103, 103.5, 104, 100, 101, 101, 102, 102]
C = [100, 101, 102, 104, 104, 105, 105, 101.5, 102, 103, 103, 103.5]
SCRIPT = {0: dict(side="LONG", tp=104.0, sl=98.0), 3: dict(side="SHORT", tp=100.0, sl=106.0),
          5: dict(side="LONG", frac=0.5, sl=103.0), 7: dict(side="LONG", max_hold=2), 10: dict(side="LONG")}


def oracle(cost, breaks):
    orders = {t: OracleOrder(d["side"], d.get("frac", 1.0), tp=d.get("tp"), sl=d.get("sl"), max_hold=d.get("max_hold"))
              for t, d in SCRIPT.items()}
    cfg = BacktestConfig(fee_bps=cost.fee_bps_taker, half_spread_bps=cost.half_spread_bps,
                         slippage_bps=cost.slippage_bps)
    return run_backtest(_Sig(), Bars(OP, H, LO, C, breaks), _Scripted(orders), cfg)


def desk_run(cost, breaks):
    """The Engine's loop of A5.9 in miniature: SimBroker + ExecutionEngine + Portfolio, orders from SCRIPT."""
    clock = SimClock(0)
    pf = Portfolio("acc", kind=AccountKind.BACKTEST, initial_cash=10_000.0, clock=clock)
    broker = SimBroker(INS, cost, pf, clock)
    eng = ExecutionEngine(broker, pf, RiskGuard(RiskLimits.unlimited(), INS, pf, clock), EventBus(), clock,
                          lambda k, p: None)
    log: list[tuple] = []  # (bar, side, mid, price, reason, entry?)
    state = {"t": 0, "entry_bar": None, "max_hold": 30, "n": 0}

    def apply(evs):
        for e in evs:
            if isinstance(e, OrderEvent) and e.order.client_id.startswith("bracket:") and eng.order(e.order.client_id) is None:
                eng._orders[e.order.client_id] = e.order  # the Engine registers a broker-made exit (CONTRACT gap)
                eng._originals[e.order.client_id] = dataclasses.replace(e.order)
            was_flat = pf.positions.get("X-s") is None or pf.positions["X-s"].qty == 0
            eng.apply(e)
            if isinstance(e, Fill):
                o = eng.order(e.client_id)
                log.append((state["t"], e.side.value, e.mid, e.price, o.tags.get("reason"), was_flat))
                if was_flat:
                    state["entry_bar"] = state["t"]

    def submit(cid, side, **kw):
        state["n"] += 1
        eng.submit(Order(client_id=f"{cid}{state['n']}", instrument_id=X, account_id="acc", strategy_id="s", side=side,
                         type=M, **kw))

    n = 12
    for t in range(n):
        b = bar(t, OP[t], H[t], LO[t], C[t])
        state["t"] = t
        clock.set(b.ts_open_ms)
        apply(broker.fill_exits(b))
        apply(broker.fill_entries(b, pf.realised_equity()))
        apply(broker.on_bar_intrabar(b))
        eng.set_marks({X: b.close})
        clock.set(b.ts_close_ms)
        pos = pf.positions.get("X-s")
        broken = breaks is not None and bool(breaks[t])
        if pos is not None and pos.qty > 0:
            held = t - state["entry_bar"] + 1
            if broken or t == n - 1:
                submit("eow", Side.SELL if pos.side.value == "LONG" else Side.BUY, qty=pos.qty, reduce_only=True,
                       tags={"fill_at": "close", "reason": "EOW"})
                apply(broker.fill_at_close(b))
            elif held >= state["max_hold"]:
                submit("time", Side.SELL if pos.side.value == "LONG" else Side.BUY, qty=pos.qty, reduce_only=True,
                       tags={"reason": "TIME"})
        elif t < n - 1 and not broken and t in SCRIPT:
            d = SCRIPT[t]
            state["max_hold"] = d.get("max_hold") or 30
            submit("entry", Side.BUY if d["side"] == "LONG" else Side.SELL, size_frac=d.get("frac", 1.0),
                   tags={"reason": "ENTRY"}, bracket={"take_profit": d.get("tp"), "stop_loss": d.get("sl"),
                                                      "tp_is_offset": False, "max_hold": d.get("max_hold")})
        pf.mark({X: b.close}, b.ts_close_ms)
    return pf, log


def trades_of(log):
    out, cur = [], None
    for bar_, side, _mid, price, reason, entry in log:
        if entry:
            cur = (bar_, side, price)
        else:
            out.append((cur[0], bar_, "LONG" if cur[1] == "BUY" else "SHORT", cur[2], price, reason))
    return out


BREAKS = np.array([False] * 8 + [True] + [False] * 3)


@pytest.mark.parametrize("breaks", [None, BREAKS], ids=["no_break", "break_after_bar_8"])
@pytest.mark.parametrize("cost", [RESEARCH_ZERO, CostProfile(name="p", fee_bps_taker=10, fee_bps_maker=10,
                                                              half_spread_bps=1, slippage_bps=2)], ids=["0bps", "10_1_2"])
def test_parity_with_run_backtest_trades_equity_and_costs(cost, breaks):
    res = oracle(cost, breaks)
    pf, log = desk_run(cost, breaks)
    mine = trades_of(log)
    assert len(mine) == len(res.trades) == 5
    for m, tr in zip(mine, res.trades):
        assert m[:3] == (tr.entry_bar, tr.exit_bar, tr.side) and m[5] == tr.exit_reason
        assert m[3] == pytest.approx(tr.entry_price, rel=1e-12) and m[4] == pytest.approx(tr.exit_price, rel=1e-12)
    assert np.allclose([e for _, e in pf.equity_curve], res.equity[1:], rtol=0, atol=1e-9)
    assert [t.costs for t in pf.trades] == pytest.approx([t.costs for t in res.trades], abs=1e-9)
    assert [t.gross_pnl for t in pf.trades] == pytest.approx([t.gross_pnl for t in res.trades], abs=1e-9)


def test_parity_example_has_the_hand_computed_trades():
    """Literals (0 bps): TP gap-free, a short stopped, a long stopped through the open, TIME, EOW."""
    _, log = desk_run(RESEARCH_ZERO, None)
    assert trades_of(log) == [(1, 3, "LONG", 100.0, 104.0, "TP"), (4, 5, "SHORT", 104.0, 106.0, "SL"),
                              (6, 7, "LONG", 105.0, 101.0, "SL"), (8, 10, "LONG", 102.0, 103.0, "TIME"),
                              (11, 11, "LONG", 103.0, 103.5, "EOW")]
    _, log = desk_run(RESEARCH_ZERO, BREAKS)  # a hole after bar 8: the long of bar 8 closes at that close, EOW
    assert trades_of(log)[3] == (8, 8, "LONG", 102.0, 102.0, "EOW")


# ------------------------------------------------------------------ causality: random bars, random submit times
def scenario(seed, n=40):
    rng = np.random.default_rng(seed)
    px = 100.0 + np.cumsum(rng.normal(0, 1.0, n + 1))
    bars = [bar(i, px[i], px[i] + abs(rng.normal(0, 1)) + 0.1, px[i] - abs(rng.normal(0, 1)) - 0.1, px[i + 1])
            for i in range(n)]
    plan = []
    for i in range(n):
        for _ in range(int(rng.integers(0, 3))):
            typ = [M, L_, SM, SL_][int(rng.integers(0, 4))]
            side = [B, S][int(rng.integers(0, 2))]
            off = float(rng.normal(0, 1.5))
            kw = dict(qty=1.0, reduce_only=bool(rng.integers(0, 4) == 0))
            if typ is L_:
                kw["price"] = px[i + 1] + off
            elif typ in (SM, SL_):
                kw["trigger_price"] = px[i + 1] + off
                if typ is SL_:
                    kw["price"] = px[i + 1] + off + (0.5 if side is B else -0.5)
            if typ is M and rng.integers(0, 6) == 0:
                kw["tags"] = {"fill_at": "close"}
            plan.append((i, side, typ, kw))
    return bars, plan


def run_scenario(seed, n):
    bars, plan = scenario(seed)
    r = Rig()
    made: dict[str, tuple[int, Order]] = {}
    out = []
    for i, b in enumerate(bars[:n]):
        evs = r.feed(b)
        # the rig's feed ends with the clock at the close; fill_at_close is the Engine's step 4
        extra = r.b.fill_at_close(b)
        r.apply(extra)
        out += [(i, e) for e in evs + extra]
        for k, (j, side, typ, kw) in enumerate(plan):
            if j == i:
                cid = f"o{k}"
                r.submit(cid, side, typ, **kw)
                made[cid] = (i, r.orders[cid])
    return out, made


@pytest.mark.parametrize("seed", range(30))
def test_no_fill_ever_precedes_the_orders_bar_and_the_broker_cannot_see_future_bars(seed):
    out, made = run_scenario(seed, 40)
    assert any(isinstance(e, Fill) for _, e in out)
    for i, e in out:
        if not isinstance(e, Fill) or e.client_id.startswith("bracket:"):
            continue
        sub_bar, o = made[e.client_id]
        assert e.ts >= o.ts_created  # never before the order existed
        if o.tags.get("fill_at") == "close":
            assert i > sub_bar and e.ts == (i + 1) * MIN  # at the close of a later bar
        else:
            assert i > sub_bar and e.ts == i * MIN  # a fill at t+1 earliest; a resting order only in later bars
    # prefix invariance: the same run stopped after 25 bars reports exactly the fills of those 25 bars
    short, _ = run_scenario(seed, 25)
    key = lambda items: [(i, repr(e)) for i, e in items if i < 25]  # noqa: E731
    assert key(short) == key(out)


def test_engine_accepts_the_broker_reports_end_to_end():
    """Submit through the real ExecutionEngine: accepted -> triggered -> filled statuses are legal edges."""
    clock = SimClock(0)
    pf = Portfolio("acc", kind=AccountKind.BACKTEST, clock=clock)
    broker = SimBroker(INS, RESEARCH_ZERO, pf, clock)
    eng = ExecutionEngine(broker, pf, RiskGuard(RiskLimits.unlimited(), INS, pf, clock), EventBus(), clock,
                          lambda k, p: None)
    eng.set_marks({X: 100.0})
    o = eng.submit(Order(client_id="s", instrument_id=X, account_id="acc", side=B, type=SM, qty=1.0,
                         trigger_price=105.0))
    assert o.status is OrderStatus.ACCEPTED and o.venue_id == "sim-s"
    b = bar(1, 107, 108, 106, 107)
    clock.set(b.ts_open_ms)
    eng.apply_all(broker.fill_exits(b) + broker.fill_entries(b, pf.realised_equity()) + broker.on_bar_intrabar(b))
    assert (o.status, o.filled_qty, o.avg_fill_price) == (OrderStatus.FILLED, 1.0, 107.0)
    c = eng.submit(Order(client_id="c", instrument_id=X, account_id="acc", side=S, type=L_, qty=1.0, price=200.0))
    eng.cancel("c")
    assert c.status is OrderStatus.CANCELED and broker.pending == []

"""NT-223 (DESK-1.2): RiskGuard pinned by literal numbers and exact reason strings."""
from __future__ import annotations

import dataclasses
import itertools
import math

import pytest


from neural_trade.desk.core.clock import SimClock
from neural_trade.desk.core.costs import RESEARCH_ZERO
from neural_trade.desk.core.fills import market_fill
from neural_trade.desk.core.model import Instrument, Order
from neural_trade.desk.core.portfolio import Portfolio
from neural_trade.desk.core.risk import (DAY_MS, RiskGuard, RiskLimits, TradingState, round_down, round_nearest)

X = "X"
_ids = itertools.count()
T0 = 5 * DAY_MS + 1_000  # shortly after a UTC midnight


def inst(**kw):
    base = dict(id=X, symbol="X/USDT", base="X", quote="USDT", price_increment=0.01, size_increment=0.001,
                min_qty=0.001, min_notional=10.0, max_qty=5.0, is_spot=True)
    base.update(kw)
    return Instrument(**base)


def order(side="BUY", qty=1.0, type="MARKET", price=None, strategy="s", **kw):
    if "size_frac" in kw:
        qty = None
    return Order(client_id=f"c{next(_ids)}", instrument_id=X, account_id="acc", strategy_id=strategy, side=side,
                 type=type, qty=qty, price=price, **kw)


def make(limits=None, ins=None, cash=10_000.0, costs=None, **pkw):
    clock = SimClock(T0)
    ins = ins or inst()
    pf = Portfolio("acc", initial_cash=cash, allow_short=not ins.is_spot, clock=clock, **pkw)
    return RiskGuard(limits or RiskLimits.unlimited(), ins, pf, clock, costs=costs), pf, clock


def hold(pf, side, qty, price, strategy="s"):
    f = dataclasses.replace(market_fill(price, 1 if side == "BUY" else -1, RESEARCH_ZERO, qty), instrument_id=X,
                            trade_id=f"t{next(_ids)}")
    pf.apply_fill(f, strategy_id=strategy)


# ---------------------------------------------------------------- rounding
def test_round_helpers_half_way():
    assert round_nearest(1.005, 0.01) == 1.01  # half-way rounds up (exact decimal, not float 1.00499..)
    assert round_nearest(1.004, 0.01) == 1.0
    assert round_nearest(0.015, 0.01) == 0.02
    assert round_down(1.2349, 0.001) == 1.234
    assert round_down(0.0019, 0.001) == 0.001
    assert round_down(1.0, 0.001) == 1.0
    assert round_nearest(1.23456, 0) == 1.23456 and round_down(1.23456, 0) == 1.23456


def test_qty_floors_price_rounds_half_up():
    g, _, _ = make()
    d = g.check(order(qty=1.2349, type="LIMIT", price=100.005), ref_price=100.0)
    assert (d.allowed, d.qty, d.price, d.reason) == (True, 1.234, 100.01, "")


def test_qty_rounds_to_zero_denied():
    g, _, _ = make()
    d = g.check(order(qty=0.0009), ref_price=100.0)
    assert (d.allowed, d.reason, d.qty) == (False, "qty_rounds_to_zero", None)


def test_price_rounds_to_zero_denied():
    g, _, _ = make()
    assert g.check(order(type="LIMIT", price=0.004), ref_price=100.0).reason == "price_rounds_to_zero"


def test_zero_increments_do_not_round():
    g, _, _ = make(ins=inst(price_increment=0, size_increment=0, min_qty=0, min_notional=0, max_qty=None))
    d = g.check(order(qty=0.123456789, type="LIMIT", price=10.123456), ref_price=10.0)
    assert (d.qty, d.price) == (0.123456789, 10.123456)


def test_invalid_orders_denied():
    g, _, _ = make()
    assert g.check(order(), ref_price=float("nan")).reason == "invalid_order"
    assert g.check(order(qty=-1.0), ref_price=100.0).reason == "invalid_order"
    assert g.check(order(qty=float("inf")), ref_price=100.0).reason == "invalid_order"
    assert g.check(order(type="LIMIT", price=-1.0), ref_price=100.0).reason == "invalid_order"
    o = order()
    o.instrument_id = "Y"
    assert g.check(o, ref_price=100.0).reason == "wrong_instrument"


def test_size_frac_resolves_on_equity_and_floors():
    g, _, _ = make(cash=1_000.0)
    d = g.check(order(size_frac=0.5), ref_price=300.0)  # 0.5 x 1000 / 300 = 1.6666..
    assert (d.allowed, d.qty) == (True, 1.666)
    assert g.check(order(size_frac=7.0), ref_price=1000.0).qty == 1.0  # clipped to 1.0 -> 1000/1000
    assert g.check(order(size_frac=float("nan")), ref_price=100.0).reason == "invalid_order"


# ---------------------------------------------------------------- instrument limits
def test_min_qty_and_max_qty():
    g, _, _ = make(ins=inst(min_qty=0.5, size_increment=0.001, min_notional=0))
    assert g.check(order(qty=0.5), ref_price=100.0).allowed
    assert g.check(order(qty=0.499), ref_price=100.0).reason == "min_qty"
    g2, _, _ = make()
    assert g2.check(order(qty=5.0), ref_price=100.0).allowed
    assert g2.check(order(qty=5.001), ref_price=100.0).reason == "max_qty"


def test_min_notional():
    g, _, _ = make()
    assert g.check(order(qty=0.1), ref_price=100.0).allowed  # exactly 10.0
    d = g.check(order(qty=0.099), ref_price=100.0)  # 9.9
    assert (d.allowed, d.reason) == (False, "min_notional")


# ---------------------------------------------------------------- RiskLimits
def test_max_order_notional():
    g, _, _ = make(RiskLimits(max_order_notional=100.0, max_position_notional=None, max_orders_per_minute=None,
                              daily_loss_limit=None))
    assert g.check(order(qty=2.0), ref_price=50.0).allowed  # notional 100.0 == limit passes
    d = g.check(order(qty=2.0), ref_price=50.01)  # 100.02
    assert (d.allowed, d.reason) == (False, "max_order_notional")


def test_max_position_buys_only_and_reductions_pass():
    lim = RiskLimits(max_order_notional=None, max_position_notional=300.0, max_orders_per_minute=None,
                     daily_loss_limit=None)
    g, pf, _ = make(lim)
    hold(pf, "BUY", 2.0, 100.0)
    assert g.check(order(qty=1.0), ref_price=100.0).allowed  # 3 x 100 = 300 == limit
    d = g.check(order(qty=1.001), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "max_position")
    assert g.check(order("SELL", qty=1.0), ref_price=100.0).allowed


def test_max_position_counts_the_orders_own_position_only():
    lim = RiskLimits(max_order_notional=None, max_position_notional=300.0, max_orders_per_minute=None,
                     daily_loss_limit=None)
    g, pf, _ = make(lim)
    hold(pf, "BUY", 3.0, 100.0, strategy="other")
    assert g.check(order(qty=3.0), ref_price=100.0).allowed  # strategy "s" holds nothing


def test_max_position_applies_to_a_short_on_margin():
    lim = RiskLimits(max_order_notional=None, max_position_notional=300.0, max_orders_per_minute=None,
                     daily_loss_limit=None)
    g, pf, _ = make(lim, ins=inst(is_spot=False))
    assert g.check(order("SELL", qty=3.0), ref_price=100.0).allowed
    assert g.check(order("SELL", qty=3.001), ref_price=100.0).reason == "max_position"


def test_orders_per_minute_sliding_window_boundary():
    lim = RiskLimits(max_order_notional=None, max_position_notional=None, max_orders_per_minute=2,
                     daily_loss_limit=None)
    g, _, clock = make(lim)
    assert g.check(order(), ref_price=100.0).allowed  # at T0
    clock.set(T0 + 30_000)
    assert g.check(order(), ref_price=100.0).allowed
    clock.set(T0 + 59_999)  # first order is 59.999 s old: still in the window
    d = g.check(order(), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "max_orders_per_minute")
    clock.set(T0 + 60_000)  # exactly 60 s old: out of the window
    assert g.check(order(), ref_price=100.0).allowed
    clock.set(T0 + 60_001)  # the second (T0+30s) is still in, the third just recorded
    assert g.check(order(), ref_price=100.0).reason == "max_orders_per_minute"


def test_denied_and_estimate_checks_take_no_rate_slot():
    lim = RiskLimits(max_order_notional=None, max_position_notional=None, max_orders_per_minute=1,
                     daily_loss_limit=None)
    g, _, _ = make(lim)
    assert g.check(order(), ref_price=100.0, record=False).allowed
    assert g.check(order(qty=0.0001), ref_price=100.0).reason == "qty_rounds_to_zero"
    assert g.check(order(), ref_price=100.0).allowed  # the first real slot
    assert g.check(order(), ref_price=100.0).reason == "max_orders_per_minute"


def test_daily_loss_limit_and_utc_rollover():
    lim = RiskLimits(max_order_notional=None, max_position_notional=None, max_orders_per_minute=None,
                     daily_loss_limit=100.0)
    g, pf, clock = make(lim)
    hold(pf, "BUY", 10.0, 100.0)  # long 10 at 100, cash 9000
    assert g.check(order(), ref_price=100.0).allowed  # day starts: equity 10000
    assert g.check(order(), ref_price=91.0).allowed  # loss 90 < 100
    d = g.check(order(), ref_price=90.0)  # loss exactly 100 -> reached
    assert (d.allowed, d.reason) == (False, "daily_loss_limit")
    assert g.check(order("SELL", qty=1.0), ref_price=90.0).allowed  # reducing is still allowed
    clock.set(6 * DAY_MS)  # next UTC day: baseline is today's equity 9000 + 10 x 90 = 9900
    assert g.check(order(), ref_price=90.0).allowed
    assert g.day_loss({X: 90.0}) == 0.0
    assert g.check(order(), ref_price=80.0).reason == "daily_loss_limit"  # 9900 - 9800 = 100, reached again


def test_daily_loss_one_ms_before_midnight_is_still_the_old_day():
    lim = RiskLimits(max_order_notional=None, max_position_notional=None, max_orders_per_minute=None,
                     daily_loss_limit=100.0)
    g, pf, clock = make(lim)
    hold(pf, "BUY", 10.0, 100.0)
    g.update({X: 100.0})
    clock.set(6 * DAY_MS - 1)
    assert g.check(order(), ref_price=90.0).reason == "daily_loss_limit"
    clock.set(6 * DAY_MS)
    assert g.check(order(), ref_price=90.0).allowed


# ---------------------------------------------------------------- spot no-short
def test_spot_sell_beyond_balance_denied():
    g, pf, _ = make()
    hold(pf, "BUY", 2.0, 100.0)
    assert g.check(order("SELL", qty=2.0), ref_price=100.0).allowed
    d = g.check(order("SELL", qty=2.001), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "spot_sell_exceeds_balance")
    assert g.check(order("SELL", qty=1.0), ref_price=100.0, marks={X: 100.0}).allowed


def test_spot_sell_with_no_holding_denied():
    g, _, _ = make()
    assert g.check(order("SELL", qty=1.0), ref_price=100.0).reason == "spot_sell_exceeds_balance"


def test_spot_short_without_own_long_denied_even_if_another_strategy_is_long():
    """Decision for the lead: CONTRACTS is silent; deny (conservative) with a distinct reason."""
    g, pf, _ = make()
    hold(pf, "BUY", 5.0, 100.0, strategy="other")
    d = g.check(order("SELL", qty=1.0, strategy="s"), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "spot_short_no_own_long")
    assert g.check(order("SELL", qty=5.0, strategy="other"), ref_price=100.0).allowed


def test_margin_instrument_may_short():
    g, _, _ = make(ins=inst(is_spot=False))
    assert g.check(order("SELL", qty=1.0), ref_price=100.0).allowed


# ---------------------------------------------------------------- states
def test_reducing_allows_only_reducing_orders():
    g, pf, _ = make()
    hold(pf, "BUY", 2.0, 100.0)
    g.reduce_only("test")
    assert g.state is TradingState.REDUCING
    d = g.check(order("BUY", qty=1.0), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "reducing_only")
    assert g.check(order("SELL", qty=1.5), ref_price=100.0).qty == 1.5
    assert g.check(order("SELL", qty=2.0), ref_price=100.0).allowed


def test_reducing_flat_position_denies_everything():
    g, _, _ = make(ins=inst(is_spot=False))
    g.reduce_only()
    assert g.check(order("BUY"), ref_price=100.0).reason == "reducing_only"
    assert g.check(order("SELL"), ref_price=100.0).reason == "reducing_only"


def test_reducing_flip_is_clipped_to_the_position():
    g, pf, _ = make(ins=inst(is_spot=False))
    hold(pf, "SELL", 2.0, 100.0)  # short 2 on margin
    g.reduce_only()
    d = g.check(order("BUY", qty=3.0), ref_price=100.0)
    assert (d.allowed, d.qty, d.reason) == (True, 2.0, "clipped_to_position")


def test_reduce_only_flag_in_active_state():
    g, pf, _ = make()
    hold(pf, "BUY", 2.0, 100.0)
    d = g.check(order("BUY", reduce_only=True), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "reduce_only_violation")
    assert g.check(order("SELL", qty=1.0, reduce_only=True), ref_price=100.0).qty == 1.0


def test_halted_denies_everything_even_reducing_and_reduce_only():
    g, pf, _ = make()
    hold(pf, "BUY", 2.0, 100.0)
    g.halt("kill")
    for o in (order("SELL", qty=1.0), order("SELL", qty=1.0, reduce_only=True), order("BUY")):
        d = g.check(o, ref_price=100.0)
        assert (d.allowed, d.reason) == (False, "halted")
    g.reduce_only()  # does not soften a halt
    assert g.state is TradingState.HALTED


def test_halt_idempotent_emits_once_and_keeps_first_reason():
    seen = []
    clock = SimClock(T0)
    g = RiskGuard(RiskLimits.unlimited(), inst(), Portfolio("acc", clock=clock), clock,
                  on_state_change=lambda o, n, r: seen.append((o, n, r)))
    g.halt("first")
    g.halt("second")
    assert seen == [(TradingState.ACTIVE, TradingState.HALTED, "first")]
    assert g.state_reason == "first"


def test_state_transitions():
    seen = []
    clock = SimClock(T0)
    g = RiskGuard(RiskLimits.unlimited(), inst(), Portfolio("acc", clock=clock), clock,
                  on_state_change=lambda o, n, r: seen.append((o.value, n.value)))
    g.reduce_only()
    g.reduce_only()  # no change, no event
    g.halt("x")
    g.resume()
    g.resume()
    g.halt("y")
    assert seen == [("ACTIVE", "REDUCING"), ("REDUCING", "HALTED"), ("HALTED", "ACTIVE"), ("ACTIVE", "HALTED")]


def test_unlimited_and_defaults():
    u = RiskLimits.unlimited()
    assert (u.max_order_notional, u.max_position_notional, u.max_orders_per_minute, u.daily_loss_limit) == (
        None, None, None, None)
    d = RiskLimits()
    assert (d.max_order_notional, d.max_position_notional, d.max_orders_per_minute, d.daily_loss_limit) == (
        1_000.0, 5_000.0, 30, 500.0)
    g, _, _ = make(u, ins=inst(is_spot=False, price_increment=0, size_increment=0, min_qty=0, min_notional=0,
                               max_qty=None))
    for _ in range(100):
        assert g.check(order("SELL", qty=1e6), ref_price=100.0).allowed


# ---------------------------------------------------------------- repair round 1 (QA FAIL on 235aa232)
def only(**kw):
    base = dict(max_order_notional=None, max_position_notional=None, max_orders_per_minute=None, daily_loss_limit=None)
    base.update(kw)
    return RiskLimits(**base)


def test_f1_flip_under_breached_daily_loss_is_clipped_to_the_position():
    g, pf, _ = make(only(daily_loss_limit=500.0), ins=inst(is_spot=False, max_qty=None))
    hold(pf, "BUY", 10.0, 100.0)  # cash 9000
    g.update({X: 100.0})  # day starts at equity 10000
    assert g.day_loss({X: 40.0}) == 600.0  # equity 9000 + 10 x 40 = 9400
    assert g.check(order("BUY", qty=1.0), ref_price=40.0).reason == "daily_loss_limit"
    d = g.check(order("SELL", qty=30.0), ref_price=40.0)
    assert (d.allowed, d.qty, d.reason) == (True, 10.0, "clipped_to_position")
    assert g.check(order("SELL", qty=4.0), ref_price=40.0).qty == 4.0


def test_f2_exit_of_a_large_position_is_exempt_from_order_notional_and_rate_limit():
    ins = inst(max_qty=None)
    g, pf, _ = make(RiskLimits(), ins=ins)
    hold(pf, "BUY", 50.0, 100.0)  # a 5000 position, the default max_position_notional
    for state in ("ACTIVE", "REDUCING"):
        if state == "REDUCING":
            g.reduce_only()
        d = g.check(order("SELL", qty=50.0, reduce_only=True), ref_price=100.0, record=False)
        assert (d.allowed, d.qty) == (True, 50.0), state
    # opening orders keep every limit
    assert g.resume() is None
    assert g.check(order("BUY", qty=11.0), ref_price=100.0).reason == "max_order_notional"


def test_f2_exit_is_exempt_from_the_rate_limit_and_takes_no_slot():
    g, pf, _ = make(only(max_orders_per_minute=1))
    hold(pf, "BUY", 2.0, 100.0)
    assert g.check(order("BUY", qty=1.0), ref_price=100.0).allowed  # takes the only slot
    assert g.check(order("BUY", qty=1.0), ref_price=100.0).reason == "max_orders_per_minute"
    assert g.check(order("SELL", qty=2.0, reduce_only=True), ref_price=100.0).allowed
    assert len(g._sent) == 1


def test_f2_halted_and_rounding_still_apply_to_exits():
    g, pf, _ = make()
    hold(pf, "BUY", 2.0, 100.0)
    assert g.check(order("SELL", qty=0.0001, reduce_only=True), ref_price=100.0).reason == "qty_rounds_to_zero"
    g.halt("x")
    assert g.check(order("SELL", qty=2.0, reduce_only=True), ref_price=100.0).reason == "halted"


def test_f3_size_frac_uses_realised_cash_not_marked_equity():
    g, pf, _ = make(cash=15_000.0, ins=inst(max_qty=None), kind="PAPER", costs=RESEARCH_ZERO)  # NT-292: explicit costs
    hold(pf, "BUY", 50.0, 200.0)  # cash 5000
    assert pf.cash == 5000.0
    # NT-289 (CONTRACTS A5.1/A5.9, lead's decision 2026-10-16): size_frac resolves on REALISED equity = cash +
    # signed qty x avg price = 5000 + 50 x 200 = 15000 -> 75 units (15000 / 200); on a PAPER spot account that costs
    # 15000 > cash 5000, so the size_frac order is clipped to cash: 5000 / 200 = 25 (NT-289 lead decision, clipped_to_cash)
    d = g.check(order(size_frac=1.0), ref_price=200.0)
    assert (d.allowed, d.qty, d.reason) == (True, 25.0, "clipped_to_cash")
    gb, pfb, _ = make(cash=15_000.0, ins=inst(max_qty=None))  # BACKTEST: no cash check, the full 75
    hold(pfb, "BUY", 50.0, 200.0)
    assert gb.check(order(size_frac=1.0), ref_price=200.0).qty == 75.0


def test_f4_stop_orders_valued_at_the_worst_price():
    g, _, _ = make(only(max_order_notional=105.0), ins=inst(is_spot=False))
    assert g.check(order("BUY", type="STOP_MARKET", trigger_price=105.0), ref_price=100.0).allowed
    d = g.check(order("BUY", type="STOP_MARKET", trigger_price=105.01), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "max_order_notional")
    # a buy stop below the market is valued at the market
    assert g.check(order("BUY", type="STOP_MARKET", trigger_price=90.0), ref_price=106.0).reason == \
        "max_order_notional"
    # a sell stop is valued at its trigger
    assert g.check(order("SELL", type="STOP_MARKET", trigger_price=105.0), ref_price=200.0).allowed
    assert g.check(order("SELL", type="STOP_MARKET", trigger_price=105.02), ref_price=1.0).reason == \
        "max_order_notional"
    # stop-limit buy: the highest of trigger, limit and market
    assert g.check(order("BUY", type="STOP_LIMIT", trigger_price=100.0, price=106.0),
                   ref_price=100.0).reason == "max_order_notional"


@pytest.mark.parametrize("trig", [float("nan"), float("inf"), 0.0, -5.0])
def test_f4_bad_trigger_is_invalid(trig):
    g, _, _ = make()
    assert g.check(order("BUY", type="STOP_MARKET", trigger_price=trig), ref_price=100.0).reason == "invalid_order"


@pytest.mark.parametrize("side,br", [
    ("BUY", dict(take_profit=99.0, stop_loss=90.0)), ("BUY", dict(take_profit=110.0, stop_loss=101.0)),
    ("BUY", dict(take_profit=float("nan"), stop_loss=90.0)), ("BUY", dict(take_profit=110.0, stop_loss=-1.0)),
    ("SELL", dict(take_profit=101.0, stop_loss=110.0)), ("SELL", dict(take_profit=90.0, stop_loss=99.0)),
    ("BUY", dict(take_profit=110.0, stop_loss=float("inf"), tp_is_offset=True)),
])
def test_f4_invalid_bracket_legs(side, br):
    g, _, _ = make(ins=inst(is_spot=False))
    assert g.check(order(side, bracket=br), ref_price=100.0).reason == "invalid_order"


def test_f4_bracket_levels_count_in_the_notional_and_valid_brackets_pass():
    g, _, _ = make(only(max_order_notional=120.0), ins=inst(is_spot=False))
    ok = dict(take_profit=110.0, stop_loss=95.0, tp_is_offset=False, max_hold=None)
    assert g.check(order("BUY", qty=1.0, bracket=ok), ref_price=100.0).allowed
    assert g.check(order("BUY", qty=1.0, bracket=dict(ok, take_profit=121.0)), ref_price=100.0).reason == \
        "max_order_notional"
    off = dict(take_profit=21.0, stop_loss=-5.0, tp_is_offset=True)  # levels 121 and 95 around entry 100
    assert g.check(order("BUY", qty=1.0, bracket=off), ref_price=100.0).reason == "max_order_notional"
    # a short's stop-loss above the entry is its worst case
    assert g.check(order("SELL", qty=1.0, bracket=dict(take_profit=90.0, stop_loss=121.0)),
                   ref_price=100.0).reason == "max_order_notional"


def test_a_ref_price_none_is_a_denial_not_a_crash():
    g, _, _ = make()
    assert g.check(order(), ref_price=None).reason == "invalid_order"


def test_b_missing_marks_for_another_held_instrument():
    g, pf, _ = make(ins=inst(is_spot=False))
    f = dataclasses.replace(market_fill(10.0, 1, RESEARCH_ZERO, 1.0), instrument_id="Y", trade_id="ty")
    pf.apply_fill(f, strategy_id="s")
    assert g.check(order(), ref_price=100.0).reason == "missing_marks"
    assert g.check(order(), ref_price=100.0, marks={X: 100.0, "Y": 10.0}).allowed


def test_c_day_start_equity_restores_the_opening_equity_after_a_restart():
    g, pf, _ = make(only(daily_loss_limit=500.0), ins=inst(is_spot=False))
    hold(pf, "BUY", 10.0, 100.0)
    marks = {X: 40.0}  # equity 9400 now
    assert g.check(order("BUY", qty=1.0), ref_price=40.0, marks=marks, day_start_equity=10_000.0).reason == \
        "daily_loss_limit"
    g2, pf2, _ = make(only(daily_loss_limit=500.0), ins=inst(is_spot=False))
    hold(pf2, "BUY", 10.0, 100.0)
    assert g2.check(order("BUY", qty=1.0), ref_price=40.0, marks=marks).allowed  # without it: baseline = 9400


def test_d_spot_buy_above_available_cash_is_denied():
    g, _, _ = make(cash=1_000.0, ins=inst(max_qty=None), kind="PAPER", costs=RESEARCH_ZERO)  # NT-292: explicit costs
    assert g.check(order(qty=10.0), ref_price=100.0).allowed
    d = g.check(order(qty=10.001), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "insufficient_cash")
    g2, _, _ = make(cash=1_000.0, ins=inst(is_spot=False, max_qty=None))
    assert g2.check(order(qty=11.0), ref_price=100.0).allowed  # margin: no cash check


# ---------------------------------------------------------------- NT-289 (DESK-1.2 completion)
def test_n1_size_frac_on_realised_equity_other_strategy_long():
    g, pf, _ = make(cash=10_000.0, ins=inst(max_qty=None))
    hold(pf, "BUY", 25.0, 200.0, strategy="other")  # cash 5000, realised equity 5000 + 25 x 200 = 10000
    assert pf.realised_equity() == 10_000.0
    d = g.check(order(size_frac=0.5), ref_price=200.0)  # 0.5 x 10000 / 200 = 25 (cash alone gave 12.5)
    assert (d.allowed, d.qty) == (True, 25.0)


def test_n1_size_frac_margin_short_is_not_levered():
    g, pf, _ = make(cash=10_000.0, ins=inst(is_spot=False, max_qty=None))
    hold(pf, "SELL", 25.0, 200.0, strategy="other")  # cash 15000, realised equity 15000 - 25 x 200 = 10000
    assert pf.cash == 15_000.0 and pf.realised_equity() == 10_000.0
    for side in ("BUY", "SELL"):  # cash alone gave 15000 / 200 = 75
        assert g.check(order(side, size_frac=1.0), ref_price=200.0, record=False).qty == 50.0


def test_n1_size_frac_flat_account_equals_cash():
    g, pf, _ = make(cash=1_000.0)
    assert pf.realised_equity() == pf.cash == 1_000.0
    assert g.check(order(size_frac=0.5), ref_price=250.0).qty == 2.0


def test_n2_flip_in_active_keeps_notional_and_rate_limits():
    ins = inst(is_spot=False, max_qty=None)
    g, pf, _ = make(only(max_order_notional=500.0), ins=ins)
    hold(pf, "BUY", 2.0, 100.0)
    d = g.check(order("SELL", qty=8.0), ref_price=100.0)  # closes 2, opens 6: notional 800 > 500
    assert (d.allowed, d.reason) == (False, "max_order_notional")
    assert g.check(order("SELL", qty=5.0), ref_price=100.0).allowed  # flip with notional 500 == limit passes
    g2, pf2, _ = make(only(max_orders_per_minute=1), ins=ins)
    hold(pf2, "BUY", 2.0, 100.0)
    assert g2.check(order("SELL", qty=3.0), ref_price=100.0).allowed and len(g2._sent) == 1  # the flip takes a slot
    assert g2.check(order("SELL", qty=3.0), ref_price=100.0).reason == "max_orders_per_minute"


def test_n3_committed_reduce_orders_limit_further_reductions():
    g, pf, _ = make(ins=inst(max_qty=None))
    hold(pf, "BUY", 10.0, 100.0)
    committed, allowed = 0.0, 0
    for _ in range(50):
        d = g.check(order("SELL", qty=10.0), ref_price=100.0, committed_reduce_qty=committed)
        if d.allowed:
            allowed += 1
            committed += d.qty
        else:
            assert d.reason == "reduce_exceeds_position"
    assert allowed == 1 and committed == 10.0


def test_n3_committed_clips_the_remainder_and_a_flip_stays_a_flip():
    g, pf, _ = make(ins=inst(is_spot=False, max_qty=None))
    hold(pf, "BUY", 10.0, 100.0)
    d = g.check(order("SELL", qty=8.0), ref_price=100.0, committed_reduce_qty=4.0)  # capacity 6
    assert (d.allowed, d.qty, d.reason) == (True, 6.0, "clipped_to_position")
    assert g.check(order("SELL", qty=6.0), ref_price=100.0, committed_reduce_qty=4.0).reason == ""
    assert g.check(order("SELL", qty=15.0), ref_price=100.0, committed_reduce_qty=4.0).qty == 15.0  # a flip
    d = g.check(order("SELL", qty=15.0, reduce_only=True), ref_price=100.0, committed_reduce_qty=4.0)
    assert (d.qty, d.reason) == (6.0, "clipped_to_position")
    assert g.check(order("SELL", qty=15.0, reduce_only=True), ref_price=100.0,
                   committed_reduce_qty=10.0).reason == "reduce_exceeds_position"
    assert g.check(order("SELL", qty=1.0), ref_price=100.0, committed_reduce_qty=-1.0).reason == "invalid_order"
    assert g.check(order("BUY", qty=1.0), ref_price=100.0, committed_reduce_qty=10.0).allowed  # opens: ignored


def test_n4_offset_bracket_legs_get_the_side_check():
    g, _, _ = make(ins=inst(is_spot=False))
    bad = [("BUY", dict(take_profit=-50.0, stop_loss=-5.0)), ("BUY", dict(take_profit=5.0, stop_loss=5.0)),
           ("SELL", dict(take_profit=50.0, stop_loss=5.0)), ("SELL", dict(take_profit=-5.0, stop_loss=-5.0)),
           ("BUY", dict(take_profit=0.0)), ("BUY", dict(stop_loss=-100.0))]  # last: level 0
    for side, br in bad:
        assert g.check(order(side, bracket=dict(br, tp_is_offset=True)), ref_price=100.0).reason == \
            "invalid_order", (side, br)
    assert g.check(order("BUY", bracket=dict(take_profit=5.0, stop_loss=-5.0, tp_is_offset=True)),
                   ref_price=100.0).allowed
    assert g.check(order("SELL", bracket=dict(take_profit=-5.0, stop_loss=5.0, tp_is_offset=True)),
                   ref_price=100.0).allowed


def test_n5_closing_a_position_with_a_sub_min_qty_remainder_closes_it_whole():
    g, pf, _ = make(ins=inst(min_notional=0))
    hold(pf, "BUY", 0.0015, 100.0)
    d = g.check(order("SELL", qty=0.0015, reduce_only=True), ref_price=100.0)  # floor 0.001 leaves 0.0005 < 0.001
    assert (d.allowed, d.qty) == (True, 0.0015)
    assert g.check(order("SELL", qty=0.0015), ref_price=100.0).qty == 0.0015  # an order asking for the whole position
    assert g.check(order("SELL", qty=0.001, reduce_only=True), ref_price=100.0).qty == 0.0015  # reduce_only closes
    assert g.check(order("SELL", qty=0.001), ref_price=100.0).qty == 0.001  # a plain partial sell is not touched
    g.reduce_only()
    assert g.check(order("SELL", qty=0.002), ref_price=100.0).qty == 0.0015  # a clipped flip closes whole too
    g2, _, _ = make(ins=inst(min_notional=0))  # an opening order still floors
    assert g2.check(order("BUY", qty=0.0015), ref_price=100.0).qty == 0.001


def test_n5_remainder_at_or_above_min_qty_is_left_alone():
    g, pf, _ = make(ins=inst(min_notional=0))
    hold(pf, "BUY", 0.0025, 100.0)
    assert g.check(order("SELL", qty=0.001, reduce_only=True), ref_price=100.0).qty == 0.001  # remainder 0.0015
    assert g.check(order("SELL", qty=0.0015, reduce_only=True), ref_price=100.0).qty == 0.001  # floors; remainder 0.0015


def test_n6_insufficient_cash_includes_the_fee():
    from neural_trade.desk.core.costs import CostProfile
    prof = CostProfile(name="p", fee_bps_taker=10.0, fee_bps_maker=2.0)
    clock = SimClock(T0)
    pf = Portfolio("acc", kind="PAPER", initial_cash=1_000.0, allow_short=False, clock=clock)
    g = RiskGuard(RiskLimits.unlimited(), inst(max_qty=None), pf, clock, costs=prof)
    # market BUY 10 at 100: 1000 + 0.1% x 1000 = 1001 > 1000
    assert g.check(order(qty=10.0), ref_price=100.0).reason == "insufficient_cash"
    assert g.check(order(qty=9.99), ref_price=100.0).allowed  # 999 + 0.999 = 999.999
    assert g.check(order("BUY", type="STOP_MARKET", trigger_price=100.0, qty=10.0), ref_price=100.0).reason == \
        "insufficient_cash"
    # LIMIT pays the maker fee, 2 bps: 1000 + 0.2 = 1000.2 > 1000
    assert g.check(order(qty=10.0, type="LIMIT", price=100.0), ref_price=100.0).reason == "insufficient_cash"
    # 9.995 x 100 = 999.5: limit 999.5 + 0.1999 passes, market 999.5 + 0.9995 = 1000.4995 does not
    assert g.check(order(qty=9.995, type="LIMIT", price=100.0), ref_price=100.0).allowed
    assert g.check(order(qty=9.995), ref_price=100.0).reason == "insufficient_cash"
    g0, _, _ = make(cash=1_000.0, ins=inst(max_qty=None))  # the zero-cost default is unchanged
    assert g0.check(order(qty=10.0), ref_price=100.0).allowed


# ---------------------------------------------------------------- NT-289 round 0: size_frac clipped to cash, dust close
def _fee10():
    from neural_trade.desk.core.costs import CostProfile
    return CostProfile(name="p", fee_bps_taker=10.0, fee_bps_maker=5.0)


def _guard(kind, ins=None, cash=10_000.0, costs=None):
    clock = SimClock(T0)
    pf = Portfolio("acc", kind=kind, initial_cash=cash, allow_short=False, clock=clock)
    return RiskGuard(RiskLimits.unlimited(), ins or inst(max_qty=None), pf, clock, costs=costs or _fee10()), pf


@pytest.mark.parametrize("kind", ["PAPER", "LIVE"])
def test_r0_size_frac_buy_is_clipped_to_cash_with_fee(kind):
    g, _ = _guard(kind)
    d = g.check(order(size_frac=1.0), ref_price=100.0)  # 10000 / (100 x 1.001) = 99.9000999 -> 99.900
    assert (d.allowed, d.qty, d.reason) == (True, 99.9, "clipped_to_cash")
    d = g.check(order(size_frac=0.5), ref_price=100.0)  # 50.0 x 100 x 1.001 = 5005 <= 10000: not clipped
    assert (d.qty, d.reason) == (50.0, "")


def test_r0_backtest_has_no_cash_check():
    g, _ = _guard("BACKTEST")
    d = g.check(order(size_frac=1.0), ref_price=100.0)
    assert (d.allowed, d.qty, d.reason) == (True, 100.0, "")
    assert g.check(order(qty=100.0), ref_price=100.0).allowed


def test_r0_explicit_qty_above_cash_is_denied_not_clipped():
    g, _ = _guard("PAPER")
    d = g.check(order(qty=100.0), ref_price=100.0)  # 10000 x 1.001 > 10000
    assert (d.allowed, d.reason) == (False, "insufficient_cash")
    assert g.check(order(qty=99.9), ref_price=100.0).allowed


def test_r0_limit_uses_the_maker_fee_for_the_clip():
    g, _ = _guard("PAPER")
    d = g.check(order(size_frac=1.0, type="LIMIT", price=100.0), ref_price=100.0)  # 10000 / 100.05 = 99.950
    assert (d.qty, d.reason) == (99.95, "clipped_to_cash")


def test_r0_clip_below_min_qty_is_denied():
    g, _ = _guard("PAPER", ins=inst(min_qty=1.0, max_qty=None), cash=100.0)
    # cash 100 at price 1000: 100 / 1001 = 0.0999 -> 0.099 < min_qty 1 -> denied; also a tiny cash rounds to zero
    d = g.check(order(size_frac=1.0), ref_price=1000.0)
    assert (d.allowed, d.reason) == (False, "insufficient_cash")


def test_r0_margin_instrument_has_no_cash_check():
    g, _ = _guard("PAPER", ins=inst(is_spot=False, max_qty=None))
    assert g.check(order(size_frac=1.0), ref_price=100.0).qty == 100.0


def test_r0_dust_position_is_closable():
    g, pf = _guard("PAPER", ins=inst(max_qty=None), costs=_fee10())
    hold(pf, "BUY", 0.0009, 100.0)  # below min_qty 0.001 and min_notional 10
    for kw in (dict(qty=0.0009), dict(qty=0.0009, reduce_only=True), dict(qty=0.5, reduce_only=True)):
        d = g.check(order("SELL", **kw), ref_price=100.0)
        assert (d.allowed, d.qty) == (True, 0.0009), kw
    # a sell of less than the dust position is still floored away
    assert g.check(order("SELL", qty=0.0005), ref_price=100.0).reason == "qty_rounds_to_zero"
    # a non-dust position is unchanged: a partial sell below min_qty is still refused
    g2, pf2 = _guard("PAPER", ins=inst(max_qty=None))
    hold(pf2, "BUY", 2.0, 100.0)
    assert g2.check(order("SELL", qty=0.0009, reduce_only=True), ref_price=100.0).reason == "qty_rounds_to_zero"


# ---------------------------------------------------------------- NT-292: RiskGuard hardening (QA of NT-289)
def _paper(costs, ins=None, cash=10_000.0, kind="PAPER", limits=None, allow_short=False):
    clock = SimClock(T0)
    pf = Portfolio("acc", kind=kind, initial_cash=cash, allow_short=allow_short, clock=clock)
    return RiskGuard(limits or RiskLimits.unlimited(), ins or inst(max_qty=None), pf, clock, costs=costs), pf


def _gross_limits(frac="auto"):
    return dataclasses.replace(RiskLimits.unlimited(), max_gross_exposure_frac=frac)


def test_292_p1_cash_clip_includes_slippage_and_the_account_never_overdrafts():
    from neural_trade.desk.core.costs import CostProfile
    prof = CostProfile(name="p", fee_bps_taker=10.0, slippage_bps=5.0)
    g, pf = _paper(prof)
    d = g.check(order(size_frac=1.0), ref_price=100.0)  # 10000 / (100 x 1.0015) = 99.85022 -> 99.850
    assert (d.allowed, d.qty, d.reason) == (True, 99.85, "clipped_to_cash")  # fee only would give 99.9
    f = dataclasses.replace(market_fill(100.0, 1, prof, d.qty), instrument_id=X, trade_id="t292")
    pf.apply_fill(f, strategy_id="s")
    assert 0.0 <= pf.cash < 0.2  # 10000 - 99.85 x 100.05 - 0.001 x 9985 = 0.1 (a fee-only clip gave -4.985)
    # an explicit quantity pays the same unit cost: 99.9 x 100 x 1.0015 = 10004.985 > 10000, 99.85 fits
    g2, _ = _paper(prof)
    assert g2.check(order(qty=99.9), ref_price=100.0).reason == "insufficient_cash"
    assert g2.check(order(qty=99.85), ref_price=100.0).allowed
    # a LIMIT order pays the maker fee and the same slip
    prof2 = CostProfile(name="p", fee_bps_taker=10.0, fee_bps_maker=2.0, half_spread_bps=1.0, slippage_bps=4.0)
    g3, _ = _paper(prof2)
    d = g3.check(order(size_frac=1.0, type="LIMIT", price=100.0), ref_price=100.0)  # 10000 / 100.07 = 99.930
    assert (d.qty, d.reason) == (99.93, "clipped_to_cash")


def test_292_p2_costs_are_implied_only_on_a_backtest_account():
    from neural_trade.desk.core.costs import BINANCE_SPOT_TAKER
    clock = SimClock(T0)
    for kind in ("PAPER", "LIVE"):
        with pytest.raises(ValueError, match="costs must be given"):
            RiskGuard(RiskLimits.unlimited(), inst(), Portfolio("acc", kind=kind, clock=clock), clock)
    bt = RiskGuard(RiskLimits.unlimited(), inst(), Portfolio("acc", clock=clock), clock)
    assert bt.costs is RESEARCH_ZERO and bt.cash_check_active is False
    explicit = RiskGuard(RiskLimits.unlimited(), inst(), Portfolio("acc", kind="PAPER", clock=clock), clock,
                         costs=RESEARCH_ZERO)  # an explicit zero profile on PAPER is a choice, and is allowed
    assert explicit.cash_check_active is True
    live = RiskGuard(RiskLimits.unlimited(), inst(), Portfolio("acc", kind="LIVE", clock=clock), clock,
                     costs=BINANCE_SPOT_TAKER)
    assert live.cash_check_active is True
    margin = RiskGuard(RiskLimits.unlimited(), inst(is_spot=False), Portfolio("acc", kind="PAPER", clock=clock),
                       clock, costs=RESEARCH_ZERO)
    assert margin.cash_check_active is False


def test_292_p3_gross_exposure_cap_denies_a_third_strategy_but_not_reductions():
    ins = inst(is_spot=False, max_qty=None)
    g, pf = _paper(RESEARCH_ZERO, ins=ins, limits=_gross_limits(), allow_short=True)
    hold(pf, "BUY", 100.0, 100.0, strategy="a")
    hold(pf, "BUY", 100.0, 100.0, strategy="b")  # cash -10000, realised equity 10000, gross 20000
    assert pf.realised_equity() == 10_000.0
    d = g.check(order("BUY", qty=1.0, strategy="c"), ref_price=100.0)
    assert (d.allowed, d.reason) == (False, "max_gross_exposure")
    assert g.check(order("SELL", qty=1.0, strategy="a"), ref_price=100.0).allowed  # reducing orders pass
    assert g.check(order("SELL", qty=100.0, strategy="a", reduce_only=True), ref_price=100.0).allowed
    # BACKTEST: unlimited by default, as before; an explicit cap applies there too
    gb, pfb = _paper(RESEARCH_ZERO, ins=ins, kind="BACKTEST", limits=_gross_limits(), allow_short=True)
    hold(pfb, "BUY", 100.0, 100.0, strategy="a")
    hold(pfb, "BUY", 100.0, 100.0, strategy="b")
    assert gb.gross_exposure_cap_frac is None
    assert gb.check(order("BUY", qty=1.0, strategy="c"), ref_price=100.0).allowed
    gc, pfc = _paper(RESEARCH_ZERO, ins=ins, kind="BACKTEST", limits=_gross_limits(1.0), allow_short=True)
    hold(pfc, "BUY", 100.0, 100.0)
    assert gc.check(order("BUY", qty=1.0, strategy="c"), ref_price=100.0).reason == "max_gross_exposure"


def test_292_p3_gross_exposure_cap_boundary_and_off_switch():
    ins = inst(is_spot=False, max_qty=None)
    g, _ = _paper(RESEARCH_ZERO, ins=ins, limits=_gross_limits(), allow_short=True)
    assert g.gross_exposure_cap_frac == 1.0
    assert g.check(order("BUY", qty=100.0), ref_price=100.0).allowed  # exactly 10000 = 1.0 x realised equity
    assert g.check(order("BUY", qty=100.001), ref_price=100.0).reason == "max_gross_exposure"
    assert g.check(order("SELL", qty=100.001), ref_price=100.0).reason == "max_gross_exposure"  # a short counts
    gh, _ = _paper(RESEARCH_ZERO, ins=ins, limits=_gross_limits(0.5), allow_short=True)
    assert gh.check(order("BUY", qty=50.0), ref_price=100.0).allowed
    assert gh.check(order("BUY", qty=50.001), ref_price=100.0).reason == "max_gross_exposure"
    goff, _ = _paper(RESEARCH_ZERO, ins=ins, limits=_gross_limits(None), allow_short=True)
    assert goff.gross_exposure_cap_frac is None and goff.check(order("BUY", qty=500.0), ref_price=100.0).allowed
    for bad in (0, -1.0, "x", True, float("nan")):
        with pytest.raises(ValueError, match="max_gross_exposure_frac"):
            _paper(RESEARCH_ZERO, limits=_gross_limits(bad))
    assert RiskLimits.unlimited().max_gross_exposure_frac is None and RiskLimits().max_gross_exposure_frac == "auto"


def test_292_p4_the_whole_position_closes_under_min_notional_a_partial_reduce_does_not():
    g, pf = _paper(RESEARCH_ZERO, kind="BACKTEST")
    hold(pf, "BUY", 0.05, 100.0)  # notional 5 < min_notional 10, quantity above min_qty 0.001
    for kw in (dict(qty=0.05, reduce_only=True), dict(qty=0.05), dict(qty=1.0, reduce_only=True)):
        d = g.check(order("SELL", **kw), ref_price=100.0)
        assert (d.allowed, d.qty) == (True, 0.05), kw
    d = g.check(order("SELL", qty=0.04, reduce_only=True), ref_price=100.0)  # a partial reduce keeps min_notional
    assert (d.allowed, d.reason) == (False, "min_notional")
    # with a resting reduce order claiming part of it, closing the rest is not closing the whole position
    d = g.check(order("SELL", qty=0.04, reduce_only=True), ref_price=100.0, committed_reduce_qty=0.01)
    assert (d.allowed, d.reason) == (False, "min_notional")
    assert g.check(order("BUY", qty=0.05), ref_price=100.0).reason == "min_notional"  # opening orders keep it


def test_292_p5_committed_capacity_is_clipped_to_a_clean_step_or_denied():
    g, pf = _paper(RESEARCH_ZERO, ins=inst(max_qty=None, min_notional=0), kind="BACKTEST")
    hold(pf, "BUY", 10.0, 100.0)
    # committed 9.9995: capacity 0.0005000000000006111 is below one step: denied, not submitted as float noise
    d = g.check(order("SELL", qty=1.0), ref_price=100.0, committed_reduce_qty=9.9995)
    assert (d.allowed, d.reason, d.qty) == (False, "reduce_exceeds_position", None)
    # committed 9.9985: capacity 0.0015000000000000568 -> the clean step value 0.001, no snap to the noisy capacity
    d = g.check(order("SELL", qty=1.0), ref_price=100.0, committed_reduce_qty=9.9985)
    assert (d.allowed, d.qty, d.reason) == (True, 0.001, "clipped_to_position")
    d = g.check(order("SELL", qty=1.0, reduce_only=True), ref_price=100.0, committed_reduce_qty=9.5)
    assert (d.allowed, d.qty, d.reason) == (True, 0.5, "clipped_to_position")
    # nothing committed: the sub-min_qty remainder of a whole close is still swept (NT-289 behaviour kept)
    g2, pf2 = _paper(RESEARCH_ZERO, ins=inst(max_qty=None, min_notional=0), kind="BACKTEST")
    hold(pf2, "BUY", 0.0015, 100.0)
    assert g2.check(order("SELL", qty=0.001, reduce_only=True), ref_price=100.0).qty == 0.0015


def test_292_p5_a_snapped_close_with_a_claim_is_not_a_whole_position_close():
    """Kills 'dust close for every snapped close': with 0.0015 committed of 0.0030 the capacity is 0.0015, a
    reduce_only SELL 0.001 floors to 0.001 and must stay 0.001 (notional 0.1 < min_notional 10: denied), not be
    widened to the capacity and waved through as a closing order."""
    g, pf = _paper(RESEARCH_ZERO, kind="BACKTEST")
    hold(pf, "BUY", 0.003, 100.0)
    d = g.check(order("SELL", qty=0.001, reduce_only=True), ref_price=100.0, committed_reduce_qty=0.0015)
    assert (d.allowed, d.reason) == (False, "min_notional")
    # committed 0.0021: capacity 0.0009 is below one step -> reduce_exceeds_position
    d = g.check(order("SELL", qty=0.001, reduce_only=True), ref_price=100.0, committed_reduce_qty=0.0021)
    assert (d.allowed, d.reason) == (False, "reduce_exceeds_position")


def test_292_p6_spot_size_frac_exit_is_clipped_to_the_holding():
    g, pf = _paper(RESEARCH_ZERO, ins=inst(max_qty=None, min_notional=0), kind="BACKTEST")
    hold(pf, "BUY", 2.0, 100.0)  # realised equity 10000: size_frac 1.0 would be 100 units
    for fr, want in ((1.0, 2.0), (0.01, 1.0)):  # 1.0 -> clipped to the 2.0 held; 0.01 x 10000 / 100 = 1.0 as asked
        d = g.check(order("SELL", size_frac=fr), ref_price=100.0)
        assert (d.allowed, d.qty) == (True, want), fr
    assert g.check(order("SELL", size_frac=1.0), ref_price=100.0).reason == "clipped_to_position"
    assert g.check(order("SELL", qty=100.0), ref_price=100.0).reason == "spot_sell_exceeds_balance"  # explicit qty
    d = g.check(order("SELL", size_frac=1.0, strategy="t"), ref_price=100.0)  # no long of its own
    assert (d.allowed, d.reason) == (False, "spot_sell_exceeds_balance")
    hold(pf, "BUY", 5.0, 100.0, strategy="other")
    assert g.check(order("SELL", size_frac=0.001, strategy="t"), ref_price=100.0).reason == "spot_short_no_own_long"
    gm, pfm = _paper(RESEARCH_ZERO, ins=inst(is_spot=False, max_qty=None, min_notional=0), kind="BACKTEST",
                     allow_short=True)
    hold(pfm, "BUY", 2.0, 100.0)  # margin: a size_frac SELL larger than the long is a flip, not clipped
    assert gm.check(order("SELL", size_frac=1.0), ref_price=100.0).qty == 100.0


def test_292_p7_a_bool_is_never_a_quantity_price_offset_or_claim():
    g, pf = _paper(RESEARCH_ZERO, kind="BACKTEST")
    assert g.check(order(qty=True), ref_price=100.0).reason == "invalid_order"
    assert g.check(order(size_frac=True), ref_price=100.0).reason == "invalid_order"
    assert g.check(order(), ref_price=True).reason == "invalid_order"
    assert g.check(order(type="LIMIT", price=True), ref_price=100.0).reason == "invalid_order"
    assert g.check(order(), ref_price=100.0, committed_reduce_qty=True).reason == "invalid_order"
    br = dict(take_profit=True, stop_loss=-5.0, tp_is_offset=True)  # True would be an offset of +1
    assert g.check(order(bracket=br), ref_price=100.0).reason == "invalid_order"
    assert g.check(order(bracket=dict(take_profit=1.0, stop_loss=-5.0, tp_is_offset=True)), ref_price=100.0).allowed


# ---------------------------------------------------------------- NT-293: follow-ups of the QA of NT-292
def _prof():
    from neural_trade.desk.core.costs import CostProfile
    return CostProfile(name="p", fee_bps_taker=10.0, slippage_bps=5.0)  # unit cost x 1.0015


def test_293_p1_spot_is_judged_by_cash_not_by_the_gross_cap_after_a_rally():
    # long 50 at average price 100; cash 10000 - 5000 = 5000; realised equity 10000; mark 200 -> gross 10000 + order
    g, pf = _paper(_prof(), limits=_gross_limits(), cash=10_000.0)
    hold(pf, "BUY", 50.0, 100.0)
    assert pf.cash == 5_000.0 and pf.realised_equity() == 10_000.0
    assert g.gross_exposure_cap_frac is None  # spot: the cash check binds
    marks = {X: 200.0}
    d = g.check(order("BUY", qty=10.0), ref_price=200.0, marks=marks)  # needs 10 x 200 x 1.0015 = 2003 of 5000
    assert (d.allowed, d.qty, d.reason) == (True, 10.0, "")  # was max_gross_exposure (gross 12000 > 10000)
    assert g.check(order("BUY", qty=24.9), ref_price=200.0, marks=marks).allowed  # 4987.5 of 5000
    assert g.check(order("BUY", qty=25.0), ref_price=200.0, marks=marks).reason == "insufficient_cash"  # 5007.5
    # size_frac is clipped by cash (not denied by the cap): cash 150 -> 150 / 200.3 = 0.74888 -> 0.748
    g2, pf2 = _paper(_prof(), limits=_gross_limits(), cash=5_150.0)
    hold(pf2, "BUY", 50.0, 100.0)
    assert pf2.cash == 150.0
    d = g2.check(order(size_frac=0.1), ref_price=200.0, marks=marks)  # asks 0.1 x 5150 / 200 = 2.575
    assert (d.allowed, d.qty, d.reason) == (True, 0.748, "clipped_to_cash")


def test_293_p1_margin_keeps_the_gross_cap_and_the_field_still_validates():
    ins = inst(is_spot=False, max_qty=None)
    g, pf = _paper(RESEARCH_ZERO, ins=ins, limits=_gross_limits(), allow_short=True)
    hold(pf, "BUY", 100.0, 100.0, strategy="a")
    hold(pf, "BUY", 100.0, 100.0, strategy="b")
    assert g.gross_exposure_cap_frac == 1.0
    assert g.check(order("BUY", qty=1.0, strategy="c"), ref_price=100.0).reason == "max_gross_exposure"
    with pytest.raises(ValueError, match="max_gross_exposure_frac"):  # validated for a spot instrument too
        _paper(RESEARCH_ZERO, limits=_gross_limits(-1.0))


def test_293_p2_cash_never_ends_below_minus_1e9_fuzz():
    import random
    from neural_trade.desk.core.costs import CostProfile
    # the exact case of the QA: cash 1e6, price 0.0123, step 1e-8 (qty about 8.1e7)
    ins = inst(price_increment=1e-8, size_increment=1e-8, min_qty=1e-8, min_notional=0.0, max_qty=None)
    prof = _prof()
    g, pf = _paper(prof, ins=ins, cash=1e6)
    d = g.check(order(size_frac=1.0), ref_price=0.0123)
    assert d.allowed and d.reason == "clipped_to_cash"
    pf.apply_fill(dataclasses.replace(market_fill(0.0123, 1, prof, d.qty), instrument_id=X, trade_id="t293"),
                  strategy_id="s")
    assert pf.cash >= -1e-9, pf.cash  # tolerance: the fill's own float arithmetic (a ULP of 1e6 is 1.2e-10)
    # cash / unit rounds UP here (found by search): 958680.08 / (2.5563 x 1.0015) floors to 374464.73953741, whose
    # cost exceeds the cash by one float step, so the clip steps one size increment down to ...39
    g, pf = _paper(prof, ins=ins, cash=958_680.08)
    d = g.check(order(size_frac=1.0), ref_price=2.5563)
    assert d.qty == 374464.73953739, d.qty
    rng = random.Random(293)
    worst, n_ok = 0.0, 0
    for i in range(5000):
        cash = 10 ** rng.uniform(2, 6)  # up to 1e6: the tolerance below is absolute for that scale
        price = 10 ** rng.uniform(-3, 5)
        step = 10 ** rng.uniform(-8, -2)
        p = CostProfile(name="f", fee_bps_taker=rng.uniform(0, 30), half_spread_bps=rng.uniform(0, 10),
                        slippage_bps=rng.uniform(0, 10))
        ins = inst(price_increment=1e-8, size_increment=step, min_qty=step, min_notional=0.0, max_qty=None)
        g, pf = _paper(p, ins=ins, cash=cash)
        unit = price * (1 + p.fee_rate(__import__("neural_trade.desk.core.model", fromlist=["Liquidity"]).Liquidity.TAKER)
                        + p.slip_rate)
        o = order(size_frac=1.0) if i % 2 else order(qty=round_down(cash / unit, step))
        d = g.check(o, ref_price=price)
        if not d.allowed:
            continue
        n_ok += 1
        pf.apply_fill(dataclasses.replace(market_fill(price, 1, p, d.qty), instrument_id=X, trade_id=f"f{i}"),
                      strategy_id="s")
        worst = min(worst, pf.cash)
    assert n_ok > 4000
    assert worst >= -1e-9, worst  # ULP noise of the fill's own arithmetic at cash <= 1e6 is about 1e-10


def test_293_p3_committed_open_notional_spot_cash_check():
    g, pf = _paper(RESEARCH_ZERO, cash=10_000.0)
    first = order("BUY", qty=100.0, type="LIMIT", price=100.0)
    assert g.check(first, ref_price=100.0, record=False).allowed  # notional 10000 = all the cash
    second = order("BUY", qty=100.0, type="LIMIT", price=100.0)
    d = g.check(second, ref_price=100.0, committed_open_notional=10_000.0)
    assert (d.allowed, d.reason) == (False, "insufficient_cash")
    assert g.check(order("BUY", qty=50.0, type="LIMIT", price=100.0), ref_price=100.0,
                   committed_open_notional=5_000.0).allowed  # 5000 + 5000 = cash
    assert g.check(order("BUY", qty=50.001, type="LIMIT", price=100.0), ref_price=100.0,
                   committed_open_notional=5_000.0).reason == "insufficient_cash"
    # size_frac: the clip sees cash - committed
    d = g.check(order(size_frac=1.0), ref_price=100.0, committed_open_notional=4_000.0)
    assert (d.allowed, d.qty, d.reason) == (True, 60.0, "clipped_to_cash")  # (10000 - 4000) / 100
    assert g.check(order(size_frac=1.0), ref_price=100.0, committed_open_notional=10_000.0).reason == "insufficient_cash"
    # a reducing order ignores it
    hold(pf, "BUY", 10.0, 100.0)
    assert g.check(order("SELL", qty=10.0), ref_price=100.0, committed_open_notional=1e9).allowed
    for bad in (-1.0, float("nan"), float("inf"), True, "1"):
        assert g.check(order("BUY", qty=1.0), ref_price=100.0, committed_open_notional=bad).reason == "invalid_order"


def test_293_p3_committed_open_notional_margin_gross_and_max_position():
    ins = inst(is_spot=False, max_qty=None)
    g, pf = _paper(RESEARCH_ZERO, ins=ins, limits=_gross_limits(), allow_short=True)
    assert g.check(order("BUY", qty=100.0, type="LIMIT", price=100.0), ref_price=100.0, record=False).allowed
    d = g.check(order("BUY", qty=100.0, type="LIMIT", price=100.0), ref_price=100.0, committed_open_notional=10_000.0)
    assert (d.allowed, d.reason) == (False, "max_gross_exposure")
    assert g.check(order("BUY", qty=50.0), ref_price=100.0, committed_open_notional=5_000.0).allowed
    hold(pf, "BUY", 10.0, 100.0)
    assert g.check(order("SELL", qty=10.0), ref_price=100.0, committed_open_notional=1e9).allowed  # reduction
    # max position is per position (NT-295): 20 x 100 = 2000 against 5000 counts committed_position_notional only
    lim = dataclasses.replace(RiskLimits.unlimited(), max_position_notional=5_000.0)
    gm = make(lim, ins=inst(max_qty=None))[0]
    assert gm.check(order("BUY", qty=20.0), ref_price=100.0, committed_position_notional=3_000.0).allowed
    assert gm.check(order("BUY", qty=20.0), ref_price=100.0,
                    committed_position_notional=3_000.01).reason == "max_position"
    assert gm.check(order("BUY", qty=20.0), ref_price=100.0, committed_open_notional=1e6).allowed  # the account total
    for bad in (-1.0, float("nan"), float("inf"), True, "1"):
        assert gm.check(order("BUY", qty=1.0), ref_price=100.0,
                        committed_position_notional=bad).reason == "invalid_order"
    # a reduction ignores it
    hold(gm.portfolio, "BUY", 10.0, 100.0)
    assert gm.check(order("SELL", qty=10.0), ref_price=100.0, committed_position_notional=1e9).allowed


def test_295_p4_size_frac_clip_steps_down_when_cash_over_unit_rounds_up_with_committed_orders():
    from neural_trade.desk.core.costs import CostProfile
    prof = CostProfile(name="p", fee_bps_taker=10.0, half_spread_bps=3.0, slippage_bps=2.0)  # unit = price x 1.0015
    # size_increment 0 (found by search): avail / unit = 2655.069337816401 and its cost exceeds avail by 5.8e-11, so
    # the old clip kept it and the final cash check then DENIED the order; now it steps down one float step
    ins0 = inst(max_qty=None, size_increment=0.0, price_increment=0.0, min_qty=0.0, min_notional=0.0)
    g, pf = _paper(prof, ins=ins0, cash=680_431.93)
    unit = 157.4165 * (1 + 0.001 + 0.0005)
    assert 2655.069337816401 * unit > 680_431.93 - 261_853.28
    d = g.check(order(size_frac=1.0), ref_price=157.4165, committed_open_notional=261_853.28)
    assert (d.allowed, d.reason) == (True, "clipped_to_cash")
    assert d.qty == math.nextafter(2655.069337816401, 0.0) and d.qty * unit + 261_853.28 <= pf.cash
    # a size increment of 1e-8 and a commitment: avail / unit floors to ...095 whose cost exceeds avail by 1.2e-10;
    # the step-down must compare with cash - committed (a cash-only comparison does not step and the order is denied)
    ins8 = inst(max_qty=None, size_increment=1e-8, price_increment=1e-8, min_qty=1e-8, min_notional=0.0)
    g, pf = _paper(prof, ins=ins8, cash=684_970.48)
    unit = 1.2056 * (1 + 0.001 + 0.0005)
    assert 545401.53108095 * unit > 684_970.48 - 26_448.09
    d = g.check(order(size_frac=1.0), ref_price=1.2056, committed_open_notional=26_448.09)
    assert (d.allowed, d.reason, d.qty) == (True, "clipped_to_cash", 545401.53108094)
    assert d.qty * unit + 26_448.09 <= pf.cash


def test_295_r2_fill_recheck_skips_the_rate_window_and_takes_no_slot():
    lim = dataclasses.replace(RiskLimits.unlimited(), max_orders_per_minute=1)
    g = make(lim, ins=inst(max_qty=None))[0]
    assert g.check(order(qty=1.0), ref_price=100.0).allowed  # takes the only slot
    assert g.check(order(qty=1.0), ref_price=100.0).reason == "max_orders_per_minute"
    assert g.check(order(qty=1.0), ref_price=100.0, fill_recheck=True).allowed
    assert len(g._sent) == 1


def test_295_r2e_prior_order_notional_counts_against_max_order_notional_only():
    lim = RiskLimits(max_order_notional=1000.0, max_position_notional=None, max_orders_per_minute=None,
                     daily_loss_limit=None)
    g, pf, _ = make(lim, ins=inst(max_qty=None), cash=1e5)
    assert g.check(order(qty=4.9), ref_price=104.0, record=False).allowed  # 509.6 alone
    d = g.check(order(qty=4.9), ref_price=104.0, record=False, prior_order_notional=515.0)  # 1024.6
    assert (d.allowed, d.reason) == (False, "max_order_notional")
    assert g.check(order(qty=4.9), ref_price=104.0, record=False, prior_order_notional=490.4).allowed  # 1000.0
    for bad in (-1.0, float("nan"), float("inf")):
        assert g.check(order(qty=1.0), ref_price=100.0, prior_order_notional=bad).reason == "invalid_order"
    hold(pf, "BUY", 5.0, 100.0)  # a reduction stays exempt
    assert g.check(order(side="SELL", qty=5.0), ref_price=104.0, prior_order_notional=1e9).allowed


# ---------------------------------------------------------------- NT-299 point 2: max position after the resting reductions
def test_299_max_position_counts_the_position_left_after_the_committed_reductions():
    """QA probe C1a: long 2, a resting SELL 2 commits the whole holding; a new SELL 5 ends at short 5 (500), not at the
    short 3 (300) of ``held + sign x qty``."""
    lim = dataclasses.replace(RiskLimits.unlimited(), max_position_notional=400.0)
    g, pf, _ = make(lim, ins=inst(is_spot=False, max_qty=None, min_notional=0.0))
    hold(pf, "BUY", 2.0, 100.0)
    d = g.check(order("SELL", qty=5.0), ref_price=100.0, committed_reduce_qty=2.0)
    assert (d.allowed, d.reason) == (False, "max_position")
    assert g.check(order("SELL", qty=5.0), ref_price=100.0).allowed  # nothing committed: short 3 = 300
    assert g.check(order("SELL", qty=6.0), ref_price=100.0, committed_reduce_qty=2.0).reason == "max_position"
    assert g.check(order("SELL", qty=4.0), ref_price=100.0, committed_reduce_qty=2.0).allowed  # short 4 = 400
    d = g.check(order("SELL", qty=5.0), ref_price=100.0, committed_reduce_qty=1.0)  # short 4 = 400 at worst
    assert d.allowed and d.qty == 5.0
    # an order on the position's own side is not helped by the reductions: the worse case (they never fill) counts
    assert g.check(order("BUY", qty=3.0), ref_price=100.0, committed_reduce_qty=2.0).reason == "max_position"
    assert g.check(order("BUY", qty=2.0), ref_price=100.0, committed_reduce_qty=2.0).allowed  # long 4 = 400

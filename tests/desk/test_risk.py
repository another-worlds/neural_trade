"""NT-223 (DESK-1.2): RiskGuard pinned by literal numbers and exact reason strings."""
from __future__ import annotations

import dataclasses
import itertools


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


def make(limits=None, ins=None, cash=10_000.0, **pkw):
    clock = SimClock(T0)
    ins = ins or inst()
    pf = Portfolio("acc", initial_cash=cash, allow_short=not ins.is_spot, clock=clock, **pkw)
    return RiskGuard(limits or RiskLimits.unlimited(), ins, pf, clock), pf, clock


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

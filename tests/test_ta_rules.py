"""NT-033 (2): the classic TA rules (strategy/ta_rules.py) on hand-made price series with known entries
and exits, their declared search ranges, and the look-ahead guard with parameters that really trade."""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.strategy import (BacktestConfig, Bars, SignalFrame, Strategies, assert_no_lookahead,
                                   build_strategy, run_backtest)
from neural_trade.strategy.strategies import strategy_search_space
from neural_trade.strategy.ta_rules import TA_RULES, price_only_frame, sma, wilder_rsi

CFG = BacktestConfig()      # costs 0 (D-044): the fills are the bars' own prices


def _run(name, close, **params):
    c = np.asarray(close, dtype=float)
    s = SignalFrame.build(price_only_frame(c), 1.0)
    return run_backtest(s, Bars.from_close(c), build_strategy(name, params), CFG)


def _trades(res):
    return [(t.side, t.entry_bar, t.exit_bar, t.exit_reason, t.entry_price, t.exit_price) for t in res.trades]


def test_the_three_rules_are_registered_with_declared_ranges():
    for name in TA_RULES:
        assert Strategies.has(name)
    assert strategy_search_space("ta_ma_cross") == {"fast": {"low": 2, "high": 30, "log": False},
                                                     "slow": {"low": 30, "high": 240, "log": False}}
    assert strategy_search_space("ta_rsi") == {"period": {"low": 2, "high": 60, "log": False},
                                               "lower": {"low": 10.0, "high": 40.0, "log": False},
                                               "upper": {"low": 60.0, "high": 90.0, "log": False}}
    assert strategy_search_space("ta_bollinger") == {"period": {"low": 5, "high": 120, "log": False},
                                                      "k": {"low": 1.0, "high": 3.0, "log": False}}
    assert strategy_search_space("buy_and_hold") == {}


@pytest.mark.parametrize("name", TA_RULES)
def test_every_textbook_default_lies_inside_its_declared_range(name):
    strat = build_strategy(name)
    for param, rule in strategy_search_space(name).items():
        assert rule["low"] <= getattr(strat, param) <= rule["high"], (name, param)


def test_ma_cross_range_never_produces_fast_above_slow():
    space = strategy_search_space("ta_ma_cross")
    assert space["fast"]["high"] <= space["slow"]["low"]


def test_ma_cross_enters_on_the_faster_average_and_reverses_on_the_cross():
    # SMA(2) vs SMA(3): up from bar 3 (12), back through at bar 7 (12 after 16, 14), down from bar 8
    close = [10, 10, 10, 12, 14, 16, 14, 12, 10, 8, 8, 8]
    res = _run("ta_ma_cross", close, fast=2, slow=3)
    assert _trades(res) == [("LONG", 4, 8, "CROSS", 12.0, 12.0),     # decided at close 3, filled at open 4
                            ("SHORT", 9, 11, "EOW", 10.0, 8.0)]      # reversal decided at close 8, still short at the end
    assert res.summary["n_trades"] == 2


def test_ma_cross_long_only_and_warmup():
    close = [10, 10, 10, 12, 14, 16, 14, 12, 10, 8, 8, 8]
    res = _run("ta_ma_cross", close, fast=2, slow=3, allow_short=False)
    assert [t[0] for t in _trades(res)] == ["LONG"]
    assert build_strategy("ta_ma_cross", {"fast": 2, "slow": 3}).warmup() == 3
    with pytest.raises(InvalidConfigurationError):
        build_strategy("ta_ma_cross", {"fast": 5, "slow": 3})


def test_wilder_rsi_matches_the_published_worked_example():
    # Wilder's RSI(14) on the StockCharts worked example (first value 70.53, then 66.32 ...). The published
    # figures come from changes rounded to 2 decimals, so they agree to about 0.07, not to the last digit.
    close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
             46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    r = wilder_rsi(close, 14)
    assert np.all(np.isnan(r[:14]))
    np.testing.assert_allclose(r[14:], [70.53, 66.32, 66.55, 69.41, 66.36, 57.97], atol=0.1)
    d = np.diff(close)                                        # the first value from the definition
    avg_gain, avg_loss = np.maximum(d[:14], 0).mean(), np.maximum(-d[:14], 0).mean()
    assert r[14] == pytest.approx(100 - 100 / (1 + avg_gain / avg_loss), rel=1e-12)


def test_wilder_rsi_edge_cases():
    assert wilder_rsi([1.0, 2.0, 3.0, 4.0], 2)[-1] == 100.0         # no losses
    assert wilder_rsi([5.0, 5.0, 5.0, 5.0], 2)[-1] == 50.0          # no change at all
    assert wilder_rsi([4.0, 3.0, 2.0, 1.0], 2)[-1] == 0.0           # no gains
    assert np.all(np.isnan(wilder_rsi([1.0, 2.0], 2)))              # fewer bars than the period


def test_rsi_rule_buys_oversold_shorts_overbought_and_exits_at_50():
    # RSI(2): 0 at bar 2 (long), 50 at bar 3 (out), 75 at bar 4 (short), 87.5 at 5, 43.75 at 6 (out)
    close = [10, 9, 8, 9, 10, 11, 10, 9]
    res = _run("ta_rsi", close, period=2)
    assert _trades(res) == [("LONG", 3, 4, "RSI_MID", 8.0, 9.0),
                            ("SHORT", 5, 7, "RSI_MID", 10.0, 10.0)]
    assert build_strategy("ta_rsi", {"period": 2}).warmup() == 2


def test_rsi_rule_rejects_inconsistent_levels():
    for bad in ({"lower": 55.0}, {"upper": 45.0}, {"lower": 0.0}, {"upper": 100.0}, {"period": 0}):
        with pytest.raises(InvalidConfigurationError):
            build_strategy("ta_rsi", bad)


def test_bollinger_rule_follows_the_breakout_and_leaves_at_the_middle_band():
    up = [10, 10, 10, 10, 13, 13, 10, 10, 10]         # bar 4: close 13 > 11 + 1.414 (period 3, k 1)
    res = _run("ta_bollinger", up, period=3, k=1.0)
    assert _trades(res) == [("LONG", 5, 7, "BB_MID", 13.0, 10.0)]
    down = [20 - x for x in up]
    assert _trades(_run("ta_bollinger", down, period=3, k=1.0)) == [("SHORT", 5, 7, "BB_MID", 7.0, 10.0)]
    # long only: the short breakout is skipped; the rebound at bar 6 (10 above 8 + 1.414) is a long breakout
    assert _trades(_run("ta_bollinger", down, period=3, k=1.0, allow_short=False)) == [
        ("LONG", 7, 8, "EOW", 10.0, 10.0)]
    assert _run("ta_bollinger", up, period=3, k=1.5).trades == []      # the band is wider than the move


def test_indicator_helpers_are_trailing():
    c = np.arange(1.0, 8.0)
    np.testing.assert_allclose(sma(c, 3)[2:], [2, 3, 4, 5, 6])
    changed = c.copy()
    changed[5:] += 100.0
    np.testing.assert_array_equal(sma(c, 3)[:5], sma(changed, 3)[:5])
    np.testing.assert_array_equal(wilder_rsi(c, 2)[:5], wilder_rsi(changed, 2)[:5])


def test_one_strategy_instance_keeps_each_frame_apart():
    rise = np.array([10, 10, 10, 12, 14, 16, 18, 20.0])
    fall = rise[::-1].copy()
    strat = build_strategy("ta_ma_cross", {"fast": 2, "slow": 3})
    for close, side in ((rise, "LONG"), (fall, "SHORT"), (rise, "LONG")):
        s = SignalFrame.build(price_only_frame(close), 1.0)
        order = strat.decide(s, 6)
        assert order is not None and order.side == side


def _walk(n=700, seed=5):
    rng = np.random.default_rng(seed)
    return 100_000 + np.cumsum(rng.normal(0, 40, n))


@pytest.mark.parametrize("name,params", [("ta_ma_cross", {"fast": 3, "slow": 12}),
                                         ("ta_rsi", {"period": 5, "lower": 35.0, "upper": 65.0}),
                                         ("ta_bollinger", {"period": 10, "k": 1.0})])
def test_rules_that_trade_pass_assert_no_lookahead(name, params):
    close = _walk()
    frame = price_only_frame(close)
    bars = Bars.from_close(close)
    base = run_backtest(SignalFrame.build(frame, 1.0), bars, build_strategy(name, params), CFG)
    assert base.summary["n_trades"] >= 3                       # the probes see real decisions and exits
    assert_no_lookahead(frame, bars, lambda: build_strategy(name, params), var_scale=1.0)

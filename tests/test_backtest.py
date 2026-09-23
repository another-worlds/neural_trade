"""Backtest engine and strategies (plan C5): hand-computed P&L after costs, stop/target on the bar's
high/low, next-open fills, look-ahead self-test for every registered strategy, causal features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.data.windowing import make_sequences_with_extended_trends, sequence_anchor_bars
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy import (BacktestConfig, Bars, Order, SignalFrame, Strategies, Strategy,
                                   assert_no_lookahead, backtest, build_strategy, run_backtest, var_scale_from)

COST = 13e-4  # 10 fee + 1 half-spread + 2 slippage bps per side


def _frame(n=400, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    close = 100_000 + np.cumsum(rng.normal(0, 40, n + 20))
    lc = close[:n]
    y = np.stack([close[h:n + h] - lc for h in (10, 15, 20)], 1)
    noise = rng.normal(0, 60, (n, 3))
    delta = (y * 0.6 + noise) if informative else noise
    prob = 1 / (1 + np.exp(-delta / 40))
    var = rng.uniform(0.2, 1.5, (n, 3))
    f = PredictionFrame(y, lc, {h: delta[:, i] for i, h in enumerate(HORIZONS)},
                        {h: prob[:, i] for i, h in enumerate(HORIZONS)},
                        {h: var[:, i] for i, h in enumerate(HORIZONS)}, 250.0)
    return f, Bars.from_close(lc)


@dataclass
class Scripted(Strategy):
    """Places given orders at given bars (test helper)."""

    name: ClassVar[str] = "scripted"
    orders: Optional[Dict[int, Order]] = None

    def decide(self, s, t):
        return (self.orders or {}).get(t)


def _signals(n):
    f, _ = _frame(n)
    return SignalFrame.build(f, 1.0)


def test_long_pnl_after_costs_matches_hand_computation():
    o = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    bars = Bars(o, o + 0.5, o - 0.5, o + 0.25)
    strat = Scripted(orders={0: Order("LONG", 1.0, max_hold=2)})
    res = run_backtest(_signals(6), bars, strat, BacktestConfig())
    (tr,) = res.trades
    assert (tr.entry_bar, tr.exit_bar, tr.exit_reason) == (1, 3, "TIME")  # decided at close 0, filled at open 1
    qty = 10_000 / 101.0
    gross = qty * (103.0 - 101.0)
    costs = 10_000 * COST + qty * 103.0 * COST
    assert tr.gross_pnl == pytest.approx(gross, rel=1e-9)
    assert tr.costs == pytest.approx(costs, rel=1e-9)
    assert res.equity[-1] == pytest.approx(10_000 + gross - costs, rel=1e-9)
    assert tr.entry_price == pytest.approx(101.0 * (1 + 3e-4), rel=1e-12)  # adverse spread + slippage
    assert res.summary["fees_paid"] == pytest.approx(10_000 * 1e-3 + qty * 103.0 * 1e-3, rel=1e-9)
    assert res.equity_gross[-1] == pytest.approx(10_000 + gross, rel=1e-9)


def test_short_pnl_after_costs():
    o = np.array([100.0, 100.0, 98.0, 97.0, 96.0])
    bars = Bars(o, o + 0.1, o - 0.1, o)
    res = run_backtest(_signals(5), bars, Scripted(orders={0: Order("SHORT", 0.5, max_hold=1)}), BacktestConfig())
    (tr,) = res.trades
    qty = 5_000 / 100.0
    assert (tr.entry_bar, tr.exit_bar) == (1, 2)
    assert tr.net_pnl == pytest.approx(qty * 2.0 - 5_000 * COST - qty * 98.0 * COST, rel=1e-9)


@pytest.mark.parametrize("tiebreak, reason, price", [("sl_first", "SL", 99.0), ("tp_first", "TP", 102.0)])
def test_stop_and_target_on_high_low_with_tiebreak(tiebreak, reason, price):
    o = np.array([100.0, 100.0, 100.5, 100.5, 100.5])
    h = np.array([100.0, 100.2, 102.5, 100.6, 100.6])   # bar 2 touches both levels
    lo = np.array([100.0, 99.8, 98.5, 100.4, 100.4])
    bars = Bars(o, h, lo, o)
    order = Order("LONG", 1.0, tp=2.0, sl=-1.0, tp_is_offset=True, max_hold=10)
    res = run_backtest(_signals(5), bars, Scripted(orders={0: order}), BacktestConfig(same_bar_tiebreak=tiebreak))
    (tr,) = res.trades
    assert tr.exit_reason == reason and tr.exit_bar == 2
    assert tr.exit_price == pytest.approx(price * (1 - 3e-4), rel=1e-12)


def test_gap_through_stop_fills_at_the_open():
    o = np.array([100.0, 100.0, 95.0, 95.0])
    bars = Bars(o, o + 0.1, o - 0.1, o)
    res = run_backtest(_signals(4), bars, Scripted(orders={0: Order("LONG", 1.0, sl=99.0, max_hold=10)}))
    (tr,) = res.trades
    assert tr.exit_reason == "SL" and tr.exit_price == pytest.approx(95.0 * (1 - 3e-4))


def test_close_only_tp_sl_ignores_wicks():
    o = np.array([100.0, 100.0, 100.0, 100.0])
    bars = Bars(o, o + 5, o - 5, o)
    res = run_backtest(_signals(4), bars, Scripted(orders={0: Order("LONG", 1.0, tp=103, sl=97, max_hold=10)}),
                       BacktestConfig(tp_sl_on="close"))
    assert res.trades[0].exit_reason == "EOW"


def test_end_of_window_marks_open_position_and_buy_and_hold():
    f, bars = _frame(200)
    res = run_backtest(SignalFrame.build(f, 1.0), bars, build_strategy("buy_and_hold"))
    (tr,) = res.trades
    assert tr.exit_reason == "EOW" and tr.exit_bar == 199 and tr.entry_bar == 1
    qty = 10_000 / bars.open[1]
    expect = qty * (bars.close[-1] - bars.open[1]) - 10_000 * COST - qty * bars.close[-1] * COST
    assert res.equity[-1] - 10_000 == pytest.approx(expect, rel=1e-9)
    assert res.summary["exposure"] == pytest.approx(199 / 200)


def test_always_flat_has_flat_equity():
    f, bars = _frame(100)
    res = run_backtest(SignalFrame.build(f, 1.0), bars, build_strategy("always_flat"))
    assert res.summary["n_trades"] == 0 and np.all(res.equity == 10_000) and res.summary["sharpe_net"] == 0


def test_no_order_on_the_last_bar():
    res = run_backtest(_signals(3), Bars.from_close([1.0, 2.0, 3.0]), Scripted(orders={2: Order("LONG")}))
    assert res.trades == [] and res.decisions == []


@pytest.mark.parametrize("name", sorted(Strategies.list_names()))
def test_no_lookahead_all_strategies(name):
    f, bars = _frame(500, seed=3)
    vs = var_scale_from(f)
    assert_no_lookahead(f, bars, lambda: build_strategy(name), var_scale=vs, probes=(120, 250, 380))


def test_lookahead_probe_catches_a_peeking_strategy():
    @dataclass
    class Peek(Strategy):
        name: ClassVar[str] = "peek"

        def decide(self, s, t):
            return Order("LONG", max_hold=2) if t + 5 < len(s) and s.close[t + 5] > s.close[t] else None

    f, bars = _frame(300, seed=4)
    with pytest.raises(AssertionError, match="look-ahead"):
        assert_no_lookahead(f, bars, Peek, var_scale=1.0, probes=(100,))


def test_trailing_features_are_causal():
    f, _ = _frame(300, seed=5)
    full = SignalFrame.build(f, 0.8)
    t = 150
    cut = PredictionFrame(f.y[:t + 1], f.last_close[:t + 1], {h: v[:t + 1] for h, v in f.delta.items()},
                          {h: v[:t + 1] for h, v in f.direction_prob.items()},
                          {h: v[:t + 1] for h, v in f.variance_scaled.items()}, f.pred_scale)
    part = SignalFrame.build(cut, 0.8)
    for name in ("weighted_direction", "strength", "agreement", "var_spike", "volatility", "consensus"):
        np.testing.assert_array_equal(getattr(full, name)[:t + 1], getattr(part, name))


def test_var_spike_uses_trailing_not_centred_mean():
    n = 100
    var = np.full((n, 3), 0.5)
    var[60, 1] = 5.0  # a spike at 60 must not flag bars before it
    f = PredictionFrame(np.zeros((n, 3)), np.full(n, 1e5), {h: np.zeros(n) for h in HORIZONS},
                        {h: np.full(n, 0.5) for h in HORIZONS}, {h: var[:, i] for i, h in enumerate(HORIZONS)}, 250.0)
    s = SignalFrame.build(f, 0.5)
    assert s.var_spike[60] and not s.var_spike[:60].any()


def test_enhanced_stop_is_sigma_dollars_not_half_the_price():
    f, bars = _frame(400, seed=6)
    s = SignalFrame.build(f, var_scale_from(f))
    strat = build_strategy("enhanced_multi_horizon", {"min_agreement": 0.0, "min_signal_strength": 0.0,
                                                      "min_confidence": 0.0})
    orders = [(t, strat.decide(s, t)) for t in range(len(s))]
    orders = [(t, o) for t, o in orders if o is not None]
    assert orders
    for t, o in orders:
        sign = 1 if o.side == "LONG" else -1
        assert o.tp_is_offset and o.sl == pytest.approx(-sign * 1.5 * s.volatility[t])
        assert abs(o.sl) < 0.05 * s.close[t]  # the notebook's stop sat at ~-50% of the price


def test_multi_horizon_strategies_trade_only_with_the_consensus():
    f, bars = _frame(600, seed=7)
    s = SignalFrame.build(f, var_scale_from(f))
    for name in ("enhanced_multi_horizon", "liberal"):
        strat = build_strategy(name, {"min_agreement": 0.0})
        for t in range(len(s)):
            o = strat.decide(s, t)
            if o is not None:
                assert s.consensus[t] == (1 if o.side == "LONG" else -1), (name, t)


def test_threshold_agreement_counts_the_trade_side():
    f, _ = _frame(300, seed=8)
    s = SignalFrame.build(f, var_scale_from(f))
    strat = build_strategy("threshold_spike", {"min_confidence": 0.0, "warmup_bars": 0})
    for t in range(len(s)):
        o = strat.decide(s, t)
        if o is not None:
            assert s.agreeing_horizons(1 if o.side == "LONG" else -1)[t] >= 2


def test_informative_signals_trade_and_rank_against_random():
    f, bars = _frame(800, seed=9)
    s = SignalFrame.build(f, var_scale_from(f))
    res = backtest(s, bars, build_strategy("enhanced_multi_horizon"), BacktestConfig(random_seeds=20))
    assert res.summary["n_trades"] > 0
    rnd = res.baselines["random_same_freq"]
    assert rnd["n_seeds"] == 20 and 0 <= rnd["percentile_total_return"] <= 100
    assert set(res.baselines) == {"buy_and_hold", "always_flat", "random_same_freq"}
    assert res.baselines["always_flat"]["n_trades"] == 0
    assert len(res.trades_frame()) == res.summary["n_trades"]


def test_params_reject_unknown_knobs_and_load_yaml(tmp_path):
    with pytest.raises(InvalidConfigurationError, match="min_agreement"):
        build_strategy("enhanced_multi_horizon", {"min_agrement": 0.3})
    p = tmp_path / "s.yaml"
    p.write_text("strategy: liberal\nparams:\n  min_agreement: 0.3\nbacktest:\n  fee_bps: 5\n", encoding="utf-8")
    from neural_trade.strategy import from_file

    strat, cfg = from_file(p)
    assert strat.min_agreement == 0.3 and cfg.fee_bps == 5


def test_backtest_config_rejects_close_fill():
    with pytest.raises(ValueError, match="look-ahead"):
        BacktestConfig(fill="close")


def test_anchor_bars_match_windowing_and_bars_from_frame():
    cfg = Config().override(LOOKBACK=20, EXTENDED_TREND_PERIODS=[30, 30, 30], HORIZON_STEPS=[2, 3, 5], MAX_SEQUENCE_COUNT=50)
    rng = np.random.default_rng(0)
    close = (1000 + np.cumsum(rng.normal(0, 1, 200))).astype("float32")
    X, y, lc, _ = make_sequences_with_extended_trends(cfg, close, cfg.LOOKBACK)
    n_total = X.shape[0]
    anchors = sequence_anchor_bars(cfg, len(close), n_total)[-50:]
    np.testing.assert_array_equal(close[anchors], lc[-50:])
    df = pd.DataFrame({"Open": close - 0.1, "High": close + 1, "Low": close - 1, "Close": close})
    bars = Bars.from_frame(df, anchors)
    assert np.allclose(bars.close, lc[-50:]) and np.allclose(bars.high - bars.close, 1.0, atol=1e-4)
    with pytest.raises(ValueError, match="consecutive"):
        Bars.from_frame(df, anchors[::2])

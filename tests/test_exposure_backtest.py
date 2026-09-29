"""The exposure mode of the backtest engine and its timing null, the SignalFrame read-outs for the
variance strategies, and the notional / break-even fields of every summary (NT-077;
docs/research/2026-09-29-strategy-architectures/README.md sections 2.0 and 5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy import (EWMA_WARMUP, BacktestConfig, Bars, ExposureStrategy, Order, SignalFrame,
                                   Strategy, assert_no_lookahead, backtest, build_strategy, circular_shift_null,
                                   random_same_frequency, run_backtest, run_exposure_backtest, summarize,
                                   var_scale_from)
from neural_trade.strategy.backtest import _ReplayFills, fill_events

COST = 13e-4  # 10 fee + 1 half-spread + 2 slippage bps per side


def _frame(n=700, seed=0, steps=(10, 15, 20)):
    rng = np.random.default_rng(seed)
    vol = 4e-4 * (1 + 0.8 * np.sin(np.arange(n + 30) / 90.0))
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, vol)))
    lc = close[:n]
    y = np.stack([close[h:n + h] - lc for h in steps], 1)
    delta = y * 0.4 + rng.normal(0, 60, (n, 3))
    prob = 1 / (1 + np.exp(-delta / 40))
    var = rng.uniform(0.2, 1.5, (n, 3))
    f = PredictionFrame(y, lc, {h: delta[:, i] for i, h in enumerate(HORIZONS)},
                        {h: prob[:, i] for i, h in enumerate(HORIZONS)},
                        {h: var[:, i] for i, h in enumerate(HORIZONS)}, 250.0, horizon_steps=tuple(steps))
    return f, Bars.from_close(lc)


def _signals(n):
    f, _ = _frame(max(n, 3))
    return SignalFrame.build(PredictionFrame(f.y[:n], f.last_close[:n], {h: v[:n] for h, v in f.delta.items()},
                                             {h: v[:n] for h, v in f.direction_prob.items()},
                                             {h: v[:n] for h, v in f.variance_scaled.items()}, f.pred_scale), 1.0)


@dataclass
class Scripted(ExposureStrategy):
    """Target exposures given per decision bar (test helper); ``default`` elsewhere (None: no decision)."""

    name: ClassVar[str] = "scripted_exposure"
    targets_by_bar: Optional[Dict[int, float]] = None
    default: Optional[float] = None
    decide_every: int = 1
    band: float = 0.0
    calls: int = 0

    def target(self, s, t, current):
        self.calls += 1
        v = (self.targets_by_bar or {}).get(t, self.default)
        return float("nan") if v is None else v


# ------------------------------------------------------------------ SignalFrame read-outs
def test_sigma_ret_and_mu_gauss_follow_their_definitions():
    f, _ = _frame(400, seed=1)
    f.direction_prob["h0"][:3] = [0.0, 1.0, 0.5]              # clipped to [1e-6, 1 - 1e-6], not +-inf
    s = SignalFrame.build(f, var_scale_from(f), calibrated=False)
    np.testing.assert_allclose(s.sigma_ret, s.sigma / s.close[:, None], rtol=1e-12)
    p = np.clip(s.p, 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(s.mu_gauss, s.sigma * ndtri(p), rtol=1e-12)
    assert np.all(np.isfinite(s.mu_gauss)) and s.mu_gauss[2, 0] == 0.0
    assert s.mu_gauss[0, 0] < 0 < s.mu_gauss[1, 0]


def test_sigma_ewma_is_the_halflife_60_ewma_of_squared_log_returns_scaled_per_horizon():
    steps = (2, 3, 5)
    f, _ = _frame(500, seed=2, steps=steps)
    s = SignalFrame.build(f, 1.0)
    c = f.last_close
    r2 = np.log(c[1:] / c[:-1]) ** 2
    lam = 0.5 ** (1 / 60)
    expect = np.full((len(c), 3), np.nan)
    for t in range(1, len(c)):                                 # normalised weights over r_1..r_t
        w = lam ** np.arange(t - 1, -1, -1)
        v = float(np.sum(w * r2[:t]) / np.sum(w))
        expect[t] = np.sqrt(v) * np.sqrt(steps) * c[t]
    expect[:EWMA_WARMUP] = np.nan
    assert EWMA_WARMUP == 240 and s.horizon_bars == steps
    assert np.isnan(s.sigma_ewma[:240]).all() and np.isfinite(s.sigma_ewma[240:]).all()
    np.testing.assert_allclose(s.sigma_ewma[240:], expect[240:], rtol=1e-9)


def test_sigma_ewma_is_causal():
    f, _ = _frame(600, seed=3)
    s = SignalFrame.build(f, 1.0)
    t = 400
    lc = f.last_close.copy()
    lc[t + 1:] *= np.exp(np.random.default_rng(0).normal(0, 0.01, len(lc) - t - 1))
    g = PredictionFrame(f.y, lc, f.delta, f.direction_prob, f.variance_scaled, f.pred_scale)
    s2 = SignalFrame.build(g, 1.0)
    np.testing.assert_array_equal(s.sigma_ewma[: t + 1], s2.sigma_ewma[: t + 1])
    assert not np.allclose(s.sigma_ewma[t + 2:], s2.sigma_ewma[t + 2:])


def test_sigma_for_selects_the_source():
    s = SignalFrame.build(_frame(300, seed=4)[0], 1.0)
    np.testing.assert_array_equal(s.sigma_for("model", 2), s.sigma[:, 2])
    np.testing.assert_array_equal(s.sigma_for("ewma", 1), s.sigma_ewma[:, 1])
    with pytest.raises(ValueError, match="sigma source"):
        s.sigma_for("garch", 0)


# ------------------------------------------------------------------ the exposure engine
def test_hand_computed_rebalances_with_drift_costs_and_close_out():
    o = np.array([100.0, 100.0, 110.0, 99.0, 105.0])
    c = np.array([100.0, 104.0, 100.0, 105.0, 110.0])
    bars = Bars(o, np.maximum(o, c) + 1, np.minimum(o, c) - 1, c)
    strat = Scripted(targets_by_bar={0: 0.5, 2: 1.0, 3: -0.5}, band=0.1)
    res = run_backtest(_signals(5), bars, strat, BacktestConfig())
    assert res.mode == "exposure" and res.trades == []
    # bar 1 open: 0 -> 0.5 of 10,000
    q1 = 0.5 * 10_000 / 100.0
    cost1 = 5_000 * COST
    cash = 10_000 - q1 * 100.0 - cost1
    assert res.equity[2] == pytest.approx(cash + q1 * 104.0, rel=1e-12)       # marked at bar 1's close
    assert res.position[1] == pytest.approx(q1 * 104.0 / (cash + q1 * 104.0), rel=1e-12)   # drifted
    # bar 3 open: the held (drifted) exposure -> 1.0; the notional is |1 - current| x equity at the open
    eq3 = cash + q1 * 99.0
    q3 = 1.0 * eq3 / 99.0
    n3 = abs(q3 - q1) * 99.0
    assert n3 == pytest.approx((1.0 - q1 * 99.0 / eq3) * eq3, rel=1e-12)
    cash = cash - (q3 - q1) * 99.0 - n3 * COST
    # bar 4 open: -> -0.5; then the close-out at bar 4's close
    eq4 = cash + q3 * 105.0
    q4 = -0.5 * eq4 / 105.0
    n4 = abs(q4 - q3) * 105.0
    cash = cash - (q4 - q3) * 105.0 - n4 * COST
    n5 = abs(q4) * 110.0
    cash = cash + q4 * 110.0 - n5 * COST
    assert res.equity[-1] == pytest.approx(cash, rel=1e-12)
    assert [(d["bar"], d["fill_bar"], d["reason"], d["side"]) for d in res.decisions] == [
        (0, 1, "target", "LONG"), (2, 3, "target", "LONG"), (3, 4, "target", "SHORT"), (4, 4, "EOW", "LONG")]
    assert [d["notional"] for d in res.decisions] == pytest.approx([5_000.0, n3, n4, n5], rel=1e-12)
    assert res.decisions[1]["from"] == pytest.approx(q1 * 99.0 / eq3, rel=1e-12)
    assert res.decisions[1]["fill"] == pytest.approx(99.0 * (1 + 3e-4), rel=1e-12)
    assert res.decisions[2]["fill"] == pytest.approx(105.0 * (1 - 3e-4), rel=1e-12)
    np.testing.assert_allclose(res.target_path, [0.0, 0.5, 0.5, 1.0, -0.5])
    sm = res.summary
    assert sm["n_rebalances"] == sm["n_trades"] == 3
    assert sm["traded_notional"] == pytest.approx(5_000 + n3 + n4 + n5, rel=1e-12)
    assert sm["costs_paid"] == pytest.approx(COST * sm["traded_notional"], rel=1e-12)
    assert sm["fees_paid"] == pytest.approx(1e-3 * sm["traded_notional"], rel=1e-12)
    gross = res.equity_gross[-1] - 10_000
    assert sm["gross_pnl"] == pytest.approx(gross, rel=1e-12)
    assert sm["breakeven_cost_bps"] == pytest.approx(gross / sm["traded_notional"] * 1e4 * 2, rel=1e-12)
    assert sm["avg_hold_bars"] == pytest.approx(np.mean([2, 1, 1]))


def test_constant_full_target_reproduces_buy_and_hold():
    f, bars = _frame(500, seed=5)
    s = SignalFrame.build(f, 1.0)
    bh = run_backtest(s, bars, build_strategy("buy_and_hold"))
    ex = run_backtest(s, bars, Scripted(default=1.0, band=0.5))     # enters at bar 1, never rebalances
    assert ex.summary["n_rebalances"] == 1 and len(ex.decisions) == 2          # the entry and the close-out
    np.testing.assert_allclose(ex.equity, bh.equity, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ex.equity_gross, bh.equity_gross, rtol=0, atol=1e-9)
    assert ex.summary["costs_paid"] == pytest.approx(bh.summary["costs_paid"], rel=1e-12)
    assert ex.summary["traded_notional"] == pytest.approx(bh.summary["traded_notional"], rel=1e-12)


def test_a_band_wider_than_any_change_never_trades():
    f, bars = _frame(300, seed=6)
    res = run_backtest(SignalFrame.build(f, 1.0), bars, Scripted(default=0.5, band=0.6))
    assert res.decisions == [] and len(res.targets) == 299 and all(d["queued"] is None for d in res.targets)
    assert np.all(res.equity == 10_000) and np.all(res.position == 0)
    assert res.summary["n_trades"] == 0 and res.summary["traded_notional"] == 0
    assert np.isnan(res.summary["breakeven_cost_bps"])


def test_costs_are_the_cost_rate_times_the_traded_notional():
    f, bars = _frame(600, seed=7)
    path = {t: float(np.sin(t / 17.0)) for t in range(600)}
    cfg = BacktestConfig(fee_bps=7.0, half_spread_bps=1.5, slippage_bps=2.5)
    res = run_backtest(SignalFrame.build(f, 1.0), bars, Scripted(targets_by_bar=path, decide_every=5, band=0.2), cfg)
    sm = res.summary
    assert sm["n_rebalances"] > 10
    rate = (7.0 + 1.5 + 2.5) / 1e4
    assert sm["costs_paid"] == pytest.approx(rate * sm["traded_notional"], rel=1e-12)
    assert sm["costs_paid"] == pytest.approx(sum(d["costs"] for d in res.decisions), rel=1e-12)
    assert res.equity_gross[-1] - res.equity[-1] == pytest.approx(sm["costs_paid"], rel=1e-9)
    assert sm["cost_drag"] == pytest.approx(sm["costs_paid"] / 10_000, rel=1e-12)
    assert sm["turnover"] == pytest.approx(sm["traded_notional"] / 10_000, rel=1e-12)
    assert sm["mean_abs_exposure"] == pytest.approx(np.mean(np.abs(res.position)), rel=1e-12)
    assert set(summarize(res.equity, [0.0], [0.0], [], 0.0, 0.0)) <= set(sm)


def test_cadence_warmup_band_edge_and_clipping():
    f, bars = _frame(200, seed=8)

    @dataclass
    class Late(Scripted):
        def warmup(self):
            return 30

    res = run_backtest(SignalFrame.build(f, 1.0), bars,
                       Late(default=3.0, decide_every=10, band=0.3, trade_to_band_edge=True))
    assert [d["bar"] for d in res.targets][:3] == [30, 40, 50] and res.targets[-1]["bar"] == 190
    first = res.targets[0]
    assert first["target"] == 1.0 and first["queued"] == pytest.approx(0.7)       # clipped, then the band's edge
    assert (res.decisions[0]["bar"], res.decisions[0]["fill_bar"], res.decisions[0]["reason"]) == (30, 31, "band_edge")
    assert all(abs(d["to"]) <= 1.0 for d in res.decisions)


def test_no_decision_on_the_last_bar_and_open_book_without_close_out():
    bars = Bars.from_close([100.0, 101.0, 102.0])
    res = run_backtest(_signals(3), bars, Scripted(targets_by_bar={2: 1.0}))
    assert res.decisions == [] and [d["bar"] for d in res.targets] == [0, 1]
    held = run_backtest(_signals(3), bars, Scripted(targets_by_bar={0: 1.0}), BacktestConfig(mark_to_market_at_end=False))
    assert [d["reason"] for d in held.decisions] == ["target"] and held.position[-1] > 0


def test_non_finite_target_is_no_decision():
    f, bars = _frame(100, seed=9)
    res = run_backtest(SignalFrame.build(f, 1.0), bars, Scripted(default=None))
    assert res.decisions == [] and all(d["target"] is None for d in res.targets)


# ------------------------------------------------------------------ summaries of discrete strategies
@dataclass
class ScriptedOrders(Strategy):
    name: ClassVar[str] = "scripted_orders"
    orders: Optional[Dict[int, Order]] = None

    def decide(self, s, t):
        return (self.orders or {}).get(t)


def test_discrete_summary_reports_notional_breakeven_and_gross_edge():
    o = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 102.0])
    bars = Bars(o, o + 0.5, o - 0.5, o + 0.25)
    strat = ScriptedOrders(orders={0: Order("LONG", 1.0, max_hold=2), 4: Order("SHORT", 0.5, max_hold=2)})
    res = run_backtest(_signals(8), bars, strat, BacktestConfig())
    long_, short = res.trades
    ql = 10_000 / 101.0                                   # long: bar 1 open 101 -> bar 3 open 103
    qs = 0.5 * (10_000 + long_.net_pnl) / 105.0           # short, half the equity: bar 5 open 105 -> bar 7 open 102
    notional = ql * 101.0 + ql * 103.0 + qs * 105.0 + qs * 102.0
    gross = ql * 2.0 + qs * 3.0
    sm = res.summary
    assert sm["traded_notional"] == pytest.approx(notional, rel=1e-12)
    assert sm["gross_pnl"] == pytest.approx(gross, rel=1e-12)
    assert sm["breakeven_cost_bps"] == pytest.approx(gross / notional * 1e4 * 2, rel=1e-12)
    assert sm["gross_edge_per_trade_bps"] == pytest.approx(np.mean([2.0 / 101.0, 3.0 / 105.0]) * 1e4, rel=1e-12)
    assert sm["costs_paid"] == pytest.approx(COST * notional, rel=1e-12)
    flat = run_backtest(_signals(8), bars, build_strategy("always_flat")).summary
    assert flat["traded_notional"] == 0 and np.isnan(flat["breakeven_cost_bps"])
    assert np.isnan(flat["gross_edge_per_trade_bps"])


# ------------------------------------------------------------------ the timing null
NULL_KEYS = {"n_seeds", "random_mean_total_return", "random_mean_sharpe_net", "random_p05_total_return",
             "random_p95_total_return", "random_mean_gross_return", "percentile_total_return",
             "percentile_sharpe_net", "percentile_gross_return"}


def _vol_strategy():
    return Scripted(targets_by_bar={t: float(0.5 + 0.5 * np.sin(t / 40.0)) for t in range(0, 3000, 20)},
                    decide_every=20, band=0.1)


def test_exposure_strategies_get_a_deterministic_circular_shift_null_with_the_same_keys():
    f, bars = _frame(900, seed=10)
    s = SignalFrame.build(f, 1.0)
    cfg = BacktestConfig(random_seeds=12)
    res = backtest(s, bars, _vol_strategy(), cfg)
    null = res.baselines["random_same_freq"]
    assert NULL_KEYS <= set(null) and null["null"] == "circular_shift" and null["n_seeds"] == 12
    assert (null["shift_min"], null["shift_max"]) == (225, 675)                   # min(720, 900 // 4)
    again = backtest(s, bars, _vol_strategy(), cfg).baselines["random_same_freq"]
    assert again == null
    # the same shifts, replayed by hand
    shifts = np.random.default_rng(0).integers(225, 676, size=12)
    events = fill_events(res)
    assert np.isfinite(events).sum() == res.summary["n_rebalances"]
    rets = [run_exposure_backtest(s, bars, _ReplayFills(events=np.roll(events, int(k))), cfg).summary["total_return"]
            for k in shifts]
    assert null["random_mean_total_return"] == pytest.approx(np.mean(rets), rel=1e-12)
    assert null["percentile_total_return"] == pytest.approx(100 * np.mean(np.array(rets) < res.summary["total_return"]))
    assert random_same_frequency(s, bars, res, cfg) == null == circular_shift_null(s, bars, res, cfg)


def test_the_replay_of_an_unshifted_path_reproduces_the_strategy():
    f, bars = _frame(700, seed=11)
    s = SignalFrame.build(f, 1.0)
    for strat in (_vol_strategy(), Scripted(default=0.8, decide_every=7, band=0.25, trade_to_band_edge=True)):
        res = run_backtest(s, bars, strat)
        replay = run_exposure_backtest(s, bars, _ReplayFills(events=fill_events(res)))
        np.testing.assert_allclose(replay.equity, res.equity, rtol=0, atol=1e-9)
        assert [d["fill_bar"] for d in replay.decisions] == [d["fill_bar"] for d in res.decisions]
        assert res.summary["n_rebalances"] > 2


def test_the_null_never_calls_the_strategy_and_discrete_strategies_keep_the_random_null():
    f, bars = _frame(600, seed=12)
    s = SignalFrame.build(f, 1.0)
    strat = _vol_strategy()
    res = backtest(s, bars, strat, BacktestConfig(random_seeds=5))
    assert strat.calls == len(res.targets)
    assert res.baselines["random_same_freq"]["n_seeds"] == 5
    disc = backtest(s, bars, ScriptedOrders(orders={300: Order("LONG", 0.5, max_hold=5)}), BacktestConfig(random_seeds=5))
    rnd = disc.baselines["random_same_freq"]
    assert "null" not in rnd and {"trade_rate", "hold_bars", "size_frac"} <= set(rnd) and NULL_KEYS <= set(rnd)


def test_no_rebalances_give_an_empty_null():
    f, bars = _frame(300, seed=13)
    res = backtest(SignalFrame.build(f, 1.0), bars, Scripted(default=None), BacktestConfig(random_seeds=5))
    null = res.baselines["random_same_freq"]
    assert null["n_seeds"] == 0 and np.isnan(null["percentile_total_return"])


# ------------------------------------------------------------------ look-ahead
def test_lookahead_probe_catches_a_peeking_exposure_strategy():
    @dataclass
    class Peek(ExposureStrategy):
        name: ClassVar[str] = "peek_exposure"
        decide_every: int = 1

        def target(self, s, t, current):
            return 1.0 if t + 3 < len(s) and s.close[t + 3] > s.close[t] else -1.0

    f, bars = _frame(300, seed=14)
    with pytest.raises(AssertionError, match="look-ahead"):
        assert_no_lookahead(f, bars, Peek, var_scale=1.0, probes=(150,))


def test_lookahead_probe_catches_a_centred_ewma():
    """A strategy reading a centred (two-sided) smoother of the close sees the future; the probe fails it."""

    @dataclass
    class Centred(ExposureStrategy):
        name: ClassVar[str] = "centred_exposure"
        decide_every: int = 1
        band: float = 0.0

        def target(self, s, t, current):
            sm = pd.Series(s.close).rolling(21, center=True, min_periods=1).mean().to_numpy()
            return 1.0 if s.close[t] > sm[t] else 0.0

    f, bars = _frame(300, seed=15)
    with pytest.raises(AssertionError, match="look-ahead"):
        assert_no_lookahead(f, bars, Centred, var_scale=1.0, probes=(100, 200))

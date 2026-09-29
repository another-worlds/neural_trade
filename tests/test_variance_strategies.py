"""The variance-driven strategies of NT-077 (docs/research/2026-09-29-strategy-architectures/README.md
section 3): knobs and defaults, thresholds fitted on the calibration block only, the rules each one
trades by, and the look-ahead self-test for every registered strategy with both sigma sources."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy import (EWMA_WARMUP, Bars, ExposureStrategy, SignalFrame, Strategies, assert_no_lookahead,
                                   backtest, build_strategy, run_backtest, var_scale_from)

NEW = ("vol_target", "net_edge_kelly", "edge_over_cost", "vol_regime_long", "gated_ta")
WITH_SOURCE = ("vol_target", "vol_regime_long", "gated_ta")


def _frame(n=900, seed=0, move=0.4):
    rng = np.random.default_rng(seed)
    vol = 4e-4 * (1 + 0.8 * np.sin(np.arange(n + 30) / 90.0))
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, vol)))
    lc = close[:n]
    y = np.stack([close[h:n + h] - lc for h in (10, 15, 20)], 1)
    delta = y * move + rng.normal(0, 60, (n, 3))
    prob = 1 / (1 + np.exp(-delta / 25))
    var = (vol[:n, None] / 4e-4) ** 2 * rng.uniform(0.5, 1.5, (n, 3))       # a variance head with some skill
    f = PredictionFrame(y, lc, {h: delta[:, i] for i, h in enumerate(HORIZONS)},
                        {h: prob[:, i] for i, h in enumerate(HORIZONS)},
                        {h: var[:, i] for i, h in enumerate(HORIZONS)}, 250.0)
    return f, Bars.from_close(lc)


@pytest.fixture(scope="module")
def blocks():
    cal_f, _ = _frame(900, seed=101)
    f, bars = _frame(900, seed=102)
    vs = var_scale_from(cal_f)
    return SignalFrame.build(cal_f, vs), SignalFrame.build(f, vs), bars, f, vs


def _sh(s, source, h=2):
    return s.sigma_for(source, h) / s.close


def _cal_q(cal, source, q):
    x = _sh(cal, source)
    return float(np.quantile(x[np.isfinite(x)], q))


def test_new_strategies_are_registered_with_the_specified_defaults(blocks):
    cal = blocks[0]
    assert set(NEW) <= set(Strategies.list_names()) and Strategies.default == "calibrated_quantile"
    vt, nk, eoc, rs, gt = (build_strategy(n, calibration=cal) for n in NEW)
    assert isinstance(vt, ExposureStrategy) and isinstance(nk, ExposureStrategy)
    assert not isinstance(eoc, ExposureStrategy) and not isinstance(rs, ExposureStrategy)
    assert (vt.decide_every, vt.band, vt.trade_to_band_edge, vt.max_abs_exposure, vt.sigma_source) == \
        (60, 0.10, False, 1.0, "model")
    assert (nk.decide_every, nk.band, nk.trade_to_band_edge, nk.f, nk.cost) == (20, 0.10, True, 0.25, 0.0026)
    assert (eoc.cost, eoc.size, eoc.max_hold) == (0.0026, 1.0, 20)
    assert (rs.q_out, rs.q_in, rs.max_hold, rs.sigma_source) == (0.80, 0.70, 10 ** 9, "model")
    assert (gt.primary, gt.q, gt.max_hold, gt.fast, gt.slow, gt.bb_window, gt.bb_k) == \
        ("ma_cross", 0.5, 60, 20, 50, 20, 2.0)
    for strat in (vt, nk, eoc, rs, gt):
        assert strat.horizon == -1                           # the longest horizon: h2 on the reference setup
        assert strat.warmup() >= EWMA_WARMUP == 240


@pytest.mark.parametrize("name", WITH_SOURCE)
def test_sigma_source_is_validated(name, blocks):
    with pytest.raises(InvalidConfigurationError, match="sigma_source"):
        build_strategy(name, {"sigma_source": "garch"}, calibration=blocks[0])


@pytest.mark.parametrize("name", NEW)
def test_new_strategies_need_the_calibration_signalframe(name):
    with pytest.raises(InvalidConfigurationError, match="calibration"):
        build_strategy(name)
    with pytest.raises(InvalidConfigurationError, match="SignalFrame"):
        build_strategy(name, calibration={"0.5": 0.5})


def _fitted(strat):
    return {k: getattr(strat, k) for k in type(strat).fitted_fields}


@pytest.mark.parametrize("name,source", [(n, s) for n in NEW for s in (("model", "ewma") if n in WITH_SOURCE
                                                                        else (None,))])
def test_fit_reads_only_the_calibration_block(name, source, blocks):
    cal, s, bars, _, _ = blocks
    params = {"sigma_source": source} if source else {}
    strat = build_strategy(name, params, calibration=cal)
    before = _fitted(strat)
    assert all(np.isfinite(v) for v in before.values())
    run_backtest(s, bars, strat)                               # trading an out-of-sample block ...
    other_f, other_bars = _frame(900, seed=103)
    run_backtest(SignalFrame.build(other_f, 1.0), other_bars, strat)
    assert _fitted(strat) == before                            # ... never moves the thresholds
    again = build_strategy(name, params, calibration=cal)
    assert _fitted(again) == before
    # a threshold passed as a parameter is ignored: the fit sets it
    for key in before:
        assert _fitted(build_strategy(name, {**params, key: 123.0}, calibration=cal)) == before


def test_vol_target_holds_sigma_star_over_sigma_hat(blocks):
    cal, s, bars, _, _ = blocks
    for source in ("model", "ewma"):
        strat = build_strategy("vol_target", {"sigma_source": source}, calibration=cal)
        assert strat.sigma_star == pytest.approx(_cal_q(cal, source, 0.5), rel=1e-12)
        res = run_backtest(s, bars, strat)
        assert res.mode == "exposure" and res.summary["n_rebalances"] > 0
        assert [d["bar"] for d in res.targets] == list(range(240, 900 - 1, 60))
        sh = _sh(s, source)
        for d in res.targets:
            assert d["target"] == pytest.approx(min(1.0, strat.sigma_star / sh[d["bar"]]), rel=1e-12)
            assert (d["queued"] is not None) == (abs(d["target"] - d["current"]) > 0.10)
        assert all(0 <= d["to"] <= 1 for d in res.decisions)


def test_net_edge_kelly_aims_at_the_net_edge_over_variance(blocks):
    cal, s, bars, _, _ = blocks
    strat = build_strategy("net_edge_kelly", {"f": 0.5, "cost": 0.0005}, calibration=cal)
    res = run_backtest(s, bars, strat)
    assert [d["bar"] for d in res.targets] == list(range(240, 900 - 1, 20))
    active = 0
    for d in res.targets:
        t = d["bar"]
        mu = s.mu_gauss[t, 2] / s.close[t]
        assert mu == pytest.approx(s.sigma_ret[t, 2] * float(np.sqrt(2) * _erfinv(2 * np.clip(s.p[t, 2], 1e-6, 1 - 1e-6) - 1)),
                                   rel=1e-9)
        aim = np.clip(np.sign(mu) * 0.5 * max(abs(mu) - 0.0005, 0.0) / s.sigma_ret[t, 2] ** 2, -1, 1)
        assert d["target"] == pytest.approx(aim, abs=1e-12)
        active += d["target"] != 0
        if d["queued"] is not None:                             # traded to the band's edge
            gap = d["target"] - d["current"]
            assert d["queued"] == pytest.approx(d["current"] + np.sign(gap) * (abs(gap) - 0.10), abs=1e-12)
    assert active > 0 and res.summary["n_rebalances"] > 0


def _erfinv(x):
    from scipy.special import erfinv

    return erfinv(x)


def test_edge_over_cost_trades_only_when_the_expected_move_beats_the_cost(blocks):
    cal, s, bars, _, _ = blocks
    strat = build_strategy("edge_over_cost", {"cost": 0.0004}, calibration=cal)
    res = run_backtest(s, bars, strat)
    assert res.summary["n_trades"] > 0
    mu = s.mu_gauss[:, 2] / s.close
    for d in res.decisions:
        assert abs(mu[d["bar"]]) > 0.0004 and d["side"] == ("LONG" if mu[d["bar"]] > 0 else "SHORT")
        assert d["bar"] >= 240
    for tr in res.trades:
        assert tr.sl is None and tr.tp is None and tr.bars_held <= 20
    flat = {d["bar"] for d in res.decisions}
    held = np.zeros(len(s), bool)
    for tr in res.trades:
        held[tr.entry_bar - 1: tr.exit_bar] = True                 # decided at entry - 1 ... exit
    for t in range(240, len(s) - 1):
        if not held[t] and t not in flat:
            assert abs(mu[t]) <= 0.0004


@pytest.mark.parametrize("source", ["model", "ewma"])
def test_vol_regime_long_enters_in_calm_and_leaves_in_storm(source, blocks):
    cal, s, bars, _, _ = blocks
    strat = build_strategy("vol_regime_long", {"q_out": 0.9, "sigma_source": source}, calibration=cal)
    assert strat.q_in == pytest.approx(0.8)
    assert strat.in_below == pytest.approx(_cal_q(cal, source, 0.8), rel=1e-12)
    assert strat.out_above == pytest.approx(_cal_q(cal, source, 0.9), rel=1e-12)
    res = run_backtest(s, bars, strat)
    sh = _sh(s, source)
    assert res.summary["n_trades"] > 0
    assert all(d["side"] == "LONG" and sh[d["bar"]] < strat.in_below for d in res.decisions)
    for tr in res.trades:
        if tr.exit_reason == "VOL":
            assert sh[tr.exit_bar - 1] > strat.out_above
        else:
            assert tr.exit_reason == "EOW"
    set_q = build_strategy("vol_regime_long", {"q_out": 0.9, "q_in": 0.5}, calibration=cal)
    assert set_q.in_below == pytest.approx(_cal_q(cal, "model", 0.5), rel=1e-12)
    with pytest.raises(InvalidConfigurationError, match="q_in"):
        build_strategy("vol_regime_long", {"q_out": 0.5, "q_in": 0.7}, calibration=cal)


def _sma(c, t, w):
    return float(np.mean(c[t - w + 1: t + 1]))


@pytest.mark.parametrize("source", ["model", "ewma"])
def test_gated_ta_ma_cross(source, blocks):
    cal, s, bars, _, _ = blocks
    strat = build_strategy("gated_ta", {"q": 0.3, "sigma_source": source}, calibration=cal)
    assert strat.gate == pytest.approx(_cal_q(cal, source, 0.3), rel=1e-12)
    res = run_backtest(s, bars, strat)
    c, sh = s.close, _sh(s, source)
    assert res.summary["n_trades"] > 0
    for d in res.decisions:
        t = d["bar"]
        before, now = _sma(c, t - 1, 20) - _sma(c, t - 1, 50), _sma(c, t, 20) - _sma(c, t, 50)
        assert sh[t] >= strat.gate
        assert (before <= 0 < now) if d["side"] == "LONG" else (before >= 0 > now)
    for tr in res.trades:
        assert tr.bars_held <= 60 and tr.exit_reason in ("REV", "TIME", "EOW")
        if tr.exit_reason == "REV":
            t = tr.exit_bar - 1
            gap = _sma(c, t, 20) - _sma(c, t, 50)
            assert gap < 0 if tr.side == "LONG" else gap > 0


def test_gated_ta_bollinger(blocks):
    cal, s, bars, _, _ = blocks
    strat = build_strategy("gated_ta", {"primary": "bollinger", "q": 0.3}, calibration=cal)
    res = run_backtest(s, bars, strat)
    c = s.close
    assert res.summary["n_trades"] > 0
    for d in res.decisions:
        t = d["bar"]
        w = c[t - 19: t + 1]
        mid, sd = w.mean(), w.std()
        assert c[t] > mid + 2 * sd if d["side"] == "LONG" else c[t] < mid - 2 * sd
    for tr in res.trades:
        assert tr.exit_reason in ("MID", "TIME", "EOW")
        if tr.exit_reason == "MID":
            t = tr.exit_bar - 1
            mid = c[t - 19: t + 1].mean()
            assert c[t] < mid if tr.side == "LONG" else c[t] > mid
    with pytest.raises(InvalidConfigurationError, match="primary"):
        build_strategy("gated_ta", {"primary": "rsi"}, calibration=cal)


@pytest.mark.parametrize("name,source", [(n, s) for n in sorted(Strategies.list_names())
                                         for s in (("model", "ewma") if n in WITH_SOURCE else (None,))])
def test_no_lookahead_every_strategy_both_sigma_sources(name, source, blocks):
    """Every registered strategy, fitted on a synthetic calibration frame; the variance ones with
    both sigma sources (the EWMA is rebuilt from the perturbed closes). Probes on decision bars."""
    cal, _, _, f, vs = blocks
    fields = {fl.name for fl in dataclasses.fields(Strategies.get(name))} if dataclasses.is_dataclass(
        Strategies.get(name)) else set()
    params = {"sigma_source": source} if source else {}
    if name == "net_edge_kelly":
        params = {"cost": 0.0003}                               # so that it trades on this frame
    elif name == "edge_over_cost":
        params = {"cost": 0.0003}
    assert not params or set(params) <= fields
    make = lambda: build_strategy(name, params, calibration=cal)  # noqa: E731
    if name in NEW:                                             # the probe is not vacuous: it trades
        assert run_backtest(SignalFrame.build(f, vs), blocks[2], make()).summary["n_trades"] > 0
    assert_no_lookahead(f, blocks[2], make, var_scale=vs, probes=(300, 480, 720))


def test_exposure_strategies_carry_the_timing_null_through_backtest(blocks):
    cal, s, bars, _, _ = blocks
    res = backtest(s, bars, build_strategy("vol_target", {"sigma_source": "ewma"}, calibration=cal))
    null = res.baselines["random_same_freq"]
    assert null["null"] == "circular_shift" and null["n_seeds"] == 100
    assert set(res.baselines) == {"buy_and_hold", "always_flat", "random_same_freq"}
    disc = backtest(s, bars, build_strategy("vol_regime_long", calibration=cal))
    assert "null" not in disc.baselines["random_same_freq"]

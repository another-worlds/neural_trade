"""Correlation noise bands are read in r units wherever they are drawn or printed next to an r.

``stats.corr_null`` is the 95% half-width of atanh(r) (the Fisher-z scale). Used directly as a band
for an r, a Spearman rho or an MCC it is too wide, most of all for a small n: 0.74 instead of 0.63
at n = 10, 0.98 instead of 0.75 at n = 7. ``stats.corr_null_r`` back-transforms it with tanh. Each
test below pins a value the old code printed or drew differently.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats as sps

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.visualization import stats as S


def _exact_r(n: int) -> float:
    """Two-sided 5% critical |r| of a Pearson r on n pairs with no relationship (t test)."""
    t = sps.t.ppf(0.975, n - 2)
    return float(t / math.sqrt(n - 2 + t * t))


# ------------------------------------------------------------------ stats
@pytest.mark.parametrize("n", [7, 10, 19, 50, 286, 3000])
def test_corr_null_r_is_the_fisher_band_back_transformed_to_r(n):
    r = S.corr_null_r(n)
    assert r == pytest.approx(math.tanh(S.corr_null(n)), abs=1e-12)
    assert 0 < r < 1 and r < S.corr_null(n)
    assert r == pytest.approx(_exact_r(n), abs=0.003)               # the exact t-test value


def test_corr_null_keeps_its_fisher_z_value_and_both_count_effective_samples():
    """comparison.py multiplies corr_null by (1 - r^2) as a z-scale half-width: it must not change."""
    assert S.corr_null(19) == pytest.approx(1.959964 / 4)
    assert S.corr_null(7) == pytest.approx(1.959964 / 2)
    assert f"{S.corr_null_r(19):.2f}" == "0.45" and f"{S.corr_null_r(10):.2f}" == "0.63"
    assert f"{S.corr_null_r(7):.2f}" == "0.75"
    assert S.corr_null_r(200, steps=20) == S.corr_null_r(10)
    assert S.corr_null_r(2) < 1                                       # never reaches 1, even at n_eff 1


# ------------------------------------------------------------------ trade analytics (Spearman rho)
def test_trade_analytics_chance_band_is_in_rho_units_for_few_trades(viz_backtest):
    """10 trades: 'chance ±0.63', not the Fisher-z ±0.74."""
    from neural_trade.visualization.trade_analytics import _disjoint_windows, trade_analytics_figure

    res, bars, sig, _ = viz_backtest
    few = dataclasses.replace(res, trades=res.trades[:10])
    fig = trade_analytics_figure(few, bars, signals=sig)
    assert "(chance ±0.63)" in fig.layout.annotations[6].text       # conviction vs gross move
    assert "(chance ±0.63)" in fig.layout.annotations[8].text       # predicted vs realised over the hold
    fig = trade_analytics_figure(few, bars, signals=sig, horizon_steps=(10, 15, 20))
    d = np.array([t.entry_bar - 1 for t in few.trades])
    k = _disjoint_windows(d[d + 15 < len(sig.close)], 15)
    assert k <= 10
    assert f"(chance ±{S.corr_null_r(max(k, 1)):.2f})" in fig.layout.annotations[8].text
    assert f"(chance ±{S.corr_null(max(k, 1)):.2f})" not in fig.layout.annotations[8].text


# ------------------------------------------------------------------ training dashboard (MCC)
def test_training_mcc_band_is_in_mcc_units_on_a_small_validation_block(viz_config):
    """n_val 200 gives n_eff 20 / 13 / 10: bands ±0.44 / ±0.55 / ±0.63 (Fisher z: ±0.48 / ±0.62 / ±0.74)."""
    from tests.test_viz_training import _panel_axes, _served_rows

    from neural_trade.visualization.training_dashboard import training_dashboard_figure, training_health

    want = [S.corr_null_r(n) for n in (20, 13, 10)]
    assert [round(w, 2) for w in want] == [0.44, 0.55, 0.63]
    rows = _served_rows(6, best=5)
    rows[-1]["val_dir_mcc_h0"] = 0.46                                # beyond ±0.44, inside the old ±0.48
    tile = training_health(rows, viz_config, n_val=200)["direction h0"]
    assert tile["status"] == "good" and "above chance (±0.44)" in tile["value"]
    fig = training_dashboard_figure(rows, viz_config, n_val=200)
    for (r, c), center, scale in (((2, 1), 0.0, 1.0), ((2, 2), 0.0, 1.0), ((3, 1), 0.5, 0.5)):
        _, ya = _panel_axes(fig, r, c)
        rects = [s for s in fig.layout.shapes if s.type == "rect" and s.yref == ya]
        np.testing.assert_allclose(sorted(s.y1 for s in rects), [center + scale * w for w in want], atol=1e-9)


# ------------------------------------------------------------------ coherence (P(up) correlation)
def test_coherence_chance_note_is_in_r_units():
    """200 samples, h2 = 20 bars: n_eff 10, so |r| < 0.63 is within chance (the Fisher-z value is 0.74)."""
    from neural_trade.visualization.analytics_confidence import coherence_analytics_figure

    rng = np.random.default_rng(1)
    n = 200
    sig = rng.normal(0, 1, (n, 3))
    y = 100 * sig + rng.normal(0, 100, (n, 3))
    frame = PredictionFrame(y=y, last_close=np.full(n, 100_000.0),
                            delta={h: 30 * sig[:, i] for i, h in enumerate(HORIZONS)},
                            direction_prob={h: 1 / (1 + np.exp(-0.5 * sig[:, i])) for i, h in enumerate(HORIZONS)},
                            variance_scaled={h: np.ones(n) for h in HORIZONS}, pred_scale=100.0)
    fig = coherence_analytics_figure(frame, n_boot=0)
    note = next(a.text for a in fig.layout.annotations if "is within chance" in (a.text or ""))
    assert "|r| &lt; 0.63 is within chance (n_eff = N / 20 = 10)" in note


# ------------------------------------------------------------------ delta analytics (rolling correlation)
def test_delta_corr_band_is_in_r_units():
    from neural_trade.visualization.analytics_delta import corr_band

    assert corr_band(40, 2.0) == pytest.approx(S.corr_null_r(20))
    assert f"{corr_band(40, 2.0):.2f}" == "0.44"                     # Fisher z: 0.48
    assert f"{corr_band(500, 50.0):.2f}" == "0.63"                   # a 500-bar window at deff 50: n_eff 10


# ------------------------------------------------------------------ tables
def test_table_noise_bands_are_in_r_units(viz_config):
    from tests.test_viz_tables import _windows_frame

    from neural_trade.visualization import analytics_tables as AT

    frame, _ = _windows_frame(n=200, seed=4)
    col = "corr noise band +/- (95%, n_eff)"
    dq = AT.delta_quality_table(frame, viz_config, digits=None, usd_digits=None)
    tm = AT.trailing_move_table({"test": frame}, viz_config, digits=None)
    for h, steps in zip(HORIZONS, (10, 15, 20)):
        assert dq.loc[col, h] == pytest.approx(S.corr_null_r(200, steps=steps))
        assert tm.loc[("test", h), col] == pytest.approx(S.corr_null_r(200, steps=steps))
    assert f"{dq.loc[col, 'h2']:.2f}" == "0.63" and f"{tm.loc[('test', 'h2'), col]:.2f}" == "0.63"   # z: 0.74

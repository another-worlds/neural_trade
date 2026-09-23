"""Calibration package: the split guard, conformal coverage, temperature direction, save/load.

Pure numpy - runs without TensorFlow.
"""
from __future__ import annotations

import numpy as np
import pytest

from calibration import CalibrationPipeline

H = ("h0", "h1", "h2")


SHARPEN = 3.0  # how over-confident the direction head is; temperature scaling must undo it


def _synthetic(rng, n, sharpen=SHARPEN):
    """Predictions in the TrainResult layout whose direction head is over-confident BY CONSTRUCTION.

    ``z`` is a latent signal. The honest probability of an up move is ``sigmoid(z)`` and the
    label is drawn from exactly that, so a perfectly calibrated head would report
    ``sigmoid(z)``. The head instead reports ``sigmoid(sharpen * z)`` - the same ranking,
    but too sure of itself. Temperature scaling must then fit ``T ~ sharpen`` and pull every
    probability back toward 0.5.

    The earlier version built the probabilities from ``sign(y) * U(0.2, 1) + N(0, 0.5)``,
    which produced a head that was *under*-confident relative to its own accuracy, so
    fitting correctly sharpened it and the "must move toward 0.5" assertion was simply
    the wrong expectation for that data.
    """
    lc = np.full(n, 110_000.0)
    sigma = np.array([40.0, 90.0, 160.0])
    z = rng.normal(0.0, 1.0, (n, 3))
    p_true = 1.0 / (1.0 + np.exp(-z))
    up = rng.uniform(size=(n, 3)) < p_true
    # sign carries the label; magnitude is independent, so y stays marginally symmetric
    y = np.abs(rng.normal(0.0, 1.0, (n, 3))) * sigma * np.where(up, 1.0, -1.0)
    delta_pred = y + rng.normal(0.0, 1.0, (n, 3)) * sigma * 0.8
    probs = 1.0 / (1.0 + np.exp(-sharpen * z))
    preds = {
        "delta": {h: delta_pred[:, i] for i, h in enumerate(H)},
        "direction_prob": {h: probs[:, i] for i, h in enumerate(H)},
        "variance": {h: np.full(n, float(sigma[i] ** 2)) for i, h in enumerate(H)},
    }
    return preds, y, lc, p_true


def test_fit_refuses_the_test_split_and_requires_a_calibration_block():
    class _Result:
        config = type("C", (), {"DIR_DEADBAND_BPS": 0.0})()

    with pytest.raises(ValueError, match="refused"):
        CalibrationPipeline().fit(_Result(), split="test")
    with pytest.raises(ValueError, match="no calibration split"):
        CalibrationPipeline().fit(_Result(), split="cal")


def test_fit_apply_coverage_temperature_and_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    cal_preds, y_cal, lc_cal, _ = _synthetic(rng, 3_000)
    pipe = CalibrationPipeline().fit_from_arrays(cal_preds, y_cal, lc_cal, deadband_bps=0.0, conformal_alpha=0.1)

    te_preds, y_te, _, p_true = _synthetic(rng, 20_000)
    out = pipe.apply(te_preds, alpha=0.1)
    for i, h in enumerate(H):
        lo, hi = out["intervals"][h]
        cov = float(np.mean((y_te[:, i] >= lo) & (y_te[:, i] <= hi)))
        assert 0.88 <= cov <= 0.92, (h, cov)  # split-conformal: coverage >= 1 - alpha, not far above
        raw, cal = te_preds["direction_prob"][h], out["direction_prob"][h]
        assert np.mean(np.abs(cal - 0.5)) < np.mean(np.abs(raw - 0.5)), "over-confident probs must move toward 0.5"
        # and they must move toward the probability that actually generated the labels
        assert np.mean(np.abs(cal - p_true[:, i])) < np.mean(np.abs(raw - p_true[:, i]))

    pipe.save(str(tmp_path / "calib"))
    loaded = CalibrationPipeline.load(str(tmp_path / "calib"))
    out2 = loaded.apply(te_preds, alpha=0.1)
    for h in H:
        np.testing.assert_allclose(out2["direction_prob"][h], out["direction_prob"][h], atol=1e-6)
        np.testing.assert_allclose(out2["intervals"][h][0], out["intervals"][h][0], atol=1e-6)
        np.testing.assert_allclose(out2["intervals"][h][1], out["intervals"][h][1], atol=1e-6)


def _regime(rng, n, vol, lookback=60, steps=(10, 15, 20)):
    """Random-walk windows at per-bar volatility ``vol`` and forward deltas at the same volatility;
    the model predicts zero change."""
    win = 110_000 + np.cumsum(rng.normal(0, vol, (n, lookback)), axis=1)
    y = np.stack([rng.normal(0, vol * np.sqrt(k), n) for k in steps], 1)
    preds = {"delta": {h: np.zeros(n) for h in H}, "direction_prob": {h: np.full(n, 0.5) for h in H},
             "variance": {h: np.ones(n) for h in H}}
    return preds, y, win[:, -1], win


@pytest.mark.parametrize("mode, lo_cov, hi_cov", [("none", 0.975, 1.0), ("realized_vol", 0.87, 0.93)])
def test_realized_vol_conformal_survives_a_volatility_regime_shift(mode, lo_cov, hi_cov):
    rng = np.random.default_rng(7)
    cal_preds, y_cal, lc_cal, win_cal = _regime(rng, 4_000, vol=30.0)   # calm-to-busy: cal is 1.5x more volatile
    te_preds, y_te, _, win_te = _regime(rng, 20_000, vol=20.0)
    pipe = CalibrationPipeline(conformal_scale=mode).fit_from_arrays(
        cal_preds, y_cal, lc_cal, windows=win_cal, horizon_steps=(10, 15, 20))
    out = pipe.apply(te_preds, windows=win_te)
    for i, h in enumerate(H):
        lo, hi = out["intervals"][h]
        cov = float(np.mean((y_te[:, i] >= lo) & (y_te[:, i] <= hi)))
        assert lo_cov <= cov <= hi_cov, (mode, h, cov)


def test_normalized_conformal_needs_windows_and_round_trips(tmp_path):
    rng = np.random.default_rng(8)
    preds, y, lc, win = _regime(rng, 2_000, vol=25.0)
    pipe = CalibrationPipeline(conformal_scale="realized_vol").fit_from_arrays(preds, y, lc, windows=win)
    with pytest.raises(ValueError, match="raw input windows"):
        pipe.apply(preds)
    pipe.save(str(tmp_path / "c"))
    loaded = CalibrationPipeline.load(str(tmp_path / "c"))
    assert loaded.conformal_scale == "realized_vol"
    a, b = pipe.apply(preds, windows=win), loaded.apply(preds, windows=win)
    for h in H:
        np.testing.assert_allclose(a["intervals"][h][1], b["intervals"][h][1], rtol=1e-12)
    with pytest.raises(ValueError, match="conformal_scale"):
        CalibrationPipeline(conformal_scale="bogus")

"""Calibration package: the split guard, conformal coverage, temperature direction, save/load.

Pure numpy - runs without TensorFlow.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.calibration import CalibrationPipeline

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


# ---------------------------------------------------------------------------- scales, online, diagnostics
def test_interval_scale_modes_and_errors():
    from neural_trade.calibration.conformal import interval_scale

    rng = np.random.default_rng(0)
    win = 100 + np.cumsum(rng.normal(0, 2.0, (50, 60)), axis=1)
    var = {h: np.full(50, 0.25) for h in H}
    assert all((v == 1).all() and len(v) == 50 for v in interval_scale("none", windows=win).values())
    assert len(interval_scale("none", variance_scaled=var)["h1"]) == 50
    np.testing.assert_allclose(interval_scale("sigma", variance_scaled=var, pred_scale=200.0)["h2"], 100.0)
    rv = interval_scale("realized_vol", windows=win, horizon_steps=(4, 9, 16))
    np.testing.assert_allclose(rv["h1"] / rv["h0"], 1.5) and np.testing.assert_allclose(rv["h2"] / rv["h0"], 2.0)
    for mode, kw, match in (("bogus", {}, "one of"), ("none", {}, "needs n"), ("sigma", {"variance_scaled": var}, "pred_scale"),
                            ("realized_vol", {}, "raw input windows")):
        with pytest.raises(ValueError, match=match):
            interval_scale(mode, **kw)


def test_conformal_regressor_scale_contract():
    from neural_trade.calibration.conformal import ConformalRegressor

    plain = ConformalRegressor().fit(np.arange(10.0), np.zeros(10))
    with pytest.raises(ValueError, match="without a scale"):
        plain.predict_interval(np.zeros(3), scale=np.ones(3))
    scaled = ConformalRegressor().fit(np.arange(10.0), np.zeros(10), scale=np.ones(10))
    with pytest.raises(ValueError, match="pass scale"):
        scaled.predict_interval(np.zeros(3))
    with pytest.raises(ValueError, match="entries"):
        scaled.predict_interval(np.zeros(3), scale=np.ones(4))
    with pytest.raises(RuntimeError):
        ConformalRegressor().predict_interval(np.zeros(3))
    lo, hi = scaled.predict_interval(np.zeros(2), scale=np.array([1.0, 1e-9]))
    assert hi[1] - lo[1] > 0  # the floor keeps a near-zero scale from collapsing the interval


def test_online_calibrator_learns_to_soften_an_overconfident_head(tmp_path):
    from neural_trade.calibration.online_calibrator import OnlineTemperatureCalibrator

    rng = np.random.default_rng(1)
    oc = OnlineTemperatureCalibrator(lr=0.05, ema_decay=0.9)
    assert oc.calibrate(0.8) == 0.8 and np.allclose(oc.calibrate_array(np.array([0.2, 0.8])), [0.2, 0.8])
    for _ in range(3000):
        z = rng.normal(0, 1)
        y = float(rng.uniform() < 1 / (1 + np.exp(-z)))
        oc.update(1 / (1 + np.exp(-3 * z)), y, "h1")   # reports sigmoid(3z): over-confident
    assert oc.state["h1"]["T_ema"] > 1.5 and oc.state["h1"]["n_updates"] == 3000
    assert abs(oc.calibrate(0.9, "h1") - 0.5) < 0.4
    np.testing.assert_allclose(oc.calibrate_array(np.array([0.9]), "h1")[0], oc.calibrate(0.9, "h1"), rtol=1e-9)
    oc.save(str(tmp_path / "online.json"))
    again = OnlineTemperatureCalibrator.from_file(str(tmp_path / "online.json"))
    assert again.state["h1"]["T_ema"] == pytest.approx(oc.state["h1"]["T_ema"])


def test_pipeline_online_intervals_summary_and_legacy_load(tmp_path, capsys):
    import os

    rng = np.random.default_rng(2)
    preds, y, lc, _ = _synthetic(rng, 2_000)
    with pytest.raises(RuntimeError):
        CalibrationPipeline().apply(preds)
    pipe = CalibrationPipeline().fit_from_arrays(preds, y, lc)
    iv = pipe.predict_intervals(preds)
    np.testing.assert_allclose(iv["h1"][1], pipe.apply(preds)["intervals"]["h1"][1])
    before = pipe.calibrate_online(0.9)
    for _ in range(200):
        pipe.update_online(0.95, 0.0)
    assert pipe.calibrate_online(0.9) < before
    assert pipe.calibrate_online_array(np.array([0.9]))[0] == pytest.approx(pipe.calibrate_online(0.9))
    capsys.readouterr()
    pipe.summary()
    out = capsys.readouterr().out
    assert "Temperature scaling" in out and "Conformal regressors" in out
    d = tmp_path / "old"
    pipe.save(str(d))
    os.remove(d / "pipeline_meta.json")        # bundles written before the meta file existed
    os.remove(d / "online_calibrator.json")
    old = CalibrationPipeline.load(str(d))
    assert old.conformal_scale == "none" and old.online is not None
    np.testing.assert_allclose(old.apply(preds)["intervals"]["h0"][0], pipe.apply(preds)["intervals"]["h0"][0])


def test_delta_shrinkage_scales_the_price_head_to_its_calibration_value(tmp_path):
    rng = np.random.default_rng(9)
    n = 5_000
    y = rng.normal(0, 200, (n, 3))
    signal = y + rng.normal(0, 400, (n, 3))          # informative but too loud: LS scale ~0.2
    noise = rng.normal(0, 300, (n, 3))               # pure noise: LS scale ~0
    for d, lo_beta, hi_beta in ((signal, 0.15, 0.25), (noise, 0.0, 0.03)):
        preds = {"delta": {h: d[:, i] for i, h in enumerate(H)}, "direction_prob": {h: np.full(n, 0.5) for h in H},
                 "variance": {h: np.ones(n) for h in H}}
        pipe = CalibrationPipeline(shrink_delta=True).fit_from_arrays(preds, y, np.full(n, 110_000.0))
        for h in H:
            assert lo_beta <= pipe.delta_scale[h] <= hi_beta, (h, pipe.delta_scale[h])
        out = pipe.apply(preds)
        np.testing.assert_allclose(out["delta"]["h1"], pipe.delta_scale["h1"] * d[:, 1])
        lo, hi = out["intervals"]["h1"]
        np.testing.assert_allclose((lo + hi) / 2, out["delta"]["h1"], atol=1e-6)   # centred on the served delta
        ev_raw = 1 - np.var(y[:, 1] - d[:, 1]) / np.var(y[:, 1])
        ev_served = 1 - np.var(y[:, 1] - out["delta"]["h1"]) / np.var(y[:, 1])
        assert ev_served >= ev_raw and ev_served > -0.01
    pipe.save(str(tmp_path / "s"))
    assert CalibrationPipeline.load(str(tmp_path / "s")).delta_scale == pytest.approx(pipe.delta_scale)
    off = CalibrationPipeline().fit_from_arrays(preds, y, np.full(n, 110_000.0))
    assert off.delta_scale == {h: 1.0 for h in H}

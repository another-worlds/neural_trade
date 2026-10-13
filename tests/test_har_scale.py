"""Opt-in CONFORMAL_SCALE='har' (tactical findings H85 / H87): a HAR forecast of the absolute return as the conformal
interval scale. The default 'realized_vol' path is untouched (golden run; see the logs named in the PR)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_trade.calibration.conformal import SCALE_MODES, interval_scale
from neural_trade.calibration.har_scale import RIDGE_ALPHA, HarScale


def _series(n=900, seed=3):
    rng = np.random.default_rng(seed)
    vol = 0.0004 * np.exp(0.6 * np.sin(np.arange(n) / 80.0))
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, vol)))
    ts = (pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n), unit="min")).to_numpy("datetime64[ns]")
    return close, ts


def test_hand_computed_har_on_a_small_series():
    """Features, ridge coefficients and sigma against an independent loop-based computation."""
    close, ts = _series()
    har = HarScale(bar_minutes=1.0, windows_minutes=(3, 6, 12, 24))
    train = np.arange(60, 500)
    H = 5
    har.fit(close, ts, train, (H,))
    lc = np.log(close)
    r2 = np.r_[0.0, np.diff(lc) ** 2]

    def feats(t):
        row = [np.log(max(r2[t - w + 1:t + 1].sum() / w, 1e-12)) for w in (3, 6, 12, 24)]
        tod = (ts[t].astype("int64") % (86_400 * 10**9)) / (86_400 * 10**9) * 2 * np.pi
        return row + [np.sin(tod), np.cos(tod)]

    t = train[train + H <= len(close) - 1]
    X = np.array([feats(i) for i in t])
    y = np.array([np.log(abs(lc[i + H] - lc[i]) + 1e-5) for i in t])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.clip((X - mu) / sd, -5, 5)
    coef = np.linalg.solve(Z.T @ Z + RIDGE_ALPHA * np.eye(6), Z.T @ (y - y.mean()))
    m = har.models[H]
    np.testing.assert_allclose(m["coef"], coef, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(har.features(close, ts, t), X, rtol=1e-10)
    ylo, yhi = y.min() - 1.0, y.max() + 1.0
    e = np.exp(np.clip(Z @ coef + y.mean(), ylo, yhi))
    c = np.mean(np.abs(lc[t + H] - lc[t]) / e)
    probe = np.array([520, 600, 700])
    want = np.exp(np.clip(np.clip((np.array([feats(i) for i in probe]) - mu) / sd, -5, 5) @ coef + y.mean(), ylo, yhi)) * c * close[probe]
    np.testing.assert_allclose(har.sigma(close, ts, probe, H), want, rtol=1e-9)


def test_scale_is_causal_finite_positive():
    close, ts = _series()
    har = HarScale(windows_minutes=(3, 6, 12, 24)).fit(close, ts, np.arange(60, 500), (5, 8))
    t = np.arange(520, 800)
    base = har.sigma(close, ts, t, 5)
    assert np.isfinite(base).all() and (base > 0).all()
    # the future of each decision bar never matters: changing every later bar leaves earlier scales identical
    cut = 650
    close2 = close.copy()
    close2[cut + 1:] *= np.exp(np.random.default_rng(1).normal(0, 0.05, len(close) - cut - 1))
    early = t <= cut
    np.testing.assert_array_equal(har.sigma(close2, ts, t[early], 5), base[early])
    assert not np.array_equal(har.sigma(close2, ts, t[~early], 5), base[~early])


def test_fit_reads_only_the_training_decision_bars():
    """Cal / test bars (after the training decisions + horizon) cannot move the fit."""
    close, ts = _series()
    train = np.arange(60, 400)
    a = HarScale(windows_minutes=(3, 6, 12, 24)).fit(close, ts, train, (5,))
    close2 = close.copy()
    close2[train[-1] + 5 + 1:] *= 3.0          # everything after the last training label
    b = HarScale(windows_minutes=(3, 6, 12, 24)).fit(close2, ts, train, (5,))
    for k in ("coef", "mu", "sd"):
        np.testing.assert_array_equal(a.models[5][k], b.models[5][k])
    assert a.models[5]["c"] == b.models[5]["c"]


def test_too_few_training_rows_is_refused():
    close, ts = _series()
    with pytest.raises(ValueError, match="training rows"):
        HarScale(windows_minutes=(3, 6, 12, 24)).fit(close, ts, np.array([60, 300, 500]), (5,))


def test_interval_scale_har_mode():
    assert "har" in SCALE_MODES
    sig = {h: np.full(7, 2.0) for h in ("h0", "h1", "h2")}
    out = interval_scale("har", har_sigma=sig, horizon_steps=(10, 15, 20))
    assert all((v == 2.0).all() for v in out.values())
    with pytest.raises(ValueError, match="har_sigma"):
        interval_scale("har")
    sig["h1"][0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        interval_scale("har", har_sigma=sig)


def test_config_accepts_har_and_default_is_realized_vol():
    from neural_trade.core.config import Config

    assert Config().CONFORMAL_SCALE == "realized_vol"
    cfg = Config()
    cfg.CONFORMAL_SCALE = "har"
    cfg.validate()
    cfg.CONFORMAL_SCALE = "iqr"
    with pytest.raises(Exception, match="CONFORMAL_SCALE"):
        cfg.validate()


def test_pipeline_har_needs_sigma_and_realized_vol_path_is_bit_identical():
    from neural_trade.calibration.pipeline import CalibrationPipeline

    rng = np.random.default_rng(5)
    n = 400
    H = ("h0", "h1", "h2")
    win = 100 + np.cumsum(rng.normal(0, 1.0, (n, 60)), axis=1)
    y = rng.normal(0, 3.0, (n, 3))
    preds = {"delta": {h: rng.normal(0, 1, n) for h in H}, "direction_prob": {h: rng.uniform(0.3, 0.7, n) for h in H},
             "variance": {h: np.ones(n) for h in H}}
    base = CalibrationPipeline(conformal_scale="realized_vol").fit_from_arrays(preds, y, win[:, -1], windows=win)
    again = CalibrationPipeline(conformal_scale="realized_vol").fit_from_arrays(
        preds, y, win[:, -1], windows=win, har_sigma=None)
    a, b = base.apply(preds, windows=win), again.apply(preds, windows=win, har_sigma=None)
    for h in H:
        np.testing.assert_array_equal(a["intervals"][h][0], b["intervals"][h][0])
        np.testing.assert_array_equal(a["intervals"][h][1], b["intervals"][h][1])
    sig = {h: np.abs(rng.normal(2.0, 0.5, n)) + 0.1 for h in H}
    har = CalibrationPipeline(conformal_scale="har").fit_from_arrays(preds, y, win[:, -1], har_sigma=sig)
    with pytest.raises(ValueError, match="har_sigma"):
        har.apply(preds)
    out = har.apply(preds, har_sigma=sig)
    for h in H:
        lo, hi = out["intervals"][h]
        assert np.isfinite(lo).all() and (hi > lo).all()


def test_short_training_span_is_refused_not_clipped():
    """A training block shorter than two longest windows cannot make the long RV / time of day vary: refuse."""
    close, ts = _series()
    with pytest.raises(ValueError, match="span"):
        HarScale(windows_minutes=(3, 6, 12, 24)).fit(close, ts, np.arange(60, 100), (5,))
    # the default 1440-minute window needs 2880 bars of decisions
    close, ts = _series(n=6000)
    with pytest.raises(ValueError, match="2880"):
        HarScale().fit(close, ts, np.arange(1500, 3000), (5,))
    HarScale().fit(close, ts, np.arange(1500, 4500), (5,))


def test_trainer_hook_aligns_with_blocks_and_refuses_the_tiny_layout(tf, tiny_close_only_config, tmp_path):
    """_har_scales: the blocks' decision bars (anchor - 1) are the blocks' last_close bars; sigma has no clip saturation
    on a layout with a multi-day training block; the 3-hour tiny layout is refused."""
    from neural_trade.data.processor import DataProcessor
    from neural_trade.training.trainer import _har_scales

    bundled = Path(__file__).resolve().parents[1] / "binance_btcusdt_1min_ccxt.csv"
    if not bundled.exists():
        pytest.skip("bundled csv not present")
    cfg = tiny_close_only_config
    cfg.CSV_PATH = str(bundled)
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.CONFORMAL_SCALE = "har"
    cfg.MAX_SEQUENCE_COUNT = 600
    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    dp.prepare_datasets(df, close)
    with pytest.raises(ValueError, match="span"):
        _har_scales(cfg, dp, df, close)
    cfg.MAX_SEQUENCE_COUNT = 12000
    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    dp.prepare_datasets(df, close)
    sig_cal, sig_test = _har_scales(cfg, dp, df, close)
    f = dp.fold
    anchors = np.asarray(dp._plan.anchors)[int(dp._lo or 0):]
    np.testing.assert_allclose(close[anchors[f.cal] - 1], dp.cal_block["last_close"])
    for sig, n in ((sig_cal, len(f.cal)), (sig_test, len(f.test))):
        assert set(sig) == {"h0", "h1", "h2"}
        for v in sig.values():
            assert len(v) == n and np.isfinite(v).all() and (v > 0).all()
            assert v.std() / v.mean() > 0.01        # varies: not a clip-saturated constant

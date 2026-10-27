"""DESK-5.6: prediction overlays. One row per bar; direction_signal per horizon; "none" is flagged and carries no
forecast values (D-066); a zero shrink beta reports the served delta as n/a (D-007); the overlay at bar t reads bars up to
t only. The fast tests use a fake Predictor (no TensorFlow); the slow test uses a real tiny bundle."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neural_trade.desk.indicators import predictions
from neural_trade.desk.indicators.predictions import NO_SIGNAL_REASON, overlay, raw_frame, warmup_bars

CFG = SimpleNamespace(LOOKBACK=5, EXTENDED_TREND_PERIODS=[2, 3, 4], RESAMPLE_MINUTES=1)
STEPS = (10, 15, 20)


class FakeBatch:
    """The part of serving.PredictionBatch the overlay reads. delta at a window = 0.001 x its last close."""

    def __init__(self, last_close, signal, with_raw=True):
        self.last_close = last_close
        self.horizon_steps = STEPS
        self.direction_signal = signal
        self.horizons = ("h0", "h1", "h2")
        self.delta_raw = {h: last_close * 0.002 for h in self.horizons} if with_raw else None
        self._cols = {"last_close": last_close}
        for h in self.horizons:
            self._cols[f"{h}_delta"] = last_close * 0.001
            self._cols[f"{h}_price"] = last_close * 1.001
            self._cols[f"{h}_p_up"] = np.full(len(last_close), 0.6)
            self._cols[f"{h}_p_up_calibrated"] = 0.5 + 0.1 * np.tanh(last_close - last_close.mean())
            self._cols[f"{h}_sigma"] = np.full(len(last_close), 3.0)
            self._cols[f"{h}_lo90"] = last_close - 5.0
            self._cols[f"{h}_hi90"] = last_close + 5.0

    def to_frame(self, index=None):
        return pd.DataFrame(self._cols, index=index)


class FakePredictor:
    config = CFG

    def __init__(self, signal=None, beta=None):
        self.signal = signal
        self.bundle = SimpleNamespace(calibration_pipeline=None if beta is None else SimpleNamespace(
            shrink_delta=True, delta_scale=beta))
        self.calls = []

    def predict_windows_frame(self, frame, alpha=0.1, batch_size=None):
        self.calls.append(len(frame))
        df = frame.reset_index(drop=True)
        start = predictions.warmup_bars(CFG)
        anchors = np.arange(start - 1, len(df))
        return FakeBatch(df["close"].to_numpy(float)[anchors], self.signal), df, anchors


def make_bars(n=60, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 1, "low": close - 1, "close": close,
                         "volume": 1.0})


def pairs(overlay_):
    return {h["key"]: h for h in overlay_["horizons"]}


# ---------------------------------------------------------------------------------------------- acceptance 2
def test_one_row_per_bar_and_a_direction_signal_per_horizon():
    bars = make_bars(60)
    out = overlay(FakePredictor({"h0": "ok", "h1": "ok", "h2": "ok"}), bars, 30)
    assert out["n"] == 30 == len(out["t"]) == len(out["last_close"]) == len(out["has_window"])
    assert out["t"][-1] == int(bars["timestamp"].iloc[-1].value // 10**6) and out["bar_ms"] == 60_000
    for h in out["horizons"]:
        assert h["direction_signal"] == "ok" and h["horizon_bars"] in STEPS
        for k in ("target_t", "delta", "price", "delta_raw", "price_raw", "lo90", "hi90", "sigma"):
            assert len(h[k]) == 30, k
        assert len(h["p_up"]["values"]) == 30
        assert h["target_t"][3] == out["t"][3] + h["horizon_bars"] * 60_000
    assert [h["key"] for h in out["horizons"]] == ["h0", "h1", "h2"]


def test_only_the_needed_history_reaches_the_predictor():
    p = FakePredictor()
    overlay(p, make_bars(500), 20)
    assert p.calls == [20 + warmup_bars(CFG)]


def test_bars_without_a_complete_window_are_null_rows():
    out = overlay(FakePredictor(), make_bars(12), 12)      # first anchor is bar 4: rows 0..3 have no window
    assert out["has_window"] == [False] * 4 + [True] * 8
    h = out["horizons"][0]
    assert h["delta"][:4] == [None] * 4 and h["price"][3] is None and h["lo90"][3] is None
    assert h["delta"][4] is not None and h["p_up"]["values"][:4] == [None] * 4


def test_fewer_bars_than_n_gives_fewer_rows_and_the_store_frame_is_accepted():
    store = pd.DataFrame({"ts": np.arange(20) * 60_000 + 1_700_000_000_000, "o": 1.0, "h": 2.0, "l": 0.5,
                          "c": np.linspace(1, 2, 20), "v": 1.0})
    out = overlay(FakePredictor(), store, 100)
    assert out["n"] == 20 and out["t"][0] == 1_700_000_000_000 and out["t"][1] - out["t"][0] == 60_000
    assert list(raw_frame(store).columns[:1]) == ["timestamp"]


def test_non_finite_values_become_null_and_the_payload_is_json_safe():
    class Nan(FakePredictor):
        def predict_windows_frame(self, frame, alpha=0.1, batch_size=None):
            batch, df, anchors = super().predict_windows_frame(frame, alpha, batch_size)
            batch._cols["h0_sigma"] = batch._cols["h0_sigma"].copy()
            batch._cols["h0_sigma"][2] = np.nan
            batch._cols["h0_hi90"] = batch._cols["h0_hi90"].copy()
            batch._cols["h0_hi90"][1] = np.inf
            return batch, df, anchors

    out = overlay(Nan(), make_bars(30), 20)
    assert out["horizons"][0]["sigma"].count(None) == 1 and out["horizons"][0]["hi90"].count(None) == 1
    json.dumps(out, allow_nan=False)


# ---------------------------------------------------------------------------------------------- acceptance 3
def test_direction_signal_none_flags_the_p_up_series_and_carries_no_forecast_values():
    out = overlay(FakePredictor({"h0": "ok", "h1": "none", "h2": "ok"}), make_bars(40), 20)
    h = pairs(out)
    assert h["h1"]["direction_signal"] == "none"
    p = h["h1"]["p_up"]
    assert p["flagged"] is True and p["draw_as_forecast"] is False and p["values"] is None
    assert p["reason"] == NO_SIGNAL_REASON
    assert "none" not in (h["h0"]["direction_signal"], h["h2"]["direction_signal"])
    for ok in ("h0", "h2"):
        assert h[ok]["p_up"]["flagged"] is False and h[ok]["p_up"]["draw_as_forecast"] is True
        assert len(h[ok]["p_up"]["values"]) == 20
    # the whole serialised payload of the flagged horizon holds no P(up) number: the calibrated and the raw value are absent
    flagged = json.dumps(h["h1"]["p_up"])
    assert "0.6" not in flagged and "null" in flagged
    # the price band of that horizon is still shown (it is not the direction head)
    assert h["h1"]["lo90"][-1] is not None and h["h1"]["price"][-1] is not None


def test_without_a_calibration_the_signal_is_na_and_the_raw_head_is_drawn_as_uncalibrated():
    out = overlay(FakePredictor(signal=None), make_bars(40), 10)
    for h in out["horizons"]:
        assert h["direction_signal"] == "n/a"
        assert h["p_up"]["calibrated"] is False and h["p_up"]["flagged"] is False
        assert set(v for v in h["p_up"]["values"] if v is not None) == {0.6}      # the raw head


def test_a_zero_shrink_beta_reports_the_served_delta_as_na_with_the_raw_heads_alongside():
    out = overlay(FakePredictor({"h0": "ok", "h1": "ok", "h2": "ok"}, beta={"h0": 0.0, "h1": 0.5, "h2": 1.0}),
                  make_bars(40), 10)
    h = pairs(out)
    assert h["h0"]["delta_available"] is False and h["h0"]["delta"] is None and h["h0"]["price"] is None
    assert "beta is 0" in h["h0"]["delta_note"]
    assert h["h0"]["delta_raw"][-1] is not None and h["h0"]["price_raw"][-1] is not None
    assert h["h0"]["lo90"][-1] is not None and h["h0"]["sigma"][-1] is not None
    for k in ("h1", "h2"):
        assert h[k]["delta_available"] is True and h[k]["delta"][-1] is not None and h[k]["delta_note"] is None


# ---------------------------------------------------------------------------------------------- causality
def test_the_overlay_at_bar_t_uses_only_bars_up_to_t():
    """The fake's delta at a window is 0.001 x its last close, so alignment and causality are both visible: row i must equal
    the value computed from bar i, and a trap value planted AFTER bar t must not reach any row up to t."""
    bars = make_bars(60)
    base = overlay(FakePredictor(), bars, 40)
    close = bars["close"].to_numpy()[-40:]
    d = np.array(pairs(base)["h0"]["delta"], dtype=float)
    ok = np.array(base["has_window"])
    np.testing.assert_allclose(d[ok], 0.001 * close[ok])            # row i belongs to bar i, no shift
    cut = 45                                                          # bar index inside the frame
    trapped = bars.copy()
    trapped.loc[cut + 1:, "close"] = 1e9                              # the trap marker, strictly after bar `cut`
    out = overlay(FakePredictor(), trapped, 60)
    ref = overlay(FakePredictor(), bars, 60)
    for key in ("delta", "lo90", "hi90"):
        a, b = pairs(out)["h0"][key][: cut + 1], pairs(ref)["h0"][key][: cut + 1]
        assert a == b, key
    assert pairs(out)["h0"]["delta"][cut + 1] != pairs(ref)["h0"]["delta"][cut + 1]      # the trap does show later


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        overlay(FakePredictor(), make_bars(10), 0)


# ---------------------------------------------------------------------------------------------- real tiny bundle
@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory, synthetic_bars):
    """A tiny trained, calibrated bundle, built ONCE (tests/test_predictor_round_trip.py builds it the same way)."""
    import os

    import tensorflow as tf

    from neural_trade.core.config import Config
    from neural_trade.training.trainer import train_and_evaluate

    tmp = tmp_path_factory.mktemp("overlay_bundle")
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        tf.keras.utils.set_random_seed(0)
        synthetic_bars.to_csv(tmp / "bars.csv", index=False)
        cfg = Config(EPOCHS=1, BATCH_SIZE=16, MAX_SEQUENCE_COUNT=1200, PATIENCE=1, EARLY=1,
                     CSV_PATH=str(tmp / "bars.csv"), MODEL_PATH=str(tmp / "w.h5"),
                     SCALER_PATH=str(tmp / "s.joblib"), ARTIFACTS_DIR=str(tmp / "artifacts"))
        result = train_and_evaluate(config=cfg, epochs=1, force=True, calibrate=True, save_artifacts=True)
    finally:
        os.chdir(cwd)
    return result.artifacts_dir


@pytest.mark.slow
def test_a_real_bundle_gives_one_row_per_bar_and_a_causal_overlay(tiny_bundle, synthetic_bars):
    from neural_trade.desk.lab.models import PredictorCache

    cache = PredictorCache(gpu_visible=lambda: False)
    predictor = cache.get(tiny_bundle)
    assert cache.get(tiny_bundle) is predictor
    raw = synthetic_bars.iloc[:400].reset_index(drop=True)
    n = 40
    out = overlay(predictor, raw, n)
    assert out["n"] == n and all(out["has_window"])
    pipeline = predictor.bundle.calibration_pipeline
    expected = pipeline.direction_signal()
    assert len(out["horizons"]) == len(predictor.config.HORIZON_STEPS)
    for h in out["horizons"]:
        assert h["direction_signal"] == expected[h["key"]]
        assert len(h["lo90"]) == len(h["hi90"]) == len(h["sigma"]) == n
        assert h["p_up"]["flagged"] == (expected[h["key"]] == "none")
        assert (h["p_up"]["values"] is None) == (expected[h["key"]] == "none")
        zero = pipeline.shrink_delta and pipeline.delta_scale[h["key"]] == 0.0
        assert h["delta_available"] == (not zero)
    json.dumps(out, allow_nan=False)
    # causality on the real model: bars planted after bar `cut` change nothing up to `cut`
    cut = 300
    trapped = raw.copy()
    trapped.loc[cut + 1:, ["open", "high", "low", "close"]] = 1.0e7
    w = warmup_bars(predictor.config)
    end = cut + 1 + 30
    a = overlay(predictor, trapped.iloc[:end], end - w)
    b = overlay(predictor, raw.iloc[:end], end - w)
    k = cut - w                                  # the overlay row of bar `cut` (rows start at bar `w`)
    for ha, hb in zip(a["horizons"], b["horizons"]):
        for key in ("delta_raw", "lo90", "hi90", "sigma"):
            assert ha[key][: k + 1] == hb[key][: k + 1], (ha["key"], key)
        assert ha["delta_raw"][k + 1] != hb["delta_raw"][k + 1]

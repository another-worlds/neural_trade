"""NT-124 repair round 1 (D-066): the no-signal state reaches the calibration explorer, the frame built from a
TrainResult and the served path; a fit at EITHER search bound is flagged "none"."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neural_trade.calibration import CalibrationPipeline
from neural_trade.calibration.temperature_scaling import T_MAX, T_MIN, TemperatureScaler
from neural_trade.core.config import Config
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.notebook.calibration_ui import CalibrationExplorer, no_signal_note

N = 1500


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _preds(rng, kind):
    """kind[h] in {"ok", "upper", "lower"}: the head's logits against labels drawn to fit that bound."""
    probs, ys = {}, np.zeros((N, 3))
    for i, h in enumerate(HORIZONS):
        if kind[h] == "ok":
            z = rng.normal(0, 1.0, N)
            up = rng.uniform(size=N) < _sig(z)
        elif kind[h] == "upper":          # labels independent of the head: T runs to the upper bound
            z = rng.normal(0, 1.5, N)
            up = rng.uniform(size=N) < 0.5
        else:                              # a near-flat head whose labels are extremely sure: T to the lower bound
            z = rng.normal(0, 0.01, N)
            up = z > 0
        probs[h] = _sig(z)
        ys[:, i] = np.abs(rng.normal(0, 50, N)) * np.where(up, 1.0, -1.0)
    delta = {h: rng.normal(0, 20, N) for h in HORIZONS}
    return {"delta": delta, "direction_prob": probs, "variance": {h: np.full(N, 0.3) for h in HORIZONS}}, ys


def _raw(rng, kind):
    p, y = _preds(rng, kind)
    return SimpleNamespace(y=y, last_close=np.full(N, 100_000.0), delta=p["delta"], direction_prob=p["direction_prob"],
                           variance_scaled=p["variance"], X_raw=None)


def _explorer(kind):
    rng = np.random.default_rng(7)
    cfg = Config()
    blocks = {"config": cfg, "cal_raw": _raw(rng, kind), "test_raw": _raw(np.random.default_rng(8), kind),
              "predictor": SimpleNamespace(bundle=SimpleNamespace(pred_scale=50.0, calibration_pipeline=None)),
              "test": None, "cal": None}
    return CalibrationExplorer(blocks)


def test_explorer_flags_a_horizon_fit_at_the_upper_bound_and_leaves_the_others_alone():
    ex = _explorer({"h0": "ok", "h1": "upper", "h2": "ok"})
    t = ex.refit("none", shrink_delta=False)
    assert ex.pipeline.direction_signal() == {"h0": "ok", "h1": "none", "h2": "ok"}
    assert t.loc["h1", "temperature"] == pytest.approx(T_MAX, rel=1e-2)
    assert t.loc["h1", "direction signal"].startswith("none")
    assert np.isnan(t.loc["h1", "ECE calibrated"]) and np.isnan(t.loc["h1", "ECE calibrated cal (in-sample)"])
    assert np.isfinite(t.loc["h1", "ECE raw"])                     # the raw number stays
    for h in ("h0", "h2"):
        assert t.loc[h, "direction signal"] == "ok"
        assert np.isfinite(t.loc[h, "ECE calibrated"]) and np.isfinite(t.loc[h, "ECE calibrated cal (in-sample)"])
    # the figure note says it in words
    rel, _ = ex.figures("h1")
    note = rel.layout.title.text
    assert "no usable direction signal on h1 (temperature fit at a bound: T = 1000.00)" in note
    assert "refit temperature" not in note
    rel0, _ = ex.figures("h0")
    note0 = rel0.layout.title.text
    assert "refit temperature" in note0 and "no usable direction signal" not in note0


def test_explorer_flags_the_lower_bound_too_and_a_normal_fit_has_no_flag():
    t = _explorer({"h0": "ok", "h1": "lower", "h2": "ok"}).refit("none", shrink_delta=False)
    assert t.loc["h1", "temperature"] == pytest.approx(T_MIN, rel=1e-2)
    assert t.loc["h1", "direction signal"].startswith("none") and np.isnan(t.loc["h1", "ECE calibrated"])
    ok = _explorer({h: "ok" for h in HORIZONS}).refit("none", shrink_delta=False)
    assert (ok["direction signal"] == "ok").all() and ok["ECE calibrated"].notna().all()


def test_no_signal_note_text():
    assert no_signal_note("h2", 1000.0).startswith("no usable direction signal on h2 (temperature fit at a bound: T = 1000.00)")


@pytest.mark.parametrize("status,bound", [("upper_bound", T_MAX), ("lower_bound", T_MIN)])
def test_a_fit_at_either_bound_is_flagged_none(status, bound):
    """D-066: 'a bound is reached' flags the horizon, whichever bound it is (not only the upper one)."""
    sc = TemperatureScaler()
    sc.fit_status["h1"] = status
    assert sc.direction_signal() == {"h0": "ok", "h1": "none", "h2": "ok"}
    rng = np.random.default_rng(2)
    p, y = _preds(rng, {"h0": "ok", "h1": "upper" if status == "upper_bound" else "lower", "h2": "ok"})
    fitted = TemperatureScaler()
    lab = {h: (y[:, i] > 0).astype(float) for i, h in enumerate(HORIZONS)}
    fitted.fit(p["direction_prob"]["h0"], lab["h0"], p["direction_prob"]["h1"], lab["h1"],
               p["direction_prob"]["h2"], lab["h2"])
    assert fitted.fit_status["h1"] == status and fitted.temperatures["h1"] == pytest.approx(bound, rel=1e-2)
    assert fitted.direction_signal()["h1"] == "none"
    assert fitted.direction_signal()["h0"] == fitted.direction_signal()["h2"] == "ok"


def _train_result(signal):
    rng = np.random.default_rng(4)
    p, y = _preds(rng, {h: "ok" for h in HORIZONS})
    pipe = SimpleNamespace(direction_signal=lambda: signal, delta_scale={})
    return SimpleNamespace(
        target_scaler=SimpleNamespace(scale_=[50.0], mean_=[0.0]), predictions=p, y_test=y,
        last_close_test=np.full(N, 100_000.0), predictions_calibrated=None, windows_test=None,
        config=SimpleNamespace(HORIZON_STEPS=(10, 15, 20)), calibration_pipeline=pipe)


def test_frame_from_result_carries_the_signal_flag():
    """Kills the mutation that drops the flag in PredictionFrame.from_result."""
    sig = {"h0": "ok", "h1": "none", "h2": "none"}
    assert PredictionFrame.from_result(_train_result(sig), "test").meta["direction_signal"] == sig
    ok = {h: "ok" for h in HORIZONS}
    assert PredictionFrame.from_result(_train_result(ok), "test").meta["direction_signal"] == ok


def test_served_batch_and_its_frame_carry_the_signal_flag(monkeypatch):
    """Kills the mutation that drops the flag on the served path (Predictor.predict -> PredictionBatch -> frame)."""
    from neural_trade.serving import predictor as pm

    rng = np.random.default_rng(5)
    cfg = Config()
    n = 4
    p, _ = _preds(rng, {h: "ok" for h in HORIZONS})
    preds = {k: {h: np.asarray(v[h])[:n] for h in HORIZONS} for k, v in p.items()}
    sig = {"h0": "ok", "h1": "none", "h2": "ok"}

    class Pipe:
        def apply(self, preds_, alpha=0.1, windows=None):
            return {"direction_prob": preds_["direction_prob"], "delta": preds_["delta"],
                    "intervals": {h: (np.zeros(n), np.ones(n)) for h in HORIZONS}}

        def direction_signal(self):
            return dict(sig)

    bundle = SimpleNamespace(config=cfg, pred_scale=50.0, pred_mean=0.0, calibration_pipeline=Pipe(),
                             normalizer=SimpleNamespace(transform=lambda X, lc: np.asarray(X, "float32")),
                             build_model=lambda: None)
    model = SimpleNamespace(predict=lambda ds, verbose=0: None)
    monkeypatch.setattr(pm, "heads_to_predictions", lambda *a, **k: preds)
    monkeypatch.setattr("neural_trade.utils.seeding.set_arithmetic_rewrite", lambda c: None)
    pr = pm.Predictor(bundle, model=model)
    series = tuple(cfg.INPUT_SERIES or ["close"])
    shape = (n, cfg.LOOKBACK) if len(series) == 1 else (n, cfg.LOOKBACK, len(series))
    X = np.full(shape, 100.0, "float32")
    batch = pr.predict(X)
    assert batch.direction_signal == sig
    assert batch.to_prediction_frame(50.0).meta["direction_signal"] == sig
    assert pr.predict(X, calibrated=False).direction_signal is None


def test_calibration_pipeline_reports_both_bounds_after_a_real_fit():
    rng = np.random.default_rng(9)
    p, y = _preds(rng, {"h0": "upper", "h1": "lower", "h2": "ok"})
    pipe = CalibrationPipeline().fit_from_arrays(p, y, np.full(N, 100_000.0), deadband_bps=0.0, conformal_alpha=0.1)
    assert pipe.direction_signal() == {"h0": "none", "h1": "none", "h2": "ok"}

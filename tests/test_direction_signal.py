"""NT-124 criterion 7 (D-066): a horizon whose temperature fit ends at a bound has no usable direction signal.

pipeline_meta.json records it, the SignalFrame gives that horizon a neutral P(up) with no weight, the
strategies reading P(up) stay flat on it, the report says so; with a normal fit nothing changes.
"""
import copy
import json
import logging

import numpy as np

from neural_trade.calibration.pipeline import CalibrationPipeline
from neural_trade.calibration.temperature_scaling import TemperatureScaler
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy import Bars, SignalFrame, backtest, build_strategy

from tests.test_temperature_scaling import _capture  # noqa: E402

N = 3000


def _frame(seed=0, informative=True):
    rng = np.random.default_rng(seed)
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, N)))
    y = rng.normal(0, 50, (N, 3))
    sig = {h: rng.normal(0, 1.5, N) for h in HORIZONS}
    prob = {h: 1 / (1 + np.exp(-sig[h])) for h in HORIZONS}
    delta = {h: rng.normal(0, 20, N) for h in HORIZONS}
    var = {h: np.full(N, 0.3) for h in HORIZONS}
    fr = PredictionFrame(y, close, delta, prob, var, 50.0, 0.0, (10, 15, 20), "test", copy.deepcopy(prob))
    fr.meta["delta_raw"] = {h: delta[h].copy() for h in HORIZONS}
    return fr, close


def _flatten_h1(fr):
    """The reference-like state: T at the upper bound, so the calibrated P(up) of h1 is ~0.5."""
    rng = np.random.default_rng(3)
    fr.direction_prob_calibrated["h1"] = 0.5 + rng.normal(0, 1e-5, len(fr))
    fr.meta["direction_signal"] = {"h0": "ok", "h1": "none", "h2": "ok"}


def _trades(fr, close, name):
    sig = SignalFrame.build(fr, 1.0)
    strat = build_strategy(name, calibration=sig) if name == "calibrated_quantile" else build_strategy(name)
    return len(backtest(sig, Bars.from_close(close), strat).trades)


def test_normal_fit_changes_nothing():
    fr, _ = _frame()
    a = SignalFrame.build(fr, 1.0)
    fr.meta["direction_signal"] = {h: "ok" for h in HORIZONS}
    b = SignalFrame.build(fr, 1.0)
    for f in ("p", "weighted_direction", "strength", "agreement", "consensus", "avg_confidence"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_h1_at_the_bound_is_neutral_and_weightless():
    fr, _ = _frame()
    _flatten_h1(fr)
    records = _capture("neural_trade.strategy.signals")
    with records:
        s = SignalFrame.build(fr, 1.0)
    assert any(r.levelno == logging.WARNING and "no usable direction signal on ['h1']" in r.getMessage()
               for r in records.items)
    assert np.all(s.p[:, 1] == 0.5)
    # the weighted direction is the confidence-weighted mean of h0 and h2 only
    from neural_trade.strategy.signals import DEFAULT_LAMBDAS
    lam = np.array([DEFAULT_LAMBDAS[h] for h in HORIZONS])
    w = lam * np.exp(-0.3)
    expect = (w[0] * fr.direction_prob_calibrated["h0"] + w[2] * fr.direction_prob_calibrated["h2"]) / (w[0] + w[2])
    np.testing.assert_allclose(s.weighted_direction, expect, atol=1e-12)


def test_no_signal_on_every_horizon_keeps_the_p_up_strategies_flat():
    fr, close = _frame()
    for name in ("calibrated_quantile", "enhanced_multi_horizon", "liberal", "threshold_spike"):
        assert _trades(fr, close, name) > 0, name          # the same frame trades with a usable signal
    for h in HORIZONS:
        fr.direction_prob_calibrated[h] = 0.5 + np.random.default_rng(1).normal(0, 1e-5, N)
    fr.meta["direction_signal"] = {h: "none" for h in HORIZONS}
    for name in ("calibrated_quantile", "enhanced_multi_horizon", "liberal", "threshold_spike"):
        assert _trades(fr, close, name) == 0, name


def test_uncalibrated_readout_is_not_gated():
    fr, _ = _frame()
    _flatten_h1(fr)
    s = SignalFrame.build(fr, 1.0, calibrated=False)
    np.testing.assert_array_equal(s.p[:, 1], fr.direction_prob["h1"])


def test_frame_round_trip_keeps_the_state(tmp_path):
    fr, _ = _frame()
    _flatten_h1(fr)
    path = fr.save_npz(tmp_path / "f.npz")
    assert PredictionFrame.load_npz(path).meta["direction_signal"] == fr.meta["direction_signal"]
    fr2, _ = _frame()
    path2 = fr2.save_npz(tmp_path / "g.npz")
    assert "direction_signal" not in PredictionFrame.load_npz(path2).meta


def test_scaler_and_pipeline_meta_record_the_state(tmp_path):
    sc = TemperatureScaler()
    sc.fit_status["h1"] = "upper_bound"
    assert sc.direction_signal() == {"h0": "ok", "h1": "none", "h2": "ok"}
    pipe = CalibrationPipeline()
    pipe.temperature_scaler = sc
    assert pipe.direction_signal()["h1"] == "none"


def test_pipeline_meta_json_has_direction_signal(tmp_path):
    from tests.test_temperature_scaling import _synthetic
    rng = np.random.default_rng(5)
    preds, yy, lc, _ = _synthetic(rng, 2_000)
    pipe = CalibrationPipeline().fit_from_arrays(preds, yy, lc, deadband_bps=0.0, conformal_alpha=0.1)
    pipe.temperature_scaler.fit_status["h1"] = "upper_bound"
    pipe.save(str(tmp_path / "c"))
    meta = json.loads((tmp_path / "c" / "pipeline_meta.json").read_text())
    assert meta["direction_signal"]["h1"] == "none"
    assert all(v == "ok" for h, v in meta["direction_signal"].items() if h != "h1")


def test_report_says_no_usable_direction_signal():
    from neural_trade.core.config import Config
    from neural_trade.evaluation.report import evaluate
    fr, _ = _frame()
    _flatten_h1(fr)
    rep = evaluate(fr, Config())
    assert rep.meta["direction_signal_none"] == ["h1"]
    md = rep.to_markdown()
    assert "No usable direction signal on h1" in md and "well-calibrated" not in md
    fr2, _ = _frame()
    assert "No usable direction signal" not in evaluate(fr2, Config()).to_markdown()

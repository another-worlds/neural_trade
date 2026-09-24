"""A frame built from a TrainResult carries its raw price heads and betas, so every evaluate() caller scores them."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from neural_trade.core.config import Config
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.evaluation.report import evaluate
from neural_trade.experiments.compare import _flat


def _result(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 200, (n, 3))
    raw = {h: 0.3 * y[:, i] + rng.normal(0, 150, n) for i, h in enumerate(HORIZONS)}
    betas = {"h0": 0.2, "h1": 0.05, "h2": 0.3}
    prob = {h: 1 / (1 + np.exp(-raw[h] / 100)) for h in HORIZONS}
    var = {h: np.ones(n) for h in HORIZONS}
    preds = {"delta": raw, "direction_prob": prob, "variance": var}
    served = {"delta": {h: betas[h] * raw[h] for h in HORIZONS}, "direction_prob": prob, "intervals": {}}
    return SimpleNamespace(
        target_scaler=SimpleNamespace(scale_=[200.0], mean_=[0.0]), predictions=preds, predictions_calibrated=served,
        y_test=y, last_close_test=np.full(n, 110_000.0), config=Config(),
        calibration_pipeline=SimpleNamespace(delta_scale=betas)), raw, betas


def test_from_result_frames_are_scored_on_raw_heads_and_served_deltas():
    result, raw, betas = _result()
    frame = PredictionFrame.from_result(result, "test")
    assert np.allclose(frame.delta["h1"], betas["h1"] * raw["h1"])            # the frame holds what is served
    rep = evaluate(frame, result.config)                                       # no extra arguments
    assert rep.meta["delta_scale"] == betas
    h1 = rep.model["horizons"]["h1"]
    assert math.isclose(h1["delta_raw"]["rmse"], math.sqrt(np.mean((result.y_test[:, 1] - raw["h1"]) ** 2)))
    assert "mag_order_full_raw" in rep.model["coherence"]
    md = rep.to_markdown()
    assert "RMSE ($), raw heads" in md and "shrink beta" in md
    assert "h1/delta_raw/rmse" in _flat(rep.to_dict())                          # compare_runs sees them too

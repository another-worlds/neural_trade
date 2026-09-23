"""Train -> save the artifact bundle -> load a Predictor -> identical predictions (plan B5/B13)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_bars):
    import os

    import tensorflow as tf

    from neural_trade.core.config import Config
    from neural_trade.training.trainer import train_and_evaluate

    tmp = tmp_path_factory.mktemp("rt")
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        tf.keras.utils.set_random_seed(0)
        synthetic_bars.to_csv(tmp / "bars.csv", index=False)
        cfg = Config(EPOCHS=1, BATCH_SIZE=16, MAX_SEQUENCE_COUNT=1200, PATIENCE=1, EARLY=1,
                     CSV_PATH=str(tmp / "bars.csv"), MODEL_PATH=str(tmp / "w.h5"),
                     SCALER_PATH=str(tmp / "s.joblib"), ARTIFACTS_DIR=str(tmp / "artifacts"))
        result = train_and_evaluate(config=cfg, epochs=1, force=True, calibrate=False, save_artifacts=True)
    finally:
        os.chdir(cwd)
    return cfg, result


def _raw_test_windows(cfg, result):
    from neural_trade.data.processor import DataProcessor
    from neural_trade.data.windowing import make_sequences_with_extended_trends

    _, close = DataProcessor(cfg).load_and_prepare_data()
    X = make_sequences_with_extended_trends(cfg, close, cfg.LOOKBACK)[0]
    X = X[-cfg.MAX_SEQUENCE_COUNT:] if len(X) > cfg.MAX_SEQUENCE_COUNT else X
    return X[result.fold.test]


def test_bundle_contents(trained):
    cfg, result = trained
    from pathlib import Path

    d = Path(result.artifacts_dir)
    assert (d / "weights.h5").exists() and (d / "config.yaml").exists() and (d / "meta.json").exists()
    assert (d / "calibration").is_dir()


def test_served_predictions_equal_the_reported_ones(trained):
    from neural_trade.serving.predictor import Predictor

    cfg, result = trained
    p = Predictor.from_artifacts(result.artifacts_dir)
    batch = p.predict(_raw_test_windows(cfg, result), result.last_close_test)
    for kind, served in (("delta", batch.delta), ("direction_prob", batch.direction_prob),
                         ("variance", batch.variance_scaled)):
        for h in ("h0", "h1", "h2"):
            np.testing.assert_array_equal(served[h], result.predictions[kind][h], err_msg=f"{kind}/{h}")
    for h in ("h0", "h1", "h2"):
        np.testing.assert_allclose(batch.direction_prob_calibrated[h],
                                   result.predictions_calibrated["direction_prob"][h], rtol=1e-12)
        lo, hi = batch.interval[h]
        assert np.all(lo < hi)


def test_predict_frame_and_predict_last(trained, synthetic_bars):
    from neural_trade.serving.predictor import Predictor

    cfg, result = trained
    p = Predictor.from_artifacts(result.artifacts_dir)
    frame = p.predict_frame(synthetic_bars)
    assert frame.index[-1] == synthetic_bars["datetime"].iloc[-1]
    assert {"h1_delta", "h1_p_up_calibrated", "h1_lo90", "h1_hi90", "h1_gauss_p_up"} <= set(frame.columns)
    last = p.predict_last(synthetic_bars["close"].to_numpy())
    assert set(last) == {"h0", "h1", "h2"} and last["h1"]["horizon_bars"] == cfg.HORIZON_STEPS[1]
    np.testing.assert_allclose(last["h2"]["delta"], frame["h2_delta"].iloc[-1], rtol=1e-6)

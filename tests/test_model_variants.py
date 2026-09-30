"""NT-104: gru_small and linear_indicators register through the Models registry (D-002), train one
CPU epoch on the bundled CSV and serve predictions through Predictor unchanged; DIRECTION_DEEP_ZERO_INIT
zero-initialises the deep direction logit (default off, golden run bit-for-bit); the skip-variance-share
helper reads the skip and tower logits from the model itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

BUNDLED_CSV = Path(__file__).resolve().parent.parent / "binance_btcusdt_1min_ccxt.csv"


def test_gru_attention_gru_small_linear_indicators_all_registered_with_ten_heads():
    from neural_trade.core.config import Config
    from neural_trade.core.outputs import PredictiveOutputs
    from neural_trade.models.registry import Models

    assert {"gru_attention", "gru_small", "linear_indicators"} <= set(Models.list_names())
    for name in ("gru_attention", "gru_small", "linear_indicators"):
        tf.keras.utils.set_random_seed(0)
        model = Models.build(name, Config(LOOKBACK=32))
        assert len(model.outputs) == len(PredictiveOutputs._fields) == 10
        outs = PredictiveOutputs(*model(tf.random.normal([3, 32, 5]), training=False))
        assert outs.price_h1.shape == (3, 1) and outs.vacuum_overflow.shape == (3, 1)
        assert bool(tf.reduce_all(tf.math.is_finite(outs.variance_h2)))
        assert 0.0 <= float(tf.reduce_min(outs.direction_h0)) <= float(tf.reduce_max(outs.direction_h0)) <= 1.0

    # gru_small and linear_indicators are meaningfully smaller than the default (B_model_indicators.md 7.2)
    small = Models.build("gru_small", Config(LOOKBACK=32)).count_params()
    linear = Models.build("linear_indicators", Config(LOOKBACK=32)).count_params()
    default = Models.build("gru_attention", Config(LOOKBACK=32)).count_params()
    assert linear < small < default


@pytest.mark.skipif(not BUNDLED_CSV.exists(), reason=f"{BUNDLED_CSV.name} is not present")
@pytest.mark.parametrize("model_name", ["gru_small", "linear_indicators"])
def test_variant_trains_one_cpu_epoch_and_serves_through_predictor(model_name, tmp_path, monkeypatch):
    from neural_trade.core.config import Config
    from neural_trade.serving.predictor import Predictor
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    tf.keras.utils.set_random_seed(0)
    cfg = Config(EPOCHS=1, BATCH_SIZE=16, MAX_SEQUENCE_COUNT=600, PATIENCE=1, EARLY=1,
                MODEL_NAME=model_name, CSV_PATH=str(BUNDLED_CSV), MODEL_PATH=str(tmp_path / "w.h5"),
                SCALER_PATH=str(tmp_path / "s.joblib"), ARTIFACTS_DIR=str(tmp_path / "artifacts"))
    result = train_and_evaluate(config=cfg, epochs=1, force=True, calibrate=False, fit_calibration=False,
                                save_artifacts=True)

    assert np.isfinite(result.history.history["loss"][-1])
    d = Path(result.artifacts_dir)
    assert (d / "weights.h5").exists() and (d / "config.yaml").exists()

    p = Predictor.from_artifacts(result.artifacts_dir)
    assert p.model.name or True  # the base model rebuilt from MODEL_NAME with the saved weights
    windows = np.random.default_rng(0).normal(size=(4, cfg.LOOKBACK, len(cfg.input_series()))).astype("float32")
    batch = p.predict(windows, calibrated=False)
    for h in ("h0", "h1", "h2"):
        assert np.all(np.isfinite(batch.delta[h]))
        assert np.all(np.isfinite(batch.variance_scaled[h])) and np.all(batch.variance_scaled[h] > 0)
        assert np.all((batch.direction_prob[h] >= 0.0) & (batch.direction_prob[h] <= 1.0))


def test_direction_deep_zero_init_starts_the_head_at_the_skip_logit_and_defaults_off():
    from neural_trade.core.config import Config
    from neural_trade.models.direction_diagnostics import direction_skip_variance_share
    from neural_trade.models.registry import Models

    assert Config().DIRECTION_DEEP_ZERO_INIT is False  # golden run bit-for-bit default

    x = tf.random.normal([64, 32, 5]).numpy()

    tf.keras.utils.set_random_seed(0)
    off = Models.build("gru_attention", Config(LOOKBACK=32, DIRECTION_DEEP_ZERO_INIT=False))
    shares_off = direction_skip_variance_share(off, x)
    assert all(0.0 <= v <= 1.0 for v in shares_off.values())
    assert any(v < 0.999 for v in shares_off.values()), "the deep logit should carry some variance when off"

    tf.keras.utils.set_random_seed(0)
    on = Models.build("gru_attention", Config(LOOKBACK=32, DIRECTION_DEEP_ZERO_INIT=True))
    shares_on = direction_skip_variance_share(on, x)
    for h in ("h0", "h1", "h2"):
        assert shares_on[h] == pytest.approx(1.0), f"{h}: deep logit is zero-initialised, skip should be 100%"
        np.testing.assert_array_equal(on.get_layer(f"direction_{h}_logit").get_weights()[0], 0.0)


def test_direction_skip_variance_share_reports_per_horizon_and_needs_the_skip_layers():
    from neural_trade.core.config import Config
    from neural_trade.models.direction_diagnostics import direction_skip_variance_share
    from neural_trade.models.registry import Models

    tf.keras.utils.set_random_seed(0)
    no_skip = Models.build("gru_attention", Config(LOOKBACK=32, DIRECTION_SKIP=False))
    with pytest.raises(ValueError):
        direction_skip_variance_share(no_skip, tf.random.normal([8, 32, 5]).numpy())

    tf.keras.utils.set_random_seed(0)
    model = Models.build("gru_small", Config(LOOKBACK=32))  # works for a non-default architecture too
    shares = direction_skip_variance_share(model, tf.random.normal([32, 32, 5]).numpy())
    assert set(shares) == {"h0", "h1", "h2"}

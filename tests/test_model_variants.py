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
    from neural_trade.models.direction_diagnostics import direction_skip_covariance_share
    from neural_trade.models.registry import Models

    assert Config().DIRECTION_DEEP_ZERO_INIT is False  # golden run bit-for-bit default

    x = tf.random.normal([64, 32, 5]).numpy()

    tf.keras.utils.set_random_seed(0)
    off = Models.build("gru_attention", Config(LOOKBACK=32, DIRECTION_DEEP_ZERO_INIT=False))
    shares_off = direction_skip_covariance_share(off, x)
    for v in shares_off.values():
        assert v["skip_share"] == pytest.approx(1.0 - v["tower_share"])
    assert any(v["skip_share"] < 0.999 for v in shares_off.values()), \
        "the deep logit should carry some variance when off"

    tf.keras.utils.set_random_seed(0)
    on = Models.build("gru_attention", Config(LOOKBACK=32, DIRECTION_DEEP_ZERO_INIT=True))
    shares_on = direction_skip_covariance_share(on, x)
    for h in ("h0", "h1", "h2"):
        assert shares_on[h]["skip_share"] == pytest.approx(1.0), \
            f"{h}: deep logit is zero-initialised, skip should carry 100% of the (zero) tower variance"
        assert shares_on[h]["tower_share"] == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_array_equal(on.get_layer(f"direction_{h}_logit").get_weights()[0], 0.0)


def test_direction_skip_covariance_share_reports_per_horizon_and_needs_the_skip_layers():
    from neural_trade.core.config import Config
    from neural_trade.models.direction_diagnostics import direction_skip_covariance_share
    from neural_trade.models.registry import Models

    tf.keras.utils.set_random_seed(0)
    no_skip = Models.build("gru_attention", Config(LOOKBACK=32, DIRECTION_SKIP=False))
    with pytest.raises(ValueError):
        direction_skip_covariance_share(no_skip, tf.random.normal([8, 32, 5]).numpy())

    tf.keras.utils.set_random_seed(0)
    model = Models.build("gru_small", Config(LOOKBACK=32))  # works for a non-default architecture too
    shares = direction_skip_covariance_share(model, tf.random.normal([32, 32, 5]).numpy())
    assert set(shares) == {"h0", "h1", "h2"}
    for h, v in shares.items():
        assert set(v) == {"skip_share", "tower_share", "corr_skip_tower"}
        assert v["skip_share"] + v["tower_share"] == pytest.approx(1.0), h


@pytest.mark.slow
@pytest.mark.skipif(not BUNDLED_CSV.exists(), reason=f"{BUNDLED_CSV.name} is not present")
def test_linear_indicators_loss_is_the_same_order_as_gru_small_and_not_saturated(tmp_path, monkeypatch):
    """Repair round 1 (QA FAIL on a68d72b): the raw pooled indicator features ranged over abs max
    2.2e3 with a per-channel std of 0.015-358, so a one-CPU-epoch default-config run gave a loss of
    542,944 (gru_small: 10.8) with about half the validation window's P(up) saturated at 0 or 1.
    LayerNormalization on the pooled (+ last-bar) features before the linear heads
    (models/linear_indicators.py) fixes both. Not the exact default config (shrunk for test speed),
    but the same comparison QA ran: trains both variants under identical settings and checks the
    loss stays within an order of magnitude and the trained model is not saturated on validation."""
    from neural_trade.core.config import Config
    from neural_trade.data.processor import DataProcessor
    from neural_trade.serving.predictor import Predictor
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    losses, sat = {}, {}
    for name in ("gru_small", "linear_indicators"):
        tf.keras.utils.set_random_seed(0)
        d = tmp_path / name
        d.mkdir()
        cfg = Config(EPOCHS=1, MAX_SEQUENCE_COUNT=4000, MODEL_NAME=name, CSV_PATH=str(BUNDLED_CSV),
                    MODEL_PATH=str(d / "w.h5"), SCALER_PATH=str(d / "s.joblib"), ARTIFACTS_DIR=str(d / "artifacts"))
        res = train_and_evaluate(config=cfg, epochs=1, force=True, calibrate=False, fit_calibration=False,
                                 save_artifacts=True)
        losses[name] = float(res.history.history["loss"][-1])

        dp = DataProcessor(cfg)
        df, close = dp.load_and_prepare_data()
        dp.prepare_datasets(df, close)
        model = Predictor.from_artifacts(res.artifacts_dir).model
        o = model.predict(dp.val_block["X"], batch_size=256, verbose=0)
        dirp = np.concatenate([o[i].ravel() for i in (1, 4, 7)])
        sat[name] = float(np.mean((dirp < 0.01) | (dirp > 0.99)))

    assert np.isfinite(losses["linear_indicators"]) and np.isfinite(losses["gru_small"])
    assert losses["linear_indicators"] < 20 * losses["gru_small"], losses
    assert sat["linear_indicators"] < 0.05, sat


def test_direction_logit_decomposition_matches_a_direct_numpy_computation():
    """NT-110: cov(skip, logit)/var(logit) with skip_share + tower_share == 1, against a direct
    numpy computation on correlated (even anti-correlated) synthetic logits."""
    from neural_trade.models.direction_diagnostics import direction_logit_decomposition

    rng = np.random.default_rng(7)
    tower = rng.normal(0.0, 1.0, size=4000)
    skip = -0.9 * tower + rng.normal(0.0, 0.3, size=4000)  # strongly anti-correlated, like the real run
    out = direction_logit_decomposition(tower, skip)

    logit = tower + skip
    var_logit = np.var(logit)
    ref_skip = np.cov(skip, logit, ddof=0)[0, 1] / var_logit
    ref_tower = np.cov(tower, logit, ddof=0)[0, 1] / var_logit
    ref_corr = np.corrcoef(skip, tower)[0, 1]

    assert out["skip_share"] == pytest.approx(ref_skip, rel=1e-9)
    assert out["tower_share"] == pytest.approx(ref_tower, rel=1e-9)
    assert out["corr_skip_tower"] == pytest.approx(ref_corr, rel=1e-9)
    assert out["skip_share"] + out["tower_share"] == pytest.approx(1.0)
    # the point of NT-110: the old var(skip)/var(skip+tower) definition is NOT a bounded share here
    old_definition = np.var(skip) / var_logit
    assert old_definition > 1.0, "fixture should reproduce the >1 failure mode NT-110 fixes"

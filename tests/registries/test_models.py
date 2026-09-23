"""Models registry: gru_attention builds with the 10 PredictiveOutputs heads; contracts enforced."""
from __future__ import annotations

import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.core.outputs import PredictiveOutputs
from neural_trade.registries.models import Models, ensure_predictive_outputs


def test_default_model_builds_with_ten_named_heads_at_a_small_lookback():
    tf.keras.utils.set_random_seed(0)
    cfg = Config(LOOKBACK=32)
    model = Models.build(None, cfg)
    assert len(model.outputs) == len(PredictiveOutputs._fields) == 10
    outs = PredictiveOutputs(*model(tf.random.normal([3, 32]), training=False))
    assert outs.price_h1.shape == (3, 1) and outs.vacuum_overflow.shape == (3, 1)
    assert float(tf.reduce_min(outs.variance_h2)) > 0.0
    assert 0.0 <= float(tf.reduce_min(outs.direction_h0)) <= float(tf.reduce_max(outs.direction_h0)) <= 1.0


def test_builder_signature_and_output_contract_are_enforced():
    with pytest.raises(ComponentValidationError):
        Models.register(name="no_config_param")(lambda cfg: None)
    inp = tf.keras.Input((8,))
    wrong = tf.keras.Model(inp, [tf.keras.layers.Dense(1)(inp)] * 9)
    with pytest.raises(ComponentValidationError, match="expected 10 outputs"):
        ensure_predictive_outputs(wrong)


def test_price_predictor_facade_uses_the_registry():
    from neural_trade.models.facade import PricePredictor

    tf.keras.utils.set_random_seed(0)
    m = PricePredictor(Config(LOOKBACK=32)).build_model()
    assert len(m.outputs) == 10


def test_direction_skip_adds_a_linear_logit_and_can_be_turned_off():
    import numpy as np

    from neural_trade.models.gru_attention import SKIP_LAGS, _trailing_return_features

    tf.keras.utils.set_random_seed(0)
    plain = Models.build(None, Config(LOOKBACK=32, DIRECTION_SKIP=False))
    names = {layer.name for layer in plain.layers}
    assert "direction_skip_features" not in names and "direction_h1_skip" not in names

    tf.keras.utils.set_random_seed(0)
    skip = Models.build(None, Config(LOOKBACK=32))   # on by default
    names = {layer.name for layer in skip.layers}
    assert {"direction_skip_features", "direction_h0_skip", "direction_h1_logit", "direction_h2_skip"} <= names
    outs = PredictiveOutputs(*skip(tf.random.normal([4, 32]), training=False))
    assert 0.0 <= float(tf.reduce_min(outs.direction_h1)) <= float(tf.reduce_max(outs.direction_h1)) <= 1.0

    x = np.random.default_rng(0).normal(size=(5, 32)).astype(np.float32)
    feats = _trailing_return_features(tf.constant(x)).numpy()
    expect = [x[:, -1] - x[:, -1 - k] for k in SKIP_LAGS] + [x[:, -1] - x[:, 0],
                                                            np.log(np.std(np.diff(x, axis=1), axis=1) + 1e-6)]
    np.testing.assert_allclose(feats, np.stack(expect, 1), rtol=1e-5, atol=1e-5)

    # the skip is the only path it adds: the tower's direction weights exist in both builds
    assert skip.get_layer("direction_h1_skip").kernel.shape == (len(SKIP_LAGS) + 2, 1)

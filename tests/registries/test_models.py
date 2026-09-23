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
    from model import PricePredictor

    tf.keras.utils.set_random_seed(0)
    m = PricePredictor(Config(LOOKBACK=32)).build_model()
    assert len(m.outputs) == 10

"""NT-114: Config.DETERMINISTIC_GRU builds every recurrent layer off the cuDNN-fused kernel
(``unroll=True``, identical maths) so pre-registered comparison studies can get a bit-for-bit
reproducible GPU path (NT-074 GPU check: the cuDNN GRU kernel is the one op TF 2.10's
``enable_op_determinism()`` does not cover). Default stays off (today's cuDNN path, unchanged
numbers and speed, golden run). These tests run on CPU; they pin the layer configuration and the
forward-pass equivalence, not GPU reproducibility itself (that is the GPU check already on record).
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf


@pytest.mark.parametrize("model_name,layer_name,get_cells", [
    ("gru_attention", "gru_attention_backbone",
     lambda layer: [layer.forward_layer, layer.backward_layer]),
    ("gru_small", "gru_small_backbone", lambda layer: [layer]),
])
def test_deterministic_gru_default_off_switch_on_configures_unroll(model_name, layer_name, get_cells):
    from neural_trade.core.config import Config
    from neural_trade.models.registry import Models

    assert Config().DETERMINISTIC_GRU is False  # golden run bit-for-bit default

    off = Models.build(model_name, Config(LOOKBACK=32, DETERMINISTIC_GRU=False))
    for cell in get_cells(off.get_layer(layer_name)):
        assert cell.unroll is False

    on = Models.build(model_name, Config(LOOKBACK=32, DETERMINISTIC_GRU=True))
    for cell in get_cells(on.get_layer(layer_name)):
        assert cell.unroll is True


@pytest.mark.parametrize("model_name", ["gru_attention", "gru_small"])
def test_deterministic_gru_same_weights_same_forward_output_on_cpu(model_name):
    """unroll=True only changes HOW the recurrence is computed (an unrolled Python loop of the
    identical GRU cell instead of the single cuDNN-eligible op), not the maths: given identical
    weights and the same input, the two configurations must agree on every output head."""
    from neural_trade.core.config import Config
    from neural_trade.models.registry import Models

    tf.keras.utils.set_random_seed(0)
    off = Models.build(model_name, Config(LOOKBACK=32, DETERMINISTIC_GRU=False))
    tf.keras.utils.set_random_seed(0)
    on = Models.build(model_name, Config(LOOKBACK=32, DETERMINISTIC_GRU=True))
    on.set_weights(off.get_weights())

    x = tf.random.normal([5, 32, 5], seed=0)
    outs_off = off(x, training=False)
    outs_on = on(x, training=False)
    assert len(outs_off) == len(outs_on) == 10
    for a, b in zip(outs_off, outs_on):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)


def test_deterministic_gru_does_not_change_golden_default():
    """Default False must leave the layer on the cuDNN-eligible configuration (unroll=False) for
    both model variants, matching the pre-NT-114 graph (scripts/golden_run.py verify pins the
    numbers; this pins the config-level mechanism)."""
    from neural_trade.core.config import Config
    from neural_trade.models.registry import Models

    for name, layer_name in [("gru_attention", "gru_attention_backbone"),
                             ("gru_small", "gru_small_backbone")]:
        model = Models.build(name, Config(LOOKBACK=32))
        layer = model.get_layer(layer_name)
        cells = [layer.forward_layer, layer.backward_layer] if hasattr(layer, "forward_layer") else [layer]
        for cell in cells:
            assert cell.unroll is False

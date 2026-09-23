"""Layers registry: the four custom layers are registered, strict, buildable by role."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.registries.layers import Layers


def test_the_four_layers_are_registered_and_the_registry_is_strict():
    assert set(Layers.list_names()) >= {"learnable_indicators", "positional_encoding",
                                        "vacuum_saturation_noise", "energy_gate"}
    assert Layers.strict
    with pytest.raises(ComponentValidationError):
        Layers.register(name="not_a_layer")(lambda **kw: None)


def test_build_by_role_from_config():
    tf.keras.utils.set_random_seed(0)  # TF_DETERMINISTIC_OPS requires seeded random ops
    cfg = Config()
    ind = Layers.for_role(cfg, "indicators", cfg)
    out = ind([tf.random.normal([4, cfg.LOOKBACK]), tf.zeros([4, 18])])
    assert out.shape == (4, cfg.LOOKBACK, 31)
    assert set(Layers.as_custom_objects()) >= {"LearnableIndicators", "EnergyGate"}


def test_energy_gate_matches_the_original_functional_ops():
    """Same math as the inline Keras ops it replaced (GAP/centre/var/max -> Dense softmax -> blend)."""
    rng = np.random.default_rng(0)
    window = tf.constant(rng.normal(0, 1, (5, 60)).astype(np.float32))
    branches = [tf.constant(rng.normal(0, 1, (5, 60, 16)).astype(np.float32)) for _ in range(3)]
    gate = Layers.build("energy_gate", n_branches=3)
    out = gate([window, *branches]).numpy()

    w, b = [v.numpy() for v in gate.gate.weights]
    x = window.numpy()[..., None]
    feats = np.concatenate([np.mean((x - x.mean(1, keepdims=True)) ** 2, 1), x.max(1)], -1)
    logits = feats @ w + b
    g = np.exp(logits - logits.max(-1, keepdims=True))
    g /= g.sum(-1, keepdims=True)
    ref = sum(branches[k].numpy() * g[:, k][:, None, None] for k in range(3))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

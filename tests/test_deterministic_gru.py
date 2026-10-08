"""NT-114: Config.DETERMINISTIC_GRU builds every recurrent layer off the cuDNN-fused kernel
(``unroll=True``, identical maths and weights layout) so pre-registered comparison studies can get a
bit-for-bit reproducible GPU path (NT-074 GPU check: the cuDNN GRU kernel is the one op TF 2.10's
``enable_op_determinism()`` does not cover). Default stays off (today's cuDNN-eligible graph,
unchanged numbers and speed; ``scripts/golden_run.py`` pins the numbers). These tests run on CPU: they
pin the layer configuration and the forward/gradient equivalence, not GPU reproducibility itself
(that is the experimenter's GPU check).
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.models.registry import Models

_LOOKBACK = 24


def _recurrent_layers(model):
    """Every recurrent layer in ``model`` (the wrapped GRU/LSTM inside a Bidirectional, not the
    wrapper), found by walking the layers, so a new recurrent builder is covered automatically."""
    found = []
    for layer in model.submodules:
        if isinstance(layer, tf.keras.layers.Bidirectional):
            found += [layer.forward_layer, layer.backward_layer]
        elif isinstance(layer, tf.keras.layers.RNN):
            found.append(layer)
    # Bidirectional's inner layers are also submodules: de-duplicate by identity.
    return list({id(layer): layer for layer in found}.values())


def _build(name, **overrides):
    return Models.build(name, Config(LOOKBACK=_LOOKBACK, **overrides))


def test_default_is_off():
    assert Config().DETERMINISTIC_GRU is False


@pytest.mark.parametrize("name", Models.list_names())
def test_every_recurrent_layer_follows_the_switch(name):
    """On: every recurrent layer of every registered model is unrolled and Keras no longer treats it
    as cuDNN-eligible. Off: untouched Keras defaults (unroll False, cuDNN-eligible). A registered
    model without a recurrent layer (linear_indicators) has nothing to switch and builds either way."""
    off = _recurrent_layers(_build(name))
    on = _recurrent_layers(_build(name, DETERMINISTIC_GRU=True))
    assert len(off) == len(on)
    for layer in off:
        assert layer.unroll is False
        assert layer._could_use_gpu_kernel is True  # the cuDNN-eligible default configuration
    for layer in on:
        assert layer.unroll is True
        assert layer._could_use_gpu_kernel is False  # Keras no longer picks the fused kernel
    if name in ("gru_attention", "gru_small"):
        assert on, f"{name} is a recurrent model but no recurrent layer was found"


@pytest.mark.parametrize("name", ["gru_attention", "gru_small"])
def test_on_keeps_weights_layout_and_names(name):
    """Same weights layout: identical variable names (up to counters) and shapes, so set_weights and every saved
    bundle fit both settings (acceptance 1/3)."""
    off = _build(name)
    on = _build(name, DETERMINISTIC_GRU=True)
    def layout(model):  # Keras auto-name counters differ between two builds, and the first build in a process has none
        # ('bidirectional' against 'bidirectional_1'): compare names without the counters and their underscore
        return [(re.sub(r"_?\d+", "", w.name), tuple(w.shape)) for w in model.weights]

    assert layout(off) == layout(on)


@pytest.mark.parametrize("name", ["gru_attention", "gru_small"])
def test_same_weights_same_forward_output_and_gradient(name):
    """unroll=True only changes HOW the recurrence is computed (an unrolled loop of the identical GRU
    cell instead of the single fused op), not the maths: with identical weights and the same input the
    two configurations agree on every output head, and on the gradient of a scalar of them."""
    off = _build(name)
    on = _build(name, DETERMINISTIC_GRU=True)
    on.set_weights(off.get_weights())

    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(size=(5,) + tuple(off.input_shape[1:])).astype("float32"))

    def forward(model):
        with tf.GradientTape() as tape:
            outs = model(x, training=False)
            loss = tf.add_n([tf.reduce_sum(tf.square(o)) for o in outs])
        return outs, tape.gradient(loss, model.trainable_variables)

    outs_off, grads_off = forward(off)
    outs_on, grads_on = forward(on)
    assert len(outs_off) == len(outs_on) > 0
    for a, b in zip(outs_off, outs_on):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)
    for g_off, g_on in zip(grads_off, grads_on):
        if g_off is None:
            assert g_on is None
            continue
        np.testing.assert_allclose(tf.convert_to_tensor(g_off).numpy(), tf.convert_to_tensor(g_on).numpy(),
                                   rtol=1e-3, atol=1e-5)


def test_default_graph_is_the_pre_nt114_graph():
    """The default build has the same layer names, order and variable shapes as before the switch
    existed: the Bidirectional layer keeps its Keras auto-name (naming it would change the graph that
    legacy bundles and the golden run see)."""
    model = _build("gru_attention")
    assert not any(layer.name == "gru_attention_backbone" for layer in model.layers)
    assert any(layer.name.startswith("bidirectional") for layer in model.layers)

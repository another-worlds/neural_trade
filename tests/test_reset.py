"""``training/reset.py`` (NT-092, NT-108): resetting a stochastic layer's random-number state between
reused screen-mode trials.

NT-108: :func:`reset_stateful_rngs` used to derive each stochastic layer's seed from its numeric
position in ``model.submodules`` (``enumerate(model.submodules)``). ``tf.Module.submodules`` orders its
traversal by attribute name, so adding an unrelated ``tf.Module`` attribute anywhere on the model (a new
Metric, a new sub-layer, ...) whose name sorts earlier than an existing stochastic layer's owner shifts
that layer's enumerate index and silently changes its derived seed - the NT-037 finding (adding 18
``tf.keras.metrics.Mean`` objects to ``CustomTrainModel`` shifted every later layer's screen-mode
``lambda_t_perp`` from 0.89 to 1.29). The fix derives the seed from the layer's own Keras ``name``
instead, which does not depend on what else is attached to the model.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf  # import neural_trade before tensorflow (CUDA DLLs on PATH) - see conftest

from neural_trade.training.reset import reset_stateful_rngs, seeded_stochastic_layers


def _built_dropout_pair():
    """Two ``Dropout`` layers built under :func:`seeded_stochastic_layers`, each called once so its
    ``_random_generator._generator`` exists (Keras creates it lazily, on first call)."""
    with seeded_stochastic_layers():
        a = tf.keras.layers.Dropout(0.5, name="dropout_a")
        b = tf.keras.layers.Dropout(0.5, name="dropout_b")
    a(tf.zeros([1, 4]), training=True)
    b(tf.zeros([1, 4]), training=True)
    return a, b


def _generator_state(layer) -> np.ndarray:
    return layer._random_generator._generator.state.numpy().copy()


def _hold(layer):
    """Wrap one layer in a bare tf.Module: ``model.submodules`` lists a module's CHILDREN, not the
    module itself, so reset_stateful_rngs(layer, ...) alone would always see 0 children."""
    class Holder(tf.Module):
        pass

    holder = Holder()
    holder.layer = layer
    return holder


def test_seeded_stochastic_layers_gives_dropout_a_resettable_generator():
    a, _ = _built_dropout_pair()
    assert reset_stateful_rngs(_hold(a), seed=1) == 1, "a Dropout built under seeded_stochastic_layers must be resettable"


def test_dropout_built_outside_seeded_stochastic_layers_is_not_touched():
    """The default training path never enables the flag, so its Dropout layers use Keras's legacy
    stateful ops (no ``tf.random.Generator`` at all, module docstring) and reset_stateful_rngs finds
    nothing to reset - exactly the ordinary-training case this module must leave alone. An explicit
    op-level ``seed=`` avoids tripping the op-determinism check (``TF_DETERMINISTIC_OPS=1``,
    tests/conftest.py) without touching the module's own random-number state."""
    plain = tf.keras.layers.Dropout(0.5, name="plain_dropout", seed=1)
    plain(tf.zeros([1, 4]), training=True)
    assert reset_stateful_rngs(_hold(plain), seed=1) == 0


def test_derived_seed_depends_on_layer_identity_not_submodules_enumeration_position():
    """NT-108 acceptance 1: adding an unrelated tf.Module attribute to the model, positioned so it
    would have shifted the OLD position-based enumerate index of the two Dropout layers, leaves every
    derived seed - and so the exact post-reset generator state - unchanged."""

    class Holder(tf.Module):
        pass

    h1 = Holder()
    h1.dropout_a, h1.dropout_b = _built_dropout_pair()
    assert reset_stateful_rngs(h1, seed=7) == 2
    state_a_before = _generator_state(h1.dropout_a)
    state_b_before = _generator_state(h1.dropout_b)

    h2 = Holder()
    h2.dropout_a, h2.dropout_b = _built_dropout_pair()
    # "aaa_unrelated" sorts before "dropout_a"/"dropout_b" alphabetically: under the old,
    # position-based code this shifts both dropout layers' enumerate index in model.submodules and so
    # changes their derived seed. A tf.Variable makes it a real, tracked tf.Module attribute (like the
    # NT-037 Metric objects), not an inert Python attribute submodules() would skip.
    h2.aaa_unrelated = tf.Module()
    h2.aaa_unrelated.v = tf.Variable(0.0, name="unrelated")
    assert reset_stateful_rngs(h2, seed=7) == 2
    state_a_after = _generator_state(h2.dropout_a)
    state_b_after = _generator_state(h2.dropout_b)

    np.testing.assert_array_equal(state_a_before, state_a_after)
    np.testing.assert_array_equal(state_b_before, state_b_after)
    # The two layers still get different streams from each other (never share one seed).
    assert not np.array_equal(state_a_before, state_b_before)


def test_different_seeds_give_different_generator_states():
    a, _ = _built_dropout_pair()
    holder = _hold(a)
    reset_stateful_rngs(holder, seed=1)
    state_1 = _generator_state(a)
    reset_stateful_rngs(holder, seed=2)
    state_2 = _generator_state(a)
    assert not np.array_equal(state_1, state_2)

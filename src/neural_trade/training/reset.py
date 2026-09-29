"""Deterministic per-trial RNG reset for screen phase 2 (NT-092).

Screen phase 2 reuses one persistent ``CustomTrainModel`` (and its compiled ``train_function`` /
``test_function``) across every trial of a structural group, resetting only the WEIGHTS, optimizer
state and continuous hyperparameters between trials (:mod:`neural_trade.experiments.screen`). Model
weights are reset with ``set_weights`` and optimizer state with a zero-assign of every optimizer
variable - neither touches a stochastic layer's (``layers.Dropout``, ``layers.MultiHeadAttention``'s
internal attention dropout) own ``tf.random.Generator``, which is a separate, non-"weight" trackable
that Keras seeds once, at layer construction, from whatever global seed was active then.

Left alone, a reused trial's dropout draws would continue the PREVIOUS trial's stream instead of
starting fresh from the reused trial's own seed, breaking reproducibility with a freshly-built model
of the same seed (NT-092 acceptance 2). :func:`reset_stateful_rngs` resets every such generator found
under a model directly from the trial's seed, and is called from BOTH the reused-graph path AND the
fresh-graph path (:func:`neural_trade.experiments.screen._run_trial_light`) - not from
:class:`neural_trade.training.custom_model.CustomTrainModel` or :mod:`neural_trade.training.trainer`
- so the two paths draw the exact same dropout stream for a given seed, and ordinary training
(``scenario run``, the golden run) is untouched: it never calls this function, so its RNG draws are
whatever Keras's own construction-time seeding already gave them.
"""
from __future__ import annotations

import tensorflow as tf

__all__ = ["reset_stateful_rngs"]


def reset_stateful_rngs(model: tf.keras.Model, seed: int) -> int:
    """Reset every ``tf.random.Generator``-backed stochastic layer under ``model`` from ``seed``.

    Walks ``model.submodules`` (every nested layer, including a ``MultiHeadAttention``'s internal
    dropout layers) in a fixed, deterministic order and, for each with a Keras
    ``_random_generator._generator`` (the stateful RNG behind ``layers.Dropout`` and similar layers
    since TF/Keras made per-layer RNGs stateful), calls ``reset_from_seed`` with a seed derived from
    ``seed`` and the layer's position, so different layers do not share one stream. Returns the
    number of generators reset (0 for a model with no stochastic layers - the common case; nothing
    to do, and nothing wrong, either).
    """
    n = 0
    for i, layer in enumerate(model.submodules):
        holder = getattr(layer, "_random_generator", None)
        generator = getattr(holder, "_generator", None)
        reset_from_seed = getattr(generator, "reset_from_seed", None)
        if callable(reset_from_seed):
            reset_from_seed(int(seed) * 1_000_003 + i)
            n += 1
    return n

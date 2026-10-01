"""Deterministic per-trial RNG reset for screen phase 2 (NT-092).

Screen phase 2 reuses one persistent ``CustomTrainModel`` (and its compiled ``train_function`` /
``test_function``) across every trial of a structural group, resetting only the WEIGHTS, optimizer
state and continuous hyperparameters between trials (:mod:`neural_trade.experiments.screen`). Model
weights are reset with ``set_weights`` and optimizer state with a zero-assign of every optimizer
variable - neither touches a stochastic layer's own random-number state.

**QA repair round 1 finding:** by DEFAULT (``Config.SEEDED_STOCHASTIC_LAYERS = False``, unchanged),
Keras 2.10's ``layers.Dropout`` and ``layers.MultiHeadAttention``'s internal attention dropout use
``rng_type='legacy_stateful'`` - a plain ``tf.nn.dropout`` call backed by TF's LEGACY stateful random
ops, which have NO ``tf.random.Generator`` (or any other Python-visible state) at all: their
randomness lives inside the TF runtime's hidden per-op counter, which only ever advances, and cannot
be reset from Python. :func:`reset_stateful_rngs` alone found 0 such generators on the default model
(round 1 QA, ``state_dump.py``) - the model's own noise layer
(:class:`neural_trade.models.layers.vacuum_saturation_noise.VacuumSaturationNoise`) has the same
problem with its own ``tf.random.normal`` call. Left as they were, a reused trial's dropout/noise
draws continued the PREVIOUS trial's stream instead of starting fresh from the reused trial's own
seed - reused matched fresh only when both were coincidentally noise-free (dropout rate 0, LAMBDA_T_PERP
0 so the noise layer is skip-gated to identity).

**The fix**, gated behind ``Config.SEEDED_STOCHASTIC_LAYERS`` (default ``False`` - the normal training
path, ``scenario run`` and the golden run, never sets it, so their graphs and RNG behaviour are
UNCHANGED bit-for-bit):

- :func:`seeded_stochastic_layers` is a context manager around every ``Models.build(...)`` call
  screen mode makes. Inside it, Keras's global
  ``tf.keras.backend.experimental.enable_tf_random_generator()`` flag is on, so every
  ``layers.Dropout``/``layers.MultiHeadAttention`` LAYER CONSTRUCTED WHILE IT IS ON (this flag is read
  once, at each layer's OWN construction, not at call time - it never touches layers built outside the
  context, i.e. every layer the normal training path builds) gets ``rng_type='stateful'`` instead:
  backed by a real ``tf.random.Generator`` with a ``.reset_from_seed`` method.
- ``VacuumSaturationNoise`` takes an explicit ``seeded=`` constructor argument
  (``models/gru_attention.py`` passes ``Config.SEEDED_STOCHASTIC_LAYERS``) and, when true, draws from
  its OWN ``tf.random.Generator`` (a plain attribute, not tracked as a layer weight - the same pattern
  Keras's Dropout itself uses) instead of the unseeded, unresettable ``tf.random.normal``.
- :func:`reset_stateful_rngs` (extended, round 2) finds and resets BOTH shapes of generator: Keras's
  own ``layer._random_generator._generator`` (Dropout, MultiHeadAttention's internal dropout) and a
  bare ``layer._generator`` (``VacuumSaturationNoise``), covering every stochastic op the model's
  ``train_step``/``test_step`` graph can execute (checked by grepping ``tf.random\\.`` and
  ``GaussianNoise`` across ``src/neural_trade/models/``: the two found above are the only ones).

Screen mode (:mod:`neural_trade.experiments.screen`) forces ``SEEDED_STOCHASTIC_LAYERS = True`` and
wraps every ``Models.build`` call in :func:`seeded_stochastic_layers` for EVERY trial it builds, fresh
or reused, so the two paths draw the identical stochastic-layer stream for a given seed. Called from
neither :class:`neural_trade.training.custom_model.CustomTrainModel` nor
:mod:`neural_trade.training.trainer`, so ordinary training is untouched.

**NT-108 / NT-074:** round 2's :func:`reset_stateful_rngs` derived each layer's seed offset from its
Keras-assigned ``name`` instead of its position in ``model.submodules`` (NT-108: that position shifts
whenever an unrelated ``tf.Module`` attribute is added anywhere on the model). That traded one process-
dependence for another: an unnamed ``Dropout``/``MultiHeadAttention`` is auto-named from a GLOBAL,
process-wide Keras counter, so the SAME architecture built twice in the SAME PROCESS - screen phase 2's
persistent group model versus the throwaway shadow model ``Models.build`` rebuilds every trial for fresh
initial weights, or versus an independent ``_run_trial_light`` call made afterwards for the bit-for-bit
comparison test - gets DIFFERENT auto-generated names, hence a different offset and a different noise
stream for the identical config and seed (NT-074: `tests/test_screen.py`'s reused-vs-fresh test failed,
final_train_loss 7.4939 vs 7.7375). :func:`_identity_offset` now keys on neither: it uses the resettable
generator's POSITION AMONG RESETTABLE GENERATORS ONLY, in :func:`reset_stateful_rngs`'s own traversal
order - a property of the model-building code's attribute structure, unaffected by an unrelated
non-stochastic attribute (NT-108's complaint: never enters the filtered sequence) and unaffected by any
other model built earlier in the same process (NT-074: no global counter involved).
"""
from __future__ import annotations

import contextlib
import hashlib

import tensorflow as tf

__all__ = ["reset_stateful_rngs", "seeded_stochastic_layers"]


@contextlib.contextmanager
def seeded_stochastic_layers():
    """While active, a newly-constructed ``layers.Dropout``/``layers.MultiHeadAttention`` gets a
    resettable ``tf.random.Generator``-backed RNG instead of Keras's legacy stateful ops (see this
    module's docstring). Restores the PREVIOUS global state on exit (not just "off"), so nesting or a
    caller that already enabled it elsewhere is never clobbered."""
    was_enabled = tf.keras.backend.experimental.is_tf_random_generator_enabled()
    tf.keras.backend.experimental.enable_tf_random_generator()
    try:
        yield
    finally:
        if not was_enabled:
            tf.keras.backend.experimental.disable_tf_random_generator()


def _identity_offset(index: int) -> int:
    """A deterministic offset from a resettable generator's POSITION among resettable generators only
    (NT-074), not from the layer's Keras-assigned ``name`` (NT-108's own fix) and not from its raw
    enumeration position in ``model.submodules`` (NT-108's original complaint).

    NT-108 moved off ``model.submodules`` position because an unrelated ``tf.Module`` attribute added
    anywhere on the model shifts every later index. But the replacement - the layer's own ``name`` -
    has the same disease from a different angle: Keras auto-names an unnamed ``Dropout`` /
    ``MultiHeadAttention`` from a GLOBAL, process-wide counter (``dropout``, ``dropout_1``,
    ``dropout_2``, ...), so the identical architecture built a second time in the SAME PROCESS - for
    example screen phase 2's persistent group model, built once, versus a throwaway shadow model
    ``Models.build`` rebuilds on every trial to get fresh initial weights, or an independent
    ``_run_trial_light`` call made afterwards for the bit-for-bit comparison test - gets DIFFERENT
    auto-generated names for the SAME layer, hence a different ``_identity_offset`` and a different
    noise/dropout stream for an identical config and seed (measured: ``tests/test_screen.py``'s reused-
    vs-fresh test, final_train_loss 7.4939 vs 7.7375; reproduced directly by building the same
    ``Config`` three times in one process and printing each resettable layer's name - position 0 reads
    'dropout', 'dropout_3', 'dropout_6' on the three builds).

    ``index`` is this generator's position in the FILTERED sequence of resettable candidates only (built
    by :func:`reset_stateful_rngs`'s own traversal order of ``model.submodules``, which is a property of
    the model-building code's attribute structure, not of a global naming counter or of unrelated
    non-stochastic attributes elsewhere on the model - those never enter the filtered sequence, so they
    cannot shift it). ``hashlib.sha256`` (not the builtin ``hash()``, which is salted per process by
    ``PYTHONHASHSEED``) spreads nearby indices to well-separated 64-bit offsets."""
    digest = hashlib.sha256(f"stochastic_layer#{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def reset_stateful_rngs(model: tf.keras.Model, seed: int) -> int:
    """Reset every resettable stochastic layer under ``model`` from ``seed``.

    Walks ``model.submodules`` (every nested layer, including a ``MultiHeadAttention``'s internal
    dropout layers) and, for each with a Keras ``_random_generator._generator`` (Dropout and similar
    layers, only present when built under :func:`seeded_stochastic_layers`) OR a bare ``_generator``
    attribute that is itself a ``tf.random.Generator`` (``VacuumSaturationNoise``, when
    ``seeded=True``), calls ``reset_from_seed`` with a seed derived from ``seed`` and the generator's
    own POSITION among resettable generators (:func:`_identity_offset`, NT-074), so different layers do
    not share one stream, and neither an unrelated ``tf.Module`` attribute elsewhere on the model
    (NT-108) nor another model build earlier in the same process (NT-074) changes any derived seed.
    Returns the number of generators reset (0 for a model with no stochastic layers, or one built with
    ``SEEDED_STOCHASTIC_LAYERS`` off - the common case for ordinary training; nothing to do, and
    nothing wrong, either)."""
    n = 0
    index = 0
    for layer in model.submodules:
        candidates = (getattr(getattr(layer, "_random_generator", None), "_generator", None),
                     getattr(layer, "_generator", None))
        for generator in candidates:
            reset_from_seed = getattr(generator, "reset_from_seed", None)
            if callable(reset_from_seed):
                offset = _identity_offset(index)
                reset_from_seed((int(seed) * 1_000_003 + offset) % (2**31 - 1))
                n += 1
                index += 1
    return n

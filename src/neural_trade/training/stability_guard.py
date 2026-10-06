"""Fail loudly, naming the loss term (NT-038, D-026).

:class:`StabilityGuard` is a Keras callback the engine's trainer adds in strict mode
(``Config.STRICT_LOSS_MASKS``; ``experiments.runner.train_cell``). At the end of every epoch it reads the
epoch logs' NT-036 counters (``masked_<term>``: steps on which that loss term's value was non-finite;
``nonfinite_grad_steps``: steps the finite-gradient guard zeroed) and, when any fired, stops the run with
:class:`UnstableTrainingError` whose message names the terms. Nothing is added to the per-step path: the
counters already exist and the logs are read once per epoch.

Attribution rules (the message says which one applied):

* a masked term counter fired: those terms are blamed. ``total_loss`` is the sum of the others, so it is
  named only when no other counter fired; the ``head_*`` counters (a head output that was non-finite) name
  the head;
* no term counter fired but a step was non-finite: a gradient was non-finite while every loss value was
  finite (the per-term probe's shares are all NaN then, see ``CustomTrainModel._run_gradient_probe``), so no
  single term can be named from the counters: the message says so.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import tensorflow as tf

#: ``total_loss`` is derived from the other terms; it is blamed only when it is the only one that fired.
DERIVED_TERMS = ("total_loss",)
#: this many loss-term counters firing together (or any head_* counter) points at the inputs, not at one term
INPUT_SUSPECT_TERMS = 5


class UnstableTrainingError(RuntimeError):
    """A training run produced a non-finite loss term, loss or gradient step; ``terms`` are the blamed terms."""

    def __init__(self, message: str, terms: Sequence[str] = (), epoch: Optional[int] = None,
                 nonfinite_steps: float = 0.0):
        super().__init__(message)
        self.terms = tuple(terms)
        self.epoch = epoch
        self.nonfinite_steps = nonfinite_steps


def blame(masked: Mapping[str, float]) -> List[str]:
    """The loss terms to name from the fired ``masked_*`` counters (term name -> steps), most steps first."""
    fired = {t: float(v) for t, v in masked.items() if v and float(v) > 0}
    primary = {t: v for t, v in fired.items() if t not in DERIVED_TERMS}
    chosen = primary or fired
    return [t for t, _ in sorted(chosen.items(), key=lambda kv: (-kv[1], kv[0]))]


def describe(masked: Mapping[str, float], nonfinite_steps: float, epoch: Optional[int]) -> Tuple[str, List[str]]:
    terms = blame(masked)
    fired = {t for t, v in masked.items() if v and float(v) > 0}
    # every term non-finite on the first bad step, or a head output non-finite: the cause is upstream of the losses
    upstream = any(t.startswith("head_") for t in fired) or len(fired - set(DERIVED_TERMS)) >= INPUT_SUSPECT_TERMS
    where = f"epoch {epoch}" if epoch is not None else "training"
    if terms:
        fired = ", ".join(f"{t} ({float(masked[t]):g} step(s))" for t in terms)
        return (f"unstable training in {where}: non-finite loss term(s), blamed: {fired}; "
                f"{nonfinite_steps:g} non-finite step(s)"
                + ("; inputs may be non-finite (a head output or most loss terms are non-finite at once)"
                   if upstream else ""), terms)
    return (f"unstable training in {where}: {nonfinite_steps:g} non-finite step(s) with every loss term finite "
            "(a non-finite gradient or input; no loss term can be blamed from the counters)", [])


def _logged(logs: Mapping[str, Any], prefix: str) -> Dict[str, float]:
    out = {}
    for k, v in (logs or {}).items():
        if k.startswith(prefix):
            try:
                out[k[len(prefix):]] = float(v)
            except (TypeError, ValueError):
                continue
    return out


class StabilityGuard(tf.keras.callbacks.Callback):
    """Raise :class:`UnstableTrainingError` at the end of the first epoch with a non-finite term or step.

    It reads the epoch ``logs``, not the model's counters: CustomTrainModel stashes the train-epoch aggregates
    (``masked_<term>``, ``nonfinite_grad_steps``) right after the last training batch and merges them into the
    logs ahead of every other callback, because the counters themselves are reset before the validation pass
    (and the validation pass counts its own non-finite terms into them)."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        masked = {t: v for t, v in _logged(logs, "masked_").items() if v > 0}
        if not masked:                                    # a non-finite term only in the validation pass
            masked = {t: v for t, v in _logged(logs, "val_masked_").items() if v > 0}
        try:
            nonfinite = float(logs.get("nonfinite_grad_steps") or 0.0)
        except (TypeError, ValueError):
            nonfinite = 0.0
        loss = logs.get("loss")
        bad_loss = loss is not None and not math.isfinite(float(loss))
        if nonfinite > 0 or masked or bad_loss:
            msg, terms = describe(masked, nonfinite, int(epoch) + 1)
            raise UnstableTrainingError(msg, terms, int(epoch) + 1, nonfinite)


__all__ = ["StabilityGuard", "UnstableTrainingError", "blame", "describe"]

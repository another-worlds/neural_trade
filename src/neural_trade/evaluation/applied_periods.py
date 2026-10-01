"""Applied-period statistics (NT-097): p5 / p50 / p95 of the PER-WINDOW applied period of every
learnable indicator, on real evaluation windows, plus a small JSON writer for a run's result/meta
files.

``B_model_indicators.md`` sections 2.2, 4.2 and 7 items 5, 7: ``get_learned_parameters()``
(``models/layers/learnable_indicators.py``) reports only the base period ``2/sigmoid(logit) - 1``.
The per-window ``meta_adjust`` shift moves the period the model actually applies away from that
base value - sometimes outside ``[MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX]`` (measured: 1.6-1.8 bars
below the floor of 2, up to 74.7 above a ceiling of 60, on the dev blocks of six runs). This module
recomputes the applied period from the model's own ``meta_adjust`` tensor
(:meth:`LearnableIndicators.applied_period_samples`) on a batch of real windows, so the numbers
match what the model applied, not just the logged base value.

This module does not wire itself into the trainer or the scorer (``training/trainer.py``,
``evaluation/report.py`` and ``experiments/scorer.py`` are owned by other items, NT-037 in
particular); :func:`write_applied_period_report` is a small, self-contained writer a run's report
step can call once it has a trained model and an evaluation-block window array.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import tensorflow as tf


def _meta_adjust_model(model: tf.keras.Model) -> tf.keras.Model:
    """A sub-model of ``model`` mapping its input(s) to the ``meta_adjust`` tensor.

    Requires the Dense layer named ``'meta_adjust'`` (``models/gru_attention.py``, NT-097).
    """
    return tf.keras.Model(model.inputs, model.get_layer("meta_adjust").output)


def _indicator_layer(model: tf.keras.Model):
    """The model's :class:`LearnableIndicators` layer, the way ``scripts/golden_run.py`` finds it."""
    layer = getattr(model, "_indicator_layer", None)
    if layer is not None:
        return layer
    for lyr in model.layers:
        if getattr(lyr, "name", "").startswith("learnable_indicators"):
            return lyr
    raise ValueError("no learnable_indicators layer found on this model")


def applied_period_samples(model: tf.keras.Model, x, batch_size: int = 2048) -> Dict[str, np.ndarray]:
    """Per-window applied period for every learnable parameter, on evaluation windows ``x``.

    ``x`` is an array shaped like the model's input: ``[N, LOOKBACK]`` (close-only) or
    ``[N, LOOKBACK, len(INPUT_SERIES)]`` (NT-047). Returns
    ``{family.learned_name(i, p): np.ndarray[N]}`` (bars).
    """
    layer = _indicator_layer(model)
    meta_model = _meta_adjust_model(model)
    meta = meta_model.predict(np.asarray(x), batch_size=batch_size, verbose=0)
    return layer.applied_period_samples(tf.constant(meta, dtype=tf.float32))


def applied_period_stats(model: tf.keras.Model, x, batch_size: int = 2048) -> Dict[str, Dict[str, float]]:
    """p5 / p50 / p95 of every learnable parameter's applied period over ``x``'s windows."""
    samples = applied_period_samples(model, x, batch_size=batch_size)
    stats: Dict[str, Dict[str, float]] = {}
    for name, values in samples.items():
        v = np.asarray(values, dtype=np.float64)
        stats[name] = {
            "p5": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
        }
    return stats


def write_applied_period_report(model: tf.keras.Model, x, path: Union[str, Path],
                                batch_size: int = 2048) -> Dict[str, Dict[str, float]]:
    """Compute :func:`applied_period_stats` and write it as JSON to ``path``.

    ``path`` is meant to sit next to a run's other result/meta files (for example
    ``<run_dir>/applied_periods.json``). Returns the stats dict as well, so a caller can fold it
    into an existing JSON structure instead of (or in addition to) writing this file.
    """
    stats = applied_period_stats(model, x, batch_size=batch_size)
    Path(path).write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats

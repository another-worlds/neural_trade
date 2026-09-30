"""Report the DIRECTION_SKIP logit's share of the direction-logit variance (NT-104).

Reads the deep tower logit (``direction_h*_logit``) and the DIRECTION_SKIP linear logit
(``direction_h*_skip``) straight from the built model's own layers as a small sub-model, so the
number reflects exactly what the architecture computes on the given input, with no separate
re-implementation of the skip features to drift out of sync. Works for any model built through the
Models registry whose heads follow the ``gru_attention``/``head_kit`` naming (``gru_attention``,
``gru_small`` and ``linear_indicators`` today), and for both the close-only and the OHLCV
(Config.INPUT_SERIES, NT-047) input modes, since the head names and the model's own ``model.input``
do not depend on the input mode.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf

DEFAULT_HORIZONS = ("h0", "h1", "h2")


def direction_skip_variance_share(model: tf.keras.Model, x, horizons: Optional[Sequence[str]] = None,
                                  batch_size: int = 512) -> Dict[str, float]:
    """``{horizon: Var(skip logit) / Var(skip logit + deep logit)}`` on ``x``.

    ``model`` must have a ``direction_{h}_skip`` layer for every horizon in ``horizons`` (i.e. it
    was built with ``Config.DIRECTION_SKIP = True``); a :class:`ValueError` from
    ``model.get_layer`` otherwise, naming the missing layer. ``x`` is a batch of raw model inputs:
    ``[N, LOOKBACK]`` (close-only) or ``[N, LOOKBACK, len(INPUT_SERIES)]`` (OHLCV), already
    normalised the way the model expects (the same tensor ``model(x)`` would take). A horizon whose
    combined-logit variance is exactly 0 on this batch (a degenerate input) reports ``nan`` rather
    than dividing by zero.
    """
    horizons = tuple(horizons) if horizons is not None else DEFAULT_HORIZONS
    layer_names = []
    for h in horizons:
        deep_name, skip_name = f"direction_{h}_logit", f"direction_{h}_skip"
        model.get_layer(deep_name)  # raises ValueError, naming the layer, if DIRECTION_SKIP is off
        model.get_layer(skip_name)
        layer_names.extend([deep_name, skip_name])

    probe = tf.keras.Model(inputs=model.input, outputs=[model.get_layer(n).output for n in layer_names])
    logits = probe.predict(x, batch_size=batch_size, verbose=0)

    shares: Dict[str, float] = {}
    for i, h in enumerate(horizons):
        deep = np.asarray(logits[2 * i], dtype="float64").reshape(-1)
        skip = np.asarray(logits[2 * i + 1], dtype="float64").reshape(-1)
        combined_var = float(np.var(deep + skip))
        shares[h] = float(np.var(skip) / combined_var) if combined_var > 0 else float("nan")
    return shares


__all__ = ["direction_skip_variance_share", "DEFAULT_HORIZONS"]

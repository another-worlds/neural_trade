"""The DIRECTION_SKIP logit's share of the direction-logit variance (NT-104, NT-110).

Reads the deep tower logit (``direction_h*_logit``) and the DIRECTION_SKIP linear logit
(``direction_h*_skip``) straight from the built model's own layers as a small sub-model, so the
number reflects exactly what the architecture computes on the given input, with no separate
re-implementation of the skip features to drift out of sync. Works for any model built through the
Models registry whose heads follow the ``gru_attention``/``head_kit`` naming (``gru_attention``,
``gru_small`` and ``linear_indicators`` today), and for both the close-only and the OHLCV
(Config.INPUT_SERIES, NT-047) input modes, since the head names and the model's own ``model.input``
do not depend on the input mode.

NT-110: the direction logit is ``logit = tower + skip``, so the only decomposition whose two parts
sum to 1 is each path's share of ``var(logit)`` through its *covariance* with the logit, not its own
variance over the combined variance - the ``var(skip) / var(skip + tower)`` ratio NT-037 shipped is
not a share at all once ``skip`` and ``tower`` are correlated (it exceeded 1 on a real run where they
were anti-correlated, QA of NT-037, 18b3479). This is the one implementation of that number;
``neural_trade.evaluation.report.direction_skip_share`` is a thin wrapper around it for the training
report (unwraps ``CustomTrainModel.base_model``, never raises).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf

DEFAULT_HORIZONS = ("h0", "h1", "h2")


def direction_logit_decomposition(tower_logit, skip_logit) -> Dict[str, float]:
    """The covariance decomposition of ``logit = tower_logit + skip_logit`` into
    ``{"skip_share": cov(skip, logit) / var(logit), "tower_share": cov(tower, logit) / var(logit),
    "corr_skip_tower": corr(skip, tower)}``.

    ``skip_share + tower_share == 1`` identically (``cov(skip, logit) + cov(tower, logit) =
    cov(skip + tower, logit) = var(logit)``), unlike ``var(skip) / var(logit)`` alone, which is not
    bounded by 1 when ``skip`` and ``tower`` are correlated (they were found anti-correlated, corr
    -0.76 to -0.92, on a real run: NT-110). All three are ``nan`` when ``var(logit)`` is 0 on this
    batch (a degenerate input); ``corr_skip_tower`` is ``nan`` when either input is constant.
    """
    tower = np.asarray(tower_logit, dtype=np.float64).reshape(-1)
    skip = np.asarray(skip_logit, dtype=np.float64).reshape(-1)
    logit = tower + skip
    var_logit = float(np.var(logit))
    std_skip, std_tower = float(np.std(skip)), float(np.std(tower))
    if std_skip > 0 and std_tower > 0:
        corr = float(np.cov(skip, tower, ddof=0)[0, 1] / (std_skip * std_tower))
    else:
        corr = float("nan")
    if var_logit <= 0:
        return {"skip_share": float("nan"), "tower_share": float("nan"), "corr_skip_tower": corr}
    skip_share = float(np.cov(skip, logit, ddof=0)[0, 1] / var_logit)
    tower_share = float(np.cov(tower, logit, ddof=0)[0, 1] / var_logit)
    return {"skip_share": skip_share, "tower_share": tower_share, "corr_skip_tower": corr}


def direction_skip_covariance_share(model: tf.keras.Model, x, horizons: Optional[Sequence[str]] = None,
                                    batch_size: int = 512) -> Dict[str, Dict[str, float]]:
    """``{horizon: direction_logit_decomposition(tower_logit, skip_logit)}`` on ``x``.

    ``model`` (the base functional model, e.g. ``CustomTrainModel.base_model`` or a
    ``Models.build(...)`` result) must have a ``direction_{h}_logit`` and ``direction_{h}_skip``
    layer for every horizon in ``horizons`` (i.e. it was built with ``Config.DIRECTION_SKIP =
    True``); a :class:`ValueError` from ``model.get_layer`` otherwise, naming the missing layer.
    ``x`` is a batch of raw model inputs: ``[N, LOOKBACK]`` (close-only) or
    ``[N, LOOKBACK, len(INPUT_SERIES)]`` (OHLCV), already normalised the way the model expects (the
    same tensor ``model(x)`` would take).
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

    return {h: direction_logit_decomposition(logits[2 * i], logits[2 * i + 1]) for i, h in enumerate(horizons)}


__all__ = ["direction_logit_decomposition", "direction_skip_covariance_share", "DEFAULT_HORIZONS"]

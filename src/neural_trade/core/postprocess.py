"""Model heads -> the predictions dict (shared by training evaluation and serving).

Moved here from ``neural_trade.serving.postprocess`` in NT-027 (the layering fix): both
``training/trainer.py`` and ``serving/predictor.py`` need it, and it needs nothing back from
either, so it belongs below both, in ``core/`` (``neural_trade.serving.postprocess`` re-exports
everything below for its old callers).

``{"delta": {h: raw $ deltas}, "direction_prob": {h: P(up)}, "variance": {h: scaled var}}``.
Price heads are inverse-transformed exactly as ``StandardScaler.inverse_transform`` does it
(in-place float32 arithmetic against the float64 scale and mean), so a served prediction is
bit-identical to the one training reported. Direction and variance heads are sanitised
(NaN/Inf -> neutral) and clipped to [0, 1] and [VAR_FLOOR, VAR_CAP].
"""
from __future__ import annotations

import numpy as np

from neural_trade.core.horizons import REFERENCE_HORIZONS, horizon_keys, n_from_outputs

#: The reference three horizon keys (kept for old importers; a run's own keys come from its heads).
HORIZONS = REFERENCE_HORIZONS


def inverse_scale(scaled, pred_scale: float, pred_mean: float) -> np.ndarray:
    x = np.array(np.asarray(scaled).reshape(-1, 1), dtype=np.asarray(scaled).dtype
                 if np.asarray(scaled).dtype in (np.float32, np.float64) else np.float64, copy=True)
    x *= np.array([pred_scale], dtype=np.float64)
    x += np.array([pred_mean], dtype=np.float64)
    return x.ravel()


def sanitize_prob(head, n: int) -> np.ndarray:
    p = np.asarray(head, dtype=float).reshape(-1)[:n]
    return np.nan_to_num(p, nan=0.5, posinf=0.5, neginf=0.5).clip(0.0, 1.0)


def sanitize_var(head, n: int, var_floor: float, var_cap: float) -> np.ndarray:
    v = np.asarray(head, dtype=float).reshape(-1)[:n]
    return np.nan_to_num(v, nan=1.0, posinf=1.0, neginf=1.0).clip(float(var_floor), float(var_cap))


def heads_to_predictions(heads, n: int, pred_scale: float, pred_mean: float, config) -> dict:
    """``heads`` is the model's 3N + 1 output list (price, direction, variance per horizon, then
    the overflow; NT-042); ``n`` trims batch padding. The dicts are keyed h0 .. h{N-1}."""
    heads = list(heads)
    keys = horizon_keys(n_from_outputs(len(heads)))
    n = int(n)

    def delta(head):
        return inverse_scale(np.asarray(head).reshape(-1)[:n], pred_scale, pred_mean)

    vf, vc = float(config.VAR_FLOOR), float(config.VAR_CAP)
    return {
        "delta": {h: delta(heads[3 * i]) for i, h in enumerate(keys)},
        "direction_prob": {h: sanitize_prob(heads[3 * i + 1], n) for i, h in enumerate(keys)},
        "variance": {h: sanitize_var(heads[3 * i + 2], n, vf, vc) for i, h in enumerate(keys)},
    }

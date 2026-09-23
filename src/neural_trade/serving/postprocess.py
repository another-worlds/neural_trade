"""Model heads -> the predictions dict (shared by training evaluation and serving).

``{"delta": {h: raw $ deltas}, "direction_prob": {h: P(up)}, "variance": {h: scaled var}}``.
Price heads are inverse-transformed exactly as ``StandardScaler.inverse_transform`` does it
(in-place float32 arithmetic against the float64 scale and mean), so a served prediction is
bit-identical to the one training reported. Direction and variance heads are sanitised
(NaN/Inf -> neutral) and clipped to [0, 1] and [VAR_FLOOR, VAR_CAP].
"""
from __future__ import annotations

import numpy as np

from neural_trade.core.outputs import PredictiveOutputs

HORIZONS = ("h0", "h1", "h2")


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
    """``heads`` is the model's 10-output list; ``n`` trims batch padding."""
    heads = PredictiveOutputs(*heads)
    n = int(n)

    def delta(head):
        return inverse_scale(np.asarray(head).reshape(-1)[:n], pred_scale, pred_mean)

    vf, vc = float(config.VAR_FLOOR), float(config.VAR_CAP)
    return {
        "delta": {"h0": delta(heads.price_h0), "h1": delta(heads.price_h1), "h2": delta(heads.price_h2)},
        "direction_prob": {"h0": sanitize_prob(heads.direction_h0, n), "h1": sanitize_prob(heads.direction_h1, n),
                           "h2": sanitize_prob(heads.direction_h2, n)},
        "variance": {"h0": sanitize_var(heads.variance_h0, n, vf, vc),
                     "h1": sanitize_var(heads.variance_h1, n, vf, vc),
                     "h2": sanitize_var(heads.variance_h2, n, vf, vc)},
    }

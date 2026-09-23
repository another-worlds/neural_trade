"""Direction labels and the Gaussian direction readout, in NumPy (no TensorFlow).

The rule, shared by every path (training loss, step metrics, evaluation, calibration):
a move whose |return| = |delta / last_close| is at most the deadband is NEUTRAL and
masked out; otherwise it is "up" when return > deadband. The TF twin is
:func:`neural_trade.metrics.tf_direction.direction_labels_tf`.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def direction_labels_np(
    y_true_delta_raw: np.ndarray,
    last_close: np.ndarray,
    deadband_bps: float = 0.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """NumPy equivalent of the direction labeling logic (with deadband).

    Centralizes the previously duplicated "compute returns, apply deadband mask,
    produce binary UP labels" code that lived in model.py (_compute_all_horizon_metrics,
    train/test_step), registries/losses.py:custom_loss, and calibration/pipeline.py.

    Returns a dict mapping 'h0'/'h1'/'h2' to (labels, mask) numpy arrays.
    Matches the shape/semantics expected by CalibrationPipeline and the py metrics path.
    """
    lc = np.asarray(last_close, dtype=float).reshape(-1)
    y = np.asarray(y_true_delta_raw, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    deadband = deadband_bps / 10_000.0
    result = {}
    horizons = ("h0", "h1", "h2")
    for i, h in enumerate(horizons):
        if i >= y.shape[1]:
            break
        ret = y[:, i] / (lc + 1e-12)
        mask = (np.abs(ret) > deadband)
        labels = (ret > deadband).astype(float)
        result[h] = (labels, mask)
    return result


# Name used by the calibration package and model.py before Phase B5.
compute_direction_labels_np = direction_labels_np


def gaussian_up_prob_given_move_np(delta_pred_raw, var_scaled, last_close, deadband_bps, pred_scale):
    """NumPy twin of tf_direction.gaussian_up_prob_given_move for raw-dollar predictions.

    P(delta > d | |delta| > d) for delta ~ N(mu, sigma^2), mu = delta_pred_raw,
    sigma = sqrt(var_scaled) * pred_scale, d = deadband * last_close. Stable in the tails
    (evaluated as sigmoid(log Phi(a) - log Phi(b))).
    """
    from scipy.special import expit, log_ndtr

    mu = np.asarray(delta_pred_raw, dtype=float)
    sigma = np.sqrt(np.maximum(np.asarray(var_scaled, dtype=float), 0.0)) * float(pred_scale) + 1e-8
    d = float(deadband_bps) / 1e4 * np.asarray(last_close, dtype=float)
    return expit(log_ndtr((mu - d) / sigma) - log_ndtr((-mu - d) / sigma))

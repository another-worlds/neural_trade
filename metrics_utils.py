"""Utility metrics for delta/price forecasting.

This module intentionally avoids deleting or replacing existing evaluation paths.
It provides stable alternatives to vanilla MAPE when targets can be near zero
(e.g., minute-level deltas).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _to_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def apply_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    if mask is None:
        n = min(len(y_true), len(y_pred))
        return y_true[:n], y_pred[:n]

    mask = np.asarray(mask).reshape(-1).astype(bool)
    n = min(len(y_true), len(y_pred), len(mask))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = mask[:n]
    return y_true[mask], y_pred[mask]


def safe_mape(
    y_true,
    y_pred,
    *,
    eps: float = 1e-8,
    min_abs_y: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    """MAPE that won’t explode on near-zero targets.

    Returns a fraction (e.g., 0.02 for 2%).

    - If `min_abs_y` is provided, only samples where |y_true| >= min_abs_y contribute.
    - If a `mask` is provided, it is applied first.
    """

    y_t, y_p = apply_mask(y_true, y_pred, mask)
    if y_t.size == 0:
        return float("nan")

    denom = np.abs(y_t)
    if min_abs_y is not None:
        keep = denom >= float(min_abs_y)
        if not np.any(keep):
            return float("nan")
        y_t = y_t[keep]
        y_p = y_p[keep]
        denom = denom[keep]

    denom = np.maximum(denom, float(eps))
    return float(np.mean(np.abs(y_t - y_p) / denom))


def smape(y_true, y_pred, *, eps: float = 1e-8, mask: Optional[np.ndarray] = None) -> float:
    """Symmetric MAPE.

    Returns a fraction in [0, 2] (typical). Multiply by 100 for percent.
    """

    y_t, y_p = apply_mask(y_true, y_pred, mask)
    if y_t.size == 0:
        return float("nan")

    denom = np.abs(y_t) + np.abs(y_p)
    denom = np.maximum(denom, float(eps))
    return float(np.mean(2.0 * np.abs(y_t - y_p) / denom))


def wape(y_true, y_pred, *, eps: float = 1e-8, mask: Optional[np.ndarray] = None) -> float:
    """Weighted absolute percentage error.

    Stable even when many y_true values are near zero.
    Returns a fraction. Multiply by 100 for percent.
    """

    y_t, y_p = apply_mask(y_true, y_pred, mask)
    if y_t.size == 0:
        return float("nan")

    denom = np.sum(np.abs(y_t))
    denom = max(float(denom), float(eps))
    return float(np.sum(np.abs(y_t - y_p)) / denom)


def reconstruct_prices(last_close, delta) -> np.ndarray:
    """Convert delta predictions/targets back into future price levels."""

    lc = _to_1d(last_close)
    d = _to_1d(delta)
    n = min(len(lc), len(d))
    return lc[:n] + d[:n]


@dataclass(frozen=True)
class Coverage:
    used: int
    total: int

    @property
    def frac(self) -> float:
        return float(self.used / self.total) if self.total else float("nan")


def mask_by_min_abs_y(y_true, *, min_abs_y: float) -> Tuple[np.ndarray, Coverage]:
    y_t = _to_1d(y_true)
    keep = np.abs(y_t) >= float(min_abs_y)
    used = int(np.sum(keep))
    total = int(len(y_t))
    return keep, Coverage(used=used, total=total)


def pit_uniformity(
    y_true,
    mu,
    sigma,
    *,
    eps: float = 1e-8,
) -> float:
    """Kolmogorov-Smirnov statistic between PIT values and Uniform[0,1].

    For a well-calibrated Gaussian predictive distribution N(mu, sigma^2),
    the probability integral transform (PIT) u_i = Phi((y_i - mu_i) / sigma_i)
    should be Uniform[0,1].  The KS statistic measures the maximum absolute
    deviation between the empirical CDF of u and the ideal diagonal.

    Returns a value in [0, 1]; 0 = perfectly calibrated, 1 = maximally off.
    Smaller is better.
    """
    from scipy.special import ndtr  # normal CDF

    y = np.asarray(y_true, dtype=float).reshape(-1)
    m = np.asarray(mu, dtype=float).reshape(-1)
    s = np.asarray(sigma, dtype=float).reshape(-1)
    n = min(len(y), len(m), len(s))
    y, m, s = y[:n], m[:n], s[:n]

    s = np.maximum(s, float(eps))
    u = ndtr((y - m) / s)              # PIT values in [0, 1]
    u_sorted = np.sort(u)
    n_f = float(n)
    # Empirical CDF at each sorted point
    ecdf = np.arange(1, n + 1) / n_f
    # KS statistic: max deviation from uniform diagonal
    ks = float(np.max(np.abs(ecdf - u_sorted)))
    return ks

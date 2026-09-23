"""Utility metrics for delta/price forecasting (moved from the root metrics_utils.py in B5).

This module intentionally avoids deleting or replacing existing evaluation paths.
It provides stable alternatives to vanilla MAPE when targets can be near zero
(e.g., minute-level deltas).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# The labelling rule lives in metrics.direction_labels; re-exported for old callers.
from neural_trade.metrics.direction_labels import compute_direction_labels_np, direction_labels_np  # noqa: F401


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


# =============================================================================== numpy metric tier
# Signature contract (Metrics registry): f(y_true, y_pred, *, mask=None, ...) -> float.
# Regression metrics take raw deltas; direction metrics take 0/1 labels and P(up); interval
# metrics take y_pred as an (n, 2) array of [lower, upper].


def _masked(y_true, y_pred, mask):
    yt = _to_1d(y_true)
    yp = np.asarray(y_pred, dtype=float)
    if yp.ndim == 1 or yp.shape[-1] != 2:
        yp = yp.reshape(-1)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]
    if mask is not None:
        mk = np.asarray(mask).reshape(-1).astype(bool)[:n]
        yt, yp = yt[mk], yp[mk]
    return yt, yp


def mse(y_true, y_pred, *, mask=None) -> float:
    """Mean squared error."""
    yt, yp = _masked(y_true, y_pred, mask)
    return float(np.mean((yt - yp) ** 2)) if len(yt) else float("nan")


def rmse(y_true, y_pred, *, mask=None) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred, mask=mask)))


def mae(y_true, y_pred, *, mask=None) -> float:
    """Mean absolute error."""
    yt, yp = _masked(y_true, y_pred, mask)
    return float(np.mean(np.abs(yt - yp))) if len(yt) else float("nan")


def explained_variance(y_true, y_pred, *, mask=None) -> float:
    """1 - Var(y - yhat) / Var(y) (sklearn semantics; negative when worse than the mean)."""
    from sklearn.metrics import explained_variance_score

    yt, yp = _masked(y_true, y_pred, mask)
    return float(explained_variance_score(yt, yp)) if len(yt) > 1 else float("nan")


def corr(y_true, y_pred, *, mask=None) -> float:
    """Pearson correlation; 0.0 when either side is constant."""
    yt, yp = _masked(y_true, y_pred, mask)
    if len(yt) < 2 or yt.std() == 0 or yp.std() == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def r2(y_true, y_pred, *, mask=None) -> float:
    """Coefficient of determination."""
    from sklearn.metrics import r2_score

    yt, yp = _masked(y_true, y_pred, mask)
    return float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")


def _confusion(y_true, y_pred, mask, threshold):
    t, p = _masked(y_true, y_pred, mask)
    t = t > 0.5
    b = p > threshold
    tp, tn = float(np.sum(b & t)), float(np.sum(~b & ~t))
    fp, fn = float(np.sum(b & ~t)), float(np.sum(~b & t))
    return tp, tn, fp, fn


def direction_accuracy(y_true, y_pred, *, mask=None, threshold=0.5) -> float:
    """Accuracy of P(up) > threshold against 0/1 labels."""
    tp, tn, fp, fn = _confusion(y_true, y_pred, mask, threshold)
    n = tp + tn + fp + fn
    return (tp + tn) / n if n else float("nan")


def direction_f1(y_true, y_pred, *, mask=None, threshold=0.5) -> float:
    """F1 of the UP class."""
    tp, tn, fp, fn = _confusion(y_true, y_pred, mask, threshold)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def mcc(y_true, y_pred, *, mask=None, threshold=0.5) -> float:
    """Matthews correlation coefficient (0.0 when a margin is empty)."""
    tp, tn, fp, fn = _confusion(y_true, y_pred, mask, threshold)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return float((tp * tn - fp * fn) / np.sqrt(denom)) if denom > 0 else 0.0


def brier(y_true, y_pred, *, mask=None) -> float:
    """Brier score of P(up) against 0/1 labels."""
    t, p = _masked(y_true, y_pred, mask)
    return float(np.mean((p - t) ** 2)) if len(t) else float("nan")


def ece_pos(y_true, y_pred, *, mask=None, n_bins=10) -> float:
    """Positive-class expected calibration error: |observed up-rate - mean P(up)| per bin."""
    t, p = _masked(y_true, y_pred, mask)
    if not len(t):
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    idx = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(t[sel].mean() - p[sel].mean())
    return float(ece)


def pit_ks(y_true, y_pred, *, variance, mask=None) -> float:
    """KS distance between the PIT values Phi((y - mu) / sigma) and U[0, 1] (exact)."""
    from scipy.special import ndtr

    y, mu = _masked(y_true, y_pred, mask)
    var = _to_1d(variance)[: len(_to_1d(y_true))]
    if mask is not None:
        var = var[np.asarray(mask).reshape(-1).astype(bool)[: len(var)]]
    if not len(y):
        return float("nan")
    u = np.sort(ndtr((y - mu) / np.sqrt(np.maximum(var, 1e-12))))
    n = len(u)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def coverage(y_true, y_pred, *, mask=None) -> float:
    """Share of y_true inside [lower, upper]; y_pred is an (n, 2) array or a (lower, upper) pair."""
    yp = np.asarray(y_pred, dtype=float)
    if yp.ndim == 2 and yp.shape[0] == 2 and yp.shape[1] != 2:
        yp = yp.T
    yt = _to_1d(y_true)
    n = min(len(yt), len(yp))
    inside = (yt[:n] >= yp[:n, 0]) & (yt[:n] <= yp[:n, 1])
    if mask is not None:
        inside = inside[np.asarray(mask).reshape(-1).astype(bool)[:n]]
    return float(np.mean(inside)) if len(inside) else float("nan")


REGRESSION_METRICS = ("mse", "rmse", "mae", "explained_variance", "corr", "r2", "safe_mape", "smape", "wape")
DIRECTION_METRICS = ("direction_accuracy", "direction_f1", "mcc", "brier", "ece_pos")
DISTRIBUTION_METRICS = ("pit_ks", "coverage")


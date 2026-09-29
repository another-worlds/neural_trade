"""The shared metrics and statistics module (NT-027): one AUC, one DeLong variance, one block
bootstrap, one long-run variance, one set of effective-sample helpers.

Before this module, the same statistics were computed independently in several places (survey of
2026-09-28): AUC in ``evaluation/report.py``, ``visualization/analytics_common.py`` (``roc_curve``)
and ``visualization/analytics_direction.py`` (``roc_points``, the DeLong placements); the
effective-sample helpers (``n_eff``, Wilson, the AUC CI) in ``visualization/stats.py``; the paired
block bootstrap and the Diebold-Mariano long-run variance in ``evaluation/report.py``. Every one of
those moved here verbatim (the numbers are unchanged - see ``scripts/golden_run.py`` and the
figure/report tests); their old modules re-export what they used to define, so no caller's import
path breaks. ``evaluation/report.py`` and the figure modules (``visualization/analytics_common.py``,
``visualization/analytics_direction.py``) now call this module instead of computing their own.

NT-015 later moves the remaining overlap-aware interval helpers (HAC reliability bands, the PIT
band per bin) that still live in individual figure modules into this one.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

Z95 = 1.959964

# ------------------------------------------------------------------------- effective-sample helpers
# Consecutive 1-minute samples are not independent: the target of the sample at bar t and of the
# sample at bar t+1 share ``h - 1`` of their ``h`` bars (h = the horizon in bars). Treating the N
# samples as independent makes every interval about sqrt(h) times too narrow; every helper below
# counts ``N / steps`` effective samples instead.


def n_eff(n, steps: int = 1) -> float:
    """Effective number of independent samples among ``n`` samples whose targets span ``steps`` bars."""
    return max(1.0, float(n) / max(1, int(steps)))


def wilson(k, n, *, steps: int = 1, z: float = Z95):
    """(p, lo, hi): the observed proportion k/n and its Wilson interval on n / steps effective samples."""
    k = np.asarray(k, float)
    n = np.maximum(np.asarray(n, float), 1.0)
    p = k / n
    ne = np.maximum(n / max(1, int(steps)), 1.0)
    den = 1 + z * z / ne
    centre = (p + z * z / (2 * ne)) / den
    half = z * np.sqrt(p * (1 - p) / ne + z * z / (4 * ne * ne)) / den
    return p, centre - half, centre + half


def mean_ci(y, *, steps: int = 1, z: float = Z95) -> Tuple[float, float, float]:
    """(mean, lo, hi) of ``y`` with the standard error on len(y) / steps effective samples."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        m = float(y.mean()) if len(y) else np.nan
        return m, np.nan, np.nan
    se = y.std(ddof=1) / np.sqrt(n_eff(len(y), steps))
    m = float(y.mean())
    return m, m - z * se, m + z * se


def corr_null(n, *, steps: int = 1, z: float = Z95) -> float:
    """Half-width, on the Fisher-z scale (atanh r), of the band a sample correlation stays inside by chance
    when there is no relationship: z / sqrt(n_eff - 3).

    This is NOT a threshold for r itself: for 19 samples it is 0.49 where the r-scale value is 0.45, and
    for 7 samples 0.98 where it is 0.75. Compare an r, a Spearman rho or an MCC with :func:`corr_null_r`.
    Use this value only on the z scale, e.g. times (1 - r^2) for the delta-method interval of an r."""
    return float(z / np.sqrt(max(n_eff(n, steps) - 3.0, 1.0)))


def corr_null_r(n, *, steps: int = 1, z: float = Z95) -> float:
    """95% no-relation half-width on the r scale: the band a sample correlation r (Pearson, Spearman rho,
    or an MCC) stays inside by chance, on n / steps effective samples.

    The Fisher-z half-width :func:`corr_null` back-transformed with tanh, so it is always below 1: 0.45
    for 19 samples (exact t test 0.456), 0.63 for 10 (0.632), 0.75 for 7 (0.754)."""
    return float(np.tanh(corr_null(n, steps=steps, z=z)))


def auc_ci(auc: float, n_pos: int, n_neg: int, *, steps: int = 1, z: float = Z95) -> Tuple[float, float]:
    """Hanley-McNeil 95% interval for an AUC, on effective class counts."""
    a = float(auc)
    npos, nneg = n_eff(n_pos, steps), n_eff(n_neg, steps)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg)
    se = float(np.sqrt(max(var, 0.0)))
    return a - z * se, a + z * se


def thin(n: int, max_points: int = 1500) -> np.ndarray:
    """Evenly spaced indices keeping at most ``max_points`` of ``n`` (first and last always kept)."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def horizon_steps(frame_or_config, h: str) -> int:
    """Bars ahead for horizon ``h`` from a PredictionFrame (``horizon_steps``) or a Config (``HORIZON_STEPS``)."""
    steps = getattr(frame_or_config, "horizon_steps", None) or getattr(frame_or_config, "HORIZON_STEPS", None)
    i = ("h0", "h1", "h2").index(h)
    return int(steps[i]) if steps is not None and i < len(steps) else 1


# --------------------------------------------------------------------------------------------- AUC
def auc_score(labels, scores) -> float:
    """ROC AUC (NaN if a class is missing or fewer than 2 labelled samples): the one AUC implementation
    in src/neural_trade (the frozen D-023 scripts are exempt; see tests/test_layering.py)."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if len(labels) < 2 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def roc_curve(labels, scores, max_points: int = 400):
    """(fpr, tpr, auc) with ties handled; thinned to ``max_points`` for plotting."""
    labels = np.asarray(labels, float)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    P, N = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    tpr, fpr = np.r_[0, tps / P], np.r_[0, fps / N]
    auc = float(np.trapz(tpr, fpr))
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr, auc


_trapz = getattr(np, "trapezoid", None) or np.trapz


def roc_points(labels, scores, max_points: int = 400):
    """(fpr, tpr, threshold, auc): the ROC with ties handled, thinned to ``max_points`` for plotting.

    ``threshold[k]`` is the score at or above which a sample is called up at point k (inf at the origin).
    The AUC is computed on the full curve before thinning (equal to the Mann-Whitney AUC); it is NaN when
    one class is missing.
    """
    labels = np.asarray(labels, float)
    scores = np.asarray(scores, float)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.zeros(1), np.zeros(1), np.full(1, np.inf), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    tpr, fpr = np.r_[0.0, tps / n_pos], np.r_[0.0, fps / n_neg]
    thr = np.r_[np.inf, s[distinct]]
    auc = float(_trapz(tpr, fpr))
    keep = thin(len(fpr), max_points)
    return fpr[keep], tpr[keep], thr[keep], auc


def delong_placements(labels, scores):
    """DeLong placement values: per positive, the share of negatives it outranks; per negative, the
    share of positives that outrank it (ties count one half). Both average to the AUC."""
    from scipy.stats import rankdata

    pos = np.asarray(labels) > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    r = rankdata(scores)
    v10 = (r[pos] - rankdata(scores[pos])) / n0
    v01 = 1.0 - (r[~pos] - rankdata(scores[~pos])) / n1
    return v10, v01


def auc_difference(labels, scores_a, scores_b, *, steps: int = 1):
    """(AUC_a - AUC_b, lo, hi): paired DeLong 95% interval on n / steps effective samples per class."""
    labels = np.asarray(labels, float)
    n1 = int((labels > 0.5).sum())
    n0 = len(labels) - n1
    if n1 < 2 or n0 < 2:
        return float("nan"), float("nan"), float("nan")
    a10, a01 = delong_placements(labels, np.asarray(scores_a, float))
    b10, b01 = delong_placements(labels, np.asarray(scores_b, float))
    d = float(a10.mean() - b10.mean())
    var = np.var(a10 - b10, ddof=1) / n_eff(n1, steps) + np.var(a01 - b01, ddof=1) / n_eff(n0, steps)
    half = Z95 * float(np.sqrt(max(var, 0.0)))
    return d, d - half, d + half


# ----------------------------------------------------------------- long-run variance (Diebold-Mariano)
DM_LAG_PER_STEP = 2      # Bartlett lag of the Diebold-Mariano long-run variance, in multiples of bars ahead


def long_run_variance(x: np.ndarray, lag: int) -> float:
    """Bartlett (Newey-West) long-run variance of a time-ordered series; NaN entries (unscored bars) keep their
    place in time and add nothing. Divide by the number of finite entries for the variance of their mean."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n < 2:
        return float("nan")
    z = np.where(ok, x - x[ok].mean(), 0.0)
    lrv = float(z @ z) / n
    for k in range(1, min(int(lag), len(z) - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(z[k:] @ z[:-k]) / n
    return lrv


def dm_z(loss_model: np.ndarray, loss_base: np.ndarray, steps: int, *, lag: Optional[int] = None) -> Optional[float]:
    """Diebold-Mariano z of mean(loss_base - loss_model) (> 0: the model is better).

    Both arrays are per bar, time-ordered (NaN = unscored). The variance of the mean difference is the
    Bartlett (Newey-West) long-run variance with ``lag`` bars (default DM_LAG_PER_STEP x ``steps``): h-bar
    targets overlap for h - 1 bars, so the differences are autocorrelated at least that far. None when fewer
    than 2 non-overlapping outcomes are scored or the difference is constant.
    """
    diff = np.asarray(loss_base, float) - np.asarray(loss_model, float)
    steps = max(1, int(steps))
    n = int(np.isfinite(diff).sum())
    if n // steps < 2:
        return None
    lrv = long_run_variance(diff, DM_LAG_PER_STEP * steps if lag is None else int(lag))
    if not lrv > 0:
        return None
    return float(np.nanmean(diff) / math.sqrt(lrv / n))


# --------------------------------------------------------------------- paired block bootstrap (ranked metrics)
BLOCK = 80
BOOT_N = 500             # resamples of the paired block bootstrap of the ranked baseline metrics


def block_bootstrap_counts(n: int, *, block: int = BLOCK, n_boot: int = BOOT_N, seed: int = 0) -> np.ndarray:
    """[n_boot, n] multiplicities of each bar in moving-block bootstrap resamples of a length-n series."""
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    flat = (idx + (np.arange(n_boot) * n)[:, None]).ravel()
    return np.bincount(flat, minlength=n_boot * n).reshape(n_boot, n).astype(float)


def ties(x: np.ndarray):
    """(order, starts, group): the sort order of x, where each tie group starts in it, and each sorted item's group."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    return order, starts, np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(x)]))


def w_group_midranks(W: np.ndarray, starts: np.ndarray, order: np.ndarray):
    """(G, mid): per resample (rows of W: multiplicities), the weight of each tie group and its mid-rank."""
    G = np.add.reduceat(np.take(W, order, axis=1), starts, axis=1)
    return G, np.cumsum(G, axis=1) - G + (G + 1.0) / 2.0


def w_pearson(W: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Weighted Pearson correlation of two [n] series per row of W; 0 where either side is constant."""
    a, b = a - a.mean(), b - b.mean()
    sw = W.sum(1)
    ma, mb = W @ a / sw, W @ b / sw
    cov, va, vb = W @ (a * b) / sw - ma * mb, W @ (a * a) / sw - ma * ma, W @ (b * b) / sw - mb * mb
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    return np.where((va > 0) & (vb > 0), r, 0.0)


def w_spearman(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation (Pearson on tie-averaged ranks) per weighted resample; 0 where a side is constant."""
    ox, sx, gx = ties(x)
    oy, sy, gy = ties(y)
    Gx, mx = w_group_midranks(W, sx, ox)
    Gy, my = w_group_midranks(W, sy, oy)
    group_x = np.empty(len(x), int)
    group_x[ox] = gx
    sw = W.sum(1)
    m = (sw + 1.0) / 2.0                                       # the mean rank of every resample
    cross = (np.take(W, oy, axis=1) * my[:, gy] * np.take(mx, group_x[oy], axis=1)).sum(1) / sw - m * m
    vx, vy = (Gx * mx * mx).sum(1) / sw - m * m, (Gy * my * my).sum(1) / sw - m * m
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cross / np.sqrt(vx * vy)
    return np.where((vx > 1e-9) & (vy > 1e-9), r, 0.0)

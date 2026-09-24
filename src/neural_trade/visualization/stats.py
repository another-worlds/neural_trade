"""Uncertainty for the figures: how much of what a chart shows is noise.

Consecutive 1-minute samples are not independent: the target of the sample at bar t and of the
sample at bar t+1 share ``h - 1`` of their ``h`` bars (h = the horizon in bars). Treating the N
samples as independent makes every interval about sqrt(h) times too narrow. As in the evaluation
report (``n_eff = N // h_steps``), the helpers here count ``N / steps`` effective samples.

* :func:`n_eff` - effective sample size.
* :func:`wilson` - binomial proportion interval on the effective count.
* :func:`mean_ci` - normal interval of a mean on the effective count.
* :func:`corr_null_r` - the +/- band a correlation r stays inside by chance (no relationship), in r units.
* :func:`corr_null` - the same band on the Fisher-z scale (atanh r), for intervals built on that scale.
* :func:`auc_ci` - Hanley-McNeil interval for a ROC AUC on effective class counts.
* :func:`thin` - indices that keep at most ``max_points`` of a long, ordered series (for plotting).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

Z95 = 1.959964


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

"""Helpers shared by the analytics figures (analytics_direction / _delta / _variance / _confidence)."""
from __future__ import annotations

import numpy as np

from neural_trade.visualization import theme as T

UP_COLOR = T.UP_COLOR       # realised class colours (violet / yellow): not a horizon colour
DOWN_COLOR = T.DOWN_COLOR


def _labels(frame, config):
    from neural_trade.metrics.direction_labels import direction_labels_np

    deadband = float(getattr(config, "DIR_DEADBAND_BPS", 0.0) or 0.0) if config is not None else 0.0
    return direction_labels_np(frame.y, frame.last_close, deadband)


def _wilson(k, n, z=1.96):
    n = np.maximum(np.asarray(n, float), 1)
    p = np.asarray(k, float) / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


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


def _binned(x, y, n_bins=10):
    """Equal-count bins of x: (mean x, mean y, 95% CI of mean y, count)."""
    order = np.argsort(x)
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx) < 2:
            continue
        yy = y[idx]
        half = 1.96 * yy.std(ddof=1) / np.sqrt(len(idx))
        rows.append((x[idx].mean(), yy.mean(), yy.mean() - half, yy.mean() + half, len(idx)))
    return np.array(rows)


def _rolling_mean(v, window):
    v = np.asarray(v, float)
    if len(v) < window:
        return np.arange(len(v)), np.full(len(v), np.nan)
    k = np.ones(window) / window
    return np.arange(window - 1, len(v)), np.convolve(v, k, mode="valid")


def _rolling_corr(a, b, window):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < window:
        return np.arange(0), np.array([])
    x, ma = _rolling_mean(a, window)
    _, mb = _rolling_mean(b, window)
    _, mab = _rolling_mean(a * b, window)
    _, maa = _rolling_mean(a * a, window)
    _, mbb = _rolling_mean(b * b, window)
    cov = mab - ma * mb
    den = np.sqrt(np.maximum(maa - ma ** 2, 0) * np.maximum(mbb - mb ** 2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return x, np.where(den > 0, cov / den, np.nan)


def _grid(rows, titles, *, row_heights=None, vspace=0.09, specs=None):
    from plotly.subplots import make_subplots

    return make_subplots(rows=rows, cols=3, subplot_titles=titles, vertical_spacing=vspace,
                         horizontal_spacing=0.06, row_heights=row_heights, specs=specs)


def _diag(fig, row, col, lo, hi, name="perfect"):
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name=name, showlegend=(row, col) == (1, 1),
                             legendgroup="ref", line=dict(color=T.NEUTRAL, dash="dot", width=1), hoverinfo="skip"),
                  row, col)

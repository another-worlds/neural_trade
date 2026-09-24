"""Calibration figures: direction reliability and conformal-interval coverage over time."""
from __future__ import annotations

from typing import Optional

import numpy as np

from neural_trade.visualization.theme import apply


def reliability_table(labels, probs, n_bins: int = 10):
    """Equal-count bins of predicted P(up): mean prediction, observed up-rate, count, 95% band."""
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    order = np.argsort(probs)
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        p, y = probs[idx].mean(), labels[idx].mean()
        half = 1.96 * np.sqrt(max(y * (1 - y), 1e-12) / len(idx))
        rows.append((p, y, len(idx), y - half, y + half))
    return np.array(rows)


def reliability_figure(labels, p_raw, p_cal=None, *, n_bins: int = 10, title: Optional[str] = None):
    """Reliability diagram (equal-count bins) for the raw and, if given, calibrated P(up)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    lo = min(np.min(p_raw), np.min(p_cal) if p_cal is not None else 1.0)
    hi = max(np.max(p_raw), np.max(p_cal) if p_cal is not None else 0.0)
    pad = max(0.01, (hi - lo) * 0.1)
    fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines", name="perfect",
                             line=dict(dash="dot", color="#9a9890")))
    for name, p, color in (("raw", p_raw, "#c98500"), ("calibrated", p_cal, "#3987e5")):
        if p is None:
            continue
        t = reliability_table(labels, p, n_bins)
        fig.add_trace(go.Scatter(x=t[:, 0], y=t[:, 1], mode="lines+markers", name=name, line=dict(color=color),
                                 error_y=dict(type="data", symmetric=False, array=t[:, 4] - t[:, 1],
                                              arrayminus=t[:, 1] - t[:, 3], thickness=1),
                                 hovertext=[f"n={int(n)}" for n in t[:, 2]]))
    fig.update_layout(title=title or "Direction reliability (equal-count bins, 95% bands)",
                      xaxis_title="predicted P(up)", yaxis_title="observed up-rate (outside the deadband)",
                      height=420)
    return apply(fig)


def coverage_over_time_figure(y, lo, hi, *, window: int = 500, target: float = 0.9, title: Optional[str] = None):
    """Rolling share of outcomes inside the interval, and the interval width, along the block."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    inside = ((y >= lo) & (y <= hi)).astype(float)
    k = np.ones(window) / window
    roll = np.convolve(inside, k, mode="valid")          # trailing windows, plotted at their end
    x = np.arange(window - 1, len(y))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        subplot_titles=(f"Coverage, trailing {window} samples", "Interval width ($)"))
    fig.add_trace(go.Scatter(x=x, y=roll, name="coverage", line=dict(color="#3987e5")), 1, 1)
    fig.add_hline(y=target, line_dash="dot", row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=hi - lo, name="width", line=dict(width=1, color="#898781")),
                  2, 1)
    fig.update_layout(title=title or f"Interval coverage: {inside.mean():.3f} overall (target {target:.2f})",
                      height=460, showlegend=False)
    return apply(fig)


# ------------------------------------------------------------------ registry entries (data, config)
def reliability(data, config=None, **kw):
    """``data``: {"labels", "p_raw", "p_cal"?}."""
    return reliability_figure(data["labels"], data["p_raw"], data.get("p_cal"), **kw)


def interval_coverage(data, config=None, **kw):
    """``data``: {"y", "lo", "hi"}."""
    return coverage_over_time_figure(data["y"], data["lo"], data["hi"], **kw)

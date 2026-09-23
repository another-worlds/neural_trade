"""Evaluation figures (registered in the Visualizations registry as ``eval_report``)."""
from __future__ import annotations

import numpy as np

from neural_trade.evaluation.frame import HORIZONS
from neural_trade.metrics.direction_labels import direction_labels_np


def eval_report_figure(data, config, **_):
    """Reliability diagram, PIT histogram and interval coverage for a PredictionFrame."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.special import ndtr

    frame = data
    deadband = float(getattr(config, "DIR_DEADBAND_BPS", 0.0))
    labels = direction_labels_np(frame.y, frame.last_close, deadband)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Reliability (direction head)", "PIT histogram",
                                                        "90% interval coverage"))
    edges = np.linspace(0, 1, 11)
    for i, h in enumerate(HORIZONS):
        lab, mask = labels[h]
        p = frame.prob(h)[mask]
        t = lab[mask]
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        conf = [p[idx == b].mean() if (idx == b).any() else None for b in range(10)]
        obs = [t[idx == b].mean() if (idx == b).any() else None for b in range(10)]
        fig.add_trace(go.Scatter(x=conf, y=obs, mode="lines+markers", name=f"{h} reliability"), row=1, col=1)
        u = ndtr((frame.y[:, i] - frame.delta[h]) / np.maximum(frame.sigma(h), 1e-12))
        fig.add_trace(go.Histogram(x=u, nbinsx=20, histnorm="probability density", name=f"{h} PIT",
                                   opacity=0.5), row=1, col=2)
        if frame.intervals and h in frame.intervals:
            lo, hi = frame.intervals[h]
            cov = float(np.mean((frame.y[:, i] >= lo) & (frame.y[:, i] <= hi)))
            fig.add_trace(go.Bar(x=[h], y=[cov], name=f"{h} coverage"), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot"), name="ideal"), row=1, col=1)
    fig.add_hline(y=0.9, line_dash="dot", row=1, col=3)
    fig.update_layout(height=420, barmode="overlay", title=f"Evaluation - {frame.split} split")
    return fig

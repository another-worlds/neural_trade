"""Confidence (accuracy by confidence, selective accuracy, confusion) and cross-horizon coherence."""
from __future__ import annotations


import numpy as np

from neural_trade.visualization import stats as S  # noqa: F401  (noise bands: effective sample size)
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import (  # noqa: F401
    DOWN_COLOR, UP_COLOR, _binned, _diag, _grid, _labels, _rolling_corr, _rolling_mean, _wilson, roc_curve,
)


# ------------------------------------------------------------------ confidence
def confidence_analytics_figure(frame, config=None, *, height: int = 1000):
    import plotly.graph_objects as go

    labels = _labels(frame, config)
    titles = [f"{T.horizon_label(h, config)}: accuracy by confidence decile" for h in T.HORIZONS]
    titles += ["Selective accuracy: trade only the most confident x%"] * 3
    titles += ["Confusion matrix (outside the deadband)"] * 3
    fig = _grid(3, titles, row_heights=[0.36, 0.36, 0.28])
    for j, h in enumerate(T.HORIZONS, start=1):
        lab, mask = labels[h]
        lab, p = lab[mask], frame.prob(h, True)[mask]
        c = T.HORIZON_COLORS[h]
        pred = (p > 0.5).astype(float)
        correct = (pred == lab).astype(float)
        conf = np.abs(p - 0.5)
        order = np.argsort(conf)
        rows = []
        for idx in np.array_split(order, 10):
            k, n = correct[idx].sum(), len(idx)
            a, lo, hi = _wilson(k, n)
            rows.append((conf[idx].mean() + 0.5, a, lo, hi, n))
        t = np.array(rows)
        fig.add_trace(go.Bar(x=[str(k + 1) for k in range(len(t))], y=t[:, 1], name="accuracy", legendgroup="acc",
                             showlegend=j == 1, marker=dict(color=T.rgba(c, 0.85)),
                             error_y=dict(type="data", symmetric=False, array=t[:, 3] - t[:, 1],
                                          arrayminus=t[:, 1] - t[:, 2], thickness=1, width=3, color=T.INK_2),
                             customdata=np.column_stack([t[:, 4], t[:, 0]]),
                             hovertemplate="decile %{x}: mean max(p, 1-p) %{customdata[1]:.3f}<br>accuracy %{y:.3f}"
                                           "<br>n %{customdata[0]:.0f}<extra></extra>"), 1, j)
        fig.update_xaxes(type="category", row=1, col=j)
        fig.add_hline(y=0.5, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=1, col=j)
        fig.update_yaxes(range=[max(0, t[:, 2].min() - 0.05), min(1, t[:, 3].max() + 0.05)], row=1, col=j)
        # selective accuracy
        desc = order[::-1]
        kept = np.arange(1, len(desc) + 1)
        acc = np.cumsum(correct[desc]) / kept
        share = kept / len(desc)
        sel = share >= 0.02
        a, lo, hi = _wilson(np.cumsum(correct[desc])[sel], kept[sel])
        fig.add_trace(go.Scatter(x=np.r_[share[sel], share[sel][::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                 fillcolor=T.rgba(c, 0.15), line=dict(width=0), name="95% band",
                                 legendgroup="band", showlegend=j == 1, hoverinfo="skip"), 2, j)
        fig.add_trace(go.Scatter(x=share[sel], y=acc[sel], mode="lines", name="accuracy of the top x%",
                                 legendgroup="sel", showlegend=j == 1, line=dict(color=c, width=2),
                                 customdata=kept[sel],
                                 hovertemplate="top %{x:.0%} (n %{customdata})<br>accuracy %{y:.3f}<extra></extra>"),
                      2, j)
        fig.add_hline(y=0.5, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=2, col=j)
        fig.update_xaxes(type="log", tickvals=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0], tickformat=".0%", row=2, col=j,
                         title_text="share of samples kept (most confident first)")
        # confusion matrix
        cm = np.array([[np.sum((lab == 0) & (pred == 0)), np.sum((lab == 0) & (pred == 1))],
                       [np.sum((lab == 1) & (pred == 0)), np.sum((lab == 1) & (pred == 1))]])
        pct = cm / max(cm.sum(), 1)
        fig.add_trace(go.Heatmap(z=pct, x=["pred down", "pred up"], y=["real down", "real up"],
                                 colorscale=[[0, T.SURFACE], [1, c]], showscale=False, zmin=0, zmax=0.5,
                                 text=[[f"{cm[r, k]:,}<br>{100 * pct[r, k]:.1f}%" for k in range(2)] for r in range(2)],
                                 texttemplate="%{text}", textfont=dict(color=T.INK, size=13),
                                 hovertemplate="%{y}, %{x}: %{text}<extra></extra>"), 3, j)
        fig.update_xaxes(title_text="confidence decile (1 = least confident)", row=1, col=j)
    T.apply(fig, title="Confidence", height=height,
            subtitle="does a more confident P(up) mean a more accurate call? (calibrated P(up), deadband excluded)")
    fig.update_layout(margin=dict(t=120), legend=dict(y=1.04))
    return fig


# ------------------------------------------------------------------ coherence
def coherence_analytics_figure(frame, config=None, *, height: int = 460):
    import plotly.graph_objects as go

    labels = _labels(frame, config)
    P = np.column_stack([frame.prob(h, True) for h in T.HORIZONS])
    corr = np.corrcoef(P.T)
    votes = (P > 0.5).sum(axis=1)
    lab, mask = labels["h1"]
    pred1 = (P[:, 1] > 0.5).astype(float)
    unanimous = (votes == 0) | (votes == 3)
    fig = _grid(1, ("P(up) correlation across horizons", "Horizons predicting up (per sample)",
                    "h1 accuracy: horizons agree vs split"))
    fig.add_trace(go.Heatmap(z=corr, x=list(T.HORIZONS), y=list(T.HORIZONS), zmin=-1, zmax=1,
                             colorscale=[[0, T.SERIES[1]], [0.5, "#383835"], [1, T.SERIES[0]]], showscale=False,
                             text=[[f"{v:+.2f}" for v in r] for r in corr], texttemplate="%{text}",
                             textfont=dict(color=T.INK), hovertemplate="%{y} vs %{x}: %{text}<extra></extra>"), 1, 1)
    counts = np.bincount(votes, minlength=4)
    fig.add_trace(go.Bar(x=["0 (all down)", "1", "2", "3 (all up)"], y=counts / len(votes), name="share",
                         marker=dict(color=[T.SERIES[1], T.rgba(T.SERIES[1], 0.5), T.rgba(T.SERIES[0], 0.5),
                                            T.SERIES[0]]),
                         customdata=counts, showlegend=False,
                         hovertemplate="%{x}: %{y:.1%} (n %{customdata:,})<extra></extra>"), 1, 2)
    rows = []
    for name, sel in (("all 3 agree", unanimous), ("split", ~unanimous)):
        m = mask & sel
        k, n = float(np.sum(pred1[m] == lab[m])), int(m.sum())
        a, lo, hi = _wilson(k, n)
        rows.append((name, float(a), float(lo), float(hi), n))
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows], showlegend=False,
                         marker=dict(color=T.HORIZON_COLORS["h1"]),
                         error_y=dict(type="data", symmetric=False, array=[r[3] - r[1] for r in rows],
                                      arrayminus=[r[1] - r[2] for r in rows], color=T.INK_2, thickness=1),
                         customdata=[r[4] for r in rows],
                         hovertemplate="%{x}: accuracy %{y:.3f} (n %{customdata:,})<extra></extra>"), 1, 3)
    fig.add_hline(y=0.5, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=1, col=3)
    fig.update_xaxes(type="category", row=1, col=2)
    fig.update_xaxes(type="category", row=1, col=3)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    lo = min(r[2] for r in rows)
    fig.update_yaxes(range=[max(0, lo - 0.05), min(1, max(r[3] for r in rows) + 0.05)], row=1, col=3)
    T.apply(fig, title="Cross-horizon coherence", height=height, legend_top=False,
            subtitle="calibrated P(up); a horizon votes up when P(up) > 0.5")
    fig.update_layout(margin=dict(t=100))
    return fig

"""Direction heads: P(up) by realised class, ROC, reliability (one column per horizon)."""
from __future__ import annotations


import numpy as np

from neural_trade.visualization import stats as S  # noqa: F401  (noise bands: effective sample size)
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import (  # noqa: F401
    DOWN_COLOR, UP_COLOR, _binned, _diag, _grid, _labels, _rolling_corr, _rolling_mean, _wilson, roc_curve,
)


# ------------------------------------------------------------------ direction
def direction_analytics_figure(frame, config=None, *, bins: int = 40, height: int = 1020):
    import plotly.graph_objects as go

    labels = _labels(frame, config)
    titles, stats = [], {}
    for h in T.HORIZONS:
        lab, mask = labels[h]
        p_cal, p_raw = frame.prob(h, True)[mask], frame.direction_prob[h][mask]
        g = frame.gauss_prob(h, float(getattr(config, "DIR_DEADBAND_BPS", 0.0) or 0.0))[mask]
        stats[h] = dict(lab=lab[mask], p_cal=p_cal, p_raw=p_raw, g=g)
    for h in T.HORIZONS:
        titles.append(f"{T.horizon_label(h, config)}: P(up) by realised move")
    for h in T.HORIZONS:
        s = stats[h]
        titles.append(f"ROC - head AUC {roc_curve(s['lab'], s['p_raw'])[2]:.3f}, "
                      f"Gaussian {roc_curve(s['lab'], s['g'])[2]:.3f}")
    from neural_trade.metrics.numpy_metrics import ece_pos

    for h in T.HORIZONS:
        s = stats[h]
        titles.append(f"Reliability - ECE {ece_pos(s['lab'], s['p_raw']):.3f} raw, "
                      f"{ece_pos(s['lab'], s['p_cal']):.3f} calibrated")
    fig = _grid(3, titles)
    from neural_trade.visualization.calibration_plots import reliability_table

    for j, h in enumerate(T.HORIZONS, start=1):
        s, c = stats[h], T.HORIZON_COLORS[h]
        lo, hi = np.percentile(s["p_cal"], [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        for cls, name, color in ((1, "realised up", UP_COLOR), (0, "realised down", DOWN_COLOR)):
            dens, _ = np.histogram(s["p_cal"][s["lab"] == cls], bins=edges, density=True)
            fig.add_trace(go.Scatter(x=mids, y=dens, mode="lines", line_shape="hvh", name=name, legendgroup=name,
                                     showlegend=j == 1, line=dict(color=color, width=2), fill="tozeroy",
                                     fillcolor=T.rgba(color, 0.12),
                                     hovertemplate=f"{name}<br>P(up) %{{x:.3f}}<br>density %{{y:.2f}}<extra></extra>"),
                          1, j)
        fig.add_vline(x=0.5, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=1, col=j)
        # ROC
        for key, name, dash in (("p_raw", "direction head", T.VAL_DASH), ("g", "Gaussian readout", T.ALT_DASH)):
            fpr, tpr, auc = roc_curve(s["lab"], s[key])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name, legendgroup=f"roc-{key}",
                                     showlegend=j == 1, line=dict(color=c, dash=dash, width=2),
                                     hovertemplate=f"{h} {name} (AUC {auc:.3f})<br>FPR %{{x:.2f}} TPR %{{y:.2f}}"
                                                   "<extra></extra>"), 2, j)
        _diag(fig, 2, j, 0, 1, name="chance")
        # reliability
        allp = np.r_[s["p_raw"], s["p_cal"]]
        _diag(fig, 3, j, float(allp.min()), float(allp.max()))
        for key, name, dash in (("p_raw", "raw", T.TRAIN_DASH), ("p_cal", "calibrated", T.VAL_DASH)):
            t = reliability_table(s["lab"], s[key], 10)
            fig.add_trace(go.Scatter(
                x=t[:, 0], y=t[:, 1], mode="lines+markers", name=f"{name} P(up)", legendgroup=f"rel-{key}",
                showlegend=j == 1, line=dict(color=c, dash=dash, width=2), marker=dict(size=7, color=c),
                error_y=dict(type="data", symmetric=False, array=t[:, 4] - t[:, 1], arrayminus=t[:, 1] - t[:, 3],
                             thickness=1, width=3, color=c),
                customdata=t[:, 2],
                hovertemplate=f"{h} {name}<br>predicted %{{x:.3f}}<br>observed %{{y:.3f}}<br>n %{{customdata:.0f}}"
                              "<extra></extra>"), 3, j)
        fig.update_xaxes(title_text="P(up)", row=1, col=j)
        fig.update_xaxes(title_text="false positive rate", row=2, col=j)
        fig.update_yaxes(title_text="true positive rate" if j == 1 else None, row=2, col=j)
        fig.update_xaxes(title_text="predicted P(up)", row=3, col=j)
        fig.update_yaxes(title_text="observed up-rate" if j == 1 else None, row=3, col=j)
    fig.update_layout(barmode="overlay")
    T.apply(fig, title="Direction heads", height=height,
            subtitle=f"{frame.split} block, {len(frame):,} samples; moves inside the deadband excluded; "
                     "ROC ranks by the raw head (calibration does not change the ranking)")
    fig.update_layout(margin=dict(t=120), legend=dict(y=1.04))
    return fig

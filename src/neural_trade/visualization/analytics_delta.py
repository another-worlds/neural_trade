"""Price heads (delta): realised vs predicted move, binned calibration, rolling correlation."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from neural_trade.visualization import stats as S  # noqa: F401  (noise bands: effective sample size)
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import (  # noqa: F401
    DOWN_COLOR, UP_COLOR, _binned, _diag, _grid, _labels, _rolling_corr, _rolling_mean, _wilson, roc_curve,
)


# ------------------------------------------------------------------ delta
def delta_analytics_figure(frame, config=None, *, raw_delta: Optional[Dict[str, np.ndarray]] = None,
                           window: int = 500, height: int = 980):
    """``raw_delta``: the price heads before delta shrinkage (default: ``frame.delta``)."""
    import plotly.graph_objects as go

    raw = raw_delta or frame.delta
    titles = []
    for i, h in enumerate(T.HORIZONS):
        d, y = np.asarray(raw[h], float), frame.y[:, i]
        corr = np.corrcoef(d, y)[0, 1] if d.std() > 0 else np.nan
        beta = float(np.dot(y, d) / np.dot(d, d)) if np.dot(d, d) > 0 else np.nan
        titles.append(f"{T.horizon_label(h, config)}: corr {corr:+.3f}, LS slope {beta:+.2f}")
    titles += ["Realised vs predicted, binned (95% CI)"] * 3
    titles += [f"Rolling {window}-sample correlation"] * 3
    fig = _grid(3, titles)
    for j, h in enumerate(T.HORIZONS, start=1):
        i, c = j - 1, T.HORIZON_COLORS[h]
        d, y = np.asarray(raw[h], float), frame.y[:, i]
        served = np.asarray(frame.delta[h], float)
        fig.add_trace(go.Scattergl(x=d, y=y, mode="markers", name="samples", legendgroup="samples",
                                   showlegend=j == 1, marker=dict(size=3, color=T.rgba(c, 0.25)),
                                   hovertemplate="predicted %{x:$,.1f}<br>realised %{y:$,.1f}<extra></extra>"), 1, j)
        lim = float(np.percentile(np.abs(y), 99.5))
        fig.add_trace(go.Scatter(x=[-lim, lim], y=[0, 0], mode="lines", name="zero prediction baseline",
                                 legendgroup="zero", showlegend=j == 1,
                                 line=dict(color=T.NEUTRAL, dash="dot", width=1), hoverinfo="skip"), 1, j)
        if d.std() > 0:
            beta = float(np.dot(y, d) / np.dot(d, d))
            xs = np.array([d.min(), d.max()])
            fig.add_trace(go.Scatter(x=xs, y=beta * xs, mode="lines", name="least-squares fit", legendgroup="ls",
                                     showlegend=j == 1, line=dict(color=T.INK, width=1.5),
                                     hovertemplate=f"slope {beta:+.3f}<extra></extra>"), 1, j)
        fig.update_yaxes(range=[-lim, lim], row=1, col=j)
        # binned calibration of the price head
        if d.std() > 0:
            t = _binned(d, y, 10)
            fig.add_trace(go.Scatter(
                x=t[:, 0], y=t[:, 1], mode="lines+markers", name="mean realised per predicted decile",
                legendgroup="binned", showlegend=j == 1, line=dict(color=c, width=2),
                error_y=dict(type="data", symmetric=False, array=t[:, 3] - t[:, 1], arrayminus=t[:, 1] - t[:, 2],
                             thickness=1, width=3, color=c),
                customdata=t[:, 4],
                hovertemplate="predicted %{x:$,.2f}<br>realised %{y:$,.2f}<br>n %{customdata:.0f}<extra></extra>"),
                2, j)
            _diag(fig, 2, j, float(t[:, 0].min()), float(t[:, 0].max()))
        # rolling correlation of raw and served delta with the outcome
        x, rc = _rolling_corr(d, y, window)
        fig.add_trace(go.Scatter(x=x, y=rc, mode="lines", name="raw price head", legendgroup="rc-raw",
                                 showlegend=j == 1, line=dict(color=c, width=1.5),
                                 hovertemplate="sample %{x}<br>corr %{y:+.3f}<extra></extra>"), 3, j)
        if served.std() > 0 and not np.allclose(served, d):
            fig.add_trace(go.Scatter(x=x, y=_rolling_corr(served, y, window)[1], mode="lines", name="served (shrunk)",
                                     legendgroup="rc-served", showlegend=j == 1,
                                     line=dict(color=c, width=1, dash=T.TRAIN_DASH), hoverinfo="skip"), 3, j)
        fig.add_hline(y=0, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=3, col=j)
        fig.update_xaxes(title_text="predicted move ($)", row=1, col=j)
        fig.update_yaxes(title_text="realised move ($)" if j == 1 else None, row=1, col=j)
        fig.update_xaxes(title_text="predicted move ($), decile mean", row=2, col=j)
        fig.update_xaxes(title_text="sample (time order)", row=3, col=j)
    T.note_on_empty(fig, "constant prediction: nothing to bin")
    T.apply(fig, title="Price heads (delta)", height=height,
            subtitle="raw heads before delta shrinkage; a slope near 0 or negative means the served delta is "
                     "shrunk towards zero")
    fig.update_layout(margin=dict(t=120), legend=dict(y=1.04))
    return fig

"""Variance heads: predicted sigma vs realised error, PIT, interval coverage."""
from __future__ import annotations


import numpy as np

from neural_trade.visualization import stats as S  # noqa: F401  (noise bands: effective sample size)
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import (  # noqa: F401
    DOWN_COLOR, UP_COLOR, _binned, _diag, _grid, _labels, _rolling_corr, _rolling_mean, _wilson, roc_curve,
)


# ------------------------------------------------------------------ variance
def variance_analytics_figure(frame, config=None, *, window: int = 500, height: int = 980):
    import plotly.graph_objects as go
    from scipy.stats import norm

    titles = []
    for i, h in enumerate(T.HORIZONS):
        sig, err = frame.sigma(h), frame.y[:, i] - frame.delta[h]
        from scipy.stats import spearmanr

        rho = spearmanr(sig, err ** 2).correlation if np.ptp(sig) > 0 else np.nan
        titles.append(f"{T.horizon_label(h, config)}: realised vs predicted sigma (Spearman {rho:+.3f})")
    titles += ["PIT histogram (flat = calibrated)"] * 3
    titles += [f"Interval coverage, trailing {window}"] * 3
    fig = _grid(3, titles)
    for j, h in enumerate(T.HORIZONS, start=1):
        i, c = j - 1, T.HORIZON_COLORS[h]
        sig, mu, y = frame.sigma(h), frame.delta[h], frame.y[:, i]
        t = _binned(sig, np.abs(y - mu), 12)
        rms = np.array([np.sqrt(np.mean((y - mu)[idx] ** 2)) for idx in np.array_split(np.argsort(sig), 12)])
        fig.add_trace(go.Scatter(x=t[:, 0], y=rms[:len(t)], mode="lines+markers", name="RMS error per sigma bin",
                                 legendgroup="rms", showlegend=j == 1, line=dict(color=c, width=2),
                                 customdata=t[:, 4],
                                 hovertemplate="predicted sigma %{x:$,.1f}<br>RMS error %{y:$,.1f}<br>n %{customdata:.0f}"
                                               "<extra></extra>"), 1, j)
        fig.add_trace(go.Scatter(x=t[:, 0], y=t[:, 1], mode="lines", name="mean |error|", legendgroup="mae",
                                 showlegend=j == 1, line=dict(color=c, width=1, dash=T.TRAIN_DASH),
                                 hoverinfo="skip"), 1, j)
        _diag(fig, 1, j, float(t[:, 0].min()), float(t[:, 0].max()), name="RMS error = sigma")
        # PIT
        pit = norm.cdf((y - mu) / np.maximum(sig, 1e-9))
        fig.add_trace(go.Histogram(x=pit, xbins=dict(start=0, end=1.0001, size=0.05), histnorm="probability density",
                                   name="PIT", legendgroup="pit", showlegend=j == 1,
                                   marker=dict(color=T.rgba(c, 0.8)),
                                   hovertemplate="PIT %{x}<br>density %{y:.2f}<extra></extra>"), 2, j)
        fig.add_hline(y=1.0, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=2, col=j)
        # rolling coverage
        if frame.intervals is not None and h in frame.intervals:
            lo, hi = frame.intervals[h]
            inside = ((y >= lo) & (y <= hi)).astype(float)
            x, cov = _rolling_mean(inside, window)
            fig.add_trace(go.Scatter(x=x, y=cov, mode="lines", name="conformal 90% interval", legendgroup="cov",
                                     showlegend=j == 1, line=dict(color=c, width=1.5),
                                     hovertemplate="sample %{x}<br>coverage %{y:.3f}<extra></extra>"), 3, j)
        inside_g = (np.abs(y - mu) <= 1.645 * sig).astype(float)
        x, covg = _rolling_mean(inside_g, window)
        fig.add_trace(go.Scatter(x=x, y=covg, mode="lines", name="raw Gaussian +/-1.645 sigma", legendgroup="covg",
                                 showlegend=j == 1, line=dict(color=c, width=1, dash=T.TRAIN_DASH),
                                 hovertemplate="sample %{x}<br>coverage %{y:.3f}<extra></extra>"), 3, j)
        fig.add_hline(y=0.9, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=3, col=j)
        fig.update_xaxes(title_text="predicted sigma ($)", row=1, col=j)
        fig.update_yaxes(title_text="realised error ($)" if j == 1 else None, row=1, col=j)
        fig.update_xaxes(title_text="PIT = Phi((y - mu) / sigma)", row=2, col=j)
        fig.update_xaxes(title_text="sample (time order)", row=3, col=j)
    fig.update_layout(barmode="overlay")
    T.apply(fig, title="Variance heads", height=height,
            subtitle="the dotted diagonal is a perfectly calibrated sigma; a PIT hump in the middle = sigma too "
                     "wide, a U shape = too narrow")
    fig.update_layout(margin=dict(t=120), legend=dict(y=1.04))
    return fig

"""Per-trade analytics and the strategy comparison, from BacktestResults.

* :func:`trade_analytics_figure` - net P&L distribution, gross vs net (the cost drag), P&L by exit
  reason, cumulative P&L, holding time vs P&L, and each trade's maximum favourable / adverse
  excursion (from the bars' high / low).
* :func:`strategy_comparison_figure` - several strategies' equity curves and net vs gross return.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from neural_trade.visualization import stats as S  # noqa: F401
from neural_trade.visualization import theme as T


# ------------------------------------------------------------------ per-trade analytics
def excursions(trades, bars):
    """(MFE, MAE) per trade in % of the entry price, from the bars' high / low while it was open."""
    hi, lo = np.asarray(bars.high, float), np.asarray(bars.low, float)
    mfe, mae = [], []
    for t in trades:
        a, b = t.entry_bar + 1, max(t.exit_bar, t.entry_bar + 1) + 1
        h, low = hi[a:b], lo[a:b]
        if len(h) == 0:
            mfe.append(0.0), mae.append(0.0)
            continue
        if t.side == "LONG":
            mfe.append(100 * (h.max() - t.entry_price) / t.entry_price)
            mae.append(100 * (low.min() - t.entry_price) / t.entry_price)
        else:
            mfe.append(100 * (t.entry_price - low.min()) / t.entry_price)
            mae.append(100 * (t.entry_price - h.max()) / t.entry_price)
    return np.array(mfe), np.array(mae)


def trade_analytics_figure(result, bars=None, *, height: int = 820):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    trades = list(result.trades)
    titles = ("Net P&L per trade ($)", "Gross vs net P&L per trade (cost drag)", "Net P&L by exit reason ($)",
              "Cumulative P&L by trade", "Holding time vs net P&L", "Max favourable vs adverse excursion (%)")
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, vertical_spacing=0.14, horizontal_spacing=0.07)
    if not trades:
        T.note_on_empty(fig, "no trades")
        T.apply(fig, title=f"{result.strategy}: no trades", height=height)
        return fig
    net = np.array([t.net_pnl for t in trades])
    gross = np.array([t.gross_pnl for t in trades])
    held = np.array([t.bars_held for t in trades])
    side = np.array([t.side for t in trades])
    win = net > 0
    edges = np.linspace(min(net.min(), gross.min()), max(net.max(), gross.max()), 31)
    for sel, name, color in ((win, "winning trades", T.GOOD), (~win, "losing trades", T.CRITICAL)):
        fig.add_trace(go.Histogram(x=net[sel], xbins=dict(start=edges[0], end=edges[-1], size=edges[1] - edges[0]),
                                   name=name, legendgroup=name, marker=dict(color=T.rgba(color, 0.8)),
                                   hovertemplate="net %{x}<br>%{y} trades<extra></extra>"), 1, 1)
    dens, _ = np.histogram(gross, bins=edges)
    fig.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=dens, mode="lines", line_shape="hvh",
                             name="gross P&L (before costs)", line=dict(color=T.INK_2, width=1.5),
                             hovertemplate="gross %{x:$,.0f}<br>%{y} trades<extra></extra>"), 1, 1)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=1, col=1)
    for s_name, color, symbol in (("LONG", T.LONG_COLOR, "triangle-up"), ("SHORT", T.SHORT_COLOR, "triangle-down")):
        m = side == s_name
        if m.any():
            fig.add_trace(go.Scatter(x=gross[m], y=net[m], mode="markers", name=s_name.lower(), legendgroup=s_name,
                                     marker=dict(symbol=symbol, size=8, color=color, line=dict(color=T.PAPER, width=1)),
                                     hovertemplate="gross %{x:+$,.2f}<br>net %{y:+$,.2f}<extra></extra>"), 1, 2)
            fig.add_trace(go.Scatter(x=held[m], y=net[m], mode="markers", name=s_name.lower(), legendgroup=s_name,
                                     showlegend=False,
                                     marker=dict(symbol=symbol, size=8, color=color, line=dict(color=T.PAPER, width=1)),
                                     hovertemplate="held %{x} bars<br>net %{y:+$,.2f}<extra></extra>"), 2, 2)
    lim = [min(gross.min(), net.min()), max(gross.max(), net.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="no costs", line=dict(color=T.NEUTRAL, dash="dot", width=1),
                             hoverinfo="skip"), 1, 2)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=1, col=2)
    reasons = sorted({t.exit_reason for t in trades})
    for i, r in enumerate(reasons):
        m = np.array([t.exit_reason == r for t in trades])
        fig.add_trace(go.Box(y=net[m], name=f"{r} ({int(m.sum())})", boxpoints="all", jitter=0.4, pointpos=0,
                             marker=dict(size=4, color=T.SERIES[(i + 3) % len(T.SERIES)]),
                             line=dict(color=T.SERIES[(i + 3) % len(T.SERIES)], width=1), showlegend=False,
                             hovertemplate=f"{r}<br>net %{{y:+$,.2f}}<extra></extra>"), 1, 3)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=1, col=3)
    k = np.arange(1, len(trades) + 1)
    fig.add_trace(go.Scatter(x=k, y=np.cumsum(gross), mode="lines", name="cumulative gross",
                             line=dict(color=T.INK_2, width=1.5, dash="dot"),
                             hovertemplate="trade %{x}<br>gross %{y:+$,.0f}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=k, y=np.cumsum(net), mode="lines", name="cumulative net", line=dict(color=T.INK, width=2),
                             hovertemplate="trade %{x}<br>net %{y:+$,.0f}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=k, y=np.cumsum(gross - net), mode="lines", name="cumulative costs",
                             line=dict(color=T.CRITICAL, width=1.5),
                             hovertemplate="trade %{x}<br>costs %{y:$,.0f}<extra></extra>"), 2, 1)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, dash="dot", width=1), row=2, col=2)
    if bars is not None:
        mfe, mae = excursions(trades, bars)
        for sel, name, color, symbol in ((win, "win", T.GOOD, "circle"), (~win, "loss", T.CRITICAL, "x")):
            fig.add_trace(go.Scatter(x=mae[sel], y=mfe[sel], mode="markers", name=f"exit {name}", legendgroup=f"x-{name}",
                                     marker=dict(symbol=symbol, size=7, color=color),
                                     hovertemplate="adverse %{x:.3f}%<br>favourable %{y:.3f}%<extra></extra>"), 2, 3)
        rt = 2 * (result.config.fee_bps + result.config.half_spread_bps + result.config.slippage_bps) / 100
        fig.add_hline(y=rt, line=dict(color=T.WARNING, dash="dash", width=1), row=2, col=3,
                      annotation_text=f"round-trip cost {rt:.2f}%", annotation_position="top right",
                      annotation_font=dict(color=T.WARNING, size=11))
    for (r, c), (xt, yt) in {(1, 1): ("net P&L ($)", "trades"), (1, 2): ("gross P&L ($)", "net P&L ($)"),
                             (2, 1): ("trade #", "$"), (2, 2): ("bars held", "net P&L ($)"),
                             (2, 3): ("worst move against (%)", "best move in favour (%)")}.items():
        fig.update_xaxes(title_text=xt, row=r, col=c)
        fig.update_yaxes(title_text=yt, row=r, col=c)
    fig.update_layout(barmode="stack")
    T.note_on_empty(fig, "needs the bars")
    s = result.summary
    T.apply(fig, title=f"{result.strategy}: {len(trades)} trades", height=height,
            subtitle=f"hit rate {100 * (s.get('hit_rate') or 0):.1f}% after costs, "
                     f"{100 * (s.get('hit_rate_gross') or 0):.1f}% before; profit factor {s.get('profit_factor', 0):.2f}")
    fig.update_layout(margin=dict(t=120), legend=dict(y=1.04))
    return fig


# ------------------------------------------------------------------ several strategies
def strategy_comparison_figure(results: Dict[str, object], bars=None, *, height: int = 620):
    """``results``: {name: BacktestResult} on the same block."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], horizontal_spacing=0.08,
                        subplot_titles=("Equity after costs", "Return: before vs after costs"))
    names = list(results)
    for i, name in enumerate(names):
        r = results[name]
        eq = np.asarray(r.equity, float)
        fig.add_trace(go.Scatter(x=np.arange(-1, len(eq) - 1), y=eq, mode="lines", name=name,
                                 line=dict(color=T.SERIES[i % len(T.SERIES)], width=1.5),
                                 hovertemplate=f"{name}<br>bar %{{x}}: %{{y:$,.0f}}<extra></extra>"), 1, 1)
    init = float(results[names[0]].config.initial_equity) if names else 1.0
    gross = [100 * results[n].summary.get("gross_pnl", 0) / init for n in names]
    net = [100 * results[n].summary["total_return"] for n in names]
    labels = [f"{n} ({results[n].summary['n_trades']})" for n in names]
    fig.add_trace(go.Bar(y=labels, x=gross, orientation="h", name="before costs", marker=dict(color=T.rgba(T.INK_2, 0.5)),
                         hovertemplate="%{y}<br>gross %{x:+.2f}%<extra></extra>"), 1, 2)
    fig.add_trace(go.Bar(y=labels, x=net, orientation="h", name="after costs",
                         marker=dict(color=[T.GOOD if v > 0 else T.CRITICAL for v in net]),
                         hovertemplate="%{y}<br>net %{x:+.2f}%<extra></extra>"), 1, 2)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, width=1), row=1, col=2)
    fig.update_xaxes(title_text="bar (test block)", row=1, col=1)
    fig.update_xaxes(title_text="% of initial equity", row=1, col=2)
    fig.update_layout(barmode="group")
    T.apply(fig, title="Strategies on the same block", height=height,
            subtitle="default knobs and costs; the number in brackets is the trade count")
    fig.update_layout(margin=dict(t=110), legend=dict(y=1.06))
    return fig

"""Trading dashboards drawn from a BacktestResult (and the SignalFrame it traded on).

* :func:`trading_dashboard_figure` - six stacked panels on ONE time axis with synchronised hover
  (one vertical cursor, every panel's values in one tooltip): price with entries / exits, the
  take-profit and stop levels and win / loss connectors, holding periods shaded; the per-horizon
  P(up) and the strategy's entry lines; confidence and signal strength; predicted sigma with the
  variance spikes; equity (net, gross, buy-and-hold); drawdown.
* :func:`trade_analytics_figure` - per-trade view: net P&L distribution, gross vs net (the cost
  drag), P&L by exit reason, cumulative P&L, holding time vs P&L, and the maximum favourable /
  adverse excursion of each trade (from the bars' high / low).
* :func:`strategy_comparison_figure` - several strategies' equity curves and net vs gross return.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from neural_trade.visualization import theme as T

_THRESHOLD_FIELDS = (("long_above", "long entry"), ("short_below", "short entry"), ("median", "exit (median)"),
                     ("p_long", "long entry"), ("p_short", "short entry"))


# ------------------------------------------------------------------ stacked panels on one x axis
def _stack(fig, heights: Sequence[float], titles: Sequence[str], gap: float = 0.028):
    """Lay out ``len(heights)`` y axes top to bottom, all anchored to the single x axis (so hover
    and zoom are shared). Returns the y-axis refs ['y', 'y2', ...]."""
    total = sum(heights) + gap * (len(heights) - 1)
    top = 1.0
    refs = []
    for i, (h, title) in enumerate(zip(heights, titles)):
        size = h / total
        dom = [max(0.0, top - size), top]
        name = "yaxis" if i == 0 else f"yaxis{i + 1}"
        fig.update_layout({name: dict(domain=dom, anchor="x")})
        # each panel's heading is its own legend: the title, then the keys of that panel's traces
        legend = "legend" if i == 0 else f"legend{i + 1}"
        fig.update_layout({legend: dict(
            title=dict(text=f"<b>{title}</b>   ", side="left", font=dict(color=T.INK_2, size=12)),
            orientation="h", x=0, xanchor="left", y=dom[1] + 0.002, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=T.INK_2), itemwidth=30, itemsizing="constant")})
        refs.append("y" if i == 0 else f"y{i + 1}")
        top = dom[0] - gap / total
    fig.update_layout(xaxis=dict(anchor=refs[-1], domain=[0, 1]))
    return refs


def _trade_rows(trades, lo, hi):
    return [t for t in trades if lo <= t.entry_bar < hi]


def trading_dashboard_figure(result, bars=None, signals=None, strategy=None, *, start: Optional[int] = None,
                             end: Optional[int] = None, title: Optional[str] = None, height: int = 1350):
    """``result``: BacktestResult; ``bars``: its Bars; ``signals``: its SignalFrame; ``strategy``:
    the Strategy instance (its entry lines are drawn on the signal panel); ``start``/``end``: a bar
    window (default: the whole block)."""
    import plotly.graph_objects as go

    n = len(result.position)
    lo, hi = max(0, start or 0), min(n, end if end is not None else n)
    x = np.arange(lo, hi)
    has_sig = signals is not None
    heights = [0.34] + ([0.17, 0.11, 0.09] if has_sig else []) + [0.17, 0.09]
    titles = ["Price, trades, take-profit / stop levels"] + (
        ["P(up) per horizon and the strategy's entry lines", "Confidence and signal strength",
         "Predicted sigma h1 ($) and variance spikes"] if has_sig else []) + ["Equity", "Drawdown"]
    fig = go.Figure()
    refs = _stack(fig, heights, titles)
    ax = dict(zip(["price"] + (["signal", "conf", "sigma"] if has_sig else []) + ["equity", "dd"], refs))

    def add(trace, panel):
        ref = ax[panel]
        trace.update(xaxis="x", yaxis=ref, legend="legend" + ref[1:])
        fig.add_trace(trace)

    # --- price
    if bars is not None:
        close = np.asarray(bars.close, float)[lo:hi]
        add(go.Scatter(x=x, y=np.asarray(bars.high, float)[lo:hi], mode="lines", line=dict(width=0),
                       name="high", showlegend=False, hoverinfo="skip"), "price")
        add(go.Scatter(x=x, y=np.asarray(bars.low, float)[lo:hi], mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor=T.rgba(T.INK_2, 0.10), name="bar high / low range",
                       hoverinfo="skip"), "price")
        add(go.Scatter(x=x, y=close, mode="lines", name="close", line=dict(color=T.INK_2, width=1.2),
                       hovertemplate="close %{y:$,.2f}<extra></extra>"), "price")
    trades = _trade_rows(result.trades, lo, hi)
    for t in trades:   # holding periods
        fig.add_shape(type="rect", xref="x", yref=f"{ax['price']} domain", x0=t.entry_bar,
                      x1=max(t.exit_bar, t.entry_bar + 1), y0=0, y1=1, line_width=0, layer="below",
                      fillcolor=T.rgba(T.LONG_COLOR if t.side == "LONG" else T.SHORT_COLOR, 0.10))
    for outcome, color, dash in (("win", T.GOOD, "dot"), ("loss", T.CRITICAL, "dot")):
        sel = [t for t in trades if (t.net_pnl > 0) == (outcome == "win")]
        if not sel:
            continue
        xs, ys = [], []
        for t in sel:
            xs += [t.entry_bar, t.exit_bar, None]
            ys += [t.entry_price, t.exit_price, None]
        add(go.Scatter(x=xs, y=ys, mode="lines", name=f"{outcome} (entry to exit)", legendgroup=f"trade-{outcome}",
                       line=dict(color=color, width=1.2, dash=dash), hoverinfo="skip"), "price")
    for level, name, color in (("tp", "take-profit", T.GOOD), ("sl", "stop", T.CRITICAL)):
        xs, ys = [], []
        for t in trades:
            v = getattr(t, level)
            if v is not None and np.isfinite(v):
                xs += [t.entry_bar, t.exit_bar, None]
                ys += [v, v, None]
        if xs:
            add(go.Scatter(x=xs, y=ys, mode="lines", name=f"{name} level", legendgroup=f"lvl-{level}",
                           line=dict(color=T.rgba(color, 0.7), width=1, dash="dash"), hoverinfo="skip"), "price")
    for side, color, symbol in (("LONG", T.LONG_COLOR, "triangle-up"), ("SHORT", T.SHORT_COLOR, "triangle-down")):
        sel = [t for t in trades if t.side == side]
        if not sel:
            continue
        add(go.Scatter(
            x=[t.entry_bar for t in sel], y=[t.entry_price for t in sel], mode="markers", name=f"{side.lower()} entry",
            marker=dict(symbol=symbol, size=11, color=color, line=dict(color=T.PAPER, width=1)),
            customdata=[[t.tp if t.tp is not None else np.nan, t.sl if t.sl is not None else np.nan, t.notional,
                         t.entry_reason] for t in sel],
            hovertemplate=f"<b>{side} entry</b> %{{y:$,.2f}}<br>TP %{{customdata[0]:$,.2f}} · SL %{{customdata[1]:$,.2f}}"
                          "<br>size %{customdata[2]:$,.0f} · %{customdata[3]}<extra></extra>"), "price")
    for outcome, color, symbol in (("win", T.GOOD, "circle"), ("loss", T.CRITICAL, "x")):
        sel = [t for t in trades if (t.net_pnl > 0) == (outcome == "win")]
        if not sel:
            continue
        add(go.Scatter(
            x=[t.exit_bar for t in sel], y=[t.exit_price for t in sel], mode="markers", name=f"exit, {outcome}",
            marker=dict(symbol=symbol, size=9, color=color, line=dict(color=T.PAPER, width=1)),
            customdata=[[t.exit_reason, t.net_pnl, t.gross_pnl, t.bars_held, t.side] for t in sel],
            hovertemplate="<b>%{customdata[4]} exit (%{customdata[0]})</b> %{y:$,.2f}<br>net %{customdata[1]:+$,.2f}"
                          " · gross %{customdata[2]:+$,.2f}<br>held %{customdata[3]} bars<extra></extra>"), "price")

    # --- signals
    if has_sig:
        p = np.asarray(signals.p, float)[lo:hi]
        for i, h in enumerate(T.HORIZONS):
            add(go.Scatter(x=x, y=p[:, i], mode="lines", name=f"P(up) {h}", legendgroup=h,
                           line=dict(color=T.rgba(T.HORIZON_COLORS[h], 0.75), width=1),
                           hovertemplate=f"P(up) {h} %{{y:.3f}}<extra></extra>"), "signal")
        add(go.Scatter(x=x, y=np.asarray(signals.weighted_direction, float)[lo:hi], mode="lines",
                       name="weighted P(up)", line=dict(color=T.INK, width=1.5),
                       hovertemplate="weighted P(up) %{y:.3f}<extra></extra>"), "signal")
        seen = set()
        for field, label in _THRESHOLD_FIELDS:
            v = getattr(strategy, field, None) if strategy is not None else None
            if v is None or label in seen:
                continue
            seen.add(label)
            color = {"long entry": T.LONG_COLOR, "short entry": T.SHORT_COLOR}.get(label, T.NEUTRAL)
            add(go.Scatter(x=[lo, hi - 1], y=[v, v], mode="lines", name=f"{label} {v:.3f}",
                           line=dict(color=color, width=1, dash="dash"), hoverinfo="skip"), "signal")
        add(go.Scatter(x=x, y=np.asarray(signals.avg_confidence, float)[lo:hi], mode="lines", name="avg confidence",
                       line=dict(color=T.SERIES[4], width=1.2),
                       hovertemplate="confidence %{y:.3f}<extra></extra>"), "conf")
        add(go.Scatter(x=x, y=np.asarray(signals.strength, float)[lo:hi], mode="lines", name="signal strength",
                       line=dict(color=T.SERIES[3], width=1), fill="tozeroy", fillcolor=T.rgba(T.SERIES[3], 0.2),
                       hovertemplate="strength %{y:.3f}<extra></extra>"), "conf")
        sig1 = np.asarray(signals.sigma, float)[lo:hi, 1]
        add(go.Scatter(x=x, y=sig1, mode="lines", name="sigma h1", line=dict(color=T.HORIZON_COLORS["h1"], width=1),
                       hovertemplate="sigma h1 %{y:$,.1f}<extra></extra>"), "sigma")
        spikes = np.where(np.asarray(signals.var_spike, bool)[lo:hi])[0]
        if len(spikes):
            add(go.Scatter(x=x[spikes], y=sig1[spikes], mode="markers", name="variance spike",
                           marker=dict(symbol="diamond", size=6, color=T.WARNING),
                           hovertemplate="variance spike<extra></extra>"), "sigma")

    # --- equity and drawdown
    eq = np.asarray(result.equity, float)[1:][lo:hi]
    eqg = np.asarray(result.equity_gross, float)[1:][lo:hi]
    init = float(result.config.initial_equity)
    add(go.Scatter(x=x, y=eqg, mode="lines", name="equity before costs", line=dict(color=T.INK_2, width=1, dash="dot"),
                   hovertemplate="gross equity %{y:$,.0f}<extra></extra>"), "equity")
    add(go.Scatter(x=x, y=eq, mode="lines", name="equity after costs", line=dict(color=T.INK, width=2),
                   hovertemplate="net equity %{y:$,.0f}<extra></extra>"), "equity")
    if bars is not None:
        c = np.asarray(bars.close, float)
        bh = eq[0] * c[lo:hi] / c[lo]   # from the strategy's equity at the window start
        add(go.Scatter(x=x, y=bh, mode="lines", name="buy and hold", line=dict(color=T.NEUTRAL, width=1, dash="dash"),
                       hovertemplate="buy & hold %{y:$,.0f}<extra></extra>"), "equity")
    peak = np.maximum.accumulate(np.asarray(result.equity, float))[1:][lo:hi]
    dd = eq / peak - 1.0
    add(go.Scatter(x=x, y=dd, mode="lines", name="from the running peak", line=dict(color=T.CRITICAL, width=1), fill="tozeroy",
                   fillcolor=T.rgba(T.CRITICAL, 0.25), hovertemplate="drawdown %{y:.2%}<extra></extra>"), "dd")
    fig.update_layout({"yaxis" + ax["dd"][1:]: dict(tickformat=".1%")})

    s = result.summary
    subtitle = (f"{s['n_trades']} trades · net {100 * s['total_return']:+.2f}% · gross "
                f"{100 * s.get('gross_pnl', 0) / init:+.2f}% · costs ${s.get('costs_paid', 0):,.0f} · hit rate "
                f"{100 * (s.get('hit_rate') or 0):.1f}% net / {100 * (s.get('hit_rate_gross') or 0):.1f}% gross · "
                f"max DD {100 * s['max_drawdown']:.2f}% · bars {lo}-{hi - 1}")
    T.apply(fig, title=title or f"{result.strategy}", subtitle=subtitle, height=height, legend_top=False)
    fig.update_layout(hovermode="x unified", hoversubplots="axis", margin=dict(t=100),
                      xaxis=dict(title_text="bar (test block)", showspikes=True, spikemode="across"))
    return fig


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


# ------------------------------------------------------------------ registry entries (data, config)
def trading_dashboard(data, config=None, *, bars=None, signals=None, strategy=None, **kw):
    return trading_dashboard_figure(data, bars, signals, strategy, **kw)


def trade_analytics(data, config=None, *, bars=None, **kw):
    return trade_analytics_figure(data, bars, **kw)


def strategy_comparison(data, config=None, *, bars=None, **kw):
    return strategy_comparison_figure(data, bars, **kw)

"""Trading dashboards drawn from a BacktestResult (and the SignalFrame it traded on).

* :func:`trading_dashboard_figure` - six stacked panels on ONE time axis with synchronised hover
  (one vertical cursor, every panel's values in one tooltip): price with entries / exits, the
  take-profit and stop levels and win / loss connectors, holding periods shaded; the per-horizon
  P(up) and the strategy's entry lines; confidence and signal strength; predicted sigma with the
  variance spikes; equity (net, gross, buy-and-hold); drawdown.
* per-trade analytics and the strategy comparison live in :mod:`trade_analytics` (re-exported here).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from neural_trade.visualization import stats as S  # noqa: F401
from neural_trade.visualization import theme as T
from neural_trade.visualization.trade_analytics import (  # noqa: F401  (re-exported)
    excursions, strategy_comparison_figure, trade_analytics_figure,
)

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


# ------------------------------------------------------------------ registry entries (data, config)
def trading_dashboard(data, config=None, *, bars=None, signals=None, strategy=None, **kw):
    return trading_dashboard_figure(data, bars, signals, strategy, **kw)


def trade_analytics(data, config=None, *, bars=None, **kw):
    return trade_analytics_figure(data, bars, **kw)


def strategy_comparison(data, config=None, *, bars=None, **kw):
    return strategy_comparison_figure(data, bars, **kw)

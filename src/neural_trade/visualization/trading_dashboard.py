"""Trading dashboards drawn from a BacktestResult (and the SignalFrame it traded on).

:func:`trading_dashboard_figure` stacks panels on ONE time axis with one unified hover (a vertical
cursor, every panel's value at that bar in one tooltip):

1. price with the bar high-low band, entries (triangles) and exits (circle = won after costs,
   x = lost); on a short view also the holding periods, stop / take-profit levels and the
   entry-to-exit lines;
2. P(up) per horizon (faint) and the weighted P(up), with the strategy's entry / exit lines, the
   entry zones shaded and a triangle on each decision bar;
3. confidence and signal strength; 4. predicted sigma h1 with the variance spikes;
5. net P&L after costs, $ since the start of the view, with each exit marked on the curve;
6. before costs (the same trades with the costs added back) vs buy & hold, $ since the start of
   the view; 7. drawdown from the running peak since the start of the view.

Every number is for the bars shown. A window (``start`` / ``end``) is sliced, not zoomed: all y
axes fit the window, P&L restarts at $0 at the window start and the subtitle gives the window's
own result, then the whole block's on a separate line.

Timing (the engine's): a strategy decides at bar t's close and fills at bar t+1's open, so the
decision triangle sits one bar before the entry triangle.

Per-trade analytics and the strategy comparison live in :mod:`trade_analytics` (re-exported here).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from neural_trade.visualization import theme as T
from neural_trade.visualization.trade_analytics import (  # noqa: F401  (re-exported)
    excursions, strategy_comparison_figure, trade_analytics_figure,
)

# Trade detail (holding periods, stop / take-profit levels, entry-to-exit lines) is drawn when the
# view is at most this many bars: at ~1300 px that is >= 1.6 px per bar, so a 10-bar trade is ~16 px.
DETAIL_MAX_BARS = 800
# Longer views aggregate the bar high-low band into about this many buckets (max high / min low).
_BAND_POINTS = 1500

# Fixed lines a strategy exposes: (field, the series it applies to, role, label drawn at the line).
_THRESHOLDS = (
    ("long_above", "weighted", "long", "long > {v:.3f}"),
    ("short_below", "weighted", "short", "short < {v:.3f}"),
    ("median", "weighted", "exit", "exit {v:.3f}"),
    ("p_long", "h1", "long", "long > {v:.3f}"),
    ("p_short", "h1", "short", "short < {v:.3f}"),
    ("exit_long_below", "h1", "exit", "exit L < {v:.3f}"),
    ("exit_short_above", "h1", "exit", "exit S > {v:.3f}"),
)
_RIGHT_MARGIN = 104     # room for the labels at the right end of the lines
# The P(up) axis fits the traded series and the strategy's lines (the decision band). The horizon
# lines may widen it by at most this much on each side, so one multi-bar spike of a single horizon
# does not squeeze the band; the horizon points left outside are counted in the subtitle.
_PUP_REACH = 0.05
_PUP_PAD = 0.02
# A $ panel whose data span less than this spans at least this (whole-dollar ticks, a flat line mid-panel).
_MIN_USD_SPAN = 20.0
# The shallowest drawdown axis: a flat or tiny drawdown is drawn on 0 to -1%, not stretched to fill the panel.
_MIN_DD_SPAN = 0.01


# ------------------------------------------------------------------ small helpers
def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _usd(v, *, signed: bool = False, decimals: int = 0) -> str:
    """'$1,234' / '-$1,234' / '+$1,234' (formatted here: plotly.js rejects d3 formats like '+$,.2f')."""
    if v is None or not np.isfinite(v):
        return "n/a"
    body = f"${abs(v):,.{decimals}f}"
    if round(abs(v), decimals) == 0:
        return body
    return ("-" if v < 0 else "+" if signed else "") + body


def _pct(v, decimals: int = 2) -> str:
    """'+1.23%' / '-1.23%'; a value that rounds to zero prints '0.00%' (no sign, never '-0.00%')."""
    if v is None or not np.isfinite(v):
        return "n/a"
    if round(100 * v, decimals) == 0:
        return f"{0:.{decimals}f}%"
    return f"{100 * v:+.{decimals}f}%"


def _count(k: int, word: str) -> str:
    """'1 trade' / '2 trades' / '0 trades'."""
    return f"{k} {word}{'' if k == 1 else 's'}"


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


def _usd_range(values, min_span: float = _MIN_USD_SPAN):
    """[lo, hi] a $ panel's autorange must include when its data (and $0) span less than ``min_span``:
    centred on the data, ``min_span`` wide, so the '$,.0f' ticks are whole, distinct dollars (a flat
    series would get plotly's +/-$1 and print '$1, $1, $0, -$1, -$1'). None when the data span more."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lo_v, hi_v = min(float(v.min()) if len(v) else 0.0, 0.0), max(float(v.max()) if len(v) else 0.0, 0.0)
    if hi_v - lo_v >= min_span:
        return None
    mid = (hi_v + lo_v) / 2
    return [mid - min_span / 2, mid + min_span / 2]


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
            font=dict(size=10, color=T.INK_2), itemwidth=30, itemsizing="constant", groupclick="togglegroup")})
        refs.append("y" if i == 0 else f"y{i + 1}")
        top = dom[0] - gap / total
    fig.update_layout(xaxis=dict(anchor=refs[-1], domain=[0, 1]))
    return refs


def _trade_rows(trades, lo, hi):
    """Trades that overlap bars [lo, hi): opened before the window and closed inside it included."""
    return [t for t in trades if t.exit_bar >= lo and t.entry_bar < hi]


def _levels_text(t) -> str:
    parts = [f"{k} {_usd(v, decimals=2)} ({100 * (v / t.entry_price - 1):+.2f}%)"
             for k, v in (("take-profit", t.tp), ("stop", t.sl)) if _finite(v)]
    return " · ".join(parts) or "no take-profit or stop"


def _end_labels(fig, yref: str, items, span: float):
    """Label lines at the right edge of the plot (outside it, in the right margin), nudged apart."""
    items = sorted((it for it in items if np.isfinite(it[0])), key=lambda it: it[0])
    if not items:
        return
    gap = 0.16 * (span if span > 0 else 1.0)
    ys = [items[0][0]]
    for y, _ in items[1:]:
        ys.append(max(y, ys[-1] + gap))
    shift = (ys[-1] - items[-1][0]) / 2 if len(ys) > 1 else 0.0     # centre the nudged cluster
    for (_, text), y in zip(items, ys):
        fig.add_annotation(xref="x domain", x=1.0, xanchor="left", xshift=6, yref=yref, y=y - shift, text=text,
                           showarrow=False, yanchor="middle", align="left", font=dict(size=10, color=T.INK_2))


def _wrap(clauses, width: int = 160):
    """Join clauses with ' · ' into lines of at most ~``width`` characters."""
    lines, cur = [], ""
    for c in clauses:
        if cur and len(cur) + 3 + len(c) > width:
            lines.append(cur)
            cur = c
        else:
            cur = f"{cur} · {c}" if cur else c
    return lines + ([cur] if cur else [])


def _hline(fig, yref: str, y: float, color: str, dash: str, width: float = 1.0, layer: str = "above"):
    fig.add_shape(type="line", xref="x domain", x0=0, x1=1, yref=yref, y0=y, y1=y, layer=layer,
                  line=dict(color=color, width=width, dash=dash))


def _band(values, lo, hi, k, fn):
    v = np.asarray(values, float)[lo:hi]
    if k <= 1:
        return v
    pad = (-len(v)) % k
    v = np.concatenate([v, np.full(pad, np.nan)]).reshape(-1, k)
    return fn(v, axis=1)


def _pup_range(pv, wv, line_values, *, reach: float = _PUP_REACH, pad: float = _PUP_PAD):
    """(y_lo, y_hi, n_clipped) of the P(up) axis. The core is the traded (weighted) series and the
    strategy's lines, all always inside. The horizons widen it to their 0.2% / 99.8% quantiles, by at
    most ``reach`` beyond the core on each side, so a spike of one horizon cannot squeeze the decision
    band; ``n_clipped`` counts the horizon points left outside. (None, None, 0) with no finite data."""
    pv = np.asarray(pv, float)
    wv = np.asarray(wv, float)
    core = np.concatenate([wv[np.isfinite(wv)], np.asarray(line_values, float).ravel()])
    hz = pv[np.isfinite(pv)]
    if not len(core) and not len(hz):
        return None, None, 0
    if len(hz):
        q_lo, q_hi = np.quantile(hz, [0.002, 0.998])
    if not len(core):
        y_lo, y_hi = q_lo, q_hi
    else:
        c_lo, c_hi = float(core.min()), float(core.max())
        y_lo = max(min(q_lo, c_lo), c_lo - reach) if len(hz) else c_lo
        y_hi = min(max(q_hi, c_hi), c_hi + reach) if len(hz) else c_hi
    y_lo, y_hi = float(y_lo) - pad, float(y_hi) + pad
    with np.errstate(invalid="ignore"):
        clipped = int(np.sum((pv < y_lo) | (pv > y_hi)))
    return y_lo, y_hi, clipped


def detail_window(result, n_bars: int = 600, *, around: str = "steepest_fall"):
    """(start, end) of an ``n_bars`` window worth reading in detail: ``"steepest_fall"`` - where net
    equity fell most over ``n_bars`` bars; ``"worst_trade"`` - centred on the entry of the trade
    with the largest net loss; ``"last"`` - the last ``n_bars`` bars."""
    if around not in ("steepest_fall", "worst_trade", "last"):
        raise ValueError(f"around must be 'steepest_fall', 'worst_trade' or 'last', got {around!r}")
    n = len(result.position)
    k = max(1, min(int(n_bars), n))
    if around == "last" or n <= k:
        return n - k, n
    if around == "worst_trade" and result.trades:
        t = min(result.trades, key=lambda t: t.net_pnl)
        s = int(np.clip(t.entry_bar - k // 2, 0, n - k))
        return s, s + k
    E = np.asarray(result.equity, float)
    s = int(np.argmin(E[k:] - E[:-k]))            # E[s + k] - E[s]: the change over bars [s, s + k)
    return s, s + k


# ------------------------------------------------------------------ the dashboard
def trading_dashboard_figure(result, bars=None, signals=None, strategy=None, *, start: Optional[int] = None,
                             end: Optional[int] = None, title: Optional[str] = None, height: Optional[int] = None,
                             config=None, times=None, detail_max_bars: int = DETAIL_MAX_BARS):
    """``result``: BacktestResult; ``bars``: its Bars; ``signals``: its SignalFrame; ``strategy``:
    the Strategy instance (its entry / exit lines go on the P(up) panel); ``start``/``end``: the
    bars to show, [start, end) (default: the whole block; the data are sliced so every y axis fits
    the window); ``config``: the run's Config (horizon lengths in the labels); ``times``: one
    timestamp per bar of the block (the window's time span goes in the subtitle, and on a view of
    at most ``detail_max_bars`` bars into the hover); ``detail_max_bars``: the longest view that
    still draws holding periods, stop / take-profit levels and entry-to-exit lines."""
    import plotly.graph_objects as go

    n = len(result.position)
    lo = max(0, int(start or 0))
    hi = min(n, int(end) if end is not None else n)
    if hi <= lo:
        raise ValueError(f"empty window: start={start}, end={end} on a block of {n} bars")
    nv = hi - lo
    whole = (lo, hi) == (0, n)
    detail = nv <= detail_max_bars
    x = np.arange(lo, hi)
    has_sig = signals is not None
    has_bars = bars is not None
    E = np.asarray(result.equity, float)          # [n + 1]: E[i] = equity before bar i (after bar i - 1's close)
    G = np.asarray(result.equity_gross, float)
    init = float(result.config.initial_equity)
    since = "since the start" if whole else f"since bar {lo}"

    # ---------------------------------------------------------------- trades in the view
    trades = _trade_rows(result.trades, lo, hi)
    entered = [t for t in trades if lo <= t.entry_bar < hi]
    exited = [t for t in trades if lo <= t.exit_bar < hi]
    open_at_start = [t for t in trades if t.entry_bar < lo]
    has_tp = any(_finite(t.tp) for t in trades)
    has_sl = any(_finite(t.sl) for t in trades)
    levels = {(True, True): ", take-profit / stop levels", (False, True): ", stop levels",
              (True, False): ", take-profit levels"}.get((has_tp, has_sl), "") if detail else ""

    # ---------------------------------------------------------------- strategy lines on the P(up) panel
    lines = []          # (series, role, value, label)
    if has_sig and strategy is not None:
        for field, series, role, label in _THRESHOLDS:
            v = getattr(strategy, field, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                lines.append((series, role, float(v), label.format(v=v)))
    line_series = lines[0][0] if lines else "weighted"
    decide_series = "h1" if line_series == "h1" else "weighted"
    hl = lambda h: T.horizon_label(h, config)  # noqa: E731
    steps = getattr(config, "HORIZON_STEPS", None) if config is not None else None
    hz = "h0-h2" + (f" ({'/'.join(str(s) for s in steps[:3])} bars)" if steps is not None else "")
    sig_title = f"P(up): {hz}, weighted (white)"
    if lines:
        sig_title += "; lines on the weighted" if line_series == "weighted" else "; lines on h1"
    elif strategy is not None and hasattr(strategy, "base_entry_threshold"):
        sig_title += "; the entry line varies per bar"

    heights = [0.30] + ([0.15, 0.08, 0.08] if has_sig else []) + [0.13, 0.10, 0.08]
    panels = ["price"] + (["signal", "conf", "sigma"] if has_sig else []) + ["equity", "pre", "dd"]
    whole_dd_at_lo = E[lo] / np.max(E[:lo + 1]) - 1.0
    titles = {
        "price": f"Price and trades{levels}" if trades else "Price (no trades in this view)",
        "signal": sig_title,
        "conf": "Confidence and signal strength (0 to 1)",
        "sigma": f"Predicted sigma {hl('h1')}, $",
        "equity": f"Net P&L after costs, $ {since} (equity {_usd(E[lo])} at the start)",
        "pre": f"Before costs (same trades, costs added back){' vs buy & hold' if has_bars else ''}, $ {since}",
        "dd": ("Drawdown from the running peak" if whole else
               f"Drawdown from the running peak since bar {lo} (whole-block drawdown at bar {lo}: "
               f"{100 * whole_dd_at_lo:.1f}%)"),
    }
    fig = go.Figure()
    refs = _stack(fig, heights, [titles[p] for p in panels])
    ax = dict(zip(panels, refs))

    def add(trace, panel):
        ref = ax[panel]
        trace.update(xaxis="x", yaxis=ref, legend="legend" + ref[1:])
        fig.add_trace(trace)

    def yaxis(panel, **kw):
        fig.update_layout({"yaxis" + ax[panel][1:]: kw})

    # ---------------------------------------------------------------- per-bar hover text on the close
    events = {}

    def event(bar, text):
        if lo <= bar < hi:
            events.setdefault(bar, []).append(text)

    p = np.asarray(signals.p, float) if has_sig else None
    w = np.asarray(signals.weighted_direction, float) if has_sig else None
    thr = {(s_, r): v for s_, r, v, _ in lines}
    for d in (getattr(result, "decisions", None) or ()) if detail else ():   # long views: the triangles only
        bar, side = int(d["bar"]), d["side"]
        if has_sig:
            val = p[bar, 1] if decide_series == "h1" else w[bar]
            name = "P(up) h1" if decide_series == "h1" else "weighted P(up)"
            line = thr.get((decide_series, "long" if side == "LONG" else "short"))
            cmp = f" vs line {line:.3f}" if line is not None else ""
            event(bar, f"<b>decided {side}</b> at this close: {name} {val:.3f}{cmp}; fills at the next open")
        else:
            event(bar, f"<b>decided {side}</b> at this close ({d.get('reason', '')}); fills at the next open")
    if detail:
        for t in entered:
            event(t.entry_bar, f"<b>{t.side} entry</b> {_usd(t.entry_price, decimals=2)} · size {_usd(t.notional)}"
                               f" · {_levels_text(t)}")
        for t in exited:
            event(t.exit_bar, f"<b>{t.side} exit ({t.exit_reason})</b> {_usd(t.exit_price, decimals=2)} · net "
                              f"{_usd(t.net_pnl, signed=True, decimals=2)} · before costs "
                              f"{_usd(t.gross_pnl, signed=True, decimals=2)} · held {t.bars_held} bars "
                              f"(entered bar {t.entry_bar})")
    else:   # a long view: one compact line per trade, on its exit bar (keeps the figure small)
        for t in exited:
            event(t.exit_bar, f"<b>{t.side} exit ({t.exit_reason})</b>, entered bar {t.entry_bar}: net "
                              f"{_usd(t.net_pnl, signed=True, decimals=2)} · before costs "
                              f"{_usd(t.gross_pnl, signed=True, decimals=2)}")
    tstr = None
    if times is not None:
        import pandas as pd

        tt = pd.to_datetime(np.asarray(times)[:n])
        tstr = [f"{v:%Y-%m-%d %H:%M}" for v in tt]

    # ---------------------------------------------------------------- 1. price
    if has_bars:
        k = 1 if nv <= 2 * _BAND_POINTS else int(np.ceil(nv / _BAND_POINTS))
        band = dict(x0=lo + (k - 1) / 2, dx=k, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        add(go.Scatter(y=_f32(_band(bars.high, lo, hi, k, np.nanmax)), name="bar high", **band), "price")
        add(go.Scatter(y=_f32(_band(bars.low, lo, hi, k, np.nanmin)), name="bar high-low range", fill="tonexty",
                       fillcolor=T.rgba(T.INK_2, 0.13), **band), "price")
        text = []
        for i in range(lo, hi):
            s = (f" · {tstr[i]}" if (tstr is not None and detail) else "")
            s += "".join("<br>" + e for e in events.get(i, ()))
            text.append(s)
        add(go.Scatter(x0=lo, dx=1, y=_f32(np.asarray(bars.close, float)[lo:hi]), mode="lines", name="close",
                       line=dict(color=T.INK_2, width=1.2), text=text if any(text) else None,
                       hovertemplate="close %{y:$,.2f}" + ("%{text}" if any(text) else "") + "<extra></extra>"),
            "price")
        hi_px, lo_px = np.nanmax(bars.high[lo:hi]), np.nanmin(bars.low[lo:hi])
        pad = 0.05 * max(hi_px - lo_px, 1e-9)
        # stops far from price would stretch the axis: the axis fits the bars (levels stay in the hover)
        yaxis("price", autorangeoptions=dict(clipmin=float(lo_px - pad), clipmax=float(hi_px + pad)))
    yaxis("price", tickformat="$,.0f")
    if detail:
        for t in trades:   # holding periods: the position is exposed from the entry open to the exit
            fig.add_shape(type="rect", xref="x", yref=f"{ax['price']} domain", x0=t.entry_bar - 0.5,
                          x1=max(t.exit_bar - 0.5, t.entry_bar + 0.5), y0=0, y1=1, line_width=0, layer="below",
                          fillcolor=T.rgba(T.LONG_COLOR if t.side == "LONG" else T.SHORT_COLOR, 0.12))
        # take-profit: long light dashes; stop: short grey dashes of equal gap (never dotted: dotted
        # means training). With both kinds in the view each segment on the axis also carries its name,
        # at its start, on the side facing the entry price (inside the panel even at its edge).
        y_in = (lo_px - pad, hi_px + pad) if has_bars else (-np.inf, np.inf)
        for level, name, tag, color, dash in (("tp", "take-profit level", "TP", T.INK_2, "14px,4px"),
                                              ("sl", "stop level", "stop", T.MUTED, "6px,6px")):
            xs, ys, lab = [], [], []
            for t in trades:
                v = getattr(t, level)
                if _finite(v):
                    xs += [t.entry_bar, t.exit_bar, None]
                    ys += [v, v, None]
                    if y_in[0] <= v <= y_in[1]:
                        lab.append((max(t.entry_bar, lo), v, "bottom right" if v > t.entry_price else "top right"))
            if xs:
                add(go.Scatter(x=xs, y=ys, mode="lines", name=name, legendgroup=f"lvl-{level}",
                               line=dict(color=color, width=1.3, dash=dash), hoverinfo="skip"), "price")
            if lab and has_tp and has_sl:
                add(go.Scatter(x=[a for a, _, _ in lab], y=_f32([b for _, b, _ in lab]), mode="text",
                               text=[tag] * len(lab), textposition=[c for _, _, c in lab], name=f"{name} label",
                               legendgroup=f"lvl-{level}", showlegend=False, hoverinfo="skip",
                               textfont=dict(size=9, color=color)), "price")
    outcomes = (("win", "exit, won after costs", T.GOOD, "circle"), ("loss", "exit, lost after costs", T.CRITICAL, "x"))
    for outcome, _, color, _ in outcomes:   # entry-to-exit lines, one legend key with the exit marker
        sel = [t for t in trades if (t.net_pnl > 0) == (outcome == "win")]
        if detail and sel:
            xs, ys = [], []
            for t in sel:
                xs += [t.entry_bar, t.exit_bar, None]
                ys += [t.entry_price, t.exit_price, None]
            add(go.Scatter(x=xs, y=ys, mode="lines", name=f"{outcome} (entry to exit)", legendgroup=f"trade-{outcome}",
                           showlegend=False, line=dict(color=T.rgba(color, 0.7), width=1.3), hoverinfo="skip"),
                "price")
    size_in, size_out = (11, 9) if detail else (7, 6)
    marker_hover = dict(hoverinfo="skip") if has_bars else {}
    for side, color, symbol in (("LONG", T.LONG_COLOR, "triangle-up"), ("SHORT", T.SHORT_COLOR, "triangle-down")):
        sel = [t for t in entered if t.side == side]
        if sel:
            hv = marker_hover or dict(hovertemplate=f"<b>{side} entry</b> at bar %{{x}}: %{{y:$,.2f}}<extra></extra>")
            add(go.Scatter(x=[t.entry_bar for t in sel], y=_f32([t.entry_price for t in sel]), mode="markers",
                           name=f"{side.lower()} entry", legendgroup=f"entry-{side}",
                           marker=dict(symbol=symbol, size=size_in, color=color, line=dict(color=T.PAPER, width=1)),
                           **hv), "price")
    for outcome, name, color, symbol in outcomes:
        sel = [t for t in exited if (t.net_pnl > 0) == (outcome == "win")]
        if sel:
            hv = marker_hover or dict(hovertemplate="<b>exit</b> at bar %{x}: %{y:$,.2f}<extra></extra>")
            add(go.Scatter(x=[t.exit_bar for t in sel], y=_f32([t.exit_price for t in sel]), mode="markers", name=name,
                           legendgroup=f"trade-{outcome}", opacity=1.0 if detail else 0.85,
                           marker=dict(symbol=symbol, size=size_out, color=color, line=dict(color=T.PAPER, width=1)),
                           **hv), "price")
    if not has_bars and not trades:
        T.note_on_empty(fig, "no bars given and no trades in this view")

    # ---------------------------------------------------------------- 2-4. signals
    if has_sig:
        pv, wv = p[lo:hi], w[lo:hi]
        for role, color in (("long", T.LONG_COLOR), ("short", T.SHORT_COLOR)):   # entry zones
            for _series, r, v, _ in lines:
                if r == role:
                    fig.add_shape(type="rect", xref="x domain", x0=0, x1=1, yref=ax["signal"],
                                  y0=v, y1=1.0 if role == "long" else 0.0, line_width=0, layer="below",
                                  fillcolor=T.rgba(color, 0.12))
        for _series, role, v, _ in lines:          # exit lines go under the traces, entry lines over them
            if role == "exit":
                _hline(fig, ax["signal"], v, T.NEUTRAL, "8px,4px", 1.2, layer="below")
        if detail:
            for i, h in enumerate(T.HORIZONS):
                add(go.Scatter(x0=lo, dx=1, y=_f32(pv[:, i]), mode="lines", name=h, legendgroup=h,
                               line=dict(color=T.rgba(T.HORIZON_COLORS[h], 0.55), width=1),
                               hovertemplate=f"P(up) {hl(h)} %{{y:.3f}}<extra></extra>"), "signal")
        else:   # a long view cannot resolve three lines per bar: the range across the horizons (per bucket)
            k = int(np.ceil(nv / _BAND_POINTS))
            rng = dict(x0=lo + (k - 1) / 2, dx=k, mode="lines", line=dict(width=0), legendgroup="h-range")
            over = "" if k == 1 else f" over {k} bars"
            band_name = ("h0-h2 range (min-max across horizons)" if k == 1 else
                         f"h0-h2 range (min-max across horizons and {k}-bar buckets)")
            add(go.Scatter(y=_f32(_band(np.nanmax(pv, 1), 0, nv, k, np.nanmax)), name="h0-h2 max", showlegend=False,
                           hovertemplate=f"h0-h2 max{over} %{{y:.3f}}<extra></extra>", **rng), "signal")
            add(go.Scatter(y=_f32(_band(np.nanmin(pv, 1), 0, nv, k, np.nanmin)), name=band_name,
                           fill="tonexty", fillcolor=T.rgba(T.INK_2, 0.28),
                           hovertemplate=f"h0-h2 min{over} %{{y:.3f}}<extra></extra>", **rng), "signal")
        add(go.Scatter(x0=lo, dx=1, y=_f32(wv), mode="lines", name="weighted", line=dict(color=T.INK, width=1.5),
                       hovertemplate="weighted P(up) %{y:.3f}<extra></extra>"), "signal")
        for _series, role, v, _ in lines:
            if role != "exit":
                _hline(fig, ax["signal"], v, T.INK_2, "6px,3px", 1.5)
        # the decision bar (the close the strategy decided at; the fill is the next bar's open)
        dser = p[:, 1] if decide_series == "h1" else w
        dec = [(int(d["bar"]), d["side"]) for d in (getattr(result, "decisions", None) or ())
               if lo <= int(d["bar"]) < hi]
        for side, color, symbol in (("LONG", T.LONG_COLOR, "triangle-up"), ("SHORT", T.SHORT_COLOR, "triangle-down")):
            bars_ = [b for b, s_ in dec if s_ == side]
            if bars_:   # one key per side (own groups: a horizontal legend stacks a group's items)
                add(go.Scatter(x=bars_, y=_f32(dser[bars_]), mode="markers", name=f"decided {side.lower()}",
                               legendgroup=f"decision-{side}",
                               marker=dict(symbol=symbol, size=9 if detail else 6, color=color,
                                           line=dict(color=T.PAPER, width=1)), hoverinfo="skip"), "signal")
        y_lo, y_hi, clipped = _pup_range(pv, wv, [v for _, _, v, _ in lines])
        if y_lo is not None:
            yaxis("signal", range=[y_lo, y_hi])
            _end_labels(fig, ax["signal"], [(v, lab) for _, _, v, lab in lines], y_hi - y_lo)
        yaxis("signal", tickformat=".2f")

        add(go.Scatter(x0=lo, dx=1, y=_f32(np.asarray(signals.avg_confidence, float)[lo:hi]), mode="lines",
                       name="avg confidence", line=dict(color=T.SERIES[4], width=1.2),
                       hovertemplate="confidence %{y:.3f}<extra></extra>"), "conf")
        add(go.Scatter(x0=lo, dx=1, y=_f32(np.asarray(signals.strength, float)[lo:hi]), mode="lines",
                       name="signal strength", line=dict(color=T.INK_2, width=1), fill="tozeroy",
                       fillcolor=T.rgba(T.INK_2, 0.15), hovertemplate="strength %{y:.3f}<extra></extra>"), "conf")
        sig1 = np.asarray(signals.sigma, float)[lo:hi, 1]
        spike = np.asarray(signals.var_spike, bool)[lo:hi]
        spikes = np.where(spike)[0]
        add(go.Scatter(x0=lo, dx=1, y=_f32(sig1), mode="lines", name="sigma h1",
                       line=dict(color=T.HORIZON_COLORS["h1"], width=1),
                       text=np.where(spike, " · variance spike", "").tolist() if len(spikes) else None,
                       hovertemplate=f"sigma {hl('h1')} %{{y:$,.1f}}" + ("%{text}" if len(spikes) else "")
                                     + "<extra></extra>"), "sigma")
        if len(spikes):   # the flag is in the sigma hover of that exact bar, not on a nearby marker
            add(go.Scatter(x=x[spikes], y=_f32(sig1[spikes]), mode="markers", name="variance spike",
                           marker=dict(symbol="diamond", size=6, color=T.WARNING), hoverinfo="skip"), "sigma")
        yaxis("sigma", tickformat="$,.0f")

    # ---------------------------------------------------------------- 5. net P&L since the start of the view
    e_w, g_w = E[lo + 1:hi + 1], G[lo + 1:hi + 1]
    net = e_w - E[lo]
    pre = g_w - G[lo]
    _hline(fig, ax["equity"], 0.0, T.NEUTRAL, "4px,3px", 1.0, layer="below")
    add(go.Scatter(x0=lo, dx=1, y=_f32(net), mode="lines", name="net P&L", line=dict(color=T.INK, width=2),
                   hovertemplate=f"net P&L {since} %{{y:$,.0f}}<extra></extra>"), "equity")
    for outcome, _, color, symbol in outcomes:
        sel = [t for t in exited if (t.net_pnl > 0) == (outcome == "win")]
        if sel:
            add(go.Scatter(x=[t.exit_bar for t in sel], y=_f32([E[t.exit_bar + 1] - E[lo] for t in sel]),
                           mode="markers", name=f"exit, {outcome}", legendgroup=f"trade-{outcome}", showlegend=False,
                           marker=dict(symbol=symbol, size=8 if detail else 5, color=color,
                                       line=dict(color=T.PAPER, width=1)), hoverinfo="skip"), "equity")
    yaxis("equity", tickformat="$,.0f")
    r_eq = _usd_range(net)
    if r_eq is not None:        # e.g. no trades: a flat $0 line mid-panel, whole-dollar ticks
        yaxis("equity", autorangeoptions=dict(include=r_eq))
    span_eq = r_eq[1] - r_eq[0] if r_eq is not None else float(np.nanmax(net) - np.nanmin(net))
    _end_labels(fig, ax["equity"], [(net[-1], f"net {_usd(net[-1], signed=True)}")], span_eq)

    # ---------------------------------------------------------------- 6. before costs vs buy & hold
    _hline(fig, ax["pre"], 0.0, T.NEUTRAL, "4px,3px", 1.0, layer="below")
    add(go.Scatter(x0=lo, dx=1, y=_f32(pre), mode="lines", name="strategy before costs",
                   line=dict(color=T.INK_2, width=1.5),
                   hovertemplate=f"before costs {since} %{{y:$,.0f}}<extra></extra>"), "pre")
    ends = [(pre[-1], f"before costs {_usd(pre[-1], signed=True)}")]
    bh_ret = np.nan
    if has_bars:
        c = np.asarray(bars.close, float)
        p0 = c[max(lo - 1, 0)]      # the close before the view (bar 0's close for the whole block)
        bh = E[lo] * (c[lo:hi] / p0 - 1.0)       # the equity at the start of the view, held, no costs
        bh_ret = c[hi - 1] / p0 - 1.0
        add(go.Scatter(x0=lo, dx=1, y=_f32(bh), mode="lines", name="buy & hold (no costs)",
                       line=dict(color=T.NEUTRAL, width=1.5, dash="6px,3px"),
                       hovertemplate=f"buy & hold {since} %{{y:$,.0f}} ({_usd(E[lo])} held, no costs)<extra></extra>"),
            "pre")
        ends.append((bh[-1], f"buy & hold {_usd(bh[-1], signed=True)}"))
    yaxis("pre", tickformat="$,.0f")
    allpre = np.concatenate([pre] + ([bh] if has_bars else []))
    r_pre = _usd_range(allpre)
    if r_pre is not None:
        yaxis("pre", autorangeoptions=dict(include=r_pre))
    span_pre = r_pre[1] - r_pre[0] if r_pre is not None else float(np.nanmax(allpre) - np.nanmin(allpre))
    _end_labels(fig, ax["pre"], ends, span_pre)

    # ---------------------------------------------------------------- 7. drawdown from the running peak in the view
    peak = np.maximum.accumulate(E[lo:hi + 1])[1:]      # includes the equity at the start of the view
    dd = e_w / peak - 1.0
    max_dd = max(0.0, -float(np.nanmin(dd)))            # 0.0, never -0.0, when there is no drawdown
    add(go.Scatter(x0=lo, dx=1, y=_f32(dd), mode="lines", name="drawdown",
                   line=dict(color=T.CRITICAL, width=1), fill="tozeroy", fillcolor=T.rgba(T.CRITICAL, 0.25),
                   hovertemplate="drawdown %{y:.2%}<extra></extra>"), "dd")
    yaxis("dd", tickformat=".1%")
    if max_dd < _MIN_DD_SPAN:   # a flat or tiny drawdown is drawn on 0 to -1%, not stretched over the panel
        # (a fixed range: plotly autoranges an all-zero series to +/-1 even with a small
        # autorangeoptions.include); the top sits a little above 0 so the line at 0 is not hidden on
        # the panel's edge
        yaxis("dd", range=[-1.05 * _MIN_DD_SPAN, 0.03 * _MIN_DD_SPAN])

    # ---------------------------------------------------------------- subtitle: the view's numbers first
    e0 = E[lo]
    net_ret, pre_ret = (E[hi] - e0) / e0, (G[hi] - G[lo]) / e0
    costs = (G[hi] - G[lo]) - (E[hi] - e0)
    wins = sum(t.net_pnl > 0 for t in entered)
    wins_g = sum(t.gross_pnl > 0 for t in entered)
    n_sl = sum(t.exit_reason == "SL" for t in exited)
    stop_d = [abs(t.sl - t.entry_price) for t in entered if _finite(t.sl)]
    span = f"bars {lo}-{hi - 1}" + (f" ({tstr[lo]} to {tstr[hi - 1]})" if tstr is not None else "")
    scope = f"whole test block, {span}" if whole else f"window {span}, {nv} bars"
    groups = [[f"{scope}: {_count(len(entered), 'trade')}{'' if whole else ' entered'}",
               f"net {_pct(net_ret)} ({_usd(E[hi] - e0, signed=True)})", f"before costs {_pct(pre_ret)}",
               f"costs {_usd(costs)}"] + ([f"buy & hold {_pct(bh_ret)} (no costs)"] if has_bars else [])
              + [f"max DD {100 * max_dd:.2f}%"]
              + ([] if whole else [f"net and before-costs % are of the window's starting equity {_usd(e0)}"])]
    if not whole:
        s = result.summary
        groups.append([f"whole test block: {_count(int(s['n_trades']), 'trade')}", f"net {_pct(s['total_return'])}",
                       f"before costs {_pct(s.get('gross_pnl', 0) / init)}", f"costs {_usd(s.get('costs_paid', 0))}",
                       f"max DD {100 * max(0.0, float(s['max_drawdown'])):.2f}%"])
    third = []
    if entered:
        third.append(f"{wins} of {_count(len(entered), 'trade')} won after costs, {wins_g} before costs")
    if stop_d:
        third.append(f"{n_sl} of {_count(len(exited), 'exit')} on the stop (median stop "
                     f"{_usd(float(np.median(stop_d)))} from entry)")
    if open_at_start:
        third.append(f"{_count(len(open_at_start), 'trade')} open at the window start")
    if has_sig and clipped:
        third.append(f"the P(up) axis leaves out {clipped} of {pv.size:,} horizon points (hover shows them)")
    if not detail:
        third.append(f"holding periods, stop / take-profit levels and entry-to-exit lines are drawn on views of "
                     f"{detail_max_bars} bars or fewer (start= / end=)")
    groups.append(third)
    lines_sub = [ln for g in groups for ln in _wrap(g)]
    T.apply(fig, title=title or f"{result.strategy}", subtitle="<br>".join(lines_sub),
            height=height or (1500 if has_sig else 950), legend_top=False)
    bar_min = getattr(result.config, "bar_minutes", 1.0)
    fig.update_layout(hovermode="x unified", hoversubplots="axis", margin=dict(t=76 + 16 * len(lines_sub),
                                                                              r=_RIGHT_MARGIN),
                      xaxis=dict(title_text=f"bar of the test block ({bar_min:g}-minute bars)", showspikes=True,
                                 spikemode="across", range=[lo - 0.5, hi - 0.5], autorange=False))
    return fig


# ------------------------------------------------------------------ registry entries (data, config)
def trading_dashboard(data, config=None, *, bars=None, signals=None, strategy=None, **kw):
    kw.setdefault("config", config)
    return trading_dashboard_figure(data, bars, signals, strategy, **kw)


def trade_analytics(data, config=None, *, bars=None, **kw):
    return trade_analytics_figure(data, bars, **kw)


def strategy_comparison(data, config=None, *, bars=None, **kw):
    return strategy_comparison_figure(data, bars, **kw)

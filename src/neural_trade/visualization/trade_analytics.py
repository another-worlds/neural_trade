"""Per-trade analytics and the strategy comparison, from BacktestResults.

* :func:`trade_analytics_figure` - nine panels, each with its own key and a one-line readout under
  its title: the return per trade before / after costs, the cost drag, net return by exit reason,
  cumulative P&L, holding time, each trade's best / worst move while it was open (MFE / MAE, from
  the bars' high / low over the bars the position was actually exposed to), gross return by the
  model's conviction at the decision bar, long vs short, and the predicted vs realised move.
* :func:`strategy_comparison_figure` - several strategies' equity curves and their return before
  and after costs, each next to random entries with the SAME trade rate, holding time and size.
* :func:`excursions` - (MFE, MAE) per trade, in % of the entry mid.

Units: per-trade panels are in % of the trade's notional (the engine sizes each trade as a share
of the equity at entry, so $ P&L mixes trade sizes); totals (cumulative P&L, long vs short) are in $,
because summed $ P&L is the equity change.

Hover formats: a signed d3 spec must not START with '+' (plotly prefixes it with '~', which d3
rejects, and the raw float is printed): put an align char first, ``>+.2f``, never a bare ``+.2f``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T

# Exits at the exit bar's CLOSE (EOW: the whole bar was held). SL / TP fill INSIDE the exit bar (the
# order of its high and low around the fill is unknown), and every other reason (REV, REV_H1, TIME,
# TP1, INCOH, SPIKE, ...) is flagged at a close and filled at the exit bar's OPEN: of those exit bars
# only the open and the exit mid count.
_CLOSE_EXITS = ("EOW",)

# Strategy colours. Model strategies take the non-horizon categorical slots that carry no status
# meaning: pink, violet, amber (not the green that reads as "good", not the red #e66767 next to
# CRITICAL, not the grey #8a8984 next to MUTED / NEUTRAL). Baselines are drawn in neutral inks.
# Past three model strategies the colours repeat with a dashed, then a dash-dot line.
_STRATEGY_COLORS = (T.OTHER_SERIES[1], T.OTHER_SERIES[3], T.OTHER_SERIES[0])
_BASELINE_COLORS = {"buy_and_hold": T.NEUTRAL, "random_signal": T.INK_2}
_BENCHMARK = "buy_and_hold"

# Explicit dash patterns: plotly's named "dash" at width 1 is a 9 px dash, so a 30 px legend swatch
# shows one short bar that looks solid. These read as dashed in the plot and in the keys.
_REF_DASH = "6px,4px"            # reference lines: break-even, after costs, time limit, y = x, benchmark
_ALT_DASH = "8px,3px,2px,3px"    # a third series of the same quantity (cumulative costs)


# ------------------------------------------------------------------ per-trade quantities
def _sign(t) -> int:
    return 1 if t.side == "LONG" else -1


def _entry_mid(t, bars, slip_rate: Optional[float]) -> float:
    if bars is not None:
        return float(bars.open[t.entry_bar])        # every entry fills at the entry bar's open
    return float(t.entry_price / (1 + _sign(t) * (slip_rate or 0.0)))


def _exit_mid(t, bars, slip_rate: Optional[float]) -> float:
    """The mid the trade closed at (fills are the mid moved adversely by spread + slippage)."""
    s = _sign(t)
    if slip_rate is None:
        # the engine's slip rate, read back from this trade's own entry fill
        slip_rate = s * (t.entry_price / float(bars.open[t.entry_bar]) - 1.0) if bars is not None else 0.0
    return float(t.exit_price / (1 - s * slip_rate))


def excursions(trades, bars, *, slip_rate: Optional[float] = None):
    """(MFE, MAE) per trade: the best and worst move while the position was open, in % of the
    entry mid (the entry bar's open; costs are charged on mids, so the break-even line is exact).

    The window follows the engine (``strategy/backtest.py``): the entry fills at the OPEN of
    ``entry_bar``, so that bar's whole range belongs to the trade; bars up to ``exit_bar - 1`` are
    held throughout; of ``exit_bar`` only its open and the exit mid count, except for an ``EOW``
    exit (at that bar's close: the whole bar was held). MFE >= 0 >= MAE, and the realised move
    lies between them. ``slip_rate`` (spread + slippage as a fraction) recovers the exit mid from
    the exit fill; by default it is read back from each trade's entry fill.
    """
    hi, lo, op = (np.asarray(bars.high, float), np.asarray(bars.low, float), np.asarray(bars.open, float))
    mfe, mae = [], []
    for t in trades:
        s = _sign(t)
        a = t.entry_bar
        last = t.exit_bar if t.exit_reason in _CLOSE_EXITS else t.exit_bar - 1
        ref = op[a]
        x_mid = _exit_mid(t, bars, slip_rate)
        highs = [ref, op[t.exit_bar], x_mid]
        lows = [ref, op[t.exit_bar], x_mid]
        if last >= a:
            highs.append(hi[a:last + 1].max())
            lows.append(lo[a:last + 1].min())
        h, low = max(highs), min(lows)
        if s > 0:
            mfe.append(100 * (h - ref) / ref)
            mae.append(100 * (low - ref) / ref)
        else:
            mfe.append(100 * (ref - low) / ref)
            mae.append(100 * (ref - h) / ref)
    return np.array(mfe, dtype=float), np.array(mae, dtype=float)


def cost_per_side(config) -> float:
    """Fee + half-spread + slippage per side, as a fraction of the mid notional."""
    return (config.fee_bps + config.half_spread_bps + config.slippage_bps) / 1e4


def break_even_pct(config, side: str = "LONG") -> float:
    """The mid-to-mid move (% of the entry mid) a trade needs to break even after costs:
    2c / (1 - c) for a long, 2c / (1 + c) for a short (c = cost per side)."""
    c = cost_per_side(config)
    return 100 * 2 * c / ((1 - c) if side == "LONG" else (1 + c))


def _spearman(a, b) -> float:
    import pandas as pd

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def _money(v: float, decimals: int = 0) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):,.{decimals}f}"


def _pct(v: float, decimals: int = 2) -> str:
    return f"{v:+.{decimals}f}%"


# ------------------------------------------------------------------ layout helpers
def _panel_key(fig, legend_id: str, row: int, col: int):
    """A panel's key: one row of legend entries directly above the panel, under its title."""
    sp = fig.get_subplot(row, col)
    xd, yd = sp.xaxis.domain, sp.yaxis.domain
    fig.update_layout({legend_id: dict(
        orientation="h", x=xd[0], xanchor="left", y=yd[1], yanchor="bottom", font=dict(size=10, color=T.INK_2),
        bgcolor="rgba(0,0,0,0)", itemsizing="constant", itemwidth=30, groupclick="togglegroup", tracegroupgap=0,
        title=None)})


def _note(fig, row: int, col: int, text: str):
    sp = fig.get_subplot(row, col)
    xd, yd = sp.xaxis.domain, sp.yaxis.domain
    fig.add_annotation(x=(xd[0] + xd[1]) / 2, y=(yd[0] + yd[1]) / 2, xref="paper", yref="paper", text=text,
                       showarrow=False, font=dict(color=T.MUTED, size=12))


def _title(text: str, readout: str) -> str:
    return f"{text}<br><span style='font-size:10px;color:{T.MUTED}'>{readout}</span>"


def _hline(fig, row, col, x0, x1, y, *, name, legend, dash=_REF_DASH, color=T.NEUTRAL, showlegend=True, group=None):
    """A horizontal reference line as a trace, so its label lives in the panel key (never on the data)."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", name=name, legend=legend, legendgroup=group or name,
                             showlegend=showlegend, line=dict(color=color, dash=dash, width=1), hoverinfo="skip"),
                  row, col)


# ------------------------------------------------------------------ per-trade analytics
_PANELS = (("Return per trade (% of notional)", "legend"), ("Gross vs net return (cost drag, %)", "legend2"),
           ("Net return by exit reason (%)", "legend3"), ("Cumulative P&L ($)", "legend4"),
           ("Holding time vs net return", "legend5"), ("Best vs worst move while open (%)", "legend6"),
           ("Gross return by entry conviction", "legend7"), ("Long vs short: total P&L ($)", "legend8"),
           ("Predicted vs realised h1 move ($)", "legend9"))


def _h1_steps(horizon_steps, signals) -> Optional[int]:
    """Bars ahead of the h1 forecast: from ``horizon_steps`` (e.g. a PredictionFrame's), else from the
    signals if they carry it; None when unknown."""
    steps = horizon_steps if horizon_steps is not None else getattr(signals, "horizon_steps", None)
    if steps is None:
        return None
    if np.ndim(steps) == 0:
        return int(steps)
    return int(steps[1]) if len(steps) > 1 else None


def _disjoint_windows(starts, span: int) -> int:
    """How many of the windows [start, start + span) can be picked without overlap (greedy, in time
    order): the effective sample count of outcomes that share bars."""
    k, last = 0, -np.inf
    for b in np.sort(np.asarray(starts, float)):
        if b >= last + span:
            k, last = k + 1, b
    return k


def _no_trades_figure(result):
    """One short figure with one message (not nine empty panels)."""
    import plotly.graph_objects as go

    orders = len(result.decisions or [])
    why = (f"{result.strategy} placed {orders} order{'s' if orders != 1 else ''} but none became a trade"
           if orders else f"{result.strategy} placed no orders")
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, align="center",
                       font=dict(color=T.INK_2, size=13),
                       text=f"{why} on this block, so there is nothing to analyse per trade.<br>"
                            f"<span style='color:{T.MUTED};font-size:12px'>Compare its entry thresholds with the "
                            f"signal ranges: BacktestExplorer.signal_summary().</span>")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    T.apply(fig, title=f"{result.strategy}: 0 trades", height=200, legend_top=False)
    fig.update_layout(plot_bgcolor=T.PAPER, margin=dict(t=56, b=8, l=24, r=24), showlegend=False)
    return fig


def trade_analytics_figure(result, bars=None, *, signals=None, horizon_steps: Optional[Sequence[int]] = None,
                           height: int = 1380, seed: int = 0):
    """Per-trade analytics of one BacktestResult.

    ``bars``: its Bars (for the MFE / MAE panel); ``signals``: the SignalFrame it traded on (for the
    conviction and predicted-move panels, read at the DECISION bar ``entry_bar - 1``: the order
    fills at the next open, so the entry bar itself would be look-ahead). ``horizon_steps``: the
    bars ahead of (h0, h1, h2), e.g. ``PredictionFrame.horizon_steps``. With it, the predicted h1
    move is compared with the realised move over the SAME h1 bars from the decision close; without
    it, with the trade's own move from entry to exit (a different span, said so on the panel).
    Panels without their input say so instead of drawing nothing. A result with no trades gives
    one short figure with one message.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    trades = list(result.trades)
    n = len(trades)
    if not n:
        return _no_trades_figure(result)
    cfg = result.config
    rt = 100 * 2 * cost_per_side(cfg)                  # round-trip cost, % of notional (mid to mid)
    be = break_even_pct(cfg, "LONG")
    readouts = [""] * 9
    sign = np.array([_sign(t) for t in trades])
    side = np.array([t.side for t in trades])
    notional = np.array([t.notional for t in trades], dtype=float)
    gross = np.array([t.gross_pnl for t in trades], dtype=float)
    net = np.array([t.net_pnl for t in trades], dtype=float)
    gross_pct, net_pct = 100 * gross / notional, 100 * net / notional
    held = np.array([t.bars_held for t in trades])
    reason = np.array([t.exit_reason for t in trades])
    win = net > 0
    g_mean, g_lo, g_hi = S.mean_ci(gross_pct)          # consecutive trades never overlap: one sample each
    readouts[0] = f"mean gross {_pct(g_mean)} [{g_lo:+.2f}, {g_hi:+.2f}], net {_pct(net_pct.mean())}"
    readouts[1] = f"gross beat the {rt:.2f}% costs on {int((gross_pct > rt).sum())} of {n} trades"
    readouts[2] = "box: median, quartiles, 1.5 IQR whiskers"
    readouts[3] = (f"gross {_money(gross.sum())} - costs ${(gross - net).sum():,.0f} = net {_money(net.sum())}")
    time_exits = held[reason == "TIME"]
    t_lim = int(time_exits.max()) if len(time_exits) else None
    n_lim = int((reason == "TIME").sum())
    readouts[4] = (f"{n_lim} of {n} trades ({100 * n_lim / n:.0f}%) hit the {t_lim}-bar time limit" if t_lim
                   else "no trade hit a time limit")

    fig = make_subplots(rows=3, cols=3, subplot_titles=[_title(p, r) for (p, _), r in zip(_PANELS, readouts)],
                        vertical_spacing=0.15, horizontal_spacing=0.075)
    for a in fig.layout.annotations:              # the title and readout sit above the panel's key row
        a.update(yshift=24)
    for i, (_, lid) in enumerate(_PANELS):
        _panel_key(fig, lid, i // 3 + 1, i % 3 + 1)
    s = result.summary

    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    lg = dict(zip(range(1, 10), [lid for _, lid in _PANELS]))

    # (1,1) return per trade: won / lost after costs, with the gross distribution over it --------
    lo_x, hi_x = min(net_pct.min(), gross_pct.min()), max(net_pct.max(), gross_pct.max())
    width = max((hi_x - lo_x) / 32, 1e-3)
    start = np.floor(lo_x / width) * width                       # a multiple of the width: 0 is a bin edge
    edges = start + width * np.arange(int(np.ceil((hi_x - start) / width)) + 2)
    centres = (edges[:-1] + edges[1:]) / 2
    for sel, name, color in ((win, f"won ({int(win.sum())})", T.GOOD),
                             (~win, f"lost ({int((~win).sum())})", T.CRITICAL)):
        cnt, _ = np.histogram(net_pct[sel], bins=edges)
        fig.add_trace(go.Bar(x=f32(centres), y=cnt, width=width * 0.95, name=name, legend=lg[1],
                             marker=dict(color=T.rgba(color, 0.8), line=dict(width=0)),
                             hovertemplate="net %{x:>+.2f}%<br>%{y} trades<extra>" + name + "</extra>"), 1, 1)
    dens, _ = np.histogram(gross_pct, bins=edges)
    fig.add_trace(go.Scatter(x=f32(centres), y=dens, mode="lines", line_shape="hvh", name="gross",
                             legend=lg[1], line=dict(color=T.INK_2, width=1.5),
                             hovertemplate="gross %{x:>+.2f}%<br>%{y} trades<extra></extra>"), 1, 1)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, width=1), row=1, col=1)

    # (1,2) gross vs net: every trade pays the same round trip -----------------------------------
    usd = np.stack([gross, net, notional], 1).astype(np.float32)
    fig.add_trace(go.Scatter(x=f32(gross_pct), y=f32(net_pct), mode="markers", name="trade", legend=lg[2], showlegend=False,
                             marker=dict(size=6, color=T.rgba(T.INK_2, 0.75), line=dict(width=0)), customdata=usd,
                             hovertemplate="gross %{x:>+.3f}% (%{customdata[0]:>+$,.2f})<br>net %{y:>+.3f}% "
                                           "(%{customdata[1]:>+$,.2f})<br>notional %{customdata[2]:$,.0f}"
                                           "<extra></extra>"), 1, 2)
    lim = np.array([lo_x, hi_x])
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="no costs", legend=lg[2],
                             line=dict(color=T.NEUTRAL, width=1), hoverinfo="skip"), 1, 2)
    fig.add_trace(go.Scatter(x=lim, y=lim - rt, mode="lines", name=f"after {rt:.2f}% costs", legend=lg[2],
                             line=dict(color=T.NEUTRAL, width=1, dash=_REF_DASH), hoverinfo="skip"), 1, 2)

    # (1,3) net return by exit reason: neutral boxes, points coloured by outcome ------------------
    outcomes = (("won", T.GOOD, "circle", win), ("lost", T.CRITICAL, "x", ~win))
    for r in sorted(set(reason)):
        m = reason == r
        lab = f"{r} · {int(m.sum())}<br>{int((m & win).sum())} won · {_money(net[m].sum())}"
        # the summary box keeps its (invisible) points so the whiskers stay at 1.5 IQR, not min / max
        fig.add_trace(go.Box(y=f32(net_pct[m]), x=[lab] * int(m.sum()), name=r, boxpoints="all", jitter=0,
                             marker=dict(opacity=0), line=dict(color=T.INK_2, width=1),
                             fillcolor=T.rgba(T.INK_2, 0.08), hoveron="boxes", showlegend=False, legend=lg[3]), 1, 3)
        for name, color, symbol, outcome in outcomes:
            sel = m & outcome
            k = int(sel.sum())
            if not k:
                continue
            fig.add_trace(go.Box(y=f32(net_pct[sel]), x=[lab] * k, name=name, legend=lg[3], legendgroup=name,
                                 showlegend=False, boxpoints="all", jitter=0.6, pointpos=0,
                                 fillcolor="rgba(0,0,0,0)", line=dict(width=0), hoveron="points",
                                 marker=dict(symbol=symbol, size=5, color=color),
                                 hovertemplate=f"{r}<br>net %{{y:>+.2f}}%<extra>{name}</extra>"), 1, 3)
    for name, color, symbol, outcome in outcomes:      # the key (a box trace's own swatch is its box)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{name} ({int(outcome.sum())})",
                                 legend=lg[3], legendgroup=name, marker=dict(symbol=symbol, size=7, color=color),
                                 hoverinfo="skip"), 1, 3)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=1, col=3)

    # (2,1) cumulative P&L in $ (summed $ is the equity change) -----------------------------------
    for y, name, color, dash in ((np.cumsum(gross), "gross", T.INK_2, _REF_DASH), (np.cumsum(net), "net", T.INK, "solid"),
                                 (np.cumsum(gross - net), "costs", T.OTHER_SERIES[1], _ALT_DASH)):
        fig.add_trace(go.Scatter(x0=1, dx=1, y=f32(y), mode="lines", name=name, legend=lg[4],
                                 line=dict(color=color, width=2 if name == "net" else 1.5, dash=dash),
                                 hovertemplate="trade %{x}<br>" + name + " %{y:>+$,.0f}<extra></extra>"), 2, 1)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=2, col=1)

    # (2,2) holding time: dodge long / short, jitter, true values in the hover --------------------
    rng = np.random.default_rng(seed)
    for s_name, color, symbol, off in (("LONG", T.LONG_COLOR, "triangle-up", -0.18),
                                       ("SHORT", T.SHORT_COLOR, "triangle-down", 0.18)):
        m = side == s_name
        if not m.any():
            continue
        xj = held[m] + off + rng.uniform(-0.1, 0.1, int(m.sum()))
        cd = np.stack([held[m], net[m]], 1).astype(np.float32)
        fig.add_trace(go.Scatter(x=f32(xj), y=f32(net_pct[m]), mode="markers", name=s_name.lower(),
                                 legend=lg[5], customdata=cd, opacity=0.75,
                                 marker=dict(symbol=symbol, size=7, color=color, line=dict(color=T.PAPER, width=0.5)),
                                 hovertemplate="held %{customdata[0]:.0f} bars<br>net %{y:>+.2f}% "
                                               "(%{customdata[1]:>+$,.2f})<extra>" + s_name.lower() + "</extra>"),
                      2, 2)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=2, col=2)
    if t_lim:
        y0, y1 = float(net_pct.min()), float(net_pct.max())
        fig.add_trace(go.Scatter(x=[t_lim + 0.45, t_lim + 0.45], y=[y0, y1], mode="lines", legend=lg[5],
                                 name="time limit", line=dict(color=T.NEUTRAL, width=1, dash=_REF_DASH),
                                 hoverinfo="skip"), 2, 2)

    # (2,3) MFE / MAE from the entry mid over the bars actually held ------------------------------
    if bars is not None:
        mfe, mae = excursions(trades, bars, slip_rate=cfg.slip_rate)
        be_side = np.where(sign > 0, break_even_pct(cfg, "LONG"), break_even_pct(cfg, "SHORT"))
        reached = int(((mfe >= be_side - 1e-9) & ~win).sum())
        info = np.stack([net_pct, gross_pct], 1).astype(np.float32)
        for sel, name, color, symbol in ((win, "won", T.GOOD, "circle"), (~win, "lost", T.CRITICAL, "x")):
            txt = [f"{t.side.lower()} {t.entry_bar}->{t.exit_bar} {t.exit_reason}" for t, k in zip(trades, sel) if k]
            fig.add_trace(go.Scatter(x=f32(mae[sel]), y=f32(mfe[sel]), mode="markers", name=name, legend=lg[6],
                                     text=txt, customdata=info[sel], marker=dict(symbol=symbol, size=6, color=color),
                                     hovertemplate="%{text}<br>worst %{x:.3f}% · best %{y:.3f}%<br>realised "
                                                   "%{customdata[1]:>+.3f}% gross, %{customdata[0]:>+.3f}% net"
                                                   "<extra></extra>"), 2, 3)
        _hline(fig, 2, 3, float(min(mae.min(), -0.01)), 0.0, be, name=f"break-even {be:.2f}%", legend=lg[6])
        fig.layout.annotations[5].text = _title(_PANELS[5][0], f"{reached} of {int((~win).sum())} losers were once "
                                                                f"past break-even")
    else:
        _note(fig, 2, 3, "needs the bars")

    # (3,1), (3,3) the entry signal at the decision bar -------------------------------------------
    if signals is not None:
        d = np.array([t.entry_bar - 1 for t in trades])
        wdir = np.asarray(signals.weighted_direction, float)[d]
        conv = sign * (wdir - 0.5)
        k = int(min(5, max(1, n // 8)))
        qs = np.quantile(conv, np.linspace(0, 1, k + 1))
        b = np.clip(np.searchsorted(qs[1:-1], conv, side="right"), 0, k - 1)
        xs, means, lo_ci, hi_ci, labels, hover = [], [], [], [], [], []
        for j in range(k):
            m = b == j
            if not m.any():
                continue
            mu, lo_, hi_ = S.mean_ci(gross_pct[m])
            xs.append(j)
            means.append(mu)
            lo_ci.append(mu - lo_ if np.isfinite(lo_) else 0.0)
            hi_ci.append(hi_ - mu if np.isfinite(hi_) else 0.0)
            labels.append(f"Q{j + 1} · {int(m.sum())}")
            hover.append(f"conviction {conv[m].min():+.3f} to {conv[m].max():+.3f}<br>{int(m.sum())} trades, "
                         f"{100 * np.mean(gross_pct[m] > 0):.0f}% up before costs<br>mean net {net_pct[m].mean():+.3f}%")
        fig.add_trace(go.Scatter(x=labels, y=means, mode="markers+lines", name="mean, 95% CI", legend=lg[7],
                                 line=dict(color=T.INK_2, width=1), marker=dict(size=8, color=T.INK),
                                 error_y=dict(type="data", symmetric=False, array=hi_ci, arrayminus=lo_ci,
                                              color=T.INK_2, thickness=1.2, width=5),
                                 text=hover, hovertemplate="%{x}<br>mean gross %{y:>+.3f}%<br>%{text}<extra></extra>"),
                      3, 1)
        _hline(fig, 3, 1, labels[0], labels[-1], be, name=f"break-even {be:.2f}%", legend=lg[7])
        fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=3, col=1)
        rho, band = _spearman(conv, gross_pct), S.corr_null(n)
        fig.layout.annotations[6].text = _title(_PANELS[6][0], f"Spearman ρ {rho:+.2f} (chance ±{band:.2f}), "
                                                                f"{k} bins")
        # (3,3): the h1 forecast at the decision bar, in the trade's direction, against what followed
        pred = sign * np.asarray(signals.delta, float)[d, 1]
        sig_h1 = np.asarray(signals.sigma, float)[d, 1]
        entry_mid = np.array([_entry_mid(t, bars, cfg.slip_rate) for t in trades])
        exit_mid = np.array([_exit_mid(t, bars, cfg.slip_rate) for t in trades])
        hold_move = sign * (exit_mid - entry_mid)          # what the position itself moved, entry to exit mid
        steps = _h1_steps(horizon_steps, signals)
        if steps:
            # like for like: the realised move over the SAME h1 bars from the decision close
            close = np.asarray(signals.close, float)
            ok = d + steps < len(close)
            real = np.full(n, np.nan)
            real[ok] = sign[ok] * (close[d[ok] + steps] - close[d[ok]])
            n_eff = _disjoint_windows(d[ok], steps)       # overlapping h1 windows share bars: count them once
            y9_title = f"realised move over the same {steps} bars ($)"
            real_hover = (f"realised over the next {steps} bars %{{y:>+$,.2f}}<br>the trade itself: held "
                          f"%{{customdata[2]:.0f}} bars, moved %{{customdata[1]:>+$,.2f}} (entry to exit)")
        else:
            ok = np.ones(n, dtype=bool)
            real = hold_move
            n_eff = n                                      # positions never overlap
            y9_title = f"realised over the hold ($; median {np.median(held):.0f} bars)"
            real_hover = "realised over the hold (%{customdata[2]:.0f} bars, entry to exit) %{y:>+$,.2f}"
        cd9 = np.stack([sig_h1, hold_move, held], 1).astype(np.float32)
        for s_name, color, symbol in (("LONG", T.LONG_COLOR, "triangle-up"), ("SHORT", T.SHORT_COLOR, "triangle-down")):
            m = (side == s_name) & ok
            if not m.any():
                continue
            fig.add_trace(go.Scatter(x=f32(pred[m]), y=f32(real[m]), mode="markers", name=s_name.lower(), legend=lg[9],
                                     customdata=cd9[m], opacity=0.8,
                                     marker=dict(symbol=symbol, size=7, color=color, line=dict(color=T.PAPER, width=0.5)),
                                     hovertemplate="predicted h1 %{x:>+$,.2f} (sigma %{customdata[0]:$,.0f})<br>"
                                                   + real_hover + "<extra>" + s_name.lower() + "</extra>"), 3, 3)
        span = np.array([min(pred.min(), 0.0), max(pred.max(), 0.0)])
        # no zero line here: the realised = predicted line IS almost flat when the predicted moves are small
        fig.add_trace(go.Scatter(x=span, y=span, mode="lines", name="y = x (perfect)", legend=lg[9],
                                 line=dict(color=T.NEUTRAL, width=1.5, dash=_REF_DASH), hoverinfo="skip"), 3, 3)
        rho2, band2 = _spearman(pred[ok], real[ok]), S.corr_null(max(n_eff, 1))
        right = 100 * np.mean(np.sign(pred[ok]) * np.sign(real[ok]) > 0) if ok.any() else float("nan")
        late = int((~ok).sum())
        head = _PANELS[8][0] if steps else "Predicted h1 vs move over the hold ($)"
        fig.layout.annotations[8].text = _title(
            head, f"ρ {rho2:+.2f} (chance ±{band2:.2f}) · sign right {right:.0f}% · median "
                  f"${np.median(np.abs(pred[ok])):,.1f} vs ${np.median(np.abs(real[ok])):,.0f}"
                  + (f" · {late} too late to score" if late else ""))
    else:
        y9_title = "realised move ($)"
        for c in (1, 3):
            _note(fig, 3, c, "needs the signals (signals=)")

    # (3,2) long vs short totals in $ -------------------------------------------------------------
    cats, g_tot, n_tot, cols, hov = [], [], [], [], []
    for s_name, color, tri in (("LONG", T.LONG_COLOR, "▲"), ("SHORT", T.SHORT_COLOR, "▼")):
        m = side == s_name
        if not m.any():
            continue
        mu, lo_, hi_ = S.mean_ci(gross_pct[m])
        cats.append(f"{tri} {s_name.lower()} · {int(m.sum())}")
        g_tot.append(gross[m].sum())
        n_tot.append(net[m].sum())
        cols.append(color)
        ci = f" [{lo_:+.3f}, {hi_:+.3f}]" if np.isfinite(lo_) else ""
        hov.append(f"{int(m.sum())} trades, {100 * np.mean(gross[m] > 0):.0f}% up before costs, "
                   f"{100 * np.mean(win[m]):.0f}% after<br>mean gross {mu:+.3f}%{ci} per trade")
    xpos = np.arange(len(cats), dtype=float)
    fig.add_trace(go.Bar(x=xpos - 0.2, y=g_tot, width=0.38, showlegend=False, legend=lg[8], name="before costs",
                         marker=dict(color=[T.rgba(c, 0.15) for c in cols], line=dict(color=cols, width=1.5)),
                         text=[_money(v) for v in g_tot], textposition="outside", textfont=dict(size=10, color=T.INK_2),
                         customdata=hov, hovertemplate="before costs %{y:>+$,.0f}<br>%{customdata}<extra></extra>"), 3, 2)
    fig.add_trace(go.Bar(x=xpos + 0.2, y=n_tot, width=0.38, showlegend=False, legend=lg[8], name="after costs",
                         marker=dict(color=cols, line=dict(width=0)), text=[_money(v) for v in n_tot],
                         textposition="outside", textfont=dict(size=10, color=T.INK_2), customdata=hov,
                         hovertemplate="after costs %{y:>+$,.0f}<br>%{customdata}<extra></extra>"), 3, 2)
    for name, marker in (("before costs", dict(color=T.rgba(T.NEUTRAL, 0.15), line=dict(color=T.NEUTRAL, width=1.5))),
                         ("after costs", dict(color=T.NEUTRAL))):
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, legend=lg[8], marker=marker, hoverinfo="skip"), 3, 2)
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=3, col=2)
    fig.update_xaxes(tickvals=xpos, ticktext=cats, range=[-0.7, len(cats) - 0.3], row=3, col=2)
    top = max(max(g_tot + n_tot), 0.0)
    bot = min(min(g_tot + n_tot), 0.0)
    pad = 0.18 * (top - bot or 1.0)
    fig.update_yaxes(range=[bot - pad, top + pad], row=3, col=2)
    hit_l = [100 * np.mean(gross[side == sd] > 0) for sd in ("LONG", "SHORT") if (side == sd).any()]
    fig.layout.annotations[7].text = _title(_PANELS[7][0], "up before costs: " + ", ".join(
        f"{c.split(' · ')[0][2:]} {h:.0f}%" for c, h in zip(cats, hit_l)))

    # axes -----------------------------------------------------------------------------------------
    for (r, c), (xt, yt) in {(1, 1): ("return per trade (% of notional)", "trades"),
                             (1, 2): ("gross return (%)", "net return (%)"), (1, 3): (None, "net return (%)"),
                             (2, 1): ("trade #", "cumulative $"), (2, 2): ("bars held", "net return (%)"),
                             (2, 3): ("worst move against (% of entry mid)", "best move in favour (%)"),
                             (3, 1): ("conviction quintile at the decision bar (weak to strong)", "mean gross return (%)"),
                             (3, 2): (None, "total P&L ($)"),
                             (3, 3): ("predicted h1 move, in the trade's direction ($)", y9_title)}.items():
        fig.update_xaxes(title_text=xt, row=r, col=c)
        fig.update_yaxes(title_text=yt, row=r, col=c)
    fig.update_layout(barmode="overlay", boxmode="overlay")

    wins, losses = net[win], net[~win]
    avg_win = _money(wins.mean(), 2) if len(wins) else "n/a"
    avg_loss = _money(losses.mean(), 2) if len(losses) else "n/a"
    line1 = (f"hit rate {100 * (s.get('hit_rate') or 0):.1f}% after costs, {100 * (s.get('hit_rate_gross') or 0):.1f}% "
             f"before · profit factor {s.get('profit_factor', float('nan')):.2f} · avg win {avg_win}, avg loss "
             f"{avg_loss}, expectancy {_money(net.mean(), 2)} per trade")
    line2 = (f"costs {cost_per_side(cfg) * 1e4:.0f} bps per side ({rt:.2f}% round trip) · per-trade panels in % of "
             f"notional (size x equity at entry: ${notional[0]:,.0f} first trade, ${notional[-1]:,.0f} last)")
    line3 = "won / lost = net P&L after costs · [a, b] = 95% CI (positions never overlap: one sample per trade)"
    T.apply(fig, title=f"{result.strategy}: {n} trades", subtitle=f"{line1}<br>{line2}<br>{line3}", height=height,
            legend_top=False)
    fig.update_layout(margin=dict(t=186, b=56))
    return fig


# ------------------------------------------------------------------ several strategies
def _mean_size(result) -> float:
    sizes = [float(d.get("size", 1.0)) for d in (result.decisions or [])]
    return float(np.mean(sizes)) if sizes else 1.0


def _usable_null(result, null) -> Optional[dict]:
    """The matched random null for ``result``, or None when it is absent or not size-matched."""
    if not null or not null.get("n_seeds") or "random_mean_total_return" not in null:
        return None
    size = null.get("size_frac")
    if size is None and abs(_mean_size(result) - 1.0) > 0.01:
        return None                      # random entries at full size vs a strategy that sizes down
    return null


def _default_label(name: str, result) -> str:
    """A display name. A ``random_signal`` run is ONE random draw at its own knobs, not the matched
    null: say so, with its trade rate and hold read back from the result."""
    if name != "random_signal":
        return name
    s = result.summary
    n_tr = int(s.get("n_trades", 0))
    if not n_tr:
        return "random_signal (one draw, not a matched null)"
    n_bars = max(1, len(result.equity) - 1)
    flat_bars = max(1, n_bars - int(s.get("exposure", 0.0) * n_bars))
    return (f"random_signal (one draw: {100 * n_tr / flat_bars:.0f}%/flat bar, {s.get('avg_hold_bars', 0):.0f}-bar "
            f"hold; not matched)")


def _strategy_styles(traded):
    """{name: (colour, dash)}: baselines in neutral inks, model strategies in the strategy colours
    (repeating with a dashed, then a dash-dot line past three)."""
    out, k = {}, 0
    dashes = ("solid", _REF_DASH, _ALT_DASH)
    for n in traded:
        if n in _BASELINE_COLORS:
            out[n] = (_BASELINE_COLORS[n], "solid")
        else:
            out[n] = (_STRATEGY_COLORS[k % len(_STRATEGY_COLORS)], dashes[(k // len(_STRATEGY_COLORS)) % len(dashes)])
            k += 1
    return out


def strategy_comparison_figure(results: Dict[str, object], bars=None, *, height: int = 660,
                               nulls: Optional[Dict[str, dict]] = None, labels: Optional[Dict[str, str]] = None):
    """``results``: {name: BacktestResult} on the same block.

    ``nulls``: {name: matched random baseline} (see ``notebook.backtest_ui.matched_random_null``;
    default: each result's ``baselines['random_same_freq']``), drawn as a hollow diamond with a
    5-95% whisker on that strategy's after-cost bar. ``labels``: display names (e.g. the knobs of an
    explicitly requested ``random_signal``; without one, a ``random_signal`` run is labelled as one
    unmatched draw). Strategies with no trades share one flat line and one row.

    Layout: one key under the plots (the strategies, then the bar encodings); it reserves its own
    rows at the bottom, so it cannot grow into the title however long the names are.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.035,
                        subplot_titles=("Equity after costs", "Return before and after costs (% of initial equity)"))
    names = list(results)
    if not names:
        T.note_on_empty(fig, "no strategies")
        T.apply(fig, title="Strategies on the same block", height=height)
        return fig
    labels = {n: (labels or {}).get(n) or _default_label(n, results[n]) for n in names}
    init = float(results[names[0]].config.initial_equity)
    flat = [n for n in names if results[n].summary.get("n_trades", 0) == 0]
    traded = [n for n in names if n not in flat]
    style = _strategy_styles(traded)

    # equity (after costs): the strategies' key ---------------------------------------------------
    for n in traded:
        eq = np.asarray(results[n].equity, dtype=np.float32)
        bench = n == _BENCHMARK
        color, dash = style[n]
        fig.add_trace(go.Scatter(x0=-1, dx=1, y=eq, mode="lines", name=labels[n] + (" (benchmark)" if bench else ""),
                                 legendgroup=n,
                                 line=dict(color=color, width=1.2 if bench else 1.6, dash=dash),
                                 hovertemplate=f"{labels[n]}<br>bar %{{x}}: %{{y:$,.0f}}<extra></extra>"), 1, 1)
    flat_name = None
    if flat:
        flat_name = "no trades: " + ", ".join(labels[n] for n in flat)
        m = len(results[flat[0]].equity)
        fig.add_trace(go.Scatter(x=[-1, m - 2], y=[init, init], mode="lines", name=flat_name,
                                 legendgroup="flat", line=dict(color=T.MUTED, width=1.2, dash=_REF_DASH),
                                 hoverinfo="skip"), 1, 1)

    # return bars, sorted by net return; one row for every no-trade strategy -----------------------
    rows = [(n, 100 * results[n].summary.get("gross_pnl", 0.0) / init, 100 * results[n].summary["total_return"])
            for n in traded]
    rows.sort(key=lambda r: r[2])
    if flat:
        rows.insert(next((i for i, r in enumerate(rows) if r[2] > 0), len(rows)), ("__flat__", 0.0, 0.0))
    ypos = np.arange(len(rows), dtype=float)
    ticktext, g_x, n_x, cols, hov = [], [], [], [], []
    null_x, null_y, null_lo, null_hi, null_txt, seeds = [], [], [], [], [], set()
    for i, (n, g, v) in enumerate(rows):
        if n == "__flat__":
            ticktext.append(f"no trades ({len(flat)})<br>{_pct(0.0, 1)}")
            g_x.append(0.0), n_x.append(0.0), cols.append(T.MUTED), hov.append(flat_name)
            continue
        r = results[n]
        nt = r.summary["n_trades"]
        lab = f"{labels[n]} · {nt} trade{'s' if nt != 1 else ''}<br>net {_pct(v, 1)} · gross {_pct(g, 1)}"
        null = (nulls or {}).get(n) if nulls is not None else r.baselines.get("random_same_freq")
        null = _usable_null(r, null)
        if null is not None:
            pct = null.get("percentile_total_return", float("nan"))
            lab += f"<br>beats {pct:.0f}% of random" + (f" ({null['percentile_gross_return']:.0f}% before costs)"
                                                         if "percentile_gross_return" in null else "")
            mu = 100 * null["random_mean_total_return"]
            p05 = 100 * null.get("random_p05_total_return", null["random_mean_total_return"])
            p95 = 100 * null.get("random_p95_total_return", null["random_mean_total_return"])
            null_x.append(mu), null_y.append(i - 0.2)
            null_lo.append(max(mu - p05, 0.0)), null_hi.append(max(p95 - mu, 0.0))
            seeds.add(int(null["n_seeds"]))
            null_txt.append(f"{labels[n]}: random with {100 * null.get('trade_rate', 0):.2f}% entries/flat bar, "
                            f"{null.get('hold_bars', 0)}-bar hold, size {null.get('size_frac', 1.0):.2f}<br>"
                            f"mean {mu:+.2f}%, 5-95% {p05:+.2f}% to {p95:+.2f}% ({null['n_seeds']} seeds)<br>"
                            f"strategy {v:+.2f}%: beats {pct:.0f}% of them"
                            + (f"; before costs beats {null['percentile_gross_return']:.0f}%"
                               if "percentile_gross_return" in null else ""))
        ticktext.append(lab)
        g_x.append(g), n_x.append(v), cols.append(style[n][0])
        hov.append(f"{labels[n]}: {nt} trades, costs ${r.summary.get('costs_paid', 0):,.0f}")
    # the values are in the row labels (text on the bars would sit on the random-null markers)
    fig.add_trace(go.Bar(y=ypos + 0.2, x=g_x, width=0.36, orientation="h", name="before costs", showlegend=False,
                         marker=dict(color=[T.rgba(c, 0.15) for c in cols], line=dict(color=cols, width=1.5)),
                         customdata=hov, hovertemplate="%{customdata}<br>before costs %{x:>+.2f}%<extra></extra>"), 1, 2)
    fig.add_trace(go.Bar(y=ypos - 0.2, x=n_x, width=0.36, orientation="h", name="after costs", showlegend=False,
                         marker=dict(color=cols, line=dict(width=0)), customdata=hov,
                         hovertemplate="%{customdata}<br>after costs %{x:>+.2f}%<extra></extra>"), 1, 2)
    # the bar panel's key: neutral swatches (the bar colours are the strategies')
    for name, marker in (("before costs", dict(color=T.rgba(T.NEUTRAL, 0.15), line=dict(color=T.NEUTRAL, width=1.5))),
                         ("after costs", dict(color=T.NEUTRAL))):
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name=name, marker=marker,
                             hoverinfo="skip"), 1, 2)
    if null_x:
        fig.add_trace(go.Scatter(x=null_x, y=null_y, mode="markers", name="random entries, same rate / hold / size "
                                 "(mean, 5-95%)",
                                 marker=dict(symbol="diamond-open", size=11, color=T.INK, line=dict(width=2)),
                                 error_x=dict(type="data", symmetric=False, array=null_hi, arrayminus=null_lo,
                                              color=T.INK, thickness=1.2, width=5),
                                 text=null_txt, hovertemplate="%{text}<extra></extra>"), 1, 2)
    if _BENCHMARK in traded:
        v = 100 * results[_BENCHMARK].summary["total_return"]
        fig.add_trace(go.Scatter(x=[v, v], y=[-0.6, len(rows) - 0.4], mode="lines", name="buy and hold's return",
                                 line=dict(color=T.NEUTRAL, width=1, dash=_REF_DASH),
                                 hoverinfo="skip"), 1, 2)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, width=1), row=1, col=2)
    allx = g_x + n_x + [a - b for a, b in zip(null_x, null_lo)] + [a + b for a, b in zip(null_x, null_hi)]
    lo_x, hi_x = min(min(allx), 0.0), max(max(allx), 0.0)
    pad = 0.16 * (hi_x - lo_x or 1.0)
    fig.update_xaxes(range=[lo_x - pad, hi_x + pad], title_text="% of initial equity", row=1, col=2)
    fig.update_yaxes(tickvals=ypos, ticktext=ticktext, range=[-0.6, len(rows) - 0.4], side="right", automargin=True,
                     showgrid=False, row=1, col=2)
    fig.update_xaxes(title_text="bar (test block)", row=1, col=1)
    fig.update_yaxes(title_text="equity ($)", row=1, col=1)
    fig.update_layout(barmode="overlay")
    cfg = results[names[0]].config
    rt = 100 * 2 * cost_per_side(cfg)
    sub = (f"costs {cost_per_side(cfg) * 1e4:.0f} bps per side ({rt:.2f}% round trip) · rows sorted by "
           f"return after costs")
    if null_x:
        k_seeds = f"{min(seeds)}" if len(seeds) == 1 else f"{min(seeds)}-{max(seeds)}"
        sub += (f"<br>diamond: random entries at the strategy's trade rate, holding time and size (mean of {k_seeds} "
                f"seeds, whisker 5-95%); a bar inside the whisker is no better than chance")
    T.apply(fig, title="Strategies on the same block", height=height, subtitle=sub, legend_top=False)
    # The key goes UNDER the plots, anchored to the bottom of the figure: plotly reserves its height
    # below the axis titles, so however many rows it wraps to it pushes the panels up and never
    # reaches the title (a key anchored above the panels grows upward into the subtitle). The top
    # margin is the fixed title band (one bold line + the subtitle lines) plus the panel titles.
    n_lines = 1 + sub.count("<br>") + 1
    fig.update_layout(
        margin=dict(t=14 + 20 * n_lines + 28, b=8, r=24),
        legend=dict(orientation="h", xref="paper", x=0, xanchor="left", yref="container", y=0, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=T.INK_2), itemsizing="constant", itemwidth=30,
                    groupclick="togglegroup", tracegroupgap=0))
    return fig

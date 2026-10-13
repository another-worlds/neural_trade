"""Geometry figures of the tactical report: every geometry variant drawn on the same three real BTC windows.

All constructions follow the definitions in the code that was tested: IndicatorGeometry (src/neural_trade/models/layers/
indicator_geometry.py) for H24/H43/H45/H47 and runs/tactical/novel/features.py for N1-N7 (the numbers in the tables are
that code's own output on the window). Windows are 60 one-minute bars, all before the cutoff."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rcommon import (NF, T, UP, DOWN, PINK, GOLD, TAN, VIOLET, SALMON, INK, INK2, MUTED, GRID, SURFACE, AXIS, S,
                     span_text, table, ru)

X = np.arange(60)
NEUT = T.NEUTRAL


# ------------------------------------------------------------------ indicators on a 60-bar window
def ema(x, p):
    return NF.ema(np.asarray(x, float)[None], p)[0]


def sma(x, p):
    out = np.full(len(x), np.nan)
    for i in range(p - 1, len(x)):
        out[i] = x[i - p + 1:i + 1].mean()
    return out


def rstd(x, p):
    out = np.full(len(x), np.nan)
    for i in range(p - 1, len(x)):
        out[i] = x[i - p + 1:i + 1].std()
    return out


def indicators(W):
    o, h, l, c, v = W.T
    r = NF.rsi_series(c[None], 14)[0]
    rsi = 50 + 50 * r
    m = ema(c, 12) - ema(c, 26)
    sig = ema(m, 9)
    k = np.full(60, np.nan)
    for i in range(13, 60):
        lo, hi = l[i - 13:i + 1].min(), h[i - 13:i + 1].max()
        k[i] = 100 * (c[i] - lo) / (hi - lo) if hi > lo else 50.0
    dline = sma(np.nan_to_num(k, nan=50.0), 3)
    dline[:15] = np.nan
    s20 = sma(c, 20)
    sd20 = rstd(c, 20)
    return dict(ema20=ema(c, 20), sma20=s20, up=s20 + 2 * sd20, lo=s20 - 2 * sd20, rsi=rsi, macd=m, sig=sig,
                hist=m - sig, k=k, d=dline)


def chan_feats(x, c, slopes=(3, 10)):
    """The IndicatorGeometry features of one channel x against the close c (non-NaN part of x only)."""
    ok = ~np.isnan(x)
    xv = x[ok]
    sx = np.sqrt(np.var(np.diff(xv)) + 1e-8)
    sc = np.sqrt(np.var(np.diff(c)) + 1e-8)
    out = {f"slope{k}": (xv[-1] - xv[-1 - k]) / (k * sx) for k in slopes}
    d = (c[ok] - xv) / sc
    out["distance"] = float(np.arcsinh(d[-1]))
    out["cross5"] = float((np.tanh(d[-1]) - np.tanh(d[-1 - 5])) / 2)
    out["squeeze"] = float(np.sqrt(np.var(xv[-10:]) + 1e-8) / np.sqrt(np.var(xv) + 1e-8))
    out["_sx"] = sx
    return out


def crossings(x, y, bars=20):
    """Indices j (last `bars` bars) where series x crosses series y: (j, +1 up / -1 down)."""
    d = x - y
    res = []
    for j in range(60 - bars, 60):
        if np.isnan(d[j]) or np.isnan(d[j - 1]):
            continue
        if d[j - 1] < 0 <= d[j]:
            res.append((j, 1))
        elif d[j - 1] > 0 >= d[j]:
            res.append((j, -1))
    return res


def candle(fig, W, row=1, col=1, name="Свечи"):
    o, h, l, c, v = W.T
    fig.add_trace(go.Candlestick(x=X, open=o, high=h, low=l, close=c, name=name,
                                 increasing=dict(line=dict(color=UP), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
                                 hoverlabel=dict(namelength=-1), showlegend=False), row=row, col=col)


def seg(fig, x, k, color, label, row, col, width, dash, legend=False, norm=None):
    """Slope segment: chord of the channel between bar 59-k and bar 59."""
    y0, y1 = x[59 - k], x[59]
    txt = f"{label}: наклон за {k} бар = {(y1 - y0) / k / norm:+.2f} (в ед. своей волатильности)" if norm else label
    fig.add_trace(go.Scatter(x=[59 - k, 59], y=[y0, y1], mode="lines+markers", line=dict(color=color, width=width, dash=dash),
                             marker=dict(size=[5, 8], color=color, line=dict(color=INK, width=1)), showlegend=False,
                             hovertemplate=txt.replace("%", "%%") + "<extra></extra>"), row=row, col=col)


def markers_cross(fig, pts, series, color, name, row, col):
    if not pts:
        return
    xs = [j for j, _ in pts]
    ago = [59 - j for j in xs]
    fig.add_trace(go.Scatter(x=xs, y=[series[j] for j in xs], mode="markers", name=name, showlegend=False,
                             marker=dict(symbol=["triangle-up" if s > 0 else "triangle-down" for _, s in pts],
                                         size=[14 if a < 5 else 11 if a < 10 else 8 for a in ago], color=color,
                                         opacity=[1.0 if a < 5 else 0.8 if a < 10 else 0.5 for a in ago],
                                         line=dict(color=INK, width=1)),
                             customdata=[[a, "снизу вверх" if s > 0 else "сверху вниз"] for a, (_, s) in zip(ago, pts)],
                             hovertemplate=name + ": %{customdata[1]}, %{customdata[0]} бар назад<extra></extra>"),
                  row=row, col=col)


def legend_entry(fig, name, color, dash="solid", symbol=None, width=3, size=10):
    fig.add_trace(go.Scatter(x=[None], y=[None], name=name, mode="lines+markers" if symbol else "lines",
                             line=dict(color=color, width=width, dash=dash),
                             marker=dict(symbol=symbol or "circle", size=size, color=color) if symbol else None))


# ------------------------------------------------------------------ H24 / H43 / H45 / H47: indicator geometry
def h24_figure(w):
    W = w["W"]
    o, h, l, c, v = W.T
    ind = indicators(W)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.52, 0.16, 0.16, 0.16])
    candle(fig, W)
    # Bollinger band and channels
    fig.add_trace(go.Scatter(x=X, y=ind["up"], mode="lines", line=dict(color=INK2, width=1, dash="dot"), name="Bollinger 20 ±2σ",
                             legendgroup="bb", hovertemplate="верхняя полоса %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=X, y=ind["lo"], mode="lines", line=dict(color=INK2, width=1, dash="dot"), name="нижняя полоса",
                             legendgroup="bb", showlegend=False, fill="tonexty", fillcolor="rgba(195,194,183,0.07)",
                             hovertemplate="нижняя полоса %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=X, y=ind["sma20"], mode="lines", line=dict(color=TAN, width=2), name="SMA20 (середина Bollinger)",
                             hovertemplate="SMA20 %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=X, y=ind["ema20"], mode="lines", line=dict(color=PINK, width=2), name="EMA20",
                             hovertemplate="EMA20 %{y:.2f}<extra></extra>"), row=1, col=1)
    feats = {}
    for key, col, lab in (("ema20", PINK, "EMA20"), ("sma20", TAN, "SMA20"), ("up", INK2, "Bollinger верх"), ("lo", INK2, "Bollinger низ")):
        f = chan_feats(ind[key], c)
        feats[lab] = f
        seg(fig, ind[key], 10, col, lab, 1, 1, 3, "dash", norm=f["_sx"])
        seg(fig, ind[key], 3, col, lab, 1, 1, 5, "solid", norm=f["_sx"])
    markers_cross(fig, crossings(c, ind["ema20"]), ind["ema20"], PINK, "Цена пересекла EMA20", 1, 1)
    markers_cross(fig, crossings(c, ind["sma20"]), ind["sma20"], TAN, "Цена пересекла SMA20", 1, 1)
    # distance arrows
    for key, col, lab, dx in (("ema20", PINK, "EMA20", 0.55), ("sma20", TAN, "SMA20", 1.5)):
        f = feats[lab]
        fig.add_annotation(x=59 + dx, y=ind[key][59], ax=59 + dx, ay=c[59], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=col, row=1, col=1)
        fig.add_annotation(x=59 + dx + 0.3, y=(ind[key][59] + c[59]) / 2, text=f"{lab}: расст. {f['distance']:+.2f}", showarrow=False,
                           xanchor="left", yshift=16 if lab == "EMA20" else -16, font=dict(size=10, color=col), xref="x", yref="y", row=1, col=1)
    # squeeze band over the last 10 bars (all rows)
    sq = feats["EMA20"]["squeeze"]
    for r in (1, 2, 3, 4):
        fig.add_vrect(x0=49.5, x1=59.5, fillcolor="rgba(242,241,236,0.07)", line_width=0, row=r, col=1)
    fig.add_annotation(x=54.5, y=1.0, yref="y domain", xref="x", text=f"сжатие EMA20: σ(10)/σ(60) = {sq:.2f}", showarrow=False,
                       yanchor="bottom", font=dict(size=10, color=INK2), row=1, col=1)
    # RSI
    r_f = chan_feats(ind["rsi"], c)
    fig.add_trace(go.Scatter(x=X, y=ind["rsi"], mode="lines", line=dict(color=VIOLET, width=2), name="RSI14",
                             hovertemplate="RSI %{y:.1f}<extra></extra>"), row=2, col=1)
    for lvl in (30, 50, 70):
        fig.add_hline(y=lvl, line=dict(color=NEUT, width=1, dash="dot"), row=2, col=1)
    seg(fig, ind["rsi"], 10, VIOLET, "RSI", 2, 1, 3, "dash", norm=r_f["_sx"])
    seg(fig, ind["rsi"], 3, VIOLET, "RSI", 2, 1, 5, "solid", norm=r_f["_sx"])
    markers_cross(fig, crossings(ind["rsi"], np.full(60, 50.0)), ind["rsi"], VIOLET, "RSI пересёк 50", 2, 1)
    # MACD
    m_f = chan_feats(ind["macd"], c)
    fig.add_trace(go.Bar(x=X, y=ind["hist"], marker_color=[UP if x >= 0 else DOWN for x in ind["hist"]], opacity=0.5,
                         name="гистограмма MACD", hovertemplate="гист. %{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=X, y=ind["macd"], mode="lines", line=dict(color=SALMON, width=2), name="MACD 12/26",
                             hovertemplate="MACD %{y:.2f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=X, y=ind["sig"], mode="lines", line=dict(color=INK2, width=1.5, dash="dash"), name="сигнал 9",
                             hovertemplate="сигнал %{y:.2f}<extra></extra>"), row=3, col=1)
    seg(fig, ind["macd"], 10, SALMON, "MACD", 3, 1, 3, "dash", norm=m_f["_sx"])
    seg(fig, ind["macd"], 3, SALMON, "MACD", 3, 1, 5, "solid", norm=m_f["_sx"])
    markers_cross(fig, crossings(ind["macd"], ind["sig"]), ind["macd"], SALMON, "MACD пересёк сигнал", 3, 1)
    # Stochastic
    k_f = chan_feats(ind["k"], c)
    fig.add_trace(go.Scatter(x=X, y=ind["k"], mode="lines", line=dict(color=GOLD, width=2), name="Stoch %K14",
                             hovertemplate="%%K %{y:.1f}<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=X, y=ind["d"], mode="lines", line=dict(color=INK2, width=1.5, dash="dash"), name="%D (SMA3)",
                             hovertemplate="%%D %{y:.1f}<extra></extra>"), row=4, col=1)
    for lvl in (20, 80):
        fig.add_hline(y=lvl, line=dict(color=NEUT, width=1, dash="dot"), row=4, col=1)
    seg(fig, ind["k"], 10, GOLD, "%K", 4, 1, 3, "dash", norm=k_f["_sx"])
    seg(fig, ind["k"], 3, GOLD, "%K", 4, 1, 5, "solid", norm=k_f["_sx"])
    markers_cross(fig, crossings(ind["k"], ind["d"]), ind["k"], GOLD, "%K пересёк %D", 4, 1)
    legend_entry(fig, "отрезок наклона за 10 баров (штрих)", INK, "dash", width=3)
    legend_entry(fig, "отрезок наклона за 3 бара (сплошной)", INK, "solid", width=5)
    legend_entry(fig, "пересечение: ▲ снизу вверх, ▼ сверху вниз; размер - давность (5/10/20 баров)", INK, symbol="triangle-up", width=0)
    fig.update_xaxes(range=[-1, 70], showgrid=False)
    fig.update_xaxes(title_text="бар окна (0 - самый старый, 59 - последний)", row=4, col=1)
    fig.update_yaxes(title_text="цена, USDT", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stoch", row=4, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(bargap=0.1, hovermode="x unified")
    rows = []
    for lab, f in feats.items():
        rows.append([lab, ru(f["slope3"], 2, True), ru(f["slope10"], 2, True), ru(f["distance"], 2, True), ru(f["cross5"], 2, True),
                     ru(f["squeeze"], 2)])
    for lab, f in (("RSI14", r_f), ("MACD", m_f), ("Stoch %K", k_f)):
        rows.append([lab, ru(f["slope3"], 2, True), ru(f["slope10"], 2, True), "-", "-", ru(f["squeeze"], 2)])
    tab = table(rows, ["канал", "наклон 3 бар", "наклон 10 бар", "расстояние до цены (asinh)", "пересечение за 5 бар", "сжатие σ10/σ60"])
    return fig, tab


# ------------------------------------------------------------------ N1: curvature and torsion of the indicator trajectory
def n1_figure(ws):
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3], subplot_titles=[f"{w['label']}: {span_text(w)}" for w in ws],
                        horizontal_spacing=0.02)
    all_k = []
    pre = []
    for w in ws:
        o, h, l, c, v = (w["W"][None, :, i].astype(float) for i in range(5))
        ref = np.abs(c).mean(1)
        ax = [c - NF.ema(c, 20), NF.rsi_series(c, 14)]
        m = NF.ema(c, 12) - NF.ema(c, 26)
        ax.append(m - NF.ema(m, 9))
        refs = [ref, np.ones_like(ref), ref]
        P = np.stack([NF._zs(a, r_)[:, -20:] for a, r_ in zip(ax, refs)], 2)
        kappa, tau, phi, arc, chord = NF.frenet(P)
        pre.append((P[0], kappa[0], tau[0], phi[0], float(arc[0]), float(chord[0])))
        all_k.extend(kappa[0].tolist())
    cmax = float(np.quantile(all_k, 0.98))
    for j, (w, (P, kappa, tau, phi, arc, chord)) in enumerate(zip(ws, pre), 1):
        kv = np.concatenate([[kappa[0]], kappa, [kappa[-1]]])
        bars_ago = 19 - np.arange(20)
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines+markers",
                                   line=dict(color=kv, colorscale="Viridis", cmin=0, cmax=cmax, width=7),
                                   marker=dict(size=4, color=kv, colorscale="Viridis", cmin=0, cmax=cmax,
                                               showscale=(j == 3), colorbar=dict(title="кривизна κ", len=0.7, x=1.0)),
                                   customdata=np.stack([bars_ago, kv], 1), showlegend=False,
                                   hovertemplate="баров назад %{customdata[0]}<br>κ = %{customdata[1]:.2f}<br>(close−EMA20) %{x:.2f}σ"
                                                 "<br>RSI %{y:.2f}σ<br>MACD гист. %{z:.2f}σ<extra></extra>"), row=1, col=j)
        e = P[-1] - P[-2]
        unit = e / (np.linalg.norm(e) + 1e-12)
        seglen = np.linalg.norm(np.diff(P, axis=0), axis=1).mean()
        tip = P[-1] + unit * seglen * 3
        fig.add_trace(go.Scatter3d(x=[P[-1, 0], tip[0]], y=[P[-1, 1], tip[1]], z=[P[-1, 2], tip[2]], mode="lines+markers",
                                   line=dict(color=PINK, width=9), marker=dict(size=[0.1, 7], color=PINK, symbol="diamond"),
                                   name="касательная в конце (направление последнего шага)", showlegend=(j == 1),
                                   hovertemplate="касательная в последней точке<extra></extra>"), row=1, col=j)
        fig.add_trace(go.Scatter3d(x=[P[0, 0]], y=[P[0, 1]], z=[P[0, 2]], mode="markers", marker=dict(size=7, color=INK, symbol="diamond"),
                                   name="начало (20 баров назад)", showlegend=(j == 1), hoverinfo="name"), row=1, col=j)
        sc = f"scene{'' if j == 1 else j}"
        fig.layout[sc].update(xaxis_title="close−EMA20 (σ)", yaxis_title="RSI14 (σ)", zaxis_title="MACD гист. (σ)",
                              bgcolor=SURFACE, aspectmode="cube")
        for a in ("xaxis", "yaxis", "zaxis"):
            fig.layout[sc][a].update(gridcolor=GRID, backgroundcolor=SURFACE, color=MUTED)
    fig.update_layout(margin=dict(l=10, r=10, t=80, b=10))
    rows = []
    for w in ws:
        pass
    return fig, pre


# ------------------------------------------------------------------ N2: Levy areas
def n2_figure(ws, k=30):
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{w['label']}: {span_text(w)}" for w in ws] + [""] * 3,
                        vertical_spacing=0.14, horizontal_spacing=0.06)
    ylabs = ["накопленный знаковый объём, BTC", "RSI14 (шкала −1..1)"]
    for j, w in enumerate(ws, 1):
        Wd = w["W"]
        c = Wd[:, 3]
        dd = np.diff(c)
        cumv = np.concatenate([[0], np.cumsum(np.sign(dd) * Wd[1:, 4])])
        rsi = NF.rsi_series(c[None], 14)[0]
        Xlog = np.log(c)
        for r, Y in ((1, cumv), (2, rsi)):
            xx = (Xlog[-(k + 1):] - Xlog[-(k + 1)]) * 1e4
            yy = Y[-(k + 1):]
            area = 0.0
            pos, neg = [], []
            for i in range(len(xx) - 1):
                tri = [(xx[0], yy[0]), (xx[i], yy[i]), (xx[i + 1], yy[i + 1])]
                a = 0.5 * ((tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0]))
                area += a
                (pos if a >= 0 else neg).append((tri, a))
            for lst, col, pat, nm in ((pos, UP, "/", "знаковая площадь +"), (neg, DOWN, "\\", "знаковая площадь −")):
                xs, ys = [], []
                for tri, a in lst:
                    xs += [p[0] for p in tri] + [tri[0][0], None]
                    ys += [p[1] for p in tri] + [tri[0][1], None]
                if xs:
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=col, opacity=0.35,
                                             line=dict(color=col, width=0.3), fillpattern=dict(shape=pat, fgcolor=col, size=6, solidity=0.3),
                                             name=nm, legendgroup=nm, showlegend=(j == 1 and r == 1), hoverinfo="skip"), row=r, col=j)
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines+markers", line=dict(color=INK, width=2),
                                     marker=dict(size=5, color=np.arange(len(xx)), colorscale=[[0, "#5a5a56"], [1, PINK]]),
                                     customdata=np.arange(len(xx)) - len(xx) + 1, name="путь", legendgroup="путь",
                                     showlegend=(j == 1 and r == 1),
                                     hovertemplate="баров назад %{customdata}<br>лог-доходность от старта %{x:.1f} б.п.<br>%{y:.3f}<extra></extra>"),
                          row=r, col=j)
            fig.add_trace(go.Scatter(x=[xx[-1], xx[0]], y=[yy[-1], yy[0]], mode="lines", line=dict(color=NEUT, width=2, dash="dash"),
                                     name="хорда (замыкание)", legendgroup="хорда", showlegend=(j == 1 and r == 1), hoverinfo="skip"),
                          row=r, col=j)
            lev = NF.levy(Xlog[None, -(k + 1):], Y[None, -(k + 1):])[0]
            fig.add_annotation(text=f"A = {area:+.3g}; Леви 2A/(TVx·TVy) = {lev:+.2f}", xref="x domain", yref="y domain", x=0.02, y=0.98,
                               showarrow=False, xanchor="left", yanchor="top", font=dict(size=11, color=INK),
                               bgcolor="rgba(18,18,17,0.7)", row=r, col=j)
            fig.update_xaxes(title_text="лог-доходность с начала отрезка, б.п." if r == 2 else None, row=r, col=j)
            fig.update_yaxes(title_text=ylabs[r - 1] if j == 1 else None, row=r, col=j)
    return fig


# ------------------------------------------------------------------ N3: phase portrait with winding
def n3_figure(ws):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{w['label']}: {span_text(w)}" for w in ws], horizontal_spacing=0.07)
    res = []
    for j, w in enumerate(ws, 1):
        c = w["W"][:, 3][None]
        s = NF.ema(c, 5)
        vel = np.diff(s, axis=1)
        acc = np.diff(vel, axis=1)
        sv = vel.std(1, keepdims=True) + 1e-12 * np.abs(c).mean(1, keepdims=True)
        sa = acc.std(1, keepdims=True) + 1e-12 * np.abs(c).mean(1, keepdims=True)
        vn, an = (vel[:, 1:] / sv)[0], (acc / sa)[0]
        th = np.arctan2(an, vn)
        dth = np.arctan2(np.sin(np.diff(th)), np.cos(np.diff(th)))
        wind = float(dth[-30:].sum())
        lim = float(max(np.abs(vn).max(), np.abs(an).max()) * 1.15)
        quads = [((0, 0), (lim, lim), "ускоряется вверх"), ((-lim, 0), (0, lim), "падение тормозит"),
                 ((-lim, -lim), (0, 0), "ускоряется вниз"), ((0, -lim), (lim, 0), "рост тормозит")]
        for (x0, y0), (x1, y1), name in quads:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor="rgba(242,241,236,0.04)", line_width=0, layer="below",
                          row=1, col=j)
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=name, showarrow=False, font=dict(size=10, color=MUTED), row=1, col=j)
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=j)
        fig.add_vline(x=0, line=dict(color=AXIS, width=1), row=1, col=j)
        bars_ago = np.arange(len(vn))[::-1]
        fig.add_trace(go.Scatter(x=vn, y=an, mode="lines+markers", line=dict(color="rgba(195,194,183,0.5)", width=1.5),
                                 marker=dict(symbol="arrow", size=11, angleref="previous", color=bars_ago, colorscale=[[0, PINK], [1, "#4a4a46"]],
                                             showscale=False),
                                 customdata=np.stack([bars_ago, th, np.hypot(vn, an)], 1), showlegend=False,
                                 hovertemplate="баров назад %{customdata[0]}<br>скорость %{x:.2f}σ, ускорение %{y:.2f}σ"
                                               "<br>угол θ %{customdata[1]:.2f} рад, радиус %{customdata[2]:.2f}<extra></extra>"), row=1, col=j)
        fig.add_trace(go.Scatter(x=[vn[-1]], y=[an[-1]], mode="markers", marker=dict(symbol="star", size=15, color=INK, line=dict(color=PINK, width=2)),
                                 name="последний бар", showlegend=(j == 1), hovertemplate="конец: θ = %{customdata:.2f} рад<extra></extra>",
                                 customdata=[th[-1]]), row=1, col=j)
        fig.add_trace(go.Scatter(x=[0, vn[-1]], y=[0, an[-1]], mode="lines", line=dict(color=INK, width=1.5, dash="dot"), showlegend=False,
                                 hoverinfo="skip"), row=1, col=j)
        fig.add_annotation(text=f"намотка за 30 баров: {wind:+.2f} рад ({wind / (2 * np.pi):+.2f} оборота)<br>θ в конце = {th[-1]:+.2f} рад",
                           xref="x domain", yref="y domain", x=0.02, y=0.02, showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=11, color=INK), bgcolor="rgba(18,18,17,0.7)", row=1, col=j)
        fig.update_xaxes(title_text="скорость EMA5 (σ)", range=[-lim, lim], row=1, col=j)
        fig.update_yaxes(title_text="ускорение EMA5 (σ)" if j == 1 else None, range=[-lim, lim], row=1, col=j)
        res.append(wind)
    return fig


# ------------------------------------------------------------------ N4: convex hulls
def hull_vertices(y):
    """Upper concave hull vertices of the points (t, y[t]) (gift wrapping, ties go to the farthest) - as features.upper_hull."""
    n = len(y)
    cur = 0
    out = [0]
    while cur < n - 1:
        t = np.arange(cur + 1, n)
        sl = (y[cur + 1:] - y[cur]) / (t - cur)
        m = sl.max()
        cand = t[sl >= m - 1e-12 * (1 + abs(m))]
        cur = int(cand.max())
        out.append(cur)
    return np.array(out), float(m)


def n4_figure(ws):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{w['label']}: {span_text(w)}" for w in ws], horizontal_spacing=0.06)
    info = []
    for j, w in enumerate(ws, 1):
        W = w["W"]
        o, h, l, c, v = W.T
        candle(fig, W, 1, j)
        hh, ll = h[:-1], l[:-1]
        iu, su = hull_vertices(hh)
        il, sl_ = hull_vertices(-ll)
        sl_ = -sl_
        U59 = hh[-1] + su
        L59 = ll[-1] + sl_
        pos = (c[-1] - L59) / (U59 - L59) if U59 - L59 > 1e-9 * abs(c).mean() else 0.5
        fig.add_trace(go.Scatter(x=iu, y=hh[iu], mode="lines+markers", line=dict(color=PINK, width=2), marker=dict(size=7, color=PINK, symbol="circle"),
                                 name="верхняя оболочка (по high)", legendgroup="u", showlegend=(j == 1),
                                 hovertemplate="вершина: бар %{x}, high %{y:.2f}<extra></extra>"), row=1, col=j)
        fig.add_trace(go.Scatter(x=il, y=ll[il], mode="lines+markers", line=dict(color=GOLD, width=2), marker=dict(size=7, color=GOLD, symbol="circle"),
                                 name="нижняя оболочка (по low)", legendgroup="l", showlegend=(j == 1),
                                 hovertemplate="вершина: бар %{x}, low %{y:.2f}<extra></extra>"), row=1, col=j)
        fig.add_trace(go.Scatter(x=[iu[-1], 59], y=[hh[iu[-1]], U59], mode="lines", line=dict(color=PINK, width=3, dash="dash"),
                                 name="сопротивление: последнее ребро, продолженное до бара 59", legendgroup="ur", showlegend=(j == 1),
                                 hovertemplate="сопротивление к бару 59: %{y:.2f}<extra></extra>"), row=1, col=j)
        fig.add_trace(go.Scatter(x=[il[-1], 59], y=[ll[il[-1]], L59], mode="lines", line=dict(color=GOLD, width=3, dash="dash"),
                                 name="поддержка: последнее ребро, продолженное до бара 59", legendgroup="lr", showlegend=(j == 1),
                                 hovertemplate="поддержка к бару 59: %{y:.2f}<extra></extra>"), row=1, col=j)
        fig.add_trace(go.Scatter(x=[59], y=[c[-1]], mode="markers", marker=dict(symbol="star", size=14, color=INK, line=dict(color=INK2, width=1)),
                                 name="закрытие бара 59", legendgroup="cl", showlegend=(j == 1),
                                 hovertemplate=f"закрытие {c[-1]:.2f}<br>положение между линиями {pos:.2f}<extra></extra>"), row=1, col=j)
        fig.add_annotation(text=f"положение = {pos:.2f} (0 - поддержка, 1 - сопротивление)<br>вершин: {len(iu) + len(il)}" + ("<br>линии к бару 59 пересеклись: нейтральное 0,5 (как в коде)" if U59 - L59 <= 1e-9 * abs(c).mean() else ""),
                           xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=11, color=INK), bgcolor="rgba(18,18,17,0.7)", row=1, col=j)
        fig.update_xaxes(range=[-1, 62], rangeslider_visible=False, title_text="бар окна", row=1, col=j)
        info.append(pos)
    fig.update_yaxes(title_text="цена, USDT", row=1, col=1)
    return fig


# ------------------------------------------------------------------ N5: Wasserstein shift
def ecdf_xy(a):
    s = np.sort(a)
    return s, np.arange(1, len(s) + 1) / len(s)


def n5_figures(ws):
    figs = []
    pairs = [("первая половина против второй (30 + 29 доходностей)", lambda r: (r[:29], r[30:])),
             ("последние 15 баров против предыдущих 44", lambda r: (r[:-15], r[-15:]))]
    names = [("первая половина", "вторая половина"), ("предыдущие 44", "последние 15")]
    tt = [f"{w['label']} · {w['t0']:%m-%d %H:%M} · половины" for w in ws] + [f"{w['label']} · {w['t0']:%m-%d %H:%M} · 15 против 44" for w in ws]
    f_ecdf = make_subplots(rows=2, cols=3, subplot_titles=tt, vertical_spacing=0.16, horizontal_spacing=0.06)
    f_hist = make_subplots(rows=2, cols=3, subplot_titles=tt, vertical_spacing=0.16, horizontal_spacing=0.06)
    for j, w in enumerate(ws, 1):
        c = w["W"][:, 3]
        r = np.diff(c) / c[:-1] * 1e4
        for i, (title, fn) in enumerate(pairs, 1):
            a, b = fn(r)
            qa, qb = np.quantile(a, NF._Q), np.quantile(b, NF._Q)
            w1 = float(np.abs(qa - qb).mean())
            w1n = float(NF._w1(a[None], b[None], np.array([r.std()]))[0])
            grid = np.sort(np.concatenate([a, b]))
            Fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
            Fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
            na, nb = names[i - 1]
            ca, cb = (VIOLET, GOLD)
            fig = f_ecdf
            fig.add_trace(go.Scatter(x=grid, y=Fa, mode="lines", line=dict(color=ca, width=2, shape="hv"), name=na, legendgroup=na,
                                     showlegend=(j == 1), hovertemplate=f"{na}: F(%{{x:.1f}} б.п.) = %{{y:.2f}}<extra></extra>"), row=i, col=j)
            fig.add_trace(go.Scatter(x=grid, y=Fb, mode="lines", line=dict(color=cb, width=2, shape="hv"), name=nb, legendgroup=nb,
                                     showlegend=(j == 1), fill="tonexty", fillcolor="rgba(213,81,129,0.28)",
                                     hovertemplate=f"{nb}: F(%{{x:.1f}} б.п.) = %{{y:.2f}}<extra></extra>"), row=i, col=j)
            fig.add_annotation(text=f"W1 = {w1:.2f} б.п. ({w1n:.2f}σ); площадь между кривыми = W1<br>медианный сдвиг {np.median(b) - np.median(a):+.2f} б.п.",
                               xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, xanchor="left", yanchor="top",
                               font=dict(size=10, color=INK), bgcolor="rgba(18,18,17,0.7)", row=i, col=j)
            f_ecdf.update_xaxes(title_text="1-барная доходность, б.п." if i == 2 else None, row=i, col=j)
            f_ecdf.update_yaxes(title_text="доля баров ≤ x" if j == 1 else None, row=i, col=j)
            lo, hi = float(np.min(r)), float(np.max(r))
            bins = dict(start=lo, end=hi, size=(hi - lo) / 14)
            f_hist.add_trace(go.Histogram(x=a, xbins=bins, marker_color=ca, opacity=0.6, name=na, legendgroup=na, showlegend=(j == 1),
                                          hovertemplate=f"{na}: %{{x:.1f}} б.п., баров %{{y}}<extra></extra>"), row=i, col=j)
            f_hist.add_trace(go.Histogram(x=b, xbins=bins, marker_color=cb, opacity=0.6, name=nb, legendgroup=nb, showlegend=(j == 1),
                                          hovertemplate=f"{nb}: %{{x:.1f}} б.п., баров %{{y}}<extra></extra>"), row=i, col=j)
            f_hist.add_annotation(text=f"W1 = {w1:.2f} б.п.", xref="x domain", yref="y domain", x=0.98, y=0.98, showarrow=False, xanchor="right",
                                  yanchor="top", font=dict(size=10, color=INK), row=i, col=j)
            f_hist.update_xaxes(title_text="1-барная доходность, б.п." if i == 2 else None, row=i, col=j)
            f_hist.update_yaxes(title_text="число баров" if j == 1 else None, row=i, col=j)
    f_hist.update_layout(barmode="overlay")
    return f_ecdf, f_hist


# ------------------------------------------------------------------ N6: Fisher-Rao path on the Poincare half-plane
def n6_figure(ws):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{w['label']}: {span_text(w)}" for w in ws], horizontal_spacing=0.06)
    for j, w in enumerate(ws, 1):
        c = w["W"][:, 3]
        r = np.diff(c) / c[:-1]
        r = r / (r.std() + NF.EPS)
        mus = np.array([r[10 * k:10 * k + 9].mean() for k in range(6)])
        sig = np.maximum(np.array([r[10 * k:10 * k + 9].std() for k in range(6)]), 0.05)
        x = mus / np.sqrt(2)

        def arc(i0, i1):
            x1, s1, x2, s2 = x[i0], sig[i0], x[i1], sig[i1]
            if abs(x2 - x1) < 1e-9:
                return np.array([x1, x2]), np.array([s1, s2])
            cc = (x2 ** 2 + s2 ** 2 - x1 ** 2 - s1 ** 2) / (2 * (x2 - x1))
            rad = np.sqrt((x1 - cc) ** 2 + s1 ** 2)
            a1, a2 = np.arctan2(s1, x1 - cc), np.arctan2(s2, x2 - cc)
            aa = np.linspace(a1, a2, 40)
            return cc + rad * np.cos(aa), rad * np.sin(aa)

        d = NF.fr_dist(mus[:-1], sig[:-1], mus[1:], sig[1:])
        net = float(NF.fr_dist(mus[0], sig[0], mus[-1], sig[-1]))
        for i in range(5):
            ax_, ay_ = arc(i, i + 1)
            fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color=PINK, width=2.5), showlegend=False,
                                     hovertemplate=f"геодезическая {i + 1}→{i + 2}: расстояние Фишера-Рао {d[i]:.2f}<extra></extra>"), row=1, col=j)
        ax_, ay_ = arc(0, 5)
        fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color=INK2, width=2, dash="dash"), name="прямая геодезическая 1→6 (чистое смещение)",
                                 showlegend=(j == 1), hovertemplate=f"чистое смещение {net:.2f}<extra></extra>"), row=1, col=j)
        fig.add_trace(go.Scatter(x=x, y=sig, mode="markers+text", text=[str(i + 1) for i in range(6)], textposition="top center",
                                 marker=dict(size=12, color=np.arange(6), colorscale=[[0, "#4a4a46"], [1, PINK]], line=dict(color=INK, width=1)),
                                 customdata=np.stack([mus, sig, np.arange(6) * 10, np.arange(6) * 10 + 9], 1), showlegend=False,
                                 hovertemplate="подокно %{text} (бары %{customdata[2]}-%{customdata[3]})<br>μ = %{customdata[0]:.2f}σ_окна<br>σ = %{customdata[1]:.2f}<extra></extra>"),
                      row=1, col=j)
        fig.add_annotation(text=f"длина пути {d.sum():.2f}; чистое смещение {net:.2f}", xref="x domain", yref="y domain", x=0.02, y=0.98,
                           showarrow=False, xanchor="left", yanchor="top", font=dict(size=11, color=INK), bgcolor="rgba(18,18,17,0.7)", row=1, col=j)
        pad = 0.25 * max(np.ptp(x), 0.3)
        fig.update_xaxes(title_text="μ/√2 (среднее доходности в σ окна)", range=[x.min() - pad, x.max() + pad], row=1, col=j)
        fig.update_yaxes(title_text="σ подокна (полуплоскость σ>0)" if j == 1 else None, range=[0, sig.max() * 1.4], row=1, col=j)
    return fig


# ------------------------------------------------------------------ N7: convexity (control)
def n7_figure(ws):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{w['label']}: {span_text(w)}" for w in ws], horizontal_spacing=0.06)
    colors = {15: PINK, 30: GOLD, 60: VIOLET}
    for j, w in enumerate(ws, 1):
        W = w["W"]
        c = W[:, 3]
        y = np.log(c)
        sd = np.diff(y).std() + NF.EPS
        fig.add_trace(go.Scatter(x=X, y=c, mode="lines", line=dict(color=INK, width=1.5), name="закрытие", legendgroup="c", showlegend=(j == 1),
                                 hovertemplate="close %{y:.2f}<extra></extra>"), row=1, col=j)
        for k in (60, 30, 15):
            t = np.arange(k) - (k - 1) / 2.0
            A = np.vstack([t ** 2, t, np.ones(k)]).T
            coef = np.linalg.lstsq(A, y[-k:], rcond=None)[0]
            fit = np.exp(A @ coef)
            qn = coef[0] / sd
            fig.add_trace(go.Scatter(x=X[-k:], y=fit, mode="lines", line=dict(color=colors[k], width=3), name=f"парабола по {k} барам",
                                     legendgroup=str(k), showlegend=(j == 1),
                                     hovertemplate=f"{k} баров: выпуклость a/σ = {qn:+.3f}<br>fit %{{y:.2f}}<extra></extra>"), row=1, col=j)
        t = np.arange(60) - 29.5
        a60 = np.polyfit(t, y, 2)[0] / sd
        fig.add_annotation(text="выпуклость a/σ: " + ", ".join(f"{k}: {NF.quad_coef(y[None, -k:])[0] / sd:+.3f}" for k in (15, 30, 60)),
                           xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=11, color=INK), bgcolor="rgba(18,18,17,0.7)", row=1, col=j)
        fig.update_xaxes(title_text="бар окна", row=1, col=j)
    fig.update_yaxes(title_text="цена, USDT", row=1, col=1)
    return fig


# ------------------------------------------------------------------ H67: the next hour's path in its own clock
def h67_figure(ws):
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{w['label']}: следующий час после {w['t1']:%Y-%m-%d %H:%M}" for w in ws] + [""] * 3,
                        vertical_spacing=0.17, horizontal_spacing=0.06)
    stats = []
    rng = np.random.default_rng(0)
    for j, w in enumerate(ws, 1):
        nxt = w["next"]
        c0 = w["W"][-1, 3]
        lc = np.log(nxt[:, 3])
        r = np.diff(np.concatenate([[np.log(c0)], lc]))
        q = float((r ** 2).sum())
        sr = float(r.sum())
        top = max(np.log(c0), np.log(nxt[:, 1]).max())
        bot = min(np.log(c0), np.log(nxt[:, 2]).min())
        rho = abs(sr) / (top - bot)
        vr = sr * sr / q
        lag1 = float((r[1:] * r[:-1]).sum())
        z2 = ((sr * sr - q) / 2 - lag1) / q
        path = np.concatenate([[0], np.cumsum(r)]) * 1e4
        tau = np.concatenate([[0], np.cumsum(r ** 2)]) / q
        tt = np.arange(61) / 60
        for row, xx, ttl, env_x in ((1, np.arange(61), "минуты следующего часа", tt), (2, tau, "собственные часы: накопленная доля квадратичной вариации", tau)):
            band = 1.645 * np.sqrt(q * env_x) * 1e4
            fig.add_trace(go.Scatter(x=xx, y=band, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=row, col=j)
            fig.add_trace(go.Scatter(x=xx, y=-band, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(137,135,129,0.22)",
                                     name="90% коридор броуновского пути с той же дисперсией", legendgroup="bm",
                                     showlegend=(j == 1 and row == 1), hoverinfo="skip"), row=row, col=j)
            fig.add_trace(go.Scatter(x=xx, y=path, mode="lines+markers", line=dict(color=PINK if row == 1 else VIOLET, width=2),
                                     marker=dict(size=4), name="путь (лог-цена от закрытия окна)", legendgroup="p" + str(row),
                                     showlegend=(j == 1), customdata=np.arange(61),
                                     hovertemplate="бар %{customdata}: %{y:.1f} б.п.<br>" + ("" if row == 1 else "доля квадр. вариации %{x:.2f}<br>") + "<extra></extra>"),
                          row=row, col=j)
            fig.update_xaxes(title_text=ttl if True else None, row=row, col=j)
        fig.add_annotation(text=f"ρ = {rho:.2f}, VR = {vr:.2f}, Z2 = {z2:+.2f}", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=11, color=INK), bgcolor="rgba(18,18,17,0.7)", row=1, col=j)
        stats.append((rho, vr, z2))
    fig.update_yaxes(title_text="лог-цена от закрытия окна, б.п.", row=1, col=1)
    fig.update_yaxes(title_text="то же, по собственным часам", row=2, col=1)
    return fig, stats


# ------------------------------------------------------------------ H80: synthetic indicators from a random causal TCN (control C1)
def random_tcn(W, K=12, seed=7, ch=16):
    """A causal dilated TCN with random weights: [60,5] -> K series [K,60] (the control C1 of H80)."""
    rng = np.random.default_rng(seed)
    o, h, l, c, v = W.T
    r = np.concatenate([[0], np.diff(np.log(c))])
    inp = np.stack([r / (r.std() + 1e-12), np.log(h / l + 1e-9), np.log1p(v)], 0)
    inp = (inp - inp.mean(1, keepdims=True)) / (inp.std(1, keepdims=True) + 1e-9)
    x = inp
    cin = 3
    for dil in (1, 2, 4):
        Wt = rng.normal(0, 1 / np.sqrt(cin * 3), (ch, cin, 3))
        out = np.zeros((ch, 60))
        for k in range(3):
            sh = (2 - k) * dil
            xs = np.concatenate([np.zeros((cin, sh)), x[:, :60 - sh]], 1) if sh else x
            out += np.einsum("oc,ct->ot", Wt[:, :, k], xs)
        x = np.tanh(out)
        cin = ch
    Wo = rng.normal(0, 1 / np.sqrt(ch), (K, ch))
    ser = Wo @ x
    return np.cumsum(ser, axis=1) * 0.3 + ser           # a level-like series: the cumulative sum plus its own value


def h80_figure(ws, show=6):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{w['label']}: {span_text(w)}" for w in ws], horizontal_spacing=0.05)
    heat_rows = []
    pal = [PINK, GOLD, VIOLET, TAN, SALMON, "#c3c2b7"]
    for j, w in enumerate(ws, 1):
        W = w["W"]
        c = W[:, 3]
        cz = (c - c.mean()) / (c.std() + 1e-12)
        ser = random_tcn(W)
        fig.add_trace(go.Scatter(x=X, y=cz, mode="lines", line=dict(color=INK, width=3), name="закрытие (z-оценка)", legendgroup="c",
                                 showlegend=(j == 1), hovertemplate="close z = %{y:.2f}<extra></extra>"), row=1, col=j)
        feats = []
        for i in range(ser.shape[0]):
            x = (ser[i] - ser[i].mean()) / (ser[i].std() + 1e-12)
            f = chan_feats(x, cz)
            feats.append([f["slope3"], f["slope10"], f["distance"], f["cross5"], f["squeeze"]])
            if i < show:
                col = pal[i]
                fig.add_trace(go.Scatter(x=X, y=x, mode="lines", line=dict(color=col, width=1.6), name=f"синтетический ряд {i + 1} (случайная TCN)",
                                         legendgroup=f"s{i}", showlegend=(j == 1), opacity=0.9,
                                         hovertemplate=f"ряд {i + 1}: %{{y:.2f}}<extra></extra>"), row=1, col=j)
                for k, wd, dsh in ((10, 3, "dash"), (3, 5, "solid")):
                    fig.add_trace(go.Scatter(x=[59 - k, 59], y=[x[59 - k], x[59]], mode="lines+markers", line=dict(color=col, width=wd, dash=dsh),
                                             marker=dict(size=[4, 7], color=col, line=dict(color=INK, width=1)), showlegend=False,
                                             hovertemplate=f"ряд {i + 1}: наклон за {k} бар = {(x[59] - x[59 - k]) / k / f['_sx']:+.2f}<extra></extra>"),
                                  row=1, col=j)
                markers_cross(fig, crossings(cz, x, 20), x, col, f"Цена пересекла ряд {i + 1}", 1, j)
        fig.add_vrect(x0=49.5, x1=59.5, fillcolor="rgba(242,241,236,0.07)", line_width=0, row=1, col=j)
        fig.update_xaxes(title_text="бар окна", range=[-1, 62], row=1, col=j)
        heat_rows.append(np.array(feats))
    fig.update_yaxes(title_text="z-оценка (ряды и цена в одних единицах)", row=1, col=1)
    # heat map of the geometry layer's output: series x feature
    names = ["наклон 3", "наклон 10", "расстояние", "пересечение 5", "сжатие"]
    hm = make_subplots(rows=1, cols=3, subplot_titles=[f"{w['label']}" for w in ws], horizontal_spacing=0.09, shared_yaxes=True)
    for j, a in enumerate(heat_rows, 1):
        zn = a.T / np.maximum(np.abs(a.T).max(1, keepdims=True), 1e-9)
        hm.add_trace(go.Heatmap(z=zn, customdata=a.T, x=[f"ряд {i + 1}" for i in range(a.shape[0])], y=names,
                                colorscale=[[0, DOWN], [0.5, SURFACE], [1, UP]], zmin=-1, zmax=1, showscale=(j == 3),
                                colorbar=dict(title="значение / макс. модуль в строке"), text=np.round(a.T, 2), texttemplate="%{text}",
                                textfont=dict(size=9, color=INK), hovertemplate="%{x}, %{y}: %{customdata:.3f}<extra></extra>", xgap=1, ygap=1),
                     row=1, col=j)
    return fig, hm

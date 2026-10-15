"""T10 figures in the tactical_report style (theme.py): plotly JSON + PNG (headless Edge). Reads results.json / look_rows.npz (and the forward file
through T09's guard for the price window). Models: the chosen one solid amber, the others in the other-series slots; horizon colours are not used.
usage: python make_figs.py"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import t10_lib as L

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
spec = importlib.util.spec_from_file_location("nt_theme", os.path.join(ROOT, "src", "neural_trade", "visualization", "theme.py"))
TH = importlib.util.module_from_spec(spec); sys.modules["nt_theme"] = TH; spec.loader.exec_module(TH); TH.register()
OS = TH.OTHER_SERIES
RES = json.load(open(os.path.join(HERE, "results.json"))); ROW = np.load(os.path.join(HERE, "look_rows.npz"))
CH = RES["chosen"]
COL = {"B1": OS[5], "B2": OS[2], "H1": OS[3], "H2": OS[1], "H3": OS[0], "B0": "#8a8984"}
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
LOOKS = (("look1", "Look 1: 2024-07..2025-07-20, Bitstamp history, monthly walk-forward"), ("look2", "Look 2: 2025-11..2026-09, Binance forward file, frozen fit"))
VENUE = {"look1": "Bitstamp BTC/USD 1-minute history (the cut file; not Binance).", "look2": "Binance BTCUSDT forward file (a different venue from the Bitstamp history)."}


def save(fig, name, w=1500, h=900):
    fig.write_json(os.path.join(FIG, name + ".json"))
    html = os.path.join(tempfile.gettempdir(), f"t10_{name}.html")
    fig.write_html(html, include_plotlyjs=True, full_html=True, config=dict(displayModeBar=False))
    png = os.path.join(FIG, name + ".png")
    if os.path.exists(png):
        os.remove(png)
    prof = os.path.join(tempfile.gettempdir(), "edge_prof_t10")
    subprocess.run([EDGE, "--headless=new", "--user-data-dir=" + prof, "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    "--hide-scrollbars", f"--screenshot={png}", f"--window-size={w},{h}", "--virtual-time-budget=30000", "file:///" + html.replace("\\", "/")],
                   capture_output=True, text=True)
    print(name, os.path.exists(png), flush=True)


def style(fig, title, subtitle, height, width=1500, b=56):
    lines = textwrap.wrap(subtitle, int(width / 6.6)); sub = "<br>".join(lines)
    t = 62 + 17 * len(lines) + 84
    TH.apply(fig, title=title, subtitle=sub, height=height)
    fig.update_layout(margin=dict(t=t, b=b, l=64, r=24), legend=dict(orientation="h", yanchor="bottom", y=1 + 36 / max(height - t - b, 200), xanchor="left", x=0.0))
    return fig


def ax(r, c):
    k = (r - 1) * 3 + c
    return "" if k == 1 else str(k)


def fig_reliability(nm, title):
    ttl = [f"X = {X} bps, T = {T} min" for T in L.TS for X in L.XS]
    fig = make_subplots(rows=3, cols=3, subplot_titles=ttl, vertical_spacing=0.09, horizontal_spacing=0.06)
    for i, T in enumerate(L.TS):
        for j, X in enumerate(L.XS):
            r, c = i + 1, j + 1; y = ROW[f"{nm}|{T}|y|{X}"].astype(float)
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TH.NEUTRAL, width=1), showlegend=False, hoverinfo="skip"), r, c)
            for m in ("B1", CH):
                p = ROW[f"{nm}|{T}|p|{m}|{X}"].astype(float); rel = L.reliability(p, y)
                xs = [d["p_mean"] for d in rel if d["n"] > 0]; ys = [d["y_mean"] for d in rel if d["n"] > 0]; ns = [d["n"] for d in rel if d["n"] > 0]
                se = [1.96 * np.sqrt(max(a * (1 - a), 1e-6) / n) for a, n in zip(ys, ns)]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=m + (" (chosen)" if m == CH else ""), legendgroup=m, showlegend=(i == 0 and j == 0),
                                         line=dict(color=COL[m], width=2.4 if m == CH else 1.6),
                                         marker=dict(color=COL[m], size=[5 + 4 * np.log10(n) for n in ns], symbol="circle" if m == CH else "square"),
                                         error_y=dict(type="data", array=se, color=COL[m], thickness=1, width=2) if m == CH else None,
                                         customdata=ns, hovertemplate=f"{m} p=%{{x:.3f}} freq=%{{y:.3f}} n=%{{customdata}}<extra></extra>"), r, c)
            e1 = L.ece(ROW[f"{nm}|{T}|p|B1|{X}"].astype(float), y); ec = L.ece(ROW[f"{nm}|{T}|p|{CH}|{X}"].astype(float), y)
            sk = RES[nm][f"{X}|{T}"][CH]["skill_vs_B1"]
            fig.add_annotation(x=0.02, y=0.98, xref=f"x{ax(r, c)} domain", yref=f"y{ax(r, c)} domain",
                               text=f"n={len(y)}, base {y.mean():.3f}<br>ECE {CH} {100 * ec:.2f}pp, B1 {100 * e1:.2f}pp<br>skill {100 * sk:+.1f}%", showarrow=False,
                               xanchor="left", yanchor="top", font=dict(size=10, color=TH.INK), bgcolor="rgba(18,18,17,0.7)")
            fig.update_xaxes(range=[0, 1], title_text="predicted P(hit)" if i == 2 else None, row=r, col=c)
            fig.update_yaxes(range=[0, 1], title_text="observed frequency" if j == 0 else None, row=r, col=c)
    style(fig, title=f"T10 reliability: P(|move| >= X bps within T minutes). {title}",
          subtitle=f"Ten equal-width bins of predicted probability; marker size grows with the bin's anchor count; bars are 95% binomial errors for the chosen model "
                   f"({CH}) assuming independent anchors (they do not overlap). Grey diagonal = perfect. B1 = Brownian first passage with the trailing realised volatility. "
                   f"Primary cell: X = 50, T = 60 (middle row, middle column). Data: {VENUE[nm]}", height=1150, width=1500)
    return fig


def fig_skill_by_month():
    ttl = [f"X = {X} bps, T = {T} min" for T in L.TS for X in L.XS]
    fig = make_subplots(rows=3, cols=3, subplot_titles=ttl, vertical_spacing=0.09, horizontal_spacing=0.06)
    for i, T in enumerate(L.TS):
        for j, X in enumerate(L.XS):
            r, c = i + 1, j + 1
            fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1), row=r, col=c)
            for m in ("B2", "H1", "H2", "H3"):
                xs, ys = [], []
                for nm in ("look1", "look2"):
                    for k, v in RES[nm][f"{X}|{T}"][m]["monthly"].items():
                        xs.append(k); ys.append(100 * (1 - v["brier"] / v["brier_b1"]))
                    if nm == "look1":
                        xs.append(xs[-1]); ys.append(None)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=m + (" (chosen)" if m == CH else ""), legendgroup=m, showlegend=(i == 0 and j == 0),
                                         line=dict(color=COL[m], width=2.6 if m == CH else 1.3), marker=dict(color=COL[m], size=5 if m == CH else 3),
                                         hovertemplate=m + " %{x}: %{y:+.1f}%<extra></extra>", connectgaps=False), r, c)
            fig.update_xaxes(type="category", tickangle=-60, tickfont=dict(size=8), row=r, col=c)
            fig.update_yaxes(title_text="Brier skill over B1, %" if j == 0 else None, row=r, col=c)
    n15 = int(RES["look1"]["50|15"]["B1"]["n"] / 13); n240 = int(RES["look1"]["50|240"]["B1"]["n"] / 13)
    style(fig, title="T10 Brier skill over B1 (trailing-volatility first passage) by month",
          subtitle=f"Skill = 1 - Brier(model) / Brier(B1) within each month; the left block is Look 1 (Bitstamp history, monthly walk-forward), the right block Look 2 (Binance forward file, "
                   f"frozen fit; 2025-08..10 absent by design). {CH} is the model chosen on 2019-01..2024-06. A month holds about {n240} (T = 240) to {n15} (T = 15) anchors "
                   f"on average, so the T = 240 panels are noisy.", height=1150)
    return fig


def window_rule():
    """Look 2, T = 60, X = 50. Among UTC days with 24 valid hourly anchors, the day whose number of hits is the median (lower median, earliest day on ties)."""
    ta = pd.DatetimeIndex(ROW["look2|60|ta"]); y = ROW["look2|60|y|50"]
    df = pd.DataFrame({"day": ta.normalize(), "y": y}); g = df.groupby("day").agg(n=("y", "size"), hits=("y", "sum")); g = g[g.n == 24]
    med = np.sort(g.hits.values)[(len(g) - 1) // 2]
    return g.index[g.hits.values == med][0], int(med), len(g)


def fig_window():
    d, med, ndays = window_rule()
    ta = pd.DatetimeIndex(ROW["look2|60|ta"]); y = ROW["look2|60|y|50"]; p = ROW[f"look2|60|p|{CH}|50"]; pb = ROW["look2|60|p|B1|50"]
    k = np.where(ta.normalize() == d)[0]
    fwd = L.guard.load_ohlcv("fwd"); seg = fwd[(fwd.index >= d) & (fwd.index < d + pd.Timedelta(hours=25))]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=(f"BTC/USDT 1-minute close, {d.date()} (UTC), +-50 bps band from each hourly close", f"P(hit within 60 min): {CH} and B1; marker = outcome"))
    fig.add_trace(go.Scatter(x=seg.index, y=seg["close"], mode="lines", name="close", line=dict(color=TH.INK, width=1.3), hovertemplate="%{x|%H:%M} %{y:,.1f}<extra></extra>"), 1, 1)
    for t, yy, pp in zip(ta[k], y[k], p[k]):
        c0 = float(fwd["close"].loc[t - pd.Timedelta(minutes=1)]); col = TH.SERIES[1] if yy else TH.SERIES[0]
        fig.add_shape(type="rect", x0=t, x1=t + pd.Timedelta(minutes=60), y0=c0 * np.exp(-0.005), y1=c0 * np.exp(0.005), line=dict(color=col, width=1),
                      fillcolor=col, opacity=0.16, row=1, col=1)
        fig.add_annotation(x=t + pd.Timedelta(minutes=30), y=c0 * np.exp(0.005), text=f"{pp:.2f}", showarrow=False, yshift=9, font=dict(size=10, color=TH.INK), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="band touched (hit)", marker=dict(color=TH.SERIES[1], symbol="diamond", size=10)), 1, 1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="band not touched", marker=dict(color=TH.SERIES[0], symbol="circle", size=9)), 1, 1)
    fig.add_trace(go.Scatter(x=ta[k], y=pb[k], mode="lines+markers", name="B1", line=dict(color=COL["B1"], width=1.5), marker=dict(symbol="square", size=6, color=COL["B1"])), 2, 1)
    fig.add_trace(go.Scatter(x=ta[k], y=p[k], mode="lines", name=CH, line=dict(color=COL[CH], width=2.4)), 2, 1)
    for t, yy, pp in zip(ta[k], y[k], p[k]):
        fig.add_trace(go.Scatter(x=[t], y=[pp], mode="markers", showlegend=False,
                                 marker=dict(color=TH.SERIES[1] if yy else TH.SERIES[0], symbol="diamond" if yy else "circle", size=10, line=dict(color=TH.INK, width=1)),
                                 hovertemplate=f"{CH} P={pp:.3f} hit={bool(yy)}<extra></extra>"), 2, 1)
    c0s = fwd["close"].reindex(ta[k] - pd.Timedelta(minutes=1)).values
    fig.update_yaxes(title_text="price", tickformat=",.0f", range=[float(min(seg["close"].min(), (c0s * np.exp(-0.005)).min())) * 0.9994, float(max(seg["close"].max(), (c0s * np.exp(0.005)).max())) * 1.0012], row=1, col=1); fig.update_yaxes(title_text="P(hit)", range=[0, 1], row=2, col=1)
    fig.update_xaxes(range=[d, d + pd.Timedelta(hours=25)])
    style(fig, title=f"T10 one real day: P(|move| >= 50 bps within 60 minutes), {d.date()}",
          subtitle=f"Chosen by rule, before looking at it: among the {ndays} Look 2 UTC days with 24 hourly anchors, the day whose hit count ({med} of 24) is the median (lower median, "
                   f"earliest on ties). Each shaded box starts at an anchor (the close of the hour's last minute), spans 60 minutes and +-50 bps around that close; the number above it is "
                   f"{CH}'s P(hit); orange boxes were hit. Data: {VENUE['look2']}", height=900)
    return fig, d, med, ndays


def main():
    for nm, ttl in LOOKS:
        save(fig_reliability(nm, ttl), f"reliability_{nm}", 1500, 1150)
    save(fig_skill_by_month(), "brier_skill_by_month", 1500, 1150)
    f, d, med, nd = fig_window(); save(f, "btc_window", 1500, 900)
    print("window", d, med, nd)


if __name__ == "__main__":
    main()

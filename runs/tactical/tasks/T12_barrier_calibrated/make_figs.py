"""T12 figures in the tactical_report style (theme.py): plotly JSON + PNG (headless Edge). Reads resultsA.json / rowsA.npz (Look A).
Figures: reliability before (H3) and after (H3c) calibration, with B1; Brier skill over B1 by month.
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import t12_lib as T12

L = T12.L
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
spec = importlib.util.spec_from_file_location("nt_theme", os.path.join(ROOT, "src", "neural_trade", "visualization", "theme.py"))
TH = importlib.util.module_from_spec(spec); sys.modules["nt_theme"] = TH; spec.loader.exec_module(TH); TH.register()
OS = TH.OTHER_SERIES
RES = json.load(open(os.path.join(HERE, "resultsA.json"))); ROW = np.load(os.path.join(HERE, "rowsA.npz"))
COL = {"B1": OS[5], "H3": OS[0], "H3c": OS[1]}
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
VENUE = "Bitstamp BTC/USD 1-minute history (the cut file; not Binance), scored months 2017-10..2018-12."


def save(fig, name, w=1500, h=900):
    fig.write_json(os.path.join(FIG, name + ".json"))
    html = os.path.join(tempfile.gettempdir(), f"t12_{name}.html")
    fig.write_html(html, include_plotlyjs=True, full_html=True, config=dict(displayModeBar=False))
    png = os.path.join(FIG, name + ".png")
    if os.path.exists(png):
        os.remove(png)
    prof = os.path.join(tempfile.gettempdir(), "edge_prof_t12")
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


def fig_reliability():
    ttl = [f"X = {X} bps, T = {T} min" for T in T12.TS for X in T12.XS]
    fig = make_subplots(rows=3, cols=3, subplot_titles=ttl, vertical_spacing=0.09, horizontal_spacing=0.06)
    for i, T in enumerate(T12.TS):
        for j, X in enumerate(T12.XS):
            r, c = i + 1, j + 1; y = ROW[f"{T}|y|{X}"].astype(float)
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TH.NEUTRAL, width=1), showlegend=False, hoverinfo="skip"), r, c)
            txt = [f"n={len(y)}, base {y.mean():.3f}"]
            for m in ("B1", "H3", "H3c"):
                p = ROW[f"{T}|p|{m}|{X}"].astype(float); rel = L.reliability(p, y)
                xs = [d["p_mean"] for d in rel if d["n"] > 0]; ys = [d["y_mean"] for d in rel if d["n"] > 0]; ns = [d["n"] for d in rel if d["n"] > 0]
                se = [1.96 * np.sqrt(max(a * (1 - a), 1e-6) / n) for a, n in zip(ys, ns)]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name={"B1": "B1", "H3": "H3 (before calibration)", "H3c": "H3c (after calibration)"}[m], legendgroup=m,
                                         showlegend=(i == 0 and j == 0), line=dict(color=COL[m], width=2.4 if m != "B1" else 1.4),
                                         marker=dict(color=COL[m], size=[5 + 4 * np.log10(n) for n in ns], symbol={"B1": "square", "H3": "circle", "H3c": "diamond"}[m]),
                                         error_y=dict(type="data", array=se, color=COL[m], thickness=1, width=2) if m == "H3c" else None,
                                         customdata=ns, hovertemplate=f"{m} p=%{{x:.3f}} freq=%{{y:.3f}} n=%{{customdata}}<extra></extra>"), r, c)
                d = RES["cells"][f"{X}|{T}"][m]; txt.append(f"{m}: ECE {100 * d['ece']:.2f}pp [{100 * d['ece_ci'][0]:.2f},{100 * d['ece_ci'][1]:.2f}], skill {100 * d['skill_vs_B1']:+.1f}%")
            fig.add_annotation(x=0.02, y=0.98, xref=f"x{ax(r, c)} domain", yref=f"y{ax(r, c)} domain", text="<br>".join(txt), showarrow=False, xanchor="left", yanchor="top",
                               font=dict(size=9, color=TH.INK), bgcolor="rgba(18,18,17,0.7)")
            fig.update_xaxes(range=[0, 1], title_text="predicted P(hit)" if i == 2 else None, row=r, col=c)
            fig.update_yaxes(range=[0, 1], title_text="observed frequency" if j == 0 else None, row=r, col=c)
    style(fig, title="T12 reliability before and after the monthly isotonic calibration (Look A)",
          subtitle=f"Ten equal-width bins of predicted probability; marker size grows with the bin's anchor count; bars are 95% binomial errors for H3c assuming independent anchors. "
                   f"H3 = T10's model; H3c = H3 mapped by an isotonic regression fitted on the previous calendar month; B1 = Brownian first passage with trailing volatility. "
                   f"ECE in brackets is the day-block bootstrap 95% CI. Primary cell: X = 50, T = 60 (middle row, middle column). Data: {VENUE}", height=1150, width=1500)
    return fig


def fig_skill_by_month():
    ttl = [f"X = {X} bps, T = {T} min" for T in T12.TS for X in T12.XS]
    fig = make_subplots(rows=3, cols=3, subplot_titles=ttl, vertical_spacing=0.09, horizontal_spacing=0.06)
    for i, T in enumerate(T12.TS):
        for j, X in enumerate(T12.XS):
            r, c = i + 1, j + 1
            fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1), row=r, col=c)
            for m in ("H3", "H3c"):
                mo = RES["cells"][f"{X}|{T}"][m]["monthly"]
                fig.add_trace(go.Scatter(x=list(mo), y=[1000 * (v["brier_b1"] - v["brier"]) for v in mo.values()], mode="lines+markers", name=m, legendgroup=m,
                                         showlegend=(i == 0 and j == 0), line=dict(color=COL[m], width=2.4), marker=dict(color=COL[m], size=5),
                                         hovertemplate=m + " %{x}: %{y:+.1f}e-3<extra></extra>"), r, c)
            fig.update_xaxes(type="category", tickangle=-60, tickfont=dict(size=8), row=r, col=c)
            fig.update_yaxes(title_text="Brier(B1) - Brier(model), x1e-3" if j == 0 else None, row=r, col=c)
    style(fig, title="T12 Brier improvement over B1 by month (Look A)",
          subtitle=f"Brier(B1) - Brier(model) within each month, in units of 1e-3 (positive = better than B1; the ratio is not shown because a month with almost no misses has a B1 Brier near zero and the ratio explodes), H3 and H3c. The first scored month is 2017-10 (2017-09 only calibrates); H3 is refitted at each month start "
                   f"on the 365 days before it, or on everything since 2017-08-17 (15 to 349 days for 2017-09..2018-08). T = 240 panels hold about 180 anchors a month and are noisy. Data: {VENUE}",
          height=1150)
    return fig


def main():
    save(fig_reliability(), "reliability_before_after", 1500, 1150)
    save(fig_skill_by_month(), "brier_skill_by_month", 1500, 1150)


if __name__ == "__main__":
    main()

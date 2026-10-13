"""T09 figures in the tactical_report style (theme.py): plotly JSON + PNG (headless Edge). Horizon colours are NOT used (no axis here is the forecast
horizon); solid lines (nothing here is training). Reads results_minute.pkl / results_forward.pkl / results_hourly.json / sanity.json.
usage: python make_figs.py [name ...]   (default: all)"""
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lib
import guard

ROOT = os.path.abspath(os.path.join(lib.TAC, "..", ".."))
spec = importlib.util.spec_from_file_location("nt_theme", os.path.join(ROOT, "src", "neural_trade", "visualization", "theme.py"))
TH = importlib.util.module_from_spec(spec); sys.modules["nt_theme"] = TH; spec.loader.exec_module(TH); TH.register()
OS = TH.OTHER_SERIES
PCOL = {"2017-19": OS[2], "2020-23": OS[1], "2024-25": OS[0]}             # warm sequence, old to new
ECOL = {"close": OS[0], "nopen": OS[1], "mid": OS[3]}
ENAME = {"close": "close entry (H51)", "nopen": "next-open entry (H59)", "mid": "mid of next bar"}
DIV = [[0, TH.SERIES[6]], [0.5, "#2c2c2a"], [1, TH.SERIES[3]]]             # purple - neutral - amber (the up / down slots, used as signed colours)
SEQ = [[0, "#1f1f1d"], [1, TH.SERIES[3]]]
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
M = pickle.load(open(os.path.join(HERE, "results_minute.pkl"), "rb")); R = M["R"]
FW = pickle.load(open(os.path.join(HERE, "results_forward.pkl"), "rb"))["R"]
HR = json.load(open(os.path.join(HERE, "results_hourly.json")))
YEARS = R["q1"]["close"]["years"]


def save(fig, name, w=1500, h=900):
    fig.write_json(os.path.join(FIG, name + ".json"))
    html = os.path.join(tempfile.gettempdir(), f"t09_{name}.html")
    fig.write_html(html, include_plotlyjs=True, full_html=True, config=dict(displayModeBar=False))
    png = os.path.join(FIG, name + ".png")
    if os.path.exists(png):
        os.remove(png)
    prof = os.path.join(tempfile.gettempdir(), "edge_prof_t09")
    subprocess.run([EDGE, "--headless=new", "--user-data-dir=" + prof, "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    "--hide-scrollbars", f"--screenshot={png}", f"--window-size={w},{h}", "--virtual-time-budget=30000", "file:///" + html.replace("\\", "/")],
                   capture_output=True, text=True)
    print(name, os.path.exists(png), flush=True)


def heat(fig, z, x, y, row, col, zmin=None, zmax=None, scale=DIV, text=None, fmt=None, showscale=False, cbtitle=None, hover=None, textsize=10):
    z = np.asarray(z, float)
    tr = go.Heatmap(z=z, x=x, y=y, colorscale=scale, zmin=zmin, zmax=zmax, xgap=1, ygap=1, showscale=showscale,
                    colorbar=dict(title=dict(text=cbtitle or "", font=dict(size=11)), thickness=12, len=0.5) if showscale else None,
                    hovertemplate=hover or "%{y} | %{x}: %{z:.4f}<extra></extra>")
    if text is not None:
        tr.update(text=np.asarray(text, object), texttemplate="%{text}", textfont=dict(size=textsize, color=TH.INK))
    elif fmt:
        tr.update(texttemplate=fmt, textfont=dict(size=textsize, color=TH.INK))
    fig.add_trace(tr, row, col)
    if len(x) <= 30 and isinstance(x[0], str):
        fig.update_xaxes(type="category", tickmode="array", tickvals=list(x), ticktext=list(x), row=row, col=col)


def style(fig, title, subtitle, height, b=56, width=1500, venue="cut"):
    """theme.apply with a wrapped subtitle, a top margin that clears it, and the legend above the first subplot titles."""
    import textwrap
    subtitle = subtitle + (" Data: Bitstamp BTC/USD 1-minute history (the cut file; not Binance)." if venue == "cut" else " Data: Binance BTCUSDT forward file (a different venue from the Bitstamp history).")
    lines = textwrap.wrap(subtitle, int(width / 6.6)); sub = "<br>".join(lines)
    t = 62 + 17 * len(lines) + 84
    TH.apply(fig, title=title, subtitle=sub, height=height)
    fig.update_layout(margin=dict(t=t, b=b, l=64, r=24), legend=dict(orientation="h", yanchor="bottom", y=1 + 36 / max(height - t - b, 200), xanchor="left", x=0.0))
    return fig


def seam_marks(fig, rows_cols, labels=True):
    """Vertical marks for the data events on monthly axes: Bitstamp tick $0.01 -> $1 (2022-08-10), the file splice (2024-09-13: offset change, about
    4 h missing, candle construction change) and the 1,160-minute gap (2025-03-15)."""
    for (r, c) in rows_cols:
        fig.update_xaxes(type="date", row=r, col=c)
        for x, nm in (("2022-08-10", "tick $0.01 to $1"), ("2024-09-13", "file splice"), ("2025-03-15", "1,160-min gap")):
            fig.add_vline(x=x, line=dict(color=TH.INK_2, width=1, dash="dashdot"), row=r, col=c)
            if labels:
                fig.add_annotation(x=x, y=0.98, xref="x" + str((r - 1) * 2 + c if (r, c) != (1, 1) else "") + "", yref="y" + str((r - 1) * 2 + c if (r, c) != (1, 1) else "") + " domain",
                                   text=nm, showarrow=False, textangle=-90, xanchor="right", yanchor="top", font=dict(size=9, color=TH.MUTED))


def yearcat(fig, row, col):
    fig.update_xaxes(type="category", row=row, col=col)


def month_ticks(months):
    idx = [i for i, m in enumerate(months) if m.endswith("-01")]
    return [months[i] for i in idx], [months[i][:4] for i in idx]


# ------------------------------------------------------------------ 1 coefficient heatmap
def fig_coef():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.09,
                        subplot_titles=("Close entry (H51 label): standardised coefficient of the monthly logistic regression, feature x month",
                                        "Next-open entry (H59 label)"))
    for r, e in enumerate(("close", "nopen"), 1):
        ms = [m["month"] for m in M["Wd"][e]["months"]]; Z = np.array([m["coef"] for m in M["Wd"][e]["months"]]).T
        lim = float(np.nanpercentile(np.abs(Z), 99))
        heat(fig, Z, ms, lib.FEAT_NAMES, r, 1, -lim, lim, showscale=r == 1, cbtitle="coefficient (per 1 sd)",
             hover="%{y} | %{x}: %{z:>+.3f}<extra></extra>")
        tv, tt = month_ticks(ms); fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=1); fig.update_yaxes(autorange="reversed", row=r, col=1)
        for x in ("2022-08-10", "2024-09-13"):
            fig.add_vline(x=x, line=dict(color=TH.INK, width=1, dash="dashdot"), row=r, col=1)
    style(fig, title="Q1: the regression's coefficients month by month",
             subtitle="13 z-scored features, C = 0.1, trained on the 60 days before each month. Purple = negative weight (fade), amber = positive. The weight moves between "
                      "collinear features (position: ema20, bb20, stoch14, cci20 vs the MACD histogram) while the fitted score stays similar. Dash-dot lines: 2022-08-10 (tick $0.01 to $1) and 2024-09-13 (file splice).", height=900)
    return fig


# ------------------------------------------------------------------ 2 importance by year
def fig_importance():
    q = R["q1"]["close"]; qn = R["q1"]["nopen"]; ys = YEARS; names = lib.FEAT_NAMES
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.12, horizontal_spacing=0.07,
                        subplot_titles=("Univariate AUC - 0.5 of the raw feature (negative = reversal)", "Single-feature permutation AUC loss (fitted model)",
                                        "Drop-one AUC loss (refit without the feature)", "Grouped permutation loss, by group",
                                        "AUC of a model using ONLY the group (full model: row 'all 13')", "Next-open entry: single-feature permutation loss"))
    U = [[q["uni"][nm][y]["mean"] - 0.5 for y in ys] for nm in names]
    heat(fig, U, ys, names, 1, 1, -0.1, 0.1, fmt="%{z:>+.3f}", hover="%{y} | %{x}: %{z:>+.4f}<extra></extra>")
    P_ = [[q["perm"][nm][y]["mean"] for y in ys] for nm in names]; heat(fig, P_, ys, names, 1, 2, -0.05, 0.05, fmt="%{z:.3f}")
    Dr = [[q["drop"][nm][y]["mean"] for y in ys] for nm in names]; heat(fig, Dr, ys, names, 1, 3, -0.005, 0.005, fmt="%{z:>+.4f}", textsize=9)
    gs = list(q["gperm"]); G = [[q["gperm"][g][y]["mean"] for y in ys] for g in gs]; heat(fig, G, ys, gs, 2, 1, -0.08, 0.08, fmt="%{z:.3f}")
    ggo = list(q["gonly"]); GO = [[q["gonly"][g][y]["mean"] for y in ys] for g in ggo] + [[q["auc"][y]["mean"] for y in ys]]
    heat(fig, GO, ys, ggo + ["all 13"], 2, 2, 0.38, 0.62, fmt="%{z:.3f}")
    Pn = [[qn["perm"][nm][y]["mean"] for y in ys] for nm in names]; heat(fig, Pn, ys, names, 2, 3, -0.05, 0.05, fmt="%{z:.3f}")
    for r in (1, 2):
        for c in (1, 2, 3):
            fig.update_yaxes(autorange="reversed", row=r, col=c)
    style(fig, title="Q1: which features carry it? (yearly means over months, close entry unless stated)",
             subtitle="Every feature is a version of 'how stretched is the last 20-60 minutes'; the features are collinear, so the drop-one loss is ~0 (the others cover for the dropped one) "
                      "while shuffling a feature or a group breaks the fitted score. Position features carry the most; volatility (atr14) carries nothing.", height=1100, width=1700)
    return fig


# ------------------------------------------------------------------ 3 partial dependence
def fig_pd():
    q2 = R["q2"]["close"]; eras = list(q2); feats = list(q2[eras[0]])
    fig = make_subplots(rows=len(feats), cols=3, vertical_spacing=0.05, horizontal_spacing=0.06,
                        subplot_titles=[f"{f}, {e}" for f in feats for e in eras])
    for i, f in enumerate(feats, 1):
        for j, e in enumerate(eras, 1):
            v = q2[e][f]; b = pd.DataFrame(v["bins"])
            fig.add_trace(go.Scatter(x=b["x_median"], y=b["p_up"], mode="markers", marker=dict(size=7, color=PCOL[e]), name="empirical P(up) (20 quantile bins, 95% band)",
                                     error_y=dict(type="data", symmetric=False, array=b["hi"] - b["p_up"], arrayminus=b["p_up"] - b["lo"], color=TH.rgba(PCOL[e], 0.6), thickness=1),
                                     legendgroup="emp", showlegend=(i == 1 and j == 1), hovertemplate="x %{x:.3f}: P(up) %{y:.3f}<extra></extra>"), i, j)
            fig.add_trace(go.Scatter(x=b["x_median"], y=b["pd_model"], mode="lines", line=dict(color=TH.INK_2, width=1.6), name="model partial dependence (conditional)",
                                     legendgroup="pd", showlegend=(i == 1 and j == 1), hovertemplate="x %{x:.3f}: model %{y:.3f}<extra></extra>"), i, j)
            fig.add_hline(y=v["mean_p_up"], line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=i, col=j)
            fig.add_annotation(text=f"R2 line {v['r2_linear']:.2f}, slope lo/hi {v['slope_lower_half']:+.3f}/{v['slope_upper_half']:+.3f}", x=0.02, y=0.97, xref=f"x{'' if (i - 1) * 3 + j == 1 else (i - 1) * 3 + j} domain",
                               yref=f"y{'' if (i - 1) * 3 + j == 1 else (i - 1) * 3 + j} domain", showarrow=False, xanchor="left", yanchor="top", font=dict(size=10, color=TH.MUTED))
            fig.update_yaxes(range=[0.40, 0.66], row=i, col=j, tickformat=".2f")
    style(fig, title="Q2: the shape. Marginal P(up) against each top feature is close to a straight line, with a fatter edge in the tails",
             subtitle="Dots = share of up moves in 20 quantile bins of the era's pool (95% Wilson band, n_eff = n x 10/15); dashed = era mean P(up). The model line is the CONDITIONAL "
                      "dependence: for macd and cci20 it has the opposite sign to the marginal dots (suppressors in a collinear set).", height=1700)
    return fig


def fig_tails():
    q2 = R["q2"]["close"]; eras = list(q2); feats = list(q2[eras[0]])
    fig = make_subplots(rows=1, cols=len(feats), horizontal_spacing=0.04, subplot_titles=feats)
    for j, f in enumerate(feats, 1):
        for e in eras:
            t = q2[e][f]["tails"]
            fig.add_trace(go.Scatter(x=[x["decile"] + 1 for x in t], y=[x["fade_hit"] for x in t], mode="lines+markers", line=dict(color=PCOL[e], width=2),
                                     name=e, legendgroup=e, showlegend=j == 1, hovertemplate="decile %{x}: fade hit %{y:.3f}<extra></extra>"), 1, j)
        fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=1, col=j)
        fig.update_xaxes(title_text="|feature| decile (10 = most stretched)", row=1, col=j); fig.update_yaxes(range=[0.47, 0.66], row=1, col=j)
    fig.update_yaxes(title_text="share of moves in the 'fade' direction", row=1, col=1)
    style(fig, title="Q2: the edge sits in the tail. Fading the most stretched decile wins 56-61% of moves in 2017-19 and 53-55% in 2024-25; the middle deciles are coin flips",
             subtitle="Hit rate of the rule 'predict the move against the sign of the feature' by decile of |feature| (deadband-masked anchors, close entry). Dashed = 0.5. "
                      "Symmetry: the same fade works above and below (edge of the 3 lowest bins vs the 3 highest in tables.txt).", height=640, width=1900)
    return fig


# ------------------------------------------------------------------ 4 impulse response
def fig_impulse(entry, stat):
    q3 = R["q3"][entry]; d = q3["by_year_auc" if stat == "auc" else "by_year_bps"]; hw = R["q3"]["auc_halfwidth"][entry]
    fig = make_subplots(rows=3, cols=3, vertical_spacing=0.09, horizontal_spacing=0.06, subplot_titles=[f"{y}" for y in YEARS])
    for n, y in enumerate(YEARS):
        r, c = n // 3 + 1, n % 3 + 1; Z = np.array(d[y])
        txt = np.array([[f"{x * 100:.1f}" if stat == "auc" else f"{x:+.1f}" for x in row] for row in Z], object)
        if stat == "auc":
            heat(fig, Z, [f"{h}m" for h in lib.HS], [f"{k}m" for k in lib.KS], r, c, 0.40, 0.60, text=txt, showscale=n == 8, cbtitle="AUC (0.5 = none)", textsize=11,
                 hover="fade last %{y} | horizon %{x}: AUC %{z:.4f}<extra></extra>")
        else:
            heat(fig, Z, [f"{h}m" for h in lib.HS], [f"{k}m" for k in lib.KS], r, c, -6, 6, text=txt, showscale=n == 8, cbtitle="bps per signal", textsize=11,
                 hover="fade last %{y} | horizon %{x}: %{z:>+.2f} bps<extra></extra>")
        fig.update_yaxes(autorange="reversed", row=r, col=c)
        if r == 3:
            fig.update_xaxes(title_text="target horizon", row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text="fade the last k min", row=r, col=c)
    what = "AUC x 100 of 'fade the last k minutes'" if stat == "auc" else "mean signed return of 'fade the last k minutes' (bps per signal, no costs)"
    style(fig, title=f"Q3 impulse response, {ENAME[entry]}: {what}",
             subtitle="Rows: the lag k of the return that is faded; columns: the horizon h of the move predicted. Yearly mean of monthly values; the k = 1 row has a 95% half-width of "
                      f"{np.mean([hw[y][0][3] for y in YEARS]):.3f} AUC per cell on average over years (tables.txt has every cell's width). Amber = fading pays (AUC above 0.5, positive bps), purple = it loses.", height=1150)
    return fig


# ------------------------------------------------------------------ 5 conditions
def fig_conditions(entry):
    df = pd.DataFrame(R["q5_period"][entry]); conds = ["vol", "volume", "hl", "cs", "move1sig", "move1bps", "weekend", "hour"]
    titles = {"vol": "volatility (window sigma) tercile", "volume": "volume tercile", "hl": "1-min high-low range tercile (spread proxy)", "cs": "Corwin-Schultz spread (zero | low | high)",
              "move1sig": "size of the last 1-min move (in window sigmas): quantile bins", "move1bps": "size of the last 1-min move (bps): quantile bins", "weekend": "weekday vs weekend",
              "hour": "hour of day, UTC (file hours converted: minus 4 h in US daylight saving, minus 5 h otherwise, until the 2024-09-13 splice)"}
    fig = make_subplots(rows=4, cols=2, vertical_spacing=0.08, horizontal_spacing=0.07, subplot_titles=[titles[c] for c in conds])
    for n, c in enumerate(conds):
        r, k = n // 2 + 1, n % 2 + 1; d = df[df.cond == c]
        for g in ("2017-19", "2020-23", "2024-25"):
            x = d[d.grp == g]
            if c == "hour":
                fig.add_trace(go.Scatter(x=x["bin"], y=x["auc_hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), r, k)
                fig.add_trace(go.Scatter(x=x["bin"], y=x["auc_lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=TH.rgba(PCOL[g], 0.15), showlegend=False, hoverinfo="skip"), r, k)
                fig.add_trace(go.Scatter(x=x["bin"], y=x["auc"], mode="lines+markers", line=dict(color=PCOL[g], width=2), marker=dict(size=5), name=g, legendgroup=g, showlegend=False,
                                         hovertemplate=f"{g} hour %{{x}}: AUC %{{y:.4f}}<extra></extra>"), r, k)
            else:
                fig.add_trace(go.Bar(x=x["bin"], y=x["auc"], marker_color=PCOL[g], name=g, legendgroup=g, showlegend=n == 0,
                                     error_y=dict(type="data", symmetric=False, array=x["auc_hi"] - x["auc"], arrayminus=x["auc"] - x["auc_lo"], color=TH.INK_2, thickness=1),
                                     hovertemplate=f"{g} %{{x}}: AUC %{{y:.4f}}<extra></extra>"), r, k)
        fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=r, col=k); fig.update_yaxes(range=[0.49, 0.65], row=r, col=k)
    fig.update_layout(barmode="group")
    style(fig, title=f"Q5: when does it work? AUC by condition and period, {ENAME[entry]}",
             subtitle="Bins from the training span's quantiles (no look-ahead); bars = mean of monthly AUCs with a 95% t-interval over months; dashed = 0.5. "
                      "In 2024-25 most bins sit at 0.52-0.53; weekends keep 0.547 and, within wide noise, the largest last moves 0.55-0.57.", height=1500)
    return fig


# ------------------------------------------------------------------ 6 decay decomposition
def fig_decay():
    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.10, horizontal_spacing=0.08,
                        subplot_titles=("Decay vs 2019, close entry: where the AUC went", "Decay vs 2019, next-open entry",
                                        "Monthly AUC of the regression, by entry", "Monthly AUC against the lag-1 return autocorrelation (close entry; colour = period)",
                                        "Market quality, monthly: autocorrelation and variance ratio", "Spread and bounce proxies, monthly (bps)"))
    comps = (("strength", "feature strength (same coefficients, same condition mix)", OS[3]), ("cond", "condition mix (volatility x volume cells)", OS[2]), ("coef", "coefficient adaptation", OS[1]))
    for c, e in enumerate(("close", "nopen"), 1):
        w = R["q6"][e]["waterfall"]; ys = [y for y in YEARS]
        for key, label, col in comps:
            fig.add_trace(go.Bar(x=ys, y=[w[y][key] for y in ys], name=label, marker_color=col, legendgroup=key, showlegend=c == 1,
                                 hovertemplate=f"{label} %{{x}}: %{{y:+.4f}}<extra></extra>"), 1, c)
        fig.add_trace(go.Scatter(x=ys, y=[w[y]["total"] for y in ys], mode="markers", marker=dict(symbol="diamond", size=10, color=TH.INK, line=dict(color=TH.PAPER, width=1)),
                                 name="total change vs 2019", legendgroup="tot", showlegend=c == 1), 1, c)
    fig.update_layout(barmode="relative"); fig.update_yaxes(title_text="AUC change vs 2019", row=1, col=1); fig.update_yaxes(zeroline=True, row=1, col=1)
    yearcat(fig, 1, 1); yearcat(fig, 1, 2)
    for e in ("close", "nopen", "mid"):
        ma = R["monthly_auc"][e]; ms = sorted(ma)
        fig.add_trace(go.Scatter(x=ms, y=[ma[m] for m in ms], mode="lines", line=dict(color=ECOL[e], width=1.6), name=ENAME[e], legendgroup="m" + e), 2, 1)
    fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=2, col=1)
    MQ = R["monthly_mq"]; ma = R["monthly_auc"]["close"]; ms = [m for m in sorted(ma) if m in MQ]
    per = lambda m: "2017-19" if m[:4] <= "2019" else ("2020-23" if m[:4] <= "2023" else "2024-25")
    for g in PCOL:
        sel = [m for m in ms if per(m) == g]
        fig.add_trace(go.Scatter(x=[MQ[m]["ac1"] for m in sel], y=[ma[m] for m in sel], mode="markers", marker=dict(size=7, color=PCOL[g]), name=g, legendgroup="p" + g,
                                 text=sel, hovertemplate="%{text}: ac1 %{x:.3f}, AUC %{y:.3f}<extra></extra>"), 2, 2)
    co = R["q6"]["correlations_close"]["ac1"]
    fig.update_xaxes(title_text="lag-1 autocorrelation of 1-minute returns", row=2, col=2); fig.update_yaxes(title_text="monthly AUC", row=2, col=2)
    fig.add_annotation(text=f"Spearman {co['spearman']:+.2f}; detrended Pearson {co['detrended_pearson']:+.2f} (n = {co['n_months']})", xref="x4 domain", yref="y4 domain", x=0.02, y=0.04,
                       showarrow=False, xanchor="left", font=dict(size=11, color=TH.MUTED))
    allm = sorted(MQ)
    fig.add_trace(go.Scatter(x=allm, y=[MQ[m]["ac1"] for m in allm], mode="lines", line=dict(color=OS[0], width=1.6), name="lag-1 autocorrelation", legendgroup="ac"), 3, 1)
    fig.add_trace(go.Scatter(x=allm, y=[MQ[m]["vr_1_15"] - 1 for m in allm], mode="lines", line=dict(color=OS[3], width=1.6), name="variance ratio VR(1,15) - 1", legendgroup="vr"), 3, 1)
    fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=3, col=1)
    for key, nm, col in (("roll_bps", "Roll spread", OS[0]), ("hl_median_bps", "median 1-min high-low range", OS[1]), ("cs_median_bps", "Corwin-Schultz median", OS[3])):
        fig.add_trace(go.Scatter(x=allm, y=[MQ[m][key] for m in allm], mode="lines", line=dict(color=col, width=1.6), name=nm, legendgroup=key), 3, 2)
    for r in (2, 3):
        for c in (1, 2):
            if not (r == 2 and c == 2):
                tv, tt = month_ticks(sorted(MQ)); fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=c)
    seam_marks(fig, [(2, 1)]); seam_marks(fig, [(3, 1)], labels=False); seam_marks(fig, [(3, 2)], labels=False)
    style(fig, title="Q6: why did it decay? The signal itself got weaker, and the market got less 'bouncy'",
             subtitle="Top: AUC change vs 2019 split into feature strength / condition mix / coefficient adaptation (sum = total, diamonds). Refitting adds <0.01 and the condition mix is minor; "
                      "the remainder is the same coefficients predicting less. Bottom: the decay tracks the fading of the minute-scale negative autocorrelation (bid-ask bounce and thin-book reversal). Dash-dot lines: Bitstamp tick $0.01 to $1 (2022-08-10), file splice (2024-09-13), 1,160-minute gap (2025-03-15); "
                      "the 2022 and 2024 yearly bars contain the tick change and the splice, so a step in those years is confounded with a data change.", height=1500)
    return fig


# ------------------------------------------------------------------ 7 break-even
def fig_breakeven():
    q7 = R["q7"]; ys = YEARS
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, subplot_titles=("Gross bps per trade of the top-10% signals = break-even round-trip cost", "What it costs: spread proxies of the traded bars (bps)"))
    names = {"bps_close": ("close", "close entry (H51)"), "bps_nopen": ("nopen", "next-open entry (H59)"), "bps_mid": ("mid", "mid of the next bar"),
             "bps_close_model_at_nopen": ("close", None)}
    for key, (e, label) in names.items():
        if label is None:
            continue
        t = q7[key]
        fig.add_trace(go.Bar(x=ys, y=[t[y]["mean"] for y in ys], marker_color=ECOL[e], name=label, error_y=dict(type="data", symmetric=False, array=[t[y]["hi"] - t[y]["mean"] for y in ys],
                                                                                                               arrayminus=[t[y]["mean"] - t[y]["lo"] for y in ys], color=TH.INK_2, thickness=1),
                             hovertemplate=f"{label} %{{x}}: %{{y:.2f}} bps<extra></extra>"), 1, 1)
    fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1), row=1, col=1)
    for key, nm, col in (("hl_trades", "high-low range of the signal bars (traded)", OS[1]), ("hl_all", "high-low range, all bars", OS[2]), ("cs_trades", "Corwin-Schultz of the traded bars", OS[3])):
        t = q7[key]; fig.add_trace(go.Scatter(x=ys, y=[t[y]["mean"] for y in ys], mode="lines+markers", line=dict(color=col, width=2), name=nm), 1, 2)
    mq = R["q6"]["market_quality_by_year"]
    fig.add_trace(go.Scatter(x=ys, y=[mq[y]["roll_bps"] for y in ys], mode="lines+markers", line=dict(color=OS[0], width=2), name="Roll spread (bounce)"), 1, 2)
    yearcat(fig, 1, 1); yearcat(fig, 1, 2)
    fig.update_layout(barmode="group"); fig.update_yaxes(title_text="bps", row=1, col=1); fig.update_yaxes(title_text="bps", row=1, col=2)
    style(fig, title="Q7: economic size. The gross edge was about 4 bps at the close and 1.7-3.6 at the next open in 2018-21, and is 0-1 bps since 2022",
             subtitle="Top 10% of each month's signals (threshold from the training predictions), non-overlap not enforced, no costs. Bars: mean over months with a 95% t-interval. "
                      "Break-even cost = the bar height; since 2022 it is inside its own noise band and below any real cost.", height=620)
    return fig


# ------------------------------------------------------------------ 8 a real window
def fig_window(month, day, hour0, hours=4, name="window"):
    s = guard.load_ohlcv("cut"); mods = pickle.load(open("D:/nt/nt_t09_cache/models.pkl", "rb"))
    sc, mdl = mods["close"][month]; thr = next(x["thr"] for x in M["Wd"]["close"]["months"] if x["month"] == month)
    t0 = pd.Timestamp(f"{month}-{day:02d} {hour0:02d}:00"); t1 = t0 + pd.Timedelta(hours=hours)
    i0 = s.index.get_loc(t0); i1 = s.index.get_loc(t1)
    A = s.values.astype(np.float64); lab = lib.import_lab(); idx = np.arange(i0, i1)          # window ends at bar idx (the signal bar)
    W = A[idx[:, None] + np.arange(-lib.L + 1, 1)].astype(np.float32); F = np.nan_to_num(lab.fs_tb7(W))
    p = mdl.predict_proba(sc.transform(F))[:, 1]
    t = s.index[idx]; c = A[idx, 3]
    trades = []; busy = -1
    for k in range(len(idx) - 17):
        if k < busy or abs(p[k] - 0.5) < thr:
            continue
        a = idx[k] + 1; side = 1 if p[k] > 0.5 else -1; ent, ex = A[a, 0], A[a + lib.H0, 3]
        trades.append((a, a + lib.H0, side, ent, ex, side * (ex / ent - 1) * 1e4)); busy = k + lib.H0 + 2
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.04,
                        subplot_titles=("Bitstamp BTC/USD 1-minute close, the regression's top-10% signals as trades (enter next open, hold 15 minutes)", "P(up) - 0.5 (dashed: the top-10% threshold of the month)",
                                        "What it reads: bb20 (position in the 20-minute band) and the 30-minute return in sigmas"))
    fig.add_trace(go.Scatter(x=t, y=c, mode="lines", line=dict(color=TH.INK_2, width=1.6), name="close", hovertemplate="%{x}: %{y:.2f}<extra></extra>"), 1, 1)
    wins = 0; shown = set()
    for n, (a, b, side, ent, ex, bps) in enumerate(trades):
        col = TH.UP_COLOR if side > 0 else TH.DOWN_COLOR; ta, tb = s.index[a], s.index[b]
        fig.add_trace(go.Scatter(x=[ta, tb], y=[ent, ex], mode="lines", line=dict(color=col, width=3), name="long (regression says up)" if side > 0 else "short (regression says down)",
                                 legendgroup="L" if side > 0 else "S", showlegend=side not in shown,
                                 hovertemplate=f"{'long' if side > 0 else 'short'} {bps:+.1f} bps<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=[ta], y=[ent], mode="markers", marker=dict(symbol="triangle-up" if side > 0 else "triangle-down", size=11, color=col, line=dict(color=TH.PAPER, width=1)),
                                 showlegend=False, hovertemplate=f"entry {'long' if side > 0 else 'short'}<extra></extra>"), 1, 1)
        shown.add(side); win = bps > 0; wins += win
        fig.add_trace(go.Scatter(x=[tb], y=[ex], mode="markers", marker=dict(symbol="circle" if win else "x", size=9, color=TH.GOOD if win else TH.CRITICAL, line=dict(color=TH.PAPER, width=1)),
                                 showlegend=False, hovertemplate=f"exit {bps:+.1f} bps<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=p - 0.5, mode="lines", line=dict(color=TH.INK_2, width=1.4), name="P(up) - 0.5"), 2, 1)
    for sgn in (1, -1):
        fig.add_hline(y=sgn * thr, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1), row=2, col=1)
    f_bb = F[:, lib.FEAT_NAMES.index("bb20")]; f_r30 = F[:, lib.FEAT_NAMES.index("ret30")]
    fig.add_trace(go.Scatter(x=t, y=f_bb, mode="lines", line=dict(color=OS[0], width=1.4), name="bb20 (band position; > 0.5 = above the band)"), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=f_r30 / 10, mode="lines", line=dict(color=OS[3], width=1.4), name="30-minute return in sigmas / 10"), 3, 1)
    fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1), row=3, col=1)
    net = np.mean([x[5] for x in trades]) if trades else float("nan")
    mrow = next(r for r in R["q7_months"] if r["month"] == month)
    fig.add_annotation(text="triangle = entry, circle = winning exit, x = losing exit; purple line = long, amber = short", xref="paper", yref="paper", x=0.0, y=-0.07, showarrow=False, xanchor="left", font=dict(size=11, color=TH.MUTED))
    style(fig, title=f"One real window: {month}-{day:02d} {hour0:02d}:00-{(hour0 + hours) % 24:02d}:00 file time ({int(lib.utc_hour(np.array([t0.value]), 'cut')[0]):02d}:00 UTC start), the regression of {month} and its trades",
             subtitle=f"{len(trades)} non-overlapping top-10% trades, {wins} won, mean {net:+.1f} bps gross per trade in this window (an illustration, not an estimate: the whole month's top-10% trades at the next open "
                      f"averaged {mrow['bps_close_model_at_nopen']:+.2f} bps, win rate {mrow['hit_close_model_at_nopen']:.0%}). "
                      "The model fades stretched moves: it sells after a run above the band and buys after a drop below it.", height=900)
    return fig


# ------------------------------------------------------------------ 9 microstructure
def fig_micro():
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=("Regression AUC by entry convention, by year", "Close-entry minus next-open AUC against the Roll spread, monthly",
                                        "AUC by high-low range tercile (spread proxy), close vs next-open entry", "Signal bars are the wide-range bars (median high-low range, bps)"))
    for e in ("close", "nopen", "mid"):
        t = R["q4"]["auc_by_year"][e]; ys = list(t)
        fig.add_trace(go.Scatter(x=ys, y=[t[y]["mean"] for y in ys], mode="lines+markers", line=dict(color=ECOL[e], width=2), name=ENAME[e], legendgroup=e,
                                 error_y=dict(type="data", symmetric=False, array=[t[y]["hi"] - t[y]["mean"] for y in ys], arrayminus=[t[y]["mean"] - t[y]["lo"] for y in ys], color=TH.rgba(ECOL[e], 0.5), thickness=1)), 1, 1)
    t = R["q4"]["close_model_on_nopen_labels"]
    fig.add_trace(go.Scatter(x=list(t), y=[t[y]["mean"] for y in t], mode="lines+markers", line=dict(color=TH.INK_2, width=1.4), marker=dict(symbol="square", size=6), name="close-entry model scored on next-open labels", legendgroup="x"), 1, 1)
    fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=1, col=1)
    MQ = R["monthly_mq"]; ac, an = R["monthly_auc"]["close"], R["monthly_auc"]["nopen"]; ms = [m for m in sorted(ac) if m in an and m in MQ]
    per = lambda m: "2017-19" if m[:4] <= "2019" else ("2020-23" if m[:4] <= "2023" else "2024-25")
    for g in PCOL:
        sel = [m for m in ms if per(m) == g]
        fig.add_trace(go.Scatter(x=[MQ[m]["roll_bps"] for m in sel], y=[ac[m] - an[m] for m in sel], mode="markers", marker=dict(size=7, color=PCOL[g]), name=g, text=sel, legendgroup="p" + g,
                                 hovertemplate="%{text}: Roll %{x:.1f} bps, close-nopen %{y:>+.3f}<extra></extra>"), 1, 2)
    fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1), row=1, col=2)
    fig.update_xaxes(title_text="Roll spread (bps)", row=1, col=2); fig.update_yaxes(title_text="AUC(close) - AUC(next open)", row=1, col=2)
    for e in ("close", "nopen"):
        df = pd.DataFrame(R["q5_period"][e]); d = df[df.cond == "hl"]
        for g in PCOL:
            x = d[d.grp == g]
            fig.add_trace(go.Bar(x=[f"{b} ({g})" for b in x["bin"]], y=x["auc"], marker_color=PCOL[g] if e == "close" else TH.rgba(PCOL[g], 0.45), name=f"{g} {e}", showlegend=False,
                                 error_y=dict(type="data", symmetric=False, array=x["auc_hi"] - x["auc"], arrayminus=x["auc"] - x["auc_lo"], color=TH.INK_2, thickness=1),
                                 hovertemplate=f"{e} {g} %{{x}}: %{{y:.4f}}<extra></extra>", offsetgroup=e), 2, 1)
    fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=2, col=1); fig.update_yaxes(range=[0.49, 0.64], row=2, col=1)
    fig.update_xaxes(tickangle=-60, row=2, col=1)
    yearcat(fig, 1, 1); yearcat(fig, 2, 2)
    q7 = R["q7"]; ys = YEARS
    for key, nm, col in (("hl_trades", "traded (top-10%) signal bars", OS[1]), ("hl_all", "all bars", OS[2])):
        fig.add_trace(go.Scatter(x=ys, y=[q7[key][y]["mean"] for y in ys], mode="lines+markers", line=dict(color=col, width=2), name=nm), 2, 2)
    style(fig, title="Q4: how much is bid-ask bounce? About 0.02-0.04 AUC in 2017-20, nothing from 2021 on",
             subtitle="Entering at the close of the signal bar is flattered by the bounce (the close is as likely a bid as an ask print); entering at the next open or at the mid removes it. "
                      "Light bars in the lower-left panel = next-open entry, solid = close entry; the spread proxy is the signal bar's own high-low range (terciles from the training span).", height=1050)
    fig.update_layout(barmode="group")
    return fig


# ------------------------------------------------------------------ 10 hourly
def fig_hourly():
    q = HR["q1"]; ys = q["years"]; gs = q["groups"]
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=("Hourly rich logistic (H62): AUC by year, 4-hour horizon", "Family permutation AUC loss", "AUC of a model using ONLY the family",
                                        "Impulse AUC (next-open), 2019", "2022", "2025"))
    t = q["auc"]
    fig.add_trace(go.Scatter(x=ys, y=[t[y]["mean"] for y in ys], mode="lines+markers", line=dict(color=OS[1], width=2), name="regression AUC",
                             error_y=dict(type="data", symmetric=False, array=[t[y]["hi"] - t[y]["mean"] for y in ys], arrayminus=[t[y]["mean"] - t[y]["lo"] for y in ys], color=TH.rgba(OS[1], 0.5), thickness=1)), 1, 1)
    sh = HR["q3"]["rule_share_nopen_h4"]
    fig.add_trace(go.Scatter(x=list(sh), y=[sh[y]["best_auc"] for y in sh], mode="lines+markers", line=dict(color=OS[3], width=2), name="best simple fade rule (k chosen in the year)"), 1, 1)
    fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=1, col=1); yearcat(fig, 1, 1)
    heat(fig, [[q["gperm"][g][y]["mean"] for y in ys] for g in gs], ys, gs, 1, 2, -0.1, 0.1, fmt="%{z:.3f}", textsize=9)
    heat(fig, [[q["gonly"][g][y]["mean"] for y in ys] for g in gs], ys, gs, 1, 3, 0.38, 0.62, fmt="%{z:.3f}", textsize=9)
    for c in (2, 3):
        fig.update_yaxes(autorange="reversed", row=1, col=c)
    r3 = HR["q3"]["nopen"]
    for c, y in enumerate(("2019", "2022", "2025"), 1):
        Z = np.array(r3["by_year"][y]["auc"]); txt = np.array([[f"{x * 100:.1f}" for x in row] for row in Z], object)
        heat(fig, Z, [f"{h}h" for h in r3["hs"]], [f"{k}h" for k in r3["ks"]], 2, c, 0.40, 0.60, text=txt, showscale=c == 3, cbtitle="AUC", textsize=11,
             hover="fade last %{y} | horizon %{x}: AUC %{z:.4f}<extra></extra>")
        fig.update_yaxes(autorange="reversed", row=2, col=c); fig.update_xaxes(title_text="target horizon", row=2, col=c)
    fig.update_yaxes(title_text="fade the last k hours", row=2, col=1)
    style(fig, title="Q8: the hourly cousin is also reversal. Rows k = lag faded, columns h = horizon",
             subtitle="Fading the last 1-6 hours predicts the next 1-8 hours (AUC 0.53-0.58 in 2019 and 2022), and the 24-hour fade predicts the next 8-24 hours (0.55-0.60). By 2025 the short fades are at 0.50-0.53 and only the 24-hour fade still holds (0.54-0.56). "
                      "The hourly regression went from 0.58-0.59 (2019-20) to 0.54 (2024-25); its best single fade rule reaches 0.76-1.6 of the regression's edge.", height=980, width=1700)
    return fig


# ------------------------------------------------------------------ 11 forward
def fig_forward():
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.15, horizontal_spacing=0.08,
                        subplot_titles=("Monthly AUC, forward year 2025-11..2026-09", "Permutation AUC loss by feature (close entry, mean over months)", "AUC of a model using ONLY the group",
                                        "Impulse AUC, next-open entry (forward months)", "AUC by high-low range tercile and weekend", "Top-10% gross bps per trade, by month"))
    for e in ("close", "nopen", "mid"):
        d = FW["monthly"][e]
        fig.add_trace(go.Scatter(x=[r["month"] for r in d], y=[r["auc"] for r in d], mode="lines+markers", line=dict(color=ECOL[e], width=2), name=ENAME[e]), 1, 1)
    fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=1, col=1)
    q = FW["q1"]["close"]; names = lib.FEAT_NAMES
    fig.add_trace(go.Bar(x=names, y=[q["perm"][nm]["mean"] for nm in names], marker_color=OS[0], showlegend=False,
                         error_y=dict(type="data", symmetric=False, array=[q["perm"][nm]["hi"] - q["perm"][nm]["mean"] for nm in names], arrayminus=[q["perm"][nm]["mean"] - q["perm"][nm]["lo"] for nm in names], color=TH.INK_2, thickness=1)), 1, 2)
    gs = list(q["gonly"])
    fig.add_trace(go.Bar(x=gs + ["all 13"], y=[q["gonly"][g]["mean"] for g in gs] + [q["auc"]["mean"]], marker_color=OS[1], showlegend=False,
                         error_y=dict(type="data", symmetric=False, array=[q["gonly"][g]["hi"] - q["gonly"][g]["mean"] for g in gs] + [q["auc"]["hi"] - q["auc"]["mean"]],
                                      arrayminus=[q["gonly"][g]["mean"] - q["gonly"][g]["lo"] for g in gs] + [q["auc"]["mean"] - q["auc"]["lo"]], color=TH.INK_2, thickness=1)), 1, 3)
    fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=1, col=3); fig.update_yaxes(range=[0.48, 0.58], row=1, col=3)
    Z = np.array(FW["q3"]["nopen"]["auc"]); txt = np.array([[f"{x * 100:.1f}" for x in row] for row in Z], object)
    heat(fig, Z, [f"{h}m" for h in lib.HS], [f"{k}m" for k in lib.KS], 2, 1, 0.40, 0.60, text=txt, showscale=True, cbtitle="AUC", textsize=11, hover="fade last %{y} | horizon %{x}: AUC %{z:.4f}<extra></extra>")
    fig.update_yaxes(autorange="reversed", row=2, col=1); fig.update_xaxes(title_text="target horizon", row=2, col=1)
    df = pd.DataFrame(FW["q4"]["by_spread_proxy"]["close"]); sel = df[df.cond.isin(["hl", "weekend"])]
    lab = [f"{c}:{b}" for c, b in zip(sel["cond"], sel["bin"])]
    fig.add_trace(go.Bar(x=lab, y=sel["auc"], marker_color=OS[2], showlegend=False, error_y=dict(type="data", symmetric=False, array=sel["auc_hi"] - sel["auc"], arrayminus=sel["auc"] - sel["auc_lo"], color=TH.INK_2, thickness=1)), 2, 2)
    fig.add_hline(y=0.5, line=dict(color=TH.NEUTRAL, width=1, dash="dash"), row=2, col=2); fig.update_yaxes(range=[0.48, 0.60], row=2, col=2)
    tr = FW["q7_months"]
    for key, nm, e in (("bps_close", "close entry", "close"), ("bps_nopen", "next-open entry", "nopen")):
        fig.add_trace(go.Bar(x=[r["month"] for r in tr], y=[r[key] for r in tr], marker_color=ECOL[e], name=nm), 2, 3)
    fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1), row=2, col=3)
    fig.update_layout(barmode="group"); fig.update_xaxes(tickangle=-50, row=1, col=2); yearcat(fig, 1, 1); yearcat(fig, 2, 3)
    style(fig, title="Q9: the present. A reversal signal of AUC 0.535-0.538 remains in 2025-11..2026-09, worth 0.2-0.7 bps gross per top-10% trade",
             subtitle="Binance forward file, descriptive only: 11 scored months, the first trained on 30 days. The signal is the same fade (ema20 and the position features), a little better than 2024 and 2025 (0.531 and 0.521 on the old file), "
                      "and bounce-free (close and next-open entries agree). The top-10% trades earn a gross mean that its 95% interval cannot separate from zero.", height=1000, width=1700, venue="fwd")
    return fig


FIGS = {"fig1_coef_heatmap": lambda: (fig_coef(), 1500, 900), "fig2_importance_by_year": lambda: (fig_importance(), 1700, 1100),
        "fig3_partial_dependence": lambda: (fig_pd(), 1500, 1700), "fig3b_tail_edge": lambda: (fig_tails(), 1900, 700),
        "fig4a_impulse_auc_close": lambda: (fig_impulse("close", "auc"), 1500, 1150), "fig4b_impulse_auc_nopen": lambda: (fig_impulse("nopen", "auc"), 1500, 1150),
        "fig4c_impulse_bps_nopen": lambda: (fig_impulse("nopen", "bps"), 1500, 1150),
        "fig5a_conditions_close": lambda: (fig_conditions("close"), 1500, 1500), "fig5b_conditions_nopen": lambda: (fig_conditions("nopen"), 1500, 1500),
        "fig6_decay_decomposition": lambda: (fig_decay(), 1500, 1500), "fig7_breakeven_cost": lambda: (fig_breakeven(), 1500, 620),
        "fig8a_window_2019": lambda: (fig_window("2019-06", 17, 8), 1500, 900), "fig8b_window_2025": lambda: (fig_window("2025-05", 13, 12), 1500, 900),
        "fig9_microstructure": lambda: (fig_micro(), 1500, 1050), "fig10_hourly": lambda: (fig_hourly(), 1700, 980), "fig11_forward_year": lambda: (fig_forward(), 1700, 1000)}


if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for n in want:
        fig, w, h = FIGS[n](); save(fig, n, w, h)

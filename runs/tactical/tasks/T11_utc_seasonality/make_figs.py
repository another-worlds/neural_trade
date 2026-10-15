"""T11 figures in the tactical_report style (visualization/theme.py): plotly JSON + PNG (headless Edge). Descriptive only; nothing is chosen from them.
fig1_seasonality: volatility seasonality of the 1-minute bars by UTC hour, New York hour, minute of hour, weekday / weekend, file-time hour versus UTC,
and the 96 UTC 15-minute slots, before / after 2021-11 (cut file, file time converted by timefix.py; seam bars excluded) and on the Binance file.
fig2_variants: the monthly walk-forward of V0..V3 (skill vs ar, difference over V0, secondary Spearman and QLIKE differences) for the selection span,
the Binance months and 2017-18. usage: python make_figs.py [seasonality|variants]  (default both; seasonality.json is cached)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t11_lib as L  # noqa: E402
T = L.T; TF = L.TF
ROOT = os.path.abspath(os.path.join(L.TAC, "..", ".."))
spec = importlib.util.spec_from_file_location("nt_theme", os.path.join(ROOT, "src", "neural_trade", "visualization", "theme.py"))
TH = importlib.util.module_from_spec(spec); sys.modules["nt_theme"] = TH; spec.loader.exec_module(TH); TH.register()
OS = TH.OTHER_SERIES; SER = TH.SERIES
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SPLIT = pd.Timestamp("2021-11-01")
VCOL = {"V0": TH.NEUTRAL, "V1": SER[0], "V2": SER[3], "V3": SER[6]}          # categorical colours for the variants (not horizons)
VNAME = {"V0": "V0 file-time hour (T01)", "V1": "V1 UTC hour", "V2": "V2 + New York basis", "V3": "V3 + 96 UTC slots"}


def save(fig, name, w=1700, h=1000):
    fig.write_json(os.path.join(FIG, name + ".json"))
    html = os.path.join(tempfile.gettempdir(), f"t11_{name}.html")
    fig.write_html(html, include_plotlyjs=True, full_html=True, config=dict(displayModeBar=False))
    png = os.path.join(FIG, name + ".png")
    if os.path.exists(png):
        os.remove(png)
    prof = os.path.join(tempfile.gettempdir(), "edge_prof_t11")
    subprocess.run([EDGE, "--headless=new", "--user-data-dir=" + prof, "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars",
                    f"--screenshot={png}", f"--window-size={w},{h}", "--virtual-time-budget=30000", "file:///" + html.replace("\\", "/")], capture_output=True, text=True)
    print(name, os.path.exists(png), flush=True)


def style(fig, title, subtitle, height, width=1700, b=56):
    lines = textwrap.wrap(subtitle, int(width / 6.6)); sub = "<br>".join(lines); t = 62 + 17 * len(lines) + 84
    TH.apply(fig, title=title, subtitle=sub, height=height)
    fig.update_layout(margin=dict(t=t, b=b, l=64, r=24), legend=dict(orientation="h", yanchor="bottom", y=1 + 36 / max(height - t - b, 200), xanchor="left", x=0.0))
    return fig


# ---------------------------------------------------------------- seasonality data
def _bin_stats(r, key, day, nb):
    """mean |r| per bin, relative to the mean over the sample, with a day-block standard error (days are the independent units)."""
    g = pd.DataFrame({"r": r, "k": key, "d": day}).groupby(["d", "k"])["r"].mean().unstack().reindex(columns=range(nb))
    mean = g.mean().values; se = (g.std() / np.sqrt(g.count())).values; sc = float(np.nanmean(mean))
    return {"rel": (mean / sc).tolist(), "se": (se / sc).tolist(), "mean_bps": mean.tolist(), "n_days": int(len(g))}


def seasonality():
    path = os.path.join(HERE, "seasonality.json")
    if os.path.exists(path):
        return json.load(open(path))
    cut = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2017-01-01"), T.CUTOFF); fw = L.load_freeze_bars(T.FWD_CSV)
    out = {}
    for name, bars, src in (("cut", cut, "bitstamp"), ("binance", fw, "binance")):
        c = bars["close"].values; r = np.abs(np.diff(np.log(c), prepend=np.nan)) * 1e4
        ts = bars.index; ok = np.isfinite(r) & ~TF.seam_mask(ts, L.SEAM_PAD, src) & np.concatenate([[False], np.diff(ts.values) == np.timedelta64(1, "m")])
        utc = TF.to_utc(ts, src); ny = TF.ny_time(utc); wk = utc.dayofweek.values >= 5
        sets = {"cut": {"before 2021-11": np.asarray(utc < SPLIT), "after 2021-11": np.asarray(utc >= SPLIT)}, "binance": {"Binance 2025-10..2026-09": np.ones(len(ts), bool)}}[name]
        for lab, m in sets.items():
            k = ok & m; d = {}
            u = utc[k]; day = u.normalize().values
            d["utc_hour"] = _bin_stats(r[k], u.hour.values, day, 24); d["ny_hour"] = _bin_stats(r[k], ny[k].hour.values, day, 24)
            d["minute"] = _bin_stats(r[k], u.minute.values, day, 60); d["slot96"] = _bin_stats(r[k], (u.hour.values * 60 + u.minute.values) // 15, day, 96)
            d["ny_hour_weekday"] = _bin_stats(r[k & ~wk], ny[k & ~wk].hour.values, utc[k & ~wk].normalize().values, 24)
            d["ny_hour_weekend"] = _bin_stats(r[k & wk], ny[k & wk].hour.values, utc[k & wk].normalize().values, 24)
            d["n_bars"] = int(k.sum()); d["mean_abs_bps"] = float(r[k].mean()); out[lab] = d
        if name == "cut":                                                       # file-time hour, only where the file is still shifted
            m = ok & np.asarray((utc >= SPLIT) & (ts < TF.SPLICE)); fh = ts[m].hour.values
            out["file_time_2021-11..2024-09"] = {"file_hour": _bin_stats(r[m], fh, utc[m].normalize().values, 24), "utc_hour": _bin_stats(r[m], utc[m].hour.values, utc[m].normalize().values, 24)}
    json.dump(out, open(path, "w"))
    return out


def fig_seasonality():
    S = seasonality(); B0, A0, BN = S["before 2021-11"], S["after 2021-11"], S["Binance 2025-10..2026-09"]; F = S["file_time_2021-11..2024-09"]
    series = [("before 2021-11", B0, OS[1], "solid"), ("after 2021-11", A0, SER[0], "solid"), ("Binance 2025-10..2026-09", BN, SER[2], "dash")]
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.14, horizontal_spacing=0.06,
                        subplot_titles=("UTC hour", "New York hour (DST-aware)", "minute of the hour (UTC)", "New York hour, weekday vs weekend (after 2021-11 and Binance)",
                                        "file-time hour vs UTC hour (cut file 2021-11..2024-09, still shifted)", "96 UTC 15-minute slots"))

    def add(row, col, key, x, name, colr, dash, d, show=True, width=2):
        y = np.array(d[key]["rel"]); se = np.array(d[key]["se"])
        fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=np.r_[y + 1.96 * se, (y - 1.96 * se)[::-1]], fill="toself", fillcolor=TH.rgba(colr, 0.18), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False, legendgroup=name), row, col)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=colr, dash=dash, width=width), legendgroup=name, showlegend=show,
                                 hovertemplate=f"{name}<br>%{{x}}: %{{y:.2f}}x the average bar<extra></extra>"), row, col)
    xs = {"utc_hour": list(range(24)), "ny_hour": list(range(24)), "minute": list(range(60)), "slot96": list(range(96))}
    for i, (nm, d, colr, dash) in enumerate(series):
        add(1, 1, "utc_hour", xs["utc_hour"], nm, colr, dash, d, show=True)
        for (rw, cl, key) in ((1, 2, "ny_hour"), (1, 3, "minute"), (2, 3, "slot96")):
            add(rw, cl, key, xs[key] if key != "slot96" else [x for x in xs[key]], nm, colr, dash, d, show=False)
    for nm, d, colr, dash in ((" weekday", A0, SER[0], "solid"), (" weekend", A0, SER[3], "solid")):
        add(2, 1, "ny_hour_weekday" if "day" in nm else "ny_hour_weekend", xs["ny_hour"], "after 2021-11," + nm, colr, dash, d)
    for nm, d, colr, dash in (("Binance," + " weekday", BN, SER[0], "dash"), ("Binance," + " weekend", BN, SER[3], "dash")):
        add(2, 1, "ny_hour_weekday" if "day" in nm else "ny_hour_weekend", xs["ny_hour"], nm, colr, dash, d, width=1.5)
    add(2, 2, "file_hour", xs["utc_hour"], "file-time hour", OS[2], "solid", F); add(2, 2, "utc_hour", xs["utc_hour"], "UTC hour", SER[0], "solid", F)
    for (rw, cl) in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig.update_xaxes(dtick=3, row=rw, col=cl)
    fig.update_xaxes(dtick=5, row=1, col=3); fig.update_xaxes(dtick=8, row=2, col=3, tickvals=list(range(0, 96, 8)), ticktext=[f"{h // 4:02d}:{15 * (h % 4):02d}" for h in range(0, 96, 8)])
    for rw, cl in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)):
        fig.add_hline(y=1.0, line=dict(color=TH.NEUTRAL, width=1, dash="dot"), row=rw, col=cl); fig.update_yaxes(title_text="mean |1-minute return| / average", row=rw, col=cl)
    n = lambda d: f"{d['n_bars'] / 1e6:.2f} M bars"
    style(fig, title="T11: when is Bitcoin volatile? The clock matters, and file time is the wrong clock",
          subtitle=f"Mean absolute 1-minute log return per clock bin, as a multiple of the series' own average (bands: 1.96 standard errors across days). Cut file (Bitstamp) with file time "
                   f"converted to UTC by timefix.py, seam bars excluded ({n(B0)} before 2021-11, {n(A0)} after); Binance forward file in UTC ({n(BN)}). Weekend = Saturday / Sunday UTC. "
                   "The bottom-middle panel shows the same data under the two clocks: in file time the US-session peak sits 4-5 hours late and is smeared by the daylight-saving switch.", height=1000)
    return fig


def fig_variants():
    sel = json.load(open(os.path.join(HERE, "selection_results.json")))["rows"]; rep = json.load(open(os.path.join(HERE, "results.json")))
    sets = [("selection 2019-01..2024-06 (66 months, cut file)", sel), ("Binance 2025-11..2026-09 (report only; monthly refit on the forward file)", rep["binance_2025_26"]["rows"]),
            ("cut file 2017-09..2018-12 (report only)", rep["A_2017_18"]["rows"])]
    fig = make_subplots(rows=4, cols=3, vertical_spacing=0.07, horizontal_spacing=0.05, column_widths=[0.5, 0.25, 0.25],
                        row_titles=["target A: skill vs ar", "A: skill minus V0", "|r_15|: Spearman minus V0", "|r_15|: QLIKE minus V0 (lower is better)"],
                        subplot_titles=[s[0] for s in sets] + [""] * 9)
    for j, (nm, rows) in enumerate(sets, 1):
        ms = [r["month"] for r in rows]
        for v in L.VARIANTS:
            kw = dict(mode="lines+markers" if len(rows) < 20 else "lines", name=VNAME[v], legendgroup=v, showlegend=(j == 1), line=dict(color=VCOL[v], width=2 if v != "V0" else 1.5, dash="dot" if v == "V0" else "solid"),
                      marker=dict(size=5))
            fig.add_trace(go.Scatter(x=ms, y=[r["skill_ar"][v] for r in rows], hovertemplate="%{x}: %{y:.4f}<extra>" + v + "</extra>", **kw), 1, j)
            if v == "V0":
                continue
            kw["showlegend"] = False
            fig.add_trace(go.Scatter(x=ms, y=[r["skill_ar"][v] - r["skill_ar"]["V0"] for r in rows], hovertemplate="%{x}: %{y:>+.4f}<extra>" + v + "</extra>", **kw), 2, j)
            fig.add_trace(go.Scatter(x=ms, y=[r["sec"][v]["spearman"] - r["sec"]["V0"]["spearman"] for r in rows], hovertemplate="%{x}: %{y:>+.4f}<extra>" + v + "</extra>", **kw), 3, j)
            fig.add_trace(go.Scatter(x=ms, y=[r["sec"][v]["qlike"] - r["sec"]["V0"]["qlike"] for r in rows], hovertemplate="%{x}: %{y:>+.5f}<extra>" + v + "</extra>", **kw), 4, j)
        for rw in (2, 3, 4):
            fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1, dash="dot"), row=rw, col=j)
        fig.add_hline(y=0.01, line=dict(color=TH.INK_2, width=1, dash="dash"), row=2, col=j)
        fig.add_hline(y=0.005, line=dict(color=TH.INK_2, width=1, dash="dash"), row=3, col=j)
        step = 12 if len(rows) > 30 else 3
        for rw in range(1, 5):
            fig.update_xaxes(type="category", tickvals=ms[::step], ticktext=ms[::step], tickangle=-45, row=rw, col=j)
    style(fig, title="T11: the four clock variants month by month (walk-forward, 365 training days)",
          subtitle="V0 = T01's S3 with the hour taken from FILE time; V1 = the hour in UTC; V2 = S3 + a 14-term New York basis; V3 = S3 + 96 UTC 15-minute slots. Left: the selection span, on which V3 won "
                   "(mean skill gain over V0 +0.0199, V2 +0.0098, V1 +0.0004); the dashed line is the +0.01 minimum effect of the prospective test (+0.005 for the Spearman rows). "
                   "Middle and right are report only: Binance months were seen by H83, 2017-18 is partly seen (T04-A) and has little training history.", height=1250, width=1800)
    return fig


if __name__ == "__main__":
    which = sys.argv[1:] or ["seasonality", "variants"]
    if "seasonality" in which:
        save(fig_seasonality(), "fig1_seasonality", 1700, 1000)
    if "variants" in which:
        save(fig_variants(), "fig2_variants", 1800, 1250)

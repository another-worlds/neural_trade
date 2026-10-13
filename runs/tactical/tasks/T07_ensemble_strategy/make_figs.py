"""T07 figures in the style of runs/tactical/report/tactical_report.html (theme.py): written as plotly JSON (for a later report / deck section)
and as PNG (headless Edge, like report/shoot.py). Solid lines; no horizon colours for strategies (theme.OTHER_SERIES slots).
Reads D:/nt/nt_t07_cache/{sel,look,fwd}_hourly.npz and results.json. usage: python make_figs.py"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import t07_lib as T

ROOT = os.path.abspath(os.path.join(T.TAC, "..", ".."))
spec = importlib.util.spec_from_file_location("nt_theme", os.path.join(ROOT, "src", "neural_trade", "visualization", "theme.py"))
TH = importlib.util.module_from_spec(spec); sys.modules["nt_theme"] = TH; spec.loader.exec_module(TH); TH.register()
S = TH.OTHER_SERIES
MCOL = {"M1": S[0], "M2": S[1], "M3": S[2], "M4": S[3], "M5": S[4], "M6": S[6]}
CHOSEN_COL, BH_COL = S[0], TH.INK_2
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def thin(fig, cap=9000):
    """Keep the JSON small: a trace with more than `cap` points is decimated (every k-th point, the last point kept)."""
    for t in fig.data:
        n = len(t.x) if getattr(t, "x", None) is not None and t.type == "scatter" else 0
        if n > cap:
            k = int(np.ceil(n / cap)); idx = np.unique(np.concatenate([np.arange(0, n, k), [n - 1]]))
            t.x = np.asarray(t.x)[idx]; t.y = np.asarray(t.y)[idx]
    return fig


def save(fig, name, w=1400, h=900):
    thin(fig)
    fig.write_json(os.path.join(FIG, name + ".json"))
    html = os.path.join(tempfile.gettempdir(), f"t07_{name}.html")
    fig.write_html(html, include_plotlyjs=True, full_html=True, config=dict(displayModeBar=False))
    png = os.path.join(FIG, name + ".png")
    if os.path.exists(png):
        os.remove(png)
    prof = os.path.join(tempfile.gettempdir(), "edge_prof_t07")
    subprocess.run([EDGE, "--headless=new", "--user-data-dir=" + prof, "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    "--hide-scrollbars", f"--screenshot={png}", f"--window-size={w},{h}", "--virtual-time-budget=30000", "file:///" + html.replace("\\", "/")],
                   capture_output=True, text=True)
    print(name, os.path.exists(png), flush=True)


def load(tag):
    z = np.load(f"{T.CACHE}/{tag}_hourly.npz")
    return {k: z[k] for k in z.files}


def null_paths(w, r, cost, draws=300, seed=0):
    """Equity paths of the random-side null (signs of runs flipped; exposure and cost of the actual path). [draws, nh+1]"""
    from scipy import sparse
    s, rid, _ = T.runs_of(w); nh = len(w) // 60; nr = int(rid.max())
    cost_h = cost * 0.5e-4 * T.hsum(np.abs(np.diff(w, prepend=0.0)))
    df = pd.DataFrame({"h": np.arange(len(w)) // 60, "run": rid, "g": w * r}).groupby(["h", "run"])["g"].sum().reset_index()
    sh = sparse.csr_matrix((np.ones(len(df)), (df["h"].values, np.arange(len(df)))), shape=(nh, len(df)))
    rng = np.random.default_rng(seed); fl = np.where(rng.random((draws, nr + 1)) < 0.5, 1.0, -1.0); fl[:, 0] = 1.0
    gross = sh @ (df["g"].values[:, None] * fl[:, df["run"].values].T)
    return np.vstack([np.ones((1, draws)), np.cumprod(1 + gross - cost_h[:, None], axis=0)]).T


def dd_of(eq):
    return 1.0 - eq / np.maximum.accumulate(eq)


def period_slice(d, tag, start, stop):
    ht = d["hour_t"].astype("datetime64[m]")
    i0 = int(np.searchsorted(ht, np.datetime64(start))); i1 = int(np.searchsorted(ht, np.datetime64(stop)))
    return slice(i0, i1), pd.DatetimeIndex(ht[i0:i1])


def fig_main(tag, chosen, start, stop, title):
    d = load(tag); sl, ht = period_slice(d, tag, start, stop)
    w = d[f"c_{chosen}"].astype(np.float64)[sl.start * 60:sl.stop * 60]; r = d["r"][sl.start * 60:sl.stop * 60]
    rh = T.net(*T.book(w, r)[:2], 2.0)
    eq = np.concatenate([[1.0], np.cumprod(1 + rh)]); bh = np.concatenate([[1.0], np.cumprod(1 + T.hsum(r))])
    nul = null_paths(w, r, 2.0); lo, med, hi = np.quantile(nul, [0.05, 0.5, 0.95], axis=0)
    x = np.concatenate([[ht[0] - pd.Timedelta(hours=1)], ht])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                        subplot_titles=("Equity (start 1.0, net of 2 bps round trip)", "Drawdown", "Exposure (hourly mean position, 1.0 = long 1x)"))
    fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=lo, fill="tonexty", fillcolor=TH.rgba(TH.NEUTRAL, 0.22), line=dict(width=0), name="random-side null, 5-95% (300 draws)"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=med, line=dict(color=TH.MUTED, width=1), name="null median"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=bh, line=dict(color=BH_COL, width=2), name="buy-and-hold (no cost)"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=eq, line=dict(color=CHOSEN_COL, width=2.5), name=f"{chosen} (chosen)"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=-dd_of(bh), line=dict(color=BH_COL, width=1.5), name="buy-and-hold", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=-dd_of(eq), line=dict(color=CHOSEN_COL, width=2), name=f"{chosen}", showlegend=False, fill="tozeroy",
                             fillcolor=TH.rgba(CHOSEN_COL, 0.2)), 2, 1)
    fig.add_trace(go.Scatter(x=ht, y=T.hmean(w), line=dict(color=CHOSEN_COL, width=1), name="position", showlegend=False, fill="tozeroy",
                             fillcolor=TH.rgba(CHOSEN_COL, 0.2)), 3, 1)
    fig.update_yaxes(title_text="equity", row=1, col=1, type="log" if max(eq.max(), bh.max()) / max(min(eq.min(), bh.min()), 1e-9) > 4 else "linear")
    fig.update_yaxes(title_text="drawdown", tickformat=".0%", row=2, col=1); fig.update_yaxes(title_text="position", row=3, col=1, range=[-1.05, 1.05])
    st = T.equity_stats(rh); stb = T.equity_stats(T.hsum(r))
    TH.apply(fig, title=title, subtitle=(f"{chosen}: return {st['total_return']:+.1%}, max DD {st['max_dd']:.1%}, Calmar {st['calmar']:.2f}  |  buy-and-hold: "
                                         f"{stb['total_return']:+.1%}, max DD {stb['max_dd']:.1%}, Calmar {stb['calmar']:.2f}"), height=900)
    fig.update_layout(margin=dict(t=125), legend=dict(y=1.09))
    return fig


def fig_members(tag, chosen, start, stop, title):
    d = load(tag); sl, ht = period_slice(d, tag, start, stop)
    r = d["r"][sl.start * 60:sl.stop * 60]
    x = np.concatenate([[ht[0] - pd.Timedelta(hours=1)], ht])
    fig = go.Figure()
    for i, m in enumerate(T.MEMBERS):
        rh = T.net(d["g"][i, sl], d["u"][i, sl], 2.0)
        eq = np.concatenate([[1.0], np.cumprod(1 + rh)]); st = T.equity_stats(rh)
        fig.add_trace(go.Scatter(x=x, y=eq, line=dict(color=MCOL[m], width=1.6), name=f"{m} ({st['total_return']:+.0%}, DD {st['max_dd']:.0%})"))
    w = d[f"c_{chosen}"].astype(np.float64)[sl.start * 60:sl.stop * 60]
    rh = T.net(*T.book(w, r)[:2], 2.0); fig.add_trace(go.Scatter(x=x, y=np.concatenate([[1.0], np.cumprod(1 + rh)]), line=dict(color=TH.INK, width=3), name=f"{chosen} (chosen)"))
    fig.add_trace(go.Scatter(x=x, y=np.concatenate([[1.0], np.cumprod(1 + T.hsum(r))]), line=dict(color=BH_COL, width=2), name="buy-and-hold (no cost)"))
    fig.update_yaxes(title_text="equity at 2 bps (own book, 1x)", type="log"); TH.apply(fig, title=title, subtitle="each member alone, 1x notional", height=620)
    return fig


def fig_corr(tag, start, stop, title):
    d = load(tag); sl, ht = period_slice(d, tag, start, stop)
    rn = T.net(d["g"][:, sl], d["u"][:, sl], 2.0); cm = np.corrcoef(rn)
    cs = [[0.0, TH.DOWN_COLOR], [0.5, TH.SURFACE], [1.0, TH.UP_COLOR]]
    fig = go.Figure(go.Heatmap(z=cm, x=list(T.MEMBERS), y=list(T.MEMBERS), zmin=-1, zmax=1, colorscale=cs, text=np.round(cm, 2), texttemplate="%{text}",
                               textfont=dict(color=TH.INK, size=13), colorbar=dict(title="r"), hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); TH.apply(fig, title=title, subtitle="Pearson r of the members' hourly own-book net returns (2 bps); yellow negative, violet positive", height=560)
    return fig


def fig_monthly(tag, chosen, start, stop, title):
    d = load(tag); sl, ht = period_slice(d, tag, start, stop)
    w = d[f"c_{chosen}"].astype(np.float64)[sl.start * 60:sl.stop * 60]; r = d["r"][sl.start * 60:sl.stop * 60]
    ms = T.monthly_table(T.net(*T.book(w, r)[:2], 2.0), ht); mb = T.monthly_table(T.hsum(r), ht)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(ms.index), y=ms.values, name=f"{chosen} (net 2 bps)", marker_color=CHOSEN_COL, text=[f"{v:+.1%}" for v in ms.values], textposition="outside"))
    fig.add_trace(go.Bar(x=list(mb.index), y=mb.values, name="buy-and-hold", marker_color=BH_COL, text=[f"{v:+.1%}" for v in mb.values], textposition="outside"))
    fig.add_hline(y=0, line=dict(color=TH.NEUTRAL, width=1))
    fig.update_yaxes(tickformat=".0%", title_text="monthly return"); fig.update_layout(barmode="group")
    TH.apply(fig, title=title, subtitle=f"{chosen}: mean {ms.mean():+.2%} per month, {int((ms.values > 0).sum())}/{len(ms)} months positive", height=520)
    return fig


def main():
    res = json.load(open(f"{HERE}/results.json")); ch = res["chosen"]
    jobs = [("fwd", "2025-11-01", "2026-10-01", "forward 2025-11 .. 2026-09"), ("look", "2024-07-01", "2025-07-21", "Look 1 2024-07 .. 2025-07-20"),
            ("sel", "2019-01-01", "2024-07-01", "selection 2019-01 .. 2024-06")]
    for tag, a, b, lab in jobs:
        src = "look" if tag == "sel" else tag                                     # the look file holds the whole history from 2019
        t = lambda k: f"T07 {k}: {lab}"
        save(fig_main(src, ch, a, b, t(f"{ch} against buy-and-hold and the null")), f"equity_{tag}", h=900)
        save(fig_members(src, ch, a, b, t("members and the chosen combination")), f"members_{tag}", h=620)
        save(fig_corr(src, a, b, t("member correlation")), f"corr_{tag}", h=560)
        save(fig_monthly(src, ch, a, b, t("monthly returns")), f"monthly_{tag}", h=520)


if __name__ == "__main__":
    main()

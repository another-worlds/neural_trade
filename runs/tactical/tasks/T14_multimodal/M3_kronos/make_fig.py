"""M3 figure (mm_env, CPU): one real Binance hour (the 2026-03 ok anchor whose realised next-hour log RV is the median of that month),
its 4 Kronos paths over the real 5-minute path, the +-50 bps band, kr_phit50 against the H3 P(hit) (H3 fitted as in evaluate.run_B for 2026-03).
Colours from src/neural_trade/visualization/theme.py (loaded by path); PNG + plotly figure JSON."""
import os, sys, json, glob, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import anchors as AN, view_cache as VC, evaluate as EV
from _libs import ROOT
import make_view as MV
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("nt_theme", os.path.join(ROOT, "src", "neural_trade", "visualization", "theme.py"))
th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
TAG = sys.argv[1] if len(sys.argv) > 1 else "n4"; NAME = sys.argv[2] if len(sys.argv) > 2 else "M3_kronos"
src = "fwd"; mo = pd.Period("2026-03", "M")
base = AN.load_base(src); ta = base.ta; st = AN.get_store(src)
view = VC.read_view(NAME, src); rv = AN.rv5_series(src)
cand = ta[(ta >= mo.start_time) & (ta < (mo + 1).start_time) & base.ok & ta.isin(view.index)]
true_lrv = rv["logrv_1h"].reindex(cand + pd.Timedelta(minutes=55)).to_numpy()
order = np.argsort(true_lrv); pick = cand[order[len(order) // 2]]                 # the median realised-vol hour (rule fixed before looking)
# H3 probability for that anchor
R = EV.run_B(base, base.ok & np.isfinite(VC.align_view(view, ta)[0]).all(1), [mo], pd.Timestamp("2025-10-02"), {"H3": None})
te = np.where(EV.month_mask(ta, base.ok & np.isfinite(VC.align_view(view, ta)[0]).all(1), mo))[0]
p_h3 = float(R["p"]["H3"][list(ta[te]).index(pick)]); y_real = int(R["y"][list(ta[te]).index(pick)])
# paths
for f in sorted(glob.glob(os.path.join(AN.CACHE_DIR, "m3", f"paths_{TAG}_{src}", "chunk_*.npz"))):
    z = np.load(f); t = pd.DatetimeIndex(z["ta_ns"].astype("datetime64[ns]"))
    if pick in t:
        P = z["paths"][list(t).index(pick)]; break
ctx = st.bars_before(pick, 12, "5min"); m5 = st.m5(); real = m5.loc[pick:pick + pd.Timedelta(minutes=55)]
c0 = float(ctx["close"].iloc[-1]); row = view.loc[pick]
fut = pd.date_range(pick, periods=12, freq="5min")
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=th.PAPER, gridspec_kw={"width_ratios": [3, 1]})
a = ax[0]; a.set_facecolor(th.SURFACE); x_ctx = np.arange(-12, 0); x_f = np.arange(0, 12)
def bps(v): return (v / c0 - 1) * 1e4
for k in range(P.shape[0]):
    c = th.OTHER_SERIES[k % len(th.OTHER_SERIES)]
    hi = np.maximum.reduce([P[k, :, 0], P[k, :, 1], P[k, :, 2], P[k, :, 3]]); lo = np.minimum.reduce([P[k, :, 0], P[k, :, 1], P[k, :, 2], P[k, :, 3]])
    a.vlines(x_f + 0.12 * (k - 1.5), bps(lo), bps(hi), color=c, lw=1.2, alpha=0.7)
    a.plot(x_f + 0.12 * (k - 1.5), bps(P[k, :, 3]), "-o", ms=3, lw=1.3, color=c, label=f"Kronos path {k + 1} (close; whiskers high/low)")
a.vlines(x_ctx, bps(ctx["low"]), bps(ctx["high"]), color=th.INK_2, lw=1); a.plot(x_ctx, bps(ctx["close"]), "-o", ms=3, color=th.INK, lw=1.5, label="real, before the anchor")
a.vlines(x_f[:len(real)], bps(real["low"]), bps(real["high"]), color=th.SERIES[0], lw=2.2); a.plot(x_f[:len(real)], bps(real["close"]), "-s", ms=4, color=th.SERIES[0], lw=2.2, label="real hour (close; whiskers high/low)")
for s in (-50, 50): a.axhline(s, color=th.CRITICAL, ls="--", lw=1)
a.axhline(0, color=th.NEUTRAL, lw=0.8); a.axvline(-0.5, color=th.MUTED, lw=0.8, ls=":")
a.set_xlabel("5-minute bars from the anchor (bar 0 opens at the anchor)", color=th.INK_2); a.set_ylabel("bps from the last close before the anchor", color=th.INK_2)
a.set_title(f"Binance {pick:%Y-%m-%d %H:%M} UTC (median realised-vol hour of 2026-03, ln RV {true_lrv[order[len(order)//2]]:.2f}); dashed = +-50 bps", color=th.INK, fontsize=10)
a.tick_params(colors=th.MUTED); a.grid(color=th.GRID, lw=0.5); a.legend(fontsize=7, facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2, loc="upper left")
for sp in a.spines.values(): sp.set_color(th.AXIS)
real_hit = int((np.maximum(np.abs(bps(real["high"])), np.abs(bps(real["low"]))).max() >= 50))
b = ax[1]; b.set_facecolor(th.SURFACE); vals = [row["kr_phit50"], p_h3, real_hit]; labs = ["Kronos kr_phit50\n(fraction of 4 paths)", "H3 P(hit)\n(HAR first passage + logit)", "realised\nhit (0/1)"]
bars = b.bar(range(3), vals, color=[th.OTHER_SERIES[0], th.OTHER_SERIES[2], th.NEUTRAL]); b.set_xticks(range(3)); b.set_xticklabels(labs, color=th.INK_2, fontsize=8)
for i, v in enumerate(vals): b.text(i, v + 0.02, f"{v:.2f}", ha="center", color=th.INK, fontsize=9)
b.set_ylim(0, 1.15); b.set_title("P(max |move| >= 50 bps within 60 min)", color=th.INK, fontsize=10); b.tick_params(colors=th.MUTED); b.grid(color=th.GRID, lw=0.5, axis="y")
for sp in b.spines.values(): sp.set_color(th.AXIS)
fig.suptitle(f"T14 M3 Kronos-small: 4 sampled 12-bar paths vs the real hour; kr_range_med {row['kr_range_med']*1e4:.0f} bps, kr_vol {row['kr_vol']*1e4:.1f} bps/5m", color=th.INK, fontsize=11)
fig.tight_layout(); png = os.path.join(HERE, f"fig_{NAME}_example.png"); fig.savefig(png, dpi=130, facecolor=th.PAPER); plt.close(fig)
pl = {"data": [], "layout": {"paper_bgcolor": th.PAPER, "plot_bgcolor": th.SURFACE, "font": {"color": th.INK, "family": th.FONT}, "height": 520,
      "title": {"text": f"T14 M3 Kronos: 4 sampled paths vs the real hour {pick} UTC"}, "xaxis": {"title": "5-minute bars from the anchor"}, "yaxis": {"title": "bps from last close"}}}
pl["data"].append({"type": "scatter", "mode": "lines+markers", "name": "real before", "x": [int(v) for v in x_ctx], "y": [float(v) for v in bps(ctx["close"])], "line": {"color": th.INK}})
pl["data"].append({"type": "scatter", "mode": "lines+markers", "name": "real hour", "x": list(range(len(real))), "y": [float(v) for v in bps(real["close"])], "line": {"color": th.SERIES[0]}})
for k in range(P.shape[0]):
    pl["data"].append({"type": "scatter", "mode": "lines+markers", "name": f"Kronos path {k+1}", "x": list(range(12)), "y": [float(v) for v in bps(P[k, :, 3])], "line": {"color": th.OTHER_SERIES[k % 7]}})
pl["layout"]["shapes"] = [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": s, "y1": s, "line": {"color": th.CRITICAL, "dash": "dash"}} for s in (-50, 50)]
pl["layout"]["annotations"] = [{"text": f"kr_phit50 {row['kr_phit50']:.2f} | H3 P(hit) {p_h3:.2f} | realised hit {real_hit}", "xref": "paper", "yref": "paper", "x": 0.5, "y": 1.06, "showarrow": False}]
json.dump(pl, open(os.path.join(HERE, f"fig_{NAME}_example.json"), "w"))
print("anchor", pick, "kr_phit50", row["kr_phit50"], "H3", p_h3, "realised hit", real_hit, "png", png)

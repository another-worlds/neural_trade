"""The one explanatory figure of M6: for one real Binance hour (rule: the anchor of 2026-03 whose realised next-hour volatility is the median among the
served ok anchors of that month) the 50 analog windows' next-hour paths (faint) around the real path (bold), with the analogs' dates listed.
Run: D:/nt/nt_mm_env/python.exe make_figure.py  -> fig_analogs.png, fig_analogs.json, fig_analogs.txt (dates and numbers)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m6_lib as M  # noqa: E402
import anchors as AN  # noqa: E402
import view_cache as VC  # noqa: E402
import evaluate as EV  # noqa: E402

th = EV.theme_colors()
src = "fwd"; base = AN.load_base(src); st = AN.get_store(src); m1 = st.m1; ta = base.ta
desc = pd.read_parquet(os.path.join(AN.CACHE_DIR, f"desc_{src}.parquet")).reindex(ta).to_numpy(float)
view = VC.read_view("M6_faiss", src)
lab = np.column_stack([base.arr["A_dA"], base.arr["B_y50"].astype(float)]); ok = base.ok & np.isfinite(lab).all(1)
c = m1["close"].astype(float)


def path(t0):
    """bps path of minutes 1..60 after anchor t0 relative to the close of the last bar before t0; None when a bar is missing."""
    idx = pd.date_range(t0 - pd.Timedelta(minutes=1), periods=61, freq="min")
    s = c.reindex(idx)
    return None if s.isna().any() else 1e4 * np.log(s.values[1:] / s.values[0])


def realised(t0):
    p = path(t0); return np.nan if p is None else float(np.sqrt(np.sum(np.diff(np.r_[0.0, p]) ** 2)))   # bps, sum of squared minute returns


# the median-realised-vol served hour of 2026-03
mar = [i for i in range(len(ta)) if ta[i].year == 2026 and ta[i].month == 3 and ok[i] and ta[i] in view.index]
rv = np.array([realised(ta[i]) for i in mar]); good = np.isfinite(rv); mar = np.array(mar)[good]; rv = rv[good]
med = np.median(rv); i0 = int(mar[np.argmin(np.abs(rv - med))]); t0 = ta[i0]
nbr, dist, pool = M.knn_pool_index(ta, ok, desc, kmax=50, queries=(np.arange(len(ta)) == i0))
nb = nbr[i0, :50]; assert (nb >= 0).all()
assert (ta[nb] + pd.Timedelta(minutes=60) <= t0 - pd.Timedelta(days=1)).all()
p_real = path(t0); paths = [path(ta[j]) for j in nb]; keep = [k for k, p in enumerate(paths) if p is not None]
P = np.array([paths[k] for k in keep]); tt = np.arange(1, 61)
nb_mean_dA = float(lab[nb, 0].mean()); nb_hit = float(lab[nb, 1].mean())
real_dA = float(lab[i0, 0]); real_hit = float(lab[i0, 1])
sd_real = realised(t0); sd_nb = np.array([realised(ta[j]) for j in nb])
facts = (f"anchor {t0} UTC (Binance, fwd file); realised next-hour vol {sd_real:.1f} bps (median of {len(rv)} served ok hours of 2026-03 = {med:.1f} bps); pool {int(pool[i0])} past anchors; "
         f"50 analogs: mean realised dA {nb_mean_dA:+.3f} (real {real_dA:+.3f}), barrier-hit rate {nb_hit:.2f} (real {real_hit:.0f}); "
         f"analog realised vol median {np.median(sd_nb):.1f} bps; mean distance {dist[i0, :50].mean():.2f}; {len(keep)}/50 paths complete")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig = plt.figure(figsize=(13, 7.2), facecolor=th.PAPER)
ax = fig.add_axes([0.06, 0.12, 0.58, 0.75], facecolor=th.SURFACE); tx = fig.add_axes([0.66, 0.05, 0.33, 0.85], facecolor=th.PAPER); tx.axis("off")
for p in P:
    ax.plot(tt, p, color=th.SERIES[0], alpha=0.22, lw=0.9)
ax.plot(tt, np.median(P, 0), color=th.SERIES[2], lw=1.6, ls="--", label="median of the 50 analogs")
ax.plot(tt, p_real, color=th.SERIES[1], lw=2.6, label=f"the real hour {t0:%Y-%m-%d %H:%M}")
for y in (50, -50):
    ax.axhline(y, color=th.NEUTRAL, lw=0.8, ls=":")
ax.axhline(0, color=th.NEUTRAL, lw=0.8)
ax.set_xlabel("minutes after the anchor", color=th.INK_2); ax.set_ylabel("cumulative log return, bps (barrier +-50 dotted)", color=th.INK_2)
ax.tick_params(colors=th.MUTED); ax.grid(color=th.GRID, lw=0.5)
for sp in ax.spines.values():
    sp.set_color(th.AXIS)
ax.legend(facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2, fontsize=9, loc="upper left")
fig.suptitle("T14 M6 regime retrieval: the 50 nearest past hours (faint) and what the next hour did", color=th.INK, fontsize=12, x=0.06, ha="left")
fig.text(0.06, 0.935, "analogs come only from anchors whose label window ended at least a day before the anchor; keys: vol percentile, session, stretch, S/R distance, last 3 hourly log RV",
         color=th.MUTED, fontsize=8, ha="left")
order = np.argsort(ta[nb].values); lines = [f"{ta[nb[o]]:%Y-%m-%d %H:%M}  d={dist[i0, o]:.2f}  {sd_nb[o]:5.1f}bp" for o in order]
tx.text(0.0, 1.0, "50 analog hours (UTC), sorted by date;\nd = distance, realised next-hour vol", color=th.INK_2, fontsize=8, va="top", family="monospace")
half = 25
for col in range(2):
    tx.text(0.0 + col * 0.52, 0.93, "\n".join(lines[col * half:(col + 1) * half]), color=th.INK, fontsize=5.8, va="top", family="monospace")
fig.text(0.06, 0.02, facts, color=th.MUTED, fontsize=6.5, ha="left", wrap=True)
png = os.path.join(HERE, "fig_analogs.png"); fig.savefig(png, dpi=140, facecolor=th.PAPER); plt.close(fig)

pl = {"data": [{"type": "scatter", "mode": "lines", "name": "analog", "x": tt.tolist(), "y": [float(v) for v in p], "line": {"color": th.rgba(th.SERIES[0], 0.25) if hasattr(th, "rgba") else th.SERIES[0], "width": 1},
                "legendgroup": "analogs", "showlegend": bool(k == 0), "hovertext": f"{ta[nb[keep[k]]]}"} for k, p in enumerate(P)]
      + [{"type": "scatter", "mode": "lines", "name": "median of analogs", "x": tt.tolist(), "y": np.median(P, 0).tolist(), "line": {"color": th.SERIES[2], "dash": "dash", "width": 2}},
         {"type": "scatter", "mode": "lines", "name": f"real hour {t0}", "x": tt.tolist(), "y": p_real.tolist(), "line": {"color": th.SERIES[1], "width": 3}}],
      "layout": {"paper_bgcolor": th.PAPER, "plot_bgcolor": th.SURFACE, "font": {"color": th.INK, "family": th.FONT}, "height": 620,
                 "title": {"text": "T14 M6 regime retrieval: 50 nearest past hours and the real hour"}, "xaxis": {"title": "minutes after anchor", "gridcolor": th.GRID},
                 "yaxis": {"title": "cumulative log return (bps)", "gridcolor": th.GRID, "zerolinecolor": th.AXIS},
                 "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": y, "y1": y, "line": {"color": th.NEUTRAL, "dash": "dot"}} for y in (50, -50)],
                 "annotations": [{"text": facts, "xref": "paper", "yref": "paper", "x": 0, "y": -0.2, "showarrow": False, "font": {"size": 9, "color": th.MUTED}}]},
      "analog_dates": [str(ta[nb[o]]) for o in order]}
json.dump(pl, open(os.path.join(HERE, "fig_analogs.json"), "w"))
open(os.path.join(HERE, "fig_analogs.txt"), "w", encoding="utf-8").write(facts + "\n" + "\n".join(lines) + "\n")
print(facts)

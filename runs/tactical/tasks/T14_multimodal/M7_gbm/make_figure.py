"""The one explanatory figure of M7: partial dependence of gbm_dA_hat (next-hour change of log RV) from the FINAL model (the one for the last month, 2026-09:
fitted on the 365 days before it, labels ended). Left: vol percentile x UTC hour. Middle: what is left of that surface after the best additive fit
(row effect + column effect): the interaction a ridge on the same keys cannot form. Right: PD on stretch (str20, str60) against the linear (ridge) PD.
Vol percentile = vp_h and vp_d set to the same value (the other keys stay at their observed values, 500 background pool rows).
Run: D:/nt/nt_mm_env/python.exe make_figure.py -> fig_pd.png, fig_pd.json, fig_pd.txt."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m7_lib as L  # noqa: E402
import build_m7 as B  # noqa: E402  (also puts M6 on the path)
import m6_lib as M6L  # noqa: E402
import evaluate as EV  # noqa: E402

th = EV.theme_colors()
D = B.load_all(); ta = D["ta"]; K = D["keys"]; ok = D["ok"] & B.finite_rows(K, D["dA"], D["hit"])
m0 = pd.Period("2026-09", "M").start_time
tr = np.where(ok & np.asarray(ta >= m0 - pd.Timedelta(days=365)) & np.asarray(ta + pd.Timedelta(minutes=60) <= m0))[0]
_, _, mdl = L.gbm_fit_predict(K[tr], D["dA"][tr], K[tr[:3]], False)
rng = np.random.default_rng(0); bg = K[rng.choice(tr, 500, replace=False)]
ix = {k: i for i, k in enumerate(M6L.KEYS)}
vgrid = np.linspace(0.05, 0.95, 19); hours = np.arange(24)

Z = np.zeros((len(vgrid), len(hours)))
for a, v in enumerate(vgrid):
    for b, h in enumerate(hours):
        X = bg.copy(); X[:, ix["vp_h"]] = v; X[:, ix["vp_d"]] = v
        X[:, ix["hsin"]] = np.sin(2 * np.pi * h / 24); X[:, ix["hcos"]] = np.cos(2 * np.pi * h / 24)
        Z[a, b] = mdl.predict(X).mean()
add = Z.mean(1, keepdims=True) + Z.mean(0, keepdims=True) - Z.mean()
I = Z - add

# linear PD: ridge on the same keys and pool
mu, sd = K[tr].mean(0), K[tr].std(0) + 1e-9
Xs = (K[tr] - mu) / sd
y = D["dA"][tr]
w = np.linalg.solve(Xs.T @ Xs + 1.0 * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
base_pred = (((bg - mu) / sd) @ w + y.mean()).mean()


def lin_pd(key, grid):
    j = ix[key]
    return np.array([base_pred + w[j] * ((g - mu[j]) / sd[j] - ((bg[:, j] - mu[j]) / sd[j]).mean()) for g in grid])


def gbm_pd(key, grid):
    out = []
    for g in grid:
        X = bg.copy(); X[:, ix[key]] = g; out.append(mdl.predict(X).mean())
    return np.array(out)


curves = {}
for key in ("str20", "str60"):
    lo, hi = np.quantile(K[tr][:, ix[key]], [0.01, 0.99]); g = np.linspace(lo, hi, 41)
    curves[key] = {"x": g, "gbm": gbm_pd(key, g), "lin": lin_pd(key, g)}
marg = np.array([gbm_pd("vp_h", [v])[0] for v in vgrid])

facts = [f"final model: month 2026-09, pool {len(tr)} anchors ({ta[tr].min():%Y-%m-%d}..{ta[tr].max():%Y-%m-%d}), LightGBM 200 rounds, 15 leaves",
         f"PD surface vol percentile x hour: range {Z.min():+.3f}..{Z.max():+.3f} (log RV change units); additive part explains {1 - I.var() / Z.var():.1%} of its variance; "
         f"interaction range {I.min():+.3f}..{I.max():+.3f}, sd {I.std():.3f}"]
for key in ("str20", "str60"):
    c = curves[key]
    facts.append(f"{key}: PD range gbm {c['gbm'].min():+.3f}..{c['gbm'].max():+.3f} vs linear {c['lin'].min():+.3f}..{c['lin'].max():+.3f}; PD at the low end {c['gbm'][0]:+.3f}, middle {c['gbm'][20]:+.3f}, "
                 f"high end {c['gbm'][-1]:+.3f}; correlation(gbm, linear) {np.corrcoef(c['gbm'], c['lin'])[0, 1]:.2f}; gbm minus linear sd {np.std(c['gbm'] - c['lin']):.3f}")
flat = np.dstack(np.unravel_index(np.argsort(-np.abs(I).ravel())[:5], I.shape))[0]
facts.append("largest interaction cells (vol pct, UTC hour, value): " + "; ".join(f"({vgrid[a]:.2f}, {hours[b]:02d}h, {I[a, b]:+.3f})" for a, b in flat))
facts.append("hour profile at vol pct 0.15 / 0.50 / 0.85 (peak hour, trough hour, swing): "
             + "; ".join(f"{vgrid[a]:.2f}: {hours[Z[a].argmax()]:02d}h, {hours[Z[a].argmin()]:02d}h, {Z[a].max() - Z[a].min():.3f}" for a in (2, 9, 16)))
facts.append(f"vp_h alone (vp_d untouched): {marg[0]:+.3f} at 0.05, {marg[9]:+.3f} at 0.50, {marg[-1]:+.3f} at 0.95")
print("\n".join(facts))
open(os.path.join(HERE, "fig_pd.txt"), "w", encoding="utf-8").write("\n".join(facts) + "\n")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
seq = LinearSegmentedColormap.from_list("nt_seq", [th.SURFACE, th.SERIES[0], th.INK])
div = LinearSegmentedColormap.from_list("nt_div", [th.SERIES[3], th.SURFACE, th.SERIES[0]])
fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=th.PAPER, gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
for ax in axs:
    ax.set_facecolor(th.SURFACE); ax.tick_params(colors=th.MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(th.AXIS)
ext = [-0.5, 23.5, vgrid[0] - 0.025, vgrid[-1] + 0.025]
im = axs[0].imshow(Z, origin="lower", aspect="auto", extent=ext, cmap=seq); cb = fig.colorbar(im, ax=axs[0], fraction=0.046); cb.ax.tick_params(colors=th.MUTED, labelsize=8)
axs[0].set_title("A. predicted change of next-hour log RV (partial dependence)", color=th.INK, fontsize=10)
axs[0].set_xlabel("UTC hour of the anchor", color=th.INK_2); axs[0].set_ylabel("vol percentile (hour and day, together)", color=th.INK_2)
v = np.abs(I).max()
im = axs[1].imshow(I, origin="lower", aspect="auto", extent=ext, cmap=div, vmin=-v, vmax=v); cb = fig.colorbar(im, ax=axs[1], fraction=0.046); cb.ax.tick_params(colors=th.MUTED, labelsize=8)
axs[1].set_title("B. interaction left after the best additive fit (row + column)", color=th.INK, fontsize=10); axs[1].set_xlabel("UTC hour of the anchor", color=th.INK_2)
for key, ci in (("str20", 0), ("str60", 1)):
    c = curves[key]
    axs[2].plot(c["x"], c["gbm"], color=th.SERIES[ci], lw=2.2, label=f"GBM, {key}")
    axs[2].plot(c["x"], c["lin"], color=th.SERIES[ci], lw=1.4, ls="--", label=f"ridge on the keys, {key}")
axs[2].axvline(0, color=th.NEUTRAL, lw=0.8, ls=":"); axs[2].grid(color=th.GRID, lw=0.5)
axs[2].set_title("C. stretch: ln(close/EMA) in sigma units", color=th.INK, fontsize=10)
axs[2].set_xlabel("stretch (sigma)", color=th.INK_2); axs[2].set_ylabel("PD of gbm_dA_hat", color=th.INK_2)
axs[2].legend(facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2, fontsize=8)
fig.suptitle("T14 M7 nonlinear regime stage: what the GBM on the 12 regime keys learned (final model, 2026-09)", color=th.INK, fontsize=12, x=0.01, ha="left")
fig.text(0.01, 0.01, facts[1], color=th.MUTED, fontsize=7, ha="left")
fig.tight_layout(rect=[0, 0.04, 1, 0.94]); fig.savefig(os.path.join(HERE, "fig_pd.png"), dpi=130, facecolor=th.PAPER); plt.close(fig)

pl = {"data": [{"type": "heatmap", "z": Z.tolist(), "x": hours.tolist(), "y": vgrid.tolist(), "colorscale": [[0, th.SURFACE], [0.5, th.SERIES[0]], [1, th.INK]],
                "xaxis": "x", "yaxis": "y", "name": "PD", "colorbar": {"x": 0.30, "len": 0.8}},
               {"type": "heatmap", "z": I.tolist(), "x": hours.tolist(), "y": vgrid.tolist(), "colorscale": [[0, th.SERIES[3]], [0.5, th.SURFACE], [1, th.SERIES[0]]], "zmid": 0,
                "xaxis": "x2", "yaxis": "y2", "name": "interaction", "colorbar": {"x": 0.65, "len": 0.8}}]
      + [{"type": "scatter", "mode": "lines", "x": curves[k]["x"].tolist(), "y": curves[k][t].tolist(), "name": f"{'GBM' if t == 'gbm' else 'ridge'}, {k}", "xaxis": "x3", "yaxis": "y3",
          "line": {"color": th.SERIES[i], "dash": "solid" if t == "gbm" else "dash"}} for i, k in enumerate(("str20", "str60")) for t in ("gbm", "lin")],
      "layout": {"paper_bgcolor": th.PAPER, "plot_bgcolor": th.SURFACE, "font": {"color": th.INK, "family": th.FONT}, "height": 560,
                 "title": {"text": "T14 M7: partial dependence of gbm_dA_hat (final model, 2026-09)"},
                 "xaxis": {"domain": [0, 0.28], "title": "UTC hour", "gridcolor": th.GRID}, "yaxis": {"title": "vol percentile", "gridcolor": th.GRID, "anchor": "x"},
                 "xaxis2": {"domain": [0.36, 0.64], "title": "UTC hour", "anchor": "y2"}, "yaxis2": {"anchor": "x2"},
                 "xaxis3": {"domain": [0.74, 1], "title": "stretch (sigma)", "gridcolor": th.GRID, "anchor": "y3"}, "yaxis3": {"title": "PD", "gridcolor": th.GRID, "anchor": "x3"},
                 "annotations": [{"text": facts[1], "xref": "paper", "yref": "paper", "x": 0, "y": -0.2, "showarrow": False, "font": {"size": 9, "color": th.MUTED}}]}}
json.dump(pl, open(os.path.join(HERE, "fig_pd.json"), "w"))

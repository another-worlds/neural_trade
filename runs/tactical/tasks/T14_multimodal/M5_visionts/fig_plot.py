"""M5 figure step 2 (nt env, theme.py): fig_vts.png (matplotlib) and fig_vts.json (plotly) from fig_data.npz, day_data.csv, day_price.csv."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
from neural_trade.visualization import theme as T
Z = np.load(os.path.join(HERE, "fig_data.npz")); npi = int(Z["npi"])
d = pd.read_csv(os.path.join(HERE, "day_data.csv"), index_col=0, parse_dates=True)
TA = pd.Timestamp("2026-03-20 14:00")
img_in = Z["img_in"][:, :npi].copy(); img_rec = Z["img_rec"].copy()
ctx_idx = pd.DatetimeIndex(Z["ctx_idx"]); ctx_rv = Z["ctx_rv"]; fut = pd.DatetimeIndex(Z["fut"])
last = ctx_idx >= TA - pd.Timedelta(hours=24)
C_REAL, C_V3, C_M5 = T.INK, T.SERIES[0], T.SERIES[1]
vmin, vmax = -1.6, 1.6
rows_lab = [(0, "14:00"), (56, "18:00"), (112, "22:00"), (168, "02:00"), (224 - 1, "14:00 +1d")]   # 224 image rows span 288 five-minute bars (24 h) from 14:00 UTC
x = list(d.index) + [d.index[-1] + pd.Timedelta(hours=1)]
def step(c): return list(d[c]) + [d[c].iloc[-1]]
tt = "VisionTS on one anchor: the picture of a week of 5-minute log RV, its reconstruction, and the hourly forecast"
sub = "Anchor 2026-03-20 14:00 UTC (lower-median-vol day of 2026-03, by rule). Zero-shot MAE-base, period 288 (rows = time of day from 14:00, columns = the 7 context days resized)."
# ---- plotly
fig = make_subplots(rows=2, cols=2, subplot_titles=("Input image (7 days; masked part = the future)", "Reconstruction (right of the line = the forecast region)",
                    "Next hour: forecast vs realised 5-min ln RV", "Hourly vol that day, bps"), vertical_spacing=0.14, horizontal_spacing=0.08)
fig.add_trace(go.Heatmap(z=img_in, zmin=vmin, zmax=vmax, colorscale="Greys", showscale=False, name="input"), row=1, col=1)
fig.add_trace(go.Heatmap(z=img_rec, zmin=vmin, zmax=vmax, colorscale="Greys", showscale=False, name="reconstruction"), row=1, col=2)
fig.add_vline(x=npi - 0.5, line=dict(color=C_M5, width=2), row=1, col=2)
for r in (1,):
    for c in (1, 2):
        fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=[a for a, _ in rows_lab], ticktext=[b for _, b in rows_lab], row=r, col=c)
fig.add_trace(go.Scatter(x=ctx_idx[last], y=ctx_rv[last], name="context, last 24 h", line=dict(color=T.NEUTRAL, width=1.2)), row=2, col=1)
fig.add_trace(go.Scatter(x=fut, y=Z["real_rv"], name="realised next hour", line=dict(color=C_REAL, width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=fut, y=Z["y_rv"], name="VisionTS forecast", line=dict(color=C_M5, width=2, dash="dash")), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=step("vol_bps_vts_logsum"), name="VisionTS implied hourly vol (debiased)", line=dict(color=C_M5, width=2, shape="hv")), row=2, col=2)
fig.add_trace(go.Scatter(x=x, y=step("v3_vol_bps"), name="V3 forecast", line=dict(color=C_V3, width=2, dash="dash", shape="hv")), row=2, col=2)
fig.add_trace(go.Scatter(x=x, y=step("real_vol_bps"), name="realised hourly vol", line=dict(color=C_REAL, width=2.2, shape="hv")), row=2, col=2)
fig.update_yaxes(title_text="ln(rv5 + 1e-12)", row=2, col=1); fig.update_yaxes(title_text="bps (sqrt RV)", row=2, col=2)
T.apply(fig, title=tt, subtitle=sub, height=760)
fig.write_json(os.path.join(HERE, "fig_vts.json"))
# ---- matplotlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as md
f, ax = plt.subplots(2, 2, figsize=(13, 8.4), facecolor=T.PAPER, gridspec_kw={"height_ratios": [1, 0.8]})
for a in ax.ravel():
    a.set_facecolor(T.SURFACE); [s.set_color(T.AXIS) for s in a.spines.values()]; a.tick_params(colors=T.MUTED)
ax[0, 0].imshow(img_in, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
ax[0, 1].imshow(img_rec, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"); ax[0, 1].axvline(npi - 0.5, color=C_M5, lw=2)
ax[0, 0].set_title("Input image: ln rv5, 7 days x time of day (masked part cut)", color=T.INK_2, fontsize=10, loc="left")
ax[0, 1].set_title("Reconstruction (right of the line = forecast region)", color=T.INK_2, fontsize=10, loc="left")
for a in ax[0]:
    a.set_yticks([p for p, _ in rows_lab]); a.set_yticklabels([q for _, q in rows_lab], color=T.MUTED); a.set_ylabel("UTC time of day (from 14:00)", color=T.INK_2); a.set_xlabel("columns: 7 days (old to new) resized to 64 px; right: 224 px", color=T.INK_2)
b = ax[1, 0]
b.plot(ctx_idx[last], ctx_rv[last], color=T.NEUTRAL, lw=1.1, label="context, last 24 h"); b.plot(fut, Z["real_rv"], color=C_REAL, lw=2, label="realised next hour")
b.plot(fut, Z["y_rv"], color=C_M5, lw=2, ls="--", label="VisionTS forecast"); b.axvline(TA, color=T.AXIS, lw=1)
b.xaxis.set_major_formatter(md.DateFormatter("%H:%M")); b.set_ylabel("ln(rv5 + 1e-12)", color=T.INK_2); b.set_xlabel("UTC, 2026-03-19/20", color=T.INK_2)
c = ax[1, 1]; xi = pd.DatetimeIndex(x)
c.step(xi, step("vol_bps_vts_logsum"), where="post", color=C_M5, lw=2, label="VisionTS implied hourly vol (debiased)")
c.step(xi, step("v3_vol_bps"), where="post", color=C_V3, lw=2, ls="--", label="V3 forecast"); c.step(xi, step("real_vol_bps"), where="post", color=C_REAL, lw=2.2, label="realised hourly vol")
c.axvline(TA, color=T.AXIS, lw=1); c.xaxis.set_major_formatter(md.DateFormatter("%H")); c.set_ylabel("hourly vol, bps (sqrt RV)", color=T.INK_2); c.set_xlabel("UTC hour, 2026-03-20", color=T.INK_2)
for a in (b, c):
    a.grid(color=T.GRID); l = a.legend(loc="upper left", facecolor=T.SURFACE, edgecolor=T.AXIS, fontsize=8.5); [t.set_color(T.INK_2) for t in l.get_texts()]
f.suptitle("VisionTS on one anchor, 2026-03-20 14:00 UTC (lower-median-vol day of 2026-03)", color=T.INK, x=0.01, ha="left", fontweight="bold", fontsize=13)
f.tight_layout(rect=(0, 0, 1, 0.96)); f.savefig(os.path.join(HERE, "fig_vts.png"), dpi=130, facecolor=T.PAPER)
print("ok")

"""Step 2 of the figure (nt env, theme.py colours): fig_day.png (matplotlib) and fig_day.json (plotly figure JSON) from day_data.csv / day_price.csv."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
from neural_trade.visualization import theme as T
d = pd.read_csv(os.path.join(HERE, "day_data.csv"), index_col=0, parse_dates=True); pr = pd.read_csv(os.path.join(HERE, "day_price.csv"), index_col=0, parse_dates=True)["close"]
day = d.index[0].date()
# hourly series are plotted as steps over their forecast hour [ta, ta+1h)
C_REAL, C_V3, C_CH = T.INK, T.SERIES[0], T.SERIES[1]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6], vertical_spacing=0.06)
fig.add_trace(go.Scatter(x=pr.index, y=pr.values, name="BTC/USDT close (5-min)", line=dict(color=T.NEUTRAL, width=1.5)), row=1, col=1)
x = list(d.index) + [d.index[-1] + pd.Timedelta(hours=1)]
def step(col): return list(d[col]) + [d[col].iloc[-1]]
fig.add_trace(go.Scatter(x=x, y=step("vol_bps_ttm_logsum"), name="TTM r2 implied hourly vol (debiased)", line=dict(color=C_CH, width=2, shape="hv")), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=step("v3_vol_bps"), name="V3 forecast", line=dict(color=C_V3, width=2, dash="dash", shape="hv")), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=step("real_vol_bps"), name="realised hourly vol", line=dict(color=C_REAL, width=2.2, shape="hv")), row=2, col=1)
fig.update_yaxes(title_text="price", row=1, col=1); fig.update_yaxes(title_text="hourly vol, bps (sqrt RV)", row=2, col=1); fig.update_xaxes(title_text="UTC", row=2, col=1)
T.apply(fig, title=f"TTM r2 against V3 on a median-vol day, {day}", subtitle="Day chosen by rule: the median mean-hourly-vol day of 2026-03. Forecasts are made at the start of each hour from the bars before it; vol = sqrt(exp(log RV)) in bps.", height=640)
fig.write_json(os.path.join(HERE, "fig_day.json"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [0.4, 0.6]}, facecolor=T.PAPER)
for a in (a1, a2):
    a.set_facecolor(T.SURFACE); a.grid(color=T.GRID); [s.set_color(T.AXIS) for s in a.spines.values()]; a.tick_params(colors=T.MUTED)
a1.plot(pr.index, pr.values, color=T.NEUTRAL, lw=1.4, label="BTC/USDT close (5-min)"); a1.set_ylabel("price", color=T.INK_2)
xi = pd.DatetimeIndex(x)
a2.step(xi, step("vol_bps_ttm_logsum"), where="post", color=C_CH, lw=2, label="TTM r2 implied hourly vol (debiased)")
a2.step(xi, step("v3_vol_bps"), where="post", color=C_V3, lw=2, ls="--", label="V3 forecast")
a2.step(xi, step("real_vol_bps"), where="post", color=C_REAL, lw=2.2, label="realised hourly vol")
a2.set_ylabel("hourly vol, bps (sqrt RV)", color=T.INK_2); a2.set_xlabel("UTC", color=T.INK_2)
for a in (a1, a2):
    l = a.legend(loc="upper left", facecolor=T.SURFACE, edgecolor=T.AXIS, fontsize=9); [t.set_color(T.INK_2) for t in l.get_texts()]
f.suptitle(f"TTM r2 against V3 on a median-vol day, {day}", color=T.INK, x=0.01, ha="left", fontweight="bold", fontsize=13)
f.tight_layout(rect=(0, 0, 1, 0.95)); f.savefig(os.path.join(HERE, "fig_day.png"), dpi=130, facecolor=T.PAPER)

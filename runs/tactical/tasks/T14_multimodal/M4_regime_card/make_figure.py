"""Figure: three real regime cards (Binance anchors at the 5th / 50th / 95th percentile of realised next-hour vol in 2026-03), each with the rendered image and its
caption, and below them the predicted vs realised hourly vol of V3 and V3 + image (M4_img, the pre-registered R1 view; predictions from the same monthly
walk-forward as the harness, month 2026-03). Run: D:/nt/nt_mm_env/python.exe make_figure.py  -> fig_regime_cards.png / .json"""
import base64, io, json, os, sys, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import m4_lib as L, build_m4 as B, anchors as AN, view_cache as VC, evaluate as EV
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
th = EV.theme_colors()
base, m1, ta, sigma, pos, rows = B.prep("fwd")
view = VC.read_view("M4_img", "fwd"); M, cols = VC.align_view(view, ta); valid = base.ok & np.isfinite(M).all(1)
mo = pd.Period("2026-03", "M"); start = AN.LOOKS["look2"]["train_from"]
r = EV.run_A(base, valid, [mo], start, {"V3": None, "view": M}, keep_preds=True)[0]
te = r["te"]; lv0 = base.A["LV0"][te]; y = base.A["T"]["A"][te]
real = lv0 + y; p3 = lv0 + r["preds"]["V3"]; pv = lv0 + r["preds"]["view"]
vol = lambda lv: np.sqrt(np.exp(lv)) * 1e4                                       # hourly realised vol in bps (sqrt of the sum of squared 1-minute log returns)
cap = pd.read_parquet(os.path.join(AN.CACHE_DIR, "m4_captions_fwd.parquet"))["caption"]
picks = []
for q in (0.05, 0.5, 0.95):
    k = int(np.argmin(np.abs(real - np.quantile(real, q)))); picks.append((q, k))
o, h, l, c, v = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
fig, axs = plt.subplots(2, 3, figsize=(15, 9.2), facecolor=th.PAPER, gridspec_kw={"height_ratios": [3, 1.55]})
pl = {"data": [], "layout": {"paper_bgcolor": th.PAPER, "plot_bgcolor": th.SURFACE, "font": {"color": th.INK, "family": th.FONT}, "height": 900, "images": [], "annotations": [],
      "title": {"text": "M4 regime cards, Binance 2026-03: the card the model sees, its caption, and V3 vs V3 + image on that hour"}}}
info = []
for j, (q, k) in enumerate(picks):
    i = te[k]; t = ta[i]; im = L.render_images(o, h, l, c, v, np.array([pos[i]]), np.array([sigma[i]]))
    img = next(iter(im))[1][0]; txt = cap.iloc[i]
    ax = axs[0][j]; ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"realised next-hour vol p{int(q * 100)}: anchor {t:%Y-%m-%d %H:%M} UTC", color=th.INK, fontsize=10)
    ax.text(0.0, -0.03, "\n".join(textwrap.wrap(txt, 52)), transform=ax.transAxes, va="top", ha="left", color=th.INK_2, fontsize=8.5)
    for sp in ax.spines.values():
        sp.set_color(th.AXIS)
    vals = [vol(real[k]), vol(p3[k]), vol(pv[k])]; labs = ["realised", "V3", "V3 + image"]; cl = [th.INK, th.OTHER_SERIES[0], th.OTHER_SERIES[1]]
    b = axs[1][j]; b.set_facecolor(th.SURFACE); b.bar(labs, vals, color=cl)
    for xx, vv in zip(labs, vals):
        b.text(xx, vv, f"{vv:.0f}", ha="center", va="bottom", color=th.INK_2, fontsize=8)
    b.set_ylabel("hourly vol, bps", color=th.INK_2, fontsize=8); b.tick_params(colors=th.MUTED, labelsize=8); b.grid(color=th.GRID, lw=0.5, axis="y")
    for sp in b.spines.values():
        sp.set_color(th.AXIS)
    buf = io.BytesIO(); Image.fromarray(img).resize((448, 448), Image.NEAREST).save(buf, "PNG")
    d = 1 / 3
    pl["layout"]["images"].append({"source": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "xref": "paper", "yref": "paper", "x": j * d + 0.01,
                                   "y": 0.98, "sizex": d - 0.03, "sizey": 0.55, "xanchor": "left", "yanchor": "top"})
    pl["layout"]["annotations"].append({"text": "<br>".join(textwrap.wrap(f"p{int(q * 100)} | {t:%Y-%m-%d %H:%M} UTC | " + txt, 55)), "x": j * d + 0.01, "y": 0.41, "xref": "paper", "yref": "paper",
                                        "showarrow": False, "xanchor": "left", "yanchor": "top", "align": "left", "font": {"size": 10, "color": th.INK_2}})
    sfx = "" if j == 0 else str(j + 1)
    pl["data"].append({"type": "bar", "x": labs, "y": [float(x) for x in vals], "marker": {"color": cl}, "xaxis": "x" + sfx, "yaxis": "y" + sfx, "showlegend": False})
    pl["layout"]["xaxis" + sfx] = {"domain": [j * d + 0.04, (j + 1) * d - 0.03], "anchor": "y" + sfx}
    pl["layout"]["yaxis" + sfx] = {"domain": [0.0, 0.2], "anchor": "x" + sfx, "title": "hourly vol, bps", "gridcolor": th.GRID}
    info.append({"q": q, "anchor": str(t), "caption": txt, "real_bps": float(vals[0]), "V3_bps": float(vals[1]), "V3_image_bps": float(vals[2])})
fig.suptitle("M4 regime cards, Binance 2026-03: the 224x224 card (candles, volume, T13 hull / pivot / channel lines), its caption, and V3 vs V3 + image on that hour", color=th.INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.subplots_adjust(hspace=0.32)
fig.savefig(os.path.join(HERE, "fig_regime_cards.png"), dpi=130, facecolor=th.PAPER)
json.dump(pl, open(os.path.join(HERE, "fig_regime_cards.json"), "w")); json.dump(info, open(os.path.join(HERE, "fig_regime_cards_info.json"), "w"), indent=1)
print(json.dumps(info, indent=1))

"""T02 looks. Refuses to run unless selection.json is committed. Look 1: 2024-07..2025-07-20 (monthly refit, descriptive).
Look 2: Binance forward, models and thresholds frozen on the 365 days to 2025-07-20 (the verdict)."""
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t02lib as L  # noqa: E402

sel_path = os.path.join(HERE, "selection.json")
tracked = subprocess.run(["git", "ls-files", "--error-unmatch", sel_path], capture_output=True).returncode == 0
clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", sel_path]).returncode == 0
assert tracked and clean, "selection.json must be committed before the looks"
S = json.load(open(sel_path))
cfg = tuple(S["cfg"])
name = S["selected"]
t0 = time.time()
df, _ = L.load_main()
B = L.Bars(df)

# Look 1
sig1 = L.monthly_signals(B, "2024-07", "2025-07")
f1 = L.hour_frame(B, sig1)
# Look 2: frozen on the 365 days to 2025-07-20
span_lo = np.datetime64(L.CUT.normalize() - pd.Timedelta(days=L.TRAIN_DAYS - 1), "m")
span_hi = np.datetime64(L.CUT + pd.Timedelta(minutes=1), "m")
M = L.Models(B, span_lo, span_hi)
dff, _ = L.load_fwd()
Bf = L.Bars(dff)
a = L.hour_anchors(Bf)
a = a[(Bf.t[a] >= np.datetime64(L.FWD_START, "m")) & (Bf.t[a] < np.datetime64(L.FWD_END, "m"))]
sg = L.signals_for(Bf, M, a)
sg["pos"] = a
sg["month"] = Bf.index[a].strftime("%Y-%m").values
f2 = L.hour_frame(Bf, pd.DataFrame(sg))
print("look1 hours", len(f1), f1.time.min(), f1.time.max(), "look2 hours", len(f2), f2.time.min(), f2.time.max(), flush=True)

res = {"selected": name, "cfg": list(cfg), "thresholds_frozen": M.thresholds(), "n_train_s1": M.n_c, "look1_hours": int(len(f1)),
       "look2_hours": int(len(f2)), "look2_first": str(f2.time.min()), "look2_last": str(f2.time.max())}
lines = []
hourly = {}
for tag, f in (("look1", f1), ("look2", f2)):
    res[tag] = {}
    lines.append(f"== {tag.upper()} {f.time.min()} .. {f.time.max()} ({len(f)} hours) {'(descriptive)' if tag == 'look1' else '(VERDICT)'} ==")
    lines.append("bps  series        total_ret  ann_ret   maxDD  Calmar   mean_w  turnover/yr")
    for bps in (0, 2, 5):
        c, (w, r, rn, rb) = L.compare(f, cfg, bps)
        res[tag][str(bps)] = c
        for k, lab in (("strategy", name), ("null", f"const {c['mean_w']:.3f}"), ("bh", "buy&hold")):
            s = c[k]
            lines.append(f"{bps:>3}  {lab:13s} {s['total_return']:+9.4f} {s['ann_return']:+8.4f} {s['max_dd']:7.4f} {s['calmar']:+8.3f}  "
                         f"{c['mean_w'] if k != 'bh' else 1.0:.3f}   {c['turnover_per_year'] if k == 'strategy' else 0:.0f}")
        if bps == 2:
            hourly[tag] = np.column_stack([w, r, rn, rb])
            res[tag]["boot_calmar_diff"] = L.boot_calmar_diff(r, rn)
            res[tag]["exposure_dd5_bh"] = L.exposure_for_dd(f["ret"].values)
            res[tag]["scale_dd5_strategy"] = L.scale_for_dd(w, f["ret"].values, 2)
            r5, _ = L.strat_returns(res[tag]["scale_dd5_strategy"] * w, f["ret"].values, 2)
            res[tag]["scaled_strategy_at_dd5"] = L.equity_stats(r5)
            res[tag]["scaled_mean_w"] = float((res[tag]["scale_dd5_strategy"] * w).mean())
            mo = pd.DataFrame({"m": f["month"].values, "s": r, "n": rn, "b": rb}).groupby("m").agg(lambda x: float(np.prod(1 + x) - 1))
            res[tag]["monthly"] = mo.to_dict("index")
            pd.DataFrame({"time": f["time"].values, "w": w, "r_strategy": r, "r_null": rn, "r_bh": rb}).to_csv(
                os.path.join(HERE, f"{tag}_hourly.csv"), index=False)
    bc = res[tag]["boot_calmar_diff"]
    lines.append(f"bootstrap (1-week blocks, 2000, paired) Calmar(strategy)-Calmar(null) at 2 bps: mean {bc['mean']:+.3f} "
                 f"95% [{bc['lo']:+.3f}, {bc['hi']:+.3f}] (n={bc['n']})")
    sc = res[tag]["scaled_strategy_at_dd5"]
    lines.append(f"constant exposure for max DD < 5% on buy-and-hold: {res[tag]['exposure_dd5_bh']:.3f}; strategy weights scaled by "
                 f"s={res[tag]['scale_dd5_strategy']:.3f} (mean exposure {res[tag]['scaled_mean_w']:.3f}) -> ann {sc['ann_return']:+.4f}, maxDD {sc['max_dd']:.4f}")
    lines.append("monthly returns (strategy / null / buy&hold):")
    for m, v in res[tag]["monthly"].items():
        lines.append(f"  {m}  {v['s']:+.4f} {v['n']:+.4f} {v['b']:+.4f}")
    lines.append("")

c2 = res["look2"]["2"]
s, n, b = c2["strategy"], c2["null"], c2["bh"]
crit = {"1_dd_le_0.8_null": bool(s["max_dd"] <= 0.8 * n["max_dd"]), "2_calmar_gt_null": bool(s["calmar"] > n["calmar"]),
        "3_calmar_gt_bh": bool(s["calmar"] > b["calmar"])}
res["criterion_look2_2bps"] = crit
res["success"] = bool(all(crit.values()))
res["owner_dd5_met_by_strategy_look2"] = bool(s["max_dd"] < 0.05)
lines.append(f"CRITERION (Look 2, 2 bps): strategy DD {s['max_dd']:.4f} vs 0.8 x null DD {0.8 * n['max_dd']:.4f} -> {crit['1_dd_le_0.8_null']}; "
             f"Calmar {s['calmar']:+.3f} vs null {n['calmar']:+.3f} -> {crit['2_calmar_gt_null']}; vs buy&hold {b['calmar']:+.3f} -> {crit['3_calmar_gt_bh']}")
lines.append(f"SUCCESS: {res['success']}   owner's DD<5% line met by the strategy: {res['owner_dd5_met_by_strategy_look2']}")
res["run_seconds"] = time.time() - t0
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
open(os.path.join(HERE, "tables.txt"), "w").write("\n".join(lines))
print("\n".join(lines))
print(f"{time.time() - t0:.0f}s")

# figures (theme colours)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from neural_trade.visualization import theme as T  # noqa: E402

for tag, f in (("look1", f1), ("look2", f2)):
    h = hourly[tag]
    eq = np.cumprod(1 + h[:, 1:], axis=0)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor=T.PAPER, gridspec_kw={"height_ratios": [3, 1]})
    for a_ in ax:
        a_.set_facecolor(T.SURFACE)
        a_.grid(color=T.GRID, lw=.6)
        for sp in a_.spines.values():
            sp.set_color(T.AXIS)
        a_.tick_params(colors=T.MUTED)
    t = f["time"].values
    for i, (lab, col, ls) in enumerate(((name + " (2 bps)", T.SERIES[0], "-"), (f"constant null {h[:, 0].mean():.2f}", T.SERIES[1], "-"),
                                        ("buy-and-hold", T.NEUTRAL, "--"))):
        ax[0].plot(t, eq[:, i], color=col, ls=ls, lw=1.4, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("equity", color=T.INK)
    ax[0].legend(facecolor=T.SURFACE, edgecolor=T.AXIS, labelcolor=T.INK)
    ax[0].set_title(f"T02 {tag}: volatility-timed exposure ({name})", color=T.INK)
    ax[1].plot(t, h[:, 0], color=T.SERIES[0], lw=.7)
    ax[1].set_ylabel("exposure w", color=T.INK)
    ax[1].set_ylim(-.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"equity_{tag}.png"), dpi=110, facecolor=T.PAPER)
    plt.close(fig)

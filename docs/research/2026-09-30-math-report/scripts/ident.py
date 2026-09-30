import json, glob, os
import numpy as np
import pandas as pd

BASE = "D:/neural_trade/runs/scenarios/long_360d_stab"
runs = sorted(d for d in glob.glob(BASE + "/2026*") if os.path.isdir(d))
names = None
rows = {}
for d in runs:
    rid = os.path.basename(d)
    tag = rid.split("default__")[1]
    init = json.load(open(d + "/period_init.json"))["configured"]
    st = json.load(open(d + "/status.json"))
    h = pd.read_csv(d + "/indicator_params_history.csv")
    names = list(init)
    we = st["weights_epoch"]
    served = h[h.epoch == we - 1].iloc[0]; assert np.allclose([served[k] for k in names], [h.iloc[-1][k] for k in names], rtol=1e-4), rid
    last = h.iloc[-2]
    rows[tag] = dict(init=init, served={k: served[k] for k in names}, last={k: last[k] for k in names},
                     ep0={k: h.iloc[0][k] for k in names}, we=we, nep=len(h),
                     traj={k: h[k].tolist() for k in names},
                     lr_ind=h["log_lr_indicator_used"].tolist(), lr=h["log_lr_used"].tolist(),
                     gn=h["log_grad_global_norm"].tolist())
tags = list(rows)
print("runs", tags)
print("served epoch / epochs:", [(t, rows[t]["we"], rows[t]["nep"]) for t in tags])
print("lr_ind per run (first,last):", [(t, rows[t]["lr_ind"][0], rows[t]["lr_ind"][-1]) for t in tags])
out = []
for k in names:
    p0 = rows[tags[0]]["init"][k]
    sv = np.array([rows[t]["served"][k] for t in tags])
    ls = np.array([rows[t]["last"][k] for t in tags])
    e0 = np.array([rows[t]["ep0"][k] for t in tags])
    sgn = np.sign(sv - p0)
    agree = int(max((sgn > 0).sum(), (sgn < 0).sum()))
    # within-fold agreement
    f3 = sv[:3]; f2 = sv[3:]
    # max abs relative move along trajectory
    maxmove = max(max(abs(np.array(rows[t]["traj"][k]) - p0)) / p0 for t in tags)
    out.append(dict(param=k, init=p0,
                    served=" / ".join(f"{v:.2f}" for v in sv),
                    mean=sv.mean(), std=sv.std(ddof=1), rel_move=(sv.mean() - p0) / p0,
                    cv=sv.std(ddof=1) / sv.mean(), agree=f"{agree}/6 {'+' if (sgn>0).sum()>=(sgn<0).sum() else '-'}",
                    mean_f3=f3.mean(), mean_f2=f2.mean(),
                    last_mean=ls.mean(), last_std=ls.std(ddof=1),
                    ep0_mean=e0.mean(), maxrel=maxmove,
                    at_floor=int((sv <= 2.05).sum()), at_ceiling=int((sv >= 59.5).sum())))
df = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.round(3).to_string(index=False))
df.to_csv("D:/nt_math_scratch/ident.csv", index=False)
# t-stat of the mean move across 6 runs
df["t"] = (df["mean"] - df["init"]) / (df["std"] / np.sqrt(6))
print(df[["param", "init", "mean", "std", "t"]].round(2).to_string(index=False))
# trajectories for the served epoch-by-epoch of one param for the biggest movers
for k in ["macd_1_fast", "macd_1_slow", "ma_period_0", "bb_period_0"]:
    for t in tags:
        print(k, t, [round(v, 2) for v in rows[t]["traj"][k]])
# pairwise correlation of the served move vectors between runs
M = np.array([[ (rows[t]["served"][k] - rows[t]["init"][k]) / rows[t]["init"][k] for k in names] for t in tags])
print("corr of relative-move vectors between runs:")
print(pd.DataFrame(np.corrcoef(M), index=tags, columns=tags).round(2))

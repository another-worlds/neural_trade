"""Score volatility forecasts (mean 1-min high-low range over next h minutes) vs classical baselines.

Baselines (all causal): EWMA of range, rolling means (15/60/240 min), previous-day same-minute
mean (intraday seasonality), and a HAR regression on log past means fitted on pre-holdout data only.
Metrics: mean |log(pred/real)| (lower better), Spearman rank corr, QLIKE on squared range.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from common import HORIZONS, RESULTS, STRIDE, load, origins  # noqa: E402

OUT = RESULTS + "_vol"
df, _ = load()
rng = np.log(df["high"].to_numpy(np.float64) / df["low"].to_numpy(np.float64)) * 1e4
n = len(rng)
cs = np.concatenate([[0.0], np.cumsum(rng)])


def past_mean(t, w):          # mean of rng[t-w+1 .. t]
    return (cs[t + 1] - cs[t + 1 - w]) / w


def fut_mean(t, h):           # mean of rng[t+1 .. t+h]
    return (cs[t + 1 + h] - cs[t + 1]) / h


ewma = np.zeros(n)
ewma[0] = rng[0]
for i in range(1, n):
    ewma[i] = 0.94 * ewma[i - 1] + 0.06 * rng[i]

# HAR fitted on pre-holdout origins only
train_org = np.arange(1440, origins(n)[0] - max(HORIZONS), STRIDE)   # strictly before the first test origin


def har_x(t):
    return np.column_stack([np.ones(len(t))] + [np.log(past_mean(t, w) + 1e-3) for w in (1, 5, 15, 60, 240, 1440)])


har_coef = {h: np.linalg.lstsq(har_x(train_org), np.log(fut_mean(train_org, h) + 1e-3), rcond=None)[0]
            for h in HORIZONS}
# log-OLS predicts the geometric mean; scale by the train-set mean of exp(residual) (smearing estimator)
har_smear = {h: float(np.mean(np.exp(np.log(fut_mean(train_org, h) + 1e-3) - har_x(train_org) @ har_coef[h])))
             for h in HORIZONS}
rs = np.random.default_rng(0)


def paired_ci(d, block=20, nboot=1000):
    n = len(d)
    st = rs.integers(0, n - block + 1, size=(nboot, int(np.ceil(n / block))))
    idx = (st[:, :, None] + np.arange(block)).reshape(nboot, -1)[:, :n]
    return [round(float(v), 4) for v in np.percentile(d[idx].mean(axis=1), [2.5, 97.5])]


def per_origin(pred, real):
    pred, real = np.maximum(pred, 1e-3), np.maximum(real, 1e-3)
    ratio = real ** 2 / pred ** 2
    return np.abs(np.log(pred / real)), ratio - np.log(ratio) - 1


def scores(pred, real):
    pred = np.maximum(pred, 1e-3)
    real = np.maximum(real, 1e-3)
    ratio = real ** 2 / pred ** 2
    return {"mean_abs_log_err": round(float(np.mean(np.abs(np.log(pred / real)))), 4),
            "spearman": round(float(spearmanr(pred, real).statistic), 4),
            "qlike": round(float(np.mean(ratio - np.log(ratio) - 1)), 4),
            "bias_log": round(float(np.mean(np.log(pred / real))), 4)}


rows = []
done = set()
BASE = {}
files = sorted(glob.glob(f"{OUT}/*.npz"))
for path in files:
    z = np.load(path)
    org = z["origins"]
    key = (len(org), int(org[0]))
    if key not in done:
        done.add(key)
        for hi, h in enumerate(HORIZONS):
            real = fut_mean(org, h)
            base = {"ewma_0.94": ewma[org], "rollmean_15": past_mean(org, 15), "rollmean_60": past_mean(org, 60),
                    "rollmean_240": past_mean(org, 240),
                    "prev_day_same_time": np.array([fut_mean(t - 1440, h) for t in org]),
                    "har_ols_train": np.exp(har_x(org) @ har_coef[h]),
                    "har_ols_train_smeared": np.exp(har_x(org) @ har_coef[h]) * har_smear[h]}
            BASE[(key, h)] = base
            for name, p in base.items():
                rows.append({"set": f"n{len(org)}", "model": name, "h": h, **scores(p, real)})
    for hi, h in enumerate(HORIZONS):
        real = fut_mean(org, h)
        block = max(5, int(round(60 / np.median(np.diff(org)))))
        for kk, suffix in [("pred", ""), ("pred_median", "_medianpath")]:
            if kk in z:
                row = {"set": f"n{len(org)}", "model": os.path.basename(path)[:-4] + suffix, "h": h,
                       **scores(z[kk][:, hi], real), "elapsed_s": round(float(z["elapsed"]), 1)}
                male_m, ql_m = per_origin(z[kk][:, hi], real)
                for ref, tag in (("ewma_0.94", "ewma"), ("har_ols_train_smeared", "harS"), ("har_ols_train", "har")):
                    male_r, ql_r = per_origin(BASE[(key, h)][ref], real)
                    row[f"qlike_vs_{tag}_pct"] = round(float(100 * (ql_m - ql_r).mean() / ql_r.mean()), 2)
                    row[f"qlike_vs_{tag}_ci95_pct"] = [round(100 * v / float(ql_r.mean()), 2) for v in paired_ci(ql_m - ql_r, block)]
                    row[f"male_vs_{tag}_pct"] = round(float(100 * (male_m - male_r).mean() / male_r.mean()), 2)
                    row[f"male_vs_{tag}_ci95_pct"] = [round(100 * v / float(male_r.mean()), 2) for v in paired_ci(male_m - male_r, block)]
                rows.append(row)

json.dump(rows, open(f"{OUT}/metrics_vol.json", "w"), indent=1)
cols = ["set", "model", "h", "mean_abs_log_err", "spearman", "qlike", "bias_log", "qlike_vs_harS_pct",
        "qlike_vs_harS_ci95_pct", "male_vs_harS_pct", "male_vs_harS_ci95_pct", "qlike_vs_ewma_pct", "qlike_vs_ewma_ci95_pct", "male_vs_har_pct", "male_vs_har_ci95_pct"]
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in sorted(rows, key=lambda r: (r["set"], r["h"], r["mean_abs_log_err"])):
    lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
open(f"{OUT}/metrics_vol.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

"""Noise of candidate A/B metrics, from the 84-run physics ablation (3 seeds x 2 periods x 14 conditions).

Caveat: the grid predates the served-epoch fix (D-011): every run was scored on its LAST epoch.
seed sigma: pooled SD across seeds within (condition, period).
paired SD: SD of (condition - all_off) differences over (seed, period) pairs, pooled over 13 conditions
(true term effects are included, so this is an upper-side estimate of pure noise; v1 found no VALUE).
Decomposition of the paired differences: within-period (seed) vs between-period (fold) variance.
Detectable non-inferiority margin at 80% power, one-sided alpha 0.05, true difference 0:
  margin = (t_.95 + t_.80)(n-1) * SD_d / sqrt(n)   (independent pairs; see the fold caveat below).
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("C:/Users/Step/Documents/neural_trade/runs/ablations/ablate_physics_v1-full/results.csv")
for h in ("h0", "h1", "h2"):
    df[f"{h}/variance/log_crps"] = np.log(df[f"{h}/variance/crps"])
metrics = ["h1/direction/auc", "h1/direction/mcc", "h1/variance/log_crps", "h1/variance/crpss",
           "h1/variance/pit_ks", "h1/variance/coverage90", "h1/delta/rmse", "h1/delta/skill_vs_zero",
           "backtest/sharpe_net", "backtest/total_return_net", "backtest/n_trades", "wall_s", "epochs_run"]
metrics = [m for m in metrics if m in df.columns]
print("columns with backtest/:", [c for c in df.columns if c.startswith("backtest/")][:30])
print(f"runs: {len(df)}, conditions: {df.condition.nunique()}, seeds {sorted(df.seed.unique())}, "
      f"periods {sorted(df.period.unique())}")

base = df[df.condition == "all_off"].set_index(["seed", "period"])
rows = []
for m in metrics:
    g = df.groupby(["condition", "period"])[m]
    seed_sd = float(np.sqrt(np.nanmean(g.var(ddof=1))))
    diffs, within, between = [], [], []
    for c, sub in df[df.condition != "all_off"].groupby("condition"):
        s = sub.set_index(["seed", "period"])[m]
        d = (s - base[m].reindex(s.index)).dropna()
        diffs += list(d.values)
        dd = d.reset_index()
        per = dd.groupby("period")[m]
        within.append(np.nanmean(per.var(ddof=1)))
        between.append(np.var(per.mean().values, ddof=1))
    diffs = np.array(diffs)
    sd_d = float(np.std(diffs - np.repeat([np.mean(diffs)], len(diffs)), ddof=1))
    w = float(np.sqrt(np.nanmean(within)))
    b = float(np.sqrt(np.nanmean(between)))
    row = {"metric": m, "mean": float(df[m].mean()), "seed_sd": seed_sd, "paired_sd": sd_d,
           "within_period_sd_of_diff": w, "sd_of_period_mean_diff": b}
    for n in (6, 9, 12, 18, 24):
        k = stats.t.ppf(0.95, n - 1) + stats.t.ppf(0.80, n - 1)
        row[f"margin80_n{n}"] = k * sd_d / np.sqrt(n)
    rows.append(row)
out = pd.DataFrame(rows).set_index("metric")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(out.round(4).to_string())

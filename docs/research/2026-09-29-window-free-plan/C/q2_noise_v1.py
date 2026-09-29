"""Q2: measured noise of the A/B metrics on the v1 physics grid (84 runs: 14 conditions x 3 seeds x
2 folds; P1 = fold -2 = a DEV fold, P2 = fold -1 = the TEST fold), plus arm A's dev-fold edge.

Reads (read-only) runs/ablations/ablate_physics_v1-full/results.csv and each run's
eval_report_test.json (baselines) and metrics.jsonl (epoch_seconds). CPU only, numpy/pandas.

Variance model per metric y, per fold p:   y[c,s,p] = mu[c,p] + beta[s,p] + e[c,s,p]
  e     : seed noise of one run (residual of a condition x seed two-way layout, per fold)
  beta  : a seed effect shared by all conditions at the same seed (the same graph and init here;
          two different architectures would not share it: 'nominal' seed pairing)
  mu    : condition x fold means; their interaction is the arm x fold effect (effect heterogeneity).
Paired SD of one (seed, fold) pair of two arms:
  shared-seed pairing  sd_pair = sqrt(2 s_e^2)
  nominal pairing      sd_pair = sqrt(2 (s_e^2 + s_beta^2))
  across folds (adds the arm x fold interaction) sd_pair_folds = sqrt(sd_pair^2 + 2 s_cp^2)

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q2_noise_v1.py
Writes q2_noise_v1.json and q2_v1_per_run.csv.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

GRID = Path("D:/neural_trade/runs/ablations/ablate_physics_v1-full")
OUT = Path(__file__).with_name("q2_noise_v1.json")
PER_RUN = Path(__file__).with_name("q2_v1_per_run.csv")


def per_run_table():
    df = pd.read_csv(GRID / "results.csv")
    rows = []
    for _, r in df.iterrows():
        rd = GRID / "runs" / r["run_id"]
        ev = json.loads((rd / "eval_report_test.json").read_text(encoding="utf-8"))
        mets = [json.loads(line) for line in (rd / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        ep = np.array([m["epoch_seconds"] for m in mets if m.get("epoch", 0) >= 1 and "epoch_seconds" in m], float)
        val = np.array([m["val_loss"] for m in mets], float)
        vcrps = np.array([m["val_crps_h0"] + m["val_crps_h1"] + m["val_crps_h2"] for m in mets], float)
        row = {"run_id": r["run_id"], "condition": r["condition"], "seed": int(r["seed"]), "period": r["period"],
               "epochs_run": int(r["epochs_run"]), "wall_s": float(r["wall_s"])}
        for h in ("h0", "h1", "h2"):
            m = ev["model"]["horizons"][h]
            cv = ev["baselines"]["const_var"]["horizons"][h]["variance"]["crps"]
            lr_auc = ev["baselines"]["logreg_lags"]["horizons"][h]["direction"]["auc"]
            row[f"logcrps_{h}"] = math.log(m["variance"]["crps"])
            row[f"log_edge_{h}"] = math.log(cv) - math.log(m["variance"]["crps"])   # >0: beats const_var
            row[f"crpss_{h}"] = m["variance"]["crpss"]
            row[f"crpss_recomputed_{h}"] = 1.0 - m["variance"]["crps"] / cv
            row[f"auc_{h}"] = m["direction"]["auc"]
            row[f"logreg_auc_{h}"] = lr_auc
            row[f"auc_minus_logreg_{h}"] = m["direction"]["auc"] - lr_auc
            row[f"cov90_{h}"] = m["variance"]["coverage90"]
            row[f"covdist_{h}"] = abs(m["variance"]["coverage90"] - 0.90)
            row[f"pit_ks_{h}"] = m["variance"]["pit_ks"]
        row["logcrps_mean"] = float(np.mean([row[f"logcrps_{h}"] for h in ("h0", "h1", "h2")]))
        row["log_edge_mean"] = float(np.mean([row[f"log_edge_{h}"] for h in ("h0", "h1", "h2")]))
        row["covdist_mean"] = float(np.mean([row[f"covdist_{h}"] for h in ("h0", "h1", "h2")]))
        bt = ev["backtest"]["summary"]
        row["sharpe_net"] = bt["sharpe_net"]
        row["sharpe_gross"] = bt["sharpe_gross"]
        row["n_trades"] = bt["n_trades"]
        row["median_epoch_s"] = float(np.median(ep))
        row["log_median_epoch_s"] = math.log(float(np.median(ep)))
        row["max_epoch_over_median"] = float(ep.max() / np.median(ep))
        row["n_epochs_over_1.5x_median"] = int((ep > 1.5 * np.median(ep)).sum())
        row["best_val_epoch_1based"] = int(np.argmin(val)) + 1
        row["best_valcrps_epoch_1based"] = int(np.argmin(vcrps)) + 1
        rows.append(row)
    return pd.DataFrame(rows)


def components(t, metric):
    """Variance components of `metric` in the condition x seed (x fold) layout."""
    res = {}
    s_e2, s_b2, df_e = [], [], []
    cell_means = {}
    for p, g in t.groupby("period"):
        piv = g.pivot(index="condition", columns="seed", values=metric)   # 14 x 3
        piv = piv.dropna()
        C, S = piv.shape
        grand = piv.values.mean()
        rowm = piv.values.mean(axis=1, keepdims=True)
        colm = piv.values.mean(axis=0, keepdims=True)
        resid = piv.values - rowm - colm + grand
        ms_res = (resid ** 2).sum() / ((C - 1) * (S - 1))
        ms_seed = C * ((colm - grand) ** 2).sum() / (S - 1)
        sb2 = max(0.0, (ms_seed - ms_res) / C)
        # plain within-cell seed SD (ignores the shared seed effect): pooled over conditions
        within = piv.values - rowm
        s_within2 = (within ** 2).sum() / (C * (S - 1))
        res[p] = dict(s_e=math.sqrt(ms_res), s_beta=math.sqrt(sb2), s_seed_within_cell=math.sqrt(s_within2),
                      df_e=(C - 1) * (S - 1), n_runs=int(C * S), cond_means=piv.mean(axis=1).to_dict(),
                      mean=float(grand))
        s_e2.append(ms_res); s_b2.append(sb2); df_e.append((C - 1) * (S - 1))
        cell_means[p] = piv.mean(axis=1)
    # pooled over folds
    s_e_pool = math.sqrt(np.average(s_e2, weights=df_e))
    s_b_pool = math.sqrt(np.mean(s_b2))
    # condition x fold interaction from the cell means (2 folds x 14 conditions), minus seed noise
    cm = pd.concat(cell_means, axis=1).dropna()        # conditions x folds
    if cm.shape[1] == 2:
        g = cm.values
        inter = g - g.mean(axis=1, keepdims=True) - g.mean(axis=0, keepdims=True) + g.mean()
        ms_int = (inter ** 2).sum() / ((g.shape[0] - 1) * (g.shape[1] - 1))
        # each cell mean averages 3 seeds: its noise variance is (s_e^2 + s_beta^2... beta cancels in the
        # interaction contrast within a fold) -> s_e^2 / 3
        s_cp2 = max(0.0, ms_int - s_e_pool ** 2 / 3.0)
        fold_mean_diff = float(g[:, 1].mean() - g[:, 0].mean())
    else:
        ms_int, s_cp2, fold_mean_diff = float("nan"), float("nan"), float("nan")
    sd_pair_shared = math.sqrt(2 * s_e_pool ** 2)
    sd_pair_nominal = math.sqrt(2 * (s_e_pool ** 2 + s_b_pool ** 2))
    sd_pair_folds = math.sqrt(sd_pair_nominal ** 2 + 2 * s_cp2) if not math.isnan(s_cp2) else float("nan")
    res["pooled"] = dict(s_e=s_e_pool, s_beta=s_b_pool, s_cp=math.sqrt(s_cp2) if not math.isnan(s_cp2) else None,
                         ms_interaction=ms_int, sd_pair_shared_seed=sd_pair_shared,
                         sd_pair_nominal_seed=sd_pair_nominal, sd_pair_across_folds=sd_pair_folds,
                         fold_mean_P2_minus_P1=fold_mean_diff, n_runs=int(t[metric].notna().sum()))
    return res


def direct_pair_sd(t, metric):
    """SD of paired differences between every pair of conditions at the same (seed, fold) -- the
    first research's estimator (includes shared-seed pairing and the small real condition effects)."""
    out = {}
    for p, g in t.groupby("period"):
        piv = g.pivot(index="seed", columns="condition", values=metric)
        conds = list(piv.columns)
        sds = []
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                d = (piv[conds[j]] - piv[conds[i]]).dropna().values
                if len(d) >= 2:
                    sds.append(np.var(d, ddof=1))
        out[p] = math.sqrt(float(np.mean(sds)))
    return out


METRICS = (["logcrps_mean", "log_edge_mean", "covdist_mean", "logcrps_h0", "logcrps_h1", "logcrps_h2", "log_edge_h0", "log_edge_h1", "log_edge_h2",
            "crpss_h0", "crpss_h1", "crpss_h2", "auc_h0", "auc_h1", "auc_h2",
            "auc_minus_logreg_h0", "auc_minus_logreg_h1", "auc_minus_logreg_h2",
            "covdist_h0", "covdist_h1", "covdist_h2", "cov90_h1", "pit_ks_h1",
            "sharpe_net", "sharpe_gross", "n_trades", "log_median_epoch_s"])


def main():
    t = per_run_table()
    t.to_csv(PER_RUN, index=False)
    out = {"n_runs": int(len(t)), "folds": {"P1": "fold -2 (dev)", "P2": "fold -1 (test)"},
           "caveats": ["v1 grid: commit 6dec27a, LAST-epoch weights (before D-011), lambdas calibrated once and frozen",
                       "14 conditions differ only in physics-term weights: arm differences are small, so the "
                       "arm x fold interaction is a lower bound for a larger change (series vs window memory)",
                       "P2 is the test fold: its numbers are used only to size noise, never to choose"],
           "metrics": {}}
    for m in METRICS:
        out["metrics"][m] = {"components": components(t, m), "direct_pair_sd": direct_pair_sd(t, m)}
    # arm A's edge on the DEV fold (P1) and, for sizing only, on the test fold (P2)
    edge = {}
    for p, g in t.groupby("period"):
        e = {}
        for h in ("h0", "h1", "h2"):
            e[f"crpss_{h}_all_runs_mean"] = float(g[f"crpss_{h}"].mean())
            e[f"log_edge_{h}_all_runs_mean"] = float(g[f"log_edge_{h}"].mean())
            e[f"log_edge_{h}_min_run"] = float(g[f"log_edge_{h}"].min())
            e[f"runs_beating_const_var_{h}"] = int((g[f"log_edge_{h}"] > 0).sum())
            gv = g[g.condition == "without:LAMBDA_VAC"]
            e[f"log_edge_{h}_today_like(without:LAMBDA_VAC)"] = float(gv[f"log_edge_{h}"].mean())
            e[f"auc_{h}_mean"] = float(g[f"auc_{h}"].mean())
            e[f"logreg_auc_{h}"] = float(g[f"logreg_auc_{h}"].mean())
            e[f"cov90_{h}_mean"] = float(g[f"cov90_{h}"].mean())
            e[f"cov90_{h}_range"] = [float(g[f"cov90_{h}"].min()), float(g[f"cov90_{h}"].max())]
        e["log_edge_mean_all_runs_mean"] = float(g["log_edge_mean"].mean())
        e["log_edge_mean_today_like(without:LAMBDA_VAC)"] = float(g[g.condition == "without:LAMBDA_VAC"]["log_edge_mean"].mean())
        e["runs_beating_const_var_all_horizons"] = int(((g["log_edge_h0"] > 0) & (g["log_edge_h1"] > 0) & (g["log_edge_h2"] > 0)).sum())
        e["runs"] = int(len(g))
        e["sharpe_net_mean"] = float(g["sharpe_net"].mean())
        e["n_trades_mean"] = float(g["n_trades"].mean())
        e["epochs_run_mean"] = float(g["epochs_run"].mean())
        e["best_val_epoch_1based_median"] = float(g["best_val_epoch_1based"].median())
        e["best_val_epoch_at_last_epoch_share"] = float((g["best_val_epoch_1based"] == g["epochs_run"]).mean())
        e["median_epoch_s_median"] = float(g["median_epoch_s"].median())
        e["runs_with_an_epoch_over_1.5x_median"] = int((g["n_epochs_over_1.5x_median"] > 0).sum())
        edge[p] = e
    out["arm_A_edge"] = edge
    # timing: robust spread of log median epoch time within cells
    lt = t.groupby(["condition", "period"])["log_median_epoch_s"]
    out["timing"] = {
        "within_cell_sd_log_median_epoch": float(np.sqrt((lt.transform(lambda x: x - x.mean()) ** 2).sum() / (len(t) - lt.ngroups))),
        "within_cell_robust_sd_log_median_epoch(1.4826 MAD)": float(1.4826 * np.median(np.abs(lt.transform(lambda x: x - x.median())))),
        "per_run_max_epoch_over_median_quantiles": {q: float(t["max_epoch_over_median"].quantile(q)) for q in (0.5, 0.9, 1.0)},
        "runs_with_any_epoch_over_1.5x_median": int((t["n_epochs_over_1.5x_median"] > 0).sum()),
    }
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    # summary print
    print(f"{'metric':<22} {'s_e':>8} {'s_beta':>8} {'s_cp':>8} | {'sd_pair shared':>14} {'nominal':>9} {'+folds':>9} | direct P1/P2")
    for m in METRICS:
        c = out["metrics"][m]["components"]["pooled"]
        d = out["metrics"][m]["direct_pair_sd"]
        print(f"{m:<22} {c['s_e']:8.4f} {c['s_beta']:8.4f} {(c['s_cp'] or 0):8.4f} | {c['sd_pair_shared_seed']:14.4f} "
              f"{c['sd_pair_nominal_seed']:9.4f} {c['sd_pair_across_folds']:9.4f} | {d.get('P1', float('nan')):.4f} / {d.get('P2', float('nan')):.4f}")
    print(json.dumps(out["arm_A_edge"], indent=1))
    print(json.dumps(out["timing"], indent=1))


if __name__ == "__main__":
    main()

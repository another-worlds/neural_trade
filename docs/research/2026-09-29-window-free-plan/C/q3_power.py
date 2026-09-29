"""Q3: pair counts that power the A/B margins, from the measured noise (q2_noise_v1.json,
q2_block_scaling.json) and the margins stated as fractions of arm A's edge.

One-sided paired non-inferiority test at alpha = 0.05, power 0.80 at a true difference of 0:
  naive      : n i.i.d. (seed, fold) pairs, t-test with n-1 df, per-pair SD s_pair
  clustered  : F folds x S seeds, the fold means carry the arm x fold interaction (variance 2 s_cp^2)
               plus s_pair^2 / S; t-test on the F fold means (F - 1 df)
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q3_power.py
Writes q3_power.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "q3_power.json"
ALPHA, POWER = 0.05, 0.80


def power_t(delta, se, df):
    """P(reject H0: mean <= -delta) when the true mean is 0: T = (mean + delta)/se ~ nct(df, delta/se)."""
    crit = stats.t.ppf(1 - ALPHA, df)
    return 1 - stats.nct.cdf(crit, df, delta / se)


def n_naive(delta, s_pair, n_max=2000):
    for n in range(2, n_max):
        if power_t(delta, s_pair / math.sqrt(n), n - 1) >= POWER:
            return n
    return None


def seeds_clustered(delta, s_pair, s_cp, F, s_max=200):
    """Seeds per fold so that the clustered test on F fold means has 80% power (None: impossible)."""
    for S in range(1, s_max):
        se = math.sqrt(2 * s_cp ** 2 / F + s_pair ** 2 / (F * S))
        if power_t(delta, se, F - 1) >= POWER:
            return S
    return None


def naive_false_positive_rate(s_pair, s_cp, F, S):
    """Size of the naive i.i.d.-pairs test when the arm x fold interaction exists (true mean = -delta boundary)."""
    n = F * S
    se_true = math.sqrt(2 * s_cp ** 2 / F + s_pair ** 2 / n)
    se_naive = math.sqrt((s_pair ** 2 + 2 * s_cp ** 2) / n)       # what the naive test estimates on average
    crit = stats.t.ppf(1 - ALPHA, n - 1)
    return float(1 - stats.norm.cdf(crit * se_naive / se_true))


def main():
    noise = json.loads((HERE / "q2_noise_v1.json").read_text(encoding="utf-8"))
    scal = json.loads((HERE / "q2_block_scaling.json").read_text(encoding="utf-8")) if (HERE / "q2_block_scaling.json").exists() else None
    nb = json.loads((HERE / "q2_notebook_runs.json").read_text(encoding="utf-8"))
    comp = lambda m: noise["metrics"][m]["components"]["pooled"]
    edge_dev = noise["arm_A_edge"]["P1"]
    edge_test_v1 = noise["arm_A_edge"]["P2"]
    latest = nb["20260924T182915Z-1aeff1c-dirty-af67ee43"]["TEST_block_for_sizing_only"]
    out = {"alpha": ALPHA, "power": POWER, "edges": {}, "designs": {}}

    # block-length inflation factor of the paired SD (5-day -> 1-day and 2-day evaluation blocks)
    infl = {}
    if scal:
        for h in ("h0", "h1", "h2", "mean"):
            s = scal["scaling"][h]
            infl[h] = {L: s[L]["sd_null_pairs(nominal seed)"] / s["7236"]["sd_null_pairs(nominal seed)"] for L in ("1440", "2880", "7236")}
    out["block_length_inflation_of_paired_sd"] = infl

    for h in ("mean", "h0", "h1", "h2"):
        latest_e = (sum(latest[k]["log_edge"] for k in ("h0", "h1", "h2")) / 3.0) if h == "mean" else latest[h]["log_edge"]
        e = {"dev_fold_-2_v1_all_runs": edge_dev[f"log_edge_{h}_all_runs_mean"],
             "dev_fold_-2_v1_today_like": edge_dev[f"log_edge_{h}_today_like(without:LAMBDA_VAC)"],
             "TEST_fold_-1_v1_sizing_only": edge_test_v1[f"log_edge_{h}_all_runs_mean"],
             "TEST_fold_-1_latest_run_sizing_only": latest_e}
        out["edges"][h] = e
        c = comp(f"logcrps_{h}")
        rows = []
        # per-pair SD options: the v1 grid's component estimates at 5-day blocks (shared / nominal seed
        # pairing), and the directly measured null-pair SD of the CPU re-prediction at 5, 2 and 1-day blocks.
        sd_opts = [("v1_grid_shared_seed_5d", c["sd_pair_shared_seed"], 7236),
                   ("v1_grid_nominal_seed_5d", c["sd_pair_nominal_seed"], 7236)]
        if scal:
            for L in ("7236", "2880", "1440"):
                sd_opts.append((f"measured_null_pairs_{int(L) // 1440}d", scal["scaling"][h][L]["sd_null_pairs(nominal seed)"], int(L)))
            sd_opts.append(("v1_nominal_x_measured_inflation_1d", c["sd_pair_nominal_seed"] * infl[h]["1440"], 1440))
            sd_opts.append(("v1_nominal_x_measured_inflation_2d", c["sd_pair_nominal_seed"] * infl[h]["2880"], 2880))
        for edge_name, E in e.items():
            for frac in (1 / 3, 1 / 2):
                delta = E * frac
                for s_name, sp, L in sd_opts:
                    row = {"edge": edge_name, "edge_value": E, "fraction": round(frac, 3), "margin": delta,
                           "sd_source": s_name, "eval_block_anchors": L, "s_pair": sp, "n_naive": n_naive(delta, sp)}
                    for s_cp_mult in (1.0, 2.0):
                        s_cp = (c["s_cp"] or 0.0) * s_cp_mult      # measured at 5-day blocks; not rescaled
                        row[f"s_cp_x{s_cp_mult:g}"] = s_cp
                        row[f"seeds_per_fold_clustered_s_cp_x{s_cp_mult:g}"] = {F: seeds_clustered(delta, sp, s_cp, F) for F in (3, 4, 6, 8, 10, 15)}
                    rows.append(row)
        out["designs"][f"logcrps_{h}"] = rows

    # other metrics: detectable margins at given pair counts (naive, nominal pairing)
    det = {}
    for m in ("covdist_h0", "covdist_h1", "covdist_h2", "auc_h1", "sharpe_net", "n_trades", "log_median_epoch_s"):
        c = comp(m)
        det[m] = {"s_pair_nominal": c["sd_pair_nominal_seed"], "s_cp": c["s_cp"],
                  "detectable_margin_80pct": {n: detectable(c["sd_pair_nominal_seed"], n) for n in (6, 12, 18, 24, 30)}}
    out["detectable_margins_naive"] = det

    # size of the naive test under the measured interaction
    c = comp("logcrps_h1")
    out["naive_test_size_with_interaction_logcrps_h1"] = {
        f"F{F}xS{S}": {f"s_cp_x{k:g}": naive_false_positive_rate(c["sd_pair_nominal_seed"], (c["s_cp"] or 0) * k, F, S) for k in (1, 2, 4)}
        for F, S in ((3, 5), (3, 7), (6, 3), (10, 3), (15, 2))}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("block-length inflation:", json.dumps(infl))
    print("edges (log CRPS vs const_var):", json.dumps(out["edges"], indent=1))
    for key in ("logcrps_h1", "logcrps_mean"):
      for r in out["designs"][key]:
        if abs(r["fraction"] - 1 / 3) < 1e-3 and r["edge"] in ("dev_fold_-2_v1_all_runs", "TEST_fold_-1_latest_run_sizing_only"):
            print(f"{key} {r['edge']:<36} {r['sd_source']:<36} margin {r['margin']:.4f} s_pair {r['s_pair']:.4f} "
                  f"n_naive {r['n_naive']:>4}  clustered S per fold (s_cp x1) {r['seeds_per_fold_clustered_s_cp_x1']}  (x2) {r['seeds_per_fold_clustered_s_cp_x2']}")
    print(json.dumps(out["detectable_margins_naive"], indent=1))
    print(json.dumps(out["naive_test_size_with_interaction_logcrps_h1"], indent=1))


def detectable(s_pair, n):
    """Smallest margin with 80% power at n pairs (naive paired t, true difference 0)."""
    lo, hi = 0.0, 50 * s_pair
    for _ in range(80):
        mid = (lo + hi) / 2
        if power_t(mid, s_pair / math.sqrt(n), n - 1) >= POWER:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    main()

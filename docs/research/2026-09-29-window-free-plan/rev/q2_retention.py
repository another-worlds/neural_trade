"""M5: retention judged per fold. r_f = d_f - E_A,f / 3 (d_f = paired horizon-mean log-CRPS loss of B vs A on
fold f's out-of-sample block, positive = B worse; E_A,f = A's horizon-mean log edge over constant variance on
that block, from the SAME A run). Non-inferior iff the one-sided 95% upper bound of mean(r_f) < 0.

Measured inputs: DEV fold -2 (P1) only, from C/q2_v1_per_run.csv (fold -1 not used):
  s_e, s_beta of horizon-mean log CRPS in the condition x seed layout of fold -2; A's edge on fold -2.
Empirical check: SD of r over the 14 x 13 ordered condition pairs at the same seed (shared-seed pairing) and at
different seeds (nominal), fold -2.
Not measurable on one dev fold (so scenarios, labelled Estimate): s_cp (arm x fold interaction of d),
s_E (between-fold SD of A's true edge), the 7-day-training inflation (C's x1.0-1.5), the judged folds' mean edge.
Model (one seed per fold):  r_f = Delta_f - E_f/3 + e_B - (2/3) e_A (+ beta/3 shared-seed),
  Var(r_f) = 13/9 s_e^2 + s_beta^2/9 (shared) or 13/9 (s_e^2 + s_beta^2) (nominal) + 2 s_cp^2 + s_E^2 / 9.
Fixed-delta design (the draft): d_f against delta = E_dev/3, Var(d_f) = 2 s_e^2 (+ 2 s_beta^2) + 2 s_cp^2.
Fold-clustered: S seeds per fold, the t-test on fold means (F - 1 df); seed part / S.
Monte Carlo check of size at the two retention boundaries (Delta_f = E_f/3 per fold; Delta = mean E/3).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q2_retention.py
Writes q2_retention.json.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, PER_RUN

ALPHA = 0.05


def components_P1(t, col):
    piv = t.pivot(index="condition", columns="seed", values=col)
    C, S = piv.shape
    g = piv.values.mean(); rm = piv.values.mean(1, keepdims=True); cm = piv.values.mean(0, keepdims=True)
    res = piv.values - rm - cm + g
    ms_res = (res ** 2).sum() / ((C - 1) * (S - 1))
    ms_seed = C * ((cm - g) ** 2).sum() / (S - 1)
    return math.sqrt(ms_res), math.sqrt(max(0.0, (ms_seed - ms_res) / C))


def power(effect, sd, F, df=None):
    df = F - 1 if df is None else df
    crit = stats.t.ppf(1 - ALPHA, df)
    return float(1 - stats.nct.cdf(crit, df, effect / (sd / math.sqrt(F))))


def F80(effect, sd, Fmax=500):
    for F in range(3, Fmax):
        if power(effect, sd, F) >= 0.8:
            return F
    return None


def main():
    t = pd.read_csv(PER_RUN)
    t = t[t.period == "P1"]
    s_e, s_b = components_P1(t, "logcrps_mean")
    E_dev = float(t.log_edge_mean.mean())
    out = {"source": "DEV fold -2 (P1) only", "measured": {"s_e": s_e, "s_beta": s_b, "E_dev_mean": E_dev,
                                                           "sd_E_run_within_fold": float(t.log_edge_mean.std(ddof=1)),
                                                           "delta_dev": E_dev / 3}}
    # empirical r on fold -2
    piv_L = t.pivot(index="condition", columns="seed", values="logcrps_mean")
    piv_E = t.pivot(index="condition", columns="seed", values="log_edge_mean")
    conds = list(piv_L.index)
    r_sh, r_nom, d_sh, d_nom = [], [], [], []
    for a, b in itertools.permutations(conds, 2):
        for s in (0, 1, 2):
            r_sh.append(piv_L.loc[b, s] - piv_L.loc[a, s] - piv_E.loc[a, s] / 3); d_sh.append(piv_L.loc[b, s] - piv_L.loc[a, s])
            for s2 in (0, 1, 2):
                if s2 != s:
                    r_nom.append(piv_L.loc[b, s2] - piv_L.loc[a, s] - piv_E.loc[a, s] / 3); d_nom.append(piv_L.loc[b, s2] - piv_L.loc[a, s])
    out["measured"]["empirical_sd_r_shared_seed"] = float(np.std(r_sh, ddof=1))
    out["measured"]["empirical_sd_r_nominal"] = float(np.std(r_nom, ddof=1))
    out["measured"]["empirical_sd_d_shared_seed"] = float(np.std(d_sh, ddof=1))
    out["measured"]["empirical_sd_d_nominal"] = float(np.std(d_nom, ddof=1))
    out["measured"]["model_sd_r_shared"] = math.sqrt(13 / 9 * s_e ** 2 + s_b ** 2 / 9)
    out["measured"]["model_sd_r_nominal"] = math.sqrt(13 / 9 * (s_e ** 2 + s_b ** 2))
    out["measured"]["model_sd_d_shared"] = math.sqrt(2 * s_e ** 2)
    out["measured"]["model_sd_d_nominal"] = math.sqrt(2 * (s_e ** 2 + s_b ** 2))

    rows = []
    for pairing in ("shared", "nominal"):
        for infl in (1.0, 1.25, 1.5):
            se, sb = s_e * infl, s_b * infl
            seed_r = 13 / 9 * se ** 2 + (sb ** 2 / 9 if pairing == "shared" else 13 / 9 * sb ** 2)
            seed_d = 2 * se ** 2 + (0 if pairing == "shared" else 2 * sb ** 2)
            for s_cp in (0.0, 0.0025, 0.005):
                for s_E in (0.0, 0.005, 0.010, 0.015):
                    for Ej in (0.040, 0.030, 0.020, 0.015):
                        sd_r = math.sqrt(seed_r + 2 * s_cp ** 2 + s_E ** 2 / 9)
                        sd_d = math.sqrt(seed_d + 2 * s_cp ** 2)
                        row = {"pairing": pairing, "inflation": infl, "s_cp": s_cp, "s_E": s_E, "E_judged": Ej,
                               "sd_r": sd_r, "sd_d": sd_d, "F80_per_fold": F80(Ej / 3, sd_r)}
                        for S in (2, 3):
                            sdS = math.sqrt(seed_r / S + 2 * s_cp ** 2 + s_E ** 2 / 9)
                            row[f"F80_clustered_S{S}"] = F80(Ej / 3, sdS)
                        # fixed delta from the dev folds (E_dev = .040 measured on fold -2)
                        delta = 0.040 / 3
                        row["fixed_F80_if_E_judged=E_dev"] = F80(delta, sd_d)
                        for F in (10, 16, 20):
                            se_d = sd_d / math.sqrt(F)
                            crit = stats.t.ppf(1 - ALPHA, F - 1)
                            # P(declare NI) for the fixed design when B truly keeps exactly 2/3 of the JUDGED folds' edge
                            row[f"fixed_size_vs_retention_F{F}"] = float(stats.nct.cdf((delta / se_d) - 0, F - 1, (delta - Ej / 3) / se_d) if False else
                                                                       1 - stats.nct.cdf(crit, F - 1, (delta - Ej / 3) / se_d))
                            row[f"fixed_power_d0_F{F}"] = power(delta, sd_d, F)
                            row[f"perfold_power_d0_F{F}"] = power(Ej / 3, sd_r, F)
                        rows.append(row)
    out["rows"] = rows

    # Monte Carlo: size of the per-fold test at the two boundaries, power at d = 0 (F = 16, a middle scenario)
    rng = np.random.default_rng(5)
    mc = {}
    for F in (10, 16, 20):
        for bound in ("proportional", "constant", "d0"):
            se, sb, s_cp, s_E, Ebar = s_e * 1.25, s_b * 1.25, 0.0025, 0.010, 0.030
            rej = 0; REPS = 20000
            for _ in range(REPS):
                Ef = rng.normal(Ebar, s_E, F)
                Delta = {"proportional": Ef / 3, "constant": np.full(F, Ebar / 3), "d0": np.zeros(F)}[bound]
                Delta = Delta + rng.normal(0, math.sqrt(2) * s_cp, F)
                beta = rng.normal(0, sb, F)
                eA = rng.normal(0, se, F); eB = rng.normal(0, se, F)
                d = Delta + eB - eA                           # shared seed: beta cancels in d
                EA = Ef - beta - eA                            # A's measured edge carries its run noise
                r = d - EA / 3
                ub = r.mean() + stats.t.ppf(0.95, F - 1) * r.std(ddof=1) / math.sqrt(F)
                rej += ub < 0
            mc[f"F{F}_{bound}"] = rej / REPS
    out["monte_carlo_perfold_P(NI)_(infl1.25,s_cp.0025,s_E.010,Ebar.030,shared)"] = mc
    (OUT / "q2_retention.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["measured"], indent=1))
    print(json.dumps(mc, indent=1))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    sel = df[(df.inflation == 1.25)]
    print(sel[["pairing", "s_cp", "s_E", "E_judged", "sd_r", "sd_d", "F80_per_fold", "F80_clustered_S2", "F80_clustered_S3",
               "fixed_F80_if_E_judged=E_dev", "fixed_size_vs_retention_F16", "perfold_power_d0_F16", "fixed_power_d0_F16"]].to_string(index=False))


if __name__ == "__main__":
    main()

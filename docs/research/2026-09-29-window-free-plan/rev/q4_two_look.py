"""Should-fix 8: the INCONCLUSIVE re-run as a pre-registered two-look group-sequential design on the per-fold
retention statistic r_f (q2_retention.py), against the draft's unspecified re-run.

r_f ~ N(mu, sd_r), one per judgement fold, independent across folds. Look 1 after F folds, look 2 after 2F
(the F new folds pooled with the first F). t-statistic T_k = mean / (s / sqrt(n_k)).
NI (ADOPT on P1) at look k iff T_k < -c_k; breach (REJECT) iff T_k > +c_k (non-binding for the size of NI);
otherwise continue (look 1) or INCONCLUSIVE (look 2, then the owner).
Designs:
  naive_pooled   c_1 = c_2 = t_.95 (each look at .05, pooled): the draft read one way
  standalone     look 2 uses ONLY the 2F new folds at .05 (the draft read the other way)
  standalone_025 look 1 and the standalone re-run each at .025 (Bonferroni)
  obf            Lan-DeMets O'Brien-Fleming spending: alpha_1 = 2(1 - Phi(z_.975 / sqrt(.5))) = .0056 at look 1;
                 look 2's nominal level calibrated by simulation so that P(NI | mu = 0) = .05
  pocock         equal nominal levels at both looks, calibrated to .05 overall
Scenarios: sd_r = .0095 (shared-seed, inflation x1.25, s_cp .0025, s_E .010: q2_retention's middle scenario,
Estimate) and .0114 (nominal pairing, same); mean edge of the judged folds E in {.040 (dev fold -2), .030, .020};
mu = -E/3 (true d = 0), mu = 0 (retention boundary: size), mu = +E/3 (B keeps only 1/3 of the edge).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q4_two_look.py
Writes q4_two_look.json."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import optimize, stats

from common import OUT

REPS = 100000


def sim_T(F, mu, sd, rng):
    x = rng.normal(mu, sd, (REPS, 2 * F))
    def T(a):
        return a.mean(1) / (a.std(1, ddof=1) / math.sqrt(a.shape[1]))
    return T(x[:, :F]), T(x), T(x[:, F:])


def decide(T1, T2, T2s, c1, c2, design):
    ni1 = T1 < -c1; br1 = T1 > c1; cont = ~ni1 & ~br1
    T2u = T2s if design.startswith("standalone") else T2
    ni2 = cont & (T2u < -c2); br2 = cont & (T2u > c2)
    return {"NI_look1": float(ni1.mean()), "NI_total": float((ni1 | ni2).mean()),
            "breach_total": float((br1 | br2).mean()), "inconclusive_final": float((cont & ~ni2 & ~br2).mean()),
            "P(second look)": float(cont.mean()), "E[folds]": float((1 + cont.mean() * (2 if design.startswith("standalone") else 1)))}


def calibrate(F, design, rng):
    T1, T2, T2s = sim_T(F, 0.0, 1.0, rng)          # size does not depend on sd
    df1, df2 = F - 1, 2 * F - 1
    if design == "obf":
        a1 = 2 * (1 - stats.norm.cdf(stats.norm.ppf(0.975) / math.sqrt(0.5)))
        c1 = stats.t.ppf(1 - a1, df1)
        f = lambda a2: decide(T1, T2, T2s, c1, stats.t.ppf(1 - a2, df2), design)["NI_total"] - 0.05
        a2 = optimize.brentq(f, 0.001, 0.05)
        return c1, stats.t.ppf(1 - a2, df2), {"alpha1_nominal": a1, "alpha2_nominal": a2}
    if design == "pocock":
        f = lambda a: decide(T1, T2, T2s, stats.t.ppf(1 - a, df1), stats.t.ppf(1 - a, df2), design)["NI_total"] - 0.05
        a = optimize.brentq(f, 0.005, 0.05)
        return stats.t.ppf(1 - a, df1), stats.t.ppf(1 - a, df2), {"alpha_nominal_each": a}
    if design == "naive_pooled":
        return stats.t.ppf(0.95, df1), stats.t.ppf(0.95, df2), {"alpha_nominal_each": 0.05}
    if design == "standalone":
        return stats.t.ppf(0.95, df1), stats.t.ppf(0.95, df2), {"alpha_nominal_each": 0.05}
    if design == "standalone_025":
        return stats.t.ppf(0.975, df1), stats.t.ppf(0.975, df2), {"alpha_nominal_each": 0.025}


def main():
    rng = np.random.default_rng(11)
    out = {"reps": REPS, "designs": {}}
    for F in (6, 8, 10, 12):
        for design in ("naive_pooled", "standalone", "standalone_025", "pocock", "obf"):
            c1, c2, info = calibrate(F, design, rng)
            d = {"c1": c1, "c2": c2, **info, "scenarios": {}}
            for sd in (0.0095, 0.0114):
                for E in (0.040, 0.030, 0.020):
                    for lab, mu in (("d0", -E / 3), ("boundary", 0.0), ("keeps_1/3", E / 3)):
                        T1, T2, T2s = sim_T(F, mu, sd, rng)
                        d["scenarios"][f"sd{sd}|E{E}|{lab}"] = decide(T1, T2, T2s, c1, c2, design)
            out["designs"][f"F{F}|{design}"] = d
            s = d["scenarios"]
            print(f"F{F:<3}{design:<15} c1 {c1:5.2f} c2 {c2:5.2f} size {s['sd0.0095|E0.03|boundary']['NI_total']:.3f} | "
                  + " | ".join(f"E{E}: pow {s[f'sd0.0095|E{E}|d0']['NI_total']:.3f} (L1 {s[f'sd0.0095|E{E}|d0']['NI_look1']:.3f}) inc {s[f'sd0.0095|E{E}|d0']['inconclusive_final']:.3f} Ef {s[f'sd0.0095|E{E}|d0']['E[folds]']:.2f} brch1/3 {s[f'sd0.0095|E{E}|keeps_1/3']['breach_total']:.2f}"
                               for E in (0.04, 0.03, 0.02)))
    (OUT / "q4_two_look.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

"""M1 part 1: how often does an EQUIVALENT run (a condition-mate: same physics weights, another seed) fail
to reach a quality level built from arm A's runs, per candidate rule, on the 42 v1 DEV fold -2 curves.

Rules (R = the level on J~ = smoothed val CRPS; E = first epoch with J~ <= R):
  old        R = mean(min J~_A) + 0.05 mean(span_A)                         (the draft)
  exp_d      R = mean(min J~_A) x exp(delta), delta = dev-fold -2 edge / 3   (the NI margin)
  span_p     R = mean(J~_A(1)) - p mean(span_A), p = .5 / .65 / .8           (p of A's improvement span)
  self_p     each run's own level J~(1) - p span_own (never misses; quality judged separately)
  served     E = the served (best val_loss) epoch; quality judged separately by the NI guard-rail
A = the other seed(s) of B's condition: 'loo2' (the 2 other seeds, as the review's r3), 'pair1' (one other
seed, as A/B-2's per-fold level from a single A run: 84 ordered pairs).
Censoring: the v1 runs stop at 20 epochs. A B run that has not reached R by its last epoch is a MISS if early
stopping fired (converged without reaching), CENSORED if it was capped at 20 (unknown). Both are reported.
Also: a two-way (condition + seed) ANOVA of ln min J~ and ln E to test condition effects (whether the 42 runs
can be pooled as exchangeable 'equivalent' runs in the probe simulation).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_rules.py
Writes q1_rules.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from common import OUT, load_dev_runs, reach_epoch

recs = load_dev_runs()
DELTA_DEV = float(np.mean([r["log_edge_mean"] for r in recs])) / 3.0      # fold -2, 42 runs, last-epoch weights

RULES = ["old", "exp_d", "span_0.5", "span_0.65", "span_0.8", "self_0.5", "self_0.65", "self_0.8", "served"]


def level(rule, A):
    mn = np.mean([a["min"] for a in A]); sp = np.mean([a["span"] for a in A]); j1 = np.mean([a["J1"] for a in A])
    if rule == "old":
        return mn + 0.05 * sp
    if rule == "exp_d":
        return mn * np.exp(DELTA_DEV)
    if rule.startswith("span_"):
        return j1 - float(rule.split("_")[1]) * sp
    raise ValueError(rule)


def E_of(rule, B, A=None, interp=False):
    if rule == "served":
        return float(B["served"])
    if rule.startswith("self_"):
        p = float(rule.split("_")[1])
        return reach_epoch(B["Js"], B["J1"] - p * B["span"], interp)
    return reach_epoch(B["Js"], level(rule, A), interp)


def two_way(y, cond, seed):
    conds = sorted(set(cond)); seeds = sorted(set(seed))
    M = np.full((len(conds), len(seeds)), np.nan)
    for v, c, s in zip(y, cond, seed):
        M[conds.index(c), seeds.index(s)] = v
    C, S = M.shape
    g = M.mean(); rm = M.mean(1, keepdims=True); cm = M.mean(0, keepdims=True)
    res = M - rm - cm + g
    ms_res = (res ** 2).sum() / ((C - 1) * (S - 1))
    ms_c = S * ((rm - g) ** 2).sum() / (C - 1)
    ms_s = C * ((cm - g) ** 2).sum() / (S - 1)
    return {"F_condition": ms_c / ms_res, "p_condition": float(1 - stats.f.cdf(ms_c / ms_res, C - 1, (C - 1) * (S - 1))),
            "F_seed": ms_s / ms_res, "p_seed": float(1 - stats.f.cdf(ms_s / ms_res, S - 1, (C - 1) * (S - 1))),
            "s_resid": float(np.sqrt(ms_res)), "sd_within_condition": float(np.sqrt(((M - rm) ** 2).sum() / (C * (S - 1))))}


def main():
    out = {"source": "42 v1 runs, DEV fold -2 (P1) only; fold -1 not read", "delta_dev": DELTA_DEV}
    conds = sorted({r["cond"] for r in recs})
    lnmin = [np.log(r["min"]) for r in recs]
    lnspan = [np.log(r["J1"] / r["min"]) for r in recs]
    out["descriptives"] = {
        "ln_span_ln(J1/minJ)": {"median": float(np.median(lnspan)), "min": float(np.min(lnspan)), "max": float(np.max(lnspan))},
        "delta_over_ln_span_median": float(DELTA_DEV / np.median(lnspan)),
        "anova_ln_min": two_way(lnmin, [r["cond"] for r in recs], [r["seed"] for r in recs]),
        "served_epoch_counts": {int(k): int(v) for k, v in zip(*np.unique([r["served"] for r in recs], return_counts=True))},
        "early_stopped_runs": int(sum(r["early_stopped"] for r in recs)),
        "early_stopped_by_seed": {s: int(sum(r["early_stopped"] for r in recs if r["seed"] == s)) for s in (0, 1, 2)},
        "min_Js_epoch_in_last3_share": float(np.mean([int(np.argmin(r["Js"])) + 1 >= r["n"] - 2 for r in recs])),
        "late_gain_min20_vs_min14_over_span_median": float(np.median([(r["Js"][:14].min() - r["min"]) / r["span"] for r in recs])),
    }
    # per-rule single-run reach epochs against the run's own condition level (loo2 and pair1)
    rules_out = {}
    for rule in RULES:
        for interp in (False, True):
            key = f"{rule}{'_interp' if interp else ''}"
            loo = {"reached": 0, "miss_converged": 0, "censored_capped": 0}
            pair = {"reached": 0, "miss_converged": 0, "censored_capped": 0}
            lnE_by_cond, pair_l = {}, []
            Eself = []
            for c in conds:
                g = [r for r in recs if r["cond"] == c]
                for i, B in enumerate(g):
                    others = [g[j] for j in range(len(g)) if j != i]
                    E = E_of(rule, B, others, interp)
                    lab = "reached" if np.isfinite(E) else ("miss_converged" if B["early_stopped"] else "censored_capped")
                    loo[lab] += 1
                    Es = E_of(rule, B, [B], interp) if rule not in ("served",) and not rule.startswith("self_") else E
                    Eself.append(Es)
                    for A in others:                               # ordered pairs: A single run, B the other
                        EB = E_of(rule, B, [A], interp)
                        EA = E_of(rule, A, [A], interp)
                        labp = "reached" if np.isfinite(EB) else ("miss_converged" if B["early_stopped"] else "censored_capped")
                        pair[labp] += 1
                        if np.isfinite(EB) and np.isfinite(EA):
                            pair_l.append(np.log(EA / EB))
                    if np.isfinite(Es):
                        lnE_by_cond.setdefault(c, []).append(np.log(Es))
            # within-condition SD of ln E (each run against its own-condition level incl. itself: 'self' levels)
            resid = [x - np.mean(v) for v in lnE_by_cond.values() if len(v) > 1 for x in v]
            dof = sum(len(v) - 1 for v in lnE_by_cond.values() if len(v) > 1)
            rules_out[key] = {
                "loo2_single_B": loo, "loo2_miss_or_censored_share": (loo["miss_converged"] + loo["censored_capped"]) / 42,
                "pair1_single_A_single_B(84)": pair, "pair1_miss_or_censored_share": (pair["miss_converged"] + pair["censored_capped"]) / 84,
                "sd_ln_E_within_condition(own-level)": float(np.sqrt(np.sum(np.square(resid)) / dof)) if dof else None,
                "sd_pair_ln(EA/EB)_reached_pairs": float(np.std(pair_l, ddof=1)) if len(pair_l) > 2 else None,
                "n_reached_pairs": len(pair_l),
                "median_E_own_level": float(np.median([e for e in Eself if np.isfinite(e)])),
                "own_level_unreached": int(sum(~np.isfinite(Eself))),
            }
            if rule in ("self_0.5", "self_0.65", "self_0.8", "served") and not interp:
                y = [np.log(E_of(rule, r, [r])) for r in recs]
                rules_out[key]["anova_ln_E"] = two_way(y, [r["cond"] for r in recs], [r["seed"] for r in recs])
    out["rules"] = rules_out
    (OUT / "q1_rules.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out["descriptives"], indent=1, default=float))
    print(f"delta_dev={DELTA_DEV:.4f}")
    print(f"{'rule':<18}{'loo2 miss/cens':>16}{'pair1 miss/cens':>18}{'sd lnE':>8}{'sd pair':>9}{'medE':>6}")
    for k, v in rules_out.items():
        l, p = v["loo2_single_B"], v["pair1_single_A_single_B(84)"]
        print(f"{k:<18}{l['miss_converged']:>7}/{l['censored_capped']:<8}{p['miss_converged']:>9}/{p['censored_capped']:<8}"
              f"{(v['sd_ln_E_within_condition(own-level)'] or 0):8.3f}{(v['sd_pair_ln(EA/EB)_reached_pairs'] or 0):9.3f}{v['median_E_own_level']:6.1f}")


if __name__ == "__main__":
    main()

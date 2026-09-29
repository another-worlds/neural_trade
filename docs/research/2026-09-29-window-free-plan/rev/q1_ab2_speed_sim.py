"""M1 part 3: A/B-2's speed primary with TARGET MISSES INCLUDED, per reach rule, on the v1 DEV fold -2 curves.

Pair = (A run a, B run b), b an equivalent run (same learning per epoch) whose epochs are k times faster:
t_A = fit wall-clock of a to the end of its reach epoch (interpolated within the epoch when fractional),
t_B = wall-clock of b to its reach epoch / k. l = ln(t_A / t_B). The per-fold level comes from A's single run.
Rules: old (min_A + .05 span_A: the draft), exp_d (min_A exp(delta)), span_p (J~_A(1) - p span_A),
self_p (each run's own p of its own span; quality judged by the G1 guard-rail), served (time to the served,
best-val_loss epoch; censored at the 20-epoch cap in this data).
Misses: b never reaches within its observed curve: early-stopped -> t_B = inf (l = -inf);
capped at 20 -> 'censored': spec handling = the pair is INCONCLUSIVE and dropped (verdict INCONCLUSIVE if more
than 1/4 of pairs are); 'pess' = counted as t_B = inf.
Test: one-sided Wilcoxon signed-rank on l - ln 2 at alpha .05 (= HL lower bound >= ln 2). Pair populations:
'mates' = ordered condition-mate pairs (84); 'pooled' = any two different runs (42 x 41).
Only one fold exists: fold-to-fold variation of the curves is NOT in these numbers.
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_ab2_speed_sim.py
Writes q1_ab2_speed_sim.json.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats

from common import OUT, load_dev_runs, reach_epoch

recs = load_dev_runs()
DELTA_DEV = float(np.mean([r["log_edge_mean"] for r in recs])) / 3.0
REPS = 4000
RULES = ["old", "exp_d", "span_0.5", "span_0.65", "span_0.8", "self_0.5", "self_0.65", "self_0.8", "served"]
LN2 = math.log(2)


def wall_at(run, E):
    """Wall-clock to fractional epoch E (1-based end-of-epoch stamps; linear within an epoch)."""
    w = run["wall"]
    if E <= 1:
        return w[0] * E
    k = int(math.floor(E)); f = E - k
    if k >= len(w):
        return w[-1]
    return w[k - 1] + f * (w[k] - w[k - 1])


def lvl(rule, A):
    if rule == "old":
        return A["min"] + 0.05 * A["span"]
    if rule == "exp_d":
        return A["min"] * math.exp(DELTA_DEV)
    p = float(rule.split("_")[1])
    return A["J1"] - p * A["span"]


def E_of(rule, run, A):
    if rule == "served":
        return float(run["served"])
    if rule.startswith("self_"):
        p = float(rule.split("_")[1])
        return reach_epoch(run["Js"], run["J1"] - p * run["span"], interp=True)
    return reach_epoch(run["Js"], lvl(rule, A), interp=True)


def pair_value(rule, a, b):
    """(l at k = 1, status) with status in ok / miss / censored."""
    EA = E_of(rule, a, a)
    EB = E_of(rule, b, a)
    if not np.isfinite(EB):
        return None, ("miss" if b["early_stopped"] else "censored")
    return math.log(wall_at(a, EA) / wall_at(b, EB)), "ok"


_CRIT = {}


def crit_wplus(n, alpha=0.05):
    """Smallest c with P(W+ >= c) <= alpha under H0 (exact, no ties)."""
    if n not in _CRIT:
        m = n * (n + 1) // 2
        cnt = np.zeros(m + 1); cnt[0] = 1
        for r in range(1, n + 1):
            cnt[r:] = cnt[r:] + cnt[:-r].copy() if r <= m else cnt[r:]
        p = cnt / cnt.sum()
        tail = np.cumsum(p[::-1])[::-1]          # tail[c] = P(W+ >= c)
        _CRIT[n] = int(np.nonzero(tail <= alpha)[0][0])
    return _CRIT[n]


def wilcoxon_pass(x):
    """One-sided exact signed-rank test of median(x) > 0 at .05 (equivalently HL lower bound > 0).
    -inf values rank as the largest negatives; ties among them do not change W+."""
    x = np.asarray(x, float)
    x = x[x != 0]
    n = len(x)
    if n < 5:
        return False
    a = np.where(np.isfinite(x), np.abs(x), np.inf)
    ranks = np.empty(n); ranks[np.argsort(a, kind="stable")] = np.arange(1, n + 1)
    return ranks[x > 0].sum() >= crit_wplus(n)


def main():
    rng = np.random.default_rng(29)
    N = len(recs)
    pops = {"mates": [(i, j) for i in range(N) for j in range(N) if i != j and recs[i]["cond"] == recs[j]["cond"]],
            "pooled": [(i, j) for i in range(N) for j in range(N) if i != j]}
    out = {"source": "42 v1 runs, DEV fold -2 only", "reps": REPS, "delta_dev": DELTA_DEV, "rules": {}}
    for rule in RULES:
        ro = {}
        for pname, pop in pops.items():
            vals = [pair_value(rule, recs[i], recs[j]) for i, j in pop]
            l1 = np.array([v for v, s in vals if s == "ok"])
            st = [s for _, s in vals]
            pv = {"share_miss": st.count("miss") / len(st), "share_censored": st.count("censored") / len(st),
                  "sd_l_ok_pairs": float(np.std(l1, ddof=1)), "mean_l_ok_pairs(k=1)": float(np.mean(l1)),
                  "median_l_ok_pairs(k=1)": float(np.median(l1))}
            for F in (12, 20, 30):
                for k in (1.0, 2.0, 2.5, 3.0, 3.5, 4.0):
                    lk = math.log(k)
                    res = {"spec": 0, "pess": 0, "spec_inconclusive_verdict": 0}
                    for _ in range(REPS):
                        idx = rng.integers(0, len(pop), F)
                        xs, xp, ninc = [], [], 0
                        for t in idx:
                            v, s = vals[t]
                            if s == "ok":
                                xs.append(v + lk - LN2); xp.append(v + lk - LN2)
                            elif s == "miss":
                                xs.append(-np.inf); xp.append(-np.inf)
                            else:
                                ninc += 1; xp.append(-np.inf)
                        if ninc > F / 4:
                            res["spec_inconclusive_verdict"] += 1
                        elif wilcoxon_pass(xs):
                            res["spec"] += 1
                        res["pess"] += int(wilcoxon_pass(xp))
                    pv[f"F{F}_k{k:g}"] = {kk: round(vv / REPS, 4) for kk, vv in res.items()}
            ro[pname] = pv
        out["rules"][rule] = ro
        m = ro["mates"]
        print(f"{rule:<10} miss {m['share_miss']:.3f} cens {m['share_censored']:.3f} sd_l {m['sd_l_ok_pairs']:.3f} | F20 pass@k2 {m['F20_k2']['spec']:.3f}/{m['F20_k2']['pess']:.3f}"
              f" k2.5 {m['F20_k2.5']['spec']:.3f} k3 {m['F20_k3']['spec']:.3f} k3.5 {m['F20_k3.5']['spec']:.3f}/{m['F20_k3.5']['pess']:.3f} inc@3.5 {m['F20_k3.5']['spec_inconclusive_verdict']:.3f}"
              f" | pooled k2 {ro['pooled']['F20_k2']['spec']:.3f} k3.5 {ro['pooled']['F20_k3.5']['spec']:.3f}")
    (OUT / "q1_ab2_speed_sim.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

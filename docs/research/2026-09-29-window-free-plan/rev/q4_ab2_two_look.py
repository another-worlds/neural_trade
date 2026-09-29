"""Should-fix 8 for A/B-2: the speed primary (one-sided exact Wilcoxon of l - ln 2 > 0) as a two-look design,
F folds then F more (pooled), each look at the Pocock-type nominal level (calibrated by simulation to an overall
.05 at k = 2 on the pair populations) and at the OBF-type levels (.0056, then calibrated). Pair values l from
q1_ab2_speed_sim.py (DEV fold -2 curves, condition-mate pairs; no misses for the self and served rules).
Also: the single-look designs at 2F. Run: ...python q4_ab2_two_look.py. Writes q4_ab2_two_look.json."""
import json, math
import numpy as np
from scipy import optimize
import q1_ab2_speed_sim as S

recs = S.recs; N = len(recs)
pop = [(i, j) for i in range(N) for j in range(N) if i != j and recs[i]["cond"] == recs[j]["cond"]]
REPS = 6000
crit_cache = {}


def crit(n, a):
    key = (n, round(a, 6))
    if key not in crit_cache:
        m = n * (n + 1) // 2
        cnt = np.zeros(m + 1); cnt[0] = 1
        for r in range(1, n + 1):
            cnt[r:] = cnt[r:] + cnt[:-r].copy()
        tail = np.cumsum((cnt / cnt.sum())[::-1])[::-1]
        crit_cache[key] = int(np.nonzero(tail <= a)[0][0])
    return crit_cache[key]


def wplus(x):
    a = np.abs(x); ranks = np.empty(len(x)); ranks[np.argsort(a, kind="stable")] = np.arange(1, len(x) + 1)
    return ranks[x > 0].sum()


out = {}
rng = np.random.default_rng(3)
for rule in ("self_0.5", "self_0.65", "served"):
    vals = np.array([S.pair_value(rule, recs[i], recs[j])[0] for i, j in pop], float)
    for F in (10, 12):
        idx = rng.integers(0, len(vals), (REPS, 2 * F))
        base = vals[idx]
        def run(k, a1, a2):
            x = base + math.log(k) - math.log(2)
            p1 = np.array([wplus(r[:F]) >= crit(F, a1) for r in x])
            p2 = np.array([wplus(r) >= crit(2 * F, a2) for r in x])
            return float(p1.mean()), float((p1 | p2).mean())
        f = lambda a: run(2.0, a, a)[1] - 0.05
        # the exact test is discrete: search a grid instead of brentq
        grid = np.linspace(0.01, 0.05, 41)
        sizes = [run(2.0, a, a)[1] for a in grid]
        a_poc = float(max([a for a, s in zip(grid, sizes) if s <= 0.05], default=0.01))
        res = {"pocock_alpha_each": a_poc}
        for k in (1.0, 2.0, 2.5, 3.0, 3.5):
            l1, tot = run(k, a_poc, a_poc)
            single = float(np.mean([wplus(r) >= crit(2 * F, 0.05) for r in base + math.log(k) - math.log(2)]))
            res[f"k{k:g}"] = {"pass_look1": round(l1, 3), "pass_total_two_look": round(tot, 3), "single_look_2F_at_.05": round(single, 3)}
        out[f"{rule}|F{F}"] = res
        print(rule, F, json.dumps(res))
open("q4_ab2_two_look.json", "w").write(json.dumps(out, indent=1))

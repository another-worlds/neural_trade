"""M1 part 2b: the probe with the recommended self-referenced rule and B1024 capped at 4 x A's epochs (so each
run's own span is complete: B's cap covers the whole underlying 20-epoch curve at rho <= 4), plus the quality
guard: B's seed-mean ln min J~ must be within delta of A's (false guard-failure rate for equivalent B).
Reuses q1_probe_sim.py's functions. DEV fold -2 only.
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_probe_sim_bigcap.py
Writes q1_probe_sim_bigcap.json."""
import json, math
import numpy as np
import q1_probe_sim as P

REPS = 4000
out = {"reps": REPS, "table": {}}
rng = np.random.default_rng(7)
N = len(P.recs)
for S in (3, 4, 6):
    draws = [rng.choice(N, 2 * S, replace=False) for _ in range(REPS)]
    guard_fail = np.mean([np.mean([math.log(P.recs[i]["min"]) for i in d[S:]]) - np.mean([math.log(P.recs[i]["min"]) for i in d[:S]]) > P.DELTA_DEV for d in draws])
    out["table"][f"S{S}|guard_false_fail(rho=1)"] = float(guard_fail)
    for rule in ("self_0.5", "self_0.65", "self_0.8"):
        for interp in (False, True):
            res = {}
            for rho in (1.0, 1.5, 2.0, 3.0, 4.0):
                cnt = {"epoch": 0, "update": 0, "inconclusive": 0, "A_miss": 0}
                for d in draws:
                    EA = np.array([P.E_rule(rule, i, None, 1.0, P.EMAX, interp, "pess") for i in d[:S]])
                    EB = np.array([P.E_rule(rule, i, None, rho, 4 * 20, interp, "pess") for i in d[S:]])
                    cnt[P.classify(EA, EB, "geo")] += 1
                res[f"rho{rho:g}"] = {k: v / REPS for k, v in cnt.items()}
            key = f"S{S}|{rule}{'_interp' if interp else ''}|capB=4xA|geo"
            out["table"][key] = res
            print(f"{key:<40} ep|1 {res['rho1']['epoch']:.3f} up|1 {res['rho1']['update']:.3f} up|4 {res['rho4']['update']:.3f} ep|4 {res['rho4']['epoch']:.3f} ep|1.5 {res['rho1.5']['epoch']:.3f} up|3 {res['rho3']['update']:.3f} ep|2 {res['rho2']['epoch']:.3f} up|2 {res['rho2']['update']:.3f}")
    print(S, "guard false fail", guard_fail)
open("q1_probe_sim_bigcap.json", "w").write(json.dumps(out, indent=1))

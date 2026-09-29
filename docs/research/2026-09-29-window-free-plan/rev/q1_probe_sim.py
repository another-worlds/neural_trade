"""M1 part 2: probe (spec 0) classification error rates with TARGET MISSES INCLUDED, per reach rule.

Resampling on the 42 v1 DEV fold -2 curves (fold -1 not read). Arm A = S runs, arm B = S other runs, drawn
without replacement from the 42 (nominal seed pairing; conditions pooled: the condition effect on ln min J~ is
not significant, q1_rules.json, F = 1.58, p = .16; the seed effect is, and is kept as noise). B's curve is an
equivalent run slowed by rho in epochs: J~_B(e) = J~_run(e / rho) (linear interpolation; J~_run(1) held for
e / rho < 1). rho = 1: epoch-bound (B1024 learns as much per epoch); rho = 4: update-bound.
Caps: A as observed (20 epochs, the data limit); B = ceil(3 mean E_A) + 3 epochs (the draft).
Censoring beyond an underlying curve's last epoch (20, or earlier if it early-stopped):
  pess: never reached (E = inf);  opt: a capped (not early-stopped) curve reaches at underlying epoch n + 1.
Statistic: rho_hat = mean E_B / mean E_A ('mean'; any B miss -> inf) or exp(mean ln E_B - mean ln E_A)
('geo'; any B miss -> inf). An A miss (A cannot reach its own arm's level) -> 'A_miss' (inconclusive).
Epoch-bound iff rho_hat <= 1.5 (the wall-clock condition W_B < W_A is assumed to hold); update-bound iff >= 3.
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q1_probe_sim.py
Writes q1_probe_sim.json.
"""
from __future__ import annotations

import json
import math

import numpy as np

from common import OUT, load_dev_runs

recs = load_dev_runs()
DELTA_DEV = float(np.mean([r["log_edge_mean"] for r in recs])) / 3.0
REPS = 4000
RULES = ["old", "exp_d", "span_0.5", "span_0.65", "span_0.8", "self_0.5", "self_0.65", "self_0.8", "served"]


def level(rule, A):
    mn = np.mean([a["min"] for a in A]); sp = np.mean([a["span"] for a in A]); j1 = np.mean([a["J1"] for a in A])
    if rule == "old":
        return mn + 0.05 * sp
    if rule == "exp_d":
        return mn * math.exp(DELTA_DEV)
    return j1 - float(rule.split("_")[1]) * sp


EMAX = 400


def stretched(run, rho, cens):
    """v[e-1] = J~ of `run` slowed by rho at B-epoch e (e = 1..EMAX); +inf beyond the data (pess), or the
    last value for one underlying epoch past the data if the run was capped (opt: 'reaches just after')."""
    Js = run["Js"]; n = len(Js)
    e = np.arange(1, EMAX + 1, dtype=float); u = e / rho
    k = np.clip(np.floor(u).astype(int), 1, n); f = u - np.floor(u)
    nxt = np.minimum(k, n - 1)
    v = np.where(k >= n, Js[n - 1], Js[k - 1] * (1 - f) + Js[nxt] * f)
    v = np.where(u <= 1, Js[0], v)
    beyond = u > n
    if cens == "opt" and not run["early_stopped"]:
        v = np.where(beyond & (u <= n + 1), -np.inf, v)        # reaches any level just after the data
        v = np.where(u > n + 1, np.inf, v)
    else:
        v = np.where(beyond, np.inf, v)
    return v


CACHE = {}


def curve(run_i, rho, cens):
    key = (run_i, rho, cens)
    if key not in CACHE:
        CACHE[key] = stretched(recs[run_i], rho, cens)
    return CACHE[key]


def reach(v, R, cap, interp):
    cap = int(min(cap, EMAX))
    idx = np.nonzero(v[:cap] <= R)[0]
    if len(idx) == 0:
        return math.inf
    i = int(idx[0])
    if not interp or i == 0 or not np.isfinite(v[i]) or not np.isfinite(v[i - 1]):
        return float(i + 1)
    a, b = v[i - 1], v[i]
    return float(i + ((a - R) / (a - b) if a != b else 1.0))


def E_rule(rule, i, A_level, rho, cap, interp, cens):
    run = recs[i]
    if rule == "served":
        E = rho * run["served"]
        return float(E) if E <= cap else math.inf
    v = curve(i, rho, cens)
    if rule.startswith("self_"):
        p = float(rule.split("_")[1])
        return reach(v, run["J1"] - p * run["span"], cap, interp)
    return reach(v, A_level, cap, interp)


def classify(EA, EB, stat):
    if not all(np.isfinite(EA)):
        return "A_miss"
    if not all(np.isfinite(EB)):
        return "update"
    r = (np.mean(EB) / np.mean(EA)) if stat == "mean" else math.exp(np.mean(np.log(EB)) - np.mean(np.log(EA)))
    return "epoch" if r <= 1.5 else ("update" if r >= 3 else "inconclusive")


def main():
    rng = np.random.default_rng(20260929)
    out = {"source": "42 v1 runs, DEV fold -2 only", "reps": REPS, "delta_dev": DELTA_DEV, "table": {}}
    N = len(recs)
    for S in (3, 4, 6):
        draws = [rng.choice(N, 2 * S, replace=False) for _ in range(REPS)]
        for rule in RULES:
            for interp in ((False,) if rule == "served" else (False, True)):
                for cens in ("pess", "opt"):
                    res = {"mean": {}, "geo": {}}
                    for rho in (1.0, 1.5, 2.0, 3.0, 4.0):
                        cnt = {s: {"epoch": 0, "update": 0, "inconclusive": 0, "A_miss": 0} for s in ("mean", "geo")}
                        for d in draws:
                            A = list(d[:S]); B = list(d[S:])
                            R = None if rule == "served" or rule.startswith("self_") else level(rule, [recs[i] for i in A])
                            EA = np.array([E_rule(rule, i, R, 1.0, EMAX, interp, cens) for i in A])
                            capB = math.ceil(3 * np.mean(EA)) + 3 if np.all(np.isfinite(EA)) else 60
                            EB = np.array([E_rule(rule, i, R, rho, capB, interp, cens) for i in B])
                            for s in ("mean", "geo"):
                                cnt[s][classify(EA, EB, s)] += 1
                        for s in ("mean", "geo"):
                            res[s][f"rho{rho:g}"] = {k: round(v / REPS, 4) for k, v in cnt[s].items()}
                    for s in ("mean", "geo"):
                        out["table"][f"S{S}|{rule}{'_interp' if interp else ''}|{cens}|{s}"] = res[s]
    (OUT / "q1_probe_sim.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{'key':<34}{'P(ep|1)':>8}{'P(up|1)':>8}{'Amiss|1':>8}{'P(up|4)':>8}{'P(ep|4)':>8}{'P(ep|2)':>8}{'P(up|2)':>8}")
    for k, v in out["table"].items():
        print(f"{k:<34}{v['rho1']['epoch']:8.3f}{v['rho1']['update']:8.3f}{v['rho1']['A_miss']:8.3f}{v['rho4']['update']:8.3f}"
              f"{v['rho4']['epoch']:8.3f}{v['rho2']['epoch']:8.3f}{v['rho2']['update']:8.3f}")


if __name__ == "__main__":
    main()

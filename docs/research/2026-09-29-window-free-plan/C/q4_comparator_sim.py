"""Q4: error rates of candidate paired tests for the A/B specifications, by simulation calibrated to the
measured noise (q2_noise_v1.json, q2_block_scaling.json). numpy/scipy only, fixed seed.

Quality criterion (horizon-mean log CRPS, 5-day out-of-sample blocks):
  d[f,s] = Delta + u_f + e[f,s],  u_f ~ N(0, 2 s_cp^2) (arm x fold interaction), e ~ N(0, s_pair^2)
  per-anchor layer for the pooled test: Delta(t) = d[f,s] + eta(t), eta piecewise constant on 80-bar
  blocks with the SD that reproduces the measured within-run block-bootstrap SE (h1: 0.0044 at 7,236 bars).
  Non-inferiority: H0 E[d] >= margin (B worse by the margin); 'non-inferior' iff the one-sided 95% upper
  bound < margin. Size is the pass rate at E[d] = margin; power the pass rate at E[d] = 0.
Tests: naive (i.i.d. pairs, t with n-1 df), clustered (t on the F fold means, F-1 df), pooled-anchor
  (anchors treated as i.i.d.; the anti-conservative mistake NT-032 must refuse).
Speed criterion: r = ln(epoch_B / epoch_A) = rho + N(0, 0.005^2) per pair, plus whole-run contention on
  24% of runs (v1 measured: 20 of 84 runs more than 2% off their fold median; shock ln-factor ~ U(0.1, 1.9)).
  Pass iff the one-sided 95% upper bound <= ln 1.05. mean/t versus Hodges-Lehmann/Wilcoxon versus t after
  the whole-run re-timing rule (a run > 1.10x its arm's median is re-timed once).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q4_comparator_sim.py
Writes q4_comparator_sim.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "q4_comparator_sim.json"
REPS = 4000
RNG = np.random.default_rng(20260928)


def upper_t(x, alpha=0.05):
    n = len(x)
    return x.mean() + stats.t.ppf(1 - alpha, n - 1) * x.std(ddof=1) / math.sqrt(n)


def upper_hl_wilcoxon(x, alpha=0.05):
    """One-sided upper confidence bound of the Hodges-Lehmann median from Walsh averages (exact for small n)."""
    n = len(x)
    w = np.sort([(x[i] + x[j]) / 2 for i in range(n) for j in range(i, n)])
    m = len(w)
    # critical value of the signed-rank statistic (normal approximation with continuity correction)
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    k = int(math.floor(mu - stats.norm.ppf(1 - alpha) * sd - 0.5))
    k = max(k, 0)
    return w[m - 1 - k]


def quality_sim(s_pair, s_cp, F, S, margin, eta_block_sd=None, n_bars=7236, block=80):
    res = {}
    for truth, label in ((margin, "size_at_margin"), (0.0, "power_at_zero")):
        naive = clustered = pooled = 0
        for _ in range(REPS):
            u = RNG.normal(0, math.sqrt(2) * s_cp, F)
            e = RNG.normal(0, s_pair, (F, S))
            d = truth + u[:, None] + e
            if F * S >= 2 and upper_t(d.ravel()) < margin:
                naive += 1
            if F >= 2 and upper_t(d.mean(axis=1)) < margin:
                clustered += 1
            if eta_block_sd is not None:
                # pooled-anchor test: per-anchor differences with 80-bar block noise around each pair's d
                nb = n_bars // block
                eta = RNG.normal(0, eta_block_sd, (F * S, nb))
                per_anchor_mean = d.ravel()[:, None] + eta                 # block means
                x = np.repeat(per_anchor_mean, block, axis=1).ravel()      # anchors (block-constant)
                # the mistake: anchors as i.i.d. with their raw SD
                se = x.std(ddof=1) / math.sqrt(len(x))
                if x.mean() + 1.645 * se < margin:
                    pooled += 1
        res[label] = {"naive": naive / REPS, "clustered": clustered / REPS if F >= 2 else None,
                      "pooled_anchor": pooled / REPS if eta_block_sd is not None else None}
    return res


def speed_sim(n_pairs, rho, contention_p=0.24, retime=False):
    fails = {"mean_t": 0, "hl_wilcoxon": 0}
    for _ in range(REPS):
        a = RNG.normal(0, 0.005 / math.sqrt(2), n_pairs)
        b = RNG.normal(rho, 0.005 / math.sqrt(2), n_pairs)
        for arr in (a, b):
            hit = RNG.random(n_pairs) < contention_p
            shock = RNG.uniform(0.1, 1.9, n_pairs)
            arr += np.where(hit, shock, 0.0)
            if retime:
                med = np.median(arr)
                again = arr > med + math.log(1.10)
                # re-timed once: a fresh draw with the same contention probability
                redo = np.where(RNG.random(n_pairs) < contention_p, RNG.uniform(0.1, 1.9, n_pairs), 0.0)
                arr[again] = (rho if arr is b else 0.0) + RNG.normal(0, 0.005 / math.sqrt(2), again.sum()) + redo[again]
        r = b - a
        if upper_t(r) > math.log(1.05):
            fails["mean_t"] += 1
        if upper_hl_wilcoxon(r) > math.log(1.05):
            fails["hl_wilcoxon"] += 1
    return {k: v / REPS for k, v in fails.items()}


def main():
    noise = json.loads((HERE / "q2_noise_v1.json").read_text(encoding="utf-8"))
    scal = json.loads((HERE / "q2_block_scaling.json").read_text(encoding="utf-8"))
    c = noise["metrics"]["logcrps_mean"]["components"]["pooled"]
    s_pair, s_cp = c["sd_pair_nominal_seed"], c["s_cp"]
    boot = scal["scaling"]["h1"]["within_run_block_bootstrap_sd_of_d_whole_block(80-bar blocks)"]
    eta_block_sd = boot * math.sqrt(7236 // 80)          # SD per 80-bar block that gives the measured SE
    out = {"reps": REPS, "s_pair": s_pair, "s_cp": s_cp, "eta_block_sd": eta_block_sd, "quality": {}, "speed": {}, "iut": {}}
    margin = 0.0065                                      # 1/3 of the planning edge (latest run, sizing only)
    for F, S in ((1, 6), (3, 5), (6, 3), (10, 2), (16, 1), (20, 1)):
        for k in (1, 2, 4):
            key = f"F{F}xS{S}_s_cp_x{k}"
            out["quality"][key] = quality_sim(s_pair, s_cp * k, F, S, margin,
                                              eta_block_sd if (k == 1 and F in (1, 10)) else None)
            print(key, out["quality"][key], flush=True)
    for n in (6, 12, 20):
        for retime in (False, True):
            key = f"n{n}_retime{retime}"
            out["speed"][key] = {"false_fail_at_rho0": speed_sim(n, 0.0, retime=retime),
                                 "pass_at_rho_ln1.10_(should_fail)": {k: 1 - v for k, v in speed_sim(n, math.log(1.10), retime=retime).items()}}
            print(key, out["speed"][key], flush=True)
    # joint ADOPT probability of an intersection-union rule with k independent criteria at power p
    out["iut"] = {f"k{k}": {f"p{p}": p ** k for p in (0.8, 0.9, 0.95, 0.99)} for k in (1, 2, 3, 4, 5, 6)}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["iut"], indent=1))


if __name__ == "__main__":
    main()

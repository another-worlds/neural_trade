"""Q4 checks (CPU, float64, read-only):

1. Empirical offset invariance of the full 31-channel series layer, mode (a), newest run's weights: start
   the recursion cold at bar s0 and compare with the run from bar 0 (warm at s0). k*(eps) = first k after
   which every channel differs by <= eps x its standard deviation for good. Against the analytic M_layer(eps)
   (q4_drift_warmup.m_layer, with the -0.5 shift bound).
2. The per-epoch span margin: M at logit_min - 0.5 - S * lr (the Adam bound within one epoch).
3. d log(period) / d logit (is the logit already a log-timescale parameter?).
4. Random-walk scale law of d = EMA - close: RMS(d) against sigma_1bar * sqrt((1-a)^2 / (a (2 - a))).
5. M(1e-3) of the periods the measured drift rates project after 20 / 40 epochs.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q4b_checks.py
Writes q4b_checks.json.
"""
import json
import os

import numpy as np

import common as C
from candidates import reference64
from q4_drift_warmup import m_layer, m_single

NEWEST = f"{C.RUNS}/20260924T182915Z-1aeff1c-dirty-af67ee43/"


def offset_invariance(info, close, eps_list=(1e-2, 1e-3, 1e-4), starts=(6000, 14000, 22000, 30000), span=4000):
    s = info["scale"]
    dx = np.diff(close) / s
    ctx = C.rolling_context(close, s)[1:]
    delta = C.meta_shift(ctx, info["W"], info["b"]).T
    lg = info["logits"]
    end = max(starts) + span
    full = reference64("a", dx[:end], lg, delta[:, :end])
    out = {"per_start": {}}
    for s0 in starts:
        cold = reference64("a", dx[s0:s0 + span], lg, delta[:, s0:s0 + span])
        warm = full[s0:s0 + span]
        sd = warm.std(0)
        cols = sd > 0
        rel = np.abs(cold[:, cols] - warm[:, cols]) / sd[cols]
        worst = rel.max(1)                                         # over channels, per bar k
        row = {}
        for eps in eps_list:
            bad = np.nonzero(worst > eps)[0]
            row[f"eps{eps:g}"] = int(bad[-1] + 1) if len(bad) else 0
        out["per_start"][str(s0)] = row
    per = C.period_of_logit(lg)
    for eps in eps_list:
        k = max(v[f"eps{eps:g}"] for v in out["per_start"].values())
        out[f"eps{eps:g}"] = {"empirical_max_kstar": k, "analytic_M_layer_shift": m_layer(per, eps)["M"],
                              "analytic_M_layer_no_shift": m_layer(per, eps, 0.0)["M"]}
    return out


def main():
    info = C.run_info(NEWEST)
    close = C.blocks(-1)["close"]
    res = {"offset_invariance_mode_a_newest": offset_invariance(info, close)}
    lg_min = float(info["logits"].min())
    margin = {}
    for S in (119, 39):
        for lr in (5e-3, 1e-3):
            a = C.sigmoid(lg_min - 0.5 - S * lr)
            margin[f"S{S}_lr{lr:g}"] = {"M_eps1e-3_epoch_bound": C.m_eps(a, 1e-3),
                                        "M_eps1e-3_now": C.m_eps(C.sigmoid(lg_min - 0.5), 1e-3),
                                        "factor": C.m_eps(a, 1e-3) / C.m_eps(C.sigmoid(lg_min - 0.5), 1e-3)}
    res["per_epoch_span_margin"] = margin
    res["dlogp_dlogit"] = {str(p): float(-2 * (1 - 2 / (p + 1)) / (2 - 2 / (p + 1))) for p in (2, 5, 10, 30, 60, 240, 1440)}
    dx = np.diff(close) / info["scale"]
    sig = float(np.sqrt(np.mean(dx ** 2)))
    law = {}
    q4 = json.load(open(os.path.join(C.HERE, "q4_drift_warmup.json")))
    for p, v in q4["curse_of_memory_sensitivity"].items():
        a = 2 / (int(p) + 1)
        pred = sig * np.sqrt((1 - a) ** 2 / (a * (2 - a)))
        law[p] = {"rms_d_measured": v["rms_feature"], "random_walk_prediction": float(pred), "ratio": v["rms_feature"] / pred}
    res["random_walk_scale_law"] = {"sigma_1bar_scaled": sig, "by_period": law}
    proj = {}
    for tag in ("projection_measured_rates_local", "projection_measured_rates_ablation"):
        top = q4[tag]["slowest_instances"][0]
        proj[tag] = {k: {"period": top[k], "M_eps1e-3_shift": m_single(top[k], 1e-3)}
                     for k in top if k.startswith("p_after")} | {"instance": f"{top['run'][-40:]} {top['name']}"}
    res["projected_periods_warmup"] = proj
    print(json.dumps(res, indent=1)[:3000])
    print("wrote", C.dump(res, "q4b_checks.json"))


if __name__ == "__main__":
    main()

"""Q4: warm-up with no period ceiling. CPU, read-only on the runs.

1. Drift of the learned periods per epoch, from indicator_params_history.csv of the 7 local runs and the
   84 v1 ablation runs, against Adam's per-step bound (|step| ~ lr_indicator = LR x INDICATOR_LR_MULT
   when the gradient sign is consistent): ratio |d logit| / (steps_per_epoch x lr_indicator).
2. Extrapolation without a ceiling: projected periods after 20 and 40 epochs (measured drift rate, and
   the Adam bound), at today's 119 steps per epoch and at a 7-day training block (~39 steps).
3. M(eps): bars until the initial state's weight falls below eps, for single EWMAs and for the
   two-stage channels (MACD signal, Bollinger variance), after the maximal adaptive shift (-0.5 logit).
4. Curse of memory: sensitivity of an EWMA feature to its logit vs the period (series mode, float64),
   and the feature change caused by one Adam step at lr_indicator = 5e-3 and 1e-3.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q4_drift_warmup.py
Writes q4_drift_warmup.json.
"""
import json
import os

import numpy as np
import pandas as pd

import common as C

LR_IND = 5e-3
CAP_HI, CAP_LO = 60.0, 2.0


def steps_per_epoch(run_dir):
    rows = [json.loads(l) for l in open(run_dir + "metrics.jsonl") if l.strip()]
    s = [r["epoch_seconds"] / r["sec_per_step"] for r in rows if r.get("sec_per_step")]
    lr = [r.get("lr_indicator_used") for r in rows if r.get("lr_indicator_used") is not None]
    return float(np.median(s)), (sorted(set(np.round(lr, 8))) if lr else None), len(rows)


def start_periods(run_dir):
    p = run_dir + "period_init.json"
    if os.path.exists(p):
        return json.load(open(p))["periods"], "period_init.json"
    return {k: float(v) for k, v in C.TEXTBOOK.items()}, "configured (textbook)"


def drift(run_dirs, label):
    recs, per_run = [], {}
    for r in run_dirs:
        h = pd.read_csv(r + "indicator_params_history.csv")
        S, lrs, n_ep = steps_per_epoch(r)
        start, src = start_periods(r)
        bound = S * LR_IND
        run = os.path.basename(r.rstrip("/\\"))
        P = np.vstack([[start[n] for n in C.NAMES], h[C.NAMES].to_numpy(np.float64)])   # [E+1, 18]
        Lg = C.logit_of_period(P)
        dL = np.diff(Lg, axis=0)
        capped = (P[1:] >= CAP_HI - 0.05) | (P[:-1] >= CAP_HI - 0.05) | (P[1:] <= CAP_LO + 0.01) | (P[:-1] <= CAP_LO + 0.01)
        for j, n in enumerate(C.NAMES):
            for e in range(dL.shape[0]):
                recs.append({"set": label, "run": run, "name": n, "epoch": e, "dlogit": dL[e, j],
                             "ratio": abs(dL[e, j]) / bound, "capped": bool(capped[e, j])})
        free = ~capped
        rate = np.array([dL[free[:, j], j].mean() if free[:, j].any() else np.nan for j in range(18)])
        per_run[run] = {"steps_per_epoch": S, "lr_indicator_logged": lrs, "epochs": int(dL.shape[0]),
                        "start_source": src, "adam_bound_per_epoch": bound,
                        "start": dict(zip(C.NAMES, P[0].round(3).tolist())),
                        "end": dict(zip(C.NAMES, P[-1].round(3).tolist())),
                        "max_period": float(P.max()), "argmax": C.NAMES[int(np.argmax(P.max(0)))],
                        "hit_cap_60": [n for j, n in enumerate(C.NAMES) if (P[:, j] >= CAP_HI - 0.05).any()],
                        "first_epoch_at_cap": {n: int(np.argmax(P[1:, j] >= CAP_HI - 0.05)) for j, n in enumerate(C.NAMES)
                                               if (P[1:, j] >= CAP_HI - 0.05).any()},
                        "mean_uncapped_rate_logit_per_epoch": dict(zip(C.NAMES, np.round(rate, 5).tolist()))}
    return pd.DataFrame(recs), per_run


def summarize(df):
    d = df[~df.capped]
    out = {"n_epoch_changes": int(len(d)),
           "ratio_quantiles": {f"p{q}": float(np.percentile(d.ratio, q)) for q in (50, 90, 99, 100)},
           "share_ratio_above_0.5": float((d.ratio > 0.5).mean()), "share_ratio_above_0.9": float((d.ratio > 0.9).mean()),
           "top5": d.sort_values("ratio", ascending=False).head(5)[["run", "name", "epoch", "dlogit", "ratio"]].to_dict("records")}
    fam = {}
    for pre in ("ma_", "macd", "rsi", "bb_"):
        x = d[d.name.str.startswith(pre)]
        fam[pre] = {"ratio_p50": float(np.percentile(x.ratio, 50)), "ratio_p99": float(np.percentile(x.ratio, 99)),
                    "ratio_max": float(x.ratio.max())}
    out["by_family"] = fam
    by_slow = d[d.name.str.endswith("_slow")]
    out["slow_periods"] = {"ratio_p50": float(np.percentile(by_slow.ratio, 50)), "ratio_max": float(by_slow.ratio.max()),
                           "share_dlogit_negative(longer)": float((by_slow.dlogit < 0).mean())}
    return out


def project(per_run, epochs=(20, 40), steps=(119, 39)):
    """Periods after E epochs if the measured uncapped mean rate continued with no ceiling."""
    rows = []
    for run, v in per_run.items():
        S = v["steps_per_epoch"]
        for n in C.NAMES:
            rate = v["mean_uncapped_rate_logit_per_epoch"][n]
            if rate is None or not np.isfinite(rate):
                continue
            l0 = C.logit_of_period(v["start"][n])
            rows.append({"run": run, "name": n, "rate": rate, "rate_per_step": rate / S,
                         **{f"p_after_{E}ep_at_{s}steps": float(C.period_of_logit(l0 + rate / S * s * E))
                            for E in epochs for s in steps}})
    df = pd.DataFrame(rows)
    top = df.sort_values("rate").head(8)
    return {"slowest_instances": top.to_dict("records"),
            "median_rate_logit_per_epoch": float(df.rate.median()),
            "most_negative_rate_per_step": float(df.rate_per_step.min())}


def bound_projection(p0_list=(26, 35, 60), epochs=(20, 40), steps=(119, 39), lr=(5e-3, 1e-3)):
    out = {}
    for p0 in p0_list:
        for s in steps:
            for E in epochs:
                for l in lr:
                    out[f"p0={p0} steps={s} E={E} lr={l:g}"] = float(C.period_of_logit(C.logit_of_period(p0) - s * E * l))
    return out


# --------------------------------------------------------------------------------------------- M(eps)
def m_single(p, eps, shift=0.5):
    a = C.sigmoid(C.logit_of_period(p) - shift)
    return C.m_eps(a, eps)


def m_cascade(p1, p2, eps, shift=0.5, factor=1.0, tmax=2_000_000):
    """Stage-2 z_t = (1-b) z_{t-1} + b u_t, u_t perturbed by (1-a)^t (stage-1 initial state) plus z's own
    initial state: response r_t = (1-b)^t + factor * b * sum_k (1-b)^(t-k) (1-a)^k. First t after which
    r stays <= eps (both terms decay monotonically after their peak)."""
    a = C.sigmoid(C.logit_of_period(p1) - shift)
    b = C.sigmoid(C.logit_of_period(p2) - shift)
    # closed form of the convolution: b (1-a) ((1-a)^t - (1-b)^t) / ((1-a) - (1-b)) for a != b
    t = np.arange(1, tmax, dtype=np.float64)
    if abs(a - b) > 1e-12:
        conv = b * (1 - a) * ((1 - a) ** t - (1 - b) ** t) / ((1 - a) - (1 - b))
    else:
        conv = b * t * (1 - a) ** t
    r = (1 - b) ** t + factor * np.abs(conv)
    above = np.nonzero(r > eps)[0]
    return int(above[-1] + 2) if len(above) else 1


def m_layer(periods, eps, shift=0.5):
    """Warm-up of today's 31-channel layer: max over its single and two-stage channels."""
    p = dict(zip(C.NAMES, periods))
    ms = {}
    for n in C.NAMES:
        ms[n] = m_single(p[n], eps, shift)
    for i in range(3):
        ms[f"macd_{i}_signal_cascade"] = max(m_cascade(p[f"macd_{i}_slow"], p[f"macd_{i}_signal"], eps, shift),
                                             m_cascade(p[f"macd_{i}_fast"], p[f"macd_{i}_signal"], eps, shift))
        ms[f"bb_{i}_variance_cascade"] = m_cascade(p[f"bb_period_{i}"], p[f"bb_period_{i}"], eps, shift, factor=2.0)
    worst = max(ms, key=ms.get)
    return {"M": ms[worst], "worst": worst, "all": ms}


def warmup_tables(newest_end):
    out = {}
    textbook = [C.TEXTBOOK[n] for n in C.NAMES]
    learned = [newest_end[n] for n in C.NAMES]
    for eps in (1e-2, 1e-3, 1e-4, 1e-6):
        out[f"eps{eps:g}"] = {
            "textbook_no_shift": m_layer(textbook, eps, 0.0)["M"],
            "textbook_shift": m_layer(textbook, eps)["M"],
            "newest_run_learned_shift": m_layer(learned, eps),
            "single_ewma_shift": {str(p): m_single(p, eps) for p in (60, 240, 1440, 10080, 43200, 100000)},
            "single_ewma_no_shift": {str(p): m_single(p, eps, 0.0) for p in (60, 240, 1440, 10080, 43200, 100000)}}
    # share of a 7-day training block (10,080 bars) masked at the data start if no earlier history exists
    blk = 10080
    out["share_of_7day_block_masked_eps1e-3"] = {str(p): min(1.0, (m_single(p, 1e-3) + 59) / blk)
                                                 for p in (60, 240, 1440, 10080)}
    return out


# ----------------------------------------------------------------------------------- curse of memory
def sensitivity(close, scale, periods=(5, 10, 30, 60, 240, 1440, 10080, 43200)):
    """d_t = EMA_t - close_t (scaled), series mode; dd/dlogit by forward-mode recursion (float64)."""
    dx = np.diff(close) / scale
    out = {}
    burn = 0
    for p in periods:
        a = 2.0 / (p + 1.0)
        d = 0.0; s = 0.0
        D = np.empty(len(dx)); Sg = np.empty(len(dx))
        for t, x in enumerate(dx):
            prev = d - x
            s = -prev + (1 - a) * s                 # d d_t / d a  (d_t = (1-a)(d_{t-1} - dx_t))
            d = (1 - a) * prev
            D[t] = d; Sg[t] = s * a * (1 - a)        # chain rule to the logit
        m = C.m_eps(a, 1e-3)
        sl = slice(min(m, len(dx) // 2), None)      # after the warm-up (half the file at most)
        rms_d, rms_g = float(np.sqrt(np.mean(D[sl] ** 2))), float(np.sqrt(np.mean(Sg[sl] ** 2)))
        out[str(p)] = {"rms_feature": rms_d, "rms_dfeature_dlogit": rms_g, "rel_sensitivity": rms_g / rms_d,
                       "feature_change_per_adam_step_lr5e-3_rel": 5e-3 * rms_g / rms_d,
                       "feature_change_per_epoch_119_steps_rel": 119 * 5e-3 * rms_g / rms_d,
                       "warmup_used_bars": int(sl.start), "bars_scored": int(len(dx) - sl.start)}
    return out


def main():
    loc, per_loc = drift(C.run_dirs(), "local")
    abl, per_abl = drift(C.ablation_dirs(), "ablation")
    newest = per_loc["20260924T182915Z-1aeff1c-dirty-af67ee43"]
    B = C.blocks(-1)
    res = {"lr_indicator_assumed": LR_IND,
           "drift_local": summarize(loc), "drift_ablation": summarize(abl),
           "per_run_local": per_loc,
           "per_run_ablation_summary": {"steps_per_epoch_values": sorted({round(v["steps_per_epoch"]) for v in per_abl.values()}),
                                        "max_period_over_runs": max(v["max_period"] for v in per_abl.values()),
                                        "runs_hitting_cap": sum(bool(v["hit_cap_60"]) for v in per_abl.values()),
                                        "cap_hits_by_name": pd.Series([n for v in per_abl.values() for n in v["hit_cap_60"]]).value_counts().to_dict()},
           "projection_measured_rates_local": project(per_loc), "projection_measured_rates_ablation": project(per_abl),
           "projection_adam_bound": bound_projection(),
           "warmup": warmup_tables(newest["end"]),
           "curse_of_memory_sensitivity": sensitivity(B["close"], 257.51813253642973)}
    path = C.dump(res, "q4_drift_warmup.json")
    for k in ("drift_local", "drift_ablation"):
        print(k, json.dumps({kk: res[k][kk] for kk in ("ratio_quantiles", "share_ratio_above_0.5", "slow_periods")}))
    print("top local", res["drift_local"]["top5"][:3])
    print("projection local", json.dumps(res["projection_measured_rates_local"]["slowest_instances"][:3], default=float))
    print("warmup eps1e-3", json.dumps(res["warmup"]["eps0.001"], default=float)[:800])
    print("sensitivity", json.dumps({p: round(v["rel_sensitivity"], 3) for p, v in res["curse_of_memory_sensitivity"].items()}))
    print("wrote", path)


if __name__ == "__main__":
    main()

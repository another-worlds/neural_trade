"""Review check: the probe (spec 0) and A/B-2 judge reach against a COMMON target built from arm A's
seeds (R = mean(min J~) + 0.05 x mean span), but C/ q3_probe_target measured reach epochs against each
run's OWN minimum (always reached). How often does a run of the SAME condition never reach a common
target built from the OTHER seeds of its condition (a stand-in for an equivalent arm B)?

Data: the 42 v1 runs on dev fold -2 (14 conditions x 3 seeds, batch 256, 20-epoch cap; the physics
weights differ between conditions, so the same-condition comparison is the equivalent-arm null).
J = val CRPS summed over horizons, J~ = centred 3-epoch mean (C/ q3_probe_target.py's definitions).
Run: PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python r3_common_target.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID = Path("D:/neural_trade/runs/ablations/ablate_physics_v1-full")


def curve(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return np.array([r["val_crps_h0"] + r["val_crps_h1"] + r["val_crps_h2"] for r in rows], float)


def smooth(J):
    return np.array([J[max(0, i - 1): i + 2].mean() for i in range(len(J))])


def main():
    df = pd.read_csv(GRID / "results.csv")
    df = df[df.period == "P1"]
    recs = []
    for _, r in df.iterrows():
        Js = smooth(curve(GRID / "runs" / r["run_id"] / "metrics.jsonl"))
        recs.append({"cond": r["condition"], "seed": int(r["seed"]), "Js": Js, "min": Js.min(), "span": Js[0] - Js.min()})
    out = {}
    for frac in (0.05, 0.10):
        never, total, sd_ratio = 0, 0, []
        for cond in sorted({x["cond"] for x in recs}):
            g = [x for x in recs if x["cond"] == cond]
            mins = np.array([x["min"] for x in g]); spans = np.array([x["span"] for x in g])
            sd_ratio.append(mins.std(ddof=1) / spans.mean())
            for i, x in enumerate(g):
                others = [j for j in range(len(g)) if j != i]
                R = mins[others].mean() + frac * spans[others].mean()
                total += 1
                never += int(not np.any(x["Js"] <= R))
        out[f"frac{frac}"] = {"runs": total, "never_reach_leave_one_out_target": never,
                              "share_never": never / total,
                              "median_sd_min_over_mean_span": float(np.median(sd_ratio))}
    print(json.dumps(out, indent=1))
    Path("D:/nt_research/wfp/review/r3_common_target.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

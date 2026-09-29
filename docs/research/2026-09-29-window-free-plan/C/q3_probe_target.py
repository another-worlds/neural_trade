"""Q3 (0): the probe's lambda-free target on real curves. J = val CRPS summed over horizons (logged per
epoch; grouping-free, q3_val_grouping_check.json). J~ = 3-epoch centred running mean. Target per arm-fold:
T = min_e J~(e); span = J~(1) - T; reach level R = T + 5% (or 10%) of span; E = first epoch with J~ <= R.

Curves: the 42 v1 runs on DEV fold -2 (P1; batch 256, 20-epoch cap, fixed lambdas) and the four 20-epoch
notebook runs (fold -1's VAL block, not test). Read-only. CPU.
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q3_probe_target.py
Writes q3_probe_target.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "q3_probe_target.json"
GRID = Path("D:/neural_trade/runs/ablations/ablate_physics_v1-full")


def curve(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    J = np.array([r["val_crps_h0"] + r["val_crps_h1"] + r["val_crps_h2"] for r in rows], float)
    t = np.array([r["time"] for r in rows], float)
    es = np.array([r["epoch_seconds"] for r in rows], float)
    return J, t, es


def smooth(J):
    out = np.empty_like(J)
    for i in range(len(J)):
        out[i] = J[max(0, i - 1): i + 2].mean()
    return out


def reach(Js, frac):
    T = Js.min()
    span = Js[0] - T
    R = T + frac * span
    return int(np.argmax(Js <= R)) + 1, float(T), float(span), int(np.argmin(Js)) + 1


def main():
    df = pd.read_csv(GRID / "results.csv")
    out = {"v1_dev_fold_-2": {}, "notebook_val_fold_-1": {}}
    rows = []
    for _, r in df[df.period == "P1"].iterrows():
        J, t, es = curve(GRID / "runs" / r["run_id"] / "metrics.jsonl")
        Js = smooth(J)
        for frac in (0.05, 0.10):
            E, T, span, emin = reach(Js, frac)
            rows.append({"run": r["run_id"], "condition": r["condition"], "seed": int(r["seed"]), "frac": frac,
                         "epochs_run": len(J), "reach_epoch": E, "min_epoch_smoothed": emin, "T": T, "span": span,
                         "min_in_last_3_epochs": emin >= len(J) - 2,
                         "late_abs_step_over_span": float(np.median(np.abs(np.diff(J[-6:]))) / span) if span > 0 else None})
    t_ = pd.DataFrame(rows)
    for frac, g in t_.groupby("frac"):
        full = g[g.epochs_run == 20]
        out["v1_dev_fold_-2"][f"frac{frac}"] = {
            "runs": int(len(g)), "runs_with_20_epochs": int(len(full)),
            "reach_epoch_quantiles": {q: float(g.reach_epoch.quantile(q)) for q in (0.1, 0.5, 0.9, 1.0)},
            "reach_epoch_sd_log": float(np.log(g.reach_epoch).std(ddof=1)),
            "min_in_last_3_epochs_share(20-epoch runs)": float(full.min_in_last_3_epochs.mean()),
            "smoothed_min_epoch_quantiles": {q: float(g.min_epoch_smoothed.quantile(q)) for q in (0.1, 0.5, 0.9)},
            "late_abs_step_over_span_median": float(g.late_abs_step_over_span.median()),
            "within_condition_sd_log_reach_epoch": float(np.sqrt((g.groupby("condition").reach_epoch.transform(lambda x: np.log(x) - np.log(x).mean()) ** 2).sum() / (len(g) - g.condition.nunique()))),
        }
    for rd in sorted(Path("D:/neural_trade/runs").glob("2026*")):
        J, t, es = curve(rd / "metrics.jsonl")
        if len(J) < 20:
            continue
        Js = smooth(J)
        out["notebook_val_fold_-1"][rd.name] = {f"frac{f}": dict(zip(("reach_epoch", "T", "span", "min_epoch_smoothed"), reach(Js, f)))
                                                for f in (0.05, 0.10)}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

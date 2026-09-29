"""Shared loading for the rev/ scripts (window-free plan, second review round).

Data: the 42 v1 physics-ablation runs on DEV fold -2 (period P1) only. Fold -1 (P2) is the test fold
and is never read here (D-020). Read-only.
J = val CRPS summed over horizons (= val_crps_loss; grouping- and lambda-free, C/q3_val_grouping_check),
J~ = centred 3-epoch running mean (C/q3_probe_target.py's definition).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID = Path("D:/neural_trade/runs/ablations/ablate_physics_v1-full")
PER_RUN = Path("D:/nt_research/wfp/C/q2_v1_per_run.csv")
OUT = Path("D:/nt_research/wfp/rev")
EARLY = 6          # v1 EarlyStopping patience (val_loss)
CAP_V1 = 20        # v1 EPOCHS
STEPS_V1_FOLD_M2 = 90   # ceil(22,977 / 256): fold -2 training anchors (C/q1_folds_today.json)
STEPS_7DAY = 40         # ceil(10,080 / 256)


def smooth(J):
    return np.array([J[max(0, i - 1): i + 2].mean() for i in range(len(J))])


def load_dev_runs():
    """List of dicts, one per P1 run."""
    df = pd.read_csv(GRID / "results.csv")
    df = df[df.period == "P1"]
    pr = pd.read_csv(PER_RUN)
    pr = pr[pr.period == "P1"].set_index("run_id")
    recs = []
    for _, r in df.iterrows():
        rows = [json.loads(l) for l in (GRID / "runs" / r["run_id"] / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        J = np.array([x["val_crps_h0"] + x["val_crps_h1"] + x["val_crps_h2"] for x in rows], float)
        vl = np.array([x["val_loss"] for x in rows], float)
        tt = np.array([x["time"] for x in rows], float)
        es = np.array([x["epoch_seconds"] for x in rows], float)
        start = tt[0] - es[0]
        wall = tt - start                      # fit wall-clock to the END of epoch e (index 0 = epoch 1)
        Js = smooth(J)
        p = pr.loc[r["run_id"]]
        recs.append({
            "run_id": r["run_id"], "cond": r["condition"], "seed": int(r["seed"]),
            "J": J, "Js": Js, "val_loss": vl, "wall": wall, "n": len(J),
            "early_stopped": len(J) < CAP_V1,
            "served": int(np.argmin(vl)) + 1,           # D-011: best val_loss epoch (1-based)
            "min": float(Js.min()), "J1": float(Js[0]), "span": float(Js[0] - Js.min()),
            "log_edge_mean": float(p["log_edge_mean"]), "logcrps_mean": float(p["logcrps_mean"]),
            "log_edge": [float(p[f"log_edge_{h}"]) for h in ("h0", "h1", "h2")],
            "logcrps": [float(p[f"logcrps_{h}"]) for h in ("h0", "h1", "h2")],
            "median_epoch_s": float(p["median_epoch_s"]),
        })
    return recs


def reach_epoch(Js, R, interp=False, cap=None):
    """First 1-based epoch with J~ <= R within the curve (optionally only up to `cap`); np.inf if never.
    interp=True: linear interpolation of the crossing between epochs (fractional epoch)."""
    n = len(Js) if cap is None else min(len(Js), int(cap))
    idx = np.nonzero(Js[:n] <= R)[0]
    if len(idx) == 0:
        return np.inf
    i = int(idx[0])
    if not interp or i == 0:
        return float(i + 1)
    a, b = Js[i - 1], Js[i]
    frac = (a - R) / (a - b) if a != b else 1.0
    return float(i + frac)

"""capacity_v1 summary (NT-104): per-arm tables from the scenario cells (descriptive; the verdicts come
from `neural-trade compare`). Reads result.json, eval_report_dev.json, training_log.csv and
predictions_oos.npz of each cell; writes summary.json and prints markdown tables.

    PYTHONPATH=D:/nt/nt_wt_104ab/src python summarize.py
"""
import glob
import json
import os
import re

import numpy as np

ROOT = "D:/nt/nt_wt_104ab/runs/scenarios/capacity_v1"
OUT = "D:/nt/nt_wt_104ab/runs/experiments/capacity_v1/summary.json"
ARMS = ["control", "gru_small", "linear_indicators"]
FOLDS = [-96, -95, -94, -93, -92]
HS = ["h0", "h1", "h2"]


def cells():
    out = {}
    for d in sorted(glob.glob(ROOT + "/2*")):
        m = re.search(r"-(control|gru_small|linear_indicators)__f(-\d+)__s(\d+)$", d.replace("\\", "/"))
        if not m:
            continue
        out[(m.group(1), int(m.group(2)))] = d
    return out


def bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    rows = []
    for (arm, fold), d in cells().items():
        r = json.load(open(d + "/result.json"))
        row = {"arm": arm, "fold": fold, "run_id": os.path.basename(d), "status": r["status"],
               "wall_s": r.get("wall_s"), "train_s": r.get("train_s"), "score_s": r.get("score_s"),
               "sec_per_step": r.get("sec_per_step")}
        if r["status"] != "done":
            rows.append(row)
            continue
        s = r["scores"]
        for h in HS:
            for k in ("direction/auc", "direction/brier", "variance/crpss", "variance/coverage90"):
                row[f"{h}/{k}"] = s.get(f"{h}/{k}")
            row[f"{h}/logreg_auc"] = s.get(f"baseline/logreg_lags/{h}/direction/auc")
        try:
            z = np.load(d + "/predictions_oos.npz", allow_pickle=True)
            y = z["y"]
            for i, h in enumerate(HS):
                p = z[f"direction_prob_calibrated__{h}"]
                row[f"{h}/bce"] = bce(p, (y[:, i] > 0).astype(float))
        except Exception as exc:  # noqa
            row["bce_error"] = str(exc)
        try:
            ev = json.load(open(d + "/eval_report_dev.json"))
            row["skip_share"] = ev.get("health", {}).get("direction_skip_share")
        except Exception:
            pass
        try:
            import csv
            with open(d + "/training_log.csv") as fh:
                log = list(csv.DictReader(fh))
            row["epochs_run"] = len(log)
            row["nonfinite_grad_steps"] = sum(float(x.get("nonfinite_grad_steps") or 0) for x in log)
            row["grad_global_norm_mean"] = float(np.mean([float(x["grad_global_norm"]) for x in log]))
            row["grad_clip_steps_main_total"] = sum(float(x.get("grad_clip_steps_main") or 0) for x in log)
            vl = [float(x["val_loss"]) for x in log]
            row["served_epoch"] = int(np.argmin(vl)) + 1
        except Exception as exc:  # noqa
            row["log_error"] = str(exc)
        try:
            st = json.load(open(d + "/status.json"))
            row["weights_epoch"] = st.get("weights_epoch")
        except Exception:
            pass
        rows.append(row)
    json.dump(rows, open(OUT, "w"), indent=1)

    def mean_table(metric):
        print(f"\n### {metric} (per fold; mean)\n")
        print("| arm | " + " | ".join(str(f) for f in FOLDS) + " | mean |")
        print("|---|" + "---|" * (len(FOLDS) + 1))
        for arm in ARMS:
            vals = []
            for f in FOLDS:
                v = [r.get(metric) for r in rows if r["arm"] == arm and r["fold"] == f]
                vals.append(v[0] if v and v[0] is not None else float("nan"))
            print(f"| {arm} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {np.nanmean(vals):.4f} |")

    for m in ["h1/direction/auc", "h0/direction/auc", "h2/direction/auc", "h1/variance/crpss", "h0/variance/crpss",
              "h2/variance/crpss", "h1/direction/brier", "h0/bce", "h1/bce", "h2/bce", "h1/logreg_auc",
              "h0/variance/coverage90", "h1/variance/coverage90", "h2/variance/coverage90", "epochs_run",
              "served_epoch", "sec_per_step", "wall_s", "nonfinite_grad_steps", "grad_global_norm_mean"]:
        mean_table(m)
    print("\n### skip_share / tower_share / corr (h1)\n")
    for r in rows:
        ss = r.get("skip_share")
        if ss:
            h1 = ss.get("h1", {})
            print(f"{r['arm']} f{r['fold']}: skip {h1.get('skip_share'):.3f} tower {h1.get('tower_share'):.3f} "
                  f"corr {h1.get('corr_skip_tower'):.3f}" if h1 and h1.get("skip_share") is not None else f"{r['arm']} f{r['fold']}: n/a")
    print("\nwall total GPU hours:", sum(r.get("wall_s") or 0 for r in rows) / 3600)


if __name__ == "__main__":
    main()

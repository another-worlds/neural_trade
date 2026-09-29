"""Q2 support: the 7 local notebook runs (runs/2026*/, fold -1 of the bundled file, different commits).

Used for: timing (median epoch time of epochs >= 1, epoch 0, time between epochs), convergence
(best val epoch, whether the cap binds), the val-block CRPS curve that the probe's target is built
from (val, not test), and -- ONLY for sizing, as the Challenge did -- the test-block edge of the
latest run. Read-only. CPU only.
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python q2_notebook_runs.py
Writes q2_notebook_runs.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RUNS = sorted(Path("D:/neural_trade/runs").glob("2026*"))
OUT = Path(__file__).with_name("q2_notebook_runs.json")


def main():
    out = {}
    for rd in RUNS:
        mets = [json.loads(l) for l in (rd / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        st = json.loads((rd / "status.json").read_text(encoding="utf-8"))
        ep = np.array([m["epoch_seconds"] for m in mets], float)
        tt = np.array([m["time"] for m in mets], float)
        between = (np.diff(tt) - ep[1:]) if len(tt) > 1 else np.array([])
        vcrps = np.array([m["val_crps_h0"] + m["val_crps_h1"] + m["val_crps_h2"] for m in mets], float)
        vnll = np.array([m["val_nll_h0"] + m["val_nll_h1"] + m["val_nll_h2"] for m in mets], float)
        vl = np.array([m["val_loss"] for m in mets], float)
        r = {"epochs": len(mets), "status": st,
             "epoch0_s": float(ep[0]),
             "median_epoch_s_ge1": float(np.median(ep[1:])) if len(ep) > 1 else None,
             "robust_sd_log_epoch_s_ge1": float(1.4826 * np.median(np.abs(np.log(ep[1:]) - np.median(np.log(ep[1:]))))) if len(ep) > 2 else None,
             "max_over_median_ge1": float(ep[1:].max() / np.median(ep[1:])) if len(ep) > 1 else None,
             "median_between_epochs_s": float(np.median(between)) if len(between) else None,
             "val_loss_curve": vl.round(4).tolist(),
             "val_crps_sum_curve": vcrps.round(5).tolist(),
             "val_nll_sum_curve": vnll.round(4).tolist(),
             "best_val_loss_epoch_1based": int(np.argmin(vl)) + 1,
             "best_val_crps_epoch_1based": int(np.argmin(vcrps)) + 1,
             "best_val_nll_epoch_1based": int(np.argmin(vnll)) + 1}
        if len(vcrps) >= 5:
            best = vcrps.min()
            span = vcrps[0] - best
            r["val_crps_epoch1_minus_best"] = float(span)
            # first epoch within 5% / 10% of the span above the best (the probe's candidate target)
            r["first_epoch_within_5pct_of_span"] = int(np.argmax(vcrps <= best + 0.05 * span)) + 1
            r["first_epoch_within_10pct_of_span"] = int(np.argmax(vcrps <= best + 0.10 * span)) + 1
            late = vcrps[-6:]
            r["late_epoch_to_epoch_abs_change_median"] = float(np.median(np.abs(np.diff(late))))
            r["late_epoch_to_epoch_abs_change_over_span"] = float(np.median(np.abs(np.diff(late))) / span) if span > 0 else None
        evp = rd / "eval_report_test.json"
        if evp.exists():
            ev = json.loads(evp.read_text(encoding="utf-8"))
            te = {}
            for h in ("h0", "h1", "h2"):
                m = ev["model"]["horizons"][h]
                cv = ev["baselines"]["const_var"]["horizons"][h]["variance"]["crps"]
                te[h] = {"crpss": m["variance"]["crpss"], "log_edge": math.log(cv) - math.log(m["variance"]["crps"]),
                         "auc": m["direction"]["auc"],
                         "logreg_auc": ev["baselines"]["logreg_lags"]["horizons"][h]["direction"]["auc"],
                         "coverage90": m["variance"]["coverage90"]}
            te["sharpe_net"] = ((ev.get("backtest") or {}).get("summary") or {}).get("sharpe_net")
            r["TEST_block_for_sizing_only"] = te
        out[rd.name] = r
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    for k, r in out.items():
        print(k, "epochs", r["epochs"], "ep0", round(r["epoch0_s"], 1), "median>=1", r["median_epoch_s_ge1"] and round(r["median_epoch_s_ge1"], 2),
              "robust sd log", r["robust_sd_log_epoch_s_ge1"] and round(r["robust_sd_log_epoch_s_ge1"], 4),
              "between", r["median_between_epochs_s"] and round(r["median_between_epochs_s"], 2),
              "best val_loss ep", r["best_val_loss_epoch_1based"], "best val_crps ep", r["best_val_crps_epoch_1based"],
              "5%-of-span ep", r.get("first_epoch_within_5pct_of_span"), "late |d|/span", r.get("late_epoch_to_epoch_abs_change_over_span"))
        if "TEST_block_for_sizing_only" in r:
            te = r["TEST_block_for_sizing_only"]
            print("    TEST (sizing only): h1 crpss", round(te["h1"]["crpss"], 4), "log edge", round(te["h1"]["log_edge"], 4),
                  "auc", round(te["h1"]["auc"], 3), "logreg", round(te["h1"]["logreg_auc"], 3), "cov", round(te["h1"]["coverage90"], 3))
        print("    val_crps_sum:", r["val_crps_sum_curve"])


if __name__ == "__main__":
    main()

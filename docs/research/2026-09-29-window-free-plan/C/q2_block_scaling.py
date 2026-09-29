"""Q2: how the paired CRPS noise between two runs grows when the evaluation block gets shorter
(7-day training folds after NT-041 will have evaluation blocks of about 1-2 days, the v1 grid had 5).

Re-predicts 12 v1 runs on their DEV block (fold -2's out-of-sample block, P1) on the CPU from their
saved checkpoint weights (weights.h5 = the best-val_loss epoch written by model_checkpoint), refits
the calibration pipeline on the fold's cal block exactly as train_and_evaluate does, and stores the
per-anchor served predictions. Conditions: all_on, all_off, only:LAMBDA_T_PERP, without:LAMBDA_VAC
(today-like) x seeds 0-2. Two runs whose evaluated weights WERE the checkpoint (all_off s0: early
stopping restored epoch 10; only:LAMBDA_T_PERP s0: best epoch = last) validate the reproduction
against their eval_report_test.json.

Nothing in D:/neural_trade is written: weights are copied to ./scratch, SCALER/ARTIFACTS paths point
there, save_artifacts=False, no RunContext, and the process runs with cwd=./scratch.
Run (CPU): cd D:/nt_research/wfp/C/scratch && CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 \
    C:/Users/Step/miniforge3/envs/nt/python ../q2_block_scaling.py
Writes q2_block_scaling_preds/<run>.npz and q2_block_scaling.json.
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "scratch"
PRED_DIR = HERE / "q2_block_scaling_preds"
OUT = HERE / "q2_block_scaling.json"
GRID = Path("D:/neural_trade/runs/ablations/ablate_physics_v1-full")
CSV = "D:/neural_trade/binance_btcusdt_1min_ccxt.csv"
CONDS = ["all_on", "all_off", "only:LAMBDA_T_PERP", "without:LAMBDA_VAC"]
VALIDATE = {"20260923T203258Z-6dec27a-ddfc658c-all_off__s0__P1", "20260923T210911Z-6dec27a-8c390f51-only-LAMBDA_T_PERP__s0__P1"}
LENGTHS = [360, 720, 1440, 2880, 7236]   # anchors per evaluation block (0.25, 0.5, 1, 2, 5 days)
HS = ("h0", "h1", "h2")


class _Catch(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def predict_runs():
    import neural_trade  # noqa: F401  (CUDA DLL path; CPU here)
    import pandas as pd
    import tensorflow as tf
    from neural_trade.core.config import Config
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.evaluation.report import gaussian_crps
    from neural_trade.training.trainer import train_and_evaluate

    catch = _Catch()
    for name in ("", "neural_trade", "neural_trade.training", "neural_trade.training.trainer"):
        lg = logging.getLogger(name)
        lg.addHandler(catch)
    logging.getLogger("neural_trade.training.trainer").setLevel(logging.INFO)
    df = pd.read_csv(GRID / "results.csv")
    sel = df[(df.period == "P1") & (df.condition.isin(CONDS))]
    PRED_DIR.mkdir(exist_ok=True)
    checks = {}
    for _, r in sel.iterrows():
        rid = r["run_id"]
        dst = PRED_DIR / f"{rid}.npz"
        if dst.exists():
            continue
        rd = GRID / "runs" / rid
        wdir = SCRATCH / rid
        wdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rd / "weights.h5", wdir / "weights.h5")
        cfg = Config.from_yaml(rd / "config.yaml")
        cfg.CSV_PATH = CSV
        cfg.MODEL_PATH = str(wdir / "weights.h5")
        cfg.SCALER_PATH = str(wdir / "scaler.joblib")
        cfg.ARTIFACTS_DIR = str(wdir / "artifacts")
        catch.msgs.clear()
        tf.keras.backend.clear_session()
        res = train_and_evaluate(config=cfg, force=False, calibrate=False, fit_calibration=True, save_artifacts=False)
        failed = [m for m in catch.msgs if "failed to load" in m.lower()]
        loaded = [m for m in catch.msgs if "Loading existing model weights" in m]
        if failed:
            raise RuntimeError(f"{rid}: weights not loaded ({failed})")
        # the load message is not always captured (logging is reconfigured inside train_and_evaluate);
        # the validation runs' CRPS against eval_report_test.json is the real check
        fr = PredictionFrame.from_result(res)
        arrays = {"y": fr.y, "anchor_seq": np.asarray(res.fold.test)}
        for i, h in enumerate(HS):
            arrays[f"delta_{h}"] = fr.delta[h]
            arrays[f"sigma_{h}"] = fr.sigma(h)
            arrays[f"crps_{h}"] = gaussian_crps(fr.y[:, i], fr.delta[h], np.maximum(fr.sigma(h), 1e-12))
        np.savez_compressed(dst, **arrays)
        ev = json.loads((rd / "eval_report_test.json").read_text(encoding="utf-8"))
        checks[rid] = {h: {"reproduced_crps": float(arrays[f"crps_{h}"].mean()),
                           "eval_report_crps": ev["model"]["horizons"][h]["variance"]["crps"]} for h in HS}
        checks[rid]["validation_run"] = rid in VALIDATE
        checks[rid]["load_message_captured"] = bool(loaded)
        print(rid, {h: (round(checks[rid][h]["reproduced_crps"], 3), round(checks[rid][h]["eval_report_crps"], 3)) for h in HS},
              "VALIDATION" if rid in VALIDATE else "", flush=True)
    return checks


def load_preds():
    import pandas as pd
    df = pd.read_csv(GRID / "results.csv")
    sel = df[(df.period == "P1") & (df.condition.isin(CONDS))]
    runs = {}
    for _, r in sel.iterrows():
        z = np.load(PRED_DIR / f"{r['run_id']}.npz")
        runs[(r["condition"], int(r["seed"]))] = {k: z[k] for k in z.files}
    return runs


def _d(r1, r2, h, b):
    """ln(mean CRPS r2) - ln(mean CRPS r1) on slice b; h='mean' averages the three horizons' log ratios."""
    hs = HS if h == "mean" else (h,)
    return float(np.mean([math.log(r2[f"crps_{k}"][b].mean()) - math.log(r1[f"crps_{k}"][b].mean()) for k in hs]))


def analyse(runs):
    out = {}
    n = len(next(iter(runs.values()))["y"])
    for h in HS + ("mean",):
        res_h = {}
        for L in LENGTHS:
            nb = n // L
            # the last block of length 7236 is the whole block
            blocks = [slice(b * L, (b + 1) * L) for b in range(nb)]
            null_d, cross_d = [], []
            for b in blocks:
                for c in CONDS:                                       # same condition, different seeds
                    for s1, s2 in itertools.combinations(range(3), 2):
                        null_d.append(_d(runs[(c, s1)], runs[(c, s2)], h, b))
                for c1, c2 in itertools.combinations(CONDS, 2):       # different condition, same seed
                    for s in range(3):
                        cross_d.append(_d(runs[(c1, s)], runs[(c2, s)], h, b))
            # null pairs: E[d] = 0 by symmetry, so the RMS is the SD of a nominal-seed paired difference
            res_h[str(L)] = {"n_blocks": nb,
                             "sd_null_pairs(nominal seed)": float(np.sqrt(np.mean(np.square(null_d)))),
                             "sd_cross_condition_same_seed": float(np.std(cross_d, ddof=1)),
                             "n_null": len(null_d), "n_cross": len(cross_d)}
        if h == "mean":
            out[h] = res_h
            continue
        # within-run block bootstrap (80-bar blocks, D-012) of d for the whole block, null pairs
        rng = np.random.default_rng(0)
        boot_sd = []
        for c in CONDS:
            for s1, s2 in itertools.combinations(range(3), 2):
                a, bb = runs[(c, s1)][f"crps_{h}"], runs[(c, s2)][f"crps_{h}"]
                nblk = n // 80
                A = a[: nblk * 80].reshape(nblk, 80)
                B = bb[: nblk * 80].reshape(nblk, 80)
                ds = []
                for _ in range(400):
                    idx = rng.integers(0, nblk, nblk)
                    ds.append(math.log(B[idx].mean()) - math.log(A[idx].mean()))
                boot_sd.append(np.std(ds, ddof=1))
        res_h["within_run_block_bootstrap_sd_of_d_whole_block(80-bar blocks)"] = float(np.sqrt(np.mean(np.square(boot_sd))))
        out[h] = res_h
    return out


def main():
    SCRATCH.mkdir(exist_ok=True)
    os.chdir(SCRATCH)
    checks = predict_runs() if "--analyse-only" not in sys.argv else {}
    runs = load_preds()
    res = {"checks": checks, "scaling": analyse(runs),
           "note": "DEV block of fold -2 (v1 P1). Served predictions of the checkpoint (best val_loss) weights; "
                   "calibration refit on the cal block. d = ln(mean CRPS run2) - ln(mean CRPS run1) on the same "
                   "block. The training-block length is NOT varied here (still 22,977 anchors)."}
    if OUT.exists() and not checks:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        res["checks"] = old.get("checks", {})
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for h in HS + ("mean",):
        print(h, {L: (round(v["sd_null_pairs(nominal seed)"], 4), round(v["sd_cross_condition_same_seed"], 4), v["n_blocks"])
                  for L, v in res["scaling"][h].items() if isinstance(v, dict)},
              "boot", round(res["scaling"][h].get("within_run_block_bootstrap_sd_of_d_whole_block(80-bar blocks)", float("nan")), 4))


if __name__ == "__main__":
    main()

"""Direction-skill experiments (milestone M3): which change lets the direction heads learn?

    python scripts/direction_experiments.py --out runs/experiments/direction_v1 [--only NAME ...] [--jobs 2]

Each experiment is a Config override set trained in its own process on a DEVELOPMENT fold
(FOLD_INDEX -3 or -2); choices are made on that fold's test block, never on fold -1 (the block
every gate reports). Scores: direction AUC/MCC per horizon with the deadband mask, next to a
logistic regression on trailing returns fitted on the same fold's train block - the evidence
that the window carries direction signal in that period at all. (The calibration block, 2,880
sequences = ~190 independent h1 outcomes, is too small to rank models by AUC.)
Results: <out>/<name>/result.json and <out>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# name -> (overrides, epochs). Development happens on walk-forward folds -3 and -2 (their test
# blocks are the dev sets); fold -1 - the block every gate reports - is not used for choices.
# BATCH_SIZE 256 keeps iterations fast (the step is launch-latency bound on this GPU: ~200 ms per
# step at batch 64 or 256); the chosen setting is re-checked at the default 64.
B = {"BATCH_SIZE": 256}
EXPERIMENTS = {}
for _fold in (-3, -2):
    _f = {**B, "FOLD_INDEX": _fold}
    EXPERIMENTS.update({
        f"legacy_focal_dice_f{_fold}": ({**_f, "DIRECTION_LOSS": "focal_dice"}, 20),
        f"bce_f{_fold}": ({**_f}, 20),
        f"bce_no_soft_ece_f{_fold}": ({**_f, "LAMBDA_SOFT_ECE": 0.0}, 20),
    })


def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _mcc(y, p):
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y, (p > 0.5).astype(int)))


def score(frame, deadband_bps, baseline_frame=None):
    import numpy as np

    from neural_trade.evaluation.frame import HORIZONS
    from neural_trade.metrics.direction_labels import direction_labels_np

    labels = direction_labels_np(frame.y, frame.last_close, deadband_bps)
    out = {}
    for i, h in enumerate(HORIZONS):
        lab, mask = labels[h]
        p = frame.direction_prob[h][mask]
        g = frame.gauss_prob(h, deadband_bps)[mask]
        y = lab[mask].astype(int)
        row = {"auc": _auc(y, p), "mcc": _mcc(y, p), "gauss_auc": _auc(y, g), "p_std": float(np.std(p)),
               "pred_up_rate": float(np.mean(p > 0.5)), "delta_corr": float(np.corrcoef(frame.delta[h],
                                                                                        frame.y[:, i])[0, 1])}
        if baseline_frame is not None:
            row["logreg_auc"] = _auc(y, baseline_frame.direction_prob[h][mask])
        out[h] = row
    return out


def run_one(name, overrides, epochs, out_dir, csv):
    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.evaluation.baselines import BaselineSet
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.training.trainer import train_and_evaluate

    cfg = Config().override(CSV_PATH=csv, **overrides)
    ctx = RunContext.create(cfg, root=out_dir / "runs", name=name, tags=["direction_experiments", name])
    t0 = time.time()
    result = train_and_evaluate(config=ctx.config, run_context=ctx, epochs=epochs, force=True, calibrate=True,
                                fit_calibration=True, save_artifacts=False)
    wall = time.time() - t0
    blocks = split_arrays(ctx.config)
    base = BaselineSet.fit(blocks["train"]["X"], blocks["train"]["y"], blocks["train"]["last_close"],
                           ctx.config.DIR_DEADBAND_BPS)
    cal = PredictionFrame.from_result(result, "cal", X_raw=blocks["cal"]["X"])
    test = PredictionFrame.from_result(result, "test", X_raw=blocks["test"]["X"])
    hist = result.history.history
    res = {"name": name, "overrides": overrides, "epochs_run": len(hist["loss"]), "wall_s": wall,
           "run_id": ctx.run_id, "cal": score(cal, ctx.config.DIR_DEADBAND_BPS, base.predict(cal)["logreg_lags"]),
           "test": score(test, ctx.config.DIR_DEADBAND_BPS, base.predict(test)["logreg_lags"]),
           "val_dir_mcc_h1": hist.get("val_dir_mcc_h1"), "val_loss": hist.get("val_loss")}
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def summarize(out_dir):
    rows = []
    for p in sorted(out_dir.glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["test"]
        rows.append(f"| `{r['name']}` | {r['epochs_run']} | " + " | ".join(
            f"{c[h]['auc']:.4f} / {c[h]['logreg_auc']:.4f}" for h in ("h0", "h1", "h2")) +
            f" | {c['h1']['mcc']:+.4f} | {c['h1']['p_std']:.4f} | {c['h1']['delta_corr']:+.4f} | "
            f"{r['wall_s'] / 60:.0f} | `{r['run_id']}` |")
    text = "\n".join([
        "# Direction experiments - development folds", "",
        "AUC with the 5 bps deadband mask on each development fold's own test block, model / logistic regression",
        "on trailing returns fit on the same fold's train block. Fold -1 (the reported test block) is not used.", "",
        "| experiment | epochs | AUC h0 | AUC h1 | AUC h2 | MCC h1 | P(up) std h1 | delta corr h1 | min | run |",
        "|---|---|---|---|---|---|---|---|---|---|", *rows, ""])
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "runs" / "experiments" / "direction_v1"))
    ap.add_argument("--csv", default=str(REPO / "binance_btcusdt_1min_ccxt.csv"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.one:
        overrides, epochs = EXPERIMENTS[args.one]
        run_one(args.one, overrides, epochs, out, args.csv)
        return 0
    names = [n for n in (args.only or EXPERIMENTS) if not (out / n / "result.json").exists()]
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true")
    running = []
    (out / "logs").mkdir(parents=True, exist_ok=True)
    while names or running:
        while names and len(running) < args.jobs:
            n = names.pop(0)
            fh = open(out / "logs" / f"{n}.log", "w", encoding="utf-8", errors="replace")
            running.append((n, fh, subprocess.Popen([sys.executable, __file__, "--out", str(out), "--csv", args.csv,
                                                     "--one", n], stdout=fh, stderr=subprocess.STDOUT, env=env)))
            print(f"[direction] start {n}", flush=True)
        time.sleep(5)
        for item in list(running):
            n, fh, proc = item
            if proc.poll() is not None:
                fh.close()
                running.remove(item)
                print(f"[direction] {n} exit {proc.returncode}", flush=True)
    print(summarize(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

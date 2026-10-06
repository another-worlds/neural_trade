"""Save the owner's top-3 candidates (owner 2026-09-30: "сохрани 1-3", the zero-cost ranking's rows 1-3) and prove
the saved copies reproduce their numbers.

    python configs/candidates/save_candidates.py          # copy (once), write the manifest, verify

All three sit on the same six models: the 360-day model of configs/scenarios/long_360d_stab.yaml (folds -3 / -2 x
seeds 0-2, trained at dce15ed, runs/scenarios/long_360d_stab/). Zero trading costs (D-044).
  C1  each model -> calibrated_quantile, entry_quantile 0.9, size 1.0     (ranking row 1)
  C2  each model -> calibrated_quantile, entry_quantile 0.95, size 1.0    (ranking row 2)
  C3  per fold, the mean of the three seeds' predictions -> calibrated_quantile 0.9, size 0.7   (ranking row 3)
Heavy files (weights, calibration, predictions) are copied to D:/neural_trade/saved_models/long_360d/ (git-ignored,
machine-local); configs/candidates/manifest.json (committed) records every file's sha256 and every candidate's
reproduced numbers.
"""
from __future__ import annotations

import glob
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import neural_trade  # noqa: F401
from neural_trade.experiments.scorer import BlockSignals, fit_and_backtest, load_block

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "runs/scenarios/long_360d_stab"
DST = REPO / "saved_models/long_360d"
MANIFEST = Path(__file__).with_name("manifest.json")
ZERO = {"fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0, "random_seeds": 20}
CANDIDATES = {
    "C1": {"rank": 1, "mode": "per_model", "params": {"entry_quantile": 0.9, "size": 1.0}},
    "C2": {"rank": 2, "mode": "per_model", "params": {"entry_quantile": 0.95, "size": 1.0}},
    "C3": {"rank": 3, "mode": "seed_ensemble", "params": {"entry_quantile": 0.9, "size": 0.7}},
}
KEEP = ("predictions_cal.npz", "predictions_oos.npz", "config.yaml", "meta.json", "result.json", "status.json",
        "eval_report_dev.md", "eval_report_dev.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_cells() -> dict:
    files = {}
    for d in sorted(glob.glob(str(SRC / "2026*"))):
        src = Path(d)
        dst = DST / src.name
        (dst / "artifacts").mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            if (src / name).exists() and not (dst / name).exists():
                shutil.copy2(src / name, dst / name)
        if not any((dst / "artifacts").iterdir()):
            shutil.copytree(src / "artifacts", dst / "artifacts", dirs_exist_ok=True)
        for f in sorted(dst.rglob("*")):
            if f.is_file():
                rel = f.relative_to(DST).as_posix()
                files[rel] = sha(f)
                assert sha(src / f.relative_to(dst)) == files[rel], f"copy differs: {rel}"
    return files


def summarise(bt) -> dict:
    s = bt.summary
    return {"return": round(float(s["total_return"]), 4), "win_share": round(float(s["hit_rate"]), 4),
            "max_drawdown": round(float(s["max_drawdown"]), 4), "sharpe_net": round(float(s["sharpe_net"]), 2),
            "trades": int(s["n_trades"]),
            "buy_and_hold": round(float(bt.baselines["buy_and_hold"]["total_return"]), 4)}


def mean_frame(frames):
    from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
    f0 = frames[0]
    avg = lambda a: {h: np.mean([getattr(f, a)[h] for f in frames], axis=0) for h in HORIZONS}  # noqa: E731
    cal = avg("direction_prob_calibrated") if all(f.direction_prob_calibrated is not None for f in frames) else None
    out = PredictionFrame(f0.y, f0.last_close, avg("delta"), avg("direction_prob"), avg("variance_scaled"),
                          f0.pred_scale, f0.pred_mean, f0.horizon_steps, f0.split, cal)
    out.meta = dict(getattr(f0, "meta", {}) or {})
    return out


def evaluate() -> dict:
    cells = sorted(p for p in DST.iterdir() if p.is_dir())
    out = {}
    for name, cand in CANDIDATES.items():
        rows = []
        if cand["mode"] == "per_model":
            for c in cells:
                cal, _, _ = load_block(c / "predictions_cal.npz")
                oos, bars, extra = load_block(c / "predictions_oos.npz")
                bt, _ = fit_and_backtest(BlockSignals.build(cal, oos), bars, strategy="calibrated_quantile",
                                         strategy_params=cand["params"], backtest_params=ZERO,
                                         bar_minutes=float(extra["bar_minutes"]))
                rows.append({"cell": c.name, **summarise(bt)})
        else:
            for fold in ("f-3", "f-2"):
                group = [c for c in cells if f"__{fold}__" in c.name]
                cals = [load_block(c / "predictions_cal.npz")[0] for c in group]
                oos_bars = [load_block(c / "predictions_oos.npz") for c in group]
                bars = oos_bars[0][1]
                bar_minutes = float(oos_bars[0][2]["bar_minutes"])
                sig = BlockSignals.build(mean_frame(cals), mean_frame([o[0] for o in oos_bars]))
                bt, _ = fit_and_backtest(sig, bars, strategy="calibrated_quantile", strategy_params=cand["params"],
                                         backtest_params=ZERO, bar_minutes=bar_minutes)
                rows.append({"cell": f"ensemble_{fold}_seeds0-2", **summarise(bt)})
        ret = [r["return"] for r in rows]
        out[name] = {**cand, "cells": rows,
                     "mean_return": round(float(np.mean(ret)), 4), "profitable_cells": f"{sum(r > 0 for r in ret)}/{len(ret)}",
                     "mean_win_share": round(float(np.mean([r["win_share"] for r in rows])), 4),
                     "max_drawdown": round(float(max(r["max_drawdown"] for r in rows)), 4)}
    return out


def main() -> None:
    files = copy_cells()
    results = evaluate()
    MANIFEST.write_text(json.dumps({"source": "runs/scenarios/long_360d_stab (dce15ed)", "saved_to": "saved_models/long_360d",
                                    "costs": "zero (D-044)", "files_sha256": files, "candidates": results}, indent=2),
                        encoding="utf-8")
    for k, v in results.items():
        print(k, {x: v[x] for x in ("mean_return", "profitable_cells", "mean_win_share", "max_drawdown")})


if __name__ == "__main__":
    main()

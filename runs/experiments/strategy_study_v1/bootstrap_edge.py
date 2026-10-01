"""Strategy study v1 (NT-005 (b)): mean gross edge per trade in bps, with an 80-bar block-bootstrap 95% CI,
for every discrete candidate on the dev cells, against the 26 bps round trip.

    python runs/experiments/strategy_study_v1/bootstrap_edge.py <rescore dir>   # writes <rescore dir>/gross_edge_bootstrap.json

Re-runs each discrete configuration on each dev cell from the stored predictions (the rescore's own path:
fit on cal, backtest the out-of-sample block), takes every trade's gross P&L over its entry notional, and
resamples (cell, entry_bar // 80) blocks with replacement (D-012: consecutive bars share their targets).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import neural_trade  # noqa: F401  (CUDA DLLs, before anything imports TF)
from neural_trade.experiments.rescore import StrategyStudy
from neural_trade.experiments.scorer import BlockSignals, fit_and_backtest, load_block

BLOCK = 80
REPS = 5000
COST_BPS = 26.0
EXPOSURE = {"vol_target", "net_edge_kelly"}


def main(rescore_dir: str) -> None:
    out = Path(rescore_dir)
    study = StrategyStudy.from_yaml(out / "study.yaml")
    cells = pd.read_csv(out / "cells.csv")
    dev_dirs = sorted(set(cells.loc[cells.role == "dev", "run_dir"]))
    blocks = {}
    for d in dev_dirs:
        p = next(q for q in (Path("runs") / d, Path(d)) if (q / "predictions_oos.npz").exists())
        cal, _, _ = load_block(p / "predictions_cal.npz")
        oos, bars, extra = load_block(p / "predictions_oos.npz")
        blocks[d] = (BlockSignals.build(cal, oos), bars, float(extra["bar_minutes"]))
    rng = np.random.default_rng(0)
    result = {}
    for conf in study.configurations():
        if conf.strategy in EXPOSURE or "_ewma" in conf.id:
            continue
        rows = []
        for d, (sig, bars, bar_minutes) in blocks.items():
            res, _ = fit_and_backtest(sig, bars, strategy=conf.strategy, strategy_params=conf.params,
                                      backtest_params={**conf.backtest, "random_seeds": 0},
                                      bar_minutes=bar_minutes)
            rows += [(d, t.entry_bar // BLOCK, 1e4 * t.gross_pnl / t.notional) for t in res.trades if t.notional > 0]
        if not rows:
            result[conf.id] = {"n_trades": 0}
            continue
        df = pd.DataFrame(rows, columns=["cell", "block", "bps"])
        groups = [g.bps.to_numpy() for _, g in df.groupby(["cell", "block"])]
        sums = np.array([g.sum() for g in groups])
        counts = np.array([len(g) for g in groups])
        idx = rng.integers(0, len(groups), size=(REPS, len(groups)))
        boot = sums[idx].sum(1) / counts[idx].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        result[conf.id] = {"strategy": conf.strategy, "n_trades": int(len(df)), "n_blocks": len(groups),
                           "mean_gross_bps": float(df.bps.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
                           "ci_excludes_cost": bool(hi < COST_BPS)}
    (out / "gross_edge_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, v in result.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main(sys.argv[1])

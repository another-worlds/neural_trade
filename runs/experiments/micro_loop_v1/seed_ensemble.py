"""Z2: a seed ensemble of the 360-day model at zero cost (D-044). For each dev fold of long_360d_stab, average the
three seeds' stored predictions (every head, calibration and out-of-sample blocks), then trade calibrated_quantile
with the scorer's own fit-on-cal path. Pre-registered pass (the owner's target, unchanged): on EACH fold, win share
>= 0.60, net return > 0, max drawdown < 5%, at position size 1.0; the same numbers at size 0.7 are reported as
information only. Entry quantiles 0.9 / 0.95 / 0.99 reported; the pass is judged on 0.9 (the Z1 rule's config).

    python runs/experiments/micro_loop_v1/seed_ensemble.py    # writes seed_ensemble.json next to this file
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import neural_trade  # noqa: F401
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.experiments.scorer import BlockSignals, fit_and_backtest, load_block

ROOT = Path("D:/neural_trade/runs/scenarios/long_360d_stab")
OUT = Path(__file__).with_name("seed_ensemble.json")
ZERO = {"fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0, "random_seeds": 20}


def mean_frame(frames):
    f0 = frames[0]
    avg = lambda attr: {h: np.mean([getattr(f, attr)[h] for f in frames], axis=0) for h in HORIZONS}  # noqa: E731
    cal = None
    if all(f.direction_prob_calibrated is not None for f in frames):
        cal = avg("direction_prob_calibrated")
    out = PredictionFrame(f0.y, f0.last_close, avg("delta"), avg("direction_prob"), avg("variance_scaled"),
                          f0.pred_scale, f0.pred_mean, f0.horizon_steps, f0.split, cal)
    out.meta = dict(getattr(f0, "meta", {}) or {})
    return out


def main() -> None:
    res = {"rows": []}
    for fold in (-3, -2):
        dirs = sorted(glob.glob(str(ROOT / f"*__f{fold}__s*")))
        cals, oos = [], []
        for d in dirs:
            c, _, _ = load_block(Path(d) / "predictions_cal.npz")
            o, bars, extra = load_block(Path(d) / "predictions_oos.npz")
            cals.append(c)
            oos.append(o)
        assert all(np.allclose(o.last_close, oos[0].last_close) for o in oos), "blocks differ across seeds"
        sig = BlockSignals.build(mean_frame(cals), mean_frame(oos))
        bar_minutes = float(extra["bar_minutes"])
        for q in (0.9, 0.95, 0.99):
            for size in (1.0, 0.7):
                bt, _ = fit_and_backtest(sig, bars, strategy="calibrated_quantile",
                                         strategy_params={"entry_quantile": q, "size": size}, backtest_params=ZERO,
                                         bar_minutes=bar_minutes)
                s = bt.summary
                row = {"fold": fold, "n_seeds": len(dirs), "entry_quantile": q, "size": size,
                       "return": round(float(s["total_return"]), 4), "sharpe_net": round(float(s["sharpe_net"]), 2),
                       "win_share": round(float(s["hit_rate"]), 4), "max_drawdown": round(float(s["max_drawdown"]), 4),
                       "trades": int(s["n_trades"]),
                       "buy_and_hold": round(float(bt.baselines["buy_and_hold"]["total_return"]), 4)}
                row["target_met"] = bool(row["win_share"] >= 0.60 and row["return"] > 0 and row["max_drawdown"] < 0.05)
                res["rows"].append(row)
                print(json.dumps(row))
    q9 = [r for r in res["rows"] if r["entry_quantile"] == 0.9 and r["size"] == 1.0]
    res["pass_pre_registered"] = bool(all(r["target_met"] for r in q9))
    print("PASS" if res["pass_pre_registered"] else "FAIL")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

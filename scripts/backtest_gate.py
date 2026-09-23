"""Backtest a gate run's saved test predictions with every registered strategy.

    python scripts/backtest_gate.py runs/gates/m3

Rebuilds the run's splits from its config (deterministic), takes the TEST block's decision bars
from the OHLC data, sets var_scale from the CAL predictions, runs each strategy with the default
costs and writes ``backtest.json`` into the run directory. ``analytics.json`` gets
``total_trades`` (the default strategy's trade count) so scripts/check_gates.py can judge M3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--csv", default=str(REPO / "binance_btcusdt_1min_ccxt.csv"))
    ap.add_argument("--random-seeds", type=int, default=100)
    ap.add_argument("--uncalibrated", action="store_true", help="use raw instead of calibrated P(up)")
    args = ap.parse_args(argv)

    import numpy as np

    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.strategy import (BacktestConfig, Bars, SignalFrame, Strategies, backtest, build_strategy,
                                       var_scale_from)

    run = Path(args.run_dir)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    analytics = json.loads((run / "analytics.json").read_text(encoding="utf-8"))
    config = Config().override(CSV_PATH=args.csv, **meta.get("overrides", {}))
    scale, mean = float(analytics["pred_scale"]), float(analytics["pred_mean"])
    hs = tuple(config.HORIZON_STEPS)

    test = PredictionFrame.from_npz(run / "predictions_test.npz", scale, mean, hs, "test")
    cal = PredictionFrame.from_npz(run / "predictions_cal.npz", scale, mean, hs, "cal")
    blocks = split_arrays(config)
    bars = Bars.from_frame(blocks["df"], blocks["test"]["anchor_bar"])
    if len(bars) != len(test) or not np.allclose(bars.close, test.last_close, rtol=1e-6):
        raise SystemExit(f"test bars do not line up with the saved predictions ({len(bars)} vs {len(test)})")

    var_scale = var_scale_from(cal)
    signals = SignalFrame.build(test, var_scale, calibrated=not args.uncalibrated)
    cal_signals = SignalFrame.build(cal, var_scale, calibrated=not args.uncalibrated)
    cfg = BacktestConfig(random_seeds=args.random_seeds)
    out = {"run": run.name, "var_scale": var_scale, "calibrated": not args.uncalibrated, "n_bars": len(bars),
           "strategies": {}}
    for name in Strategies.list_names():
        if "baseline" in Strategies.get_metadata(name).get("tags", []):
            continue
        res = backtest(signals, bars, build_strategy(name, calibration=cal_signals), cfg)
        out["strategies"][name] = res.to_dict()
        s = res.summary
        rnd = res.baselines["random_same_freq"]
        print(f"{name:24s} trades {s['n_trades']:5d}  net {100 * s['total_return']:+7.2f}%  "
              f"gross {100 * s['gross_pnl'] / cfg.initial_equity:+7.2f}%  sharpe {s['sharpe_net']:+6.2f}  "
              f"maxDD {100 * s['max_drawdown']:5.2f}%  hit {s['hit_rate']:.2f}  "
              f"random pct {rnd.get('percentile_total_return', float('nan')):.0f}")
    bh = out["strategies"][Strategies.default]["baselines"]["buy_and_hold"]
    print(f"{'buy_and_hold':24s} net {100 * bh['total_return']:+7.2f}%")
    (run / "backtest.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    default = out["strategies"][Strategies.default]["summary"]
    analytics["total_trades"] = int(default["n_trades"])
    analytics["total_trades_note"] = (f"{Strategies.default} strategy, default costs, scripts/backtest_gate.py; "
                                      "all strategies in backtest.json")
    analytics["total_trades_by_strategy"] = {k: int(v["summary"]["n_trades"]) for k, v in out["strategies"].items()}
    (run / "analytics.json").write_text(json.dumps(analytics, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

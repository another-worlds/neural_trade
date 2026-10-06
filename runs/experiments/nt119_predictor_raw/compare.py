"""NT-119 (3): trade and INCOH counts on a real run, before (no raw heads on the frame) and after.

    CUDA_VISIBLE_DEVICES=-1 python compare.py <run_dir> <csv> <out.json>

"before" is the frame the Predictor path built until NT-119 (no meta['delta_raw']): the coherence flags
fall back to the served delta. "after" is Predictor.to_prediction_frame as it is now (raw heads carried).
CPU inference only; the backtest is the cli backtest path (default params, costs 0 per D-044).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from neural_trade.serving.predictor import Predictor
from neural_trade.strategy import (Bars, SignalFrame, Strategies, backtest, build_backtest_config, build_strategy)

run_dir, csv, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
pred = Predictor.from_artifacts(run_dir / "artifacts")
batch, df, anchors = pred.predict_windows_frame(pd.read_csv(csv), batch_size=2048)
frame = batch.to_prediction_frame(pred.bundle.pred_scale, pred.bundle.pred_mean)
var_scale = float(pred.bundle.meta["var_scale"])
bars = Bars.from_frame(df, anchors)
bcfg = build_backtest_config({"random_seeds": 0, "bar_minutes": float(pred.config.RESAMPLE_MINUTES)})
cal = pred.bundle.meta.get("weighted_direction_quantiles")

res = {"run": run_dir.name, "n_windows": int(len(frame)),
       "betas": {h: float(b) for h, b in (pred.bundle.calibration_pipeline.delta_scale or {}).items()}}
raw = frame.meta.pop("delta_raw")
for variant in ("before", "after"):
    if variant == "after":
        frame.meta["delta_raw"] = raw
    s = SignalFrame.build(frame, var_scale)
    row = {"coherence_on_raw": bool(s.coherence_on_raw), "bars_magnitude_coherent": int(s.magnitude_coherent.sum()),
           "bars_direction_aligned": int(s.direction_aligned.sum()), "strategies": {}}
    for name in ("enhanced_multi_horizon", "liberal", Strategies.default):
        r = backtest(s, bars, build_strategy(name, None, calibration=cal), bcfg)
        tf_ = r.trades_frame()
        col = next((c for c in ("exit_reason", "reason", "exit_type") if c in tf_.columns), None)
        incoh = int((tf_[col] == "INCOH").sum()) if col and len(tf_) else 0
        row["strategies"][name] = {"n_trades": int(r.summary["n_trades"]), "incoh": incoh}
    res[variant] = row
Path(out).write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))

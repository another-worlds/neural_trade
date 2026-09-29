"""Benchmark driver for the NEW code (f39593a): the same path as `neural-trade train`.
Usage: python bench_new.py BATCH_SIZE [EPOCHS]. Run with PYTHONPATH=D:/nt_bench_new/src from D:/nt_bench_new."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
import neural_trade  # noqa: E402,F401  (CUDA DLLs on PATH)
import tensorflow as tf  # noqa: E402

from bench_common import Dmon, make_timer, summarize  # noqa: E402

bs = int(sys.argv[1])
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
tag = f"new_bs{bs}"
out_root = "D:/nt_bench_runs/new"
dmon = Dmon(f"D:/nt_bench_runs/logs/dmon_{tag}.txt")
dmon.start()
try:
    from neural_trade.core.config import Config
    from neural_trade.evaluation.baselines import BaselineSet
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.evaluation.report import evaluate
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.registries import load_all
    from neural_trade.training.trainer import train_and_evaluate

    print("neural_trade from", neural_trade.__file__)
    cfg = Config().override(BATCH_SIZE=bs, EPOCHS=epochs)
    load_all(cfg, plugins_dir=getattr(cfg, "PLUGINS_DIR", None), strict=False)
    ctx = RunContext.create(cfg, root=out_root, seed=None, tags=["bench"], name=tag)
    timer = make_timer(tf)
    result = train_and_evaluate(config=ctx.config, run_context=ctx, epochs=epochs, force=True, calibrate=True,
                                fit_calibration=True, save_artifacts=True, extra_callbacks=[timer])
    t_after = time.time()
    test = PredictionFrame.from_result(result, "test")
    cal = PredictionFrame.from_result(result, "cal") if result.predictions_cal is not None else None
    from neural_trade.data.processor import split_arrays
    train = split_arrays(ctx.config)["train"]
    baselines = BaselineSet.fit(train["X"], train["y"], train["last_close"], float(ctx.config.DIR_DEADBAND_BPS))
    report = evaluate(test, ctx.config, baselines=baselines, cal_frame=cal, run_id=ctx.run_id)
    report.to_json(ctx.path("eval_report_test.json"))
    t_end = time.time()
    summary = summarize(tag, timer, t0, timer.t_train_begin, t_after, t_end,
                        extra={"run_dir": str(ctx.run_dir), "n_train_windows": int(train["X"].shape[0]),
                               "steady_windows": [[e["t_begin"], e.get("t_train_end")] for e in timer.epochs[1:]]})
    with open(f"D:/nt_bench_runs/{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "epochs"}, indent=2, default=str))
finally:
    dmon.stop()

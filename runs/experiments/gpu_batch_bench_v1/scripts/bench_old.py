"""Benchmark driver for the OLD code (master 7002a71, single-file model.py).
Usage: python bench_old.py BATCH_SIZE [EPOCHS] [MAX_SEQUENCE_COUNT]. cwd is set to D:/nt_bench_runs/old."""
import json
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
t0 = time.time()
import neural_trade  # noqa: E402,F401  (only for the CUDA DLL PATH)
import tensorflow as tf  # noqa: E402

from bench_common import Dmon, make_timer, summarize  # noqa: E402

bs = int(sys.argv[1])
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
max_seq = int(sys.argv[3]) if len(sys.argv) > 3 else 36000
tag = f"old_bs{bs}"
os.makedirs(os.path.join(here, "old"), exist_ok=True)
os.chdir(os.path.join(here, "old"))
sys.path.insert(0, "D:/nt_bench_old")
dmon = Dmon(f"D:/nt_bench_runs/logs/dmon_{tag}.txt")
dmon.start()
try:
    import model  # noqa: E402  the old single file

    print("old model from", model.__file__, "tf", tf.__version__)
    timer = make_timer(tf)
    overrides = {"CSV_PATH": "D:/nt_bench_runs/binance_ts.csv", "BATCH_SIZE": bs, "EPOCHS": epochs,
                 "PATIENCE": epochs, "MAX_SEQUENCE_COUNT": max_seq,
                 "MODEL_PATH": f"old_{tag}.weights.h5", "SCALER_PATH": f"old_{tag}_scaler.joblib"}
    for p in (overrides["MODEL_PATH"],):
        if os.path.exists(p):
            os.remove(p)
    result = model.train_and_evaluate(config=model.Config(), config_overrides=overrides, epochs=epochs,
                                      force=True, calibrate=True, extra_callbacks=[timer])
    t_after = time.time()
    t_end = time.time()
    summary = summarize(tag, timer, t0, timer.t_train_begin, t_after, t_end,
                        extra={"n_train_windows": int(len(result.y_test) * 4),
                               "n_test_windows": int(len(result.y_test)),
                               "steady_windows": [[e["t_begin"], e.get("t_train_end")] for e in timer.epochs[1:]]})
    with open(f"D:/nt_bench_runs/{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "epochs"}, indent=2, default=str))
finally:
    dmon.stop()

"""Single-series (batch 1) wall-clock latency on this CPU: the live-trading case for NFR-1.1 (< 1 s).

Usage: LATENCY_GO=1 python latency.py timestral | chronos2
Package defaults (fp32, no torch.compile), torch default threads. Per config: 3 warm-ups, then
repetitions until 200 calls or 60 s (minimum 20). Reports p50/p95/p99 ms, cold start
(load + first call) and peak RSS. Appends JSON lines to results_latency.jsonl.
"""
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import _SCRATCH, PRED_LEN, load  # noqa: E402

if os.environ.get("LATENCY_GO") != "1":
    sys.exit(0)          # run only on an idle box, after all benchmark queues
df, _ = load()
close = df["close"].to_numpy(np.float64)
OUT = f"{_SCRATCH}/results_latency.jsonl"


def bench(fn, max_reps=200, budget_s=60.0, min_reps=20, warm=3):
    for _ in range(warm):
        fn()
    ts, t_end = [], time.perf_counter() + budget_s
    while len(ts) < max_reps and (len(ts) < min_reps or time.perf_counter() < t_end):
        t = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t))
    p = np.percentile(ts, [50, 95, 99])
    return {"p50_ms": round(float(p[0]), 1), "p95_ms": round(float(p[1]), 1), "p99_ms": round(float(p[2]), 1),
            "reps": len(ts)}


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


rows = []
with torch.inference_mode():
    if sys.argv[1] == "timestral":
        from timestral import TimestralForecaster
        for size in ["5M", "28M", "123M", "209M"]:
            t0 = time.perf_counter()
            fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", local_files_only=True)
            fc.forecast(close[-2048:][None, :], PRED_LEN)
            cold = round(1000 * (time.perf_counter() - t0), 1)
            for ctx in [64, 512, 2048, 8192, 32768]:
                for sym in [True, False]:
                    x = close[-ctx:][None, :]
                    r = {"model": f"timestral-{size}", "ctx": ctx, "symmetric": sym,
                         **bench(lambda: fc.forecast(x, PRED_LEN, symmetric_averaging=sym)),
                         "cold_start_ms_ctx2048": cold, "peak_rss_mb": rss_mb(), "threads": torch.get_num_threads()}
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    elif sys.argv[1] == "chronos2":
        from chronos import BaseChronosPipeline
        t0 = time.perf_counter()
        p = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
        p.predict_quantiles(torch.tensor(close[-2048:][None, None, :], dtype=torch.float32),
                            prediction_length=PRED_LEN, quantile_levels=[0.1, 0.5, 0.9])
        cold = round(1000 * (time.perf_counter() - t0), 1)
        for ctx in [64, 512, 2048, 8192]:
            x = torch.tensor(close[-ctx:][None, None, :], dtype=torch.float32)
            r = {"model": "chronos-2 (official)", "ctx": ctx,
                 **bench(lambda: p.predict_quantiles(x, prediction_length=PRED_LEN, quantile_levels=[0.1, 0.5, 0.9])),
                 "cold_start_ms_ctx2048": cold, "peak_rss_mb": rss_mb(), "threads": torch.get_num_threads()}
            rows.append(r)
            print(json.dumps(r), flush=True)
with open(OUT, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

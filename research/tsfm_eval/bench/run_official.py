"""Zero-shot Chronos-2 / Chronos-Bolt through the official chronos-forecasting package (reference for Samay).

Usage: python run_official.py <model> <context>   model: chronos2 | bolt_small | bolt_base
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import HORIZONS, PRED_LEN, QLEVELS, RESULTS, load, origins  # noqa: E402
from chronos import BaseChronosPipeline  # noqa: E402

name, ctx = sys.argv[1], int(sys.argv[2])
batch = int(os.environ.get("BENCH_BATCH", "64"))
sub = int(os.environ.get("BENCH_SUBSTRIDE", "1"))
tag = f"official-{name}_ctx{ctx}_price" + (f"_sub{sub}" if sub > 1 else "")
repo = {"chronos2": "amazon/chronos-2", "bolt_small": "amazon/chronos-bolt-small",
        "bolt_base": "amazon/chronos-bolt-base"}[name]
df, _ = load()
close = df["close"].to_numpy(np.float64)
org = origins(len(df))[::sub]
p = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", torch_dtype=torch.float32)
hidx = [h - 1 for h in HORIZONS]
out_q = np.zeros((len(org), len(HORIZONS), 21), np.float32)
out_med = np.zeros((len(org), PRED_LEN), np.float32)
LV = list(QLEVELS)
t0 = time.perf_counter()
for i in range(0, len(org), batch):
    ob = org[i:i + batch]
    x = torch.tensor(close[np.stack([np.arange(t - ctx + 1, t + 1) for t in ob])], dtype=torch.float32)
    if name == "chronos2":
        ql, _ = p.predict_quantiles(x[:, None, :], prediction_length=PRED_LEN, quantile_levels=LV)
        q = torch.stack([t[0] for t in ql]).numpy()           # (B, H, 21)
    else:
        q, _ = p.predict_quantiles(x, prediction_length=PRED_LEN, quantile_levels=LV)
        q = q.numpy()                                          # Bolt: tails beyond q0.1/q0.9 are clamped
    out_q[i:i + len(ob)] = q[:, hidx, :]
    out_med[i:i + len(ob)] = q[:, :, 10]
elapsed = time.perf_counter() - t0
np.savez_compressed(f"{RESULTS}/{tag}.npz", q=out_q, med_path=out_med, origins=org, framing="price",
                    elapsed=elapsed, n=len(org), threads=torch.get_num_threads())
print(f"{tag}: {len(org)} origins in {elapsed:.1f}s ({1000 * elapsed / len(org):.1f} ms/series batched)")

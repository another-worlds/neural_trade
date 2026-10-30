"""Volatility role test: forecast mean 1-min high-low range (bps) over the next h minutes.

This targets the slot neural_trade fills with its variance head (Student-t NLL) and that
helper_functions.py uses for confidence, stop-loss and position size.

Usage: python run_vol.py <backend> <size_or_model> <context>
  backend: timestral (size 5M|28M|123M|209M) | chronos2 (official package; size arg ignored)
Saves results_vol/<tag>.npz with predicted mean range per horizon (median path averaged).
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import HORIZONS, PRED_LEN, QLEVELS, RESULTS, load, origins  # noqa: E402

backend, size, ctx = sys.argv[1], sys.argv[2], int(sys.argv[3])
sub = int(os.environ.get("BENCH_SUBSTRIDE", "1"))
batch = int(os.environ.get("BENCH_BATCH", "64"))
OUT = RESULTS + "_vol"
os.makedirs(OUT, exist_ok=True)
sym = "--no-sym" not in sys.argv
tag = f"{backend}-{size}_ctx{ctx}" + ("" if sym else "_nosym") + (f"_sub{sub}" if sub > 1 else "")
LV = np.array(QLEVELS)
# midpoint-rule weights over the 21 quantile levels: E[X] ~= sum_i w_i * Q(p_i)
_edges = np.concatenate([[0.0], (LV[1:] + LV[:-1]) / 2, [1.0]])
W_MEAN = np.diff(_edges)

df, _ = load()
rng_bps = np.log(df["high"].to_numpy(np.float64) / df["low"].to_numpy(np.float64)) * 1e4
org = origins(len(df))[::sub]

if backend == "timestral":
    from timestral import TimestralForecaster
    fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")

    def qpath(x):
        return fc.forecast(x, PRED_LEN, symmetric_averaging=sym).quantiles[:, 0].numpy()   # (B, H, 21)
elif backend == "chronos2":
    from chronos import BaseChronosPipeline
    p = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)

    def qpath(x):
        ql, _ = p.predict_quantiles(torch.tensor(x[:, None, :], dtype=torch.float32), prediction_length=PRED_LEN,
                                    quantile_levels=list(QLEVELS))
        return torch.stack([t[0] for t in ql]).numpy()
else:
    raise SystemExit(backend)

paths = np.zeros((len(org), PRED_LEN), np.float32)
mpaths = np.zeros((len(org), PRED_LEN), np.float32)
t0 = time.perf_counter()
with torch.inference_mode():
    for i in range(0, len(org), batch):
        ob = org[i:i + batch]
        q = qpath(rng_bps[np.stack([np.arange(t - ctx + 1, t + 1) for t in ob])])
        paths[i:i + len(ob)] = q[:, :, 10]
        mpaths[i:i + len(ob)] = q @ W_MEAN
elapsed = time.perf_counter() - t0
pred = np.stack([mpaths[:, :h].mean(axis=1) for h in HORIZONS], axis=1)          # quantile-integrated mean
pred_median = np.stack([paths[:, :h].mean(axis=1) for h in HORIZONS], axis=1)    # median path
np.savez_compressed(f"{OUT}/{tag}.npz", pred=pred, pred_median=pred_median, origins=org, elapsed=elapsed)
print(f"vol {tag}: {len(org)} origins in {elapsed:.1f}s")

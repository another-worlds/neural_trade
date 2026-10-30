"""Zero-shot Timestral forecasts at neural_trade horizons over a rolling chronological holdout.

Usage: python run_timestral.py <size> <context> <framing> [--no-sym]
  size:    5M | 28M | 123M | 209M
  framing: price      close price level, univariate
           aggret     per horizon h: non-overlapping h-min log returns (bps) ending at t, forecast 1 step
           logret     1-min log returns in bps, univariate (point/direction from summed medians)
           price_cov  close price level + past covariates (log volume, high-low range bps, log return bps)
Saves results/<tag>.npz with quantiles at HORIZONS: q (N, 3, 21), plus origins and timing.
"""
import sys
import time
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import HORIZONS, PRED_LEN, RESULTS, load, origins  # noqa: E402
from timestral import TimestralForecaster  # noqa: E402

size, ctx, framing = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sym = "--no-sym" not in sys.argv
batch = int(os.environ.get("BENCH_BATCH", "16" if framing == "price_cov" else "64"))  # covariate rows make variate attention O(rows^2)
sub = int(os.environ.get("BENCH_SUBSTRIDE", "1"))
tag = f"timestral-{size}_ctx{ctx}_{framing}" + ("" if sym else "_nosym") + (f"_sub{sub}" if sub > 1 else "")
os.makedirs(RESULTS, exist_ok=True)

df, _ = load()
close = df["close"].to_numpy(np.float64)
logret = np.zeros_like(close)
logret[1:] = np.log(close[1:] / close[:-1]) * 1e4
logvol = np.log1p(df["volume"].to_numpy(np.float64))
rng_bps = np.log(df["high"].to_numpy(np.float64) / df["low"].to_numpy(np.float64)) * 1e4
org = origins(len(df))[::sub]
assert org[0] - ctx + 1 >= 1, "context reaches before data start"

fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")
hidx = [h - 1 for h in HORIZONS]
out_q = np.zeros((len(org), len(HORIZONS), 21), np.float32)
out_med_path = np.zeros((len(org), PRED_LEN), np.float32)
t0 = time.perf_counter()
lc = np.log(close)
if framing == "aggret":
    assert org[0] - max(HORIZONS) * ctx >= 0, "context reaches before data start"
for i in range(0, len(org), batch):
    ob = org[i:i + batch]
    if framing == "aggret":       # per horizon: non-overlapping h-min log returns (bps) ending at t; forecast 1 step
        for hi, h in enumerate(HORIZONS):
            ends = ob[:, None] - h * np.arange(ctx - 1, -1, -1)[None, :]
            f = fc.forecast((lc[ends] - lc[ends - h]) * 1e4, 1, symmetric_averaging=sym)
            out_q[i:i + len(ob), hi] = f.quantiles[:, 0].numpy()[:, 0, :]
        continue
    sl = np.stack([np.arange(t - ctx + 1, t + 1) for t in ob])
    if framing == "price":
        f = fc.forecast(close[sl], PRED_LEN, symmetric_averaging=sym)
        q = f.quantiles[:, 0].numpy()                      # (B, H, Q) price levels
    elif framing == "price_cov":
        cov = np.stack([logvol[sl], rng_bps[sl], logret[sl]], axis=1)
        f = fc.forecast(close[sl][:, None, :], PRED_LEN, past_covariates=cov, symmetric_averaging=sym)
        q = f.quantiles[:, 0].numpy()
    elif framing == "logret":
        f = fc.forecast(logret[sl], PRED_LEN, symmetric_averaging=sym)
        q = f.quantiles[:, 0].numpy()                      # (B, H, Q) per-step bps returns
    else:
        raise SystemExit(f"unknown framing {framing}")
    out_q[i:i + len(ob)] = q[:, hidx, :]
    out_med_path[i:i + len(ob)] = q[:, :, 10]
    if i == 0:
        print(f"{tag}: first batch {time.perf_counter() - t0:.1f}s", flush=True)
elapsed = time.perf_counter() - t0
np.savez_compressed(f"{RESULTS}/{tag}.npz", q=out_q, med_path=out_med_path, origins=org,
                    framing=framing, elapsed=elapsed, n=len(org), threads=torch.get_num_threads())
print(f"{tag}: {len(org)} origins in {elapsed:.1f}s ({1000 * elapsed / len(org):.1f} ms/series batched)")

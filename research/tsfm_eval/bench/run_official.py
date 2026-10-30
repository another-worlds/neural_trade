"""Zero-shot Chronos-2 / Chronos-Bolt through the official chronos-forecasting package (reference for Samay).

Usage: python run_official.py <model> <context> [price|logret]   model: chronos2 | bolt_small | bolt_base
framing logret: the input is the 1-min log return in bps; the output is per-step return quantiles (only h=1 is scored
as a distribution, because quantiles of a multi-step sum cannot be built from per-step quantiles).
framing aggret: for each h, the input is the series of non-overlapping h-min log returns (bps) ending at the origin,
<context> bars long, and the model forecasts one step. That gives a return distribution for every horizon.
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
framing = sys.argv[3] if len(sys.argv) > 3 else "price"
assert framing in ("price", "logret", "aggret"), framing
batch = int(os.environ.get("BENCH_BATCH", "64"))
sub = int(os.environ.get("BENCH_SUBSTRIDE", "1"))
tag = f"official-{name}_ctx{ctx}_{framing}" + (f"_sub{sub}" if sub > 1 else "")
repo = {"chronos2": "amazon/chronos-2", "bolt_small": "amazon/chronos-bolt-small",
        "bolt_base": "amazon/chronos-bolt-base"}[name]
df, _ = load()
close = df["close"].to_numpy(np.float64)
logret = np.zeros_like(close)
logret[1:] = np.log(close[1:] / close[:-1]) * 1e4
series = close if framing == "price" else logret
org = origins(len(df))[::sub]
p = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", torch_dtype=torch.float32)
hidx = [h - 1 for h in HORIZONS]
out_q = np.zeros((len(org), len(HORIZONS), 21), np.float32)
out_med = np.zeros((len(org), PRED_LEN), np.float32)
LV = list(QLEVELS)
lc = np.log(close)
t0 = time.perf_counter()


def chronos_q(x, steps):
    if name == "chronos2":
        ql, _ = p.predict_quantiles(x[:, None, :], prediction_length=steps, quantile_levels=LV)
        return torch.stack([t[0] for t in ql]).numpy()
    q, _ = p.predict_quantiles(x, prediction_length=steps, quantile_levels=LV)
    return q.numpy()


if framing == "aggret":
    assert org[0] - max(HORIZONS) * ctx >= 0, "context reaches before data start"
    for i in range(0, len(org), batch):
        ob = org[i:i + batch]
        for hi, h in enumerate(HORIZONS):
            ends = ob[:, None] - h * np.arange(ctx - 1, -1, -1)[None, :]
            x = torch.tensor((lc[ends] - lc[ends - h]) * 1e4, dtype=torch.float32)
            out_q[i:i + len(ob), hi] = chronos_q(x, 1)[:, 0, :]
    org_loop = []
else:
    org_loop = range(0, len(org), batch)
for i in org_loop:
    ob = org[i:i + batch]
    x = torch.tensor(series[np.stack([np.arange(t - ctx + 1, t + 1) for t in ob])], dtype=torch.float32)
    if name == "chronos2":
        ql, _ = p.predict_quantiles(x[:, None, :], prediction_length=PRED_LEN, quantile_levels=LV)
        q = torch.stack([t[0] for t in ql]).numpy()           # (B, H, 21)
    else:
        q, _ = p.predict_quantiles(x, prediction_length=PRED_LEN, quantile_levels=LV)
        q = q.numpy()                                          # Bolt: tails beyond q0.1/q0.9 are clamped
    out_q[i:i + len(ob)] = q[:, hidx, :]
    out_med[i:i + len(ob)] = q[:, :, 10]
elapsed = time.perf_counter() - t0
np.savez_compressed(f"{RESULTS}/{tag}.npz", q=out_q, med_path=out_med, origins=org, framing=framing,
                    elapsed=elapsed, n=len(org), threads=torch.get_num_threads())
print(f"{tag}: {len(org)} origins in {elapsed:.1f}s ({1000 * elapsed / len(org):.1f} ms/series batched)")

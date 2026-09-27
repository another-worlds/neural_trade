"""float32 precision of the chunked two-level EWMA over the whole 30,393-bar block vs a float64 recursion,
on the real closes (window-free, shift by the first close, / scale). Also periods 60, 240, 1440."""
import os
import json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pandas as pd
import neural_trade  # noqa: F401
import tensorflow as tf

REPO = "C:/Users/Step/Documents/neural_trade"
close = pd.read_csv(os.path.join(REPO, "binance_btcusdt_1min_ccxt.csv"), usecols=["close"])["close"].to_numpy(np.float64)
scale = float(np.std(close[20:] - close[:-20]))
periods = np.array([5, 9, 12, 14, 20, 26, 30, 35, 60, 240, 1440], dtype=np.float64)
K = len(periods)
a64 = 2.0 / (periods + 1.0)
n = 30213 + 180
x64 = (close[-n:] - close[-n]) / scale
y64 = np.empty((n, K)); y64[0] = x64[0]
for t in range(1, n):
    y64[t] = a64 * x64[t] + (1 - a64) * y64[t - 1]


def ewma_chunked(x, a, C=64):
    nn = int(x.shape[0]); nc = -(-nn // C)
    xb = tf.reshape(tf.pad(x, [[0, nc * C - nn]]), [nc, C])
    la = tf.math.log1p(-a)
    t = tf.range(C, dtype=tf.float32); lag = t[:, None] - t[None, :]
    M = tf.where((lag >= 0)[None], tf.exp(tf.maximum(lag, 0.0)[None] * la[:, None, None]) * a[:, None, None], 0.0)
    yloc = tf.einsum("kts,js->jtk", M, xb)
    j = tf.range(nc, dtype=tf.float32); lagb = j[:, None] - j[None, :]
    M2 = tf.where((lagb >= 0)[None], tf.exp(tf.maximum(lagb, 0.0)[None] * (C * la)[:, None, None]), 0.0)
    S = tf.einsum("kji,ik->jk", M2, yloc[:, -1, :]) + tf.exp((j[:, None] + 1.0) * C * la[None, :]) * x[0]
    S_prev = tf.concat([tf.ones_like(S[:1]) * x[0], S[:-1]], 0)
    y = yloc + tf.exp((t[:, None] + 1.0) * la[None, :])[None] * S_prev[:, None, :]
    return tf.reshape(y, [nc * C, K])[:nn]


y = ewma_chunked(tf.constant(x64.astype(np.float32)), tf.constant(a64.astype(np.float32))).numpy()
err = np.abs(y - y64).max(0)
# the feature the network sees: EWMA minus the bar's own close
dev = np.abs(y64 - x64[:, None])
res = {"n_bars": n, "max_abs_x": float(np.abs(x64).max()),
       "max_abs_err_by_period": {f"p{int(p)}": float(e) for p, e in zip(periods, err)},
       "median_abs_(ewma-close)_by_period": {f"p{int(p)}": float(np.median(d)) for p, d in zip(periods, dev.T)}}
print(json.dumps(res, indent=1))

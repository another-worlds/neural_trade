"""CPU: scaling of the windowed EWMA (K x T^2 per window) vs the causal pass (K x n) as the catalogue
grows (K = 24 today -> ~87 estimated for 14 families x 3 instances) and as the window grows (T = 60 ->
240 for periods up to ~80 bars). Plus analytic window-start contamination. CUDA_VISIBLE_DEVICES=-1."""
import os
import time
import json
import math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import neural_trade  # noqa: F401
import tensorflow as tf
import neural_trade.utils.math as mh

rng = np.random.default_rng(0)
out = {}


def timeit(fn, args, reps=3, warm=1):
    for _ in range(warm):
        r = fn(*args)
    np.asarray(tf.nest.flatten(r)[0])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        _ = [np.asarray(t) for t in tf.nest.flatten(r)]
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def windowed(Bn, K, T):
    lv = tf.Variable(rng.normal(-2.5, 0.5, size=K).astype(np.float32))
    x = tf.constant(rng.normal(size=(Bn, T)).cumsum(1).astype(np.float32))
    w = tf.constant(rng.normal(size=(Bn, K, T)).astype(np.float32))

    @tf.function
    def fb(xx, ww):
        with tf.GradientTape() as tape:
            a = tf.sigmoid(lv)[None, :] + tf.zeros((Bn, K))
            y = mh.ewma_sequence_matrix_multi(tf.tile(xx[:, None, :], [1, K, 1]), a)
            loss = tf.reduce_sum(y * ww)
        return loss, tape.gradient(loss, lv)
    return timeit(fb, (x, w))


def chunked(K, n, C=64):
    lv = tf.Variable(rng.normal(-2.5, 0.5, size=K).astype(np.float32))
    x = tf.constant(rng.normal(size=n).cumsum().astype(np.float32))
    w = tf.constant(rng.normal(size=(n, K)).astype(np.float32))

    @tf.function
    def fb(xx, ww):
        with tf.GradientTape() as tape:
            a = tf.sigmoid(lv)
            nc = -(-n // C)
            xb = tf.reshape(tf.pad(xx, [[0, nc * C - n]]), [nc, C])
            la = tf.math.log1p(-a)
            t = tf.range(C, dtype=tf.float32)
            lag = t[:, None] - t[None, :]
            M = tf.where((lag >= 0)[None], tf.exp(tf.maximum(lag, 0.0)[None] * la[:, None, None]) * a[:, None, None], 0.0)
            yloc = tf.einsum("kts,js->jtk", M, xb)
            j = tf.range(nc, dtype=tf.float32)
            lagb = j[:, None] - j[None, :]
            M2 = tf.where((lagb >= 0)[None], tf.exp(tf.maximum(lagb, 0.0)[None] * (C * la)[:, None, None]), 0.0)
            S = tf.einsum("kji,ik->jk", M2, yloc[:, -1, :]) + tf.exp((j[:, None] + 1.0) * C * la[None, :]) * xx[0]
            S_prev = tf.concat([tf.ones_like(S[:1]) * xx[0], S[:-1]], 0)
            y = yloc + tf.exp((t[:, None] + 1.0) * la[None, :])[None] * S_prev[:, None, :]
            y = tf.reshape(y, [nc * C, K])[:n]
            loss = tf.reduce_sum(y * ww)
        return loss, tape.gradient(loss, lv)
    return timeit(fb, (x, w))


rows = {}
for K, T, Bn in ((24, 60, 256), (87, 60, 256), (24, 240, 64), (87, 240, 64)):
    t = windowed(Bn, K, T)
    rows[f"windowed K={K} T={T} B={Bn}"] = {"ms": round(t * 1000, 1), "us_per_pred": round(t / Bn * 1e6, 1),
                                           "epoch_30213_est_s": round(t / Bn * 30213, 1),
                                           "weights_tensor_MB_per_256": round(256 * K * T * T * 4 / 1e6, 1)}
    print("windowed", K, T, Bn, rows[f"windowed K={K} T={T} B={Bn}"], flush=True)
for K in (24, 87):
    t = chunked(K, 30213 + 720)
    rows[f"causal chunked K={K} n=30213+720"] = {"ms": round(t * 1000, 1), "us_per_pred": round(t / 30213 * 1e6, 3)}
    print("causal", K, rows[f"causal chunked K={K} n=30213+720"], flush=True)
out["scaling"] = rows

# analytic: weight of the window's first bar in the windowed EWMA, at the last bar and averaged over the window
LB = 60
contam = {}
for p in (5, 9, 14, 20, 26, 30, 35, 60):
    a = 2.0 / (p + 1)
    at_last = (1 - a) ** (LB - 1)
    mean_over_window = float(np.mean((1 - a) ** np.arange(LB)))
    contam[f"p{p}"] = {"at_last_bar": round(at_last, 4), "mean_over_60_positions": round(mean_over_window, 3)}
out["window_start_weight"] = contam
print(json.dumps(contam))
json.dump(out, open(os.path.join(SCRATCH, "bench_scaling.json"), "w"), indent=2)

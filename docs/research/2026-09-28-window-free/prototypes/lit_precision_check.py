"""Small CPU check of float32 pitfalls for windowless (whole-series) EMA recurrences.

Synthetic BTC-like random walk: start 60,000 USD, 1-minute steps with std 30 USD, T = 20,000 bars.
(a) sequential float32 EMA on raw prices vs float64 reference;
(b) exponentially weighted variance as EMA(x^2) - EMA(x)^2 in float32 on raw prices
    vs float64 reference (catastrophic cancellation), and the same on (x - x_0) centred input;
(c) Heinsen log-space scan (tf.math.cumulative_logsumexp, TF 2.10) for a constant-alpha EMA over
    the full series in float32, vs float64 reference, on centred input (needs complex log for negatives:
    here split positive / negative parts).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import time
import numpy as np
import tensorflow as tf

rng = np.random.default_rng(0)
T = 20_000
x64 = 60_000.0 + np.cumsum(rng.normal(0.0, 30.0, T))
alpha = 2.0 / (60 + 1)  # period-60 EMA


def ema_seq(x, a, dtype):
    x = x.astype(dtype)
    a = dtype(a)
    out = np.empty_like(x)
    s = x[0]
    for t in range(len(x)):
        s = a * x[t] + (dtype(1) - a) * s
        out[t] = s
    return out


ref = ema_seq(x64, alpha, np.float64)
f32 = ema_seq(x64, alpha, np.float32)
print(f"(a) float32 EMA on raw prices: max |err| = {np.max(np.abs(f32 - ref)):.4f} USD "
      f"(1-min std 30 USD; float32 spacing at 6e4 = {np.spacing(np.float32(6e4)):.4f})")

# (b) exponentially weighted variance
def ewvar(x, a, dtype):
    m = ema_seq(x, a, dtype)
    m2 = ema_seq(x.astype(dtype) ** 2, a, dtype)
    return m2 - m * m

v_ref = ewvar(x64, alpha, np.float64)
v_raw32 = ewvar(x64, alpha, np.float32)
xc = x64 - x64[0]
v_c32 = ewvar(xc, alpha, np.float32)
v_cref = ewvar(xc, alpha, np.float64)
sl = slice(1000, None)
print(f"(b) EW variance, median true var = {np.median(v_ref[sl]):.1f} USD^2")
print(f"    float32 raw prices:   median rel err = {np.median(np.abs(v_raw32[sl] - v_ref[sl]) / v_ref[sl]):.3g}, "
      f"negative values = {int(np.sum(v_raw32[sl] < 0))}")
print(f"    float32 centred x-x0: median rel err = {np.median(np.abs(v_c32[sl] - v_cref[sl]) / v_cref[sl]):.3g}")

# (c) Heinsen log-space scan with TF 2.10 primitives, constant alpha, full series in one shot.
# y_t = (1-a) y_{t-1} + a x_t, y_0 = x_0  ->  split x into positive and negative parts (log needs > 0).
def heinsen_ema_tf(x, a):
    x = tf.constant(x, tf.float32)
    t = tf.cast(tf.range(tf.shape(x)[0]), tf.float32)
    log_decay = tf.math.log1p(-tf.constant(a, tf.float32))
    a_star = t * log_decay                                   # cumulative sum of log(1-a)
    w = tf.concat([tf.ones(1), tf.fill([tf.shape(x)[0] - 1], tf.constant(a, tf.float32))], 0)
    def part(v):
        v = tf.maximum(v, 1e-30)
        return tf.exp(a_star + tf.math.cumulative_logsumexp(tf.math.log(w * v) - a_star))
    pos = part(tf.nn.relu(x))
    neg = part(tf.nn.relu(-x))
    return (pos - neg).numpy()

xc_ref = ema_seq(xc, alpha, np.float64)
t0 = time.perf_counter()
h32 = heinsen_ema_tf(xc, alpha)
dt = time.perf_counter() - t0
err = np.abs(h32 - xc_ref)
print(f"(c) log-space scan float32 (T={T}, centred): max |err| = {np.max(err):.4g} USD, "
      f"median = {np.median(err):.3g} USD; note a_star at T = {T * np.log1p(-alpha):.1f} (float32 exp range ~ +-88) ; {dt*1e3:.1f} ms")

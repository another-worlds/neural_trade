"""Effective sample size of per-position losses/gradients: sliding-window batches vs contiguous chunks.

Proxy per-position gradient series on the bundled 30-day BTC/USDT file (same anchors/targets as
data/windowing.py: target_h = close[i+h-1] - close[i-1], anchor i from 60):
  price : y_h                               (MSE gradient of a price-head bias at a zero prediction)
  dir   : 1{y_h > db} - mean, 0 in deadband  (BCE gradient of a direction-head bias)
  var   : 1 - y_h^2 / s_h^2                  (Gaussian-NLL gradient wrt log-variance, s_h^2 = h x trailing
                                              60-bar variance of 1-bar changes, the realized_vol scale)
Integrated autocorrelation time tau by batch means; design effect of batch layouts by simulation:
Var(batch mean under layout) / (Var(x)/n).  n_eff = n / deff.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.read_csv("C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv")
close = df["close"].to_numpy(float)
L, H = 60, (10, 15, 20)
DB_BPS = 5.0
N_TRAIN = 30213          # train anchors of run 20260924T182915Z-1aeff1c (artifacts/meta.json)

start, end = L, len(close) - (max(H) - 1)
anchors = np.arange(start, end)
lc = close[anchors - 1]
d1 = np.diff(close)
# trailing 60-bar std of 1-bar changes, known at the decision bar (window close[i-60:i] -> 59 diffs)
csum = np.concatenate([[0], np.cumsum(d1)])
csum2 = np.concatenate([[0], np.cumsum(d1 ** 2)])
def trailing_var(i):  # diffs d1[i-60 .. i-2] (between bars i-60..i-1)
    a, b = i - L, i - 1
    n = b - a
    m = (csum[b] - csum[a]) / n
    return (csum2[b] - csum2[a]) / n - m * m
tv = np.array([trailing_var(i) for i in anchors])

series = {}
for h in H:
    y = close[anchors + h - 1] - lc
    ret_bps = 1e4 * y / lc
    move = np.abs(ret_bps) > DB_BPS
    up = (y > 0).astype(float)
    g_dir = np.where(move, up - up[move].mean(), 0.0)
    s2 = h * np.maximum(tv, 1e-12)
    series[f"price_h{h}"] = y
    series[f"dir_h{h}"] = g_dir
    series[f"var_h{h}"] = 1.0 - y ** 2 / s2
for k in series:
    series[k] = series[k][:N_TRAIN]

def tau_batch_means(x, b):
    x = x - x.mean()
    m = len(x) // b
    bm = x[: m * b].reshape(m, b).mean(1)
    return b * bm.var(ddof=1) / x.var(ddof=1)

print("# Integrated autocorrelation time tau(b) = b * Var(block mean) / Var(x) (plateau = tau)")
bs = (1, 10, 30, 100, 300, 1000, 3000)
print("series      " + "".join(f"{('b=' + str(b)):>9s}" for b in bs))
for k, x in series.items():
    print(f"{k:11s} " + "".join(f"{tau_batch_means(x, b):9.1f}" for b in bs))

# ---------------------------------------------------------------- batch layouts
def tf_shuffle_order(n, buffer, rng):
    """Emulates tf.data shuffle(buffer_size): fill buffer, emit a random slot, refill from the stream."""
    buf = list(range(min(buffer, n)))
    nxt = len(buf)
    out = []
    while buf:
        j = rng.integers(len(buf))
        out.append(buf[j])
        if nxt < n:
            buf[j] = nxt
            nxt += 1
        else:
            buf[j] = buf[-1]
            buf.pop()
    return np.array(out)

def batches_shuffle(n_pos, bs, buffer, epochs, rng):
    out = []
    for _ in range(epochs):
        o = tf_shuffle_order(n_pos, buffer, rng)
        m = len(o) // bs
        out += list(o[: m * bs].reshape(m, bs))
    return out

def batches_random(n_pos, bs, n_batches, rng):
    return [rng.choice(n_pos, bs, replace=False) for _ in range(n_batches)]

def batches_chunks(n_pos, n_chunks, T, epochs, rng):
    """Tile the block with contiguous chunks of T positions at a random phase, shuffle chunks, group n_chunks per batch."""
    out = []
    for _ in range(epochs):
        phase = rng.integers(T)
        starts = np.arange(phase, n_pos - T + 1, T)
        rng.shuffle(starts)
        m = len(starts) // n_chunks
        for g in starts[: m * n_chunks].reshape(m, n_chunks):
            out.append((g[:, None] + np.arange(T)[None, :]).ravel())
    return out

def deff(x, batches):
    means = np.array([x[b].mean() for b in batches])
    n = len(batches[0])
    return means.var(ddof=1) / (x.var(ddof=1) / n)

layouts = {}
n = N_TRAIN
layouts["random 256"] = batches_random(n, 256, 3000, rng)
layouts["tf.shuffle(2048) 256 (today)"] = batches_shuffle(n, 256, 2048, 20, rng)
layouts["chunks 16 x 16"] = batches_chunks(n, 16, 16, 30, rng)
layouts["chunks 4 x 64"] = batches_chunks(n, 4, 64, 60, rng)
layouts["chunks 1 x 256"] = batches_chunks(n, 1, 256, 20, rng)
layouts["random 4096"] = batches_random(n, 4096, 1500, rng)
layouts["chunks 64 x 64"] = batches_chunks(n, 64, 64, 150, rng)
layouts["chunks 16 x 256"] = batches_chunks(n, 16, 256, 150, rng)
layouts["chunks 4 x 1024"] = batches_chunks(n, 4, 1024, 150, rng)
layouts["chunks 1 x 4096"] = batches_chunks(n, 1, 4096, 300, rng)

keys = ["price_h15", "dir_h15", "var_h15", "price_h10", "var_h20"]
print("\n# Design effect deff (and n_eff = n / deff) of the batch mean, per proxy gradient series")
print(f"{'layout':32s} {'n':>5s} {'batches':>7s} " + "".join(f"{k:>20s}" for k in keys))
for name, b in layouts.items():
    nn = len(b[0])
    row = []
    for k in keys:
        d = deff(series[k], b)
        row.append(f"{d:7.2f} (n_eff {nn / d:6.0f})")
    print(f"{name:32s} {nn:5d} {len(b):7d} " + "".join(f"{r:>20s}" for r in row))

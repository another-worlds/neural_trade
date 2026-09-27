"""CPU micro-benchmark: 24 learnable EWMAs for N predictions, windowed (today) vs one causal pass.

(a) today: N windows of 60 bars, neural_trade.utils.math.ewma_sequence_matrix_multi, alpha per sample
    [N, 24] = sigmoid(logit + 0.5 * meta), forward + gradient w.r.t. the 24 logits; also the real
    LearnableIndicators layer (all 31 features) for reference.
(b) one causal pass over a chunk of N + W bars (W = warm-up), alpha [24] = sigmoid(logit):
    b1 tf.scan (sequential); b2 Hillis-Steele associative scan (log depth); b3 chunked two-level
    matrix form (exact); b4 FFT convolution (full-length kernel); b5 depthwise conv with a truncated
    exponential kernel; b2v associative scan with a PER-BAR alpha [n, 24].
Checks: every (b) against a float64 numpy recursion; b3 on a single 60-bar window against
ewma_sequence_matrix_multi; the long pass against today's windowed values on the last 60 bars
(difference = window-start initialisation term, verified analytically); gradients against float64.
All numbers are CPU measurements (CUDA_VISIBLE_DEVICES=-1).
"""
import os
import sys
import time
import json
import math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = "C:/Users/Step/Documents/neural_trade"

import numpy as np
import pandas as pd
import neural_trade  # noqa: F401
import tensorflow as tf
import neural_trade.utils.math as mh

K = 24
LOOKBACK = 60
# the 24 periods today: stage 1 (MA 5,10,30; MACD fast 12, slow 26 x3 settings; BB 10,20,25; RSI 9,14,21 x2)
# + stage 2 (MACD signal 9 x3; BB 10,20,25). The MACD settings are read from the config.
from neural_trade.core.config import Config
cfg = Config()
macd = cfg.MACD_SETTINGS
periods = (list(cfg.MA_SPANS) + [m["fast"] for m in macd] + [m["slow"] for m in macd] + list(cfg.BB_PERIODS)
           + list(cfg.RSI_PERIODS) * 2 + [m["signal"] for m in macd] + list(cfg.BB_PERIODS))
assert len(periods) == K, periods
periods = np.array(periods, dtype=np.float64)
alpha0 = 2.0 / (periods + 1.0)
logit0 = np.log(alpha0) - np.log1p(-alpha0)
print("periods", periods.tolist())

close = pd.read_csv(os.path.join(REPO, "binance_btcusdt_1min_ccxt.csv"), usecols=["close"])["close"].to_numpy(np.float64)
scale = float(np.std(close[20:] - close[:-20]))   # ~ the target scale (20-bar deltas), for units only
print("bars", len(close), "scale(20-bar delta std)", round(scale, 2))

rng = np.random.default_rng(0)
out = {"periods": periods.tolist(), "scale": scale, "cpu_threads": os.cpu_count()}


def timeit(fn, args, reps=5, warm=2):
    for _ in range(warm):
        r = fn(*args)
    np.asarray(tf.nest.flatten(r)[0])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        _ = [np.asarray(t) for t in tf.nest.flatten(r)]
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


def n_ops(tf_fn, *args):
    gd = tf_fn.get_concrete_function(*args).graph.as_graph_def()
    skip = {"Const", "NoOp", "Identity", "Placeholder", "ReadVariableOp", "_Arg", "_Retval"}
    top = sum(1 for n in gd.node if n.op not in skip)
    lib = sum(1 for f in gd.library.function for n in f.node_def if n.op not in skip)
    return top, lib


# ============================================================ (a) today: windowed matrix form
def windows_for(n_pred, end):
    """n_pred windows of 60 ending at bars end-n_pred .. end-1 (window-relative, / scale)."""
    idx = np.arange(end - n_pred, end)
    W = np.stack([close[i - LOOKBACK + 1:i + 1] for i in idx])
    return ((W - W[:, -1:]) / scale).astype(np.float32)


logit_var = tf.Variable(logit0.astype(np.float32))


@tf.function
def today_fb(xw, meta, wout):
    # xw [N, 60] -> the 24 series are all the (window-relative) close here; per-sample alpha like today
    with tf.GradientTape() as tape:
        a = tf.sigmoid(logit_var[None, :] + 0.5 * meta)                           # [N, 24]
        x = tf.tile(xw[:, None, :], [1, K, 1])                                   # [N, 24, 60]
        y = mh.ewma_sequence_matrix_multi(x, a)                                  # [N, 24, 60]
        loss = tf.reduce_sum(y * wout)
    return loss, tape.gradient(loss, logit_var)


res_a = {}
for n_pred in (256, 1024):
    xw = tf.constant(windows_for(n_pred, len(close)))
    meta = tf.constant(np.tanh(rng.normal(size=(n_pred, K))).astype(np.float32) * 0.0)  # meta=0 -> same alpha as (b)
    wout = tf.constant(rng.normal(size=(n_pred, K, LOOKBACK)).astype(np.float32))
    med, lo, hi = timeit(today_fb, (xw, meta, wout), reps=5)
    res_a[n_pred] = med
    print(f"(a) today matrix-multi fwd+grad  N={n_pred:6d}: {med*1000:9.2f} ms  ({med/n_pred*1e6:7.2f} us/pred)", flush=True)
out["a_today_ms"] = {str(k): round(v * 1000, 2) for k, v in res_a.items()}
out["a_today_ops"] = n_ops(today_fb, xw, meta, wout)
# the real layer (31 features; 2 einsum groups) for reference
from neural_trade.registries.layers import Layers
li = Layers.for_role(cfg, "indicators", cfg, name="li_bench")
xw256 = tf.constant(windows_for(256, len(close)))
meta256 = tf.constant(np.tanh(rng.normal(size=(256, 18))).astype(np.float32))
_ = li([xw256, meta256])
w31 = tf.constant(rng.normal(size=(256, LOOKBACK, 31)).astype(np.float32))


@tf.function
def layer_fb(xx, mm, ww):
    with tf.GradientTape() as tape:
        tape.watch(mm)
        y = li([xx, mm])
        loss = tf.reduce_sum(y * ww)
    g = tape.gradient(loss, li.trainable_variables + [mm])
    return loss, g


med, lo, hi = timeit(layer_fb, (xw256, meta256, w31), reps=5)
out["a_real_layer_B256_ms"] = round(med * 1000, 2)
out["a_real_layer_ops"] = n_ops(layer_fb, xw256, meta256, w31)
print(f"(a') real LearnableIndicators fwd+grad N=256: {med*1000:.2f} ms", flush=True)
# per epoch: 30,213 training windows = 119 batches of 256 (measured per-batch time x 119)
out["a_today_epoch_est_ms_from_B256"] = round(res_a[256] * 1000 * 119, 1)
out["a_real_layer_epoch_est_ms"] = round(med * 1000 * 119, 1)


# ============================================================ (b) one causal pass over a chunk
def chunk(n):
    x = close[len(close) - n:]
    return ((x - x[0]) / scale).astype(np.float32)    # any constant shift is exact for an EWMA


def ref64(x, a):
    """float64 recursion, y[0] = x[0]; x [n], a [K] or [n, K] -> [n, K]."""
    x = np.asarray(x, np.float64)
    n = len(x)
    A = np.broadcast_to(np.asarray(a, np.float64), (n, K)) if np.ndim(a) == 2 else np.broadcast_to(a, (n, K))
    y = np.empty((n, K))
    y[0] = x[0]
    for t in range(1, n):
        y[t] = A[t] * x[t] + (1 - A[t]) * y[t - 1]
    return y


def ewma_scan(x, a):                         # b1: x [n], a [K] -> [n, K]
    xk = tf.tile(x[:, None], [1, K])
    rest = tf.scan(lambda p, c: a * c + (1.0 - a) * p, xk[1:], initializer=xk[0])
    return tf.concat([xk[:1], rest], 0)


def ewma_assoc(x, a):                        # b2: Hillis-Steele doubling; a [K] or [n, K]
    n = tf.shape(x)[0]
    xk = tf.tile(x[:, None], [1, K])
    a_full = a + tf.zeros_like(xk)
    A = 1.0 - a_full
    b = a_full * xk
    # y[0] = x[0]: element 0 is (A=0, b=x0)
    A = tf.concat([tf.zeros_like(A[:1]), A[1:]], 0)
    b = tf.concat([xk[:1], b[1:]], 0)
    d = 1
    nn = int(x.shape[0])
    while d < nn:
        b = b + A * tf.concat([tf.zeros_like(b[:d]), b[:-d]], 0)
        A = A * tf.concat([tf.ones_like(A[:d]), A[:-d]], 0)
        d *= 2
    return b


def ewma_chunked(x, a, C=64):                # b3: exact two-level matrix form, a [K]
    nn = int(x.shape[0])
    nc = -(-nn // C)
    pad = nc * C - nn
    xk = tf.pad(x, [[0, pad]])
    xb = tf.reshape(xk, [nc, C])                                           # [nc, C]
    la = tf.math.log1p(-a)                                                  # [K]
    t = tf.range(C, dtype=tf.float32)
    lag = t[:, None] - t[None, :]
    lower = lag >= 0
    M = tf.exp(tf.maximum(lag, 0.0)[None] * la[:, None, None]) * a[:, None, None]
    M = tf.where(lower[None], M, 0.0)                                       # [K, C, C]
    yloc = tf.einsum("kts,js->jtk", M, xb)                                  # [nc, C, K], zero start
    # block-end states: S_j = D^C S_{j-1} + yloc[j, C-1],  S_{-1} = x0  (so y[0] = x0)
    j = tf.range(nc, dtype=tf.float32)
    lagb = j[:, None] - j[None, :]
    M2 = tf.where((lagb >= 0)[None], tf.exp(tf.maximum(lagb, 0.0)[None] * (C * la)[:, None, None]), 0.0)  # [K, nc, nc]
    S = tf.einsum("kji,ik->jk", M2, yloc[:, -1, :]) + tf.exp((j[:, None] + 1.0) * C * la[None, :]) * x[0]  # [nc, K]
    S_prev = tf.concat([tf.ones_like(S[:1]) * x[0], S[:-1]], 0)            # state entering block j
    carry = tf.exp((t[:, None] + 1.0) * la[None, :])                        # [C, K] (1-a)^(t+1)
    y = yloc + carry[None] * S_prev[:, None, :]
    return tf.reshape(y, [nc * C, K])[:nn]


def ewma_fft(x, a):                          # b4: FFT convolution, full-length kernel (exact up to round-off)
    nn = int(x.shape[0])
    L = 1 << int(math.ceil(math.log2(2 * nn)))
    j = tf.range(nn, dtype=tf.float32)
    la = tf.math.log1p(-a)
    h = a[:, None] * tf.exp(j[None, :] * la[:, None])                      # [K, nn]
    X = tf.signal.rfft(tf.pad(x, [[0, L - nn]])[None, :], fft_length=[L])
    H = tf.signal.rfft(tf.pad(h, [[0, 0], [0, L - nn]]), fft_length=[L])
    y = tf.signal.irfft(X * H, fft_length=[L])[:, :nn]                      # sum_{j<=t} a(1-a)^j x[t-j]
    y = y + tf.exp((j[None, :] + 1.0) * la[:, None]) * x[0]                # initial state x0
    return tf.transpose(y)


def make_trunc(eps=1e-7):
    return int(math.ceil(math.log(eps) / math.log(1 - alpha0.min())))


KT = make_trunc()


def ewma_conv(x, a):                          # b5: depthwise conv, truncated kernel of KT taps
    nn = int(x.shape[0])
    j = tf.range(KT, dtype=tf.float32)
    la = tf.math.log1p(-a)
    h = a[:, None] * tf.exp(j[None, :] * la[:, None])                       # [K, KT], h[k, j] weight of x[t-j]
    xk = tf.pad(x, [[KT - 1, 0]], constant_values=0.0)                       # causal pad
    inp = tf.reshape(tf.tile(xk[:, None], [1, K]), [1, 1, nn + KT - 1, K])
    filt = tf.reshape(tf.transpose(tf.reverse(h, [1])), [1, KT, K, 1])
    y = tf.nn.depthwise_conv2d(inp, filt, strides=[1, 1, 1, 1], padding="VALID")[0, 0]   # [nn, K]
    y = y + tf.transpose(tf.exp((tf.range(nn, dtype=tf.float32)[None, :] + 1.0) * la[:, None])) * x[0]
    return y


impls = {"b1 tf.scan": ewma_scan, "b2 assoc scan": ewma_assoc, "b3 chunked matrix C=64": ewma_chunked,
         "b4 FFT conv": ewma_fft, "b5 depthwise conv trunc": ewma_conv}
W_WARM = 180   # warm-up = 3 x the 60-bar period ceiling: (1-2/61)^180 = 0.0025 of the start value left
out["warmup_bars"] = W_WARM
out["trunc_taps_b5"] = KT
res_b = {}
for name, fn in impls.items():
    res_b[name] = {}
    for n_pred in (256, 4096, 30213):
        if name == "b5 depthwise conv trunc" and n_pred > 4096:
            pass
        n = n_pred + W_WARM
        xc = tf.constant(chunk(n))
        wout = tf.constant(rng.normal(size=(n, K)).astype(np.float32) * (np.arange(n) >= W_WARM)[:, None].astype(np.float32))

        @tf.function
        def fb(xx, ww, fn=fn):
            with tf.GradientTape() as tape:
                a = tf.sigmoid(logit_var)
                y = fn(xx, a)
                loss = tf.reduce_sum(y * ww)
            return loss, tape.gradient(loss, logit_var)

        reps = 3 if (name.startswith("b1") and n_pred > 4096) else 5
        med, lo, hi = timeit(fb, (xc, wout), reps=reps, warm=1)
        top, lib = n_ops(fb, xc, wout)
        res_b[name][n_pred] = {"ms": round(med * 1000, 2), "min": round(lo * 1000, 2), "max": round(hi * 1000, 2),
                               "us_per_pred": round(med / n_pred * 1e6, 3), "ops_top": top, "ops_lib": lib}
        print(f"({name:26s}) fwd+grad N={n_pred:6d} (+{W_WARM} warm-up): {med*1000:9.2f} ms "
              f"({med/n_pred*1e6:8.3f} us/pred)  ops {top}+{lib}", flush=True)
out["b_long_pass"] = res_b

# b2v: per-bar alpha (time-varying) with the associative scan
res_v = {}
for n_pred in (4096, 30213):
    n = n_pred + W_WARM
    xc = tf.constant(chunk(n))
    meta = tf.constant((0.5 * np.tanh(rng.normal(size=(n, K)))).astype(np.float32))
    wout = tf.constant(rng.normal(size=(n, K)).astype(np.float32))

    @tf.function
    def fbv(xx, mm, ww):
        with tf.GradientTape() as tape:
            a = tf.sigmoid(logit_var[None, :] + mm)
            y = ewma_assoc(xx, a)
            loss = tf.reduce_sum(y * ww)
        return loss, tape.gradient(loss, logit_var)

    med, lo, hi = timeit(fbv, (xc, meta, wout), reps=5, warm=1)
    res_v[n_pred] = round(med * 1000, 2)
    print(f"(b2v assoc scan, per-bar alpha) N={n_pred}: {med*1000:.2f} ms", flush=True)
out["b2v_perbar_alpha_ms"] = res_v

# ============================================================ numerical agreement
chk = {}
n = 4096 + W_WARM
xc_np = chunk(n)
xc = tf.constant(xc_np)
a_tf = tf.sigmoid(tf.constant(logit0.astype(np.float32)))
a64 = 1.0 / (1.0 + np.exp(-logit0))
y64 = ref64(xc_np.astype(np.float64), a64)
yscale = np.abs(y64).max()
for name, fn in impls.items():
    y = fn(xc, a_tf).numpy()
    chk[name] = {"max_abs_err_vs_float64": float(np.abs(y - y64).max()), "max_abs_value": float(yscale)}
    print(f"agreement {name:26s}: max|y - y64| = {chk[name]['max_abs_err_vs_float64']:.2e} (|y| up to {yscale:.1f})")
# b2v per-bar alpha vs float64
meta_np = (0.5 * np.tanh(rng.normal(size=(n, K))))
av = 1.0 / (1.0 + np.exp(-(logit0[None, :] + meta_np)))
yv = ewma_assoc(xc, tf.constant(av.astype(np.float32))).numpy()
chk["b2v per-bar alpha"] = {"max_abs_err_vs_float64": float(np.abs(yv - ref64(xc_np, av)).max())}
print("agreement b2v per-bar alpha:", chk["b2v per-bar alpha"])

# the same recurrence: long-pass implementations run on ONE 60-bar window == ewma_sequence_matrix_multi
w1 = windows_for(1, len(close))[0]
ymat = mh.ewma_sequence_matrix_multi(tf.constant(np.tile(w1[None, None, :], [1, K, 1])),
                                     tf.constant(alpha0[None, :].astype(np.float32))).numpy()[0].T   # [60, K]
for name, fn in impls.items():
    if name.startswith("b5"):
        continue
    y = fn(tf.constant(w1), a_tf).numpy()
    chk[f"{name} on one 60-bar window vs ewma_sequence_matrix_multi"] = float(np.abs(y - ymat).max())
    print(f"same-window check {name:26s}: max diff {np.abs(y - ymat).max():.2e}")

# long pass (warm) vs today's windowed values on the last 60 bars, re-referenced to the last close
ylong = y64[-LOOKBACK:] - xc_np[-1]              # EWMA is shift-equivariant: subtract the anchor close
xwin = xc_np[-LOOKBACK:] - xc_np[-1]
ywin = ref64(xwin, a64)                           # today's windowed EWMA (starts at the window's first bar)
diff = ylong - ywin                              # [60, K]
# analytic: diff[t] = (1-a)^t * (y_long[t0] - x[t0])
pred = ((1 - a64)[None, :] ** np.arange(LOOKBACK)[:, None]) * (ylong[0] - xwin[0])[None, :]
chk["window_vs_long_analytic_max_err"] = float(np.abs(diff - pred).max())
last = np.abs(diff[-1])
chk["window_vs_long_last_bar_absdiff_by_period"] = {f"p{int(p)}": round(float(d), 4) for p, d in zip(periods, last)}
chk["init_weight_at_last_bar_by_period"] = {f"p{int(p)}": round(float((1 - a) ** (LOOKBACK - 1)), 4) for p, a in zip(periods, a64)}
print("window vs long pass: analytic residual", chk["window_vs_long_analytic_max_err"])
print("last-bar |long - windowed| by period (scale units):", chk["window_vs_long_last_bar_absdiff_by_period"])
print("weight of the window's first bar at the last bar (1-a)^59:", chk["init_weight_at_last_bar_by_period"])
# over many windows: distribution of the last-bar discrepancy relative to the EWMA's own deviation from the close
m = 2000
idx = np.arange(len(close) - m, len(close))
full = ref64(((close - close[0]) / scale), a64)                              # long pass over the whole file
rel = []
for i in idx[::10]:
    xw_ = (close[i - LOOKBACK + 1:i + 1] - close[i]) / scale
    yw_ = ref64(xw_, a64)[-1]
    yl_ = full[i] - (close[i] - close[0]) / scale
    rel.append(np.abs(yl_ - yw_) / (np.abs(yl_) + 1e-9))
rel = np.array(rel)
chk["last_bar_rel_discrepancy_median_by_period"] = {f"p{int(p)}": round(float(v), 4) for p, v in zip(periods, np.median(rel, 0))}
print("median |long-windowed| / |long - close| at the last bar, by period:", chk["last_bar_rel_discrepancy_median_by_period"])

# gradients vs float64 (b2, b3, b4 against a float64 scan through tf)
wg = rng.normal(size=(n, K))
with tf.GradientTape() as tape:
    lv = tf.Variable(logit0)
    a = tf.sigmoid(lv)
    xk = tf.constant(xc_np.astype(np.float64))
    y = tf.concat([tf.tile(xk[:1, None], [1, K]),
                   tf.scan(lambda p, c: a * c + (1.0 - a) * p, tf.tile(xk[1:, None], [1, K]),
                           initializer=tf.tile(xk[:1], [K]))], 0)
    loss = tf.reduce_sum(y * wg)
g64 = tape.gradient(loss, lv).numpy()
for name, fn in impls.items():
    lv32 = tf.Variable(logit0.astype(np.float32))
    with tf.GradientTape() as tape:
        y = fn(xc, tf.sigmoid(lv32))
        loss = tf.reduce_sum(y * tf.constant(wg.astype(np.float32)))
    g = tape.gradient(loss, lv32).numpy()
    chk[f"grad_rel_err {name}"] = float(np.max(np.abs(g - g64) / (np.abs(g64) + 1e-9)))
    print(f"grad agreement {name:26s}: max rel err {chk[f'grad_rel_err {name}']:.2e}")
out["checks"] = chk
json.dump(out, open(os.path.join(SCRATCH, "bench_ewma.json"), "w"), indent=2, default=str)
print("saved", os.path.join(SCRATCH, "bench_ewma.json"))

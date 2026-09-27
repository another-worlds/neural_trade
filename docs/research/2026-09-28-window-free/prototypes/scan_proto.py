"""Prototype: first-order linear recurrence h_t = a_t * h_{t-1} + b_t over long series in TF 2.10.

Implementations compared (CPU only, CUDA_VISIBLE_DEVICES=-1):
  seq     : tf.scan (the reference; sequential, what a naive port would do)
  hs      : Hillis-Steele parallel associative scan (log2 T levels of shift/mul/add)
  chunk   : chunked matrix form (intra-chunk C x C decay matrices via local log-cumsum,
            inter-chunk carry by a Hillis-Steele scan over the chunk states)
  logcum  : naive global log-cumsum form h_t = sum_k exp(S_t - S_k) b_k (to show float32 cancellation)
Reference: numpy float64 sequential loop.
Selective (time-varying) alpha_t: a_t = 1 - alpha_t, b_t = alpha_t * x_t.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(4)

CSV = "C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv"


# ------------------------------------------------------------------ implementations
def scan_seq(a, b, h0=None):
    """a, b: [K, T]. tf.scan over time."""
    K = tf.shape(a)[0]
    init = tf.zeros([K], a.dtype) if h0 is None else h0
    out = tf.scan(lambda h, ab: ab[0] * h + ab[1], (tf.transpose(a), tf.transpose(b)), initializer=init)
    return tf.transpose(out)


def scan_hs(a, b, h0=None):
    """Hillis-Steele inclusive scan of the affine maps (a_t, b_t) along the last axis."""
    if h0 is not None:
        b = tf.concat([a[..., :1] * h0[..., None] + b[..., :1], b[..., 1:]], axis=-1)
    T = a.shape[-1]
    d = 1
    while d < T:
        a_prev = tf.concat([tf.ones_like(a[..., :d]), a[..., :-d]], axis=-1)
        b_prev = tf.concat([tf.zeros_like(b[..., :d]), b[..., :-d]], axis=-1)
        b = a * b_prev + b
        a = a * a_prev
        d *= 2
    return b


def scan_chunk(a, b, C=64):
    """Chunked matrix form. a in (0, 1], b any. [K, T] with T % C == 0."""
    K, T = a.shape
    n = T // C
    la = tf.math.log(tf.reshape(a, [K, n, C]))                  # log decay per step
    bb = tf.reshape(b, [K, n, C])
    cs = tf.cumsum(la, axis=-1)                                 # local cumsum inside each chunk
    diff = cs[..., :, None] - cs[..., None, :]                  # [K, n, C, C]: sum_{j<k<=i} log a_k
    idx = tf.range(C)
    lower = idx[:, None] >= idx[None, :]
    diff = tf.where(lower, diff, tf.fill(tf.shape(diff), tf.constant(-1e30, diff.dtype)))
    L = tf.exp(diff)                                            # 0 above the diagonal, no inf
    intra = tf.einsum("knij,knj->kni", L, bb)                   # state if each chunk started from 0
    # carry across chunks: H_c = A_c * H_{c-1} + s_c, A_c = prod of a over chunk c
    A = tf.exp(cs[..., -1])                                     # [K, n]
    s = intra[..., -1]                                          # [K, n]
    H = scan_hs(A, s)                                           # state at the END of every chunk
    H_prev = tf.concat([tf.zeros_like(H[..., :1]), H[..., :-1]], axis=-1)   # state entering chunk c
    carry = tf.exp(cs) * H_prev[..., None]                      # decayed incoming state
    return tf.reshape(intra + carry, [K, T])


def scan_logcum(a, b):
    """Global log-cumsum: h_t = sum_{k<=t} exp(S_t - S_k) b_k (only for the precision demo, via cumsum of
    exp(-S) b which overflows; we instead do the dense form on a short tail)."""
    S = tf.cumsum(tf.math.log(a), axis=-1)
    return S


# ------------------------------------------------------------------ helpers
def ref64(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    h = np.zeros_like(b); prev = np.zeros(a.shape[0])
    for t in range(a.shape[1]):
        prev = a[:, t] * prev + b[:, t]
        h[:, t] = prev
    return h


def graph_ops(fn, *args):
    cf = tf.function(fn).get_concrete_function(*args)
    n = len(cf.graph.get_operations())
    return n


def timed(fn, *args, reps=5):
    f = tf.function(fn)
    f(*args)  # trace + warm
    t0 = time.perf_counter()
    for _ in range(reps):
        f(*args)
    return (time.perf_counter() - t0) / reps


def make_inputs(T, periods, seed=0, level=False):
    """Selective alphas from real BTC closes: alpha_t = sigmoid(logit(period) + 0.5 * tanh(z_t)),
    z_t a causal standardised 1-bar change (a stand-in for meta_adjust per bar)."""
    close = pd.read_csv(CSV, usecols=["close"])["close"].to_numpy(np.float64)
    close = close[-(T + 1):]
    dx = np.diff(close)                                         # increments, $ per bar
    scale = np.std(close[20:] - close[:-20])                    # rough stand-in for the target scale
    z = (dx - dx.mean()) / dx.std()
    K = len(periods)
    base = np.log(2.0 / (np.array(periods) + 1.0)) - np.log(1 - 2.0 / (np.array(periods) + 1.0))
    logit = base[:, None] + 0.5 * np.tanh(z)[None, :]
    alpha = 1.0 / (1.0 + np.exp(-logit))
    alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
    if level:
        x = np.broadcast_to(close[1:], (K, T))                  # raw price level
    else:
        x = np.broadcast_to(dx / scale, (K, T))                 # scaled increments
    return alpha, np.ascontiguousarray(x), scale, close


def main():
    periods = [2, 5, 14, 30, 60, 240, 1440, 5, 14, 30, 60, 240]   # K = 12 channels per call (x2 below)
    out = {}
    for T in (10240, 43008):
        alpha, x, scale, close = make_inputs(T, periods)
        a64 = 1.0 - alpha; b64 = alpha * x
        h_ref = ref64(a64, b64)
        a = tf.constant(a64, tf.float32); b = tf.constant(b64, tf.float32)
        res = {}
        for name, fn in (("hs", scan_hs), ("chunk", scan_chunk)):
            h = fn(a, b).numpy()
            res[name] = float(np.max(np.abs(h - h_ref)))
        if T <= 10240:
            h = scan_seq(a, b).numpy(); res["seq"] = float(np.max(np.abs(h - h_ref)))
        # naive global log-cumsum precision: error of exp(S_i - S_j) for |i-j| <= 64 at the END of the series
        S32 = tf.cumsum(tf.math.log(a), axis=-1).numpy()
        S64 = np.cumsum(np.log(a64), axis=-1)
        tail = slice(T - 64, T)
        d32 = S32[:, tail][:, :, None] - S32[:, tail][:, None, :]
        d64 = S64[:, tail][:, :, None] - S64[:, tail][:, None, :]
        mask = np.tril(np.ones((64, 64), bool))
        rel = np.abs(np.exp(np.where(mask, d32, -1e30)) - np.exp(np.where(mask, d64, -1e30))) / \
            np.maximum(np.exp(np.where(mask, d64, -1e30)), 1e-30)
        rel = np.where(np.exp(np.where(mask, d64, -1e30)) > 1e-6, rel, 0.0)
        res["logcum_rel_err_max"] = float(rel.max())
        res["logcum_abs_S_max"] = float(np.abs(S64[:, -1]).max())
        out[T] = res
        print(f"T={T}: max |h - ref64| in scaled-increment units:", {k: f"{v:.3g}" for k, v in res.items()})

    # level form vs increment form (float32 error of EMA - close in $)
    T = 43008
    alpha, xl, scale, close = make_inputs(T, periods, level=True)
    a64 = 1.0 - alpha; b64 = alpha * xl
    # first bar: start the level recurrence at x_0 (h_{-1} = x_0) so it is an EMA of prices
    h0 = xl[:, 0].astype(np.float64)
    h_ref = ref64(np.concatenate([np.zeros((len(periods), 1)), a64], 1),
                  np.concatenate([h0[:, None], b64], 1))[:, 1:]
    h32 = scan_hs(tf.constant(a64, tf.float32), tf.constant(b64, tf.float32),
                  h0=tf.constant(h0, tf.float32)).numpy()
    err_level = np.abs((h32 - xl) - (h_ref - xl))               # error of (EMA - close) in $
    # increment form: d_t = EMA_t - x_t = (1 - alpha_t) (d_{t-1} - dx_t)
    dx = np.diff(close)                                         # dx_t = x_t - x_{t-1}, len T
    a_d = 1.0 - alpha; b_d = -(1.0 - alpha) * (dx / scale)[None, :]
    d_ref = ref64(a_d, b_d) * scale
    d32 = scan_hs(tf.constant(a_d, tf.float32), tf.constant(b_d, tf.float32)).numpy() * scale
    # the two references agree (same math, different start): compare after 5000 bars of burn-in
    print("level-form float32: max |err(EMA - close)| $ =", f"{err_level[:, 5000:].max():.3g}",
          "; in target-scale units:", f"{err_level[:, 5000:].max() / scale:.3g}", "(scale $", f"{scale:.1f})")
    print("increment-form float32: max |err(EMA - close)| $ =", f"{np.abs(d32 - d_ref)[:, 5000:].max():.3g}",
          "; in target-scale units:", f"{np.abs(d32 - d_ref)[:, 5000:].max() / scale:.3g}")
    print("level vs increment reference (after 5000 bars burn-in, $):",
          f"{np.abs((h_ref - xl) - d_ref)[:, 5000:].max():.3g}")

    # gradients: hs / chunk vs tf.scan in float64 (T = 2048)
    T = 2048
    alpha, x, scale, close = make_inputs(T, periods)
    w = np.random.default_rng(0).normal(size=(len(periods), T))
    logit64 = np.log(alpha) - np.log1p(-alpha)

    def loss_fn(fn, dtype):
        lg = tf.Variable(logit64.astype(dtype))
        xx = tf.constant(x.astype(dtype)); ww = tf.constant(w.astype(dtype))
        with tf.GradientTape() as tape:
            al = tf.sigmoid(lg)
            h = fn(1.0 - al, al * xx)
            L = tf.reduce_sum(ww * h)
        return tape.gradient(L, lg).numpy()

    g_ref = loss_fn(scan_seq, np.float64)
    for name, fn in (("hs", scan_hs), ("chunk", scan_chunk), ("seq32", scan_seq)):
        g = loss_fn(fn, np.float32)
        rel = np.max(np.abs(g - g_ref)) / np.max(np.abs(g_ref))
        print(f"grad wrt per-step logits, {name} (float32) vs tf.scan float64: max rel err {rel:.3g}, "
              f"finite: {np.isfinite(g).all()}")

    # extreme alphas: gradient finiteness at the clamp (alpha = 1 - 1e-6 and 1e-6)
    for al in (1e-6, 1 - 1e-6):
        lg = tf.Variable(np.full((2, 4096), np.log(al) - np.log1p(-al), np.float32))
        xx = tf.constant(np.random.default_rng(1).normal(size=(2, 4096)).astype(np.float32))
        for name, fn in (("hs", scan_hs), ("chunk", scan_chunk)):
            with tf.GradientTape() as tape:
                a_ = tf.clip_by_value(tf.sigmoid(lg), 1e-6, 1 - 1e-6)
                L = tf.reduce_sum(fn(1.0 - a_, a_ * xx))
            g = tape.gradient(L, lg).numpy()
            print(f"alpha={al:g} {name}: grad finite {np.isfinite(g).all()}, max|g| {np.abs(g).max():.3g}")

    # timings (CPU) forward+backward, K = 24 channels, and graph op counts
    print("\nCPU timings, forward+backward, K=24 channels (seconds per call):")
    rng = np.random.default_rng(2)
    for T in (10240, 30720):
        lg = tf.Variable(rng.normal(-2, 1, size=(24, T)).astype(np.float32))
        xx = tf.constant(rng.normal(size=(24, T)).astype(np.float32))

        def fb(fn):
            def g():
                with tf.GradientTape() as tape:
                    al = tf.sigmoid(lg)
                    L = tf.reduce_sum(tf.square(fn(1.0 - al, al * xx)))
                return tape.gradient(L, lg)
            return g
        row = {}
        for name, fn in (("hs", scan_hs), ("chunk64", lambda a, b: scan_chunk(a, b, 64)),
                         ("chunk128", lambda a, b: scan_chunk(a, b, 128))):
            row[name] = timed(fb(fn), reps=3)
            row[name + "_ops"] = graph_ops(fb(fn))
        if T == 10240:
            row["seq"] = timed(fb(scan_seq), reps=1)
        print(f"T={T}:", {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()

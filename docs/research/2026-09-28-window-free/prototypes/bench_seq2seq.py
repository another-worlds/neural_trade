"""CPU sketch: today's windowed model vs a causal sequence-to-sequence model of similar width.

Both trained with the SAME simple per-prediction loss (MSE + BCE + Gaussian NLL over 3 horizons),
so the comparison isolates the architecture/windowing (the real 34-term loss and ~150 metrics are a
fixed per-step add-on in either design). Measured: CPU ms per training step, predictions per second,
and graph op count (a proxy for GPU kernel launches). CPU only (CUDA_VISIBLE_DEVICES=-1).
"""
import os
import time
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = "C:/Users/Step/Documents/neural_trade"

import numpy as np
import pandas as pd
import neural_trade  # noqa: F401
import tensorflow as tf
from tensorflow.keras import layers

from neural_trade.core.config import Config
from neural_trade.registries.models import Models

tf.keras.utils.set_random_seed(0)
cfg = Config()
H = [int(h) for h in cfg.HORIZON_STEPS]
LB = cfg.LOOKBACK
close = pd.read_csv(os.path.join(REPO, "binance_btcusdt_1min_ccxt.csv"), usecols=["close"])["close"].to_numpy(np.float64)
scale = float(np.std(close[20:] - close[:-20]))
rng = np.random.default_rng(0)
out = {}


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
    # only count the library functions actually used on this device (cuDNN vs standard GRU both appear)
    lib = sum(1 for f in gd.library.function for n in f.node_def if n.op not in skip)
    return top, lib


def simple_loss(price, dirp, var, y, mask=None):
    """price/dir/var/y: lists of 3 tensors of the same shape; mask broadcastable (1 = counted)."""
    tot = 0.0
    for p, d, v, t in zip(price, dirp, var, y):
        up = tf.cast(t > 0, tf.float32)
        v = tf.maximum(v, 1e-3)
        per = tf.square(p - t) + tf.keras.backend.binary_crossentropy(up, tf.clip_by_value(d, 1e-6, 1 - 1e-6)) \
            + 0.5 * (tf.math.log(v) + tf.square(t - p) / v)
        if mask is not None:
            tot += tf.reduce_sum(per * mask) / tf.reduce_sum(mask)
        else:
            tot += tf.reduce_mean(per)
    return tot


# ------------------------------------------------------------------ today: windowed gru_attention
base = Models.build(getattr(cfg, "MODEL_NAME", None), cfg)
opt_a = tf.keras.optimizers.Adam(1e-3)
B = 256
idx = rng.integers(LB, len(close) - max(H), size=B)
Xw = np.stack([(close[i - LB:i] - close[i - 1]) / scale for i in idx]).astype(np.float32)
Yw = np.stack([[(close[i + h - 1] - close[i - 1]) / scale for h in H] for i in idx]).astype(np.float32)
Xw, Yw = tf.constant(Xw), tf.constant(Yw)


@tf.function
def step_today(x, y):
    with tf.GradientTape() as tape:
        o = base(x, training=True)
        loss = simple_loss([o[0], o[3], o[6]], [o[1], o[4], o[7]], [o[2], o[5], o[8]],
                           [y[:, 0:1], y[:, 1:2], y[:, 2:3]])
    g = tape.gradient(loss, base.trainable_variables)
    opt_a.apply_gradients([(gg, v) for gg, v in zip(g, base.trainable_variables) if gg is not None])
    return loss


med, lo, hi = timeit(step_today, (Xw, Yw), reps=7)
ops = n_ops(step_today, Xw, Yw)
out["today_B256"] = {"ms": round(med * 1000, 1), "min": round(lo * 1000, 1), "max": round(hi * 1000, 1),
                     "preds_per_step": B, "preds_per_s": round(B / med), "ops_top_lib": ops,
                     "params": int(base.count_params())}
print("TODAY windowed gru_attention, B=256:", out["today_B256"], flush=True)


# ------------------------------------------------------------------ seq2seq sketch
MA, RSI, BBP = list(cfg.MA_SPANS), list(cfg.RSI_PERIODS), list(cfg.BB_PERIODS)
MACD = cfg.MACD_SETTINGS


def logit_p(p):
    a = 2.0 / (p + 1.0)
    return float(np.log(a) - np.log1p(-a))


def ewma_chunked_b(x, a, C=64):
    """x [Bc, n, K], a [K] -> [Bc, n, K]; exact two-level matrix form, y[:,0] = x[:,0]."""
    Bc, nn, Kc = x.shape[0], int(x.shape[1]), int(x.shape[2])
    nc = -(-nn // C)
    xp = tf.pad(x, [[0, 0], [0, nc * C - nn], [0, 0]])
    xb = tf.reshape(xp, [-1, nc, C, Kc])
    la = tf.math.log1p(-a)
    t = tf.range(C, dtype=tf.float32)
    lag = t[:, None] - t[None, :]
    M = tf.where((lag >= 0)[None], tf.exp(tf.maximum(lag, 0.0)[None] * la[:, None, None]) * a[:, None, None], 0.0)
    yloc = tf.einsum("kts,bjsk->bjtk", M, xb)
    j = tf.range(nc, dtype=tf.float32)
    lagb = j[:, None] - j[None, :]
    M2 = tf.where((lagb >= 0)[None], tf.exp(tf.maximum(lagb, 0.0)[None] * (C * la)[:, None, None]), 0.0)
    x0 = x[:, 0, :]                                                                   # [Bc, K]
    S = tf.einsum("kji,bik->bjk", M2, yloc[:, :, -1, :]) + tf.exp((j[:, None] + 1.0) * C * la[None, :])[None] * x0[:, None, :]
    S_prev = tf.concat([x0[:, None, :], S[:, :-1, :]], 1)
    carry = tf.exp((t[:, None] + 1.0) * la[None, :])
    y = yloc + carry[None, None] * S_prev[:, :, None, :]
    return tf.reshape(y, [-1, nc * C, Kc])[:, :nn]


class CausalIndicators(layers.Layer):
    """The same 18 + 6 learnable EWMAs and 31 channels, as one causal pass; level channels are
    re-referenced to each bar's own close (what window-relative input gives at the last bar)."""

    def build(self, s):
        p1 = MA + [m["fast"] for m in MACD] + [m["slow"] for m in MACD] + BBP + RSI + RSI
        p2 = [m["signal"] for m in MACD] + BBP
        self.l1 = self.add_weight("l1", shape=(len(p1),), initializer=tf.constant_initializer([logit_p(p) for p in p1]))
        self.l2 = self.add_weight("l2", shape=(len(p2),), initializer=tf.constant_initializer([logit_p(p) for p in p2]))

    def call(self, x):                                                   # x [Bc, n]
        nma, nm, nb, nr = len(MA), len(MACD), len(BBP), len(RSI)
        d = tf.concat([tf.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
        gains, losses = tf.nn.relu(d), tf.nn.relu(-d)
        s1 = tf.stack([x] * (nma + 2 * nm + nb) + [gains] * nr + [losses] * nr, -1)
        e1 = ewma_chunked_b(s1, tf.sigmoid(self.l1))
        o = 0
        ma = e1[..., o:o + nma]; o += nma
        fast = e1[..., o:o + nm]; o += nm
        slow = e1[..., o:o + nm]; o += nm
        bbm = e1[..., o:o + nb]; o += nb
        g = e1[..., o:o + nr]; o += nr
        lo_ = e1[..., o:o + nr]
        line = fast - slow
        sq = tf.square(x[..., None] - bbm)
        e2 = ewma_chunked_b(tf.concat([line, sq], -1), tf.sigmoid(self.l2))
        sig, var = e2[..., :nm], e2[..., nm:]
        hist = line - sig
        rsi = 100.0 - 100.0 / (1.0 + g / (lo_ + 1e-8))
        std = tf.sqrt(var + 1e-8)
        xc = x[..., None]
        feats = [ma - xc, line, sig, hist, tf.tanh(10.0 * hist), rsi / 100.0, bbm - xc,
                 bbm + 2 * std - xc, bbm - 2 * std - xc, (xc - (bbm - 2 * std)) / (4 * std + 1e-8), d[..., None]]
        return tf.concat(feats, -1), var                                 # [Bc, n, 31], [Bc, n, 3]


def band_mask(n, width=LB):
    i = tf.range(n)[:, None]
    j = tf.range(n)[None, :]
    return tf.logical_and(j <= i, j > i - width)                        # causal, last `width` bars


def build_seq2seq(n):
    inp = layers.Input(shape=(n,), batch_size=None)
    ind, var = CausalIndicators(name="causal_ind")(inp)
    mem = layers.GRU(128, return_sequences=True)(ind)                    # causal (unidirectional)
    mem = layers.Dropout(0.1)(mem)
    mask = band_mask(n)[None]
    att = layers.MultiHeadAttention(num_heads=8, key_dim=32)(mem, mem, attention_mask=mask)
    x = layers.LayerNormalization()(layers.Add()([mem, att]))
    mix = layers.Dense(128, activation="gelu")(x)                       # stands in for cross-indicator attention
    x = layers.LayerNormalization()(layers.Add()([x, mix]))
    xs = layers.Conv1D(16, 3, padding="causal", activation="gelu")(x)
    xm = layers.Conv1D(16, 7, padding="causal", activation="gelu")(x)
    xl = layers.Conv1D(16, 15, padding="causal", activation="gelu")(x)
    gate = layers.Dense(3, activation="softmax")(tf.math.log(var + 1e-6))  # per-bar energy gate from EWMA variance
    x = xs * gate[..., 0:1] + xm * gate[..., 1:2] + xl * gate[..., 2:3]
    x = layers.LayerNormalization()(x)
    for _ in range(2):
        a = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x, attention_mask=mask)
        x = layers.LayerNormalization()(layers.Add()([x, a]))
        ff = layers.Dense(16)(layers.Dropout(0.1)(layers.Dense(32, activation="gelu")(x)))
        x = layers.LayerNormalization()(layers.Add()([x, ff]))
    sh = layers.Dense(32, activation="gelu")(x)
    outs = []
    for k in range(3):
        tw = layers.Dense(16, activation="gelu")(sh)
        outs += [layers.Dense(1)(tw), layers.Dense(1, activation="sigmoid")(tw), layers.Dense(1, activation="softplus")(tw)]
    return tf.keras.Model(inp, outs)


W = 180   # warm-up bars per chunk: masked out of the loss (indicator and GRU state warm-up)
res = {}
for Bc, L in ((16, 256), (8, 512), (4, 1024), (16, 512)):
    n = W + L + max(H)
    m = build_seq2seq(n - max(H))
    opt = tf.keras.optimizers.Adam(1e-3)
    starts = rng.integers(0, len(close) - n, size=Bc)
    xs_ = np.stack([close[s:s + n - max(H)] for s in starts])
    X = tf.constant(((xs_ - xs_[:, :1]) / scale).astype(np.float32))
    full = np.stack([close[s:s + n] for s in starts])
    Y = np.stack([(full[:, h:h + n - max(H)] - full[:, :n - max(H)]) / scale for h in H], -1).astype(np.float32)  # delta to bar t+h
    Y = tf.constant(Y)
    lm = tf.constant((np.arange(n - max(H)) >= W).astype(np.float32)[None, :, None])

    @tf.function
    def step_s2s(x, y, m=m, opt=opt, lm=lm):
        with tf.GradientTape() as tape:
            o = m(x, training=True)
            loss = simple_loss([o[0], o[3], o[6]], [o[1], o[4], o[7]], [o[2], o[5], o[8]],
                               [y[..., 0:1], y[..., 1:2], y[..., 2:3]], mask=lm)
        g = tape.gradient(loss, m.trainable_variables)
        opt.apply_gradients([(gg, v) for gg, v in zip(g, m.trainable_variables) if gg is not None])
        return loss

    med, lo, hi = timeit(step_s2s, (X, Y), reps=5, warm=2)
    ops = n_ops(step_s2s, X, Y)
    preds = Bc * L
    res[f"Bc{Bc}_L{L}"] = {"ms": round(med * 1000, 1), "min": round(lo * 1000, 1), "max": round(hi * 1000, 1),
                           "preds_per_step": preds, "bars_processed": Bc * (n - max(H)), "preds_per_s": round(preds / med),
                           "ops_top_lib": ops, "params": int(m.count_params())}
    print(f"SEQ2SEQ Bc={Bc} L={L} (+{W} warm-up):", res[f"Bc{Bc}_L{L}"], flush=True)
out["seq2seq"] = res
out["warmup"] = W
json.dump(out, open(os.path.join(SCRATCH, "bench_seq2seq.json"), "w"), indent=2)
print("saved")

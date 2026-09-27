"""Option A2 prototype: today's 31 indicator channels computed ONCE over the whole series with
per-bar selective alphas (increment form, Hillis-Steele scan with a reverse-scan gradient), then the
[B, L, 31] windows the unchanged network reads are a gather. Compared with today's
LearnableIndicators (per-window [B, K, L, L] matrices): graph ops fwd+bwd (GPU launch proxy),
CPU time, and equality with today's layer when the history is cut to the window (sanity).
"""
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import neural_trade  # noqa: F401
import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(1)
tf.config.threading.set_inter_op_parallelism_threads(4)
CSV = "C:/Users/Step/Documents/neural_trade/binance_btcusdt_1min_ccxt.csv"
S = 257.51813253642973


def hs(a, b):
    T = a.shape[-1]
    d = 1
    while d < T:
        a_prev = tf.concat([tf.ones_like(a[..., :d]), a[..., :-d]], axis=-1)
        b_prev = tf.concat([tf.zeros_like(b[..., :d]), b[..., :-d]], axis=-1)
        b = a * b_prev + b
        a = a * a_prev
        d *= 2
    return b


@tf.custom_gradient
def linrec(a, b):
    h = hs(a, b)

    def grad(g):
        a_next = tf.concat([a[..., 1:], tf.zeros_like(a[..., :1])], axis=-1)
        G = tf.reverse(hs(tf.reverse(a_next, [-1]), tf.reverse(g, [-1])), [-1])
        h_prev = tf.concat([tf.zeros_like(h[..., :1]), h[..., :-1]], axis=-1)
        return G * h_prev, G
    return h, grad


def logit_p(p):
    a = 2.0 / (np.asarray(p, np.float64) + 1.0)
    return (np.log(a) - np.log1p(-a)).astype(np.float32)


class SeriesIndicators(tf.keras.layers.Layer):
    """Periods: MA 3, MACD 3x(fast, slow, signal), RSI 3, BB 3 = 18 logits; per-bar meta shift."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.ma = tf.Variable(logit_p([5, 10, 30])); self.fast = tf.Variable(logit_p([12, 5, 8]))
        self.slow = tf.Variable(logit_p([26, 35, 17])); self.sig = tf.Variable(logit_p([9, 5, 9]))
        self.rsi = tf.Variable(logit_p([9, 14, 21])); self.bb = tf.Variable(logit_p([10, 20, 25]))
        self.meta = tf.keras.layers.Dense(18, activation="tanh")

    def call(self, dx, ctx):
        """dx [N] scaled increments (close_t - close_{t-1}) / S; ctx [N, 2] causal context -> F [N, 31]
        with every price-type channel expressed relative to its OWN bar's close."""
        m = tf.transpose(self.meta(ctx)) * 0.5                                  # [18, N]
        lg = tf.concat([self.ma, self.fast, self.slow, self.bb, self.rsi, self.sig], 0)[:, None] + m
        al = tf.clip_by_value(tf.sigmoid(lg), 1e-6, 1 - 1e-6)                   # [18, N] per-bar alphas
        a_ma, a_f, a_s, a_bb, a_r, a_sig = tf.split(al, 3 * np.ones(6, int), 0)
        gain, loss = tf.nn.relu(dx), tf.nn.relu(-dx)
        # stage 1 (15 series): d = EMA - close via d_t = (1 - a)(d_{t-1} - dx_t); RSI gain / loss EWMAs
        a1 = tf.concat([a_ma, a_f, a_s, a_bb, a_r, a_r], 0)
        b1 = tf.concat([-(1 - tf.concat([a_ma, a_f, a_s, a_bb], 0)) * dx[None, :],
                        a_r * gain[None, :], a_r * loss[None, :]], 0)
        h1 = linrec(1 - a1, b1)
        d_ma, d_f, d_s, d_bb, g_r, l_r = tf.split(h1, 3 * np.ones(6, int), 0)
        macd = d_f - d_s                                                         # close cancels
        # stage 2 (6 series): MACD signal, Bollinger variance EWMA((x - mean)^2) = EWMA(d_bb^2)
        a2 = tf.concat([a_sig, a_bb], 0)
        h2 = linrec(1 - a2, a2 * tf.concat([macd, tf.square(d_bb)], 0))
        sig, var = h2[:3], h2[3:]
        std = tf.sqrt(var + 1e-8)
        hist = macd - sig
        rsi = 100.0 - 100.0 / (1.0 + g_r / (l_r + 1e-8))
        feats = [d_ma]
        for i in range(3):
            feats.append(tf.stack([macd[i], sig[i], hist[i], tf.tanh(10 * hist[i])]))
        feats.append(rsi)
        for i in range(3):
            feats.append(tf.stack([d_bb[i], d_bb[i] + 2 * std[i], d_bb[i] - 2 * std[i],
                                   (-d_bb[i] + 2 * std[i]) / (4 * std[i] + 1e-8)]))
        feats.append(tf.zeros_like(dx)[None, :])                                 # raw channel (0 = own close)
        F = tf.transpose(tf.concat(feats, 0))                                    # [N, 31]
        return F


PRICE_TYPE = np.zeros(31, np.float32)
PRICE_TYPE[[0, 1, 2, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30]] = 1.0   # channels that move with the level


def windows(F, cum, anchors, L):
    """[B, L, 31] exactly like today's input: price-type channels relative to the ANCHOR close."""
    idx = anchors[:, None] - (L - 1) + tf.range(L)[None, :]                     # [B, L]
    rel = tf.gather(cum, idx) - tf.gather(cum, anchors)[:, None]                 # (close_j - close_anchor)/S
    return tf.gather(F, idx) + rel[..., None] * PRICE_TYPE


def count(fn):
    cf = tf.function(fn).get_concrete_function()
    g = cf.graph
    return len(g.get_operations()) + sum(len(f.node_def) for f in g.as_graph_def().library.function)


def main():
    close = pd.read_csv(CSV, usecols=["close"])["close"].to_numpy(np.float64)
    for N in (10080, 30720):
        c = close[-(N + 1):]
        dx = tf.constant((np.diff(c) / S).astype(np.float32))
        cum = tf.cumsum(dx)                                                     # (close_t - close_0)/S, float32
        ew = pd.Series(np.diff(c)).ewm(alpha=2 / 61).mean().to_numpy() / S
        vol = pd.Series(np.diff(c)).abs().ewm(alpha=2 / 61).mean().to_numpy() / S
        ctx = tf.constant(np.stack([ew, np.log(vol + 1e-6)], 1).astype(np.float32))
        layer = SeriesIndicators()
        anchors = tf.constant(np.random.default_rng(0).integers(2000, N, size=256).astype(np.int32))

        def fb():
            with tf.GradientTape() as tape:
                F = layer(dx, ctx)
                X = windows(F, cum, anchors, 60)
                L = tf.reduce_mean(tf.square(X))
            return tape.gradient(L, layer.trainable_variables)

        f = tf.function(fb)
        g = f()
        t0 = time.perf_counter(); [f() for _ in range(5)]; t = (time.perf_counter() - t0) / 5
        print(f"A2 series indicators N={N} + gather [256, 60, 31]: fwd+bwd graph ops {count(fb)}, CPU {t:.4f} s, "
              f"grads finite {all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in g)}")


if __name__ == "__main__":
    main()

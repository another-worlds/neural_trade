"""A2 prototype layer: today's 18 learnable periods / 31 channels computed ONCE over the pass span of the
series (two kernel stages), with a per-bar meta shift, then [B, L, 31] windows at the batch's anchors
assembled with the deterministic D6 design (fwd gather; bwd transpose-by-gather with a host inverse map).

Per-bar meta shift (replacement for today's per-window meta_adjust, gru_attention.py:47-56): today's
meta input is [mean, max] of the window-relative close of the window ending at the anchor. Here the same
two numbers are computed for EVERY bar t as trailing 60-bar statistics of the data (numpy, causal, no
gradient), and the same Dense(18, tanh) x 0.5 maps them to a logit shift per bar and period.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from kernel import linrec

L = 60
PERIODS = {"ma": [5, 10, 30], "fast": [12, 5, 8], "slow": [26, 35, 17], "sig": [9, 5, 9],
           "rsi": [9, 14, 21], "bb": [10, 20, 25]}
PRICE_TYPE = np.zeros(31, np.float32)
PRICE_TYPE[[0, 1, 2, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30]] = 1.0


def logit_p(p):
    a = 2.0 / (np.asarray(p, np.float64) + 1.0)
    return (np.log(a) - np.log1p(-a)).astype(np.float32)


def context_features(close, scale, L=L):
    """[N, 2] per bar: mean and max over the trailing L bars of (close_j - close_t) / scale (today's meta
    input for the window ending at t). The first L-1 bars use the bars available (causal)."""
    x = np.asarray(close, np.float64)
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.maximum(np.arange(n) - L + 1, 0)
    cnt = np.arange(n) - lo + 1
    mean = (cs[np.arange(n) + 1] - cs[lo]) / cnt
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(L - 1, -np.inf), x])
    mx = sliding_window_view(pad, L).max(axis=1)
    return np.stack([(mean - x) / scale, (mx - x) / scale], 1).astype(np.float32)


class SeriesIndicatorsA2(tf.Module):
    def __init__(self, grad_mult=5.0, meta_scale=0.5, C=64, G=32, contract="mulsum", seed=0):
        super().__init__()
        self.v = {k: tf.Variable(logit_p(p), name=k) for k, p in PERIODS.items()}
        rng = np.random.default_rng(seed)
        self.W = tf.Variable((rng.normal(size=(2, 18)) * 0.5).astype(np.float32))
        self.b = tf.Variable(np.zeros(18, np.float32))
        self.k, self.meta_scale = float(grad_mult), float(meta_scale)
        self.kw = dict(C=C, G=G, contract=contract)

    def _ste(self, v):
        return self.k * v - tf.stop_gradient((self.k - 1.0) * v)

    def series(self, dx, ctx):
        """dx [N] scaled increments (dx_t = close_t - close_{t-1}) / S; ctx [N, 2]. -> F [N, 31] with every
        price-level channel relative to its own bar's close."""
        # per-bar shift without a matmul (2 inputs): exact fp32 on every device
        m = tf.tanh(ctx[:, 0:1] * self.W[0] + ctx[:, 1:2] * self.W[1] + self.b)       # [N, 18]
        m = tf.transpose(m) * self.meta_scale                                          # [18, N]
        base = tf.concat([self._ste(self.v[k]) for k in ("ma", "fast", "slow", "bb", "rsi", "sig")], 0)
        lam = base[:, None] + m                                                        # [18, N]
        la = -tf.nn.softplus(lam)
        dec, al = tf.exp(la), tf.sigmoid(lam)
        la_ma, la_f, la_s, la_bb, la_r, la_sig = tf.split(la, 6, 0)
        dec_px = dec[:12]                                                              # ma, fast, slow, bb
        al_r, al_sig, al_bb = al[12:15], al[15:18], al[9:12]
        gain, loss = tf.nn.relu(dx), tf.nn.relu(-dx)
        la1 = tf.concat([la[:12], la_r, la_r], 0)                                      # 18 stage-1 channels
        b1 = tf.concat([-dec_px * dx[None, :], al_r * gain[None, :], al_r * loss[None, :]], 0)
        h1, _ = linrec(la1, b1, **self.kw)
        d_ma, d_f, d_s, d_bb, g_r, l_r = tf.split(h1, 6, 0)
        macd = d_f - d_s
        la2 = tf.concat([la_sig, la_bb], 0)                                            # 6 stage-2 channels
        b2 = tf.concat([al_sig * macd, al_bb * tf.square(d_bb)], 0)
        h2, _ = linrec(la2, b2, **self.kw)
        sig, var = h2[:3], h2[3:]
        std = tf.sqrt(var + 1e-8)
        hist = macd - sig
        rsi = 100.0 - 100.0 / (1.0 + g_r / (l_r + 1e-8))
        feats = [d_ma]
        for i in range(3):
            feats.append(tf.stack([macd[i], sig[i], hist[i], tf.tanh(10.0 * hist[i])]))
        feats.append(rsi)
        for i in range(3):
            feats.append(tf.stack([d_bb[i], d_bb[i] + 2 * std[i], d_bb[i] - 2 * std[i],
                                   (-d_bb[i] + 2 * std[i]) / (4 * std[i] + 1e-8)]))
        feats.append(tf.zeros_like(dx)[None, :])                                      # raw close (0 = own)
        return tf.transpose(tf.concat(feats, 0))                                       # [N, 31]

    @property
    def trainable(self):
        return list(self.v.values()) + [self.W, self.b]


def assemble_D6(F, cum, anchors, inv):
    """[B, L, C] windows: F at the window bars + (close_j - close_anchor)/S on the price-level channels.
    fwd: gather; bwd: transpose by gather through the host inverse map inv [N + L - 1] (B = no anchor)."""
    B = anchors.shape[0]
    N, C = F.shape
    idx = anchors[:, None] - (L - 1) + tf.range(L)[None, :]

    @tf.custom_gradient
    def g(F):
        def grad(dy):
            nl = tf.range(N)[:, None] + (L - 1) - tf.range(L)[None, :]
            flat = tf.gather(inv, nl) * L + tf.range(L)[None, :]
            dyp = tf.concat([tf.reshape(dy, [B * L, C]), tf.zeros([L, C], dy.dtype)], 0)
            return tf.reduce_sum(tf.gather(dyp, flat), axis=1)
        return tf.gather(F, idx), grad
    rel = tf.gather(cum, idx) - tf.gather(cum, anchors)[:, None]
    return g(F) + rel[..., None] * tf.constant(PRICE_TYPE[:C])

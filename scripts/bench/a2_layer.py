"""The whole A2 series indicator layer (NT-059; window-free plan README "Answer" (1) and "Kernel V1").

Today's 18 learnable EWMA periods -> 31 indicator channels
(``src/neural_trade/models/layers/learnable_indicators.py``), computed ONCE per step over the whole
pass span with ``kernel_v1.linrec`` (two kernel stages: increment / RSI-gain-loss, then
MACD-signal / Bollinger-variance), with a per-bar meta shift - the per-window ``meta_adjust``'s
replacement: the same [mean, max] context features, computed for every bar instead of once per window,
and applied as an elementwise multiply-add (window-free plan README "Per-bar adaptive periods": "no
MatMul, so TF32 cannot touch it") - then ``[B, 60, 31]`` windows assembled at the batch's anchors with
``assembly_d6b.assemble`` (D6b).

Adapted from ``docs/research/2026-09-29-window-free-plan/A/a2layer.py`` (the NT-053 research
prototype).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

import assembly_d6b
import kernel_v1

L = 60
PERIODS = {"ma": [5, 10, 30], "fast": [12, 5, 8], "slow": [26, 35, 17], "sig": [9, 5, 9],
           "rsi": [9, 14, 21], "bb": [10, 20, 25]}
PRICE_TYPE = np.zeros(31, np.float32)
PRICE_TYPE[[0, 1, 2, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30]] = 1.0    # price-level channels


def logit_p(p):
    a = 2.0 / (np.asarray(p, np.float64) + 1.0)
    return (np.log(a) - np.log1p(-a)).astype(np.float32)


def context_features(close, scale, L=L):
    """``[N, 2]`` per bar: the mean and the max, over the trailing L bars, of
    ``(close_j - close_t) / scale`` - today's meta input for the window ending at t, computed here for
    every bar t (causal; the first L-1 bars use the bars available)."""
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
    """The whole A2 indicator layer; ``series()`` is the benchmarked forward pass over the series."""

    def __init__(self, grad_mult=5.0, meta_scale=0.5, C=16, G=32, seed=0):
        super().__init__()
        self.v = {k: tf.Variable(logit_p(p), name=k) for k, p in PERIODS.items()}
        rng = np.random.default_rng(seed)
        self.W = tf.Variable((rng.normal(size=(2, 18)) * 0.5).astype(np.float32))
        self.b = tf.Variable(np.zeros(18, np.float32))
        self.k, self.meta_scale = float(grad_mult), float(meta_scale)
        self.kw = dict(C=C, G=G)

    def _ste(self, v):
        return self.k * v - tf.stop_gradient((self.k - 1.0) * v)

    def series(self, dx, ctx):
        """``dx [N]``: scaled increments ``(close_t - close_{t-1}) / S``. ``ctx [N, 2]``:
        ``context_features``. Returns ``F [N, 31]``, every price-level channel relative to its own
        bar's close."""
        # per-bar logit shift without a matmul (2 scalar inputs): exact fp32 on every device
        m = tf.tanh(ctx[:, 0:1] * self.W[0] + ctx[:, 1:2] * self.W[1] + self.b)     # [N, 18]
        m = tf.transpose(m) * self.meta_scale                                        # [18, N]
        base = tf.concat([self._ste(self.v[k]) for k in ("ma", "fast", "slow", "bb", "rsi", "sig")], 0)
        lam = base[:, None] + m                                                      # [18, N] per-bar logits
        la = -tf.nn.softplus(lam)
        dec, al = tf.exp(la), tf.sigmoid(lam)
        la_ma, la_f, la_s, la_bb, la_r, la_sig = tf.split(la, 6, 0)
        dec_px = dec[:12]                                                            # ma, fast, slow, bb
        al_r, al_sig, al_bb = al[12:15], al[15:18], al[9:12]
        gain, loss = tf.nn.relu(dx), tf.nn.relu(-dx)
        la1 = tf.concat([la[:12], la_r, la_r], 0)                                    # 18 stage-1 channels
        b1 = tf.concat([-dec_px * dx[None, :], al_r * gain[None, :], al_r * loss[None, :]], 0)
        h1, _ = kernel_v1.linrec(la1, b1, **self.kw)
        d_ma, d_f, d_s, d_bb, g_r, l_r = tf.split(h1, 6, 0)
        macd = d_f - d_s
        la2 = tf.concat([la_sig, la_bb], 0)                                          # 6 stage-2 channels
        b2 = tf.concat([al_sig * macd, al_bb * tf.square(d_bb)], 0)
        h2, _ = kernel_v1.linrec(la2, b2, **self.kw)
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
        feats.append(tf.zeros_like(dx)[None, :])                                    # raw close (0 = own bar)
        return tf.transpose(tf.concat(feats, 0))                                    # [N, 31]

    @property
    def trainable(self):
        return list(self.v.values()) + [self.W, self.b]


def assemble(F, cum, anchors, inv, L=L):
    """``[B, L, 31]`` windows: F at the window bars plus ``(close_j - close_anchor) / S`` on the
    price-level channels; D6b (``assembly_d6b.assemble``) for the deterministic gradient."""
    idx = anchors[:, None] - (L - 1) + tf.range(L)[None, :]
    rel = tf.gather(cum, idx) - tf.gather(cum, anchors)[:, None]
    return assembly_d6b.assemble(F, anchors, inv, L) + rel[..., None] * tf.constant(PRICE_TYPE)

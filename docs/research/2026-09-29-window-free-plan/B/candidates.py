"""The three series-mode replacements for today's per-window adaptive periods (NT-053 part B).

Today's 18 learnable periods -> 31 channels (learnable_indicators.py:124-182), computed ONCE over a
series of scaled increments dx_t = (close_t - close_{t-1}) / scale, every price-type channel relative to
its OWN bar's close (increment form d_t = EMA_t - close_t = (1 - a_t)(d_{t-1} - dx_t); first round's
hybrid_proto.py). The network would then gather [B, 60, 31] windows at the anchors (not done here:
the gather is common to every candidate and is part A's design).

mode "c"  fixed: alpha_i = sigmoid(logit_i) for every bar (the global learned period; D-031's off switch).
mode "a"  per-bar: alpha_{i,t} = sigmoid(logit_i + 0.5 * tanh(ctx_t @ W + b)_i), ctx_t = 2 causal context
          features at bar t (today's meta_adjust network, applied per bar instead of per window).
mode "b"  bank: M fixed periods per instance at logit_i + o_m, o = linspace(-0.5, 0.5, M), each a
          constant-alpha EWMA; the output at bar t mixes the members with hat weights of the same
          per-bar shift delta_{i,t}: the bar gets (approximately) the fixed-period EWMA at its own period.

Order of the 18 logits = common.NAMES (the layer's variable order).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from kernels import linrec_chunked, linrec_toeplitz

G = {"ma": [0, 1, 2], "fast": [3, 6, 9], "slow": [4, 7, 10], "sig": [5, 8, 11], "rsi": [12, 13, 14],
     "bb": [15, 16, 17]}
CLAMP = 1e-6


def _alpha(z):
    return tf.clip_by_value(tf.sigmoid(z), CLAMP, 1.0 - CLAMP)


def hat_weights(delta, offsets):
    """delta [..., N] in [-0.5, 0.5], offsets [M] evenly spaced -> [..., M, N] piecewise-linear weights."""
    step = float(offsets[1] - offsets[0])
    o = tf.constant(np.asarray(offsets, np.float32))
    return tf.nn.relu(1.0 - tf.abs(delta[..., None, :] - o[:, None]) / step)


class SeriesLayer(tf.Module):
    def __init__(self, mode, logits, W, b, M=3, C=64):
        super().__init__()
        self.mode, self.M, self.C = mode, int(M), int(C)
        self.lg = tf.Variable(np.asarray(logits, np.float32), name="logits")
        self.W = tf.Variable(np.asarray(W, np.float32), name="meta_kernel")
        self.b = tf.Variable(np.asarray(b, np.float32), name="meta_bias")
        self.offsets = np.linspace(-0.5, 0.5, self.M) if self.M > 1 else np.zeros(1)

    def shift(self, ctx):
        """[N, 2] -> [18, N]: the per-bar logit shift (bounded by +-0.5 by the tanh)."""
        return tf.transpose(0.5 * tf.tanh(tf.matmul(ctx, self.W) + self.b))

    # -------------------------------------------------------------------------------- runs
    def _run_fixed(self, alpha, b):
        """constant alpha per channel [K], input b [K, N] -> states [K, N] (Toeplitz kernel)."""
        return linrec_toeplitz(tf.math.log1p(-alpha), b, self.C)

    def _run_perbar(self, alpha, b):
        return linrec_chunked(1.0 - alpha, b, self.C)

    def __call__(self, dx, ctx):
        dx = tf.convert_to_tensor(dx, tf.float32)
        gain, loss = tf.nn.relu(dx), tf.nn.relu(-dx)
        if self.mode == "a":
            return self._perbar(dx, gain, loss, self.shift(ctx))
        if self.mode == "c":
            return self._fixed(dx, gain, loss)
        if self.mode == "b":
            return self._bank(dx, gain, loss, self.shift(ctx))
        raise ValueError(self.mode)

    def alphas(self, ctx):
        """The per-bar alpha of every instance [18, N] as mode a applies it (for reporting)."""
        return _alpha(self.lg[:, None] + self.shift(ctx))

    # -------------------------------------------------------------------------------- modes
    def _outputs(self, d_ma, d_f, d_s, d_bb, g, l_, sig, var):
        macd = d_f - d_s
        std = tf.sqrt(var + 1e-8)
        hist = macd - sig
        rsi = 100.0 - 100.0 / (1.0 + g / (l_ + 1e-8))
        feats = [d_ma]
        for i in range(3):
            feats.append(tf.stack([macd[i], sig[i], hist[i], tf.tanh(10.0 * hist[i])]))
        feats.append(rsi)
        for i in range(3):
            feats.append(tf.stack([d_bb[i], d_bb[i] + 2 * std[i], d_bb[i] - 2 * std[i],
                                   (-d_bb[i] + 2 * std[i]) / (4 * std[i] + 1e-8)]))
        feats.append(tf.zeros_like(d_ma[:1]))            # raw close channel = 0 relative to its own close
        return tf.transpose(tf.concat(feats, 0))         # [N, 31]

    def _fixed(self, dx, gain, loss):
        al = _alpha(self.lg)                                                      # [18]
        grp = lambda k: tf.gather(al, G[k])
        a1 = tf.concat([grp("ma"), grp("fast"), grp("slow"), grp("bb"), grp("rsi"), grp("rsi")], 0)
        dtype_a = tf.concat([grp("ma"), grp("fast"), grp("slow"), grp("bb")], 0)
        b1 = tf.concat([-(1 - dtype_a)[:, None] * dx[None, :], grp("rsi")[:, None] * gain[None, :],
                        grp("rsi")[:, None] * loss[None, :]], 0)
        h1 = self._run_fixed(a1, b1)
        d_ma, d_f, d_s, d_bb, g, l_ = tf.split(h1, 6, 0)
        a2 = tf.concat([grp("sig"), grp("bb")], 0)
        h2 = self._run_fixed(a2, a2[:, None] * tf.concat([d_f - d_s, tf.square(d_bb)], 0))
        return self._outputs(d_ma, d_f, d_s, d_bb, g, l_, h2[:3], h2[3:])

    def _perbar(self, dx, gain, loss, delta):
        al = _alpha(self.lg[:, None] + delta)                                     # [18, N]
        grp = lambda k: tf.gather(al, G[k])
        dtype_a = tf.concat([grp("ma"), grp("fast"), grp("slow"), grp("bb")], 0)
        a1 = tf.concat([dtype_a, grp("rsi"), grp("rsi")], 0)
        b1 = tf.concat([-(1 - dtype_a) * dx[None, :], grp("rsi") * gain[None, :], grp("rsi") * loss[None, :]], 0)
        h1 = self._run_perbar(a1, b1)
        d_ma, d_f, d_s, d_bb, g, l_ = tf.split(h1, 6, 0)
        a2 = tf.concat([grp("sig"), grp("bb")], 0)
        h2 = self._run_perbar(a2, a2 * tf.concat([d_f - d_s, tf.square(d_bb)], 0))
        return self._outputs(d_ma, d_f, d_s, d_bb, g, l_, h2[:3], h2[3:])

    def _bank(self, dx, gain, loss, delta):
        M = self.M
        off = tf.constant(np.asarray(self.offsets, np.float32))
        al = _alpha(self.lg[:, None] + off[None, :])                              # [18, M] constant
        w = hat_weights(delta, self.offsets)                                      # [18, M, N]
        grp = lambda t, k: tf.gather(t, G[k])
        dtype_a = tf.concat([grp(al, "ma"), grp(al, "fast"), grp(al, "slow"), grp(al, "bb")], 0)   # [12, M]
        a_r = grp(al, "rsi")                                                      # [3, M]
        a1 = tf.reshape(tf.concat([dtype_a, a_r, a_r], 0), [-1])                  # [18 M]
        b1 = tf.concat([tf.reshape(-(1 - dtype_a)[..., None] * dx[None, None, :], [12 * M, -1]),
                        tf.reshape(a_r[..., None] * gain[None, None, :], [3 * M, -1]),
                        tf.reshape(a_r[..., None] * loss[None, None, :], [3 * M, -1])], 0)
        h1 = tf.reshape(self._run_fixed(a1, b1), [18, M, -1])                     # members
        w1 = tf.concat([grp(w, "ma"), grp(w, "fast"), grp(w, "slow"), grp(w, "bb"), grp(w, "rsi"), grp(w, "rsi")], 0)
        mix = tf.reduce_sum(w1 * h1, axis=1)                                      # [18, N]
        d_ma, d_f, d_s, d_bb, g, l_ = tf.split(mix, 6, 0)
        d_bb_m = h1[9:12]                                                         # members' own mean deviation
        a_sig, a_bb = grp(al, "sig"), grp(al, "bb")
        a2 = tf.reshape(tf.concat([a_sig, a_bb], 0), [-1])                        # [6 M]
        line = d_f - d_s
        b2 = tf.concat([tf.reshape(a_sig[..., None] * line[:, None, :], [3 * M, -1]),
                        tf.reshape(a_bb[..., None] * tf.square(d_bb_m), [3 * M, -1])], 0)
        h2 = tf.reshape(self._run_fixed(a2, b2), [6, M, -1])
        w2 = tf.concat([grp(w, "sig"), grp(w, "bb")], 0)
        mix2 = tf.reduce_sum(w2 * h2, axis=1)
        return self._outputs(d_ma, d_f, d_s, d_bb, g, l_, mix2[:3], mix2[3:])


# ------------------------------------------------------------------------------------ float64 reference
def reference64(mode, dx, logits, delta, M=3):
    """numpy float64 version of SeriesLayer (same formulas, sequential recursion). delta [18, N] or None."""
    from common import linrec64
    dx = np.asarray(dx, np.float64)
    gain, loss = np.maximum(dx, 0), np.maximum(-dx, 0)
    lg = np.asarray(logits, np.float64)
    clamp = lambda a: np.clip(a, CLAMP, 1 - CLAMP)
    sig = lambda z: 1 / (1 + np.exp(-z))

    def run(al, bb):                      # al [K] or [K, N]
        al = np.broadcast_to(al if al.ndim == 2 else al[:, None], bb.shape)
        return linrec64(1 - al, bb)

    def outputs(d_ma, d_f, d_s, d_bb, g, l_, sg, var):
        macd = d_f - d_s; std = np.sqrt(var + 1e-8); hist = macd - sg
        rsi = 100 - 100 / (1 + g / (l_ + 1e-8))
        feats = [d_ma]
        for i in range(3):
            feats.append(np.stack([macd[i], sg[i], hist[i], np.tanh(10 * hist[i])]))
        feats.append(rsi)
        for i in range(3):
            feats.append(np.stack([d_bb[i], d_bb[i] + 2 * std[i], d_bb[i] - 2 * std[i],
                                   (-d_bb[i] + 2 * std[i]) / (4 * std[i] + 1e-8)]))
        feats.append(np.zeros_like(d_ma[:1]))
        return np.concatenate(feats, 0).T

    if mode in ("a", "c"):
        al = clamp(sig(lg[:, None] + (delta if mode == "a" else 0.0))) * np.ones((1, len(dx)))
        grp = lambda k: al[G[k]]
        dA = np.concatenate([grp("ma"), grp("fast"), grp("slow"), grp("bb")])
        h1 = run(np.concatenate([dA, grp("rsi"), grp("rsi")]),
                 np.concatenate([-(1 - dA) * dx, grp("rsi") * gain, grp("rsi") * loss]))
        d_ma, d_f, d_s, d_bb, g, l_ = np.split(h1, 6)
        a2 = np.concatenate([grp("sig"), grp("bb")])
        h2 = run(a2, a2 * np.concatenate([d_f - d_s, d_bb ** 2]))
        return outputs(d_ma, d_f, d_s, d_bb, g, l_, h2[:3], h2[3:])
    # bank
    off = np.linspace(-0.5, 0.5, M)
    al = clamp(sig(lg[:, None] + off[None, :]))                                   # [18, M]
    step = off[1] - off[0]
    w = np.maximum(0, 1 - np.abs(delta[:, None, :] - off[None, :, None]) / step)  # [18, M, N]
    out = {}
    for k in G:
        out[k] = np.zeros((3, M, len(dx)))
    for m in range(M):
        for k in ("ma", "fast", "slow", "bb"):
            a = al[G[k], m]
            out[k][:, m] = run(a, -(1 - a)[:, None] * dx[None, :])
        a = al[G["rsi"], m]
        out.setdefault("g", np.zeros((3, M, len(dx))))
        out.setdefault("l", np.zeros((3, M, len(dx))))
        out["g"][:, m] = run(a, a[:, None] * gain[None, :])
        out["l"][:, m] = run(a, a[:, None] * loss[None, :])
    mixk = lambda arr, k: (w[G[k]] * arr).sum(1)
    d_ma, d_f, d_s, d_bb = (mixk(out[k], k) for k in ("ma", "fast", "slow", "bb"))
    g, l_ = mixk(out["g"], "rsi"), mixk(out["l"], "rsi")
    line = d_f - d_s
    sgm = np.zeros((3, M, len(dx))); vm = np.zeros((3, M, len(dx)))
    for m in range(M):
        a = al[G["sig"], m]; sgm[:, m] = run(a, a[:, None] * line)
        a = al[G["bb"], m]; vm[:, m] = run(a, a[:, None] * out["bb"][:, m] ** 2)
    return outputs(d_ma, d_f, d_s, d_bb, g, l_, mixk(sgm, "sig"), mixk(vm, "bb"))

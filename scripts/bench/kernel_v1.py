"""Series indicator kernel V1 (NT-059; window-free plan README "Kernel V1").

The causal recurrence ``h_t = exp(la_t) * h_{t-1} + b_t`` over K independent channels, where ``la`` is
the LOG decay (``la = -softplus(logit) * dt`` for elapsed-time gaps, or ``NEG`` for a state reset:
decay exactly 0). Implemented as a hierarchical chunked recurrence so every cumsum stays local to a
chunk (at most C bars) or a block of chunks (at most G), which is what keeps periods of 1,440 bars and
longer inside the 1e-5 x max|state|, 3e-5 x RMS tolerance
(``docs/research/2026-09-29-window-free-plan/A/FINDINGS.md`` Q1): a difference of two long cumsums
cancels at float32 precision, and rounding ``1 - alpha`` to float32 is a systematic period error; the
log-decay input with local segment sums avoids both. The recommended chunk size is C = 16 (the
smallest memory and CPU cost that still passes every tolerance). The contraction is an elementwise
multiply plus ``reduce_sum``, never a matmul or an einsum: TF 2.10 runs matmul/einsum in
TensorFloat-32 on an Ampere/Ada GPU by default, which breaks these tolerances by 20-200x
(A/FINDINGS.md Q5).

Adapted from ``docs/research/2026-09-29-window-free-plan/A/kernel.py`` (the NT-053 research
prototype); this module keeps only the recommended form (log-decay input, segment-sum decay matrix,
elementwise-multiply contraction) and drops the einsum / TF32-emulation / global-cumsum variants that
prototype kept only to demonstrate why they fail.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

NEG = -1e30   # a reset: exp(NEG) == 0.0 in float32 exactly, for any earlier state


def _masks(m):
    i = np.arange(m)
    return tf.constant(i[:, None] > i[None, :]), tf.constant(i[:, None] >= i[None, :])


def segsum(la):
    """[..., m] -> [..., m, m]: ``S[i, j] = sum_{k=j+1..i} la_k`` for i >= j (0 on the diagonal), NEG
    above the diagonal. A direct segment sum (a cumsum of the masked terms): every entry is a sum of
    at most m consecutive terms, never a difference of two long cumsums, so it never cancels."""
    m = la.shape[-1]
    strict, low = _masks(m)
    x = tf.broadcast_to(la[..., :, None], la.shape.as_list() + [m])   # x[..., i, j] = la_i
    x = tf.where(strict, x, tf.zeros_like(x))
    s = tf.cumsum(x, axis=-2)
    return tf.where(low, s, tf.constant(NEG, s.dtype))


def _contract(D, b):
    """Elementwise multiply + reduce_sum: no MatMul, BatchMatMul or Einsum (never TF32-rounded)."""
    return tf.reduce_sum(D * b[..., None, :], axis=-1)


def _level(la, b, h_in, C, G):
    K, T = la.shape.as_list()
    if T <= C:                                    # one block: the top of the hierarchy
        intra = _contract(tf.exp(segsum(la)), b)
        return intra + tf.exp(tf.cumsum(la, axis=-1)) * h_in[:, None]
    pad = (-T) % C
    if pad:                                        # future end only: causal
        la = tf.concat([la, tf.zeros([K, pad], la.dtype)], -1)
        b = tf.concat([b, tf.zeros([K, pad], b.dtype)], -1)
    n = (T + pad) // C
    LA, BB = tf.reshape(la, [K, n, C]), tf.reshape(b, [K, n, C])
    intra = _contract(tf.exp(segsum(LA)), BB)                          # [K, n, C]
    cs = tf.cumsum(LA, axis=-1)                                         # local inclusive, <= C terms
    H = _level(cs[..., -1], intra[..., -1], h_in, G, G)                # state at every chunk end
    H_prev = tf.concat([h_in[:, None], H[:, :-1]], -1)                 # state entering every chunk
    h = intra + tf.exp(cs) * H_prev[..., None]
    return tf.reshape(h, [K, n * C])[:, :T]


def linrec(la, b, h0=None, C=16, G=32, check_finite=False):
    """Returns ``(h [K, T], h_last [K])``. ``la``, ``b``: ``[K, T]`` (float32 or float64), T static.

    ``h0``: the carried state entering bar 0 (default 0). A reset within ``la`` is ``la = NEG``, which
    makes every later output bitwise independent of everything before the reset (a zero Jacobian).
    ``check_finite``: refuse non-finite ``la`` or ``b`` with an ``InvalidArgumentError`` (the
    finite-input contract; D-018 keeps this off the per-step path by default and checks at data load
    instead, since one NaN would otherwise corrupt most earlier outputs).
    """
    la = tf.convert_to_tensor(la)
    b = tf.convert_to_tensor(b, la.dtype)
    if check_finite:
        la = tf.debugging.check_numerics(la, "linrec: non-finite log decay")
        b = tf.debugging.check_numerics(b, "linrec: non-finite input")
    K = la.shape[0]
    h_in = tf.zeros([K], la.dtype) if h0 is None else tf.cast(tf.convert_to_tensor(h0), la.dtype)
    h = _level(la, b, h_in, int(C), int(G))
    return h, h[:, -1]


# --------------------------------------------------------------------------------- float64 reference
def ref_linrec(la, b, h0=None):
    """Sequential float64 recursion (numpy); the precision reference for `linrec`."""
    la = np.asarray(la, np.float64)
    b = np.asarray(b, np.float64)
    a = np.exp(la)
    K, T = la.shape
    h = np.empty((K, T))
    prev = np.zeros(K) if h0 is None else np.asarray(h0, np.float64).copy()
    for t in range(T):
        prev = a[:, t] * prev + b[:, t]
        h[:, t] = prev
    return h


def softplus64(x):
    return np.logaddexp(0.0, np.asarray(x, np.float64))


def sigmoid64(x):
    x = np.asarray(x, np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def logit_period(p):
    """logit of alpha = 2 / (p + 1) (float64)."""
    a = 2.0 / (np.asarray(p, np.float64) + 1.0)
    return np.log(a) - np.log1p(-a)


def burn_in(period_max, eps=1e-3, shift=0.5):
    """M(eps): bars until the weight of the state before the pass start falls below eps, for the
    longest period after the maximal negative meta shift (logit - shift)."""
    a = sigmoid64(logit_period(period_max) - shift)
    return int(np.ceil(np.log(eps) / np.log1p(-a)))

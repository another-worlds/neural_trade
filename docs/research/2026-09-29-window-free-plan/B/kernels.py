"""Causal first-order recurrences h_t = a_t h_{t-1} + b_t over long series, TF 2.10 (CPU prototypes).

Written for part B (the per-bar adaptation candidates); part A owns the production kernel. Both forms
are the first round's two-level chunked idea (docs/research/2026-09-28-window-free/prototypes/
scan_proto.py:scan_chunk), with cumsums local to a chunk only and the carry across chunks done by a
Hillis-Steele scan over the n chunk states (no global cumsum of log-decays, which the first round's
mat2_proto.scan_mat2 used and the Challenge flagged).

* ``linrec_chunked(a, b, C)``: general (per-bar) a_t. Intra-chunk decay matrices [K, n, C, C].
* ``linrec_toeplitz(log_a, b, C)``: a constant per channel (log_a [K]). One decay matrix [K, C, C]
  shared by every chunk (the Toeplitz specialisation a fixed period allows).
All shapes are static (T known when traced); the series is padded at its FUTURE end (causal).
"""
from __future__ import annotations

import tensorflow as tf

NEG = -1e30


def hs_scan(a, b):
    """Inclusive Hillis-Steele scan of the affine maps (a_t, b_t) along the last (static) axis."""
    T = a.shape[-1]
    d = 1
    while d < T:
        a_prev = tf.concat([tf.ones_like(a[..., :d]), a[..., :-d]], axis=-1)
        b_prev = tf.concat([tf.zeros_like(b[..., :d]), b[..., :-d]], axis=-1)
        b = a * b_prev + b
        a = a * a_prev
        d *= 2
    return b


def _pad(x, pad, value):
    if not pad:
        return x
    return tf.concat([x, tf.fill(tf.concat([tf.shape(x)[:-1], [pad]], 0), tf.constant(value, x.dtype))], axis=-1)


def linrec_chunked(a, b, C=64):
    """a, b [K, T] (a in (0, 1]) -> h [K, T], h_{-1} = 0. Per-bar a_t."""
    K, T = a.shape
    pad = (-T) % C
    a = _pad(a, pad, 1.0)
    b = _pad(b, pad, 0.0)
    n = (T + pad) // C
    cs = tf.cumsum(tf.math.log(tf.reshape(a, [K, n, C])), axis=-1)            # local: |cs| <= C |log a|
    i = tf.range(C)
    lower = (i[:, None] >= i[None, :])
    diff = cs[..., :, None] - cs[..., None, :]
    L = tf.exp(tf.where(lower, diff, tf.fill(tf.shape(diff), tf.constant(NEG, a.dtype))))
    intra = tf.einsum("knij,knj->kni", L, tf.reshape(b, [K, n, C]))           # state if the chunk started at 0
    H = hs_scan(tf.exp(cs[..., -1]), intra[..., -1])                           # state at each chunk end
    H_prev = tf.concat([tf.zeros_like(H[..., :1]), H[..., :-1]], axis=-1)
    h = intra + tf.exp(cs) * H_prev[..., None]
    return tf.reshape(h, [K, n * C])[:, :T]


def linrec_toeplitz(log_a, b, C=64):
    """log_a [K] (constant decay per channel, log a <= 0), b [K, T] -> h [K, T], h_{-1} = 0."""
    K, T = b.shape
    pad = (-T) % C
    b = _pad(b, pad, 0.0)
    n = (T + pad) // C
    i = tf.range(C)
    lag = tf.cast(i[:, None] - i[None, :], log_a.dtype)                        # [C, C]
    lower = (i[:, None] >= i[None, :])
    e = lag[None] * log_a[:, None, None]
    L = tf.exp(tf.where(lower[None], e, tf.fill(tf.shape(e), tf.constant(NEG, log_a.dtype))))   # [K, C, C]
    intra = tf.einsum("kij,knj->kni", L, tf.reshape(b, [K, n, C]))
    A = tf.exp(tf.cast(C, log_a.dtype) * log_a)[:, None] * tf.ones([1, n], log_a.dtype)
    H = hs_scan(A, intra[..., -1])
    H_prev = tf.concat([tf.zeros_like(H[..., :1]), H[..., :-1]], axis=-1)
    carry = tf.exp(tf.cast(i + 1, log_a.dtype)[None, :] * log_a[:, None])     # a^(i+1) [K, C]
    h = intra + carry[:, None, :] * H_prev[..., None]
    return tf.reshape(h, [K, n * C])[:, :T]


def linrec_factored(a, b, C=32):
    """Per-bar a_t without any C x C matrix: inside a chunk the decay factorises,
    exp(cs_i - cs_j) = exp(cs_i) * exp(-cs_j), so intra_i = exp(cs_i) * cumsum_j(exp(-cs_j) b_j).
    Memory O(K T). Needs exp(-cs) finite: C * max|log a_t| < ~80 (float32), i.e. a_t >= exp(-80 / C)
    (C = 32: alpha <= 0.918, period >= 1.18; today's applied floor is period 1.6, alpha 0.767)."""
    K, T = a.shape
    pad = (-T) % C
    a = _pad(a, pad, 1.0)
    b = _pad(b, pad, 0.0)
    n = (T + pad) // C
    cs = tf.cumsum(tf.math.log(tf.reshape(a, [K, n, C])), axis=-1)            # local, in [-C|log a|, 0]
    e = tf.exp(cs)
    intra = e * tf.cumsum(tf.exp(-cs) * tf.reshape(b, [K, n, C]), axis=-1)
    H = hs_scan(e[..., -1], intra[..., -1])
    H_prev = tf.concat([tf.zeros_like(H[..., :1]), H[..., :-1]], axis=-1)
    h = intra + e * H_prev[..., None]
    return tf.reshape(h, [K, n * C])[:, :T]


def graph_ops(fn, *args):
    """Graph size of a traced function (top-level ops + ops of nested library functions)."""
    cf = tf.function(fn).get_concrete_function(*args)
    g = cf.graph
    return len(g.get_operations()) + sum(len(f.node_def) for f in g.as_graph_def().library.function)

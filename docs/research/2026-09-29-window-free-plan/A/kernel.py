"""Series indicator kernel prototype (NT-053 part A, Q1).

    h_t = exp(la_t) * h_{t-1} + b_t,   t = 0..T-1,   h_{-1} = h0 (the carried state in; default 0)

over K independent channels ([K, T] tensors, T static). `la` is the LOG decay (<= 0): callers pass
la = -softplus(logit) (= log(1 - sigmoid(logit)), exact in float32 for any logit, also as alpha -> 0),
times the elapsed bars dt for gaps, or NEG (= -1e30, decay exactly 0) for a state reset.

Hierarchical chunked matrix form ("chunk-local cumsums only"):
  level 1: the series is cut into chunks of C bars; inside a chunk the C x C decay matrix
           D[i, j] = exp(sum_{k=j+1..i} la_k) (i >= j, else 0) is built from a SEGMENT SUM: a cumulative
           sum of the masked terms down each column, so every entry is a direct sum of consecutive
           terms (at most C of them), never a difference of two long cumsums (no cancellation);
           intra = D @ b is the state each chunk would reach from a zero start;
  level 2+: the chunk-end maps (log P_c = sum of la over the chunk, s_c = intra at the chunk end) obey
           the same recurrence H_c = P_c H_{c-1} + s_c, solved by the same function with block size G
           (so its cumsums are local to G chunks), recursively, until one block of <= G is left;
  combine: h = intra + exp(cumsum_local(la)) * H_{c-1}.
Padding (only ever at the FUTURE end, la = 0 and b = 0) keeps every earlier output unchanged.

Variants kept for the measurements: seg="segsub" (local cumsum, then cs_i - cs_j; cancels) and
G=10**9 with segsub (= the first round's mat2_proto.scan_mat2: one global cumsum over all chunk
decays); contract="einsum" (BatchMatMulV2: TensorFloat-32 on an Ampere/Ada GPU while TF32 is on, the
TF 2.10 default) or "mulsum" (elementwise multiply + reduce_sum: fp32 on every device).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

NEG = -1e30


def _masks(m):
    i = np.arange(m)
    return tf.constant(i[:, None] > i[None, :]), tf.constant(i[:, None] >= i[None, :])


def segsum(la):
    """[..., m] -> [..., m, m]: S[i, j] = sum_{k=j+1..i} la_k for i >= j (0 on the diagonal), NEG above."""
    m = la.shape[-1]
    strict, low = _masks(m)
    x = tf.broadcast_to(la[..., :, None], la.shape.as_list() + [m])     # x[..., i, j] = la_i
    x = tf.where(strict, x, tf.zeros_like(x))
    s = tf.cumsum(x, axis=-2)
    return tf.where(low, s, tf.constant(NEG, s.dtype))


def segsub(la):
    """Same matrix as cs_i - cs_j of a local inclusive cumsum (the first round's form; cancels in float32
    for short periods in long chunks, and turns a NEG reset into garbage)."""
    m = la.shape[-1]
    _, low = _masks(m)
    cs = tf.cumsum(la, axis=-1)
    s = cs[..., :, None] - cs[..., None, :]
    return tf.where(low, s, tf.constant(NEG, s.dtype))


def round_tf32(x, mode):
    """Emulate the fp32 -> TF32 input rounding of a tensor-core matmul (10 explicit mantissa bits).
    mode 'rn': round to nearest (ties away); 'rz': truncate."""
    if x.dtype != tf.float32:
        return x
    i = tf.bitcast(x, tf.int32)
    if mode == "rn":
        i = i + 0x1000
    i = tf.bitwise.bitwise_and(i, tf.constant(-8192, tf.int32))
    return tf.bitcast(i, tf.float32)


def _contract(D, b, mode, tf32):
    if tf32:
        D, b = round_tf32(D, tf32), round_tf32(b, tf32)
    if mode == "einsum":
        return tf.linalg.matvec(D, b)
    return tf.reduce_sum(D * b[..., None, :], axis=-1)


def _level(la, b, h_in, C, G, seg, mode, tf32):
    K, T = la.shape.as_list()
    if T <= C:                                   # one block: the top of the hierarchy
        intra = _contract(tf.exp(seg(la)), b, mode, tf32)
        return intra + tf.exp(tf.cumsum(la, axis=-1)) * h_in[:, None]
    pad = (-T) % C
    if pad:                                      # future end only: causal
        la = tf.concat([la, tf.zeros([K, pad], la.dtype)], -1)
        b = tf.concat([b, tf.zeros([K, pad], b.dtype)], -1)
    n = (T + pad) // C
    LA, BB = tf.reshape(la, [K, n, C]), tf.reshape(b, [K, n, C])
    intra = _contract(tf.exp(seg(LA)), BB, mode, tf32)          # [K, n, C]
    cs = tf.cumsum(LA, axis=-1)                                  # local inclusive, <= C terms
    H = _level(cs[..., -1], intra[..., -1], h_in, G, G, seg, mode, tf32)   # state at every chunk end
    H_prev = tf.concat([h_in[:, None], H[:, :-1]], -1)          # state entering every chunk
    h = intra + tf.exp(cs) * H_prev[..., None]
    return tf.reshape(h, [K, n * C])[:, :T]


def linrec(la, b, h0=None, C=64, G=32, seg="segsum", contract="mulsum", tf32=None, check_finite=False):
    """Returns (h [K, T], h_last [K]). la, b: [K, T] (float32 or float64), T static."""
    la = tf.convert_to_tensor(la)
    b = tf.convert_to_tensor(b, la.dtype)
    if check_finite:                             # refuse non-finite input (NEG resets are finite)
        la = tf.debugging.check_numerics(la, "linrec: non-finite log decay")
        b = tf.debugging.check_numerics(b, "linrec: non-finite input")
    K = la.shape[0]
    h_in = tf.zeros([K], la.dtype) if h0 is None else tf.cast(tf.convert_to_tensor(h0), la.dtype)
    segf = segsum if seg == "segsum" else segsub
    h = _level(la, b, h_in, int(C), int(G), segf, contract, tf32)
    return h, h[:, -1]


# ------------------------------------------------------------------------------------ references
def ref_linrec(la, b, h0=None):
    """float64 sequential recursion (numpy)."""
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

"""
Mathematical helper functions (moved from the root ``math_helpers.py`` in Phase B5).

The loss copies that used to live here (huber_loss, focal_loss, pinball_loss) were
unused duplicates of neural_trade.losses and have been removed.

Mathematical helper functions for neural_trade model.

This module contains pure mathematical functions extracted from model.py
for better modularity and testability. All functions work with TensorFlow tensors.
"""

import numpy as np
import tensorflow as tf


# -------------------------
# Alpha/Logit/Period Transformations
# -------------------------

def logit_from_alpha(alpha, epsilon=1e-8):
    """Convert alpha (smoothing parameter) to logit space.

    Args:
        alpha: Smoothing parameter in [0, 1]
        epsilon: Small value to prevent log(0)

    Returns:
        Logit representation of alpha
    """
    return tf.math.log(alpha + epsilon) - tf.math.log(1.0 - alpha + epsilon)


def alpha_from_logit(logit):
    """Convert logit to alpha (smoothing parameter).

    Args:
        logit: Logit value

    Returns:
        Alpha in [0, 1]
    """
    return tf.sigmoid(logit)


def logit_from_period(period, epsilon=1e-8):
    """Convert period (window size) to logit space.

    Args:
        period: Period/window size
        epsilon: Small value to prevent log(0)

    Returns:
        Logit representation
    """
    alpha = 2.0 / (period + 1.0)
    return logit_from_alpha(alpha, epsilon)


def period_from_logit(logit, epsilon=1e-8):
    """Convert logit to period (window size).

    Args:
        logit: Logit value
        epsilon: Small value for numerical stability

    Returns:
        Period/window size (>= 0)
    """
    alpha = alpha_from_logit(logit)
    period = (2.0 / (alpha + epsilon)) - 1.0
    return tf.maximum(period, 0.0)









# -------------------------
# Statistical Functions
# -------------------------

def normal_cdf(z):
    """Standard Normal CDF using error function.

    Computes P(Z <= z) where Z ~ N(0, 1)

    Args:
        z: Input value(s) (any shape)

    Returns:
        CDF values with same shape as input
    """
    z = tf.cast(z, tf.float32)
    return 0.5 * (1.0 + tf.math.erf(z / tf.constant(np.sqrt(2.0), dtype=tf.float32)))


# -------------------------
# Scaling Functions
# -------------------------

def to_scaled(raw, mean, scale, eps=1e-8):
    """Convert raw values to scaled units using standardization.

    Args:
        raw: Raw values
        mean: Mean for standardization
        scale: Standard deviation for standardization
        eps: Small value to prevent division by zero

    Returns:
        Scaled values: (raw - mean) / scale
    """
    raw = tf.cast(raw, tf.float32)
    mean = tf.cast(mean, tf.float32)
    scale = tf.cast(scale, tf.float32)
    return (raw - mean) / (scale + eps)


def to_unscaled(scaled, mean, scale):
    """Convert scaled values back to raw units.

    Args:
        scaled: Scaled values
        mean: Mean used for standardization
        scale: Standard deviation used for standardization

    Returns:
        Raw values: scaled * scale + mean
    """
    scaled = tf.cast(scaled, tf.float32)
    mean = tf.cast(mean, tf.float32)
    scale = tf.cast(scale, tf.float32)
    return scaled * scale + mean


# -------------------------
# Time Series Functions
# -------------------------

def ewma_sequence(x_seq, alpha_scalar):
    """Compute Exponentially Weighted Moving Average (EWMA) sequence.

    Applies EWMA with smoothing parameter alpha across time dimension.
    EWMA[t] = alpha * x[t] + (1 - alpha) * EWMA[t-1]

    Args:
        x_seq: Input sequence [batch, time]
        alpha_scalar: Smoothing parameter in [0, 1]

    Returns:
        EWMA sequence with same shape as input
    """
    x_seq = tf.cast(x_seq, tf.float32)
    alpha_scalar = tf.cast(alpha_scalar, tf.float32)

    def step(prev, cur):
        return alpha_scalar * cur + (1.0 - alpha_scalar) * prev

    first = x_seq[:, 0]
    rest = x_seq[:, 1:]

    ema_rest = tf.scan(
        fn=lambda prev, cur: step(prev, cur),
        elems=tf.transpose(rest, perm=[1, 0]),
        initializer=first,
        parallel_iterations=1
    )

    ema_rest = tf.transpose(ema_rest, perm=[1, 0])
    ema_full = tf.concat([tf.expand_dims(first, axis=1), ema_rest], axis=1)
    return ema_full


_EWMA_ALPHA_CLAMP = 1e-6


def ewma_sequence_matrix(x_seq, alpha):
    """EWMA as one batched matrix product - the same recurrence as ``ewma_sequence``.

    ``ewma_sequence`` runs ``tf.scan(parallel_iterations=1)``: LOOKBACK-1 strictly sequential
    steps, and LearnableIndicators calls it 24 times per forward pass (~1,400 sequential GPU
    steps per batch, forward and backward). Unrolling the recurrence
    ``ema[t] = a*x[t] + (1-a)*ema[t-1]``, ``ema[0] = x[0]`` gives

        ema[t] = (1-a)^t * x[0] + sum_{k=1..t} a*(1-a)^(t-k) * x[k]

    i.e. ``ema = M @ x`` with ``M[b,t,k] = a_b (1-a_b)^(t-k)`` for 1 <= k <= t,
    ``M[b,t,0] = (1-a_b)^t`` and 0 above the diagonal. The powers are built as
    ``exp((t-k) * log1p(-a))``, so a per-sample alpha (shape [B]) or a scalar is supported and
    the gradient w.r.t. alpha flows exactly as through the scan.

    Alpha is clamped to [1e-6, 1-1e-6] because float32 ``sigmoid`` rounds to exactly 1.0 for
    logits above ~17 and ``log1p(-1) = -inf`` would give ``0 * inf = NaN`` on the diagonal.
    At the clamp the result differs from the scan by at most 1e-6 * |x[t] - x[t-1]|.

    Cost is O(B*T^2) memory (64 x 60 x 60 floats per call), trivially parallel.
    Equivalence with the scan is pinned by tests/test_learnable_indicators.py.
    """
    x = tf.cast(x_seq, tf.float32)                                   # [B, T]
    a = tf.cast(alpha, tf.float32)
    a = a + tf.zeros_like(x[:, 0])                                   # scalar or [B] -> [B]
    a = tf.clip_by_value(a, _EWMA_ALPHA_CLAMP, 1.0 - _EWMA_ALPHA_CLAMP)

    t = tf.range(tf.shape(x)[1])
    lag = t[:, None] - t[None, :]                                    # [T, T], t - k
    lower = lag >= 0
    lag_f = tf.cast(tf.maximum(lag, 0), tf.float32)

    decay = tf.exp(lag_f[None, :, :] * tf.math.log1p(-a)[:, None, None])   # (1-a)^(t-k)
    weights = decay * a[:, None, None]                                      # a (1-a)^(t-k)
    first_col = tf.equal(t, 0)[None, None, :]                               # k == 0
    weights = tf.where(first_col, decay, weights)
    weights = tf.where(lower[None, :, :], weights, tf.zeros_like(weights))
    return tf.einsum('btk,bk->bt', weights, x)


# -------------------------
# Utility Functions
# -------------------------

def safe_reduce_mean(x):
    """Compute mean with safe float32 casting.

    Args:
        x: Input tensor

    Returns:
        Mean value
    """
    return tf.reduce_mean(tf.cast(x, tf.float32))


def safe_clip_finite(x, fallback=0.0):
    """Replace non-finite values (NaN, Inf) with fallback.

    Args:
        x: Input tensor
        fallback: Value to use for non-finite elements

    Returns:
        Tensor with finite values only
    """
    x = tf.cast(x, tf.float32)
    fallback = tf.constant(fallback, dtype=tf.float32)
    return tf.where(tf.math.is_finite(x), x, fallback)


_LOG_NDTR_SPLIT = -8.0
_HALF_LOG_2PI = 0.9189385332046727


def log_ndtr(x):
    """log(Phi(x)) for float32 that stays finite deep in the lower tail.

    TensorFlow 2.10 has no ``log_ndtr``. ``log(0.5 * erfc(-x / sqrt2))`` is accurate down to
    about x = -8 and then underflows to log(0) = -inf, which matters here: with the variance
    head at VAR_FLOOR both tail probabilities of the direction readout are ~1e-98. Below the
    split the Mills-ratio asymptotic series is used (truncation error < 1e-6 at x = -8).
    Each branch only ever sees inputs clamped into its own domain, so neither the discarded
    branch nor its gradient can produce NaN (the usual ``tf.where`` gradient trap).
    """
    x = tf.cast(x, tf.float32)
    split = tf.constant(_LOG_NDTR_SPLIT, dtype=tf.float32)
    x_hi = tf.maximum(x, split)
    upper = tf.math.log(0.5 * tf.math.erfc(-x_hi * tf.constant(0.7071067811865476, dtype=tf.float32)))
    x_lo = tf.minimum(x, split)
    inv2 = 1.0 / (x_lo * x_lo)
    series = 1.0 - inv2 + 3.0 * inv2 ** 2 - 15.0 * inv2 ** 3 + 105.0 * inv2 ** 4
    lower = (-0.5 * x_lo * x_lo - tf.math.log(-x_lo)
             - tf.constant(_HALF_LOG_2PI, dtype=tf.float32) + tf.math.log(series))
    return tf.where(x > split, upper, lower)

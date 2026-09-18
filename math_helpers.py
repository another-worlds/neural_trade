"""
Mathematical helper functions for neural_trade model.

This module contains pure mathematical functions extracted from model.py
for better modularity and testability. All functions work with TensorFlow tensors.
"""

import math
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
# Loss Functions
# -------------------------

def huber_loss(x, delta=1.0):
    """Element-wise Huber loss (returns same-shape tensor).

    Huber loss is quadratic for small errors and linear for large errors,
    providing robustness to outliers.

    Args:
        x: Error tensor (any shape)
        delta: Threshold for quadratic vs linear behavior

    Returns:
        Huber loss with same shape as input
    """
    delta = tf.cast(delta, tf.float32)
    x = tf.cast(x, tf.float32)
    abs_x = tf.abs(x)
    quadratic = 0.5 * tf.square(x)
    linear = delta * (abs_x - 0.5 * delta)
    return tf.where(abs_x <= delta, quadratic, linear)


def focal_loss(true_labels, logits, alpha=0.7, gamma=2.0, reduce=True):
    """Focal Loss for imbalanced binary classification.

    Focal Loss: -α(1-p_t)^γ * log(p_t)
    Addresses class imbalance by down-weighting easy examples and focusing on hard ones.

    Args:
        true_labels: Binary labels [B] (0 or 1)
        logits: Predicted probabilities [B] from sigmoid (0 to 1)
        alpha: Class weight for minority class (default 0.7 weights DOWN class)
        gamma: Focusing parameter (default 2.0; higher = more focus on hard examples)
        reduce: If True, return scalar mean; if False, return per-example vector

    Returns:
        If reduce=True: scalar mean focal loss
        If reduce=False: per-example focal loss vector
    """
    alpha = tf.cast(alpha, tf.float32)
    gamma = tf.cast(gamma, tf.float32)

    # Ensure inputs are float32
    true_labels = tf.cast(true_labels, tf.float32)
    logits = tf.cast(logits, tf.float32)

    # Clamp logits to prevent log(0)
    logits = tf.clip_by_value(logits, 1e-7, 1.0 - 1e-7)

    # Compute focal weight: (1 - p_t)^γ where p_t is the probability of true class
    p_t = true_labels * logits + (1.0 - true_labels) * (1.0 - logits)
    focal_weight = tf.pow(1.0 - p_t, gamma)

    # Compute base cross-entropy
    bce = -true_labels * tf.math.log(logits) - (1.0 - true_labels) * tf.math.log(1.0 - logits)

    # Apply focal weighting and class weighting
    # alpha is the weight for the DOWN class (label=0), and (1-alpha) weights UP (label=1)
    class_weight = alpha * (1.0 - true_labels) + (1.0 - alpha) * true_labels

    focal = class_weight * focal_weight * bce

    if reduce:
        return tf.reduce_mean(focal)
    return focal


def pinball_loss(y_true, y_pred, tau=0.5):
    """Pinball (quantile) loss for asymmetric prediction.

    Args:
        y_true: True values
        y_pred: Predicted values
        tau: Quantile parameter in [0, 1] (default 0.5 for median)

    Returns:
        Mean pinball loss
    """
    tau = tf.constant(tau, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    error = y_true - y_pred
    per_elem = tf.maximum(tau * error, (tau - 1.0) * error)

    result = tf.reduce_mean(per_elem)
    # Guard against NaN/Inf contamination
    result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
    return result


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

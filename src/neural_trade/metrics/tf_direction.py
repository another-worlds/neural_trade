"""TensorFlow side of the direction metrics: labels, Gaussian readout (graph-safe).

NumPy twins live in :mod:`neural_trade.metrics.direction_labels`; both implement the same
rule and are cross-checked by tests/test_direction_labels.py.
"""
from __future__ import annotations

import tensorflow as tf

from neural_trade.utils.math import log_ndtr


def direction_labels_tf(y_true_raw, last_close_squeeze, deadband_bps, eps=1e-8):
    """TF-graph version of direction labeling with deadband (single source of truth).

    The single TF labelling rule (custom_loss, train_step, test_step). |return| <= deadband
    is neutral and masked out; "up" means return > deadband.
    Returns (mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2).
    NumPy twin: neural_trade.metrics.direction_labels.direction_labels_np.
    """
    deadband = tf.cast(deadband_bps, tf.float32) / tf.constant(10000.0, dtype=tf.float32)
    ret_h0 = y_true_raw[:, 0] / (last_close_squeeze + eps)
    ret_h1 = y_true_raw[:, 1] / (last_close_squeeze + eps)
    ret_h2 = y_true_raw[:, 2] / (last_close_squeeze + eps)
    mask_h0 = tf.cast(tf.abs(ret_h0) > deadband, tf.float32)
    mask_h1 = tf.cast(tf.abs(ret_h1) > deadband, tf.float32)
    mask_h2 = tf.cast(tf.abs(ret_h2) > deadband, tf.float32)
    true_dir_h0 = tf.cast(ret_h0 > deadband, tf.float32)
    true_dir_h1 = tf.cast(ret_h1 > deadband, tf.float32)
    true_dir_h2 = tf.cast(ret_h2 > deadband, tf.float32)
    return mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2


def gaussian_up_prob_given_move(mu_scaled, var_scaled, last_close, deadband_frac,
                                pred_mean, pred_scale, eps=1e-8):
    """P(delta > d | |delta| > d) for delta ~ N(mu, sigma^2): the Gaussian direction readout.

    Direction labels mask out moves inside the deadband (|return| <= d) and call the rest
    "up" when return > d (see ``_compute_direction_labels_and_masks_tf``). The readout
    that matches those labels is therefore the probability of "up" GIVEN that the move
    left the deadband:

        p = Phi(a) / (Phi(a) + Phi(b)),  a = (mu - d) / sigma,  b = (-mu - d) / sigma

    evaluated as sigmoid(log Phi(a) - log Phi(b)) so it stays finite when sigma << d.

    The previous readout was Phi((mu - d) / sigma) alone - the UNconditional probability of
    an up move, which also counts the neutral mass as "not up". With the 5 bps deadband
    (~0.21 scaled units at BTC 110k) every sample sat at Phi(-0.21) ~ 0.42 < 0.5, nothing
    was ever predicted up and the Gaussian MCC was identically zero by construction. It also
    converted the threshold to scaled units without the pred_mean offset. This version works
    in raw dollars, where the labels are defined, and reduces to Phi(mu / sigma) when d = 0.

    Args:
        mu_scaled, var_scaled: price head and variance head in scaled-delta units, [B].
        last_close: raw last close, [B].
        deadband_frac: DIR_DEADBAND_BPS / 1e4.
        pred_mean, pred_scale: the target scaler statistics.
    """
    mu_raw = tf.cast(mu_scaled, tf.float32) * pred_scale + pred_mean
    sigma_raw = tf.sqrt(tf.maximum(tf.cast(var_scaled, tf.float32), 0.0)) * pred_scale + eps
    d_raw = tf.cast(deadband_frac, tf.float32) * tf.cast(last_close, tf.float32)
    a = (mu_raw - d_raw) / sigma_raw
    b = (-mu_raw - d_raw) / sigma_raw
    return tf.sigmoid(log_ndtr(a) - log_ndtr(b))

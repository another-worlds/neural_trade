"""TensorFlow side of the direction metrics: labels, Gaussian readout (graph-safe).

NumPy twins live in :mod:`neural_trade.metrics.direction_labels`; both implement the same
rule and are cross-checked by tests/test_direction_labels.py.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Callable, Dict, Optional

import numpy as np
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


# ---- Epoch-level aggregation of the step metrics (moved from model.py in B8) ----------
# Keras keeps only the dict returned by the LAST train/test step of an epoch
# (`logs = tmp_logs` in Model.fit/evaluate). The steps used to return per-batch tensors, so
# every logged train_*/val_* number - including the val_loss that drives EarlyStopping,
# ModelCheckpoint and ReduceLROnPlateau - described one batch (~50 of 2,866 validation
# samples). The steps now update epoch accumulators and return their running totals, so the
# last dict IS the epoch aggregate. Keras resets them via CustomTrainModel.metrics.
DIR_N_BINS = 10
PIT_N_BINS = 200
STEP_MEAN_KEYS = (
    'loss', 'point_loss', 'point_h0', 'point_h1', 'point_h2',
    'trend_loss', 'trend_h0', 'trend_h1', 'trend_h2',
    'local_h0', 'global_h0', 'extended_h0', 'local_h1', 'global_h1', 'extended_h1',
    'local_h2', 'global_h2', 'extended_h2',
    'dir_loss', 'dir_loss_h0', 'dir_loss_h1', 'dir_loss_h2',
    'nll_loss', 'nll_h0', 'nll_h1', 'nll_h2',
    'crps_loss', 'crps_h0', 'crps_h1', 'crps_h2',
    'soft_ece_loss', 'soft_ece_h0', 'soft_ece_h1', 'soft_ece_h2',
    'reg_loss', 'inter_reg', 'vol_loss',
    't_perp_loss', 'casimir_loss', 'vac_loss', 'hd_loss', 'ife_loss', 'vac_overflow_loss',
    'grad_global_norm',
)
TRAIN_ONLY_MEAN_KEYS = ('grad_global_norm',)


def direction_counts(true_dir, dir_pred, mask):
    """Sufficient statistics of one horizon's direction predictions.

    Returns ([TP, TN, FP, FN, brier_sum, prob_sum, mask_sum], bin_n, bin_true, bin_prob) where
    the bins partition [0, 1] into DIR_N_BINS (last bin inclusive) for the ECE.
    """
    t = tf.cast(tf.reshape(true_dir, [-1]), tf.float32)
    p = tf.cast(tf.reshape(dir_pred, [-1]), tf.float32)
    m = tf.cast(tf.reshape(mask, [-1]), tf.float32)
    pb = tf.cast(p > 0.5, tf.float32)
    pc = tf.clip_by_value(p, 0.0, 1.0)
    counts = tf.stack([
        tf.reduce_sum(pb * t * m),
        tf.reduce_sum((1.0 - pb) * (1.0 - t) * m),
        tf.reduce_sum(pb * (1.0 - t) * m),
        tf.reduce_sum((1.0 - pb) * t * m),
        tf.reduce_sum(tf.square(p - t) * m),
        tf.reduce_sum(pc * m),
        tf.reduce_sum(m),
    ])
    idx = tf.clip_by_value(tf.cast(tf.floor(pc * DIR_N_BINS), tf.int32), 0, DIR_N_BINS - 1)
    onehot = tf.one_hot(idx, DIR_N_BINS, dtype=tf.float32) * m[:, None]
    return (counts, tf.reduce_sum(onehot, axis=0), tf.reduce_sum(onehot * t[:, None], axis=0),
            tf.reduce_sum(onehot * pc[:, None], axis=0))


DirectionStats = namedtuple(
    "DirectionStats",
    ["TP", "TN", "FP", "FN", "brier_sum", "prob_sum", "mask_sum", "bin_n", "bin_true", "bin_prob"],
)


def direction_stats(counts, bin_n, bin_true, bin_prob) -> DirectionStats:
    return DirectionStats(counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6],
                          bin_n, bin_true, bin_prob)


# ---- the 11 step metrics (Metrics registry, TF tier): f(stats) -> scalar tensor ----------
# Graph-safe (no .numpy()); computed from epoch-accumulated sufficient statistics.
def tf_dir_acc(stats):
    """Accuracy of P(up) > 0.5 on non-neutral samples."""
    s = stats
    return (s.TP + s.TN) / (s.TP + s.TN + s.FP + s.FN + 1e-8)


def tf_dir_sensitivity(stats):
    """Recall of the UP class."""
    return stats.TP / (stats.TP + stats.FN + 1e-8)


def tf_dir_specificity(stats):
    """Recall of the DOWN class."""
    return stats.TN / (stats.TN + stats.FP + 1e-8)


def tf_dir_bal_acc(stats):
    """Mean of sensitivity and specificity."""
    return (tf_dir_sensitivity(stats) + tf_dir_specificity(stats)) / 2.0


def tf_dir_f1(stats):
    """F1 of the UP class."""
    precision = stats.TP / (stats.TP + stats.FP + 1e-8)
    recall = stats.TP / (stats.TP + stats.FN + 1e-8)
    return 2.0 * (precision * recall) / (precision + recall + 1e-8)


def tf_dir_mcc(stats):
    """Matthews correlation coefficient (0 when a margin is empty)."""
    s = stats
    num = (s.TP * s.TN) - (s.FP * s.FN)
    marginal = (s.TP + s.FP) * (s.TP + s.FN) * (s.TN + s.FP) * (s.TN + s.FN)
    return tf.where(marginal > 1e-8, num / tf.sqrt(marginal + 1e-8), tf.constant(0.0, tf.float32))


def tf_dir_brier(stats):
    """Brier score of P(up)."""
    return stats.brier_sum / (stats.mask_sum + 1e-8)


def tf_dir_ece(stats):
    """Positive-class ECE: |observed up-rate - mean P(up)| per bin, weighted by bin mass."""
    bin_acc = stats.bin_true / (stats.bin_n + 1e-8)
    bin_conf = stats.bin_prob / (stats.bin_n + 1e-8)
    return tf.reduce_sum((stats.bin_n / (stats.mask_sum + 1e-8)) * tf.abs(bin_acc - bin_conf))


def tf_pred_up_rate(stats):
    """Share of non-neutral samples predicted UP."""
    s = stats
    return (s.TP + s.FP) / (s.TP + s.TN + s.FP + s.FN + 1e-8)


def tf_true_up_rate(stats):
    """Share of non-neutral samples that were UP."""
    s = stats
    return (s.TP + s.FN) / (s.TP + s.TN + s.FP + s.FN + 1e-8)


def tf_mean_dir_prob(stats):
    """Mean P(up) over non-neutral samples."""
    return stats.prob_sum / (stats.mask_sum + 1e-8)


STEP_METRIC_FUNCTIONS: Dict[str, Callable] = {
    "dir_acc": tf_dir_acc, "dir_sensitivity": tf_dir_sensitivity, "dir_specificity": tf_dir_specificity,
    "dir_bal_acc": tf_dir_bal_acc, "dir_f1": tf_dir_f1, "dir_mcc": tf_dir_mcc, "dir_brier": tf_dir_brier,
    "dir_ece": tf_dir_ece, "pred_up_rate": tf_pred_up_rate, "true_up_rate": tf_true_up_rate,
    "mean_dir_prob": tf_mean_dir_prob,
}


def direction_metrics_from_stats(stats: DirectionStats, prefix: str, h_name: str,
                                  fns: Optional[Dict[str, Callable]] = None) -> Dict[str, tf.Tensor]:
    """``{prefix}{name}_{h}`` for each step metric; NaN when the horizon had no samples."""
    fns = fns or STEP_METRIC_FUNCTIONS
    no_samples = stats.mask_sum < 1e-8
    nan = tf.constant(np.nan, dtype=tf.float32)
    return {f"{prefix}{name}_{h_name}": tf.where(no_samples, nan, fn(stats)) for name, fn in fns.items()}


class DirectionAccumulator(tf.keras.metrics.Metric):
    """Epoch accumulator of direction statistics for the three horizons."""

    def __init__(self, name='direction_accumulator', **kwargs):
        super().__init__(name=name, **kwargs)
        self.counts = self.add_weight(name='counts', shape=(3, 7), initializer='zeros')
        self.bin_n = self.add_weight(name='bin_n', shape=(3, DIR_N_BINS), initializer='zeros')
        self.bin_true = self.add_weight(name='bin_true', shape=(3, DIR_N_BINS), initializer='zeros')
        self.bin_prob = self.add_weight(name='bin_prob', shape=(3, DIR_N_BINS), initializer='zeros')

    def update_state(self, true_dirs, dir_preds, masks, sample_weight=None):
        stats = [direction_counts(t, p, m) for t, p, m in zip(true_dirs, dir_preds, masks)]
        self.counts.assign_add(tf.stack([s[0] for s in stats]))
        self.bin_n.assign_add(tf.stack([s[1] for s in stats]))
        self.bin_true.assign_add(tf.stack([s[2] for s in stats]))
        self.bin_prob.assign_add(tf.stack([s[3] for s in stats]))

    def result(self):
        return self.counts

    def reset_state(self):
        for v in (self.counts, self.bin_n, self.bin_true, self.bin_prob):
            v.assign(tf.zeros_like(v))

    def logs(self, prefix, fns=None):
        out = {}
        for i, h in enumerate(("h0", "h1", "h2")):
            stats = direction_stats(self.counts[i], self.bin_n[i], self.bin_true[i], self.bin_prob[i])
            out.update(direction_metrics_from_stats(stats, prefix, h, fns))
        return out


class PITAccumulator(tf.keras.metrics.Metric):
    """Epoch histogram of PIT values Phi((y - mu) / sigma) per horizon; KS from the binned ECDF.

    KS is not decomposable over batches (the mean of per-batch KS on ~64 samples is dominated
    by sampling noise, ~0.1 even for a perfectly calibrated model), so the PIT values are
    binned into PIT_N_BINS and the KS distance is taken at the bin edges of the pooled
    epoch ECDF (resolution 1 / PIT_N_BINS).
    """

    def __init__(self, var_floor=1e-4, var_cap=1e3, name='pit_accumulator', **kwargs):
        super().__init__(name=name, **kwargs)
        self.var_floor, self.var_cap = float(var_floor), float(var_cap)
        self.hist = self.add_weight(name='hist', shape=(3, PIT_N_BINS), initializer='zeros')

    def update_state(self, ys, mus, variances, sample_weight=None):
        rows = []
        for y, mu, var in zip(ys, mus, variances):
            y = tf.cast(tf.reshape(y, [-1]), tf.float32)
            mu = tf.cast(tf.reshape(mu, [-1]), tf.float32)
            var = tf.clip_by_value(tf.cast(tf.reshape(var, [-1]), tf.float32), self.var_floor, self.var_cap)
            u = 0.5 * (1.0 + tf.math.erf(((y - mu) / (tf.sqrt(var) + 1e-8)) / np.sqrt(2.0).astype(np.float32)))
            idx = tf.clip_by_value(tf.cast(tf.floor(u * PIT_N_BINS), tf.int32), 0, PIT_N_BINS - 1)
            rows.append(tf.reduce_sum(tf.one_hot(idx, PIT_N_BINS, dtype=tf.float32), axis=0))
        self.hist.assign_add(tf.stack(rows))

    def result(self):
        return self.hist

    def reset_state(self):
        self.hist.assign(tf.zeros_like(self.hist))

    def logs(self):
        n = tf.reduce_sum(self.hist, axis=1, keepdims=True)                       # [3, 1]
        ecdf = tf.cumsum(self.hist, axis=1) / (n + 1e-8)                          # at right edges
        edges = tf.range(1, PIT_N_BINS + 1, dtype=tf.float32) / PIT_N_BINS
        ks = tf.reduce_max(tf.abs(ecdf - edges[None, :]), axis=1)
        ks = tf.where(tf.squeeze(n, 1) > 0, ks, tf.constant(np.nan, tf.float32))
        return {"pit_ks_h0": ks[0], "pit_ks_h1": ks[1], "pit_ks_h2": ks[2]}

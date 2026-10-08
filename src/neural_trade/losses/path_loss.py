"""Path head losses (tactical hypothesis E, ``Config.PATH_HEAD``): technical indicators at the END, as a
regulariser. The network predicts the next P closes as scaled deltas from the last close; the same fixed
indicators are computed on the predicted and on the real future path and their mismatch is a loss, so
"trend" is defined the way technical analysis defines it.

Features, in this order (``PATH_FEATURE_NAMES``), on the path ``s = [0, p_1 .. p_P]`` (the last close is
the origin, so the features are shift-free):

* ``ema_slope_<n>`` for every ``Config.PATH_IND_EMA`` period n: the least-squares slope of the EMA of s
  (alpha = 2 / (n + 1), seeded with s_0) over the path, in scaled units per bar.
* ``rsi_<n>``: a smooth RSI (n = ``Config.PATH_IND_RSI``), in [0, 1]: Wilder smoothing (alpha = 1 / n)
  of the smooth positive part ``u = (d + sqrt(d^2 + e^2)) / 2`` and negative part ``u - d`` of the bar
  differences ``d``.
* ``efficiency``: net move / path length, ``s_P / sum sqrt(d^2 + e^2)``, in [-1, 1].

Why the indicator parameters are FIXED (not learnable like the input-side indicators): a learnable
period or threshold lives on both paths, so the optimiser could drive it to a value where the feature is
(almost) a constant of both and the loss is trivially 0 without the path being right. Fixed parameters
make the target the same function of the real path from the first step to the last.

Each feature's squared error is divided by the batch variance of its REAL value (stop-gradient), so no
feature dominates and the loss is in "real-path standard deviations": a constant predicted path (no
trend, RSI 0.5, efficiency 0) is not rewarded when the real paths trend. The EMA slope and the
efficiency are linear / smooth in the path, the smoothing ``e`` keeps the RSI and efficiency
differentiable at flat steps. Everything here is static-shaped TF ops on small matrices: no loops over
the batch, cost O(B * P).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf

#: smoothing of |d| / max(d, 0) in scaled units (one unit = the std of a horizon delta; a bar step is
#: typically 0.2-0.3 of it). Fixed, like every other indicator parameter here.
SMOOTH_EPS = 0.05
#: floor of the batch std of a real feature used as a divisor (degenerate batches only).
STD_FLOOR = 1e-3


def path_feature_names(ema_periods: Sequence[int], rsi_period: int) -> List[str]:
    return [f"ema_slope_{int(n)}" for n in ema_periods] + [f"rsi_{int(rsi_period)}", "efficiency"]


def _ema_weights(n_points: int, period: int, alpha: float) -> np.ndarray:
    """[n_points, n_points] E with ``ema = s @ E.T`` for ``ema_0 = s_0; ema_t = (1-a) ema_{t-1} + a s_t``."""
    e = np.zeros((n_points, n_points), dtype="float64")
    e[0, 0] = 1.0
    for t in range(1, n_points):
        e[t] = (1.0 - alpha) * e[t - 1]
        e[t, t] += alpha
    return e


def _ema_slope_vector(n_points: int, period: int) -> np.ndarray:
    """[n_points] c with ``slope = s @ c``: least-squares slope over t = 0..P of the period-n EMA of s."""
    e = _ema_weights(n_points, period, 2.0 / (float(period) + 1.0))
    t = np.arange(n_points, dtype="float64")
    centred = t - t.mean()
    return (centred / np.sum(centred ** 2)) @ e


def _wilder_weights(n_steps: int, period: int) -> np.ndarray:
    """[n_steps] w with ``rma_last = x @ w`` for ``r_1 = x_1; r_t = (1-a) r_{t-1} + a x_t``, a = 1/period."""
    a = 1.0 / float(period)
    r = np.zeros((n_steps, n_steps), dtype="float64")
    r[0, 0] = 1.0
    for t in range(1, n_steps):
        r[t] = (1.0 - a) * r[t - 1]
        r[t, t] += a
    return r[-1]


def path_features(path, ema_periods: Sequence[int] = (5, 10), rsi_period: int = 10):
    """``[B, F]`` shape features of ``path`` ``[B, P]`` (scaled deltas from the last close), F =
    ``len(ema_periods) + 2`` in the order of :func:`path_feature_names`."""
    path = tf.cast(path, tf.float32)
    p = int(path.shape[-1])
    if p < 2:
        raise ValueError(f"the path indicators need a path of at least 2 bars, got {p}")
    s = tf.concat([tf.zeros_like(path[:, :1]), path], axis=1)         # [B, P+1], origin = the last close
    feats = []
    for n in ema_periods:
        c = tf.constant(_ema_slope_vector(p + 1, int(n)), dtype=tf.float32)
        feats.append(tf.linalg.matvec(s, c))
    d = s[:, 1:] - s[:, :-1]                                           # [B, P]
    eps = tf.constant(SMOOTH_EPS, tf.float32)
    root = tf.sqrt(tf.square(d) + tf.square(eps))
    up = 0.5 * (d + root)
    down = up - d
    w = tf.constant(_wilder_weights(p, int(rsi_period)), dtype=tf.float32)
    avg_up, avg_down = tf.linalg.matvec(up, w), tf.linalg.matvec(down, w)
    feats.append(avg_up / (avg_up + avg_down + 1e-6))
    feats.append(s[:, -1] / (tf.reduce_sum(root, axis=1) + 1e-6))
    return tf.stack(feats, axis=1)


def path_indicator_loss(pred_path, real_path, ema_periods: Sequence[int] = (5, 10), rsi_period: int = 10):
    """Mean over features of the batch-mean squared feature mismatch, each in units of the real
    feature's batch std. Returns ``(loss, per_feature [F])``."""
    f_pred = path_features(pred_path, ema_periods, rsi_period)
    f_real = path_features(real_path, ema_periods, rsi_period)
    std = tf.stop_gradient(tf.maximum(tf.math.reduce_std(f_real, axis=0), STD_FLOOR))
    per_feature = tf.reduce_mean(tf.square((f_pred - f_real) / std), axis=0)
    return tf.reduce_mean(per_feature), per_feature


def path_logcosh_loss(pred_path, real_path):
    """Mean log-cosh of the path error (scaled units): quadratic near 0, linear in the tails."""
    x = tf.cast(pred_path, tf.float32) - tf.cast(real_path, tf.float32)
    return tf.reduce_mean(x + tf.math.softplus(-2.0 * x) - tf.math.log(2.0))


def path_loss_terms(pred_path, real_path, config) -> Tuple[tf.Tensor, tf.Tensor]:
    """``(path_loss, path_ind_loss)``: the log-cosh and the indicator shape loss, UNWEIGHTED. A term
    whose weight is 0 is not computed (a Python branch on the Config value: a constant 0)."""
    zero = tf.constant(0.0, tf.float32)
    path_term = path_logcosh_loss(pred_path, real_path) if float(config.LAMBDA_PATH) > 0 else zero
    ind_term = zero
    if float(config.LAMBDA_PATH_IND) > 0:
        ind_term, _ = path_indicator_loss(pred_path, real_path, config.PATH_IND_EMA, int(config.PATH_IND_RSI))
    return path_term, ind_term


__all__ = ["path_feature_names", "path_features", "path_indicator_loss", "path_logcosh_loss",
           "path_loss_terms", "SMOOTH_EPS", "STD_FLOOR"]

"""Centralized Loss registry.

This module provides a loss function registry using the BaseRegistry pattern.
Decorating any function with `@Losses.register()` will add it to the registry
under the function name (or a provided name). This allows discovery and
programmatic lookup of losses without changing the implementation sites.

The registry intentionally does not change the behavior of the decorated
function; it only records it for programmatic use. Helpers are provided
to normalize common return shapes for logging (e.g., scalar, dict, tuple).
"""
from __future__ import annotations
from typing import Callable, Dict, Any, Tuple
import tensorflow as tf

from core.registry import BaseRegistry


class Losses(BaseRegistry):
    """Registry for loss functions.

    Usage:
        @Losses.register()
        def my_loss(...):
            return scalar_or_dict_or_tuple

        loss_fn = Losses.get('my_loss')
        loss_fn(...)  # returns the loss value

    Or with metadata:
        @Losses.register(name="focal_loss", tags=["classification", "imbalanced"])
        def focal_loss(...):
            return loss_value
    """

    registry = {}

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        """Validate that component is a callable loss function."""
        return callable(component)

    @classmethod
    def normalize_return(cls, out) -> Tuple[Any, Dict[str, Any]]:
        """Normalize common loss return patterns into (total, components).

        - If `out` is a dict: will try to find a 'loss' key as total, or sum
          keys that end with '_loss' as a fallback.
        - If `out` is a scalar/tensor: return (out, {'loss': out})
        - If `out` is a tuple and the second element is a dict: return it as-is
        - Otherwise, return (first_element, {}) for tuples.
        """
        # Tensor or numeric scalar
        if isinstance(out, (tf.Tensor, float, int)):
            return out, {'loss': out}

        # Dict: ensure a total key 'loss' exists
        if isinstance(out, dict):
            comps = dict(out)
            if 'loss' not in comps:
                # Try summing keys that look like losses
                loss_keys = [k for k in comps.keys() if k.endswith('_loss') or k == 'loss']
                if loss_keys:
                    total = None
                    for k in loss_keys:
                        v = comps[k]
                        v_mean = tf.reduce_mean(v) if isinstance(v, tf.Tensor) else v
                        total = (v_mean if total is None else total + v_mean)
                    comps['loss'] = total
                else:
                    # Pick the first numeric-like entry as a fallback
                    for k, v in comps.items():
                        if isinstance(v, (float, int)) or isinstance(v, tf.Tensor):
                            comps['loss'] = tf.reduce_mean(v) if isinstance(v, tf.Tensor) else v
                            break
                    comps.setdefault('loss', 0.0)
            return comps['loss'], comps

        # Tuple-like
        if isinstance(out, tuple):
            if len(out) == 2 and isinstance(out[1], dict):
                return out[0], out[1]
            # otherwise, assume first element is total
            total = out[0]
            return total, {}

        # Unknown
        return out, {'loss': out}

    @classmethod
    def as_logging_dict(cls, out, total_key: str = 'loss') -> Dict[str, Any]:
        """Return a flat dict of values suitable for logging (Keras, CSV).

        Ensures `total_key` is present and returns Tensor values as-is; callers
        are responsible for converting tensors to Python floats if needed.
        """
        total, comps = cls.normalize_return(out)
        result = dict(comps)
        # Overwrite or set total
        result[total_key] = total
        return result


# -------------------------
# Concrete loss implementations (centralized)
# -------------------------
import numpy as np


@Losses.register(name="focal_loss", tags=["classification", "imbalanced", "focal"])
def focal_loss(model, true_labels, logits, alpha=None, gamma=None, reduce=True):
    """Focal loss implementation that accepts a `model` context for hyperparams.

    The signature mirrors the original `CustomTrainModel.focal_loss` but is
    implemented centrally to avoid duplication.
    """
    if alpha is None:
        alpha = getattr(getattr(model, 'config', None), 'FOCAL_ALPHA', 0.5)
    if gamma is None:
        gamma = getattr(getattr(model, 'config', None), 'FOCAL_GAMMA', 2.0)

    alpha = tf.cast(alpha, tf.float32)
    gamma = tf.cast(gamma, tf.float32)

    true_labels = tf.cast(true_labels, tf.float32)
    logits = tf.cast(logits, tf.float32)

    logits = tf.clip_by_value(logits, 1e-7, 1.0 - 1e-7)

    p_t = true_labels * logits + (1.0 - true_labels) * (1.0 - logits)
    focal_weight = tf.pow(1.0 - p_t, gamma)

    bce = -true_labels * tf.math.log(logits) - (1.0 - true_labels) * tf.math.log(1.0 - logits)
    class_weight = alpha * (1.0 - true_labels) + (1.0 - alpha) * true_labels
    focal = class_weight * focal_weight * bce

    if reduce:
        return tf.reduce_mean(focal)
    return focal


@Losses.register(name="dice_loss", tags=["classification", "overlap", "segmentation"])
def dice_loss(model, true_labels, logits, smooth=1.0, reduce=True):
    true_labels = tf.cast(true_labels, tf.float32)
    logits = tf.cast(logits, tf.float32)
    smooth = tf.cast(smooth, tf.float32)

    intersection = true_labels * logits
    numerator = 2.0 * intersection + smooth
    denominator = true_labels + logits + smooth
    dice_per_sample = numerator / (denominator + 1e-8)
    dice_loss_per_sample = 1.0 - dice_per_sample

    if reduce:
        return tf.reduce_mean(dice_loss_per_sample)
    return dice_loss_per_sample


@Losses.register(name="combined_direction_loss", tags=["classification", "composite", "direction"])
def combined_direction_loss(model, true_labels, logits, alpha=None, gamma=None,
                             focal_weight=0.5, dice_weight=0.5, reduce=True):
    focal = focal_loss(model, true_labels, logits, alpha=alpha, gamma=gamma, reduce=reduce)
    dice = dice_loss(model, true_labels, logits, reduce=reduce)
    return focal_weight * focal + dice_weight * dice


@Losses.register(name="compute_dynamic_alpha", tags=["utility", "adaptive", "helper"])
def compute_dynamic_alpha(model_or_labels, true_labels=None, min_alpha=0.3, max_alpha=0.7):
    """If called as compute_dynamic_alpha(model, labels) or compute_dynamic_alpha(labels).
    Returns clipped alpha in [min_alpha, max_alpha]."""
    if true_labels is None:
        true_labels = model_or_labels
    true_labels = tf.cast(true_labels, tf.float32)
    up_ratio = tf.reduce_mean(true_labels)
    alpha = up_ratio
    alpha = tf.clip_by_value(alpha, min_alpha, max_alpha)
    return alpha


@Losses.register(name="point_huber", tags=["regression", "robust", "logcosh"])
def point_huber(model, y_true_scaled, y_pred_scaled, last_close_scaled=None, delta=None):
    y_true = tf.squeeze(y_true_scaled, axis=1)
    y_pred = tf.squeeze(y_pred_scaled, axis=1)
    diffs = y_true - y_pred
    per_elem = tf.math.log(tf.cosh(diffs))
    per_elem = tf.clip_by_value(per_elem, -10.0, 10.0)
    result = tf.reduce_mean(tf.cast(per_elem, tf.float32))
    result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
    return result


@Losses.register(name="local_trend_loss", tags=["trend", "local", "regression"])
def local_trend_loss(model, x_window, y_true_raw, y_pred_raw, last_close_raw):
    last_close = tf.squeeze(last_close_raw, axis=1)
    last_close_scaled = model._to_scaled_static(last_close, model.pred_mean, model.pred_scale, model.eps)

    y_true_scaled = model._to_scaled_static(y_true_raw, model.pred_mean, model.pred_scale, model.eps)
    y_pred_scaled = model._to_scaled_static(y_pred_raw, model.pred_mean, model.pred_scale, model.eps)

    actual_trend = y_true_scaled - last_close_scaled
    pred_trend = y_pred_scaled - last_close_scaled
    trend_diffs = actual_trend - pred_trend

    per_elem = tf.math.log(tf.cosh(trend_diffs))
    per_elem = tf.clip_by_value(per_elem, -10.0, 10.0)

    result = model._reduce_mean(per_elem)
    result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
    return result


@Losses.register(name="extended_trend_loss", tags=["trend", "extended", "multi_scale"])
def extended_trend_loss(model, x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw):
    start_of_window = tf.squeeze(x_window[:, 0:1], axis=1)
    start_scaled = model._to_scaled_static(start_of_window, model.pred_mean, model.pred_scale, model.eps)

    y_true_scaled = model._to_scaled_static(y_true_raw, model.pred_mean, model.pred_scale, model.eps)
    y_pred_scaled = model._to_scaled_static(y_pred_raw, model.pred_mean, model.pred_scale, model.eps)

    global_diffs = (y_true_scaled - start_scaled) - (y_pred_scaled - start_scaled)
    global_logcosh = tf.math.log(tf.cosh(global_diffs))
    global_logcosh = tf.clip_by_value(global_logcosh, -10.0, 10.0)
    global_loss = model._reduce_mean(global_logcosh)
    global_loss = tf.where(tf.math.is_finite(global_loss), global_loss, tf.constant(0.0, dtype=tf.float32))

    last_close = tf.squeeze(last_close_raw, axis=1)
    last_close_scaled = model._to_scaled(last_close)
    pred_trend_scaled = y_pred_scaled - last_close_scaled

    n_trend_features = tf.shape(extended_trends)[1]

    def compute_extended():
        eps = tf.cast(1e-8, tf.float32)
        long_term_trend = tf.cast(extended_trends[:, -1], tf.float32)
        long_term_trend = tf.clip_by_value(long_term_trend, -0.999, 1e6)
        past_price_long = last_close / (1.0 + long_term_trend + eps)
        long_price_diff_raw = last_close - past_price_long
        long_price_diff_scaled = long_price_diff_raw / (model.pred_scale + model.eps)

        long_diffs = pred_trend_scaled - long_price_diff_scaled
        long_logcosh = tf.math.log(tf.cosh(long_diffs))
        long_logcosh = tf.clip_by_value(long_logcosh, -10.0, 10.0)
        extended_loss_long = model._reduce_mean(long_logcosh)
        extended_loss_long = tf.where(tf.math.is_finite(extended_loss_long), extended_loss_long, tf.constant(0.0, dtype=tf.float32))

        def compute_multi():
            short_trends = tf.cast(extended_trends[:, :-1], tf.float32)
            short_trends = tf.clip_by_value(short_trends, -0.999, 1e6)

            past_prices = last_close[:, None] / (1.0 + short_trends + eps)
            short_price_diff_raw = last_close[:, None] - past_prices
            short_price_diff_scaled = short_price_diff_raw / (model.pred_scale + model.eps)

            short_diffs = tf.expand_dims(pred_trend_scaled, 1) - short_price_diff_scaled
            logcosh_losses = tf.math.log(tf.cosh(short_diffs))
            logcosh_losses = tf.clip_by_value(logcosh_losses, -10.0, 10.0)

            per_scale_mean = tf.reduce_mean(logcosh_losses, axis=0)
            denom = tf.reduce_mean(per_scale_mean) + model.eps
            normalized_per_scale = per_scale_mean / denom
            result = tf.reduce_mean(normalized_per_scale)
            result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
            return result

        def no_multi():
            return tf.constant(0.0, dtype=tf.float32)

        multi_scale_loss = tf.cond(tf.greater(tf.shape(extended_trends)[1], 1), compute_multi, no_multi)
        return extended_loss_long + multi_scale_loss

    def no_extended():
        return tf.constant(0.0, dtype=tf.float32)

    extended_loss = tf.cond(tf.greater(n_trend_features, 0), compute_extended, no_extended)

    return global_loss, extended_loss


@Losses.register(name="custom_loss", tags=["composite", "default", "multi_output"])
def custom_loss(model, x_window, y_true, y_pred, last_close, extended_trends):
    # Convert to expected types and shapes exactly as original model implementation
    y_true = tf.cast(y_true, tf.float32)
    y_true_raw = y_true * model.pred_scale + model.pred_mean
    last_close_squeeze = tf.squeeze(last_close, axis=1)

    y_true_h0 = y_true[:, 0:1]
    y_true_h1 = y_true[:, 1:2]
    y_true_h2 = y_true[:, 2:3]

    y_true_raw_h0 = y_true_raw[:, 0]
    y_true_raw_h1 = y_true_raw[:, 1]
    y_true_raw_h2 = y_true_raw[:, 2]

    deadband_bps = tf.cast(getattr(model.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
    deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

    ret_h0 = (y_true_raw_h0) / (last_close_squeeze + model.eps)
    ret_h1 = (y_true_raw_h1) / (last_close_squeeze + model.eps)
    ret_h2 = (y_true_raw_h2) / (last_close_squeeze + model.eps)

    mask_h0 = tf.cast(tf.abs(ret_h0) > deadband, tf.float32)
    mask_h1 = tf.cast(tf.abs(ret_h1) > deadband, tf.float32)
    mask_h2 = tf.cast(tf.abs(ret_h2) > deadband, tf.float32)

    true_dir_h0 = tf.cast(ret_h0 > deadband, tf.float32)
    true_dir_h1 = tf.cast(ret_h1 > deadband, tf.float32)
    true_dir_h2 = tf.cast(ret_h2 > deadband, tf.float32)

    # y_pred unpacking
    price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2 = y_pred

    point_loss_h0_val = model.lambda_short * point_huber(model, y_true_h0, price_h0)
    point_loss_h1_val = model.lambda_point * point_huber(model, y_true_h1, price_h1)
    point_loss_h2_val = model.lambda_long * point_huber(model, y_true_h2, price_h2)
    point_loss_val = point_loss_h0_val + point_loss_h1_val + point_loss_h2_val

    pred_scale = tf.cast(model.pred_scale + model.eps, tf.float32)
    extended_trends_scaled_h0 = extended_trends[:, 0:1] / pred_scale
    extended_trends_scaled_h1 = extended_trends[:, 1:2] / pred_scale
    extended_trends_scaled_h2 = extended_trends[:, 2:3] / pred_scale

    trend_loss_h0 = tf.reduce_mean(tf.square(price_h0 - extended_trends_scaled_h0))
    trend_loss_h1 = tf.reduce_mean(tf.square(price_h1 - extended_trends_scaled_h1))
    trend_loss_h2 = tf.reduce_mean(tf.square(price_h2 - extended_trends_scaled_h2))

    sign_pred_h0 = tf.sign(price_h0)
    sign_pred_h1 = tf.sign(price_h1)
    sign_pred_h2 = tf.sign(price_h2)

    dir_agree_h01 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h0, sign_pred_h1), tf.float32))
    dir_agree_h12 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h1, sign_pred_h2), tf.float32))
    dir_disagree_loss = 1.0 - (dir_agree_h01 + dir_agree_h12) / 2.0

    abs_pred_h0 = tf.abs(price_h0)
    abs_pred_h1 = tf.abs(price_h1)
    abs_pred_h2 = tf.abs(price_h2)

    magnitude_h01_violation = tf.nn.relu(abs_pred_h0 - abs_pred_h1)
    magnitude_h12_violation = tf.nn.relu(abs_pred_h1 - abs_pred_h2)
    magnitude_loss = tf.reduce_mean(magnitude_h01_violation + magnitude_h12_violation)

    sign_target_h0 = tf.sign(y_true_raw_h0)
    sign_target_h1 = tf.sign(y_true_raw_h1)
    sign_target_h2 = tf.sign(y_true_raw_h2)
    target_smoothness_loss = tf.reduce_mean(
        tf.cast(tf.math.logical_xor(sign_target_h1 == sign_target_h0, 
                                     sign_target_h1 == sign_target_h2), tf.float32)
    )

    coherence_penalty = (dir_disagree_loss + magnitude_loss + target_smoothness_loss) / 3.0

    local_trend_h0 = tf.constant(0.0, dtype=tf.float32)
    global_trend_h0 = tf.constant(0.0, dtype=tf.float32)
    extended_trend_h0 = model.lambda_extended_trend * trend_loss_h0
    local_trend_h1 = tf.constant(0.0, dtype=tf.float32)
    global_trend_h1 = tf.constant(0.0, dtype=tf.float32)
    extended_trend_h1 = model.lambda_extended_trend * trend_loss_h1
    local_trend_h2 = tf.constant(0.0, dtype=tf.float32)
    global_trend_h2 = tf.constant(0.0, dtype=tf.float32)
    extended_trend_h2 = model.lambda_extended_trend * trend_loss_h2

    trend_loss_val = extended_trend_h0 + extended_trend_h1 + extended_trend_h2 + coherence_penalty * 0.01

    dir_pred_h0 = tf.squeeze(dir_h0, axis=1)
    dir_pred_h1 = tf.squeeze(dir_h1, axis=1)
    dir_pred_h2 = tf.squeeze(dir_h2, axis=1)

    alpha_h0 = compute_dynamic_alpha(true_dir_h0)
    alpha_h1 = compute_dynamic_alpha(true_dir_h1)
    alpha_h2 = compute_dynamic_alpha(true_dir_h2)

    per_ex_h0 = combined_direction_loss(model, true_dir_h0, dir_pred_h0, alpha=alpha_h0, 
                                        focal_weight=0.5, dice_weight=0.5, reduce=False)
    per_ex_h1 = combined_direction_loss(model, true_dir_h1, dir_pred_h1, alpha=alpha_h1,
                                        focal_weight=0.5, dice_weight=0.5, reduce=False)
    per_ex_h2 = combined_direction_loss(model, true_dir_h2, dir_pred_h2, alpha=alpha_h2,
                                        focal_weight=0.5, dice_weight=0.5, reduce=False)

    dir_loss_h0 = tf.reduce_sum(per_ex_h0 * mask_h0) / (tf.reduce_sum(mask_h0) + model.eps)
    dir_loss_h1 = tf.reduce_sum(per_ex_h1 * mask_h1) / (tf.reduce_sum(mask_h1) + model.eps)
    dir_loss_h2 = tf.reduce_sum(per_ex_h2 * mask_h2) / (tf.reduce_sum(mask_h2) + model.eps)
    total_dir_loss = model.lambda_dir * (dir_loss_h0 + dir_loss_h1 + dir_loss_h2)

    var_floor = tf.cast(getattr(model.config, 'VAR_FLOOR', 1e-4), tf.float32)
    var_cap = tf.cast(getattr(model.config, 'VAR_CAP', 1e4), tf.float32)
    var_h0_c = tf.clip_by_value(var_h0, var_floor, var_cap)
    var_h1_c = tf.clip_by_value(var_h1, var_floor, var_cap)
    var_h2_c = tf.clip_by_value(var_h2, var_floor, var_cap)

    log_2pi = tf.constant(1.8378770664093453, dtype=tf.float32)

    nll_h0 = 0.5 * (log_2pi + tf.math.log(var_h0_c + model.eps)) + 0.5 * tf.square(y_true_h0 - price_h0) / (var_h0_c + model.eps)
    nll_h0_val = tf.reduce_mean(nll_h0)
    nll_h1 = 0.5 * (log_2pi + tf.math.log(var_h1_c + model.eps)) + 0.5 * tf.square(y_true_h1 - price_h1) / (var_h1_c + model.eps)
    nll_h1_val = tf.reduce_mean(nll_h1)
    nll_h2 = 0.5 * (log_2pi + tf.math.log(var_h2_c + model.eps)) + 0.5 * tf.square(y_true_h2 - price_h2) / (var_h2_c + model.eps)
    nll_h2_val = tf.reduce_mean(nll_h2)
    total_nll = model.lambda_var * (nll_h0_val + nll_h1_val + nll_h2_val)

    mu_h0 = tf.squeeze(price_h0, axis=1)
    mu_h1 = tf.squeeze(price_h1, axis=1)
    mu_h2 = tf.squeeze(price_h2, axis=1)
    sigma_h0 = tf.sqrt(tf.squeeze(var_h0_c, axis=1) + model.eps)
    sigma_h1 = tf.sqrt(tf.squeeze(var_h1_c, axis=1) + model.eps)
    sigma_h2 = tf.sqrt(tf.squeeze(var_h2_c, axis=1) + model.eps)
    deadband_delta_scaled = (deadband * last_close_squeeze) / (model.pred_scale + model.eps)
    z_up_h0 = (mu_h0 - deadband_delta_scaled) / (sigma_h0 + model.eps)
    z_up_h1 = (mu_h1 - deadband_delta_scaled) / (sigma_h1 + model.eps)
    z_up_h2 = (mu_h2 - deadband_delta_scaled) / (sigma_h2 + model.eps)
    gauss_p_up_h0 = 0.5 * (1.0 + tf.math.erf(z_up_h0 / tf.constant(np.sqrt(2.0), dtype=tf.float32)))
    gauss_p_up_h1 = 0.5 * (1.0 + tf.math.erf(z_up_h1 / tf.constant(np.sqrt(2.0), dtype=tf.float32)))
    gauss_p_up_h2 = 0.5 * (1.0 + tf.math.erf(z_up_h2 / tf.constant(np.sqrt(2.0), dtype=tf.float32)))

    lambda_dir_align = tf.constant(float(getattr(model.config, 'LAMBDA_DIR_ALIGN', 0.0)), dtype=tf.float32)

    align_h0 = tf.keras.losses.binary_crossentropy(gauss_p_up_h0, dir_pred_h0)
    align_h1 = tf.keras.losses.binary_crossentropy(gauss_p_up_h1, dir_pred_h1)
    align_h2 = tf.keras.losses.binary_crossentropy(gauss_p_up_h2, dir_pred_h2)
    align_h0 = tf.reduce_sum(align_h0 * mask_h0) / (tf.reduce_sum(mask_h0) + model.eps)
    align_h1 = tf.reduce_sum(align_h1 * mask_h1) / (tf.reduce_sum(mask_h1) + model.eps)
    align_h2 = tf.reduce_sum(align_h2 * mask_h2) / (tf.reduce_sum(mask_h2) + model.eps)
    dir_align_loss = lambda_dir_align * (align_h0 + align_h1 + align_h2)

    reg_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)
    inter_reg = model.config.LAMBDA_INTER * reg_loss

    actual_trend = y_true[:, 1]
    pred_trend_scaled = tf.squeeze(price_h1, axis=1)
    actual_std = tf.math.reduce_std(actual_trend)
    pred_std = tf.math.reduce_std(pred_trend_scaled)
    vol_diff = tf.abs(pred_std - actual_std)
    vol_diff_clipped = tf.minimum(vol_diff, 10.0)
    vol_loss = vol_diff_clipped * model.lambda_vol
    vol_loss = tf.where(tf.math.is_finite(vol_loss), vol_loss, tf.constant(0.0, dtype=tf.float32))

    total = (
        0.2 * point_loss_val +
        0.1 * trend_loss_val +      # Trend baseline consistency (auxiliary)
        1.0 * total_dir_loss +      # Direction classification (INCREASED from 0.2)
        0.1 * dir_align_loss +     # Distribution alignment (increased for better calibration)
        reg_loss +
        0.05 * inter_reg +           # Indicator correlation (weak regularization)
        0.05 * vol_loss +           # Volatility penalty (very weak)
        0.05 * coherence_penalty +   # STRENGTHENED: Cross-horizon coherence (0.01 → 0.1)
        1.0 * total_nll             # Variance NLL (INCREASED from 0.5 for better calibration)
    )

    return (
        total,
        point_loss_h0_val, point_loss_h1_val, point_loss_h2_val,
        local_trend_h0, global_trend_h0, extended_trend_h0,
        local_trend_h1, global_trend_h1, extended_trend_h1,
        local_trend_h2, global_trend_h2, extended_trend_h2,
        dir_loss_h0, dir_loss_h1, dir_loss_h2,
        nll_h0_val, nll_h1_val, nll_h2_val,
        reg_loss, inter_reg, vol_loss
    )

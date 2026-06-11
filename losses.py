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
from collections import namedtuple
import tensorflow as tf

from core.registry import BaseRegistry

# Robust representation of the 34-component loss return from custom_loss.
# NamedTuple is a tuple subclass, so all existing positional unpacks in
# CustomTrainModel.train_step / test_step, lambda calibration sampler,
# and tests continue to work unchanged. New code can use attribute access.
LossComponents = namedtuple(
    'LossComponents',
    [
        'total',
        'point_h0', 'point_h1', 'point_h2',
        'local_h0', 'global_h0', 'extended_h0',
        'local_h1', 'global_h1', 'extended_h1',
        'local_h2', 'global_h2', 'extended_h2',
        'dir_h0', 'dir_h1', 'dir_h2',
        'nll_h0', 'nll_h1', 'nll_h2',
        'reg_loss', 'inter_reg', 'vol_loss',
        'crps_h0', 'crps_h1', 'crps_h2',
        'soft_ece_h0', 'soft_ece_h1', 'soft_ece_h2',
        't_perp_total', 'casimir_val', 'vac_val', 'hd_val', 'ife_val',
        'vac_overflow_val',
    ]
)


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
    """Log-cosh point loss (smooth, robust regression loss).

    Registered name "point_huber" is retained for backward compatibility with
    tests, the Losses registry, and any external code that does Losses.get('point_huber').
    The implementation has always been log(cosh) (see exhaustive tests and
    CustomTrainModel point_huber delegation comment). A separate piecewise Huber
    implementation exists on CustomTrainModel.huber but is not wired into the
    primary point supervision path.
    """
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
        # Extended trends are now absolute deltas (price[t] - price[t-p]) per the
        # contract in DataProcessor.compute_extended_trend_features (see model.py:282).
        # Previously this code treated them as fractional returns (last / (1 + trend)),
        # which was invalidated by the data-layer refactor. Fix: use direct delta arithmetic.
        long_term_trend = tf.cast(extended_trends[:, -1], tf.float32)
        long_term_trend = tf.clip_by_value(long_term_trend, -1e6, 1e6)  # deltas, not returns
        # implied delta from the trend feature is exactly the feature value itself
        long_price_diff_raw = long_term_trend
        long_price_diff_scaled = model._to_scaled_static(
            long_price_diff_raw, model.pred_mean, model.pred_scale, model.eps
        )

        long_diffs = pred_trend_scaled - long_price_diff_scaled
        long_logcosh = tf.math.log(tf.cosh(long_diffs))
        long_logcosh = tf.clip_by_value(long_logcosh, -10.0, 10.0)
        extended_loss_long = model._reduce_mean(long_logcosh)
        extended_loss_long = tf.where(tf.math.is_finite(extended_loss_long), extended_loss_long, tf.constant(0.0, dtype=tf.float32))

        def compute_multi():
            short_trends = tf.cast(extended_trends[:, :-1], tf.float32)
            short_trends = tf.clip_by_value(short_trends, -1e6, 1e6)

            # For each short period the feature value *is* the absolute delta over that lag.
            short_price_diff_raw = short_trends
            short_price_diff_scaled = model._to_scaled_static(
                short_price_diff_raw, model.pred_mean, model.pred_scale, model.eps
            )

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


@Losses.register(name="crps_gaussian_loss", tags=["calibration", "regression", "probabilistic"])
def crps_gaussian_loss(model, y_true_scaled, mu_scaled, var_scaled):
    """Continuous Ranked Probability Score for a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [omega*(2*Phi(omega)-1) + 2*phi(omega) - 1/sqrt(pi)]
    where omega = (y - mu) / sigma.

    Unlike NLL, CRPS simultaneously rewards both sharpness and reliability, making it
    harder for the variance head to collapse sigma to game the loss.
    Operates in the same scaled space as regression targets.
    """
    y = tf.cast(tf.squeeze(y_true_scaled, axis=1), tf.float32)
    mu = tf.cast(tf.squeeze(mu_scaled, axis=1), tf.float32)
    var = tf.cast(tf.squeeze(var_scaled, axis=1), tf.float32)
    sigma = tf.sqrt(tf.maximum(var, 1e-8))

    sqrt_2 = tf.constant(1.4142135623730951, dtype=tf.float32)
    inv_sqrt_pi = tf.constant(0.5641895835477563, dtype=tf.float32)
    inv_sqrt_2pi = tf.constant(0.3989422804014327, dtype=tf.float32)

    omega = (y - mu) / (sigma + 1e-8)
    Phi_omega = 0.5 * (1.0 + tf.math.erf(omega / sqrt_2))
    phi_omega = inv_sqrt_2pi * tf.exp(-0.5 * tf.square(omega))

    crps_per_sample = sigma * (omega * (2.0 * Phi_omega - 1.0) + 2.0 * phi_omega - inv_sqrt_pi)
    crps_per_sample = tf.clip_by_value(crps_per_sample, 0.0, 100.0)
    return tf.reduce_mean(crps_per_sample)


@Losses.register(name="soft_ece_loss", tags=["calibration", "classification", "ece"])
def soft_ece_loss(model, true_dir, dir_pred, mask, n_bins=10, bandwidth=None):
    """Differentiable Expected Calibration Error via Gaussian kernel soft binning.

    Standard ECE uses hard histogram bins whose discontinuities block gradient flow.
    This replaces hard membership with a Gaussian kernel:
        w_{bi} = exp(-(p_i - c_b)^2 / (2*h^2))
    where c_b are bin centers and h is the bandwidth (default = half bin width).

    soft_ece = sum_b |acc_b - conf_b| * (sum_w_b / N_eff)

    Applied with the deadband mask so neutral samples are excluded, consistent
    with direction loss treatment.
    """
    p = tf.cast(dir_pred, tf.float32)
    y = tf.cast(true_dir, tf.float32)
    m = tf.cast(mask, tf.float32)

    if bandwidth is None:
        bandwidth = 1.0 / (2.0 * n_bins)
    h2 = tf.constant(2.0 * bandwidth ** 2, dtype=tf.float32)

    total_eff = tf.reduce_sum(m) + 1e-8
    ece = tf.constant(0.0, dtype=tf.float32)
    for i in range(n_bins):
        c = tf.constant((float(i) + 0.5) / float(n_bins), dtype=tf.float32)
        w = tf.exp(-tf.square(p - c) / h2) * m
        sum_w = tf.reduce_sum(w) + 1e-8
        soft_acc = tf.reduce_sum(w * y) / sum_w
        soft_conf = tf.reduce_sum(w * p) / sum_w
        bin_weight = tf.reduce_sum(w) / total_eff
        ece = ece + bin_weight * tf.abs(soft_acc - soft_conf)
    return ece


@Losses.register(name="t_perp_calibration_loss", tags=["calibration", "t_perp", "perpendicular"])
def t_perp_calibration_loss(model, y_true_h, price_h, var_h):
    """T_⊥ Calibration Loss — 'zero does not exist, only T_⊥'.

    In the QBOX ontology, anything that vanishes is not destroyed — it flows into
    the perpendicular tensor T_⊥ (hidden dimensions).  Semantically transposed
    to trading: the variance head IS our estimate of T_⊥ magnitude (unexplained
    residual energy / hidden order-flow).

    Calibration target: predicted σ should track empirical residual std of the batch.
    Prevents variance from collapsing to a constant by anchoring it to actual errors.

    Loss = (empirical_std(residuals) - mean(predicted_σ))²
    """
    eps = tf.constant(1e-8, dtype=tf.float32)
    y = tf.cast(tf.squeeze(y_true_h, axis=1), tf.float32)      # [B]
    mu = tf.cast(tf.squeeze(price_h, axis=1), tf.float32)       # [B]
    var = tf.cast(tf.squeeze(var_h, axis=1), tf.float32)         # [B]

    residuals = y - mu
    empirical_std = tf.math.reduce_std(residuals) + eps
    predicted_sigma = tf.reduce_mean(tf.sqrt(var + eps))
    return tf.square(empirical_std - predicted_sigma)


@Losses.register(name="casimir_interference_loss", tags=["calibration", "casimir", "multi_scale", "t_perp"])
def casimir_interference_loss(model, price_h0, price_h1, price_h2,
                               var_h0, var_h1, var_h2):
    """Casimir Inter-Scale Interference Loss.

    Casimir effect in QBOX: at boundaries between layers (scales), destructive
    interference causes energy to flow into T_⊥.  Transposed to trading:
    when short-horizon and long-horizon predictions point in OPPOSITE directions
    (destructive interference), the model is genuinely uncertain — the variance
    MUST be high at those crossing points.

    Penalises low predicted variance when cross-horizon interference is high.

    Loss = mean(interference_h01 / avg_σ² + interference_h12 / avg_σ²)
    where interference = relu(-sign(p_a) * sign(p_b))  [high when opposite signs]
    """
    eps = tf.constant(1e-8, dtype=tf.float32)
    p0 = tf.cast(tf.squeeze(price_h0, axis=1), tf.float32)   # [B]
    p1 = tf.cast(tf.squeeze(price_h1, axis=1), tf.float32)
    p2 = tf.cast(tf.squeeze(price_h2, axis=1), tf.float32)
    v0 = tf.cast(tf.squeeze(var_h0, axis=1), tf.float32)
    v1 = tf.cast(tf.squeeze(var_h1, axis=1), tf.float32)
    v2 = tf.cast(tf.squeeze(var_h2, axis=1), tf.float32)

    # Destructive interference = positive when predictions disagree in sign
    interf_h01 = tf.nn.relu(-p0 * p1)                         # [B]
    interf_h12 = tf.nn.relu(-p1 * p2)

    avg_var_h01 = 0.5 * (v0 + v1) + eps
    avg_var_h12 = 0.5 * (v1 + v2) + eps

    # High interference with low variance → large penalty (T_⊥ should be high here)
    casimir = tf.reduce_mean(interf_h01 / avg_var_h01 + interf_h12 / avg_var_h12)
    return casimir


@Losses.register(name="vacuum_bandwidth_loss", tags=["regulation", "vacuum", "t_perp", "self_limiting"])
def vacuum_bandwidth_loss(model, price_h0, price_h1, price_h2, lambda_vac=None):
    """Vacuum Bandwidth Loss — Λ_vac self-limiting term.

    QBOX vacuum: the vacuum has limited bandwidth Λ_vac; when local energy exceeds it,
    the excess is automatically shunted to T_⊥ producing supra-stable plateaus.
    Transposed to trading: the cross-horizon spread of predictions (h0–h2) should not
    exceed Λ_vac.  When it does, the model is over-extrapolating across timescales
    (the market's 'vacuum' bandwidth is saturated → uncertainty should rise).

    Only penalises violations ABOVE Λ_vac (relu clamp = supra-stable self-limiting).

    Loss = mean(relu(std(p0, p1, p2) - Λ_vac))
    """
    if lambda_vac is None:
        lambda_vac = tf.constant(float(getattr(getattr(model, 'config', None), 'LAMBDA_VAC', 0.0)),
                                 dtype=tf.float32)
    lv = tf.cast(lambda_vac, tf.float32)

    p0 = tf.cast(tf.squeeze(price_h0, axis=1), tf.float32)    # [B]
    p1 = tf.cast(tf.squeeze(price_h1, axis=1), tf.float32)
    p2 = tf.cast(tf.squeeze(price_h2, axis=1), tf.float32)

    stacked = tf.stack([p0, p1, p2], axis=1)                   # [B, 3]
    cross_std = tf.math.reduce_std(stacked, axis=1)             # [B]
    violation = tf.nn.relu(cross_std - lv)
    mean_viol = tf.reduce_mean(violation)

    # P0-2: opt-in via LAMBDA_VAC (default 0 in Config). Graph-safe conditional for
    # Keras fit / train_step / @tf.function autograph (OperatorNotAllowedInGraphError).
    # When lv <= 0 the term contributes exactly 0 (no penalty, no gradient from it).
    return tf.cond(
        tf.greater(lv, 0.0),
        lambda: mean_viol,
        lambda: tf.constant(0.0, dtype=tf.float32)
    )


@Losses.register(name="hyper_decoherence_coupling_loss",
                 tags=["calibration", "volatility", "decoherence", "t_perp"])
def hyper_decoherence_coupling_loss(model, x_window, var_h0, var_h1, var_h2):
    """Hyper-Decoherence Coupling Loss.

    QBOX High-T T_⊥-Computer: hyper-decoherence is a RESOURCE.  At high temperatures
    (high volatility), instead of fighting uncertainty we should EMBRACE it — the model
    must express high σ when local market volatility is high.  Penalises suppressed σ
    during volatile regimes.

    The 'ΔS = S_real − S_measured = |T_⊥|' framing: when the market is highly disordered
    (high vol) but the model claims to be certain (low σ), the entropy gap is large.
    We minimise that gap.

    Loss = −mean(local_vol · avg_σ)   [negative = reward coupling; higher vol → higher σ]

    In practice we negate so that minimising the loss drives positive coupling.
    """
    eps = tf.constant(1e-8, dtype=tf.float32)
    x = tf.cast(x_window, tf.float32)                          # [B, LOOKBACK]
    # Local volatility: std across time dimension
    local_vol = tf.math.reduce_std(x, axis=1)                  # [B]

    v0 = tf.cast(tf.squeeze(var_h0, axis=1), tf.float32)       # [B]
    v1 = tf.cast(tf.squeeze(var_h1, axis=1), tf.float32)
    v2 = tf.cast(tf.squeeze(var_h2, axis=1), tf.float32)
    avg_sigma = tf.sqrt((v0 + v1 + v2) / 3.0 + eps)            # [B]

    # Negative coupling: minimising loss → σ grows with local_vol
    return -tf.reduce_mean(local_vol * avg_sigma)


@Losses.register(name="information_flow_entropy_loss",
                 tags=["regulation", "diversity", "multi_scale", "information"])
def information_flow_entropy_loss(model, price_h0, price_h1, price_h2, rho_max=None):
    """Information Flow Entropy Loss.

    QBOX 'movement = unrolling of transcendental number digits': each horizon is
    meant to reveal ADDITIONAL new information — the next decimal of Ξ.  If
    h0, h1, h2 are simply scaled copies of each other, the network has learned to
    copy rather than integrate multi-scale information.

    Penalises cross-horizon Pearson correlation that exceeds ρ_max:
        Loss = relu(|corr(h0, h1)| − ρ_max)² + relu(|corr(h1, h2)| − ρ_max)²

    This forces horizons to carry non-redundant information (diverse views of the
    same market state), analogous to ensuring each digit of Ξ is actually new.
    """
    if rho_max is None:
        rho_max = float(getattr(getattr(model, 'config', None), 'RHO_MAX', 0.95))
    rho_max_c = tf.constant(float(rho_max), dtype=tf.float32)
    eps = tf.constant(1e-8, dtype=tf.float32)

    p0 = tf.cast(tf.squeeze(price_h0, axis=1), tf.float32)    # [B]
    p1 = tf.cast(tf.squeeze(price_h1, axis=1), tf.float32)
    p2 = tf.cast(tf.squeeze(price_h2, axis=1), tf.float32)

    def _pearson(a, b):
        ma = a - tf.reduce_mean(a)
        mb = b - tf.reduce_mean(b)
        cov = tf.reduce_mean(ma * mb)
        std_a = tf.math.reduce_std(a) + eps
        std_b = tf.math.reduce_std(b) + eps
        return cov / (std_a * std_b)

    r01 = _pearson(p0, p1)
    r12 = _pearson(p1, p2)

    viol = (tf.nn.relu(tf.abs(r01) - rho_max_c) ** 2 +
            tf.nn.relu(tf.abs(r12) - rho_max_c) ** 2)
    return viol


@Losses.register(name="vacuum_overflow_t_perp_loss",
                 tags=["calibration", "t_perp", "vacuum", "overflow", "precision"])
def vacuum_overflow_t_perp_loss(model, vacuum_overflow,
                                y_true_h0, price_h0,
                                y_true_h1, price_h1,
                                y_true_h2, price_h2):
    """Vacuum Overflow T_⊥ Precision Loss.

    Anchors the observable vacuum overflow signal to the actual prediction residual
    magnitude, making the overflow a precise, calibrated measure of T_⊥ intensity.

    Physics framing: the overflow (energy exceeding the vacuum ceiling E_max) is the
    portion of hidden-dimension energy that the network's visible subspace cannot absorb.
    This MUST equal the unexplained residual in real-space predictions, otherwise T_⊥ is
    either under-reported (overflow too small) or a hallucination (overflow too large).

    Loss = (mean_overflow - mean_residual_mag)² / (mean_residual_mag² + ε)

    Normalised by residual² so the loss is scale-invariant across training stages.
    """
    eps = tf.constant(1e-8, dtype=tf.float32)

    ov = tf.cast(tf.squeeze(vacuum_overflow, axis=1), tf.float32)   # [B]

    y0 = tf.cast(tf.squeeze(y_true_h0, axis=1), tf.float32)         # [B]
    p0 = tf.cast(tf.squeeze(price_h0,  axis=1), tf.float32)
    y1 = tf.cast(tf.squeeze(y_true_h1, axis=1), tf.float32)
    p1 = tf.cast(tf.squeeze(price_h1,  axis=1), tf.float32)
    y2 = tf.cast(tf.squeeze(y_true_h2, axis=1), tf.float32)
    p2 = tf.cast(tf.squeeze(price_h2,  axis=1), tf.float32)

    # Per-sample mean absolute residual across all three horizons
    residual_mag = (tf.abs(y0 - p0) + tf.abs(y1 - p1) + tf.abs(y2 - p2)) / 3.0  # [B]

    mean_ov  = tf.reduce_mean(ov)           # scalar
    mean_res = tf.reduce_mean(residual_mag) # scalar

    # Scale-invariant alignment loss
    return tf.square(mean_ov - mean_res) / (tf.square(mean_res) + eps)


def _compute_direction_labels_and_masks_tf(y_true_raw, last_close_squeeze, deadband_bps, eps=1e-8):
    """TF-graph version of direction labeling with deadband (single source of truth).

    Used by custom_loss (and can be reused by train_step/test_step direction metric code).
    Returns (mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2).
    Mirrors the logic previously duplicated in custom_loss, train/test_step, and _compute_...
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


@Losses.register(name="custom_loss", tags=["composite", "default", "multi_output"])
def custom_loss(model, x_window, y_true, y_pred, last_close, extended_trends,
               vacuum_overflow=None):
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
    mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2 = (
        _compute_direction_labels_and_masks_tf(
            y_true_raw, last_close_squeeze, deadband_bps, model.eps
        )
    )

    # The scaled deadband is still needed later for dir_align and gauss_p_up calculations.
    deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

    # y_pred unpacking
    price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2 = y_pred

    # Sanitize heads (belt-and-suspenders with output clips in build_model).
    # Ensures no NaN/Inf reaches *any* loss term (point, dir, nll, casimir, t_perp, ife, vacuum, hd, align, etc.).
    # Prevents 0*nan pollution in total even for "inactive" (lambda=0) terms, and keeps all components finite.
    price_h0 = tf.where(tf.math.is_finite(price_h0), price_h0, tf.zeros_like(price_h0))
    dir_h0   = tf.where(tf.math.is_finite(dir_h0),   dir_h0,   tf.ones_like(dir_h0) * 0.5)
    var_h0   = tf.where(tf.math.is_finite(var_h0),   var_h0,   tf.ones_like(var_h0))
    price_h1 = tf.where(tf.math.is_finite(price_h1), price_h1, tf.zeros_like(price_h1))
    dir_h1   = tf.where(tf.math.is_finite(dir_h1),   dir_h1,   tf.ones_like(dir_h1) * 0.5)
    var_h1   = tf.where(tf.math.is_finite(var_h1),   var_h1,   tf.ones_like(var_h1))
    price_h2 = tf.where(tf.math.is_finite(price_h2), price_h2, tf.zeros_like(price_h2))
    dir_h2   = tf.where(tf.math.is_finite(dir_h2),   dir_h2,   tf.ones_like(dir_h2) * 0.5)
    var_h2   = tf.where(tf.math.is_finite(var_h2),   var_h2,   tf.ones_like(var_h2))

    point_loss_h0_val = model.lambda_short * point_huber(model, y_true_h0, price_h0)
    point_loss_h1_val = model.lambda_point * point_huber(model, y_true_h1, price_h1)
    point_loss_h2_val = model.lambda_long * point_huber(model, y_true_h2, price_h2)
    point_loss_val = point_loss_h0_val + point_loss_h1_val + point_loss_h2_val
    point_loss_val = tf.where(tf.math.is_finite(point_loss_val), point_loss_val, tf.constant(0.0, dtype=tf.float32))
    point_loss_h0_val = tf.where(tf.math.is_finite(point_loss_h0_val), point_loss_h0_val, tf.constant(0.0, dtype=tf.float32))
    point_loss_h1_val = tf.where(tf.math.is_finite(point_loss_h1_val), point_loss_h1_val, tf.constant(0.0, dtype=tf.float32))
    point_loss_h2_val = tf.where(tf.math.is_finite(point_loss_h2_val), point_loss_h2_val, tf.constant(0.0, dtype=tf.float32))

    # Use the registered trend loss fns (now fixed for absolute-delta extended_trends
    # per DataProcessor contract). This activates the intended multi-scale "global from
    # window start" + feature-implied trend matching instead of the previous naive
    # (and incorrectly scaled) direct regression of future delta heads onto past deltas.
    # The returned values populate the per-horizon slots for logging/metrics and
    # contribute (via extended_trend_h*) to trend_loss_val.
    # last_close (the [B,1] tensor from the dataset) is passed; the fns squeeze internally.
    # Squeeze the per-horizon price outputs to 1D for the registered trend loss fns
    # (they expect 1D y_pred_raw for the per-horizon calls; model outputs are [B,1]).
    price_h0_s = tf.squeeze(price_h0, axis=1)
    price_h1_s = tf.squeeze(price_h1, axis=1)
    price_h2_s = tf.squeeze(price_h2, axis=1)

    local_trend_h0 = local_trend_loss(model, x_window, y_true_raw_h0, price_h0_s, last_close)
    g0, ext0 = extended_trend_loss(model, x_window, y_true_raw_h0, price_h0_s, extended_trends, last_close)
    local_trend_h1 = local_trend_loss(model, x_window, y_true_raw_h1, price_h1_s, last_close)
    g1, ext1 = extended_trend_loss(model, x_window, y_true_raw_h1, price_h1_s, extended_trends, last_close)
    local_trend_h2 = local_trend_loss(model, x_window, y_true_raw_h2, price_h2_s, last_close)
    g2, ext2 = extended_trend_loss(model, x_window, y_true_raw_h2, price_h2_s, extended_trends, last_close)

    # For backward compatibility of the "trend_loss_val" formula we keep the previous
    # structure (extended components + coherence). The globals are available in the
    # returned components for richer logging.
    trend_loss_h0 = ext0
    trend_loss_h1 = ext1
    trend_loss_h2 = ext2

    sign_pred_h0 = tf.sign(price_h0)
    sign_pred_h1 = tf.sign(price_h1)
    sign_pred_h2 = tf.sign(price_h2)

    dir_agree_h01 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h0, sign_pred_h1), tf.float32))
    dir_agree_h12 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h1, sign_pred_h2), tf.float32))
    dir_disagree_loss = 1.0 - (dir_agree_h01 + dir_agree_h12) / 2.0
    dir_disagree_loss = tf.where(tf.math.is_finite(dir_disagree_loss), dir_disagree_loss, tf.constant(0.0, dtype=tf.float32))

    abs_pred_h0 = tf.abs(price_h0)
    abs_pred_h1 = tf.abs(price_h1)
    abs_pred_h2 = tf.abs(price_h2)

    magnitude_h01_violation = tf.nn.relu(abs_pred_h0 - abs_pred_h1)
    magnitude_h12_violation = tf.nn.relu(abs_pred_h1 - abs_pred_h2)
    magnitude_loss = tf.reduce_mean(magnitude_h01_violation + magnitude_h12_violation)
    magnitude_loss = tf.where(tf.math.is_finite(magnitude_loss), magnitude_loss, tf.constant(0.0, dtype=tf.float32))

    sign_target_h0 = tf.sign(y_true_raw_h0)
    sign_target_h1 = tf.sign(y_true_raw_h1)
    sign_target_h2 = tf.sign(y_true_raw_h2)
    target_smoothness_loss = tf.reduce_mean(
        tf.cast(tf.math.logical_xor(sign_target_h1 == sign_target_h0, 
                                     sign_target_h1 == sign_target_h2), tf.float32)
    )
    target_smoothness_loss = tf.where(tf.math.is_finite(target_smoothness_loss), target_smoothness_loss, tf.constant(0.0, dtype=tf.float32))

    coherence_penalty = (dir_disagree_loss + magnitude_loss + target_smoothness_loss) / 3.0
    coherence_penalty = tf.where(tf.math.is_finite(coherence_penalty), coherence_penalty, tf.constant(0.0, dtype=tf.float32))

    # Assign from the registered calls above (scaled correctly, using fixed delta math).
    # Multiply the extended components by the per-horizon lambda here for consistency
    # with prior behavior (the registered fn returns the raw component loss).
    global_trend_h0 = g0
    extended_trend_h0 = model.lambda_extended_trend * trend_loss_h0
    global_trend_h1 = g1
    extended_trend_h1 = model.lambda_extended_trend * trend_loss_h1
    global_trend_h2 = g2
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
    dir_loss_h0 = tf.where(tf.math.is_finite(dir_loss_h0), dir_loss_h0, tf.constant(0.0, dtype=tf.float32))
    dir_loss_h1 = tf.where(tf.math.is_finite(dir_loss_h1), dir_loss_h1, tf.constant(0.0, dtype=tf.float32))
    dir_loss_h2 = tf.where(tf.math.is_finite(dir_loss_h2), dir_loss_h2, tf.constant(0.0, dtype=tf.float32))
    total_dir_loss = model.lambda_dir * (dir_loss_h0 + dir_loss_h1 + dir_loss_h2)
    total_dir_loss = tf.where(tf.math.is_finite(total_dir_loss), total_dir_loss, tf.constant(0.0, dtype=tf.float32))

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
    nll_h0_val = tf.where(tf.math.is_finite(nll_h0_val), nll_h0_val, tf.constant(0.0, dtype=tf.float32))
    nll_h1_val = tf.where(tf.math.is_finite(nll_h1_val), nll_h1_val, tf.constant(0.0, dtype=tf.float32))
    nll_h2_val = tf.where(tf.math.is_finite(nll_h2_val), nll_h2_val, tf.constant(0.0, dtype=tf.float32))
    total_nll = model.lambda_var * (nll_h0_val + nll_h1_val + nll_h2_val)
    total_nll = tf.where(tf.math.is_finite(total_nll), total_nll, tf.constant(0.0, dtype=tf.float32))

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

    # === CRPS LOSSES (Gaussian Continuous Ranked Probability Score) ===
    # Controlled by lambda_crps (default 0 → no effect on existing runs).
    # CRPS is a proper scoring rule that jointly rewards sharpness and calibration,
    # preventing the variance head from collapsing sigma to game NLL.
    lambda_crps = tf.constant(float(getattr(model, 'lambda_crps', 0.0)), dtype=tf.float32)
    crps_h0_val = crps_gaussian_loss(model, y_true_h0, price_h0, var_h0_c)
    crps_h1_val = crps_gaussian_loss(model, y_true_h1, price_h1, var_h1_c)
    crps_h2_val = crps_gaussian_loss(model, y_true_h2, price_h2, var_h2_c)
    total_crps = lambda_crps * (crps_h0_val + crps_h1_val + crps_h2_val)
    total_crps = tf.where(tf.math.is_finite(total_crps), total_crps, tf.constant(0.0, dtype=tf.float32))

    # === SOFT-ECE LOSSES (differentiable Expected Calibration Error) ===
    # Directly minimizes direction-head calibration error w.r.t. realized outcomes.
    # Complements dir-align (which aligns dir_head to Gaussian-implied P(up)) by also
    # aligning to actual labels. Controlled by lambda_soft_ece (default 0).
    lambda_soft_ece = tf.constant(float(getattr(model, 'lambda_soft_ece', 0.0)), dtype=tf.float32)
    soft_ece_h0_val = soft_ece_loss(model, true_dir_h0, dir_pred_h0, mask_h0)
    soft_ece_h1_val = soft_ece_loss(model, true_dir_h1, dir_pred_h1, mask_h1)
    soft_ece_h2_val = soft_ece_loss(model, true_dir_h2, dir_pred_h2, mask_h2)
    total_soft_ece = lambda_soft_ece * (soft_ece_h0_val + soft_ece_h1_val + soft_ece_h2_val)
    total_soft_ece = tf.where(tf.math.is_finite(total_soft_ece), total_soft_ece, tf.constant(0.0, dtype=tf.float32))

    # === T_⊥ / QBOX LOSSES ===
    # T_⊥ calibration: predicted σ tracks empirical residual std (T_⊥ is what we can't explain)
    lambda_t_perp = tf.constant(float(getattr(model, 'lambda_t_perp', 0.0)), dtype=tf.float32)
    t_perp_h0_val = t_perp_calibration_loss(model, y_true_h0, price_h0, var_h0_c)
    t_perp_h1_val = t_perp_calibration_loss(model, y_true_h1, price_h1, var_h1_c)
    t_perp_h2_val = t_perp_calibration_loss(model, y_true_h2, price_h2, var_h2_c)
    total_t_perp = lambda_t_perp * (t_perp_h0_val + t_perp_h1_val + t_perp_h2_val)
    total_t_perp = tf.where(tf.math.is_finite(total_t_perp), total_t_perp, tf.constant(0.0, dtype=tf.float32))

    # Casimir: destructive cross-horizon interference → T_⊥ (σ) must be high
    lambda_casimir = tf.constant(float(getattr(model, 'lambda_casimir', 0.0)), dtype=tf.float32)
    casimir_val = lambda_casimir * casimir_interference_loss(
        model, price_h0, price_h1, price_h2, var_h0_c, var_h1_c, var_h2_c)
    casimir_val = tf.where(tf.math.is_finite(casimir_val), casimir_val, tf.constant(0.0, dtype=tf.float32))

    # Vacuum bandwidth: cross-horizon spread must not exceed Λ_vac (self-limiting)
    # P0-2: now opt-in (default 0 in Config + helper). When 0 the term is 0.
    lambda_vac_cfg = tf.constant(float(getattr(getattr(model, 'config', None), 'LAMBDA_VAC', 0.0)),
                                 dtype=tf.float32)
    vac_val = vacuum_bandwidth_loss(model, price_h0, price_h1, price_h2, lambda_vac_cfg)
    vac_val = tf.where(tf.math.is_finite(vac_val), vac_val, tf.constant(0.0, dtype=tf.float32))

    # Hyper-decoherence: high local volatility should couple to high σ
    lambda_hd = tf.constant(float(getattr(model, 'lambda_hd', 0.0)), dtype=tf.float32)
    hd_val = lambda_hd * hyper_decoherence_coupling_loss(
        model, x_window, var_h0_c, var_h1_c, var_h2_c)
    hd_val = tf.where(tf.math.is_finite(hd_val), hd_val, tf.constant(0.0, dtype=tf.float32))

    # Information flow entropy: each horizon must carry non-redundant information
    lambda_ife = tf.constant(float(getattr(model, 'lambda_ife', 0.0)), dtype=tf.float32)
    ife_val = lambda_ife * information_flow_entropy_loss(model, price_h0, price_h1, price_h2)
    ife_val = tf.where(tf.math.is_finite(ife_val), ife_val, tf.constant(0.0, dtype=tf.float32))

    # Vacuum overflow T_⊥ precision: overflow tracks prediction residual magnitude
    # Active only when lambda_vac_overflow > 0 AND vacuum_overflow tensor is provided.
    lambda_vac_overflow = tf.constant(
        float(getattr(model, 'lambda_vac_overflow', 0.0)), dtype=tf.float32)
    if vacuum_overflow is not None:
        vac_overflow_val = lambda_vac_overflow * vacuum_overflow_t_perp_loss(
            model, vacuum_overflow,
            y_true_h0, price_h0,
            y_true_h1, price_h1,
            y_true_h2, price_h2,
        )
    else:
        vac_overflow_val = tf.constant(0.0, dtype=tf.float32)
    vac_overflow_val = tf.where(tf.math.is_finite(vac_overflow_val), vac_overflow_val, tf.constant(0.0, dtype=tf.float32))

    total = (
        point_loss_val +
        model.lambda_trend_outer * trend_loss_val +
        model.lambda_dir_outer * total_dir_loss +
        model.lambda_dir_align_outer * dir_align_loss +
        0  * reg_loss +
        0.1 * inter_reg +           # Indicator correlation (weak regularization)
        0.1 * vol_loss +           # Volatility penalty (very weak)
        model.lambda_coherence_outer * coherence_penalty +
        model.lambda_nll_outer * total_nll +
        total_crps +                # CRPS calibration (active only when lambda_crps > 0)
        total_soft_ece +            # Soft-ECE calibration (active only when lambda_soft_ece > 0)
        total_t_perp +              # T_⊥ calibration (active only when lambda_t_perp > 0)
        casimir_val +               # Casimir interference (active only when lambda_casimir > 0)
        vac_val +                   # Vacuum bandwidth self-limiting (always active, weight via Λ_vac)
        hd_val +                    # Hyper-decoherence coupling (active only when lambda_hd > 0)
        ife_val +                   # Information flow entropy (active only when lambda_ife > 0)
        vac_overflow_val            # Vacuum overflow T_⊥ precision (active when lambda_vac_overflow > 0)
    )
    total = tf.where(tf.math.is_finite(total), total, tf.constant(0.0, dtype=tf.float32))

    # Return as LossComponents (NamedTuple subclass). This preserves exact
    # 34-element tuple shape / positional unpacking for all callers while
    # enabling named attribute access (e.g. lc.extended_trend_h1).
    # Update the component list only by extending the namedtuple definition above.
    return LossComponents(
        total,
        point_loss_h0_val, point_loss_h1_val, point_loss_h2_val,
        local_trend_h0, global_trend_h0, extended_trend_h0,
        local_trend_h1, global_trend_h1, extended_trend_h1,
        local_trend_h2, global_trend_h2, extended_trend_h2,
        dir_loss_h0, dir_loss_h1, dir_loss_h2,
        nll_h0_val, nll_h1_val, nll_h2_val,
        reg_loss, inter_reg, vol_loss,
        crps_h0_val, crps_h1_val, crps_h2_val,
        soft_ece_h0_val, soft_ece_h1_val, soft_ece_h2_val,
        total_t_perp, casimir_val, vac_val, hd_val, ife_val,
        vac_overflow_val,
    )

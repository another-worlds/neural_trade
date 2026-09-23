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


def _logcosh_safe(x):
    """log(cosh(x)) that is finite for every float32 input and has a bounded gradient.

    ``tf.math.log(tf.cosh(x))`` overflows to +inf for |x| > ~89 and its backward pass is
    ``grad * sinh(x)``, which becomes ``0 * inf = NaN`` once the forward value has been
    clipped. This identity is exact, saturates gracefully (value -> |x| - log 2) and its
    gradient is tanh(x), so |grad| <= 1 everywhere.
    """
    x = tf.cast(x, tf.float32)
    return x + tf.math.softplus(-2.0 * x) - tf.constant(0.6931471805599453, dtype=tf.float32)


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
    per_elem = _logcosh_safe(diffs)
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

    per_elem = _logcosh_safe(trend_diffs)

    result = model._reduce_mean(per_elem)
    result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
    return result


@Losses.register(name="extended_trend_loss", tags=["trend", "extended", "multi_scale"])
def extended_trend_loss(model, x_window, y_true_raw, y_pred_scaled, extended_trends, last_close_raw,
                        horizon_idx=None):
    """Momentum prior: agreement between the horizon-k price head and the realised past
    delta over EXTENDED_TREND_PERIODS[k], computed in consistent scaled-delta units.

    Units contract (the reason this function was rewritten):
      * ``y_pred_scaled`` is the model's price head, which is ALREADY a scaled delta
        ((raw_delta - pred_mean) / pred_scale). It is compared as-is; re-scaling it
        attenuated its gradient by 1/pred_scale.
      * ``extended_trends[:, k]`` is a raw dollar delta (price[t] - price[t - p_k]) and
        is brought into the same units with ``_to_scaled_static``.
      * ``x_window`` and ``last_close_raw`` are accepted for signature compatibility and
        are NOT used. ``last_close`` is a raw price: passing it through the delta scaler
        gave ~421 in scaled units, ``cosh`` overflowed, ``clip_by_value`` pinned the
        term at 10.0 (the constant 1.333295 seen in every logged epoch) and the backward
        pass produced ``0 * inf = NaN`` that poisoned every weight on step 1.

    Returns ``(global_loss, extended_loss)`` for call-site compatibility. The "global"
    half is retired (it cancelled algebraically to the point loss) and is always 0; the
    old "multi-scale" sub-term (mean of per-scale means over their own mean) was
    identically 1.0 with zero gradient and is dropped.
    """
    zero = tf.constant(0.0, dtype=tf.float32)
    n_cols = extended_trends.shape[-1]
    if n_cols is not None and int(n_cols) == 0:
        return zero, zero
    k = -1 if horizon_idx is None else int(horizon_idx)
    if n_cols is not None and k >= int(n_cols):
        k = int(n_cols) - 1

    past_delta_raw = tf.cast(extended_trends[:, k], tf.float32)
    past_delta_scaled = model._to_scaled_static(past_delta_raw, model.pred_mean, model.pred_scale, model.eps)
    y_pred = tf.cast(y_pred_scaled, tf.float32)

    ext = model._reduce_mean(_logcosh_safe(y_pred - past_delta_scaled))
    ext = tf.where(tf.math.is_finite(ext), ext, zero)
    return zero, ext


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
    """T_perp calibration: batch-mean predicted variance tracks batch-mean residual energy.

    Loss = (log(mean(residual^2) + eps) - log(mean(var) + eps))^2

    Why this form (the previous one rewarded the collapse it claimed to prevent):
      * (std(residual) - mean(sqrt(var)))^2 compared a root-mean-square against a mean of
        roots. By Jensen the two agree only when sigma is CONSTANT across the batch, so a
        perfectly calibrated heteroscedastic model was penalised and a constant sigma was
        the unique zero - exactly the variance-head collapse observed in training.
      * The residual statistic is the TARGET, so it is wrapped in stop_gradient: the term
        cannot move mu to make the residuals fit sigma (it previously leaked gradient into
        the price head).
      * The log-ratio is scale-free with bounded gradient, and zero iff the batch-mean
        variance equals the batch-mean squared residual.
    """
    eps = tf.constant(1e-8, dtype=tf.float32)
    y = tf.cast(tf.squeeze(y_true_h, axis=1), tf.float32)      # [B]
    mu = tf.cast(tf.squeeze(price_h, axis=1), tf.float32)       # [B]
    var = tf.cast(tf.squeeze(var_h, axis=1), tf.float32)         # [B]

    residual_energy = tf.stop_gradient(tf.reduce_mean(tf.square(y - mu))) + eps
    predicted_energy = tf.reduce_mean(var) + eps
    return tf.square(tf.math.log(residual_energy) - tf.math.log(predicted_energy))


@Losses.register(name="casimir_interference_loss", tags=["calibration", "casimir", "multi_scale", "t_perp"])
def casimir_interference_loss(model, price_h0, price_h1, price_h2,
                               var_h0, var_h1, var_h2):
    """Casimir inter-scale interference: where adjacent horizons disagree in sign, the
    predicted variance must not be small.

    Loss = mean( interf_01 * relu(log v_ref - log avg_var_01)
               + interf_12 * relu(log v_ref - log avg_var_12) )
    with interf_ab = stop_gradient(relu(-s(p_a) * s(p_b))) in [0, 1], s(p) = tanh(p / tau),
    v_ref = 1.0 (unit variance in scaled-delta units), tau = 0.5.

    Fixes relative to the previous relu(-p_a * p_b) / avg_var:
      * the interference is a bounded SOFT-SIGN product, as the docstring always said -
        not a raw product that scaled like |p_a||p_b| (up to 1e4 under the +-100 clip);
      * it is stop_gradient-ed, so the term can no longer be minimised by shrinking the
        price heads toward zero;
      * the variance penalty is a hinge in log space that is zero once avg_var >= v_ref,
        so sigma is lifted up to "uncertain" and never inflated without bound (the old
        1/avg_var had no minimum and drove sigma toward VAR_CAP, where the clip froze it).
    Bounded above by log(v_ref / VAR_FLOOR) ~ 9.2.
    """
    eps = tf.constant(1e-8, dtype=tf.float32)
    tau = tf.constant(0.5, dtype=tf.float32)
    log_v_ref = tf.constant(0.0, dtype=tf.float32)   # log(1.0)

    def _soft_sign(p):
        return tf.tanh(tf.cast(tf.squeeze(p, axis=1), tf.float32) / tau)

    s0, s1, s2 = _soft_sign(price_h0), _soft_sign(price_h1), _soft_sign(price_h2)
    v0 = tf.cast(tf.squeeze(var_h0, axis=1), tf.float32)
    v1 = tf.cast(tf.squeeze(var_h1, axis=1), tf.float32)
    v2 = tf.cast(tf.squeeze(var_h2, axis=1), tf.float32)

    interf_h01 = tf.stop_gradient(tf.nn.relu(-s0 * s1))      # [B], in [0, 1]
    interf_h12 = tf.stop_gradient(tf.nn.relu(-s1 * s2))

    under_h01 = tf.nn.relu(log_v_ref - tf.math.log(0.5 * (v0 + v1) + eps))
    under_h12 = tf.nn.relu(log_v_ref - tf.math.log(0.5 * (v1 + v2) + eps))

    return tf.reduce_mean(interf_h01 * under_h01 + interf_h12 * under_h12)


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
    """Hyper-decoherence coupling: predicted variance should be ORDERED like the window's
    realised volatility across the batch.

    Loss = 1 - Pearson( z(log local_vol), z(log mean_var) ) in [0, 2]

    Fixes relative to the previous -mean(local_vol * sigma):
      * that form was linear and UNBOUNDED BELOW in sigma - a pure bounty on inflating the
        variance head with no coupling to being right about volatility - and it mixed
        input-normaliser units (local_vol) with target-scaler units (sigma);
      * standardising both quantities within the batch makes the term a scale-free
        ordering constraint: it is minimised when high-volatility windows get the highest
        sigma, and it cannot push the overall level of sigma anywhere;
      * local_vol is data, so it is stop_gradient-ed explicitly.
    """
    eps = tf.constant(1e-3, dtype=tf.float32)
    x = tf.cast(x_window, tf.float32)                                  # [B, LOOKBACK]
    local_vol = tf.stop_gradient(tf.math.reduce_std(x, axis=1))         # [B]
    log_vol = tf.math.log(local_vol + eps)

    v0 = tf.cast(tf.squeeze(var_h0, axis=1), tf.float32)
    v1 = tf.cast(tf.squeeze(var_h1, axis=1), tf.float32)
    v2 = tf.cast(tf.squeeze(var_h2, axis=1), tf.float32)
    log_var = tf.math.log((v0 + v1 + v2) / 3.0 + eps)                   # [B]

    def _z(a):
        return (a - tf.reduce_mean(a)) / (tf.math.reduce_std(a) + eps)

    pearson = tf.reduce_mean(_z(log_vol) * _z(log_var))
    return 1.0 - pearson


@Losses.register(name="information_flow_entropy_loss",
                 tags=["regulation", "diversity", "multi_scale", "information"])
def information_flow_entropy_loss(model, price_h0, price_h1, price_h2, rho_max=None):
    """Information Flow Entropy Loss.

    QBOX 'movement = unrolling of transcendental number digits': each horizon is
    meant to reveal ADDITIONAL new information — the next decimal of Ξ.  If
    h0, h1, h2 are simply scaled copies of each other, the network has learned to
    copy rather than integrate multi-scale information.

    Penalises cross-horizon Pearson correlation that exceeds ρ_max:
        Loss = relu(|corr(h0, h1)| - rho_max) + relu(|corr(h1, h2)| - rho_max)   [linear hinge]

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

    # Linear hinge. The squared hinge had a gradient of at most 2*(1-rho_max) = 0.1 inside the
    # band it polices and was bounded by 2*(1-rho_max)^2 = 0.005 by construction, which is why
    # the lambda auto-calibration pinned it at CALIB_LAMBDA_MAX every run (now excluded).
    viol = (tf.nn.relu(tf.abs(r01) - rho_max_c) +
            tf.nn.relu(tf.abs(r12) - rho_max_c))
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
    # Target statistic: stop_gradient, otherwise the term rewards WORSE predictions whenever
    # mean overflow exceeds mean residual (d/d residual of (ov/res - 1)^2 is then negative).
    residual_mag = tf.stop_gradient((tf.abs(y0 - p0) + tf.abs(y1 - p1) + tf.abs(y2 - p2)) / 3.0)  # [B]

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

    # Trend supervision, in consistent scaled-delta units.
    # The local and global trend terms are retired: both reduced algebraically to the
    # point loss (last_close / window-start cancelled out) and neither ever entered
    # `total`. Their LossComponents slots are kept as 0 so positional unpacking in
    # train_step / test_step / the calibration sampler stays valid.
    # The price heads are ALREADY scaled deltas; extended_trend_loss scales the matching
    # past delta into the same units. It must never see last_close (a raw price): that
    # unit mismatch overflowed cosh and produced the NaN gradients that froze training.
    price_h0_s = tf.squeeze(price_h0, axis=1)
    price_h1_s = tf.squeeze(price_h1, axis=1)
    price_h2_s = tf.squeeze(price_h2, axis=1)

    _zero = tf.constant(0.0, dtype=tf.float32)
    local_trend_h0 = local_trend_h1 = local_trend_h2 = _zero
    g0, ext0 = extended_trend_loss(model, x_window, y_true_raw_h0, price_h0_s, extended_trends, last_close, horizon_idx=0)
    g1, ext1 = extended_trend_loss(model, x_window, y_true_raw_h1, price_h1_s, extended_trends, last_close, horizon_idx=1)
    g2, ext2 = extended_trend_loss(model, x_window, y_true_raw_h2, price_h2_s, extended_trends, last_close, horizon_idx=2)

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

    trend_loss_val = extended_trend_h0 + extended_trend_h1 + extended_trend_h2  # coherence enters `total` once, via lambda_coherence_outer

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
    # Floor only: clip_by_value has zero gradient outside its range, so the old upper cap (1e3)
    # permanently detached any variance head that drifted to it. With the physics terms now
    # bounded nothing pushes sigma to +inf; the metric paths keep the cap for gauss_p_up.
    var_h0_c = tf.maximum(var_h0, var_floor)
    var_h1_c = tf.maximum(var_h1, var_floor)
    var_h2_c = tf.maximum(var_h2, var_floor)

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

    # Direction/Gaussian alignment. Skipped entirely unless its outer weight is > 0
    # (LAMBDA_DIR_ALIGN_OUTER defaults to 0): otherwise three BCE evaluations per step
    # for a term that is multiplied by zero. NOTE when re-enabling:
    # tf.keras.losses.binary_crossentropy on 1-D arguments reduces over the batch axis
    # and returns a scalar, so the per-sample masks below are a no-op - compute the
    # per-example BCE by hand before masking.
    if float(getattr(model, 'lambda_dir_align_outer', 0.0)) > 0.0:
        lambda_dir_align = tf.constant(float(getattr(model.config, 'LAMBDA_DIR_ALIGN', 0.0)), dtype=tf.float32)
        align_h0 = tf.keras.losses.binary_crossentropy(gauss_p_up_h0, dir_pred_h0)
        align_h1 = tf.keras.losses.binary_crossentropy(gauss_p_up_h1, dir_pred_h1)
        align_h2 = tf.keras.losses.binary_crossentropy(gauss_p_up_h2, dir_pred_h2)
        align_h0 = tf.reduce_sum(align_h0 * mask_h0) / (tf.reduce_sum(mask_h0) + model.eps)
        align_h1 = tf.reduce_sum(align_h1 * mask_h1) / (tf.reduce_sum(mask_h1) + model.eps)
        align_h2 = tf.reduce_sum(align_h2 * mask_h2) / (tf.reduce_sum(mask_h2) + model.eps)
        dir_align_loss = lambda_dir_align * (align_h0 + align_h1 + align_h2)
    else:
        dir_align_loss = tf.constant(0.0, dtype=tf.float32)

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
    lambda_crps = tf.cast(getattr(model, 'lambda_crps', 0.0), tf.float32)
    crps_h0_val = crps_gaussian_loss(model, y_true_h0, price_h0, var_h0_c)
    crps_h1_val = crps_gaussian_loss(model, y_true_h1, price_h1, var_h1_c)
    crps_h2_val = crps_gaussian_loss(model, y_true_h2, price_h2, var_h2_c)
    total_crps = lambda_crps * (crps_h0_val + crps_h1_val + crps_h2_val)
    total_crps = tf.where(tf.math.is_finite(total_crps), total_crps, tf.constant(0.0, dtype=tf.float32))

    # === SOFT-ECE LOSSES (differentiable Expected Calibration Error) ===
    # Directly minimizes direction-head calibration error w.r.t. realized outcomes.
    # Complements dir-align (which aligns dir_head to Gaussian-implied P(up)) by also
    # aligning to actual labels. Controlled by lambda_soft_ece (default 0).
    lambda_soft_ece = tf.cast(getattr(model, 'lambda_soft_ece', 0.0), tf.float32)
    soft_ece_h0_val = soft_ece_loss(model, true_dir_h0, dir_pred_h0, mask_h0)
    soft_ece_h1_val = soft_ece_loss(model, true_dir_h1, dir_pred_h1, mask_h1)
    soft_ece_h2_val = soft_ece_loss(model, true_dir_h2, dir_pred_h2, mask_h2)
    total_soft_ece = lambda_soft_ece * (soft_ece_h0_val + soft_ece_h1_val + soft_ece_h2_val)
    total_soft_ece = tf.where(tf.math.is_finite(total_soft_ece), total_soft_ece, tf.constant(0.0, dtype=tf.float32))

    # === T_⊥ / QBOX LOSSES ===
    # T_⊥ calibration: predicted σ tracks empirical residual std (T_⊥ is what we can't explain)
    lambda_t_perp = tf.cast(getattr(model, 'lambda_t_perp', 0.0), tf.float32)
    t_perp_h0_val = t_perp_calibration_loss(model, y_true_h0, price_h0, var_h0_c)
    t_perp_h1_val = t_perp_calibration_loss(model, y_true_h1, price_h1, var_h1_c)
    t_perp_h2_val = t_perp_calibration_loss(model, y_true_h2, price_h2, var_h2_c)
    total_t_perp = lambda_t_perp * (t_perp_h0_val + t_perp_h1_val + t_perp_h2_val)
    total_t_perp = tf.where(tf.math.is_finite(total_t_perp), total_t_perp, tf.constant(0.0, dtype=tf.float32))

    # Casimir: destructive cross-horizon interference → T_⊥ (σ) must be high
    lambda_casimir = tf.cast(getattr(model, 'lambda_casimir', 0.0), tf.float32)
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
    lambda_hd = tf.cast(getattr(model, 'lambda_hd', 0.0), tf.float32)
    hd_val = lambda_hd * hyper_decoherence_coupling_loss(
        model, x_window, var_h0_c, var_h1_c, var_h2_c)
    hd_val = tf.where(tf.math.is_finite(hd_val), hd_val, tf.constant(0.0, dtype=tf.float32))

    # Information flow entropy: each horizon must carry non-redundant information
    lambda_ife = tf.cast(getattr(model, 'lambda_ife', 0.0), tf.float32)
    ife_val = lambda_ife * information_flow_entropy_loss(model, price_h0, price_h1, price_h2)
    ife_val = tf.where(tf.math.is_finite(ife_val), ife_val, tf.constant(0.0, dtype=tf.float32))

    # Vacuum overflow T_⊥ precision: overflow tracks prediction residual magnitude
    # Active only when lambda_vac_overflow > 0 AND vacuum_overflow tensor is provided.
    lambda_vac_overflow = tf.cast(getattr(model, 'lambda_vac_overflow', 0.0), tf.float32)
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

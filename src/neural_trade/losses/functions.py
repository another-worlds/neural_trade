"""Loss functions of the training objective (moved from the root ``losses.py`` in Phase B3).

Every function registers itself in :class:`neural_trade.losses.registry.Losses`:
component losses (first parameter ``model``) with ``Losses.register`` and the full
training objective ``custom_loss`` with ``Losses.register_objective``. The registry class
lives in ``neural_trade/losses/registry.py`` (moved from ``registries/losses.py`` in NT-027,
which now re-exports it); ``LossComponents`` in ``neural_trade/core/outputs.py``.
"""
from __future__ import annotations

import math

import tensorflow as tf

from neural_trade.core.horizons import (REFERENCE_N_HORIZONS, horizon_keys, loss_components_type, n_horizons,
                                        neighbour_pairs, pair_sum_scale, point_weight_role, primary_index)
from neural_trade.core.outputs import LossComponents  # noqa: F401  (re-exported: the N = 3 class)
from neural_trade.losses.registry import Losses
from neural_trade.metrics.tf_direction import direction_labels_tf, gaussian_up_prob_given_move  # noqa: F401
from neural_trade.utils.math import log_ndtr  # noqa: F401  (re-exported for callers of this module)

# Old private name of the TF labelling rule, kept for callers of this module.
_compute_direction_labels_and_masks_tf = direction_labels_tf

#: Every non-finite guard in this module, wired through :func:`_finite_or_zero` (NT-036, D-026,
#: QA repair round 1): the 15 addends of ``total`` (``custom_loss``) and ``pnl_utility``'s own
#: extra total, plus every intermediate site upstream of them (point_huber/local_trend/
#: extended_trend's own guards, the per-horizon head sanitisation, the coherence sub-terms, the
#: per-horizon dir/nll pre-guards, and pnl_utility's internal window/sigma/per-horizon guards) -
#: so ``STRICT_LOSS_MASKS`` genuinely lifts every mask, not only the ones ``total`` sees directly.
#: Each name is also a ``model._mask_counters`` key (a plain ``_Accum("sum")``,
#: ``training/custom_model.py`` - not a ``tf.keras.metrics.Metric``, see ``_Accum``'s docstring
#: for why) and shows up in ``metrics.jsonl`` as ``masked_<name>`` / ``val_masked_<name>``.
def mask_term_names(n_horizons: int = REFERENCE_N_HORIZONS) -> tuple:
    """Every mask-counter name for a model with ``n_horizons`` horizons (NT-042); the reference
    N = 3 gives exactly the tuple below, in the same order."""
    hs = horizon_keys(n_horizons)
    return (
        "point_loss", "dir_loss", "dir_align_loss", "nll_loss", "vol_loss", "crps_loss",
        "soft_ece_loss", "t_perp_loss", "casimir_loss", "vac_loss", "hd_loss", "ife_loss",
        "vac_overflow_loss", "coherence_penalty", "total_loss",
        # QA repair round 1 (NT-036/037 fix 1): every remaining tf.where(is_finite, x, 0) site in this
        # module, so STRICT_LOSS_MASKS truly lifts EVERY mask, not only the ones that feed `total`
        # directly, and every one of them counts its own steps.
        "point_huber_raw", "local_trend_raw", "extended_trend_raw",
        *(f"head_price_{h}" for h in hs),
        *(f"head_dir_{h}" for h in hs),
        *(f"head_var_{h}" for h in hs),
        *(f"point_loss_{h}" for h in hs),
        # dir_disagree_loss/target_smoothness_loss (NT-096): kept unconditionally, even though
        # Config.COHERENCE_MAGNITUDE_ONLY=True skips computing both - this tuple is built once,
        # independently of that config value, so the names must stay static; when skipped their
        # counters simply read 0 rather than being dropped.
        "dir_disagree_loss", "magnitude_loss", "target_smoothness_loss",
        *(f"dir_loss_{h}" for h in hs),
        *(f"nll_{h}" for h in hs),
        # pnl_utility (NT-087) only:
        "pnl_raw_window", "pnl_step_ret", "pnl_step_sd", "pnl_sigma", "pnl_loss_h", "pnl_term",
        "pnl_total",
    )


#: The reference (N = 3) mask-counter names; ``mask_term_names(n)`` for another horizon count.
MASK_TERM_NAMES = mask_term_names(REFERENCE_N_HORIZONS)


def _sum_in_order(terms):
    """``((t0 + t1) + t2) + ...``: the left-to-right sum the hand-written three-horizon formulas
    used, so N = 3 builds the same ops in the same order (NT-042, golden run). A generator is
    consumed lazily: each term is built just before it is added, as in ``a + b + c``."""
    it = iter(terms)
    acc = next(it)
    for t in it:
        acc = acc + t
    return acc


def _pair_scaled(value, n):
    """A pairwise term's sum over the neighbouring pairs, times ``pair_sum_scale(n)`` (2 / (n - 1):
    the per-pair mean times the reference pair count). At N = 3 the factor is exactly 1 and no op
    is added (D-022, NT-042)."""
    scale = pair_sum_scale(n)
    return value if scale == 1.0 else value * tf.constant(scale, dtype=tf.float32)


def _horizon_count(model, default=None) -> int:
    """The model's horizon count: ``len(model.config.HORIZON_STEPS)``, else ``default``, else 3."""
    if default is not None:
        return int(default)
    return n_horizons(getattr(model, "config", None))


def _finite_or_zero(model, x, term, fallback=0.0):
    """The non-finite guard for one addend of ``total`` (D-026, NT-036).

    Replaces the inline ``tf.where(tf.math.is_finite(x), x, fallback)`` this loss module used to
    repeat at every term (``fallback`` defaults to 0, matching most of them; the per-horizon head
    sanitisation sites pass their ORIGINAL fallback - 0.5 for a direction head, 1.0 for a variance
    head. QA repair round 2 caught hard-coding 0 there: it left every all-finite run, including
    scripts/golden_run.py's, bit-for-bit, but changed a real run's trajectory the moment NT-047's
    larger indicator set produced its first transient non-finite value): same result by default,
    plus two things the inline form could not do:

    * it counts the step into ``model._mask_counters[term]`` (a plain ``_Accum("sum")``, never a
      ``tf.keras.metrics.Metric`` - see that class's docstring for why) whenever the mask actually
      fired, so an unstable run can be attributed to its term instead of only seeing ``total`` (or
      the gradient) go non-finite;
    * in strict mode (``Config.STRICT_LOSS_MASKS``) it does NOT mask: ``x`` passes through
      unchanged, so a non-finite term makes ``total`` non-finite too and ``train_step``'s
      finite-gradient guard fires and counts the step, instead of the term being silently
      replaced by ``fallback`` before it ever reaches ``total``.

    Strict mode defaults off (``STRICT_LOSS_MASKS: False``): with it off this is bit-for-bit the
    old inline guard (the counting is a pure side read of ``x``, so it changes no numbers).

    ``STRICT_LOSS_MASKS`` is read as a plain Python ``bool`` off ``model.config`` (a trace-time
    branch, NOT a graph-level ``tf.cond`` on a live ``tf.Variable``): a real repair-round-1
    regression, found only after merging NT-047's larger default model, showed that wrapping
    EVERY one of this module's ~46 guard sites (up from the original 15) in a ``tf.cond`` - even
    one whose predicate is a constant-``False`` Variable - measurably changes a real training
    run's trajectory (``scripts/golden_run.py verify`` failed on ``period/*``, ``coverage/*`` and
    ``pred/*``; calibration itself, which never calls train_step, was unaffected). A Python branch
    costs nothing when strict is off (confirmed bit-for-bit) and means toggling
    ``Config.STRICT_LOSS_MASKS`` on an already-traced model needs a fresh model - matching every
    other purely-structural Config switch in this codebase (e.g. ``LAMBDA_DIR_ALIGN_OUTER`` in
    ``custom_loss``); every caller (the stability tests, ``CustomTrainModel.strict_loss_masks``)
    already builds a fresh model per value.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    finite = tf.math.is_finite(x)
    all_finite = tf.reduce_all(finite) if x.shape.rank else finite
    counters = getattr(model, "_mask_counters", None)
    if counters is not None and term in counters:
        counters[term].update_state(tf.cast(tf.logical_not(all_finite), tf.float32))
    if bool(getattr(getattr(model, "config", None), "STRICT_LOSS_MASKS", False)):
        return x
    return tf.where(finite, x, tf.ones_like(x) * fallback)


#: Added to the variance before its square root in :func:`_safe_std` (NT-096, D-045,
#: ``docs/research/2026-09-30-math-report/A_losses.md`` section 3). ``tf.math.reduce_std`` is
#: ``sqrt(reduce_variance(x))`` with no guard: ``d/dvar sqrt(var) = 1 / (2*sqrt(var))`` is
#: infinite at ``var = 0``, so a batch (or, for an axis-1 reduction, one example's horizons)
#: that is exactly constant gives an infinite upstream gradient - a NaN training step hidden by
#: the ``_finite_or_zero`` guards above and never actually observed, because a real batch is
#: never perfectly constant. 1e-12 is far below float32's representable precision at the
#: variances these losses see in practice (>= ~1e-4: ``x + 1e-12 == x`` there in float32), so it
#: changes no normal run's forward value or gradient (``scripts/golden_run.py verify``); it only
#: bounds the gradient at the degenerate point the fix targets.
_STD_EPS = tf.constant(1e-12, dtype=tf.float32)


def _safe_std(x, axis=None):
    """A gradient-safe ``tf.math.reduce_std`` (see ``_STD_EPS``).

    Every call site below used to add its own small eps AFTER the square root (to keep a
    denominator off exactly zero); that does nothing for the gradient, since the derivative of
    ``sqrt`` is taken at the UNGUARDED variance. The guard has to be on the variance, before the
    ``sqrt``.
    """
    variance = tf.math.reduce_variance(tf.cast(x, tf.float32), axis=axis)
    return tf.sqrt(variance + _STD_EPS)


def _std(model, x, axis=None):
    """``tf.math.reduce_std``, or :func:`_safe_std` when ``Config.LOSS_SAFE_STD`` is set
    (NT-096 repair round 1, D-045).

    A trace-time Python branch, NOT a ``tf.cond`` - same reasoning as ``STRICT_LOSS_MASKS``
    (``_finite_or_zero``'s docstring): wrapping these call sites in a ``tf.cond``, even one whose
    predicate never changes, was found (via ``scripts/golden_run.py verify``) to change a real
    run's trajectory through CPU floating-point reduction-order sensitivity, not through any
    value or gradient difference. ``False`` (the default) is bit-for-bit today's graph - no
    ``reduce_variance``/``sqrt(+eps)`` node is even built. ``True`` (``Config.LOSS_SAFE_STD``)
    swaps in the gradient-safe form; see its field docstring for which terms this actually
    protects against an observable NaN versus an unbounded-but-finite gradient.
    """
    if bool(getattr(getattr(model, "config", None), "LOSS_SAFE_STD", False)):
        return _safe_std(x, axis=axis)
    return tf.math.reduce_std(x, axis=axis)


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


@Losses.register(name="binary_cross_entropy_loss", tags=["classification", "direction", "proper"])
def binary_cross_entropy_loss(model, true_labels, probs, reduce=True):
    """Binary cross-entropy on probabilities: the direction heads' default objective.

    A strictly proper scoring rule - its expected value over y ~ Bernoulli(q) is minimised
    exactly at p = q - so a head that finds a weak signal is rewarded for reporting it.
    The previous default, 0.5 focal + 0.5 dice, is not: dice's expected loss has an interior
    MAXIMUM, so its optimum is always p = 0 or p = 1 (a constant extreme when samples look
    alike), and focal (BCE minus an entropy bonus) rewards staying near 0.5. Together they
    held the direction heads near a constant. Kept as Config.DIRECTION_LOSS = "focal_dice".
    """
    y = tf.cast(true_labels, tf.float32)
    p = tf.clip_by_value(tf.cast(probs, tf.float32), 1e-7, 1.0 - 1e-7)
    per_example = -(y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
    return tf.reduce_mean(per_example) if reduce else per_example


@Losses.register(name="combined_direction_loss", tags=["classification", "composite", "direction"])
def combined_direction_loss(model, true_labels, logits, alpha=None, gamma=None,
                             focal_weight=0.5, dice_weight=0.5, reduce=True):
    focal = focal_loss(model, true_labels, logits, alpha=alpha, gamma=gamma, reduce=reduce)
    dice = dice_loss(model, true_labels, logits, reduce=reduce)
    return focal_weight * focal + dice_weight * dice


# Not a loss (a label-balance helper), so it is not registered.
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
    result = _finite_or_zero(model, result, "point_huber_raw")
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
    result = _finite_or_zero(model, result, "local_trend_raw")
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
    ext = _finite_or_zero(model, ext, "extended_trend_raw")
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
    # All bins at once ([B, n_bins]): the per-bin Python loop issued ~30 small kernels per bin and
    # was ~3/4 of the whole loss's forward+backward time.
    centers = (tf.range(n_bins, dtype=tf.float32) + 0.5) / float(n_bins)
    w = tf.exp(-tf.square(p[:, None] - centers[None, :]) / h2) * m[:, None]      # [B, n_bins]
    w_sum = tf.reduce_sum(w, axis=0)                                              # [n_bins]
    soft_acc = tf.reduce_sum(w * y[:, None], axis=0) / (w_sum + 1e-8)
    soft_conf = tf.reduce_sum(w * p[:, None], axis=0) / (w_sum + 1e-8)
    return tf.reduce_sum(w_sum / total_eff * tf.abs(soft_acc - soft_conf))


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
def casimir_interference_loss(model, *heads):
    """Casimir inter-scale interference: where adjacent horizons disagree in sign, the
    predicted variance must not be small.

    ``heads`` are the N price heads followed by the N variance heads (``price_h0, ...,
    price_h{N-1}, var_h0, ..., var_h{N-1}``; the reference call is
    ``casimir_interference_loss(model, price_h0, price_h1, price_h2, var_h0, var_h1, var_h2)``).

    Loss = s_N * mean( sum over neighbouring pairs (a, b) of interf_ab * relu(log v_ref - log avg_var_ab) )
    with interf_ab = stop_gradient(relu(-s(p_a) * s(p_b))) in [0, 1], s(p) = tanh(p / tau),
    v_ref = 1.0 (unit variance in scaled-delta units), tau = 0.5, and s_N = 2 / (N - 1) (the
    per-pair mean times the reference pair count, so N = 3 is exactly the sum of the (h0, h1)
    and (h1, h2) terms; D-022, NT-042).

    Fixes relative to the previous relu(-p_a * p_b) / avg_var:
      * the interference is a bounded SOFT-SIGN product, as the docstring always said -
        not a raw product that scaled like |p_a||p_b| (up to 1e4 under the +-100 clip);
      * it is stop_gradient-ed, so the term can no longer be minimised by shrinking the
        price heads toward zero;
      * the variance penalty is a hinge in log space that is zero once avg_var >= v_ref,
        so sigma is lifted up to "uncertain" and never inflated without bound (the old
        1/avg_var had no minimum and drove sigma toward VAR_CAP, where the clip froze it).
    Each pair's hinge is bounded above by log(v_ref / VAR_FLOOR) ~ 9.2.
    """
    if len(heads) % 2 or len(heads) < 4:
        raise ValueError(f"casimir_interference_loss: expected N price heads then N variance heads (N >= 2), "
                         f"got {len(heads)} tensors")
    n = len(heads) // 2
    prices, variances = heads[:n], heads[n:]
    eps = tf.constant(1e-8, dtype=tf.float32)
    tau = tf.constant(0.5, dtype=tf.float32)
    log_v_ref = tf.constant(0.0, dtype=tf.float32)   # log(1.0)

    def _soft_sign(p):
        return tf.tanh(tf.cast(tf.squeeze(p, axis=1), tf.float32) / tau)

    s = [_soft_sign(p) for p in prices]
    v = [tf.cast(tf.squeeze(var, axis=1), tf.float32) for var in variances]
    pairs = neighbour_pairs(n)

    interf = [tf.stop_gradient(tf.nn.relu(-s[a] * s[b])) for a, b in pairs]      # [B], in [0, 1]
    under = [tf.nn.relu(log_v_ref - tf.math.log(0.5 * (v[a] + v[b]) + eps)) for a, b in pairs]

    return _pair_scaled(tf.reduce_mean(_sum_in_order(i * u for i, u in zip(interf, under))), n)


@Losses.register(name="vacuum_bandwidth_loss", tags=["regulation", "vacuum", "t_perp", "self_limiting"])
def vacuum_bandwidth_loss(model, *prices, lambda_vac=None):
    """Vacuum Bandwidth Loss — Λ_vac self-limiting term.

    QBOX vacuum: the vacuum has limited bandwidth Λ_vac; when local energy exceeds it,
    the excess is automatically shunted to T_⊥ producing supra-stable plateaus.
    Transposed to trading: the cross-horizon spread of predictions (h0–h2) should not
    exceed Λ_vac.  When it does, the model is over-extrapolating across timescales
    (the market's 'vacuum' bandwidth is saturated → uncertainty should rise).

    Only penalises violations ABOVE Λ_vac (relu clamp = supra-stable self-limiting).

    Loss = mean(relu(std(p0, ..., p{N-1}) - Λ_vac)), over the N price heads passed positionally.
    N rule (NT-042): the cross-horizon standard deviation is already a per-sample dispersion over
    however many horizons there are, so it is not rescaled.
    """
    if lambda_vac is None:
        lambda_vac = tf.constant(float(getattr(getattr(model, 'config', None), 'LAMBDA_VAC', 0.0)),
                                 dtype=tf.float32)
    lv = tf.cast(lambda_vac, tf.float32)

    ps = [tf.cast(tf.squeeze(p, axis=1), tf.float32) for p in prices]    # N x [B]

    stacked = tf.stack(ps, axis=1)                             # [B, N]
    cross_std = _std(model, stacked, axis=1)                   # [B] (NT-096: gradient-safe when LOSS_SAFE_STD)
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
def hyper_decoherence_coupling_loss(model, x_window, *variances):
    """Hyper-decoherence coupling: predicted variance should be ORDERED like the window's
    realised volatility across the batch.

    Loss = 1 - Pearson( z(log local_vol), z(log mean_var) ) in [0, 2]

    ``mean_var`` is the mean of the N variance heads passed positionally (N rule, NT-042: a mean
    over the horizons, so its scale does not depend on N).

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
    local_vol = tf.stop_gradient(_std(model, x, axis=1))                # [B]
    log_vol = tf.math.log(local_vol + eps)

    vs = [tf.cast(tf.squeeze(v, axis=1), tf.float32) for v in variances]
    log_var = tf.math.log(_sum_in_order(vs) / float(len(vs)) + eps)     # [B]

    def _z(a):
        return (a - tf.reduce_mean(a)) / (_std(model, a) + eps)  # NT-096: eps guards the sqrt when LOSS_SAFE_STD

    pearson = tf.reduce_mean(_z(log_vol) * _z(log_var))
    return 1.0 - pearson


@Losses.register(name="information_flow_entropy_loss",
                 tags=["regulation", "diversity", "multi_scale", "information"])
def information_flow_entropy_loss(model, *prices, rho_max=None):
    """Information Flow Entropy Loss.

    QBOX 'movement = unrolling of transcendental number digits': each horizon is
    meant to reveal ADDITIONAL new information — the next decimal of Ξ.  If
    h0, h1, h2 are simply scaled copies of each other, the network has learned to
    copy rather than integrate multi-scale information.

    Penalises cross-horizon Pearson correlation that exceeds ρ_max, over every neighbouring pair
    of the N price heads passed positionally:
        Loss = s_N * sum over (a, b) of relu(|corr(h_a, h_b)| - rho_max)   [linear hinge]
    with s_N = 2 / (N - 1) (the per-pair mean times the reference pair count: N = 3 is exactly
    relu(|corr(h0, h1)| - rho_max) + relu(|corr(h1, h2)| - rho_max); D-022, NT-042).

    This forces horizons to carry non-redundant information (diverse views of the
    same market state), analogous to ensuring each digit of Ξ is actually new.
    """
    if rho_max is None:
        rho_max = float(getattr(getattr(model, 'config', None), 'RHO_MAX', 0.95))
    rho_max_c = tf.constant(float(rho_max), dtype=tf.float32)
    eps = tf.constant(1e-8, dtype=tf.float32)

    ps = [tf.cast(tf.squeeze(p, axis=1), tf.float32) for p in prices]    # N x [B]

    def _pearson(a, b):
        ma = a - tf.reduce_mean(a)
        mb = b - tf.reduce_mean(b)
        cov = tf.reduce_mean(ma * mb)
        std_a = _std(model, a) + eps  # NT-096: eps guards the sqrt when LOSS_SAFE_STD
        std_b = _std(model, b) + eps
        return cov / (std_a * std_b)

    pairs = neighbour_pairs(len(ps))
    rs = [_pearson(ps[a], ps[b]) for a, b in pairs]

    # Linear hinge. The squared hinge had a gradient of at most 2*(1-rho_max) = 0.1 inside the
    # band it polices and was bounded by 2*(1-rho_max)^2 = 0.005 by construction, which is why
    # the lambda auto-calibration pinned it at CALIB_LAMBDA_MAX every run (now excluded).
    viol = _sum_in_order([tf.nn.relu(tf.abs(r) - rho_max_c) for r in rs])
    return _pair_scaled(viol, len(ps))


@Losses.register(name="vacuum_overflow_t_perp_loss",
                 tags=["calibration", "t_perp", "vacuum", "overflow", "precision"])
def vacuum_overflow_t_perp_loss(model, vacuum_overflow, *targets_and_prices):
    """Vacuum Overflow T_⊥ Precision Loss.

    Anchors the observable vacuum overflow signal to the actual prediction residual
    magnitude, making the overflow a precise, calibrated measure of T_⊥ intensity.

    Physics framing: the overflow (energy exceeding the vacuum ceiling E_max) is the
    portion of hidden-dimension energy that the network's visible subspace cannot absorb.
    This MUST equal the unexplained residual in real-space predictions, otherwise T_⊥ is
    either under-reported (overflow too small) or a hallucination (overflow too large).

    ``targets_and_prices`` is ``y_true_h0, price_h0, y_true_h1, price_h1, ...`` for the N
    horizons (the reference call passes three pairs).

    Loss = (mean_overflow - mean_residual_mag)² / (mean_residual_mag² + ε)

    Normalised by residual² so the loss is scale-invariant across training stages. N rule
    (NT-042): the residual magnitude is the mean over the N horizons, so it does not grow with N.
    """
    if len(targets_and_prices) % 2 or not targets_and_prices:
        raise ValueError("vacuum_overflow_t_perp_loss: expected (y_true_h, price_h) pairs, one per horizon")
    eps = tf.constant(1e-8, dtype=tf.float32)

    ov = tf.cast(tf.squeeze(vacuum_overflow, axis=1), tf.float32)   # [B]

    ys, ps = [], []
    for y_h, p_h in zip(targets_and_prices[0::2], targets_and_prices[1::2]):
        ys.append(tf.cast(tf.squeeze(y_h, axis=1), tf.float32))      # [B]
        ps.append(tf.cast(tf.squeeze(p_h, axis=1), tf.float32))

    # Per-sample mean absolute residual across all horizons
    # Target statistic: stop_gradient, otherwise the term rewards WORSE predictions whenever
    # mean overflow exceeds mean residual (d/d residual of (ov/res - 1)^2 is then negative).
    residual_mag = tf.stop_gradient(
        _sum_in_order(tf.abs(y - p) for y, p in zip(ys, ps)) / float(len(ys)))  # [B]

    mean_ov  = tf.reduce_mean(ov)           # scalar
    mean_res = tf.reduce_mean(residual_mag) # scalar

    # Scale-invariant alignment loss
    return tf.square(mean_ov - mean_res) / (tf.square(mean_res) + eps)


@Losses.register_objective(name="custom_loss", tags=["composite", "default", "multi_output"])
def custom_loss(model, x_window, y_true, y_pred, last_close, extended_trends,
               vacuum_overflow=None):
    """The training objective over N horizons (NT-042, D-022).

    ``y_pred`` is the 3N price/direction/variance heads in ``PredictiveOutputs`` order (without
    ``vacuum_overflow``, passed separately); ``y_true`` is ``[B, N]`` scaled deltas. Every
    per-horizon term runs once per horizon; the pairwise terms (Casimir, IFE, the coherence
    magnitude ordering) run over every neighbouring pair and are scaled so N = 3 is today's sum
    (``core.horizons.pair_sum_scale``); the point-loss weight of horizon i is LAMBDA_SHORT for h0,
    LAMBDA_LONG for the last horizon and LAMBDA_POINT in between (``core.horizons.point_weight_role``);
    the variance penalty reads the primary horizon (``core.horizons.primary_index``: h1 at N = 3).
    At N = 3 the graph is the hand-written three-horizon one, op for op (golden run).
    Returns ``core.horizons.loss_components_type(N)`` (``LossComponents`` at N = 3).
    """
    y_pred = list(y_pred)
    if not y_pred or len(y_pred) % 3:
        raise ValueError(f"custom_loss: y_pred must hold 3 heads per horizon, got {len(y_pred)} tensors")
    n = len(y_pred) // 3
    hs = range(n)
    keys = horizon_keys(n)

    # Convert to expected types and shapes exactly as original model implementation
    y_true = tf.cast(y_true, tf.float32)
    y_true_raw = y_true * model.pred_scale + model.pred_mean
    last_close_squeeze = tf.squeeze(last_close, axis=1)

    y_true_h = [y_true[:, i:i + 1] for i in hs]
    y_true_raw_h = [y_true_raw[:, i] for i in hs]

    deadband_bps = tf.cast(getattr(model.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
    labels = _compute_direction_labels_and_masks_tf(
        y_true_raw, last_close_squeeze, deadband_bps, model.eps, n_horizons=n
    )
    mask_h, true_dir_h = list(labels[:n]), list(labels[n:])

    # The scaled deadband is still needed later for dir_align and gauss_p_up calculations.
    deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

    # Sanitize heads (belt-and-suspenders with output clips in build_model).
    # Ensures no NaN/Inf reaches *any* loss term (point, dir, nll, casimir, t_perp, ife, vacuum, hd, align, etc.).
    # Prevents 0*nan pollution in total even for "inactive" (lambda=0) terms, and keeps all components finite.
    price_h, dir_h, var_h = [], [], []
    for i, h in enumerate(keys):
        price_h.append(_finite_or_zero(model, y_pred[3 * i], f"head_price_{h}"))
        dir_h.append(_finite_or_zero(model, y_pred[3 * i + 1], f"head_dir_{h}", fallback=0.5))
        var_h.append(_finite_or_zero(model, y_pred[3 * i + 2], f"head_var_{h}", fallback=1.0))

    # Point-loss weight per horizon: LAMBDA_SHORT (h0), LAMBDA_POINT (between), LAMBDA_LONG (last).
    point_loss_h = [getattr(model, f"lambda_{point_weight_role(i, n)}") * point_huber(model, y_true_h[i], price_h[i])
                    for i in hs]
    point_loss_val = _sum_in_order(point_loss_h)
    point_loss_val = _finite_or_zero(model, point_loss_val, "point_loss")
    point_loss_h = [_finite_or_zero(model, point_loss_h[i], f"point_loss_{h}") for i, h in enumerate(keys)]

    # Trend supervision, in consistent scaled-delta units.
    # The local and global trend terms are retired: both reduced algebraically to the
    # point loss (last_close / window-start cancelled out) and neither ever entered
    # `total`. Their LossComponents slots are kept as 0 so positional unpacking in
    # train_step / test_step / the calibration sampler stays valid.
    # The price heads are ALREADY scaled deltas; extended_trend_loss scales the matching
    # past delta into the same units. It must never see last_close (a raw price): that
    # unit mismatch overflowed cosh and produced the NaN gradients that froze training.
    price_h_s = [tf.squeeze(price_h[i], axis=1) for i in hs]

    _zero = tf.constant(0.0, dtype=tf.float32)
    local_trend_h = [_zero] * n
    global_ext = [extended_trend_loss(model, x_window, y_true_raw_h[i], price_h_s[i], extended_trends, last_close,
                                      horizon_idx=i) for i in hs]

    # For backward compatibility of the "trend_loss_val" formula we keep the previous
    # structure (extended components + coherence). The globals are available in the
    # returned components for richer logging.
    trend_loss_h = [ext for _g, ext in global_ext]

    # Coherence averages three sub-terms. Two of them have ZERO GRADIENT everywhere
    # (NT-096, D-045, A_losses.md section 8):
    #   * dir_disagree_loss compares tf.sign(price_h*) across horizons with tf.equal - both
    #     non-differentiable (tf.sign's gradient is 0 a.e., tf.equal/tf.cast have none at all),
    #     so it can never receive or pass a gradient to the price heads. Its VALUE, though, is
    #     NOT a per-batch constant: it depends on the price heads' sign agreement, which moves
    #     as the weights train (QA of repair round 1: its own val_loss contribution moved from
    #     0.2197 to 0.1793 between the two golden-run epochs - a real run's epoch-to-epoch
    #     val_loss gaps near the best epoch are about 0.06, so dropping this term CAN change
    #     which epoch is served, D-011, and ReduceLROnPlateau/EarlyStopping's state).
    #   * target_smoothness_loss reads only tf.sign(y_true_raw_h*) - the LABELS, which are a
    #     true per-batch constant as far as the model's trainable variables are concerned, so
    #     it is both zero-gradient and value-constant across weight updates (within a batch).
    # Removing dir_disagree_loss is therefore a value-moving change (even though it moves no
    # gradient): D-045 ships it behind Config.COHERENCE_MAGNITUDE_ONLY (default False = today's
    # exact three-term graph, bit-for-bit), chosen at trace time in Python, not a tf.cond (same
    # reasoning as LOSS_SAFE_STD's docstring). True drops both dead terms and keeps only
    # magnitude_loss (the ordering hinge on |price_h0| <= |price_h1| <= ...), still /3.0
    # (the old three-term average's weight on it), so that path's own gradient is unchanged too
    # (tests/test_custom_loss.py::test_coherence_dead_parts_removal_does_not_change_any_gradient
    # compares both switch settings' gradients on a fixed batch). MASK_TERM_NAMES keeps
    # "dir_disagree_loss"/"target_smoothness_loss" unconditionally (the counters dict is built
    # once, independently of this config value); when True they are simply never touched, so
    # they read 0 rather than being dropped.
    # N rules (NT-042): the magnitude ordering is a pairwise hinge over the neighbouring pairs,
    # scaled by pair_sum_scale (N = 3: today's sum of two hinges); dir_disagree is 1 - the mean
    # sign agreement over the pairs (N = 3: today's /2); target_smoothness is the mean, over the
    # interior horizons h_1..h_{N-2}, of the label-sign xor with both neighbours (N = 3: today's
    # single h1 term; N = 2: no interior horizon, so 0).
    pairs = neighbour_pairs(n)
    abs_pred_h = [tf.abs(price_h[i]) for i in hs]

    magnitude_violations = [tf.nn.relu(abs_pred_h[a] - abs_pred_h[b]) for a, b in pairs]
    magnitude_loss = _pair_scaled(tf.reduce_mean(_sum_in_order(magnitude_violations)), n)
    magnitude_loss = _finite_or_zero(model, magnitude_loss, "magnitude_loss")

    if bool(getattr(getattr(model, "config", None), "COHERENCE_MAGNITUDE_ONLY", False)):
        coherence_penalty = magnitude_loss / 3.0
    else:
        sign_pred_h = [tf.sign(price_h[i]) for i in hs]

        dir_agree = [tf.reduce_mean(tf.cast(tf.equal(sign_pred_h[a], sign_pred_h[b]), tf.float32))
                     for a, b in pairs]
        dir_disagree_loss = 1.0 - _sum_in_order(dir_agree) / float(len(pairs))
        dir_disagree_loss = _finite_or_zero(model, dir_disagree_loss, "dir_disagree_loss")

        sign_target_h = [tf.sign(y_true_raw_h[i]) for i in hs]
        interior = range(1, n - 1)
        smooth_terms = [
            tf.reduce_mean(
                tf.cast(tf.math.logical_xor(sign_target_h[i] == sign_target_h[i - 1],
                                             sign_target_h[i] == sign_target_h[i + 1]), tf.float32)
            )
            for i in interior
        ]
        if not smooth_terms:
            target_smoothness_loss = tf.constant(0.0, dtype=tf.float32)
        elif len(smooth_terms) == 1:
            target_smoothness_loss = smooth_terms[0]
        else:
            target_smoothness_loss = _sum_in_order(smooth_terms) / float(len(smooth_terms))
        target_smoothness_loss = _finite_or_zero(model, target_smoothness_loss, "target_smoothness_loss")

        coherence_penalty = (dir_disagree_loss + magnitude_loss + target_smoothness_loss) / 3.0
    coherence_penalty = _finite_or_zero(model, coherence_penalty, "coherence_penalty")

    # Assign from the registered calls above (scaled correctly, using fixed delta math).
    # Multiply the extended components by the per-horizon lambda here for consistency
    # with prior behavior (the registered fn returns the raw component loss).
    global_trend_h, extended_trend_h = [], []
    for i in hs:
        global_trend_h.append(global_ext[i][0])
        extended_trend_h.append(model.lambda_extended_trend * trend_loss_h[i])

    trend_loss_val = _sum_in_order(extended_trend_h)  # coherence enters `total` once, via lambda_coherence_outer

    dir_pred_h = [tf.squeeze(dir_h[i], axis=1) for i in hs]

    if str(getattr(model.config, 'DIRECTION_LOSS', 'bce')) == 'focal_dice':  # legacy, improper
        alpha_h = [compute_dynamic_alpha(true_dir_h[i]) for i in hs]
        per_ex_h = [combined_direction_loss(model, true_dir_h[i], dir_pred_h[i], alpha=alpha_h[i],
                                            focal_weight=0.5, dice_weight=0.5, reduce=False) for i in hs]
    else:
        per_ex_h = [binary_cross_entropy_loss(model, true_dir_h[i], dir_pred_h[i], reduce=False) for i in hs]

    dir_loss_h = [tf.reduce_sum(per_ex_h[i] * mask_h[i]) / (tf.reduce_sum(mask_h[i]) + model.eps) for i in hs]
    dir_loss_h = [_finite_or_zero(model, dir_loss_h[i], f"dir_loss_{h}") for i, h in enumerate(keys)]
    total_dir_loss = model.lambda_dir * _sum_in_order(dir_loss_h)
    total_dir_loss = _finite_or_zero(model, total_dir_loss, "dir_loss")

    var_floor = tf.cast(getattr(model.config, 'VAR_FLOOR', 1e-4), tf.float32)
    var_cap = tf.cast(getattr(model.config, 'VAR_CAP', 1e4), tf.float32)  # noqa: F841 - unused, but removing the op shifts graph op seeds
    # Floor only: clip_by_value has zero gradient outside its range, so the old upper cap (1e3)
    # permanently detached any variance head that drifted to it. With the physics terms now
    # bounded nothing pushes sigma to +inf; the metric paths keep the cap for gauss_p_up.
    var_h_c = [tf.maximum(var_h[i], var_floor) for i in hs]

    # NLL_KIND is read from the config once, at trace time (a Python if, not tf.cond): the 'gaussian'
    # branch below is the original formula, kept byte-for-byte so the default graph (and the golden
    # run) is unaffected by this option's existence (D-023, NT-100 criterion 1).
    nll_kind = str(getattr(model.config, 'NLL_KIND', 'gaussian'))
    nll_h_val = []
    if nll_kind == 'student_t':
        # A_losses.md section 5: the gaussian NLL's mean-gradient e/v and variance-gradient e^2/v
        # are bounded only by the variance floor and the +-100 output clip, so a single large
        # residual can dominate a batch. A fixed-dof Student-t has heavier tails: its mean gradient
        # saturates at +-(dof+1)/(2*|e|) as |e| -> infinity instead of growing like e/v, and its
        # scale gradient is bounded likewise. The variance head keeps predicting Var[Y|x] in both
        # cases: the implied Student-t scale^2 is derived from it so the served variance, the
        # conformal intervals and the CRPS path (all built on var_h_c) keep their meaning.
        dof = float(getattr(model.config, 'NLL_STUDENT_DOF', 5.0))
        # log of the dof-only normalising constant of the Student-t density, -lgamma((dof+1)/2) +
        # lgamma(dof/2) + 0.5*log(dof*pi): a fixed Python float (dof is not learned), so this adds a
        # plain scalar constant to the graph, not a tf op per call.
        nll_const = tf.constant(
            math.lgamma(dof / 2.0) - math.lgamma((dof + 1.0) / 2.0) + 0.5 * math.log(dof * math.pi),
            dtype=tf.float32)
        scale_factor = (dof - 2.0) / dof  # scale^2 = var_c * scale_factor; dof > 2 (Config.validate)
        half_dof_p1 = (dof + 1.0) / 2.0

        def _student_t_nll(y_true, price, var_c):
            scale_sq = var_c * scale_factor
            resid_sq = tf.square(y_true - price)
            return (nll_const + 0.5 * tf.math.log(scale_sq + model.eps)
                    + half_dof_p1 * tf.math.log1p(resid_sq / ((dof - 2.0) * var_c + model.eps)))

        for i in hs:
            nll_i = _student_t_nll(y_true_h[i], price_h[i], var_h_c[i])
            nll_h_val.append(tf.reduce_mean(nll_i))
    else:
        log_2pi = tf.constant(1.8378770664093453, dtype=tf.float32)

        for i in hs:
            nll_i = (0.5 * (log_2pi + tf.math.log(var_h_c[i] + model.eps))
                     + 0.5 * tf.square(y_true_h[i] - price_h[i]) / (var_h_c[i] + model.eps))
            nll_h_val.append(tf.reduce_mean(nll_i))
    nll_h_val = [_finite_or_zero(model, nll_h_val[i], f"nll_{h}") for i, h in enumerate(keys)]
    total_nll = model.lambda_var * _sum_in_order(nll_h_val)
    total_nll = _finite_or_zero(model, total_nll, "nll_loss")

    mu_h = [tf.squeeze(price_h[i], axis=1) for i in hs]
    # Gaussian-implied P(up | move left the deadband), consistent with the masked labels.
    def _gauss_up(mu, var_c):
        return gaussian_up_prob_given_move(mu, tf.squeeze(var_c, axis=1), last_close_squeeze,
                                           deadband, model.pred_mean, model.pred_scale, model.eps)
    gauss_p_up_h = [_gauss_up(mu_h[i], var_h_c[i]) for i in hs]

    # Direction/Gaussian alignment. Skipped entirely unless its outer weight is > 0
    # (LAMBDA_DIR_ALIGN_OUTER defaults to 0): otherwise N BCE evaluations per step
    # for a term that is multiplied by zero. The BCE is computed per example by hand:
    # tf.keras.losses.binary_crossentropy on 1-D arguments reduces over the batch and
    # returned a scalar, which made the masks a no-op.
    if float(getattr(model, 'lambda_dir_align_outer', 0.0)) > 0.0:
        lambda_dir_align = tf.constant(float(getattr(model.config, 'LAMBDA_DIR_ALIGN', 0.0)), dtype=tf.float32)
        def _bce(target, pred):  # per-example, so the masks below weight samples
            pred = tf.clip_by_value(pred, 1e-7, 1.0 - 1e-7)
            return -(target * tf.math.log(pred) + (1.0 - target) * tf.math.log(1.0 - pred))
        align_h = [_bce(gauss_p_up_h[i], dir_pred_h[i]) for i in hs]
        align_h = [tf.reduce_sum(align_h[i] * mask_h[i]) / (tf.reduce_sum(mask_h[i]) + model.eps) for i in hs]
        dir_align_loss = lambda_dir_align * _sum_in_order(align_h)
        dir_align_loss = _finite_or_zero(model, dir_align_loss, "dir_align_loss")
    else:
        dir_align_loss = tf.constant(0.0, dtype=tf.float32)

    reg_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)
    inter_reg = model.config.LAMBDA_INTER * reg_loss

    # The volatility penalty reads the primary horizon (h1 at N = 3; core.horizons.primary_index).
    primary = primary_index(n)
    actual_trend = y_true[:, primary]
    pred_trend_scaled = tf.squeeze(price_h[primary], axis=1)
    actual_std = _std(model, actual_trend)
    pred_std = _std(model, pred_trend_scaled)  # NT-096: finite gradient when the primary price head is batch-constant (if LOSS_SAFE_STD)
    vol_diff = tf.abs(pred_std - actual_std)
    vol_diff_clipped = tf.minimum(vol_diff, 10.0)
    vol_loss = vol_diff_clipped * model.lambda_vol
    vol_loss = _finite_or_zero(model, vol_loss, "vol_loss")

    # === CRPS LOSSES (Gaussian Continuous Ranked Probability Score) ===
    # Controlled by lambda_crps (default 0 → no effect on existing runs).
    # CRPS is a proper scoring rule that jointly rewards sharpness and calibration,
    # preventing the variance head from collapsing sigma to game NLL.
    lambda_crps = tf.cast(getattr(model, 'lambda_crps', 0.0), tf.float32)
    crps_h_val = [crps_gaussian_loss(model, y_true_h[i], price_h[i], var_h_c[i]) for i in hs]
    total_crps = lambda_crps * _sum_in_order(crps_h_val)
    total_crps = _finite_or_zero(model, total_crps, "crps_loss")

    # === SOFT-ECE LOSSES (differentiable Expected Calibration Error) ===
    # Directly minimizes direction-head calibration error w.r.t. realized outcomes.
    # Complements dir-align (which aligns dir_head to Gaussian-implied P(up)) by also
    # aligning to actual labels. Controlled by lambda_soft_ece (default 0).
    lambda_soft_ece = tf.cast(getattr(model, 'lambda_soft_ece', 0.0), tf.float32)
    soft_ece_h_val = [soft_ece_loss(model, true_dir_h[i], dir_pred_h[i], mask_h[i]) for i in hs]
    total_soft_ece = lambda_soft_ece * _sum_in_order(soft_ece_h_val)
    total_soft_ece = _finite_or_zero(model, total_soft_ece, "soft_ece_loss")

    # === T_⊥ / QBOX LOSSES ===
    # T_⊥ calibration: predicted σ tracks empirical residual std (T_⊥ is what we can't explain)
    lambda_t_perp = tf.cast(getattr(model, 'lambda_t_perp', 0.0), tf.float32)
    t_perp_h_val = [t_perp_calibration_loss(model, y_true_h[i], price_h[i], var_h_c[i]) for i in hs]
    total_t_perp = lambda_t_perp * _sum_in_order(t_perp_h_val)
    total_t_perp = _finite_or_zero(model, total_t_perp, "t_perp_loss")

    # Casimir: destructive cross-horizon interference → T_⊥ (σ) must be high
    lambda_casimir = tf.cast(getattr(model, 'lambda_casimir', 0.0), tf.float32)
    casimir_val = lambda_casimir * casimir_interference_loss(model, *price_h, *var_h_c)
    casimir_val = _finite_or_zero(model, casimir_val, "casimir_loss")

    # Vacuum bandwidth: cross-horizon spread must not exceed Λ_vac (self-limiting)
    # P0-2: now opt-in (default 0 in Config + helper). When 0 the term is 0.
    lambda_vac_cfg = tf.constant(float(getattr(getattr(model, 'config', None), 'LAMBDA_VAC', 0.0)),
                                 dtype=tf.float32)
    vac_val = vacuum_bandwidth_loss(model, *price_h, lambda_vac=lambda_vac_cfg)
    vac_val = _finite_or_zero(model, vac_val, "vac_loss")

    # Hyper-decoherence: high local volatility should couple to high σ
    lambda_hd = tf.cast(getattr(model, 'lambda_hd', 0.0), tf.float32)
    hd_val = lambda_hd * hyper_decoherence_coupling_loss(model, x_window, *var_h_c)
    hd_val = _finite_or_zero(model, hd_val, "hd_loss")

    # Information flow entropy: each horizon must carry non-redundant information
    lambda_ife = tf.cast(getattr(model, 'lambda_ife', 0.0), tf.float32)
    ife_val = lambda_ife * information_flow_entropy_loss(model, *price_h)
    ife_val = _finite_or_zero(model, ife_val, "ife_loss")

    # Vacuum overflow T_⊥ precision: overflow tracks prediction residual magnitude
    # Active only when lambda_vac_overflow > 0 AND vacuum_overflow tensor is provided.
    lambda_vac_overflow = tf.cast(getattr(model, 'lambda_vac_overflow', 0.0), tf.float32)
    if vacuum_overflow is not None:
        pairs_yp = []
        for i in hs:
            pairs_yp += [y_true_h[i], price_h[i]]
        vac_overflow_val = lambda_vac_overflow * vacuum_overflow_t_perp_loss(model, vacuum_overflow, *pairs_yp)
    else:
        vac_overflow_val = tf.constant(0.0, dtype=tf.float32)
    vac_overflow_val = _finite_or_zero(model, vac_overflow_val, "vac_overflow_loss")

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
    total = _finite_or_zero(model, total, "total_loss")

    # Return as the N-horizon LossComponents (LossComponents itself at N = 3). This preserves
    # exact positional unpacking for all callers while enabling named attribute access (e.g.
    # lc.extended_h1). The field order is core.horizons.loss_component_fields(N).
    per_trend = []
    for i in hs:
        per_trend += [local_trend_h[i], global_trend_h[i], extended_trend_h[i]]
    return loss_components_type(n)(
        total,
        *point_loss_h,
        *per_trend,
        *dir_loss_h,
        *nll_h_val,
        reg_loss, inter_reg, vol_loss,
        *crps_h_val,
        *soft_ece_h_val,
        total_t_perp, casimir_val, vac_val, hd_val, ife_val,
        vac_overflow_val,
        tf.constant(0.0, dtype=tf.float32),  # pnl_val: 0 unless wrapped by pnl_utility (NT-087)
        dir_align_loss, coherence_penalty,
    )


@Losses.register_objective(name="pnl_utility", tags=["composite", "pnl"])
def pnl_utility(model, x_window, y_true, y_pred, last_close, extended_trends, vacuum_overflow=None):
    """``custom_loss`` plus a mean-variance P&L utility, with a linear cost, on the direction
    heads' implied positions a_i = 2*p_i - 1 (NT-087, ``docs/research/2026-09-29-pnl-target/``
    section 1.1, option A).

    ``U_i = mean( a*r~ - c~*|a| - 0.5*gamma*a^2*r~^2 )``, ``r~`` the volatility-scaled forward
    return of horizon i, winsorised at +-5 sigma; ``loss_pnl_i = -U_i``. The maximiser of U per
    sample is the net-edge fractional-Kelly aim: a* = 0 unless the expected move exceeds the cost
    (Garleanu and Pedersen 2013), and the quadratic term keeps a* inside (-1, 1).

    ``x_window`` is the NORMALISED input window ``[B, LOOKBACK]`` (``Config.WINDOW_NORMALIZER``,
    default ``"window_relative"``: ``x = (raw_close - last_close) / pred_scale``,
    ``data/scaling.py``); it is NOT the raw close-price window (NT-087 repair round 1: an earlier
    version of ``PNL_SIGMA_SOURCE='ewma'`` treated it as raw prices and computed step returns
    ~3,500x too large, since the last window column is exactly 0 by construction). ``'realized_vol'``
    reconstructs the raw window as ``x * model.pred_scale + last_close`` (exact for
    ``window_relative``; not calibrated for the legacy ``per_lag_standard`` normaliser, which this
    function cannot invert from ``x_window`` alone) before taking bar-to-bar returns. The dir/var
    heads are sanitised for non-finite values exactly as ``custom_loss`` does internally, since
    ``y_pred`` here is the model's raw (unsanitised) output.
    """
    base = custom_loss(model, x_window, y_true, y_pred, last_close, extended_trends,
                       vacuum_overflow=vacuum_overflow)

    y_true = tf.cast(y_true, tf.float32)
    y_true_raw = y_true * model.pred_scale + model.pred_mean
    last_close_sq = tf.squeeze(last_close, axis=1)
    y_pred = list(y_pred)
    n = len(y_pred) // 3
    keys = horizon_keys(n)

    # Sanitize dir/var heads for non-finite values, mirroring custom_loss's own copies (it does
    # not expose them): dir falls back to 0.5 (no signal), var to 1.0.
    dir_h = [_finite_or_zero(model, y_pred[3 * i + 1], f"head_dir_{h}", fallback=0.5) for i, h in enumerate(keys)]
    var_h = [_finite_or_zero(model, y_pred[3 * i + 2], f"head_var_{h}", fallback=1.0) for i, h in enumerate(keys)]

    default_horizons = [10, 15, 20]
    horizon_steps_cfg = list(getattr(model.config, 'HORIZON_STEPS', default_horizons) or default_horizons)
    # one step count per horizon (NT-042); padded from the reference steps only when the config
    # holds fewer entries than the model has horizons (defensive: Config.validate prevents it)
    horizon_steps = (horizon_steps_cfg + default_horizons * n)[:n]
    var_floor = tf.cast(getattr(model.config, 'VAR_FLOOR', 1e-4), tf.float32)
    cost = tf.cast(getattr(model.config, 'PNL_COST_BPS', 0.0), tf.float32) / 10000.0
    gamma = tf.cast(getattr(model.config, 'PNL_GAMMA', 1.0), tf.float32)
    sigma_source = str(getattr(model.config, 'PNL_SIGMA_SOURCE', 'realized_vol'))
    eps = model.eps

    step_sd = None
    if sigma_source == 'realized_vol':
        # Reconstruct the RAW close-price window from the normalised one (window_relative:
        # x = (raw - last_close) / pred_scale => raw = x * pred_scale + last_close), sanitising
        # every stage so a single bad bar (+-inf, or a near-zero raw price) cannot turn into a
        # non-finite sigma that would otherwise reach the direction heads as a NaN gradient
        # (NT-087 repair round 1, P2).
        x = tf.cast(x_window, tf.float32)
        last_close_col = tf.expand_dims(last_close_sq, axis=1)
        raw = x * model.pred_scale + last_close_col                        # [B, LOOKBACK]
        raw = _finite_or_zero(model, raw, "pnl_raw_window")
        step_ret = (raw[:, 1:] - raw[:, :-1]) / (raw[:, :-1] + eps)        # [B, LOOKBACK-1]
        step_ret = _finite_or_zero(model, step_ret, "pnl_step_ret")
        step_sd = tf.stop_gradient(tf.math.reduce_std(step_ret, axis=1))  # [B]
        step_sd = _finite_or_zero(model, step_sd, "pnl_step_sd")

    def _one_horizon(i, dir_head, var_head, h_steps):
        r = y_true_raw[:, i] / (last_close_sq + eps)                       # fractional forward return
        if sigma_source == 'realized_vol':
            sigma = step_sd * tf.sqrt(tf.cast(h_steps, tf.float32))
        else:
            var_c = tf.maximum(tf.squeeze(var_head, axis=1), var_floor)
            sigma_raw = tf.sqrt(var_c) * model.pred_scale
            sigma = tf.stop_gradient(sigma_raw / (last_close_sq + eps))
        # Sanitise BEFORE the floor: tf.maximum(nan, 1e-6) is still nan, so a non-finite sigma
        # (e.g. from an extreme window) must be caught here, not only on the final loss (P2).
        sigma = _finite_or_zero(model, sigma, "pnl_sigma")
        sigma = tf.maximum(sigma, 1e-6)
        r_tilde = tf.clip_by_value(r / sigma, -5.0, 5.0)
        # Demeaned over the batch (not a persistent training-block statistic): with a full reshuffle
        # (Config.SHUFFLE_BUFFER: 0, the E2 scenario's setting) a batch is drawn uniformly from the
        # whole training block, so its mean is an unbiased, stateless estimator of the block mean,
        # and this avoids contaminating a running estimate with validation-block batches (the
        # objective has no train/eval flag; test_step calls it too). NT-087 repair round 1, P2:
        # this equality holds ONLY under a full reshuffle. With the default SHUFFLE_BUFFER (2048, a
        # partial, order-preserving shuffle) a training batch is drawn from a local window of the
        # block, not the whole block, so its mean tracks local drift instead of the block mean; and
        # validation/test batches are never shuffled at all (test_step iterates them in order), so
        # r_tilde there is demeaned by a short local window, not by anything comparable to the
        # training-block statistic. val_loss under pnl_utility is therefore not on the same
        # demeaning basis as the training loss unless SHUFFLE_BUFFER is 0.
        r_tilde = r_tilde - tf.stop_gradient(tf.reduce_mean(r_tilde))
        c_tilde = cost / sigma
        a = 2.0 * tf.squeeze(dir_head, axis=1) - 1.0
        util = tf.reduce_mean(a * r_tilde - c_tilde * tf.abs(a) - 0.5 * gamma * tf.square(a) * tf.square(r_tilde))
        loss_i = -util
        return _finite_or_zero(model, loss_i, "pnl_loss_h")

    loss_h = [_one_horizon(i, dir_h[i], var_h[i], horizon_steps[i]) for i in range(n)]

    lambda_pnl = tf.cast(getattr(model, 'lambda_pnl', 0.0), tf.float32)
    pnl_term = lambda_pnl * _sum_in_order(loss_h)
    pnl_term = _finite_or_zero(model, pnl_term, "pnl_term")

    total = base.total + pnl_term
    total = _finite_or_zero(model, total, "pnl_total")
    return base._replace(total=total, pnl_val=pnl_term)

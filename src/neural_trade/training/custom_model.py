"""CustomTrainModel: two-optimizer training loop over the multi-horizon objective (moved in B10).

* ``train_step``: objective -> gradients -> finite-gradient guard -> split into NN weights and
  indicator logits -> per-group global-norm clip -> two optimizers -> clip learned periods in
  logit space -> epoch-level metric accumulators.
* ``test_step``: the same objective and metrics, no update.
* Loss weights are live tf.Variables behind ``lambda_*`` properties (training.lambdas).
"""
from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import models

import neural_trade.losses.functions as _losses
import neural_trade.utils.math as mh
from neural_trade.core.config import Config
from neural_trade.core.outputs import PredictiveOutputs
from neural_trade.metrics.tf_direction import (DirectionAccumulator, PITAccumulator, STEP_MEAN_KEYS,
                                               TRAIN_ONLY_MEAN_KEYS, direction_counts,
                                               direction_labels_tf, direction_metrics_from_stats,
                                               direction_stats)
from neural_trade.registries.losses import Losses
from neural_trade.registries.metrics import Metrics
from neural_trade.registries.models import Models
from neural_trade.training.lambdas import install_lambda_properties
from neural_trade.training.optim import build_indicator_optimizer

logger = logging.getLogger(__name__)


class CustomTrainModel(models.Model):
    def __init__(self, base_model, pred_scale, pred_mean,
                 lambda_point=1.0, lambda_local_trend=1.0, lambda_global_trend=0.2,
                 lambda_extended_trend=0.16, lambda_dir=1.0, config=None,
                 objective=None, indicator_optimizer=None, **kwargs):
        super().__init__(**kwargs)
        config = config or Config()
        self.base_model = base_model
        self.epsilon = 1e-8
        # Per-term loss weights live in non-trainable tf.Variables behind properties (see
        # _make_lambda_property after this class). Reads in losses.py (`model.lambda_x`) and
        # writes in the calibration pass (`model.lambda_x = v`) are unchanged, but a write is
        # now an in-place .assign() that takes effect on the next step without retracing.
        # Created with attribute tracking off so Keras does not add them to the weights file.
        self._setattr_tracking = False
        self._lambda_vars = {}
        self._setattr_tracking = True

        # Cast important scalars to float32 early
        self.pred_scale = tf.cast(pred_scale, tf.float32)
        self.pred_mean = tf.cast(pred_mean, tf.float32)

        # Basic numeric guard
        if tf.keras.backend.get_value(self.pred_scale) < 1e-6:
            raise ValueError("pred_scale is too small, which may cause numerical instability.")

        self.lambda_point = float(lambda_point)
        self.lambda_local_trend = float(lambda_local_trend)
        self.lambda_global_trend = float(lambda_global_trend)
        self.lambda_extended_trend = float(lambda_extended_trend)
        self.lambda_dir = float(lambda_dir)  # New
        self.lambda_vol = config.LAMBDA_VOL
        self.lambda_short = config.LAMBDA_SHORT
        self.lambda_long = config.LAMBDA_LONG
        self.lambda_var = config.LAMBDA_VAR
        self.lambda_trend_outer = float(getattr(config, 'LAMBDA_TREND_OUTER', 0.5))
        self.lambda_dir_outer = float(getattr(config, 'LAMBDA_DIR_OUTER', 0.5))
        self.lambda_dir_align_outer = float(getattr(config, 'LAMBDA_DIR_ALIGN_OUTER', 0.0))
        self.lambda_coherence_outer = float(getattr(config, 'LAMBDA_COHERENCE', 1.0))
        self.lambda_nll_outer = float(getattr(config, 'LAMBDA_NLL_OUTER', 1.0))
        self.lambda_crps = float(getattr(config, 'LAMBDA_CRPS', 0.0))
        self.lambda_soft_ece = float(getattr(config, 'LAMBDA_SOFT_ECE', 0.0))
        # === T_⊥ / QBOX lambdas (default 0.0 → backward compatible; enable explicitly) ===
        self.lambda_t_perp       = float(getattr(config, 'LAMBDA_T_PERP',       0.0))
        self.lambda_casimir      = float(getattr(config, 'LAMBDA_CASIMIR',      0.0))
        self.lambda_hd           = float(getattr(config, 'LAMBDA_HD',           0.0))
        self.lambda_ife          = float(getattr(config, 'LAMBDA_IFE',          0.0))
        self.lambda_vac_overflow = float(getattr(config, 'LAMBDA_VAC_OVERFLOW', 0.0))
        self.config = config or Config()

        # Dedicated optimizer for indicator logit vars (LR = main LR * INDICATOR_LR_MULT).
        # Adam normalizes gradient magnitudes, so scaling grads is insufficient — a higher LR
        # is the only way to give indicator params a genuinely larger step size.
        # Built through the Optimizers registry (Config.INDICATOR_OPTIMIZER_NAME) unless the caller
        # passes one (Trainer builds both optimizers together: training.optim.build_optimizers).
        self.indicator_optimizer = indicator_optimizer or build_indicator_optimizer(self.config)
        # The training objective, resolved ONCE from the Losses registry (Config.LOSS_NAME).
        self._setattr_tracking = False
        self.objective = objective or Losses.get_objective(getattr(self.config, 'LOSS_NAME', None))
        self._setattr_tracking = True

        # Single source-of-truth for Huber delta (in *scaled* units)
        self.huber_delta = float(self.config.HUBER_DELTA)

        # Numerical epsilon used in denominators
        self.eps = tf.constant(1e-8, dtype=tf.float32)
        # Counts training steps whose update was zeroed by the finite-gradient guard (reset each epoch by Keras).
        self.nonfinite_grad_steps = tf.keras.metrics.Sum(name='nonfinite_grad_steps')
        # Epoch accumulators behind every logged step metric (see neural_trade.metrics.tf_direction.STEP_MEAN_KEYS).
        # Created with attribute tracking off; the `metrics` property below hands them to
        # Keras so they are reset at every epoch and before every evaluation.
        self._setattr_tracking = False
        self._step_means = {k: tf.keras.metrics.Mean(name=k) for k in STEP_MEAN_KEYS}
        # Step metrics resolved ONCE from the Metrics registry (TF tier), never inside tf.function.
        self._step_metric_fns = Metrics.tf_functions(getattr(self.config, 'STEP_METRICS', None))
        self._dir_head_acc = DirectionAccumulator(name='dir_head_accumulator')
        self._dir_gauss_acc = DirectionAccumulator(name='dir_gauss_accumulator')
        self._pit_acc = PITAccumulator(var_floor=float(getattr(self.config, 'VAR_FLOOR', 1e-4)),
                                        var_cap=float(getattr(self.config, 'VAR_CAP', 1e3)),
                                        name='pit_accumulator')
        self._setattr_tracking = True

        # Robust (non-string) collection of indicator vars for gradient routing
        # (to indicator_optimizer) and post-step period clipping.
        # Falls back gracefully if the layer is not present (e.g. during some tests).
        self._indicator_var_ids = set()
        self._indicator_layer = None
        try:
            if base_model is not None:
                for layer in base_model.layers:
                    if getattr(layer, 'name', '').startswith('learnable_indicators'):
                        ind_vars = getattr(layer, 'get_indicator_trainable_variables', lambda: [])()
                        self._indicator_var_ids = {id(v) for v in ind_vars}
                        self._indicator_layer = layer
                        break
        except Exception:
            # Without the indicator variables the second optimizer silently gets nothing.
            logger.exception("could not locate the learnable-indicator layer's variables")

        # NOTE: We no longer use tf.keras.losses.Huber in the primary point supervision path;
        # point loss delegates to the registered "point_huber" which implements log(cosh).
        # A separate piecewise Huber lives in CustomTrainModel.huber (unused for the main loss).
        # Config.USE_HUBER is legacy and not consulted by the active custom_loss.
    def _logit_from_alpha(self, alpha): return mh.logit_from_alpha(alpha, self.epsilon)
    def _alpha_from_logit(self, logit): return mh.alpha_from_logit(logit)
    def _logit_from_period(self, period): return mh.logit_from_period(period, self.epsilon)
    def _period_from_logit(self, logit): return mh.period_from_logit(logit, self.epsilon)
    # -------------------------
    # Unified element-wise Huber
    # -------------------------
    def huber(self, x, delta=None):
        """Element-wise Huber (returns same-shape tensor). Works on scaled differences."""
        if delta is None:
            delta = tf.cast(self.huber_delta, tf.float32)
        else:
            delta = tf.cast(delta, tf.float32)

        x = tf.cast(x, tf.float32)
        abs_x = tf.abs(x)
        quadratic = 0.5 * tf.square(x)
        linear = delta * (abs_x - 0.5 * delta)
        return tf.where(abs_x <= delta, quadratic, linear)

    @property
    def metrics(self):
        """Metrics Keras resets at each epoch / evaluation: the built-in ones plus the epoch
        accumulators behind the step logs."""
        base = list(super().metrics)
        extra = list(getattr(self, '_step_means', {}).values())
        extra += [m for m in (getattr(self, '_dir_head_acc', None), getattr(self, '_dir_gauss_acc', None),
                              getattr(self, '_pit_acc', None)) if m is not None]
        seen = {id(m) for m in base}
        return base + [m for m in extra if id(m) not in seen]

    def _epoch_logs(self, loss_components, y_true, y_pred_9, true_dirs, head_probs, gauss_probs, masks,
                    head_prefix, gauss_prefix, training, grad_global_norm=None):
        """Update the epoch accumulators with this batch and return their running aggregates."""
        c = loss_components
        batch = tf.cast(tf.shape(y_true)[0], tf.float32)
        scalars = {
            'loss': c.total,
            'point_loss': c.point_h0 + c.point_h1 + c.point_h2,
            'point_h0': c.point_h0, 'point_h1': c.point_h1, 'point_h2': c.point_h2,
            'trend_h0': c.local_h0 + c.global_h0 + c.extended_h0,
            'trend_h1': c.local_h1 + c.global_h1 + c.extended_h1,
            'trend_h2': c.local_h2 + c.global_h2 + c.extended_h2,
            'local_h0': c.local_h0, 'global_h0': c.global_h0, 'extended_h0': c.extended_h0,
            'local_h1': c.local_h1, 'global_h1': c.global_h1, 'extended_h1': c.extended_h1,
            'local_h2': c.local_h2, 'global_h2': c.global_h2, 'extended_h2': c.extended_h2,
            'dir_loss': c.dir_h0 + c.dir_h1 + c.dir_h2,
            'dir_loss_h0': c.dir_h0, 'dir_loss_h1': c.dir_h1, 'dir_loss_h2': c.dir_h2,
            'nll_loss': c.nll_h0 + c.nll_h1 + c.nll_h2,
            'nll_h0': c.nll_h0, 'nll_h1': c.nll_h1, 'nll_h2': c.nll_h2,
            'crps_loss': c.crps_h0 + c.crps_h1 + c.crps_h2,
            'crps_h0': c.crps_h0, 'crps_h1': c.crps_h1, 'crps_h2': c.crps_h2,
            'soft_ece_loss': c.soft_ece_h0 + c.soft_ece_h1 + c.soft_ece_h2,
            'soft_ece_h0': c.soft_ece_h0, 'soft_ece_h1': c.soft_ece_h1, 'soft_ece_h2': c.soft_ece_h2,
            'reg_loss': c.reg_loss, 'inter_reg': c.inter_reg, 'vol_loss': c.vol_loss,
            't_perp_loss': c.t_perp_total, 'casimir_loss': c.casimir_val, 'vac_loss': c.vac_val,
            'hd_loss': c.hd_val, 'ife_loss': c.ife_val, 'vac_overflow_loss': c.vac_overflow_val,
        }
        scalars['trend_loss'] = scalars['trend_h0'] + scalars['trend_h1'] + scalars['trend_h2']
        if training and grad_global_norm is not None:
            scalars['grad_global_norm'] = grad_global_norm
        for k, v in scalars.items():
            self._step_means[k].update_state(tf.cast(v, tf.float32), sample_weight=batch)
        self._dir_head_acc.update_state(true_dirs, head_probs, masks)
        self._dir_gauss_acc.update_state(true_dirs, gauss_probs, masks)
        self._pit_acc.update_state([y_true[:, 0], y_true[:, 1], y_true[:, 2]],
                                   [y_pred_9[0], y_pred_9[3], y_pred_9[6]],
                                   [y_pred_9[2], y_pred_9[5], y_pred_9[8]])
        logs = {k: m.result() for k, m in self._step_means.items()
                if training or k not in TRAIN_ONLY_MEAN_KEYS}
        logs.update(self._pit_acc.logs())
        logs.update(self._dir_head_acc.logs(head_prefix, self._step_metric_fns))
        logs.update(self._dir_gauss_acc.logs(gauss_prefix, self._step_metric_fns))
        return logs


    # Small utility: reduce-mean with safe casting
    def _reduce_mean(self, x):
        return tf.reduce_mean(tf.cast(x, tf.float32))

    @staticmethod
    def _normal_cdf(z):
        """Standard Normal CDF using erf; z can be any float tensor."""
        z = tf.cast(z, tf.float32)
        return 0.5 * (1.0 + tf.math.erf(z / tf.constant(np.sqrt(2.0), dtype=tf.float32)))

    # -------------------------
    # Utility / transforms (moved outside class to avoid tracing issues)
    # -------------------------
    def _pit_ks(self, y, mu, var):
        """Kolmogorov-Smirnov distance between the PIT values Phi((y - mu) / sigma) and U[0, 1].

        Computed in-graph. The previous implementation called .numpy() on symbolic tensors
        inside the traced train step, was swallowed by a bare except, and logged NaN in
        every epoch; test_step did not compute it at all.
        """
        y = tf.cast(tf.reshape(y, [-1]), tf.float32)
        mu = tf.cast(tf.reshape(mu, [-1]), tf.float32)
        var = tf.cast(tf.reshape(var, [-1]), tf.float32)
        var = tf.clip_by_value(var, float(getattr(self.config, 'VAR_FLOOR', 1e-4)),
                               float(getattr(self.config, 'VAR_CAP', 1e3)))
        u = tf.sort(self._normal_cdf((y - mu) / (tf.sqrt(var) + self.eps)))
        n = tf.cast(tf.shape(u)[0], tf.float32)
        i = tf.range(1.0, n + 1.0, dtype=tf.float32)
        d_plus = tf.reduce_max(i / n - u)
        d_minus = tf.reduce_max(u - (i - 1.0) / n)
        return tf.maximum(d_plus, d_minus)

    @staticmethod
    def _to_scaled_static(raw, pred_mean, pred_scale, eps=1e-8):
        """Convert raw prices to scaled units (same domain as dataset scaling)."""
        raw = tf.cast(raw, tf.float32)
        return (raw - pred_mean) / (pred_scale + eps)

    def _to_scaled(self, raw):
        """Instance helper that uses the stored scaling parameters."""
        return self._to_scaled_static(raw, self.pred_mean, self.pred_scale, self.eps)

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    # -------------------------
    # Focal Loss for imbalanced binary classification
    # -------------------------
    def focal_loss(self, true_labels, logits, alpha=None, gamma=None, reduce=True):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.focal_loss(self, true_labels, logits, alpha=alpha, gamma=gamma, reduce=reduce)

    # -------------------------
    # Dice Loss for F1-like optimization (differentiable)
    # -------------------------
    def dice_loss(self, true_labels, logits, smooth=1.0, reduce=True):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.dice_loss(self, true_labels, logits, smooth=smooth, reduce=reduce)

    # -------------------------
    # Combined Focal + Dice Loss for balanced optimization
    # -------------------------
    def combined_direction_loss(self, true_labels, logits, alpha=None, gamma=None,
                                 focal_weight=0.5, dice_weight=0.5, reduce=True):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.combined_direction_loss(self, true_labels, logits, alpha=alpha, gamma=gamma,
                                               focal_weight=focal_weight, dice_weight=dice_weight, reduce=reduce)

    # -------------------------
    # Dynamic Alpha Computation for Class Balancing
    # -------------------------
    def compute_dynamic_alpha(self, true_labels, min_alpha=0.3, max_alpha=0.7):
        """
        Compute dynamic focal alpha based on actual class distribution in batch.

        Alpha weights the DOWN class (label=0), so:
        - If batch has more UP (label=1), alpha should be higher (weight DOWN more)
        - If batch has more DOWN (label=0), alpha should be lower (weight UP more)

        Args:
            true_labels: Binary labels [B]
            min_alpha: Minimum alpha (clips to prevent instability)
            max_alpha: Maximum alpha (clips to prevent instability)

        Returns:
            Dynamic alpha value clipped to [min_alpha, max_alpha]
        """
        true_labels = tf.cast(true_labels, tf.float32)

        # Compute proportion of UP class (label=1)
        up_ratio = tf.reduce_mean(true_labels)

        # Alpha = up_ratio means: weight DOWN inversely to its frequency
        # If up_ratio=0.6 (60% UP), alpha=0.6 → DOWN gets 0.6 weight, UP gets 0.4
        # This balances the classes
        alpha = up_ratio

        # Clip for stability
        alpha = tf.clip_by_value(alpha, min_alpha, max_alpha)

        return alpha

    # -------------------------
    # Point loss (log-cosh)
    # -------------------------
    def point_huber(self, y_true_scaled, y_pred_scaled, last_close_scaled=None, delta=None):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.point_huber(self, y_true_scaled, y_pred_scaled, last_close_scaled=last_close_scaled, delta=delta)


    # -------------------------
    # Local trend loss
    # -------------------------
    def local_trend_loss(self, x_window, y_true_raw, y_pred_raw, last_close_raw):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.local_trend_loss(self, x_window, y_true_raw, y_pred_raw, last_close_raw)


    # -------------------------
    # Extended & global trends
    # -------------------------
    def extended_trend_loss(self, x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.extended_trend_loss(self, x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw)

    # -------------------------
    # Combined custom loss (NEW: Per-horizon outputs with focal loss)
    # -------------------------
    def custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends,
                    vacuum_overflow=None):
        """The configured training objective (Losses registry, objective tier)."""
        return self.objective(self, x_window, y_true, y_pred, last_close, extended_trends,
                              vacuum_overflow=vacuum_overflow)

    def train_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        with tf.GradientTape() as tape:
            y_pred_list = self(x_window, training=True)
            # Named view for the 10 outputs (robust to future aux heads / reordering).
            heads = PredictiveOutputs(*y_pred_list)
            # Still provide the 9-tuple expected by current custom_loss signature + the vacuum separately.
            y_pred_9 = y_pred_list[:9]
            vac_overflow_pred = heads.vacuum_overflow
            loss_components = self.custom_loss(x_window, y_true, y_pred_9, last_close,
                                               extended_trends,
                                               vacuum_overflow=vac_overflow_pred)

        # loss_components is now a LossComponents NamedTuple (see registries/losses.py).
        # Positional unpack is preserved for compatibility; attribute access is also available.
        (total_loss_val,
         point_h0, point_h1, point_h2,
         local_h0, global_h0, extended_h0,
         local_h1, global_h1, extended_h1,
         local_h2, global_h2, extended_h2,
         dir_h0, dir_h1, dir_h2,
         nll_h0, nll_h1, nll_h2,
         reg_val, inter_reg, vol_loss,
         crps_h0, crps_h1, crps_h2,
         soft_ece_h0, soft_ece_h1, soft_ece_h2,
         t_perp_total, casimir_val, vac_val, hd_val, ife_val,
         vac_overflow_val) = loss_components

        grads = tape.gradient(total_loss_val, self.trainable_variables)

        # ---- Finite-gradient guard --------------------------------------------
        # One non-finite gradient anywhere used to poison EVERY weight in a single
        # step: tf.clip_by_global_norm computed a NaN global norm and rescaled every
        # gradient in the group by it. Zero the whole update instead, and count it.
        _present = [g for g in grads if g is not None]
        grad_global_norm = tf.linalg.global_norm(_present) if _present else tf.constant(0.0, dtype=tf.float32)
        step_finite = tf.math.is_finite(total_loss_val)
        if _present:
            step_finite = tf.logical_and(
                step_finite,
                tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in _present])),
            )
        self.nonfinite_grad_steps.update_state(tf.cast(tf.logical_not(step_finite), tf.float32))
        # tf.where, not `g * mask`: NaN * 0 is still NaN.
        grads = [None if g is None else tf.where(step_finite, g, tf.zeros_like(g)) for g in grads]

        # Split gradients into NN weights vs. indicator logit vars using id() set
        # (populated in __init__ from the layer's get_indicator_trainable_variables).
        # This replaces fragile substring matching on variable names.
        nn_gvs, ind_gvs = [], []
        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue
            (ind_gvs if id(v) in self._indicator_var_ids else nn_gvs).append((g, v))

        # Clip NN grads by global norm only (indicator grads are small scalars; Adam handles scale)
        # Also clip indicator grads for stability (high INDICATOR_LR_MULT + STE can produce large updates
        # on the scalar logit vars, leading to extreme alphas/periods and NaN cascade in features/preds).
        clip_norm = float(getattr(self.config, 'GRAD_CLIP_NORM', 0.0) or 0.0)
        if clip_norm > 0.0:
            if nn_gvs:
                nn_gs_clipped, _ = tf.clip_by_global_norm(
                    [g for g, v in nn_gvs], clip_norm)
                nn_gvs = list(zip(nn_gs_clipped, [v for g, v in nn_gvs]))
            if ind_gvs:
                ind_gs_clipped, _ = tf.clip_by_global_norm(
                    [g for g, v in ind_gvs], clip_norm)
                ind_gvs = list(zip(ind_gs_clipped, [v for g, v in ind_gvs]))

        # Apply gradients with separate optimizers
        self.optimizer.apply_gradients(nn_gvs)
        self.indicator_optimizer.apply_gradients(ind_gvs)

        # Clip learned indicator periods by delegating to the layer that owns them.
        # This encapsulates the period <-> logit conversion and removes duplicated
        # name-based string checks that used to live in train_step.
        min_p = self.config.MOMENTUM_CLIP_MIN
        max_p = self.config.MOMENTUM_CLIP_MAX
        if self._indicator_layer is not None:
            self._indicator_layer.clip_learned_periods(min_p, max_p)
        # Fallback for any legacy 'momentum_raw' style vars that might still be
        # attached directly to the base model (rare).
        for var in self.base_model.trainable_variables:
            if 'momentum_raw' in getattr(var, 'name', '').lower():
                p = tf.nn.softplus(var) + 1.0
                clipped = tf.clip_by_value(p, min_p, max_p)
                raw = tf.math.asinh((clipped - 1.0) / 2.0)
                var.assign(raw)

        # === COMPUTE DIRECTION METRICS FOR ALL 3 HORIZONS ===
        y_true = tf.cast(y_true, tf.float32)
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, 3] (delta_raw)
        last_close_squeeze = tf.squeeze(last_close, axis=1)
        # Match training direction labeling (including deadband if enabled)
        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        # One labelling rule for every path (neural_trade.metrics.tf_direction).
        mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2 = direction_labels_tf(
            y_true_raw, last_close_squeeze, deadband_bps, self.eps)

        # Extract direction predictions for all 3 horizons
        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred_9
        dir_pred_h0 = tf.squeeze(dir_pred_h0, axis=1)
        dir_pred_h1 = tf.squeeze(dir_pred_h1, axis=1)
        dir_pred_h2 = tf.squeeze(dir_pred_h2, axis=1)

        # Gaussian-implied P(up) from (mu, var): interpretable and consistent with regression.
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e3), tf.float32)
        var_h0_c = tf.clip_by_value(tf.squeeze(var_h0, axis=1), var_floor, var_cap)
        var_h1_c = tf.clip_by_value(tf.squeeze(var_h1, axis=1), var_floor, var_cap)
        var_h2_c = tf.clip_by_value(tf.squeeze(var_h2, axis=1), var_floor, var_cap)
        mu_h0 = tf.squeeze(price_h0, axis=1)
        mu_h1 = tf.squeeze(price_h1, axis=1)
        mu_h2 = tf.squeeze(price_h2, axis=1)
        # P(up | the move left the deadband): matches the masked labels (see losses.gaussian_up_prob_given_move).
        _lc = tf.squeeze(last_close, axis=1)
        gauss_p_up_h0 = _losses.gaussian_up_prob_given_move(mu_h0, var_h0_c, _lc, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h1 = _losses.gaussian_up_prob_given_move(mu_h1, var_h1_c, _lc, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h2 = _losses.gaussian_up_prob_given_move(mu_h2, var_h2_c, _lc, deadband, self.pred_mean, self.pred_scale, self.eps)

        logs = self._epoch_logs(
            loss_components, y_true, y_pred_9,
            (true_dir_h0, true_dir_h1, true_dir_h2),
            (dir_pred_h0, dir_pred_h1, dir_pred_h2),
            (gauss_p_up_h0, gauss_p_up_h1, gauss_p_up_h2),
            (mask_h0, mask_h1, mask_h2),
            head_prefix="train_", gauss_prefix="train_gauss_", training=True,
            grad_global_norm=grad_global_norm,
        )
        logs["nonfinite_grad_steps"] = self.nonfinite_grad_steps.result()
        return logs

    def _compute_direction_metrics(self, true_dir_h0, true_dir_h1, true_dir_h2, dir_pred_h0, dir_pred_h1, dir_pred_h2, mask_h0=None, mask_h1=None, mask_h2=None, prefix=""):
        """Direction metrics (acc, sensitivity, specificity, balanced acc, F1, MCC, Brier,
        positive-class ECE, predicted/true up rates, mean prob) over the given arrays, per horizon.

        Same formulas as the epoch accumulators (tf_direction.direction_metrics_from_stats); this form
        evaluates one set of arrays in full.
        """
        metrics = {}
        for h_name, true_dir, dir_pred, mask in (("h0", true_dir_h0, dir_pred_h0, mask_h0),
                                                 ("h1", true_dir_h1, dir_pred_h1, mask_h1),
                                                 ("h2", true_dir_h2, dir_pred_h2, mask_h2)):
            m = tf.ones_like(tf.cast(true_dir, tf.float32)) if mask is None else mask
            stats = direction_stats(*direction_counts(true_dir, dir_pred, m))
            metrics.update(direction_metrics_from_stats(stats, prefix, h_name,
                                                        getattr(self, '_step_metric_fns', None)))
        return metrics

    def test_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        y_pred_list = self(x_window, training=False)
        y_pred_9 = y_pred_list[:9]
        loss_components = self.custom_loss(x_window, y_true, y_pred_9, last_close,
                                           extended_trends,
                                           vacuum_overflow=None)  # identically 0 at eval (tanh^2 < E_max): the term would be a constant lambda in every val_loss

        # Unpack 34-component tuple (LossComponents NamedTuple; positional ok)
        (total_loss_val,
         point_h0, point_h1, point_h2,
         local_h0, global_h0, extended_h0,
         local_h1, global_h1, extended_h1,
         local_h2, global_h2, extended_h2,
         dir_h0, dir_h1, dir_h2,
         nll_h0, nll_h1, nll_h2,
         reg_val, inter_reg, vol_loss,
         crps_h0, crps_h1, crps_h2,
         soft_ece_h0, soft_ece_h1, soft_ece_h2,
         t_perp_total, casimir_val, vac_val, hd_val, ife_val,
         vac_overflow_val) = loss_components

        # Compute direction labels with the same trade-aware deadband used in training loss.
        y_true = tf.cast(y_true, tf.float32)
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, 3]
        last_close_squeeze = tf.squeeze(last_close, axis=1)

        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        # One labelling rule for every path (neural_trade.metrics.tf_direction).
        mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2 = direction_labels_tf(
            y_true_raw, last_close_squeeze, deadband_bps, self.eps)

        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred_9
        dir_pred_h0 = tf.squeeze(dir_pred_h0, axis=1)
        dir_pred_h1 = tf.squeeze(dir_pred_h1, axis=1)
        dir_pred_h2 = tf.squeeze(dir_pred_h2, axis=1)

        # Gaussian-implied P(up) from (mu, var)
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e3), tf.float32)
        var_h0_c = tf.clip_by_value(tf.squeeze(var_h0, axis=1), var_floor, var_cap)
        var_h1_c = tf.clip_by_value(tf.squeeze(var_h1, axis=1), var_floor, var_cap)
        var_h2_c = tf.clip_by_value(tf.squeeze(var_h2, axis=1), var_floor, var_cap)
        mu_h0 = tf.squeeze(price_h0, axis=1)
        mu_h1 = tf.squeeze(price_h1, axis=1)
        mu_h2 = tf.squeeze(price_h2, axis=1)
        # P(up | the move left the deadband): matches the masked labels (see losses.gaussian_up_prob_given_move).
        gauss_p_up_h0 = _losses.gaussian_up_prob_given_move(mu_h0, var_h0_c, last_close_squeeze, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h1 = _losses.gaussian_up_prob_given_move(mu_h1, var_h1_c, last_close_squeeze, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h2 = _losses.gaussian_up_prob_given_move(mu_h2, var_h2_c, last_close_squeeze, deadband, self.pred_mean, self.pred_scale, self.eps)

        # IMPORTANT: do NOT prefix with "val_" here. Keras automatically prefixes
        # validation metrics with "val_"; adding it ourselves creates "val_val_*" keys.
        return self._epoch_logs(
            loss_components, y_true, y_pred_9,
            (true_dir_h0, true_dir_h1, true_dir_h2),
            (dir_pred_h0, dir_pred_h1, dir_pred_h2),
            (gauss_p_up_h0, gauss_p_up_h1, gauss_p_up_h2),
            (mask_h0, mask_h1, mask_h2),
            head_prefix="", gauss_prefix="gauss_", training=False,
        )


    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'pred_scale': float(self.pred_scale.numpy()) if isinstance(self.pred_scale, tf.Tensor) else float(self.pred_scale),
            'pred_mean': float(self.pred_mean.numpy()) if isinstance(self.pred_mean, tf.Tensor) else float(self.pred_mean),
            'lambda_point': float(self.lambda_point),
            'lambda_local_trend': float(self.lambda_local_trend),
            'lambda_global_trend': float(self.lambda_global_trend),
            'lambda_extended_trend': float(self.lambda_extended_trend),
            'lambda_dir': float(self.lambda_dir)
        })
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config_instance = Config()
        base_model = Models.build(config_instance.MODEL_NAME, config_instance)
        pred_scale = config.pop('pred_scale', 1.0)
        pred_mean = config.pop('pred_mean', 0.0)
        lambda_point = config.pop('lambda_point', 1.0)
        lambda_local_trend = config.pop('lambda_local_trend', 1.0)
        lambda_global_trend = config.pop('lambda_global_trend', 0.2)
        lambda_extended_trend = config.pop('lambda_extended_trend', 0.16)
        lambda_dir = config.pop('lambda_dir', 1.0)
        instance = cls(base_model=base_model,
                       pred_scale=pred_scale,
                       pred_mean=pred_mean,
                       lambda_point=lambda_point,
                       lambda_local_trend=lambda_local_trend,
                       lambda_global_trend=lambda_global_trend,
                       lambda_extended_trend=lambda_extended_trend,
                       lambda_dir=lambda_dir,
                       config=config_instance,
                       **config)
        return instance


install_lambda_properties(CustomTrainModel)

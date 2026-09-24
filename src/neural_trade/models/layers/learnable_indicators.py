"""LearnableIndicators layer (moved from model.py in Phase B6)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import initializers, layers, regularizers

import neural_trade.utils.math as mh

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from neural_trade.core.config import Config


class LearnableIndicators(layers.Layer):
    """18 learnable EWMA periods -> 31 indicator channels (MA, MACD, RSI, Bollinger, raw).

    Each period is a trainable logit (alpha = sigmoid(logit), period = 2/alpha - 1),
    adjusted per sample by ``meta_adjust`` and trained through a straight-through
    gradient multiplier. Inputs: ``[close_window [B, L], meta_adjust [B, 18]]``;
    output ``[B, L, 31]``.
    """

    def __init__(self, config: "Config", **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.epsilon = 1e-8
        self.alpha_vars_ma = []
        self.macd_alpha_vars = {}
        self.rsi_alpha_vars = []
        self.bb_alpha_vars = []
        self.all_logit_vars = []  # New: collect all for metacalibration
        self.meta_scale = 0.5  # Increased from 0.1 for stronger adjustments
        self.grad_multiplier = config.INDICATOR_GRAD_MULT  # Apply gradient boost

    # Period <-> alpha <-> logit transforms: one definition, in neural_trade.utils.math.
    def _logit_from_alpha(self, alpha):
        return mh.logit_from_alpha(alpha, self.epsilon)

    def _alpha_from_logit(self, logit):
        return mh.alpha_from_logit(logit)

    def _logit_from_period(self, period):
        return mh.logit_from_period(period, self.epsilon)

    def _period_from_logit(self, logit):
        return mh.period_from_logit(logit, self.epsilon)

    def build(self, input_shape):
        # input_shape[0] is close_seq, [1] is meta_adjust [B, num_logits]
        for i, s in enumerate(self.config.MA_SPANS):
            init_logit = self._logit_from_period(s)
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(init_logit),
                                trainable=True,
                                name=f'alpha_ma_{i}',
                                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            self.alpha_vars_ma.append(v)
            self.all_logit_vars.append(v)

        for i, settings in enumerate(self.config.MACD_SETTINGS):
            v_fast = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['fast'])),
                trainable=True,
                name=f'macd_{i}_fast',
                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            v_slow = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['slow'])),
                trainable=True,
                name=f'macd_{i}_slow',
                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            v_signal = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['signal'])),
                trainable=True,
                name=f'macd_{i}_signal',
                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            self.macd_alpha_vars[f'macd_{i}_fast'] = v_fast
            self.macd_alpha_vars[f'macd_{i}_slow'] = v_slow
            self.macd_alpha_vars[f'macd_{i}_signal'] = v_signal
            self.all_logit_vars.extend([v_fast, v_slow, v_signal])

        for i, p in enumerate(self.config.RSI_PERIODS):
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(self._logit_from_period(p)),
                                trainable=True,
                                name=f'rsi_alpha_{i}',
                                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            self.rsi_alpha_vars.append(v)
            self.all_logit_vars.append(v)

        for i, p in enumerate(self.config.BB_PERIODS):
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(self._logit_from_period(p)),
                                trainable=True,
                                name=f'bb_alpha_{i}',
                                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            self.bb_alpha_vars.append(v)
            self.all_logit_vars.append(v)

        super().build(input_shape)

    def ewma_seq(self, x_seq, alpha_scalar):
        if str(getattr(self.config, 'EWMA_IMPL', 'matrix')).lower() == 'scan':
            return mh.ewma_sequence(x_seq, alpha_scalar)
        return mh.ewma_sequence_matrix(x_seq, alpha_scalar)

    def _alpha(self, logit, meta_adjust, col):
        """Per-sample alpha for one learned period: STE-scaled logit + the meta adjustment."""
        # Correct STE gradient trick: forward=logit (unchanged), backward=logit * grad_multiplier
        logit_for_alpha = (self.grad_multiplier * logit
                           - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
        return self._alpha_from_logit(logit_for_alpha + meta_adjust[:, col] * self.meta_scale)

    def call(self, inputs, training=None):
        x, meta_adjust = inputs
        x = tf.cast(x, tf.float32)
        if str(getattr(self.config, 'EWMA_IMPL', 'matrix')).lower() == 'scan':
            return self._call_per_indicator(x, meta_adjust)
        return self._call_batched(x, meta_adjust)

    def _call_batched(self, x, meta_adjust):
        """All 24 moving averages as two batched matrix products (Config.EWMA_IMPL = "matrix").

        Stage 1: the 18 averages of the input or its gains/losses (MA, MACD fast/slow, Bollinger
        mean, RSI gains/losses). Stage 2: the 6 that depend on stage 1 (MACD signal, Bollinger
        variance). Same features, order and meta-adjust columns as ``_call_per_indicator``; the two
        agree to float32 round-off (tests/test_learnable_indicators.py).
        """
        n_ma, n_macd = len(self.alpha_vars_ma), len(self.config.MACD_SETTINGS)
        n_rsi, n_bb = len(self.rsi_alpha_vars), len(self.bb_alpha_vars)
        col = 0
        ma = [self._alpha(v, meta_adjust, col + i) for i, v in enumerate(self.alpha_vars_ma)]
        col += n_ma
        macd = []
        for i in range(n_macd):
            macd.append(tuple(self._alpha(self.macd_alpha_vars[f'macd_{i}_{part}'], meta_adjust, col + j)
                              for j, part in enumerate(('fast', 'slow', 'signal'))))
            col += 3
        rsi = [self._alpha(v, meta_adjust, col + i) for i, v in enumerate(self.rsi_alpha_vars)]
        col += n_rsi
        bb = [self._alpha(v, meta_adjust, col + i) for i, v in enumerate(self.bb_alpha_vars)]

        diffs = x[:, 1:] - x[:, :-1]
        zero = tf.zeros((tf.shape(x)[0], 1), dtype=x.dtype)
        gains = tf.concat([zero, tf.where(diffs > 0, diffs, tf.zeros_like(diffs))], axis=1)
        losses = tf.concat([zero, tf.where(diffs < 0, -diffs, tf.zeros_like(diffs))], axis=1)

        seqs = [x] * (n_ma + 2 * n_macd + n_bb) + [gains] * n_rsi + [losses] * n_rsi
        alphas = ma + [f for f, _, _ in macd] + [s_ for _, s_, _ in macd] + bb + rsi + rsi
        e1 = mh.ewma_sequence_matrix_multi(tf.stack(seqs, axis=1), tf.stack(alphas, axis=1))
        e1 = tf.unstack(e1, axis=1)
        o = 0
        ema_ma, o = e1[o:o + n_ma], o + n_ma
        ema_fast, o = e1[o:o + n_macd], o + n_macd
        ema_slow, o = e1[o:o + n_macd], o + n_macd
        bb_mean, o = e1[o:o + n_bb], o + n_bb
        gains_ema, o = e1[o:o + n_rsi], o + n_rsi
        losses_ema = e1[o:o + n_rsi]

        macd_lines = [f - s_ for f, s_ in zip(ema_fast, ema_slow)]
        sq_devs = [tf.square(x - m) for m in bb_mean]
        e2 = mh.ewma_sequence_matrix_multi(tf.stack(macd_lines + sq_devs, axis=1),
                                           tf.stack([g for _, _, g in macd] + bb, axis=1))
        e2 = tf.unstack(e2, axis=1)
        macd_sig, bb_var = e2[:n_macd], e2[n_macd:]

        features = list(ema_ma)
        for line, sig in zip(macd_lines, macd_sig):
            hist = line - sig
            features.extend([line, sig, hist, tf.tanh(hist * 10.0)])
        for g, l_ in zip(gains_ema, losses_ema):
            features.append(100.0 - (100.0 / (1.0 + g / (l_ + 1e-8))))
        for mean, var in zip(bb_mean, bb_var):
            std = tf.sqrt(var + 1e-8)
            features.extend([mean, mean + 2.0 * std, mean - 2.0 * std, (x - (mean - 2.0 * std)) / (4.0 * std + 1e-8)])
        features.append(x)
        output = tf.stack(features, axis=-1)
        output.set_shape([None, self.config.LOOKBACK, len(features)])
        return output

    def _call_per_indicator(self, x, meta_adjust):
        """The reference form: one EWMA call per indicator (used with EWMA_IMPL = "scan")."""
        features = []
        idx = 0  # Index for slicing meta_adjust

        for logit in self.alpha_vars_ma:
            # Correct STE gradient trick: forward=logit (unchanged), backward=logit * grad_multiplier
            # Old (broken): logit + stop_grad(logit)*(k-1) => forward=logit*k (saturates sigmoid!)
            # New (correct): k*logit - stop_grad((k-1)*logit) => forward=logit, backward=k
            logit_for_alpha = (self.grad_multiplier * logit
                               - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
            adjusted_logit = logit_for_alpha + meta_adjust[:, idx] * self.meta_scale
            alpha = self._alpha_from_logit(adjusted_logit)
            ema_seq = self.ewma_seq(x, alpha)
            features.append(ema_seq)
            idx += 1

        for i in range(len(self.config.MACD_SETTINGS)):
            fast_var = self.macd_alpha_vars[f'macd_{i}_fast']
            slow_var = self.macd_alpha_vars[f'macd_{i}_slow']
            sig_var = self.macd_alpha_vars[f'macd_{i}_signal']
            # Correct STE gradient trick (see MA block above for explanation)
            fast_for_alpha = (self.grad_multiplier * fast_var
                              - tf.stop_gradient((self.grad_multiplier - 1.0) * fast_var))
            slow_for_alpha = (self.grad_multiplier * slow_var
                              - tf.stop_gradient((self.grad_multiplier - 1.0) * slow_var))
            sig_for_alpha  = (self.grad_multiplier * sig_var
                              - tf.stop_gradient((self.grad_multiplier - 1.0) * sig_var))
            fast_logit = fast_for_alpha + meta_adjust[:, idx] * self.meta_scale
            slow_logit = slow_for_alpha + meta_adjust[:, idx+1] * self.meta_scale
            sig_logit  = sig_for_alpha  + meta_adjust[:, idx+2] * self.meta_scale
            fast = self._alpha_from_logit(fast_logit)
            slow = self._alpha_from_logit(slow_logit)
            sig = self._alpha_from_logit(sig_logit)
            ema_f = self.ewma_seq(x, fast)
            ema_s = self.ewma_seq(x, slow)
            macd_line = ema_f - ema_s
            macd_sig = self.ewma_seq(macd_line, sig)
            macd_hist = macd_line - macd_sig
            features.extend([macd_line, macd_sig, macd_hist])
            # Soft sign: tf.sign has zero gradient; tanh*10 is a differentiable approximation
            macd_cross = tf.tanh(macd_hist * 10.0)
            features.append(macd_cross)
            idx += 3

        diffs = x[:, 1:] - x[:, :-1]
        gains = tf.where(diffs > 0, diffs, tf.zeros_like(diffs))
        losses = tf.where(diffs < 0, -diffs, tf.zeros_like(diffs))
        gains_padded = tf.concat([tf.zeros((tf.shape(gains)[0], 1), dtype=gains.dtype), gains], axis=1)
        losses_padded = tf.concat([tf.zeros((tf.shape(losses)[0], 1), dtype=losses.dtype), losses], axis=1)

        for logit in self.rsi_alpha_vars:
            # Correct STE gradient trick (see MA block above for explanation)
            logit_for_alpha = (self.grad_multiplier * logit
                               - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
            adjusted_logit = logit_for_alpha + meta_adjust[:, idx] * self.meta_scale
            rsi_alpha = self._alpha_from_logit(adjusted_logit)
            gains_ema = self.ewma_seq(gains_padded, rsi_alpha)
            losses_ema = self.ewma_seq(losses_padded, rsi_alpha)
            rs = gains_ema / (losses_ema + 1e-8)
            rsi_seq = 100.0 - (100.0 / (1.0 + rs))
            features.append(rsi_seq)
            idx += 1

        for logit in self.bb_alpha_vars:
            # Correct STE gradient trick (see MA block above for explanation)
            logit_for_alpha = (self.grad_multiplier * logit
                               - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
            adjusted_logit = logit_for_alpha + meta_adjust[:, idx] * self.meta_scale
            bb_alpha = self._alpha_from_logit(adjusted_logit)
            ema_mean = self.ewma_seq(x, bb_alpha)
            sq_dev = tf.square(x - ema_mean)
            ema_var = self.ewma_seq(sq_dev, bb_alpha)
            ema_std = tf.sqrt(ema_var + 1e-8)
            features.extend([ema_mean, ema_mean + 2.0 * ema_std, ema_mean - 2.0 * ema_std])
            # Add Bollinger %B
            bb_percent = (x - (ema_mean - 2.0 * ema_std)) / (4.0 * ema_std + 1e-8)
            features.append(bb_percent)
            idx += 1

        features.append(x)  # Add raw close as a "indicator" sequence

        output = tf.stack(features, axis=-1)  # [B, LOOKBACK, num_features]
        n_features = len(features)
        tf.ensure_shape(output, [None, self.config.LOOKBACK, n_features])
        output.set_shape([None, self.config.LOOKBACK, n_features])
        return output

    def get_learned_parameters(self):
        learned = {}
        for i, v in enumerate(self.alpha_vars_ma):
            period = self._period_from_logit(v).numpy()
            learned[f'ma_period_{i}'] = float(period)
        for k, v in self.macd_alpha_vars.items():
            period = self._period_from_logit(v).numpy()
            learned[f'{k}'] = float(period)
        for i, v in enumerate(self.rsi_alpha_vars):
            period = self._period_from_logit(v).numpy()
            learned[f'rsi_period_{i}'] = float(period)
        for i, v in enumerate(self.bb_alpha_vars):
            period = self._period_from_logit(v).numpy()
            learned[f'bb_period_{i}'] = float(period)
        return learned

    def get_indicator_trainable_variables(self):
        """Return all trainable logit/period variables owned by this indicator layer.

        Used by CustomTrainModel for robust (id-based, not string-based) routing
        of gradients to the dedicated indicator optimizer and for clipping.
        """
        vs = list(self.alpha_vars_ma)
        vs.extend(self.macd_alpha_vars.values())
        vs.extend(self.rsi_alpha_vars)
        vs.extend(self.bb_alpha_vars)
        return vs

    def clip_learned_periods(self, min_p, max_p):
        """Clip every learned period into [min_p, max_p] by clipping its LOGIT.

        Clipping happens in logit space (no period -> logit -> period round trip), so a
        clip can never write a saturated logit. The old round trip with
        MOMENTUM_CLIP_MIN = 1.0 mapped period 1 to alpha = 1 and logit ~ +18.4, where the
        float32 sigmoid is exactly 1.0 and its derivative exactly 0: any indicator that
        touched the bound was frozen for the rest of training. Logit is decreasing in
        period, so the period floor is the logit ceiling.
        Called from CustomTrainModel.train_step after the optimizer step.
        """
        logit_hi = self._logit_from_period(tf.cast(min_p, tf.float32))
        logit_lo = self._logit_from_period(tf.cast(max_p, tf.float32))
        for var in self.get_indicator_trainable_variables():
            var.assign(tf.clip_by_value(var, logit_lo, logit_hi))

        # Legacy momentum_raw support (if any such vars exist on the layer)
        # The original clipping lived in train_step string checks; we keep the
        # logic here for encapsulation even if 'momentum_raw' is no longer primary.
        for var in getattr(self, 'momentum_raw_vars', []):
            p = tf.nn.softplus(var) + 1.0
            clipped = tf.clip_by_value(p, min_p, max_p)
            raw = tf.math.asinh((clipped - 1.0) / 2.0)
            var.assign(raw)

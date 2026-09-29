"""LearnableIndicators layer: assembles the registered indicator families (NT-046).

The families come from the Indicators registry (``neural_trade.indicators``); the config
lists each family's instances (``indicator_instances``: MA_SPANS / MACD_SETTINGS /
RSI_PERIODS / BB_PERIODS plus ``INDICATOR_FAMILIES``). With the defaults this is exactly the
pre-NT-046 layer: 18 learnable EWMA periods -> 31 channels (MA, MACD, RSI, Bollinger, raw
close), pinned by tests/test_learnable_indicators.py and scripts/golden_run.py.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import initializers, layers, regularizers

import neural_trade.utils.math as mh
from neural_trade.indicators import FamilyContext, Indicators, indicator_instances

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from neural_trade.core.config import Config


class LearnableIndicators(layers.Layer):
    """Learnable indicator channels built from the configured family instances.

    Each family parameter is a trainable logit (alpha = sigmoid(logit), period = 2/alpha - 1),
    adjusted per sample by ``meta_adjust`` (off with ``Config.ADAPTIVE_INDICATORS = False``:
    every applied period then equals the learned global value) and trained through a
    straight-through gradient multiplier. Inputs: ``[close_window [B, L],
    meta_adjust [B, num_logits]]``; output ``[B, L, num_channels + 1]`` (raw close appended).
    """

    def __init__(self, config: "Config", **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.epsilon = 1e-8
        # (family, parsed instances, [{param: logit variable}]) in configured order
        self._families = []
        # legacy views of the four families' variables (telemetry and older callers)
        self.alpha_vars_ma = []
        self.macd_alpha_vars = {}
        self.rsi_alpha_vars = []
        self.bb_alpha_vars = []
        self.all_logit_vars = []  # every logit, in creation order (metacalibration, optimizer)
        self.meta_scale = 0.5  # == neural_trade.indicators.META_SCALE (kept as attribute)
        self.grad_multiplier = config.INDICATOR_GRAD_MULT  # Apply gradient boost
        self.adaptive = bool(getattr(config, "ADAPTIVE_INDICATORS", True))

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
        for name, raw in indicator_instances(self.config).items():
            family = Indicators.get(name)
            insts = [family.parse_instance(v) for v in raw]
            varmaps = []
            for i, inst in enumerate(insts):
                vm = {}
                for p in family.params:
                    v = self.add_weight(shape=(),
                                        initializer=initializers.Constant(
                                            self._logit_from_period(inst[p.name])),
                                        trainable=True,
                                        name=family.logit_name(i, p.name),
                                        regularizer=regularizers.L2(self.config.INDICATOR_L2))
                    vm[p.name] = v
                    self.all_logit_vars.append(v)
                varmaps.append(vm)
            self._families.append((family, insts, varmaps))
            if name == "ma":
                self.alpha_vars_ma = [vm["period"] for vm in varmaps]
            elif name == "macd":
                self.macd_alpha_vars = {f"macd_{i}_{p}": vm[p] for i, vm in enumerate(varmaps)
                                        for p in ("fast", "slow", "signal")}
            elif name == "rsi":
                self.rsi_alpha_vars = [vm["period"] for vm in varmaps]
            elif name == "bb":
                self.bb_alpha_vars = [vm["period"] for vm in varmaps]

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
        if not self.adaptive:
            # multiply by zero (instead of dropping the input) so the meta network keeps a
            # defined, exactly-zero gradient: logit + 0.0 * meta == logit bit-for-bit
            meta_adjust = meta_adjust * 0.0
        if str(getattr(self.config, 'EWMA_IMPL', 'matrix')).lower() == 'scan':
            return self._call_per_indicator(x, meta_adjust)
        return self._call_batched(x, meta_adjust)

    # ------------------------------------------------------------------ the engine
    def _plans(self, meta_adjust):
        """One (family, {param: alpha [B]}) plan per configured instance, in column order."""
        plans = []
        col = 0
        for family, _insts, varmaps in self._families:
            for vm in varmaps:
                alphas = {p.name: self._alpha(vm[p.name], meta_adjust, col + j)
                          for j, p in enumerate(family.params)}
                col += len(family.params)
                plans.append((family, alphas))
        return plans

    @staticmethod
    def _stage_rows(plans, stage_of):
        """The stage's requests as flat rows ``(plan index, key, seq, alpha)``.

        Rows are grouped family by family and, inside a family, request key by request key
        across its instances (all MACD 'fast' rows, then all 'slow' rows, ...). This exact
        order, not the per-instance one, reproduces the pre-NT-046 gradient aggregation
        order bit-for-bit (the determinism NOTE in ``neural_trade.indicators.base``)."""
        rows = []
        for group in plans:  # one group of (plan index, family, alphas, cache) per family
            per_inst = [stage_of(entry) for entry in group]
            for key in (per_inst[0] if per_inst else ()):
                for (k, _, _, _), reqs in zip(group, per_inst):
                    seq, a = reqs[key]
                    rows.append((k, key, seq, a))
        # Rows that read the same series are batched adjacently, in the order the series
        # first appears (a stable sort): with the default families this is the pre-NT-046
        # layout (the close rows: MA, MACD fast, MACD slow, Bollinger mean; then the RSI
        # gains rows; then the losses rows), part of reproducing it bit-for-bit.
        first_seen = {}
        for _, _, seq, _ in rows:
            first_seen.setdefault(id(seq), len(first_seen))
        rows.sort(key=lambda row: first_seen[id(row[2])])
        return rows

    def _run(self, x, meta_adjust, batched: bool):
        """Assemble every family's channels; ``batched`` batches each EWMA stage into one
        matrix product (Config.EWMA_IMPL = "matrix"), otherwise one scan per request.

        The two agree to float32 round-off (tests/test_learnable_indicators.py); a batched
        row's value does not depend on its position (each EWMA only reads its own sequence
        and alpha)."""
        ctx = FamilyContext(x)
        flat = self._plans(meta_adjust)
        caches = [{} for _ in flat]
        groups, k = [], 0
        for _family, _insts, varmaps in self._families:
            groups.append([(k + j, flat[k + j][0], flat[k + j][1], caches[k + j])
                           for j in range(len(varmaps))])
            k += len(varmaps)

        def run_stage(rows):
            if not rows:
                return []
            if batched:
                e = mh.ewma_sequence_matrix_multi(tf.stack([seq for _, _, seq, _ in rows], axis=1),
                                                  tf.stack([a for _, _, _, a in rows], axis=1))
                return tf.unstack(e, axis=1)
            return [mh.ewma_sequence(seq, a) for _, _, seq, a in rows]

        s1 = [{} for _ in flat]
        rows1 = self._stage_rows(groups, lambda e: e[1].stage1(ctx, e[2], e[3]))
        for (k, key, _, _), res in zip(rows1, run_stage(rows1)):
            s1[k][key] = res

        s2 = [{} for _ in flat]
        rows2 = self._stage_rows(groups, lambda e: e[1].stage2(ctx, e[2], s1[e[0]], e[3]))
        for (k, key, _, _), res in zip(rows2, run_stage(rows2)):
            s2[k][key] = res

        features = []
        for k, (family, alphas) in enumerate(flat):
            features.extend(family.outputs(ctx, alphas, s1[k], s2[k], caches[k]))
        features.append(x)  # raw close as the last "indicator" sequence
        output = tf.stack(features, axis=-1)
        output.set_shape([None, self.config.LOOKBACK, len(features)])
        return output

    def _call_batched(self, x, meta_adjust):
        """All EWMAs as two batched matrix products (Config.EWMA_IMPL = "matrix")."""
        return self._run(x, meta_adjust, batched=True)

    def _call_per_indicator(self, x, meta_adjust):
        """The reference form: one EWMA call per request (used with EWMA_IMPL = "scan")."""
        return self._run(x, meta_adjust, batched=False)

    # ------------------------------------------------------------------ introspection
    def get_learned_parameters(self):
        """Learned period per logit, keyed as telemetry reports them (``ma_period_0``, ...)."""
        learned = {}
        for family, _insts, varmaps in self._families:
            for i, vm in enumerate(varmaps):
                for p in family.params:
                    period = self._period_from_logit(vm[p.name]).numpy()
                    learned[family.learned_name(i, p.name)] = float(period)
        return learned

    def get_indicator_trainable_variables(self):
        """Return all trainable logit/period variables owned by this indicator layer.

        Used by CustomTrainModel for robust (id-based, not string-based) routing
        of gradients to the dedicated indicator optimizer and for clipping.
        """
        return list(self.all_logit_vars)

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

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
from neural_trade.indicators import (DERIVED_SERIES, FamilyContext, Indicators,
                                     indicator_instances)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from neural_trade.core.config import Config

# ------------------------------------------------------------------ NT-106 repair round 1
# A ParamSpec.kind == "ratio" learnable parameter (today only MACDRatioFamily's 'ratio') is a
# dimensionless fraction r in (0, 1), not a period: clipping it through the period<->logit
# transform (as clip_learned_periods and the bound_applied path in _alpha do for every other
# learnable parameter) silently re-floors it in PERIOD space - QA of NT-106 measured r pinned
# to [0.0328, 0.6667] at the MOMENTUM_CLIP_MIN/MAX defaults (2, 60), which never let the fast
# leg of MACDRatioFamily below about 1.82, defeating the item's purpose (B_model_indicators.md
# 4.1: fast legs pinned at the floor of 2). A ratio parameter gets its OWN bound instead,
# independent of MOMENTUM_CLIP_MIN/MAX: r in [_RATIO_CLIP_MIN, _RATIO_CLIP_MAX].
_RATIO_CLIP_MIN = 1e-3  # eps_r: keeps r (and so every ratio-derived period) away from 0
_RATIO_CLIP_MAX = 1.0 - 2e-4  # keeps r below 1 by the same margin families._MACD_RATIO_EPS
                              # uses, so a ratio-derived period stays float32-distinguishable
                              # from whatever it is a fraction of (see that module's comment)


class LearnableIndicators(layers.Layer):
    """Learnable indicator channels built from the configured family instances.

    Each family parameter is a trainable logit (alpha = sigmoid(logit), period = 2/alpha - 1),
    adjusted per sample by ``meta_adjust`` (off with ``Config.ADAPTIVE_INDICATORS = False``:
    every applied period then equals the learned global value) and trained through a
    straight-through gradient multiplier. Inputs: ``[window, meta_adjust [B, num_logits]]``
    where the window is the close sequence ``[B, L]`` (``Config.INPUT_SERIES = ['close']``,
    the pre-NT-047 input) or the multi-series window ``[B, L, len(INPUT_SERIES)]``; output
    ``[B, L, num_channels + 1]`` (the raw close sequence appended).
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
        # NT-097 (B_model_indicators.md 2.2, 4.2, 7.5): off (default) reproduces today's behaviour,
        # where only the base logit is clipped to [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX]
        # (CustomTrainModel.train_step -> clip_learned_periods) and the per-window meta shift can
        # push the APPLIED period outside that bound (measured 1.6-1.8 below the floor of 2, up to
        # 74.7 above a ceiling of 60). On, the combined logit (base + shift) is clipped into the
        # same bound before the sigmoid, every forward pass.
        self.bound_applied = bool(getattr(config, "INDICATOR_BOUND_APPLIED", False))
        self._applied_logit_lo = None  # set in build(): logit(period=MOMENTUM_CLIP_MAX)
        self._applied_logit_hi = None  # set in build(): logit(period=MOMENTUM_CLIP_MIN)

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
        # input_shape[0] is the window ([B, L] close or [B, L, C] multi-series),
        # [1] is meta_adjust [B, num_logits]
        available = set(getattr(self.config, "INPUT_SERIES", None) or ["close"])
        macd_ratio = (str(getattr(self.config, "MACD_PARAM", "independent")).lower() == "ratio")
        for name, raw in indicator_instances(self.config).items():
            # NT-106: Config.MACD_PARAM = "ratio" substitutes the ratio-parametrised family for
            # the "macd" instances (MACD_SETTINGS); off (default) resolves "macd" as always, so
            # the default path is byte-for-byte today's lookup.
            registry_name = "macd_ratio" if (name == "macd" and macd_ratio) else name
            family = Indicators.get(registry_name)
            missing = sorted(set(family.inputs) - available
                             - {d for d, req in DERIVED_SERIES.items() if set(req) <= available})
            if missing:
                raise ValueError(
                    f"indicator family '{name}' reads {missing}, which Config.INPUT_SERIES "
                    f"{sorted(available)} does not carry (NT-047)")
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
                param_names = ("slow", "ratio", "signal") if macd_ratio else ("fast", "slow", "signal")
                self.macd_alpha_vars = {f"macd_{i}_{p}": vm[p] for i, vm in enumerate(varmaps)
                                        for p in param_names}
            elif name == "rsi":
                self.rsi_alpha_vars = [vm["period"] for vm in varmaps]
            elif name == "bb":
                self.bb_alpha_vars = [vm["period"] for vm in varmaps]

        if self.bound_applied:
            min_p = float(self.config.MOMENTUM_CLIP_MIN)
            max_p = float(getattr(self.config, "MOMENTUM_CLIP_MAX", None) or self.config.LOOKBACK)
            # logit is decreasing in period: the period floor is the logit ceiling and vice versa
            # (the same convention as clip_learned_periods below).
            self._applied_logit_hi = float(self._logit_from_period(min_p).numpy())
            self._applied_logit_lo = float(self._logit_from_period(max_p).numpy())

        super().build(input_shape)

    def ewma_seq(self, x_seq, alpha_scalar):
        if str(getattr(self.config, 'EWMA_IMPL', 'matrix')).lower() == 'scan':
            return mh.ewma_sequence(x_seq, alpha_scalar)
        return mh.ewma_sequence_matrix(x_seq, alpha_scalar)

    def _alpha(self, logit, meta_adjust, col, kind="period"):
        """Per-sample alpha for one learned parameter: STE-scaled logit + the meta adjustment.

        ``kind`` (``ParamSpec.kind``, NT-106) says what the resulting value MEANS:

        * ``"period"`` (default): with ``INDICATOR_BOUND_APPLIED`` on (NT-097), the combined
          logit is clipped into [logit(MOMENTUM_CLIP_MAX), logit(MOMENTUM_CLIP_MIN)] first, so
          the APPLIED period this alpha implies can never leave
          [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX] even though the shift itself is unbounded;
          off (default), this reproduces today's behaviour exactly.
        * ``"ratio"``: the result is a dimensionless fraction r in (0, 1), not a period, so
          ``INDICATOR_BOUND_APPLIED`` (if on) clips r itself into
          [_RATIO_CLIP_MIN, _RATIO_CLIP_MAX] instead of period-clipping the logit (repair round
          1 of NT-106: clipping the ratio logit as a period logit re-floored it, so r never
          reached near 0 and the fast leg this parametrisation exists for never reached p = 1).
        """
        # Correct STE gradient trick: forward=logit (unchanged), backward=logit * grad_multiplier
        logit_for_alpha = (self.grad_multiplier * logit
                           - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
        combined = logit_for_alpha + meta_adjust[:, col] * self.meta_scale
        if kind == "ratio":
            alpha = self._alpha_from_logit(combined)
            if self.bound_applied:
                alpha = tf.clip_by_value(alpha, _RATIO_CLIP_MIN, _RATIO_CLIP_MAX)
            return alpha
        if self.bound_applied:
            combined = tf.clip_by_value(combined, self._applied_logit_lo, self._applied_logit_hi)
        return self._alpha_from_logit(combined)

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
                alphas = {p.name: self._alpha(vm[p.name], meta_adjust, col + j,
                                              getattr(p, "kind", "period"))
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

    def _context(self, x):
        """The FamilyContext of the input window: ``[B, L]`` is the close sequence alone;
        ``[B, L, C]`` carries Config.INPUT_SERIES in order (NT-047)."""
        if x.shape.rank == 2:
            return FamilyContext(x)
        series = list(getattr(self.config, "INPUT_SERIES", None) or ["close"])
        cols = {name: x[:, :, i] for i, name in enumerate(series)}
        return FamilyContext(cols.pop("close"), **cols)

    def _run(self, x, meta_adjust, batched: bool):
        """Assemble every family's channels; ``batched`` batches each EWMA stage into one
        matrix product (Config.EWMA_IMPL = "matrix"), otherwise one scan per request.

        The two agree to float32 round-off (tests/test_learnable_indicators.py); a batched
        row's value does not depend on its position (each EWMA only reads its own sequence
        and alpha)."""
        ctx = self._context(x)
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
        features.append(ctx.close)  # raw close as the last "indicator" sequence
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
    @staticmethod
    def _report_entries(family, index, alphas):
        """One instance's reporting entries, keyed as telemetry reports them.

        ``alphas`` carries the per-param alpha of this instance (``{param name: tensor}``,
        scalar or per-window), computed the SAME way by both callers below. A family with its
        own ``applied_report(index, alphas)`` hook (NT-106: a derived parametrisation, such as
        MACDRatioFamily's fast = r x slow) reports its own, interpretable names instead of the
        identity map; every other family (unchanged) reports ``learned_name(i, p): 2/alpha - 1``
        exactly as before NT-106.
        """
        hook = getattr(family, "applied_report", None)
        if hook is not None:
            return hook(index, alphas)
        return {family.learned_name(index, p.name): tf.maximum(2.0 / (alphas[p.name] + 1e-8) - 1.0, 0.0)
                for p in family.params}

    def get_learned_parameters(self):
        """Learned period per logit, keyed as telemetry reports them (``ma_period_0``, ...)."""
        learned = {}
        for family, _insts, varmaps in self._families:
            for i, vm in enumerate(varmaps):
                alphas = {p.name: self._alpha_from_logit(vm[p.name]) for p in family.params}
                for key, value in self._report_entries(family, i, alphas).items():
                    learned[key] = float(tf.convert_to_tensor(value).numpy())
        return learned

    def applied_period_samples(self, meta_adjust):
        """Per-window APPLIED period for every learnable parameter (NT-097 point 5, 7).

        ``meta_adjust`` is ``[N, num_logits]`` (the model's ``meta_adjust`` tensor evaluated on a
        batch of real windows, ``neural_trade.evaluation.applied_periods``). Reuses ``_alpha``
        exactly as the forward pass does (respecting ``ADAPTIVE_INDICATORS`` and
        ``INDICATOR_BOUND_APPLIED``), so the numbers match what the model actually applied, not
        just the logged base period of ``get_learned_parameters``. Returns
        ``{name: np.ndarray[N]}`` of applied periods (``2/alpha - 1``, or a family's own
        ``applied_report`` conversion, NT-106).
        """
        meta_adjust = tf.cast(meta_adjust, tf.float32)
        if not self.adaptive:
            meta_adjust = meta_adjust * 0.0
        out = {}
        col = 0
        for family, _insts, varmaps in self._families:
            for i, vm in enumerate(varmaps):
                alphas = {}
                for p in family.params:
                    alphas[p.name] = self._alpha(vm[p.name], meta_adjust, col,
                                                 getattr(p, "kind", "period"))
                    col += 1
                for key, value in self._report_entries(family, i, alphas).items():
                    out[key] = tf.convert_to_tensor(value).numpy()
        return out

    def get_indicator_trainable_variables(self):
        """Return all trainable logit/period variables owned by this indicator layer.

        Used by CustomTrainModel for robust (id-based, not string-based) routing
        of gradients to the dedicated indicator optimizer and for clipping.
        """
        return list(self.all_logit_vars)

    def clip_learned_periods(self, min_p, max_p):
        """Clip every learned PERIOD into [min_p, max_p] by clipping its LOGIT; a ``"ratio"``
        parameter (``ParamSpec.kind``, NT-106) is clipped as the fraction it is instead
        (``[_RATIO_CLIP_MIN, _RATIO_CLIP_MAX]``, independent of ``min_p`` / ``max_p``).

        Clipping happens in logit space (no period -> logit -> period round trip), so a
        clip can never write a saturated logit. The old round trip with
        MOMENTUM_CLIP_MIN = 1.0 mapped period 1 to alpha = 1 and logit ~ +18.4, where the
        float32 sigmoid is exactly 1.0 and its derivative exactly 0: any indicator that
        touched the bound was frozen for the rest of training. Logit is decreasing in
        period, so the period floor is the logit ceiling.

        Repair round 1 of NT-106: this used to clip EVERY variable (``get_indicator_trainable_
        variables()``, flat, kind-blind) through the period transform above; for a 'ratio'
        variable that re-floored r into the alpha interval [2/(max_p+1), 2/(min_p+1)] (about
        [0.033, 0.667] at the MOMENTUM_CLIP_MIN/MAX defaults), which never let the fast leg of
        MACDRatioFamily below about 1.82 - defeating the item's purpose. Branching by kind
        here (instead of by variable identity) leaves every existing 'period' variable's
        clip byte-for-byte unchanged (same bounds, same op, same per-variable independence, so
        the default path is unaffected regardless of iteration order).
        Called from CustomTrainModel.train_step after the optimizer step.
        """
        logit_hi = self._logit_from_period(tf.cast(min_p, tf.float32))
        logit_lo = self._logit_from_period(tf.cast(max_p, tf.float32))
        ratio_logit_hi = self._logit_from_alpha(tf.cast(_RATIO_CLIP_MAX, tf.float32))
        ratio_logit_lo = self._logit_from_alpha(tf.cast(_RATIO_CLIP_MIN, tf.float32))
        for family, _insts, varmaps in self._families:
            kind_of = {p.name: getattr(p, "kind", "period") for p in family.params}
            for vm in varmaps:
                for name, var in vm.items():
                    if kind_of[name] == "ratio":
                        var.assign(tf.clip_by_value(var, ratio_logit_lo, ratio_logit_hi))
                    else:
                        var.assign(tf.clip_by_value(var, logit_lo, logit_hi))

        # Legacy momentum_raw support (if any such vars exist on the layer)
        # The original clipping lived in train_step string checks; we keep the
        # logic here for encapsulation even if 'momentum_raw' is no longer primary.
        for var in getattr(self, 'momentum_raw_vars', []):
            p = tf.nn.softplus(var) + 1.0
            clipped = tf.clip_by_value(p, min_p, max_p)
            raw = tf.math.asinh((clipped - 1.0) / 2.0)
            var.assign(raw)

"""The indicator-family contract (D-031, NT-046).

One registry entry per family. An entry is an :class:`IndicatorFamily` instance that declares

* its **inputs** (which base series of the bar it reads: ``open`` / ``high`` / ``low`` /
  ``close`` / ``volume``; the derived series a :class:`FamilyContext` computes from them,
  such as ``gains`` / ``losses``, ``typical_price`` and ``true_range``, are declared through
  their base series),
* its **learnable parameters** (:class:`ParamSpec`: textbook default and bounds, in bars),
* its **output channels** (:class:`ChannelSpec`) and its **drawing spec** (``draw``: on the
  price chart or in its own panel),
* its **computation** as at most two stages of EWMA requests plus an output map (so the
  model layer can batch every EWMA of every family into two matrix calls, D-018), and
* **M(eps)** (D-037 amendment): the number of bars after which the state's dependence on its
  start is below ``eps``, computed from the current periods after the maximal per-window
  shift (for a single EWMA ``ceil(ln eps / ln(1 - alpha_min))``; cascades add their tails).

The config lists the instances of each family (three per family by default, NT-046);
``neural_trade.indicators.instances.indicator_instances`` reads them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import tensorflow as tf

#: the largest per-window logit shift ``meta_adjust`` can apply: LearnableIndicators adds
#: ``tanh(...) * META_SCALE`` to each logit, so the shift lies in [-META_SCALE, META_SCALE].
META_SCALE = 0.5

DRAW_TARGETS = ("price", "panel")

#: the base bar series a family may declare in ``inputs`` (order = channel order of the
#: model input; ``Config.INPUT_SERIES`` selects a subsequence containing at least 'close')
BASE_SERIES = ("open", "high", "low", "close", "volume")

#: derived context series -> the base series they are computed from (used by the layer's
#: input check; families declare only base series in ``inputs``)
DERIVED_SERIES = {
    "gains": ("close",),
    "losses": ("close",),
    "typical_price": ("high", "low", "close"),
    "true_range": ("high", "low", "close"),
}

# ------------------------------------------------------------------ soft extremum (NT-047)
#: inverse temperature of the Boltzmann weighting inside the rolling window, PER SERIES
#: UNIT (the model input is in target-scaler units, so this is a fixed sharpness in those
#: units; larger = closer to the hard rolling max / min)
SOFT_EXTREMUM_BETA = 4.0
#: exponent clip of the Boltzmann weights ``exp(clip(+-beta*(x - x_first)))``: bounds the
#: weights' dynamic range to e^(2*CLIP) so the float64 prefix-sum differences stay
#: well-conditioned for any input (values further than CLIP/BETA units from the window's
#: first value saturate the weighting toward an average of their region, never overflow)
SOFT_EXTREMUM_CLIP = 14.0
#: sharpness of the smooth sign / gate ``tanh(k*dx)`` / ``sigmoid(k*dx)`` used where the
#: textbook indicator branches on the sign of a one-bar move (OBV, MFI, ADX/DMI); the same
#: scale as the MACD soft cross of NT-046
SOFT_SIGN_SHARPNESS = 10.0


@dataclass(frozen=True)
class ParamSpec:
    """One learnable parameter of a family: an EWMA period in bars.

    ``default`` is the textbook value; ``minimum`` / ``maximum`` bound the learned period
    (``maximum=None``: the configured ceiling, ``MOMENTUM_CLIP_MAX`` or ``LOOKBACK`` today).
    """

    name: str
    default: float
    minimum: float = 2.0
    maximum: Optional[float] = None


@dataclass(frozen=True)
class ChannelSpec:
    """One output channel of a family instance and where it is drawn."""

    name: str
    draw: str = "panel"  # 'price' (overlaid on the price chart) or 'panel' (its own panel)


class FamilyContext:
    """The tensors one forward pass shares between the families.

    ``close`` is the (window-relative) close sequence ``[B, L]``. ``open`` / ``high`` /
    ``low`` / ``volume`` are the other bar series when the model input carries them
    (``Config.INPUT_SERIES``, NT-047); reading an absent one raises. Derived series are
    computed lazily and CACHED, so every consumer shares one op (the determinism NOTE
    below): ``gains`` / ``losses`` (zero-padded one-bar up/down close moves, as RSI reads
    them), ``prev_close`` (the close shifted one bar, first bar repeated),
    ``typical_price`` ((high + low + close) / 3) and ``true_range``
    (max(high - low, |high - prev_close|, |low - prev_close|); the first bar's true range
    is high - low).
    """

    def __init__(self, close: tf.Tensor, open: Optional[tf.Tensor] = None,  # noqa: A002
                 high: Optional[tf.Tensor] = None, low: Optional[tf.Tensor] = None,
                 volume: Optional[tf.Tensor] = None):
        self.close = close
        self._series = {"open": open, "high": high, "low": low, "volume": volume}
        self._cache: Dict[str, tf.Tensor] = {}

    def _base(self, name: str) -> tf.Tensor:
        t = self._series.get(name)
        if t is None:
            raise ValueError(f"the model input carries no {name!r} series: add it to "
                             "Config.INPUT_SERIES (NT-047)")
        return t

    @property
    def open(self) -> tf.Tensor:
        return self._base("open")

    @property
    def high(self) -> tf.Tensor:
        return self._base("high")

    @property
    def low(self) -> tf.Tensor:
        return self._base("low")

    @property
    def volume(self) -> tf.Tensor:
        return self._base("volume")

    def available(self) -> Tuple[str, ...]:
        """The base series this context carries (close always; the rest when present)."""
        return tuple(["close"] + [n for n in ("open", "high", "low", "volume")
                                  if self._series.get(n) is not None])

    def _split_moves(self) -> None:
        x = self.close
        diffs = x[:, 1:] - x[:, :-1]
        zero = tf.zeros((tf.shape(x)[0], 1), dtype=x.dtype)
        self._cache["gains"] = tf.concat(
            [zero, tf.where(diffs > 0, diffs, tf.zeros_like(diffs))], axis=1)
        self._cache["losses"] = tf.concat(
            [zero, tf.where(diffs < 0, -diffs, tf.zeros_like(diffs))], axis=1)

    @property
    def gains(self) -> tf.Tensor:
        if "gains" not in self._cache:
            self._split_moves()
        return self._cache["gains"]

    @property
    def losses(self) -> tf.Tensor:
        if "losses" not in self._cache:
            self._split_moves()
        return self._cache["losses"]

    @property
    def prev_close(self) -> tf.Tensor:
        if "prev_close" not in self._cache:
            x = self.close
            self._cache["prev_close"] = tf.concat([x[:, :1], x[:, :-1]], axis=1)
        return self._cache["prev_close"]

    @property
    def typical_price(self) -> tf.Tensor:
        if "typical_price" not in self._cache:
            self._cache["typical_price"] = (self.high + self.low + self.close) / 3.0
        return self._cache["typical_price"]

    @property
    def true_range(self) -> tf.Tensor:
        if "true_range" not in self._cache:
            pc = self.prev_close
            self._cache["true_range"] = tf.maximum(
                self.high - self.low,
                tf.maximum(tf.abs(self.high - pc), tf.abs(self.low - pc)))
        return self._cache["true_range"]


def alpha_from_period(period: float) -> float:
    """EWMA smoothing factor of a period in bars (``alpha = 2 / (period + 1)``)."""
    return 2.0 / (float(period) + 1.0)


def shifted_alpha(period: float, logit_shift: float = 0.0) -> float:
    """``sigmoid(logit(alpha) + logit_shift)``: the alpha after a per-window logit shift."""
    alpha = min(max(alpha_from_period(period), 1e-12), 1.0 - 1e-12)
    logit = math.log(alpha / (1.0 - alpha)) + float(logit_shift)
    return 1.0 / (1.0 + math.exp(-logit))


def m_single_ewma(period: float, eps: float = 1e-3, logit_shift: float = -META_SCALE) -> int:
    """M(eps) of one EWMA: bars until the start state's weight ``(1 - alpha)^M`` is below eps.

    ``logit_shift`` defaults to the maximal *slowing* shift (-META_SCALE), the worst case of
    the per-window adaptation (D-037: ``alpha_min = sigmoid(min logit - 0.5)``).
    """
    alpha = shifted_alpha(period, logit_shift)
    if alpha >= 1.0 - 1e-12:
        return 1
    return int(math.ceil(math.log(eps) / math.log(1.0 - alpha)))


# An EWMA request: {key: (sequence [B, L], alpha [B] or scalar)}. Stage-2 requests may read
# stage-1 results, so cascades (MACD signal, Bollinger variance) stay batchable.
Stage = Dict[str, Tuple[tf.Tensor, tf.Tensor]]

# NOTE on determinism: the engine batches the requests of every instance into one matrix
# call per stage, ordered family by family and, inside a family, request key by request key
# across its instances. That grouping, and tensors computed once and cached (below), keep
# the gradient aggregation order of the pre-NT-046 layer, so training is reproduced
# bit-for-bit (scripts/golden_run.py), not only the forward pass.


class IndicatorFamily:
    """Base class of the registered families (one instance registered per family)."""

    #: registry key ('ma', 'macd', 'rsi', 'bb', ...)
    name: str = ""
    #: which context series the family reads
    inputs: Tuple[str, ...] = ("close",)
    #: the learnable parameters of one instance, in declaration (meta-adjust column) order
    params: Tuple[ParamSpec, ...] = ()
    #: the output channels of one instance, in output order
    channels: Tuple[ChannelSpec, ...] = ()
    #: where the family is drawn by default ('price' or 'panel')
    draw: str = "panel"

    # ------------------------------------------------------------------ configuration
    def parse_instance(self, value) -> Dict[str, float]:
        """A config instance (a scalar period, or a dict of parameter periods) -> full mapping.

        Missing parameters of a dict instance take their textbook defaults."""
        if isinstance(value, Mapping):
            out = {p.name: float(value.get(p.name, p.default)) for p in self.params}
            unknown = set(value) - {p.name for p in self.params}
            if unknown:
                raise ValueError(f"indicator family '{self.name}': unknown parameters {sorted(unknown)}")
            return out
        if len(self.params) != 1:
            raise ValueError(f"indicator family '{self.name}' takes {len(self.params)} parameters; "
                             f"got the scalar {value!r}")
        return {self.params[0].name: float(value)}

    # ------------------------------------------------------------------ naming
    def logit_name(self, index: int, param: str) -> str:
        """The weight name of instance ``index``'s ``param`` logit (stable: bundles load by it)."""
        return f"{self.name}_{index}_{param}"

    def learned_name(self, index: int, param: str) -> str:
        """The reporting key of the learned period (``get_learned_parameters`` and telemetry)."""
        if len(self.params) == 1:
            return f"{self.name}_period_{index}"
        return f"{self.name}_{index}_{param}"

    # ------------------------------------------------------------------ computation
    # ``cache`` is a per-instance dict the engine passes to all three methods: a tensor a
    # later method reuses (MACD's line) is computed ONCE and cached, never recomputed - a
    # duplicate op would change the gradient aggregation order (see the NOTE above).
    def stage1(self, ctx: FamilyContext, alphas: Dict[str, tf.Tensor], cache: dict) -> Stage:
        """The EWMA requests that read only the context series."""
        raise NotImplementedError

    def stage2(self, ctx: FamilyContext, alphas: Dict[str, tf.Tensor],
               s1: Dict[str, tf.Tensor], cache: dict) -> Stage:
        """The EWMA requests that read stage-1 results (empty for most families)."""
        return {}

    def outputs(self, ctx: FamilyContext, alphas: Dict[str, tf.Tensor],
                s1: Dict[str, tf.Tensor], s2: Dict[str, tf.Tensor], cache: dict) -> List[tf.Tensor]:
        """The channel tensors ``[B, L]`` of one instance, in ``channels`` order."""
        raise NotImplementedError

    # ------------------------------------------------------------------ warm-up (D-037)
    def m_eps(self, periods: Mapping[str, float], eps: float = 1e-3,
              logit_shift: float = -META_SCALE) -> int:
        """Bars after which the output's dependence on the start state is below ``eps``.

        ``periods`` maps each parameter to its current period. The default is conservative
        (the sum of every parameter's single-EWMA tail); families with a known tighter
        composition override it."""
        return sum(m_single_ewma(periods[p.name], eps, logit_shift) for p in self.params)


# ------------------------------------------------------------------ soft extremum (NT-047)
# The smooth rolling max / min: a Boltzmann-weighted average over an EXACT rolling window
# whose (fractional) length is the learned period,
#     soft_ext_t = sum_{k in window(t, p)} e^{+-beta x_k} x_k / sum e^{+-beta x_k},
# computed from float64 prefix sums; the window edge bar enters with the fractional weight
# p - floor(p), which makes the value differentiable in the learned period (the gradient is
# the edge bar's weight density). It approaches the hard rolling extremum as beta grows,
# always lies inside the window's value range (a convex combination), forgets a peak the
# moment it leaves the window (an EWMA-decayed Boltzmann never does - measured mean %K
# errors above 17 points at every beta), and is SHIFT-INVARIANT in the series (a constant
# shift scales every weight alike). For conditioning, the exponent is taken against the
# window's first value (a per-window constant: cancels exactly, keeps causality) and
# clipped at +-SOFT_EXTREMUM_CLIP.


def soft_rolling_extremum(series: tf.Tensor, alpha: tf.Tensor, sign: float) -> tf.Tensor:
    """Smooth rolling max (``sign`` +1) / min (-1) of ``series`` ``[B, L]`` over the
    learned period ``2 / alpha - 1`` bars (``alpha`` scalar or ``[B]``), truncated at the
    window start like the textbook expanding-start rolling extremum."""
    x = tf.cast(series, tf.float32)
    a = tf.clip_by_value(tf.cast(alpha, tf.float32), 1e-6, 1.0 - 1e-6)
    a = a + tf.zeros_like(x[:, 0])                                  # scalar or [B] -> [B]
    p = 2.0 / a - 1.0                                               # learned period, bars
    ref = tf.stop_gradient(x[:, :1])
    z = tf.clip_by_value(sign * SOFT_EXTREMUM_BETA * (x - ref),
                         -SOFT_EXTREMUM_CLIP, SOFT_EXTREMUM_CLIP)
    w = tf.exp(tf.cast(z, tf.float64))
    zeros = tf.zeros_like(w[:, :1])
    qw = tf.concat([zeros, tf.cumsum(w, axis=1)], axis=1)           # exclusive prefix sums
    qxw = tf.concat([zeros, tf.cumsum(tf.cast(x, tf.float64) * w, axis=1)], axis=1)
    n = tf.shape(x)[1]
    t = tf.cast(tf.range(n), tf.float32)[None, :]                   # [1, L]
    start = tf.clip_by_value(t + 1.0 - p[:, None], 0.0, tf.cast(n, tf.float32))  # [B, L]
    lo = tf.floor(start)
    frac = tf.cast(start - lo, tf.float64)
    lo_i = tf.cast(lo, tf.int32)
    hi_i = tf.minimum(lo_i + 1, n)
    end_i = tf.tile(tf.range(1, n + 1)[None, :], [tf.shape(x)[0], 1])

    def cut(q):
        head = (1.0 - frac) * tf.gather(q, lo_i, batch_dims=1) + frac * tf.gather(q, hi_i, batch_dims=1)
        return tf.gather(q, end_i, batch_dims=1) - head

    # the window always holds at least one bar of weight >= e^-CLIP, so flooring the
    # denominator there is exact in real arithmetic and absorbs any prefix-sum round-off
    den = tf.maximum(cut(qw), math.exp(-SOFT_EXTREMUM_CLIP))
    return tf.cast(cut(qxw) / den, tf.float32)


def m_soft_extremum(period: float, eps: float = 1e-3, logit_shift: float = -META_SCALE) -> int:
    """M(eps) of one soft rolling extremum: the window itself (the learned period after the
    maximal slowing shift), plus the fractional edge bar and one bar of headroom."""
    alpha = shifted_alpha(period, logit_shift)
    return int(math.ceil(2.0 / alpha - 1.0)) + 2


def compute_reference(family: IndicatorFamily, close, periods: Mapping[str, float],
                      logit_shift: float = 0.0, *, series: Optional[Mapping[str, "object"]] = None):
    """Eagerly compute one instance's channels with fixed periods (scan EWMAs, no layer).

    ``close`` is ``[L]`` or ``[B, L]``; ``series`` optionally maps the other base series
    ('open' / 'high' / 'low' / 'volume') to arrays of the same shape (NT-047). Used by the
    offset-invariance tests of M(eps) and by drawing code that needs a family's channels
    outside the model."""
    import numpy as np

    import neural_trade.utils.math as mh

    def _prep(arr):
        a = np.asarray(arr, dtype=np.float32)
        return tf.constant(a[None, :] if a.ndim == 1 else a)

    extra = {k: _prep(v) for k, v in (series or {}).items()}
    ctx = FamilyContext(_prep(close), **extra)
    alphas = {p.name: tf.constant(shifted_alpha(periods[p.name], logit_shift), tf.float32)
              for p in family.params}
    cache: dict = {}
    s1 = {k: mh.ewma_sequence(seq, a) for k, (seq, a) in family.stage1(ctx, alphas, cache).items()}
    s2 = {k: mh.ewma_sequence(seq, a) for k, (seq, a) in family.stage2(ctx, alphas, s1, cache).items()}
    return [t.numpy() for t in family.outputs(ctx, alphas, s1, s2, cache)]

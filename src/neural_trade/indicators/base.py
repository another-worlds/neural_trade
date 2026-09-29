"""The indicator-family contract (D-031, NT-046).

One registry entry per family. An entry is an :class:`IndicatorFamily` instance that declares

* its **inputs** (which series of the bar it reads; only ``close`` today, and the derived
  ``gains`` / ``losses`` a :class:`FamilyContext` computes from it),
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

    ``close`` is the (window-relative) close sequence ``[B, L]``; ``gains`` / ``losses`` are
    derived from it lazily (zero-padded one-bar up/down moves, as RSI reads them). NT-047
    extends this with the other OHLCV series.
    """

    def __init__(self, close: tf.Tensor):
        self.close = close
        self._gains: Optional[tf.Tensor] = None
        self._losses: Optional[tf.Tensor] = None

    def _split_moves(self) -> None:
        x = self.close
        diffs = x[:, 1:] - x[:, :-1]
        zero = tf.zeros((tf.shape(x)[0], 1), dtype=x.dtype)
        self._gains = tf.concat([zero, tf.where(diffs > 0, diffs, tf.zeros_like(diffs))], axis=1)
        self._losses = tf.concat([zero, tf.where(diffs < 0, -diffs, tf.zeros_like(diffs))], axis=1)

    @property
    def gains(self) -> tf.Tensor:
        if self._gains is None:
            self._split_moves()
        return self._gains

    @property
    def losses(self) -> tf.Tensor:
        if self._losses is None:
            self._split_moves()
        return self._losses


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


def compute_reference(family: IndicatorFamily, close, periods: Mapping[str, float],
                      logit_shift: float = 0.0):
    """Eagerly compute one instance's channels with fixed periods (scan EWMAs, no layer).

    ``close`` is ``[L]`` or ``[B, L]``. Used by the offset-invariance tests of M(eps) and by
    drawing code that needs a family's channels outside the model."""
    import numpy as np

    import neural_trade.utils.math as mh

    x = np.asarray(close, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    ctx = FamilyContext(tf.constant(x))
    alphas = {p.name: tf.constant(shifted_alpha(periods[p.name], logit_shift), tf.float32)
              for p in family.params}
    cache: dict = {}
    s1 = {k: mh.ewma_sequence(seq, a) for k, (seq, a) in family.stage1(ctx, alphas, cache).items()}
    s2 = {k: mh.ewma_sequence(seq, a) for k, (seq, a) in family.stage2(ctx, alphas, s1, cache).items()}
    return [t.numpy() for t in family.outputs(ctx, alphas, s1, s2, cache)]

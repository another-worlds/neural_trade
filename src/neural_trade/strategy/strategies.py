"""Trading strategies: the notebook strategies as classes with their knobs as fields, plus baselines.

A strategy sees one bar's signals at that bar's CLOSE and returns an :class:`Order` (filled by the
engine at the next open) or ``None``; while a position is open, ``exit_signal`` may request an
exit (also filled at the next open). Take-profit and stop-loss levels on the order are enforced
by the engine on each bar's high/low.

Ported from the notebooks, with these fixes:
* ThresholdSpikeStrategy (trade.ipynb): its "agreement" was ``max(up, down)/3 >= 0.67`` - with
  three votes that silently meant unanimity and ignored the trade's side. It is now
  ``min_agreeing_horizons`` deltas pointing the SAME way as the trade. TP/SL were computed and
  never used; they are now enforced.
* EnhancedMultiHorizonStrategy (inference.ipynb): ``quality_ok`` ignored its require_* knobs;
  the stop was ``price - scaled_sigma * 1.5 * price`` (about -50% of the price); both fixed.
* LiberalStrategy (inference.ipynb, two identical cells): entry thresholds below 0.5 let a
  LONG fire on a bearish signal; the consensus must now match the side
  (``require_consensus_side``, on by default for both multi-horizon strategies).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple

import numpy as np

from neural_trade.core.registry import BaseRegistry
from neural_trade.strategy.signals import SignalFrame
from neural_trade.strategy.trades import Order


class Strategy:
    name: ClassVar[str] = "strategy"
    max_hold: int = 30

    def warmup(self) -> int:
        """Bars at the start with no decisions (trailing features still filling)."""
        return 0

    def decide(self, s: SignalFrame, t: int) -> Optional[Order]:
        raise NotImplementedError

    def exit_signal(self, s: SignalFrame, t: int, side: str, bars_held: int, entry_price: float,
                    order: Order) -> Optional[str]:
        return None


class Strategies(BaseRegistry):
    """Registry of strategy classes (outside the nine component registries)."""

    registry = {}
    strict = True
    default = "calibrated_quantile"

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        return inspect.isclass(component) and issubclass(component, Strategy)


def _side(sign: int) -> str:
    return "LONG" if sign > 0 else "SHORT"


@Strategies.register(name="threshold_spike", tags=["notebook", "trade.ipynb"])
@dataclass
class ThresholdSpikeStrategy(Strategy):
    """h1 direction threshold with a variance-spike veto and exit (trade.ipynb)."""

    name: ClassVar[str] = "threshold_spike"
    p_long: float = 0.65
    p_short: float = 0.35
    min_confidence: float = 0.5          # confidence = 1 / (1 + var_h1)
    exit_long_below: float = 0.45
    exit_short_above: float = 0.55
    min_agreeing_horizons: int = 2
    tp_pct: float = 0.015
    sl_pct: float = 0.01
    max_hold: int = 30
    warmup_bars: int = 20

    def warmup(self) -> int:
        return self.warmup_bars

    def decide(self, s, t):
        p, conf = s.p[t, 1], 1.0 / (1.0 + s.var_scaled[t, 1])
        if s.var_spike[t] or conf <= self.min_confidence:
            return None
        for sign, ok in ((1, p > self.p_long), (-1, p < self.p_short)):
            if ok and s.agreeing_horizons(sign)[t] >= self.min_agreeing_horizons:
                c = s.close[t]
                return Order(_side(sign), 1.0, tp=c * (1 + sign * self.tp_pct), sl=c * (1 - sign * self.sl_pct),
                             reason="threshold", max_hold=self.max_hold)
        return None

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        if s.var_spike[t]:
            return "SPIKE"
        p = s.p[t, 1]
        if (side == "LONG" and p < self.exit_long_below) or (side == "SHORT" and p > self.exit_short_above):
            return "REV"
        return None


@Strategies.register(name="enhanced_multi_horizon", tags=["notebook", "inference.ipynb", "default"])
@dataclass
class EnhancedMultiHorizonStrategy(Strategy):
    """Confidence-weighted multi-horizon consensus; delta-based take-profit, sigma-based stop."""

    name: ClassVar[str] = "enhanced_multi_horizon"
    min_agreement: float = 0.50
    min_signal_strength: float = 0.15
    base_entry_threshold: float = 0.45
    min_confidence: float = 0.15
    require_magnitude_coherence: bool = False
    require_direction_alignment: bool = False
    require_consensus_side: bool = True
    size_base: float = 0.5
    size_confidence_mult: float = 0.3
    size_strength_mult: float = 0.2
    tp1_delta_mult: float = 0.5
    tp2_delta_mult: float = 1.0
    sl_vol_mult: float = 1.5
    tp1_min_bars: int = 5
    reversal_long: float = 0.40
    reversal_short: float = 0.60
    incoherence_min_bars: int = 3
    max_hold: int = 30
    min_move_sigma: float = 0.5   # floor on the move that sizes the take-profits, in predicted h1 sigmas

    def _threshold(self, s, t):
        conf, strength = s.avg_confidence[t], s.strength[t]
        if conf > 0.7 and strength > 0.5:
            return self.base_entry_threshold - 0.05
        if conf > 0.5 and strength > 0.35:
            return self.base_entry_threshold
        return self.base_entry_threshold + 0.07

    def decide(self, s, t):
        quality = ((not self.require_magnitude_coherence or s.magnitude_coherent[t])
                   and (not self.require_direction_alignment or s.direction_aligned[t]))
        if not quality or s.agreement[t] < self.min_agreement or s.avg_confidence[t] < self.min_confidence:
            return None
        if s.strength[t] < self.min_signal_strength:
            return None
        thr, wdir, d1 = self._threshold(s, t), s.weighted_direction[t], s.delta[t, 1]
        size = float(np.clip(self.size_base + s.avg_confidence[t] * self.size_confidence_mult
                             + s.strength[t] * self.size_strength_mult, 0.1, 1.0))
        for sign, ok in ((1, wdir > thr and d1 > 0), (-1, wdir < 1.0 - thr and d1 < 0)):
            if ok and (not self.require_consensus_side or s.consensus[t] == sign):
                move = max(abs(d1), self.min_move_sigma * s.sigma[t, 1])
                return Order(_side(sign), size, tp=sign * move * self.tp2_delta_mult,
                             sl=-sign * s.volatility[t] * self.sl_vol_mult, tp_is_offset=True, reason="consensus",
                             max_hold=self.max_hold,
                             info={"tp1_offset": move * self.tp1_delta_mult, "confidence": s.avg_confidence[t]})
        return None

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        sign = 1 if side == "LONG" else -1
        tp1 = order.info.get("tp1_offset")
        if tp1 is not None and bars_held >= self.tp1_min_bars and sign * (s.close[t] - entry_price) >= tp1:
            return "TP1"
        p1 = s.p[t, 1]
        if (side == "LONG" and p1 < self.reversal_long) or (side == "SHORT" and p1 > self.reversal_short):
            return "REV_H1"
        if not s.magnitude_coherent[t] and bars_held >= self.incoherence_min_bars:
            return "INCOH"
        return None


@Strategies.register(name="liberal", tags=["notebook", "inference.ipynb"])
@dataclass
class LiberalStrategy(Strategy):
    """The notebook's 'liberal' variant: low thresholds, delta-based TP and SL."""

    name: ClassVar[str] = "liberal"
    require_magnitude_coherence: bool = False
    require_direction_alignment: bool = True
    require_consensus_side: bool = True
    min_agreement: float = 0.40
    min_signal_strength: float = 0.10
    min_confidence: float = 0.10
    base_entry_threshold: float = 0.35
    high_quality_threshold_reduction: float = 0.15
    low_quality_threshold_increase: float = 0.10
    high_quality_confidence: float = 0.60
    high_quality_strength: float = 0.20
    standard_quality_agreement: float = 0.50
    tp1_delta_mult: float = 0.5
    tp2_delta_mult: float = 1.0
    sl_delta_mult: float = 0.5
    tp1_min_bars: int = 3
    max_hold: int = 30
    # Floor on the move that sizes TP/SL, in predicted h1 sigmas: with a zero (shrunk or flat) delta
    # the stop sat at the entry price and every trade stopped out in its entry bar.
    min_move_sigma: float = 0.5

    def decide(self, s, t):
        if (self.require_magnitude_coherence and not s.magnitude_coherent[t]) or \
                (self.require_direction_alignment and not s.direction_aligned[t]):
            return None
        conf, strength, agree = s.avg_confidence[t], s.strength[t], s.agreement[t]
        if conf >= self.high_quality_confidence and strength >= self.high_quality_strength \
                and agree >= self.standard_quality_agreement:
            thr = self.base_entry_threshold - self.high_quality_threshold_reduction
        elif conf >= self.min_confidence and strength >= self.min_signal_strength and agree >= self.min_agreement:
            thr = self.base_entry_threshold
        else:
            thr = self.base_entry_threshold + self.low_quality_threshold_increase
        if strength < self.min_signal_strength or conf < self.min_confidence or agree < self.min_agreement:
            return None
        d1 = max(abs(s.delta[t, 1]), self.min_move_sigma * s.sigma[t, 1])
        wdir = s.weighted_direction[t]
        for sign, ok in ((1, wdir > thr), (-1, wdir < 1.0 - thr)):
            if ok and (not self.require_consensus_side or s.consensus[t] == sign):
                return Order(_side(sign), 1.0, tp=sign * d1 * self.tp2_delta_mult, sl=-sign * d1 * self.sl_delta_mult,
                             tp_is_offset=True, reason="liberal", max_hold=self.max_hold,
                             info={"tp1_offset": d1 * self.tp1_delta_mult})
        return None

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        sign = 1 if side == "LONG" else -1
        tp1 = order.info.get("tp1_offset")
        if tp1 is not None and bars_held >= self.tp1_min_bars and sign * (s.close[t] - entry_price) >= tp1:
            return "TP1"
        return None


@Strategies.register(name="calibrated_quantile", tags=["calibrated", "default-for-weak-edges"])
@dataclass
class QuantileSignalStrategy(Strategy):
    """Trade only the model's most convinced bars, with thresholds set on the CALIBRATION block.

    The notebook strategies use fixed probability lines (a horizon "votes" beyond 0.55 / 0.45).
    A calibrated head with a weak edge almost never crosses them, so they never trade. Here the
    entry lines are quantiles of the confidence-weighted P(up) on the calibration block
    (``from_calibration``): long above its ``entry_quantile``, short below ``1 - entry_quantile``,
    optionally only when the h1 price head points the same way. Exit after ``max_hold`` bars (the
    h1 horizon by default), on the stop (``sl_sigma`` x predicted sigma), or when the signal crosses
    back through the calibration median.
    """

    name: ClassVar[str] = "calibrated_quantile"
    long_above: float = 0.55
    short_below: float = 0.45
    median: float = 0.5
    entry_quantile: float = 0.9
    require_delta_agreement: bool = False
    sl_sigma: float = 2.0
    size: float = 1.0
    max_hold: int = 15

    @classmethod
    def from_calibration(cls, calibration, entry_quantile: float = 0.9, **kwargs) -> "QuantileSignalStrategy":
        """``calibration``: the calibration block's SignalFrame, or a stored quantile table
        ``{quantile: weighted-direction value}`` (ArtifactBundle meta "weighted_direction_quantiles")."""
        if isinstance(calibration, SignalFrame):
            w = np.asarray(calibration.weighted_direction, dtype=float)
            q = lambda x: float(np.quantile(w, x))  # noqa: E731
        else:
            table = {round(float(k), 6): float(v) for k, v in dict(calibration).items()}
            missing = [x for x in (entry_quantile, 1.0 - entry_quantile, 0.5) if round(x, 6) not in table]
            if missing:
                raise ValueError(f"quantile table lacks {missing}; it has {sorted(table)}")
            q = lambda x: table[round(x, 6)]  # noqa: E731
        return cls(long_above=q(entry_quantile), short_below=q(1.0 - entry_quantile), median=q(0.5),
                   entry_quantile=entry_quantile, **kwargs)

    def decide(self, s, t):
        w = s.weighted_direction[t]
        for sign, ok in ((1, w > self.long_above), (-1, w < self.short_below)):
            if ok and (not self.require_delta_agreement or np.sign(s.delta[t, 1]) == sign):
                return Order(_side(sign), self.size, sl=-sign * self.sl_sigma * s.sigma[t, 1], tp_is_offset=True,
                             reason="quantile", max_hold=self.max_hold)
        return None

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        w = s.weighted_direction[t]
        if (side == "LONG" and w < self.median) or (side == "SHORT" and w > self.median):
            return "REV"
        return None


# ------------------------------------------------------------------ baselines
@Strategies.register(name="buy_and_hold", tags=["baseline"])
@dataclass
class BuyAndHold(Strategy):
    """Long at the first bar, held to the end."""

    name: ClassVar[str] = "buy_and_hold"
    max_hold: int = 10 ** 9

    def decide(self, s, t):
        return Order("LONG", 1.0, reason="buy_and_hold", max_hold=self.max_hold) if t == 0 else None


@Strategies.register(name="always_flat", tags=["baseline"])
@dataclass
class AlwaysFlat(Strategy):
    """Never trades (the zero line every strategy must beat after costs)."""

    name: ClassVar[str] = "always_flat"

    def decide(self, s, t):
        return None


@Strategies.register(name="random_signal", tags=["baseline"])
@dataclass
class RandomSignal(Strategy):
    """Random side at a fixed entry rate and holding time (the 'same frequency' null model)."""

    name: ClassVar[str] = "random_signal"
    trade_rate: float = 0.05
    hold_bars: int = 10
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._draws: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        self.max_hold = self.hold_bars

    def decide(self, s, t):
        if len(self._draws[0]) != len(s):
            self._draws = (self._rng.uniform(size=len(s)), self._rng.integers(0, 2, size=len(s)))
        if self._draws[0][t] < self.trade_rate:
            return Order("LONG" if self._draws[1][t] else "SHORT", 1.0, reason="random", max_hold=self.hold_bars)
        return None

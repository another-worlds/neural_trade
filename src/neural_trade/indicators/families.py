"""Today's four indicator families (NT-046): MA/EMA, MACD, RSI, Bollinger.

Each family computes exactly what ``LearnableIndicators`` computed before NT-046 (the layer
now assembles these entries; the numbers are pinned by tests/test_learnable_indicators.py and
scripts/golden_run.py). The logit weight names and reporting keys keep their historical
spellings so saved serving bundles and telemetry stay unchanged. NT-047 adds the OHLCV
families on top of this contract.
"""
from __future__ import annotations

from typing import List, Mapping

import tensorflow as tf

from .base import ChannelSpec, FamilyContext, IndicatorFamily, ParamSpec, Stage, m_single_ewma
from .registry import Indicators

# ------------------------------------------------------------------ MACD ratio mode (NT-106)
#: keeps r away from 1 by enough that fast = 1 + r*(slow_eff - 1) is float32-DISTINGUISHABLE
#: from slow_eff, not just mathematically smaller: at slow_eff = _MACD_RATIO_SLOW_FLOOR (the
#: tightest case) the gap is _MACD_RATIO_EPS * (slow_eff - 1), about 2e-4 absolute - far above
#: float32's ~1.2e-7 relative epsilon at these magnitudes, so no rounding collision (an eps of
#: 1e-6 here is not: at slow_eff = 1.001 it gave a gap of ~1e-9, which float32 rounds away -
#: found empirically by sweeping extreme logit pairs, see tests/test_macd_ratio.py)
_MACD_RATIO_EPS = 2e-4
#: floors the EFFECTIVE slow period well clear of the fast leg's own floor of 1 - a hardcoded
#: numerical safety margin (not Config.MOMENTUM_CLIP_MIN, which the family has no access to;
#: it happens to equal MOMENTUM_CLIP_MIN's own default of 2 bars). Guarantees
#: 1 <= fast_period < slow_eff for ANY input alphas with a float32-representable gap.
_MACD_RATIO_SLOW_FLOOR = 2.0


def _macd_ratio_periods(alphas: Mapping[str, tf.Tensor]):
    """``(fast_period, slow_eff, signal_period)`` of the ratio parametrisation (NT-106).

    ``alphas`` carries the per-param alphas exactly as the engine computes them for every
    family (``LearnableIndicators._alpha`` / ``_alpha_from_logit``: the STE-scaled logit,
    the optional per-window meta_adjust shift, and the optional INDICATOR_BOUND_APPLIED
    clip - the ratio parametrisation does not change any of that machinery, only how its
    three alphas ('slow', 'ratio', 'signal') combine into periods).

    ``fast = 1 + r * (slow_eff - 1)``, ``r = alphas['ratio']`` clipped into
    ``[0, 1 - _MACD_RATIO_EPS]`` and ``slow_eff = max(slow_period, _MACD_RATIO_SLOW_FLOOR)``:
    for ANY real-valued input logit or meta shift this guarantees
    ``1 <= fast_period < slow_eff`` with a float32-representable gap, so 'fast < slow' always
    holds (Config.MACD_PARAM's docstring and config-reference.md document how
    MOMENTUM_CLIP_MIN / MOMENTUM_CLIP_MAX and INDICATOR_BOUND_APPLIED interact with this
    floor: both still clip the raw 'slow' and 'ratio' logits, narrowing the practical range of
    r and of the effective slow period, but the fast < slow guarantee itself comes from this
    formula, not from either clip).
    """
    slow_period = 2.0 / (alphas["slow"] + 1e-8) - 1.0
    slow_eff = tf.maximum(slow_period, _MACD_RATIO_SLOW_FLOOR)
    r = tf.clip_by_value(alphas["ratio"], 0.0, 1.0 - _MACD_RATIO_EPS)
    fast_period = 1.0 + r * (slow_eff - 1.0)
    signal_period = 2.0 / (alphas["signal"] + 1e-8) - 1.0
    return fast_period, slow_eff, signal_period


class MovingAverageFamily(IndicatorFamily):
    """Learnable EWMA moving average of the close (one period, drawn on price)."""

    name = "ma"
    inputs = ("close",)
    params = (ParamSpec("period", default=20.0, minimum=2.0),)
    channels = (ChannelSpec("ma", draw="price"),)
    draw = "price"

    def logit_name(self, index: int, param: str) -> str:
        return f"alpha_ma_{index}"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {"ema": (ctx.close, alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [s1["ema"]]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return m_single_ewma(periods["period"], eps, logit_shift)


class MACDFamily(IndicatorFamily):
    """Learnable MACD: fast/slow EWMAs, signal EWMA of their difference (its own panel)."""

    name = "macd"
    inputs = ("close",)
    params = (ParamSpec("fast", default=12.0, minimum=2.0),
              ParamSpec("slow", default=26.0, minimum=2.0),
              ParamSpec("signal", default=9.0, minimum=2.0))
    channels = (ChannelSpec("macd_line"), ChannelSpec("macd_signal"),
                ChannelSpec("macd_hist"), ChannelSpec("macd_cross"))
    draw = "panel"

    def logit_name(self, index: int, param: str) -> str:
        return f"macd_{index}_{param}"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {"fast": (ctx.close, alphas["fast"]), "slow": (ctx.close, alphas["slow"])}

    def stage2(self, ctx, alphas, s1, cache) -> Stage:
        # the line is cached and REUSED by outputs: computing it twice would change the
        # gradient aggregation order (see the determinism NOTE in base.py)
        cache["line"] = s1["fast"] - s1["slow"]
        return {"signal": (cache["line"], alphas["signal"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        line = cache["line"]
        sig = s2["signal"]
        hist = line - sig
        # Soft sign: tf.sign has zero gradient; tanh*10 is a differentiable approximation.
        return [line, sig, hist, tf.tanh(hist * 10.0)]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        line = max(m_single_ewma(periods["fast"], eps, logit_shift),
                   m_single_ewma(periods["slow"], eps, logit_shift))
        return line + m_single_ewma(periods["signal"], eps, logit_shift)


#: logit_from_period(this) == logit_from_alpha(fast/slow) for the textbook 12/26 settings
#: (2/(P+1) = fast/slow => P = 2*slow/fast - 1): the 'ratio' ParamSpec is stored as this kind
#: of pseudo-period so LearnableIndicators.build() (which converts EVERY instance value
#: through the same period->logit transform) initialises it to the textbook ratio
#: (NT-106; see MACDRatioFamily.parse_instance).
_MACD_RATIO_TEXTBOOK_PSEUDO_PERIOD = 2.0 * 26.0 / 12.0 - 1.0


class MACDRatioFamily(IndicatorFamily):
    """MACD under the ratio parametrisation (NT-106; D-045 B_model_indicators.md 4.1, 7 item 8).

    ``macd_1_fast`` sat at the floor of 2 in 5 of 6 real runs, and ``ma_period_0`` /
    ``macd_2_fast`` in 2 of 6 (section 4.1): fast < slow was never enforced (a mirror symmetry
    up to the sign of the downstream weights), and the independent floor (MOMENTUM_CLIP_MIN)
    kept any fast leg from ever reaching the raw price (period 1).

    Here the slow leg is learned exactly as in :class:`MACDFamily`; the fast leg is derived,
    ``fast = 1 + r * (slow_eff - 1)`` (:func:`_macd_ratio_periods`), with ``r`` its own
    learned logit (``sigmoid``) instead of an independently clipped period. This guarantees
    fast < slow for ANY logit or meta_adjust shift (structurally, not by a clip) and lets the
    fast leg reach p = 1 explicitly as r -> 0, where the previous floor of 2 never let it go.
    MOMENTUM_CLIP_MIN / MOMENTUM_CLIP_MAX and INDICATOR_BOUND_APPLIED still clip the raw
    'slow' and 'ratio' logits (Config.MACD_PARAM's docstring has the exact interaction); they
    narrow what training can reach, not the invariant itself.

    Selected by ``Config.MACD_PARAM = "ratio"`` (``LearnableIndicators.build()`` substitutes
    this family for 'macd' instances; off by default, so the default path never constructs
    it and the golden run is unaffected). ``parse_instance`` accepts the same fast/slow/signal
    config dicts as 'independent' (``MACD_SETTINGS``), translating fast into its initial
    ratio, or an explicit 'ratio' key instead of 'fast'.
    """

    name = "macd_ratio"
    inputs = ("close",)
    params = (ParamSpec("slow", default=26.0, minimum=2.0),
              ParamSpec("ratio", default=_MACD_RATIO_TEXTBOOK_PSEUDO_PERIOD, minimum=0.0),
              ParamSpec("signal", default=9.0, minimum=2.0))
    channels = (ChannelSpec("macd_line"), ChannelSpec("macd_signal"),
                ChannelSpec("macd_hist"), ChannelSpec("macd_cross"))
    draw = "panel"

    def logit_name(self, index: int, param: str) -> str:
        return f"macd_{index}_{param}"

    def parse_instance(self, value) -> dict:
        if not isinstance(value, Mapping):
            raise ValueError(f"indicator family '{self.name}' takes a mapping instance "
                             f"(fast/slow/signal or slow/ratio/signal), got the scalar {value!r}")
        unknown = set(value) - {"fast", "slow", "ratio", "signal"}
        if unknown:
            raise ValueError(f"indicator family '{self.name}': unknown parameters {sorted(unknown)}")
        if "fast" in value and "ratio" in value:
            raise ValueError(f"indicator family '{self.name}': give 'fast' or 'ratio', not both")
        slow = float(value.get("slow", 26.0))
        signal = float(value.get("signal", 9.0))
        if "ratio" in value:
            r = float(value["ratio"])
        else:
            r = float(value.get("fast", 12.0)) / slow
        r = min(max(r, _MACD_RATIO_EPS), 1.0 - _MACD_RATIO_EPS)
        pseudo_period = 2.0 / r - 1.0  # see _MACD_RATIO_TEXTBOOK_PSEUDO_PERIOD
        return {"slow": slow, "ratio": pseudo_period, "signal": signal}

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        fast_period, slow_eff, _signal_period = _macd_ratio_periods(alphas)
        fast_alpha = 2.0 / (fast_period + 1.0)
        slow_alpha = 2.0 / (slow_eff + 1.0)
        cache["fast_period"] = fast_period
        cache["slow_eff"] = slow_eff
        return {"fast": (ctx.close, fast_alpha), "slow": (ctx.close, slow_alpha)}

    def stage2(self, ctx, alphas, s1, cache) -> Stage:
        # same determinism note as MACDFamily.stage2: the line is cached and REUSED
        cache["line"] = s1["fast"] - s1["slow"]
        return {"signal": (cache["line"], alphas["signal"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        line = cache["line"]
        sig = s2["signal"]
        hist = line - sig
        return [line, sig, hist, tf.tanh(hist * 10.0)]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        # fast's effective period is always < the slow leg's (by construction), so the slow
        # leg's single-EWMA tail bounds the fast leg's too (conservative, same shape as
        # MACDFamily.m_eps with the fast tail replaced by the slow one it is always below).
        # ``periods['ratio']`` is the pseudo-period of the ParamSpec (see parse_instance), not
        # itself part of this bound.
        slow_m = m_single_ewma(periods["slow"], eps, logit_shift)
        return slow_m + m_single_ewma(periods["signal"], eps, logit_shift)

    def applied_report(self, index: int, alphas) -> dict:
        """Reporting hook (``LearnableIndicators._report_entries``, NT-106): the SAME
        ``macd_{index}_fast`` / ``_slow`` / ``_signal`` keys 'independent' mode reports, with
        'fast' and 'slow' the ACTUAL applied periods of :func:`_macd_ratio_periods` (not the
        raw 'ratio' logit, which is not itself a period) - so telemetry, the period-history
        CSV and the applied-period report (``evaluation/applied_periods.py``) read sensible
        numbers under either mode."""
        fast_period, slow_eff, signal_period = _macd_ratio_periods(alphas)
        return {f"macd_{index}_fast": fast_period, f"macd_{index}_slow": slow_eff,
                f"macd_{index}_signal": signal_period}


class RSIFamily(IndicatorFamily):
    """Learnable RSI: one smoothing period over the gain and loss EWMAs (its own panel)."""

    name = "rsi"
    inputs = ("close", "gains", "losses")
    params = (ParamSpec("period", default=14.0, minimum=2.0),)
    channels = (ChannelSpec("rsi"),)
    draw = "panel"

    def logit_name(self, index: int, param: str) -> str:
        return f"rsi_alpha_{index}"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {"gains": (ctx.gains, alphas["period"]), "losses": (ctx.losses, alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        rs = s1["gains"] / (s1["losses"] + 1e-8)
        return [100.0 - (100.0 / (1.0 + rs))]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return m_single_ewma(periods["period"], eps, logit_shift)


class BollingerFamily(IndicatorFamily):
    """Learnable Bollinger bands: EWMA mean, EWMA variance, +/- 2 sigma and %B (on price)."""

    name = "bb"
    inputs = ("close",)
    params = (ParamSpec("period", default=20.0, minimum=2.0),)
    channels = (ChannelSpec("bb_mean", draw="price"), ChannelSpec("bb_upper", draw="price"),
                ChannelSpec("bb_lower", draw="price"), ChannelSpec("bb_percent"))
    draw = "price"

    def logit_name(self, index: int, param: str) -> str:
        return f"bb_alpha_{index}"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {"mean": (ctx.close, alphas["period"])}

    def stage2(self, ctx, alphas, s1, cache) -> Stage:
        return {"var": (tf.square(ctx.close - s1["mean"]), alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        # channel order AND op creation order match the pre-NT-046 code (upper and lower
        # before %B): creating std's consumers in another order changes the gradient
        # aggregation order (the determinism NOTE in base.py)
        mean, std = s1["mean"], tf.sqrt(s2["var"] + 1e-8)
        upper = mean + 2.0 * std
        lower = mean - 2.0 * std
        percent_b = (ctx.close - (mean - 2.0 * std)) / (4.0 * std + 1e-8)
        return [mean, upper, lower, percent_b]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        # mean EWMA feeding the variance EWMA: two tails of the same period
        return 2 * m_single_ewma(periods["period"], eps, logit_shift)


Indicators.register(name="ma", tags=["trend", "price", "default"],
                    description="Learnable EWMA moving average (MA/EMA) on the close")(MovingAverageFamily())
Indicators.register(name="macd", tags=["momentum", "panel", "default"],
                    description="Learnable MACD (fast/slow/signal EWMAs) with histogram and soft cross")(MACDFamily())
Indicators.register(name="macd_ratio", tags=["momentum", "panel"],
                    description="Learnable MACD, ratio parametrisation (NT-106: fast = r x slow, "
                               "r its own logit; selected by Config.MACD_PARAM = 'ratio'")(MACDRatioFamily())
Indicators.register(name="rsi", tags=["momentum", "panel", "default"],
                    description="Learnable RSI (EWMA-smoothed gains/losses)")(RSIFamily())
Indicators.register(name="bb", tags=["volatility", "price", "default"],
                    description="Learnable Bollinger bands (EWMA mean/variance, +/- 2 sigma, %B)")(BollingerFamily())

# The OHLCV families (NT-047) register in families_ohlcv; the package __init__ imports
# both and then marks the registry initialized.

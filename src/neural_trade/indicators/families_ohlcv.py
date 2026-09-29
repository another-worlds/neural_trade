"""The ten OHLCV indicator families (NT-047, D-031): range / volatility (ATR, Stochastic,
Williams %R, Keltner), volume (OBV, VWAP, MFI) and trend strength / channels (ADX/DMI, CCI,
Donchian).

Every family is fully learnable (owner, D-031): its periods are trainable logits like the
NT-046 four. Where the textbook form is not differentiable, a smooth version replaces it:

* rolling max / min -> the smooth rolling extremum of ``base`` (a Boltzmann-weighted
  average over an exact fractional rolling window, differentiable in the period through
  the window-edge weight; window mode now, the decayed series forms come after A/B-1,
  D-037), used by Stochastic, Williams %R and Donchian;
* sign branches (OBV's sign(dClose), MFI's up/down split, ADX's +DM/-DM selection) ->
  ``tanh`` / ``sigmoid`` gates at ``SOFT_SIGN_SHARPNESS`` (the MACD soft-cross scale).

Smoothing convention: like NT-046's four, every rolling average is the EWMA with
``alpha = 2 / (period + 1)`` (the textbook EMA), including where the textbook description
uses an SMA or Wilder smoothing; the tolerance tests state this
(tests/registries/test_indicators_ohlcv.py). The families read the model's NORMALISED
input (OHLC window-relative, volume scaled by the train-mean volume,
``neural_trade.data.scaling``); each form below is shift- and scale-invariant in price
(differences and ratios of differences) or uses the volume only through ratios, except the
price-drawn channels (Keltner, VWAP, Donchian), which are in window-relative price units
like the NT-046 MA and Bollinger lines.

Each family declares its M(eps) (D-037): the bars after which the output's dependence on
the state at the start is below eps, at the maximal slowing per-window shift. The
cumulative OBV level would never forget the window start, so the OBV family outputs the
OBV oscillator (OBV minus its own EWMA), in which the level cancels and only
geometrically-decayed increments remain; its declared M carries a factor 2 of headroom for
the amplitude of the cancelled level (pinned by the empirical offset-invariance test).
"""
from __future__ import annotations

from typing import List

import tensorflow as tf

from .base import (
    ChannelSpec,
    FamilyContext,
    IndicatorFamily,
    ParamSpec,
    SOFT_SIGN_SHARPNESS,
    Stage,
    m_single_ewma,
    m_soft_extremum,
    soft_rolling_extremum,
)
from .registry import Indicators

_EPS = 1e-6


class ATRFamily(IndicatorFamily):
    """Learnable ATR: EWMA of the true range (its own panel).

    The true range max(high - low, |high - prev close|, |low - prev close|) is exact
    (differentiable in the inputs almost everywhere, and the learnable period never enters
    it); only the smoothing is learnable.
    """

    name = "atr"
    inputs = ("high", "low", "close")
    params = (ParamSpec("period", default=14.0, minimum=2.0),)
    channels = (ChannelSpec("atr"),)
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {"atr": (ctx.true_range, alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [s1["atr"]]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return m_single_ewma(periods["period"], eps, logit_shift)


class StochasticFamily(IndicatorFamily):
    """Learnable stochastic oscillator: %K over soft rolling extrema, %D its EWMA.

    %K = 100 (close - LL) / (HH - LL + eps) with HH / LL the smooth rolling max of the
    high / min of the low over the learnable k_period window (soft_rolling_extremum); %D
    is the d_period EWMA of %K. The soft extrema lie strictly inside the true range, so
    %K may slightly leave [0, 100]."""

    name = "stoch"
    inputs = ("high", "low", "close")
    params = (ParamSpec("k_period", default=14.0, minimum=2.0),
              ParamSpec("d_period", default=3.0, minimum=2.0))
    channels = (ChannelSpec("stoch_k"), ChannelSpec("stoch_d"))
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        # %K is cached and REUSED by outputs: computing it twice would change the gradient
        # aggregation order (the determinism NOTE in base.py)
        hh = soft_rolling_extremum(ctx.high, alphas["k_period"], +1.0)
        ll = soft_rolling_extremum(ctx.low, alphas["k_period"], -1.0)
        cache["k"] = 100.0 * (ctx.close - ll) / (hh - ll + _EPS)
        return {"d": (cache["k"], alphas["d_period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [cache["k"], s1["d"]]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return (m_soft_extremum(periods["k_period"], eps, logit_shift)
                + m_single_ewma(periods["d_period"], eps, logit_shift))


class WilliamsRFamily(IndicatorFamily):
    """Learnable Williams %R over the same soft rolling extrema as the stochastic."""

    name = "willr"
    inputs = ("high", "low", "close")
    params = (ParamSpec("period", default=14.0, minimum=2.0),)
    channels = (ChannelSpec("willr"),)
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        hh = soft_rolling_extremum(ctx.high, alphas["period"], +1.0)
        ll = soft_rolling_extremum(ctx.low, alphas["period"], -1.0)
        return [-100.0 * (hh - ctx.close) / (hh - ll + _EPS)]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return m_soft_extremum(periods["period"], eps, logit_shift)


class KeltnerFamily(IndicatorFamily):
    """Learnable Keltner channel: EWMA middle line +/- 2 ATR (drawn on price).

    The band multiplier stays the textbook 2 (a fixed constant, like Bollinger's 2 sigma:
    ParamSpec covers periods, D-031's learnable parameters are the periods)."""

    name = "keltner"
    inputs = ("high", "low", "close")
    params = (ParamSpec("period", default=20.0, minimum=2.0),
              ParamSpec("atr_period", default=10.0, minimum=2.0))
    channels = (ChannelSpec("kelt_mid", draw="price"), ChannelSpec("kelt_upper", draw="price"),
                ChannelSpec("kelt_lower", draw="price"))
    draw = "price"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {"mid": (ctx.close, alphas["period"]),
                "atr": (ctx.true_range, alphas["atr_period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        mid, atr = s1["mid"], s1["atr"]
        return [mid, mid + 2.0 * atr, mid - 2.0 * atr]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        # two independent stage-1 EWMAs: the slower tail dominates
        return max(m_single_ewma(periods["period"], eps, logit_shift),
                   m_single_ewma(periods["atr_period"], eps, logit_shift))


class OBVFamily(IndicatorFamily):
    """Learnable OBV oscillator: soft-signed volume flow, cumulated, minus its own EWMA.

    OBV = cumsum(tanh(k dClose) * volume) (the soft sign of the close move); the output is
    OBV - EWMA(OBV) (the OBV oscillator), in which the cumulative level cancels: only
    geometrically decayed increments remain, so the window start is forgotten (see the
    module docstring on M(eps))."""

    name = "obv"
    inputs = ("close", "volume")
    params = (ParamSpec("period", default=20.0, minimum=2.0),)
    channels = (ChannelSpec("obv_osc"),)
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        diffs = ctx.close[:, 1:] - ctx.close[:, :-1]
        zero = tf.zeros((tf.shape(ctx.close)[0], 1), dtype=ctx.close.dtype)
        flow = tf.concat([zero, tf.tanh(SOFT_SIGN_SHARPNESS * diffs) * ctx.volume[:, 1:]],
                         axis=1)
        cache["obv"] = tf.cumsum(flow, axis=1)
        return {"sig": (cache["obv"], alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [cache["obv"] - s1["sig"]]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        # one EWMA tail, doubled: headroom for the amplitude of the cancelled cumsum level
        return 2 * m_single_ewma(periods["period"], eps, logit_shift)


class VWAPFamily(IndicatorFamily):
    """Learnable rolling VWAP: EWMA(typical price x volume) / EWMA(volume), on price.

    In window-relative units like every price-drawn channel; a zero-volume stretch decays
    both sums, and an all-zero window gives 0 (the last close's level)."""

    name = "vwap"
    inputs = ("high", "low", "close", "volume")
    params = (ParamSpec("period", default=20.0, minimum=2.0),)
    channels = (ChannelSpec("vwap", draw="price"),)
    draw = "price"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        a = alphas["period"]
        return {"pv": (ctx.typical_price * ctx.volume, a), "v": (ctx.volume, a)}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [s1["pv"] / (s1["v"] + 1e-8)]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        # numerator and denominator share one tail; doubled for the ratio's amplification
        return 2 * m_single_ewma(periods["period"], eps, logit_shift)


class MFIFamily(IndicatorFamily):
    """Learnable money flow index: a volume-weighted RSI of the typical price.

    The up/down split is the soft gate sigmoid(k dTP); the flow magnitude is the volume
    (textbook uses typical price x volume, whose price factor is nearly constant within a
    window and cancels in the ratio - the tolerance test quantifies this on a hand-made
    series with a realistic level-to-range ratio)."""

    name = "mfi"
    inputs = ("high", "low", "close", "volume")
    params = (ParamSpec("period", default=14.0, minimum=2.0),)
    channels = (ChannelSpec("mfi"),)
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        tp = ctx.typical_price
        diffs = tp[:, 1:] - tp[:, :-1]
        zero = tf.zeros((tf.shape(tp)[0], 1), dtype=tp.dtype)
        up = tf.concat([zero + 0.5, tf.sigmoid(SOFT_SIGN_SHARPNESS * diffs)], axis=1)
        a = alphas["period"]
        return {"pos": (ctx.volume * up, a), "neg": (ctx.volume * (1.0 - up), a)}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [100.0 * s1["pos"] / (s1["pos"] + s1["neg"] + 1e-8)]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return m_single_ewma(periods["period"], eps, logit_shift)


class ADXFamily(IndicatorFamily):
    """Learnable ADX/DMI: soft-gated directional moves, EWMA-smoothed, ADX = EWMA(DX).

    +DM / -DM keep only the larger of the up-move relu(dHigh) and down-move relu(-dLow)
    through the soft gate sigmoid(k (udm - ddm)); DI = 100 EWMA(DM) / EWMA(TR); ADX is the
    stage-2 EWMA of DX = 100 |+DI - -DI| / (+DI + -DI)."""

    name = "adx"
    inputs = ("high", "low", "close")
    params = (ParamSpec("period", default=14.0, minimum=2.0),)
    channels = (ChannelSpec("plus_di"), ChannelSpec("minus_di"), ChannelSpec("adx"))
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        zero = tf.zeros((tf.shape(ctx.high)[0], 1), dtype=ctx.high.dtype)
        udm = tf.nn.relu(tf.concat([zero, ctx.high[:, 1:] - ctx.high[:, :-1]], axis=1))
        ddm = tf.nn.relu(tf.concat([zero, ctx.low[:, :-1] - ctx.low[:, 1:]], axis=1))
        gate = tf.sigmoid(SOFT_SIGN_SHARPNESS * (udm - ddm))
        a = alphas["period"]
        return {"pdm": (udm * gate, a), "ndm": (ddm * (1.0 - gate), a),
                "tr": (ctx.true_range, a)}

    def stage2(self, ctx, alphas, s1, cache) -> Stage:
        # the DI lines are cached and REUSED by outputs: computing them twice would change
        # the gradient aggregation order (the determinism NOTE in base.py)
        cache["pdi"] = 100.0 * s1["pdm"] / (s1["tr"] + 1e-8)
        cache["ndi"] = 100.0 * s1["ndm"] / (s1["tr"] + 1e-8)
        dx = 100.0 * tf.abs(cache["pdi"] - cache["ndi"]) / (cache["pdi"] + cache["ndi"] + 1e-8)
        return {"adx": (dx, alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [cache["pdi"], cache["ndi"], s2["adx"]]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return 2 * m_single_ewma(periods["period"], eps, logit_shift)


class CCIFamily(IndicatorFamily):
    """Learnable CCI: (tp - EWMA(tp)) / (0.015 x EWMA(|tp - EWMA(tp)|)) (its own panel)."""

    name = "cci"
    inputs = ("high", "low", "close")
    params = (ParamSpec("period", default=20.0, minimum=2.0),)
    channels = (ChannelSpec("cci"),)
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {"m": (ctx.typical_price, alphas["period"])}

    def stage2(self, ctx, alphas, s1, cache) -> Stage:
        return {"md": (tf.abs(ctx.typical_price - s1["m"]), alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        return [(ctx.typical_price - s1["m"]) / (0.015 * s2["md"] + 1e-8)]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return 2 * m_single_ewma(periods["period"], eps, logit_shift)


class DonchianFamily(IndicatorFamily):
    """Learnable Donchian channel: soft rolling max of the high / min of the low, and the
    middle line, all drawn on price."""

    name = "donchian"
    inputs = ("high", "low")
    params = (ParamSpec("period", default=20.0, minimum=2.0),)
    channels = (ChannelSpec("don_upper", draw="price"), ChannelSpec("don_lower", draw="price"),
                ChannelSpec("don_mid", draw="price"))
    draw = "price"

    def stage1(self, ctx: FamilyContext, alphas, cache) -> Stage:
        return {}

    def outputs(self, ctx, alphas, s1, s2, cache) -> List[tf.Tensor]:
        upper = soft_rolling_extremum(ctx.high, alphas["period"], +1.0)
        lower = soft_rolling_extremum(ctx.low, alphas["period"], -1.0)
        return [upper, lower, (upper + lower) / 2.0]

    def m_eps(self, periods, eps=1e-3, logit_shift=-0.5) -> int:
        return m_soft_extremum(periods["period"], eps, logit_shift)


Indicators.register(name="atr", tags=["volatility", "panel", "default"],
                    description="Learnable ATR (EWMA of the exact true range)")(ATRFamily())
Indicators.register(name="stoch", tags=["momentum", "panel", "default"],
                    description="Learnable stochastic %K/%D over Boltzmann soft extrema")(StochasticFamily())
Indicators.register(name="willr", tags=["momentum", "panel", "default"],
                    description="Learnable Williams %R over Boltzmann soft extrema")(WilliamsRFamily())
Indicators.register(name="keltner", tags=["volatility", "price", "default"],
                    description="Learnable Keltner channel (EWMA middle +/- 2 ATR)")(KeltnerFamily())
Indicators.register(name="obv", tags=["volume", "panel", "default"],
                    description="Learnable OBV oscillator (soft-signed volume flow minus its EWMA)")(OBVFamily())
Indicators.register(name="vwap", tags=["volume", "price", "default"],
                    description="Learnable rolling VWAP (EWMA(tp x vol) / EWMA(vol))")(VWAPFamily())
Indicators.register(name="mfi", tags=["volume", "panel", "default"],
                    description="Learnable money flow index (volume-weighted soft RSI of tp)")(MFIFamily())
Indicators.register(name="adx", tags=["trend", "panel", "default"],
                    description="Learnable ADX/DMI (soft-gated directional moves)")(ADXFamily())
Indicators.register(name="cci", tags=["trend", "panel", "default"],
                    description="Learnable CCI (EWMA mean and mean absolute deviation of tp)")(CCIFamily())
Indicators.register(name="donchian", tags=["trend", "price", "default"],
                    description="Learnable Donchian channel (Boltzmann soft extrema)")(DonchianFamily())

"""Today's four indicator families (NT-046): MA/EMA, MACD, RSI, Bollinger.

Each family computes exactly what ``LearnableIndicators`` computed before NT-046 (the layer
now assembles these entries; the numbers are pinned by tests/test_learnable_indicators.py and
scripts/golden_run.py). The logit weight names and reporting keys keep their historical
spellings so saved serving bundles and telemetry stay unchanged. NT-047 adds the OHLCV
families on top of this contract.
"""
from __future__ import annotations

from typing import List

import tensorflow as tf

from .base import ChannelSpec, FamilyContext, IndicatorFamily, ParamSpec, Stage, m_single_ewma
from .registry import Indicators


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
Indicators.register(name="rsi", tags=["momentum", "panel", "default"],
                    description="Learnable RSI (EWMA-smoothed gains/losses)")(RSIFamily())
Indicators.register(name="bb", tags=["volatility", "price", "default"],
                    description="Learnable Bollinger bands (EWMA mean/variance, +/- 2 sigma, %B)")(BollingerFamily())
Indicators._initialized = True

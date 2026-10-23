"""rsi_threshold: Wilder-RSI mean reversion, the rules of ``neural_trade.strategy.ta_rules`` ``ta_rsi``.

Entry (flat): BUY when RSI < ``lower``, SELL when RSI > ``upper`` (SELL only with ``allow_short``). Exit: RSI back
through ``exit_level`` ("RSI_MID"); the engine's TIME exit after ``max_hold`` bars. Warm-up = ``period`` bars.

Wilder's RSI is recursive (seeded on the first ``period`` changes of the series it is given), so ``wilder_rsi`` is run
on the last ``(RSI_DECAY + 1) x period`` closes: the seed's weight there is below e^-30 (1e-13), so the values equal
the full-history ones of the oracle to rounding, and are exactly equal while the whole history is shorter than that.
Bounds are the oracle's search ranges; ``lower < exit_level < upper`` is checked on construction.
"""
from __future__ import annotations

import math

from neural_trade.desk.core.model import PositionSide
from neural_trade.desk.core.params import ParamSpec
from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy

RSI_DECAY = 30   # periods of history beyond the first: (1 - 1/period) ** (30 * period) < e**-30


@DeskStrategies.register(name="rsi_threshold", tags=["baseline", "ta"])
class RSIThreshold(DeskStrategy):
    name = "rsi_threshold"
    PARAMS = (
        ParamSpec("period", "int", 14, 2, 60, 1, doc="Wilder's n (not the network's EMA span)"),
        ParamSpec("lower", "float", 30.0, 10.0, 40.0, 1.0, doc="buy below this RSI"),
        ParamSpec("upper", "float", 70.0, 60.0, 90.0, 1.0, doc="sell above this RSI"),
        ParamSpec("exit_level", "float", 50.0, 40.0, 60.0, 1.0, doc="exit when RSI is back through this level"),
        ParamSpec("allow_short", "bool", True, doc="False: long only"),
        ParamSpec("size", "float", 1.0, 0.01, 1.0, 0.01, doc="fraction of equity per entry"),
        ParamSpec("max_hold", "int", 1440, 1, 100_000, 1, doc="bars held at most"),
    )

    def __init__(self, **params) -> None:
        super().__init__(**params)
        p = self.params
        if not p["lower"] < p["exit_level"] < p["upper"]:
            raise ValueError(f"need lower < exit_level < upper, got {p['lower']}, {p['exit_level']}, {p['upper']}")
        self.warmup_bars = p["period"]

    def on_bar(self, ctx) -> None:
        from neural_trade.strategy.ta_rules import wilder_rsi   # lazy: the strategy package imports TensorFlow

        p = self.params
        close = ctx.bars((RSI_DECAY + 1) * p["period"])["close"].to_numpy(dtype=float)
        if len(close) <= p["period"]:
            return
        r = float(wilder_rsi(close, p["period"])[-1])
        if not math.isfinite(r):
            return
        side = ctx.position.side
        if side is PositionSide.FLAT:
            if r < p["lower"]:
                ctx.submit_market("BUY", size_frac=p["size"], max_hold=p["max_hold"], tag="rsi_oversold")
            elif r > p["upper"] and p["allow_short"]:
                ctx.submit_market("SELL", size_frac=p["size"], max_hold=p["max_hold"], tag="rsi_overbought")
        elif (side is PositionSide.LONG and r >= p["exit_level"]) or (side is PositionSide.SHORT and r <= p["exit_level"]):
            ctx.close_position(tag="RSI_MID")

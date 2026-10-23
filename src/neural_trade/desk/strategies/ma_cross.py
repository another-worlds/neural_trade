"""ma_cross: fast / slow SMA crossover, the rules of ``neural_trade.strategy.ta_rules`` ``ta_ma_cross``.

Entry (flat, at the close of bar t): BUY while SMA(fast) > SMA(slow), SELL while below (a state, so a reversal re-enters
on the next flat bar; SELL only with ``allow_short``); nothing when the two are equal. Exit: the SMAs cross back
("CROSS"); the engine's TIME exit after ``max_hold`` bars. Warm-up = ``slow`` bars. The averages come from
``ta_rules.sma`` on the last ``slow`` closes of ``ctx.bars``. Bounds are the oracle's search ranges.
"""
from __future__ import annotations

import math

from neural_trade.desk.core.model import PositionSide
from neural_trade.desk.core.params import ParamSpec
from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy


@DeskStrategies.register(name="ma_cross", tags=["baseline", "ta"])
class MACross(DeskStrategy):
    name = "ma_cross"
    PARAMS = (
        ParamSpec("fast", "int", 10, 2, 30, 1, doc="fast SMA length"),
        ParamSpec("slow", "int", 30, 30, 240, 1, doc="slow SMA length (also the warm-up)"),
        ParamSpec("allow_short", "bool", True, doc="False: long only"),
        ParamSpec("size", "float", 1.0, 0.01, 1.0, 0.01, doc="fraction of equity per entry"),
        ParamSpec("max_hold", "int", 1440, 1, 100_000, 1, doc="bars held at most"),
    )

    def __init__(self, **params) -> None:
        super().__init__(**params)           # fast <= 30 <= slow holds over the whole bounded range
        self.warmup_bars = self.params["slow"]

    def on_bar(self, ctx) -> None:
        from neural_trade.strategy.ta_rules import sma   # lazy: the strategy package imports TensorFlow

        p = self.params
        close = ctx.bars(p["slow"])["close"].to_numpy(dtype=float)
        if len(close) < p["slow"]:
            return
        diff = float(sma(close, p["fast"])[-1] - sma(close, p["slow"])[-1])
        if not math.isfinite(diff):
            return
        side = ctx.position.side
        if side is PositionSide.FLAT:
            if diff == 0.0 or (diff < 0 and not p["allow_short"]):
                return
            ctx.submit_market("BUY" if diff > 0 else "SELL", size_frac=p["size"], max_hold=p["max_hold"],
                              tag="ma_cross")
        elif (1 if side is PositionSide.LONG else -1) * diff < 0:
            ctx.close_position(tag="CROSS")

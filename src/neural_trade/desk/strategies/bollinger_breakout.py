"""bollinger_breakout: SMA +- k sd breakout, the rules of ``neural_trade.strategy.ta_rules`` ``ta_bollinger``.

Entry (flat): BUY when the close is above SMA + k sd, SELL when below SMA - k sd (population sd, ``allow_short`` for
the SELL). Exit: the close back through the middle band ("BB_MID"); the engine's TIME exit after ``max_hold`` bars.
Warm-up = ``period`` bars. The bands come from ``ta_rules.sma`` and ``rolling_std`` run on ALL history the context
offers and read at the last bar: pandas' rolling mean and std are online (running sums), so their last value depends on
the whole series, and only the same call on the same series (the block from its start) reproduces the oracle exactly
(on flat stretches a fresh window gives sd 0 where the running value is a tiny positive number). The price is O(history)
per bar. Bounds are the oracle's search ranges.
"""
from __future__ import annotations

import math

from neural_trade.desk.core.model import PositionSide
from neural_trade.desk.core.params import ParamSpec
from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy


@DeskStrategies.register(name="bollinger_breakout", tags=["baseline", "ta"])
class BollingerBreakout(DeskStrategy):
    name = "bollinger_breakout"
    PARAMS = (
        ParamSpec("period", "int", 20, 5, 120, 1, doc="band length (also the warm-up)"),
        ParamSpec("k", "float", 2.0, 1.0, 3.0, 0.1, doc="band width in standard deviations"),
        ParamSpec("allow_short", "bool", True, doc="False: long only"),
        ParamSpec("size", "float", 1.0, 0.01, 1.0, 0.01, doc="fraction of equity per entry"),
        ParamSpec("max_hold", "int", 1440, 1, 100_000, 1, doc="bars held at most"),
    )

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.warmup_bars = self.params["period"]

    def on_bar(self, ctx) -> None:
        from neural_trade.strategy.ta_rules import rolling_std, sma   # lazy: the strategy package imports TensorFlow

        p = self.params
        close = ctx.bars(10 ** 9)["close"].to_numpy(dtype=float)
        if len(close) < p["period"]:
            return
        mid, sd = float(sma(close, p["period"])[-1]), float(rolling_std(close, p["period"])[-1])
        if not math.isfinite(mid):
            return
        c, side = float(close[-1]), ctx.position.side
        if side is PositionSide.FLAT:
            if c > mid + p["k"] * sd:
                ctx.submit_market("BUY", size_frac=p["size"], max_hold=p["max_hold"], tag="bb_breakout_up")
            elif c < mid - p["k"] * sd and p["allow_short"]:
                ctx.submit_market("SELL", size_frac=p["size"], max_hold=p["max_hold"], tag="bb_breakout_down")
        elif (1 if side is PositionSide.LONG else -1) * (c - mid) < 0:
            ctx.close_position(tag="BB_MID")

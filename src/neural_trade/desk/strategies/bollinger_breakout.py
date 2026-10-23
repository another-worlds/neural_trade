"""bollinger_breakout: SMA +- k sd breakout, the rules of ``neural_trade.strategy.ta_rules`` ``ta_bollinger``.

Entry (flat): BUY when the close is above SMA + k sd, SELL when below SMA - k sd (population sd, ``allow_short`` for
the SELL). Exit: the close back through the middle band ("BB_MID"); the engine's TIME exit after ``max_hold`` bars.
Warm-up = ``period`` bars. Bounds are the oracle's search ranges.

The bands come from the online kernels of ``_online`` (ports of pandas' rolling mean and population std, bit-equal to
``ta_rules.sma`` / ``rolling_std``): seeded on the first call from the history the context offers (all of it, or the
last ``ctx.max_seed_bars``), then updated by the closes that arrived since the last call, O(1) per bar. With the whole
block as the seed (a backtest) the bands are the oracle's bit for bit; ``on_start`` resets the state.
"""
from __future__ import annotations

import math

from neural_trade.desk.core.model import PositionSide
from neural_trade.desk.core.params import ParamSpec
from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy
from neural_trade.desk.strategies._online import CloseFeed, OnlineRollingStd, OnlineSMA


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

    def on_start(self, ctx) -> None:
        self._state = None   # (CloseFeed, OnlineSMA, OnlineRollingStd, (close, mid, sd))

    def _bands(self, ctx):
        """(close, mid, sd) at the current bar; mid and sd are NaN before ``period`` closes."""
        st = getattr(self, "_state", None)
        if st is None:
            n = self.params["period"]
            st = self._state = [CloseFeed(), OnlineSMA(n), OnlineRollingStd(n), (float("nan"),) * 3]
        feed, mean, std, _ = st
        for c in feed.new_closes(ctx, ctx.max_seed_bars):
            st[3] = (float(c), mean.update(c), std.update(c))
        return st[3]

    def on_bar(self, ctx) -> None:
        p = self.params
        c, mid, sd = self._bands(ctx)
        if not math.isfinite(mid):
            return
        side = ctx.position.side
        if side is PositionSide.FLAT:
            if c > mid + p["k"] * sd:
                ctx.submit_market("BUY", size_frac=p["size"], max_hold=p["max_hold"], tag="bb_breakout_up")
            elif c < mid - p["k"] * sd and p["allow_short"]:
                ctx.submit_market("SELL", size_frac=p["size"], max_hold=p["max_hold"], tag="bb_breakout_down")
        elif (1 if side is PositionSide.LONG else -1) * (c - mid) < 0:
            ctx.close_position(tag="BB_MID")

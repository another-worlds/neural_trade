"""random_entries: random side at a fixed entry rate and holding time (oracle ``RandomSignal``): the null model.

Matched-frequency use: set ``trade_rate`` to the strategy under test's entries divided by the bars on which it was
flat (and asked), ``hold_bars`` to its mean holding time in bars (at least 1) and ``size`` to its mean position size,
exactly as ``neural_trade.strategy.backtest.random_same_frequency`` does, then run several ``seed`` values: the spread
of their results is the null the strategy is ranked against.

Reproducible and causal: the draw of a bar is a function of ``(seed, ts_open_ms)`` only (``numpy.random.default_rng``),
so the same seed on the same bars gives the same entries whatever happened before, and the draws do not depend on how
long the block is (the oracle draws the whole block's schedule at once). Both numbers are drawn on every bar, so
``allow_short`` changes the side taken, never the bars entered. With ``allow_short`` False every entry is a BUY.
"""
from __future__ import annotations

import numpy as np

from neural_trade.desk.core.model import PositionSide
from neural_trade.desk.core.params import ParamSpec
from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy


@DeskStrategies.register(name="random_entries", tags=["baseline", "null"])
class RandomEntries(DeskStrategy):
    """Enters with probability ``trade_rate`` on a flat bar, holds ``hold_bars`` bars (TIME exit by the engine)."""

    name = "random_entries"
    PARAMS = (
        ParamSpec("trade_rate", "float", 0.05, 0.0, 1.0, 0.01, doc="probability of an entry on a flat bar"),
        ParamSpec("hold_bars", "int", 10, 1, 100_000, 1, doc="bars held before the TIME exit"),
        ParamSpec("seed", "int", 0, 0, 2 ** 31 - 1, 1, doc="draws are a function of (seed, bar open time)"),
        ParamSpec("size", "float", 1.0, 0.01, 1.0, 0.01, doc="fraction of equity per entry"),
        ParamSpec("allow_short", "bool", True, doc="False: every entry is a BUY"),
    )

    def on_bar(self, ctx) -> None:
        p = self.params
        if ctx.position.side is not PositionSide.FLAT:
            return
        rng = np.random.default_rng([p["seed"], ctx.bar.ts_open_ms])
        u, long_ = rng.random(), bool(rng.integers(0, 2))
        if u < p["trade_rate"]:
            buy = long_ or not p["allow_short"]
            ctx.submit_market("BUY" if buy else "SELL", size_frac=p["size"], max_hold=p["hold_bars"], tag="random")

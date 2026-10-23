"""rsi_threshold: Wilder-RSI mean reversion, the rules of ``neural_trade.strategy.ta_rules`` ``ta_rsi``.

Entry (flat): BUY when RSI < ``lower``, SELL when RSI > ``upper`` (SELL only with ``allow_short``). Exit: RSI back
through ``exit_level`` ("RSI_MID"); the engine's TIME exit after ``max_hold`` bars. Warm-up = ``period`` bars.

Wilder's RSI is recursive, so it is kept as a state per strategy instance (avg gain, avg loss): seeded on the first
call from ALL history the context offers (``ctx.bars`` of the whole source, which the Engine limits to the block start),
with the arithmetic of ``ta_rules.wilder_rsi`` (mean of the first ``period`` changes, then ``(g (n-1) + up) / n``), and
updated bar by bar with the same recursion over the closes that arrived since the last call (bars the engine did not
ask about, and holes in the data, are caught up in order; the oracle runs the recursion over the whole close array and
ignores breaks, and so does this). The values equal ``wilder_rsi``'s bit for bit; ``on_start`` resets the state (a
restart seeds again from history). ``ctx.max_seed_bars`` is ignored: the recursion depends on its start, so the
seed is always all history.
Bounds are the oracle's search ranges; ``lower < exit_level < upper`` is checked on construction.
"""
from __future__ import annotations

import math

import numpy as np

from neural_trade.desk.core.model import PositionSide
from neural_trade.desk.core.params import ParamSpec
from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy



def _rsi(gain: float, loss: float) -> float:
    if loss == 0.0:
        return 50.0 if gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


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
        self._state = None

    def on_start(self, ctx) -> None:
        self._state = None   # (gain, loss, last close, last ts)

    def _update(self, ctx) -> "float | None":
        """The RSI at the current bar, or None while there are not ``period`` + 1 closes."""
        n, cur = self.params["period"], ctx.bar.ts_open_ms
        st = getattr(self, "_state", None)
        if st is None:
            h = ctx.bars(10 ** 9)
            c = h["close"].to_numpy(dtype=float)
            if len(c) <= n:
                return None
            d = np.diff(c)
            up, dn = np.maximum(d, 0.0), np.maximum(-d, 0.0)
            gain, loss = float(up[:n].mean()), float(dn[:n].mean())
            for t in range(n + 1, len(c)):
                gain = (gain * (n - 1) + up[t - 1]) / n
                loss = (loss * (n - 1) + dn[t - 1]) / n
            self._state = (gain, loss, float(c[-1]), cur)
            return _rsi(gain, loss)
        gain, loss, last_c, last_ts = st
        if cur > last_ts:
            gap = max(2, int((cur - last_ts) // (ctx.bar.tf_minutes * 60_000)) + 1)
            h = ctx.bars(gap)
            h = h[h["ts"] > last_ts]
            for c in h["close"].to_numpy(dtype=float):
                up, dn = max(c - last_c, 0.0), max(last_c - c, 0.0)
                gain, loss, last_c = (gain * (n - 1) + up) / n, (loss * (n - 1) + dn) / n, c
            self._state = (gain, loss, last_c, cur)
        return _rsi(gain, loss)

    def on_bar(self, ctx) -> None:
        p = self.params
        r = self._update(ctx)
        if r is None or not math.isfinite(r):
            return
        side = ctx.position.side
        if side is PositionSide.FLAT:
            if r < p["lower"]:
                ctx.submit_market("BUY", size_frac=p["size"], max_hold=p["max_hold"], tag="rsi_oversold")
            elif r > p["upper"] and p["allow_short"]:
                ctx.submit_market("SELL", size_frac=p["size"], max_hold=p["max_hold"], tag="rsi_overbought")
        elif (side is PositionSide.LONG and r >= p["exit_level"]) or (side is PositionSide.SHORT and r <= p["exit_level"]):
            ctx.close_position(tag="RSI_MID")

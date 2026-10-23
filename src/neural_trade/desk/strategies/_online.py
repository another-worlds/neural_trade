"""Online (O(1) per bar) ports of pandas' fixed-window rolling mean and rolling standard deviation, bit-equal to
``neural_trade.strategy.ta_rules.sma`` and ``rolling_std`` (the oracle of the desk's TA built-ins, NT-227).

Ported from pandas **2.3.3** (``pandas/_libs/window/aggregations.pyx``: ``add_mean`` / ``remove_mean`` /
``calc_mean`` / ``roll_mean`` and ``add_var`` / ``remove_var`` / ``calc_var`` / ``roll_var``, plus ``Rolling.std``'s
``zsqrt`` and ``_prep_values``), the version of the ``nt`` env and of CI. The pandas kernels are themselves online: a
Kahan-compensated running sum (mean) and a Kahan-compensated Welford update (variance), each with separate
compensations for adds and removes, a count of non-NaN observations, a count of negative values (mean) and a count of
consecutive equal values that snaps a constant window to its exact value (GH#42064). Their value at bar t depends on
the whole series before t, not only on the last ``window`` values, and the state cannot be rebuilt from the outputs;
so the desk replays the same arithmetic in the same order, one close at a time.

What ``ta_rules`` asks for, and what is reproduced: ``Series(x).rolling(window)`` with ``min_periods = window`` (NaN
until ``window`` non-NaN values are in the window, and whenever a NaN or an inf is in it: ``_prep_values`` turns inf
into NaN), the population sd (``ddof = 0``), negative variances clipped to 0 by ``zsqrt``.

``tests/desk/test_online_kernels.py`` is the guard: it compares these classes with ``ta_rules.sma`` /
``rolling_std`` bit for bit on hard series (flats, integer prices, 1e-6 .. 1e9 magnitudes, NaN holes, windows
2 .. 240). If a pandas upgrade changes the algorithm, that test fails, and this module is re-ported before the
desk's TA strategies are trusted again.

Seeding from fewer bars than the whole history (``StrategyContext(max_seed_bars=...)``, a live bot) gives the right
values up to floating-point rounding, not bit for bit: the window's contents are exact once ``window`` bars are in it,
but the running sum carries the rounding of every earlier add and remove (a constant window is exact either way, by
the equal-value count). The backtest seeds from the block start and is exact.
"""
from __future__ import annotations

import math
from collections import deque

_NAN = float("nan")


def _prep(x: float) -> float:
    """pandas ``_prep_values``: float64, inf becomes NaN."""
    x = float(x)
    return _NAN if math.isinf(x) else x


class OnlineSMA:
    """``pd.Series(x).rolling(window).mean()``, one value at a time: ``update(x)`` returns the value at that bar."""

    __slots__ = ("window", "_buf", "_nobs", "_neg", "_sum", "_c_add", "_c_rem", "_same", "_prev", "_started")

    def __init__(self, window: int) -> None:
        if int(window) < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self.window = int(window)
        self._buf: deque = deque()
        self._nobs = self._neg = self._same = 0
        self._sum = self._c_add = self._c_rem = 0.0
        self._prev = _NAN
        self._started = False

    def update(self, x: float) -> float:
        v = _prep(x)
        if not self._started:                 # roll_mean's setup: prev_value = values[start]
            self._prev, self._started = v, True
        if len(self._buf) == self.window:     # remove_mean of the value leaving the window
            old = self._buf.popleft()
            if old == old:
                self._nobs -= 1
                y = -old - self._c_rem
                t = self._sum + y
                self._c_rem = t - self._sum - y
                self._sum = t
                if math.copysign(1.0, old) < 0:
                    self._neg -= 1
        self._buf.append(v)
        if v == v:                            # add_mean
            self._nobs += 1
            y = v - self._c_add
            t = self._sum + y
            self._c_add = t - self._sum - y
            self._sum = t
            if math.copysign(1.0, v) < 0:
                self._neg += 1
            self._same = self._same + 1 if v == self._prev else 1
            self._prev = v
        n = self._nobs                        # calc_mean, min_periods = window
        if n < self.window or n <= 0:
            return _NAN
        if self._same >= n:
            return self._prev
        r = self._sum / n
        if self._neg == 0 and r < 0:
            return 0.0
        if self._neg == n and r > 0:
            return 0.0
        return r


class OnlineRollingStd:
    """``pd.Series(x).rolling(window).std(ddof=0)``, one value at a time: ``update(x)`` returns the value at that
    bar."""

    __slots__ = ("window", "_buf", "_nobs", "_mean", "_ssq", "_c_add", "_c_rem", "_same", "_prev", "_started")

    def __init__(self, window: int) -> None:
        if int(window) < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self.window = int(window)
        self._buf: deque = deque()
        self._nobs = 0.0                      # a float64 in roll_var
        self._mean = self._ssq = self._c_add = self._c_rem = 0.0
        self._same = 0
        self._prev = _NAN
        self._started = False

    def update(self, x: float) -> float:
        v = _prep(x)
        if not self._started:
            self._prev, self._started = v, True
        if len(self._buf) == self.window:     # remove_var
            old = self._buf.popleft()
            if old == old:
                self._nobs -= 1
                if self._nobs:
                    prev_mean = self._mean - self._c_rem
                    y = old - self._c_rem
                    t = y - self._mean
                    self._c_rem = t + self._mean - y
                    self._mean = self._mean - t / self._nobs
                    self._ssq = self._ssq - (old - prev_mean) * (old - self._mean)
                else:
                    self._mean = self._ssq = 0.0
        self._buf.append(v)
        if v == v:                            # add_var (Welford with Kahan compensation)
            self._nobs += 1
            self._same = self._same + 1 if v == self._prev else 1
            self._prev = v
            prev_mean = self._mean - self._c_add
            y = v - self._c_add
            t = y - self._mean
            self._c_add = t + self._mean - y
            self._mean = self._mean + t / self._nobs if self._nobs else 0.0
            self._ssq = self._ssq + (v - prev_mean) * (v - self._mean)
        n = self._nobs                        # calc_var, min_periods = window, ddof 0
        if n < self.window or n <= 0:
            return _NAN
        var = 0.0 if (n == 1 or self._same >= n) else self._ssq / n
        return math.sqrt(var) if var >= 0 else (0.0 if var < 0 else _NAN)   # zsqrt


class CloseFeed:
    """The closes a strategy has not seen yet, in order: the first call reads up to ``seed_bars`` bars of history
    (None: all the context offers), later calls only the bars after the last one seen (bars the engine did not ask
    about and holes in the data are caught up in order)."""

    __slots__ = ("last_ts",)

    def __init__(self) -> None:
        self.last_ts = None

    def new_closes(self, ctx, seed_bars=None):
        cur = ctx.bar.ts_open_ms
        if self.last_ts is None:
            h = ctx.bars(10 ** 9 if seed_bars is None else int(seed_bars))
        elif cur > self.last_ts:
            gap = max(2, int((cur - self.last_ts) // (ctx.bar.tf_minutes * 60_000)) + 1)
            h = ctx.bars(gap)
            h = h[h["ts"] > self.last_ts]
        else:
            return ()
        self.last_ts = cur
        return h["close"].to_numpy(dtype=float)


__all__ = ["CloseFeed", "OnlineRollingStd", "OnlineSMA"]

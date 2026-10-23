"""NT-227: the desk's online rolling kernels (``desk/strategies/_online.py``) against the oracle's pandas calls.

The guard: ``OnlineSMA`` / ``OnlineRollingStd`` must equal ``ta_rules.sma`` / ``rolling_std`` bit for bit on hard
series. If a pandas upgrade changes its rolling algorithms, this fails and the port is redone (module docstring).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from neural_trade.desk.strategies._online import CloseFeed, OnlineRollingStd, OnlineSMA
from neural_trade.strategy.ta_rules import rolling_std, sma


def hard_series(rng: np.random.Generator, n: int) -> np.ndarray:
    """Prices from 1e-6 to 1e9, flat stretches, tick-rounded and small-integer prices, periodic blocks (exact ties),
    1000x jumps, negative series, NaN holes and the odd inf."""
    kind = int(rng.integers(0, 6))
    p0 = float(rng.choice([1e-6, 3e-4, 0.5, 7.0, 30000.0, 1e9]))
    c = p0 * np.exp(np.cumsum(rng.normal(0, rng.uniform(1e-5, 5e-3), n)))
    for _ in range(int(rng.integers(1, 8))):
        a = int(rng.integers(0, n - 5))
        c[a:a + int(rng.integers(2, 400))] = c[a]
    if kind == 1:
        c = np.round(c / (p0 * 1e-4)) * (p0 * 1e-4)
    elif kind == 2:
        c = np.round(c / p0 * 20) * p0 / 20
    elif kind == 3:
        per, k = int(rng.integers(2, 31)), int(rng.integers(0, n // 2))
        c[k:] = np.resize(c[:per].copy(), n - k)
    elif kind == 4:
        for _ in range(5):
            a = int(rng.integers(0, n - 2))
            c[a:] = c[a:] * rng.choice([1e-3, 1e3])
    elif kind == 5:
        c = -c
    if rng.random() < 0.4:
        for _ in range(int(rng.integers(1, 6))):
            a = int(rng.integers(0, n))
            c[a:a + int(rng.integers(1, 5))] = np.nan
        if rng.random() < 0.3:
            c[int(rng.integers(0, n))] = rng.choice([np.inf, -np.inf])
    return c


def online(cls, c, w):
    k = cls(w)
    return np.array([k.update(x) for x in c])


def test_the_module_names_the_installed_pandas_version():
    """An upgrade fails here until someone re-checks the port and writes the new version into the docstring."""
    import neural_trade.desk.strategies._online as mod
    assert f"pandas **{pd.__version__}**" in mod.__doc__


def test_online_kernels_equal_ta_rules_bit_for_bit_on_600_hard_series():
    """The guard (a pandas upgrade that changes the rolling algorithms fails here)."""
    rng = np.random.default_rng(20261023)
    t0, values, bad = time.perf_counter(), 0, []
    windows = set()
    for i in range(600):
        n = int(rng.integers(150, 1500))
        c = hard_series(rng, n)
        w = 240 if i == 0 else 2 if i == 1 else int(rng.integers(2, 241))
        windows.add(w)
        m, s = online(OnlineSMA, c, w), online(OnlineRollingStd, c, w)
        values += 2 * n
        if not np.array_equal(m, sma(c, w), equal_nan=True):
            bad.append(("mean", i, w))
        if not np.array_equal(s, rolling_std(c, w), equal_nan=True):
            bad.append(("std", i, w))
    assert not bad, bad[:10]
    assert values > 900_000 and {2, 240} <= windows
    assert time.perf_counter() - t0 < 30


def test_constant_windows_snap_to_the_exact_value_and_zero_sd():
    c = np.r_[np.linspace(1.0, 3.0, 50) * 0.1, np.full(30, 0.3)]
    w = 10
    m, s = online(OnlineSMA, c, w), online(OnlineRollingStd, c, w)
    assert (m[-20:] == 0.3).all() and (s[-20:] == 0.0).all()
    np.testing.assert_array_equal(m, sma(c, w))
    np.testing.assert_array_equal(s, rolling_std(c, w))


def test_nan_and_inf_blank_the_window_like_pandas():
    c = np.arange(1.0, 41.0)
    c[10], c[25] = np.nan, np.inf
    w = 5
    m, s = online(OnlineSMA, c, w), online(OnlineRollingStd, c, w)
    assert np.isnan(m[10:15]).all() and np.isnan(m[25:30]).all() and np.isfinite(m[15:25]).all()
    np.testing.assert_array_equal(m, sma(c, w))
    np.testing.assert_array_equal(s, rolling_std(c, w))
    assert np.isnan(m[:w - 1]).all() and np.isfinite(m[w - 1])   # min_periods = window


def test_window_below_two_is_refused():
    for cls in (OnlineSMA, OnlineRollingStd):
        with pytest.raises(ValueError, match="window must be at least 2"):
            cls(1)


def test_a_seed_of_the_last_three_windows_matches_the_full_history_up_to_rounding_only():
    """What a live bot with ``max_seed_bars`` gets: the window's contents are exact, the running sums are not. On a
    BTC-like walk the SMA differs by a few ulps and the sd by at most 1e-7 of the price (sd near 0 on quiet windows can
    differ by a large fraction of itself), so a seeded bot is close, not bit-equal: the backtest seeds from the block
    start."""
    rng = np.random.default_rng(7)
    c = np.round(60_000 * np.exp(np.cumsum(rng.normal(0, 4e-4, 12_000))), 2)
    c[3000:3040] = c[3000]
    for w in (5, 20, 240):
        full_m, full_s = online(OnlineSMA, c, w), online(OnlineRollingStd, c, w)
        start = 9000 - 3 * w
        part_m, part_s = online(OnlineSMA, c[start:], w)[3 * w:], online(OnlineRollingStd, c[start:], w)[3 * w:]
        price = np.abs(c[9000:])
        assert np.all(np.abs(part_m - full_m[9000:]) <= 1e-12 * price), w
        assert np.all(np.abs(part_s - full_s[9000:]) <= 1e-7 * price), w


class _Ctx:
    """The two things CloseFeed reads: the current bar and ``bars(n)``."""

    def __init__(self, c):
        self.df = pd.DataFrame({"ts": np.arange(len(c), dtype=np.int64) * 60_000, "close": c})
        self.t, self.asked = 0, []

    @property
    def bar(self):
        return type("B", (), {"ts_open_ms": self.t * 60_000, "tf_minutes": 1})()

    def bars(self, n):
        self.asked.append(n)
        return self.df.iloc[:self.t + 1].iloc[-n:]


def test_close_feed_seeds_once_then_catches_up_in_order_and_honours_the_seed_cap():
    c = np.arange(100.0, 200.0)
    ctx, feed, got = _Ctx(c), CloseFeed(), []
    for t in range(40, 100, 7):
        ctx.t = t
        got.extend(feed.new_closes(ctx, 25))
        assert len(feed.new_closes(ctx, 25)) == 0          # asked twice on one bar: nothing new
    assert got == list(c[40 - 24:ctx.t + 1]) and ctx.asked[0] == 25 and max(ctx.asked[1:]) <= 8

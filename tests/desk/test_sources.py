"""Data sources (DESK-1.5): closed bars in order, causal history, replay start/step/seek, live watcher."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_trade.desk.core.clock import SimClock
from neural_trade.desk.data import sources as srcmod
from neural_trade.desk.data.sources import DataSource, HistoricalSource, LiveSource, ReplaySource
from neural_trade.desk.data.store import MarketStore

SYM = "BTC/USDT"
MIN = 60_000
T0 = 1_700_000_100_000  # a 5-minute boundary


def _frame(idx) -> pd.DataFrame:
    """1-minute bars at T0 + i minutes; close = 100 + i so a bar identifies itself."""
    idx = np.asarray(list(idx), dtype=np.int64)
    c = 100.0 + idx
    return pd.DataFrame({"ts": T0 + idx * MIN, "o": c - 0.5, "h": c + 1.0, "l": c - 1.0, "c": c, "v": 10.0 + idx})


@pytest.fixture
def ms(tmp_path):
    with MarketStore(tmp_path / "market.sqlite") as s:
        yield s


@pytest.fixture
def gappy(ms):
    ms.write_bars(SYM, 1, _frame([0, 1, 2, 3, 7, 8, 9, 15]), "t")  # holes after 3 and 9
    return ms


def test_iteration_yields_closed_bars_in_order_across_gaps(gappy):
    bars = list(HistoricalSource(gappy, SYM, 1))
    assert [(b.ts_open_ms - T0) // MIN for b in bars] == [0, 1, 2, 3, 7, 8, 9, 15]
    assert all(b.is_closed and b.tf_minutes == 1 and b.instrument_id == "BINANCE:BTCUSDT" for b in bars)
    assert [b.close for b in bars] == [100, 101, 102, 103, 107, 108, 109, 115]
    assert bars[4].open == 106.5 and bars[4].high == 108.0 and bars[4].low == 106.0 and bars[4].volume == 17.0
    src = HistoricalSource(gappy, SYM, 1)
    assert isinstance(src, DataSource) and src.next_closed_bar() is not None


def test_exhausted_source_returns_none(gappy):
    src = HistoricalSource(gappy, SYM, 1)
    assert len(list(src)) == 8 and src.next_closed_bar() is None


def test_historical_range_is_start_inclusive_end_exclusive_and_matches_read_bars(gappy):
    start, end = T0 + 2 * MIN, T0 + 9 * MIN
    got = [b.ts_open_ms for b in HistoricalSource(gappy, SYM, 1, start, end)]
    assert got == [T0 + 2 * MIN, T0 + 3 * MIN, T0 + 7 * MIN, T0 + 8 * MIN]
    assert got == gappy.read_bars(SYM, 1, start, end)["ts"].tolist()
    src = HistoricalSource(gappy, SYM, 1, start, end)
    list(src)
    assert src.history(100).equals(gappy.read_bars(SYM, 1, start, end))


def test_history_is_causal_at_every_position(ms):
    ms.write_bars(SYM, 1, _frame(range(50)), "t")
    src = HistoricalSource(ms, SYM, 1)
    assert src.history(5).empty  # nothing delivered yet
    for k in range(50):
        bar = src.next_closed_bar()
        assert bar.ts_open_ms == T0 + k * MIN
        for n in (1, 3, 10, 49, 50, 1000):
            h = src.history(n)
            assert len(h) == min(n, k + 1)
            assert h["ts"].iloc[-1] == bar.ts_open_ms  # ends at the current bar
            assert h["ts"].max() <= bar.ts_open_ms  # no bar after it
            assert h["ts"].tolist() == [T0 + j * MIN for j in range(k + 1 - len(h), k + 1)]
        assert src.history(0).empty


def test_history_mutation_that_leaks_one_bar_is_caught(ms, monkeypatch):
    """Meta-check: the causality assertion above must fail for a history that looks one bar ahead."""
    ms.write_bars(SYM, 1, _frame(range(50)), "t")

    def leaky(self, n):
        return self._df.iloc[max(self._pos + 1 - n, 0):self._pos + 1].reset_index(drop=True)

    monkeypatch.setattr(srcmod._FrameSource, "history", leaky)
    src = HistoricalSource(ms, SYM, 1)
    bar = src.next_closed_bar()
    assert src.history(5)["ts"].max() > bar.ts_open_ms  # the leak is visible to the causality check


def test_replay_start_step_and_seek(ms):
    ms.write_bars(SYM, 1, _frame(range(20)), "t")
    r = ReplaySource(ms, SYM, 1, start=T0 + 5 * MIN)
    assert r.next_closed_bar().ts_open_ms == T0 + 5 * MIN
    assert [h for h in r.history(3)["ts"]] == [T0 + 3 * MIN, T0 + 4 * MIN, T0 + 5 * MIN]
    r2 = ReplaySource(ms, SYM, 1, start=T0 + 5 * MIN, step=3)
    assert [r2.next_closed_bar().ts_open_ms for _ in range(2)] == [T0 + 7 * MIN, T0 + 10 * MIN]
    assert r2.history(2)["ts"].tolist() == [T0 + 9 * MIN, T0 + 10 * MIN]  # skipped bars are history
    r2.seek(T0 + 2 * MIN)  # backward
    assert r2.next_closed_bar().ts_open_ms == T0 + 4 * MIN
    assert r2.history(10)["ts"].max() == T0 + 4 * MIN  # causal again after the seek
    r2.seek(T0 + 18 * MIN)
    assert r2.next_closed_bar().ts_open_ms == T0 + 19 * MIN  # step clipped at the end
    assert r2.next_closed_bar() is None
    r2.seek(None)
    assert r2.history(5).empty and r2.next_closed_bar().ts_open_ms == T0 + 2 * MIN
    with pytest.raises(ValueError):
        ReplaySource(ms, SYM, 1, step=0)


def test_replay_seek_lands_on_next_bar_inside_a_gap(gappy):
    r = ReplaySource(gappy, SYM, 1)
    r.seek(T0 + 5 * MIN)  # no bar at 4..6
    assert r.next_closed_bar().ts_open_ms == T0 + 7 * MIN


def test_resampled_last_bin_not_delivered_until_complete(ms):
    ms.write_bars(SYM, 1, _frame(range(13)), "t")  # bins 0-4, 5-9, 10-12 (partial)
    assert [b.ts_open_ms for b in HistoricalSource(ms, SYM, 5)] == [T0, T0 + 5 * MIN]
    ms.write_bars(SYM, 1, _frame([13, 14]), "t")
    bars = list(HistoricalSource(ms, SYM, 5))
    assert len(bars) == 3 and bars[2].high == 115.0 and bars[2].close == 114.0 and bars[2].tf_minutes == 5


def _live(ms, now):
    clock = SimClock(now)
    return clock, LiveSource(ms, SYM, 1, clock)


def test_live_emits_each_closed_bar_once_and_never_the_forming_one(ms):
    ms.write_bars(SYM, 1, _frame(range(3)), "t")
    clock, live = _live(ms, T0 + 3 * MIN)  # bars 0..2 are closed at construction: not replayed
    assert live.next_closed_bar() is None
    ms.write_bars(SYM, 1, _frame([3, 4]), "t")
    clock.set(T0 + 3 * MIN + 30_000)  # bar 3 still forming
    assert live.next_closed_bar() is None
    clock.set(T0 + 4 * MIN)  # bar 3 closes
    b = live.next_closed_bar()
    assert b.ts_open_ms == T0 + 3 * MIN and b.is_closed and b.close == 103.0
    assert live.next_closed_bar() is None  # bar 4 forming, bar 3 not repeated
    clock.set(T0 + 10 * MIN)
    assert [x.ts_open_ms for x in live] == [T0 + 4 * MIN]
    assert live.next_closed_bar() is None and live.last_ts == T0 + 4 * MIN
    assert live.history(2)["ts"].tolist() == [T0 + 3 * MIN, T0 + 4 * MIN]


def test_live_after_ts_replays_newer_bars_in_order_with_gaps(ms):
    ms.write_bars(SYM, 1, _frame([0, 1, 5, 6]), "t")
    live = LiveSource(ms, SYM, 1, SimClock(T0 + 100 * MIN), after_ts=T0)
    assert [(b.ts_open_ms - T0) // MIN for b in live] == [1, 5, 6]


def test_live_late_bar_is_not_delivered_out_of_order(ms):
    ms.write_bars(SYM, 1, _frame([0, 1, 3]), "t")
    live = LiveSource(ms, SYM, 1, SimClock(T0 + 100 * MIN), after_ts=T0 - MIN)
    assert [(b.ts_open_ms - T0) // MIN for b in live] == [0, 1, 3]
    ms.write_bars(SYM, 1, _frame([2]), "t")  # written late: older than the last delivered
    assert live.next_closed_bar() is None  # never emitted, nothing repeated
    ms.write_bars(SYM, 1, _frame([4]), "t")
    assert live.next_closed_bar().ts_open_ms == T0 + 4 * MIN
    assert live.history(10)["ts"].tolist() == [T0 + i * MIN for i in range(5)]  # the late bar is in the store


def test_live_on_empty_store_starts_with_the_first_bar(ms):
    clock, live = _live(ms, T0 + 10 * MIN)
    assert live.next_closed_bar() is None and live.history(3).empty
    ms.write_bars(SYM, 1, _frame([0]), "t")
    assert live.next_closed_bar().ts_open_ms == T0

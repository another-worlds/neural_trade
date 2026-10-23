"""Data sources (CONTRACTS ``desk/data/sources.py``, DESK-1.5): what an Engine reads closed bars from.

``DataSource`` is the protocol ``next_closed_bar() -> Optional[Bar]`` and ``history(n) -> DataFrame``. Every bar a
source hands out has ``is_closed=True`` (the collector writes closed candles only, PLAN A3); ``history(n)`` holds the
``n`` newest bars up to and including the one last delivered, never a later one (causality). Frames have the store's
columns ``ts, o, h, l, c, v`` (ascending ``ts``, UTC ms of the bar open).

Gaps in the contract (minimum implemented, nothing invented):

* iteration (``for bar in source``) and ``reset``/``seek``/``step`` are not in the protocol; they are added here;
* ``ReplaySource(step=k)`` advances ``k`` bars per call and returns the last of them (the skipped bars stay in
  ``history``); ``seek(ts)`` moves the position so the next bar delivered is the first with open ``>= ts`` (backward
  too; the replay session then builds a new Engine and SimClock, PLAN 3.7);
* a resampled last bin that the 1-minute data does not cover completely is not delivered (not closed yet);
* ``LiveSource`` starts after the newest closed bar present at construction unless ``after_ts`` is given; a bar written
  late with ``ts <= `` the last delivered one is never delivered (the position only moves forward; it stays in
  ``history``); a bar is closed when ``ts_open + tf <= clock.now_ms()``; it polls, it starts no thread.
"""
from __future__ import annotations

from typing import Iterator, Optional, Protocol, runtime_checkable

import pandas as pd

from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.model import Bar
from neural_trade.desk.data.store import BASE_TF, TimeLike, MarketStore, _ms

__all__ = ["DataSource", "HistoricalSource", "ReplaySource", "LiveSource"]

_MS_PER_MIN = 60_000
_COLS = ["ts", "o", "h", "l", "c", "v"]


@runtime_checkable
class DataSource(Protocol):
    def next_closed_bar(self) -> Optional[Bar]: ...

    def history(self, n: int) -> pd.DataFrame: ...


def _instrument_id(symbol: str) -> str:
    return "BINANCE:" + symbol.replace("/", "")


def _bar(row, instrument_id: str, tf: int) -> Bar:
    ts, o, h, lo, c, v = row
    return Bar(instrument_id=instrument_id, tf_minutes=tf, ts_open_ms=int(ts), open=float(o), high=float(h),
               low=float(lo), close=float(c), volume=float(v), is_closed=True)


def _closed_frame(store: MarketStore, symbol: str, tf: int, start, end) -> pd.DataFrame:
    df = store.read_bars(symbol, tf, start, end)
    if tf != BASE_TF and not df.empty and store.ranges(symbol, tf) == []:
        # resampled: the last bin is closed only when the 1-minute data reaches its end
        base_last = store.last_ts(symbol, BASE_TF)
        if base_last is None or base_last + _MS_PER_MIN < int(df["ts"].iloc[-1]) + tf * _MS_PER_MIN:
            df = df.iloc[:-1]
    return df.reset_index(drop=True)


class _FrameSource:
    """Bars held in a frame; ``_pos`` is the number of bars delivered (the next bar is ``_df[_pos]``)."""

    def __init__(self, df: pd.DataFrame, instrument_id: str, tf: int, pos: int = 0) -> None:
        self._df, self._id, self._tf, self._pos = df, instrument_id, int(tf), pos

    def __len__(self) -> int:
        return len(self._df)

    @property
    def position(self) -> int:
        return self._pos

    def _take(self, k: int) -> Optional[Bar]:
        if self._pos >= len(self._df):
            return None
        self._pos = min(self._pos + k, len(self._df))
        return _bar(self._df.iloc[self._pos - 1][_COLS], self._id, self._tf)

    def next_closed_bar(self) -> Optional[Bar]:
        return self._take(1)

    def history(self, n: int) -> pd.DataFrame:
        n = int(n)
        if n <= 0:
            return self._df.iloc[0:0].copy()
        return self._df.iloc[max(self._pos - n, 0):self._pos].reset_index(drop=True)

    def __iter__(self) -> Iterator[Bar]:
        while True:
            bar = self.next_closed_bar()
            if bar is None:
                return
            yield bar


class HistoricalSource(_FrameSource):
    """The bars of ``[start, end)`` (as ``MarketStore.read_bars``) delivered in order, once; ``reset()`` restarts."""

    def __init__(self, store: MarketStore, symbol: str, tf: int, start: TimeLike = None, end: TimeLike = None, *,
                 instrument_id: Optional[str] = None) -> None:
        super().__init__(_closed_frame(store, symbol, int(tf), start, end), instrument_id or _instrument_id(symbol),
                         int(tf))

    def reset(self) -> None:
        self._pos = 0


class ReplaySource(_FrameSource):
    """Interactive replay from ``start``: bars before it are history, the first delivered is the first with open
    ``>= start``. ``step`` bars per call (default 1); ``seek(ts)`` repositions in either direction."""

    def __init__(self, store: MarketStore, symbol: str, tf: int, start: TimeLike = None, step: int = 1, *,
                 end: TimeLike = None, instrument_id: Optional[str] = None) -> None:
        if int(step) < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        super().__init__(_closed_frame(store, symbol, int(tf), None, end), instrument_id or _instrument_id(symbol),
                         int(tf))
        self._step = int(step)
        self.seek(start)

    def seek(self, ts: TimeLike) -> None:
        """The next bar delivered is the first with open ``>= ts`` (None: the first bar)."""
        t = _ms(ts)
        self._pos = 0 if t is None else int(self._df["ts"].searchsorted(t, side="left"))

    def next_closed_bar(self) -> Optional[Bar]:
        return self._take(self._step)


class LiveSource:
    """Store watcher: each call returns the oldest closed bar newer than the last one delivered, or None.

    The clock decides what is closed (``ts_open + tf <= now``), so a bar of the forming candle is never delivered.
    """

    def __init__(self, store: MarketStore, symbol: str, tf: int, clock: Clock, *, after_ts: TimeLike = None,
                 instrument_id: Optional[str] = None) -> None:
        self._store, self._symbol, self._tf, self._clock = store, symbol, int(tf), clock
        self._id = instrument_id or _instrument_id(symbol)
        after = _ms(after_ts)
        self._last: Optional[int] = after if after is not None else self._newest_closed()

    def _newest_closed(self) -> Optional[int]:
        df = self._store.read_bars(self._symbol, self._tf, None, self._closed_end(), 1)
        return None if df.empty else int(df["ts"].iloc[-1])

    def _closed_end(self) -> int:
        return self._clock.now_ms() - self._tf * _MS_PER_MIN + 1  # exclusive bound on the open time

    @property
    def last_ts(self) -> Optional[int]:
        return self._last

    def next_closed_bar(self) -> Optional[Bar]:
        start = None if self._last is None else self._last + 1
        df = self._store.read_bars(self._symbol, self._tf, start, self._closed_end())
        if df.empty:
            return None
        row = df.iloc[0]
        self._last = int(row["ts"])
        return _bar(row[_COLS], self._id, self._tf)

    def history(self, n: int) -> pd.DataFrame:
        n = int(n)
        if n <= 0 or self._last is None:
            return pd.DataFrame(columns=_COLS)
        return self._store.read_bars(self._symbol, self._tf, None, self._last + 1, n)

    def __iter__(self) -> Iterator[Bar]:
        while True:
            bar = self.next_closed_bar()
            if bar is None:
                return
            yield bar

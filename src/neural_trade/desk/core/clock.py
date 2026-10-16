"""Clocks (C1). Strategies and engines read time only through a Clock, never ``time.time()``."""
from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...


class SimClock:
    """Set by the engine to each closed bar's close time; may only move forward."""

    def __init__(self, start_ms: int = 0) -> None:
        self._now = int(start_ms)

    def set(self, ts_ms: int) -> None:
        ts_ms = int(ts_ms)
        if ts_ms < self._now:
            raise ValueError(f"SimClock cannot go back: {ts_ms} < {self._now}")
        self._now = ts_ms

    def now_ms(self) -> int:
        return self._now


class LiveClock:
    """Wall clock in UTC milliseconds."""

    def now_ms(self) -> int:
        return int(time.time() * 1000)

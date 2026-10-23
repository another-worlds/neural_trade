"""Synchronous in-process event bus (C1): handlers run in subscription order.

A raising handler does not stop delivery: the failure is logged, kept in ``bus.errors`` (last 100) and
reported to ``on_error``. ``strict=True`` restores fail-fast (the exception propagates, later handlers miss the event).
"""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("neural_trade.desk.bus")

ERRORS_KEPT = 100


@dataclass(frozen=True)
class BusError:
    topic: str
    handler: str  # repr of the handler
    exc: Exception


class EventBus:
    def __init__(self, *, on_error: Callable[[str, Callable, Exception], None] | None = None, strict: bool = False) -> None:
        self._subs: dict[str, list[list]] = {}  # topic -> [[handler], ...]; the one-item list is the subscription's identity
        self._on_error = on_error
        self._strict = strict
        self.errors: deque[BusError] = deque(maxlen=ERRORS_KEPT)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> Callable[[], None]:
        """Register ``handler``; returns an idempotent unsubscribe callable (the same function twice is two subscriptions)."""
        sub = [handler]
        self._subs.setdefault(topic, []).append(sub)

        def unsubscribe() -> None:
            lst = self._subs.get(topic, [])
            for i, s in enumerate(lst):
                if s is sub:
                    del lst[i]
                    break

        return unsubscribe

    def publish(self, topic: str, event: Any) -> None:
        for sub in list(self._subs.get(topic, ())):  # snapshot: handlers may (un)subscribe while we deliver
            handler = sub[0]
            if self._strict:
                handler(event)
                continue
            try:
                handler(event)
            except Exception as exc:  # BaseException (KeyboardInterrupt, SystemExit) propagates
                self._record(topic, handler, exc)

    def _record(self, topic: str, handler: Callable, exc: Exception) -> None:
        log.error("handler %r failed on topic %r: %r", handler, topic, exc, exc_info=exc)
        self.errors.append(BusError(topic, repr(handler), exc))
        if self._on_error is not None:
            try:
                self._on_error(topic, handler, exc)
            except Exception:
                log.exception("on_error callback failed (topic %r)", topic)

"""Synchronous in-process event bus (C1): handlers run in subscription order."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[list]] = {}  # topic -> [[handler], ...]; the one-item list is the subscription's identity

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
            sub[0](event)

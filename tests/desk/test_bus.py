"""DESK-0.4: EventBus ordering and unsubscribe."""
from __future__ import annotations

from neural_trade.desk.core.bus import EventBus


def test_subscription_order_and_topics():
    bus, seen = EventBus(), []
    bus.subscribe("a", lambda e: seen.append(("1", e)))
    bus.subscribe("b", lambda e: seen.append(("x", e)))
    bus.subscribe("a", lambda e: seen.append(("2", e)))
    bus.publish("a", 7)
    assert seen == [("1", 7), ("2", 7)]
    bus.publish("none", 1)  # no subscribers: fine


def test_unsubscribe_is_idempotent_and_per_subscription():
    bus, seen = EventBus(), []

    def h(e):
        seen.append(e)

    off1 = bus.subscribe("t", h)
    bus.subscribe("t", h)  # same function twice = two subscriptions
    bus.publish("t", 1)
    off1()
    off1()
    bus.publish("t", 2)
    assert seen == [1, 1, 2]


def test_unsubscribe_during_publish():
    bus, seen = EventBus(), []
    offs = []
    offs.append(bus.subscribe("t", lambda e: (seen.append("a"), offs[1]())))
    offs.append(bus.subscribe("t", lambda e: seen.append("b")))
    bus.publish("t", 0)  # snapshot: b still gets this event
    bus.publish("t", 0)
    assert seen == ["a", "b", "a"]

"""DESK-0.4: EventBus ordering and unsubscribe."""
from __future__ import annotations

import logging

import pytest

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


@pytest.fixture
def buslog():
    """Records of the bus logger (attached directly: the package logger may not propagate to the root)."""
    recs = []

    class H(logging.Handler):
        def emit(self, record):
            recs.append(record)

    lg = logging.getLogger("neural_trade.desk.bus")
    h = H(level=logging.DEBUG)
    lg.addHandler(h)
    yield recs
    lg.removeHandler(h)


def _boom(e):
    raise ValueError(f"boom{e}")


def test_raising_handler_does_not_block_others_and_is_recorded():
    calls = []
    bus = EventBus(on_error=lambda t, h, x: calls.append((t, h, x)))
    seen = []
    bus.subscribe("t", lambda e: seen.append(("1", e)))
    bus.subscribe("t", _boom)
    bus.subscribe("t", lambda e: seen.append(("3", e)))
    bus.publish("t", 5)  # returns normally
    assert seen == [("1", 5), ("3", 5)]
    assert len(bus.errors) == 1
    err = bus.errors[0]
    assert err.topic == "t" and "_boom" in err.handler
    assert isinstance(err.exc, ValueError) and str(err.exc) == "boom5"
    assert len(calls) == 1 and calls[0][0] == "t" and calls[0][1] is _boom and calls[0][2] is err.exc


def test_one_log_record_per_failure(buslog):
    bus = EventBus()
    bus.subscribe("t", _boom)
    bus.subscribe("t", _boom)
    bus.publish("t", 1)
    recs = buslog
    assert len(recs) == 2 and "'t'" in recs[0].getMessage() and "_boom" in recs[0].getMessage()


def test_strict_keeps_fail_fast():
    bus, seen = EventBus(strict=True), []
    bus.subscribe("t", lambda e: seen.append("1"))
    bus.subscribe("t", _boom)
    bus.subscribe("t", lambda e: seen.append("3"))
    with pytest.raises(ValueError, match="boom9"):
        bus.publish("t", 9)
    assert seen == ["1"] and not bus.errors


def test_on_error_raising_is_swallowed_and_logged(buslog):
    def bad(t, h, x):
        raise RuntimeError("cb")

    bus, seen = EventBus(on_error=bad), []
    bus.subscribe("t", _boom)
    bus.subscribe("t", lambda e: seen.append(e))
    bus.publish("t", 2)
    assert seen == [2] and len(bus.errors) == 1
    assert any("on_error" in r.getMessage() for r in buslog)


def test_errors_bounded_to_last_100():
    bus = EventBus()
    bus.subscribe("t", _boom)
    for i in range(150):
        bus.publish("t", i)
    assert len(bus.errors) == 100
    assert str(bus.errors[0].exc) == "boom50" and str(bus.errors[-1].exc) == "boom149"


def test_base_exceptions_propagate():
    for exc in (KeyboardInterrupt, SystemExit):
        bus, seen = EventBus(), []

        def h(e, exc=exc):
            raise exc()

        bus.subscribe("t", h)
        bus.subscribe("t", lambda e, seen=seen: seen.append(e))
        with pytest.raises(exc):
            bus.publish("t", 1)
        assert seen == [] and not bus.errors


def test_handler_unsubscribes_itself_then_raises():
    bus, seen = EventBus(), []
    offs = []

    def h(e):
        offs[0]()
        raise ValueError("x")

    offs.append(bus.subscribe("t", h))
    bus.subscribe("t", lambda e: seen.append(e))
    bus.publish("t", 1)
    bus.publish("t", 2)
    assert seen == [1, 2] and len(bus.errors) == 1


def test_subscriber_added_during_publish_misses_current_event():
    bus, seen = EventBus(), []

    def adder(e):
        bus.subscribe("t", lambda x: seen.append(("late", x)))
        raise ValueError("after adding")

    bus.subscribe("t", adder)
    bus.publish("t", 1)
    assert seen == []
    assert len(bus.errors) == 1

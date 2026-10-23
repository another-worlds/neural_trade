"""Broker protocols (CONTRACTS C1 ``desk/broker/base.py``).

The command side (``submit`` / ``cancel`` / ``modify``) is ``core.execution.Broker``, re-exported here unchanged. The
bar-stepping side C1 lists next to it (``fill_exits``, ``fill_entries``, ``on_bar_intrabar``, ``sync``) is
``BarBroker``; ``fill_at_close`` is a gap C1 leaves open (A5.5 / A5.12 close a position at a bar's CLOSE, which no
C1 method can fill). The broker only REPORTS (``Fill`` / ``OrderEvent``); the ExecutionEngine journals and applies.
"""
from __future__ import annotations

from typing import Protocol

from neural_trade.desk.core.execution import Broker
from neural_trade.desk.core.model import Bar, DeskError, Fill, OrderEvent


class UnknownBrokerOrder(DeskError):
    """cancel / modify of an order the broker does not hold (never submitted, or already filled, cancelled, expired)."""


class BarBroker(Broker, Protocol):
    def fill_exits(self, bar: Bar) -> list[Fill | OrderEvent]: ...  # A5.9 step 1
    def fill_entries(self, bar: Bar, cash: float) -> list[Fill | OrderEvent]: ...  # step 2; ``cash`` = realised equity
    def on_bar_intrabar(self, bar: Bar) -> list[Fill | OrderEvent]: ...  # step 3
    def fill_at_close(self, bar: Bar) -> list[Fill | OrderEvent]: ...  # step 4: EOW exits at the bar's close
    def sync(self) -> list[Fill | OrderEvent]: ...


__all__ = ["BarBroker", "Broker", "UnknownBrokerOrder"]

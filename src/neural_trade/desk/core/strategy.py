"""Strategy API (CONTRACTS C1/C2, PLAN A2 ``strategy.py``, A7; DESK-1.6): ``DeskStrategy``, ``StrategyContext`` and the
``DeskStrategies`` registry. Pure logic, standard library plus pandas; no TensorFlow.

A strategy decides on the CLOSED bar ``t`` and talks to the engine only through the context's order helpers. The Engine
(DESK-1.7) owns WHEN ``on_bar`` runs (A5.9 step 4, A5.10): after ``warmup_bars`` bars, never on the last bar of a
backtest block, never on a bar flagged as a break, and not when a TIME / EOW exit is queued. The built-ins also guard
themselves: they place an entry only when their own position is flat and only with enough history.

The context hands out copies and read-only views: ``position`` and ``account`` are snapshots, ``clock`` has ``now_ms()``
only, ``bars(n)`` is a fresh frame cut at the current bar (it never shows a later bar, whatever the source returns), and
``cancel`` refuses an order this strategy did not submit. Nothing public exposes the ExecutionEngine, Portfolio or Broker.

Gaps CONTRACTS leaves open (minimum implemented; the NT-227 report lists them):

* the ``StrategyContext`` constructor is not in C1: ``StrategyContext(execution=, source=, strategy_id=, clock=)`` plus
  optional ``predictions=`` (a callable returning the dict or None) and ``log_sink=``; the Engine calls ``ctx._advance(bar)``
  once per closed bar before ``on_bar``;
* ``DataSource.history`` has the store's columns ``ts, o, h, l, c, v``; ``ctx.bars(n)`` renames them to the contract's
  ``ts, open, high, low, close, volume`` (the source is unchanged);
* a bracket dict is built whenever any of ``tp``, ``sl``, ``max_hold`` is given (``max_hold`` travels in
  ``Order.bracket["max_hold"]``, which is how the Engine reads it, A5.4); the order tags carry ``tag`` and ``reason``;
* client ids are ``<strategy_id>:<session>:<n>``: ``session`` is the constructor argument or a random 6-hex string
  (DESK-2.5 persists it), ``n`` a per-context counter; two contexts of one strategy id never collide;
* ``close_position`` builds its own reduce-only market order (not ``ExecutionEngine.close_position``) so that
  ``tags["reason"]`` is the exit tag (CROSS, RSI_MID, BB_MID, TIME ...), ``"CLOSE"`` when no tag is given;
* the Engine (DESK-1.7) must give a backtest source only the bars from the block start, so desk indicators see what the
  oracle sees (pre-block history is not exposed); the context fails loudly (RuntimeError) on a stale or out-of-order bar;
* ``max_seed_bars`` (keyword, default None = all history): how many most-recent bars a strategy with an online state
  reads when it seeds on its first ``on_bar``. A backtest leaves it None (the block from its start: ``ma_cross`` and
  ``bollinger_breakout`` are then the oracle's bit for bit). A live bot sets it (a few times the longest window) so that
  a start does not read the whole store; the SMA / sd values are then right up to floating-point rounding (the window's
  contents are exact after ``window`` bars, the running sums carry rounding from the seed start), not bit for bit.
  ``rsi_threshold`` ignores it and seeds from all history (Wilder's recursion never forgets its start);
* ``warmup_bars`` is a ClassVar default of 0; strategies whose warm-up depends on parameters set it on the instance;
* A7 mentions ``clock.now``; C1's ``Clock`` has ``now_ms()``, which is what the read-only view offers.
"""
from __future__ import annotations

import dataclasses
import inspect
import logging
import secrets
from typing import Any, Callable, ClassVar, Optional

import pandas as pd

from neural_trade.core.registry import BaseRegistry
from neural_trade.desk.core.clock import Clock
from neural_trade.desk.core.model import (AccountState, Bar, DeskError, Order, OrderEvent, OrderType, Position,
                                          PositionSide, Side, TimeInForce)
from neural_trade.desk.core.params import ParamSpec, params_schema, validate_params

BAR_COLUMNS = ("ts", "open", "high", "low", "close", "volume")
_SOURCE_COLUMNS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}


class IndicatorsUnavailable(DeskError):
    """``ctx.indicator`` before DESK-7.1 wires it to ``ChartIndicators``."""


class DeskStrategy:
    """Base class of a desk strategy (C1). Subclasses set ``name`` and ``PARAMS`` and implement ``on_bar``."""

    name: ClassVar[str]
    PARAMS: ClassVar[tuple[ParamSpec, ...]] = ()
    warmup_bars: ClassVar[int] = 0

    def __init__(self, **params: Any) -> None:
        """Validate ``params`` against ``PARAMS`` (defaults filled); out of range raises ValueError naming the bound."""
        self.params: dict[str, Any] = validate_params(self.PARAMS, params)

    @classmethod
    def params_schema(cls) -> list[dict]:
        """Every ParamSpec as a dict, bounds included (UI forms, sweep spaces)."""
        return params_schema(cls.PARAMS)

    def on_start(self, ctx: "StrategyContext") -> None:
        pass

    def on_bar(self, ctx: "StrategyContext") -> None:  # called once per CLOSED bar
        raise NotImplementedError(f"{type(self).__name__} must implement on_bar")

    def on_order_event(self, ctx: "StrategyContext", event: OrderEvent) -> None:
        pass

    def on_stop(self, ctx: "StrategyContext") -> None:
        pass


class _ReadOnlyClock:
    """The engine's clock as a strategy sees it: it can read the time, never set it."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def now_ms(self) -> int:
        return self._clock.now_ms()


class StrategyContext:
    """What ``on_bar`` sees (C1). Built by the Engine, one per strategy."""

    def __init__(self, *, execution: Any, source: Any, strategy_id: str, clock: Clock,
                 predictions: Optional[Callable[[], Optional[dict]]] = None,
                 log_sink: Optional[Callable[[str], None]] = None, session: Optional[str] = None,
                 max_seed_bars: Optional[int] = None) -> None:
        if max_seed_bars is not None and int(max_seed_bars) < 1:
            raise ValueError(f"max_seed_bars must be positive or None, got {max_seed_bars}")
        self.session = session if session is not None else secrets.token_hex(3)
        self._max_seed_bars = None if max_seed_bars is None else int(max_seed_bars)
        self._exec, self._source, self._strategy_id = execution, source, strategy_id
        self._predictions, self._log_sink = predictions, log_sink
        self._clock = _ReadOnlyClock(clock)
        self._bar: Optional[Bar] = None
        self._seq = 0
        self._mine: set[str] = set()
        self._logger = logging.getLogger(f"neural_trade.desk.strategy.{strategy_id}")

    # ------------------------------------------------------------------ engine side
    def _advance(self, bar: Bar) -> None:
        """The Engine calls this once per closed bar, before ``on_bar``."""
        if not bar.is_closed:
            raise ValueError("a strategy sees closed bars only")
        if self._bar is not None and bar.ts_open_ms <= self._bar.ts_open_ms:
            raise RuntimeError(f"stale context: bar {bar.ts_open_ms} does not come after {self._bar.ts_open_ms}")
        latest = self._source.history(1)
        if latest.empty or int(latest["ts"].iloc[-1]) != bar.ts_open_ms:
            raise RuntimeError(f"stale context: the source's latest bar is not {bar.ts_open_ms}")
        self._bar = bar

    def _need_bar(self) -> Bar:
        if self._bar is None:
            raise RuntimeError("no current bar: the engine has not started this context")
        return self._bar

    # ------------------------------------------------------------------ reads
    @property
    def bar(self) -> Bar:
        return self._need_bar()

    @property
    def max_seed_bars(self) -> Optional[int]:
        """The most bars a strategy reads to seed its online state (None: all history; module docstring)."""
        return self._max_seed_bars

    @property
    def clock(self) -> _ReadOnlyClock:
        return self._clock

    @property
    def position(self) -> Position:
        """A copy of this strategy's position (FLAT when none)."""
        pos = self._exec.portfolio.position(self._need_bar().instrument_id, self._strategy_id)
        return dataclasses.replace(pos)

    @property
    def account(self) -> AccountState:
        """A snapshot of the account marked at the current close."""
        bar = self._need_bar()
        marks = {**self._exec.marks, bar.instrument_id: bar.close}
        return self._exec.portfolio.account_state(marks, self._clock.now_ms())

    def bars(self, n: int) -> pd.DataFrame:
        """The last ``n`` closed bars up to and including the current one, columns ``ts, open, high, low, close,
        volume`` (a fresh frame; fewer rows early in the run). A row after the current bar is dropped even if the
        source returns it."""
        return self._cut(n, self._need_bar().ts_open_ms)

    def _cut(self, n: int, cur: int) -> pd.DataFrame:
        n = int(n)
        if n <= 0:
            return pd.DataFrame({c: [] for c in BAR_COLUMNS})
        k = n
        while True:   # a conforming source needs one pass; one that over-reads is asked for more until n rows survive
            h = self._source.history(k)
            kept = h[h["ts"] <= cur]
            if len(kept) >= n or len(h) < k:
                break
            k *= 2
        h = kept.iloc[-n:]
        return h.rename(columns=_SOURCE_COLUMNS).reset_index(drop=True)[list(BAR_COLUMNS)]

    def indicator(self, name: str, **params: Any) -> pd.DataFrame:
        raise IndicatorsUnavailable(
            f"indicator {name!r}: classic indicators are not wired into the strategy context yet "
            f"(DESK-7.1 connects ctx.indicator to ChartIndicators)")

    def predictions(self) -> Optional[dict]:
        """The attached model's latest prediction dict, or None when no model is attached."""
        return None if self._predictions is None else self._predictions()

    # ------------------------------------------------------------------ order helpers
    def submit_market(self, side: str, *, qty: Optional[float] = None, size_frac: Optional[float] = None, tp=None,
                      sl=None, tp_is_offset: bool = False, max_hold: Optional[int] = None, tag: str = "") -> str:
        return self._submit(side, OrderType.MARKET, qty, size_frac, tag, tp=tp, sl=sl, tp_is_offset=tp_is_offset,
                            max_hold=max_hold)

    def submit_limit(self, side: str, price: float, *, qty: Optional[float] = None, size_frac: Optional[float] = None,
                     tif: str = "GTC", tp=None, sl=None, tp_is_offset: bool = False, tag: str = "") -> str:
        return self._submit(side, OrderType.LIMIT, qty, size_frac, tag, price=price, tif=tif, tp=tp, sl=sl,
                            tp_is_offset=tp_is_offset)

    def submit_stop(self, side: str, trigger: float, *, qty: Optional[float] = None, size_frac: Optional[float] = None,
                    limit: Optional[float] = None, tag: str = "") -> str:
        typ = OrderType.STOP_MARKET if limit is None else OrderType.STOP_LIMIT
        return self._submit(side, typ, qty, size_frac, tag, price=limit, trigger_price=trigger)

    def cancel(self, client_id: str) -> None:
        if client_id not in self._mine:
            raise ValueError(f"{client_id}: not an order of strategy {self._strategy_id!r}")
        self._exec.cancel(client_id)

    def close_position(self, tag: str = "") -> None:
        """A reduce-only market order for the whole position (fills at the next open); nothing when flat."""
        bar = self._need_bar()
        pos = self._exec.portfolio.positions.get(f"{bar.instrument_id}-{self._strategy_id}")
        if pos is None or pos.qty <= 0 or pos.side is PositionSide.FLAT:
            return
        self._seq += 1
        cid = f"{self._strategy_id}:{self.session}:{self._seq}"
        self._mine.add(cid)
        self._exec.submit(Order(
            client_id=cid, instrument_id=bar.instrument_id, account_id=self._exec.portfolio.account_id,
            strategy_id=self._strategy_id, side=Side.SELL if pos.side is PositionSide.LONG else Side.BUY,
            type=OrderType.MARKET, qty=pos.qty, reduce_only=True, ts_created=self._clock.now_ms(),
            tags={"tag": tag, "reason": tag or "CLOSE"}))

    def log(self, msg: str) -> None:
        self._logger.info("%s", msg)
        if self._log_sink is not None:
            self._log_sink(str(msg))

    def _submit(self, side, typ, qty, size_frac, tag, *, price=None, trigger_price=None, tif="GTC", tp=None, sl=None,
                tp_is_offset=False, max_hold=None) -> str:
        bar = self._need_bar()
        if (qty is None) == (size_frac is None):
            raise ValueError("exactly one of qty and size_frac must be given")
        bracket = None
        if tp is not None or sl is not None or max_hold is not None:
            bracket = {"take_profit": tp, "stop_loss": sl, "tp_is_offset": bool(tp_is_offset),
                       "max_hold": None if max_hold is None else int(max_hold)}
        self._seq += 1
        cid = f"{self._strategy_id}:{self.session}:{self._seq}"
        order = Order(client_id=cid, instrument_id=bar.instrument_id, account_id=self._exec.portfolio.account_id,
                      strategy_id=self._strategy_id, side=Side(str(getattr(side, "value", side)).upper()), type=typ,
                      qty=qty, size_frac=size_frac, price=price, trigger_price=trigger_price,
                      tif=TimeInForce(tif), bracket=bracket, ts_created=self._clock.now_ms(),
                      tags={"tag": tag, "reason": tag})
        self._mine.add(cid)
        self._exec.submit(order)
        return cid


class DeskStrategies(BaseRegistry):
    """Registry of desk strategies (outside the nine component registries and the project's ``Strategies``).
    The six built-ins register when ``neural_trade.desk.strategies`` is imported; any lookup imports it first."""

    registry = {}
    strict = True
    default = None
    discovery_modules = ("neural_trade.desk.strategies",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        return (inspect.isclass(component) and issubclass(component, DeskStrategy)
                and isinstance(getattr(component, "name", None), str) and bool(component.name))

    @classmethod
    def _ensure_builtins(cls) -> None:
        if not cls._initialized:
            cls.auto_discover()

    @classmethod
    def entry(cls, name: str):
        cls._ensure_builtins()
        return super().entry(name)

    @classmethod
    def list(cls):
        cls._ensure_builtins()
        return super().list()

    @classmethod
    def list_names(cls):
        cls._ensure_builtins()
        return super().list_names()

    @classmethod
    def schemas(cls) -> dict[str, list[dict]]:
        """``{name: params_schema}`` of every registered strategy, for the UI forms and sweep spaces."""
        return {n: cls.registry[n].component.params_schema() for n in cls.list_names()}


__all__ = ["BAR_COLUMNS", "DeskStrategies", "DeskStrategy", "IndicatorsUnavailable", "StrategyContext"]

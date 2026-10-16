"""Domain model (PLAN A4): instruments, bars, orders, fills, positions, accounts, events, commands.

Pure data and the order status machine. Transitions are applied only by ``ExecutionEngine.apply`` (DESK-1.3);
this module owns the legal edge list and ``check_transition``. ``Instrument``, ``Order`` and ``Position`` are
keyword-only (their A4 field lists mix required and defaulted fields).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Optional


class DeskError(Exception):
    pass


class IllegalTransition(DeskError):
    pass


class OverfillError(DeskError):
    pass


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    GTC = "GTC"
    DAY = "DAY"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    DENIED = "DENIED"
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"
    TRIGGERED = "TRIGGERED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"


class Liquidity(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class AccountKind(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


_S = OrderStatus
# The complete A4 edge list; anything else raises IllegalTransition.
STATUS_EDGES: frozenset[tuple[OrderStatus, OrderStatus]] = frozenset({
    (_S.INITIALIZED, _S.DENIED), (_S.INITIALIZED, _S.SUBMITTED),
    (_S.SUBMITTED, _S.REJECTED), (_S.SUBMITTED, _S.ACCEPTED),
    (_S.ACCEPTED, _S.TRIGGERED), (_S.ACCEPTED, _S.PARTIALLY_FILLED), (_S.ACCEPTED, _S.FILLED),
    (_S.ACCEPTED, _S.CANCELED), (_S.ACCEPTED, _S.EXPIRED),
    (_S.TRIGGERED, _S.PARTIALLY_FILLED), (_S.TRIGGERED, _S.FILLED),
    (_S.TRIGGERED, _S.CANCELED), (_S.TRIGGERED, _S.EXPIRED),
    (_S.PARTIALLY_FILLED, _S.PARTIALLY_FILLED), (_S.PARTIALLY_FILLED, _S.FILLED),
    (_S.PARTIALLY_FILLED, _S.CANCELED), (_S.PARTIALLY_FILLED, _S.EXPIRED),
})
# The one exception edge (journal kind ``late_fill``): the venue filled before the cancel landed.
LATE_FILL_EDGES: frozenset[tuple[OrderStatus, OrderStatus]] = frozenset({
    (_S.CANCELED, _S.PARTIALLY_FILLED), (_S.CANCELED, _S.FILLED),
})
CLOSED_STATUSES: frozenset[OrderStatus] = frozenset({_S.FILLED, _S.CANCELED, _S.EXPIRED, _S.REJECTED, _S.DENIED})


def check_transition(old: OrderStatus, new: OrderStatus, *, late_fill: bool = False) -> None:
    """Raise IllegalTransition unless old->new is a listed edge (or a late_fill edge when ``late_fill``)."""
    old, new = OrderStatus(old), OrderStatus(new)
    if (old, new) in STATUS_EDGES or (late_fill and (old, new) in LATE_FILL_EDGES):
        return
    raise IllegalTransition(f"illegal order transition {old.value} -> {new.value}" + (" (late_fill)" if late_fill else ""))


def _plain(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


class _Json:
    """to_json/from_json for flat dataclasses; enum fields are named in ``_ENUMS``."""

    _ENUMS: ClassVar[dict[str, type[Enum]]] = {}

    def to_dict(self) -> dict:
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict):
        kw = dict(d)
        for name, enum in cls._ENUMS.items():
            if kw.get(name) is not None:
                kw[name] = enum(kw[name])
        return cls(**kw)

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True, kw_only=True)
class Instrument(_Json):
    id: str = "BINANCE:BTCUSDT"
    symbol: str = "BTC/USDT"
    base: str
    quote: str
    price_increment: float
    size_increment: float
    min_qty: float
    min_notional: float
    max_qty: Optional[float] = None
    is_spot: bool = True


@dataclass(frozen=True)
class Bar(_Json):
    instrument_id: str
    tf_minutes: int
    ts_open_ms: int  # Binance open time, UTC ms; the bar closes at ts_open_ms + tf*60_000
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool

    @property
    def ts_close_ms(self) -> int:
        return self.ts_open_ms + self.tf_minutes * 60_000


@dataclass(kw_only=True)
class Order(_Json):
    client_id: str
    instrument_id: str
    account_id: str
    strategy_id: Optional[str] = None
    bot_id: Optional[str] = None
    side: Side
    type: OrderType
    qty: Optional[float] = None
    size_frac: Optional[float] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    bracket: Optional[dict] = None  # {take_profit, stop_loss, tp_is_offset: bool, max_hold: int|None}
    status: OrderStatus = OrderStatus.INITIALIZED
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    reason: str = ""
    venue_id: Optional[str] = None
    ts_created: int = 0
    ts_updated: int = 0
    tags: dict = field(default_factory=dict)

    _ENUMS: ClassVar[dict[str, type[Enum]]] = {
        "side": Side, "type": OrderType, "tif": TimeInForce, "status": OrderStatus,
    }

    def __post_init__(self) -> None:
        self.side, self.type = Side(self.side), OrderType(self.type)
        self.tif, self.status = TimeInForce(self.tif), OrderStatus(self.status)
        if (self.qty is None) == (self.size_frac is None):
            raise ValueError("exactly one of qty and size_frac must be set")

    @property
    def is_closed(self) -> bool:
        return self.status in CLOSED_STATUSES


@dataclass(frozen=True)
class Fill(_Json):
    trade_id: str
    client_id: str
    instrument_id: str
    side: Side
    qty: float
    price: float  # the slipped fill price
    mid: float  # reference price before spread and slippage (== price for venue fills)
    fee: float
    fee_currency: str
    liquidity: Liquidity
    ts: int

    _ENUMS: ClassVar[dict[str, type[Enum]]] = {"side": Side, "liquidity": Liquidity}


@dataclass(kw_only=True)
class Position(_Json):
    position_id: str = ""  # default '{instrument_id}-{strategy_id}'
    instrument_id: str
    account_id: str
    strategy_id: Optional[str] = None
    side: PositionSide = PositionSide.FLAT
    qty: float = 0.0
    avg_price: float = 0.0
    realised_pnl: float = 0.0
    fees: float = 0.0
    ts_opened: int = 0
    ts_closed: Optional[int] = None

    _ENUMS: ClassVar[dict[str, type[Enum]]] = {"side": PositionSide}

    def __post_init__(self) -> None:
        self.side = PositionSide(self.side)
        if not self.position_id:
            self.position_id = f"{self.instrument_id}-{self.strategy_id}"


@dataclass
class AccountState(_Json):
    account_id: str
    kind: AccountKind
    quote_currency: str
    cash: float
    equity: float
    locked: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    ts: int = 0

    _ENUMS: ClassVar[dict[str, type[Enum]]] = {"kind": AccountKind}

    def __post_init__(self) -> None:
        self.kind = AccountKind(self.kind)


# Events: every one carries ts_event (when it happened) and ts_init (Clock.now_ms() of the engine's clock).
@dataclass(frozen=True)
class _Event:
    ts_event: int
    ts_init: int


@dataclass(frozen=True)
class BarClosed(_Event):
    bar: Bar


@dataclass(frozen=True)
class BarForming(_Event):
    bar: Bar


@dataclass(frozen=True)
class OrderEvent(_Event):
    kind: str
    order: Order


@dataclass(frozen=True)
class FillEvent(_Event):
    fill: Fill


@dataclass(frozen=True)
class PositionEvent(_Event):
    position: Position


@dataclass(frozen=True)
class AccountEvent(_Event):
    account: AccountState


@dataclass(frozen=True)
class BotEvent(_Event):
    bot_id: str
    kind: str  # 'status' | 'log'
    message: str = ""


@dataclass(frozen=True)
class AlertFired(_Event):
    alert_id: str
    message: str = ""


@dataclass(frozen=True)
class JobEvent(_Event):
    job_id: str
    state: str


@dataclass(frozen=True)
class TimeEvent(_Event):
    name: str = ""


# Commands.
@dataclass(frozen=True)
class SubmitOrder:
    order: Order


@dataclass(frozen=True)
class ModifyOrder:
    client_id: str
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    qty: Optional[float] = None


@dataclass(frozen=True)
class CancelOrder:
    client_id: str


@dataclass(frozen=True)
class CancelAll:
    account_id: str
    instrument_id: Optional[str] = None


@dataclass(frozen=True)
class ClosePosition:
    position_id: str
    tag: str = ""


@dataclass(frozen=True)
class Halt:
    reason: str

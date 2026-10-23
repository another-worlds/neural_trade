"""DESK-0.4: the A4 status machine (exhaustive), JSON round trips, SimClock."""
from __future__ import annotations

import dataclasses
import itertools

import pytest

from neural_trade.desk.core.clock import LiveClock, SimClock
from neural_trade.desk.core.model import (
    STATUS_EDGES, AccountKind, AccountState, Bar, Fill, IllegalTransition, Instrument, Liquidity,
    Order, OrderEvent, OrderStatus, OrderType, Position, PositionSide, Side, TimeInForce, check_transition,
)

S = OrderStatus
ALL_PAIRS = list(itertools.product(S, S))

# Literal copy of the 17 edges of docs/desk/CONTRACTS.md A4 (not derived from the implementation).
LITERAL_EDGES = {
    ("INITIALIZED", "DENIED"), ("INITIALIZED", "SUBMITTED"), ("SUBMITTED", "REJECTED"), ("SUBMITTED", "ACCEPTED"),
    ("ACCEPTED", "TRIGGERED"), ("ACCEPTED", "PARTIALLY_FILLED"), ("ACCEPTED", "FILLED"), ("ACCEPTED", "CANCELED"),
    ("ACCEPTED", "EXPIRED"), ("TRIGGERED", "PARTIALLY_FILLED"), ("TRIGGERED", "FILLED"), ("TRIGGERED", "CANCELED"),
    ("TRIGGERED", "EXPIRED"), ("PARTIALLY_FILLED", "PARTIALLY_FILLED"), ("PARTIALLY_FILLED", "FILLED"),
    ("PARTIALLY_FILLED", "CANCELED"), ("PARTIALLY_FILLED", "EXPIRED"),
}


def test_edge_list_equals_the_literal_a4_list():
    assert len(LITERAL_EDGES) == 17
    assert {(a.value, b.value) for a, b in STATUS_EDGES} == LITERAL_EDGES


@pytest.mark.parametrize(("old", "new"), ALL_PAIRS)
def test_every_pair_exhaustively(old, new):
    if (old.value, new.value) in LITERAL_EDGES:
        check_transition(old, new)
    else:
        with pytest.raises(IllegalTransition):
            check_transition(old, new)


@pytest.mark.parametrize(("old", "new"), ALL_PAIRS)
def test_late_fill_flag_adds_only_the_two_canceled_edges(old, new):
    legal = (old.value, new.value) in LITERAL_EDGES or (old.value, new.value) in {("CANCELED", "PARTIALLY_FILLED"), ("CANCELED", "FILLED")}
    if legal:
        check_transition(old, new, late_fill=True)
    else:
        with pytest.raises(IllegalTransition):
            check_transition(old, new, late_fill=True)


def test_late_fill_edge_needs_the_flag():
    with pytest.raises(IllegalTransition):
        check_transition(S.CANCELED, S.FILLED)


def test_is_closed():
    closed = {S.FILLED, S.CANCELED, S.EXPIRED, S.REJECTED, S.DENIED}
    for st in S:
        o = Order(client_id="c", instrument_id="i", account_id="a", side=Side.BUY, type=OrderType.MARKET, qty=1.0, status=st)
        assert o.is_closed == (st in closed)


def test_order_needs_exactly_one_of_qty_and_size_frac():
    kw = dict(client_id="c", instrument_id="i", account_id="a", side="BUY", type="MARKET")
    with pytest.raises(ValueError):
        Order(**kw)
    with pytest.raises(ValueError):
        Order(**kw, qty=1.0, size_frac=0.5)


def _full_objects():
    order = Order(client_id="c1", instrument_id="BINANCE:ETHUSDT", account_id="acc", strategy_id="s", bot_id="b7",
                  side=Side.SELL, type=OrderType.STOP_LIMIT, qty=1.5, size_frac=None, price=100.5, trigger_price=99.5,
                  tif=TimeInForce.FOK, reduce_only=True,
                  bracket={"take_profit": 2.0, "stop_loss": 1.0, "tp_is_offset": True, "max_hold": 30},
                  status=S.PARTIALLY_FILLED, filled_qty=0.4, avg_fill_price=100.1, reason="TP", venue_id="v9",
                  ts_created=5, ts_updated=6, tags={"k": [1, 2]})
    fill = Fill("t1", "c1", "i", Side.SELL, 0.5, 101.0, 100.9, 0.01, "BNB", Liquidity.MAKER, 99)
    pos = Position(position_id="pid", instrument_id="i", account_id="a", strategy_id="s", side=PositionSide.SHORT,
                   qty=2.0, avg_price=3.0, realised_pnl=4.5, fees=0.7, ts_opened=11, ts_closed=12)
    acc = AccountState("a", AccountKind.LIVE, "EUR", 1000.0, 1010.0, locked=3.0, realised_pnl=4.0, unrealised_pnl=5.0, ts=7)
    ins = Instrument(id="X:Y", symbol="Y/X", base="Y", quote="X", price_increment=0.5, size_increment=0.25, min_qty=0.1,
                     min_notional=7.0, max_qty=9.0, is_spot=False)
    bar = Bar("i", 5, 60_000, 1.0, 2.0, 0.5, 1.5, 10.0, True)
    return [order, fill, pos, acc, ins, bar]


@pytest.mark.parametrize("obj", _full_objects(), ids=lambda o: type(o).__name__)
def test_roundtrip_with_every_field_non_default(obj):
    d = obj.to_dict()
    assert set(d) == {f.name for f in dataclasses.fields(obj)}
    assert type(obj).from_dict(d) == obj
    assert type(obj).from_json(obj.to_json()) == obj
    # enum-typed fields come back as enums, not strings
    for name, enum in type(obj)._ENUMS.items():
        assert isinstance(getattr(type(obj).from_json(obj.to_json()), name), enum)
    # no field equals its dataclass default, so a dropped field cannot hide behind a default
    for f in dataclasses.fields(obj):
        if f.default is not dataclasses.MISSING and f.name != "size_frac":  # qty and size_frac are exclusive
            assert getattr(obj, f.name) != f.default, f.name


def test_order_with_size_frac_roundtrips():
    o = Order(client_id="c", instrument_id="i", account_id="a", side="BUY", type="MARKET", size_frac=0.25)
    assert Order.from_json(o.to_json()) == o and o.qty is None


def test_fill_coerces_plain_strings():
    f = Fill("t", "c", "i", "BUY", 1.0, 1.0, 1.0, 0.0, "USDT", "TAKER", 1)
    assert f.side is Side.BUY and f.liquidity is Liquidity.TAKER


def test_position_default_id():
    assert Position(instrument_id="i", account_id="a", strategy_id="s").position_id == "i-s"


def test_instrument_and_bar():
    ins = Instrument(base="BTC", quote="USDT", price_increment=0.01, size_increment=1e-5, min_qty=1e-5, min_notional=5.0)
    assert Instrument.from_json(ins.to_json()) == ins and ins.id == "BINANCE:BTCUSDT" and ins.is_spot
    bar = Bar("i", 5, 60_000, 1.0, 2.0, 0.5, 1.5, 10.0, True)
    assert bar.ts_close_ms == 60_000 + 300_000 and Bar.from_json(bar.to_json()) == bar


def test_simclock_monotonic():
    c = SimClock()
    c.set(10)
    c.set(10)  # equal is fine
    c.set(20)
    assert c.now_ms() == 20
    with pytest.raises(ValueError):
        c.set(19)
    assert c.now_ms() == 20


def test_liveclock_is_epoch_ms():
    assert LiveClock().now_ms() > 1_700_000_000_000


def test_order_event_resolved_qty_is_optional_and_additive():
    o = Order(client_id="c", instrument_id="X", account_id="a", side="BUY", type="MARKET", size_frac=0.5)
    assert OrderEvent(1, 2, "accepted", o).resolved_qty is None
    ev = OrderEvent(1, 2, "accepted", o, resolved_qty=4.0)
    assert (ev.ts_event, ev.ts_init, ev.kind, ev.order, ev.resolved_qty) == (1, 2, "accepted", o, 4.0)

"""DESK-0.4: the A4 status machine (exhaustive), JSON round trips, SimClock."""
from __future__ import annotations

import itertools

import pytest

from neural_trade.desk.core.clock import LiveClock, SimClock
from neural_trade.desk.core.model import (
    LATE_FILL_EDGES, STATUS_EDGES, AccountKind, AccountState, Bar, Fill, IllegalTransition, Instrument, Liquidity,
    Order, OrderStatus, OrderType, Position, PositionSide, Side, check_transition,
)

S = OrderStatus
ALL_PAIRS = list(itertools.product(S, S))


def test_edge_list_has_the_17_a4_edges():
    assert len(STATUS_EDGES) == 17


@pytest.mark.parametrize(("old", "new"), ALL_PAIRS)
def test_every_pair_exhaustively(old, new):
    if (old, new) in STATUS_EDGES:
        check_transition(old, new)
    else:
        with pytest.raises(IllegalTransition):
            check_transition(old, new)


@pytest.mark.parametrize(("old", "new"), ALL_PAIRS)
def test_late_fill_flag_adds_only_the_two_canceled_edges(old, new):
    legal = (old, new) in STATUS_EDGES or (old, new) in LATE_FILL_EDGES
    if legal:
        check_transition(old, new, late_fill=True)
    else:
        with pytest.raises(IllegalTransition):
            check_transition(old, new, late_fill=True)
    assert LATE_FILL_EDGES == {(S.CANCELED, S.PARTIALLY_FILLED), (S.CANCELED, S.FILLED)}


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


def test_roundtrips():
    order = Order(client_id="c1", instrument_id="BINANCE:BTCUSDT", account_id="paper", strategy_id="s", side=Side.SELL,
                  type=OrderType.LIMIT, size_frac=0.5, price=100.5, tif="IOC", status=S.ACCEPTED, filled_qty=0.1,
                  bracket={"take_profit": 2.0, "stop_loss": 1.0, "tp_is_offset": True, "max_hold": None},
                  ts_created=5, ts_updated=6, tags={"k": [1, 2]})
    fill = Fill("t1", "c1", "i", Side.BUY, 0.5, 101.0, 100.9, 0.01, "USDT", Liquidity.TAKER, 99)
    pos = Position(instrument_id="i", account_id="a", strategy_id="s", side=PositionSide.LONG, qty=2.0, avg_price=3.0, ts_closed=None)
    acc = AccountState("a", AccountKind.PAPER, "USDT", 1000.0, 1010.0, ts=7)
    for obj in (order, fill, pos, acc):
        back = type(obj).from_json(obj.to_json())
        assert back == obj
    assert pos.position_id == "i-s"
    assert Order.from_json(order.to_json()).status is S.ACCEPTED


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

"""NT-222 (DESK-1.1): the Portfolio pinned by hand-computed numbers, and against strategy/backtest.py (CONTRACTS A5)."""
from __future__ import annotations

import copy
import dataclasses
import itertools

import numpy as np
import pytest

from neural_trade.desk.core.costs import RESEARCH_ZERO, CostProfile
from neural_trade.desk.core.fills import market_fill
from neural_trade.desk.core.model import OverfillError, PositionEvent, PositionSide
from neural_trade.desk.core.portfolio import Portfolio
from neural_trade.strategy.backtest import BacktestConfig, Bars, run_backtest
from neural_trade.strategy.strategies import Strategy
from neural_trade.strategy.trades import Order

X = "X"
C10 = CostProfile(name="c", fee_bps_taker=10.0, fee_bps_maker=10.0, half_spread_bps=1.0, slippage_bps=2.0)
_ids = itertools.count()


def fill(side, qty, mid, cost=RESEARCH_ZERO, ts=0, tid=None):
    f = market_fill(mid, side, cost, qty)
    return dataclasses.replace(f, instrument_id=X, ts=ts, trade_id=f"t{next(_ids)}" if tid is None else tid)


def pf(cash=10_000.0, **kw):
    return Portfolio("acc", initial_cash=cash, **kw)


# ---------------------------------------------------------------- long, zero cost
def test_long_open_add_reduce_close():
    p = pf()
    (ev,) = p.apply_fill(fill(1, 10, 100.0, ts=1))
    pos = ev.position
    assert (pos.side, pos.qty, pos.avg_price, pos.ts_opened, pos.position_id) == (
        PositionSide.LONG, 10, 100.0, 1, "X-None")
    assert p.cash == 9000.0
    assert p.mark({X: 105.0}, 2).equity == 10_050.0
    assert p.unrealised_pnl({X: 105.0}) == 50.0

    p.apply_fill(fill(1, 10, 110.0, ts=3))  # add: avg (1000 + 1100) / 20 = 105
    pos = p.position(X)
    assert (pos.qty, pos.avg_price, pos.realised_pnl, p.cash) == (20, 105.0, 0.0, 7900.0)

    (ev,) = p.apply_fill(fill(-1, 5, 120.0, ts=4))  # reduce 5: 5 x (120 - 105) = 75
    assert (ev.position.qty, ev.position.avg_price, ev.position.realised_pnl) == (15, 105.0, 75.0)
    assert ev.position.side is PositionSide.LONG and ev.position.ts_closed is None
    assert p.cash == 8500.0
    st = p.mark({X: 100.0}, 5)  # unrealised 15 x (100 - 105) = -75
    assert (st.cash, st.realised_pnl, st.unrealised_pnl, st.equity) == (8500.0, 75.0, -75.0, 10_000.0)

    (ev,) = p.apply_fill(fill(-1, 15, 100.0, ts=6))  # close: 75 - 75 = 0
    pos = ev.position
    assert (pos.side, pos.qty, pos.realised_pnl, pos.ts_closed, p.cash) == (PositionSide.FLAT, 0, 0.0, 6, 10_000.0)
    assert [t.gross_pnl for t in p.trades] == [75.0, -75.0]
    assert p.mark({X: 100.0}, 7).equity == 10_000.0


def test_short_open_reduce_close_and_sign_of_pnl():
    p = pf()
    p.apply_fill(fill(-1, 10, 100.0))
    pos = p.position(X)
    assert (pos.side, pos.qty, pos.avg_price, p.cash) == (PositionSide.SHORT, 10, 100.0, 11_000.0)
    st = p.mark({X: 90.0}, 1)  # a falling price pays a short: +100
    assert (st.unrealised_pnl, st.equity) == (100.0, 10_100.0)
    p.apply_fill(fill(1, 4, 95.0))  # 4 x (100 - 95) = +20
    pos = p.position(X)
    assert (pos.qty, pos.realised_pnl, p.cash) == (6, 20.0, 10_620.0)
    p.apply_fill(fill(1, 6, 110.0))  # 6 x (100 - 110) = -60 -> -40
    pos = p.position(X)
    assert (pos.side, pos.realised_pnl, p.cash) == (PositionSide.FLAT, -40.0, 9_960.0)
    assert p.mark({X: 110.0}, 2).equity == 9_960.0
    assert [t.gross_pnl for t in p.trades] == [20.0, -60.0]


def test_flip_long_to_short_emits_close_then_open():
    p = pf()
    p.apply_fill(fill(1, 10, 100.0, ts=1))
    evs = p.apply_fill(fill(-1, 15, 110.0, ts=2))  # closes 10 (+100), opens short 5 @ 110
    assert len(evs) == 2 and all(isinstance(e, PositionEvent) for e in evs)
    closed, opened = (e.position for e in evs)
    assert (closed.side, closed.qty, closed.realised_pnl, closed.ts_closed) == (PositionSide.FLAT, 0, 100.0, 2)
    assert (opened.side, opened.qty, opened.avg_price, opened.realised_pnl, opened.ts_opened, opened.ts_closed) == (
        PositionSide.SHORT, 5, 110.0, 0.0, 2, None)
    assert p.cash == 10_650.0
    st = p.mark({X: 108.0}, 3)  # short 5 @ 110 marked 108: +10
    assert (st.unrealised_pnl, st.realised_pnl, st.equity) == (10.0, 100.0, 10_110.0)


def test_flip_short_to_long():
    p = pf()
    p.apply_fill(fill(-1, 10, 100.0))
    evs = p.apply_fill(fill(1, 12, 90.0))  # closes 10 (+100), opens long 2 @ 90
    assert [e.position.side for e in evs] == [PositionSide.FLAT, PositionSide.LONG]
    assert evs[0].position.realised_pnl == 100.0 and (evs[1].position.qty, evs[1].position.avg_price) == (2, 90.0)
    assert p.cash == 11_000.0 - 1080.0


# ---------------------------------------------------------------- costs
def test_long_round_trip_costs_at_10_1_2_bps():
    p = pf(20_000.0)
    p.apply_fill(fill(1, 100, 100.0, C10))  # price 100.03, fee 10.0 -> cash 20000 - 10003 - 10
    assert p.cash == pytest.approx(9987.0, abs=1e-9)
    assert p.position(X).fees == pytest.approx(10.0)
    p.apply_fill(fill(-1, 100, 102.0, C10))  # price 101.9694, fee 10.2 -> +10196.94 - 10.2
    assert p.cash == pytest.approx(20_173.74, abs=1e-9)
    (t,) = p.trades
    assert (t.gross_pnl, t.costs) == (pytest.approx(200.0), pytest.approx(26.26, abs=1e-9))
    assert t.net_pnl == pytest.approx(173.74, abs=1e-9)
    assert p.fees_paid == pytest.approx(20.2)
    assert p.position(X).fees == pytest.approx(20.2) and p.position(X).realised_pnl == pytest.approx(200.0)
    st = p.mark({X: 102.0}, 1)
    assert st.realised_pnl == pytest.approx(173.74, abs=1e-9) and st.equity == pytest.approx(20_173.74, abs=1e-9)


def test_flip_splits_costs_between_the_close_and_the_open():
    p = pf(100_000.0)
    p.apply_fill(fill(1, 10, 100.0, C10))  # entry costs: slip 3.0 x 0.1 = 0.3, fee 1.0 -> 1.3
    p.apply_fill(fill(-1, 20, 100.0, C10))  # slip 0.6 + fee 2.0 = 2.6; half (1.3) belongs to the close
    (t,) = p.trades
    assert (t.qty, t.gross_pnl, t.costs) == (10, 0.0, pytest.approx(1.3 + 1.3))
    pos = p.position(X)
    assert (pos.side, pos.qty, pos.fees) == (PositionSide.SHORT, 10, pytest.approx(1.0))
    p.apply_fill(fill(1, 10, 100.0, C10))  # close the short: its entry 1.3 + this 1.3
    assert p.trades[1].costs == pytest.approx(2.6)
    assert p.mark({X: 100.0}, 1).equity == pytest.approx(100_000.0 - 0.3 - 1.0 - 0.6 - 2.0 - 0.3 - 1.0)


def test_partial_reduce_takes_a_share_of_the_entry_costs():
    p = pf(100_000.0)
    p.apply_fill(fill(1, 10, 100.0, C10))  # entry costs 1.3
    p.apply_fill(fill(-1, 4, 100.0, C10))  # 0.4 of the entry (0.52) + this fill 0.52
    assert p.trades[0].costs == pytest.approx(0.52 + 0.52)
    p.apply_fill(fill(-1, 6, 100.0, C10))
    assert p.trades[1].costs == pytest.approx(0.78 + 0.78)


# ---------------------------------------------------------------- drawdown, identity, equity
def test_drawdown_and_equity_curve():
    p = pf()
    p.apply_fill(fill(1, 10, 100.0))
    for ts, px in enumerate([110.0, 120.0, 90.0, 100.0, 125.0]):
        p.mark({X: px}, ts)
    assert [e for _, e in p.equity_curve] == [10_100.0, 10_200.0, 9_900.0, 10_000.0, 10_250.0]
    assert p.peak_equity == 10_250.0 and p.drawdown == 0.0
    assert p.max_drawdown == pytest.approx(300.0 / 10_200.0)  # 10200 -> 9900


def test_equity_is_initial_plus_realised_plus_unrealised():
    p = pf(5_000.0)
    p.apply_fill(fill(1, 7, 31.0, C10))
    p.apply_fill(fill(-1, 3, 33.5, C10))
    p.apply_fill(fill(-1, 9, 30.0, C10))  # flips: closes 4, opens short 5
    st = p.mark({X: 28.0}, 1)
    assert st.equity == pytest.approx(5_000.0 + st.realised_pnl + st.unrealised_pnl, abs=1e-9)
    assert st.unrealised_pnl == pytest.approx(5 * (30.0 - 28.0))


def test_event_ts_init_comes_from_the_clock():
    from neural_trade.desk.core.clock import SimClock

    p = Portfolio("a", clock=SimClock(777))
    (ev,) = p.apply_fill(fill(1, 1, 10.0, ts=5))
    assert (ev.ts_event, ev.ts_init) == (5, 777)


# ---------------------------------------------------------------- edge cases
@pytest.mark.parametrize("qty", [0.0, -1.0])
def test_non_positive_qty_raises(qty):
    with pytest.raises(ValueError):
        pf().apply_fill(fill(1, qty, 100.0))


def test_repeated_trade_id_is_ignored():
    p = pf()
    f = fill(1, 10, 100.0, tid="same")
    assert len(p.apply_fill(f)) == 1
    assert p.apply_fill(f) == []
    assert (p.position(X).qty, p.cash) == (10, 9000.0)


def test_spot_sell_beyond_the_balance_raises_overfill_and_changes_nothing():
    p = pf(allow_short=False)
    p.apply_fill(fill(1, 10, 100.0))
    with pytest.raises(OverfillError):
        p.apply_fill(fill(-1, 10.5, 100.0))
    assert (p.position(X).qty, p.cash) == (10, 9000.0)
    p.apply_fill(fill(-1, 10, 100.0))  # selling exactly the balance is fine
    assert p.position(X).side is PositionSide.FLAT
    with pytest.raises(OverfillError):  # nothing held at all
        p.apply_fill(fill(-1, 0.1, 100.0))


def test_positions_net_per_strategy_and_spot_balance_spans_strategies():
    p = pf(allow_short=False)
    p.apply_fill(fill(1, 3, 100.0), strategy_id="a")
    p.apply_fill(fill(1, 2, 100.0), strategy_id="b")
    assert (p.position(X, "a").qty, p.position(X, "b").qty, p.base_balance(X)) == (3, 2, 5)
    with pytest.raises(OverfillError):
        p.apply_fill(fill(-1, 6, 100.0), strategy_id="a")
    assert p.apply_fill(fill(-1, 5, 100.0), strategy_id="a")[0].position.position_id == "X-a"


def test_reopen_after_flat_starts_a_fresh_lifecycle():
    p = pf()
    p.apply_fill(fill(1, 10, 100.0, ts=1))
    p.apply_fill(fill(-1, 10, 110.0, ts=2))
    (ev,) = p.apply_fill(fill(1, 4, 120.0, ts=3))
    pos = ev.position
    assert (pos.qty, pos.avg_price, pos.realised_pnl, pos.fees, pos.ts_opened, pos.ts_closed) == (
        4, 120.0, 0.0, 0.0, 3, None)


def test_invalid_initial_cash():
    with pytest.raises(ValueError):
        Portfolio("a", initial_cash=0)


# ---------------------------------------------------------------- against the oracle
class _Sig:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _OneTrade(Strategy):
    def __init__(self, side):
        self.side = side

    def decide(self, s, t):
        return Order(side=self.side, size_frac=1.0, max_hold=1, reason="t") if t == 0 else None


class _Alternate(Strategy):
    """Enters whenever flat, alternating sides, half the equity, held 2 bars."""

    def __init__(self):
        self.k = 0

    def decide(self, s, t):
        self.k += 1
        return Order(side="LONG" if self.k % 2 else "SHORT", size_frac=0.5, max_hold=2, reason="a")


def _replay(res, bars, cost, init):
    """Drive a Portfolio from the oracle's trade list (quantity = notional / entry open; exits at the open, EOW at
    the close) and mark every bar's close; fills are built from the bars, not from the oracle's equity."""
    p = Portfolio("acc", initial_cash=init)
    by_bar = {}
    for tr in res.trades:
        sign = 1 if tr.side == "LONG" else -1
        qty = tr.notional / bars.open[tr.entry_bar]
        exit_mid = bars.close[tr.exit_bar] if tr.exit_reason == "EOW" else bars.open[tr.exit_bar]
        by_bar.setdefault(tr.exit_bar, []).append((0, -sign, qty, exit_mid))
        by_bar.setdefault(tr.entry_bar, []).append((1, sign, qty, bars.open[tr.entry_bar]))
    out = [init]
    for t in range(len(bars)):
        for _kind, side, qty, mid in sorted(by_bar.get(t, [])):
            p.apply_fill(fill(side, qty, mid, cost, ts=t))
        out.append(p.mark({X: float(bars.close[t])}, t).equity)
    return p, np.array(out)


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
@pytest.mark.parametrize("bps", [(0, 0, 0), (10, 1, 2)])
def test_three_bar_example_equals_run_backtest(side, bps):
    fee, hs, sl = bps
    o = np.array([99.0, 100.0, 102.0])
    c = np.array([99.5, 101.0, 103.0])
    bars = Bars(open=o, high=o + 1, low=o - 1, close=c)
    res = run_backtest(_Sig(3), bars, _OneTrade(side), BacktestConfig(fee_bps=fee, half_spread_bps=hs, slippage_bps=sl))
    cost = CostProfile(fee_bps_taker=fee, fee_bps_maker=fee, half_spread_bps=hs, slippage_bps=sl)
    p, eq = _replay(res, bars, cost, 10_000.0)
    assert np.allclose(eq, res.equity, rtol=0, atol=1e-9)
    (tr,), (mine,) = res.trades, p.trades
    assert mine.gross_pnl == pytest.approx(tr.gross_pnl, abs=1e-9)
    assert mine.costs == pytest.approx(tr.costs, abs=1e-9)
    assert mine.net_pnl == pytest.approx(tr.net_pnl, abs=1e-9)
    assert mine.side.value == side


@pytest.mark.parametrize("bps", [(0, 0, 0), (10, 1, 2)])
def test_scripted_20_bar_equity_equals_run_backtest(bps):
    fee, hs, sl = bps
    rng = np.random.default_rng(7)
    o = 100.0 + np.cumsum(rng.normal(0, 1.0, 20))
    c = o + rng.normal(0, 0.5, 20)
    bars = Bars(open=o, high=np.maximum(o, c) + 0.3, low=np.minimum(o, c) - 0.3, close=c)
    res = run_backtest(_Sig(20), bars, _Alternate(), BacktestConfig(fee_bps=fee, half_spread_bps=hs,
                                                                    slippage_bps=sl, max_hold=2))
    assert len(res.trades) >= 5
    cost = CostProfile(fee_bps_taker=fee, fee_bps_maker=fee, half_spread_bps=hs, slippage_bps=sl)
    p, eq = _replay(res, bars, cost, 10_000.0)
    assert np.allclose(eq, res.equity, rtol=0, atol=1e-9)
    assert [t.net_pnl for t in p.trades] == pytest.approx([t.net_pnl for t in res.trades], abs=1e-9)
    assert p.max_drawdown == pytest.approx(float(np.max((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq))))


# ---------------------------------------------------------------- NT-286: completion
def test_account_state_is_read_only():
    p = pf()
    p.apply_fill(fill(1, 10, 100.0, ts=1))
    p.mark({X: 100.0}, 2)
    p.mark({X: 90.0}, 3)  # equity 9900, drawdown 0.01
    before = copy.deepcopy(p.__dict__)
    st = p.account_state({X: 80.0}, 4)  # equity 9800
    assert (st.cash, st.equity, st.unrealised_pnl, st.realised_pnl, st.ts) == (9000.0, 9800.0, -200.0, 0.0, 4)
    assert p.__dict__ == before
    assert (p.drawdown, p.max_drawdown, len(p.equity_curve)) == (0.01, 0.01, 2)
    assert p.mark({X: 80.0}, 4).equity == 9800.0 and p.drawdown == 0.02  # mark still records


def test_restore_long_and_short_close_like_an_uninterrupted_run():
    def run(restart):
        p = pf(cash=100_000.0)
        p.apply_fill(fill(1, 10, 100.0, C10, ts=1))  # long A
        p.apply_fill(dataclasses.replace(fill(-1, 4, 200.0, C10, ts=2), instrument_id="Y"))  # short Y
        if restart:
            q = Portfolio("acc", initial_cash=100_000.0)
            q.restore([dataclasses.replace(v) for v in p.positions.values()], p.cash,
                      entry_costs=dict(p._entry_costs))
            p = q
        p.apply_fill(fill(-1, 10, 110.0, C10, ts=3))
        p.apply_fill(dataclasses.replace(fill(1, 4, 190.0, C10, ts=3), instrument_id="Y"))
        return p
    a, b = run(False), run(True)
    assert b.cash == a.cash and [t.gross_pnl for t in b.trades] == [100.0, 40.0]
    assert [round(t.net_pnl, 9) for t in b.trades] == [round(t.net_pnl, 9) for t in a.trades]
    assert b.trades[0].costs == pytest.approx(10 * 100 * 0.0013 + 10 * 110 * 0.0013)  # entry leg carried over
    assert all(p_.qty == 0 for p_ in b.positions.values())
    with pytest.raises(ValueError):
        b.restore([], 1.0)  # not fresh


def test_restore_validates():
    from neural_trade.desk.core.model import Position
    pos = Position(instrument_id=X, account_id="acc", strategy_id=None, side=PositionSide.LONG, qty=2.0,
                   avg_price=5.0)
    with pytest.raises(ValueError):
        pf().restore([dataclasses.replace(pos, account_id="other")], 1.0)
    with pytest.raises(ValueError):
        pf().restore([pos], float("nan"))
    with pytest.raises(ValueError):
        pf().restore([pos], 1.0, entry_costs={"nope": 1.0})
    p = pf()
    p.restore([pos], 990.0)
    assert (p.cash, p.position(X).qty, p.base_balance(X)) == (990.0, 2.0, 2.0)


def test_bracket_set_get_cleared_on_close_and_flip():
    p = pf()
    with pytest.raises(ValueError):
        p.set_bracket("X-None", {"take_profit": 1.0})  # no open position
    p.apply_fill(fill(1, 10, 100.0))
    br = {"take_profit": 110.0, "stop_loss": 95.0, "tp_is_offset": False, "max_hold": None}
    p.set_bracket("X-None", br)
    br["take_profit"] = 0.0  # stored copy is independent
    assert p.get_bracket("X-None") == {"take_profit": 110.0, "stop_loss": 95.0, "tp_is_offset": False,
                                       "max_hold": None}
    p.apply_fill(fill(-1, 4, 105.0))  # reduce keeps it
    assert p.get_bracket("X-None")["stop_loss"] == 95.0
    p.apply_fill(fill(-1, 6, 105.0))  # close clears
    assert p.get_bracket("X-None") is None
    p.apply_fill(fill(1, 5, 100.0))
    p.set_bracket("X-None", {"take_profit": 120.0})
    p.apply_fill(fill(-1, 8, 100.0))  # flip to short 3: the long's bracket is gone
    assert p.position(X).side is PositionSide.SHORT and p.get_bracket("X-None") is None
    p.set_bracket("X-None", {"stop_loss": 105.0})
    p.set_bracket("X-None", None)
    assert p.get_bracket("X-None") is None
    q = pf()
    q.restore([p.position(X)], 1.0, brackets={"X-None": {"stop_loss": 1.0}})
    assert q.get_bracket("X-None") == {"stop_loss": 1.0}


@pytest.mark.parametrize("field", ["qty", "price", "mid", "fee"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_fill_refused_state_unchanged(field, bad):
    p = pf()
    p.apply_fill(fill(1, 10, 100.0))
    before = copy.deepcopy(p.__dict__)
    with pytest.raises(ValueError):
        p.apply_fill(dataclasses.replace(fill(1, 1, 100.0), **{field: bad}))
    assert p.__dict__ == before


def test_non_finite_mark_and_cash_refused():
    p = pf()
    p.apply_fill(fill(1, 10, 100.0))
    p.mark({X: 100.0}, 1)
    before = copy.deepcopy(p.__dict__)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            p.mark({X: bad}, 2)
        with pytest.raises(ValueError):
            p.account_state({X: bad}, 2)
    assert p.__dict__ == before and p.drawdown == 0.0
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Portfolio("acc", initial_cash=bad)


@pytest.mark.parametrize("held_side", [1, -1])
def test_dust_overshoot_closes_without_flipping(held_side):
    p = pf()
    p.apply_fill(fill(held_side, 0.5, 100.0, ts=1))
    evs = p.apply_fill(fill(-held_side, 0.5 + 1e-14, 100.0, ts=2))
    assert len(evs) == 1 and evs[0].position.side is PositionSide.FLAT
    assert p.position(X).qty == 0.0 and p.base_balance(X) == 0.0
    evs = p.apply_fill(fill(-held_side, 0.5, 100.0, ts=3))  # a real flip beyond dust still flips
    p.apply_fill(fill(held_side, 0.5 + 1e-9, 100.0, ts=4))
    assert p.position(X).side is (PositionSide.LONG if held_side == 1 else PositionSide.SHORT)
    assert p.position(X).qty == pytest.approx(1e-9, abs=1e-15)


def test_realised_equity_is_cash_plus_signed_qty_times_avg_price():
    p = pf()
    assert p.realised_equity() == 10_000.0  # flat: equals cash
    p.apply_fill(fill(1, 25, 200.0, ts=1))  # cash 5000, long 25 at 200
    assert p.cash == 5000.0 and p.realised_equity() == 10_000.0
    assert p.realised_equity() != p.equity({X: 250.0})  # marks do not size orders
    p2 = pf()
    p2.apply_fill(fill(-1, 25, 200.0, ts=1))  # margin short: cash 15000, realised equity 15000 - 25 x 200
    assert p2.cash == 15_000.0 and p2.realised_equity() == 10_000.0
    p3 = pf()
    p3.apply_fill(fill(1, 10, 100.0, cost=C10, ts=1))  # costs paid reduce it: slip 3 bps x 1000 + fee 10 bps x 1000
    assert p3.realised_equity() == pytest.approx(10_000.0 - 0.3 - 1.0)


def test_nt292_gross_exposure_sums_absolute_positions_without_netting_across_strategies():
    p = pf(cash=100_000.0)
    assert p.gross_exposure({X: 100.0}) == 0.0
    p.apply_fill(fill(1, 10, 100.0, ts=1), strategy_id="a")
    p.apply_fill(fill(-1, 5, 100.0, ts=2), strategy_id="b")  # a short on a margin account: |-5| counts
    assert p.gross_exposure({X: 110.0}) == 10 * 110.0 + 5 * 110.0  # 1650: the long and the short do not net
    p.apply_fill(fill(-1, 10, 100.0, ts=3), strategy_id="a")  # closed: no longer counts
    assert p.gross_exposure({X: 110.0}) == 550.0

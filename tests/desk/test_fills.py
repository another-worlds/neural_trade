"""NT-218 (DESK-0.5): cost profiles and the pure fill functions pinned by hand-computed numbers (PLAN A5)."""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.desk.config import DeskSettings
from neural_trade.desk.core.costs import BINANCE_SPOT_TAKER, PRESETS, RESEARCH_ZERO, CostProfile
from neural_trade.desk.core.fills import bracket_exit, limit_fill, market_fill, stop_fill
from neural_trade.desk.core.model import Bar, Liquidity, Side
from neural_trade.strategy.backtest import BacktestConfig, Bars, run_backtest
from neural_trade.strategy.strategies import Strategy
from neural_trade.strategy.trades import Order

ZERO = RESEARCH_ZERO
C10 = CostProfile(name="c", fee_bps_taker=10.0, fee_bps_maker=4.0, half_spread_bps=1.0, slippage_bps=2.0)


def bar(o, h, lo, c, ts=60_000):
    return Bar(instrument_id="X", tf_minutes=1, ts_open_ms=ts, open=o, high=h, low=lo, close=c, volume=1.0,
               is_closed=True)


# ------------------------------------------------------------------ profiles
def test_presets_and_settings_keys_agree():
    assert set(PRESETS) == {"research_zero", "binance_spot_taker"}
    d = DeskSettings().cost_profiles
    for name, p in PRESETS.items():
        assert p.to_dict() == {"name": name, **{k: float(v) for k, v in d[name].items()}}
        assert CostProfile.from_dict(name, d[name]) == p
    assert BINANCE_SPOT_TAKER.fee_rate() == pytest.approx(0.001)
    assert ZERO.slip_rate == 0.0


def test_slip_and_fee_rates_match_backtest_config():
    cfg = BacktestConfig(fee_bps=10, half_spread_bps=1, slippage_bps=2)
    assert C10.slip_rate == cfg.slip_rate == pytest.approx(3e-4)
    assert C10.fee_rate(Liquidity.TAKER) == cfg.fee_rate == pytest.approx(1e-3)
    assert C10.fee_rate(Liquidity.MAKER) == pytest.approx(4e-4)
    with pytest.raises(ValueError):
        CostProfile(slippage_bps=-1)
    with pytest.raises(ValueError):
        CostProfile.from_dict("x", {"fee_bps": 1})


# ------------------------------------------------------------------ market fills
def test_market_fill_literals():
    f = market_fill(100.0, 1, C10, 100.0)
    assert (f.side, f.qty, f.mid, f.liquidity) == (Side.BUY, 100.0, 100.0, Liquidity.TAKER)
    assert f.price == pytest.approx(100.03)          # 100 x (1 + 3e-4)
    assert f.fee == pytest.approx(10.0)              # 1e-3 x 100 x 100, on the mid, not the slipped price
    s = market_fill(102.0, -1, C10, 100.0)
    assert s.side is Side.SELL and s.price == pytest.approx(101.9694) and s.fee == pytest.approx(10.2)
    z = market_fill(100.0, 1, ZERO, 3.0)
    assert (z.price, z.mid, z.fee) == (100.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        market_fill(100.0, 0, ZERO, 1.0)


class _OneTrade(Strategy):
    name = "one_trade"

    def __init__(self, side):
        self.side = side

    def decide(self, s, t):
        return Order(side=self.side, size_frac=1.0, max_hold=1, reason="t") if t == 0 else None


class _Sig:
    def __len__(self):
        return 3


@pytest.mark.parametrize("bps,total_costs", [((0, 0, 0), 0.0), ((10, 1, 2), 26.26)])
@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_market_round_trip_equals_run_backtest(side, bps, total_costs):
    fee, hs, sl = bps
    o = np.array([99.0, 100.0, 102.0])
    bars = Bars(open=o, high=o + 1, low=o - 1, close=o)
    res = run_backtest(_Sig(), bars, _OneTrade(side), BacktestConfig(fee_bps=fee, half_spread_bps=hs, slippage_bps=sl))
    assert len(res.trades) == 1
    tr = res.trades[0]
    sign = 1 if side == "LONG" else -1
    cost = CostProfile(fee_bps_taker=fee, fee_bps_maker=fee, half_spread_bps=hs, slippage_bps=sl)
    qty = 10_000.0 / 100.0
    e = market_fill(100.0, sign, cost, qty)
    x = market_fill(102.0, -sign, cost, qty)
    costs = qty * abs(e.price - e.mid) + e.fee + qty * abs(x.price - x.mid) + x.fee
    gross = sign * qty * (x.mid - e.mid)
    assert tr.entry_price == pytest.approx(e.price, rel=1e-13)
    assert tr.exit_price == pytest.approx(x.price, rel=1e-13)
    assert tr.costs == pytest.approx(costs, rel=1e-12, abs=1e-12)
    assert tr.gross_pnl == pytest.approx(gross)
    # hand-computed literals: entry 3 + 10, exit 3.06 + 10.2; gross +-200
    assert tr.costs == pytest.approx(total_costs, abs=1e-9)
    assert tr.gross_pnl == pytest.approx(200.0 * sign)
    if bps == (10, 1, 2):
        assert tr.entry_price == pytest.approx(100.03 if sign > 0 else 99.97)
        assert tr.exit_price == pytest.approx(101.9694 if sign > 0 else 102.0306)
        assert tr.net_pnl == pytest.approx(200.0 * sign - 26.26)


# ------------------------------------------------------------------ limit
def test_limit_buy_touch_gap_and_miss():
    b = bar(100.0, 103.0, 98.0, 101.0)
    t = limit_fill(b, 1, 99.0, ZERO, 2.0)          # range reaches 99: fills at the limit (open is higher)
    assert t.mid == t.price == 99.0 and t.liquidity is Liquidity.MAKER
    assert limit_fill(b, 1, 98.0, ZERO, 2.0).mid == 98.0   # exact touch counts
    assert limit_fill(b, 1, 97.99, ZERO, 2.0) is None      # one tick short
    assert limit_fill(b, 1, 101.5, ZERO, 2.0).mid == 100.0  # gap through: open, better than the limit


def test_limit_sell_touch_gap_and_miss():
    b = bar(100.0, 103.0, 98.0, 101.0)
    assert limit_fill(b, -1, 102.0, ZERO, 1.0).mid == 102.0
    assert limit_fill(b, -1, 103.0, ZERO, 1.0).mid == 103.0
    assert limit_fill(b, -1, 103.01, ZERO, 1.0) is None
    assert limit_fill(b, -1, 99.0, ZERO, 1.0).mid == 100.0   # open already above the limit


def test_limit_costs_use_maker_fee_and_a51_slip():
    f = limit_fill(bar(100.0, 103.0, 98.0, 101.0), 1, 99.0, C10, 10.0)
    assert f.price == pytest.approx(99.0 * 1.0003)
    assert f.fee == pytest.approx(4e-4 * 10 * 99.0)      # maker bps on the mid notional 990
    assert f.instrument_id == "X" and f.ts == 60_000


# ------------------------------------------------------------------ stop
def test_stop_buy_trigger_and_gap():
    b = bar(100.0, 105.0, 99.0, 104.0)
    f = stop_fill(b, 1, 103.0, ZERO, 1.0)
    assert f.mid == 103.0 and f.liquidity is Liquidity.TAKER
    assert stop_fill(b, 1, 105.0, ZERO, 1.0).mid == 105.0
    assert stop_fill(b, 1, 105.01, ZERO, 1.0) is None
    assert stop_fill(b, 1, 98.0, ZERO, 1.0).mid == 100.0          # gap: open is above the trigger


def test_stop_sell_trigger_and_gap_with_slippage():
    b = bar(100.0, 105.0, 99.0, 104.0)
    assert stop_fill(b, -1, 99.5, ZERO, 1.0).mid == 99.5
    assert stop_fill(b, -1, 99.0, ZERO, 1.0).mid == 99.0
    assert stop_fill(b, -1, 98.99, ZERO, 1.0) is None
    g = stop_fill(b, -1, 101.0, C10, 10.0)                        # open 100 below trigger 101: fills at 100
    assert g.mid == 100.0
    assert g.price == pytest.approx(100.0 * (1 - 3e-4))           # 99.97
    assert g.fee == pytest.approx(1e-3 * 10 * 100.0)              # 1.0


# ------------------------------------------------------------------ bracket
def test_bracket_long_levels_and_gaps():
    b = bar(100.0, 106.0, 97.0, 101.0)
    assert bracket_exit(b, 1, 110.0, 90.0, "sl_first") is None
    assert bracket_exit(b, 1, 105.0, 90.0, "sl_first") == ("TP", 105.0)
    assert bracket_exit(b, 1, 110.0, 98.0, "sl_first") == ("SL", 98.0)
    assert bracket_exit(b, 1, None, None, "sl_first") is None
    assert bracket_exit(bar(95.0, 99.0, 94.0, 96.0), 1, 120.0, 97.0, "sl_first") == ("SL", 95.0)   # gap down
    assert bracket_exit(bar(108.0, 109.0, 107.0, 108.0), 1, 105.0, 90.0, "sl_first") == ("TP", 108.0)  # gap up


def test_bracket_short_levels_and_gaps():
    b = bar(100.0, 106.0, 97.0, 101.0)
    assert bracket_exit(b, -1, 98.0, 110.0, "sl_first") == ("TP", 98.0)
    assert bracket_exit(b, -1, 90.0, 105.0, "sl_first") == ("SL", 105.0)
    assert bracket_exit(bar(108.0, 109.0, 107.0, 108.0), -1, 90.0, 105.0, "sl_first") == ("SL", 108.0)
    assert bracket_exit(bar(92.0, 93.0, 91.0, 92.0), -1, 95.0, 110.0, "sl_first") == ("TP", 92.0)


def test_bracket_tie_break():
    b = bar(100.0, 106.0, 94.0, 101.0)   # both 105 and 95 reached by a long
    assert bracket_exit(b, 1, 105.0, 95.0, "sl_first") == ("SL", 95.0)
    assert bracket_exit(b, 1, 105.0, 95.0, "tp_first") == ("TP", 105.0)
    assert bracket_exit(b, -1, 95.0, 105.0, "sl_first") == ("SL", 105.0)
    assert bracket_exit(b, -1, 95.0, 105.0, "tp_first") == ("TP", 95.0)
    with pytest.raises(ValueError):
        bracket_exit(b, 1, 105.0, 95.0, "nope")


# ------------------------------------------------------------------ property
def test_fill_mids_stay_inside_the_bar():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        o, c = rng.uniform(50, 150, 2)
        hi = max(o, c) + rng.uniform(0, 5)
        lo = min(o, c) - rng.uniform(0, 5)
        b = bar(o, hi, lo, c)
        level = rng.uniform(40, 160)
        for side in (1, -1):
            for fn in (limit_fill, stop_fill):
                f = fn(b, side, level, C10, 1.0)
                if f is not None:
                    assert lo <= f.mid <= hi
                    if f.mid != level:          # off the level only by a gap fill at the open
                        assert f.mid == o
                    assert f.price == pytest.approx(f.mid * (1 + side * C10.slip_rate))
            for tp, sl in ((level, None), (None, level), (level, level * 0.9), (level * 1.1, level)):
                for tb in ("sl_first", "tp_first"):
                    r = bracket_exit(b, side, tp, sl, tb)
                    if r is not None:
                        assert lo <= r[1] <= hi
                        assert r[1] in (o, tp if r[0] == "TP" else sl)

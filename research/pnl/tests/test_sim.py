"""Simulator tests (SPEC 'simulator' TESTS 1-12). Synthetic bars unless stated; tests marked 'data' need PNL_DATA."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import data as D  # noqa: E402
import sim  # noqa: E402


def make_bars(n=3000, seed=0, sigma=5e-4, funding_every=480, rate=1e-4):
    rs = np.random.default_rng(seed)
    t = 1600000000 - 1600000000 % 86400 + 60 * np.arange(n)
    lc = np.log(30000) + np.cumsum(rs.normal(0, sigma, n))
    c = np.exp(lc)
    o = np.r_[c[0], c[:-1]] * np.exp(rs.normal(0, sigma / 4, n))
    h = np.maximum(o, c) * np.exp(np.abs(rs.normal(0, sigma / 2, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rs.normal(0, sigma / 2, n)))
    ft = t[::funding_every][1:]
    fr = np.full(len(ft), rate)
    z = np.zeros(n, np.float32)
    return D.Bars(t, o, h, l, c, z + 1, z + 1, z + 1, z + 0.5, 1, ft, fr, np.zeros(len(ft), bool))


class Flat:
    K, chase_entries = 1, False

    def decide(self, i, x, st):
        return 0.0


def test_zero_position_is_exactly_zero():
    b = make_bars()
    r = sim.simulate(b, Flat(), 1, len(b))
    assert r["final_E"] == 1.0 and r["fees"] == 0.0 and r["funding"] == 0.0 and np.all(r["daily"] == 0.0)


def test_buy_and_hold_nominal_matches_formula():
    b = make_bars(n=1400)                     # one UTC day: no renormalisation, absolute amounts compare directly
    r = sim.simulate(b, sim.ConstPolicy(1.0, K=1), 1, len(b), fee="C7", fill="N", close_at_end=False)
    # nominal: buy at c[1] as maker; PnL = (c_end/c_1 - 1) * (1 - fee) less funding paid on q*c_prev
    q = (1.0 - 0.0) * 1.0 / b.c[1]
    E = 1.0 - sim.MAKER * q * b.c[1]
    fund = 0.0
    for i in range(2, len(b)):
        if b.funding[i]:
            fund += b.funding[i] * q * b.c[i - 1]
    E_exp = E + q * (b.c[-1] - b.c[1]) - fund
    assert abs(r["final_E"] - E_exp) < 1e-9
    assert abs(r["funding"] - fund) < 1e-12


def test_fees_equal_sum_of_fills():
    b = make_bars(seed=1)
    targets = np.full(len(b), np.nan)
    targets[10::200] = 1.0
    targets[110::200] = -1.0
    for fee in ("C4", "C7", "C10"):
        r = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=3, chase=True), 1, len(b), fee=fee, fill="B")
        exp = sum(fr * abs(dq) * px for (_, _, dq, px, fr) in r["fills"])
        assert r["fees"] > 0 and abs(r["fees"] - exp) < 1e-12


def test_maker_never_fills_without_trade_through():
    b = make_bars(seed=2)
    # make every low above the limit after bar 100: buy limit at c[100] cannot fill
    b.l[101:111] = b.c[100] + 1.0
    b.h[101:111] = np.maximum(b.h[101:111], b.l[101:111])
    targets = np.full(len(b), np.nan)
    targets[100] = 1.0
    r = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=5, chase=False), 1, 200, fee="C7", fill="B")
    assert r["n_maker"] == 0 and r["n_cancel"] == 1 and r["final_E"] == 1.0


def test_fill_depends_only_on_window_bars():
    b = make_bars(seed=3)
    targets = np.full(len(b), np.nan)
    targets[500] = 1.0
    targets[600] = 0.0
    r1 = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=3), 1, 700)
    b2 = make_bars(seed=3)
    b2.l[504:600] *= 0.9                      # perturb bars after the maker window (t+K = 503)
    r2 = sim.simulate(b2, sim.TargetPolicy(targets[1:], 1, K=3), 1, 700)
    assert r1["n_maker"] == r2["n_maker"]
    assert r1["trades"][0]["entry_bar"] == r2["trades"][0]["entry_bar"]


def test_flip_cost_nominal_c7():
    b = make_bars(seed=4, rate=0.0)
    b.c[:] = 30000.0
    b.o[:] = 30000.0
    targets = np.full(len(b), np.nan)
    targets[10] = 1.0
    targets[20] = -1.0
    targets[30] = 0.0
    r = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=1), 1, 40, fee="C7", fill="N")
    # +1 maker (2), flip: close taker (5) + open maker (2), close taker (5): 14 bps of equity
    assert abs((1 - r["final_E"]) * 1e4 - 14.0) < 0.05


def test_funding_sign():
    b = make_bars(seed=5, rate=1e-3)
    b.c[:] = 30000.0
    b.o[:] = 30000.0
    long = sim.simulate(b, sim.ConstPolicy(1.0, K=1), 1, len(b), fill="N", close_at_end=False)
    short = sim.simulate(b, sim.ConstPolicy(-1.0, K=1), 1, len(b), fill="N", close_at_end=False)
    assert long["funding"] > 0 and short["funding"] < 0
    assert long["final_E"] < 1.0 - sim.MAKER and short["final_E"] > 1.0 - sim.MAKER


def test_drift_trim():
    b = make_bars(seed=6, rate=0.0)
    b.c[200:] = b.c[199] * 1.25               # +25% jump: a 1x short drifts to |x| > 1.10
    b.o[200:] = b.c[200:]
    b.h[200:] = b.c[200:]
    b.l[200:] = b.c[200:]
    r = sim.simulate(b, sim.ConstPolicy(-1.0, K=1), 1, 400, fill="N")
    assert r["trims"] >= 1


def test_engine_e_nominal_matches_engine_f():
    b = make_bars(seed=7, n=6000)
    rs = np.random.default_rng(0)
    targets = np.full(len(b), np.nan)
    dec = np.arange(20, len(b) - 20, 15)
    x = np.round(rs.uniform(-1, 1, len(dec)), 2)
    targets[dec] = x
    r = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=1), 1, len(b), fee="C7", fill="N", close_at_end=False)
    # engine F on the decision grid: exposure x_k held from close(dec_k) to close(dec_k+1); funding charged in between
    nxt = np.r_[dec[1:], len(b) - 1]
    R = b.c[nxt] / b.c[dec] - 1
    cf = np.r_[0.0, np.cumsum(b.funding)]
    F = cf[nxt + 1] - cf[dec + 1]
    net = sim.engine_f(x, R, F, c_in=sim.MAKER, c_out=sim.TAKER)
    gap_bps = abs(np.log(r["final_E"]) - np.log1p(net).sum()) * 1e4
    days = len(b) / 1440
    assert gap_bps / days < 0.1, gap_bps                # SPEC: engine E (N) vs engine F < 0.1 bps per day


def test_random_positions_lose_about_cost():
    b = make_bars(seed=8, n=20000, rate=0.0)
    rs = np.random.default_rng(1)
    targets = np.full(len(b), np.nan)
    dec = np.arange(30, len(b) - 30, 30)
    targets[dec] = rs.choice([-1.0, 1.0], len(dec))
    r = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=1), 1, len(b), fee="C10", fill="N")
    assert r["final_E"] < 1.0


def test_perfect_foresight_wins():
    b = make_bars(seed=9, n=6000, rate=0.0)
    targets = np.full(len(b), np.nan)
    dec = np.arange(10, len(b) - 20, 15)
    targets[dec] = np.sign(b.c[dec + 15] - b.o[dec + 1])
    r = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=1), 1, len(b), fee="C10", fill="B")
    assert r["final_E"] > 1.2


def test_no_zero_size_fills_and_ruin_stays_exact():
    b = make_bars(seed=10, n=60 * 1440, rate=0.0)
    rs = np.random.default_rng(2)
    targets = np.full(len(b), np.nan)
    dec = np.arange(5, len(b) - 5, 1)
    targets[dec] = rs.choice([-1.0, 0.0, 1.0], len(dec))
    r = sim.simulate(b, sim.TargetPolicy(targets[1:], 1, K=1), 1, len(b), fee="C10", fill="B")
    assert all(abs(dq) * px > 0 for (_, _, dq, px, _) in r["fills"])
    E_rel = []
    for (_, _, dq, px, _) in r["fills"]:
        E_rel.append(abs(dq) * px)
    E_rel = np.array(E_rel)
    assert E_rel.min() > 1e-8                 # no float-residue legs (drift corrections can be small but real)
    assert np.median(E_rel[-1000:]) > 0.5     # sizes are in start-of-day equity units: they never collapse with ruin
    assert np.isfinite(r["daily"]).all() and r["daily"].sum() < -30  # below e^-30: absolute size thresholds broke here
    n_first = sum(1 for tr in r["trades"] if tr["entry_bar"] < len(b) // 2)
    n_last = len(r["trades"]) - n_first
    assert abs(n_first - n_last) < 0.1 * len(r["trades"])


@pytest.mark.skipif(not os.path.exists(os.path.join(D.DATA, "btcusdt_perp_1m.npz")), reason="data")
def test_real_data_buy_and_hold_2021():
    b = D.load(1)
    s, e = b.index_of(1609459200), b.index_of(1640995200)        # 2021
    r = sim.simulate(b, sim.ConstPolicy(1.0, K=15), s, e, fee="C10", fill="N")
    ret = b.c[e - 1] / b.c[s] - 1
    fund = b.funding[s + 1:e].sum()
    approx = (1 + ret) - 1 - fund * (1 + ret / 2)                  # rough funding drag
    assert abs((r["final_E"] - 1) - approx) < 0.05

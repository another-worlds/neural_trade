"""Seconds-long tests of the stops study. usage: python -m pytest -q -p no:cacheprovider runs/tactical/stops/test_stops.py"""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stoplib as L  # noqa: E402


def path(o, h, l, c):
    return tuple(np.asarray(a, float) for a in (o, h, l, c))


def test_stop_and_tp_levels_and_kinds():
    # long entry 100, d 1%: stop 99, tp 101
    o, h, l, c = path([100, 100, 100], [100.5, 101.5, 100], [99.5, 100, 99.5], [100, 101, 100])
    k, p, kind = L.sim_trade(1, o, h, l, c, 0.01, 0.01, False)
    assert (k, kind) == (1, L.KIND_TP) and p == pytest.approx(101.0)            # fills at the level, not at the high
    k, p, kind = L.sim_trade(-1, o, h, l, c, 0.01, 0.01, False)                 # short: stop 101 (high 101.5 at minute 1), tp 99 (low 99.5 never)
    assert (k, kind) == (1, L.KIND_STOP) and p == pytest.approx(101.0)
    k, p, kind = L.sim_trade(1, *path([100, 100], [100.4, 100.4], [99.6, 99.6], [100.1, 100.2]), 0.01, 0.01, False)
    assert (k, kind, p) == (1, L.KIND_TIME, 100.2)                              # nothing touched: close of the last minute


def test_stop_and_tp_in_the_same_minute_takes_the_stop():
    o, h, l, c = path([100, 100], [100, 102], [100, 98], [100, 100])
    k, p, kind = L.sim_trade(1, o, h, l, c, 0.01, 0.01, False)
    assert kind == L.KIND_STOP and p == pytest.approx(99.0)


def test_gap_through_the_stop_fills_at_the_open():
    o, h, l, c = path([100, 97], [100, 98], [99.5, 96], [100, 97])              # minute 1 opens at 97, below the long stop 99
    k, p, kind = L.sim_trade(1, o, h, l, c, 0.01, None, False)
    assert (k, kind) == (1, L.KIND_STOP) and p == pytest.approx(97.0)
    o, h, l, c = path([100, 103], [100, 104], [99.5, 102], [100, 103])          # short: stop 101, minute 1 opens at 103
    k, p, kind = L.sim_trade(-1, o, h, l, c, 0.01, None, False)
    assert kind == L.KIND_STOP and p == pytest.approx(103.0)


def test_trailing_stop_ignores_its_own_minute_extreme():
    # long, trail 1%. minute 1: high 110 and low 99.5 at once. The stop at minute 1 uses the best extreme through minute 0 (100.5): 99.495,
    # so the low 99.5 does not hit it; if the minute's own high had been used the stop would be 108.9 and it would hit.
    o, h, l, c = path([100, 100.2, 105], [100.5, 110, 109], [99.8, 99.5, 104], [100, 108, 106])
    k, p, kind = L.sim_trade(1, o, h, l, c, 0.01, None, True)
    assert k == 2 and kind == L.KIND_STOP and p == pytest.approx(105.0)    # minute 2 opens at 105 below the stop 108.9 (best high 110): fills at the open (gap rule)
    # short mirror
    o, h, l, c = path([100, 99.8, 95], [100.2, 100.4, 96], [99.5, 90, 91], [100, 92, 94])
    k, p, kind = L.sim_trade(-1, o, h, l, c, 0.01, None, True)
    assert k == 2 and kind == L.KIND_STOP and p == pytest.approx(95.0)


def test_a_trade_uses_only_bars_up_to_its_exit_and_the_entry_information():
    rng = np.random.default_rng(0)
    n = 200; cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.0007, n))); op = np.concatenate([[100], cl[:-1]])
    hi = np.maximum(op, cl) * (1 + 0.0003); lo = np.minimum(op, cl) * (1 - 0.0003)
    t = (np.arange(n) * L.MIN_NS).astype(np.int64)
    mins = {"t": t, "open": op, "high": hi, "low": lo, "close": cl}
    for rule in L.STOP_RULES:
        for side in (1, -1):
            a = L.trade_paths(mins, t[0], t[120], side, 0.004, rule)
            kx = int((a[1] - t[0]) // L.MIN_NS)
            m2 = {k: v.copy() for k, v in mins.items()}
            for k in ("open", "high", "low", "close"):
                m2[k][kx + 1:] = m2[k][kx + 1:] * 3 + 5                          # corrupt every bar after the exit minute
            b = L.trade_paths(m2, t[0], t[120], side, 0.004, rule)
            if a[2] != L.KIND_TIME:
                assert a == b, rule
        # the stop distance is an input of the entry bar only: a different sigma changes the path, the same sigma reproduces it
        assert L.trade_paths(mins, t[0], t[120], 1, 0.004, rule) == L.trade_paths(mins, t[0], t[120], 1, 0.004, rule)


def test_state_inputs_ignore_the_future():
    sys.path.insert(0, os.path.join(L.TAC, "sizing")); import test_sizing as TS  # noqa: E402
    rv, times = TS.synth(); h = 4
    base = L.S.state(rv, times, h, "2021-12", "2022-02")
    j = int(np.flatnonzero(pd.DatetimeIndex(times) >= pd.Timestamp("2022-01-15"))[0])
    rv2 = rv.copy(); rv2[j + 1:] = rv2[j + 1:] * 50 + 1.0
    alt = L.S.state(rv2, times, h, "2021-12", "2022-02")
    for k in ("sigma", "dlv", "thr"):
        assert base[k][j] == alt[k][j], k
    assert base["sigma"][j] * np.sqrt(h) == alt["sigma"][j] * np.sqrt(h)


def test_non_overlap_with_early_exits():
    # three signals, entries 0h, 1h, 2h; each would hold 4h on a time exit; the first is stopped at 30 min
    H = L.HOUR
    entry = np.array([0, 1 * H, 2 * H], np.int64)
    ex_long = np.array([30 * 60 * 10 ** 9, 5 * H, 6 * H], np.int64)
    pre_exit = np.stack([ex_long, ex_long], 1)
    si = np.zeros((1, 3), int)
    assert L.sequence(entry, pre_exit, np.ones(3, bool), si)[0].tolist() == [True, True, False]       # 2nd starts after the stop, holds to 5h
    # entry == previous exit is taken; a filtered signal never blocks
    ex = np.array([1 * H, 2 * H, 3 * H], np.int64); pe = np.stack([ex, ex], 1)
    assert L.sequence(entry, pe, np.ones(3, bool), si)[0].tolist() == [True, True, True]
    assert L.sequence(entry, pe, np.array([False, True, True]), si)[0].tolist() == [False, True, True]
    # different sides, different exits: draw 0 long (exit 1h), draw 1 short (exit 10h)
    pe2 = np.stack([np.array([1 * H, 2 * H, 3 * H], np.int64), np.array([10 * H, 10 * H, 10 * H], np.int64)], 1)
    tk = L.sequence(entry, pe2, np.ones(3, bool), np.array([[0, 0, 0], [1, 1, 1]]))
    assert tk[0].tolist() == [True, True, True] and tk[1].tolist() == [True, False, False]


def test_hand_computed_equity_and_frontier():
    pnl = np.array([0.03, -0.04, 0.02, -0.01])
    # equity 1, 1.03, 0.99, 1.01, 1.00; peak 1.03 -> dd (1.03-.99)/1.03 = 3.88%
    assert L.S.dd_pct(pnl)[0] == pytest.approx((1.03 - 0.99) / 1.03 * 100)
    f = L.frontier(pnl)
    row = {r["w"]: r for r in f["rows"]}
    assert row[1.0]["cum_pct"] == pytest.approx(0.0, abs=1e-9)
    assert row[0.5]["dd_pct"] == pytest.approx(L.S.dd_pct(0.5 * pnl)[0])
    assert f["w_dd5"] == 1.0                                                    # 3.88% < 5% already at weight 1
    big = L.frontier(np.array([0.1, -0.2, 0.05]))
    assert big["w_dd5"] < 1.0 and L.S.dd_pct(big["w_dd5"] * np.array([0.1, -0.2, 0.05]))[0] < 5.0
    assert L.S.dd_pct((big["w_dd5"] + 0.01) * np.array([0.1, -0.2, 0.05]))[0] >= 5.0


def test_prev_month_threshold_depends_only_on_the_previous_month():
    rng = np.random.default_rng(1)
    mk = lambda m, shift=0.0: (m, np.arange(200, dtype=np.int64) + 1000 * int(m[-2:]), np.clip(rng.random(200) * 0.6 + 0.2 + shift, 0, 1), np.zeros(200))
    base = [mk("2022-01"), mk("2022-02"), mk("2022-03")]
    thr = L.month_thresholds_prev(base)
    assert set(thr) == {"2022-02", "2022-03"}                                    # the first month has no previous month
    alt = [base[0], (base[1][0], base[1][1], np.clip(base[1][2] + 0.3, 0, 1), base[1][3]), base[2]]
    thr2 = L.month_thresholds_prev(alt)
    assert thr2["2022-02"] == thr["2022-02"]                                     # month 2's own predictions do not enter its threshold
    assert thr2["2022-03"] != thr["2022-03"]                                     # but they set month 3's
    # a gap month: no threshold for the month after it
    assert "2022-04" not in L.month_thresholds_prev([base[0], base[2]]) and "2022-03" not in L.month_thresholds_prev([base[0], base[2]])
    sg = L.signals_v(base, 1, "prev", "2022-01")
    assert set(sg["month"]) == {"2022-02", "2022-03"}
    sg_same = L.signals_v(base, 1, "same", "2022-01")
    assert set(sg_same["month"]) == {"2022-01", "2022-02", "2022-03"} and abs(len(sg_same["month"]) - 60) <= 3


def test_null_is_deterministic_and_matches_a_hand_case():
    n = 6; entry = (np.arange(n) * L.HOUR).astype(np.int64); ex = entry + L.HOUR
    pre = {"X": {"gross": np.stack([np.full(n, 0.01), np.full(n, -0.01)], 1), "exit": np.stack([ex, ex], 1), "kind": np.zeros((n, 2), np.int8)}}
    a = L.null_stats(pre, "X", entry, np.ones(n, bool), (0, 2), {0: 0.0, 2: 0.0}, draws=200)
    b = L.null_stats(pre, "X", entry, np.ones(n, bool), (0, 2), {0: 0.0, 2: 0.0}, draws=200)
    assert a == b
    assert a["0"]["cum_median"] == pytest.approx(0.0, abs=2.01) and a["0"]["n_median"] == n
    s, det = L.actual_stats(pre, "X", entry, np.ones(n), np.ones(n, bool), 0, int(entry[0]), int(ex[-1]))
    assert s["n_trades"] == n and s["hit"] == 1.0 and s["cum_return_pct"] == pytest.approx(6.0)


# ---- real data (reads the cut file: a few seconds)
@pytest.fixture(scope="module")
def mins():
    return L.load_minutes("2024-05-01", "2024-07-01")


def test_cutoff_bounds(mins):
    assert pd.Timestamp(L.C.CUTOFF) == pd.Timestamp("2025-07-20 23:59")
    assert mins["t"].max() < pd.Timestamp("2024-07-01").value                    # the selection data end before 2024-07-01
    full = L.load_minutes("2025-07-01", "2030-01-01")
    assert full["t"].max() <= pd.Timestamp(L.C.CUTOFF).value
    with pytest.raises(AssertionError):
        L.C.assert_cut(pd.DatetimeIndex(["2025-07-21 00:00"]), "x")


def test_x0_equals_the_hourly_r_long_on_real_bars(mins):
    # the entry minute's open and the last horizon minute's close are the hourly bar's open and close
    t = mins["t"]; HOUR = L.HOUR
    hours = np.unique(t // HOUR * HOUR)[24:24 + 40]
    for H in (1, 4):
        for h0 in hours[:15]:
            res = L.trade_paths(mins, h0 + HOUR, h0 + (H + 1) * HOUR, 1, 0.003, "X0_time")
            sl = (t >= h0 + HOUR) & (t < h0 + (H + 1) * HOUR)
            hourly_open = mins["open"][sl][0]; hourly_close = mins["close"][sl][-1]
            assert res[0] == pytest.approx(hourly_close / hourly_open - 1, abs=1e-12) and res[2] == L.KIND_TIME and res[1] == h0 + (H + 1) * HOUR

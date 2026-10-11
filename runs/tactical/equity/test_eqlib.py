"""Seconds-long tests of eqlib. usage: python -m pytest -q -p no:cacheprovider runs/tactical/equity/test_eqlib.py"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eqlib as E  # noqa: E402


def test_non_overlapping_never_opens_before_previous_close():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = 200; entry = np.sort(rng.integers(0, 10 ** 4, n)); dur = rng.integers(0, 300, n); ex = entry + dur
        perm = rng.permutation(n)                                   # input order must not matter
        keep, skipped = E.non_overlapping(entry[perm], ex[perm])
        e, x = entry[perm][keep], ex[perm][keep]
        o = np.argsort(e, kind="stable"); e, x = e[o], x[o]
        assert (e[1:] >= x[:-1]).all()                                # a trade opens at or after the previous close
        assert skipped == n - keep.sum()
        # maximality: every skipped signal overlapped the position open at its entry
        for i in np.where(~keep)[0]:
            ei = entry[perm][i]; assert ((e <= ei) & (x > ei)).any()


def test_equal_entry_and_exit_is_taken_and_open_position_blocks():
    keep, sk = E.non_overlapping([0, 5, 10, 10], [10, 6, 20, 11])
    assert list(keep) == [True, False, True, False] and sk == 2


def test_pnl_uses_entry_price_and_later_prices_only():
    # the price before entry (here 1e9) must not matter: the function takes only entry and a later price
    assert abs(E.trade_pnl(1, 100.0, 101.0) - 0.01) < 1e-12
    assert abs(E.trade_pnl(-1, 100.0, 101.0) + 0.01) < 1e-12
    path = np.array([1e9, 100.0, 90.0, 110.0])        # index 0 is before entry
    for k in (1, 2, 3):
        a = E.trade_pnl(1, path[1], path[k]); path2 = path.copy(); path2[0] = 7.0
        assert a == E.trade_pnl(1, path2[1], path2[k])


def test_stats_hand_computed():
    ns = lambda d: int(d * E.NS_D)
    pnl = np.array([0.01, -0.02, 0.01, 0.03]); te = [ns(1), ns(2), ns(3), ns(40)]; tx = [ns(1.5), ns(2.5), ns(3.5), ns(40.5)]
    s = E.equity_stats(pnl, te, tx, 0, 0.0)
    assert s["n_trades"] == 4 and s["hit"] == 0.75 and abs(s["cum_return_pct"] - 3.0) < 1e-9
    assert abs(s["mean_bps"] - 75.0) < 1e-9
    # equity 1.01, 0.99, 1.00, 1.03: peak 1.01 -> trough 0.99: dd = 0.02 / 1.01
    assert abs(s["max_drawdown_pct"] - 2 / 1.01) < 1e-9
    # underwater from the peak at day 1.5 until equity exceeds 1.01 at day 40.5
    assert abs(s["longest_drawdown_days"] - 39.0) < 1e-9
    s2 = E.equity_stats(pnl, te, tx, 0, 100.0)             # a 100 bps cost makes every trade lose 1%
    assert abs(s2["mean_bps"] - (75.0 - 100.0)) < 1e-9


def test_null_is_centered_and_deterministic():
    rng = np.random.default_rng(0); pl = rng.normal(0, 0.001, 500)
    a = E.null_summary(pl, -pl, 0.0, 300); b = E.null_summary(pl, -pl, 0.0, 300)
    assert a == b and abs(a["hit_median"] - 0.5) < 0.05
    assert a["cum_return_pct_p95"] > a["cum_return_pct_median"]

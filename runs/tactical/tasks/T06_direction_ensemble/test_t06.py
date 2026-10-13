"""T06 tests (seconds, no data files): the ensembles are functions of the members' outputs only, the stacker sees only the tail rows,
thresholds come from the tail only, and the cutoff guard refuses a later bar.
usage: python -m pytest -q runs/tactical/tasks/T06_direction_ensemble/test_t06.py -p no:cacheprovider"""
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import ens_lib as E  # noqa: E402


def members(seed=0, M=4, N=500):
    rs = np.random.RandomState(seed)
    return np.clip(0.5 + 0.1 * rs.randn(M, N, 3), 0.02, 0.98)


def test_e1_e2_are_pure_functions_of_member_outputs():
    Ps = members()
    assert np.allclose(E.e1(Ps), Ps.mean(0))
    z = np.log(Ps / (1 - Ps)).mean(0)
    assert np.allclose(E.e2(Ps), 1 / (1 + np.exp(-z)))
    # no labels, returns or data argument exist: the signatures take the members' outputs and nothing else
    import inspect
    assert list(inspect.signature(E.e1).parameters) == ["Ps"] and list(inspect.signature(E.e2).parameters) == ["Ps"]
    # permuting the members does not change an equal-weight ensemble
    assert np.allclose(E.e1(Ps[::-1]), E.e1(Ps)) and np.allclose(E.e2(Ps[::-1]), E.e2(Ps))
    # a single member is returned unchanged
    assert np.allclose(E.e1(Ps[:1]), Ps[0]) and np.allclose(E.e2(Ps[:1]), Ps[0], atol=1e-5)


def test_tail_rows_are_the_last_15_percent():
    t = E.tail_rows(1000)
    assert t[0] == 850 and t[-1] == 999 and len(t) == 150
    assert E.tail_rows(16560)[0] == int(round(0.85 * 16560))


def test_stacker_is_fitted_on_tail_rows_only():
    rs = np.random.RandomState(1); n = 400
    Ps = members(1, 3, n); rtr = (rs.randn(n, 3) * 1e-3).astype(np.float32)
    db = 5e-4
    m1 = E.fit_stacker(Ps, rtr, db)
    # rows outside the tail can be anything: the stacker is a function of the tail arrays it is handed. Appending validation-like
    # rows to the apply step must not change the fitted coefficients.
    coef_before = [m.coef_.copy() for m in m1]
    _ = E.apply_stacker(m1, members(2, 3, 50))
    assert all(np.array_equal(a, m.coef_) for a, m in zip(coef_before, m1))
    # refitting on the same tail rows is deterministic; refitting on a different set of rows changes the coefficients
    m2 = E.fit_stacker(Ps, rtr, db)
    assert all(np.allclose(a.coef_, b.coef_) for a, b in zip(m1, m2))
    m3 = E.fit_stacker(Ps[:, :200], rtr[:200], db)
    assert not all(np.allclose(a.coef_, b.coef_) for a, b in zip(m1, m3))
    # rows inside the deadband are not used: zeroing them out changes nothing but their labels are never read
    rtr2 = rtr.copy(); small = np.abs(rtr2) <= db; rtr2[small] = 0.0
    m4 = E.fit_stacker(Ps, rtr2, db)
    assert all(np.allclose(a.coef_, b.coef_) for a, b in zip(m1, m4))


def test_stacker_signal_is_recovered_from_the_tail():
    rs = np.random.RandomState(3); n = 3000
    y = rs.rand(n, 3) < 0.5; sig = np.where(y, 0.3, -0.3) + rs.randn(n, 3)
    Ps = np.stack([E.sigmoid(sig), E.sigmoid(rs.randn(n, 3))])           # one informative member, one noise member
    rtr = np.where(y, 1e-3, -1e-3).astype(np.float32)
    models = E.fit_stacker(Ps, rtr, 5e-4)
    assert all(m.coef_[0][0] > 5 * abs(m.coef_[0][1]) for m in models)


def test_threshold_comes_from_the_tail_only_and_is_applied_not_refitted():
    tail = members(4, 1, 300)[0]
    thr = E.conf_threshold(tail)
    assert np.isclose(thr, np.quantile(np.abs(tail.mean(1) - 0.5), 0.9))
    r = np.random.RandomState(5).randn(200, 3) * 1e-3
    P1 = members(6, 1, 200)[0]; P2 = members(7, 1, 200)[0]
    a = E.top_read(P1, r, 5e-4, thr); b = E.top_read(P2, r, 5e-4, thr)
    # the threshold is an argument: the validation predictions cannot move it
    assert E.conf_threshold(tail) == thr
    assert a["n"] != b["n"] or a["hit"] != b["hit"]
    # a huge threshold selects nothing, a zero threshold selects every row above the deadband
    assert E.top_read(P1, r, 5e-4, 10.0)["n"] == 0
    assert E.top_read(P1, r, 5e-4, 0.0)["n"] == int((np.abs(r[:, 1]) > 5e-4).sum())


def test_top_read_uses_the_sign_of_the_mean_p():
    P = np.full((4, 3), 0.9); P[2:] = 0.1
    r = np.array([[0, 1e-3, 0], [0, -1e-3, 0], [0, -1e-3, 0], [0, 1e-3, 0]])
    out = E.top_read(P, r, 5e-4, 0.1)
    assert out["n"] == 4 and out["hit"] == 0.5 and abs(out["bps"]) < 1e-9


def test_score_matches_a_manual_auc():
    rs = np.random.RandomState(8); n = 400
    rva = rs.randn(n, 3) * 1e-3; rtr = rs.randn(n, 3) * 1e-3
    P = np.clip(0.5 + 0.3 * np.sign(rva) * rs.rand(n, 3), 0.01, 0.99)       # informative
    s = E.score(P, rva, rtr, 5e-4)
    assert s["auc3"] > 0.8 and s["ll_h1"] < s["ll_const_h1"]


def test_cutoff_guard_refuses_a_later_bar():
    import cutoff as C
    import pandas as pd
    C.assert_cut(pd.Series(["2025-07-20 23:59:00"]))
    with pytest.raises(AssertionError):
        C.assert_cut(pd.Series(["2025-07-21 00:00:00"]))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        open(p, "w").write("timestamp,open\n2025-07-20 23:59:00,1\n")
        with pytest.raises(ValueError):
            C.read_csv(p)                                   # only the cut file may be read


def test_t06_code_reads_only_the_cut_file():
    for f in ("walkforward.py", "build_cache.py"):
        src = open(os.path.join(HERE, f), encoding="utf-8").read()
        assert "C.read_csv" in src and "pd.read_csv" not in src, f
    banned = "neural_trade/" + "Bitcoin_BTCUSDT.csv"
    for f in os.listdir(HERE):
        if f.endswith(".py") and f != "test_t06.py":
            assert banned not in open(os.path.join(HERE, f), encoding="utf-8").read(), f


def test_last_look_month_is_cut_at_the_cutoff():
    import cutoff as C
    assert C.CUTOFF == "2025-07-20 23:59" and C.LAST_MONTH == "2025-07"

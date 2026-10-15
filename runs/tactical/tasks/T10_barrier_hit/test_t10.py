"""T10 tests (seconds): first-passage formula vs Monte Carlo, labels from high/low by hand, no look-ahead (features, HAR, isotonic), UTC grid,
S3 equivalence with T01, the never-read span, selection committed first.
usage: python -m pytest -q runs/tactical/tasks/T10_barrier_hit/test_t10.py -p no:cacheprovider"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import pytest

import t10_lib as L

DATA_OK = os.path.exists(L.guard.CUT_CSV) and os.path.exists(L.guard.FWD_CSV)


def synth(n=216_000, start="2025-10-01 00:00", seed=0, vol=3e-4, regime=True):
    """Synthetic 1-minute bars (UTC file) with slowly varying volatility."""
    rng = np.random.default_rng(seed)
    v = vol * (np.exp(0.5 * np.convolve(rng.standard_normal(n), np.ones(600) / 24, "same")) if regime else 1.0)
    r = rng.standard_normal(n) * v
    c = 50_000 * np.exp(np.cumsum(r)); o = np.concatenate([[c[0]], c[:-1]])
    hi = np.maximum(o, c) * np.exp(np.abs(rng.standard_normal(n)) * v * 0.5); lo = np.minimum(o, c) * np.exp(-np.abs(rng.standard_normal(n)) * v * 0.5)
    vol_ = rng.gamma(2.0, 5.0, n)
    s = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol_}, index=pd.date_range(start, periods=n, freq="min"))
    return s


# ---------------------------------------------------------------- first passage
def mc_hit(x, sigma, T, n_paths=20000, steps=2000, seed=1):
    """Monte Carlo of the two-sided hit probability with discrete monitoring; the barrier is moved in by 0.5826 sigma sqrt(dt)
    (Broadie-Glasserman-Kou) so that it estimates the continuous-monitoring probability."""
    rng = np.random.default_rng(seed); dt = T / steps; hit = 0; xb = x - 0.5826 * sigma * np.sqrt(dt)
    for _ in range(n_paths // 2000):
        w = np.cumsum(rng.standard_normal((2000, steps)) * sigma * np.sqrt(dt), 1)
        hit += int(np.sum(np.abs(w).max(1) >= xb))
    return hit / n_paths


@pytest.mark.parametrize("tau", [0.04, 0.3, 1.5])
def test_first_passage_matches_monte_carlo(tau):
    sigma, T = 1.0, 1.0; x = np.sqrt(sigma ** 2 * T / tau)
    f = float(L.first_passage(x, sigma, T)); mc = mc_hit(x, sigma, T)
    assert abs(f - mc) < 0.012, (tau, f, mc)


def test_first_passage_series_agree_and_limits():
    tau = np.array([0.02, 0.05, 0.1, 0.19, 0.2, 0.5, 1.0, 3.0])
    x = np.ones_like(tau); sg = np.sqrt(tau)
    fo = np.array([L.first_passage_fourier(1.0, s, 1.0) for s in sg if s ** 2 >= 0.05])
    re = np.array([L.first_passage_reflection(1.0, s, 1.0) for s in sg if s ** 2 >= 0.05])
    assert np.allclose(fo, re, atol=1e-9)
    assert np.allclose(L.first_passage(x, sg, 1.0), [L.first_passage_reflection(1.0, s, 1.0) for s in sg], atol=1e-9)
    assert float(L.first_passage(1.0, 0.0, 60)) == 0.0
    assert float(L.first_passage(1.0, 1e-6, 60)) < 1e-12
    assert float(L.first_passage(1e-4, 1.0, 60)) > 1 - 1e-9
    z = np.linspace(0.01, 5, 60); assert np.all(np.diff(L.first_passage(1.0, z, 1.0)) >= -1e-12)     # more vol, more hits
    # known value: P(sup_{s<=1} |B_s| < 1) = 0.3708, so the hit probability at tau = 1 is 0.6292
    p1 = float(L.first_passage(1.0, 1.0, 1.0)); assert abs(p1 - 0.6292) < 5e-4
    assert p1 <= 2 * 2 * (1 - 0.8413447460685429) + 1e-9                                           # union bound over the two barriers


# ---------------------------------------------------------------- labels by hand
def test_labels_from_high_low_by_hand():
    #            bar:   0     1     2     3     4     5     6
    c = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    h = np.array([100.0, 100.2, 100.3, 100.1, 100.0, 100.0, 101.0])
    l = np.array([100.0, 99.9, 99.8, 99.97, 99.4, 100.0, 100.0])
    # anchor a = 1: reference c[0] = 100, T = 3 -> bars 1, 2, 3: max up ln(100.3/100) = 29.96 bps, max down ln(100/99.8) = 20.02 bps
    m = L.hit_labels(h, l, c, np.array([1]), 3)[0]
    assert m == pytest.approx(np.log(100.3 / 100.0))
    assert (m >= 25e-4) and not (m >= 50e-4)
    # anchor a = 2, T = 3 -> bars 2, 3, 4: down to 99.4 = 60.2 bps beats up 30 bps
    m = L.hit_labels(h, l, c, np.array([2]), 3)[0]
    assert m == pytest.approx(np.log(100.0 / 99.4)); assert m >= 50e-4 and not m >= 100e-4
    # the bar after the window (bar 6, high 101) is not in the window of anchor 2 with T = 3, but is for T = 5
    assert L.hit_labels(h, l, c, np.array([2]), 3)[0] < 90e-4 <= L.hit_labels(h, l, c, np.array([2]), 5)[0] + 1e-12
    assert L.hit_labels(h, l, c, np.array([2]), 5)[0] == pytest.approx(max(np.log(101.0 / 100.0), np.log(100.0 / 99.4)))
    # a high equal to the barrier counts (>=): high exactly c0 * exp(X)
    h2 = h.copy(); h2[1] = 100.0 * np.exp(50e-4)
    assert L.hit_labels(h2, l, c, np.array([1]), 1)[0] >= 50e-4 - 1e-15


def test_labels_in_build_use_the_whole_window_and_reference_close_before_anchor():
    s = synth(n=20_000, regime=False)
    P = L.prep(s, "fwd"); B = L.build(P, 15)
    k = 7; a = B["a"][k]
    assert B["close0"][k] == s["close"].values[a - 1]
    w = slice(a, a + 15)
    exp = max(np.log(s["high"].values[w].max() / s["close"].values[a - 1]), np.log(s["close"].values[a - 1] / s["low"].values[w].min()))
    assert B["move"][k] == pytest.approx(exp)
    for X in L.XS:
        assert B["y"][X][k] == (exp >= X * 1e-4)


# ---------------------------------------------------------------- the anchors
def test_anchors_on_the_utc_grid_and_contiguous():
    s = synth(n=30_000, regime=False)
    s = s.drop(s.index[12_000:12_500])                                   # a 500-minute gap
    P = L.prep(s, "fwd")
    for T in L.TS:
        B = L.build(P, T); ta = B["D"].ta
        assert len(ta) > 0
        mod = (ta.hour * 60 + ta.minute) % T
        assert np.all(np.asarray(mod) == 0)
        for a in B["a"]:
            assert P.tmin[a + T - 1] - P.tmin[a - L.LOOKBACK - 1] == T + L.LOOKBACK            # no gap in history or label
        assert len(np.unique(ta)) == len(ta)


def test_utc_index_of_the_cut_file_matches_utc_hour():
    idx = pd.DatetimeIndex(["2020-01-15 12:00", "2020-07-15 12:00", "2025-03-01 12:00"])
    s = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    P = L.prep(s, "cut")
    assert list(P.hr) == [7, 8, 12]                                       # file hour - 5 (winter), - 4 (summer), UTC after the splice
    assert list(P.utc.hour) == [7, 8, 12]


# ---------------------------------------------------------------- S3 equivalence with T01
def test_s3_features_equal_t01():
    import volnet_lib as V
    s = synth(n=9000, regime=True, seed=3)
    P = L.prep(s, "fwd"); a = np.arange(1500, 8000, 53)
    har, ex10, ok = L.features(P, a)
    D = V.Data(s.values.astype(np.float64), s.index, horizons=(15,), stride=1, first=1441, with_tb7=False)
    sel = np.searchsorted(D.anch, a)
    assert np.array_equal(D.anch[sel], a)
    assert np.allclose(har, D.har[sel], atol=1e-10)
    T1 = L.T3.T1; Bt = T1.build(s, 1441)
    S3 = Bt["F"]["S3"]; sel7 = np.searchsorted(Bt["D"].anch, a); hit = Bt["D"].anch[np.minimum(sel7, len(Bt["D"].anch) - 1)] == a
    assert hit.any()
    S3mine = np.column_stack([har, ex10])
    assert np.allclose(S3mine[hit], S3[sel7[hit]], atol=1e-10)


# ---------------------------------------------------------------- no look-ahead
def test_features_trail_ewma_and_labels_do_not_look_ahead():
    s = synth(n=20_000, regime=True, seed=4); P = L.prep(s, "fwd"); B = L.build(P, 15)
    k = 40; a = B["a"][k]
    s2 = s.copy(); rng = np.random.default_rng(9)
    for col in ("open", "high", "low", "close", "volume"):
        s2.iloc[a:, s2.columns.get_loc(col)] = s[col].values[a:] * rng.uniform(0.5, 2.0, len(s) - a)       # bars >= a changed
    P2 = L.prep(s2, "fwd"); B2 = L.build(P2, 15)
    k2 = int(np.where(B2["a"] == a)[0][0])
    assert np.allclose(B["D"].har[k], B2["D"].har[k2]) and np.allclose(B["ex10"][k], B2["ex10"][k2])
    assert B["trail"][15][k] == B2["trail"][15][k2] and B["ew"][k] == B2["ew"][k2] and B["close0"][k] == B2["close0"][k2]
    assert B["move"][k] != B2["move"][k2]                                                                 # the label does depend on the future


def test_fit_and_isotonic_use_only_the_past():
    s = synth(n=216_000, regime=True, seed=5); T = 60
    P = L.prep(s, "fwd"); B = L.build(P, T)
    month = pd.Period("2026-01", "M"); m0 = month.start_time
    pos = int(np.searchsorted(s.index.values, np.datetime64(m0)))
    s2 = s.copy(); rng = np.random.default_rng(1)
    for col in ("open", "high", "low", "close", "volume"):
        s2.iloc[pos:, s2.columns.get_loc(col)] = s[col].values[pos:] * rng.uniform(0.7, 1.4, len(s) - pos)     # everything from the month start on
    B2 = L.build(L.prep(s2, "fwd"), T)
    m1 = L.fit_models(B, T, m0, days=30); m2 = L.fit_models(B2, T, m0, days=30)
    assert m1["n_train"] == m2["n_train"] and m1["last_label_end"] == m2["last_label_end"]
    assert pd.Timestamp(m1["last_label_end"]) <= m0
    assert np.allclose(m1["har"]["coef"], m2["har"]["coef"]) and m1["har"]["c"] == m2["har"]["c"]
    for X in L.XS:
        a1, a2 = m1["h3"][X], m2["h3"][X]
        assert (a1 is None) == (a2 is None)
        if a1 is not None:
            assert all(np.allclose(u, v) for u, v in zip(a1, a2))
    # the isotonic and base-rate input: the previous month's rows, labels ending before the month starts
    c1 = L.T3.cal_rows(B, T, month); c2 = L.T3.cal_rows(B2, T, month)
    assert np.array_equal(c1, c2) and len(c1) > 0
    assert np.all(B["D"].ta[c1] + pd.Timedelta(minutes=T) <= m0)
    p1, _ = L.h1_prob(B, T, c1, m1["har"]); p2, _ = L.h1_prob(B2, T, c2, m2["har"])
    for X in L.XS:
        assert np.array_equal(B["y"][X][c1], B2["y"][X][c2]) and np.allclose(p1[X], p2[X])


def test_walk_end_to_end_on_brownian_bars_is_calibrated():
    # constant-volatility Brownian bars from fine paths: the first-passage forecast with the true sigma is calibrated; B1 (trailing RV) close to it
    rng = np.random.default_rng(11); n = 216_000; sub = 10; sig = 3e-4
    x = np.cumsum(rng.standard_normal(n * sub) * sig / np.sqrt(sub)); xs = x.reshape(n, sub)
    c = 50_000 * np.exp(xs[:, -1]); o = 50_000 * np.exp(np.concatenate([[0.0], xs[:-1, -1]]))
    hi = 50_000 * np.exp(xs.max(1)); lo = 50_000 * np.exp(xs.min(1))
    s = pd.DataFrame({"open": o, "high": np.maximum(hi, np.maximum(o, c)), "low": np.minimum(lo, np.minimum(o, c)), "close": c, "volume": rng.gamma(2, 5, n)},
                     index=pd.date_range("2025-10-01", periods=n, freq="min"))
    T = 60; B = L.build(L.prep(s, "fwd"), T)
    R = L.walk(B, T, [pd.Period("2026-01", "M"), pd.Period("2026-02", "M")], days=60)
    assert len(R["month"]) > 1000
    y = R["y"][50].astype(float); p = R["p"]["B1"][50]
    assert abs(y.mean() - p.mean()) < 0.03
    assert L.ece(p, y) < 0.06
    for m in L.MODELS:
        assert np.all(np.isfinite(R["p"][m][50])) and R["p"][m][50].min() >= 0 and R["p"][m][50].max() <= 1


# ---------------------------------------------------------------- data guard and selection order
def test_never_read_span_is_refused():
    L.guard.check_index(pd.DatetimeIndex(["2025-07-20 23:59", "2025-10-01 00:00"]))
    for bad in ("2025-07-21 00:00", "2025-08-15 12:00", "2025-09-30 23:59"):
        with pytest.raises(AssertionError):
            L.guard.check_index(pd.DatetimeIndex(["2025-07-20 23:59", bad]))


def test_code_names_no_data_file_but_the_guard():
    forbidden = "neural_trade/" + "Bitcoin_BTCUSDT"
    for f in ("t10_lib.py", "run_select.py", "run_looks.py", "make_figs.py"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            src = open(p, encoding="utf-8").read()
            assert forbidden not in src and "read_csv" not in src, f


@pytest.mark.skipif(not DATA_OK, reason="data files not on this machine")
def test_loaded_files_respect_the_span():
    cut = L.guard.load_ohlcv("cut"); fwd = L.guard.load_ohlcv("fwd")
    assert cut.index.max() <= L.CUTOFF and fwd.index.min() >= pd.Timestamp("2025-10-01") and fwd.index.max() <= pd.Timestamp("2026-09-30 23:59")
    assert not ((cut.index >= "2025-07-21") & (cut.index < "2025-10-01")).any() and not ((fwd.index >= "2025-07-21") & (fwd.index < "2025-10-01")).any()


def test_selection_must_be_committed_first(tmp_path):
    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True, check=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    f = tmp_path / "selection.json"
    with pytest.raises(AssertionError):
        L.assert_selection_committed(str(f), str(tmp_path))            # missing
    f.write_text("{}")
    with pytest.raises(AssertionError):
        L.assert_selection_committed(str(f), str(tmp_path))            # untracked
    git("add", "selection.json")
    with pytest.raises(AssertionError):
        L.assert_selection_committed(str(f), str(tmp_path))            # staged, not in HEAD
    git("commit", "-q", "-m", "x")
    assert len(L.assert_selection_committed(str(f), str(tmp_path))) == 40
    f.write_text('{"a": 1}')
    with pytest.raises(AssertionError):
        L.assert_selection_committed(str(f), str(tmp_path))            # modified after the commit


def test_run_looks_checks_selection_before_anything_else():
    p = os.path.join(HERE, "run_looks.py")
    if not os.path.exists(p):
        pytest.skip("run_looks.py not written yet")
    src = open(p, encoding="utf-8").read()
    assert src.index("assert_selection_committed") < src.index("load_ohlcv")

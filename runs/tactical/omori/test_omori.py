"""Seconds. Hard cutoff (no bar after 2025-07-20), causality, event definition vs a brute-force loop, kernels, purge.
run: python -m pytest -q -p no:cacheprovider runs/tactical/omori/test_omori.py"""
import os, sys, math
import numpy as np
import pandas as pd
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import omorilib as O  # noqa: E402
S = O.S


def _bars(n=20000, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed); sig = 1e-3 * np.exp(0.5 * np.sin(np.arange(n) / 900.0))
    sh = np.where(rng.random(n) < 0.002, 6.0, 1.0)
    lc = np.log(100) + np.cumsum(sig * sh * rng.normal(size=n)); c = np.exp(lc); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.uniform(1, 10, n)},
                        index=pd.date_range(start, periods=n, freq="1min"))


def test_cutoff_constant_and_loader_drops_later_rows(tmp_path):
    assert O.CUTOFF == pd.Timestamp("2025-07-20 23:59")
    df = _bars(3000, start="2025-07-19 00:00"); df = df.copy(); df["timestamp"] = df.index.strftime("%Y-%m-%d %H:%M:%S")
    p = tmp_path / "x.csv"; df[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(p, index=False)
    out = S.load_bars(str(p))
    assert out.index.max() == pd.Timestamp("2025-07-20 23:59") and len(out) == 2880
    B = O.Bars(out); assert B.t_ns.max() == np.datetime64("2025-07-20T23:59", "ns").astype(np.int64)


def test_no_scored_month_or_window_can_pass_the_cutoff():
    df = _bars(60000, start="2025-06-01")                      # runs to 2025-07-12; extend to cross the cutoff
    df = _bars(80000, start="2025-06-01"); df = df[df.index <= O.CUTOFF]
    B = O.Bars(df)
    d = O.MonthData(B, "2025-07", 30, 3.8, train_days=20, kernels=False)
    assert d.anch.max() + 30 - 1 < B.n and B.df.index.max() <= O.CUTOFF
    assert np.isnan(O.target_event(B, np.array([B.n - 5]), 30, 3.8, np.array([1e-3]))[0])      # a window that does not fit is NaN


def test_event_matches_brute_force():
    B = O.Bars(_bars()); anch = np.array([2000, 2500, 9000, 15000]); T = 30; sig = np.full(4, 1e-3)
    ev = O.target_event(B, anch, T, 3.0, sig)
    for k, i in enumerate(anch):
        c0 = B.lc[i - 1]; exc = max(B.lh[i:i + T].max() - c0, c0 - B.ll[i:i + T].min())
        assert ev[k] == float(exc > 3.0 * 1e-3 * math.sqrt(T))


def test_power_kernel_matches_brute_force_and_excludes_the_current_bar():
    rng = np.random.default_rng(1); m = (rng.random(6000) < 0.01) * rng.uniform(1, 3, 6000)
    K = O.power_kernel(m, 0.9, 5.0, D=300)
    for x in (10, 400, 3000, 5999):
        ref = sum(m[x - d] * (d + 5.0) ** -0.9 for d in range(1, 301) if x - d >= 0)
        assert abs(K[x] - ref) < 1e-9
    E = O.exp_kernel(m, 30.0)
    for x in (10, 400, 5999):
        assert abs(E[x] - sum(m[x - d] * math.exp(-d / 30.0) for d in range(1, 3000) if x - d >= 0)) < 1e-9


def test_time_since_last_is_strictly_before():
    m = np.zeros(20); m[5] = 1
    t = O.time_since_last(m, cap=1440)
    assert t[6] == 1 and t[5] > 100 and t[10] == 5


def test_features_are_causal_and_training_never_reads_the_scored_month():
    df = _bars(60000, seed=2); B = O.Bars(df)
    d = O.MonthData(B, "2020-02", 30, 3.8, train_days=20, kernels=True)
    assert d.ok
    # edit every bar from the middle of the scored month on: the HAR fit, the training rows and the earlier test rows must not move
    cut = int(np.searchsorted(B.t_ns, np.datetime64("2020-02-15", "ns").astype(np.int64)))
    df2 = df.copy(); rng = np.random.default_rng(9)
    for col in ("open", "high", "low", "close"):
        df2.iloc[cut:, df2.columns.get_loc(col)] *= np.exp(rng.normal(0, 0.05, len(df2) - cut).cumsum() * 0.01)
    B2 = O.Bars(df2); d2 = O.MonthData(B2, "2020-02", 30, 3.8, train_days=20, kernels=True)
    assert np.allclose(d.har, d2.har)
    ok = d.anch < cut
    assert np.allclose(d.Xb[ok], d2.Xb[ok], equal_nan=True) and np.allclose(d.fs[ok], d2.fs[ok])
    assert np.allclose(d.kern[(0.9, 10.0)][ok], d2.kern[(0.9, 10.0)][ok]) and np.allclose(d.ex[ok], d2.ex[ok])
    assert np.allclose(d.y[d.tr], d2.y[d2.tr], equal_nan=True)


def test_purge_training_windows_close_before_the_month():
    B = O.Bars(_bars(60000, seed=3)); T = 30
    d = O.MonthData(B, "2020-02", T, 3.8, train_days=20, kernels=False)
    m0 = np.datetime64("2020-02-01", "ns").astype(np.int64)
    last_train_close = B.t_ns[d.anch[d.tr] + T - 1] + 60 * 10 ** 9           # close time of the last window bar
    assert (last_train_close <= m0).all()
    assert (B.t_ns[d.anch[d.te] - 1] >= m0).all()


def test_score_month_runs_and_returns_all_arms():
    B = O.Bars(_bars(60000, seed=4)); out = O.score_month(B, "2020-02", 30, 3.0, train_days=20)
    assert out is not None and out["n"] > 500
    for arm in ("const", "B", "pow", "pow_nofs", "exp", "fs"):
        assert np.isfinite(out[f"ll_{arm}"]) and np.isfinite(out[f"bs_{arm}"])
    assert out["p"] in O.P_GRID and out["c"] in O.C_GRID

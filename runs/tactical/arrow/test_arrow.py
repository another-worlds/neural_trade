"""Seconds-fast tests of the arrow study: reversal transforms, the hard cutoff, causality of the gate features, the month split."""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arrowlib as AL
import analyze as AN
import run as RUN


def rand_windows(n=5, seed=0):
    rs = np.random.RandomState(seed)
    c = 100 * np.exp(np.cumsum(rs.randn(n, AL.L) * 1e-3, axis=1)); o = c * np.exp(rs.randn(n, AL.L) * 5e-4)
    hi = np.maximum(o, c) * np.exp(np.abs(rs.randn(n, AL.L)) * 3e-4); lo = np.minimum(o, c) * np.exp(-np.abs(rs.randn(n, AL.L)) * 3e-4)
    v = np.abs(rs.randn(n, AL.L)) * 10
    return np.stack([o, hi, lo, c, v], axis=2)


def test_full_reversal_matches_reversing_the_raw_bars():
    W = rand_windows()
    Wr = W[:, ::-1].copy(); Wr[..., 0], Wr[..., 3] = W[:, ::-1, 3], W[:, ::-1, 0]       # reverse the bars, swap open and close
    X, s = AL.channels(W); Xr, sr = AL.channels(Wr)
    assert np.allclose(AL.t_full(X), Xr, atol=1e-5)
    assert np.allclose(s, sr)                                                              # the scaler is reversal-invariant
    assert np.allclose(AL.t_full(AL.t_full(X)), X)


def test_sign_transform_is_the_mirror_image_of_the_price_path():
    W = rand_windows(seed=1)
    Wm = W.copy(); Wm[..., 0], Wm[..., 1], Wm[..., 2], Wm[..., 3] = 1 / W[..., 0], 1 / W[..., 2], 1 / W[..., 1], 1 / W[..., 3]
    X, s = AL.channels(W); Xm, sm = AL.channels(Wm)
    assert np.allclose(AL.t_sign(X), Xm, atol=1e-4) and np.allclose(s, sm, rtol=1e-5)
    assert np.allclose(AL.t_sign(AL.t_sign(X)), X)


def test_order_transform_only_reverses_and_abs_is_sign_free():
    X, _ = AL.channels(rand_windows(seed=2))
    assert np.array_equal(AL.t_order(X)[..., :4], X[:, ::-1][..., :4])
    assert np.allclose(AL.t_order(X)[:, 1:, 4], X[:, 1:, 4][:, ::-1]) and np.all(AL.t_order(X)[:, 0, 4] == 0)
    assert np.array_equal(AL.t_order(AL.t_order(X)), X)
    A1 = AL.to_abs(X); A2 = AL.to_abs(AL.t_sign(X)); A3 = AL.to_abs(AL.t_full(X))
    assert np.allclose(A1, A2) and np.allclose(A3[..., :3], A1[:, ::-1][..., :3]) and np.allclose(A3, AL.t_order_abs(A1))
    assert np.allclose(A3[:, 1:, 3], A1[:, 1:, 3][:, ::-1])                          # |r| channels ignore the sign, reversal = order


def test_window_scaling_is_time_symmetric_unit_rms_body():
    X, sig = AL.channels(rand_windows(seed=3))
    rms = lambda Z: np.sqrt(np.mean(Z[..., 0] ** 2 + Z[..., 4] ** 2, axis=1))
    assert np.allclose(rms(X), 1.0, atol=1e-4)
    for T in (AL.t_full, AL.t_order, AL.t_sign):
        assert np.allclose(rms(T(X)), 1.0, atol=1e-4)


def test_hard_cutoff_bars_and_anchors():
    s = AL.load_bars()
    assert s.index.max() <= pd.Timestamp("2025-07-20 23:59:00")
    _, A, pos, idx = AL.anchor_table()
    assert len(A) == len(s) and pos.max() + AL.H < len(A)                                  # every label bar exists inside the cutoff
    assert s.index[pos.max() + AL.H] <= AL.LAST_BAR
    if os.path.exists(f"{AL.CACHE}/t.npy"):
        assert pd.Timestamp(np.load(f"{AL.CACHE}/t.npy").max()) <= AL.LAST_BAR
    # the evaluation months all close before the cutoff
    assert (pd.Period(RUN.LAST, "M") + 1).start_time <= pd.Timestamp("2025-07-21")


def test_gate_features_are_causal():
    W = rand_windows(n=4, seed=4); X, sig = AL.channels(W)
    A = np.concatenate([W[0], rand_windows(1, seed=9)[0]])                                 # 60 window bars + 60 'future' bars
    X1, s1 = AL.channels(A[None, :AL.L]); A2 = A.copy(); A2[AL.L:] *= 3.0                  # change everything after the anchor
    X2, s2 = AL.channels(A2[None, :AL.L])
    assert np.array_equal(X1, X2) and np.array_equal(s1, s2)


def test_gate_edges_use_only_earlier_months_and_assign_bins():
    rs = np.random.RandomState(0); months = np.repeat(np.arange(24), 1000) + pd.Period("2021-01", "M").ordinal
    v = rs.randn(len(months))
    e1 = AN.edges_for("2022-01", months, v)
    v2 = v.copy(); v2[months >= pd.Period("2022-01", "M").ordinal] += 100.0                # perturb the month itself and later
    assert np.array_equal(e1, AN.edges_for("2022-01", months, v2))
    v3 = v.copy(); v3[months < pd.Period("2021-07", "M").ordinal] += 100.0                 # outside the 6-month look-back
    assert np.array_equal(e1, AN.edges_for("2022-01", months, v3))
    b = AN.assign_bins(v, e1); assert set(np.unique(b)) == set(range(AN.NBIN))
    assert AN.edges_for("2021-01", months, v) is None                                      # not enough history


def test_month_split_has_no_shared_bar_between_train_and_test():
    t = (pd.date_range("2019-12-01", "2021-03-01", freq="10min").values.astype("datetime64[ns]")).astype(np.int64)
    tr, va, te = RUN.month_split(t, "2021-01")
    tt = pd.DatetimeIndex(t.astype("datetime64[ns]")); m0 = pd.Timestamp("2021-01-01")
    assert tt[va].max() <= m0 - pd.Timedelta(minutes=60) and tt[te].min() >= m0 and tt[te].max() < pd.Timestamp("2021-02-01")
    assert tt[tr].max() <= tt[va].min() - pd.Timedelta(minutes=60)                           # gap between train and validation
    assert tt[tr].min() >= m0 - pd.Timedelta(days=365)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()


def test_bce_logits_matches_definition():
    z = np.array([-30.0, -1.0, 0.0, 2.0, 30.0])
    for y in (0.0, 1.0):
        ref = -(y * np.log(1 / (1 + np.exp(-z))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-z))))
        got = AL.bce_logits(z, y)
        assert np.allclose(got[1:4], ref[1:4]) and np.all(np.isfinite(got))


def test_detrending_removes_a_shared_trend():
    n = 40; years = np.repeat([2021, 2022, 2023, 2024], 10)
    trend = np.linspace(0, 1, n); rs = np.random.RandomState(0)
    a = trend + 0.05 * rs.randn(n); b = trend + 0.05 * rs.randn(n)
    raw = AN.stats.spearmanr(a, b).correlation
    d, _, _ = AN.spearman_detrended(a, b, years, "diff"); v, _, _ = AN.spearman_detrended(a, b, years, "dev")
    assert raw > 0.9 and abs(d) < 0.5 and abs(v) < 0.8

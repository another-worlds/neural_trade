"""Seconds. Causality, the hard cutoff (no bar after 2025-07-20), sign-invariant targets, purge, brute-force target check.
run: python -m pytest -q -p no:cacheprovider runs/tactical/shape/test_shape.py"""
import os, sys
import numpy as np
import pandas as pd
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shapelib as S  # noqa: E402


def _bars(n=6000, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed); sig = 1e-3 * np.exp(0.5 * np.sin(np.arange(n) / 700.0))
    lc = np.log(100) + np.cumsum(sig * rng.normal(size=n)); c = np.exp(lc); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    v = rng.uniform(1, 10, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="1min"))


def _brute(A, i, T):
    c = np.log(A[:, 3]); r = np.diff(c[i - 1:i + T])                      # r_j for j = i..i+T-1
    top = max(np.log(A[i:i + T, 1]).max(), c[i - 1]); bot = min(np.log(A[i:i + T, 2]).min(), c[i - 1])
    rho = abs(c[i + T - 1] - c[i - 1]) / (top - bot); q = (r ** 2).sum()
    z2 = sum(r[a] * r[b] for a in range(T) for b in range(a + 2, T)) / q
    z3 = sum(r[a] * r[b] for a in range(T) for b in range(a + 3, T)) / q
    return rho, r.sum() ** 2 / q, z2, z3


def test_targets_match_a_brute_force_loop_and_rho_is_in_unit_interval():
    A = _bars().to_numpy(); anch = np.array([1500, 1510, 3000, 5000])
    for T in (30, 60):
        tg = S.targets_at(A, anch, T)
        for k, i in enumerate(anch):
            b = _brute(A, int(i), T)
            for name, v in zip(S.TARGETS, b):
                assert abs(tg[name][k] - v) < 1e-9 * max(1, abs(v)), (name, T, i)
        assert (tg["rho"] >= 0).all() and (tg["rho"] <= 1).all()


def test_targets_are_sign_invariant_under_price_reflection():
    A = _bars(seed=3).to_numpy(); k = 100.0 ** 2
    R = np.stack([k / A[:, 0], k / A[:, 2], k / A[:, 1], k / A[:, 3], A[:, 4]], 1)          # ln P -> -ln P + const
    anch = np.arange(1500, 5000, 111)
    for T in S.TS[:2]:
        a, b = S.targets_at(A, anch, T), S.targets_at(R, anch, T)
        for name in S.TARGETS:
            assert np.allclose(a[name], b[name], rtol=1e-9, atol=1e-12, equal_nan=True), (name, T)


def test_features_and_targets_ignore_later_bars():
    df = _bars(n=6000); A = df.to_numpy(); b = 3500
    A2 = A.copy(); rng = np.random.default_rng(5); A2[b + 1:] = A2[b + 1:] * rng.uniform(0.5, 1.5, (len(A) - b - 1, 5))
    f1, f2 = S.per_bar_features(A), S.per_bar_features(A2)
    for k in f1:
        assert np.array_equal(f1[k][:b + 1], f2[k][:b + 1], equal_nan=True), k
    anch = np.arange(1500, 3300, 50)
    for T in (30, 60):
        t1, t2 = S.targets_at(A, anch, T), S.targets_at(A2, anch, T)
        for name in S.TARGETS:
            assert np.array_equal(t1[name], t2[name], equal_nan=True)         # windows end by bar 3359 <= b
    anch2 = np.array([3490, 3600])
    assert not np.allclose(S.targets_at(A, anch2, 60)["rho"], S.targets_at(A2, anch2, 60)["rho"])      # and they do read the future of a later anchor


def test_loader_never_holds_a_bar_after_the_cutoff(tmp_path, monkeypatch):
    idx = pd.date_range("2025-07-19 12:00", "2025-07-23 00:00", freq="1min")
    d = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    p = tmp_path / "x.csv"; d.to_csv(p, index=False)
    s = S.load_bars(str(p), cutoff="2025-07-20 23:59", chunk=500)
    assert s.index.max() == pd.Timestamp("2025-07-20 23:59") and (s.index <= S.CUTOFF).all()
    # reading stops at the chunk that crosses the cutoff: later chunks are never requested
    seen = []; real = pd.read_csv
    def spy(*a, **k):
        for ch in real(*a, **k):
            seen.append(len(ch)); yield ch
    monkeypatch.setattr(pd, "read_csv", spy)
    S.load_bars(str(p), cutoff="2025-07-20 23:59", chunk=500)
    n_to_cut = int((idx <= pd.Timestamp("2025-07-20 23:59")).sum())
    assert sum(seen) <= n_to_cut + 500
    ds = S.build_dataset(s.assign(open=np.exp(np.cumsum(np.full(len(s), 1e-4)))).assign(high=lambda x: x.open * 1.0001, low=lambda x: x.open * 0.9999, close=lambda x: x.open)) if len(s) > 4000 else None
    if ds is not None:
        assert pd.Timestamp(ds["t"].max()) <= S.CUTOFF


def test_cutoff_constant_and_real_file_if_present():
    assert S.CUTOFF == pd.Timestamp("2025-07-20 23:59")
    if os.path.exists(S.CSV):
        s = S.load_bars()
        assert s.index.max() <= pd.Timestamp("2025-07-20 23:59") and s.index.max() >= pd.Timestamp("2025-07-20 00:00")
        n = len(s); anch = np.arange(S.WARM, n - S.HDIR, S.STRIDE)
        assert all((anch + T - 1 < n).sum() > 0 for T in S.TS)


def test_dataset_anchors_and_target_windows_stay_inside_the_data():
    df = _bars(n=5000); ds = S.build_dataset(df); n = len(df)
    assert pd.Timestamp(ds["t"].max()) <= df.index.max()
    for T in S.TS:
        late = ds["anch"] + T - 1 >= n
        assert np.isnan(ds[f"rho_{T}"][late]).all()


def test_purge_training_windows_close_before_the_month_start():
    t = pd.date_range("2020-01-01", "2021-03-01", freq="10min").values.astype("datetime64[ns]").astype(np.int64)
    for T in S.TS:
        for m in pd.period_range("2020-06", "2020-12", freq="M"):
            m0, m1 = m.start_time, (m + 1).start_time
            tr, te = S.split_masks(t, m0, m1, T)
            tt = pd.DatetimeIndex(np.asarray(t).astype("datetime64[ns]"))
            assert tr.any() and te.any() and not (tr & te).any()
            assert (tt[tr] + pd.Timedelta(minutes=T + 1)).max() <= m0
            assert tt[tr].min() >= m0 - pd.Timedelta(days=365) and tt[te].min() >= m0 and tt[te].max() < m1


def test_score_month_ignores_labels_that_reach_into_the_scored_month_and_everything_after(monkeypatch):
    monkeypatch.setattr(S, "MIN_TRAIN", 300); monkeypatch.setattr(S, "MIN_TEST", 100)
    ds = S.build_dataset(_bars(n=60000, seed=2, start="2020-01-01"))        # ~41 days
    a = S.score_month(ds, "2020-02", models=("ridge",), train_days=30)
    assert a is not None and len(a["cells"]) == 12
    m0, m1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")
    tt = pd.DatetimeIndex(np.asarray(ds["t"]).astype("datetime64[ns]"))

    def corrupt(bad_fn):
        d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ds.items()}
        for T in S.TS:
            for tg in S.TARGETS:
                d2[f"{tg}_{T}"][bad_fn(T)] *= -7.0
        d2["rdir"][bad_fn(S.HDIR)] *= -7.0
        return d2
    border = lambda T: np.asarray((tt > m0 - pd.Timedelta(minutes=T + 1)) & (tt < m0))
    b = S.score_month(corrupt(border), "2020-02", models=("ridge",), train_days=30)
    for x, y in zip(a["cells"], b["cells"]):
        assert x["sp_b"] == y["sp_b"] and x["sp_c"] == y["sp_c"]
    assert a["auc_tb7"] == b["auc_tb7"]
    legit = lambda T: np.asarray((tt <= m0 - pd.Timedelta(minutes=T + 1)) & (tt > m0 - pd.Timedelta(days=5)))
    c = S.score_month(corrupt(legit), "2020-02", models=("ridge",), train_days=30)
    assert any(x["sp_c"] != y["sp_c"] for x, y in zip(a["cells"], c["cells"]))        # the probe does bite when training labels change

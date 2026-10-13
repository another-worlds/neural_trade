"""O1 tests (SPEC "Tests"): run with pytest in this folder; seconds."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o1_lib as L  # noqa: E402
T = L.T


def synth(n=6000, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    idx = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def test_no_forbidden_bar_read():
    f = T.load_bars(T.FWD_CSV, T.FWD_START, pd.Timestamp("2025-10-03 00:00"))
    assert f.index.min() >= pd.Timestamp("2025-10-01") and not ((f.index >= T.FORBID_LO) & (f.index <= T.FORBID_HI)).any()
    tr = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2025-07-01"), T.CUTOFF)
    assert tr.index.max() <= T.CUTOFF
    tr2 = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-22"))
    assert tr2.index.max() <= T.CUTOFF                     # the cut file has no later bar
    import tempfile
    d = tempfile.mkdtemp(); p = os.path.join(d, "x.csv")
    pd.DataFrame({"timestamp": ["2025-08-01 00:00:00"], "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}).to_csv(p, index=False)
    try:
        T.load_bars(p, pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-01"))
    except AssertionError:
        pass
    else:
        raise AssertionError("the forbidden-gap guard did not fire")


def test_training_labels_end_before_the_month():
    ta = pd.date_range("2024-01-01", "2025-09-01", freq="7min").values.astype("datetime64[ns]")
    ok = np.ones(len(ta), bool)
    for H in L.HORIZONS:
        for mo in ("2024-06", "2025-02"):
            m0 = pd.Period(mo).start_time; tr = L.train_rows(ta, ok, H, m0)
            assert len(tr) > 1000
            assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(m0, "ns")).all()
            assert (ta[tr] >= np.datetime64(m0 - pd.Timedelta(days=365), "ns")).all()
            assert tr.max() < np.searchsorted(ta, np.datetime64(m0, "ns"))
        tr = L.train_rows(ta, ok, H, L.TRAIN_END)           # the frozen fit: labels end by 2025-07-21 00:00 (bar 23:59 closed)
        assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(L.TRAIN_END, "ns")).all()
    fit, hold = L.split_fit_hold(np.arange(1000), 240)
    assert hold.min() - fit.max() > 240 / 7           # a gap of H minutes between fit and hold


def test_geometry_is_causal():
    s = synth(); A = s.values.astype(np.float64)
    for a in (1500, 2600, 4001):
        g1, g15 = L.window_features(A, [a])
        A2 = A.copy(); A2[a:] = np.random.default_rng(1).uniform(1, 1000, A2[a:].shape)
        h1, h15 = L.window_features(A2, [a])
        assert np.array_equal(g1, h1) and np.array_equal(g15, h15)
        assert g15.shape[1] == g1.shape[1] == 164


def test_agg15_equals_pandas_resample():
    s = synth(n=3000, start="2024-01-01 00:00"); A = s.values.astype(np.float64)
    a = 1800 + 7 * 3                                        # bar a-900 = 900 + 21: not aligned; pick an aligned one instead
    a = 900 + 15 * 70 + 900                                 # a-900 = 1950 minutes after 00:00 = aligned to 15 (1950 % 15 == 0)
    W = L.agg15(A, [a])[0]
    r = s.iloc[a - 900:a].resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    assert len(r) == 60 and np.allclose(W, r.values)
    # unaligned anchor: groups start at a-900, so compare with a resample anchored at that bar
    a2 = a + 4
    W2 = L.agg15(A, [a2])[0]
    r2 = s.iloc[a2 - 900:a2].resample("15min", origin=s.index[a2 - 900]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    assert np.allclose(W2, r2.values)


def test_s3_matches_t01_build_row_for_row():
    s = synth(n=8000, seed=3); first = T.WARM
    B = T.build(s, first); F = L.build_features(s, first)
    assert np.array_equal(F["S3"], B["F"]["S3"].astype(np.float32))
    assert np.array_equal(F["ta"], B["D"].ta.values.astype("datetime64[ns]").astype(np.int64))
    # H = 60 target and the G1 block at the first anchor agree with a direct computation
    a = B["D"].anch; i = 5
    assert np.isclose(F["rv60"][i], (B["D"].lr[a[i]:a[i] + 60] ** 2).sum())
    g1, _ = L.window_features(s.values.astype(np.float64), a[i:i + 1])
    assert np.array_equal(F["G1"][i:i + 1], g1)


def test_ridge_equals_sklearn():
    from sklearn.linear_model import Ridge as SK
    rng = np.random.default_rng(0); X = rng.normal(size=(500, 12)) * rng.uniform(1, 5, 12) + 3; y = X[:, 0] - 2 * X[:, 3] + rng.normal(size=500)
    r = L.Ridge(X); idx = np.array([0, 2, 3, 5, 9]); m = r.fit(y, idx)
    mu, sd = X[:, idx].mean(0), X[:, idx].std(0) + 1e-12
    sk = SK(alpha=1.0).fit((X[:, idx] - mu) / sd, y)
    assert np.allclose(r.predict(m, X), sk.predict((X[:, idx] - mu) / sd))


def test_fit_and_score_run_on_synthetic():
    s = synth(n=40000, seed=5, start="2024-01-01"); F = L.build_features(s, T.WARM)
    m0 = pd.Timestamp("2024-01-25"); m1 = pd.Timestamp("2024-01-27")
    st = L.fit_state(F, 60, m0, groups=True, hgb=False)
    te = L.test_rows(F["ta"], F["ok60"], m0, m1)
    rows = L.score_state(st, F, te, "sel", "2024-01")
    mods = {r["model"] for r in rows}
    assert {"R0", "R1", "R2", "Rsh"} <= mods and any(m.startswith("grp:") for m in mods)
    assert all(r["spearman"] is not None for r in rows)


def test_qlike_uses_exp_mu_for_log_rv():
    rng = np.random.default_rng(0); mu = rng.normal(-9, 0.5, 2000); v = np.exp(mu) * rng.exponential(2.0, 2000)
    c = L.qlike_c(v, mu); assert np.isclose(c, np.mean(v / np.exp(mu)))
    q = v / (c * np.exp(mu)); assert np.isclose(L.qlike(v, mu, c), np.mean(q - np.log(q) - 1)) and np.isclose(q.mean(), 1)
    assert L.qlike(v, mu, c) < L.qlike(v, mu + 0.5, c)         # a biased forecast is worse at the same scale

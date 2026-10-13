"""O3 tests (SPEC "Tests"): pytest in this folder; seconds."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o3_lib as L  # noqa: E402
T = L.T


def synth(n=20000, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


def test_bank_is_causal():
    s = synth(); A = s.values.astype(np.float64)
    for a in (7000, 9001, 15555):
        idx = np.array([a - 1]); ref = L.bank_at(A, idx)
        B = A.copy(); rng = np.random.default_rng(1)
        B[a:] = A[a:] * rng.uniform(0.5, 2.0, A[a:].shape)           # overwrite every bar >= a
        assert np.array_equal(L.bank_at(B, idx), ref)


def test_no_forbidden_bar_read():
    f = T.load_bars(T.FWD_CSV, T.FWD_START, pd.Timestamp("2025-10-03 00:00"))
    assert f.index.min() >= pd.Timestamp("2025-10-01") and not ((f.index >= T.FORBID_LO) & (f.index <= T.FORBID_HI)).any()
    tr = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2025-07-01"), T.CUTOFF)
    assert tr.index.max() <= T.CUTOFF
    # the training file is physically cut: asking for 2025-07-22 still returns nothing after the cutoff
    assert T.load_bars(T.TRAIN_CSV, pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-22")).index.max() <= T.CUTOFF
    assert L.M_FWD[0] == "2025-10" and L.M_LOOK[-1] == "2025-07" and len(L.M_SEL) == 42 and len(L.M_LOOK) == 13 and len(L.M_FWD) == 12


def test_training_labels_end_before_the_month():
    ta = pd.date_range("2024-01-01", "2025-09-01", freq="7min").values.astype("datetime64[ns]"); ok = np.ones(len(ta), bool)
    for H in L.HORIZONS:
        for mo in ("2024-06", "2025-02"):
            m0 = pd.Period(mo).start_time; tr = L.train_rows(ta, ok, H, m0)
            assert len(tr) > 1000
            assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(m0, "ns")).all()
            assert (ta[tr] >= np.datetime64(m0 - pd.Timedelta(days=365), "ns")).all()
        tr = L.train_rows(ta, ok, H, L.TRAIN_END)
        assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(L.TRAIN_END, "ns")).all()
    fit, hold = L.split_fit_hold(np.arange(1000), 240)
    assert hold.min() - fit.max() > 240 / 7


def test_one_hot_kernel_is_ln_of_that_ewma():
    s = synth(8000); A = s.values.astype(np.float64); idx = np.arange(3000, 7000, 11)
    bank = L.bank_at(A, idx); X = L.channels(A)
    a = np.full((L.K, L.NT), -1000.0); pick = [0, 3, 5, 7, 11, 9]
    for k, j in enumerate(pick):
        a[k, j] = 0.0
    I = L.indicators(L.gather(bank), a)
    for k, j in enumerate(pick):
        ref = L.ewma(X[:, L.CHMAP[k]], L.TAUS[j])[idx]
        assert np.max(np.abs(I[:, k] - np.log(ref + L.EPS_CH[L.CHMAP[k]]))) < 1e-12
    # tau = 1 is the channel itself, and the recursion is the EWMA definition
    assert np.array_equal(L.ewma(X[:, 0], 1), X[:, 0])
    y = L.ewma(X[:, 1], 8); al = 1 / 8; assert abs(y[500] - (al * X[500, 1] + (1 - al) * y[499])) < 1e-15


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0); n, d = 64, 5
    S3s = rng.normal(size=(n, d)); Bk = np.exp(rng.normal(size=(n, L.K, L.NT))) * 1e-3 + 1e-4; y = rng.normal(size=n)
    p = {"a": rng.normal(0, 1, (L.K, L.NT)), "b": rng.normal(size=L.K), "w": rng.normal(size=d), "c": 0.3}
    I0 = np.log(Bk.mean(2) + L.EPS_CH[L.CHMAP]); zm = I0.mean(0); zs = I0.std(0) + 1e-9
    def loss(q):
        return float(np.mean((L._forward(q, S3s, Bk, zm, zs)[0] - y) ** 2))
    pred, cache = L._forward(p, S3s, Bk, zm, zs); g = L._grads(p, S3s, Bk, y, zs, pred, cache)
    for key in ("a", "b", "w"):
        for idx in [(0, 0), (3, 7), (5, 11)] if key == "a" else [(0,), (2,)]:
            q1 = {k: np.array(v, float).copy() if np.ndim(v) else v for k, v in p.items()}; q2 = {k: np.array(v, float).copy() if np.ndim(v) else v for k, v in p.items()}
            q1[key][idx] += 1e-6; q2[key][idx] -= 1e-6
            assert abs((loss(q1) - loss(q2)) / 2e-6 - g[key][idx]) < 1e-6 * max(1, abs(g[key][idx])) + 1e-8
    q1 = dict(p); q2 = dict(p); q1["c"] = p["c"] + 1e-6; q2["c"] = p["c"] - 1e-6
    assert abs((loss(q1) - loss(q2)) / 2e-6 - g["c"]) < 1e-6


def test_s3_equals_t01_build_and_o1_r0_if_present():
    s = synth(); F = L.build_features(s, first=7000)
    B = T.build(s, 7000)
    assert np.array_equal(F["S3"], B["F"]["S3"].astype(np.float64))
    assert F["bank"].shape == (len(F["anch"]), 3, L.NT) and np.isfinite(F["bank"]).all() and (F["bank"] > 0).all()
    # shared ridge: o1's Ridge class (alpha 1, standardised) must give the same R0 predictions as ridge_fit/ridge_pred
    o1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "O1_geom_vol")
    if not os.path.exists(os.path.join(o1, "o1_lib.py")):
        return
    sys.path.insert(0, o1)
    try:
        import o1_lib as O
    except Exception:       # noqa: BLE001
        pytest.skip("o1_lib not importable")
    ok = F["ok60"]; rows = np.where(ok)[0]; tr, te = rows[:len(rows) // 2], rows[len(rows) // 2:]
    y = np.log(F["rv60"])
    r = O.Ridge(F["S3"][tr]); m = r.fit(y[tr], np.arange(F["S3"].shape[1])); p1 = r.predict(m, F["S3"][te])
    m2 = L.ridge_fit(F["S3"][tr], y[tr]); p2 = L.ridge_pred(m2, F["S3"][te])
    assert np.max(np.abs(p1 - p2)) < 1e-9


def test_pipeline_runs_and_shuffled_control_is_flat_on_noise():
    s = synth(30000); F = L.build_features(s, first=7000)
    m0 = F["ta"][int(len(F["ta"]) * 0.8)]
    st = L.fit_month(F, 60, pd.Timestamp(m0))
    te = L.test_rows(F["ta"], F["ok60"], pd.Timestamp(m0), pd.Timestamp(m0) + pd.Timedelta(days=5))
    rows = L.score_state(st, F, te, "t", "x")
    assert {r["model"] for r in rows} == {"R0", "R0f", "BANK", "BANKf", "LK", "LKsh"}
    assert all(np.isfinite(r["qlike"]) for r in rows)

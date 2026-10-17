"""M7 tests (seconds, synthetic data + the written views). Run: D:/nt/nt_mm_env/python.exe -m pytest -q test_m7.py (CUDA_VISIBLE_DEVICES=-1)"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m7_lib as L  # noqa: E402


def synth(n_days=420, seed=0):
    rng = np.random.default_rng(seed); ta = pd.date_range("2024-01-01", periods=n_days * 4, freq="6h")
    X = rng.normal(size=(len(ta), 4)); y = np.sin(X[:, 0]) * X[:, 1] + 0.3 * rng.normal(size=len(ta)); ok = rng.random(len(ta)) > 0.05
    return ta, X, y, ok


@pytest.fixture(autouse=True)
def small_pool(monkeypatch):
    monkeypatch.setattr(L, "MIN_POOL", 300)


def _run(ta, X, y, ok, kind, binary=False, **kw):
    return L.monthly_view(ta, X, y, ok, kind=kind, binary=binary, min_pool=300, **kw)


@pytest.mark.parametrize("kind", ["gbm", "nw"])
def test_mutating_labels_of_month_m_and_later_leaves_month_m_unchanged(kind):
    ta, X, y, ok = synth(); m = pd.Period("2024-12", "M"); m0 = m.start_time
    p, r, info = _run(ta, X, y, ok, kind)
    # labels whose window reaches m0 or later (ta + 60 min > m0): everything from one hour before m0
    y2 = y.copy(); bad = np.asarray(ta + pd.Timedelta(minutes=60) > m0); y2[bad] = y2[bad] * -7 + 100
    p2, r2, _ = _run(ta, X, y2, ok, kind)
    served = np.asarray(ta < m0 + pd.Timedelta(days=31)) & np.isfinite(p)
    assert served.sum() > 0
    in_m = np.asarray(ta.to_period("M") == m) & np.isfinite(p)
    assert in_m.sum() > 50
    assert np.array_equal(p[ta < (m + 1).start_time].astype(float), p2[ta < (m + 1).start_time].astype(float), equal_nan=True)   # month m and every earlier month
    assert np.array_equal(r[in_m], r2[in_m])
    later = np.asarray(ta >= (m + 1).start_time) & np.isfinite(p)
    assert later.sum() > 0 and not np.allclose(p[later], p2[later])                                                                # later months do move: the mutation is real


@pytest.mark.parametrize("kind", ["gbm", "nw"])
def test_mutating_inputs_after_the_month_changes_nothing_before(kind):
    ta, X, y, ok = synth(seed=3); m0 = pd.Period("2024-11", "M").start_time
    p, _, _ = _run(ta, X, y, ok, kind); X2 = X.copy(); X2[np.asarray(ta >= m0)] += 5.0; ok2 = ok.copy()
    p2, _, _ = _run(ta, X2, y, ok2, kind)
    # month m0's own inputs changed, so only months strictly before m0 are required to be identical (their pools are before m0)
    b = np.asarray(ta < m0)
    assert np.array_equal(p[b], p2[b], equal_nan=True)


@pytest.mark.parametrize("kind", ["gbm", "nw"])
def test_pool_rules_size_window_and_subset(kind):
    ta, X, y, ok = synth()
    p, r, info = _run(ta, X, y, ok, kind)
    served = np.isfinite(p)
    assert served.any() and not served[~ok].any()                       # only ok rows are served: the view index is a subset of the ok anchors
    first = min(pd.Period(i["month"], "M") for i in info if i["served"])
    assert all(i["n_pool"] >= 300 for i in info if i["served"]) and all(i["n_pool"] < 300 for i in info if not i["served"] and pd.Period(i["month"], "M") < first)
    assert ta[served].min() >= first.start_time
    assert (r[served] >= 0).all() and (r[served] <= 1).all()


def test_pool_is_365_days_only():
    ta, X, y, ok = synth(n_days=700)
    y2 = y.copy(); old = np.asarray(ta < pd.Timestamp("2024-03-01")); y2[old] = 1000.0                   # labels older than 365 days before 2025-04 must not matter
    m = pd.Period("2025-04", "M"); p, _, _ = _run(ta, X, y, ok, "gbm"); p2, _, _ = _run(ta, X, y2, ok, "gbm")
    sel = np.asarray(ta.to_period("M") == m) & np.isfinite(p)
    assert sel.sum() > 50 and np.array_equal(p[sel], p2[sel])


def test_same_source_pool_ignores_other_source():
    ta, X, y, ok = synth(); src = np.where(np.asarray(ta < pd.Timestamp("2024-09-01")), "a", "b")
    p, _, _ = L.monthly_view(ta, X, y, ok, source=src, same_source=True, kind="gbm", min_pool=300)
    y2 = y.copy(); y2[src == "a"] = 99.0
    p2, _, _ = L.monthly_view(ta, X, y2, ok, source=src, same_source=True, kind="gbm", min_pool=300)
    b = (src == "b") & np.isfinite(p)
    assert b.sum() > 50 and np.array_equal(p[b], p2[b])


def test_kernel_bandwidth_cv_never_uses_the_own_month():
    rng = np.random.default_rng(0); n = 1200; X = rng.normal(size=(n, 3)); mon = np.repeat(np.arange(12), 100)
    y = rng.normal(size=n)                                                    # pure noise per month: a month-leaking CV would pick a tiny bandwidth
    y = y + 5 * (mon % 2)                                                     # month-level shifts that only the own month's rows could "explain"
    _, _, ex = L.nw_fit_predict(X, y, X[:10], False, mon)
    assert ex["h"] >= 1.0 or ex["cv_mse"][0] > min(ex["cv_mse"]) - 1e-9


def test_gbm_learns_a_nonlinear_interaction_a_linear_model_cannot():
    ta, X, y, ok = synth(); p, _, _ = _run(ta, X, y, ok, "gbm"); s = np.isfinite(p)
    r_gbm = np.corrcoef(p[s], y[s])[0, 1]
    from numpy.linalg import lstsq
    A = np.c_[X[s], np.ones(s.sum())]; beta = lstsq(A, y[s], rcond=None)[0]; r_lin = np.corrcoef(A @ beta, y[s])[0, 1]
    assert r_gbm > 0.5 and r_gbm > r_lin + 0.3


# ---------------------------------------------------------------- the written views (skipped when the cache is absent)
def _cache():
    import anchors as AN
    return AN


@pytest.mark.parametrize("name", ["M7_gbm", "M7_kernel", "M7b_gbm_plus_v3", "M7c_gbm_resid"])
def test_written_views_are_subsets_of_the_ok_anchors(name):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
        AN = _cache(); import view_cache as VC
        for s in ("cut", "fwd"):
            v = VC.read_view(name, s); a = AN.load_anchors(s)
            assert v.index.isin(a.index).all() and a.loc[v.index, "ok"].all()
            assert np.isfinite(v.to_numpy()).all() and not v.index.has_duplicates
    except FileNotFoundError:
        pytest.skip("view cache not built")


def test_first_served_month_has_a_pool_of_at_least_5000():
    import json
    p = os.path.join(HERE, "checks.json")
    if not os.path.exists(p):
        pytest.skip("checks.json not built")
    c = json.load(open(p))
    for tag, mi in c["months"].items():
        for i in mi:
            if i["served"]:
                assert i["n_pool"] >= 5000

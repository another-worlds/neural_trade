"""M5 tests (seconds, CPU, no model): D:/nt/nt_mm_env/python.exe -m pytest -q test_m5.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN, view_cache as VC
import m5_lib as L


def _store():
    return AN.BarStore(AN.load_bars("cut").iloc[200000:215000], "cut")


def _mutated(store, ta, rng):
    s2 = AN.BarStore(store.m1.copy(), "cut"); s2.m1 = store.m1.copy(); sel = s2.m1.index >= ta
    s2.m1.loc[sel, ["open", "high", "low", "close"]] *= rng.uniform(0.5, 2.0, (int(sel.sum()), 1))
    s2.m1.loc[sel, "volume"] *= 7.0; s2._m5 = None; s2._rv5 = None
    return s2


def test_no_information_after_the_anchor_in_the_fed_windows():
    store = _store(); rng = np.random.default_rng(0)
    f0 = L.make_frame(store); hours = f0.index[f0.index.minute == 0]
    cands = hours[hours >= f0.index[0] + pd.Timedelta(minutes=5 * (L.CTX + 40))][:-3]
    n_checked = 0
    for ta in cands[rng.choice(len(cands), 20, replace=False)]:
        c0 = L.context(f0, ta); s2 = _mutated(store, ta, rng); c2 = L.context(L.make_frame(s2), ta)
        assert (c0 is None) == (c2 is None)
        if c0 is not None:
            assert c0[0][-1] < ta and c0[1].shape == (L.CTX, 2) and np.array_equal(c0[1], c2[1]); n_checked += 1
    assert n_checked >= 10


def test_seasonal_baseline_uses_only_the_past():
    store = _store(); rng = np.random.default_rng(1)
    f0 = L.make_frame(store); hours = f0.index[f0.index.minute == 0]
    cands = hours[hours >= f0.index[0] + pd.Timedelta(days=8)][:-3]
    for ta in cands[rng.choice(len(cands), 10, replace=False)]:
        s0 = L.seasonal7(store, pd.DatetimeIndex([ta])); s2 = L.seasonal7(_mutated(store, ta, rng), pd.DatetimeIndex([ta]))
        assert np.array_equal(s0, s2, equal_nan=True)


def test_views_are_anchor_subsets_without_nan_and_point_is_debiased_logsum():
    for name in ("M5_visionts", "M5b_visionts_seasonal"):
        for src in ("cut", "fwd"):
            if not os.path.exists(VC.view_path(name, src)):
                continue
            v = VC.read_view(name, src); anc = AN.load_anchors(src)
            assert len(v.index.difference(anc.index)) == 0 and not v.index.has_duplicates and not v.isna().any().any()
            if name == "M5_visionts":
                assert np.allclose(v["point_log_rv"], v["vts_logrv_logsum"])
            else:
                assert list(v.columns) == ["vts_seasonal_resid"]


def test_debias_constant_uses_only_training_anchors():
    ta = pd.date_range("2023-06-21", periods=24 * 500, freq="h"); n = len(ta)
    imp = np.zeros(n); y = np.ones(n); ok = np.ones(n, bool)
    c0, n0, span = L.debias_constant(ta, imp, y, ok)
    assert abs(c0 - 1.0) < 1e-12 and span[1] == str(ta[0] + pd.Timedelta(days=365))
    y2 = y.copy(); y2[ta >= ta[0] + pd.Timedelta(days=365)] = 99.0
    ok2 = ok.copy(); ok2[:100] = False; y3 = y.copy(); y3[:100] = 55.0
    assert L.debias_constant(ta, imp, y2, ok)[0] == c0 and L.debias_constant(ta, imp, y3, ok2)[0] == c0
    c_cap, n_cap, _ = L.debias_constant(ta, imp, np.where(ta < ta[0] + pd.Timedelta(days=30), 2.0, 9.0), ok, cap_before=ta[0] + pd.Timedelta(days=30))
    assert abs(c_cap - 2.0) < 1e-12 and n_cap == 24 * 30


def test_columns_and_hourly_aggregation():
    fc = np.full((2, 12), np.log(1e-6)); fv = np.full((2, 12), 3.0)
    col = L.to_columns(fc, fv, np.array([np.log(1e-6)] * 2), 0.5)
    assert np.allclose(col["vts_logrv_logsum"], np.log(12e-6) + 0.5, atol=1e-6) and np.allclose(col["vts_logrv_minus_last"], 0)
    assert np.allclose(col["vts_vol_mean12"], 3.0) and np.allclose(col["point_log_rv"], col["vts_logrv_logsum"])

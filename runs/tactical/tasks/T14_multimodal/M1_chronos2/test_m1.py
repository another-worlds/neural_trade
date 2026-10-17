"""M1 tests (seconds, CPU, no model): D:/nt/nt_mm_env/python.exe -m pytest -q test_m1.py   (needs pytest; otherwise run the functions)."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN, view_cache as VC
import inputs as IN


def _store():
    bars = AN.load_bars("cut").iloc[200000:207000]
    return AN.BarStore(bars, "cut")


def _same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def test_no_information_after_the_anchor():
    store = _store(); rng = np.random.default_rng(0)
    for har in (False, True):
        f0 = IN.make_frame(store, har=har)
        hours = f0.index[(f0.index.minute == 0)]
        cands = hours[hours >= f0.index[0] + pd.Timedelta(minutes=5 * (IN.CTX + 40))][:-3]
        sample = cands[rng.choice(len(cands), 20, replace=False)]
        for ta in sample:
            w0 = IN.window(f0, ta)
            s2 = AN.BarStore(store.m1.copy(), "cut"); s2.m1 = store.m1.copy()
            sel = s2.m1.index >= ta
            s2.m1.loc[sel, ["open", "high", "low", "close"]] *= rng.uniform(0.5, 2.0, (int(sel.sum()), 1))
            s2.m1.loc[sel, "volume"] *= 7.0; s2._m5 = None; s2._rv5 = None
            f2 = IN.make_frame(s2, har=har); w2 = IN.window(f2, ta)
            assert (w0 is None) == (w2 is None)
            if w0 is None:
                continue
            assert w0["idx"][-1] < ta and _same(w0["target"], w2["target"])
            for k in w0["past"]:
                assert _same(w0["past"][k], w2["past"][k]), k
            assert _same(w0["future"]["tod_sin"], w2["future"]["tod_sin"])


def test_history_ends_before_anchor_and_future_covariates_are_known_clock_values():
    f = IN.make_frame(_store()); ta = f.index[1000].floor("h") + pd.Timedelta(hours=1)
    w = IN.window(f, ta)
    assert w["idx"][-1] == ta - pd.Timedelta(minutes=5) and len(w["idx"]) == IN.CTX
    ph = 2 * np.pi * (ta.hour * 60 + ta.minute) / 1440.0
    assert np.isclose(w["future"]["tod_sin"][0], np.sin(ph)) and np.isclose(w["future"]["tod_cos"][0], np.cos(ph))


def test_hourly_aggregation_constant_bars():
    q = np.full((2, 12), np.log(1e-6)); h = IN.hourly_from_bars(q)
    assert np.allclose(h, np.log(12e-6 + 0 * 1e-12), atol=1e-6)


def test_written_views_are_anchor_subsets_without_nan():
    for name in ("M1_chronos2", "M1b_chronos2_har"):
        for src in ("cut", "fwd"):
            if not os.path.exists(VC.view_path(name, src)):
                continue
            v = VC.read_view(name, src); anc = AN.load_anchors(src)
            assert len(v.index.difference(anc.index)) == 0 and not v.index.has_duplicates
            assert not v.isna().any().any() and {"ch_q10", "ch_q50", "ch_q90", "ch_spread", "point_log_rv"} <= set(v.columns)
            assert (v["ch_q10"] <= v["ch_q50"] + 1e-6).all() and (v["ch_q50"] <= v["ch_q90"] + 1e-6).all()
            assert np.allclose(v["point_log_rv"], v["ch_q50"])

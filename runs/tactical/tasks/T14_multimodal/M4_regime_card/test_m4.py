"""M4 tests (seconds, synthetic data, CPU). Run: CUDA_VISIBLE_DEVICES=-1 D:/nt/nt_mm_env/python.exe -m pytest -q test_m4.py"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m4_lib as L  # noqa: E402


def synth(n=50 * 1440, seed=0):
    rng = np.random.default_rng(seed); idx = pd.date_range("2025-01-01", periods=n, freq="min")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    v = rng.gamma(2.0, 10.0, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def arrs(m1):
    return tuple(m1[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))


def img_at(m1, a, sig=1e-3 * 8):
    o, h, l, c, v = arrs(m1)
    out = [x for _, x in L.render_images(o, h, l, c, v, np.array([a]), np.array([sig]))]
    return out[0][0]


def test_image_uses_only_bars_before_the_anchor():
    m1 = synth(5000); a = 3000; base = img_at(m1, a)
    m2 = m1.copy(); rng = np.random.default_rng(5)
    for k in m2.columns:                                   # destroy every bar from the anchor on (the anchor's own bar included)
        m2.iloc[a:, m2.columns.get_loc(k)] = rng.uniform(1, 1000, len(m2) - a)
    assert np.array_equal(base, img_at(m2, a))
    m3 = m1.iloc[:a].copy()                                # and the series simply ends at the anchor
    assert np.array_equal(base, img_at(m3, a))
    assert not np.array_equal(base, img_at(m1, a + 1))     # sanity: a different anchor gives a different image


def test_image_shape_scale_free():
    m1 = synth(5000); a = 3000; im = img_at(m1, a)
    assert im.shape == (224, 224, 3) and im.dtype == np.uint8 and im.any()
    m2 = m1.copy(); m2[["open", "high", "low", "close"]] *= 37.0; m2["volume"] *= 5.0
    d = np.abs(im.astype(int) - img_at(m2, a).astype(int))
    assert (d > 0).mean() < 0.001                          # price and volume scale: the picture does not change (rounding of a few pixels at most)


def test_geometry_matches_t13_features():
    m1 = synth(5000); o, h, l, c, v = arrs(m1); a = np.array([2500, 3000, 4000]); sig = np.full(3, 8e-3)
    g = L.geometry(h, l, c, a, sig); X = L.S.sr_block(h, l, c, a, sig, 1, L.NB); nm = L.S.SR_NAMES; col = lambda k: X[:, nm.index(k)]
    Rn = g["lhm"] + g["sR"]; Sn = g["llm"] + g["sS"]
    assert np.allclose(np.clip((Rn - g["c0"]) / sig, -10, 10), col("s1_dR")) and np.allclose(np.clip((g["c0"] - Sn) / sig, -10, 10), col("s1_dS"))
    dR = np.where(g["missR"], 10, np.clip((g["pivR"] - g["c0"]) / sig, -10, 10)); assert np.allclose(dR, col("s2_dR"))
    fn = g["ybar"] + g["b"] * (L.NB - g["xm"]); assert np.allclose(np.clip((fn + 2 * g["s"] - g["c0"]) / sig, -10, 10), col("s3_dU"))


def caption_inputs(m1, ta, sigma=8e-3):
    o, h, l, c, v = arrs(m1); idx = m1.index; pos = idx.searchsorted(ta, side="left"); sg = np.full(len(ta), sigma)
    X = L.S.sr_block(h, l, c, pos, sg, 1, L.NB)
    cols = {k: X[:, L.S.SR_NAMES.index(k)] for k in ("s2_dR", "s2_dS", "s2_missR", "s2_missS")}
    return L.caption_frame(m1, ta, sg, cols)


def test_captions_use_only_past_information():
    m1 = synth(50 * 1440); ta = pd.date_range("2025-02-05", "2025-02-15", freq="h"); full = caption_inputs(m1, ta)
    t0 = ta[100]; m2 = m1.copy(); rng = np.random.default_rng(1); cut = m2.index.searchsorted(t0, side="left")
    for k in m2.columns:
        m2.iloc[cut:, m2.columns.get_loc(k)] = rng.uniform(1, 1000, len(m2) - cut)
    alt = caption_inputs(m2, ta)
    assert full.loc[:t0].equals(alt.loc[:t0]) is True or np.allclose(full.loc[:t0].to_numpy(float), alt.loc[:t0].to_numpy(float), equal_nan=True)
    assert [L.caption(r) for r in full.loc[:t0].itertuples()] == [L.caption(r) for r in alt.loc[:t0].itertuples()]
    assert not np.allclose(full.loc[ta[101]:].to_numpy(float), alt.loc[ta[101]:].to_numpy(float), equal_nan=True)     # sanity: later rows do change
    txt = L.caption(full.iloc[150]); assert txt and "UTC" in txt and "volatility" in txt and len(txt.split()) < 60


def test_pca_fitted_only_on_allowed_anchors():
    rng = np.random.default_rng(0); ta = pd.date_range("2023-06-01", "2024-12-31", freq="h"); X = rng.normal(size=(len(ta), 30)); ok = np.ones(len(ta), bool)
    mu, comps, evr, n = L.fit_pca(X, ta, ok)
    inside = (ta >= L.PCA_LO) & (ta <= L.PCA_HI); assert n == int(inside.sum())
    X2 = X.copy(); X2[~inside] = rng.normal(size=(int((~inside).sum()), 30)) * 100       # poison everything outside the allowed range
    mu2, comps2, _, n2 = L.fit_pca(X2, ta, ok)
    assert np.array_equal(mu, mu2) and np.array_equal(comps, comps2) and n == n2
    ok2 = ok.copy(); ok2[inside] = False; ok2[:5] = True
    try:
        L.fit_pca(X, ta, ok2); raise SystemError("expected the size assertion")
    except AssertionError:
        pass


def test_monthly_gbm_pool_ends_before_the_month():
    rng = np.random.default_rng(0); ta = pd.date_range("2024-01-01", "2025-03-01", freq="h"); X = rng.normal(size=(len(ta), 4)); y = X[:, 0] + rng.normal(size=len(ta))
    ok = np.ones(len(ta), bool); pred, info = L.monthly_gbm(ta, X, y, ok, min_pool=2000)
    per = np.asarray(ta.to_period("M"))
    first = min(i["month"] for i in info if i["served"]); assert first >= "2024-04"                 # 2000 pool rows need at least 83 days
    for i in info:
        if i["served"]:
            m0 = pd.Period(i["month"], "M").start_time
            assert i["n_pool"] <= int(((ta + pd.Timedelta(minutes=60) <= m0) & (ta >= m0 - pd.Timedelta(days=365))).sum())
    y2 = y.copy(); late = ta >= pd.Timestamp("2025-02-01"); y2[late] = rng.normal(size=int(late.sum())) * 50           # labels of the last month change ...
    pred2, _ = L.monthly_gbm(ta, X, y2, ok, min_pool=2000)
    assert np.allclose(pred[~late], pred2[~late], equal_nan=True)                                    # ... no earlier prediction moves


def test_written_views_are_subsets_of_the_anchors():
    import view_cache as VC
    import anchors as AN
    for s in ("cut", "fwd"):
        if not os.path.exists(VC.view_path("M4_img", s)):
            continue
        v = VC.read_view("M4_img", s); anc = AN.load_anchors(s).index
        assert v.index.isin(anc).all() and v.index.is_unique and v.shape[1] == 16 and np.isfinite(v.to_numpy()).all()

"""Tests for tdconfirm: cutoff, purge, the TD target uses only the frozen stage-1 net, QLIKE MIDAS, constants untouched."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_lib as C  # noqa: E402
import tdvar_lib as L  # noqa: E402
import tdvar_nn as N  # noqa: E402

REG = dict(l2=0.1, steps=50)


def synth(n=60000, seed=0, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    vol = np.exp(0.5 * np.cumsum(rng.normal(0, 0.02, n)) * 0.2 + 0.3 * np.sin(np.arange(n) / 3000.0)) * 1e-3
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    return close, pd.date_range(start, periods=n, freq="min")


def test_cutoff_no_bar_after_2025_07_20(tmp_path):
    idx = pd.date_range("2025-07-20 22:00", periods=400, freq="min")
    df = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"), "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": np.linspace(1, 2, 400), "volume": 1.0})
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    close, ti = L.load_bars(str(p))
    assert ti.max() == pd.Timestamp("2025-07-20 23:59") and len(close) == 120
    if os.path.exists(L.CSV):
        _, ti = L.load_bars()
        assert ti.max() < pd.Timestamp("2025-07-21")
        assert L.Data(*L.load_bars()).ta.max() < pd.Timestamp("2025-07-21")


def test_periods_end_before_cutoff_and_confirmation_precedes_tdvar():
    sys.path.insert(0, HERE)
    import run_confirm as R
    for lo, hi, end in R.PERIODS.values():
        assert pd.Timestamp(end) <= L.CUTOFF
    assert pd.Period(R.PERIODS["conf"][1]) < pd.Period("2019-01")           # tdvar ran 2019-01 onward


def test_training_labels_end_before_the_month_and_fit_ignores_later_bars():
    close, idx = synth()
    D = L.Data(close, idx)
    t0 = pd.Timestamp("2019-02-10"); t1 = t0 + pd.Timedelta(days=5)
    pos0 = int(idx.searchsorted(t0))
    for H in (240, 1440):
        a_tr, a_te, p0, lo = C.anchors(D, t0, t1, 20, H, D.n)
        assert p0 == pos0 and len(a_tr) > 20 and (a_tr + H - 1 < pos0).all() and (a_tr + H // 2 - 1 < pos0).all()
        assert (a_te >= pos0).all() and (a_tr >= lo).all()
    c2 = close.copy()
    c2[pos0:] *= 1.0 + np.random.default_rng(2).normal(0, 0.05, len(c2) - pos0)       # every bar from t0 on perturbed
    D2 = L.Data(c2, idx)
    f1 = C.Fit(D, t0, t1, 20, 240, D.n, REG, seeds=(0, 1))
    f2 = C.Fit(D2, t0, t1, 20, 240, D2.n, REG, seeds=(0, 1))
    p1, p2 = f1.params(), f2.params()
    assert p1.keys() == p2.keys()
    for k in p1:
        np.testing.assert_array_equal(p1[k], p2[k], err_msg=k)               # nets, bootstrap targets, MIDAS: bit-identical
    # while the test-side data do differ
    assert not np.allclose(D.rate(f1.a_te, 240), D2.rate(f2.a_te, 240))


def test_td_target_uses_only_the_frozen_stage1_net(monkeypatch):
    close, idx = synth()
    D = L.Data(close, idx)
    t0 = pd.Timestamp("2019-02-10"); t1 = t0 + pd.Timedelta(days=5)
    H, s = 240, 120
    seen = []
    orig = N.Net.fit

    def spy(self, z, y):
        seen.append((self, np.array(y, copy=True)))
        return orig(self, z, y)

    monkeypatch.setattr(N.Net, "fit", spy)
    f = C.Fit(D, t0, t1, 20, H, D.n, REG, seeds=(0,))
    # Net constructions per seed, in order: direct, direct_reg, stage1, TD head
    assert len(seen) == 4
    (nd, yd), (nr, yr), (ns, ys_fit), (nt, z_fit) = seen
    np.testing.assert_array_equal(yd, np.maximum(f.y_tr, f.y_floor))
    ys, zn = f.ys, N.zfeat(D.feat(f.a_tr + s), f.mu, f.sd)
    expected = (s * ys + (H - s) * ns.predict(zn)) / H                           # frozen stage-1 net, realised s-rate
    np.testing.assert_allclose(z_fit, np.maximum(expected, f.y_floor), rtol=1e-12)
    # the stage-1 net is exactly the one that was trained on rate_s (stage 2 did not move it), and the target is not rate_H
    np.testing.assert_array_equal(ns.W1, f.nets[0]["stage1"].W1)
    np.testing.assert_array_equal(ys_fit, np.maximum(ys, f.y_floor * s / H))
    assert not np.allclose(z_fit, np.maximum(f.y_tr, f.y_floor))
    # the target is a pure function of (net_s, s-rate, later state): perturbing rate_H leaves it unchanged
    np.testing.assert_allclose(C.td_target(ns, ys, zn, s, H), f.nets[0]["Z"], rtol=1e-12)


def test_regularised_fit_restores_the_base_constants():
    assert (N.L2, N.STEPS, N.HID, N.LR) == (1e-3, 200, 16, 0.01) == (C.BASE_L2, C.BASE_STEPS, 16, 0.01)
    with C.net_settings(1.0, 7):
        assert (N.L2, N.STEPS) == (1.0, 7)
    assert (N.L2, N.STEPS) == (C.BASE_L2, C.BASE_STEPS)
    try:
        with C.net_settings(1.0, 7):
            raise RuntimeError
    except RuntimeError:
        pass
    assert (N.L2, N.STEPS) == (C.BASE_L2, C.BASE_STEPS)
    close, idx = synth()
    D = L.Data(close, idx)
    t0 = pd.Timestamp("2019-02-10")
    C.Fit(D, t0, t0 + pd.Timedelta(days=5), 20, 240, D.n, REG, seeds=(0,))
    assert (N.L2, N.STEPS) == (C.BASE_L2, C.BASE_STEPS)


def test_midas_q_is_qlike_fitted_and_not_worse_than_sse_on_heavy_tails():
    rng = np.random.default_rng(3)
    n = 1500
    Bk = np.abs(rng.standard_t(3, (n, L.MIDAS_K))) * 0.5 + 0.5
    z = Bk @ L.exp_almon(-0.03, -1e-3)
    y = np.maximum(z * np.exp(rng.standard_t(3, n) * 0.6), 1e-6)                  # multiplicative heavy-tailed noise
    yf, ff = 1e-4 * np.median(y), 0.02 * np.median(y)
    th, c, q = C.fit_midas_q(Bk, y, yf, ff)
    assert c[0] >= 0 and c[1] >= 0 and q > 0
    th_s, c_s = L.fit_midas(Bk, y)
    q_sse = L.qlike(y, np.maximum(L.predict_midas(th_s, c_s, Bk), ff), yf, ff)
    q_q = L.qlike(y, np.maximum(C.predict_midas_q(th, c, Bk), ff), yf, ff)
    assert q_q <= q_sse + 1e-9
    assert q == pytest.approx(q_q, rel=1e-9)                                       # the stored train objective is the QLIKE


def test_cell_confirm_runs_and_has_every_model():
    close, idx = synth()
    D = L.Data(close, idx)
    t0 = pd.Timestamp("2019-02-10")
    out = C.cell_confirm(D, t0, t0 + pd.Timedelta(days=10), 20, 240, D.n, REG, seeds=(0, 1))
    assert out is not None
    rows = {r["model"]: r for r in out[0]}
    assert set(rows) == {"nn_td", "nn_direct", "nn_direct_reg", "midas_q", "har_direct", "har_iter", "midas_sse", "naive"}
    assert all(np.isfinite(r["qlike"]) and r["qlike"] > 0 for r in rows.values())
    assert len(rows["nn_td"]["qlike_seeds"]) == 2 and rows["nn_td"]["qlike"] == pytest.approx(np.mean(rows["nn_td"]["qlike_seeds"]))

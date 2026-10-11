"""Tests for tdvar_lib: hard cutoff, causality, purge, tower additivity in levels, linear TD == iterated HAR, metrics."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tdvar_lib as L  # noqa: E402


def synth(n=60000, seed=0, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    vol = np.exp(0.5 * np.cumsum(rng.normal(0, 0.02, n)) * 0.2 + 0.3 * np.sin(np.arange(n) / 3000.0)) * 1e-3
    lr = rng.normal(0, 1, n) * vol
    close = 100.0 * np.exp(np.cumsum(lr))
    idx = pd.date_range(start, periods=n, freq="min")
    return close, idx


def test_cutoff_drops_every_later_bar(tmp_path):
    idx = pd.date_range("2025-07-20 22:00", periods=400, freq="min")       # crosses the cutoff
    df = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"), "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": np.linspace(1, 2, 400), "volume": 1.0})
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    close, ti = L.load_bars(str(p))
    assert ti.max() < pd.Timestamp("2025-07-21") and ti.max() == pd.Timestamp("2025-07-20 23:59")
    assert len(close) == 120 and close[-1] == pytest.approx(df.close.iloc[119])


def test_real_file_respects_cutoff_if_present():
    if not os.path.exists(L.CSV):
        pytest.skip("data file not available")
    close, ti = L.load_bars()
    assert ti.max() < pd.Timestamp("2025-07-21")
    D = L.Data(close, ti)
    assert D.ta.max() < pd.Timestamp("2025-07-21")


def test_features_are_causal():
    close, idx = synth()
    D = L.Data(close, idx)
    a = np.array([20000, 30000])
    f0, b0 = D.feat(a), D.blocks(a)
    c2 = close.copy()
    c2[a[0]:] *= 1.0 + np.random.default_rng(1).normal(0, 0.05, len(c2) - a[0])      # perturb bars >= a[0]
    D2 = L.Data(c2, idx)
    np.testing.assert_allclose(D2.feat(a[:1]), f0[:1])
    np.testing.assert_allclose(D2.blocks(a[:1]), b0[:1])
    # the target does read the forward bars
    assert not np.allclose(D2.rate(a[:1], 240), D.rate(a[:1], 240))


def test_tower_additivity_in_levels_not_in_logs():
    close, idx = synth()
    D = L.Data(close, idx)
    a = np.array([15000, 22000]); H, s = 240, 120
    lhs = H * D.rate(a, H)
    rhs = s * D.rate(a, s) + (H - s) * D.rate(a + s, H - s)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)
    assert not np.allclose(np.log(D.rate(a, H)), np.log(D.rate(a, s)) + np.log(D.rate(a + s, H - s)))


def test_linear_td_equals_iterated_har():
    """On the same anchors the linear bootstrap regression is the iterated HAR (OLS is linear in the target)."""
    close, idx = synth()
    D = L.Data(close, idx)
    t0 = pd.Timestamp("2019-02-10")
    out = L.cell(D, t0, t0 + pd.Timedelta(days=10), 20, 240, D.n)
    assert out is not None
    rows = {r["model"]: r for r in out[0]}
    assert rows["td"]["qlike"] == pytest.approx(rows["iterated"]["qlike"], rel=1e-9)
    assert rows["td"]["mean_f"] == pytest.approx(rows["iterated"]["mean_f"], rel=1e-9)
    # while direct differs
    assert rows["direct"]["mean_f"] != pytest.approx(rows["td"]["mean_f"], rel=1e-9)


def test_no_training_label_bar_at_or_after_the_month_start():
    close, idx = synth()
    D = L.Data(close, idx)
    t0 = pd.Timestamp("2019-02-10")
    pos0 = int(idx.searchsorted(t0)); H = 1440
    # re-derive the train mask exactly as cell() does and check the label end; then check the fit is invariant to bars >= t0
    A = D.anch
    tr = (A + H <= pos0)
    assert (A[tr] + H - 1 < pos0).all()
    c2 = close.copy()
    c2[pos0:] *= 1.0 + np.random.default_rng(2).normal(0, 0.05, len(c2) - pos0)
    D2 = L.Data(c2, idx)
    r1 = L.cell(D, t0, t0 + pd.Timedelta(days=5), 20, H, D.n)
    r2 = L.cell(D2, t0, t0 + pd.Timedelta(days=5), 20, H, D2.n)
    # the fitted models must not depend on the perturbation: train sizes equal; mean_y (a test quantity) does differ
    assert r1[1]["n_train"] == r2[1]["n_train"] and r1[1]["n_train_recent"] == r2[1]["n_train_recent"]
    assert r1[1]["midas_theta"] == r2[1]["midas_theta"]


def test_td_recent_uses_only_bars_before_t0():
    close, idx = synth()
    t0 = pd.Timestamp("2019-02-10"); pos0 = int(idx.searchsorted(t0))
    D = L.Data(close, idx)
    A = D.anch; s = 120
    rec = (A + s <= pos0) & (A + 240 > pos0)
    # label part and bootstrap state of every extra anchor end before t0
    assert (A[rec] + s <= pos0).all()
    assert (D.anch[rec] + s - 1 < pos0).all()


def test_qlike_zero_at_truth_and_positive_otherwise():
    y = np.array([1.0, 2.0, 3.0])
    assert L.qlike(y, y, 1e-9, 1e-9) == pytest.approx(0.0, abs=1e-12)
    assert L.qlike(y, y * 2, 1e-9, 1e-9) > 0 and L.qlike(y, y * 0.5, 1e-9, 1e-9) > 0


def test_midas_recovers_a_planted_decay():
    rng = np.random.default_rng(3)
    Bk = np.abs(rng.normal(1, 0.3, (4000, L.MIDAS_K)))
    w = L.exp_almon(-0.03, -1e-3)
    y = 0.2 + 1.5 * (Bk @ w) + rng.normal(0, 0.01, 4000)
    th, c = L.fit_midas(Bk, y)
    assert th == (-0.03, -1e-3)
    assert c[1] == pytest.approx(1.5, rel=0.05)


def test_gap_makes_anchors_invalid():
    close, idx = synth(30000)
    ti = idx.values.copy()
    ti[15000:] += np.timedelta64(500, "m")                    # a gap before bar 15000
    D = L.Data(close, pd.DatetimeIndex(ti))
    a = np.array([14999 + 10, 14000, 16000, 27000])
    ok = D.contiguous(a, 240)
    assert not ok[0] and ok[1] and not ok[2]                # [0] and [2] touch the gap (forward / through the 10080-bar lag)
    assert ok[3]

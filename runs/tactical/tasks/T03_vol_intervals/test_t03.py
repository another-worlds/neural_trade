"""T03 tests (seconds): forbidden span, causality, calibration month strictly before the scored month, frozen coefficients."""
import copy, os, sys
import numpy as np
import pandas as pd
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t03_lib as L  # noqa: E402
T1 = L.T1


def _bars(start, n, seed=0):
    ts = pd.date_range(start, periods=n, freq="min"); rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + 0.5 * np.sin(np.arange(n) / 700))))
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)}, index=ts)


def _write(df, p):
    d = df.copy(); d.insert(0, "timestamp", d.index.astype(str)); d.to_csv(p, index=False)


def test_never_read_span(tmp_path):
    p = tmp_path / "x.csv"; _write(_bars("2025-07-19 00:00", 6000), p)
    s = T1.load_bars(str(p), T1.TRAIN_LOAD_FROM, T1.CUTOFF)
    assert s.index.max() == T1.CUTOFF and not ((s.index >= T1.FORBID_LO) & (s.index <= T1.FORBID_HI)).any()
    q = tmp_path / "y.csv"; _write(_bars("2025-09-29 00:00", 6000), q)
    assert T1.load_bars(str(q), T1.FWD_START, T1.FWD_LOAD_END).index.min() >= T1.FWD_START
    with pytest.raises(AssertionError):
        T1.load_bars(str(q), pd.Timestamp("2025-09-29 00:00"), T1.FWD_LOAD_END)
    assert T1.FORBID_LO == pd.Timestamp("2025-07-21") and T1.CUTOFF == pd.Timestamp("2025-07-20 23:59")


def test_features_and_scale_use_only_bars_before_the_anchor():
    s = _bars("2025-10-01 00:00", 60 * 24 * 8, seed=1); B = L.make_data(s, T1.WARM)
    k = len(B["D"].anch) // 2; a = int(B["D"].anch[k])
    s2 = s.copy(); s2.iloc[a:] = s2.iloc[a:] * np.random.default_rng(5).uniform(0.5, 2.0, (len(s2) - a, 1))
    s2["high"] = np.maximum(s2["high"], s2[["open", "close"]].max(1)); s2["low"] = np.minimum(s2["low"], s2[["open", "close"]].min(1))
    B2 = L.make_data(s2, T1.WARM)
    np.testing.assert_array_equal(B["D"].har[:k + 1], B2["D"].har[:k + 1])
    for H in L.HZ:
        np.testing.assert_array_equal(B["trail"][H][:k + 1], B2["trail"][H][:k + 1])
        assert not np.allclose(B["r"][H][k:], B2["r"][H][k:])                    # the targets did change: the check can see it


def test_calibration_month_strictly_before_scored_month():
    B = L.make_data(_bars("2025-10-01 00:00", 60 * 24 * 100, seed=2), T1.WARM)
    for m in pd.period_range("2025-11", "2026-01", freq="M"):
        for H in L.HZ:
            c = L.cal_rows(B, H, m); assert len(c) > 100
            ta = B["D"].ta[c]
            assert ta.min() >= (m - 1).start_time and ta.max() < m.start_time
            assert (ta + pd.Timedelta(minutes=H)).max() <= m.start_time          # even the label bars are before the scored month
            i, _ = L.month_rows(B, H, m); assert not set(c) & set(i)


def test_frozen_coefficients_identical_across_months():
    s = _bars("2025-05-01 00:00", 60 * 24 * 150, seed=3); sT = s[s.index <= T1.CUTOFF]
    BT = L.make_data(sT, L.FIRST_TRAIN); S1B = T1.build(sT, 1441)
    BF = L.make_data(_bars("2025-10-01", 60 * 24 * 100, seed=4), T1.WARM)
    fr = L.fit_frozen(BT, S1B); snap = copy.deepcopy(fr)
    for H in L.HZ:
        assert pd.Timestamp(fr[H]["rows_last_label_end"]) <= T1.CUTOFF + pd.Timedelta(minutes=1)
    for m in pd.period_range("2025-11", "2026-01", freq="M"):
        for H in L.HZ:
            assert len(L.score_month(fr, BF, H, m)) == 6

    def eq(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                eq(a[k], b[k])
        elif isinstance(a, tuple):
            for x, y in zip(a, b):
                eq(x, y)
        else:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    eq(snap, fr)


def test_interval_score_fz0_holm_basic():
    r = np.array([0.0, 0.1, -0.1]); q = {"lo": -0.05, "hi": 0.05, "q05": -0.05, "es": -0.08}
    d = L.score(r, np.ones(3), q)
    assert abs(d["cover"] - 1 / 3) < 1e-12 and abs(d["width_bps"] - 1000) < 1e-9
    assert abs(d["iscore_bps"] - 1e4 * (0.1 + 20 * 0.05 * 2 / 3)) < 1e-6
    assert np.allclose(L.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

"""T02 tests (SPEC): never-read span, past-only forecast, next-open fills, thresholds from the training span, null exposure."""
import os
import sys
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t02lib as L  # noqa: E402


def synth(n_days=420, seed=0, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days * 1440, freq="min")
    k = np.arange(len(idx))
    vol = 0.0004 * np.exp(0.5 * np.sin(k / 5000.0) + 0.3 * np.sin(k / 700.0))
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1, len(idx)) * vol))
    o = np.concatenate([[100.0], c[:-1]])
    return pd.DataFrame({"open": o, "close": c}, index=idx)


def test_never_read_span_forward_loader_rejects(tmp_path):
    bad = pd.DataFrame({"timestamp": ["2025-09-30 23:59:00", "2025-10-01 00:00:00"], "open": [1., 1.], "close": [1., 1.]})
    p = tmp_path / "f.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(AssertionError):
        L.load_fwd(str(p))


def test_never_read_span_real_files():
    dff, _ = L.load_fwd()
    assert dff.index.min() >= pd.Timestamp("2025-10-01")
    assert not ((dff.index >= L.NEVER_LO) & (dff.index <= L.NEVER_HI)).any()
    dfm, _ = L.load_main()
    assert dfm.index.max() <= pd.Timestamp("2025-07-20 23:59")


def test_forecast_uses_only_past_bars():
    df = synth()
    B = L.Bars(df)
    a = L.hour_anchors(B)
    a0 = a[len(a) // 2]
    df2 = df.copy()
    df2.iloc[a0:, :] = df2.iloc[a0:, :] * 3.0                      # change the future of a0 (bar a0 included)
    B2 = L.Bars(df2)
    np.testing.assert_allclose(L.har_feats(B, np.array([a0])), L.har_feats(B2, np.array([a0])))
    np.testing.assert_allclose(L.s1_feats(B, np.array([a0])), L.s1_feats(B2, np.array([a0])))


def test_thresholds_from_training_span_only():
    df = synth()
    B = L.Bars(df)
    m0 = pd.Timestamp("2019-12-01")
    lo, hi = np.datetime64(m0 - pd.Timedelta(days=365), "m"), np.datetime64(m0, "m")
    M1 = L.Models(B, lo, hi)
    df2 = df.copy()
    i0 = int(np.searchsorted(df.index.values, np.datetime64(m0)))
    df2.iloc[i0:, :] = df2.iloc[i0:, :] * np.linspace(1, 5, len(df2) - i0)[:, None]   # alter everything from the month on
    M2 = L.Models(L.Bars(df2), lo, hi)
    assert M1.thresholds() == pytest.approx(M2.thresholds())


def test_next_open_fills():
    df = synth(20)
    B = L.Bars(df)
    a = L.hour_anchors(B)[:50]
    f = L.hour_frame(B, pd.DataFrame({"pos": a}))
    for p, r in zip(a, f["ret"].values):
        assert r == pytest.approx(df["open"].iloc[p + 60] / df["open"].iloc[p] - 1)
    w = np.array([1.0, 0.0, 1.0])
    ret = np.array([0.01, 0.02, -0.01])
    r, turn = L.strat_returns(w, ret, 2)
    assert r[0] == pytest.approx(0.01) and r[1] == pytest.approx(-2e-4) and r[2] == pytest.approx(-0.01 - 2e-4)


def test_null_exposure_equals_mean_w():
    df = synth()
    B = L.Bars(df)
    sig = L.monthly_signals(B, "2019-12", "2020-01")
    f = L.hour_frame(B, sig)
    for cfg in L.CONFIGS:
        c, (w, r, rn, rb) = L.compare(f, cfg, 2)
        assert c["mean_w"] == pytest.approx(w.mean())
        np.testing.assert_allclose(rn, w.mean() * f["ret"].values)
        assert np.all((w >= 0) & (w <= 1))


def test_band_rule_changes_only_beyond_band():
    f = pd.DataFrame({"sig_med": np.ones(6), "sig_hat": np.array([1.0, 1.1, 1.2, 2.0, 2.1, 1.0])})
    w = L.exposure(f, ("ab", 1.0, 0.2))
    assert w[0] == 1.0 and w[1] == w[0] and w[2] == w[0]
    assert w[3] == pytest.approx(0.5)

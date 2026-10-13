"""T01 tests (seconds): the forbidden gap, training label end, causality of forward features, frozen coefficients. Synthetic data only
except the cut/forward header checks, which read a few rows."""
import copy, os, sys
import numpy as np
import pandas as pd
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t01_lib as T  # noqa: E402


def _bars(start, n, seed=0):
    ts = pd.date_range(start, periods=n, freq="min"); rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + 0.5 * np.sin(np.arange(n) / 700))))
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)}, index=ts)


def _write(df, p):
    d = df.copy(); d.insert(0, "timestamp", d.index.astype(str)); d.to_csv(p, index=False)


def test_no_bar_of_the_forbidden_gap_is_read(tmp_path):
    p = tmp_path / "x.csv"; _write(_bars("2025-07-19 00:00", 6000), p)           # spans the cutoff and 3 days into the gap
    s = T.load_bars(str(p), T.TRAIN_LOAD_FROM, T.CUTOFF)
    assert s.index.max() == T.CUTOFF and not ((s.index >= T.FORBID_LO) & (s.index <= T.FORBID_HI)).any()
    # a file with gap bars read through the forward bounds is refused or truncated: the forward window starts 2025-10-01
    q = tmp_path / "y.csv"; _write(_bars("2025-09-29 00:00", 6000), q)
    f = T.load_bars(str(q), T.FWD_START, T.FWD_LOAD_END)
    assert f.index.min() >= T.FWD_START
    with pytest.raises(AssertionError):                                           # asking for gap bars is an error
        T.load_bars(str(q), pd.Timestamp("2025-09-29 00:00"), T.FWD_LOAD_END)


def test_real_files_respect_the_bounds():
    for path, lo, hi in ((T.TRAIN_CSV, T.TRAIN_LOAD_FROM, T.CUTOFF), (T.FWD_CSV, T.FWD_START, T.FWD_START + pd.Timedelta(days=2))):
        if not os.path.exists(path):
            pytest.skip("data file absent")
        s = T.load_bars(path, lo, hi)
        assert lo <= s.index.min() and s.index.max() <= hi
    t = pd.read_csv(T.FWD_CSV, usecols=["timestamp"], nrows=1)["timestamp"][0]
    assert pd.Timestamp(t) >= T.FWD_START


def test_training_labels_end_at_or_before_the_cutoff():
    B = T.build(_bars("2025-05-01 00:00", 60 * 24 * 90), 1441)                   # 90 days ending 2025-07-30, i.e. past the cutoff
    m0 = T.CUTOFF + pd.Timedelta(minutes=1)
    for tg, H in T.TARGETS.items():
        i = T.train_idx(B, tg, m0)
        assert len(i) > 1000
        ends = B["D"].ta[i] + pd.Timedelta(minutes=H)
        assert ends.max() <= m0                                                  # the last label bar closes by the cutoff (00:00 = 23:59 + 1 min)
        assert B["D"].ta[i].min() >= m0 - pd.Timedelta(days=T.TRAIN_DAYS)


def test_forward_features_use_only_bars_before_the_anchor():
    s = _bars("2025-10-01 00:00", 60 * 24 * 8, seed=1); B = T.build(s, T.WARM)
    k = len(B["D"].anch) // 2; a = int(B["D"].anch[k])
    s2 = s.copy(); s2.iloc[a:] = s2.iloc[a:] * np.random.default_rng(5).uniform(0.5, 2.0, (len(s2) - a, 1))   # scramble bars >= a
    s2["high"] = np.maximum(s2["high"], s2[["open", "close"]].max(1)); s2["low"] = np.minimum(s2["low"], s2[["open", "close"]].min(1))
    B2 = T.build(s2, T.WARM)
    for nm in T.SETS:
        # S1/S2 are pure functions of bars < a; S3 adds the previous four hours' log RV (bars < a). The first 8 hours before a cannot change.
        np.testing.assert_array_equal(B["F"][nm][:k + 1][-1:], B2["F"][nm][:k + 1][-1:])
        np.testing.assert_array_equal(B["F"][nm][:k + 1], B2["F"][nm][:k + 1])
    np.testing.assert_array_equal(B["AR"]["A"][:k + 1], B2["AR"]["A"][:k + 1])
    assert not np.allclose(B["T"]["A"][k:], B2["T"]["A"][k:])                    # the targets (future) did change: the check can see a change


def test_warmup_skips_the_first_1620_bars():
    s = _bars("2025-10-01 00:00", 60 * 24 * 5); B = T.build(s, T.WARM)
    assert B["D"].anch.min() == T.WARM and B["D"].ta[0] >= pd.Timestamp("2025-10-01") + pd.Timedelta(minutes=T.WARM)


def test_frozen_arm_coefficients_do_not_change_across_months():
    s = _bars("2025-05-01 00:00", 60 * 24 * 150, seed=2); BT = T.build(s[s.index <= T.CUTOFF], 1441)
    BF = T.build(_bars("2025-10-01", 60 * 24 * 100, seed=3), T.WARM)
    tr = {tg: T.train_idx(BT, tg, T.CUTOFF + pd.Timedelta(minutes=1)) for tg in T.TARGETS}
    frozen = T.fit(BT, tr); snap = copy.deepcopy(frozen)
    for m in pd.period_range("2025-11", "2026-01", freq="M"):
        T.score_month(frozen, BF, m, "F")
        for tg in T.TARGETS:
            for k, v in snap[tg].items():
                cur = frozen[tg][k]
                if isinstance(v, tuple):
                    for a, b in zip(v, cur):
                        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
                else:
                    np.testing.assert_array_equal(v, cur)
    # the coefficients depend on the training rows only: a different forward file gives the same fit
    np.testing.assert_array_equal(T.fit(BT, tr)["A"]["S3"][2], snap["A"]["S3"][2])

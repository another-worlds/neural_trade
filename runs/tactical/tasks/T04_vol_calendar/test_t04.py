"""T04 tests (seconds): data bounds, slot dummies from timestamps only, frozen artefact reload, score_B on synthetic data."""
import json, os, sys
import numpy as np
import pandas as pd
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t04_lib as L  # noqa: E402
import score_B as S  # noqa: E402
T = L.T


def _bars(start, n, seed=0):
    ts = pd.date_range(start, periods=n, freq="min"); rng = np.random.default_rng(seed)
    vol = 1 + 0.6 * np.sin(2 * np.pi * (ts.hour * 60 + ts.minute).values / 1440)          # a daily vol pattern the slots can learn
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * vol * (1 + 0.5 * np.sin(np.arange(n) / 700))))
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)}, index=ts)


def _write(df, p):
    d = df.copy(); d.insert(0, "timestamp", d.index.astype(str)); d.to_csv(p, index=False)


def test_part_a_reads_nothing_after_2018_12_31(tmp_path):
    p = tmp_path / "a.csv"; _write(_bars("2018-12-30 00:00", 60 * 24 * 5), p)            # runs 3 days into 2019
    s = T.load_bars(str(p), L.A_LO, L.A_HI)
    assert s.index.max() == L.A_HI and s.index.min() >= L.A_LO


def test_real_part_a_file_bounds():
    if not os.path.exists(T.TRAIN_CSV):
        pytest.skip("data file absent")
    s = T.load_bars(T.TRAIN_CSV, L.A_LO, L.A_LO + pd.Timedelta(days=2))
    assert s.index.min() >= L.A_LO and s.index.max() <= L.A_LO + pd.Timedelta(days=2)


def test_no_bar_of_the_forbidden_gap_is_read(tmp_path):
    q = tmp_path / "y.csv"; _write(_bars("2025-09-29 00:00", 6000), q)
    with pytest.raises(AssertionError):
        T.load_bars(str(q), pd.Timestamp("2025-09-29 00:00"), pd.Timestamp("2025-10-05"))
    assert T.load_bars(str(q), T.FWD_START, pd.Timestamp("2025-10-05")).index.min() >= T.FWD_START


def test_freeze_loader_reads_no_bar_after_2026_09_30(tmp_path):
    n = int((L.B_HI - L.B_LO).total_seconds() // 60) + 1
    p = tmp_path / "f.csv"; _write(_bars(L.B_LO, n + 3 * 1440, seed=3), p)                # three extra days past the freeze end
    bars = L.load_freeze_bars(str(p))
    assert bars.index.max() == L.B_HI and bars.index.min() == L.B_LO and len(bars) == n


def test_slot_dummies_come_from_the_timestamp_only():
    ts = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 00:14", "2026-01-01 00:15", "2026-01-01 23:59", "2026-01-02 12:07"])
    assert list(L.slots(ts)) == [0, 0, 1, 95, 48]
    s = _bars("2025-10-01 00:00", 60 * 24 * 6, seed=1); B = L.build(s, T.WARM)
    s2 = s.copy(); s2[["open", "high", "low", "close"]] = s2[["open", "high", "low", "close"]].values[::-1].copy()       # scramble all prices
    B2 = L.build(s2, T.WARM)
    assert np.array_equal(B["dum"], B2["dum"]) and np.array_equal(B["slot"], B2["slot"])
    assert B["dum"].shape[1] == 96 and np.all(B["dum"].sum(1) == 1)
    assert np.array_equal(B["slot"], L.slots(B["D"].ta))


def _frozen(seed=0):
    # synthetic freeze: bars from B_LO (a few weeks); the artefact logic is the same, only shorter
    return L.freeze(_bars(L.B_LO, 60 * 24 * 21, seed=seed))


def test_frozen_artefact_reloads_to_identical_predictions(tmp_path):
    art = _frozen(); p = tmp_path / "frozen.json"; json.dump(art, open(p, "w"), indent=1)
    art2 = S.load_frozen(str(p)); assert art2["sha256"] == art["sha256"] == L.sha_of(art2)
    B = L.build(_bars(L.B_LO, 60 * 24 * 21, seed=0), T.WARM); idx = np.where(B["ok"])[0][:500]
    p1 = L.predict(art["models"], B, idx); p2 = L.predict(art2["models"], B, idx)
    for k in p1:
        assert np.array_equal(p1[k], p2[k])
    art2["models"]["S3C"]["intercept"] += 1e-9                                          # any change breaks the hash
    json.dump(art2, open(p, "w"))
    with pytest.raises(ValueError):
        S.load_frozen(str(p))


def test_freeze_label_end_is_before_the_freeze_end():
    art = _frozen(); assert pd.Timestamp(art["train_label_end_max"]) <= L.B_M0 and art["n_train_rows"] > 500


def test_score_B_on_synthetic_data_withholds_then_decides():
    art = _frozen(); bars = _bars(S.SCORE_FROM, 60 * 24 * 40, seed=9)                   # 2026-09-28 .. 2026-11-06: October and part of November
    assert S.complete_months(bars.index.max()) == [pd.Period("2026-10", freq="M")]
    r = S.score(art, bars); assert r["verdict"] is None and r["status"].startswith("withheld") and r["n_months"] == 1
    # three scored months: the logic of the verdict on stored rows
    bars3 = _bars(S.SCORE_FROM, 60 * 24 * 110, seed=10)
    r3 = S.score(art, bars3); assert r3["n_months"] == 3 and r3["verdict"] is not None
    assert r3["verdict"]["need_positive"] == 3 and set(r3["verdict"]) >= {"pass", "mean_diff_skill_ar", "ci", "months_positive"}
    for row in r3["rows"]:
        assert row["n_te"] > 1000 and set(row["skill_ar"]) == set(L.MODELS)
    # the verdict logic itself on controlled rows
    good = [{"skill_ar": {"S3": 0.10, "S3C": 0.13}, "mse": {"S3": 1.0, "S3C": 0.97}}] * 3
    assert L.verdict(good)["need_positive"] == 3
    rows = [{"skill_ar": {"S3": 0.10, "S3C": 0.10 + d}, "mse": {"S3": 1.0, "S3C": 1.0 - d}} for d in (0.03, 0.04, 0.05)]
    v = L.verdict(rows); assert v["pass"] and v["months_positive"] == 3
    rows[2] = {"skill_ar": {"S3": 0.10, "S3C": 0.09}, "mse": {"S3": 1.0, "S3C": 1.01}}
    assert not L.verdict(rows)["pass"]


def test_slot_calendar_is_learnable_on_synthetic_daily_pattern():
    s = _bars("2025-10-01 00:00", 60 * 24 * 40, seed=4); B = L.build(s, T.WARM)
    m0 = pd.Timestamp("2025-11-05"); tr = T.train_idx(B, "A", m0, start=pd.Timestamp("2025-10-02"))
    mod = L.fit(B, tr); r = L.score_month(mod, B, pd.Period("2025-11", freq="M"))
    assert r["mse"]["S3C"] <= r["mse"]["S3"] * 1.02 and r["n_te"] > 1000

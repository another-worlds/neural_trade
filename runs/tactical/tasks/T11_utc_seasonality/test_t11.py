"""T11 tests: calendar terms from timestamps only, the V1 / V2 / V3 designs, the seam exclusion, data bounds, the Pocock verdict logic, the frozen
artefact, and score_B on synthetic data. usage: python -m pytest -q runs/tactical/tasks/T11_utc_seasonality/test_t11.py"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t11_lib as L  # noqa: E402
import score_B as S  # noqa: E402
T = L.T; TF = L.TF


def _bars(start, n, seed=0, utc_pattern=True):
    ts = pd.date_range(start, periods=n, freq="min"); rng = np.random.default_rng(seed)
    slot = np.asarray(ts.hour * 60 + ts.minute) // 15; wk = np.asarray(ts.dayofweek >= 5)
    vol = 1 + 2.0 * ((slot >= 58) & (slot < 62)) * (~wk) if utc_pattern else np.ones(n)        # UTC 14:30-15:30 on weekdays is 3x as volatile
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * vol * (1 + 0.4 * np.sin(np.arange(n) / 900))))
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)}, index=ts)


def _write(df, p):
    d = df.copy(); d.insert(0, "timestamp", d.index.astype(str)); d.to_csv(p, index=False)


def test_calendar_comes_from_the_timestamps_only():
    s = _bars("2025-10-01 00:00", 60 * 24 * 6, seed=1); B = L.build(s, T.WARM, "binance")
    s2 = s.copy(); s2[["open", "high", "low", "close"]] = s2[["open", "high", "low", "close"]].values[::-1].copy()
    B2 = L.build(s2, T.WARM, "binance")
    for k in ("hr_sc", "ny", "slot"):
        assert np.array_equal(B["cal"][k], B2["cal"][k])
    assert B["cal"]["ny"].shape[1] == len(L.NY_NAMES) == 14 and B["cal"]["hr_sc"].shape[1] == 2
    assert np.array_equal(B["cal"]["slot"], L.calendar(B["D"].ta, "binance")["slot"])


def test_v1_equals_v0_on_a_utc_file_and_differs_on_a_file_time_file():
    s = _bars("2025-10-01 00:00", 60 * 24 * 5, seed=2); B = L.build(s, T.WARM, "binance"); idx = np.arange(0, len(B["D"].anch), 50)
    assert np.allclose(L.X_A(B, "V0", idx), L.X_A(B, "V1", idx))                       # a UTC file: the hour terms coincide
    s = _bars("2022-01-10 00:00", 60 * 24 * 5, seed=2); B = L.build(s, T.WARM, "bitstamp"); idx = np.arange(0, len(B["D"].anch), 50)
    X0, X1 = L.X_A(B, "V0", idx), L.X_A(B, "V1", idx)
    assert not np.allclose(X0[:, 4:6], X1[:, 4:6]) and np.allclose(np.delete(X0, [4, 5], 1), np.delete(X1, [4, 5], 1))
    hr = (B["D"].ta[idx] - pd.Timedelta(minutes=1)).hour.values - 5                     # January: file = UTC + 5
    assert np.allclose(X1[:, 4], np.sin(2 * np.pi * (hr % 24) / 24))
    assert X0.shape[1] == 16 and L.X_A(B, "V2", idx).shape[1] == 16 + 14 and L.X_A(B, "V3", idx).shape[1] == 16 + 96
    assert np.all(L.X_A(B, "V3", idx)[:, 16:].sum(1) == 1)


def test_new_york_terms_on_known_times():
    ta = pd.DatetimeIndex(["2026-01-14 14:30", "2026-07-15 13:30", "2026-01-14 21:00", "2026-07-15 19:29", "2026-01-17 14:30", "2026-01-14 14:46", "2026-01-14 14:47"])
    ny = L.calendar(ta, "binance")["ny"]; col = {n: i for i, n in enumerate(L.NY_NAMES)}
    assert list(ny[:, col["ny_open"]]) == [1, 1, 0, 0, 1, 1, 1]        # 09:30 NY in winter and in summer; Saturday 09:30 NY is still the clock window
    assert list(ny[:, col["ny_close"]]) == [0, 0, 1, 0, 0, 0, 0]       # 16:00 NY winter (21:00 UTC) is inside; 15:29 NY summer (19:29 UTC) is not
    assert list(ny[:, col["weekend"]]) == [0, 0, 0, 0, 1, 0, 0]
    assert list(ny[:, col["qhour"]]) == [1, 1, 1, 1, 1, 1, 0]          # minutes 30, 30, 0, 29, 30, 46, 47 (+-1 around 0/15/30/45)
    assert np.allclose(ny[4, [col["wk_ny_sin1"], col["wk_ny_cos1"]]], ny[4, [col["ny_sin1"], col["ny_cos1"]]]) and ny[0, col["wk_ny_sin1"]] == 0
    h = 9.5; assert np.isclose(ny[0, col["ny_sin1"]], np.sin(2 * np.pi * h / 24)) and np.isclose(ny[0, col["ny_cos2"]], np.cos(4 * np.pi * h / 24))
    assert np.isclose(ny[0, col["min_sin"]], np.sin(np.pi)) and np.isclose(ny[5, col["min_cos"]], np.cos(2 * np.pi * 46 / 60))
    assert list(L.calendar(ta, "binance")["slot"][:3]) == [58, 54, 84]


def test_seam_anchors_are_excluded_on_the_file_time_source_only():
    s = _bars("2018-03-08 00:00", 60 * 24 * 8, seed=3)
    B = L.build(s, T.WARM, "bitstamp"); ta = B["D"].ta
    day = np.asarray((ta >= "2018-03-11") & (ta < "2018-03-12"))
    assert not B["ok"][day].any() and B["drops"]["seam_excluded"] > 0
    assert B["ok"][np.asarray(ta < "2018-03-10 20:00")].mean() > 0.99
    Bu = L.build(s, T.WARM, "binance"); assert Bu["drops"]["seam_excluded"] == 0 and Bu["ok"][day].all()


def test_secondary_target_uses_only_the_label_bars():
    s = _bars("2025-10-01 00:00", 60 * 24 * 4, seed=4); B = L.build(s, T.WARM, "binance"); a = B["D"].anch; c = s["close"].values
    assert np.allclose(B["r15"], np.log(c[a + 14]) - np.log(c[a - 1]))


def test_training_labels_end_before_the_scored_month():
    s = _bars("2025-10-01 00:00", 60 * 24 * 60, seed=5); B = L.build(s, T.WARM, "binance"); m0 = pd.Timestamp("2025-11-15")
    tra, trr = L.train_rows(B, m0, start=pd.Timestamp("2025-10-02")); ta = B["D"].ta
    assert (ta[tra] + pd.Timedelta(minutes=60) <= m0).all() and (ta[trr] + pd.Timedelta(minutes=15) <= m0).all()
    assert ta[tra].min() >= pd.Timestamp("2025-10-02") and len(trr) >= len(tra)


def test_freeze_loader_reads_no_bar_after_the_freeze_end(tmp_path):
    n = int((L.FW_HI - L.FW_LO).total_seconds() // 60) + 1
    p = tmp_path / "f.csv"; _write(_bars(L.FW_LO, n + 3 * 1440, seed=6, utc_pattern=False), p)         # three extra days past the freeze end
    bars = L.load_freeze_bars(str(p))
    assert bars.index.max() == L.FW_HI and bars.index.min() == L.FW_LO and len(bars) == n


def test_no_bar_of_the_forbidden_gap_is_read(tmp_path):
    q = tmp_path / "y.csv"; _write(_bars("2025-09-29 00:00", 6000), q)
    with pytest.raises(AssertionError):
        T.load_bars(str(q), pd.Timestamp("2025-09-29 00:00"), pd.Timestamp("2025-10-05"))


def test_selection_script_reads_only_the_selection_span():
    assert L.SEL_HI < pd.Timestamp("2025-01-01") and L.SEL_LO >= pd.Timestamp("2017-12-01") and L.FW_HI == pd.Timestamp("2026-09-30 23:59")
    assert pd.Period(L.SEL_MONTHS[0], "M").start_time == pd.Timestamp("2019-01-01") and L.SEL_MONTHS[-1] == "2024-06" and len(L.SEL_MONTHS) == 66


def _fake_rows(diffs, dsp=0.01, dq=-0.01):
    rows = []
    for d in diffs:
        rows.append({"month": "m", "skill_ar": {"V0": 0.10, "V3": 0.10 + d}, "mse": {}, "sec": {"V0": {"spearman": 0.30, "qlike": 0.20},
                                                                                              "V3": {"spearman": 0.30 + dsp, "qlike": 0.20 + dq}}})
    return rows


def test_verdict_logic_pocock_and_month_counts():
    ok = L.verdict(_fake_rows([0.020, 0.025, 0.018]), "V3", 3)
    assert ok["pass_primary"] and ok["primary_A"]["pocock_ci"][0] > 0 and ok["secondary_R"]["pass"]
    # a mean below +0.01 fails even with all months positive
    assert not L.verdict(_fake_rows([0.004, 0.006, 0.005]), "V3", 3)["pass_primary"]
    # two of three positive is enough for the month count, one of three is not (mean still above 0.01 here)
    assert L.verdict(_fake_rows([0.030, 0.032, -0.001]), "V3", 3)["primary_A"]["c3_months"] is True
    assert L.verdict(_fake_rows([0.060, -0.002, -0.001]), "V3", 3)["primary_A"]["c3_months"] is False
    # an interval that includes 0 fails, six months need five positive
    assert not L.verdict(_fake_rows([0.06, -0.05, 0.05]), "V3", 3)["primary_A"]["c2_ci_lo_gt_0"]
    six = L.verdict(_fake_rows([0.02, 0.03, 0.02, 0.025, -0.001, 0.03]), "V3", 6)
    assert six["primary_A"]["need_positive"] == 5 and six["primary_A"]["positive_months"] == 5 and six["primary_A"]["c3_months"]
    assert not L.verdict(_fake_rows([0.02, 0.03, 0.02, -0.005, -0.001, 0.03]), "V3", 6)["primary_A"]["c3_months"]
    # the secondary: a worse QLIKE or a Spearman gain below +0.005 fails it
    assert not L.verdict(_fake_rows([0.02, 0.025, 0.018], dq=0.01), "V3", 3)["secondary_R"]["pass"]
    assert not L.verdict(_fake_rows([0.02, 0.025, 0.018], dsp=0.002), "V3", 3)["secondary_R"]["c1_spearman_ge_0.005"]
    # Pocock alpha 0.0294 two-sided: the interval is wider than the 95% one
    x = np.array([0.02, 0.03, 0.01, 0.04]); mu, lo, hi, _ = L.pocock_ci(x); mu95, lo95, hi95, _ = T.ci(x)
    assert mu == mu95 and lo < lo95 and hi > hi95


def test_frozen_artefact_and_score_B_on_synthetic_data(tmp_path):
    train = _bars("2025-10-01 00:00", 60 * 24 * 75, seed=7)                       # 75 days with a planted UTC 14:30-15:30 volatility burst
    art = L.freeze(train, "V3", "0" * 64, train_from=pd.Timestamp("2025-10-02"), m0=pd.Timestamp("2025-12-15"))
    assert art["variants"] == ["V0", "V1", "V3"] and art["chosen"] == "V3" and L.sha_of(art) == art["sha256"] and art["n_train_rows_A"] > 5000
    assert pd.Timestamp(art["train_label_end_max"]) <= pd.Timestamp("2025-12-15")
    p = tmp_path / "frozen_B.json"; json.dump(art, open(p, "w"))
    assert S.load_frozen(str(p), selection_path=str(tmp_path / "none.json"))["sha256"] == art["sha256"]
    bad = dict(art); bad["chosen"] = "V1"; json.dump(bad, open(tmp_path / "bad.json", "w"))
    with pytest.raises(ValueError):
        S.load_frozen(str(tmp_path / "bad.json"), selection_path=str(tmp_path / "none.json"))
    sel = tmp_path / "selection.json"; sel.write_text("{}")
    with pytest.raises(ValueError):                                                # selection.json must still be the file the freeze used
        S.load_frozen(str(p), selection_path=str(sel))
    before = json.dumps(art["models"], sort_keys=True)

    # complete months: the month needs the 240 bars after its last minute
    assert S.complete_months(pd.Timestamp("2026-11-01 03:58")) == [] and S.complete_months(pd.Timestamp("2026-11-01 03:59")) == [pd.Period("2026-10", "M")]
    # fewer than 3 complete months: withheld, nothing written
    short = _bars("2026-09-28 00:00", 60 * 24 * 70, seed=8)                        # to 2026-12-07: 2 complete months (10, 11)
    r = S.score(art, short); assert r["status"].startswith("withheld") and r["looks"] == {} and r["n_months"] == 2
    # 3 complete months: the first look only; 6: both looks; first scored anchor is 2026-10-02 00:00
    mid = _bars("2026-09-28 00:00", 60 * 24 * 106, seed=9); r3 = S.score(art, mid)
    assert r3["n_months"] == 3 and list(r3["looks"]) == ["3"] and r3["rows"][0]["month"] == "2026-10"
    B = L.build(mid, 1441, "binance"); te = np.asarray((B["D"].ta >= L.SCORE_START) & (B["D"].ta < pd.Timestamp("2026-11-01"))) & B["ok"]
    assert r3["rows"][0]["n_te"] == int(te.sum()) and B["D"].ta[te].min() >= L.SCORE_START
    full = _bars("2026-09-28 00:00", 60 * 24 * 195, seed=10); r6 = S.score(art, full)
    assert r6["n_months"] == 6 and sorted(r6["looks"]) == ["3", "6"] and len(r6["looks"]["6"]["months"]) == 6
    assert r6["looks"]["6"]["primary_A"]["mean_diff"] > 0                           # V3 learns the planted burst that V0's hour terms cannot
    assert json.dumps(art["models"], sort_keys=True) == before                      # scoring never refits or edits the frozen models
    out = tmp_path / "out"; out.mkdir(); csv = tmp_path / "fwd.csv"; _write(mid, csv)
    old = (S.HERE, S.load_frozen)
    try:
        sys.argv = ["score_B.py", "--csv", str(csv), "--out-dir", str(out)]
        S.load_frozen = lambda *a, **k: art
        S.main(); S.main()
    finally:
        S.load_frozen = old[1]
    assert sorted(os.listdir(out)) == ["resultsB_look3.json", "resultsB_look3_followup.json"]

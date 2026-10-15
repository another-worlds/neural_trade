"""T12 tests (seconds): the isotonic map uses only the previous month with ended labels, Look A reads nothing after 2018-12-31 23:59, the never-read span,
the frozen artefact reloads identically, score_B withholds with fewer than 3 complete months, H3c reduces to T10's H3 plus the stored map.
usage: python -m pytest -q runs/tactical/tasks/T12_barrier_calibrated/test_t12.py -p no:cacheprovider"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "T10_barrier_hit"))
import numpy as np
import pandas as pd
import pytest

import t12_lib as T12
from test_t10 import synth

L = T12.L
DATA_OK = os.path.exists(L.guard.CUT_CSV) and os.path.exists(L.guard.FWD_CSV)


def rec(month, T, n=40, seed=0):
    """A month record with anchors every T minutes from the month start (the last ones have labels past the month end)."""
    rng = np.random.default_rng(seed); m0 = month.start_time
    ta = pd.DatetimeIndex([m0 + pd.Timedelta(minutes=T * k) for k in range(n)]); ta = ta + (month.end_time.floor("min") - ta[-1] + pd.Timedelta(minutes=1 - T // 2 - T * 0))  # last row straddles the end
    ta = pd.DatetimeIndex(sorted(set(ta)))
    k = len(ta); p = rng.uniform(0.05, 0.95, k)
    return {"month": month, "ta": ta, "y": {X: (rng.uniform(size=k) < p).astype(bool) for X in T12.XS}, "p": {"B1": {X: p for X in T12.XS}, "H3": {X: p for X in T12.XS}}}


# ---------------------------------------------------------------- isotonic: previous month, ended labels only
def test_isotonic_uses_only_previous_month_with_ended_labels():
    T = 240; months = [pd.Period("2018-01", "M"), pd.Period("2018-02", "M"), pd.Period("2018-03", "M")]
    def build():
        rs = []
        for i, m in enumerate(months):
            m0 = m.start_time; ta = pd.DatetimeIndex([m0 + pd.Timedelta(minutes=30 + T * k) for k in range(int((m.end_time - m0) / pd.Timedelta(minutes=T)) + 1)])
            rng = np.random.default_rng(i); p = rng.uniform(0.05, 0.95, len(ta))
            rs.append({"month": m, "ta": ta, "y": {X: (rng.uniform(size=len(ta)) < p) for X in T12.XS}, "p": {"B1": {X: p for X in T12.XS}, "H3": {X: p.copy() for X in T12.XS}}})
        return rs
    base = build(); T12.add_h3c(base, T)
    assert base[0]["p"]["H3c"] is None                                     # nothing before the first month
    # perturb every row that must NOT matter for 2018-03: all of 2018-01, and the last anchors of 2018-02 whose label ends after 2018-03-01
    pert = build(); feb = pert[1]; m0 = months[2].start_time; late = np.asarray(feb["ta"] + pd.Timedelta(minutes=T) > m0)
    assert late.sum() >= 1 and (~late).sum() > 10
    for X in T12.XS:
        pert[0]["y"][X] = ~pert[0]["y"][X]; pert[0]["p"]["H3"][X] = 1 - pert[0]["p"]["H3"][X]
        feb["y"][X] = np.where(late, ~feb["y"][X], feb["y"][X]); feb["p"]["H3"][X] = np.where(late, 1 - feb["p"]["H3"][X], feb["p"]["H3"][X])
    T12.add_h3c(pert, T)
    for X in T12.XS:
        np.testing.assert_array_equal(pert[2]["p"]["H3c"][X], base[2]["p"]["H3c"][X])
    # sensitivity: changing an ended row of February does change March
    sens = build(); k = int(np.where(~late)[0][3])
    for X in T12.XS:
        sens[1]["y"][X][k] = ~sens[1]["y"][X][k]; sens[1]["p"]["H3"][X][k] = 0.5
    T12.add_h3c(sens, T)
    assert any(not np.array_equal(sens[2]["p"]["H3c"][X], base[2]["p"]["H3c"][X]) for X in T12.XS)


def test_calibration_rows_match_t03_rule_and_isotonic_is_monotone():
    T = 60; m = pd.Period("2018-05", "M"); ta = pd.DatetimeIndex([m.start_time - pd.Timedelta(minutes=120), m.start_time - pd.Timedelta(minutes=60), m.start_time - pd.Timedelta(minutes=30)])
    assert list(T12.calib_mask(ta, T, m)) == [True, True, False]           # the label of the last row ends after the month start
    rng = np.random.default_rng(3); p = rng.uniform(size=500); y = rng.uniform(size=500) < p
    xt, yt = T12.fit_iso(p, y); out = T12.apply_iso(xt, yt, np.linspace(-0.2, 1.2, 50))
    assert np.all(np.diff(out) >= -1e-12) and out.min() >= 0 and out.max() <= 1
    from sklearn.isotonic import IsotonicRegression
    ref = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y.astype(float)).predict(np.linspace(0, 1, 101))
    np.testing.assert_allclose(T12.apply_iso(xt, yt, np.linspace(0, 1, 101)), ref, atol=1e-12)


def test_train_days_floor():
    assert T12.train_days(pd.Timestamp("2017-09-01")) == 15.0               # everything since 2017-08-17
    assert T12.train_days(pd.Timestamp("2018-12-01")) == 365.0


def test_ece_boot_matches_point_estimate():
    rng = np.random.default_rng(1); p = rng.uniform(size=3000); y = rng.uniform(size=3000) < p ** 1.3; day = np.repeat(np.arange(300), 10)
    idx = np.tile(np.arange(300), (3, 1))                                    # a "draw" that takes every day once
    np.testing.assert_allclose(T12.ece_boot(p, y, day, idx), [L.ece(p, y.astype(float))] * 3, atol=1e-12)


# ---------------------------------------------------------------- data windows
def csv_with_poison(path, first, last_ok, extra=3):
    ts = pd.date_range(first, last_ok, freq="min"); n = len(ts)
    df = pd.DataFrame({"timestamp": ts.astype(str), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    with open(path, "w") as f:
        f.write(",".join(df.columns) + "\n"); df.to_csv(f, header=False, index=False)
        for k in range(extra):
            f.write(f"{pd.Timestamp(last_ok) + pd.Timedelta(minutes=k + 1)},POISON,x,y,z,w\n")                  # unparseable numbers after the cut
    return n


def test_look_a_reads_nothing_after_2018_12_31(tmp_path):
    p = str(tmp_path / "a.csv"); n = csv_with_poison(p, T12.CUT_START, T12.A_END)
    df = T12.load_lookA(p)
    assert len(df) == n and df.index.max() == T12.A_END and df.index.min() == T12.CUT_START
    assert df.dtypes.eq(float).all()                                         # the poison rows (strings) were never parsed


def test_look_a_cut_when_file_runs_on(tmp_path):
    p = str(tmp_path / "a2.csv"); ts = pd.date_range(T12.CUT_START, T12.A_END + pd.Timedelta(days=2), freq="min")
    pd.DataFrame({"timestamp": ts.astype(str), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_csv(p, index=False)
    assert T12.load_lookA(p).index.max() == T12.A_END


def test_look_a_gappy_file_still_cut(tmp_path):
    p = str(tmp_path / "a3.csv"); ts = pd.date_range(T12.CUT_START, T12.A_END + pd.Timedelta(days=2), freq="min"); ts = ts.delete(np.arange(5000, 5100))
    pd.DataFrame({"timestamp": ts.astype(str), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_csv(p, index=False)
    df = T12.load_lookA(p)
    assert df.index.max() <= T12.A_END and len(df) == len(ts[ts <= T12.A_END])


def test_freeze_loader_stops_at_2026_09_30_and_never_span(tmp_path):
    p = str(tmp_path / "f.csv"); n = csv_with_poison(p, T12.F_LO, T12.F_HI)
    df = T12.load_freeze(p)
    assert len(df) == n and df.index.max() == T12.F_HI
    with pytest.raises(AssertionError):
        T12.guard.check_index(pd.DatetimeIndex(["2025-08-01 00:00"]), "x")
    p2 = str(tmp_path / "bad.csv"); ts = pd.date_range("2025-09-30 23:00", periods=200, freq="min")
    pd.DataFrame({"timestamp": ts.astype(str), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_csv(p2, index=False)
    with pytest.raises(AssertionError):
        T12.load_upto(p2, pd.Timestamp("2025-09-30 23:00"), pd.Timestamp("2025-10-01 02:00"))      # a bar in 2025-07-21..2025-09-30 is refused


@pytest.mark.skipif(not DATA_OK, reason="data files absent")
def test_real_files_windows():
    a = T12.load_lookA(); assert a.index.max() == T12.A_END and a.index.min() == T12.CUT_START
    f = T12.load_freeze(); assert f.index.max() == T12.F_HI and f.index.min() == T12.F_LO


# ---------------------------------------------------------------- frozen artefact and score_B
@pytest.fixture(scope="module")
def frozen():
    bars = synth(n=60 * 1440, start="2026-08-01 00:00", seed=3)               # 60 days to 2026-09-29 23:59
    bars = bars[bars.index < pd.Timestamp("2026-10-01")]
    return T12.freeze(bars, m0=pd.Timestamp("2026-10-01"), days=365), bars


def test_frozen_artefact_reloads_identically(frozen, tmp_path):
    art, bars = frozen
    assert art["sha256"] == T12.sha_of(art) and art["last_bar"] < art["m0"]
    path = tmp_path / "frozen.json"; json.dump(art, open(path, "w"), indent=1); art2 = json.load(open(path))
    assert T12.sha_of(art2) == art["sha256"]
    P = L.prep(bars, "fwd")
    for T in T12.TS:
        B = L.build(P, T); idx = np.where(B["ok"])[0][-300:]
        p1 = T12.h3_probs(T12.models_from_art(art, T), B, T, idx); p2 = T12.h3_probs(T12.models_from_art(art2, T), B, T, idx)
        for X in T12.XS:
            np.testing.assert_array_equal(p1[X], p2[X])
        for X in T12.XS:                                                     # stored map reproduces itself
            m = art2["iso_first"][str(T)][str(X)]; assert len(m["x"]) == len(m["y"]) and np.all(np.diff(m["y"]) >= 0)
    art2["h3"]["60"]["50"]["ic"] += 1e-9
    assert T12.sha_of(art2) != art["sha256"]


def test_freeze_h3_equals_t10_predict_rows(frozen):
    art, bars = frozen; P = L.prep(bars, "fwd"); T = 60; B = L.build(P, T)
    models = L.fit_models(B, T, pd.Timestamp(art["m0"]), 365); idx = np.where(B["ok"])[0][-200:]; cidx = np.where(B["ok"])[0][-400:-200]
    ref = L.predict_rows(models, B, T, idx, cidx)["H3"]; got = T12.h3_probs(T12.models_from_art(art, T), B, T, idx)
    for X in T12.XS:
        np.testing.assert_allclose(got[X], ref[X], atol=1e-12)


def test_freeze_refuses_a_bar_at_or_after_the_cut():
    bars = synth(n=3000, start="2026-09-30 22:00", seed=1)
    with pytest.raises(AssertionError):
        T12.freeze(bars, m0=pd.Timestamp("2026-10-01"))                      # bars run to 2026-10-02 00:00


def test_score_withholds_with_fewer_than_three_months(frozen):
    art, _ = frozen
    bars = synth(n=int(75 * 1440), start="2026-09-28 00:00", seed=5)           # to 2026-12-11: Oct, Nov complete, Dec not
    assert [str(m) for m in T12.complete_months(bars.index.max())] == ["2026-10", "2026-11"]
    r = T12.score(art, bars); assert r["verdict"] is None and r["status"].startswith("withheld: 2 complete months")
    bars2 = bars.iloc[:int(2.9 * 30 * 1440)]
    assert T12.score(art, bars2)["verdict"] is None


def test_score_gives_a_verdict_with_three_months(frozen):
    art, _ = frozen
    bars = synth(n=int(104 * 1440), start="2026-09-28 00:00", seed=5)          # to 2027-01-09: Oct, Nov, Dec complete
    assert len(T12.complete_months(bars.index.max())) == 3
    import time; t0 = time.time(); r = T12.score(art, bars)
    assert r["status"] == "verdict" and r["verdict"] is not None and r["frozen_sha256"] == art["sha256"]
    assert {"c1_skill", "c2_ece", "c3_not_worse", "pass"} <= set(r["verdict"])
    c = r["cells"]["50|60"]["B1"]; assert c["n"] > 0 and r["cells"]["50|60"]["H3c"]["n"] == c["n"]
    first = pd.Timestamp("2026-10-02 00:00")
    # October is scored from 2026-10-02 (the SPEC), and H3c exists for every scored row
    assert time.time() - t0 < 300 and first.year == 2026


def test_score_b_script_refuses_a_tampered_artefact(tmp_path, frozen):
    import score_B
    art, _ = frozen; art = json.loads(json.dumps(art)); art["h3"]["60"]["50"]["ic"] += 1.0
    p = tmp_path / "t.json"; json.dump(art, open(p, "w"))
    with pytest.raises(ValueError):
        score_B.load_frozen(str(p))

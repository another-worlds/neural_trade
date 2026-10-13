"""T05 tests (seconds): never-read span, causality of features, quantiles/fits only from earlier months, ACI uses only ended labels,
selection.json committed before the looks, the frozen artefact reproduces predictions (and score_B runs on synthetic data), M0 = T03's HAR."""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t05_lib as T  # noqa: E402
import run_lookA as RA  # noqa: E402
import freeze_B as FB  # noqa: E402
import score_B as SB  # noqa: E402
L3 = T.L3; T1 = T.T1


def _bars(start, n, seed=0):
    ts = pd.date_range(start, periods=n, freq="min"); rng = np.random.default_rng(seed)
    vol = 1e-3 * np.exp(0.6 * np.sin(np.arange(n) / 3000.0) + 0.3 * np.sin(np.arange(n) / 500.0))
    c = 100 * np.exp(np.cumsum(rng.standard_t(5, n) * vol))
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)}, index=ts)


def _write(df, p):
    d = df.copy(); d.insert(0, "timestamp", d.index.astype(str)); d.to_csv(p, index=False)


def test_never_read_span_and_load_bounds(tmp_path):
    q = tmp_path / "y.csv"; _write(_bars("2025-07-15 00:00", 9 * 1440), q)
    with pytest.raises(AssertionError):                                   # a window overlapping 2025-07-21..2025-09-30 is refused
        T1.load_bars(str(q), pd.Timestamp("2025-07-15"), pd.Timestamp("2025-07-23"))
    for lo, hi in (T.SEL_LOAD, T.LOOKA_LOAD):
        assert hi < T1.FORBID_LO and lo < hi
    assert T.FREEZE_LOAD[0] > T1.FORBID_HI and T.FREEZE_LOAD[1] == pd.Timestamp("2026-09-30 23:59")
    with pytest.raises(AssertionError):                                   # the forward file cannot be loaded past September 2026 for the freeze
        T.load(T.FWD_CSV, (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-03")))
    s = T.load(str(q), (pd.Timestamp("2025-07-15"), pd.Timestamp("2025-07-20 23:59")))
    assert s.index.max() <= T1.CUTOFF


def test_features_use_only_bars_before_the_anchor():
    s = _bars("2020-01-01", 1441 + 60 * 24 * 6, seed=1); B = T.make_data(s)
    k = len(B["D"].anch) // 2; a = int(B["D"].anch[k])
    s2 = s.copy(); s2.iloc[a:] = s2.iloc[a:] * np.random.default_rng(5).uniform(0.5, 2.0, (len(s2) - a, 1))
    B2 = T.make_data(s2)
    for key in ("har", "m2", "m3"):
        np.testing.assert_array_equal(B["F"][key][:k + 1], B2["F"][key][:k + 1])
    np.testing.assert_array_equal(B["ewma"][:k + 1], B2["ewma"][:k + 1]); np.testing.assert_array_equal(B["trail"][T.H][:k + 1], B2["trail"][T.H][:k + 1])
    assert not np.array_equal(B["F"]["m2"][k + 1:], B2["F"]["m2"][k + 1:])             # the perturbation was real


def test_fits_and_quantiles_come_only_from_months_before_the_scored_one():
    s = _bars("2020-01-01", 70 * 1440, seed=2); B = T.make_data(s); mo = pd.Period("2020-03", "M"); m0 = mo.start_time
    tr = L3.train_rows(B, T.H, m0); cidx = L3.cal_rows(B, T.H, mo); idx, _ = L3.month_rows(B, T.H, mo); ta = B["D"].ta
    assert (ta[tr] + pd.Timedelta(minutes=T.H) <= m0).all() and (ta[cidx] + pd.Timedelta(minutes=T.H) <= m0).all()
    assert ta[cidx].min() >= (mo - 1).start_time and (ta[idx] >= m0).all()
    cut = np.searchsorted(s.index.values, np.datetime64(m0))                            # bars of the scored month onward are replaced
    s2 = s.copy(); s2.iloc[cut:] = s2.iloc[cut:] * np.random.default_rng(3).uniform(0.4, 2.5, (len(s2) - cut, 1))
    B2 = T.make_data(s2); tr2 = L3.train_rows(B2, T.H, m0); cidx2 = L3.cal_rows(B2, T.H, mo)
    np.testing.assert_array_equal(tr, tr2); np.testing.assert_array_equal(cidx, cidx2)
    for nm in ("M0", "M1", "M2", "M3", "M5"):
        p, p2 = T.fit_model(nm, B, tr), T.fit_model(nm, B2, tr2)
        assert json.dumps(T._jsonable(p), sort_keys=True) == json.dumps(T._jsonable(p2), sort_keys=True), nm
        assert json.dumps(T.cal_summary(nm, p, B, cidx), sort_keys=True) == json.dumps(T.cal_summary(nm, p2, B2, cidx2), sort_keys=True), nm


def test_aci_uses_only_anchors_whose_label_ended():
    rng = np.random.default_rng(0); n = 400; ta = pd.date_range("2020-01-01", periods=n, freq="7min"); sg = np.full(n, 1.0)
    zs = np.sort(rng.normal(size=3000)); r = rng.normal(size=n) * 0.9
    lo, hi = T.aci_bounds(zs, sg, r, ta)
    i = 300; need = int(np.ceil(T.H / 7))                                              # labels of rows j > i - need have not ended at row i
    r2 = r.copy(); r2[i - need + 1:] = rng.normal(size=n - (i - need + 1)) * 50
    lo2, hi2 = T.aci_bounds(zs, sg, r2, ta)
    np.testing.assert_array_equal(lo[:i + 1], lo2[:i + 1]); np.testing.assert_array_equal(hi[:i + 1], hi2[:i + 1])
    r3 = r.copy(); r3[:20] = 100.0                                                       # ended labels do move the later bounds
    lo3, hi3 = T.aci_bounds(zs, sg, r3, ta)
    assert hi3[200] > hi[200]
    assert (hi > lo).all()


def _git(cwd, *a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd)


def test_selection_json_committed_before_the_looks(tmp_path):
    d = str(tmp_path); _git(d, "init", "-q"); _git(d, "config", "user.email", "t@t"); _git(d, "config", "user.name", "t")
    with pytest.raises(AssertionError):
        RA.assert_selection_committed(d)                                               # missing
    open(os.path.join(d, "selection.json"), "w").write("{}")
    with pytest.raises(AssertionError):
        RA.assert_selection_committed(d)                                               # untracked
    _git(d, "add", "selection.json"); _git(d, "commit", "-qm", "x")
    assert len(RA.assert_selection_committed(d)) == 40
    open(os.path.join(d, "selection.json"), "w").write('{"a":1}')
    with pytest.raises(AssertionError):
        RA.assert_selection_committed(d)                                               # modified after the commit


def test_real_selection_json_is_tracked_when_it_exists():
    if os.path.exists(os.path.join(HERE, "selection.json")):
        assert _git(HERE, "ls-files", "--error-unmatch", "selection.json").returncode == 0


def test_generic_ridge_equals_t03_har_and_m4_cqr_covers():
    s = _bars("2020-01-01", 80 * 1440, seed=4); B = T.make_data(s); mo = pd.Period("2020-03", "M")
    tr = L3.train_rows(B, T.H, mo.start_time); har = L3.fit_har(B, T.H, tr, None, False); gen = T.fit_ridge(B["F"]["har"][tr], B["r"][T.H][tr])
    np.testing.assert_allclose(har["coef"], gen["coef"]); assert abs(har["c"] - gen["c"]) < 1e-12 and abs(har["ic"] - gen["ic"]) < 1e-12
    cidx = L3.cal_rows(B, T.H, mo); idx, _ = L3.month_rows(B, T.H, mo)
    for nm in ("M4", "M5"):
        p = T.fit_model(nm, B, tr); lo, hi = T.month_bounds(nm, p, B, idx, cidx); sc = T.score_bounds(B["r"][T.H][idx], lo, hi)
        assert 0.8 < sc["cover"] < 0.97 and (hi >= lo).all(), (nm, sc)


def test_m0_reproduces_t03_har_on_the_real_training_file():
    if not os.path.exists(T1.TRAIN_CSV):
        pytest.skip("training file not on this machine")
    ref = json.load(open(os.path.join(T.TAC, "tasks", "T03_vol_intervals", "results.json")))["summary"]
    BT = T.make_data(T1.load_train(), horizons=(15, 60, 240)); m0 = T1.CUTOFF + pd.Timedelta(minutes=1)
    rows = L3.train_rows(BT, 60, m0); p = T.fit_model("M0", BT, rows); q = L3.quantiles(BT["r"][60][rows] / L3.sigma(p, BT, rows, None))
    assert len(rows) == ref["train_rows"]["60"]
    for k in ("lo", "hi", "q05", "es"):
        assert q[k] == ref["frozen_q"]["60"]["HAR"][k]


def test_frozen_artefact_reloads_to_identical_predictions_and_score_b_runs(tmp_path):
    full = _bars("2020-01-01", 300 * 1440, seed=7); names = ["M0", "M1", "M2", "M5"]          # 2020-01-01 .. 2020-10-26
    fb = full[full.index <= "2020-06-30 23:59"]
    fr = FB.build_frozen(fb, names, "2020-06", {"chosen": "M2"}); FB.write_frozen(fr, str(tmp_path))
    fm = json.load(open(tmp_path / "frozen_B.json")); mo = pd.Period("2020-07", "M"); m1 = mo.start_time
    B = T.make_data(full[full.index <= "2020-07-31 23:59"]); idx, _ = L3.month_rows(B, T.H, mo); cidx = L3.cal_rows(B, T.H, mo)
    tr = L3.train_rows(B, T.H, m1, days=3650)
    for nm in names:                                                                    # the stored artefact == fitting again from the same bars
        direct = T.fit_model(nm, B, tr)
        a = T.month_bounds(nm, direct, B, idx, cidx); b = T.frozen_bounds(nm, fm, B, idx, cidx)
        np.testing.assert_allclose(a[0], b[0], rtol=1e-9, atol=1e-13); np.testing.assert_allclose(a[1], b[1], rtol=1e-9, atol=1e-13)
    Bf = T.make_data(fb); cf = L3.cal_rows(Bf, T.H, mo)
    for nm in names:                                                                    # the stored calibration is what the artefact's own bars give
        assert json.dumps(T.cal_summary(nm, fm["models"][nm], Bf, cf), sort_keys=True) == json.dumps(fm["cal"][nm], sort_keys=True), nm
    first = pd.Period("2020-07", "M"); fz = str(tmp_path / "frozen_B.json")
    csv = tmp_path / "fwd.csv"; _write(full[full.index <= "2020-09-26 23:59"], csv)         # July and August complete, September not
    r = SB.score(str(csv), fz, first=first)
    assert r["status"] == "waiting" and len(r["complete_months"]) == 2
    csv2 = tmp_path / "fwd2.csv"; _write(full[full.index <= "2020-10-26 23:59"], csv2)      # July, August, September complete
    r = SB.score(str(csv2), fz, first=first)
    assert r["status"] == "scored" and len(r["months"]) == 3 and set(r["summary"]) == set(names)
    assert 0.8 < r["summary"]["M0"]["pooled_cover"] < 0.97 and r["verdict"] is not None
    (tmp_path / "frozen_B.json").write_text((tmp_path / "frozen_B.json").read_text() + " ")
    with pytest.raises(AssertionError):
        SB.score(str(csv2), fz, first=first)                                             # a changed artefact is refused

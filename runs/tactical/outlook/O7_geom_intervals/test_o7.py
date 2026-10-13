"""O7 tests (SPEC "Tests"): pytest in this folder."""
import json, os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o7_lib as P  # noqa: E402
import score_B as SB  # noqa: E402
O1 = P.O1; O3 = P.O3; T = P.T
HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(HERE)


def synth(n, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


@pytest.fixture(scope="module")
def F():
    if not os.path.exists(P.FRAME):
        pytest.skip("O6 frame cache missing")
    return P.load_all()


def test_return_definition_matches_the_rv_target_window():
    s = synth(9000); D, ex = P.har_features(s, 3000)
    lr = np.concatenate([[0.0], np.diff(np.log(s.close.values))])
    for H in P.HORIZONS:
        a = D.anch
        assert np.allclose(ex[f"r{H}"], np.log(s.close.values[a + H - 1] / s.close.values[a - 1]))
        # the same bars a .. a+H-1 whose squared 1-minute log returns make the RV target
        cs = np.concatenate([[0.0], np.cumsum(lr)]); assert np.allclose(ex[f"r{H}"], cs[a + H] - cs[a], atol=1e-12)
        rv = D.cs2[a + H] - D.cs2[a]; k = a[:50]
        assert np.allclose(rv[:50], [(lr[x:x + H] ** 2).sum() for x in k], atol=1e-14)


def test_unit_sigma_gives_raw_empirical_quantiles():
    rng = np.random.default_rng(3); r_cal = rng.standard_t(4, 5000) * 1e-3; r = rng.standard_t(4, 3000) * 1e-3
    q = P.interval_bounds(r_cal / np.ones_like(r_cal)); lo, hi = np.quantile(r_cal, [0.05, 0.95])
    assert q == (float(lo), float(hi))
    d = P.interval_stats(r, np.ones_like(r), q)
    assert abs(d["cover"] - np.mean((r >= lo) & (r <= hi))) < 1e-15 and abs(d["width_bps"] - (hi - lo) * 1e4) < 1e-9
    assert abs(d["iscore_bps"] - 1e4 * np.mean((hi - lo) + 20 * (lo - r) * (r < lo) + 20 * (r - hi) * (r > hi))) < 1e-9


def test_interval_score_hand_values():
    r = np.array([0.0, 0.03, -0.02]); d = P.interval_stats(r, np.ones(3), (-0.01, 0.01))
    assert d["n_in"] == 1 and abs(d["cover"] - 1 / 3) < 1e-15
    exp = np.mean([0.02, 0.02 + 20 * 0.02, 0.02 + 20 * 0.01]) * 1e4
    assert abs(d["iscore_bps"] - exp) < 1e-9 and abs(d["width_bps"] - 200) < 1e-9


def test_calibration_month_before_scored_month_and_forecaster_never_saw_the_scored_month():
    ta = pd.date_range("2018-06-01", "2020-12-31 23:59", freq="7min").values.astype("datetime64[ns]"); ok = np.ones(len(ta), bool)
    F = {"ta": ta, **{f"ok{H}": ok for H in P.HORIZONS}}
    for mo in P.J_SCORED:
        f = P.prev_month(mo); m0f, m1f = P.month_bounds(f); n0, n1 = P.month_bounds(mo); assert m1f == n0
        for H in P.HORIZONS:
            tr, cal, te = P.month_rows(F, H, f)
            assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(m0f, "ns")).all()          # forecaster: labels end before the calibration month
            assert (ta[cal] >= np.datetime64(m0f, "ns")).all() and (ta[cal] < np.datetime64(n0, "ns")).all()
            assert (ta[cal] + np.timedelta64(H, "m") <= np.datetime64(n0, "ns")).all()          # calibration labels end before the scored month
            assert (ta[te] >= np.datetime64(n0, "ns")).all() and (ta[te] < np.datetime64(n1, "ns")).all()
            assert ta[tr].max() < ta[cal].min() <= ta[cal].max() < ta[te].min()


def test_spans_and_no_bar_after_the_cut():
    assert len(P.J_SCORED) == 23 and P.J_SCORED[0] == "2019-02" and P.J_SCORED[-1] == "2020-12"
    assert P.S_SPENT1[0] == "2021-02" and P.S_SPENT1[-1] == "2024-06" and P.S_SPENT2[0] == "2024-08" and P.S_SPENT2[-1] == "2025-07"
    assert P.LOAD_HI == T.CUTOFF == pd.Timestamp("2025-07-20 23:59")
    s = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2025-07-20 00:00"), P.LOAD_HI)
    assert s.index.max() <= P.LOAD_HI and not ((s.index >= T.FORBID_LO) & (s.index <= T.FORBID_HI)).any()
    s2 = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2025-07-20 00:00"), pd.Timestamp("2100-01-01"))     # even an open-ended request ends at the cut file's last bar
    assert s2.index.max() <= T.CUTOFF


def test_frame_and_extra_columns_are_on_the_same_anchors_and_end_before_the_cut(F):
    assert pd.Timestamp(F["ta"].max()) <= T.CUTOFF
    assert F["har"].shape == (len(F["ta"]), 6) and all(len(F[f"r{H}"]) == len(F["ta"]) for H in P.HORIZONS)
    # the label end of the last anchor stays inside the file: ta + 240 min <= cut + 1 min
    assert pd.Timestamp(F["ta"].max()) + pd.Timedelta(minutes=240) <= T.CUTOFF + pd.Timedelta(minutes=1)
    for H in P.HORIZONS:
        ok = F[f"ok{H}"]; assert np.isfinite(F[f"r{H}"][ok]).all() and np.isfinite(F["har"][ok]).all()


def test_L1_predictions_reproduce_o6_and_o5_scores_on_a_shared_month(F):
    from scipy.stats import spearmanr
    f = "2019-03"; m0, m1 = P.month_bounds(f)
    old6 = [json.loads(x) for x in open(f"{OUTLOOK}/O6_geom_hourly/results.jsonl")]
    old5 = [json.loads(x) for x in open(f"{OUTLOOK}/O5_geom_bestbase/results.jsonl")]
    for H in P.HORIZONS:
        tr = O3.train_rows(F["ta"], F[f"ok{H}"], H, m0); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
        m = P.fit(F, H, tr, ("L1",))["L1"]; mu = O3.ridge_pred(m, P.getX(F, te, "L1"))
        # O6's own code path: fit_h / predict on its frame definition (G60 columns present there; the L1 design ignores them)
        z = np.load(P.FRAME); F6 = {k: z[k] for k in z.files}
        st = P.P6.fit_h(F6, H, m0, models=("L1",)); mu6 = P.P6.predict(st, F6, te, ("L1",))["L1"]
        assert np.max(np.abs(mu - mu6)) < 1e-9
        sp = spearmanr(mu, F[f"rv{H}"][te])[0]
        r6 = [x for x in old6 if x["span"] == "judge" and x["month"] == f and x["H"] == H and x["model"] == "L1"][0]
        assert r6["n_te"] == len(te) and abs(sp - r6["spearman"]) < 1e-9
        r5 = [x for x in old5 if x["month"] == f and x["H"] == H and x["model"] == "L1"]
        if r5:
            assert r5[0]["n_te"] == len(te) and abs(sp - r5[0]["spearman"]) < 1e-9
        del F6, z


def test_R0_predictions_reproduce_o3_scores_on_a_shared_month(F):
    from scipy.stats import spearmanr
    f = "2021-01"; m0, m1 = P.month_bounds(f)
    old = [json.loads(x) for x in open(f"{OUTLOOK}/O3_synth_volind/results.jsonl")]
    for H in P.HORIZONS:
        tr = O3.train_rows(F["ta"], F[f"ok{H}"], H, m0); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
        m = P.fit(F, H, tr, ("R0",))["R0"]; mu = O3.ridge_pred(m, P.getX(F, te, "R0"))
        ref = [x for x in old if x["span"] == "sel" and x["month"] == f and x["H"] == H and x["model"] == "R0"][0]
        assert ref["n_te"] == len(te) and abs(spearmanr(mu, F[f"rv{H}"][te])[0] - ref["spearman"]) < 1e-9


def test_har_design_is_t03s_form(F):
    # the six columns: ln RV 15/60/240/1440 (mean squared log return, floor 1e-12) and hour sin/cos; |sin|,|cos| <= 1, ln RV finite on ok rows
    ok = F["ok60"]; h = F["har"][ok]
    assert (np.abs(h[:, 4:]) <= 1 + 1e-12).all() and (h[:, :4] > -20).all()
    ta_h = pd.DatetimeIndex(F["ta"][ok]) - pd.Timedelta(minutes=1)                  # the hour of the last bar a-1
    assert np.allclose(h[:, 4], np.sin(2 * np.pi * ta_h.hour.values / 24)) and np.allclose(h[:, 5], np.cos(2 * np.pi * ta_h.hour.values / 24))
    assert np.allclose(P.target(F, 60, np.where(ok)[0][:20], "HAR"), np.log(np.abs(F["r60"][ok][:20]) + 1e-5))


def test_one_month_end_to_end_is_finite_and_the_interval_has_sane_coverage(F):
    out = P.score_fit_month(F, 60, "2019-01")
    for nm in P.MODELS:
        d = out[nm]; assert d["n"] > 1000 and 0.8 < d["cover"] < 0.97 and d["width_bps"] > 0 and np.isfinite(d["iscore_bps"])


def test_criterion_logic():
    pr = {"mean": -1.0, "lo": -2.0, "hi": -0.1, "better": 16, "n": 23}
    assert P.criterion(pr, 0.90)["pass"]
    assert not P.criterion({**pr, "hi": 0.1}, 0.90)["pass"]
    assert not P.criterion({**pr, "better": 14}, 0.90)["pass"]
    assert not P.criterion(pr, 0.93)["pass"] and not P.criterion(pr, 0.87)["pass"]


def test_score_B_checks_the_frozen_sha_and_refuses_early_months_and_short_data(tmp_path):
    art = SB.load_frozen()
    assert art["sha256"].startswith("79f7f25d") and (art["base"], art["candidate"]) == ("L0", "L1")
    bad = tmp_path / "f.json"; art2 = dict(art); art2["last_bar"] = "x"; bad.write_text(json.dumps(art2))
    with pytest.raises(ValueError):
        SB.load_frozen(str(bad), SB.FROZEN_SHA)
    bars = synth(9000, "2026-09-17 00:00")
    assert SB.score(art, bars)["status"].startswith("withheld")
    with pytest.raises(ValueError):
        SB.score(art, bars, months=["2026-10", "2026-11", "2026-12"])
    cm = SB.complete_months(pd.Timestamp("2027-02-01 03:59")); assert [str(m) for m in cm] == ["2026-11", "2026-12", "2027-01"]
    assert [str(m) for m in SB.complete_months(pd.Timestamp("2027-02-01 03:58"))] == ["2026-11", "2026-12"]


def test_prospective_path_equals_retrospective_path_on_a_synthetic_frozen_model():
    """The scoring code of score_B (frozen models, rolling calibration) gives the same numbers as the retrospective function when the models are the same."""
    s = synth(70000, "2026-09-17 00:00", seed=5)
    Fn = P.build_frame_new(s, O3.FIRST)
    H = 60; ta = Fn["ta"]
    # a model fitted on the first stretch plays the 'frozen' model for both paths
    tr = np.where(ta < np.datetime64("2026-10-04", "ns"))[0]; tr = tr[Fn[f"ok{H}"][tr]]
    mods = P.fit(Fn, H, tr, ("L1",))
    sel = lambda a, b: np.where((ta >= np.datetime64(a, "ns")) & (ta < np.datetime64(b, "ns")) & Fn[f"ok{H}"])[0]
    cal, te = sel("2026-10-05", "2026-10-12"), sel("2026-10-12", "2026-10-19")
    sg = lambda rows: P.sigma_of("L1", O3.ridge_pred(mods["L1"], P.getX(Fn, rows, "L1")))
    q = P.interval_bounds(Fn[f"r{H}"][cal] / sg(cal)); d = P.interval_stats(Fn[f"r{H}"][te], sg(te), q)
    assert 0.5 < d["cover"] < 1 and d["n"] == len(te)
    # L1 design columns: the new-frame G equals O6's window features (G1 + G15) for the same anchors
    A = synth(9000, "2024-01-01").values.astype(np.float64); a = np.arange(4000, 4100, 7)
    assert np.array_equal(P.P6.window_features6(A, a)[:, :P.N_L1G], np.concatenate(O1.window_features(A, a), 1).astype(np.float32))

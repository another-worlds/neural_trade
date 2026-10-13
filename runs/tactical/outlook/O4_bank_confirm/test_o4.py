"""O4 tests (SPEC "Tests"): pytest in this folder; seconds."""
import json, os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o4_lib as O  # noqa: E402
import score_B as SB  # noqa: E402
L = O.L; T = O.T
HERE = os.path.dirname(os.path.abspath(__file__))


def synth(n, start, seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


def test_part_a_reads_no_bar_after_the_end_and_labels_end_before_the_month():
    assert O.A_LOAD_HI == pd.Timestamp("2019-01-01 03:59") and len(O.A_MONTHS) == 16
    s = T.load_bars(T.TRAIN_CSV, O.A_LO, pd.Timestamp("2017-11-01 03:59"))
    assert s.index.min() >= O.A_LO and s.index.max() <= pd.Timestamp("2017-11-01 03:59")
    F = L.build_features(s)
    for H in L.HORIZONS:
        st = O.fit_rb(F, H, pd.Timestamp("2017-10-01"))
        assert pd.Timestamp(st["last_label_end"]) <= pd.Timestamp("2017-10-01") and st["n_tr"] > 500
        assert pd.Timestamp(st["first_ta"]) >= O.A_LO + pd.Timedelta(minutes=L.FIRST)
    # the real Part A loader never goes past the label tail
    full = T.load_bars(T.TRAIN_CSV, O.A_LO, O.A_LOAD_HI)
    assert full.index.max() <= pd.Timestamp("2019-01-01 03:59")
    ta = pd.DatetimeIndex(L.build_features(full)["ta"])
    assert ta.max() < pd.Timestamp("2019-01-01 04:00")


def test_training_rows_for_every_part_a_month_end_before_the_month():
    ta = pd.date_range("2017-08-21", "2019-01-01", freq="7min").values.astype("datetime64[ns]"); ok = np.ones(len(ta), bool)
    for m in O.A_MONTHS:
        m0 = O.month_bounds(m)[0]
        for H in L.HORIZONS:
            tr = L.train_rows(ta, ok, H, m0)
            assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(m0, "ns")).all()


def test_freeze_reads_no_bar_after_2026_09_30():
    b = O.load_freeze_bars()
    assert b.index.min() == O.B_LO and b.index.max() <= pd.Timestamp("2026-09-30 23:59")
    with pytest.raises(AssertionError):
        O.freeze(synth(8000, "2026-09-30 20:00"))                 # bars past 2026-09-30 23:59 are refused
    art = json.load(open(f"{HERE}/frozen_B.json"))
    assert pd.Timestamp(art["last_bar"]) <= pd.Timestamp("2026-09-30 23:59")
    for h, st in art["states"].items():
        assert pd.Timestamp(st["last_label_end"]) <= O.B_M0 and pd.Timestamp(st["first_ta"]) >= pd.Timestamp("2025-10-05")


def test_frozen_artefact_reloads_to_identical_predictions_and_sha():
    s = synth(22000, "2026-09-01"); F = L.build_features(s); H = 60
    st = O.fit_rb(F, H, pd.Timestamp("2026-09-12")); te = L.test_rows(F["ta"], F[f"ok{H}"], pd.Timestamp("2026-09-12"), pd.Timestamp("2026-09-16"))
    assert len(te) > 100
    art = {"states": {"60": O.state_to_json(st)}}; art["sha256"] = O.sha_of(art)
    art2 = json.loads(json.dumps(art)); assert O.sha_of(art2) == art2["sha256"]
    st2 = O.state_from_json(art2["states"]["60"]); p1 = O.predict_rb(st, F, te); p2 = O.predict_rb(st2, F, te)
    for k in O.MODELS:
        assert np.array_equal(p1[k], p2[k])
    assert st["c"] == st2["c"]
    # the real artefact verifies and reloads
    real = json.load(open(f"{HERE}/frozen_B.json")); assert O.sha_of(real) == real["sha256"]
    assert real["sha256"] == open(f"{HERE}/frozen_B.sha256").read().strip()
    for h in ("15", "60", "240"):
        assert O.state_from_json(real["states"][h])["m"]["BANK"]["w"].shape == (16 + 36,)


def test_reproduces_o3_values_on_a_selection_month():
    s = T.load_bars(T.TRAIN_CSV, L.LOAD_FROM, pd.Timestamp("2021-02-01 03:59")); F = L.build_features(s)
    old = [json.loads(x) for x in open(f"{HERE}/../O3_synth_volind/results.jsonl")]
    m0, m1 = O.month_bounds("2021-01")
    for H in L.HORIZONS:
        st = O.fit_rb(F, H, m0); te = L.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
        for r in O.score_rb(st, F, te, "sel", "2021-01"):
            ref = [x for x in old if x["span"] == "sel" and x["month"] == "2021-01" and x["H"] == H and x["model"] == r["model"]][0]
            assert r["n_te"] == ref["n_te"]
            assert abs(r["spearman"] - ref["spearman"]) < 1e-9 and abs(r["qlike"] - ref["qlike"]) < 1e-9


def test_score_B_refuses_few_months_early_months_and_bad_sha():
    bars = synth(26 * 1440 + 10 * 1440, "2026-09-17")            # 2026-09-17 .. 2026-10-22: no complete month 2026-10
    art = {"states": {str(H): O.state_to_json(O.fit_rb(L.build_features(bars), H, O.B_M0)) for H in L.HORIZONS}}
    assert SB.complete_months(bars.index.max()) == []
    r = SB.score(art, bars); assert r["verdict"] is None and r["rows"] == [] and "withheld" in r["status"]
    two = synth(76 * 1440, "2026-09-17")                         # reaches 2026-12-01 (Nov complete, Dec not): months 10, 11 complete only
    assert [str(m) for m in SB.complete_months(two.index.max())] == ["2026-10", "2026-11"]
    r = SB.score(art, two); assert r["verdict"] is None and "need 3" in r["status"]
    with pytest.raises(ValueError):
        SB.score(art, two, months=["2026-09"])
    with pytest.raises(ValueError):
        SB.score(art, two, months=["2026-12"])
    bad = json.load(open(f"{HERE}/frozen_B.json")); bad["n_bars"] += 1
    p = os.path.join(HERE, "_bad_frozen_tmp.json"); json.dump(bad, open(p, "w"))
    try:
        with pytest.raises(ValueError):
            SB.load_frozen(p)
    finally:
        os.remove(p)
    three = synth(120 * 1440, "2026-09-17")                      # through mid-January: Oct, Nov, Dec complete
    r = SB.score(art, three); assert r["status"] == "verdict" and r["months"] == ["2026-10", "2026-11", "2026-12"]
    assert set(r["verdict"]) >= {"pass", "c3_months", "need_positive"} and r["verdict"]["need_positive"] == 3


def test_criterion_rules():
    ok = O.criterion({"mean": 0.006, "lo": 0.001, "pos": 11}, {"mean": -0.001}, 11); assert ok["pass"]
    assert not O.criterion({"mean": 0.004, "lo": 0.001, "pos": 16}, {"mean": -0.001}, 11)["pass"]
    assert not O.criterion({"mean": 0.006, "lo": -0.001, "pos": 16}, {"mean": -0.001}, 11)["pass"]
    assert not O.criterion({"mean": 0.006, "lo": 0.001, "pos": 10}, {"mean": -0.001}, 11)["pass"]
    assert not O.criterion({"mean": 0.006, "lo": 0.001, "pos": 16}, {"mean": 0.001}, 11)["pass"]

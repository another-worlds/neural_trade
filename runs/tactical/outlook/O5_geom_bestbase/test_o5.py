"""O5 tests (SPEC "Tests"): pytest in this folder."""
import json, os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o5_lib as P  # noqa: E402
import score_B as SB  # noqa: E402
O1 = P.O1; O3 = P.O3; T = P.T
HERE = os.path.dirname(os.path.abspath(__file__))
HKW = dict(max_iter=15, learning_rate=0.1, max_leaf_nodes=7, min_samples_leaf=20, early_stopping=False, random_state=0)


def synth(n, start, seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


def test_no_bar_after_the_cut_and_every_training_label_ends_before_its_month():
    assert P.LOAD_HI < T.CUTOFF and P.LOAD_HI == pd.Timestamp("2024-07-01 03:59")
    assert len(P.J_MONTHS) == 30 and P.J_MONTHS[0] == "2018-07" and P.J_MONTHS[-1] == "2020-12" and len(P.S_MONTHS) == 42
    assert pd.Timestamp(P.J_MONTHS[-1]).to_period("M").end_time < P.LOAD_HI
    ta = pd.date_range("2017-08-21", "2024-07-01", freq="7min").values.astype("datetime64[ns]"); ok = np.ones(len(ta), bool)
    for m in P.J_MONTHS + P.S_MONTHS:
        m0 = P.month_bounds(m)[0]
        for H in P.HORIZONS:
            tr = O3.train_rows(ta, ok, H, m0)
            assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(m0, "ns")).all()
    s = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2019-12-01"), pd.Timestamp("2019-12-03 03:59"))
    assert s.index.max() <= pd.Timestamp("2019-12-03 03:59") and not ((s.index >= T.FORBID_LO) & (s.index <= T.FORBID_HI)).any()
    with pytest.raises(AssertionError):                                  # the freeze refuses bars past 2026-09-30 23:59
        P.freeze(synth(8000, "2026-09-30 20:00"), "L0")
    b = P.load_freeze_bars(); assert b.index.min() == P.B_LO and b.index.max() <= P.B_HI


def test_anchor_grid_is_o3s_grid():
    off = (pd.Timestamp("2019-12-01") + pd.Timedelta(minutes=O3.FIRST) - (P.LOAD_FROM + pd.Timedelta(minutes=P.FIRST5))) / pd.Timedelta(minutes=1)
    assert off % T.STRIDE == 0 and P.FIRST5 >= O3.FIRST


def test_L0_and_S3_reproduce_o3_and_N0_path_reproduces_o1_hgb_h0():
    s = T.load_bars(T.TRAIN_CSV, O3.LOAD_FROM, pd.Timestamp("2021-02-01 03:59")); F = P.build_frame(s, O3.FIRST, with_g=False)
    old = [json.loads(x) for x in open(f"{HERE}/../O3_synth_volind/results.jsonl")]
    m0, m1 = P.month_bounds("2021-01")
    for H in P.HORIZONS:
        st = P.fit_h(F, H, m0, models=("S3", "L0")); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
        for r in P.score(st, F, te, "sel", "2021-01"):
            ref = [x for x in old if x["span"] == "sel" and x["month"] == "2021-01" and x["H"] == H and x["model"] == {"S3": "R0", "L0": "BANK"}[r["model"]]][0]
            assert r["n_te"] == ref["n_te"] and abs(r["spearman"] - ref["spearman"]) < 1e-9 and abs(r["qlike"] - ref["qlike"]) < 1e-9
    # the HGB path on S3 alone against O1's H0 (O1's own grid and cache; the boosting settings and the 1-in-10 striding are the same code path)
    p = os.path.join(O1.SCRATCH, "train.npz")
    if not os.path.exists(p):
        pytest.skip("O1 feature cache missing")
    z = np.load(p); cut = np.datetime64("2021-02-01", "ns"); ta = z["ta"].view("datetime64[ns]"); keep = ta < cut
    F1 = {"ta": ta[keep], "XB": z["S3"][keep].astype(np.float64), "G": None}
    for H in (60,):
        F1[f"rv{H}"] = z[f"rv{H}"][keep]; F1[f"ok{H}"] = z[f"ok{H}"][keep]
    P.SPECS["H0"] = ("hgb", P.N_S3, False)
    try:
        st = P.fit_h(F1, 60, m0, models=("H0",)); te = O3.test_rows(F1["ta"], F1["ok60"], m0, m1); r = P.score(st, F1, te, "sel", "2021-01")[0]
    finally:
        del P.SPECS["H0"]
    old1 = [json.loads(x) for x in open(f"{HERE}/../O1_geom_vol/results.jsonl")]
    ref = [x for x in old1 if x["span"] == "sel" and x["month"] == "2021-01" and x["H"] == 60 and x["model"] == "H0" and x["target"] == "lnrv"][0]
    assert r["n_te"] == ref["n_te"] and abs(r["spearman"] - ref["spearman"]) < 1e-9 and abs(r["qlike"] - ref["qlike"]) < 1e-9


def test_frozen_artefact_reloads_to_identical_predictions_and_sha():
    s = synth(60000, "2026-08-01"); F = P.build_frame(s, O3.FIRST); H = 60; m0 = pd.Timestamp("2026-09-08")
    st = P.fit_h(F, H, m0, models=("L0", "L1", "N0", "N1"), hgb_kw=HKW); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, pd.Timestamp("2026-09-12"))
    assert len(te) > 100
    art = {"states": {"60": P.state_to_json(st)}}; art["sha256"] = P.sha_of(art)
    art2 = json.loads(json.dumps(art)); assert P.sha_of(art2) == art2["sha256"]
    st2 = P.state_from_json(art2["states"]["60"]); p1 = P.predict(st, F, te); p2 = P.predict(st2, F, te)
    for k in p1:
        assert np.array_equal(p1[k], p2[k])
    assert st["c"] == st2["c"]
    if os.path.exists(f"{HERE}/frozen_B.json"):
        real = json.load(open(f"{HERE}/frozen_B.json")); assert P.sha_of(real) == real["sha256"]
        assert real["sha256"] == open(f"{HERE}/frozen_B.sha256").read().strip()
        assert pd.Timestamp(real["last_bar"]) <= P.B_HI
        for h, stt in real["states"].items():
            assert pd.Timestamp(stt["last_label_end"]) <= P.B_M0 and pd.Timestamp(stt["first_ta"]) >= pd.Timestamp("2025-10-05")
            S = P.state_from_json(stt); assert set(S["m"]) == {real["base"], real["candidate"]}


def test_score_B_refuses_few_months_early_months_and_bad_sha():
    bars = synth(36 * 1440, "2026-09-17"); F = P.build_frame(bars, O3.FIRST)
    art = {"base": "L0", "candidate": "L1", "states": {str(H): P.state_to_json(P.fit_h(F, H, P.B_M0, models=("L0", "L1"))) for H in P.HORIZONS}}
    assert SB.complete_months(bars.index.max()) == []
    r = SB.score(art, bars); assert r["verdict"] is None and r["rows"] == [] and "withheld" in r["status"]
    two = synth(76 * 1440, "2026-09-17")
    assert [str(m) for m in SB.complete_months(two.index.max())] == ["2026-10", "2026-11"]
    r = SB.score(art, two); assert r["verdict"] is None and "need 3" in r["status"]
    with pytest.raises(ValueError):
        SB.score(art, two, months=["2026-09"])
    with pytest.raises(ValueError):
        SB.score(art, two, months=["2026-12"])
    if os.path.exists(f"{HERE}/frozen_B.json"):
        bad = json.load(open(f"{HERE}/frozen_B.json")); bad["n_bars"] += 1
        p = os.path.join(HERE, "_bad_frozen_tmp.json"); json.dump(bad, open(p, "w"))
        try:
            with pytest.raises(ValueError):
                SB.load_frozen(p)
        finally:
            os.remove(p)
    three = synth(120 * 1440, "2026-09-17")
    r = SB.score(art, three); assert r["status"] == "verdict" and r["months"] == ["2026-10", "2026-11", "2026-12"]
    assert r["verdict"]["need_positive"] == 3 and set(r["verdict"]) >= {"pass", "c3_months"}


def test_criterion_rules_and_holm():
    ok = P.criterion({"mean": 0.004, "lo": 0.001, "pos": 20}, {"mean": -0.001}, 20); assert ok["pass"]
    assert not P.criterion({"mean": 0.0029, "lo": 0.001, "pos": 25}, {"mean": -0.001})["pass"]
    assert not P.criterion({"mean": 0.004, "lo": -0.001, "pos": 25}, {"mean": -0.001})["pass"]
    assert not P.criterion({"mean": 0.004, "lo": 0.001, "pos": 19}, {"mean": -0.001})["pass"]
    assert not P.criterion({"mean": 0.004, "lo": 0.001, "pos": 25}, {"mean": 0.001})["pass"]
    assert P.holm([0.01, 0.04]) == [0.02, 0.04] and P.holm([0.04, 0.03]) == [0.06, 0.06]

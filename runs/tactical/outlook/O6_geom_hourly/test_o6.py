"""O6 tests (SPEC "Tests"): pytest in this folder."""
import json, os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o6_lib as P  # noqa: E402
import score_B as SB  # noqa: E402
O1 = P.O1; O3 = P.O3; T = P.T
HERE = os.path.dirname(os.path.abspath(__file__))


def synth(n, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


def test_g60_is_causal_and_last_hourly_group_ends_at_a_minus_1():
    A = synth(12000).values.astype(np.float64)
    for a in (3700, 6001, 9000):
        g = P.window_features6(A, [a]); A2 = A.copy(); A2[a:] = np.random.default_rng(1).uniform(1, 1000, A2[a:].shape)
        assert np.array_equal(g, P.window_features6(A2, [a])) and g.shape[1] == 3 * 164
        w = P.aggm(A, [a], 60)[0]
        assert w[-1, 0] == A[a - 60, 0] and w[-1, 3] == A[a - 1, 3] and w[-1, 1] == A[a - 60:a, 1].max() and w[-1, 2] == A[a - 60:a, 2].min()
        assert w[0, 0] == A[a - 3600, 0]
    a = 6001; A3 = A.copy(); A3[a - 1, 3] *= 1.01          # a change inside the last group changes G60 ...
    assert not np.array_equal(P.window_features6(A, [a])[:, 328:], P.window_features6(A3, [a])[:, 328:])
    A4 = A.copy(); A4[a - 3601] *= 1.5                      # ... one before the window does not
    assert np.array_equal(P.window_features6(A, [a])[:, 328:], P.window_features6(A4, [a])[:, 328:])


def test_hourly_aggregation_equals_pandas_resample_and_agg15_is_reproduced():
    s = synth(9000, "2024-01-01 00:00"); A = s.values.astype(np.float64)
    a = 7200                                                # 2024-01-06 00:00, hour aligned
    assert s.index[a].minute == 0
    w = P.aggm(A, [a], 60)[0]; seg = s.iloc[a - 3600:a]
    r = seg.resample("60min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    assert len(r) == 60 and np.allclose(w, r.values, rtol=1e-12, atol=0)
    anc = [4000, 4007, 6500]
    assert np.array_equal(P.aggm(A, anc, 15), O1.agg15(A, anc))


def test_no_bar_after_the_cut_and_training_labels_end_before_each_month():
    assert P.LOAD_HI == T.CUTOFF == pd.Timestamp("2025-07-20 23:59")
    assert len(P.J_MONTHS) == 24 and P.J_MONTHS[0] == "2019-01" and P.J_MONTHS[-1] == "2020-12"
    assert P.S_SPENT1[0] == "2021-01" and P.S_SPENT1[-1] == "2024-06" and P.S_SPENT2[0] == "2024-07" and P.S_SPENT2[-1] == "2025-07"
    ta = pd.date_range("2017-08-21", "2025-07-20", freq="7min").values.astype("datetime64[ns]"); ok = np.ones(len(ta), bool)
    for m in P.J_MONTHS + P.S_SPENT1 + P.S_SPENT2:
        m0 = P.month_bounds(m)[0]
        for H in P.HORIZONS:
            tr = O3.train_rows(ta, ok, H, m0)
            assert (ta[tr] + np.timedelta64(H, "m") <= np.datetime64(m0, "ns")).all()
    s = T.load_bars(T.TRAIN_CSV, pd.Timestamp("2025-07-19"), P.LOAD_HI)
    assert s.index.max() <= T.CUTOFF and not ((s.index >= T.FORBID_LO) & (s.index <= T.FORBID_HI)).any()
    assert P.FIRST6 >= 3601
    with pytest.raises(AssertionError):
        P.freeze(synth(8000, "2026-09-30 20:00"))
    b = P.P5.load_freeze_bars(); assert b.index.min() == P.B_LO and b.index.max() <= P.B_HI
    p = os.path.join(P.SCRATCH, "frame.npz")
    if os.path.exists(p):
        z = np.load(p); assert pd.Timestamp(z["ta"].max()) <= T.CUTOFF


def test_L1_reproduces_o5_L1_on_judgement_months():
    p = os.path.join(P.SCRATCH, "frame.npz")
    if not os.path.exists(p):
        pytest.skip("frame cache missing")
    z = np.load(p); F = {k: z[k] for k in ("ta", "XB", "G", "rv15", "ok15", "rv60", "ok60", "rv240", "ok240")}
    old = [json.loads(x) for x in open(f"{HERE}/../O5_geom_bestbase/results.jsonl")]
    for mo in ("2019-03", "2020-06"):
        m0, m1 = P.month_bounds(mo)
        for H in P.HORIZONS:
            st = P.fit_h(F, H, m0, models=("L1",)); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
            r = P.score(st, F, te, "judge", mo, models=("L1",))[0]
            ref = [x for x in old if x["span"] == "judge" and x["month"] == mo and x["H"] == H and x["model"] == "L1"][0]
            assert r["n_te"] == ref["n_te"] and r["n_tr"] == ref["n_tr"]
            assert abs(r["spearman"] - ref["spearman"]) < 1e-9 and abs(r["qlike"] - ref["qlike"]) < 1e-9


def test_L2sh_permutes_only_g60_independently_and_L1_ignores_g60():
    F = P.build_frame6(synth(40000), 6145); rows = np.arange(0, 500)
    X1 = P.getX(F, rows, "L1"); X2 = P.getX(F, rows, "L2"); pm = P.perm_for(len(rows), 60, 1); Xs = P.getX(F, rows, "L2sh", pm)
    assert X1.shape[1] == 52 + 328 and X2.shape[1] == 52 + 492
    assert np.array_equal(X2[:, :380], Xs[:, :380]) and np.array_equal(X2[:, :380], X1)
    assert np.array_equal(Xs[:, 380:], X2[pm, 380:]) and not np.array_equal(Xs[:, 380:], X2[:, 380:])
    assert not np.array_equal(P.perm_for(500, 60, 1), P.perm_for(500, 60, 2))


def test_frozen_artefact_reloads_to_identical_predictions_and_sha():
    s = synth(60000, "2026-08-01"); F = P.build_frame6(s, O3.FIRST); H = 60; m0 = pd.Timestamp("2026-09-08")
    st = P.fit_h(F, H, m0, models=("L1", "L2")); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, pd.Timestamp("2026-09-12"))
    assert len(te) > 100
    art = {"states": {"60": P.state_to_json(st)}}; art["sha256"] = P.sha_of(art)
    art2 = json.loads(json.dumps(art)); assert P.sha_of(art2) == art2["sha256"]
    st2 = P.state_from_json(art2["states"]["60"]); p1 = P.predict(st, F, te); p2 = P.predict(st2, F, te)
    for k in p1:
        assert np.array_equal(p1[k], p2[k])
    if os.path.exists(f"{HERE}/frozen_B.json"):
        real = json.load(open(f"{HERE}/frozen_B.json")); assert P.sha_of(real) == real["sha256"]
        assert real["sha256"] == open(f"{HERE}/frozen_B.sha256").read().strip()
        assert pd.Timestamp(real["last_bar"]) <= P.B_HI
        for h, stt in real["states"].items():
            assert pd.Timestamp(stt["last_label_end"]) <= P.B_M0 and pd.Timestamp(stt["first_ta"]) >= pd.Timestamp("2025-10-05")
            assert set(P.state_from_json(stt)["m"]) == {"L1", "L2"}


def test_score_B_refuses_few_months_early_months_and_bad_sha():
    bars = synth(36 * 1440, "2026-09-17"); F = P.build_frame6(bars, O3.FIRST)
    art = {"base": "L1", "candidate": "L2", "states": {str(H): P.state_to_json(P.fit_h(F, H, P.B_M0, models=("L1", "L2"))) for H in P.HORIZONS}}
    assert SB.complete_months(bars.index.max()) == []
    r = SB.score(art, bars); assert r["verdict"] is None and r["rows"] == [] and "withheld" in r["status"]
    two = synth(76 * 1440, "2026-09-17")
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
    assert r["verdict"]["need_positive"] == 3


def test_criterion_rules_and_holm():
    assert P.criterion({"mean": 0.004, "lo": 0.001, "pos": 16}, {"mean": -0.001}, 16)["pass"]
    assert not P.criterion({"mean": 0.0029, "lo": 0.001, "pos": 20}, {"mean": -0.001}, 16)["pass"]
    assert not P.criterion({"mean": 0.004, "lo": -0.001, "pos": 20}, {"mean": -0.001}, 16)["pass"]
    assert not P.criterion({"mean": 0.004, "lo": 0.001, "pos": 15}, {"mean": -0.001}, 16)["pass"]
    assert not P.criterion({"mean": 0.004, "lo": 0.001, "pos": 20}, {"mean": 0.001}, 16)["pass"]
    assert P.holm([0.01, 0.04]) == [0.02, 0.04]

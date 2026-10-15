"""O11 part d tests (SPEC "Tests"): causality of G15 and tb7 at decisions, labels end before each month, the 90-day window bounds, base equals O9's p_up,
decision labels, metrics, the choice rule and the choice gate, the frozen artefact. pytest in this folder."""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o11d_lib as D  # noqa: E402
import run_d as R  # noqa: E402
import score_P_d as SD  # noqa: E402
L9 = D.L9; L8 = D.L8; P5 = D.P5; O3 = D.O3; T = D.T; P2 = D.P2


def synth(n, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


def synth_train(n=61000, seed=0):
    """A stride-7 training frame with real tb7 / ret60 and a random (but fixed) G15 block, shaped as D.train_data expects."""
    s = synth(n, seed=seed); first = 6145
    Fr = P5.build_frame(s, first, with_g=False); TF = {k: Fr[k] for k in ("ta", "XB", "rv15", "rv60", "ok15", "ok60")}
    X = {**L9.train_extras(s, TF["ta"]), "ta": TF["ta"], "S3": TF["XB"][:, :L9.N_S3].copy(), "ok60": TF["ok60"]}
    X["pok"] = np.isfinite(X["ret60"]) & (np.abs(X["ret60"]) > L9.DEAD) & np.isfinite(X["tb7"]).all(1) & TF["ok60"]
    rng = np.random.default_rng(seed + 1); TF["G"] = rng.normal(size=(len(TF["ta"]), 2 * D.NG)).astype(np.float32)
    return s, TF, X, D.train_data(TF, X)


# ------------------------------------------------------------------ causality
def test_g15_and_tb7_at_a_decision_use_only_bars_before_it_end_to_end():
    s = synth(61000); first = 6145
    DFc = L8.build_dframe(s, first)
    DF = {"G": DFc["G"], "ok": DFc["ok"], "a": DFc["a"], "ta": DFc["ta"]}
    DX = {"tb7": L9.tb7_at(s.values.astype(float), DF["a"])}
    rows = np.where(DF["ok"])[0]; r = rows[len(rows) // 2]; a = int(DF["a"][r])
    ref_g = DF["G"][r, D.G15].copy(); ref_t = DX["tb7"][r].copy()
    s2 = s.copy(); s2.iloc[a:] = np.random.default_rng(1).uniform(1, 1000, s2.iloc[a:].shape)       # every bar at or after the decision replaced
    D2 = L8.build_dframe(s2, first); r2 = int(np.where(D2["a"] == a)[0][0])
    assert np.array_equal(D2["G"][r2, D.G15], ref_g) and np.array_equal(L9.tb7_at(s2.values.astype(float), np.array([a]))[0], ref_t)
    s3 = s.copy(); s3.iloc[a - 1, :4] *= 1.01                                                           # the last closed bar does matter
    D3 = L8.build_dframe(s3, first); r3 = int(np.where(D3["a"] == a)[0][0])
    assert not np.array_equal(D3["G"][r3, D.G15], ref_g)
    assert D.G15 == slice(164, 328) and DF["G"].shape[1] == 328


def test_decision_labels_equal_trains_ret60_and_are_nan_across_a_gap():
    s = synth(5000); a = np.array([100, 2500]); ex = L9.train_extras(s, (s.index[a - 1] + pd.Timedelta(minutes=1)).values.astype("datetime64[ns]"))
    assert np.array_equal(D.dec_ret60(s, a), ex["ret60"]) and D.dec_ret60(s, a)[0] == pytest.approx(s["open"].values[160] / s["open"].values[100] - 1)
    g = pd.concat([s.iloc[:1000], s.iloc[1010:]])
    r = D.dec_ret60(g, np.array([950, 1100])); assert np.isnan(r[0]) and np.isfinite(r[1])


def test_valid_rows_need_ok_finite_features_and_a_nonzero_finite_label():
    s = synth(3000); a = np.array([100, 200, 300, 400])
    DF = {"a": a, "ta": s.index.values[a].astype("datetime64[ns]"), "ok": np.array([True, False, True, True]), "G": np.zeros((4, 328), np.float32)}
    s2 = s.copy(); s2.iloc[460, 0] = s2.iloc[400, 0]                                                   # open[a + 60] == open[a] for the last row -> label 0
    DF["G"][2, 170] = np.nan; DX = {"tb7": np.ones((4, 13), np.float32)}
    v = D.decision_data(s2, DF, DX)["valid"]
    assert v.tolist() == [True, False, False, False]


# ------------------------------------------------------------------ fits: labels before the month, windows
@pytest.mark.parametrize("var", list(D.VARIANTS))
def test_fit_labels_end_before_the_month_and_the_feature_count_matches_the_variant(var):
    s, TF, X, TX = synth_train()
    st = D.fit_variant(TX, var, pd.Timestamp("2024-02-03 00:00"))
    assert st["last_label_end"] <= pd.Timestamp("2024-02-03 00:00")
    assert st["lr"].coef_.shape[1] == (13 + D.NG if D.VARIANTS[var][0] else 13)


def test_coefficients_ignore_rows_after_the_cut_exactly():
    s, TF, X, TX = synth_train()
    m0 = pd.Timestamp("2024-02-03 00:00"); late = TX["ta"] > np.datetime64(m0 - pd.Timedelta(minutes=D.H), "ns")
    for var in D.VARIANTS:
        st = D.fit_variant(TX, var, m0)
        TX2 = dict(TX); TX2["ret60"] = TX["ret60"].copy(); TX2["ret60"][late] = 0.5; TX2["tb7"] = TX["tb7"].copy(); TX2["tb7"][late] = 7.0
        TX2["G15"] = TX["G15"].copy(); TX2["G15"][late] = 7.0                                          # pok (the mask) is kept: the cut excludes these rows
        assert np.array_equal(st["lr"].coef_, D.fit_variant(TX2, var, m0)["lr"].coef_), var


def test_window_bounds_90_and_365_days_and_base_is_o9_pup():
    s, TF, X, TX = synth_train(200000)
    m0 = pd.Timestamp("2024-05-05 00:00")
    for var, (ug, days) in D.VARIANTS.items():
        st = D.fit_variant(TX, var, m0)
        assert st["first_ta"] >= m0 - pd.Timedelta(days=days) and st["last_label_end"] <= m0
        ok = TX["pok"] & TX["gok"] if ug else TX["pok"]
        tr = O3.train_rows(TX["ta"], ok, D.H, m0, days=days); assert st["n_tr"] == len(tr)
        lo = np.datetime64(m0 - pd.Timedelta(days=days), "ns"); hi = np.datetime64(m0 - pd.Timedelta(minutes=60), "ns")
        assert len(tr) == int(((TX["ta"] >= lo) & (TX["ta"] <= hi) & ok).sum())                       # the window is [m0 - days, m0 - 60 min]
    assert D.fit_variant(TX, "d2", m0)["n_tr"] < D.fit_variant(TX, "base", m0)["n_tr"]
    assert D.fit_variant(TX, "d2", m0)["first_ta"] > m0 - pd.Timedelta(days=91) and D.fit_variant(TX, "base", m0)["first_ta"] < m0 - pd.Timedelta(days=100)
    sp = L9.pup_fit(X, m0); sb = D.fit_variant(TX, "base", m0)
    assert np.array_equal(sp["lr"].coef_, sb["lr"].coef_) and sp["n_tr"] == sb["n_tr"] and sp["base"] == pytest.approx(sb["base"])
    assert (np.abs(TX["ret60"][O3.train_rows(TX["ta"], TX["pok"], D.H, m0)]) > 5e-4).all()             # the dead band is applied in training only


def test_g15_helps_a_fit_where_the_label_depends_on_it_and_prediction_matches_the_json():
    s, TF, X, TX = synth_train()
    TX["ret60"] = np.where(TX["G15"][:, 3] > 0, 1e-3, -1e-3); TX["pok"] = np.ones(len(TX["ta"]), bool)
    m0 = pd.Timestamp("2024-03-01 00:00"); sg = D.fit_variant(TX, "d1", m0); sb = D.fit_variant(TX, "base", m0)
    rows = np.arange(0, len(TX["ta"]), 50); y = (TX["ret60"][rows] > 0).astype(int)
    assert D.auc(y, D.predict_variant(sg, TX["tb7"][rows], TX["G15"][rows])) > 0.95 > D.auc(y, D.predict_variant(sb, TX["tb7"][rows], TX["G15"][rows])) + 0.2
    j = json.loads(json.dumps(D.st_to_json(sg)))
    assert np.allclose(D.predict_j(j, TX["tb7"][rows], TX["G15"][rows]), D.predict_variant(sg, TX["tb7"][rows], TX["G15"][rows]), atol=1e-12)


def test_predict_months_walk_forward_and_frozen_use_the_expected_cut():
    s, TF, X, TX = synth_train()
    DFc = L8.build_dframe(s, 6145); DF = {"G": DFc["G"], "ok": DFc["ok"], "a": DFc["a"], "ta": DFc["ta"]}
    DX = {"tb7": L9.tb7_at(s.values.astype(float), DF["a"])}; DD = D.decision_data(s, DF, DX)
    P = D.predict_months(TX, DD, ["2024-02"], variants=("base", "d3"))
    idx = D.month_idx(DD["ta"], "2024-02"); idx = idx[DD["valid"][idx]]
    assert np.isfinite(P["base"][idx]).all() and np.isfinite(P["d3"][idx]).all() and np.isfinite(P["base"]).sum() == len(idx)
    Pf = D.predict_months(TX, DD, ["2024-02"], variants=("base",), frozen_m0=pd.Timestamp("2024-02-01"))
    assert np.array_equal(P["base"], Pf["base"], equal_nan=True)                                       # the frozen fit at the month start equals the walk-forward one


# ------------------------------------------------------------------ metrics and rules
def test_month_metrics_on_hand_made_inputs():
    ret = np.array([1, 1, 1, -1, -1, 1] * 20, float) * 1e-3; p = np.where(ret > 0, 0.6, 0.4)
    m = D.month_metrics(p, ret, np.arange(len(ret)))
    assert m["auc"] == 1.0 and m["hit_all"] == 1.0 and m["hit_gated"] == 1.0 and m["n_gated"] == 120 and m["brier"] == pytest.approx(0.16)
    p2 = np.full(len(ret), 0.5) + 0.01 * np.sign(ret); m2 = D.month_metrics(p2, ret, np.arange(len(ret)))
    assert m2["n_gated"] == 0 and np.isnan(m2["auc_gated"]) and m2["auc"] == 1.0                      # inside the +-0.02 band: not gated
    p3 = 0.5 + 0.03 * np.sign(ret[:60]); r3 = D.month_metrics(p3, ret[:60], np.arange(60))
    assert r3["n_gated"] == 60 and r3["auc_gated"] == 1.0 and r3["hit_gated"] == 1.0
    assert D.auc(np.ones(5), np.arange(5.0)) != D.auc(np.ones(5), np.arange(5.0))                     # one class: nan


def test_mean_t_and_choice_rule():
    r = D.mean_t([0.01, 0.02, 0.0, 0.03]); assert r["mean"] == pytest.approx(0.015) and r["pos"] == 3 and r["lo"] < r["mean"] < r["hi"] and 0 < r["p"] < 1
    assert D.choose({"base": 0.60, "d1": 0.51, "d2": 0.53, "d3": 0.52}) == "d2"                       # base is never chosen
    assert D.choose({"base": 0.50, "d1": 0.51, "d2": 0.51 - 1e-9, "d3": 0.52}) == "d3"
    tab = {"a": {"m1": {"x": 0.6}, "m2": {"x": float("nan")}}, "b": {"m1": {"x": 0.5}, "m2": {"x": 0.4}}}
    assert D.paired(tab, "a", "b", "x", ["m1", "m2"]).tolist() == pytest.approx([0.1])


# ------------------------------------------------------------------ the choice gate (as O8 / O9)
def _mini_repo(tmp_path, monkeypatch):
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (tmp_path / "keep.txt").write_text("x"); g("add", "keep.txt"); g("commit", "-q", "-m", "init")
    monkeypatch.setattr(D, "HERE", str(tmp_path)); monkeypatch.setattr(D, "CHOICE", str(tmp_path / "choice_d.json"))
    return g


def test_choice_gate_refuses_until_choice_d_is_committed_and_unchanged(tmp_path, monkeypatch):
    g = _mini_repo(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="choice_d.json is not tracked"):
        D.choice_gate()
    (tmp_path / "choice_d.json").write_text(json.dumps({"variant": "d1"}))
    with pytest.raises(SystemExit, match="choice_d.json is not tracked"):                              # present but untracked
        D.choice_gate()
    with pytest.raises(SystemExit, match="not tracked"):
        R.judge()
    g("add", "choice_d.json")
    with pytest.raises(SystemExit, match="not in HEAD|differs"):                                       # staged, not committed
        D.choice_gate()
    g("commit", "-q", "-m", "c")
    assert D.choice_gate()["variant"] == "d1"
    (tmp_path / "choice_d.json").write_text(json.dumps({"variant": "d2"}))
    with pytest.raises(SystemExit, match="differs"):
        D.choice_gate()
    with pytest.raises(SystemExit, match="differs"):
        R.freeze()


# ------------------------------------------------------------------ the frozen artefact and the prospective score
def test_artefact_reload_tamper_and_independence_of_later_bars():
    s = synth(61000); m0 = pd.Timestamp("2024-02-05 00:00"); tr = s[s.index < m0]
    art = D.freeze_fit(tr, m0, "d3"); aj = json.loads(json.dumps(art))
    assert P5.sha_of(aj) == art["sha256"] and set(aj["models"]) == {"base", "d3"}
    for v in aj["models"].values():
        assert pd.Timestamp(v["last_label_end"]) <= m0
    assert aj["models"]["d3"]["days"] == 90 and aj["models"]["d3"]["use_g"] and not aj["models"]["base"]["use_g"] and pd.Timestamp(art["last_bar"]) < m0
    s2 = s.copy(); s2.loc[s2.index >= m0, :] = np.random.default_rng(2).uniform(1, 1000, s2[s2.index >= m0].shape)
    assert D.freeze_fit(s2[s2.index < m0], m0, "d3")["sha256"] == art["sha256"]
    aj["models"]["base"]["coef"][0] += 1e-9; assert P5.sha_of(aj) != art["sha256"]


def test_score_refusals_and_score_core_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["score_P_d.py"])
    res = tmp_path / "r.json"; res.write_text("{}"); monkeypatch.setattr(SD, "RESULTS", str(res))
    with pytest.raises(SystemExit, match="exists"):
        SD.main()
    monkeypatch.setattr(SD, "RESULTS", str(tmp_path / "none.json")); monkeypatch.setattr(SD, "FROZEN", str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="does not exist"):
        SD.main()
    s = synth(61000); m0 = pd.Timestamp("2024-02-05 00:00"); art = D.freeze_fit(s[s.index < m0], m0, "d1"); art["m0"] = "tampered"
    fz = tmp_path / "f.json"; fz.write_text(json.dumps(art)); monkeypatch.setattr(SD, "FROZEN", str(fz))
    with pytest.raises(SystemExit, match="sha256"):
        SD.main()
    good = D.freeze_fit(s[s.index < m0], m0, "d1"); fz.write_text(json.dumps(good))
    monkeypatch.setattr(SD.SP9, "last_bar_time", lambda csv: pd.Timestamp("2026-10-12 23:59"))
    monkeypatch.setattr(SD.T, "load_bars", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a bar was read")))
    with pytest.raises(SystemExit, match="REFUSED: 0 complete months"):
        SD.main()
    s3 = synth(110000, "2026-06-10"); m1 = pd.Timestamp("2026-07-10 00:00"); art2 = D.freeze_fit(s3[s3.index < m1], m1, "d2")
    monkeypatch.setattr(SD, "BOOT_B", 300)
    r = SD.score_core(art2, s3, ["2026-07", "2026-08"])
    assert r["months"] == ["2026-07", "2026-08"] and len(r["monthly_auc_diff"]) == 2 and set(r["pooled_auc"]) == {"base", "d2"} and r["pooled_auc_diff_ci"]["lo"] <= r["pooled_auc_diff_ci"]["hi"]
    assert 0.4 < r["pooled_auc"]["base"] < 0.6 and isinstance(r["primary_pass"], bool)

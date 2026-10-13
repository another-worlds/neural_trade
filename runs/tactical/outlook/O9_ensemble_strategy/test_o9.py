"""O9 tests, PHASE 1 (SPEC "Tests"): causality of dV and p_up, labels before the month, gates, size cap, the engine against O8's, META's training set,
the choice gate. pytest in this folder."""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o9_lib as L  # noqa: E402
import run_o9 as R  # noqa: E402
L8 = L.L8; P5 = L.P5; O3 = L.O3; T = L.T


def synth(n, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


def synth_frames(n=61000):
    s = synth(n); first = 6145
    Fr = P5.build_frame(s, first, with_g=False); TF = {k: Fr[k] for k in ("ta", "XB", "rv15", "rv60", "ok15", "ok60")}
    X = {**L.train_extras(s, TF["ta"]), "ta": TF["ta"], "S3": TF["XB"][:, :L.N_S3].copy(), "ok60": TF["ok60"]}
    X["dok"] = X["ok60"] & np.isfinite(X["dA"])
    X["pok"] = np.isfinite(X["ret60"]) & (np.abs(X["ret60"]) > L.DEAD) & np.isfinite(X["tb7"]).all(1) & TF["ok60"]
    return s, TF, X


# ------------------------------------------------------------------ features and targets
def test_tb7_at_an_anchor_uses_only_bars_before_it():
    s = synth(3000); A = s.values.astype(float); a = np.array([500, 1500])
    f = L.tb7_at(A, a)
    A2 = A.copy(); A2[1500:] = np.random.default_rng(1).uniform(1, 1000, A2[1500:].shape)
    f2 = L.tb7_at(A2, a[:1]); assert np.array_equal(f[0], f2[0])
    A3 = A.copy(); A3[1499, :4] *= 1.01
    assert not np.array_equal(L.tb7_at(A3, a[1:])[0], f[1])                            # the last closed bar matters
    assert f.shape == (2, 13)
    # equals lab.fs_tb7 on the explicit window
    w = A[1500 - 60:1500][None].astype(np.float32)
    assert np.allclose(f[1], np.nan_to_num(L.LAB.fs_tb7(w))[0], rtol=1e-5, atol=1e-6)


def test_dv_target_equals_t01_and_features_at_decisions_equal_the_frame():
    s = synth(30000); first = 3000
    B = T.build(s, first); D = B["D"]
    ex = L.train_extras(s, D.ta.values.astype("datetime64[ns]"))
    okx = B["ok"]
    assert np.allclose(ex["dA"][okx], B["T"]["A"][okx], atol=1e-9) and np.array_equal(ex["a"], D.anch)
    # ret60 is the next-open 60-minute return
    i = 100; a = ex["a"][i]; assert ex["ret60"][i] == pytest.approx(s["open"].values[a + 60] / s["open"].values[a] - 1)
    assert np.array_equal(ex["tb7"][i], L.tb7_at(s.values.astype(float), np.array([a]))[0])


def test_ret60_is_nan_across_a_data_gap():
    s = synth(3000); s = pd.concat([s.iloc[:1000], s.iloc[1010:]])
    ta = (s.index[np.array([950, 1100]) - 1] + pd.Timedelta(minutes=1)).values.astype("datetime64[ns]")
    ex = L.train_extras(s, ta); assert np.isnan(ex["ret60"][0]) and np.isfinite(ex["ret60"][1])


# ------------------------------------------------------------------ the walk-forward models
def test_dv_and_pup_fits_use_only_rows_whose_labels_end_before_the_month_and_ignore_later_data():
    s, TF, X = synth_frames()
    m0 = pd.Timestamp("2024-02-03 00:00")
    sd = L.dv_fit(X, m0); sp = L.pup_fit(X, m0)
    assert sd["last_label_end"] <= m0 and sp["last_label_end"] <= m0
    # the labels of rows after the cut are replaced by noise: both fits are unchanged
    late = X["ta"] > np.datetime64(m0 - pd.Timedelta(minutes=L.H), "ns")
    X2 = dict(X); X2["dA"] = X["dA"].copy(); X2["ret60"] = X["ret60"].copy()
    rng = np.random.default_rng(3); X2["dA"][late] = rng.normal(size=late.sum()); X2["ret60"][late] = rng.normal(0, 0.02, late.sum())
    X2["dok"] = X2["ok60"] & np.isfinite(X2["dA"]); X2["pok"] = np.isfinite(X2["ret60"]) & (np.abs(X2["ret60"]) > L.DEAD) & np.isfinite(X2["tb7"]).all(1) & X2["ok60"]
    sd2 = L.dv_fit(X2, m0); sp2 = L.pup_fit(X2, m0)
    assert np.array_equal(sd["m"]["w"], sd2["m"]["w"]) and np.array_equal(sp["lr"].coef_, sp2["lr"].coef_)


def test_pup_dead_band_labels_and_365_day_window():
    s, TF, X = synth_frames()
    m0 = pd.Timestamp("2024-02-10 00:00"); tr = O3.train_rows(X["ta"], X["pok"], L.H, m0)
    assert (np.abs(X["ret60"][tr]) > 5e-4).all()                                        # rows inside the dead band are not trained on
    assert (X["ta"][tr] >= np.datetime64(m0 - pd.Timedelta(days=365), "ns")).all()
    sp = L.pup_fit(X, m0); assert sp["base"] == pytest.approx(float((X["ret60"][tr] > 0).mean()))
    # a hand-made frame: label = sign of ret60
    ta = (pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(400) * 7, "m")).values.astype("datetime64[ns]")
    rng = np.random.default_rng(0); tb = rng.normal(size=(400, 13)).astype(np.float32); ret = np.where(tb[:, 0] > 0, 1e-3, -1e-3)
    Xh = {"ta": ta, "tb7": tb, "ret60": ret, "pok": np.ones(400, bool)}
    st = L.pup_fit(Xh, pd.Timestamp("2024-03-01")); p = L.pup_predict(st, tb)
    assert np.corrcoef(p, tb[:, 0])[0, 1] > 0.9 and ((p > 0.5) == (tb[:, 0] > 0)).mean() > 0.95


def test_signals_at_a_decision_do_not_depend_on_bars_at_or_after_it_end_to_end():
    s, TF, X = synth_frames(); first = 6145
    DFc = L8.build_dframe(s, first)
    DF = {"XB": DFc["XB"], "G": DFc["G"], "ok": DFc["ok"], "a": DFc["a"], "ta": DFc["ta"], "m": DFc["m"], **{f"trail{H}": DFc["trail"][H] for H in L8.HZ}}
    TF["G"] = np.zeros((len(TF["ta"]), L8.N_L1G), np.float32)
    mo = "2024-02"; m0, m1 = P5.month_bounds(mo); rows = L8.month_rows(DF["ta"], m0, m1); rows = rows[DF["ok"][rows]]
    r = rows[len(rows) // 2]; a = int(DF["a"][r])
    dv_ref = O3.ridge_pred(L.dv_fit(X, m0)["m"], DF["XB"][r:r + 1][:, :L.N_S3])
    pu_ref = L.pup_predict(L.pup_fit(X, m0), L.tb7_at(s.values.astype(float), np.array([a])))
    s2 = s.copy(); s2.iloc[a:] = np.random.default_rng(1).uniform(1, 1000, s2.iloc[a:].shape)       # every bar at or after the decision replaced
    D2 = L8.build_dframe(s2, first)
    r2 = int(np.where(D2["a"] == a)[0][0])
    dv2 = O3.ridge_pred(L.dv_fit(X, m0)["m"], D2["XB"][r2:r2 + 1][:, :L.N_S3])
    pu2 = L.pup_predict(L.pup_fit(X, m0), L.tb7_at(s2.values.astype(float), np.array([a])))
    assert np.array_equal(dv_ref, dv2) and np.array_equal(pu_ref, pu2)


# ------------------------------------------------------------------ gates, size
def test_gates_on_hand_made_inputs():
    p = np.array([0.60, 0.53, 0.50, 0.47, 0.40, np.nan]); dv = np.array([0.1, -0.1, 0.2, 0.0, 0.3, 0.2])
    al, as_ = L.gate_masks(p, dv, None, None); assert al.all() and as_.all()                       # both gates off: both sides everywhere
    al, as_ = L.gate_masks(p, dv, 0.02, None)
    assert list(al) == [True, True, False, False, False, False] and list(as_) == [False, False, False, True, True, False]
    al, as_ = L.gate_masks(p, dv, None, 0.0)                                                        # dV >= 0 (0.0 passes), nan blocks nothing here
    assert list(al) == [True, False, True, True, True, True] and list(as_) == list(al)
    al, as_ = L.gate_masks(p, dv, 0.02, 0.0)
    assert list(al) == [True, False, False, False, False, False] and list(as_) == [False, False, False, True, True, False]
    al, as_ = L.gate_masks(np.array([0.5 + 0.02, 0.5 - 0.02]), np.zeros(2), 0.02, None)           # the thresholds are inclusive
    assert al[0] and as_[1]
    al, as_ = L.gate_masks(np.array([0.7]), np.array([np.nan]), None, 0.0); assert not al[0] and not as_[0]        # nan dV blocks when the gate is on


def test_size_is_capped_at_two_and_uses_the_previous_months_median():
    w = L.size_weights(np.array([0.001, 0.002, 0.004, 0.02]), 0.004)
    assert list(w) == [2.0, 2.0, 1.0, pytest.approx(0.2)] and w.max() <= 2.0
    ta = (pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(0, 3 * 31 * 1440, 15), "m")).values.astype("datetime64[ns]")
    mo = pd.DatetimeIndex(ta).to_period("M").astype(str).values
    sig = np.where(mo == "2024-01", 0.01, np.where(mo == "2024-02", 0.02, 0.5)); sig[np.where(mo == "2024-01")[0][:5]] = 0.5
    ref = L.sigma_ref_by_month(sig, ta, ["2024-02", "2024-03"])
    assert ref["2024-02"] == pytest.approx(0.01) and ref["2024-03"] == pytest.approx(0.02)        # the previous month, not the month itself
    assert np.isnan(L.sigma_ref_by_month(sig, ta, ["2024-01"])["2024-01"])


# ------------------------------------------------------------------ the engine
def bars(rows, pad=80):
    r = list(rows) + [(rows[-1][3],) * 4] * pad; a = np.array(r, float); return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def flat(p=100.0):
    return (p, p, p, p)


def run9(rows, k=1.0, H=15, stop=None, sig=0.01, dec=(0,), pad=80, **kw):
    O, Hi, Lo, Cl = bars(rows, pad); d = np.array(dec)
    return L.simulate9(O, Hi, Lo, Cl, d, O[d], np.full(len(d), sig), k, H, stop, **kw)


def test_simulate9_equals_o8_brk_with_no_permissions_and_no_target_on_random_bars():
    s = synth(12000, seed=5); O, Hi, Lo, Cl = (s[c].values for c in ("open", "high", "low", "close"))
    dec = np.arange(100, 11000, 15); sig = np.random.default_rng(2).uniform(0.001, 0.004, len(dec)); tm = s.index.values.astype("datetime64[m]").astype(np.int64)
    for pess in (False, True):
        a = L8.simulate(O, Hi, Lo, Cl, dec, O[dec], sig, "BRK", 1.5, 60, 1.0, tmin=tm, pess=pess)
        b = L.simulate9(O, Hi, Lo, Cl, dec, O[dec], sig, 1.5, 60, 1.0, tmin=tm, pess=pess)
        assert len(a["dec"]) > 50
        for k_ in ("dec", "ebar", "xbar", "side", "epx", "xpx", "reason"):
            assert np.array_equal(a[k_], b[k_]), k_


def test_a_forbidden_side_places_no_order_and_the_other_side_still_fills():
    rows = [flat(), (100, 101.5, 98.0, 100), flat()]                                 # both stops (U 101, L 99) are touched in bar 1
    tr = run9(rows, k=1.0, allow_long=np.array([False]), allow_short=np.array([True]))
    assert len(tr["dec"]) == 1 and tr["side"][0] == -1
    tr = run9(rows, k=1.0, allow_long=np.array([False]), allow_short=np.array([False])); assert len(tr["dec"]) == 0
    tr = run9([flat(), (100, 101.5, 100, 101), flat(101)], k=1.0, allow_long=np.array([False]), allow_short=np.array([True]))   # only the long level is touched
    assert len(tr["dec"]) == 0


def test_target_exit_at_tau_sigma_and_stop_first_when_both_in_one_bar():
    rows = [flat(), (100, 101.2, 100, 101.1), (101.1, 103.5, 101.0, 103.0), flat(103)]     # long at 101 (bar 1); target 101 * 1.02 = 103.02
    tr = run9(rows, k=1.0, stop=1.0, tau=2.0, H=15)
    assert tr["side"][0] == 1 and tr["reason"][0] == "target" and tr["xbar"][0] == 2 and tr["xpx"][0] == pytest.approx(101.0 * 1.02)   # bar 2 trades 1 bp through 103.02; fill = max(target, open)
    rows = [flat(), (100, 101.2, 100, 101.1), (101.1, 103.5, 98.0, 100.0), flat(100)]       # bar 2 spans stop (101 * 0.99 = 99.99) and target
    tr = run9(rows, k=1.0, stop=1.0, tau=2.0, H=15)
    assert tr["reason"][0] == "stop" and tr["xpx"][0] == pytest.approx(101.0 * 0.99)
    tr = run9(rows, k=1.0, stop=1.0, tau=None, H=15); assert tr["reason"][0] == "stop"
    # a short mirrors it: entry 99 (sell stop), target 99 * 0.98 = 97.02
    rows = [flat(), (100, 100, 98.8, 98.9), (98.9, 98.95, 96.5, 97.0), flat(97)]
    tr = run9(rows, k=1.0, stop=1.0, tau=2.0, H=15)
    assert tr["side"][0] == -1 and tr["reason"][0] == "target" and tr["xpx"][0] == pytest.approx(99.0 * 0.98)


def test_metrics_apply_the_size_weight_to_returns_and_costs():
    ret = np.array([0.01, -0.02, 0.005]); w = np.array([2.0, 1.0, 0.5]); em = np.array(["2024-01", "2024-01", "2024-02"])
    mr = L.monthly9(ret, w, em, ["2024-01", "2024-02", "2024-03"], 2.0)
    assert np.allclose(mr, [2 * (0.01 - 2e-4) + (-0.02 - 2e-4), 0.5 * (0.005 - 2e-4), 0.0])
    tr = {"ret": ret, "reason": np.array(["time", "stop", "target"]), "side": np.array([1, -1, 1])}
    sm = L.summarise9(tr, w, em, ["2024-01", "2024-02", "2024-03"])
    assert sm["n_trades"] == 3 and sm["trades_per_month"] == pytest.approx(1.0) and sm["hit_2bps"] == pytest.approx(2 / 3)
    assert sm["max_dd_2bps"] == pytest.approx(0.0202)                                    # weighted nets +0.0196, -0.0202, +0.0024: peak 0.0196 to trough -0.0006


# ------------------------------------------------------------------ META
def test_meta_trains_only_on_trades_that_exited_before_the_month():
    t_exit = np.array(["2024-01-30T23:59", "2024-01-31T23:59", "2024-02-01T00:00", "2024-02-01T00:01", "2024-03-05T00:00"], "datetime64[ns]")
    m0 = pd.Timestamp("2024-02-01")
    assert list(L.meta_train_rows(t_exit, m0)) == [0, 1, 2]                          # exit known by the month start (inclusive), later exits excluded
    # end to end: labels of trades exiting after the month start are flipped; the month's gates do not change
    rng = np.random.default_rng(0); n = 6000
    ta = (pd.Timestamp("2023-06-01") + pd.to_timedelta(np.arange(n) * 60, "m")).values.astype("datetime64[ns]")
    DF = {"ta": ta, "G": rng.normal(size=(n, 328)).astype(np.float32)}
    sig = rng.uniform(0.001, 0.003, n); dv = rng.normal(size=n); p = rng.uniform(0.3, 0.7, n)
    rows = np.arange(0, n, 3); side = rng.choice([-1, 1], len(rows))
    t_ex = ta[rows] + np.timedelta64(2, "h"); y = ((p[rows] > 0.5) == (side == 1)).astype(int) ^ (rng.uniform(size=len(rows)) < 0.3)
    cand = {"row": rows, "side": side, "t_exit": t_ex, "y": y}
    months = ["2024-01", "2024-02"]; gcols = np.arange(15)
    a1 = L.meta_gate(cand, DF, sig, dv, p, gcols, months)
    cut = np.datetime64("2024-02-01", "ns"); y2 = y.copy(); y2[t_ex > cut] = 1 - y2[t_ex > cut]                 # flip labels of trades exiting after Feb 1
    a2 = L.meta_gate({**cand, "y": y2}, DF, sig, dv, p, gcols, months[:1])
    feb = np.where(pd.DatetimeIndex(ta).to_period("M").astype(str) == "2024-01")[0]
    assert np.array_equal(a1[0][feb], a2[0][feb]) and np.array_equal(a1[1][feb], a2[1][feb]) and a1[2]["2024-01"] == a2[2]["2024-01"] and a1[2]["2024-01"]["n_train"] > 500
    # the model for February (a1) is trained only on trades exiting by Feb 1: flipping later labels leaves its threshold and gates unchanged
    a3 = L.meta_gate({**cand, "y": y2}, DF, sig, dv, p, gcols, months)
    feb2 = np.where(pd.DatetimeIndex(ta).to_period("M").astype(str) == "2024-02")[0]
    assert a3[2]["2024-02"]["n_train"] == a1[2]["2024-02"]["n_train"] and a3[2]["2024-02"]["thr"] == a1[2]["2024-02"]["thr"]
    assert np.array_equal(a1[0][feb2], a3[0][feb2]) and np.array_equal(a1[1][feb2], a3[1][feb2]) and a1[0][feb2].any()
    n_in = int((t_ex <= cut).sum()); assert a1[2]["2024-02"]["n_train"] == n_in


def test_meta_threshold_is_the_previous_months_score_quantile_and_first_month_is_inactive():
    rng = np.random.default_rng(1); n = 9000
    ta = (pd.Timestamp("2023-09-01") + pd.to_timedelta(np.arange(n) * 60, "m")).values.astype("datetime64[ns]")
    DF = {"ta": ta, "G": rng.normal(size=(n, 328)).astype(np.float32)}
    sig = rng.uniform(0.001, 0.003, n); dv = rng.normal(size=n); p = rng.uniform(0.3, 0.7, n)
    rows = np.arange(0, n, 2); side = rng.choice([-1, 1], len(rows)); y = (rng.uniform(size=len(rows)) < 0.5).astype(int)
    cand = {"row": rows, "side": side, "t_exit": ta[rows] + np.timedelta64(1, "h"), "y": y}
    aL, aS, info = L.meta_gate(cand, DF, sig, dv, p, np.arange(15), ["2023-12", "2024-01", "2024-02"])
    mo = pd.DatetimeIndex(ta).to_period("M").astype(str)
    assert np.isnan(info["2023-12"]["thr"]) and not aL[mo == "2023-12"].any() and not aS[mo == "2023-12"].any()        # no previous month's scores yet
    assert np.isfinite(info["2024-01"]["thr"]) and np.isfinite(info["2024-02"]["thr"])
    # about 40% of candidate-like orders pass the 60th-percentile threshold
    share = np.mean(np.concatenate([aL[mo == "2024-02"], aS[mo == "2024-02"]])); assert 0.2 < share < 0.6


def test_meta_features_align_p_up_to_the_side():
    X = L.meta_X(np.array([0.7, 0.7]), np.zeros(2), np.array([0.002, 0.002]), np.zeros((2, 15)), np.array([6, 6]), np.array([1, -1]))
    assert X.shape == (2, 20) and X[0, 0] == pytest.approx(0.7) and X[1, 0] == pytest.approx(0.3)


def test_geometry_columns_resolve_to_the_o1_top_features():
    cols = L.geom_cols(); assert len(cols) == 15 and len(set(cols)) == 15 and cols.max() < 328


# ------------------------------------------------------------------ grid, gate
def test_grid_is_the_literal_32_with_distinct_names():
    g = L.grid(); assert len(g) == 32 and len({L.cfg_name(c) for c in g}) == 32


def test_choice_gate_refuses_without_a_committed_unchanged_choice_and_phase2_stubs_stop(tmp_path, monkeypatch):
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (tmp_path / "keep.txt").write_text("x"); g("add", "keep.txt"); g("commit", "-q", "-m", "init")
    monkeypatch.setattr(R, "HERE", str(tmp_path)); monkeypatch.setattr(R, "CHOICE", str(tmp_path / "choice.json"))
    with pytest.raises(SystemExit, match="not tracked"):
        R.describe()
    (tmp_path / "choice.json").write_text(json.dumps({"config": "x"}))
    with pytest.raises(SystemExit, match="not tracked"):
        R.freeze()
    g("add", "choice.json")
    with pytest.raises(SystemExit, match="not in HEAD|differs"):
        R.choice_gate()
    g("commit", "-q", "-m", "choice")
    with pytest.raises(SystemExit, match="PHASE 2 not implemented"):
        R.describe()
    (tmp_path / "choice.json").write_text(json.dumps({"config": "y"}))
    with pytest.raises(SystemExit, match="differs"):
        R.freeze()

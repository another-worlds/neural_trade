"""O8 tests (SPEC "Tests"): pytest in this folder. Fill rules on hand-made bars, causality of sigma, the frozen fit's data limits, SHUF, the choice gate."""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o8_lib as L  # noqa: E402
import run_o8 as R  # noqa: E402
P5 = L.P5; O3 = L.O3; T = L.T
HERE = os.path.dirname(os.path.abspath(__file__))


def synth(n, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n) * (1 + np.abs(np.sin(np.arange(n) / 900)))))
    o = np.concatenate([[c[0]], c[:-1]]); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n))); v = rng.gamma(2, 5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=pd.date_range(start, periods=n, freq="min"))


# ------------------------------------------------------------------ fill rules on hand-made bars
def bars(rows, pad=80):
    """rows: list of (open, high, low, close) for bars 0..; padded flat at the last close so every window exists."""
    r = list(rows) + [(rows[-1][3],) * 4] * pad
    a = np.array(r, float); return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def flat(p=100.0):
    return (p, p, p, p)


def run(rows, family="REV", k=1.0, H=15, stop=None, sig=0.01, dec=(0,), pad=80):
    O, Hi, Lo, Cl = bars(rows, pad)
    d = np.array(dec); m = O[d]
    return L.simulate(O, Hi, Lo, Cl, d, m, np.full(len(d), sig), family, k, H, stop)


def test_limit_touched_but_not_traded_through_does_not_fill():
    # k sigma = 1%: buy limit at 99. Bar 1 low 98.995 (only 0.5 bp through 99 -> 98.9951): no fill; bar 2 low 98.98 (2 bp through) fills at 99.
    rows = [flat(), (100, 100, 98.995, 99.5), flat(99.5), flat(99.5)]
    tr = run(rows, "REV", dec=(0,), pad=100)
    assert len(tr["dec"]) == 0
    rows2 = [flat(), (100, 100, 98.995, 99.5), (99.5, 99.5, 98.98, 99.2), flat(99.2)]
    tr = run(rows2, "REV", dec=(0,), pad=100)
    assert len(tr["dec"]) == 1 and tr["ebar"][0] == 2 and tr["side"][0] == 1 and tr["epx"][0] == pytest.approx(99.0)


def test_exactly_one_bp_through_fills():
    rows = [flat(), (100, 100, 99.0 * (1 - 1e-4), 99.5), flat(99.5)]
    tr = run(rows, "REV", dec=(0,), pad=100)
    assert len(tr["dec"]) == 1 and tr["epx"][0] == pytest.approx(99.0)


def test_gap_open_beyond_the_limit_fills_at_the_open_and_a_buy_stop_gap_fills_at_the_worse_price():
    rows = [flat(), (98.0, 98.5, 97.5, 98.0), flat(98.0)]               # opens 1% below the buy limit 99: filled at 98 (better)
    tr = run(rows, "REV", dec=(0,), pad=100)
    assert tr["ebar"][0] == 1 and tr["epx"][0] == pytest.approx(98.0) and tr["side"][0] == 1
    rows = [flat(), (102.0, 102.5, 101.5, 102.0), flat(102.0)]          # BRK: buy stop at 101 triggers; opens at 102 -> fills at 102 (worse)
    tr = run(rows, "BRK", dec=(0,), pad=100)
    assert tr["side"][0] == 1 and tr["epx"][0] == pytest.approx(102.0)
    rows = [flat(), (98.0, 98.5, 97.5, 98.0), flat(98.0)]               # BRK sell stop at 99, opens 98 -> fills at 98 (worse for a short)
    tr = run(rows, "BRK", dec=(0,), pad=100)
    assert tr["side"][0] == -1 and tr["epx"][0] == pytest.approx(98.0)


def test_brk_trigger_is_at_the_level_without_trade_through():
    rows = [flat(), (100, 101.0, 100, 100.5), flat(100.5)]
    tr = run(rows, "BRK", dec=(0,), pad=100)
    assert len(tr["dec"]) == 1 and tr["epx"][0] == pytest.approx(101.0) and tr["side"][0] == 1


def test_stop_and_target_in_the_same_bar_resolve_to_the_stop():
    # REV long: entry 99 (bar 1), target 100, stop 1 sigma below = 99 * 0.99 = 98.01. Bar 2 spans both.
    rows = [flat(), (100, 100, 98.9, 99.2), (99.2, 100.5, 98.0, 99.0), flat(99.0)]
    tr = run(rows, "REV", dec=(0,), stop=1.0, pad=100)
    assert tr["reason"][0] == "stop" and tr["xbar"][0] == 2 and tr["xpx"][0] == pytest.approx(98.01)
    # without the stop the same bar hits the target at 100
    tr = run(rows, "REV", dec=(0,), stop=None, pad=100)
    assert tr["reason"][0] == "target" and tr["xpx"][0] == pytest.approx(100.5 if False else 100.0)


def test_exits_are_checked_from_the_bar_after_the_entry_bar():
    # the entry bar itself spans the stop: ignored (SPEC). Bar 2 is calm.
    rows = [flat(), (100, 100.2, 97.0, 99.0), flat(99.0)]
    tr = run(rows, "REV", dec=(0,), stop=1.0, H=15, pad=100)
    assert tr["ebar"][0] == 1 and tr["reason"][0] == "time"


def test_gap_beyond_target_and_stop_fill_at_the_open():
    rows = [flat(), (100, 100, 98.9, 99.2), (101.0, 101.5, 100.5, 101.0), flat(101.0)]       # long entered at 99; bar 2 opens above the target 100
    tr = run(rows, "REV", dec=(0,), pad=100)
    assert tr["reason"][0] == "target" and tr["xpx"][0] == pytest.approx(101.0)
    rows = [flat(), (100, 100, 98.9, 99.2), (97.0, 97.5, 96.5, 97.0), flat(97.0)]            # opens below the stop 98.01
    tr = run(rows, "REV", dec=(0,), stop=1.0, pad=100)
    assert tr["reason"][0] == "stop" and tr["xpx"][0] == pytest.approx(97.0)


def test_time_stop_exits_at_the_open_of_entry_plus_H():
    rows = [flat(), (100, 100, 98.9, 99.2)] + [(99.2, 99.5, 98.9, 99.2)] * 30       # entry bar 1, nothing else happens; bar opens move at bar 1 + H
    O, Hi, Lo, Cl = bars(rows, 100)
    O[16] = 98.7; Lo[16] = 98.0                                                      # bar 16 = entry bar 1 + H(15): exit at ITS OPEN, not its low
    tr = L.simulate(O, Hi, Lo, Cl, np.array([0]), O[[0]], np.array([0.01]), "REV", 1.0, 15, None)
    assert tr["reason"][0] == "time" and tr["xbar"][0] == 16 and tr["xpx"][0] == pytest.approx(98.7)
    assert tr["ret"][0] == pytest.approx(np.log(98.7 / 99.0))


def test_short_side_mirrors_the_long_side():
    rows = [flat(), (100, 101.1, 100, 100.8), (100.8, 100.9, 99.9, 100.0), flat(100.0)]      # short at 101 (U); target 100
    tr = run(rows, "REV", dec=(0,), pad=100)
    assert tr["side"][0] == -1 and tr["epx"][0] == pytest.approx(101.0) and tr["reason"][0] == "target" and tr["xpx"][0] == pytest.approx(100.0)
    assert tr["ret"][0] == pytest.approx(np.log(101.0 / 100.0))
    rows = [flat(), (100, 101.1, 100, 100.8), (100.8, 102.5, 100.5, 102.0), flat(102.0)]     # short stopped at 101 * 1.01 = 102.01 -> not reached: high 102.5 ok
    tr = run(rows, "REV", dec=(0,), stop=1.0, pad=100)
    assert tr["reason"][0] == "stop" and tr["xpx"][0] == pytest.approx(101.0 * 1.01)


def test_only_one_position_at_a_time_and_a_decision_on_the_time_stop_bar_is_allowed():
    # decisions at bars 0, 15, 30. Position from decision 0 enters at bar 1 and is open until its time stop at bar 16: decision 15 is skipped.
    rows = [flat(), (100, 100, 98.9, 99.2)] + [(99.2, 99.5, 98.9, 99.2)] * 80
    O, Hi, Lo, Cl = bars(rows, 100)
    d = np.array([0, 15, 30]); m = np.full(3, 100.0)
    # decision 30's orders: make bar 31 trade through 99
    O[31], Hi[31], Lo[31], Cl[31] = 100.0, 100.0, 98.9, 99.2
    # make the price at decisions: m is the open of the decision bar
    tr = L.simulate(O, Hi, Lo, Cl, d, m, np.full(3, 0.01), "REV", 1.0, 15, None)
    assert list(tr["dec"]) == [0, 30] and tr["xbar"][0] == 16
    # a decision exactly on the time-stop bar (16) is allowed, one on an intrabar-exit bar is not
    O2, H2, L2, C2 = bars(rows, 100); O2[17] = 100.0; H2[17] = 100.0; L2[17] = 98.9
    tr = L.simulate(O2, H2, L2, C2, np.array([0, 16]), np.array([100.0, 100.0]), np.full(2, 0.01), "REV", 1.0, 15, None)
    assert list(tr["dec"]) == [0, 16]


def test_orders_are_valid_for_15_bars_only_and_nan_sigma_places_nothing():
    rows = [flat()] * 16 + [(100, 100, 98.0, 99.0), flat(99.0)]
    assert len(run(rows, "REV", dec=(0,), pad=100)["dec"]) == 0           # bar 16 is beyond decision bar + 14
    rows = [flat()] * 15 + [(100, 100, 98.0, 99.0), flat(99.0)]
    assert len(run(rows, "REV", dec=(0,), pad=100)["dec"]) == 0           # bar 15 = decision + 15 is also outside (bars 0 .. 14)
    rows = [flat()] * 14 + [(100, 100, 98.0, 99.0), flat(99.0)]
    assert len(run(rows, "REV", dec=(0,), pad=100)["dec"]) == 1           # bar 14 is the last valid one
    rows = [flat(), (100, 100, 98.0, 99.0), flat(99.0)]
    assert len(run(rows, "REV", dec=(0,), sig=np.nan, pad=100)["dec"]) == 0


def test_both_levels_in_one_bar_take_the_one_nearer_the_open_and_are_counted():
    rows = [flat(), (99.2, 101.5, 98.0, 100.0), flat(100.0)]                # open 99.2: buy limit 99 is nearer than sell limit 101
    tr = run(rows, "REV", dec=(0,), pad=100)
    assert tr["side"][0] == 1 and tr["both"][0]


def test_a_data_gap_inside_the_decision_window_places_no_orders():
    rows = [flat(), (100, 100, 98.0, 99.0), flat(99.0)]
    O, Hi, Lo, Cl = bars(rows, 100); tm = np.arange(len(O)); tm[10:] += 500                    # a gap after bar 9
    a = L.simulate(O, Hi, Lo, Cl, np.array([0]), O[[0]], np.array([0.01]), "REV", 1.0, 15, None)
    b = L.simulate(O, Hi, Lo, Cl, np.array([0]), O[[0]], np.array([0.01]), "REV", 1.0, 15, None, tmin=tm)
    assert len(a["dec"]) == 1 and len(b["dec"]) == 0


# ------------------------------------------------------------------ sigma: causality, equality with the frame code, arms
def test_features_at_a_decision_anchor_are_causal():
    s = synth(9000); a = np.array([6000, 6015, 8000])
    F = L.decision_features(s, a)
    s2 = s.copy(); s2.iloc[6000:] = np.random.default_rng(1).uniform(1, 1000, s2.iloc[6000:].shape)
    F2 = L.decision_features(s2, a[:1])                                    # the anchor 6000: bars >= 6000 replaced by noise
    assert np.array_equal(F["XB"][0], F2["XB"][0]) and np.array_equal(F["G"][0], F2["G"][0]) and bool(F["ok"][0]) == bool(F2["ok"][0])
    assert F["trail"][15][0] == F2["trail"][15][0]
    s3 = s.copy(); s3.iloc[5999, :4] *= 1.01                               # the last closed bar matters
    assert not np.array_equal(L.decision_features(s3, a[:1])["XB"][0], F["XB"][0])


def test_decision_anchors_are_on_the_clock_and_not_after_a_gap():
    s = synth(3000, "2024-01-01 00:07"); s = pd.concat([s.iloc[:1000], s.iloc[1005:]])         # a 5-minute gap before position 1000
    a = L.decision_anchors(s.index, 100)
    assert (s.index[a].minute % 15 == 0).all() and (s.index[a] - s.index[a - 1] == pd.Timedelta(minutes=1)).all()
    assert 1000 not in a and len(a) > 100


def test_decision_features_equal_the_frame_code_on_anchors_of_both_grids_and_sigma_matches_o5_prediction():
    s = synth(61000, "2024-01-01"); first = 6145
    F3 = O3.build_features(s, first); Fr = P5.build_frame(s, first)
    sel = np.arange(0, len(F3["anch"]), 37)
    DF = L.decision_features(s, F3["anch"][sel])
    assert np.array_equal(DF["XB"], Fr["XB"][sel]) and np.allclose(DF["G"], Fr["G"][sel], rtol=1e-5, atol=1e-6)
    assert np.array_equal(DF["ok"], F3["okf"][sel])
    # sigma: fit on the stride-7 frame exactly as O5 does; predict at the shared anchors = O5's prediction
    m0 = pd.Timestamp("2024-02-03 00:00"); st = P5.fit_h(Fr, 60, m0, models=("S3", "L1"))
    te = sel[Fr["ta"][sel] >= np.datetime64(m0, "ns")]
    ref = P5.predict(st, Fr, te, models=("S3", "L1"))
    dfm = L.decision_features(s, F3["anch"][te]); pr = P5.predict(st, dfm, np.arange(len(te)), models=("S3", "L1"))
    assert np.allclose(pr["L1"], ref["L1"], atol=1e-8) and np.allclose(pr["S3"], ref["S3"], atol=1e-10) and len(te) > 5


def synth_pair(n=61000):
    """Training frame (stride 7) and decision frame of one synthetic series, with the dict shapes load_train_frame / load_dframe give."""
    s = synth(n, "2024-01-01"); first = 6145
    Fr = P5.build_frame(s, first); TF = {k: Fr[k] for k in ("ta", "XB", "rv15", "rv60", "ok15", "ok60")}; TF["G"] = Fr["G"]
    DFc = L.build_dframe(s, first)
    DF = {"XB": DFc["XB"], "G": DFc["G"], "ok": DFc["ok"], "a": DFc["a"], "ta": DFc["ta"], "m": DFc["m"], **{f"trail{H}": DFc["trail"][H] for H in L.HZ}}
    return s, TF, DF


def test_sigma_arms_monthly_walk_forward_trail_and_shuffle():
    s, TF, DF = synth_pair()
    arms, states = L.sigma_arms(TF, DF, ["2024-02"], 60)
    mo = pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values
    r = np.where((mo == "2024-02") & DF["ok"])[0]
    assert len(r) > 100 and np.isfinite(arms["L1"][r]).all() and np.isnan(arms["L1"][mo != "2024-02"]).all()
    st = states["2024-02"]; assert pd.Timestamp(st["last_label_end"]) <= pd.Timestamp("2024-02-01")
    ref = P5.predict(st, DF, r, models=("L1", "S3"))
    assert np.allclose(arms["L1"][r], np.sqrt(np.exp(ref["L1"]))) and np.allclose(arms["R0"][r], np.sqrt(np.exp(ref["S3"])))
    assert not np.allclose(arms["L1"][r], arms["R0"][r])
    cs = np.concatenate([[0.0], np.cumsum(np.concatenate([[0.0], np.diff(np.log(s["close"].values))]) ** 2)])
    assert np.allclose(arms["TRAIL"][r], np.sqrt(cs[DF["a"][r]] - cs[DF["a"][r] - 60]))
    assert np.array_equal(np.sort(arms["SHUF"][r]), np.sort(arms["L1"][r])) and not np.array_equal(arms["SHUF"][r], arms["L1"][r])
    arms2, _ = L.sigma_arms(TF, DF, ["2024-02"], 60)                      # seed 0: reproducible
    assert np.array_equal(arms2["SHUF"][r], arms["SHUF"][r])


def test_shuf_keeps_each_months_sigma_multiset():
    x = np.random.default_rng(3).uniform(0.001, 0.02, 500)
    y = L.shuffle_within(x, 0, 60, "2022-05")
    assert np.array_equal(np.sort(x), np.sort(y)) and not np.array_equal(x, y)
    assert np.array_equal(L.shuffle_within(x, 0, 60, "2022-05"), y) and not np.array_equal(L.shuffle_within(x, 0, 60, "2022-06"), y)


# ------------------------------------------------------------------ the frozen fit and the data limits
def test_frozen_fit_reads_only_bars_up_to_the_cut_and_never_the_forbidden_gap(tmp_path):
    assert L.LOAD_HI == T.CUTOFF == pd.Timestamp("2025-07-20 23:59")
    assert O3.TRAIN_END == T.CUTOFF + pd.Timedelta(minutes=1)
    idx = pd.date_range("2025-07-20 23:50", periods=20, freq="min")           # crosses the cut and runs into the forbidden gap
    df = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    p = tmp_path / "x.csv"; df.to_csv(p, index=False)
    s = T.load_bars(str(p), pd.Timestamp("2025-07-20 23:50"), L.LOAD_HI)
    assert s.index.max() == T.CUTOFF and len(s) == 10
    with pytest.raises(AssertionError):
        T.load_bars(str(p), pd.Timestamp("2025-07-20 23:50"), pd.Timestamp("2025-07-21 00:05"))
    # the module loads the cut file only: the hi it passes is the cut
    import inspect
    assert "LOAD_HI" in inspect.getsource(L.load_bars_cut)


def test_frozen_sigma_uses_one_fit_that_does_not_depend_on_rows_after_it():
    s, TF, DF = synth_pair()
    m0 = pd.Timestamp("2024-02-03 00:00")
    arms, states = L.sigma_arms(TF, DF, ["2024-02", "2024-03"], 60, frozen_m0=m0)
    st = states["frozen"]; assert pd.Timestamp(st["last_label_end"]) <= m0 and list(states) == ["frozen"]
    TF2 = {k: v.copy() for k, v in TF.items()}
    late = TF2["ta"] > np.datetime64(m0, "ns")
    for k in ("XB", "G", "rv15", "rv60"):
        TF2[k][late] = TF2[k][late] * 7 + 1
    arms2, _ = L.sigma_arms(TF2, DF, ["2024-02", "2024-03"], 60, frozen_m0=m0)
    r = np.where(np.isfinite(arms["L1"]))[0]
    assert len(r) > 0 and np.array_equal(arms["L1"][r], arms2["L1"][r])               # rows after m0 never enter the fit
    mo = pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values                 # SHUF: a permutation within each month
    for m in ("2024-02", "2024-03"):
        rr = np.where((mo == m) & np.isfinite(arms["L1"]))[0]
        assert np.array_equal(np.sort(arms["SHUF"][rr]), np.sort(arms["L1"][rr]))


@pytest.mark.skipif(not os.path.exists(L.FRAME) or not os.path.exists(os.path.join(L.SCRATCH, "dframe.npz")), reason="real caches not built")
def test_real_caches_end_at_the_cut_and_decision_features_equal_the_frame_on_shared_anchors():
    z = np.load(L.FRAME); D = np.load(os.path.join(L.SCRATCH, "dframe.npz"))
    assert pd.Timestamp(z["ta"].max()) <= T.CUTOFF + pd.Timedelta(minutes=1) and pd.Timestamp(D["ta"].max()) <= T.CUTOFF
    common, i1, i2 = np.intersect1d(z["anch"], D["a"], return_indices=True); sel = np.arange(0, len(common), max(1, len(common) // 3000))
    assert len(common) > 10000
    assert np.array_equal(z["XB"][i1[sel]], D["XB"][i2[sel]]) and np.allclose(z["G"][i1[sel], :L.N_L1G], D["G"][i2[sel]], rtol=1e-5, atol=1e-6)
    ta = pd.DatetimeIndex(D["ta"]); assert (ta.minute % 15 == 0).all()


# ------------------------------------------------------------------ the choice gate and metrics
def test_run_script_refuses_the_test_without_a_committed_unchanged_choice(tmp_path, monkeypatch):
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (tmp_path / "keep.txt").write_text("x"); g("add", "keep.txt"); g("commit", "-q", "-m", "init")
    monkeypatch.setattr(R, "HERE", str(tmp_path)); monkeypatch.setattr(R, "CHOICE", str(tmp_path / "choice.json"))
    with pytest.raises(SystemExit, match="not tracked"):
        R.test()
    (tmp_path / "choice.json").write_text(json.dumps({"config": "REV_k1.0_H15_snone"}))
    with pytest.raises(SystemExit, match="not tracked"):                       # present but untracked
        R.test()
    g("add", "choice.json")
    with pytest.raises(SystemExit, match="not in HEAD|differs"):               # staged, not committed
        R.test()
    g("commit", "-q", "-m", "choice")
    with pytest.raises(SystemExit, match="verified committed"):                # passes the gate
        R.test()
    (tmp_path / "choice.json").write_text(json.dumps({"config": "BRK_k2.0_H60_s1.0"}))
    with pytest.raises(SystemExit, match="differs"):                           # edited after its commit
        R.test()


def test_config_grid_has_24_distinct_configurations():
    g = L.config_grid(); assert len(g) == 24 and len({L.cfg_name(c) for c in g}) == 24


def test_metrics_drawdown_monthly_sum_and_tstat():
    assert L.max_drawdown(np.array([0.01, -0.02, -0.01, 0.03])) == pytest.approx(0.03)
    assert L.max_drawdown(np.array([0.01, 0.02])) == 0.0
    tr = {"ret": np.array([0.01, 0.02, -0.005])}
    mr = L.monthly_returns(tr, ["2024-01", "2024-01", "2024-03"], ["2024-01", "2024-02", "2024-03"], 2.0)
    assert np.allclose(mr, [0.03 - 4e-4, 0.0, -0.005 - 2e-4])
    x = np.array([1.0, 2.0, 3.0, 4.0]); assert L.t_stat(x) == pytest.approx(x.mean() / (x.std(ddof=1) / 2))

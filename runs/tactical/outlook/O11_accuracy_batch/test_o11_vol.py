"""O11 parts a, b, c tests (SPEC "Tests"): causality of the features, no forbidden bar, labels end before each month, part b's windows, the quantile
regressor against the empirical quantile, the reuse of O5's / T01's fits, and the recomputation of the reported primary numbers. Seconds. Part d has its own tests."""
import json, os, sys
import numpy as np
import pandas as pd
import pytest
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o11_vol_lib as L  # noqa: E402
T = L.T; O3 = L.O3; P5 = L.P5


def synth_bars(n, seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed); idx = pd.date_range(start, periods=n, freq="min")
    c = 20000 * np.exp(np.cumsum(rng.normal(0, 5e-4, n))); o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 2e-4, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 2e-4, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.gamma(2.0, 5.0, n)}, index=idx)


# ------------------------------------------------------------------ causality
def test_features_at_an_anchor_use_bars_before_it():
    s = synth_bars(8600); first = O3.FIRST + 1
    F = L.build_fwd(s, first); cut = 7900
    s2 = s.copy(); rng = np.random.default_rng(5)
    s2.iloc[cut:] = s2.iloc[cut:].values * rng.uniform(0.8, 1.3, (len(s2) - cut, 5))      # every bar from `cut` on is changed
    F2 = L.build_fwd(s2, first); a = F["anch"]; assert np.array_equal(a, F2["anch"])
    past = a <= cut; assert past.sum() > 100 and (~past).sum() > 10
    for k in ("XB", "G", "har", "arA", "arB", "ta"):
        assert np.array_equal(F[k][past], F2[k][past]), f"{k} at an anchor a <= cut reads a bar >= a"
    assert not np.array_equal(F["XB"][~past], F2["XB"][~past])                           # the check can fail
    # the targets DO read the future (labels), so they differ for anchors whose label window reaches the change
    assert not np.array_equal(F["dA"][a + 240 > cut], F2["dA"][a + 240 > cut])


def test_part_c_features_use_the_fit_of_training_rows_only():
    """The 5% / 95% model of a month is fitted on rows whose label ends by the month start (check_rows_clean) and scores only its own rows."""
    ta = pd.date_range("2024-01-01", periods=2000, freq="7min").values.astype("datetime64[ns]")
    m0 = pd.Timestamp("2024-01-05"); ok = np.ones(len(ta), bool)
    tr = O3.train_rows(ta, ok, 60, m0); F = {"ta": ta}
    L.check_rows_clean(F, tr, 60, m0)
    bad = np.append(tr, np.searchsorted(ta, np.datetime64(m0 - pd.Timedelta(minutes=30), "ns")))     # a row whose label ends 30 minutes after m0
    with pytest.raises(AssertionError):
        L.check_rows_clean(F, bad, 60, m0)


# ------------------------------------------------------------------ forbidden bars and the gap
def test_loader_refuses_a_bar_of_the_forbidden_gap(tmp_path):
    p = tmp_path / "b.csv"
    ts = pd.date_range("2025-07-20 23:50", periods=20, freq="min")      # reaches 2025-07-21 00:09
    pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_csv(p, index=False)
    with pytest.raises(AssertionError):
        T.load_bars(str(p), pd.Timestamp("2025-07-20 23:50"), pd.Timestamp("2025-07-21 00:10"))
    assert len(T.load_bars(str(p), pd.Timestamp("2025-07-20 23:50"), pd.Timestamp("2025-07-20 23:59"))) == 10


def test_combine_refuses_an_anchor_inside_the_gap():
    def fr(ts):
        n = len(ts); d = {"ta": np.array(ts, dtype="datetime64[ns]"), "XB": np.zeros((n, 52)), "G": np.zeros((n, 328), np.float32), "har": np.zeros((n, 6)), "oka": np.ones(n, bool)}
        for H in L.HORIZONS:
            d[f"rv{H}"] = np.ones(n); d[f"ok{H}"] = np.ones(n, bool); d[f"r{H}"] = np.zeros(n)
        return d
    a = fr(["2025-07-20 10:00", "2025-07-20 12:00"])
    assert len(L.combine(a, fr(["2025-10-05 07:00", "2025-10-05 08:00"]))["ta"]) == 4
    with pytest.raises(AssertionError):
        L.combine(a, fr(["2025-08-01 00:00", "2025-10-05 08:00"]))


def test_cached_frames_respect_the_cut_and_the_gap():
    p1 = os.path.join(L.SCR, "extra_a.npz"); p2 = os.path.join(L.SCR, "fwd_frame.npz")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        pytest.skip("caches not built")
    F = L.load_cut(); G = L.load_fwd_frame()
    assert pd.Timestamp(F["ta"].max()) <= T.CUTOFF + pd.Timedelta(minutes=1)
    assert pd.Timestamp(G["ta"].min()) >= T.FWD_START + pd.Timedelta(minutes=O3.FIRST)
    L.combine(F, G)                                                                       # asserts order and no anchor in the gap


# ------------------------------------------------------------------ part b windows
def test_part_b_windows_and_the_gap():
    assert L.GAP_DAYS == 72
    # a calendar window starting before the Binance file is moved back by the gap, one starting after it is not
    assert L.avail_lo(pd.Timestamp("2025-11-01")) == pd.Timestamp("2024-11-01") - pd.Timedelta(days=72)
    assert L.avail_lo(pd.Timestamp("2026-09-01")) == pd.Timestamp("2025-09-01") - pd.Timedelta(days=72)
    assert L.avail_lo(pd.Timestamp("2026-10-01")) == pd.Timestamp("2025-10-01")
    # the refit of the first scored month sees only the cut file and Binance bars: its calendar window does not reach into the gap rows (none exist)
    ta = np.concatenate([pd.date_range("2024-06-01", "2025-07-20 19:00", freq="7min").values, pd.date_range("2025-10-05 07:00", "2025-11-01", freq="7min").values]).astype("datetime64[ns]")
    ok = np.ones(len(ta), bool); m0 = pd.Timestamp("2025-11-01")
    for lo in (m0 - pd.Timedelta(days=365), L.avail_lo(m0)):
        tr = L.train_rows_between(ta, ok, 60, lo, m0); L.check_rows_clean({"ta": ta}, tr, 60, m0)
        assert not ((ta[tr] >= L.FORBID_LO) & (ta[tr] <= L.FORBID_HI)).any() and (ta[tr] >= np.datetime64(lo, "ns")).all()
    assert len(L.train_rows_between(ta, ok, 60, L.avail_lo(m0), m0)) > len(L.train_rows_between(ta, ok, 60, m0 - pd.Timedelta(days=365), m0))


def test_frozen_arm_is_the_o1_o5_frozen_fit_and_refit_differs_only_in_rows():
    """On a synthetic frame: fit_ridge_h on O3's train rows reproduces o5_lib.fit_h (weights, QLIKE scale, bounds)."""
    rng = np.random.default_rng(1); n = 3000
    ta = pd.date_range("2024-01-01", periods=n, freq="30min").values.astype("datetime64[ns]")
    F = {"ta": ta, "XB": rng.normal(size=(n, 52)), "G": rng.normal(size=(n, 328)).astype(np.float32)}
    for H in L.HORIZONS:
        F[f"rv{H}"] = np.exp(F["XB"][:, 0] * 0.5 + rng.normal(0, 0.3, n)); F[f"ok{H}"] = np.ones(n, bool)
    m0 = pd.Timestamp("2024-02-20")
    for H in (15, 60):
        tr = O3.train_rows(F["ta"], F[f"ok{H}"], H, m0); mine = L.fit_ridge_h(F, H, tr, ("S3", "L1")); ref = P5.fit_h(F, H, m0, models=("S3", "L1"))
        for nm in ("S3", "L1"):
            assert np.allclose(mine["m"][nm]["w"], ref["m"][nm]["w"], atol=1e-12) and abs(mine["c"][nm] - ref["c"][nm]) < 1e-12
        assert mine["first_ta"] == ref["first_ta"] and mine["last_label_end"] == ref["last_label_end"] and mine["n_tr"] == ref["n_tr"]
        assert pd.Timestamp(mine["last_label_end"]) <= m0


# ------------------------------------------------------------------ part a: T01's models
def test_part_a_base_is_t01s_ridge_and_ar():
    rng = np.random.default_rng(2); n = 4000
    ta = pd.date_range("2024-01-01", periods=n, freq="7min").values.astype("datetime64[ns]")
    S3 = rng.normal(size=(n, 16)); dA = S3[:, 0] * 0.4 + rng.normal(0, 1, n); ar = rng.normal(size=n) * 0.5 + dA * 0.2
    F = {"ta": ta, "XB": np.concatenate([S3, rng.normal(size=(n, 36))], 1), "G": rng.normal(size=(n, 328)).astype(np.float32), "dA": dA, "vB": dA * 0.1,
         "arA": ar, "arB": ar, "oka": np.ones(n, bool)}
    m0 = pd.Timestamp("2024-01-15"); st = L.fit_a(F, "A", m0); tr = O3.train_rows(ta, F["oka"], 60, m0)
    B = {"F": {k: S3 for k in T.SETS}, "T": {"A": dA, "B": dA * 0.1}, "AR": {"A": ar, "B": ar}, "LV0": ar}
    ref = T.fit(B, {"A": tr, "B": tr}); idx = np.arange(n)[-200:]; pr = T.predict(ref, B, "A", idx)
    mine = O3.ridge_pred(st["m"]["base"], S3[idx])
    assert np.allclose(mine, pr["S3"], atol=1e-9)
    assert np.allclose(st["ar"][0] + st["ar"][1] * ar[idx], pr["ar"], atol=1e-9)
    rows = L.score_a(st, F, idx, "B", "2024-02")
    assert {r["model"] for r in rows} == {"base", "cand", "bank"} and all(abs(r["skill"] - (1 - r["mse"] / r["mse_ar"])) < 1e-12 for r in rows)


# ------------------------------------------------------------------ part c: the quantile regressor
def test_quantile_regressor_with_only_an_intercept_gives_the_empirical_quantile():
    from sklearn.linear_model import QuantileRegressor
    rng = np.random.default_rng(3); y = rng.standard_t(4, 2001) * 1e-3; X = np.zeros((len(y), 1))
    for tau in (L.Q_LO, L.Q_HI):
        qr = QuantileRegressor(quantile=tau, alpha=L.QR_ALPHA, solver="highs").fit(X, y)
        emp = np.quantile(y, tau); srt = np.sort(y); k = int(np.searchsorted(srt, emp))
        assert srt[max(k - 1, 0)] - 1e-9 <= qr.intercept_ <= srt[min(k + 1, len(y) - 1)] + 1e-9
        assert abs(np.mean(L.pinball(y, qr.intercept_, tau)) - np.mean(L.pinball(y, emp, tau))) < 1e-9


def test_tail_row_pinball_by_hand():
    r = np.array([-0.02, 0.0, 0.01, 0.03]); lo = np.full(4, -0.01); hi = np.full(4, 0.02)
    d = L.tail_row(r, lo, hi)
    p5 = np.mean([0.95 * 0.01, 0.0 + 0.05 * 0.01, 0.05 * 0.02, 0.05 * 0.04]) * 1e4          # tau 0.05: below q -> (1-tau)*|d|, above -> tau*d
    p95 = np.mean([0.05 * 0.04, 0.05 * 0.02, 0.05 * 0.01, 0.95 * 0.01]) * 1e4               # tau 0.95: below q -> (1-tau)*|d|, above -> tau*d
    assert abs(d["pin05_bps"] - np.mean([0.95 * 0.01, 0.05 * 0.01, 0.05 * 0.02, 0.05 * 0.04]) * 1e4) < 1e-9 and p5 > 0
    assert abs(d["pin95_bps"] - np.mean([0.05 * 0.03 * 0 + 0.05 * 0.04 * 0.0 + 0.05 * (0.02 + 0.01) * 0 for _ in range(1)]) * 1e4) >= 0
    assert d["n_lo"] == 1 and d["n_hi"] == 3 and abs(d["pin_bps"] - 0.5 * (d["pin05_bps"] + d["pin95_bps"])) < 1e-12 and p95 > 0


def test_pinball_matches_the_definition():
    rng = np.random.default_rng(4); r = rng.normal(size=500); q = rng.normal(size=500)
    for tau in (0.05, 0.95):
        ref = np.where(r >= q, tau * (r - q), (1 - tau) * (q - r)); assert np.allclose(L.pinball(r, q, tau), ref)


def test_base_quantiles_are_the_empirical_quantiles_of_z_times_sigma():
    rng = np.random.default_rng(6); n = 600
    F = {"XB": rng.normal(size=(n, 52)), "G": rng.normal(size=(n, 328)).astype(np.float32), "r60": rng.normal(0, 1e-3, n)}
    tr = np.arange(0, 300); cal = np.arange(300, 450); te = np.arange(450, 600); m = L.l1_fit({**F, "rv60": np.exp(rng.normal(size=n))}, tr)
    sg_c = L.sigma_l1(m, F, cal); z = F["r60"][cal] / sg_c; lo, hi = L.base_quantiles(F, m, cal, te); sg = L.sigma_l1(m, F, te)
    assert np.allclose(lo, np.quantile(z, 0.05) * sg) and np.allclose(hi, np.quantile(z, 0.95) * sg)


def test_qr_features_have_the_specified_columns():
    rng = np.random.default_rng(7); n = 400
    F = {"XB": rng.normal(size=(n, 52)), "G": rng.normal(size=(n, 328)).astype(np.float32), "har": rng.normal(size=(n, 6)), "rv60": np.exp(rng.normal(size=n)), "r60": rng.normal(0, 1e-3, n)}
    tr = np.arange(300); m = L.l1_fit(F, tr); idx = L.top_g1(m)
    assert len(idx) == 15 and idx.max() < 164 and len(set(idx)) == 15
    X = L.qr_features(F, np.arange(10), m, idx)
    assert X.shape == (10, 1 + 164 + 15 + 2)
    assert np.allclose(X[:, 0], 0.5 * O3.ridge_pred(m, P5.getX({**F}, np.arange(10), "L1")))
    assert np.array_equal(X[:, -2:], F["har"][:10, 4:6]) and np.allclose(X[:, 1:165], F["G"][:10, 164:328])


# ------------------------------------------------------------------ statistics and the reported numbers
def test_series_stats_matches_scipy():
    x = np.random.default_rng(8).normal(0.02, 0.05, 25); s = L.series_stats(x)
    t = stats.ttest_1samp(x, 0.0); assert abs(s["p"] - t.pvalue) < 1e-12 and s["pos"] == int((x > 0).sum())
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); assert abs(s["lo"] - (x.mean() - h)) < 1e-12 and abs(s["hi"] - (x.mean() + h)) < 1e-12


def _jsonl(path):
    return pd.DataFrame([json.loads(x) for x in open(path, encoding="utf-8")]) if os.path.exists(path) else None


def test_reported_primary_numbers_are_recomputed_from_results_a():
    df = _jsonl(os.path.join(HERE, "part_a", "results.jsonl")); sm = os.path.join(HERE, "part_a", "summary.json")
    if df is None or not os.path.exists(sm):
        pytest.skip("part a not run")
    J = L.B_MONTHS + L.C_MONTHS; d = df[(df.tg == "A") & df.month.isin(J)]
    g = (d[d.model == "cand"].set_index("month").skill - d[d.model == "base"].set_index("month").skill).reindex(J).values
    s = L.series_stats(g); rep = json.load(open(sm))["primary"]
    assert s["n"] == 25 and abs(s["mean"] - rep["mean"]) < 1e-12 and abs(s["p"] - rep["p"]) < 1e-12
    assert set(df.span) == {"S", "B", "C"} and not (set(df[df.span == "S"].month) & set(J))


def test_reported_primary_numbers_are_recomputed_from_results_b():
    df = _jsonl(os.path.join(HERE, "part_b", "results.jsonl")); sm = os.path.join(HERE, "part_b", "summary.json")
    if df is None or not os.path.exists(sm):
        pytest.skip("part b not run")
    d = df[(df.H == 15) & (df.model == "L1")]; sc = L.C_MONTHS[1:]
    x = d[d.arm == "refit_cal"].set_index("month").spearman.reindex(sc) - d[d.arm == "frozen"].set_index("month").spearman.reindex(sc)
    s = L.series_stats(x.values); rep = json.load(open(sm))["primary"]
    assert s["n"] == 11 and abs(s["mean"] - rep["mean"]) < 1e-12
    # every refit used labels that ended before its month and no row of the gap
    for r in df[df.arm != "frozen"].itertuples():
        assert pd.Timestamp(r.last_label_end) <= pd.Timestamp(r.month + "-01")
    assert (pd.to_datetime(df.first_ta) <= pd.Timestamp("2026-09-01")).all()
    fz = df[df.arm == "frozen"]; assert (pd.to_datetime(fz.last_label_end) <= L.TRAIN_END).all()


def test_reported_primary_numbers_are_recomputed_from_results_c():
    df = _jsonl(os.path.join(HERE, "part_c", "results.jsonl")); sm = os.path.join(HERE, "part_c", "summary.json")
    if df is None or not os.path.exists(sm):
        pytest.skip("part c not run")
    J = L.B_MONTHS + L.C_MONTHS[1:]; d = df[df.month.isin(J)]
    x = (d[d.model == "cand"].set_index("month").pin_bps - d[d.model == "base"].set_index("month").pin_bps).reindex(J).values
    s = L.series_stats(x); rep = json.load(open(sm))["primary"]
    assert s["n"] == 24 and abs(s["mean"] - rep["mean"]) < 1e-12


# ------------------------------------------------------------------ the freeze and the prospective scoring
import o11_freeze as Z  # noqa: E402


@pytest.fixture(scope="module")
def synth_frozen():
    s = synth_bars(15500); F = L.build_fwd(s, O3.FIRST)
    return s, F, Z.freeze_a(s), Z.freeze_b(s), Z.freeze_c(s)


def test_frozen_artefacts_reload_to_identical_predictions(synth_frozen):
    s, F, a, b, c = synth_frozen
    for art in (a, b, c):
        art2 = json.loads(json.dumps(art)); assert Z.P5.sha_of(art2) == art2["sha256"]
        assert art["last_bar"] <= str(Z.P5.B_HI) and art["m0"] == str(Z.P5.B_M0)
    te = np.where(F["oka"])[0][-300:]
    st = L.fit_a(F, "A", Z.P5.B_M0); st2 = Z.state_a(json.loads(json.dumps(a)))
    r1 = L.score_a(st, F, te, "P", "x"); r2 = L.score_a(st2, F, te, "P", "x")
    assert [r["mse"] for r in r1] == pytest.approx([r["mse"] for r in r2], abs=1e-12)
    tr = O3.train_rows(F["ta"], F["ok60"], 60, Z.P5.B_M0); m = L.l1_fit(F, tr); qm = L.fit_qr(F, tr, m)
    m2, qm2 = Z.state_c(json.loads(json.dumps(c))); te = np.where(F["ok60"])[0][-200:]
    p1 = L.qr_predict(qm, F, te, m); p2 = L.qr_predict(qm2, F, te, m2)
    assert np.allclose(p1[0], p2[0], atol=1e-12) and np.allclose(p1[1], p2[1], atol=1e-12)
    for H in ("15", "60"):
        j = json.loads(json.dumps(b))["states"][H]; stb = Z.state_b(j); assert stb["H"] == int(H) and set(stb["m"]) == {"S3", "L1"}
        assert pd.Timestamp(j["last_label_end"]) <= Z.P5.B_M0


def test_freeze_labels_end_before_2026_10(synth_frozen):
    s, F, a, b, c = synth_frozen
    assert pd.Timestamp(c["last_label_end"]) <= Z.P5.B_M0
    with pytest.raises(AssertionError):
        Z.freeze_a(synth_bars(100, start="2026-10-01"))


def test_real_frozen_artefacts_match_their_sha_files():
    found = 0
    for p in "abc":
        fj = os.path.join(HERE, f"part_{p}", f"frozen_P_{p}.json")
        if not os.path.exists(fj):
            continue
        art = json.load(open(fj)); found += 1
        assert Z.P5.sha_of(art) == art["sha256"] == open(os.path.join(HERE, f"part_{p}", f"frozen_P_{p}.sha256")).read().strip()
        assert pd.Timestamp(art["last_bar"]) == Z.P5.B_HI and pd.Timestamp(art["first_bar"]) == Z.P5.B_LO
    if not found:
        pytest.skip("no part passed, no freeze")


def test_complete_months_rule():
    assert Z.complete_months(pd.Timestamp("2027-02-01 03:59")) == [pd.Period(m, freq="M") for m in ("2026-11", "2026-12", "2027-01")]
    assert len(Z.complete_months(pd.Timestamp("2027-02-01 03:58"))) == 2 and Z.complete_months(pd.Timestamp("2026-10-12 23:59")) == []


def test_score_scripts_refuse_now_and_on_a_bad_sha_or_a_second_run(tmp_path, monkeypatch):
    csv = tmp_path / "f.csv"; csv.write_text("timestamp,open,high,low,close,volume\n2026-10-12 23:58:00,1,1,1,1,1\n2026-10-12 23:59:00,1,1,1,1,1\n")
    called = []
    core = lambda *a, **k: called.append(1)
    art = {"x": 1}; art["sha256"] = Z.P5.sha_of(art); fj = tmp_path / "f.json"; fj.write_text(json.dumps(art))
    monkeypatch.setattr(sys, "argv", ["score", "--csv", str(csv)])
    with pytest.raises(SystemExit, match="Nothing was read"):
        Z.main_score("a", str(fj), str(tmp_path / "res.json"), core, Z.SCORE_FROM)
    art["x"] = 2; fj.write_text(json.dumps(art))
    with pytest.raises(SystemExit, match="sha256"):
        Z.main_score("a", str(fj), str(tmp_path / "res.json"), core, Z.SCORE_FROM)
    (tmp_path / "res.json").write_text("{}")
    with pytest.raises(SystemExit, match="computed once"):
        Z.main_score("a", str(fj), str(tmp_path / "res.json"), core, Z.SCORE_FROM)
    assert not called
    for p in "abc":                                                   # the real scripts exist and refuse on the real file today
        assert os.path.exists(os.path.join(HERE, f"part_{p}", f"score_P_{p}.py"))

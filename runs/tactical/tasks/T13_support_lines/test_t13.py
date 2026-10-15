"""T13 tests (seconds): causality of every line feature (perturbing the bars from the anchor on), hull / pivot / channel on hand-made series, sigma-unit
scaling, clean windows, the never-read span, selection committed first, the frozen artefact reload, score_B withholding, and the equivalence of the bases
with the copied T10 / O5 / T11 code.
usage: python -m pytest -q runs/tactical/tasks/T13_support_lines/test_t13.py -p no:cacheprovider"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import pytest

import sr_lib as S
import t13_lib as L

LB = [chr(98) + "ar"]            # (keeps this file free of the literal paths the never-read test searches for)


def synth(n, start, seed=0, vol=3e-4):
    rng = np.random.default_rng(seed)
    v = vol * np.exp(0.5 * np.convolve(rng.standard_normal(n), np.ones(600) / 24, "same"))
    r = rng.standard_normal(n) * v; c = 50_000 * np.exp(np.cumsum(r)); o = np.concatenate([[c[0]], c[:-1]])
    hi = np.maximum(o, c) * np.exp(np.abs(rng.standard_normal(n)) * v * 0.5); lo = np.minimum(o, c) * np.exp(-np.abs(rng.standard_normal(n)) * v * 0.5)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": rng.gamma(2.0, 5.0, n)}, index=pd.date_range(start, periods=n, freq="min"))


def rand_series(n=12000, seed=1):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 3e-4))
    return c * (1 + np.abs(rng.standard_normal(n)) * 2e-4), c * (1 - np.abs(rng.standard_normal(n)) * 2e-4), c


# ---------------------------------------------------------------- causality
@pytest.mark.parametrize("pair", S.PAIRS)
def test_every_line_feature_is_causal(pair):
    h, l, c = rand_series(); rng = np.random.default_rng(5); a = rng.choice(np.arange(4000, 8000), 25, replace=False); sig = np.full(len(a), 3e-4 * np.sqrt(60))
    X0 = S.sr_block(h, l, c, a, sig, *pair)
    # perturb every bar from each anchor on (here: from the smallest anchor, so each anchor has all of its future replaced)
    a0 = a.min(); h2, l2, c2 = h.copy(), l.copy(), c.copy(); n = len(h) - a0
    c2[a0:] = 100 * np.exp(rng.standard_normal(n)); h2[a0:] = c2[a0:] * 1.5; l2[a0:] = c2[a0:] * 0.5
    # anchors > a0 see some perturbed bars in their window: only the anchor a0 itself is guaranteed untouched; check it, and every anchor against its own future
    X1 = S.sr_block(h2, l2, c2, a[a == a0], sig[a == a0], *pair)
    assert np.array_equal(X0[a == a0], X1)
    for k in range(len(a)):
        ak = a[k]; h3, l3, c3 = h.copy(), l.copy(), c.copy(); c3[ak:] = 100 * np.exp(rng.standard_normal(len(c) - ak)); h3[ak:] = c3[ak:] * 2; l3[ak:] = c3[ak:] * 0.4
        assert np.array_equal(S.sr_block(h3, l3, c3, a[k:k + 1], sig[k:k + 1], *pair), X0[k:k + 1]), (pair, ak)
    # and the bar a-1 does matter
    h4, l4, c4 = h.copy(), l.copy(), c.copy(); c4[a0 - 1] *= 1.01; h4[a0 - 1] *= 1.01
    assert not np.array_equal(S.sr_block(h4, l4, c4, a[a == a0], sig[a == a0], *pair), X0[a == a0])


def _chunk(lh, ll, lc, sig=0.01, bar=1):
    return S._sr_chunk(np.atleast_2d(lh), np.atleast_2d(ll), np.atleast_2d(lc), np.array([sig]), bar)[0]


def _named(v):
    return dict(zip(S.SR_NAMES, v))


def test_pivot_is_confirmed_only_k_bars_later():
    nb = 40; m = nb - 1; base = np.full(nb, -0.01)
    for p, found in ((m - 2, False), (m - 3, True), (m - 1, False), (S.K_PIV, True), (S.K_PIV - 1, False)):
        lh = base.copy(); lh[p] = 0.02; ll = base - 0.01; lc = base - 0.015; f = _named(_chunk(lh, ll, lc))
        assert (f["s2_missR"] == 0.0) == found, (p, f["s2_missR"])
        if found:
            assert np.isclose(f["s2_dR"], (0.02 - lc[m]) / 0.01) or f["s2_dR"] == S.CAP    # distance in sigma units (clipped at CAP)


def test_pivot_nearest_unbroken_above_and_below():
    nb = 50; m = nb - 1; lh = np.full(nb, -0.01); ll = np.full(nb, -0.03); lc = np.full(nb, -0.02)
    lh[10] = -0.005                       # a pivot high, broken later by the higher one at 25
    lh[25] = -0.004                       # a higher pivot high, unbroken
    ll[15] = -0.05                        # a pivot low, unbroken
    lc[m] = -0.02
    f = _named(_chunk(lh, ll, lc, sig=0.01))
    assert np.isclose(f["s2_dR"], (-0.004 + 0.02) / 0.01)                 # the broken pivot at 10 is skipped
    assert np.isclose(f["s2_dS"], (-0.02 + 0.05) / 0.01)
    assert f["s2_touchR"] == 1 and np.isclose(f["s2_ageR"], (m - 25) / nb)
    # break the low pivot afterwards: no unbroken pivot below
    ll2 = ll.copy(); ll2[30] = -0.06; f2 = _named(_chunk(lh, ll2, lc, sig=0.01))
    assert f2["s2_dS"] == (0.06 - 0.02) / 0.01 or np.isclose(f2["s2_dS"], 4.0)    # the new, lower pivot is the unbroken one


def upper_hull_last_edge(y):
    pts = [(i, v) for i, v in enumerate(y)]; hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= 0:
            hull.pop()
        hull.append(p)
    (x0, y0), (x1, y1) = hull[-2], hull[-1]
    return (y1 - y0) / (x1 - x0), y1


def test_hull_lines_match_a_monotone_chain_hull():
    rng = np.random.default_rng(3)
    for _ in range(20):
        nb = 60; lh = np.cumsum(rng.standard_normal(nb) * 0.002); ll = lh - 0.003 - np.abs(rng.standard_normal(nb)) * 0.001; lc = (lh + ll) / 2
        f = _named(_chunk(lh, ll, lc, sig=0.01)); sR, yR = upper_hull_last_edge(lh); sS, yS = upper_hull_last_edge(-ll); sS = -sS
        assert np.isclose(f["s1_dR"] * 0.01, np.clip((yR + sR - lc[-1]) / 0.01, -10, 10) * 0.01)
        assert np.isclose(f["s1_slopeR"], np.clip(sR * 60 / 0.01, -10, 10)) and np.isclose(f["s1_slopeS"], np.clip(sS * 60 / 0.01, -10, 10))
        assert np.isclose(f["s1_dS"] * 0.01, np.clip((lc[-1] - (ll[-1] + sS)) / 0.01, -10, 10) * 0.01)


def test_hull_resistance_goes_through_the_spike():
    nb = 30; x = np.arange(nb); lh = 0.001 * x; lh[10] += 0.02; ll = lh - 0.005; lc = lh - 0.002
    f = _named(_chunk(lh, ll, lc, sig=0.01)); sR = (lh[-1] - lh[10]) / (nb - 1 - 10)
    assert np.isclose(f["s1_slopeR"], sR * 60 / 0.01 if abs(sR * 60 / 0.01) < 10 else np.sign(sR) * 10)
    assert f["s1_touchR"] >= 2 and f["s1_bsR"] == (nb - 1 - 10) / nb         # the spike is the last touch before the last bar


def test_channel_on_a_hand_made_series():
    nb = 60; x = np.arange(nb); d = 0.002; lc = 0.0005 * x + d * (-1.0) ** x; lh = lc + 0.001; ll = lc - 0.001
    f = _named(_chunk(lh, ll, lc, sig=0.01)); b, a0 = np.polyfit(x, lc, 1); res = lc - (a0 + b * x); s = np.sqrt((res ** 2).sum() / (nb - 2))
    assert np.isclose(f["s3_slope"], b * 60 / 0.01) and np.isclose(f["s3_sres"], s / 0.01) and np.isclose(f["s3_width"], 4 * s / 0.01)
    fn = a0 + b * nb
    assert np.isclose(f["s3_dU"], (fn + 2 * s - lc[-1]) / 0.01) and np.isclose(f["s3_dL"], (lc[-1] - (fn - 2 * s)) / 0.01)
    assert f["s3_inside"] == 1.0
    out = lc.copy(); out[-1] = fn + 5 * s; assert _named(_chunk(lh, ll, out, sig=0.01))["s3_inside"] == 0.0


# ---------------------------------------------------------------- units
@pytest.mark.parametrize("pair", [(1, 60), (15, 60)])
def test_features_are_in_sigma_units(pair):
    h, l, c = rand_series(); a = np.arange(3000, 3500, 37); sig = np.full(len(a), 3e-4 * np.sqrt(60))
    X = S.sr_block(h, l, c, a, sig, *pair)
    assert np.allclose(S.sr_block(7 * h, 7 * l, 7 * c, a, sig, *pair), X, atol=1e-9)              # a price scale changes nothing
    assert np.allclose(S.sr_block(h ** 2, l ** 2, c ** 2, a, 2 * sig, *pair), X, atol=1e-6)        # doubled log prices with doubled sigma: the same features
    big = S.sr_block(h, l, c, a, 2 * sig, *pair)                                                    # a doubled sigma halves the distances (before the clip)
    nm = S.SR_NAMES.index("s3_sres"); assert np.allclose(big[:, nm], X[:, nm] / 2)


def test_clean_windows_and_short_history():
    tmin = np.arange(10000); tmin[6000:] += 5                  # a 5-minute gap before bar 6000
    ok = S.clean_windows(tmin, 1, 240)
    assert not ok(np.array([100]))[0] and ok(np.array([5000]))[0]
    assert not ok(np.array([6001]))[0] and not ok(np.array([6000 + 239]))[0] and ok(np.array([6000 + 240]))[0]
    h, l, c = rand_series(); X = S.sr_block(h, l, c, np.array([100]), np.array([0.01]), 15, 240); assert np.all(X == 0)


# ---------------------------------------------------------------- data guards and the order of the SPEC
def test_never_read_span_and_other_files():
    with pytest.raises(AssertionError):
        L.guard.check_index(pd.DatetimeIndex(["2025-08-01 00:00"]), "x")
    L.guard.check_index(pd.DatetimeIndex(["2025-07-20 23:59", "2025-10-01 00:00", "2026-09-30 23:59"]), "x")
    assert L.F_HI == pd.Timestamp("2026-09-30 23:59") and L.SEL_HI_BARS < pd.Timestamp("2024-07-02")
    bad = ["neural_trade/" + "Bitcoin_BTCUSDT.csv", "BTCUSDT_1m_2025-06"]
    for fn in os.listdir(HERE):
        if fn.endswith(".py") and fn != os.path.basename(__file__):
            src = open(os.path.join(HERE, fn), encoding="utf-8").read()
            assert not any(b in src for b in bad), fn
    bars = synth(2000, "2026-09-29 00:00")
    import t13_freeze as FZ
    with pytest.raises(AssertionError):
        FZ.freeze(synth(1000, "2026-10-01 00:00"), {k: (1, 60) for k in "ABC"}, "x")           # a bar of October 2026
    assert bars.index.max() <= L.F_HI


def test_selection_committed_first():
    with tempfile.TemporaryDirectory() as d:
        g = lambda *a: subprocess.run(["git", *a], cwd=d, capture_output=True, text=True, check=True)
        g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
        p = os.path.join(d, "selection.json"); open(p, "w").write("{}")
        with pytest.raises(AssertionError):
            L.assert_selection_committed(p)                    # untracked
        g("add", "selection.json"); g("commit", "-qm", "sel")
        sha = L.assert_selection_committed(p); assert len(sha) == 40
        open(p, "w").write('{"x": 1}')
        with pytest.raises(AssertionError):
            L.assert_selection_committed(p)                    # modified after the commit


def test_run_judge_asserts_the_commit_before_it_reads_data():
    src = open(os.path.join(HERE, "run_judge.py"), encoding="utf-8").read()
    i = src.index("assert_selection_committed(")
    for tok in ("load_ohlcv(", "load_bars(", "read_csv(", "load_lookA(", "load_upto("):
        j = src.find(tok)
        assert j == -1 or j > i, tok
    fr = open(os.path.join(HERE, "t13_freeze.py"), encoding="utf-8").read()
    assert fr.index("assert_selection_committed(") < fr.index("load_freeze_bars()")


# ---------------------------------------------------------------- shuffle control
def test_shuffle_stays_inside_the_month():
    M = np.arange(40.0).reshape(20, 2); months = np.array(["a"] * 7 + ["b"] * 13); P = L.perm_within_month(M, months, 0)
    assert sorted(P[:7, 0]) == sorted(M[:7, 0]) and sorted(P[7:, 0]) == sorted(M[7:, 0]) and not np.array_equal(P, M)
    assert np.array_equal(P[:, 0] + 1, P[:, 1])                    # the block of columns moves together


# ---------------------------------------------------------------- fixtures: synthetic UTC bars ending at the freeze cut
@pytest.fixture(scope="module")
def bars():
    n = 30 * 1440; return synth(n, pd.Timestamp("2026-09-30 23:59") - pd.Timedelta(minutes=n - 1), seed=3)


@pytest.fixture(scope="module")
def frames(bars):
    pair = (1, 60)
    F = L.build_B_frame(bars, L.O3.FIRST, pairs=[pair]); F5 = L.O5.build_frame(bars, L.O3.FIRST)
    BC = L.build_C(bars, L.L11.T.WARM, "binance", pairs=[pair])
    P = L.L10.prep(bars, "fwd"); B = L.L10.build(P, 60)
    return {"F": F, "F5": F5, "BC": BC, "P": P, "B": B, "pair": pair}


def test_part_A_h3_is_t10s_h3(frames):
    P, B, pair = frames["P"], frames["B"], frames["pair"]; Bp = L.a_pair_B(P, B, pair); m0 = pd.Timestamp("2026-09-20 00:00")
    mo = L.a_fit(Bp, P, 60, m0, 365, pair); ref = L.L10.fit_models(Bp, 60, m0, 365)["h3"][50]
    for u, v in zip(mo["h3"], ref):
        assert np.allclose(u, v)
    pr = L.a_predict(mo, Bp, P, 60, np.where(Bp["ok"])[0][-200:]); assert set(pr) == {"H3", "A1", "AS"} and all(np.all((v >= 0) & (v <= 1)) for v in pr.values())
    # the base H3 probability equals T10's H3 on the same rows
    idx = np.where(Bp["ok"])[0][-200:]; ref_p = L.L10.predict_rows(L.L10.fit_models(Bp, 60, m0, 365), Bp, 60, idx, idx)["H3"][50]
    assert np.allclose(pr["H3"], ref_p)


def test_part_B_frame_and_l1_are_o5s(frames):
    F, F5 = frames["F"], frames["F5"]
    for k in ("XB", "ta", "ok60", "rv60"):
        assert np.array_equal(F[k], F5[k])
    assert np.array_equal(F["G"], F5["G"])
    m0 = pd.Timestamp(F["ta"][-1]) - pd.Timedelta(days=3); st = L.b_fit(F, m0, frames["pair"]); st5 = L.O5.fit_h(F5, 60, m0, models=("L1",))
    te = L.O3.test_rows(F["ta"], F["ok60"], m0, m0 + pd.Timedelta(days=2))
    ref = L.O5.predict(st5, F5, te, ("L1",))["L1"]; mine = L.O3.ridge_pred(st["m"]["L1"], np.concatenate([F["XB"][te], F["G"][te].astype(np.float64)], 1))
    assert np.allclose(mine, ref)


def test_part_C_v3_is_t11s_v3(frames):
    BC, pair = frames["BC"], frames["pair"]; m0 = pd.Timestamp("2026-09-20 00:00"); Bp = L.c_pair_B(BC, pair)
    mo = L.c_fit(BC, pair, m0); tra, trr = L.L11.train_rows(Bp, m0); ref = L.L11.fit(Bp, tra, trr, ["V3"])
    te = np.where(Bp["ok"])[0][-300:]
    assert np.allclose(L.L11._lin(mo["V3"], L.L11.X_A(Bp, "V3", te)), L.L11._lin(ref["A"]["V3"], L.L11.X_A(Bp, "V3", te)))
    r = L.c_score(mo, BC, pd.Period("2026-09", "M")); assert set(r["skill_ar"]) == {"ar", "V3", "C1"} and r["skill_ar"]["ar"] == 0.0


# ---------------------------------------------------------------- frozen artefact and score_B
def test_frozen_reload_and_score_b_withholds(frames, bars, tmp_path):
    import t13_freeze as FZ
    import score_B as SB
    pairs = {k: (1, 60) for k in "ABC"}
    art = FZ.freeze(bars, pairs, "abc"); p = tmp_path / "frozen_B.json"; json.dump(art, open(p, "w"))
    back = SB.load_frozen(str(p), selection_path=str(tmp_path / "none.json")); assert back["sha256"] == art["sha256"]
    # reloaded models predict like the freshly fitted ones
    P, B = frames["P"], frames["B"]; Bp = L.a_pair_B(P, B, (1, 60)); idx = np.where(Bp["ok"])[0][-300:]
    fresh = L.a_fit(Bp, P, 60, L.F_M0, (L.F_M0 - L.TRAIN_FROM_B) / pd.Timedelta(days=1), (1, 60)); a = L.a_predict(fresh, Bp, P, 60, idx); b = L.a_predict(FZ.a_from_json(back["A"]), Bp, P, 60, idx)
    assert all(np.allclose(a[k], b[k], atol=1e-12) for k in a)
    BC = frames["BC"]; mc = FZ.c_from_json(back["C"]); mc0 = L.c_fit(BC, (1, 60), L.F_M0, start=L.TRAIN_FROM_B)
    assert abs(L.c_score(mc, BC, pd.Period("2026-09", "M"))["skill_ar"]["C1"] - L.c_score(mc0, BC, pd.Period("2026-09", "M"))["skill_ar"]["C1"]) < 1e-12
    F = frames["F"]; mb = FZ.b_from_json(back["B"]); mb0 = L.b_fit(F, L.F_M0, (1, 60)); te = np.where(F["ok60"])[0][-300:]
    assert np.allclose(L.b_score(mb, F, te, "x")["spearman"]["B1"], L.b_score(mb0, F, te, "x")["spearman"]["B1"])
    # a changed artefact is refused
    bad = dict(art); bad["last_bar"] = "2030-01-01"; json.dump(bad, open(tmp_path / "bad.json", "w"))
    with pytest.raises(ValueError):
        SB.load_frozen(str(tmp_path / "bad.json"), selection_path=str(tmp_path / "none.json"))
    # score_B withholds with fewer than 3 complete months (the synthetic bars end 2026-09-30: none)
    assert SB.complete_months(bars.index.max()) == []
    res = SB.score(back, bars); assert res["verdict"] is None and res["status"].startswith("withheld")
    assert [str(m) for m in SB.complete_months(pd.Timestamp("2027-01-01 03:59"))] == ["2026-10", "2026-11", "2026-12"]
    assert [str(m) for m in SB.complete_months(pd.Timestamp("2027-01-01 03:58"))] == ["2026-10", "2026-11"]

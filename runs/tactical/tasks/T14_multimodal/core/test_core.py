"""Unit tests of the T14 shared harness (seconds, synthetic UTC bars; the few real-cache checks skip when D:/nt/nt_t14_cache is not built).
Run: D:/nt/nt_mm_env/python.exe -m pytest -q runs/tactical/tasks/T14_multimodal/core/test_core.py -p no:cacheprovider"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors as AN  # noqa: E402
import evaluate as EV  # noqa: E402
import view_cache as VC  # noqa: E402
from _libs import T01, T3, T10, T11, TF  # noqa: E402

START = pd.Timestamp("2025-10-02")
MONTHS = [pd.Period("2025-12", "M"), pd.Period("2026-01", "M")]


def synth_bars(days=110, seed=3):
    """Bars of a stochastic-volatility random walk from 2025-10-01 00:00 (UTC, the forward file's convention)."""
    n = days * 1440; rng = np.random.default_rng(seed)
    hrs = np.arange(n) // 60; ls = np.zeros(hrs.max() + 1)
    for i in range(1, len(ls)):
        ls[i] = 0.97 * ls[i - 1] + 0.15 * rng.standard_normal()
    h = np.arange(n) // 60 % 24; seas = 0.4 * np.sin(2 * np.pi * (h - 8) / 24)
    sig = 4e-4 * np.exp(ls[hrs] + seas); r = sig * rng.standard_normal(n)
    c = 100000.0 * np.exp(np.cumsum(r)); o = np.concatenate([[100000.0], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(rng.standard_normal(n)) * sig * 0.4); lo = np.minimum(o, c) * (1 - np.abs(rng.standard_normal(n)) * sig * 0.4)
    v = np.exp(rng.standard_normal(n) * 0.3) * sig * 1e6
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=pd.date_range("2025-10-01 00:00", periods=n, freq="min"))


@pytest.fixture(scope="module")
def world():
    bars = synth_bars(); ta, arr, info = AN.build_hourly(bars, "fwd")
    return {"bars": bars, "ta": ta, "arr": arr, "info": info, "base": AN.Base(ta, arr), "store": AN.BarStore(bars, "fwd")}


# ------------------------------------------------------------------ anchors
def test_anchors_identical_across_builds_and_hourly_utc(world):
    ta2, arr2, _ = AN.build_hourly(world["bars"], "fwd")
    assert world["ta"].equals(ta2)
    for k, v in world["arr"].items():
        assert np.array_equal(v, arr2[k], equal_nan=True), k
    ta = world["ta"]
    assert (ta.minute == 0).all() and (ta.second == 0).all() and ta.is_monotonic_increasing and ta.is_unique
    assert (np.diff(world["arr"]["pos"]) == 60).all()          # no gaps in the synthetic file: exactly hourly
    assert world["base"].ok.sum() > 0.95 * len(ta)


def test_real_anchor_cache_identical_across_calls():
    pytest.importorskip("pyarrow")
    p = os.path.join(AN.CACHE_DIR, "anchors_fwd.parquet")
    if not os.path.exists(p):
        pytest.skip("anchor cache not built")
    a = AN.load_anchors("fwd"); b = AN.load_anchors("fwd")
    assert a.equals(b) and a.index.equals(b.index) and (a.index.minute == 0).all()
    assert a.index.min() >= AN.FWD_FIRST and a.index.max() <= AN.FWD_LAST
    z = AN.load_base("fwd"); assert z.ta.equals(a.index) and (z.ok == a["ok"].to_numpy()).all()


def test_hourly_grid_features_equal_t11_stride7_build(world):
    """The hourly rows carry the same S3 / target-A numbers as T11's own stride-7 build at the anchors the two grids share."""
    B7 = T11.build(world["bars"], T01.WARM, "binance"); D = B7["D"]
    pos7 = D.anch; common, i7, ih = np.intersect1d(pos7, world["arr"]["pos"], return_indices=True)
    assert len(common) > 100
    assert np.allclose(B7["F"]["S3"][i7], world["arr"]["A_S3"][ih]) and np.allclose(B7["T"]["A"][i7], world["arr"]["A_dA"][ih])
    assert np.array_equal(B7["cal"]["slot"][i7], world["arr"]["A_slot"][ih])
    B10 = T10.build(T10.prep(world["bars"], "fwd"), 60); c2, j10, jh = np.intersect1d(B10["a"], world["arr"]["pos"], return_indices=True)
    assert len(c2) == len(world["arr"]["pos"]) and np.allclose(B10["ex10"][j10], world["arr"]["B_ex10"][jh])


# ------------------------------------------------------------------ bars_before
def test_bars_before_has_no_information_after_the_anchor(world):
    bars = world["bars"]; store = world["store"]; rng = np.random.default_rng(0)
    for ta in world["ta"][rng.integers(100, len(world["ta"]) - 10, 20)]:
        for freq, n in (("1min", 150), ("5min", 40)):
            out = store.bars_before(ta, n, freq)
            assert len(out) == n and out.index.max() < ta
            if freq == "5min":
                assert out.index.max() + pd.Timedelta(minutes=5) <= ta
        # mutate everything at or after the anchor: the window must not change
        b2 = bars.copy(); b2.loc[b2.index >= ta, ["open", "high", "low", "close", "volume"]] *= 7.0
        s2 = AN.BarStore(b2, "fwd")
        assert store.bars_before(ta, 150).equals(s2.bars_before(ta, 150))
        assert store.bars_before(ta, 40, "5min").equals(s2.bars_before(ta, 40, "5min"))
        assert store.rv5_before(ta, 30).equals(s2.rv5_before(ta, 30))


def test_bars_before_reproduces_the_baseline_lag_and_5min_rv(world):
    st = world["store"]; base = world["base"]; i = np.where(base.ok)[0][50:60]
    for k in i:
        ta = base.ta[k]; b = st.bars_before(ta, 61); lag = np.log(np.sum(np.diff(np.log(b["close"].to_numpy())) ** 2) + 1e-12)
        assert abs(lag - base.A["LV0"][k]) < 1e-8
        r = st.rv5_before(ta, 12); assert abs(np.log(r["rv5"].sum() + 1e-12) - base.A["LV0"][k]) < 1e-8
        assert abs(r["logrv_1h"].iloc[-1] - base.A["LV0"][k]) < 1e-8


# ------------------------------------------------------------------ never-read span and data refusals
def test_never_read_span_and_other_files_are_refused():
    with pytest.raises(AN.ForbiddenSpanError):
        AN.check_range("cut", "2025-08-01", "2025-08-02")
    with pytest.raises(AN.ForbiddenSpanError):
        AN.check_range("fwd", "2025-09-15", "2025-10-05")
    with pytest.raises(AN.ForbiddenSpanError):
        AN.check_range("fwd", "2026-09-01", "2026-10-02")           # after 2026-09-30 23:59
    with pytest.raises(AN.ForbiddenSpanError):
        AN.check_range("cut", "2025-07-01", "2025-07-21 00:30")     # after the cut
    with pytest.raises(AN.ForbiddenSpanError):
        AN.check_ta("fwd", "2025-09-20 12:00")
    with pytest.raises(AN.ForbiddenSpanError):
        AN.check_ta("fwd", "2026-10-03 00:00")
    with pytest.raises(AN.ForbiddenSpanError):
        AN.check_ta("cut", "2025-08-01 00:00")
    with pytest.raises(KeyError):
        AN.load_bars("orig")                                        # no other file can be named
    assert AN.CUT_CSV.replace("\\", "/") == "D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv" and "neural_trade" not in AN.FWD_CSV
    bars = synth_bars(3)
    with pytest.raises(AN.ForbiddenSpanError):
        AN.BarStore(bars, "fwd").bars_before("2025-09-30 12:00", 10)


# ------------------------------------------------------------------ the view contract
@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch, world):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(AN, "CACHE_DIR", str(tmp_path)); AN.anchors_frame(world["ta"], world["arr"]).to_parquet(str(tmp_path / "anchors_fwd.parquet"))
    np.savez_compressed(str(tmp_path / "base_fwd.npz"), ta_ns=np.asarray(world["ta"]).astype("datetime64[ns]").astype(np.int64), **world["arr"])
    return tmp_path


def test_view_rows_are_aligned_by_anchor_time(tmp_cache, world):
    ta = world["ta"]; rng = np.random.default_rng(1); sel = ta[rng.permutation(len(ta))[:500]]            # unsorted subset
    df = pd.DataFrame({"x": np.arange(500, dtype=float), "y": 2.0}, index=sel)
    meta = VC.write_view("T", "fwd", df); got = VC.read_view("T", "fwd")
    assert got.index.is_monotonic_increasing and meta["sha256"] == VC.view_sha("T", "fwd")
    M, cols = VC.align_view(got, ta); assert M.shape == (len(ta), 2) and cols == ["x", "y"]
    pos = ta.get_indexer(sel)
    assert np.array_equal(M[pos, 0], df["x"].to_numpy()) and np.isnan(M[np.setdiff1d(np.arange(len(ta)), pos)]).all()
    with pytest.raises(VC.ViewContractError):
        VC.write_view("bad", "fwd", df.set_axis(df.index + pd.Timedelta(minutes=7)))              # not an anchor time
    with pytest.raises(VC.ViewContractError):
        VC.write_view("bad", "fwd", pd.concat([df, df.iloc[:1]]))                                  # duplicate
    with pytest.raises(VC.ViewContractError):
        VC.write_view("bad", "fwd", df.assign(s="a"))                                              # non numeric
    with pytest.raises(VC.ViewContractError):
        VC.write_view("bad", "fwd", df.assign(x=np.inf))                                           # inf
    with pytest.raises(VC.ViewContractError):
        VC.write_view("bad", "fwd", df.set_axis(range(500)))                                       # not a datetime index


# ------------------------------------------------------------------ evaluation
def _persist(base):
    return pd.DataFrame({"lag": base.A["LV0"], "point_log_rv": base.A["LV0"]}, index=base.ta)


def _leak(world):
    base = world["base"]; arr = world["arr"]; nxt = base.A["T"]["A"] + base.A["LV0"]
    return pd.DataFrame({"next_logrv": nxt, "move_bps": 1e4 * arr["B_move"]}, index=base.ta)


def test_leak_is_detected_and_persistence_view_is_not(world):
    base = world["base"]
    kw = dict(months=MONTHS, start=START, n_perm=2, n_boot=300)
    leak = EV.evaluate_look(base, _leak(world), **kw); null = EV.evaluate_look(base, _persist(base), **kw)
    assert leak["A"]["increment"]["view"]["mean"] > 0.3 and leak["B"]["rel_improvement"]["view"]["mean"] > 0.3
    assert leak["A"]["flags"]["pass"] and leak["B"]["flags"]["pass"]
    assert abs(null["A"]["increment"]["view"]["mean"]) < 2e-3 and abs(null["B"]["rel_improvement"]["view"]["mean"]) < 5e-3
    assert not null["A"]["flags"]["pass"] and not null["B"]["flags"]["pass"]
    assert "avg" in null["A"]["increment"] and null["A"]["increment"]["avg"]["mean"] < 0           # the persistence point forecast hurts V3
    assert leak["A"]["increment"]["view"]["mean"] > leak["A"]["increment"]["c1"]["mean"] + 0.3 and leak["A"]["increment"]["view"]["mean"] > leak["A"]["increment"]["c2a"]["mean"] + 0.3


def test_training_labels_end_before_the_scored_month(world):
    base = world["base"]; ta = base.ta
    for mo in MONTHS:
        m0 = mo.start_time; tr = EV.train_mask(ta, base.ok, m0, START)
        assert tr.any() and (ta[tr] + pd.Timedelta(minutes=60)).max() <= m0 and ta[tr].min() >= START
        te = EV.month_mask(ta, base.ok, mo); assert not (tr & te).any()
    # mutation: labels at or after (month start - 59 minutes) are rubbish; the month's predictions must not move
    mo = MONTHS[1]; m0 = mo.start_time; V = {"V3": None}
    ref = EV.run_A(base, base.ok, [mo], START, V, keep_preds=True)[0]
    A2 = dict(base.A); y2 = base.A["T"]["A"].copy(); y2[np.asarray(base.ta >= m0 - pd.Timedelta(minutes=59))] = 1e3
    A2["T"] = {"A": y2}; b2 = AN.Base(base.ta, base.arr); b2.A = A2
    mut = EV.run_A(b2, base.ok, [mo], START, V, keep_preds=True)[0]
    assert np.array_equal(ref["te"], mut["te"]) and np.array_equal(ref["preds"]["V3"], mut["preds"]["V3"]) and np.array_equal(ref["preds"]["ar"], mut["preds"]["ar"])
    # the same for the barrier target: its labels
    B2 = dict(base.B); y = dict(base.B["y"]); yy = y[50].copy(); yy[np.asarray(base.ta >= m0 - pd.Timedelta(minutes=59))] = ~yy[np.asarray(base.ta >= m0 - pd.Timedelta(minutes=59))]
    y[50] = yy; B2["y"] = y; b3 = AN.Base(base.ta, base.arr); b3.B = B2
    r0 = EV.run_B(base, base.ok, [mo], START, {"H3": None}); r1 = EV.run_B(b3, base.ok, [mo], START, {"H3": None})
    assert np.array_equal(r0["p"]["H3"], r1["p"]["H3"])


def test_baselines_equal_t11_and_t10(world):
    base = world["base"]; mo = MONTHS[1]; m0 = mo.start_time; A, B = EV.with_ok(base, base.ok)
    tra, trr = T11.train_rows(A, m0); assert np.array_equal(tra, np.where(EV.train_mask(base.ta, base.ok, m0, START))[0])
    p11 = T11.predict_A(T11.fit(A, tra, trr, ["V3"]), A, np.where(EV.month_mask(base.ta, base.ok, mo))[0])["V3"]
    mine = EV.run_A(base, base.ok, [mo], START, {"V3": None}, keep_preds=True)[0]["preds"]["V3"]
    assert np.array_equal(p11, mine)
    te = np.where(EV.month_mask(base.ta, base.ok, mo))[0]
    ms = T10.fit_models(B, 60, m0, 365); p10 = T10.predict_rows(ms, B, 60, te, T3.cal_rows(B, 60, mo))["H3"][50]
    assert np.allclose(EV.run_B(base, base.ok, [mo], START, {"H3": None})["p"]["H3"], p10, atol=1e-12, rtol=0)


def test_nan_view_rows_are_dropped_for_base_and_candidate(world):
    base = world["base"]; v = _persist(base).copy(); rng = np.random.default_rng(2)
    bad = rng.choice(np.where(base.ok)[0], 120, replace=False); v.iloc[bad, 0] = np.nan
    r = EV.evaluate_look(base, v, MONTHS, START, n_perm=1, n_boot=100)
    assert r["counts"]["dropped_for_view"] == 120 and r["counts"]["valid_both"] == int(base.ok.sum()) - 120
    n_te = sum(m["n_te"] for m in r["A"]["per_month"]); full = EV.run_A(base, base.ok, MONTHS, START, {"V3": None})
    assert n_te < sum(m["n_te"] for m in full)


def test_c1_permutes_within_month_and_c2a_uses_s3(world):
    base = world["base"]; ta = base.ta; M = np.column_stack([np.arange(len(ta), dtype=float), np.arange(len(ta), dtype=float) * 2])
    P = EV.permute_within_month(M, ta, base.ok, 5); per = np.asarray(ta.to_period("M").astype(str))
    assert not np.array_equal(P[base.ok], M[base.ok])
    for mo in np.unique(per):
        k = (per == mo) & base.ok; assert np.array_equal(np.sort(P[k, 0]), np.sort(M[k, 0])) and (P[k, 1] == 2 * P[k, 0]).mean() < 0.01
    x = np.array([-5.0, 0.0, 3.0]); assert (np.diff(EV.mono(np.sort(x))) > 0).all()


def test_cli_end_to_end_writes_results_tables_and_figure(tmp_cache, world, monkeypatch, tmp_path):
    base = world["base"]
    VC.write_view("Z_persist", "fwd", _persist(base), note="test"); VC.write_view("Z_leak", "fwd", _leak(world))
    monkeypatch.setitem(AN.LOOKS, "look2", dict(AN.LOOKS["look2"], first=MONTHS[0], last=MONTHS[1]))
    monkeypatch.setattr(AN, "load_anchors", lambda s: pd.read_parquet(str(tmp_cache / "anchors_fwd.parquet")))
    out = tmp_path / "o"
    res = EV.evaluate_view("Z_leak", looks=("look2",), out_dir=str(out), n_perm=2, n_boot=200, log=lambda *a: None)
    assert res["pass"]["candidate"] and res["looks"]["look2"]["A"]["increment"]["view"]["mean"] > 0.3
    for f in ("results.json", "tables.txt", "fig_increments.png", "fig_increments.json"):
        assert (out / f).exists() and (out / f).stat().st_size > 100
    r = json.load(open(out / "results.json")); assert r["view"] == "Z_leak" and r["pass"]["A"] is True and "view_sha256" in r
    fig = json.load(open(out / "fig_increments.json")); assert fig["data"] and fig["layout"]["paper_bgcolor"].startswith("#")
    res2 = EV.evaluate_view("Z_persist", looks=("look2",), out_dir=str(tmp_path / "o2"), n_perm=2, n_boot=200, figure=False, log=lambda *a: None)
    assert not res2["pass"]["candidate"]
    h = EV.holm_all([str(out / "results.json"), str(tmp_path / "o2" / "results.json")]); assert len(h) == 4 and all(0 <= v <= 1 for v in h.values())


def test_timefix_seam_anchor_rows_are_masked_on_the_cut_clock():
    """The cut file's hourly rows near a DST seam are marked not-ok by T10's margin; check on a synthetic Bitstamp-style frame."""
    idx = pd.date_range("2024-03-05 00:00", "2024-03-20 00:00", freq="min")        # file time; DST change on 2024-03-10 (a seam day)
    P = T10.prep(pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx), "cut")
    t = idx[np.where(P.seam)[0]]; assert len(t) > 0 and t.min() < pd.Timestamp("2024-03-10") and t.max() > pd.Timestamp("2024-03-11")
    assert not P.seam[np.searchsorted(idx, pd.Timestamp("2024-03-18"))] and TF.seam_mask(idx[:5], 61, "cut").sum() == 0

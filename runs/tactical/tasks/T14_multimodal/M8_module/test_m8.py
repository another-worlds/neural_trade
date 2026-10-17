"""M8 tests (seconds; synthetic data, the frozen files, and one replay on already-used 2026 months when the harness cache exists).
Run: D:/nt/nt_mm_env/python.exe -m pytest -q test_m8.py   (CUDA_VISIBLE_DEVICES=-1)"""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_lib as M8  # noqa: E402
import m7_lib as L  # noqa: E402
import score_P as SP  # noqa: E402
from m6_lib import KEYS  # noqa: E402


def synth(n_days=420, seed=0, d=len(KEYS)):
    rng = np.random.default_rng(seed); ta = pd.date_range("2024-01-01", periods=n_days * 4, freq="6h")
    X = rng.normal(size=(len(ta), d)); y = np.sin(X[:, 0]) * X[:, 1] + 0.3 * rng.normal(size=len(ta)); ok = rng.random(len(ta)) > 0.05
    return ta, X, y, ok


# ------------------------------------------------------------------ (a) ablation columns
@pytest.mark.parametrize("g", list(M8.GROUPS))
def test_ablation_removes_exactly_the_listed_columns(g):
    kept = M8.kept_columns(g)
    assert M8.removed_keys(g) == M8.GROUPS[g]
    assert sorted(set(KEYS[i] for i in kept) | set(M8.GROUPS[g])) == sorted(KEYS) and not (set(KEYS[i] for i in kept) & set(M8.GROUPS[g]))
    assert len(kept) == len(KEYS) - len(M8.GROUPS[g])


def test_groups_cover_every_key_once():
    allk = [k for v in M8.GROUPS.values() for k in v]
    assert sorted(allk) == sorted(KEYS) and len(allk) == len(set(allk))
    assert M8.kept_columns(None) == list(range(len(KEYS)))


def test_ablated_view_ignores_removed_columns_and_uses_kept_ones():
    ta, X, y, ok = synth(); cols = M8.kept_columns("stretch"); gone = [i for i in range(len(KEYS)) if i not in cols]
    p, _, _ = L.monthly_view(ta, X[:, cols], y, ok, kind="gbm", min_pool=300)
    X2 = X.copy(); X2[:, gone] = np.random.default_rng(5).normal(size=(len(X), len(gone))) * 9          # the removed columns can be anything
    p2, _, _ = L.monthly_view(ta, X2[:, cols], y, ok, kind="gbm", min_pool=300)
    assert np.isfinite(p).sum() > 100 and np.array_equal(p, p2, equal_nan=True)
    X3 = X.copy(); X3[:, cols[0]] = X3[:, cols[0]][::-1]                                                   # a kept column does move the prediction
    p3, _, _ = L.monthly_view(ta, X3[:, cols], y, ok, kind="gbm", min_pool=300)
    assert not np.allclose(p[np.isfinite(p)], p3[np.isfinite(p)])


# ------------------------------------------------------------------ (b) linear terms
def test_linear_terms_values():
    K = np.zeros((1, len(KEYS))); ix = {k: i for i, k in enumerate(KEYS)}
    K[0, ix["str20"]] = -2.0; K[0, ix["str60"]] = 3.0; K[0, ix["vp_h"]] = 0.9; K[0, ix["vp_d"]] = 0.2; K[0, ix["hsin"]] = 0.5; K[0, ix["hcos"]] = -1.0
    z = M8.lin_terms(K)[0]
    assert np.allclose(z, [2.0, 3.0, 4.0, 9.0, 0.4 * 0.5, 0.4 * -1.0, -0.3 * 0.5, -0.3 * -1.0]) and len(M8.LIN_COLS) == 8


def test_linear_terms_are_rowwise_so_causal():
    rng = np.random.default_rng(1); K = rng.normal(size=(500, len(KEYS))); z = M8.lin_terms(K)
    K2 = K.copy(); K2[250:] = rng.normal(size=(250, len(KEYS))) * 7                                          # change every later row
    assert np.array_equal(M8.lin_terms(K2)[:250], z[:250])
    perm = rng.permutation(500)                                                                              # and a row depends on no other row
    assert np.array_equal(M8.lin_terms(K[perm]), z[perm])
    Kn = K.copy(); Kn[3, KEYS.index("str20")] = np.nan
    assert np.isnan(M8.lin_terms(Kn)[3]).sum() >= 2 and np.isfinite(M8.lin_terms(Kn)[4]).all()


# ------------------------------------------------------------------ (c) frozen artefact
def test_frozen_reload_identical_and_sha_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "MIN_POOL", 300)
    ta, X, y, ok = synth(n_days=700); yB = (y > 0).astype(float); cut = "2025-06-01"
    meta = M8.freeze(str(tmp_path), ta, X, y, yB, ok, cut, (False, True), "synthetic")
    assert M8.verify_files(str(tmp_path), meta) == []
    tr = M8.pool_rows(ta, ok, cut)
    _, _, mA = L.gbm_fit_predict(X[tr], y[tr], X[:3], False); _, _, mB = L.gbm_fit_predict(X[tr], yB[tr], X[:3], True)
    Q = X[-200:]
    assert np.array_equal(M8.predict_gbm(str(tmp_path / "gbm_A.txt"), Q), mA.predict(Q))                  # reload == the fitted model, bit for bit
    assert np.array_equal(M8.predict_gbm(str(tmp_path / "gbm_B.txt"), Q), mB.predict_proba(Q)[:, 1])
    k1 = M8.predict_kern(str(tmp_path / "kern_A.npz"), Q); k2 = M8.predict_kern(str(tmp_path / "kern_A.npz"), Q)
    assert np.array_equal(k1, k2) and np.isfinite(k1).all()
    open(tmp_path / "gbm_A.txt", "a").write("\n")
    assert M8.verify_files(str(tmp_path), meta) == ["gbm_A.txt"]


def test_pool_rule_uses_only_label_known_anchors():
    ta = pd.date_range("2026-08-01", "2026-10-02", freq="h"); ok = np.ones(len(ta), bool)
    tr = M8.pool_rows(ta, ok, "2026-10-01")
    assert ta[tr].max() == pd.Timestamp("2026-09-30 23:00") and ta[tr].min() >= pd.Timestamp("2026-10-01") - pd.Timedelta(days=365)


def test_written_frozen_files_match_their_sha256():
    p = os.path.join(HERE, "frozen_P.json")
    if not os.path.exists(p):
        pytest.skip("not frozen yet")
    meta = json.load(open(p)); fz = os.path.join(HERE, "frozen")
    assert M8.verify_files(fz, meta["freeze"]) == []
    for f, d in meta["history"].items():
        assert M8.sha256_file(os.path.join(fz, f)) == d["sha256"]
    assert meta["module"] in ("M7_gbm", "M7c_gbm_resid") and meta["freeze"]["cut"].startswith("2026-10-01")
    assert pd.Timestamp(meta["freeze"]["last_pool_ta"]) <= pd.Timestamp("2026-09-30 23:00")                # no label after 2026-09-30
    v = M8.predict_gbm(os.path.join(fz, "gbm_A.txt"), np.zeros((2, len(KEYS))))
    assert np.isfinite(v).all()


def test_module_rule_picks_the_larger_mean_with_ties_to_m7_gbm(tmp_path):
    import freeze_m8 as F
    for nm, (a, b) in {"M7_gbm": (0.03, 0.01), "M7c_gbm_resid": (0.02, 0.02)}.items():
        os.makedirs(tmp_path / nm); json.dump({"looks": {"look1": {"A": {"increment": {"view": {"mean": a}}}}, "look2": {"A": {"increment": {"view": {"mean": b}}}}}}, open(tmp_path / nm / "results.json", "w"))
    assert F.choose_module(str(tmp_path))[0] == "M7_gbm"                                                      # 0.02 vs 0.02: tie -> M7_gbm
    json.dump({"looks": {"look1": {"A": {"increment": {"view": {"mean": 0.03}}}}, "look2": {"A": {"increment": {"view": {"mean": 0.03}}}}}}, open(tmp_path / "M7c_gbm_resid" / "results.json", "w"))
    assert F.choose_module(str(tmp_path))[0] == "M7c_gbm_resid"


# ------------------------------------------------------------------ score_P
def test_complete_months_rule():
    ta = pd.date_range("2026-10-01", "2026-12-31 17:00", freq="h")
    assert M8.complete_months(ta) == [pd.Period("2026-10", "M"), pd.Period("2026-11", "M")]                 # December ends 6 h later than the table
    assert len(M8.complete_months(pd.date_range("2026-10-01", "2026-12-31 18:00", freq="h"))) == 3
    assert M8.complete_months(pd.DatetimeIndex([])) == [] and M8.complete_months(pd.date_range("2026-10-01", "2026-10-20", freq="h")) == []


def _stub_base(end):
    ta = pd.date_range("2026-10-01", end, freq="h"); return types.SimpleNamespace(ta=ta, ok=np.ones(len(ta), bool))


def test_score_withholds_with_two_complete_months_and_computes_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not compute")
    monkeypatch.setattr(SP.EV, "evaluate_look", boom); monkeypatch.setattr(SP, "apply_frozen", boom)
    b = _stub_base("2026-11-30 23:00")
    r = SP.score(None, None, b, np.zeros((len(b.ta), len(KEYS))), None, {"view_columns": ["a", "b"]})
    assert r["status"] == "withheld" and len(r["complete_months"]) == 2 and "looks" not in r
    r0 = SP.score(None, None, _stub_base("2026-10-15"), np.zeros((349, len(KEYS))), None, {})
    assert r0["status"] == "withheld"


def _fake_look(inc=0.01, p=0.01, rel=0.02, pb=0.01):
    a = {"view": {"mean": inc, "p_one_sided": p}, "c1": {"mean": 0.0}, "c2a": {"mean": 0.001}, "c2b": {"mean": 0.002}}
    b = {"view": {"mean": rel, "p_one_sided": pb}, "c1": {"mean": 0.0}, "c2a": {"mean": 0.001}, "c2b": {"mean": 0.002}}
    return {"counts": {}, "A": {"increment": a}, "B": {"rel_improvement": b}, "_B_raw": 1}


def test_score_runs_the_right_looks(monkeypatch):
    calls = []
    monkeypatch.setattr(SP.EV, "evaluate_look", lambda base, view, months, start, c2, *a, **k: calls.append(len(months)) or _fake_look())
    monkeypatch.setattr(SP, "apply_frozen", lambda fd, meta, K, mask: (np.zeros((len(K), 2)), np.zeros((len(K), 2))))
    monkeypatch.setattr(SP, "concat_base", lambda a, b: None)
    hist = {"gbm": pd.DataFrame(columns=["a", "b"]), "kern": pd.DataFrame(columns=["kern_A_hat", "kern_B_hat"])}
    for end, expect in (("2026-12-31 23:00", [3]), ("2027-03-31 23:00", [3, 6]), ("2027-04-30 23:00", [3, 6])):
        calls.clear(); b = _stub_base(end)
        r = SP.score(None, hist, b, np.zeros((len(b.ta), len(KEYS))), None, {"view_columns": ["a", "b"]}, n_boot=10)
        assert r["status"] == "scored" and calls == expect and list(r["looks"]) == [f"{k}_months" for k in expect]
        assert r["looks"]["3_months"]["months"] == ["2026-10", "2026-11", "2026-12"]


def test_criterion_thresholds_and_pocock_level():
    assert M8.POCOCK_ALPHA == 0.0294 and M8.LOOK_MONTHS == (3, 6) and M8.MIN_COMPLETE_MONTHS == 3
    assert M8.apply_criterion(_fake_look())["pass"]
    c = M8.apply_criterion(_fake_look(inc=0.004, rel=0.009)); assert not c["pass"] and not c["A"]["inc_ge_0.005"] and not c["B"]["rel_ge_1pct"]
    c = M8.apply_criterion(_fake_look(p=0.03, pb=0.03)); assert not c["pass"] and not c["A"]["p_lt_alpha"] and not c["B"]["p_lt_alpha"]   # 0.03 > 0.0294
    c = M8.apply_criterion(_fake_look(p=0.0293)); assert c["A"]["p_lt_alpha"] and c["pass"]
    c = M8.apply_criterion(_fake_look(inc=0.0015, p=0.001, pb=0.5)); assert not c["A"]["pass"] and not c["pass"]                          # below C2b (0.002) and below the 0.005 floor
    assert M8.apply_criterion(_fake_look(pb=0.5))["pass"]                                                                                   # either target passes the candidate


# ------------------------------------------------------------------ replay on already-used months (cache needed)
def test_replay_end_to_end_on_used_2026_months(tmp_path):
    try:
        import anchors as AN
        import build_m7 as B7
        import freeze_m8 as F
        bf = AN.load_base("fwd"); desc = pd.read_parquet(os.path.join(AN.CACHE_DIR, "desc_fwd.parquet"))
        json.load(open(os.path.join(HERE, "frozen_P.json")))
    except (FileNotFoundError, OSError):
        pytest.skip("harness cache or frozen files absent")
    cut = pd.Timestamp("2026-04-01"); D = B7.load_all(resid=True)
    meta = {"module": "M7c_gbm_resid", "view_columns": ["gbm_resA_hat", "gbm_resB_hat"], "freeze": F.freeze_module(D, "M7c_gbm_resid", cut, str(tmp_path))}
    hist = {nm: pd.read_parquet(os.path.join(HERE, "frozen", f"history_{nm}_fwd.parquet")) for nm in ("gbm", "kern")}
    hist = {k: v[v.index < cut] for k, v in hist.items()}
    bh = SP.subset_base(bf, np.asarray(bf.ta < cut)); bn = SP.subset_base(bf, np.asarray(bf.ta >= cut)); K = desc.reindex(bn.ta)[KEYS].to_numpy(float)
    assert bh.ta.max() < cut <= bn.ta.min() and meta["freeze"]["last_pool_ta"] < str(cut)
    r = SP.score(bh, hist, bn, K, str(tmp_path), meta, cut="2026-04-01", n_perm=1, n_boot=200, log=lambda s: None)
    assert r["status"] == "scored" and list(r["looks"]) == ["3_months", "6_months"] and r["looks"]["6_months"]["months"][-1] == "2026-09"
    cr = r["looks"]["3_months"]["criterion"]
    assert set(cr) == {"A", "B", "pass", "alpha"} and np.isfinite(r["looks"]["3_months"]["A_increment"]["view"]["mean"])

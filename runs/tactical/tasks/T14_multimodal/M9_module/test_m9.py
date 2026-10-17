"""M9 tests (seconds; synthetic data, stored views and frozen files when present, one replay on already-used 2026 months when the harness cache exists).
Run: D:/nt/nt_mm_env/python.exe -m pytest -q test_m9.py   (CUDA_VISIBLE_DEVICES=-1)"""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m9_lib as M9  # noqa: E402
import score_P9 as SP  # noqa: E402
from m6_lib import KEYS  # noqa: E402


def rnd_keys(n, seed=0):
    rng = np.random.default_rng(seed); K = rng.normal(size=(n, len(KEYS)))
    for k in ("vp_h", "vp_d"):
        K[:, KEYS.index(k)] = rng.random(n)
    return K


# ------------------------------------------------------------------ definitions
def test_column_counts_names_and_nesting():
    assert {k: len(v) for k, v in M9.VIEW_COLS.items()} == {"M9_prof": 47, "M9_prof_str": 51, "M9_prof_str_vp": 57, "M9x_B": 73, "M9x_A": 68, "M9x_AB": 84,
                                                           "M9h3_prof_str": 17, "M9h3_prof_str_vp": 23, "M9h6_prof_str": 29, "M9h6_prof_str_vp": 35}
    assert M9.VIEW_COLS["M9_prof"] == [f"h{h}" for h in range(1, 24)] + ["wkend"] + [f"wk_h{h}" for h in range(1, 24)]
    assert M9.VIEW_COLS["M9_prof_str"] == M9.VIEW_COLS["M9_prof"] + ["abs_str20", "abs_str60", "sq_str20", "sq_str60"]
    assert M9.VIEW_COLS["M9_prof_str_vp"] == M9.VIEW_COLS["M9_prof_str"] + ["vph_c", "vpd_c", "vph_c_x_hsin", "vph_c_x_hcos", "vpd_c_x_hsin", "vpd_c_x_hcos"]
    for v in M9.VIEW_COLS.values():
        assert len(v) == len(set(v))
    for big, small in (("M9x_B", "M9_prof_str_vp"), ("M9x_A", "M9_prof_str_vp"), ("M9x_AB", "M9x_B")):
        assert M9.VIEW_COLS[big][:len(M9.VIEW_COLS[small])] == M9.VIEW_COLS[small]
    assert set(M9.VIEW_COLS["M9x_A"]) <= set(M9.VIEW_COLS["M9x_AB"])
    ta = pd.date_range("2026-03-01", periods=50, freq="h"); K = rnd_keys(50)
    for nm in M9.VIEW_COLS:
        Z, names = M9.view_matrix(nm, ta, K)
        assert Z.shape == (50, len(M9.VIEW_COLS[nm])) and names == M9.VIEW_COLS[nm] and np.isfinite(Z).all()


def test_hour_and_weekend_dummies_exact_on_known_timestamps():
    ta = pd.DatetimeIndex(["2026-03-04 13:00", "2026-03-06 23:00", "2026-03-07 00:00", "2026-03-08 23:00", "2026-03-09 00:00", "2026-03-07 05:00"])
    assert list(ta.dayofweek) == [2, 4, 5, 6, 0, 5]                                # Wed, Fri, Sat, Sun, Mon, Sat
    Z, names = M9.view_matrix("M9_prof", ta, rnd_keys(len(ta))); col = lambda c: Z[:, names.index(c)]
    assert list(col("wkend")) == [0, 0, 1, 1, 0, 1]
    assert col("h13")[0] == 1 and col("h23")[1] == 1 and col("h23")[3] == 1 and col("h5")[5] == 1
    assert Z[2, :23].sum() == 0 and Z[4, :23].sum() == 0                            # hour 0 is the dropped level
    assert col("wk_h23")[3] == 1 and col("wk_h23")[1] == 0 and col("wk_h5")[5] == 1 and col("wk_h13")[0] == 0
    assert Z[0, 24:].sum() == 0 and Z[1, 24:].sum() == 0 and Z[4, 24:].sum() == 0   # no weekend interaction on weekdays
    assert Z[:, :23].sum(1).tolist() == [1, 1, 0, 1, 0, 1]                           # exactly one hour dummy (or none for hour 0)
    assert Z[:, 24:].sum(1).tolist() == [0, 0, 0, 1, 0, 1]


def test_harmonic_and_block_columns_exact():
    ta = pd.DatetimeIndex(["2026-03-07 06:00", "2026-03-04 15:00"]); K = rnd_keys(2)
    Z, n = M9.view_matrix("M9h3_prof_str", ta, K); c = lambda x: Z[:, n.index(x)]
    assert np.allclose(c("sin1"), np.sin(2 * np.pi * np.array([6, 15]) / 24)) and np.allclose(c("cos3"), np.cos(2 * np.pi * 3 * np.array([6, 15]) / 24))
    assert np.allclose(c("wk_sin2"), [np.sin(2 * np.pi * 2 * 6 / 24), 0]) and list(c("wkend")) == [1, 0]
    Zb, nb = M9.view_matrix("M9x_B", ta, K); ix = {k: i for i, k in enumerate(KEYS)}
    assert np.isclose(Zb[0, nb.index("blk2_x_vph_c")], K[0, ix["vp_h"]] - 0.5) and Zb[0, nb.index("blk5_x_vph_c")] == 0          # hour 6 -> block 2
    assert np.isclose(Zb[1, nb.index("blk5_x_vpd_c")], K[1, ix["vp_d"]] - 0.5) and np.isclose(Zb[0, nb.index("wk_x_vph_c")], K[0, ix["vp_h"]] - 0.5)
    assert Zb[1, nb.index("wk_x_vph_c")] == 0


def test_stretch_and_vp_terms_values():
    ta = pd.date_range("2026-03-04", periods=1, freq="h"); K = np.zeros((1, len(KEYS))); ix = {k: i for i, k in enumerate(KEYS)}
    K[0, ix["str20"]] = -2.0; K[0, ix["str60"]] = 3.0; K[0, ix["vp_h"]] = 0.9; K[0, ix["vp_d"]] = 0.2; K[0, ix["hsin"]] = 0.5; K[0, ix["hcos"]] = -1.0
    Z, n = M9.view_matrix("M9_prof_str_vp", ta, K); g = lambda c: Z[0, n.index(c)]
    assert (g("abs_str20"), g("abs_str60"), g("sq_str20"), g("sq_str60")) == (2.0, 3.0, 4.0, 9.0)
    assert np.isclose(g("vph_c"), 0.4) and np.isclose(g("vpd_c"), -0.3) and np.isclose(g("vph_c_x_hsin"), 0.2) and np.isclose(g("vpd_c_x_hcos"), 0.3)


# ------------------------------------------------------------------ causality
@pytest.mark.parametrize("name", list(M9.VIEW_COLS))
def test_columns_depend_only_on_timestamp_and_the_rows_own_keys(name):
    n = 300; ta = pd.date_range("2026-01-01", periods=n, freq="h"); K = rnd_keys(n); Z, _ = M9.view_matrix(name, ta, K)
    K2 = K.copy(); K2[150:] = rnd_keys(150, 9) * 7                                  # mutate every key from row 150 on
    Z2, _ = M9.view_matrix(name, ta, K2)
    assert np.array_equal(Z[:150], Z2[:150])
    perm = np.random.default_rng(3).permutation(n)                                  # a row depends on no other row
    Zp, _ = M9.view_matrix(name, ta[perm], K[perm])
    assert np.array_equal(Zp, Z[perm])
    ta2 = ta.copy().to_series(); ta2.iloc[200:] += pd.Timedelta(days=400); Z3, _ = M9.view_matrix(name, pd.DatetimeIndex(ta2), K)    # later timestamps do not move earlier rows
    assert np.array_equal(Z3[:200], Z[:200])


def test_nan_keys_give_nan_only_in_their_columns():
    ta = pd.date_range("2026-01-01", periods=5, freq="h"); K = rnd_keys(5); K[2, KEYS.index("str20")] = np.nan
    Z, n = M9.view_matrix("M9_prof_str_vp", ta, K)
    assert np.isnan(Z[2, [n.index("abs_str20"), n.index("sq_str20")]]).all() and np.isfinite(Z[2, :47]).all() and np.isfinite(np.delete(Z, 2, 0)).all()


def test_stored_views_equal_the_recipe():
    import view_cache as VC
    import anchors as AN
    try:
        v = VC.read_view("M9_prof_str_vp", "fwd"); desc = pd.read_parquet(os.path.join(AN.CACHE_DIR, "desc_fwd.parquet"))
    except (FileNotFoundError, OSError):
        pytest.skip("harness cache or stored view absent")
    Z, names = M9.view_matrix("M9_prof_str_vp", v.index, desc.reindex(v.index)[KEYS].to_numpy(float))
    assert list(v.columns) == names and np.allclose(v.to_numpy(), Z, equal_nan=True, atol=0, rtol=0)


# ------------------------------------------------------------------ GBM7 and the rules
def test_gbm7_uses_exactly_the_seven_listed_keys():
    cols = M9.gbm7_columns()
    assert [KEYS[i] for i in cols] == ["hsin", "hcos", "wkend", "str20", "str60", "vp_h", "vp_d"] and len(set(cols)) == 7
    left_out = {k for i, k in enumerate(KEYS) if i not in cols}
    assert left_out == {"sr_near", "sr_asym", "lrv0", "lrv1", "lrv2"}


def test_freeze_rule_and_labels():
    rows = {"a": {"pass": True, "R_A": 1.0, "R_B": 0.8, "n_cols": 51}, "b": {"pass": True, "R_A": 0.9, "R_B": 0.9, "n_cols": 47}, "c": {"pass": False, "R_A": 2, "R_B": 2, "n_cols": 10}}
    assert M9.choose_freeze(rows) == "b"                                           # highest min(R): b 0.9 > a 0.8; the non-passing c never wins
    rows["a"]["R_B"] = 0.898
    assert M9.choose_freeze(rows) == "b"                                           # within 0.005: fewer columns
    rows["a"]["R_B"] = 0.88; rows["b"]["n_cols"] = 99
    assert M9.choose_freeze(rows) == "b"                                           # 0.9 vs 0.88: not a tie, b by value
    assert M9.choose_freeze({"x": {"pass": False, "R_A": 1, "R_B": 1, "n_cols": 1}}) is None
    assert [M9.recovery_label(v) for v in (0.75, 0.7499, 0.40, 0.39)] == ["recovers", "partly", "partly", "no"]


# ------------------------------------------------------------------ frozen artefact and score_P9
def test_frozen_recipe_hash_and_loader(tmp_path):
    p = os.path.join(HERE, "frozen_P.json")
    if os.path.exists(p):
        meta = json.load(open(p)); assert meta["status"] == "none" or meta["recipe_sha256"] == M9.recipe_sha256()
    for f in [f for f in os.listdir(HERE) if f.startswith("frozen_P_") and f.endswith(".json")]:
        m = json.load(open(os.path.join(HERE, f)))
        assert m["recipe_sha256"] == M9.recipe_sha256() and m["columns"] == M9.VIEW_COLS[m["view"]] and m["pool"]["last_pool_ta"] <= "2026-09-30 23:00"
        assert SP.load_frozen(os.path.join(HERE, f))["view"] == m["view"]
    good = {"status": "post hoc candidate", "view": "M9_prof", "recipe_sha256": M9.recipe_sha256(), "columns": M9.VIEW_COLS["M9_prof"]}
    json.dump(good, open(tmp_path / "ok.json", "w")); assert SP.load_frozen(str(tmp_path / "ok.json"))["view"] == "M9_prof"
    json.dump({**good, "recipe_sha256": "0" * 64}, open(tmp_path / "bad.json", "w"))
    with pytest.raises(RuntimeError):
        SP.load_frozen(str(tmp_path / "bad.json"))
    json.dump({"status": "none"}, open(tmp_path / "none.json", "w"))
    with pytest.raises(RuntimeError):
        SP.load_frozen(str(tmp_path / "none.json"))


def _stub_base(end):
    ta = pd.date_range("2026-10-01", end, freq="h"); return types.SimpleNamespace(ta=ta, ok=np.ones(len(ta), bool))


def _fake_look(inc=0.01, p=0.01, rel=0.02, pb=0.01):
    a = {"view": {"mean": inc, "p_one_sided": p}, "c1": {"mean": 0.0}, "c2a": {"mean": 0.001}}
    b = {"view": {"mean": rel, "p_one_sided": pb}, "c1": {"mean": 0.0}, "c2a": {"mean": 0.001}}
    return {"counts": {}, "A": {"increment": a}, "B": {"rel_improvement": b}, "_B_raw": 1}


def test_score_withholds_with_two_complete_months_and_computes_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not compute")
    monkeypatch.setattr(SP.EV, "evaluate_look", boom); monkeypatch.setattr(SP, "build_view", boom)
    b = _stub_base("2026-11-30 23:00")
    r = SP.score(None, None, b, np.zeros((len(b.ta), len(KEYS))), {"view": "M9_prof"})
    assert r["status"] == "withheld" and len(r["complete_months"]) == 2 and "looks" not in r
    assert SP.score(None, None, _stub_base("2026-10-15"), np.zeros((349, len(KEYS))), {"view": "M9_prof"})["status"] == "withheld"


def test_score_runs_the_right_looks_and_pocock_level(monkeypatch):
    calls = []
    monkeypatch.setattr(SP.EV, "evaluate_look", lambda base, view, months, start, c2, *a, **k: calls.append((len(months), c2)) or _fake_look())
    monkeypatch.setattr(SP, "build_view", lambda v, ta, K: pd.DataFrame(index=ta)); monkeypatch.setattr(SP.SP, "concat_base", lambda a, b: None)
    hist = types.SimpleNamespace(ta=pd.DatetimeIndex([]))
    for end, expect in (("2026-12-31 23:00", [3]), ("2027-03-31 23:00", [3, 6]), ("2027-04-30 23:00", [3, 6])):
        calls.clear(); b = _stub_base(end)
        r = SP.score(hist, np.zeros((0, len(KEYS))), b, np.zeros((len(b.ta), len(KEYS))), {"view": "M9_prof"}, n_boot=10)
        assert r["status"] == "scored" and [c[0] for c in calls] == expect and all(c[1] is None for c in calls) and r["alpha"] == 0.0294
        assert r["looks"]["3_months"]["months"] == ["2026-10", "2026-11", "2026-12"] and r["looks"]["3_months"]["criterion"]["pass"]
    c = SP.M8.apply_criterion(_fake_look(p=0.03, pb=0.03)); assert not c["pass"]


# ------------------------------------------------------------------ replay on already-used months (cache needed)
def test_replay_end_to_end_on_used_2026_months():
    try:
        import anchors as AN
        bf = AN.load_base("fwd"); desc = pd.read_parquet(os.path.join(AN.CACHE_DIR, "desc_fwd.parquet"))
    except (FileNotFoundError, OSError):
        pytest.skip("harness cache absent")
    cut = pd.Timestamp("2026-04-01")
    meta = {"view": "M9h6_prof_str"}
    bh = SP.SP.subset_base(bf, np.asarray(bf.ta < cut)); bn = SP.SP.subset_base(bf, np.asarray(bf.ta >= cut))
    Kh = desc.reindex(bh.ta)[KEYS].to_numpy(float); Kn = desc.reindex(bn.ta)[KEYS].to_numpy(float)
    assert bh.ta.max() < cut <= bn.ta.min()
    r = SP.score(bh, Kh, bn, Kn, meta, cut="2026-04-01", n_perm=1, n_boot=200, log=lambda s: None)
    assert r["status"] == "scored" and list(r["looks"]) == ["3_months", "6_months"] and r["looks"]["6_months"]["months"][-1] == "2026-09"
    cr = r["looks"]["3_months"]["criterion"]
    assert set(cr) == {"A", "B", "pass", "alpha"} and np.isfinite(r["looks"]["3_months"]["A_increment"]["view"]["mean"])

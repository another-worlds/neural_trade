"""Seconds-long checks of the T15 helpers (run before any job)."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t15_lib as T
import m9_lib as M9

def test_harm_equals_m9():
    ta = pd.date_range("2024-03-01", periods=500, freq="h"); rng = np.random.default_rng(0)
    K = rng.normal(size=(500, 12)); K[5, 5] = np.nan
    h, w = T.hour_wk(ta)
    for k, nm in ((3, "M9h3_prof_str"), (6, "M9h6_prof_str")):
        a, na = T.harm_prof_str(k, h, w, K); b, nb = M9.view_matrix(nm, ta, K)
        assert na == nb, (na, nb)
        assert np.allclose(a, b, equal_nan=True)

def test_k12_columns():
    n = T.harm_names(12) + T.STR_COLS
    assert len(n) == 51 and "sin12" not in n and "cos12" in n
    h = np.arange(24); assert np.allclose(np.sin(2 * np.pi * 12 * h / 24), 0, atol=1e-12)

def test_shuffle_within_month():
    ta = pd.date_range("2024-01-15", periods=2000, freq="h"); mask = np.ones(2000, bool); mask[::7] = False
    p = T.shuffle_within_month(ta, mask, 1)
    per = np.asarray(ta.year * 12 + ta.month)
    assert (per[p] == per).all() and (p[~mask] == np.arange(2000)[~mask]).all() and (np.sort(p[mask]) == np.where(mask)[0]).all()
    assert (p != np.arange(2000)).sum() > 1000

def test_guard():
    c = T.checks(); assert c["free_ram_gb"] > 0 and c["free_disk_D_gb"] > 0

if __name__ == "__main__":
    for f in (test_harm_equals_m9, test_k12_columns, test_shuffle_within_month, test_guard):
        f(); print("ok", f.__name__)

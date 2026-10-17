"""T14 candidate M9: the profile views (pure helpers). Every column is a function of the anchor timestamp `ta` (UTC) and of the M6 keys at the anchor,
computed ROW-WISE (a row depends on no other row), so it is causal by construction. CPU only; run with D:/nt/nt_mm_env/python.exe. No data access here.

Views (SPEC_M9): M9_prof (47 columns), M9_prof_str (51), M9_prof_str_vp (57); post hoc M9x_B, M9x_A, M9x_AB.
"""
import hashlib
import inspect
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T14 = os.path.dirname(HERE)
for p in (os.path.join(T14, "M8_module"), os.path.join(T14, "M7_gbm"), os.path.join(T14, "M6_faiss"), os.path.join(T14, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
from m6_lib import KEYS  # noqa: E402

IX = {k: i for i, k in enumerate(KEYS)}
HOUR_NAMES = [f"h{h}" for h in range(1, 24)]
WK_HOUR_NAMES = [f"wk_h{h}" for h in range(1, 24)]
PROF_COLS = HOUR_NAMES + ["wkend"] + WK_HOUR_NAMES
STR_COLS = ["abs_str20", "abs_str60", "sq_str20", "sq_str60"]
VP_COLS = ["vph_c", "vpd_c", "vph_c_x_hsin", "vph_c_x_hcos", "vpd_c_x_hsin", "vpd_c_x_hcos"]
BLOCKS = list(range(1, 8))                                  # three-hour blocks 1..7 (block 0 = hours 0-2 dropped)
XB_COLS = [f"blk{b}_x_{v}" for v in ("vph_c", "vpd_c") for b in BLOCKS] + ["wk_x_vph_c", "wk_x_vpd_c"]
XA_COLS = ["wk_x_abs_str20", "wk_x_abs_str60", "wk_x_sq_str20", "wk_x_sq_str60"] + [f"blk{b}_x_abs_str20" for b in BLOCKS]
VIEW_COLS = {"M9_prof": PROF_COLS, "M9_prof_str": PROF_COLS + STR_COLS, "M9_prof_str_vp": PROF_COLS + STR_COLS + VP_COLS,
             "M9x_B": PROF_COLS + STR_COLS + VP_COLS + XB_COLS, "M9x_A": PROF_COLS + STR_COLS + VP_COLS + XA_COLS,
             "M9x_AB": PROF_COLS + STR_COLS + VP_COLS + XB_COLS + XA_COLS}
HARM_VIEWS = {"M9h3_prof_str": (3, False), "M9h3_prof_str_vp": (3, True), "M9h6_prof_str": (6, False), "M9h6_prof_str_vp": (6, True)}   # addendum 1 (post hoc)


def harm_cols(k):
    return [f"sin{j}" for j in range(1, k + 1)] + [f"cos{j}" for j in range(1, k + 1)] + ["wkend"] + [f"wk_sin{j}" for j in range(1, k + 1)] + [f"wk_cos{j}" for j in range(1, k + 1)]


for _n, (_k, _vp) in HARM_VIEWS.items():
    VIEW_COLS[_n] = harm_cols(_k) + STR_COLS + (VP_COLS if _vp else [])
PRE_REGISTERED = ["M9_prof", "M9_prof_str", "M9_prof_str_vp"]
POST_HOC = ["M9x_B", "M9x_A", "M9x_AB"]
POST_HOC_H = list(HARM_VIEWS)
GBM7_KEYS = ["hsin", "hcos", "wkend", "str20", "str60", "vp_h", "vp_d"]
R_YES, R_PART = 0.75, 0.40
TIE = 0.005


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def prof_matrix(ta):
    """[N, 47]: hour dummies h1..h23 (hour 0 dropped), wkend, wkend * h1..h23. Only the timestamp is read."""
    ta = pd.DatetimeIndex(ta); hr = np.asarray(ta.hour); wk = (np.asarray(ta.dayofweek) >= 5).astype(float)
    H = (hr[:, None] == np.arange(1, 24)[None, :]).astype(float)
    return np.hstack([H, wk[:, None], H * wk[:, None]])


def view_matrix(name, ta, K):
    """(matrix [N, c] float, column names) of a view from the anchor times and the M6 keys [N, 12] (KEYS order). NaN where an input key is NaN
    (the timestamp columns are never NaN; a NaN in the stretch / vp keys makes that row's stretch / vp columns NaN)."""
    if name not in VIEW_COLS:
        raise KeyError(name)
    K = np.asarray(K, float); ta = pd.DatetimeIndex(ta); N = len(ta)
    cols = {}
    P = prof_matrix(ta)
    for j, c in enumerate(PROF_COLS):
        cols[c] = P[:, j]
    s20, s60, vh, vd, hs, hc = (K[:, IX[k]] for k in ("str20", "str60", "vp_h", "vp_d", "hsin", "hcos"))
    cols.update(abs_str20=np.abs(s20), abs_str60=np.abs(s60), sq_str20=s20 ** 2, sq_str60=s60 ** 2)
    vhc, vdc = vh - 0.5, vd - 0.5
    cols.update(vph_c=vhc, vpd_c=vdc, vph_c_x_hsin=vhc * hs, vph_c_x_hcos=vhc * hc, vpd_c_x_hsin=vdc * hs, vpd_c_x_hcos=vdc * hc)
    wk = P[:, PROF_COLS.index("wkend")]; blk = np.asarray(ta.hour) // 3
    for b in BLOCKS:
        ind = (blk == b).astype(float)
        cols[f"blk{b}_x_vph_c"] = ind * vhc; cols[f"blk{b}_x_vpd_c"] = ind * vdc; cols[f"blk{b}_x_abs_str20"] = ind * np.abs(s20)
    cols.update(wk_x_vph_c=wk * vhc, wk_x_vpd_c=wk * vdc, wk_x_abs_str20=wk * np.abs(s20), wk_x_abs_str60=wk * np.abs(s60),
                wk_x_sq_str20=wk * s20 ** 2, wk_x_sq_str60=wk * s60 ** 2)
    if name in HARM_VIEWS:
        kk = HARM_VIEWS[name][0]; hr = np.asarray(ta.hour); w = (np.asarray(ta.dayofweek) >= 5).astype(float)
        for j in range(1, kk + 1):
            a = 2 * np.pi * j * hr / 24
            cols[f"sin{j}"] = np.sin(a); cols[f"cos{j}"] = np.cos(a); cols[f"wk_sin{j}"] = w * np.sin(a); cols[f"wk_cos{j}"] = w * np.cos(a)
        cols["wkend"] = w
    names = list(VIEW_COLS[name])
    return np.column_stack([cols[c] for c in names]).astype(float), names


def recipe_sha256():
    """sha256 of the recipe: the sources of the column definitions (view_matrix, prof_matrix) and the column lists."""
    src = inspect.getsource(prof_matrix) + inspect.getsource(view_matrix) + repr(sorted((k, v) for k, v in VIEW_COLS.items())) + repr(KEYS)
    return hashlib.sha256(src.encode()).hexdigest()


def gbm7_columns():
    """indices into KEYS of the 7 keys of the M9_gbm7 view."""
    return [IX[k] for k in GBM7_KEYS]


# ------------------------------------------------------------------ readout rules
def recovery_label(min_r):
    return "recovers" if min_r >= R_YES else ("partly" if min_r >= R_PART else "no")


def choose_freeze(rows):
    """rows: {view: {"pass": bool, "R_A": float, "R_B": float, "n_cols": int}} for the eligible views. Highest min(R_A, R_B) among those that pass;
    ties (within TIE) to the fewer columns. None if no view passes."""
    ok = {k: v for k, v in rows.items() if v["pass"]}
    if not ok:
        return None
    best = max(min(v["R_A"], v["R_B"]) for v in ok.values())
    near = [k for k, v in ok.items() if best - min(v["R_A"], v["R_B"]) < TIE]
    return min(near, key=lambda k: (ok[k]["n_cols"], -min(ok[k]["R_A"], ok[k]["R_B"])))

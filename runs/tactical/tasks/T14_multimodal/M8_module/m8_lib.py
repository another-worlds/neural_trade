"""T14 candidate M8: pure helpers (ablation columns, cheap linear terms, frozen module save / load / apply, the prospective criterion).
CPU only; run with D:/nt/nt_mm_env/python.exe. No data access here: everything is passed in."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T14 = os.path.dirname(HERE)
for p in (os.path.join(T14, "M7_gbm"), os.path.join(T14, "M6_faiss"), os.path.join(T14, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
import m7_lib as L  # noqa: E402
from m6_lib import KEYS  # noqa: E402

GROUPS = {"stretch": ["str20", "str60"], "volpct": ["vp_h", "vp_d"], "hourwk": ["hsin", "hcos", "wkend"],
          "sr": ["sr_near", "sr_asym"], "lrv": ["lrv0", "lrv1", "lrv2"]}
LIN_COLS = ["abs_str20", "abs_str60", "sq_str20", "sq_str60", "vph_x_hsin", "vph_x_hcos", "vpd_x_hsin", "vpd_x_hcos"]
POCOCK_ALPHA = 0.0294
MIN_COMPLETE_MONTHS = 3
LOOK_MONTHS = (3, 6)
TAIL_HOURS = 6
CRIT_A, CRIT_B = 0.005, 0.01


def kept_columns(group):
    """indices into KEYS of the keys that remain when `group` (a name of GROUPS, or None) is removed."""
    drop = set(GROUPS[group]) if group else set()
    return [i for i, k in enumerate(KEYS) if k not in drop]


def removed_keys(group):
    return [k for i, k in enumerate(KEYS) if i not in set(kept_columns(group))]


def lin_terms(K):
    """[N, 8] cheap nonlinear terms computed ROW-WISE from the keys (str20, str60, vp_h, vp_d, hsin, hcos); LIN_COLS order. NaN where an input is NaN."""
    K = np.asarray(K, float); ix = {k: i for i, k in enumerate(KEYS)}
    s20, s60, vh, vd, hs, hc = (K[:, ix[k]] for k in ("str20", "str60", "vp_h", "vp_d", "hsin", "hcos"))
    return np.column_stack([np.abs(s20), np.abs(s60), s20 ** 2, s60 ** 2, (vh - 0.5) * hs, (vh - 0.5) * hc, (vd - 0.5) * hs, (vd - 0.5) * hc])


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------ frozen module
def pool_rows(ta, ok, cut, days=L.POOL_DAYS):
    """M7's pool rule at m0 = cut: ta >= cut - 365 days and ta + 60 min <= cut, ok."""
    ta = pd.DatetimeIndex(ta); cut = pd.Timestamp(cut)
    return np.where(np.asarray(ok) & np.asarray(ta >= cut - pd.Timedelta(days=days)) & np.asarray(ta + pd.Timedelta(minutes=L.LABEL_MIN) <= cut))[0]


def freeze(out_dir, ta, K, yA, yB, ok, cut, binary_pair, tag, seed=0):
    """Fit the final GBM (A, B) and the kernel control (A, B) on the pool at `cut` and save them in out_dir. `binary_pair` = (A is a classifier?, B ...).
    Returns the metadata dict (sha256 of every file, pool size and hash, parameters). Files: gbm_A.txt, gbm_B.txt, kern_A.npz, kern_B.npz."""
    import lightgbm as lgb
    os.makedirs(out_dir, exist_ok=True); tr = pool_rows(ta, ok, cut)
    if len(tr) < L.MIN_POOL:
        raise ValueError(f"pool {len(tr)} < {L.MIN_POOL}")
    meta = {"tag": tag, "cut": str(pd.Timestamp(cut)), "n_pool": int(len(tr)), "first_pool_ta": str(pd.DatetimeIndex(ta)[tr].min()), "last_pool_ta": str(pd.DatetimeIndex(ta)[tr].max()),
            "pool_ta_sha256": hashlib.sha256(np.asarray(pd.DatetimeIndex(ta)[tr]).astype("datetime64[ns]").astype(np.int64).tobytes()).hexdigest(),
            "pool_X_sha256": hashlib.sha256(np.ascontiguousarray(K[tr]).tobytes()).hexdigest(),
            "gbm_params": L.gbm_params(seed), "lightgbm": lgb.__version__, "keys": list(KEYS), "files": {}}
    for nm, y, binary in (("A", yA, binary_pair[0]), ("B", yB, binary_pair[1])):
        _, _, m = L.gbm_fit_predict(K[tr], y[tr], K[tr[:3]], binary, seed)
        p = os.path.join(out_dir, f"gbm_{nm}.txt"); m.booster_.save_model(p)
        meta["files"][f"gbm_{nm}.txt"] = {"sha256": sha256_file(p), "binary": bool(binary)}
        mu = K[tr].mean(0); sd = K[tr].std(0) + 1e-9
        month_id = np.asarray(pd.DatetimeIndex(ta)[tr].year * 12 + pd.DatetimeIndex(ta)[tr].month)
        _, _, ex = L.nw_fit_predict(K[tr], y[tr], K[tr[:3]], binary, month_id, seed)
        q = os.path.join(out_dir, f"kern_{nm}.npz")
        np.savez_compressed(q, A=((K[tr] - mu) / sd).astype(np.float64), y=np.asarray(y[tr], float), mu=mu, sd=sd, h=np.array(ex["h"]))
        meta["files"][f"kern_{nm}.npz"] = {"sha256": sha256_file(q), "h": float(ex["h"])}
    return meta


def load_gbm(path):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)


def predict_gbm(path, X):
    return np.asarray(load_gbm(path).predict(np.asarray(X, float)))


def predict_kern(path, X):
    z = np.load(path)
    T = ((np.asarray(X, float) - z["mu"]) / z["sd"]).astype(np.float64)
    return L._nw(L._d2(T, z["A"]), z["y"], float(z["h"]))


def verify_files(out_dir, meta):
    """sha256 of every frozen file against the metadata; returns the list of mismatches."""
    return [f for f, d in meta["files"].items() if sha256_file(os.path.join(out_dir, f)) != d["sha256"]]


# ------------------------------------------------------------------ prospective months
def complete_months(ta_new, first="2026-10", tail_hours=TAIL_HOURS):
    """Consecutive complete UTC months from `first` on. A month is complete when the new anchor table reaches (month end - tail_hours)."""
    ta_new = pd.DatetimeIndex(ta_new)
    if len(ta_new) == 0:
        return []
    last = ta_new.max(); out = []; mo = pd.Period(first, "M")
    while last >= (mo + 1).start_time - pd.Timedelta(hours=tail_hours):
        out.append(mo); mo = mo + 1
    return out


def apply_criterion(look_res, alpha=POCOCK_ALPHA):
    """Pre-registered prospective criterion on ONE look's harness result (evaluate_look output): A: increment >= 0.005, one-sided p < alpha, above C1 and C2
    (C2a and, when present, C2b); B: relative Brier improvement >= 1%, one-sided p (day-block bootstrap) < alpha, above C1 and C2. Pass = either target."""
    a = look_res["A"]["increment"]; v = a["view"]
    c2 = [k for k in ("c2a", "c2b") if k in a]
    fa = {"inc_ge_0.005": bool(v["mean"] is not None and v["mean"] >= CRIT_A), "p_lt_alpha": bool(v["p_one_sided"] is not None and v["p_one_sided"] < alpha),
          "above_c1": bool(v["mean"] > a["c1"]["mean"])}
    for c in c2:
        fa[f"above_{c}"] = bool(v["mean"] > a[c]["mean"])
    fa["pass"] = bool(all(fa.values()))
    b = look_res["B"]["rel_improvement"]; w = b["view"]
    fb = {"rel_ge_1pct": bool(w["mean"] >= CRIT_B), "p_lt_alpha": bool(w["p_one_sided"] < alpha), "above_c1": bool(w["mean"] > b["c1"]["mean"])}
    for c in [k for k in ("c2a", "c2b") if k in b]:
        fb[f"above_{c}"] = bool(w["mean"] > b[c]["mean"])
    fb["pass"] = bool(all(fb.values()))
    return {"A": fa, "B": fb, "pass": bool(fa["pass"] or fb["pass"]), "alpha": alpha}

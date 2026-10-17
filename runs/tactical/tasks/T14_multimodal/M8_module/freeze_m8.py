"""Freeze the M8 module for the prospective period (SPEC_M8 (c)). CPU. Run: D:/nt/nt_mm_env/python.exe freeze_m8.py -> frozen/, frozen_P.json.
Module rule (fixed in SPEC_M8): of M7_gbm and M7c_gbm_resid, the larger mean over the two looks of the stored target-A increment; ties to M7_gbm."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); T14 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m8_lib as M8  # noqa: E402
import m7_lib as L  # noqa: E402
import build_m7 as B7  # noqa: E402
import view_cache as VC  # noqa: E402

CUT = "2026-10-01"
COLS = {"M7_gbm": ["gbm_dA_hat", "gbm_hit_hat", "gbm_dA_rank", "gbm_hit_rank"], "M7c_gbm_resid": ["gbm_resA_hat", "gbm_resB_hat"]}


def choose_module(t14=T14):
    m = {}
    for nm in ("M7_gbm", "M7c_gbm_resid"):
        r = json.load(open(os.path.join(t14, nm, "results.json")))
        m[nm] = float(np.mean([r["looks"][lk]["A"]["increment"]["view"]["mean"] for lk in ("look1", "look2")]))
    return ("M7_gbm" if m["M7_gbm"] >= m["M7c_gbm_resid"] else "M7c_gbm_resid"), m


def labels(D, module):
    K = D["keys"]; okK = D["ok"] & B7.finite_rows(K)
    if module == "M7c_gbm_resid":
        return D["resA"], D["resB"], (False, False), okK & B7.finite_rows(D["dA"], D["hit"]) & B7.finite_rows(D["resA"], D["resB"])
    return D["dA"], D["hit"], (False, True), okK & B7.finite_rows(D["dA"]) & B7.finite_rows(D["hit"])


def freeze_module(D, module, cut, out_dir):
    yA, yB, bp, ok = labels(D, module)
    meta = M8.freeze(out_dir, D["ta"], D["keys"], yA, yB, ok, cut, bp, module)
    if module == "M7_gbm":                      # the rank copy: empirical CDF among the final model's own pool predictions
        tr = M8.pool_rows(D["ta"], ok, cut)
        for nm, f in (("A", "gbm_A.txt"), ("B", "gbm_B.txt")):
            p = os.path.join(out_dir, f"rank_ref_{nm}.npy"); np.save(p, np.sort(M8.predict_gbm(os.path.join(out_dir, f), D["keys"][tr])))
            meta["files"][f"rank_ref_{nm}.npy"] = {"sha256": M8.sha256_file(p)}
    return meta


def history_views(D, module, out_dir):
    """the walk-forward module view and the walk-forward kernel control (same labels), forward-file rows, saved with sha256."""
    yA, yB, bp, ok = labels(D, module); ta = D["ta"]; K = D["keys"]; hist = {}
    g = VC.read_view("M7c_gbm_resid" if module == "M7c_gbm_resid" else "M7_gbm", "fwd")
    ka, _, _ = L.monthly_view(ta, K, yA, ok, binary=bp[0], kind="nw"); kb, _, _ = L.monthly_view(ta, K, yB, ok, binary=bp[1], kind="nw")
    B7.split_write("M8_kern_" + module, D, {"kern_A_hat": ka, "kern_B_hat": kb}, f"M8 frozen-kernel control history (walk-forward NW on the {module} labels)", {"views": {}})
    k = VC.read_view("M8_kern_" + module, "fwd")
    for nm, df in (("gbm", g), ("kern", k)):
        p = os.path.join(out_dir, f"history_{nm}_fwd.parquet"); df.to_parquet(p)
        hist[f"history_{nm}_fwd.parquet"] = {"sha256": M8.sha256_file(p), "rows": int(len(df)), "columns": list(df.columns)}
    return hist


def main():
    L.below_normal(); t0 = time.time(); D = B7.load_all(resid=True)
    module, scores = choose_module(); print("module rule:", scores, "->", module, flush=True)
    out = os.path.join(HERE, "frozen"); os.makedirs(out, exist_ok=True)
    meta = freeze_module(D, module, CUT, out)
    hist = history_views(D, module, out)
    tr = M8.pool_rows(D["ta"], labels(D, module)[3], CUT)
    n_cut_pool = int(np.sum(np.asarray(D["src"][tr] == "cut")))
    full = {"module": module, "module_rule": "larger mean over the two looks of the stored target-A increment; ties to M7_gbm", "rule_values": scores, "cut": CUT,
            "view_columns": COLS[module], "n_pool_rows_from_cut_file": n_cut_pool, "freeze": meta, "history": hist,
            "criterion": {"A_inc_min": M8.CRIT_A, "B_rel_min": M8.CRIT_B, "pocock_alpha": M8.POCOCK_ALPHA, "looks_months": list(M8.LOOK_MONTHS),
                          "min_complete_months": M8.MIN_COMPLETE_MONTHS, "controls": "C1 permuted, C2a harness S3 duplicate, C2b frozen kernel"},
            "seconds": round(time.time() - t0, 1)}
    json.dump(full, open(os.path.join(HERE, "frozen_P.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in full.items() if k != "freeze"}, indent=1)); print({f: d["sha256"] for f, d in meta["files"].items()})


if __name__ == "__main__":
    main()

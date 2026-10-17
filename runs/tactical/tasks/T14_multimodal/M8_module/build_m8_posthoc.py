"""M8 POST HOC (written after seeing the pre-registered ablation, which put the gain on the hour / weekend group; labelled, not part of SPEC_M8):
M8x_abl_hour (M7_gbm without hsin, hcos), M8x_abl_wkend (without wkend), M8x_lin_wk (M8_lin's eight terms plus wkend, wkend*hsin, wkend*hcos).
Run: D:/nt/nt_mm_env/python.exe build_m8_posthoc.py, then evaluate.py --view <name> from ../core."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_lib as M8  # noqa: E402
import m7_lib as L  # noqa: E402
import build_m7 as B7  # noqa: E402
from m6_lib import KEYS  # noqa: E402


def main():
    L.below_normal(); D = B7.load_all(); ta = D["ta"]; K = D["keys"]
    ok = D["ok"] & B7.finite_rows(K) & B7.finite_rows(D["dA"]) & B7.finite_rows(D["hit"]); info = {"views": {}}
    for nm, drop in (("M8x_abl_hour", ["hsin", "hcos"]), ("M8x_abl_wkend", ["wkend"])):
        cols = [i for i, k in enumerate(KEYS) if k not in drop]
        pa, ra, _ = L.monthly_view(ta, K[:, cols], D["dA"], ok, binary=False, kind="gbm"); ph, rh, _ = L.monthly_view(ta, K[:, cols], D["hit"], ok, binary=True, kind="gbm")
        B7.split_write(nm, D, {"gbm_dA_hat": pa, "gbm_hit_hat": ph, "gbm_dA_rank": ra, "gbm_hit_rank": rh}, f"M8 POST HOC: M7_gbm without {drop}", info)
    ix = {k: i for i, k in enumerate(KEYS)}; w = K[:, ix["wkend"]]
    Z = np.column_stack([M8.lin_terms(K), w, w * K[:, ix["hsin"]], w * K[:, ix["hcos"]]]); Z[~ok] = np.nan
    B7.split_write("M8x_lin_wk", D, {c: Z[:, j] for j, c in enumerate(M8.LIN_COLS + ["wkend", "wk_x_hsin", "wk_x_hcos"])}, "M8 POST HOC: M8_lin plus weekend terms", info)
    print("done")


if __name__ == "__main__":
    main()

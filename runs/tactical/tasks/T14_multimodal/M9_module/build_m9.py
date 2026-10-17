"""Build the M9 views (CPU, below-normal priority). Usage: D:/nt/nt_mm_env/python.exe build_m9.py [pre|post|harm]
  pre   M9_prof, M9_prof_str, M9_prof_str_vp (row-wise from ta and the keys) and M9_gbm7 (M7's GBM scheme on 7 keys)
  post  M9x_B, M9x_A, M9x_AB (post hoc, SPEC_M9 menu)
then evaluate.py --view <name> from ../core."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m9_lib as M9  # noqa: E402
import m7_lib as L  # noqa: E402
import build_m7 as B7  # noqa: E402


def main(which="pre"):
    L.below_normal(); t0 = time.time(); D = B7.load_all(); ta = D["ta"]; K = D["keys"]
    ok = D["ok"] & B7.finite_rows(K) & B7.finite_rows(D["dA"]) & B7.finite_rows(D["hit"])
    info = {"n_anchors": int(len(ta)), "n_ok_rows": int(ok.sum()), "views": {}, "seconds": {}}
    for nm in {"pre": M9.PRE_REGISTERED, "post": M9.POST_HOC, "harm": M9.POST_HOC_H}[which]:
        Z, names = M9.view_matrix(nm, ta, K); Z[~ok] = np.nan
        B7.split_write(nm, D, {c: Z[:, j] for j, c in enumerate(names)}, f"M9 {nm}: {len(names)} row-wise columns from the timestamp and the M6 keys", info)
        print(nm, Z.shape[1], flush=True)
    if which == "pre":
        t = time.time(); cols = M9.gbm7_columns(); Kg = K[:, cols]
        pa, ra, _ = L.monthly_view(ta, Kg, D["dA"], ok, binary=False, kind="gbm"); ph, rh, _ = L.monthly_view(ta, Kg, D["hit"], ok, binary=True, kind="gbm")
        B7.split_write("M9_gbm7", D, {"gbm_dA_hat": pa, "gbm_hit_hat": ph, "gbm_dA_rank": ra, "gbm_hit_rank": rh}, f"M9 GBM on the 7 keys {M9.GBM7_KEYS}", info)
        info["seconds"]["gbm7"] = round(time.time() - t, 1)
    info["recipe_sha256"] = M9.recipe_sha256(); info["total_seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(os.path.join(HERE, f"checks_m9_{which}.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done", info["total_seconds"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pre")

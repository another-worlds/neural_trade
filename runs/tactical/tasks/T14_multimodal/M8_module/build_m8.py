"""Build the M8 views (CPU, below-normal priority): the five ablations M8_abl_<group>, the sanity refit with nothing removed (compared with the stored M7_gbm view),
and M8_lin. Run: D:/nt/nt_mm_env/python.exe build_m8.py   then   evaluate.py --view <name> from ../core (run_evals.sh / run_evals.py)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_lib as M8  # noqa: E402
import m7_lib as L  # noqa: E402
import build_m7 as B7  # noqa: E402
import view_cache as VC  # noqa: E402


def main():
    L.below_normal(); t0 = time.time(); D = B7.load_all(); ta = D["ta"]; K = D["keys"]
    ok = D["ok"] & B7.finite_rows(K) & B7.finite_rows(D["dA"]) & B7.finite_rows(D["hit"])
    info = {"n_anchors": int(len(ta)), "n_ok_rows": int(ok.sum()), "removed": {g: M8.removed_keys(g) for g in M8.GROUPS}, "views": {}, "seconds": {}}
    log = lambda s: print(s, flush=True)
    for g in [None] + list(M8.GROUPS):
        t = time.time(); cols = M8.kept_columns(g); Kg = K[:, cols]
        pa, ra, _ = L.monthly_view(ta, Kg, D["dA"], ok, binary=False, kind="gbm"); ph, rh, _ = L.monthly_view(ta, Kg, D["hit"], ok, binary=True, kind="gbm")
        out = {"gbm_dA_hat": pa, "gbm_hit_hat": ph, "gbm_dA_rank": ra, "gbm_hit_rank": rh}
        if g is None:                                    # sanity: identical to the stored M7_gbm view
            diffs = {}
            for s, sl in (("cut", slice(0, D["n_cut"])), ("fwd", slice(D["n_cut"], None))):
                new = pd.DataFrame(out, index=ta).iloc[sl].dropna(); old = VC.read_view("M7_gbm", s)
                assert new.index.equals(old.index), f"{s}: row sets differ"
                diffs[s] = float(np.max(np.abs(new.to_numpy() - old[new.columns].to_numpy())))
            info["sanity_full_vs_M7_gbm_max_abs_diff"] = diffs; log(f"sanity: refit with nothing removed vs stored M7_gbm: {diffs}")
            assert max(diffs.values()) < 1e-9
        else:
            B7.split_write(f"M8_abl_{g}", D, out, f"M8 ablation: M7_gbm without {M8.GROUPS[g]}", info)
            info["seconds"][g] = round(time.time() - t, 1); log(f"  {g}: removed {M8.GROUPS[g]} ({time.time() - t:.0f}s)")
    # cheap linear terms on the same rows
    Z = M8.lin_terms(K); Z[~ok] = np.nan
    B7.split_write("M8_lin", D, {c: Z[:, j] for j, c in enumerate(M8.LIN_COLS)}, "M8 cheap linear terms: |str|, str^2, (vp-0.5) x hour harmonics, computed row-wise from the M6 keys", info)
    info["total_seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(os.path.join(HERE, "checks_m8.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"done {info['total_seconds']}s")


if __name__ == "__main__":
    main()

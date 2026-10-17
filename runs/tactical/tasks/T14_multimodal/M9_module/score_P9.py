"""T14 M9 prospective scorer. Scores the months from 2026-10 on, ONCE, with the profile module named in a frozen_P*.json, when >= 3 complete months exist.

    D:/nt/nt_mm_env/python.exe score_P9.py --base-new NEW_base.npz --keys-new NEW_desc.parquet --frozen frozen_P_M9h6_prof_str.json [--cut 2026-10-01] [--out DIR]

Same inputs as M8's score_P.py (made by a later item with the harness' own builders): `--base-new` an npz in the format of base_{source}.npz for the anchors from
the cut on (365 days of earlier bars as warm-up), `--keys-new` the M6 descriptors of the same anchors. History: the harness' forward base before the cut and its
descriptors (desc_fwd). The module has NO weights: the view columns are computed from the anchor times and keys by the frozen recipe (m9_lib.view_matrix, its sha256
checked against the artefact); V3 / H3 and the ridge / logistic including the view coefficients are refitted monthly by the unchanged harness `evaluate_look`.
Looks: the first 3 complete months, then the first 6; Pocock alpha 0.0294 (one-sided) at both. With < 3 complete months: status "withheld", nothing computed.
Controls: C1 (permuted) and C2a (S3 duplicate); there is no C2b for a linear view.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); T14 = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(T14, "M8_module"))
import m9_lib as M9  # noqa: E402
import m8_lib as M8  # noqa: E402
import score_P as SP  # noqa: E402  (M8's helpers: subset_base, concat_base)
import anchors as AN  # noqa: E402
import evaluate as EV  # noqa: E402
from m6_lib import KEYS  # noqa: E402


def load_frozen(path):
    meta = json.load(open(path))
    if meta.get("status", "none") == "none" or not meta.get("view"):
        raise RuntimeError("nothing is frozen in " + str(path))
    if meta["recipe_sha256"] != M9.recipe_sha256():
        raise RuntimeError("the recipe code changed since the freeze (recipe sha256 mismatch)")
    if list(M9.VIEW_COLS[meta["view"]]) != list(meta["columns"]):
        raise RuntimeError("column list differs from the frozen one")
    return meta


def build_view(view, ta, K):
    """DataFrame (index ta) of the view's columns for the rows whose columns are all finite."""
    Z, names = M9.view_matrix(view, ta, K); df = pd.DataFrame(Z, index=pd.DatetimeIndex(ta), columns=names)
    return df[df.notna().all(axis=1)]


def score(base_hist, K_hist, base_new, K_new, meta, cut="2026-10-01", n_perm=5, n_boot=10000, log=print):
    """The prospective scoring. `base_hist` / `K_hist`: history rows before `cut`; `base_new` / `K_new` the new anchors."""
    cm = M8.complete_months(base_new.ta, first=pd.Timestamp(cut).strftime("%Y-%m")) if len(base_new.ta) else []
    res = {"cut": str(cut), "complete_months": [str(m) for m in cm], "alpha": M8.POCOCK_ALPHA, "view": meta.get("view") if meta else None}
    if len(cm) < M8.MIN_COMPLETE_MONTHS:
        res.update(status="withheld", reason=f"{len(cm)} complete months < {M8.MIN_COMPLETE_MONTHS}")
        return res
    view_df = pd.concat([build_view(meta["view"], base_hist.ta, K_hist), build_view(meta["view"], base_new.ta, K_new)])
    base = SP.concat_base(base_hist, base_new); start = AN.LOOKS["look2"]["train_from"]
    res["status"] = "scored"; res["looks"] = {}
    for k in M8.LOOK_MONTHS:
        if len(cm) < k:
            break
        r = EV.evaluate_look(base, view_df, cm[:k], start, None, n_perm, n_boot, 0, log); r.pop("_B_raw", None)
        res["looks"][f"{k}_months"] = {"months": [str(m) for m in cm[:k]], "counts": r["counts"], "A_increment": r["A"]["increment"],
                                       "B_rel_improvement": r["B"]["rel_improvement"], "criterion": M8.apply_criterion(r)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-new", required=True); ap.add_argument("--keys-new", required=True); ap.add_argument("--cut", default="2026-10-01")
    ap.add_argument("--frozen", required=True); ap.add_argument("--out", default=os.path.join(HERE, "score_P9_out"))
    ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--n-perm", type=int, default=5)
    a = ap.parse_args(argv); meta = load_frozen(a.frozen)
    z = np.load(a.base_new); arr = {k: z[k] for k in z.files if k != "ta_ns"}
    base_new = AN.Base(pd.DatetimeIndex(z["ta_ns"].astype("datetime64[ns]")), arr)
    K = pd.read_parquet(a.keys_new).reindex(base_new.ta)[KEYS].to_numpy(float)
    bf = AN.load_base("fwd"); bh = SP.subset_base(bf, np.asarray(bf.ta < pd.Timestamp(a.cut)))
    Kh = pd.read_parquet(os.path.join(AN.CACHE_DIR, "desc_fwd.parquet")).reindex(bh.ta)[KEYS].to_numpy(float)
    res = score(bh, Kh, base_new, K, meta, a.cut, a.n_perm, a.n_boot)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "score_P9_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in res.items() if k != "looks"}, indent=1))
    for k, v in res.get("looks", {}).items():
        print(k, v["months"], v["criterion"])


if __name__ == "__main__":
    main()

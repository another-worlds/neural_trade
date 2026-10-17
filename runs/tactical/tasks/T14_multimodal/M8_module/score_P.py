"""T14 M8 prospective scorer. Scores the months from 2026-10 on, ONCE, with the FROZEN module (frozen_P.json + frozen/), when >= 3 complete months exist.

    D:/nt/nt_mm_env/python.exe score_P.py --base-new NEW_base.npz --keys-new NEW_desc.parquet [--cut 2026-10-01] [--frozen DIR] [--out DIR]

Inputs (made by a later item that builds a 'prosp' source with the harness' own builders, never by this script):
  --base-new   an npz in the format of base_{source}.npz (`ta_ns` + the flat arrays of anchors.build_hourly) for the anchors from the cut on, with the
               365 days of earlier bars as warm-up, as the harness builds the forward file;
  --keys-new   the M6 descriptors (m6_lib.KEYS columns, index = anchor time) of the same anchors, as desc_{source}.parquet.
History: the harness' forward base (D:/nt/nt_t14_cache/base_fwd.npz) and the stored walk-forward views of the module and of its kernel control, copied in
frozen/ (history_*_fwd.parquet, sha256 in frozen_P.json). New months use the frozen models (no GBM refit); V3 / H3 and the ridge / logistic are refitted
monthly by the unchanged harness `evaluate_look`.
Looks: the first 3 complete months, then the first 6; Pocock alpha 0.0294 (one-sided) at both. With < 3 complete months: status "withheld", nothing computed.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_lib as M8  # noqa: E402
import anchors as AN  # noqa: E402
import evaluate as EV  # noqa: E402
from m6_lib import KEYS  # noqa: E402


def load_frozen(frozen_dir):
    meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(frozen_dir)), "frozen_P.json")))
    bad = M8.verify_files(frozen_dir, meta["freeze"])
    if bad:
        raise RuntimeError(f"frozen files changed: {bad}")
    return meta


def apply_frozen(frozen_dir, meta, K, mask):
    """(view [N, c], kern [N, 2]) for the rows `mask` (all keys finite); NaN elsewhere."""
    mod = meta["module"]; N = len(K); out = np.full((N, len(meta["view_columns"])), np.nan); kern = np.full((N, 2), np.nan)
    rows = np.where(mask)[0]
    if len(rows) == 0:
        return out, kern
    pa = M8.predict_gbm(os.path.join(frozen_dir, "gbm_A.txt"), K[rows]); pb = M8.predict_gbm(os.path.join(frozen_dir, "gbm_B.txt"), K[rows])
    out[rows, 0] = pa; out[rows, 1] = pb
    if mod == "M7_gbm":
        for j, nm in ((2, "rank_ref_A.npy"), (3, "rank_ref_B.npy")):
            ref = np.load(os.path.join(frozen_dir, nm)); out[rows, j] = np.searchsorted(ref, out[rows, j - 2], side="right") / max(len(ref), 1)
    kern[rows, 0] = M8.predict_kern(os.path.join(frozen_dir, "kern_A.npz"), K[rows]); kern[rows, 1] = M8.predict_kern(os.path.join(frozen_dir, "kern_B.npz"), K[rows])
    return out, kern


def subset_base(base, mask):
    return AN.Base(base.ta[mask], {k: v[mask] for k, v in base.arr.items()})


def concat_base(b1, b2):
    keys = set(b1.arr) & set(b2.arr)
    return AN.Base(b1.ta.append(b2.ta), {k: np.concatenate([b1.arr[k], b2.arr[k]]) for k in keys})


def load_history(frozen_dir, cut):
    h = {}
    for nm in ("gbm", "kern"):
        df = pd.read_parquet(os.path.join(frozen_dir, f"history_{nm}_fwd.parquet")); df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
        h[nm] = df[df.index < pd.Timestamp(cut)]
    return h


def score(base_hist, hist, base_new, K_new, frozen_dir, meta, cut="2026-10-01", n_perm=5, n_boot=10000, log=print):
    """The prospective scoring. `base_hist` / `hist` are the history (rows before `cut`); `base_new` / `K_new` the new anchors. Returns the result dict."""
    cm = M8.complete_months(base_new.ta, first=pd.Timestamp(cut).strftime("%Y-%m")) if len(base_new.ta) else []
    res = {"cut": str(cut), "complete_months": [str(m) for m in cm], "alpha": M8.POCOCK_ALPHA}
    if len(cm) < M8.MIN_COMPLETE_MONTHS:
        res.update(status="withheld", reason=f"{len(cm)} complete months < {M8.MIN_COMPLETE_MONTHS}")
        return res
    ok = base_new.ok & np.isfinite(K_new).all(1)
    V, Kn = apply_frozen(frozen_dir, meta, K_new, ok)
    cols = meta["view_columns"]
    vnew = pd.DataFrame(V, index=base_new.ta, columns=cols); knew = pd.DataFrame(Kn, index=base_new.ta, columns=["kern_A_hat", "kern_B_hat"])
    vnew = vnew[vnew.notna().all(axis=1)]; knew = knew[knew.notna().all(axis=1)]
    view = pd.concat([hist["gbm"][cols], vnew]); c2 = pd.concat([hist["kern"][["kern_A_hat", "kern_B_hat"]], knew])
    base = concat_base(base_hist, base_new); start = AN.LOOKS["look2"]["train_from"]
    res["status"] = "scored"; res["looks"] = {}
    for k in M8.LOOK_MONTHS:
        if len(cm) < k:
            break
        r = EV.evaluate_look(base, view, cm[:k], start, c2, n_perm, n_boot, 0, log); r.pop("_B_raw", None)
        res["looks"][f"{k}_months"] = {"months": [str(m) for m in cm[:k]], "counts": r["counts"], "A_increment": r["A"]["increment"],
                                       "B_rel_improvement": r["B"]["rel_improvement"], "criterion": M8.apply_criterion(r)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-new", required=True); ap.add_argument("--keys-new", required=True); ap.add_argument("--cut", default="2026-10-01")
    ap.add_argument("--frozen", default=os.path.join(HERE, "frozen")); ap.add_argument("--out", default=os.path.join(HERE, "score_P_out"))
    ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--n-perm", type=int, default=5)
    a = ap.parse_args(argv); meta = load_frozen(a.frozen)
    z = np.load(a.base_new); arr = {k: z[k] for k in z.files if k != "ta_ns"}
    base_new = AN.Base(pd.DatetimeIndex(z["ta_ns"].astype("datetime64[ns]")), arr)
    K = pd.read_parquet(a.keys_new).reindex(base_new.ta)[KEYS].to_numpy(float)
    bh = AN.load_base("fwd"); bh = subset_base(bh, np.asarray(bh.ta < pd.Timestamp(a.cut)))
    res = score(bh, load_history(a.frozen, a.cut), base_new, K, a.frozen, meta, a.cut, a.n_perm, a.n_boot)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "score_P_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in res.items() if k != "looks"}, indent=1))
    for k, v in res.get("looks", {}).items():
        print(k, v["months"], v["criterion"])


if __name__ == "__main__":
    main()

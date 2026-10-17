"""Build the M7 views (both sources, one pooled walk-forward) with view_cache.write_view. Run: D:/nt/nt_mm_env/python.exe build_m7.py (CPU only, below-normal priority).

M7_gbm              pre-registered. LightGBM on the 12 M6 keys: gbm_dA_hat (A: next-hour change of log RV), gbm_hit_hat (B: barrier label 50 bps / 60 min), + rank copies
M7_kernel           control. Nadaraya-Watson smoother, same keys and pool (non-retrieval, information-equivalent): kern_dA_hat, kern_hit_hat, + rank copies
M7b_gbm_plus_v3     POST HOC. GBM on the keys AND the harness numeric columns (A: the 16 S3 columns; B: the 6 HAR + 10 extra columns)
M7c_gbm_resid       POST HOC. GBM on the keys fitted to the walk-forward residual of V3 (A) / H3 (B) (as M6b did): gbm_resA_hat, gbm_resB_hat
M7s_gbm, M7s_kernel SENSITIVITY (labelled): the same, with the pool restricted to the anchor's own source file (no cross-exchange pool)
Pool of month m: anchors of BOTH source files (Bitstamp cut file, then the Binance forward file) with label window ended before m's start and within the 365 days
before it; this is the pre-registered rule read literally. The cut file ends 2025-07-20 and the forward file starts 2025-10-01, so Look 2 months have a pool.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.join(os.path.dirname(HERE), "M6_faiss")
for p in (HERE, M6):
    sys.path.insert(0, p)
import m7_lib as L  # noqa: E402
import m6_lib as M6L  # noqa: E402
import anchors as AN  # noqa: E402
import view_cache as VC  # noqa: E402
import build_views as BV6  # noqa: E402  (M6's walk-forward residuals)

SRC = (("cut", "look1"), ("fwd", "look2"))


def load_all(resid=False):
    """pooled arrays over both sources, sorted by time (cut before fwd)."""
    parts = []
    for s, lk in SRC:
        base = AN.load_base(s); ta = base.ta
        d = pd.read_parquet(os.path.join(AN.CACHE_DIR, f"desc_{s}.parquet")).reindex(ta)[M6L.KEYS].to_numpy(float)
        a = base.arr
        P = {"ta": ta, "src": np.full(len(ta), s), "keys": d, "ok": base.ok.copy(), "dA": a["A_dA"].astype(float), "hit": a["B_y50"].astype(float),
             "S3": a["A_S3"].astype(float), "BF": np.hstack([a["B_har"], a["B_ex10"]]).astype(float)}
        if resid:
            rA, rB, _ = BV6.residuals(base, lk); P["resA"] = rA; P["resB"] = rB
        parts.append(P)
    out = {k: (parts[0][k].append(parts[1][k]) if k == "ta" else np.concatenate([parts[0][k], parts[1][k]])) for k in parts[0]}
    out["n_cut"] = len(parts[0]["ta"]); return out


def finite_rows(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        a = a.reshape(len(a), -1); m &= np.isfinite(a).all(1)
    return m


def split_write(name, D, cols, note, info):
    ta = D["ta"]; n = D["n_cut"]; df = pd.DataFrame(cols, index=ta)
    meta = {}
    for s, sl in (("cut", slice(0, n)), ("fwd", slice(n, None))):
        part = df.iloc[sl]; part = part[part.notna().all(axis=1)]
        m = VC.write_view(name, s, part, note=note); meta[s] = {"n_rows": m["n_rows"], "n_ok_anchors_covered": m["n_ok_anchors_covered"], "sha256": m["sha256"],
                                                                "first_row": str(part.index.min()), "last_row": str(part.index.max())}
    info["views"][name] = meta


def main():
    L.below_normal(); t0 = time.time(); D = load_all(resid=True); ta = D["ta"]; K = D["keys"]
    info = {"n_anchors": int(len(ta)), "views": {}, "months": {}}
    okK = D["ok"] & finite_rows(K)
    okA = okK & finite_rows(D["dA"]); okB = okK & finite_rows(D["hit"])
    ok = okA & okB
    info["n_ok_pool_rows"] = int(ok.sum())
    log = lambda s: print(s, flush=True)

    def run(tag, kind, X, y, okm, binary, same=False):
        t = time.time(); p, r, mi = L.monthly_view(ta, X, y, okm, source=D["src"], same_source=same, binary=binary, kind=kind)
        info["months"][tag] = mi; log(f"  {tag}: {int(np.isfinite(p).sum())} rows served, months {[m['month'] for m in mi if m['served']][:1]}..{[m['month'] for m in mi if m['served']][-1:]} ({time.time()-t:.0f}s)")
        return p, r

    for kind, nm, pre, same_list in (("gbm", "M7_gbm", "gbm", (False, True)), ("nw", "M7_kernel", "kern", (False, True))):
        for same in same_list:
            tag = nm if not same else nm.replace("M7_", "M7s_")
            pa, ra = run(tag + ":A", kind, K, D["dA"], ok, False, same); ph, rh = run(tag + ":B", kind, K, D["hit"], ok, True, same)
            cols = {f"{pre}_dA_hat": pa, f"{pre}_hit_hat": ph, f"{pre}_dA_rank": ra, f"{pre}_hit_rank": rh}
            split_write(tag, D, cols, f"M7 {tag}: monthly walk-forward on the 12 M6 keys, pool {'own source' if same else 'both sources'}", info)
    # M7b: keys + numeric harness columns
    XA = np.hstack([K, D["S3"]]); XB = np.hstack([K, D["BF"]])
    okb = ok & finite_rows(XA, XB)
    pa, _ = run("M7b:A", "gbm", XA, D["dA"], okb, False); ph, _ = run("M7b:B", "gbm", XB, D["hit"], okb, True)
    split_write("M7b_gbm_plus_v3", D, {"gbm_dA_hat": pa, "gbm_hit_hat": ph}, "M7b POST HOC: GBM on keys + S3 (A) / HAR + ex10 (B)", info)
    # M7c: GBM on the walk-forward residuals of V3 / H3
    okr = ok & finite_rows(D["resA"], D["resB"])
    ra_, _, mia = L.monthly_view(ta, K, D["resA"], okr, binary=False, kind="gbm"); rb_, _, mib = L.monthly_view(ta, K, D["resB"], okr, binary=False, kind="gbm")
    info["months"]["M7c:A"] = mia; info["months"]["M7c:B"] = mib
    split_write("M7c_gbm_resid", D, {"gbm_resA_hat": ra_, "gbm_resB_hat": rb_}, "M7c POST HOC: GBM regressor on the keys for the walk-forward residual of V3 (A) and H3 (B)", info)
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(os.path.join(HERE, "checks.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"done {info['seconds']}s")


if __name__ == "__main__":
    main()

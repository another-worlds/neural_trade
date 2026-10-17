"""Build the M6 views (both sources) and write them with view_cache.write_view. Run: D:/nt/nt_mm_env/python.exe build_views.py   (CPU only).

M6_faiss / M6_faiss_shuf          pre-registered: k = 50, columns knn_dA_mean, knn_dA_sd, knn_hit50_mean, knn_dist_mean (control: labels shuffled across neighbours)
M6b_faiss_resid_k{20,50,100}      POST HOC variant: mean walk-forward residual of V3 (target A) and of H3 (target B) of the k neighbours, plus knn_dist_mean (+ _shuf controls)
Writes checks.json (the causal-pool rule verified on every served neighbour) and desc_{source}.parquet in the cache.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m6_lib as M  # noqa: E402
import anchors as AN  # noqa: E402
import view_cache as VC  # noqa: E402
import evaluate as EV  # noqa: E402
import sr_lib as S  # noqa: E402

RESID_KS = (20, 50, 100)


def residuals(base, look):
    """walk-forward residuals of V3 (A) and H3 (B) on the base's ok rows over the look's training+test months: y - prediction of the model fitted before the row's month."""
    cfg = AN.LOOKS[look]; start = cfg["train_from"]
    months = list(pd.period_range(start.to_period("M"), cfg["last"], freq="M")); N = len(base.ta); ta = base.ta
    resA = np.full(N, np.nan); resB = np.full(N, np.nan)
    rows = EV.run_A(base, base.ok, months, start, {"V3": None}, keep_preds=True)
    y = base.A["T"]["A"]
    for r in rows:
        if not r["skipped"]:
            resA[r["te"]] = y[r["te"]] - r["preds"]["V3"]
    R = EV.run_B(base, base.ok, months, start, {"H3": None})
    yb = base.B["y"][AN.X_B]; off = 0
    for inf in R["info"]:
        if inf["skipped"]:
            continue
        mo = pd.Period(inf["month"], "M"); te = np.where(EV.month_mask(ta, base.ok, mo))[0]
        resB[te] = yb[te] - R["p"]["H3"][off:off + len(te)]; off += len(te)
    assert off == len(R["p"]["H3"])
    return resA, resB, [inf for inf in R["info"] if inf["skipped"]]


def build_source(source, look, log=print):
    t0 = time.time(); base = AN.load_base(source); store = AN.get_store(source); ta = base.ta; start = AN.LOOKS[look]["train_from"]
    sigma = S.sigma_blend(base.arr["A_S3"])
    p = os.path.join(AN.CACHE_DIR, f"desc_{source}.parquet")
    if os.path.exists(p):
        desc = pd.read_parquet(p).reindex(ta).to_numpy(float)
    else:
        desc = M.descriptors(store.m1, ta, sigma, S.sr_block, S.clean_windows)
        pd.DataFrame(desc, index=ta, columns=M.KEYS).to_parquet(p)
    log(f"{source}: descriptors {desc.shape}, finite rows {int(np.isfinite(desc).all(1).sum())} ({time.time() - t0:.0f}s)")
    queries = np.asarray(ta >= start)
    lab = np.column_stack([base.arr["A_dA"], base.arr["B_y50"].astype(float)])
    ok = base.ok.copy()
    out = {}; info = {"source": source, "look": look, "n_anchors": int(len(ta)), "n_ok": int(ok.sum())}
    # ---- pre-registered view
    nbr, dist, pool_n = M.knn_pool_index(ta, ok & np.isfinite(lab).all(1), desc, kmax=50, queries=queries)
    served = (nbr[:, :50] >= 0).all(1)
    mean, sd, _ = M.neighbour_stats(nbr, 50, lab); smean, ssd, _ = M.neighbour_stats(nbr, 50, lab, ta=ta, shuffle=True, seed=7)
    dm = dist[:, :50].mean(1)
    cols = lambda mu, s: pd.DataFrame({"knn_dA_mean": mu[:, 0], "knn_dA_sd": s[:, 0], "knn_hit50_mean": mu[:, 1], "knn_dist_mean": dm}, index=ta)[served]
    out["M6_faiss"] = cols(mean, sd); out["M6_faiss_shuf"] = cols(smean, ssd)
    info["pre_registered"] = {"served_rows": int(served.sum()), "served_ok_rows": int((served & ok).sum()), "queries": int(queries.sum()),
                              "pool_min_served": int(pool_n[served].min()), "pool_max": int(pool_n.max()),
                              "dropped_lt_500_pool": int((queries & np.isfinite(desc).all(1) & ~served).sum()),
                              "dropped_nonfinite_desc": int((queries & ~np.isfinite(desc).all(1)).sum()), "first_served": str(ta[served][0])}
    info["rule_check"] = rule_check(ta, nbr, ok & np.isfinite(lab).all(1), 50)
    # ---- post-hoc residual variant
    resA, resB, skipped = residuals(base, look)
    rl = np.column_stack([resA, resB]); okr = ok & np.isfinite(rl).all(1)
    nbr2, dist2, pool2 = M.knn_pool_index(ta, okr, desc, kmax=100, queries=queries)
    info["residual_pool_ok_rows"] = int(okr.sum()); info["residual_skipped_months"] = skipped
    for k in RESID_KS:
        sv = (nbr2[:, :k] >= 0).all(1); mu, _, _ = M.neighbour_stats(nbr2, k, rl); smu, _, _ = M.neighbour_stats(nbr2, k, rl, ta=ta, shuffle=True, seed=11)
        d2 = dist2[:, :k].mean(1)
        mk = lambda m: pd.DataFrame({"knn_resA_mean": m[:, 0], "knn_resB_mean": m[:, 1], "knn_dist_mean": d2}, index=ta)[sv]
        out[f"M6b_faiss_resid_k{k}"] = mk(mu); out[f"M6b_faiss_resid_k{k}_shuf"] = mk(smu)
        info.setdefault("residual_variants", {})[f"k{k}"] = {"served_rows": int(sv.sum()), "rule_check": rule_check(ta, nbr2, okr, k)}
    for nm, df in out.items():
        meta = VC.write_view(nm, source, df, note=f"M6 {nm}; {look}; rows with < {M.MIN_POOL} past anchors dropped")
        info.setdefault("views", {})[nm] = {"n_rows": meta["n_rows"], "n_ok_anchors_covered": meta["n_ok_anchors_covered"], "sha256": meta["sha256"]}
    info["seconds"] = round(time.time() - t0, 1)
    return info


def rule_check(ta, nbr, pool_ok, k):
    """verify on every served neighbour: ta_j + 60 min <= ta_i - 1 day, j in the ok pool; returns counts."""
    ta = pd.DatetimeIndex(ta); i, c = np.where(nbr[:, :k] >= 0); j = nbr[i, c]
    gap = (ta[i] - ta[j]).values.astype("timedelta64[m]").astype(np.int64)
    return {"served_pairs": int(len(i)), "min_gap_minutes_ta_i_minus_ta_j": int(gap.min()) if len(gap) else None,
            "all_label_windows_ended_1_day_before": bool(len(gap) == 0 or (gap >= 24 * 60 + M.LABEL_MIN).all()),
            "all_neighbours_in_pool": bool(pool_ok[j].all()), "self_neighbours": int((i == j).sum())}


if __name__ == "__main__":
    t0 = time.time(); allinfo = {}
    for src, lk in (("cut", "look1"), ("fwd", "look2")):
        allinfo[src] = build_source(src, lk)
    allinfo["total_seconds"] = round(time.time() - t0, 1)
    json.dump(allinfo, open(os.path.join(HERE, "checks.json"), "w"), indent=1)
    print(json.dumps(allinfo, indent=1)[:4000])

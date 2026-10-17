"""M3b (POST HOC, labelled; written after the pre-registered M3_kronos result): N = 8 paths (the n4 paths plus 4 more independent ones, tag n4b)
and log-scale / smoothed-logit forms of the same path statistics, so they enter H3's logit and V3's ridge on the scale those models live on.
columns: kr_log_range (ln median-over-paths range), kr_log_vol, kr_phit_logit = logit((hits + 0.5) / (N + 1)), kr_log_retsd (ln sd across paths of the hour return),
point_log_rv (N = 8 median, K refitted). The first-passage probability of H3 itself is not recomputed here: H3's own logit is in the base design and the
view columns are appended to it (this is the blend). usage: make_view_b.py"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import anchors as AN, view_cache as VC, make_view as MV

NAME = "M3b_kronos_blend"
rep = {}
for source in ("cut", "fwd"):
    t1, P1 = MV.load_paths("n4", source); t2, P2 = MV.load_paths("n4b", source)
    common = t1.intersection(t2); P = np.concatenate([P1[t1.get_indexer(common)], P2[t2.get_indexer(common)]], axis=1); ta = common
    inp = np.load(os.path.join(AN.CACHE_DIR, "m3", f"inputs_{source}.npz")); pos = pd.Series(np.arange(len(inp["ta_ns"])), index=pd.DatetimeIndex(inp["ta_ns"].astype("datetime64[ns]")))
    c0 = inp["x"][pos.reindex(ta).to_numpy(), -1, 3]; f = MV.path_features(P, c0); N = P.shape[1]
    K, Ksd, nK = MV.real_ratio_constant(source, AN.load_anchors(source).index[AN.load_anchors(source)["ok"].values])
    ok = np.isfinite(P).all(axis=(1, 2, 3)); hits = f["kr_phit50"] * N; q = (hits + 0.5) / (N + 1)
    df = pd.DataFrame({"kr_log_range": np.log(f["kr_range_med"]), "kr_log_vol": np.log(f["kr_vol"]), "kr_phit_logit": np.log(q / (1 - q)),
                       "kr_log_retsd": np.log(f["kr_ret_sd_across_paths"] + 1e-6), "point_log_rv": np.log(f["rv5_med"] + AN.EPS_RV) + K}, index=ta)[ok]
    meta = VC.write_view(NAME, source, df, note=f"POST HOC M3b: N={N} paths, log forms; K={K:.4f}")
    rep[source] = {"n_rows": len(df), "N": N, "K": K, "sha256": meta["sha256"], "covered": meta["n_ok_anchors_covered"]}; print(source, rep[source])
json.dump(rep, open(os.path.join(HERE, f"make_view_{NAME}.json"), "w"), indent=1)

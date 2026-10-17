"""M3 step 3 (mm_env): sampled Kronos paths -> view columns -> harness view file.

per anchor ta, N sampled paths of 12 five-minute bars (open high low close), c0 = last real close before ta:
  kr_range_med  median over paths of (max high - min low) / c0
  kr_vol        sd of all path 5-minute log returns (log(close_k / close_k-1), close_-1 = c0), pooled over paths and bars
  kr_phit50     fraction of paths whose max over the 12 bars of max(|high/c0 - 1|, |low/c0 - 1|) is >= 50 bps
  kr_ret_sd_across_paths  sd (ddof 1) across paths of the hour return log(close_11 / c0)
  point_log_rv  ln(median over paths of sum of squared path 5-minute log returns + 1e-12) + K_source,
                K_source = mean over TRAINING anchors of [ln RV_1min(next hour) - ln RV_5min(next hour)] of the REAL bars (a constant per source;
                look 1: anchors whose hour ended before 2024-07-01, look 2: before 2025-11-01), T01 scale.
usage: make_view.py --tag n4 --name M3_kronos"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import anchors as AN, view_cache as VC

TRAIN_END = {"cut": pd.Timestamp("2024-07-01"), "fwd": pd.Timestamp("2025-11-01")}
HIT_BPS = 50.0


def load_paths(tag, source):
    fs = sorted(glob.glob(os.path.join(AN.CACHE_DIR, "m3", f"paths_{tag}_{source}", "chunk_*.npz")))
    ta, P = [], []
    for f in fs:
        z = np.load(f); ta.append(z["ta_ns"]); P.append(z["paths"])
    return pd.DatetimeIndex(np.concatenate(ta).astype("datetime64[ns]")), np.concatenate(P)


def path_features(P, c0):
    """P [n, N, 12, 4] ohlc, c0 [n] -> dict of arrays (+ 'rv5_med' median over paths of the sum of squared 5-min returns)."""
    o, h, l, c = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    hi = np.maximum.reduce([o, h, l, c]); lo = np.minimum.reduce([o, h, l, c])
    c0_ = c0[:, None, None]
    rng = (hi.max(2) - lo.min(2)) / c0[:, None]
    prev = np.concatenate([np.broadcast_to(c0_, c[..., :1].shape), c[..., :-1]], axis=2)
    lr = np.log(c / prev)
    move = np.maximum(np.abs(hi / c0_ - 1), np.abs(lo / c0_ - 1)).max(2)
    out = {"kr_range_med": np.median(rng, axis=1), "kr_vol": lr.reshape(len(lr), -1).std(axis=1),
           "kr_phit50": (move * 1e4 >= HIT_BPS).mean(axis=1),
           "kr_ret_sd_across_paths": np.log(c[..., -1] / c0[:, None]).std(axis=1, ddof=1) if P.shape[1] > 1 else np.full(len(c0), np.nan),
           "rv5_med": np.median((lr ** 2).sum(axis=2), axis=1)}
    return out


def real_ratio_constant(source, anchors_ta):
    """K_source from real bars on training anchors only (ln RV_1min - ln RV_5min of the next hour)."""
    st = AN.get_store(source); m1 = st.m1; rv = AN.rv5_series(source)
    keep = anchors_ta[(anchors_ta + pd.Timedelta(minutes=60)) <= TRAIN_END[source]]
    # 1-minute RV of the next hour: logrv_1h on the bin opening at ta + 55 min
    l1 = rv["logrv_1h"].reindex(keep + pd.Timedelta(minutes=55)).to_numpy()
    # 5-minute RV of the next hour: closes at ta-1min, ta+4min, ..., ta+59min
    cl = m1["close"]
    grid = (keep.values[:, None] + (np.arange(13)[None, :] * 5 - 1) * np.timedelta64(1, "m")).reshape(-1)
    c = cl.reindex(pd.DatetimeIndex(grid)).to_numpy().reshape(len(keep), 13)
    l5 = np.log(np.nansum(np.diff(np.log(c), axis=1) ** 2, axis=1) + AN.EPS_RV); bad = np.isnan(c).any(1)
    d = (l1 - l5)[~bad & ~np.isnan(l1)]
    return float(d.mean()), float(d.std()), int(len(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="n4"); ap.add_argument("--name", default="M3_kronos"); ap.add_argument("--note", default="")
    a = ap.parse_args(); rep = {}
    for source in ("cut", "fwd"):
        ta, P = load_paths(a.tag, source); inp = np.load(os.path.join(AN.CACHE_DIR, "m3", f"inputs_{source}.npz"))
        pos = pd.Series(np.arange(len(inp["ta_ns"])), index=pd.DatetimeIndex(inp["ta_ns"].astype("datetime64[ns]")))
        c0 = inp["x"][pos.reindex(ta).to_numpy(), -1, 3]; assert np.isfinite(c0).all()
        ok = np.isfinite(P).all(axis=(1, 2, 3)); f = path_features(P, c0)
        K, Ksd, nK = real_ratio_constant(source, AN.load_anchors(source).index[AN.load_anchors(source)["ok"].values])
        df = pd.DataFrame({k: v for k, v in f.items() if k != "rv5_med"}, index=ta)
        df["point_log_rv"] = np.log(f["rv5_med"] + AN.EPS_RV) + K
        df = df[ok]
        # sanity vs the harness' own previous-hour log RV
        base = AN.load_base(source); lv0 = pd.Series(base.arr["A_LV0"], index=base.ta)
        meta = VC.write_view(a.name, source, df, note=f"{a.note} K={K:.4f} (sd {Ksd:.3f}, n {nK}); tag {a.tag}; N paths {P.shape[1]}; n anchors with paths {len(ta)}")
        j = df.join(lv0.rename("lv0"), how="inner"); rep[source] = {"K": K, "Ksd": Ksd, "nK": nK, "n_paths_anchors": int(len(ta)), "meta": meta,
            "corr_point_vs_LV0": float(np.corrcoef(j["point_log_rv"], j["lv0"])[0, 1]), "describe": df.describe().round(5).to_dict()}
        print(source, json.dumps({k: v for k, v in rep[source].items() if k != "describe"}, default=str)); print(df.describe().round(5).to_string())
    json.dump(rep, open(os.path.join(HERE, f"make_view_{a.name}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

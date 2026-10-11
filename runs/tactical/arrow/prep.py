"""Build the window cache for the arrow study: D:/nt/nt_arrow_cache/{X,sig,t,pos,idx,r_no}.npy (anchors <= cutoff, sigma > 0).
usage: python prep.py"""
import os, sys, json
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arrowlib as AL

if __name__ == "__main__":
    os.makedirs(AL.CACHE, exist_ok=True)
    s, A, pos, idx = AL.anchor_table()
    W = A[pos[:, None] + np.arange(-AL.L + 1, 1)]
    X, sig = AL.channels(W)
    r = np.load(f"{AL.CACHE_IN}/1m_r.npy")[idx]
    chk = A[pos + AL.H, 3] / A[pos, 3] - 1
    assert np.allclose(r, chk, rtol=1e-6, atol=1e-9), "cache anchors do not line up with the bars"
    t = s.index[pos].values.astype("datetime64[ns]").astype(np.int64)
    assert pd.Timestamp(t.max()) <= AL.LAST_BAR
    keep = sig > 0
    gaps = np.abs(np.log(A[1:, 0] / A[:-1, 3])); gapinfo = {"share_gap_over_1e-4": float((gaps > 1e-4).mean()),
                                                              "median_gap": float(np.median(gaps)), "max_gap": float(gaps.max())}
    r_no = AL.forward_returns_nextopen(A, pos)
    for name, v in (("X", X[keep]), ("sig", sig[keep]), ("t", t[keep]), ("pos", pos[keep]), ("idx", idx[keep]), ("r_no", r_no[keep])):
        np.save(f"{AL.CACHE}/{name}.npy", v)
    info = {"n_anchors": int(keep.sum()), "dropped_flat": int((~keep).sum()), "first": str(pd.Timestamp(t[keep].min())),
            "last_anchor": str(pd.Timestamp(t[keep].max())), "last_bar_loaded": str(s.index.max()), "gap_diagnostic": gapinfo}
    json.dump(info, open(f"{HERE}/prep_info.json", "w"), indent=1); print(info)

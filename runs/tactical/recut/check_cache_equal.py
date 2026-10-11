"""Selection-unchanged check (H66 recut): the recomputed feature caches (cut data) equal the old caches (full data) on every
anchor whose forward-label bar lies before the selection end 2024-06-30 23:59. If so, a selection made on 2021-01..2024-06 from the old
cache is the selection the cut data gives. usage: python check_cache_equal.py recent|flow"""
import sys, json
import numpy as np, pandas as pd
which = sys.argv[1]
old, new = {"recent": ("D:/nt/nt_recent_cache", "D:/nt/nt_recut_cache/recent"), "flow": ("D:/nt/nt_flow_cache", "D:/nt/nt_recut_cache/flow")}[which]
bars = ("1m", "1h") if which == "recent" else ("1m", "5m")
blocks = ("tb7", "rich", "ctx", "gR", "novel", "har") if which == "recent" else ("tb7", "flow")
lim = pd.Timestamp("2024-06-30 23:59").value
res = {}
for bar in bars:
    to, tn = np.load(f"{old}/{bar}_t.npy"), np.load(f"{new}/{bar}_t.npy")
    co, cn = np.searchsorted(to, lim, side="right"), np.searchsorted(tn, lim, side="right")
    ok = bool(co == cn and np.array_equal(to[:co], tn[:cn]))
    # labels: the 15-bar label of an anchor within the last 20 minutes of the span reads bars up to lim+..; compare all rows <= lim - 2 h
    m = int(np.searchsorted(to, lim - 2 * 3600 * 10 ** 9, side="right"))
    out = {"t_equal": ok, "rows": int(cn)}
    ro, rn = np.load(f"{old}/{bar}_r.npy"), np.load(f"{new}/{bar}_r.npy")
    out["r_equal_before_lim_minus_2h"] = bool(np.array_equal(ro[:m], rn[:m]) or np.allclose(ro[:m], rn[:m], rtol=0, atol=1e-12))
    for b in blocks:
        xo, xn = np.load(f"{old}/{bar}_{b}.npy", mmap_mode="r"), np.load(f"{new}/{bar}_{b}.npy", mmap_mode="r")
        out[b] = bool(np.array_equal(np.asarray(xo[:cn]), np.asarray(xn[:cn])))
    res[bar] = out; print(bar, out, flush=True)
json.dump(res, open(f"cache_equal_{which}.json", "w"), indent=1)

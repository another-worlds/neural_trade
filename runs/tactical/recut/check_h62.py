"""H62 verification (H66 recut): rebuild the forward-candidate feature cache from the CUT csv and compare it with the cache the frozen
grid used. fwdlib.load_bars already stops at 2024-07-01, so the grid, the choice and the frozen per-year table cannot have read the
protected span; this shows the rebuilt cache is bit-identical to the old one (hence every grid row, choice.json and frozen_by_year are unchanged)."""
import sys, json
sys.path.insert(0, "D:/nt/nt_recut/runs/tactical/fwd")
import numpy as np
import fwdlib as F
n = F.build_cache()
old = "D:/nt/nt_fwd_cache"; res = {"anchors": int(n), "last_bar_of_cache": str(np.datetime64(int(np.load(f"{F.CACHE}/t.npy")[-1]), "ns"))}
for k in ("tb7", "rich", "t", "anch", "open", "close"):
    res[k + "_equal"] = bool(np.array_equal(np.load(f"{F.CACHE}/{k}.npy"), np.load(f"{old}/{k}.npy")))
print(res); json.dump(res, open("D:/nt/nt_recut/runs/tactical/recut/check_h62.json", "w"), indent=1)

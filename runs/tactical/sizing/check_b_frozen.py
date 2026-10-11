"""Consistency: the R0 sequence of side signal (b) on the selection months equals fwdlib.trades_from (the frozen pipeline)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sizing as R, sizelib as S, fwdlib as F
t = np.load(f"{F.CACHE}/t.npy"); raw = list(R.F.month_predictions(np.asarray(np.load(f"{F.CACHE}/rich.npy")), t, F.next_open_ret(np.load(f"{F.CACHE}/open.npy"), np.load(f"{F.CACHE}/close.npy"), np.load(f"{F.CACHE}/anch.npy"), 4), "logreg", 730, 4, "2021-01", "2024-06"))
ent, ex, side, ret, mon = F.trades_from(raw, t, 4, 0.10)
sg = R.signals(list(R.b_preds("2021-01", "2024-06", F.CACHE)), 4)
k, _ = S.E.non_overlapping(sg["entry"], sg["exit"])
print("fwdlib trades", len(ent), "ours R0", int(k.sum()), "equal entries", bool(np.array_equal(ent, sg["entry"][k])), "equal exits", bool(np.array_equal(ex, sg["exit"][k])))
assert np.array_equal(ent, sg["entry"][k]) and np.array_equal(ex, sg["exit"][k]) and np.allclose(ret, (sg["side"] * sg["r_long"])[k])

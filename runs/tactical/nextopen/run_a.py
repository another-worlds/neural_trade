"""Strategy (a): H48 minute triple barrier, boosting primary on tb7 + boosting meta; train 2020-01..2024-06, test 2024-07..2025-07-20 (recut).
NEXT-OPEN entry (H61): labels from geom2/meta_nextopen.py (c0 = A[anch, 0]); exit recomputation uses the same c0.
Same configuration and code path as runs/tactical/geom2/holdout_min.py (its `trades`), plus the exit time of each trade.
usage: python run_a.py"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); GEOM = os.path.join(os.path.dirname(HERE), "geom2")
sys.path.insert(0, GEOM); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'equity')); sys.path.insert(0, HERE)
import meta_nextopen as M  # noqa: E402
import eqlib as E  # noqa: E402

t0 = time.time()
M.SETUP["1min_full"] = dict(M.SETUP["1min"], end="2100-01-01")
S = dict(M.SETUP["1min_full"]); S["primary"] = "hgb"
D, A, anch = M.labels("1min_full", S["T"], S["tp"], S["sl"])
F = []
for k in range(0, len(anch), 20000):
    W = np.stack([A[i - M.L:i] for i in anch[k:k + 20000]]).astype(np.float32); F.append(np.nan_to_num(M.lab.fs_tb7(W)))
F = np.concatenate(F)
hold = M.HOLD.value; gap_ns = 2 * S["T"] * 60 * 10 ** 9
tr = np.where(D["t_exit"] < hold - gap_ns)[0]; te = np.where(D["t_entry"] >= hold)[0]
print(f"anchors {len(anch)}, train {len(tr)}, test {len(te)}", flush=True)

# --- the selection of holdout_min.trades (copied logic; HGB determinism: random_state 0)
from sklearn.preprocessing import StandardScaler  # noqa: E402
y = (D["rT"] > 0).astype(int); gap = max(1, 2 * S["T"] // S["stride"]); blocks = np.array_split(tr, 4)
X = StandardScaler().fit(F[tr]).transform(F); clf = lambda: M.hgb(S["min_leaf"])
oi, op = [], []
for k in (1, 2, 3):
    fi = np.concatenate(blocks[:k]); fi = fi[fi < blocks[k][0] - gap]
    oi.append(blocks[k]); op.append(clf().fit(X[fi], y[fi]).predict_proba(X[blocks[k]])[:, 1])
p = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
mx = lambda idx, q: np.hstack([X[idx], np.abs(q - 0.5)[:, None], np.log(D["sig"][idx])[:, None]])
pnl = lambda side, idx: np.where(side > 0, D["pl_long"][idx], D["pl_short"][idx])
i23, p23 = np.concatenate(oi[:2]), np.concatenate(op[:2]); i4, p4 = oi[2], op[2]
meta = M.hgb(S["min_leaf"]).fit(mx(i23, p23), (pnl(np.where(p23 > 0.5, 1, -1), i23) > 0).astype(int))
thr_m = np.quantile(meta.predict_proba(mx(i4, p4))[:, 1], 1 - S["q"])
side = np.where(p > 0.5, 1, -1); sel = meta.predict_proba(mx(te, p))[:, 1] >= thr_m
pl = pnl(side, te)
print(f"meta selection: n {sel.sum()}, hit {np.mean(pl[sel] > 0):.4f}, bps {pl[sel].mean() * 1e4:.3f}  (H48: n 13818, hit 0.5227, bps 0.540)", flush=True)

# --- exit times: the bar in which the first barrier was touched (or the label's last bar), end of that bar
T = S["T"]; c0 = A[anch, 0]; unit = D["unit"]
Hf = np.lib.stride_tricks.sliding_window_view(A[:, 1], T)[anch]; Lf = np.lib.stride_tricks.sliding_window_view(A[:, 2], T)[anch]
ft = M.first_touch
lt, ls = ft(Hf >= (c0 * (1 + S["tp"] * unit))[:, None]), ft(Lf <= (c0 * (1 - S["sl"] * unit))[:, None])
st_, ss = ft(Lf <= (c0 * (1 - S["tp"] * unit))[:, None]), ft(Hf >= (c0 * (1 + S["sl"] * unit))[:, None])
rT = A[anch + T - 1, 3] / c0 - 1
chk_l = np.where(ls <= lt, np.where(ls < T, -S["sl"] * unit, rT), np.where(lt < T, S["tp"] * unit, rT))
assert np.allclose(chk_l, D["pl_long"]), "recomputed long P&L differs from the labels"
jl, js = np.minimum(np.minimum(ls, lt), T - 1), np.minimum(np.minimum(ss, st_), T - 1)
bar_ns = 60 * 10 ** 9
j_te = np.where(side > 0, jl[te], js[te])
entry_ns = D["t_entry"][te] + bar_ns                               # position opens at the close of the anchor's last bar
exit_ns = D["t_entry"][te] + (j_te + 2) * bar_ns                   # bar anch+j ends 2+j bars after the open of bar anch-1
k = sel
res = E.finish("a_h48_minute_barrier_meta_nextopen", "(a) H48 minute triple barrier, boosting primary + meta, 2024-07..2025-07-20 (recut)", entry_ns[k], exit_ns[k],
               D["pl_long"][te][k], D["pl_short"][te][k], side[k], int(hold), HERE,
               extra={"signals_all_selected": int(k.sum()), "config": "1min T15 M1 q0.10 hgb tb7 + hgb meta (holdout_min.py)",
                      "hold_bars_median": float(np.median(j_te[k] + 1)), "nextopen_meta_selection_(H48_close: n 13818 hit 0.5227 bps 0.540)": {"n": int(k.sum()), "hit": float(np.mean(pl[k] > 0)), "bps": float(pl[k].mean() * 1e4)}})
print(json.dumps({c: v["stats"] for c, v in res["levels"].items()}, indent=1)); print(f"n_signals {res['n_signals']} trades {res['n_trades']} skipped {res['n_skipped']}; {time.time() - t0:.0f}s")

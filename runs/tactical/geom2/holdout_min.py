"""ONE held-out check of the minute triple barrier with a boosting primary (lead, 2026-10-11 ~05:30; fixed before the run).
Why: H46/H47 found the only consistent minute edge on dev folds: a HistGradientBoosting primary on tb7 + a boosting meta model
(H47: HGB primary base, excess over the random null +1.06 [+0.32,+1.79] bps, 8 dev folds 2020-01..2024-06).
Configuration (frozen, from meta.py SETUP '1min'): 1-minute bars, anchors every 5 bars, T 15, tp = sl = 1.0 sigma sqrt(T), q 0.10,
primary = HGB on tb7 (sign of the 15-bar return), meta = HGB on tb7 + |P-0.5| + log sigma trained on out-of-fold primary predictions.
Train: entries from 2020-01-01 whose exits end before 2024-07-01 (minus the 2T gap); test: entries from 2024-07-01 to the data end.
Success (fixed now): the META selection's mean bps per trade is above its random-side null95 AND the 95% t-interval over calendar
months is above 0. The primary selection is reported, not judged. Caveat: H33's walk-forward folds (logistic primary, in-sample meta)
already covered this period, so it is not pristine for the minute barrier; no choice in this configuration used it.
usage: python holdout_min.py"""
import json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import meta as M  # noqa: E402

M.SETUP["1min_full"] = dict(M.SETUP["1min"], end="2100-01-01")
t0 = time.time(); kind = "1min_full"; S = dict(M.SETUP[kind]); S["primary"] = "hgb"
lab_, A, anch = M.labels(kind, S["T"], S["tp"], S["sl"])
F = []
for k in range(0, len(anch), 20000):
    W = np.stack([A[i - M.L:i] for i in anch[k:k + 20000]]).astype(np.float32); F.append(np.nan_to_num(M.lab.fs_tb7(W)))
F = np.concatenate(F)
D = dict(lab_); hold = M.HOLD.value; gap_ns = 2 * S["T"] * 60 * 10 ** 9
tr = np.where(D["t_exit"] < hold - gap_ns)[0]; te = np.where(D["t_entry"] >= hold)[0]
print(f"anchors {len(anch)}, train {len(tr)}, test {len(te)}, prep {time.time() - t0:.0f}s", flush=True)
rng = np.random.default_rng(0)
out = M.split(D, S, F, F, tr, te, rng)
# monthly interval for each selection: recompute the selection's per-trade P&L by month
res = {"config": "1min T15 M1 q0.10 primary hgb tb7, meta hgb base", "train_n": int(len(tr)), "test_n": int(len(te))}
for name, v in out.items():
    res[name] = v
print(json.dumps(res, indent=1), flush=True)


# per-month P&L of the selections: repeat the selection inside split's logic to get trade-level rows
def trades(D, S, F, tr, te):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression  # noqa: F401
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
    thr_m = np.quantile(meta.predict_proba(mx(i4, p4))[:, 1], 1 - S["q"]); thr_c = np.quantile(np.abs(p4 - 0.5), 1 - S["q"])
    pl = pnl(np.where(p > 0.5, 1, -1), te)
    return pl, {"primary": np.abs(p - 0.5) >= thr_c, "meta": meta.predict_proba(mx(te, p))[:, 1] >= thr_m}


pl, sel = trades(D, S, F, tr, te)
t = pd.to_datetime(D["t_entry"][te])
for name, k in sel.items():
    mon = pd.Series(pl[k] * 1e4, index=t[k]).groupby(pd.Grouper(freq="M")).mean().dropna()
    se = mon.std(ddof=1) / math.sqrt(len(mon)); tq = {15: 2.14, 13: 2.18}.get(len(mon), 2.0)   # t_.975,df (13 months in the recut: 2.179)
    res[name + "_monthly"] = {"months": int(len(mon)), "mean": float(mon.mean()), "ci": [float(mon.mean() - tq * se), float(mon.mean() + tq * se)],
                              "positive_months": int((mon > 0).sum())}
m = res["meta"]; mm = res["meta_monthly"]
res["success"] = bool(m and m["bps"] > m["null95"] and mm["ci"][0] > 0)
print(json.dumps({k: res[k] for k in res if k.endswith("monthly") or k == "success"}, indent=1))
open(os.path.join(HERE, "holdout_min_recut.json"), "w").write(json.dumps(res, indent=1))
print(f"total {time.time() - t0:.0f}s")

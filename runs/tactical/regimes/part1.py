"""Part 1 (SPEC.md): regime cells of the minute direction signal (tb7 logistic regression) over the 24 cached slices."""
import glob, json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "D:/nt/nt_tactical/runs/tactical/lab")
import lab  # noqa: E402  (chdirs to D:/nt/nt_tactical; read-only use)
import regime_vars as R  # noqa: E402

OUT = os.path.join(HERE, "results.jsonl"); GAP = 20; MINB = 300


def load_minutes():
    s = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    return s.index, s.values.astype(np.float32)


def locate(A, W):
    """position of W[0]'s last bar in A, verified on the first, a middle and the last window (stride 1)."""
    w0 = W[0]; cand = np.where(np.isclose(A[:, 3], w0[-1, 3], rtol=0, atol=1e-3) & np.isclose(A[:, 0], w0[-1, 0], rtol=0, atol=1e-3))[0]
    for i in cand:
        if i >= 59 and all(np.allclose(A[i - 59 + k:i + 1 + k], W[k], rtol=1e-5) for k in (0, len(W) // 2, len(W) - 1)):
            return int(i)
    raise RuntimeError("window not found")


def t_ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return [round(float(m), 5), round(float(m - t * se), 5), round(float(m + t * se), 5)]


def tail(P, rva, db, cellmask, rng):
    n = len(P); A = np.zeros(n, bool); A[:n // 2] = True; B = np.zeros(n, bool); B[n // 2 + GAP:] = True
    s = P.mean(1); conf = np.abs(s - 0.5); ret = rva[:, 1]; m = np.abs(ret) > db
    thr = np.quantile(conf[A], 0.9); k = B & (conf >= thr) & m & cellmask
    if k.sum() < 20:
        return None
    mv = np.sign(s[k] - 0.5) * ret[k] * 1e4
    null = [np.mean(np.sign(s[k] - 0.5) * rng.choice([-1, 1], k.sum()) * ret[k] * 1e4) for _ in range(200)]
    return {"n": int(k.sum()), "hit": float(np.mean(mv > 0)), "bps": float(np.mean(mv)), "null95": float(np.quantile(null, 0.95))}


def main():
    t0 = time.time(); idx, A = load_minutes(); rng = np.random.default_rng(0)
    cells = [(v, j) for v in R.CELLS for j in range(len(R.CELLS[v]))]
    d_auc = {c: {} for c in cells}; tails = {c: {} for c in cells}; tail_all = {}; slices = []
    for p in sorted(glob.glob(lab.CACHE)):
        D = np.load(p); key = str(D["data_end"])[:13]; db = float(D["deadband"]); rtr, rva = D["rtr"], D["rva"]
        p_tr = locate(A, D["Wtr"]) + np.arange(len(D["Wtr"])); p_va = locate(A, D["Wva"]) + np.arange(len(D["Wva"]))
        Ftr, Fva = np.nan_to_num(lab.fs_tb7(D["Wtr"])), np.nan_to_num(lab.fs_tb7(D["Wva"]))
        sc = StandardScaler().fit(Ftr); Ftr, Fva = sc.transform(Ftr), sc.transform(Fva)
        P = np.zeros((len(Fva), 3))
        for i in range(3):
            mtr = np.abs(rtr[:, i]) > db
            P[:, i] = lab.model_of("logreg", 0.1).fit(Ftr[mtr], (rtr[mtr, i] > 0).astype(int)).predict_proba(Fva)[:, 1]
        codes, cut = R.cell_codes(A[:, 3], p_va, idx[p_va] + pd.Timedelta("1min"), train_pos=p_tr)

        def auc3(mask):
            a = []
            for i in range(3):
                m = mask & (np.abs(rva[:, i]) > db); y = (rva[m, i] > 0).astype(int)
                if m.sum() < MINB or y.min() == y.max():
                    return None
                a.append(roc_auc_score(y, P[m, i]))
            return float(np.mean(a))
        allm = np.ones(len(P), bool); overall = auc3(allm); ta = tail(P, rva, db, allm, rng); tail_all[key] = ta
        row = {"slice": key, "overall_auc3": overall, "cut_vol": [float(x) for x in cut["vol"]], "n_val": len(P)}
        for c in cells:
            mask = codes[c[0]] == c[1]; a = auc3(mask)
            if a is not None:
                d_auc[c][key] = a - overall; row[f"auc_{c[0]}_{R.CELLS[c[0]][c[1]]}"] = a
            tt = tail(P, rva, db, mask, rng)
            if tt and ta:
                tails[c][key] = {**tt, "d_hit": tt["hit"] - ta["hit"], "d_bps": tt["bps"] - ta["bps"]}
        slices.append(row); print(key, round(overall, 4), flush=True)
    res = []
    for c in cells:
        v = np.array(list(d_auc[c].values())); r = {"var": c[0], "cell": R.CELLS[c[0]][c[1]], "n_slices": len(v)}
        if len(v) >= 12:
            tstat = v.mean() / (v.std(ddof=1) / math.sqrt(len(v))); r["mean_d_auc"], r["ci_lo"], r["ci_hi"] = t_ci(v)
            r["p_greater"] = float(stats.t.sf(tstat, len(v) - 1)); r["p_less"] = float(stats.t.cdf(tstat, len(v) - 1))
            r["n_pos"] = int((v > 0).sum())
        tv = list(tails[c].values())
        if len(tv) >= 6:
            for m in ("hit", "bps", "null95", "d_hit", "d_bps"):
                r[f"tail_{m}"] = t_ci([x[m] for x in tv])
            r["tail_slices"] = len(tv)
        res.append(r)
    tested = [r for r in res if "p_greater" in r]; order = sorted(range(len(tested)), key=lambda i: tested[i]["p_greater"]); m = len(tested); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * tested[i]["p_greater"])); tested[i]["p_holm_greater"] = run
        tested[i]["helps"] = bool(tested[i]["mean_d_auc"] > 0 and run < 0.05)
    order = sorted(range(m), key=lambda i: tested[i]["p_less"]); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * tested[i]["p_less"])); tested[i]["p_holm_less"] = run; tested[i]["weaker"] = bool(tested[i]["mean_d_auc"] < 0 and run < 0.05)
    overall_ci = t_ci([s["overall_auc3"] for s in slices])
    rec = {"part": 1, "n_slices": len(slices), "overall_auc3": overall_ci, "tail_all_bps": t_ci([x["bps"] for x in tail_all.values() if x]),
           "tail_all_hit": t_ci([x["hit"] for x in tail_all.values() if x]), "cells": res, "slices": slices, "seconds": round(time.time() - t0, 1)}
    open(OUT, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
    print("overall", overall_ci)
    for r in res:
        print(r["var"], r["cell"], r["n_slices"], r.get("mean_d_auc"), r.get("ci_lo"), r.get("ci_hi"), "pH", r.get("p_holm_greater"), r.get("helps"), r.get("weaker"))


if __name__ == "__main__":
    main()

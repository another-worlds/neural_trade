"""Night experiment 5: geometry on HOURLY bars for direction (8 dev folds before 2024-07-01, hourly.py's fold edges).
Target: sign of the T-bar return, T in (1, 4, 12, 24) hours. Sets: tb7, rich, ctx (rich + hourly context), g5, gR and unions.
Models: logistic regression (C 0.1) and HistGradientBoosting. Per fold: AUC; top-20% confidence hit rate and mean signed return (bps)
against a random-sign null95 on the same bars. Paired fold differences against tb7 logreg.
usage: python hdir.py run [--shard i --of k]   |   python hdir.py report"""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(TAC, "lab")); sys.path.insert(0, os.path.join(TAC, "vol"))
import lab  # noqa: E402
import geomlib as G  # noqa: E402
import volbench as V  # noqa: E402

L = 60; HOLD = pd.Timestamp("2024-07-01"); HORIZONS = (1, 4, 12, 24)
SETS = ["tb7", "rich", "ctx", "g5", "gR", "tb7+g5", "tb7+gR", "rich+gR", "ctx+gR"]
FC = os.path.join(HERE, "featcache"); os.makedirs(FC, exist_ok=True)


def data():
    p = f"{FC}/hdir.npz"
    if os.path.exists(p):
        return dict(np.load(p))
    b = V.hourly_data(); A = b.values.astype(np.float64)
    anch = np.arange(L + 168, len(A) - 24)          # one anchor set for every horizon (the last 24 bars never anchor)
    F, hp = V.hourly_features(b, anch)               # rich (43) + ctx (rest)
    W = np.stack([A[i - L:i] for i in anch]).astype(np.float32)
    nr = lab.fs_rich(W[:2]).shape[1]
    D = dict(anch=anch, t_entry=b.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64), c0=A[anch - 1, 3],
             F_tb7=np.nan_to_num(lab.fs_tb7(W)).astype(np.float32), F_rich=F[:, :nr].astype(np.float32), F_ctx=F.astype(np.float32),
             F_g5=G.geom_base5(W), F_gR=G.geom_rich(W))
    for T in HORIZONS:
        D[f"r{T}"] = A[anch + T - 1, 3] / A[anch - 1, 3] - 1
        D[f"t_exit{T}"] = b.index[anch + T - 1].values.astype("datetime64[ns]").astype(np.int64)
    np.savez(p, **D); return D


def feats(D, name):
    return np.column_stack([D[f"F_{n}"] for n in name.split("+")])


def model(kind):
    if kind == "logreg":
        return LogisticRegression(C=0.1, max_iter=1000)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1.0, random_state=0)


def run(shard, of):
    D = data(); hold = np.datetime64(HOLD, "ns").astype(np.int64); rng = np.random.default_rng(0)
    out = os.path.join(HERE, f"hdir_rows_{shard}.jsonl"); jobs = [(T, s, m) for T in HORIZONS for s in SETS for m in ("logreg", "hgb")][shard::of]
    for T, sname, kind in jobs:
        t0 = time.time(); X = feats(D, sname); r = D[f"r{T}"]; y = (r > 0).astype(int)
        dev = np.arange(len(X))[D[f"t_exit{T}"] < hold]; edges = np.linspace(len(dev) * 0.3, len(dev), 9).astype(int); res = []
        for f in range(8):
            tr = dev[:max(1, edges[f] - 2 * T)]; te = dev[edges[f]:edges[f + 1]]
            sc = StandardScaler().fit(X[tr]); m = model(kind).fit(sc.transform(X[tr]), y[tr]); p = m.predict_proba(sc.transform(X[te]))[:, 1]
            conf = np.abs(p - 0.5); ptr = m.predict_proba(sc.transform(X[tr]))[:, 1]
            k = conf >= np.quantile(np.abs(ptr - 0.5), 0.8); sgn = np.where(p > 0.5, 1, -1); mv = sgn[k] * r[te][k] * 1e4
            null = [np.mean(rng.choice([-1, 1], k.sum()) * r[te][k] * 1e4) for _ in range(200)]
            res.append(dict(auc=float(roc_auc_score(y[te], p)), hit=float(np.mean(mv > 0)), bps=float(np.mean(mv)), null95=float(np.quantile(null, 0.95)), n=int(k.sum())))
        open(out, "a", encoding="utf-8").write(json.dumps(dict(T=T, set=sname, model=kind, folds=res)) + "\n")
        print(f"T{T} {sname} {kind}: auc {np.mean([x['auc'] for x in res]):.4f} {time.time() - t0:.0f}s", flush=True)


def report():
    rows = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith("hdir_rows_"):
            rows += [json.loads(l) for l in open(os.path.join(HERE, fn))]
    by = {(r["T"], r["set"], r["model"]): r["folds"] for r in rows}
    tq = 2.36; fmt = lambda v: f"{np.mean(v):+.4f} [{np.mean(v) - tq * np.std(v, ddof=1) / math.sqrt(8):+.4f},{np.mean(v) + tq * np.std(v, ddof=1) / math.sqrt(8):+.4f}]"
    lines = []
    for T in HORIZONS:
        lines.append(f"\n== T = {T} h; AUC mean over 8 dev folds [95% CI]; d = paired diff vs tb7 logreg; top-20% bps minus random-sign null95")
        base = [x["auc"] for x in by.get((T, "tb7", "logreg"), [])]
        for s in SETS:
            for m in ("logreg", "hgb"):
                fo = by.get((T, s, m))
                if not fo:
                    continue
                a = np.array([x["auc"] for x in fo]); d = a - np.array(base) if base else None
                ex = np.array([x["bps"] - x["null95"] for x in fo])
                lines.append(f"{s:9s} {m:6s} AUC {fmt(a)} | d {fmt(d) if d is not None and (s, m) != ('tb7', 'logreg') else '-':28s} | top20 hit {np.mean([x['hit'] for x in fo]):.3f} "
                             f"bps {np.mean([x['bps'] for x in fo]):+.1f} excess over null95 {fmt(ex)}")
    txt = "\n".join(lines); print(txt); open(os.path.join(HERE, "hdir_report.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    run(a.shard, a.of) if a.cmd == "run" else report()

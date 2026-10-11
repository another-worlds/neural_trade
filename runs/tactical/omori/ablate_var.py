"""POST-HOC, descriptive, not gating (written after the #6 selection result): which feature group carries the challenger's FZ0 gain at T = 15?
Arms (lower and upper tail, a = 2.5%, 5%): C (all 8 features), sv+clock (semivariance ratio and harmonics only), shock (jump age, Omori K, foreshock only).
usage: python ablate_var.py [--workers 4] -> results/var_ablate.jsonl, results/var_ablate_table.txt"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import omorilib as O, varlib as V
RES = os.path.join(HERE, "results"); _G = {}
GROUPS = {"all": slice(0, 8), "sv_clock": [0, 4, 5, 6, 7], "shock": [1, 2, 3]}


def _init():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    _G["B"] = O.Bars(); st = json.load(open(f"{RES}/state.json")); _G["p"], _G["c"] = st["p_star"], st["c_star"]


def _job(m):
    T = 15; arr = V.month_arrays(_G["B"], m, T, _G["p"], _G["c"])
    if arr is None:
        return m, None
    d = arr["d"]; z = arr["z"]; ok = np.isfinite(z); tr = d.tr & ok; te = d.te & ok
    if tr.sum() < 5000 or te.sum() < 500:
        return m, None
    rows = []
    for tail in ("left", "right"):
        y = z if tail == "left" else -z; X = V.tail_features(arr, tail)
        for al in (0.025, 0.05):
            q, es = V.fhs(y[tr], al); rec = {"tail": tail, "alpha": al, "L_fhs": float(V.fz0(y[te], q, es, al).mean())}
            for g, cols in GROUPS.items():
                Xs_tr = O.standardise(X[tr][:, cols], X[tr][:, cols]); Xs_te = O.standardise(X[tr][:, cols], X[te][:, cols])
                th = V.fit_fz0(Xs_tr, y[tr], al, q, es); v, e = V.predict(th, Xs_te); rec[f"L_{g}"] = float(V.fz0(y[te], v, e, al).mean())
            rows.append(rec)
    return m, rows


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    months = [str(m) for m in pd.period_range("2019-01", "2024-06", freq="M")]; t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(w, initializer=_init) as pool:
        res = [r for r in pool.map(_job, months) if r[1] is not None]
    with open(f"{RES}/var_ablate.jsonl", "w") as fh:
        for m, rows in res:
            fh.write(json.dumps({"month": m, "rows": rows}) + "\n")
    L = [f"post-hoc ablation, T=15, {len(res)} months, {time.time()-t0:.0f}s (descriptive; not a verdict)"]
    for tail in ("left", "right"):
        for al in (0.025, 0.05):
            for g in GROUPS:
                gains = np.array([[r for r in rows if r["tail"] == tail and r["alpha"] == al][0] for m, rows in res])
                d = np.array([x["L_fhs"] - x[f"L_{g}"] for x in gains]); lf = np.array([x["L_fhs"] for x in gains])
                se = d.std(ddof=1) / math.sqrt(len(d)); t = d.mean() / se
                L.append(f"{tail:5s} a={al:.3f} {g:9s} gain vs FHS {100*d.mean()/lf.mean():+6.2f}%  t {t:+6.2f}  p1 {1-stats.t.cdf(t, len(d)-1):.4f}")
    open(f"{RES}/var_ablate_table.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))

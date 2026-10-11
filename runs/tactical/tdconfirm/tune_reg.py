"""The one tuning of the regularised direct net (SPEC.md): L2 x steps on 2020-02..2021-12, W 30/60/90, H 240/1440, seed 0.
Writes reg_config.json (chosen config + the full grid of scores). usage: python tune_reg.py [--part i --nparts n | --merge]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_lib as C  # noqa: E402
import tdvar_lib as L  # noqa: E402
import tdvar_nn as N  # noqa: E402

L2S = (1e-3, 1e-2, 1e-1, 1.0)
STEPS = (25, 50, 100, 200)
MONTHS = list(pd.period_range("2020-02", "2021-12", freq="M"))
TUNE_W = (30, 60, 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    close, idx = L.load_bars()
    D = L.Data(close, idx)
    end_pos = int(idx.searchsorted(pd.Timestamp("2024-07-01")))
    out = f"{HERE}/tune_part{a.part}.jsonl"
    t00 = time.time()
    open(out, "w").close()
    for m in MONTHS[a.part::a.nparts]:
        t0, t1 = L.month_bounds(m)
        for W in TUNE_W:
            for H in C.HORIZONS:
                a_tr, a_te, pos0, lo = C.anchors(D, t0, t1, W, H, end_pos)
                if len(a_tr) < 20 or len(a_te) < 3:
                    continue
                X = D.feat(a_tr)
                lx = np.log(X + N.EPS); mu, sd = lx.mean(0), lx.std(0) + 1e-9
                z, zte = N.zfeat(X, mu, sd), N.zfeat(D.feat(a_te), mu, sd)
                y_tr, y_te = D.rate(a_tr, H), D.rate(a_te, H)
                med = float(np.median(y_tr))
                if med <= 0:
                    continue
                ff, yf = L.FLOOR_REL * med, L.Y_FLOOR_REL * med
                for l2 in L2S:
                    for st in STEPS:
                        with C.net_settings(l2, st):
                            net = N.Net(z, y_tr, yf, seed=0)
                        q = L.qlike(y_te, np.maximum(net.predict(zte), ff), yf, ff)
                        with open(out, "a") as f:
                            f.write(json.dumps(dict(month=str(m), W=W, H=H, l2=l2, steps=st, qlike=q)) + "\n")
        print(m, f"{time.time() - t00:.0f}s", flush=True)


def merge():
    import glob
    rows = []
    for p in sorted(glob.glob(f"{HERE}/tune_part*.jsonl")):
        rows += [json.loads(line) for line in open(p)]
    df = pd.DataFrame(rows)
    base = df[(df.l2 == 1e-3) & (df.steps == 200)].set_index(["month", "W", "H"])["qlike"]
    df["ratio"] = df.apply(lambda r: r.qlike / base[(r.month, r.W, r.H)], axis=1)
    g = df.groupby(["l2", "steps"])["ratio"].mean().reset_index()
    g["n_cells"] = df.groupby(["l2", "steps"])["ratio"].size().values
    lo = g["ratio"].min()
    cand = g[g["ratio"] <= lo + 1e-4].sort_values(["l2", "steps"], ascending=[False, True])
    best = cand.iloc[0]
    cfg = dict(l2=float(best.l2), steps=int(best.steps), score=float(best.ratio),
               grid=[dict(l2=float(r.l2), steps=int(r.steps), mean_ratio=float(r.ratio), n_cells=int(r.n_cells))
                     for r in g.itertuples()])
    json.dump(cfg, open(f"{HERE}/reg_config.json", "w"), indent=1)
    print(g.to_string())
    print("CHOSEN", cfg["l2"], cfg["steps"], cfg["score"])


if __name__ == "__main__":
    main()

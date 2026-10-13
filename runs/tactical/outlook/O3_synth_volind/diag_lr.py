"""Exploratory diagnostic (NOT the verdict): does a slower optimiser (lr 1e-3, patience 5, 60 epochs) let the kernel model do better than the
pre-registered lr 1e-2 on the look months at H=60? usage: python diag_lr.py"""
import json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o3_lib as L  # noqa: E402
T = L.T
tr = T.load_bars(T.TRAIN_CSV, L.LOAD_FROM, T.CUTOFF); F = L.build_features(tr); out = []
for m in L.M_LOOK:
    p = pd.Period(m, freq="M"); m0, m1 = p.start_time, (p + 1).start_time
    st = L.fit_month(F, 60, m0, lk_kwargs=dict(lr=1e-3, patience=5, max_epochs=60)); te = L.test_rows(F["ta"], F["ok60"], m0, m1)
    r = {x["model"]: x for x in L.score_state(st, F, te, "look", m)}
    out.append({"month": m, "epochs": st["m"]["LK"]["epochs"], **{k: (r[k]["spearman"], r[k]["qlike"]) for k in ("R0", "BANK", "LK", "LKsh")}})
    print(out[-1], flush=True)
json.dump(out, open(f"{HERE}/diag_lr.json", "w"))
d = lambda a, b, i: np.array([o[a][i] - o[b][i] for o in out])
for a, b in (("LK", "R0"), ("LK", "BANK")):
    for i, nm in enumerate(("Spearman", "QLIKE")):
        x = d(a, b, i); mu, lo, hi, t = T.ci(x); print(f"lr1e-3 look H60 {a}-{b} d{nm} {mu:+.4f} [{lo:+.4f},{hi:+.4f}] +{int((x > 0).sum())}/{len(x)}")

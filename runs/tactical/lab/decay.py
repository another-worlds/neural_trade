"""Is minute-scale direction predictability decaying? (lead, 2026-10-11, after H50: the barrier edge on 2017-2019 is ~4x the
2020-2024 one, and the lab slices' regression AUC correlates -0.73 with the year.)
Monthly walk-forward over the whole 2017-01..2025-09 minute file: anchors every 10 bars; tb7 features on the 60-bar window;
target = the sign of the 15-minute return beyond the 5-bps deadband; for each calendar month M, a logistic regression (C 0.1)
is trained on the 60 days before M (ending 15 bars before M starts) and scored on M: AUC, the top-10% hit rate and mean bps
(threshold from the training predictions), and the month's realised volatility. Descriptive; no choice is made from it.
usage: python decay.py -> runs/tactical/lab/decay.json"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/lab")
from lab import fs_tb7  # noqa: E402

t0 = time.time(); L, H, STRIDE, DB = 60, 15, 10, 5e-4
s = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"])
s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index(); A = s.values.astype(np.float64)
anch = np.arange(L, len(A) - H, STRIDE); t = s.index[anch - 1]
r = A[anch + H - 1, 3] / A[anch - 1, 3] - 1
F = np.concatenate([np.nan_to_num(fs_tb7(np.stack([A[i - L:i] for i in anch[k:k + 20000]]).astype(np.float32)))
                    for k in range(0, len(anch), 20000)])
print(f"anchors {len(anch)}, features {time.time() - t0:.0f}s", flush=True)
mask = np.abs(r) > DB; y = (r > 0).astype(int); out = []
for m in pd.period_range(t[0].to_period("M") + 3, t[-1].to_period("M"), freq="M"):
    m0, m1 = m.start_time, (m + 1).start_time
    tr = (t >= m0 - pd.Timedelta(days=60)) & (t < m0 - pd.Timedelta(minutes=H)) & mask
    te = (t >= m0) & (t < m1)
    if tr.sum() < 2000 or (te & mask).sum() < 500:
        continue
    sc = StandardScaler().fit(F[tr]); clf = LogisticRegression(C=0.1, max_iter=1000).fit(sc.transform(F[tr]), y[tr])
    pte = clf.predict_proba(sc.transform(F[te]))[:, 1]; ptr = clf.predict_proba(sc.transform(F[tr]))[:, 1]
    thr = np.quantile(np.abs(ptr - 0.5), 0.9); k = np.abs(pte - 0.5) >= thr
    side = np.where(pte > 0.5, 1, -1); mv = side[k] * r[te][k] * 1e4
    out.append({"month": str(m), "auc": float(roc_auc_score(y[te & mask], clf.predict_proba(sc.transform(F[te & mask]))[:, 1])),
                "top10_hit": float(np.mean(mv > 0)) if k.sum() else None, "top10_bps": float(np.mean(mv)) if k.sum() else None,
                "n_top10": int(k.sum()), "vol_bps": float(np.std(r[te]) * 1e4)})
json.dump(out, open("runs/tactical/lab/decay.json", "w"), indent=1)
df = pd.DataFrame(out); df["year"] = df["month"].str[:4]
print(df.groupby("year")[["auc", "top10_hit", "top10_bps", "vol_bps"]].mean().round(4).to_string())
print("corr(month index, AUC)", round(np.corrcoef(np.arange(len(df)), df["auc"])[0, 1], 3), f"{time.time() - t0:.0f}s")

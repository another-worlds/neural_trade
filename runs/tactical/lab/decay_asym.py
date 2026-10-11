"""Excursion asymmetry (lead, 2026-10-11, after H53: |move| is more predictable than ever while the sign is not). Target: over the
next 15 one-minute bars, is the upside excursion (max high - entry close) larger than the downside one (entry close - min low)?
Monthly walk-forward as decay.py (60-day training, anchors every 10 bars), logistic regression on (a) tb7, (b) tb7 + HAR log-RV
(15/60/240/1440) + hour of day + upper/lower wick and range features of the window; AUC per month, yearly means; plus, for
comparison, the same models on the sign of the 15-minute close-to-close return. Descriptive. usage: python decay_asym.py"""
import json, os, sys, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/lab")
from lab import fs_tb7  # noqa: E402

t0 = time.time(); L, H, STRIDE = 60, 15, 10
s = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"])
s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index(); A = s.values.astype(np.float64)
lr = np.concatenate([[0.0], np.diff(np.log(A[:, 3]))]); cs2 = np.concatenate([[0.0], np.cumsum(lr ** 2)])
anch = np.arange(1441, len(A) - H, STRIDE); t = s.index[anch - 1]; c0 = A[anch - 1, 3]
up = swv(A[:, 1], H)[anch].max(1) / c0 - 1; dn = 1 - swv(A[:, 2], H)[anch].min(1) / c0
y_asym = (up > dn).astype(int); keep_asym = np.abs(up - dn) > 1e-6
r = A[anch + H - 1, 3] / c0 - 1; y_sign = (r > 0).astype(int); keep_sign = np.abs(r) > 5e-4
rv = lambda k: np.log((cs2[anch] - cs2[anch - k]) / k + 1e-12); hr = t.hour.values
W_last = A[anch - 1]; rng_ = W_last[:, 1] - W_last[:, 2] + 1e-12
wick_up = (W_last[:, 1] - np.maximum(W_last[:, 0], W_last[:, 3])) / rng_; wick_dn = (np.minimum(W_last[:, 0], W_last[:, 3]) - W_last[:, 2]) / rng_
TB = np.concatenate([np.nan_to_num(fs_tb7(np.stack([A[i - L:i] for i in anch[k:k + 20000]]).astype(np.float32))) for k in range(0, len(anch), 20000)])
X = {"tb7": TB, "tb7+": np.column_stack([TB, rv(15), rv(60), rv(240), rv(1440), np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), wick_up, wick_dn])}
print(f"anchors {len(anch)}, {time.time() - t0:.0f}s", flush=True)
out = []
for m in pd.period_range(t[0].to_period("M") + 3, t[-1].to_period("M"), freq="M"):
    m0, m1 = m.start_time, (m + 1).start_time
    trb = (t >= m0 - pd.Timedelta(days=60)) & (t < m0 - pd.Timedelta(minutes=H)); teb = (t >= m0) & (t < m1)
    row = {"month": str(m)}
    for tgt, y, keep in (("asym", y_asym, keep_asym), ("sign", y_sign, keep_sign)):
        tr, te = trb & keep, teb & keep
        if tr.sum() < 2000 or te.sum() < 500:
            continue
        for fn, F in X.items():
            sc = StandardScaler().fit(F[tr]); mdl = LogisticRegression(C=0.1, max_iter=1000).fit(sc.transform(F[tr]), y[tr])
            row[f"{tgt}_{fn}"] = float(roc_auc_score(y[te], mdl.predict_proba(sc.transform(F[te]))[:, 1]))
    if len(row) > 1:
        out.append(row)
json.dump(out, open("runs/tactical/lab/decay_asym.json", "w"), indent=1)
df = pd.DataFrame(out); df["year"] = df["month"].str[:4]
print(df.groupby("year")[[c for c in df.columns if c not in ("month", "year")]].mean().round(3).to_string(), f"\n{time.time() - t0:.0f}s")

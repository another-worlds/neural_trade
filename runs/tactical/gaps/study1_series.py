"""Study 1, step 1 (SPEC.md): the two monthly series recomputed on the cutoff data (no bar after 2025-07-20) plus the daily notional volume and
realised variance used for the event checks. Protocols are copies of lab/decay_nextopen.py (AUC) and lab/decay_vol.py (HAR Spearman).
usage: python study1_series.py -> series_auc.json, series_har.json, daily.csv, series_check.txt (reproduction check vs the stored decay_*.json)"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaps_lib as G  # noqa: E402
from lab import fs_tb7  # noqa: E402

os.chdir(os.path.join(G.TAC, ".."))      # lab chdirs on import; keep outputs addressed by HERE
G.nice(); t0 = time.time(); L, H, STRIDE, DB = 60, 15, 10, 5e-4
s = G.load_bars(); A = s.values.astype(np.float64)
print("bars", len(A), "last", s.index[-1], flush=True)

# ---- daily series for the event checks
close = s["close"]; notional = (s["volume"] * s["close"]).resample("1D").sum()
lr = np.log(close).diff().fillna(0.0); rvd = (lr ** 2).resample("1D").sum()
pd.DataFrame({"notional": notional, "rv": rvd}).to_csv(f"{HERE}/daily.csv")

# ---- S_auc: lab/decay_nextopen.py
anch = np.arange(L, len(A) - H, STRIDE); t = s.index[anch - 1]
r = A[anch + H, 3] / A[anch, 0] - 1
F = np.concatenate([np.nan_to_num(fs_tb7(np.stack([A[i - L:i] for i in anch[k:k + 20000]]).astype(np.float32))) for k in range(0, len(anch), 20000)])
mask = np.abs(r) > DB; y = (r > 0).astype(int); out = []
for m in pd.period_range(t[0].to_period("M") + 3, t[-1].to_period("M"), freq="M"):
    m0, m1 = m.start_time, (m + 1).start_time
    tr = (t >= m0 - pd.Timedelta(days=60)) & (t < m0 - pd.Timedelta(minutes=H)) & mask; te = (t >= m0) & (t < m1)
    if tr.sum() < 2000 or (te & mask).sum() < 500:
        continue
    sc = StandardScaler().fit(F[tr]); clf = LogisticRegression(C=0.1, max_iter=1000).fit(sc.transform(F[tr]), y[tr])
    out.append({"month": str(m), "auc": float(roc_auc_score(y[te & mask], clf.predict_proba(sc.transform(F[te & mask]))[:, 1])),
                "n_te": int((te & mask).sum())})
json.dump(out, open(f"{HERE}/series_auc.json", "w"), indent=1); print("auc series", len(out), f"{time.time() - t0:.0f}s", flush=True)

# ---- S_har: lab/decay_vol.py
lrr = np.concatenate([[0.0], np.diff(np.log(A[:, 3]))]); cs2 = np.concatenate([[0.0], np.cumsum(lrr ** 2)])
anch = np.arange(1440 + 1, len(A) - H, STRIDE); t = s.index[anch - 1]
rr = A[anch + H - 1, 3] / A[anch - 1, 3] - 1; yv = np.log(np.abs(rr) + 1e-5)
rv = lambda k: np.log((cs2[anch] - cs2[anch - k]) / k + 1e-12)
hr = t.hour.values
HAR = np.column_stack([rv(15), rv(60), rv(240), rv(1440), np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])
out = []
for m in pd.period_range(t[0].to_period("M") + 3, t[-1].to_period("M"), freq="M"):
    m0, m1 = m.start_time, (m + 1).start_time
    tr = (t >= m0 - pd.Timedelta(days=60)) & (t < m0 - pd.Timedelta(minutes=H)); te = (t >= m0) & (t < m1)
    if tr.sum() < 2000 or te.sum() < 500:
        continue
    sc = StandardScaler().fit(HAR[tr]); mdl = Ridge(alpha=1.0).fit(sc.transform(HAR[tr]), yv[tr])
    out.append({"month": str(m), "har": float(spearmanr(mdl.predict(sc.transform(HAR[te])), np.abs(rr[te]))[0]),
                "rv60": float(spearmanr(HAR[te, 1], np.abs(rr[te]))[0]), "n_te": int(te.sum())})
json.dump(out, open(f"{HERE}/series_har.json", "w"), indent=1); print("har series", len(out), f"{time.time() - t0:.0f}s", flush=True)

# ---- reproduction check against the stored (to 2025-09) files, months <= 2025-06
lines = []
for name, mine, stored, col in (("auc", "series_auc.json", "../lab/decay_nextopen.json", "auc"), ("har", "series_har.json", "../lab/decay_vol.json", "har")):
    a = pd.DataFrame(json.load(open(f"{HERE}/{mine}"))).set_index("month")[col]; b = pd.DataFrame(json.load(open(f"{HERE}/{stored}"))).set_index("month")[col]
    j = [m for m in a.index if m <= "2025-06" and m in b.index]; d = (a[j] - b[j]).abs()
    lines.append(f"{name}: months {len(a)} (to {a.index[-1]}), compared {len(j)} months <= 2025-06: max |diff| {d.max():.2e}, mean {d.mean():.2e}; "
                 f"months only mine {sorted(set(a.index) - set(b.index))[:3]}, only stored {sorted(m for m in set(b.index) - set(a.index) if m <= '2025-06')[:3]}")
open(f"{HERE}/series_check.txt", "w").write("\n".join(lines) + f"\nrun time {time.time() - t0:.0f}s\n"); print("\n".join(lines))

"""Has volatility predictability decayed too? (lead, 2026-10-11, after H51/H52: direction predictability fell from AUC ~0.60 to
~0.52.) Monthly walk-forward over 2017-04..2025-09, 1-minute bars, anchors every 10 bars, horizon 15 minutes, target log|r| (+1e-5):
(a) HAR-style ridge on log realised variance over the last 15, 60, 240, 1440 bars + hour-of-day sin/cos; (b) the same + tb7;
trained on the 60 days before each month, scored on the month by Spearman(prediction, |r|); also (c) trailing RV60 alone.
Descriptive; yearly means. usage: python decay_vol.py -> runs/tactical/lab/decay_vol.json"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/lab")
from lab import fs_tb7  # noqa: E402

t0 = time.time(); L, H, STRIDE = 60, 15, 10
s = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"])
s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index(); A = s.values.astype(np.float64)
lr = np.concatenate([[0.0], np.diff(np.log(A[:, 3]))]); cs2 = np.concatenate([[0.0], np.cumsum(lr ** 2)])
anch = np.arange(1440 + 1, len(A) - H, STRIDE); t = s.index[anch - 1]
r = A[anch + H - 1, 3] / A[anch - 1, 3] - 1; y = np.log(np.abs(r) + 1e-5)
rv = lambda k: np.log((cs2[anch] - cs2[anch - k]) / k + 1e-12)
hr = t.hour.values
HAR = np.column_stack([rv(15), rv(60), rv(240), rv(1440), np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])
TB = np.concatenate([np.nan_to_num(fs_tb7(np.stack([A[i - L:i] for i in anch[k:k + 20000]]).astype(np.float32)))
                     for k in range(0, len(anch), 20000)])
print(f"anchors {len(anch)}, {time.time() - t0:.0f}s", flush=True)
out = []
for m in pd.period_range(t[0].to_period("M") + 3, t[-1].to_period("M"), freq="M"):
    m0, m1 = m.start_time, (m + 1).start_time
    tr = (t >= m0 - pd.Timedelta(days=60)) & (t < m0 - pd.Timedelta(minutes=H)); te = (t >= m0) & (t < m1)
    if tr.sum() < 2000 or te.sum() < 500:
        continue
    row = {"month": str(m)}
    for name, X in (("har", HAR), ("har_tb7", np.column_stack([HAR, TB]))):
        sc = StandardScaler().fit(X[tr]); mdl = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        row[name] = float(spearmanr(mdl.predict(sc.transform(X[te])), np.abs(r[te]))[0])
    row["rv60"] = float(spearmanr(HAR[te, 1], np.abs(r[te]))[0]); out.append(row)
json.dump(out, open("runs/tactical/lab/decay_vol.json", "w"), indent=1)
df = pd.DataFrame(out); df["year"] = df["month"].str[:4]
print(df.groupby("year")[["rv60", "har", "har_tb7"]].mean().round(3).to_string())
print("corr(month, har)", round(np.corrcoef(np.arange(len(df)), df["har"])[0, 1], 3), f"{time.time() - t0:.0f}s")

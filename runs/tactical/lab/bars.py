"""Bar-size check (owner, 2026-10-09: the old project was "daily, not minute, and reached 70%+"; git and GitHub hold no such
version - the 66.7% of 2026-01-15 was 16 wins of 24 trades with a net loss - so the idea itself is tested here).
Resample the full 2017-2025 minute file to 1h / 4h / 1d bars, build the lab's tb7 features on 60-bar windows, and predict the
direction of the next 1 and 5 bars with a logistic regression, walk-forward: 8 folds, each trained on everything before it
(minus a 2*horizon gap) and tested on the next block. Reported per bar size and horizon: AUC and accuracy (mean over folds,
95% t-interval over folds), the up-rate of the test labels (the accuracy of always guessing the majority), and the number
of independent test samples (test bars / horizon).
usage: python bars.py"""
import math, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/lab")
from lab import fs_tb7, fs_returns  # noqa: E402

TQ = {8: 2.36}
src = pd.read_csv("D:/nt/neural_trade/Bitcoin_BTCUSDT.csv")
tcol = [c for c in src.columns if c.lower() in ("date", "timestamp", "time", "open_time", "datetime")][0]
src.index = pd.to_datetime(src[tcol]); src = src.sort_index()
cols = {c.lower(): c for c in src.columns}
src = src[[cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]]; src.columns = ["o", "h", "l", "c", "v"]
print("minute rows", len(src), src.index[0], src.index[-1], flush=True)

for rule in ("1h", "4h", "1d"):
    b = src.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    A = b.values.astype(np.float64); L = 60
    W = np.stack([A[i - L:i] for i in range(L, len(A))])               # window ends at bar i-1
    last = A[L - 1:len(A) - 1, 3]
    for name, fs in (("tb7", fs_tb7), ("returns", fs_returns)):
        F = np.nan_to_num(fs(W.astype(np.float32)))
        for hz in (1, 5):
            n = len(F) - hz; fwd = A[L - 1 + hz:len(A), 3][:n] / last[:n] - 1; X = F[:n]; y = (fwd > 0).astype(int)
            edges = np.linspace(n * 0.3, n, 9).astype(int); aucs, accs, ups, neff = [], [], [], []
            for k in range(8):
                tr_end, te0, te1 = edges[k] - 2 * hz, edges[k], edges[k + 1]
                sc = StandardScaler().fit(X[:tr_end]); m = LogisticRegression(C=0.1, max_iter=1000).fit(sc.transform(X[:tr_end]), y[:tr_end])
                p = m.predict_proba(sc.transform(X[te0:te1]))[:, 1]; yt = y[te0:te1]
                aucs.append(roc_auc_score(yt, p)); accs.append(np.mean((p > 0.5) == yt)); ups.append(yt.mean()); neff.append((te1 - te0) // hz)
            ci = lambda v: f"{np.mean(v):.3f} [{np.mean(v) - TQ[8] * np.std(v, ddof=1) / math.sqrt(8):.3f},{np.mean(v) + TQ[8] * np.std(v, ddof=1) / math.sqrt(8):.3f}]"
            print(f"{rule:3s} bars {len(A):6d} | {name:7s} next {hz} bar(s): AUC {ci(aucs)} acc {ci(accs)} | up-rate {np.mean(ups):.3f} "
                  f"| independent test samples per fold ~{int(np.mean(neff))}", flush=True)

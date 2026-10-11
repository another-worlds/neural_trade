"""Shared pieces of the decay study (nt-tactical-decay): bar caches, the monthly walk-forward, yearly summaries.
Protocol = decay.py (H51): for every calendar month M a model is trained on the `train_days` before M (labels must END before M starts)
and scored on M; tb7 features on 60-bar windows; target = sign of the H-bar return beyond a deadband (5 bps at a 15-minute horizon,
scaled by sqrt(horizon minutes / 15)); AUC per month; yearly means with 95% t-intervals over months. Descriptive only."""
import sys as _sys; _sys.path.insert(0, "D:/nt/nt_recut/runs/tactical/common")  # noqa: E402
import cutoff as C  # noqa: E402  (H66: no bar after C.CUTOFF is read)
import os, sys, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CSV = C.CSV
CACHE = "D:/nt/nt_recut_cache/decay"
L = 60
BAR_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
STRIDE = {"1m": 10, "5m": 2, "15m": 1, "1h": 1}
FEAT = {"ret1": 7, "bb": 3}                      # column index in fs_tb7


def deadband(H, bar):
    return 5e-4 * math.sqrt(H * BAR_MIN[bar] / 15.0)


def load_bars(bar):
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    if bar != "1m":
        s = s.resample(f"{BAR_MIN[bar]}min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return s


def build_cache(bar):
    """features (tb7), anchor times (open time of the window's last bar), closes, sigma (std of 1-bar log returns in the window)."""
    from lab import fs_tb7
    os.makedirs(CACHE, exist_ok=True)
    s = load_bars(bar); A = s.values.astype(np.float64); close = A[:, 3]
    anch = np.arange(L, len(A) - 1, STRIDE[bar])            # anchor i: window A[i-L:i], last bar i-1
    lr = np.diff(np.log(close), prepend=np.nan)
    F, sig = [], []
    for k in range(0, len(anch), 20000):
        a = anch[k:k + 20000]
        W = np.stack([A[i - L:i] for i in a]).astype(np.float32)
        F.append(np.nan_to_num(fs_tb7(W)))
        sig.append(np.array([np.std(lr[i - L + 1:i]) for i in a]))
    np.savez(f"{CACHE}/{bar}.npz", F=np.concatenate(F).astype(np.float32), sig=np.concatenate(sig), anch=anch,
             t=s.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64), close=close)
    return len(anch)


def load_cache(bar):
    z = np.load(f"{CACHE}/{bar}.npz")
    return z["F"], z["sig"], z["anch"], pd.DatetimeIndex(z["t"].astype("datetime64[ns]")), z["close"]


def split_masks(t, m0, m1, H_td, train_days):
    """train: anchors whose label has ended by the month start (t + H bars <= m0) and not older than train_days; test: anchors in [m0, m1)."""
    tr = (t >= m0 - pd.Timedelta(days=train_days)) & (t <= m0 - H_td)
    te = (t >= m0) & (t < m1)
    return np.asarray(tr), np.asarray(te)


def make_model(kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    if kind == "logreg":
        return LogisticRegression(C=0.1, max_iter=1000)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)


def walk_forward(bar, H, train_days=60, kind="logreg", k=0.0, first="2017-04", last=None, data=None):
    """returns a list of monthly dicts. k = 0: deadband mask; k > 0: keep only |r| > k * sigma_window * sqrt(H) (train and test)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    F, sig, anch, t, close = data if data is not None else load_cache(bar)
    n = len(close); ok = anch + H - 1 < n
    F, sig, anch, t = F[ok], sig[ok], anch[ok], t[ok]
    r = close[anch + H - 1] / close[anch - 1] - 1
    y = (r > 0).astype(int)
    base = np.abs(r) > deadband(H, bar)
    mask = base if k == 0 else np.abs(r) > k * sig * math.sqrt(H)
    H_td = pd.Timedelta(minutes=H * BAR_MIN[bar])
    out = []
    first_m = max(pd.Period(first, "M"), t[0].to_period("M") + 1)
    for m in pd.period_range(first_m, last or t[-1].to_period("M"), freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = split_masks(t, m0, m1, H_td, train_days)
        trm, tem = tr & mask, te & mask
        if trm.sum() < 500 or tem.sum() < 300 or len(np.unique(y[tem])) < 2:
            continue
        sc = StandardScaler().fit(F[trm]); mdl = make_model(kind).fit(sc.transform(F[trm]), y[trm])
        p = mdl.predict_proba(sc.transform(F[tem]))[:, 1]
        row = {"month": str(m), "auc": float(roc_auc_score(y[tem], p)), "n_train": int(trm.sum()), "n_test": int(tem.sum()),
               "up_rate": float(y[tem].mean()), "keep_frac": float(tem.sum() / max(te.sum(), 1))}
        if k > 0:    # the model trained with the deadband target, scored on the same large-move bars
            trb = tr & base
            if trb.sum() >= 500:
                sb = StandardScaler().fit(F[trb]); mb = make_model(kind).fit(sb.transform(F[trb]), y[trb])
                row["auc_deadband_model"] = float(roc_auc_score(y[tem], mb.predict_proba(sb.transform(F[tem]))[:, 1]))
        if kind == "logreg":
            row["coef_ret1"] = float(mdl.coef_[0][FEAT["ret1"]]); row["coef_bb"] = float(mdl.coef_[0][FEAT["bb"]])
        for nm, j in FEAT.items():   # univariate AUC of the raw feature on the month: >0.5 momentum, <0.5 reversal
            row[f"uni_auc_{nm}"] = float(roc_auc_score(y[tem], F[tem][:, j]))
        out.append(row)
    return out


def tint(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float); n = len(v)
    if n < 2:
        return {"mean": float(v.mean()) if n else None, "lo": None, "hi": None, "n": n}
    from scipy.stats import t as T
    h = T.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return {"mean": float(v.mean()), "lo": float(v.mean() - h), "hi": float(v.mean() + h), "n": n}


def by_year(rows, key="auc", months=None):
    d = {}
    for r in rows:
        if months and not (months[0] <= r["month"] <= months[1]):
            continue
        d.setdefault(r["month"][:4], []).append(r.get(key))
    return {y: tint(v) for y, v in sorted(d.items())}

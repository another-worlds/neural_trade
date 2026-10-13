"""T01 (SPEC.md): forward test of the next-hour volatility-change forecast (H72, gaps study 3). Library: guarded loaders, features and
targets exactly as gaps/study3_vov.py, the frozen fit and the monthly scoring. CPU only."""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TAC, "volnet"))
import volnet_lib as V  # noqa: E402  (Data: features at anchor a use bars < a; month_masks)

TRAIN_CSV = "D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv"
FWD_CSV = "D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv"
CUTOFF = pd.Timestamp("2025-07-20 23:59")
FWD_START = pd.Timestamp("2025-10-01 00:00")
FORBID_LO, FORBID_HI = pd.Timestamp("2025-07-21 00:00"), pd.Timestamp("2025-09-30 23:59")
FWD_LOAD_END = pd.Timestamp("2026-10-02 00:00")        # one day past the last scored month, for the 4-hour labels
TRAIN_LOAD_FROM = pd.Timestamp("2024-06-01 00:00")     # 365 training days + the 1,440-bar feature history
STRIDE = 7; WARM = 1440 + 180; TRAIN_DAYS = 365
SETS = ["S1", "S2", "S3"]; TARGETS = {"A": 60, "B": 240}
MODELS_A = ["rw", "ar", "arlvl"] + SETS; MODELS_B = ["rw", "ar"] + SETS
FWD_MONTHS = [str(m) for m in pd.period_range("2025-10", "2026-09", freq="M")]
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_bars(path, lo, hi, chunk=500_000):
    """OHLCV indexed by time with bars lo <= t <= hi only (chunked read; stops at the first chunk past hi). Asserts the bounds and that
    no bar of the forbidden gap 2025-07-21..2025-09-30 is returned."""
    parts = []
    for ch in pd.read_csv(path, usecols=COLS, chunksize=chunk):
        ts = pd.to_datetime(ch["timestamp"]); keep = ((ts >= lo) & (ts <= hi)).to_numpy()
        if keep.any():
            c = ch[keep].copy(); c.index = ts[keep]; parts.append(c.drop(columns="timestamp"))
        if (ts > hi).any():
            break
    s = pd.concat(parts).sort_index()
    assert s.index.min() >= lo and s.index.max() <= hi
    assert not ((s.index >= FORBID_LO) & (s.index <= FORBID_HI)).any(), "a bar of 2025-07-21..2025-09-30 was read"
    return s


def load_train(path=TRAIN_CSV):
    return load_bars(path, TRAIN_LOAD_FROM, CUTOFF)


def load_fwd(path=FWD_CSV):
    return load_bars(path, FWD_START, FWD_LOAD_END)


def build(s, first):
    """Data (stride 7), feature sets, targets and baselines' regressors; identical definitions to gaps/study3_vov.build."""
    D = V.Data(s.values.astype(np.float64), s.index, stride=STRIDE, first=first, with_tb7=False); a = D.anch; cs = D.cs2
    rvw = lambda lo, hi: cs[a + hi] - cs[a + lo]
    prev = [rvw(-60 * (k + 1), -60 * k) for k in range(4)]; nxt = [rvw(60 * j, 60 * (j + 1)) for j in range(4)]
    ln = lambda x: np.log(x + 1e-12)
    L = np.column_stack([ln(p) for p in prev]); N = np.column_stack([ln(x) for x in nxt])
    vp = np.log(L.std(1, ddof=1) + 0.01); vB = np.log(N.std(1, ddof=1) + 0.01); dA = N[:, 0] - L[:, 0]
    okx = np.all([p > 0 for p in prev], 0) & np.all([x > 0 for x in nxt], 0) & np.all(D.har[:, :4] > -20, 1)
    S1 = D.har; S2 = np.column_stack([S1, D.extra]); S3 = np.column_stack([S2, L[:, 1:4], vp])
    return {"D": D, "F": {"S1": S1, "S2": S2, "S3": S3}, "T": {"A": dA, "B": vB}, "AR": {"A": L[:, 0] - L[:, 1], "B": vp},
            "LV0": L[:, 0], "ok": okx}


def train_idx(B, tg, m0, days=TRAIN_DAYS, start=None):
    """rows available before m0 whose label ends at or before m0 (and not older than `days`, or than `start`)."""
    ta = B["D"].ta; H = TARGETS[tg]
    lo = (m0 - pd.Timedelta(days=days)) if start is None else start
    m = np.asarray((ta >= lo) & (ta + pd.Timedelta(minutes=H) <= m0))
    return np.where(m & B["ok"])[0]


def ols1(x, y):
    X = np.column_stack([np.ones(len(x)), x]); return np.linalg.lstsq(X, y, rcond=None)[0]


def fit(B, tr):
    """All models for both targets on rows tr[tg]. Returns a plain dict of arrays (the frozen arm keeps it unchanged)."""
    out = {}
    for tg in TARGETS:
        i = tr[tg]; y = B["T"][tg][i]; m = {"ar": ols1(B["AR"][tg][i], y)}
        if tg == "A":
            m["arlvl"] = ols1(B["LV0"][i], y)
        for nm in SETS:
            X = B["F"][nm][i]; mu, sd = X.mean(0), X.std(0) + 1e-12
            r = Ridge(alpha=1.0).fit((X - mu) / sd, y); m[nm] = (mu, sd, r.coef_.copy(), float(r.intercept_))
        out[tg] = m
    return out


def predict(models, B, tg, idx):
    m = models[tg]; p = {"rw": np.zeros(len(idx)) if tg == "A" else B["AR"][tg][idx]}
    b = m["ar"]; p["ar"] = b[0] + b[1] * B["AR"][tg][idx]
    if tg == "A":
        b2 = m["arlvl"]; p["arlvl"] = b2[0] + b2[1] * B["LV0"][idx]
    for nm in SETS:
        mu, sd, coef, ic = m[nm]; p[nm] = ((B["F"][nm][idx] - mu) / sd) @ coef + ic
    return p


def score_month(models, B, month, label_arm):
    """Rows for one scored month: every model's MSE, Spearman, R2 vs rw, skill vs ar (as gaps/study3_vov), plus row counts."""
    m0 = month.start_time; m1 = (month + 1).start_time; ta = B["D"].ta
    te = np.asarray((ta >= m0) & (ta < m1)); all_idx = np.where(te)[0]; idx = np.where(te & B["ok"])[0]; rows = []
    for tg in TARGETS:
        yt = B["T"][tg][idx]; preds = predict(models, B, tg, idx); mse = {k: float(np.mean((yt - p) ** 2)) for k, p in preds.items()}
        for k, p in preds.items():
            sp = float(spearmanr(p, yt)[0]) if np.std(p) > 0 else float("nan")
            rows.append({"arm": label_arm, "month": str(month), "target": tg, "model": k, "mse": mse[k], "spearman": sp,
                         "r2_rw": 1 - mse[k] / mse["rw"], "skill_ar": 1 - mse[k] / mse["ar"], "n_te": int(len(idx)),
                         "n_anchors": int(len(all_idx)), "n_dropped": int(len(all_idx) - len(idx))})
    return rows


def ci(x):
    x = np.asarray(x, float); n = len(x); mu, se = x.mean(), x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return float(mu), float(mu - h), float(mu + h), float(mu / se if se > 0 else 0.0)

"""T04 (SPEC.md): the 15-minute calendar for the next-hour volatility change. Reuses T01's guarded loaders, S3 features, ar, rw and the
drop filter; adds the 96 slot dummies, the S3C / C / HAR-levels models, the frozen artefact and the monthly scoring. CPU only."""
import hashlib, json, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TAC, "tasks", "T01_vol_forward"))
import t01_lib as T  # noqa: E402

A_LO, A_HI = pd.Timestamp("2017-08-17 00:00"), pd.Timestamp("2018-12-31 23:59")
A_MONTHS = [str(m) for m in pd.period_range("2017-09", "2018-12", freq="M")]
B_LO, B_HI = pd.Timestamp("2025-10-01 00:00"), pd.Timestamp("2026-09-30 23:59")
B_TRAIN_FROM = pd.Timestamp("2025-10-02 00:00"); B_M0 = pd.Timestamp("2026-10-01 00:00")
MODELS = ["rw", "ar", "harlvl", "S3", "S3C", "C"]; RIDGE = ["harlvl", "S3", "S3C", "C"]
N_SLOT = 96; BASE = 1441


def slots(ta):
    """15-minute slot 0..95 of the anchor's UTC time of day, from the anchor timestamp (the time its last bar closes) only."""
    ta = pd.DatetimeIndex(ta); return (ta.hour.values * 60 + ta.minute.values) // 15


def build(s, first=BASE):
    B = T.build(s, first); D = B["D"]; B["slot"] = slots(D.ta)
    B["dum"] = np.eye(N_SLOT)[B["slot"]]
    return B


def drop_reasons(B):
    """Why T01's filter drops anchors: a zero-RV window among the 4 hours before / after the anchor, or a feature RV at the floor (< -20)."""
    D = B["D"]; a = D.anch; cs = D.cs2; rvw = lambda lo, hi: cs[a + hi] - cs[a + lo]
    prev0 = np.any([rvw(-60 * (k + 1), -60 * k) <= 0 for k in range(4)], 0); next0 = np.any([rvw(60 * j, 60 * (j + 1)) <= 0 for j in range(4)], 0)
    feat = np.any(D.har[:, :4] <= -20, 1)
    return {"anchors": int(len(a)), "kept": int(B["ok"].sum()), "dropped": int((~B["ok"]).sum()), "zero_rv_prev_windows": int(prev0.sum()),
            "zero_rv_next_windows": int(next0.sum()), "feature_rv_floor": int(feat.sum()),
            "feature_floor_only": int((feat & ~prev0 & ~next0).sum())}


def mats(B):
    """Feature matrix per ridge model (rows = all anchors). harlvl = the four log-RV levels (15/60/240/1440), no hour harmonics."""
    S3 = B["F"]["S3"]; dum = B["dum"]; ar = B["AR"]["A"][:, None]
    return {"harlvl": B["D"].har[:, :4], "S3": S3, "S3C": np.hstack([S3, dum]), "C": np.hstack([dum, ar])}


def _ols1(x, y):
    X = np.column_stack([np.ones(len(x)), x]); return np.linalg.lstsq(X, y, rcond=None)[0]


def fit(B, tr):
    y = B["T"]["A"][tr]; M = mats(B); out = {"ar": [float(v) for v in _ols1(B["AR"]["A"][tr], y)]}
    for nm in RIDGE:
        X = M[nm][tr]; mu, sd = X.mean(0), X.std(0) + 1e-12
        r = Ridge(alpha=1.0).fit((X - mu) / sd, y)
        out[nm] = {"mu": mu.tolist(), "sd": sd.tolist(), "coef": r.coef_.tolist(), "intercept": float(r.intercept_)}
    return out


def predict(models, B, idx):
    M = mats(B); p = {"rw": np.zeros(len(idx))}; b = models["ar"]; p["ar"] = b[0] + b[1] * B["AR"]["A"][idx]
    for nm in RIDGE:
        m = models[nm]; p[nm] = ((M[nm][idx] - np.array(m["mu"])) / np.array(m["sd"])) @ np.array(m["coef"]) + m["intercept"]
    return p


def score_month(models, B, month):
    ta = B["D"].ta; m0, m1 = month.start_time, (month + 1).start_time
    te = np.asarray((ta >= m0) & (ta < m1)); idx = np.where(te & B["ok"])[0]; yt = B["T"]["A"][idx]
    p = predict(models, B, idx); mse = {k: float(np.mean((yt - v) ** 2)) for k, v in p.items()}
    return {"month": str(month), "n_te": int(len(idx)), "n_anchors": int(te.sum()), "mse": mse, "skill_ar": {k: 1 - mse[k] / mse["ar"] for k in mse}}


def stats_for(rows, a, b, label):
    """Paired monthly differences of skill vs ar (a minus b), and the mse-ratio skill 1 - mse_a / mse_b."""
    d = np.array([r["skill_ar"][a] - r["skill_ar"][b] for r in rows]); r2 = np.array([1 - r["mse"][a] / r["mse"][b] for r in rows])
    mu, lo, hi, _ = T.ci(d); mu2, lo2, hi2, _ = T.ci(r2)
    return {"label": label, "n_months": len(d), "mean_diff_skill_ar": mu, "ci": [lo, hi], "months_positive": int((d > 0).sum()),
            "mse_ratio_skill": [mu2, lo2, hi2], "months_positive_ratio": int((r2 > 0).sum())}


def verdict(rows, min_mean=0.02):
    """Criterion A / B: mean paired difference of skill vs ar (S3C - S3) >= 0.02, 95% t lower bound > 0, positive in >= 11 of 16 (A) or 80% of months."""
    n = len(rows); s = stats_for(rows, "S3C", "S3", "S3C-S3")
    need = 11 if n == 16 else int(np.ceil(0.8 * n))
    c1, c2, c3 = s["mean_diff_skill_ar"] >= min_mean, s["ci"][0] > 0, s["months_positive"] >= need
    return {**s, "need_positive": need, "c1_mean_ge_0.02": bool(c1), "c2_lo_gt_0": bool(c2), "c3_months": bool(c3), "pass": bool(c1 and c2 and c3)}


def sha_of(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_freeze_bars(path):
    """Binance bars B_LO..B_HI only. Reads exactly the rows up to B_HI (nrows) so no later row is parsed when the file has no gaps."""
    n = int((B_HI - B_LO).total_seconds() // 60) + 1
    df = pd.read_csv(path, usecols=T.COLS, nrows=n); ts = pd.to_datetime(df["timestamp"]); df.index = ts; df = df.drop(columns="timestamp")
    assert ts.iloc[0] == B_LO
    if ts.iloc[-1] < B_HI:                                      # the file has gaps: fall back to the guarded chunked loader
        return T.load_bars(path, B_LO, B_HI, chunk=100_000)
    df = df[(df.index >= B_LO) & (df.index <= B_HI)]
    assert df.index.min() >= B_LO and df.index.max() <= B_HI
    assert not ((df.index >= T.FORBID_LO) & (df.index <= T.FORBID_HI)).any()
    return df.sort_index()


def freeze(bars):
    """Fit S3, S3C, C, harlvl, ar, rw ONCE on rows with ta >= B_TRAIN_FROM and label end <= B_M0; returns the artefact dict with its sha256."""
    assert bars.index.max() <= B_HI
    B = build(bars, T.WARM); tr = T.train_idx(B, "A", B_M0, start=B_TRAIN_FROM)
    models = fit(B, tr); ta = B["D"].ta[tr]
    art = {"spec": "T04 SPEC.md Part B freeze", "train_from": str(B_TRAIN_FROM), "train_label_end_max": str(ta.max() + pd.Timedelta(minutes=60)),
           "last_bar": str(bars.index.max()), "n_bars": int(len(bars)), "n_train_rows": int(len(tr)), "stride": T.STRIDE, "warm": T.WARM,
           "ridge_alpha": 1.0, "n_slots": N_SLOT, "drops": drop_reasons(B), "models": models}
    art["sha256"] = sha_of(art); return art

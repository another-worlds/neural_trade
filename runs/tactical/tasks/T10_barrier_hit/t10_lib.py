"""T10 (SPEC.md): probability that |price - close_t| reaches X bps within T minutes (an unsigned barrier hit). CPU only.
Reuses T03's HAR (fit_har / sigma / train_rows / cal_rows / month_rows / holm), T01's S3 feature definitions (re-evaluated at arbitrary anchors,
equivalence tested in test_t10.py) and T09's utc_hour and seam mask. Data only through T09's guard (never 2025-07-21..2025-09-30)."""
import os
import sys
import types

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); TASKS = os.path.dirname(HERE)
for p in (os.path.join(TASKS, "T03_vol_intervals"), os.path.join(TASKS, "T09_regression_anatomy")):
    sys.path.insert(0, p)
import t03_lib as T3  # noqa: E402
import guard  # noqa: E402
import lib as L9  # noqa: E402

XS = (25, 50, 100)                 # barrier distance, bps
TS = (15, 60, 240)                 # horizon, minutes
PRIMARY = (50, 60)                 # (X, T)
MODELS = ("B0", "B1", "B2", "H1", "H2", "H3")
CANDIDATES = ("H1", "H2", "H3")
TRAIN_DAYS = 365
LOOKBACK = 1440                    # feature history in bars (rv1440)
SEAM_MARGIN = 1500                 # minutes around a seam / splice in which an anchor is dropped (> the 1,440-bar history and the 240-minute label)
EW_LAMBDA = 0.94
SQRT_PI2 = np.sqrt(np.pi / 2)      # E|Z| = sqrt(2/pi) sd: T03's HAR level is a mean absolute return, the barrier formula needs the sd
SEL_FIRST, SEL_LAST = pd.Period("2019-01", "M"), pd.Period("2024-06", "M")
LOOK1_FIRST, LOOK1_LAST = pd.Period("2024-07", "M"), pd.Period("2025-07", "M")
LOOK2_FIRST, LOOK2_LAST = pd.Period("2025-11", "M"), pd.Period("2026-09", "M")
CUTOFF = guard.CUT_END
BOOT = 10000


# ------------------------------------------------------------------ Brownian two-sided first passage
def first_passage(x, sigma, T):
    """P(max_{s<=T} |W_s| >= x) for driftless Brownian motion W with variance sigma^2 per minute (x, sigma arrays or scalars, T minutes).
    Staying probability, with tau = sigma^2 T / x^2:
      Fourier series   (4/pi) sum_{k>=0} (-1)^k / (2k+1) exp(-(2k+1)^2 pi^2 tau / 8)           (fast for large tau)
      reflection series  sum_{k in Z} (-1)^k [Phi((2k+1)/sqrt(tau)) - Phi((2k-1)/sqrt(tau))]   (fast for small tau)
    The Fourier series is used for tau >= 0.2, the reflection series below."""
    x = np.asarray(x, float); sg = np.asarray(sigma, float); x, sg = np.broadcast_arrays(x, sg)
    tau = np.where(sg > 0, (sg ** 2) * T / np.where(x > 0, x, 1.0) ** 2, 0.0)
    out = np.zeros(tau.shape)
    big = tau >= 0.2; sml = (tau > 0) & ~big
    if big.any():
        t = tau[big]; s = np.zeros(t.shape)
        for k in range(40):
            s += (-1) ** k / (2 * k + 1) * np.exp(-((2 * k + 1) ** 2) * np.pi ** 2 * t / 8)
        out[big] = 1 - 4 / np.pi * s
    if sml.any():
        r = np.sqrt(tau[sml]); s = np.zeros(r.shape)
        for k in range(-12, 13):
            s += (-1) ** k * (norm.cdf((2 * k + 1) / r) - norm.cdf((2 * k - 1) / r))
        out[sml] = 1 - s
    return np.clip(out, 0.0, 1.0)


def first_passage_fourier(x, sigma, T):
    x = np.asarray(x, float); sg = np.asarray(sigma, float); tau = (sg ** 2) * T / x ** 2; s = np.zeros(np.shape(tau))
    for k in range(400):
        s = s + (-1) ** k / (2 * k + 1) * np.exp(-((2 * k + 1) ** 2) * np.pi ** 2 * tau / 8)
    return 1 - 4 / np.pi * s


def first_passage_reflection(x, sigma, T):
    x = np.asarray(x, float); sg = np.asarray(sigma, float); r = np.sqrt((sg ** 2) * T / x ** 2); s = np.zeros(np.shape(r))
    for k in range(-30, 31):
        s = s + (-1) ** k * (norm.cdf((2 * k + 1) / r) - norm.cdf((2 * k - 1) / r))
    return 1 - s


# ------------------------------------------------------------------ prepared series
def prep(s, which):
    """Per-bar arrays of a bar file (DataFrame open, high, low, close, volume, indexed by bar open time in FILE time).
    which = 'cut' (file time is UTC+4/+5 before the 2024-09-13 splice) or 'fwd' (UTC). Everything per bar is the bar's own value; the anchor
    at array position a uses bars < a only."""
    idx = s.index; A = s[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    o, h, l, c, v = A.T.copy()
    t64 = idx.values.astype("datetime64[ns]")
    hr = L9.utc_hour(t64, which)
    off = (np.asarray(idx.hour) - hr) % 24
    utc = idx - pd.to_timedelta(off, unit="h")
    lr = np.concatenate([[0.0], np.diff(np.log(c))])
    P = types.SimpleNamespace(n=len(c), idx=idx, utc=utc, hr=hr, minute=np.asarray(idx.minute), o=o, h=h, l=l, c=c, v=v, lr=lr,
                              cs2=np.concatenate([[0.0], np.cumsum(lr ** 2)]), cv=np.concatenate([[0.0], np.cumsum(v)]),
                              cr=np.concatenate([[0.0], np.cumsum((h - l) / c)]),
                              tmin=t64.astype("datetime64[m]").astype(np.int64), which=which)
    P.ew = lfilter([1 - EW_LAMBDA], [1, -EW_LAMBDA], lr ** 2)          # EWMA variance per minute; value at bar i uses bars <= i
    P.seam = L9.seam_mask(t64, which, margin_min=SEAM_MARGIN)
    return P


def features(P, a):
    """T01's S3 definitions (study3_vov.build / volnet Data) at anchor positions a (bars < a only), with the time-of-day terms in UTC.
    Returns har (rv15, rv60, rv240, rv1440 as logs, sin, cos), extra10 (the 10 columns S3 adds to S1) and the validity flag."""
    cs = P.cs2; hr = P.hr[a - 1]
    rv = lambda k: np.log(np.maximum(cs[a] - cs[a - k], 0) / k + 1e-12)
    har = np.column_stack([rv(15), rv(60), rv(240), rv(1440), np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])
    cv, cr = P.cv, P.cr
    lvm = lambda k: np.log(np.maximum(cv[a] - cv[a - k], 0) / k + 1e-8)
    lrg = lambda k: np.log(np.maximum(cr[a] - cr[a - k], 0) / k + 1e-8)
    extra = np.column_stack([lvm(15) - lvm(1440), lvm(60) - lvm(1440), lvm(240) - lvm(1440), lrg(15), lrg(60), rv(5)])
    prev = [cs[a - 60 * k] - cs[a - 60 * (k + 1)] for k in range(4)]
    Lm = np.column_stack([np.log(p + 1e-12) for p in prev]); vp = np.log(Lm.std(1, ddof=1) + 0.01)
    ok = np.all([p > 0 for p in prev], 0) & np.all(har[:, :4] > -20, 1)
    return har, np.column_stack([extra, Lm[:, 1:4], vp]), ok


def hit_labels(h, l, c, a, T):
    """max over the T bars a .. a+T-1 of |ln high - ln c[a-1]| and |ln low - ln c[a-1]|, from the 1-minute high/low."""
    w = a[:, None] + np.arange(T)
    c0 = c[a - 1]
    return np.maximum(np.log(h[w].max(1) / c0), np.log(c0 / l[w].min(1)))


def build(P, T, first=LOOKBACK + 1):
    """Anchors for horizon T: close time (UTC) on the T-minute grid from the hour start, whole history contiguous, no seam, valid features.
    Returns the dict B used by the walk-forward (B['D'] carries har and ta like T03's duck-typed data)."""
    n = P.n
    a = np.arange(first, n - T + 1)
    mod = (P.hr[a - 1] * 60 + P.minute[a - 1] + 1) % 1440
    a = a[(mod % T == 0)]
    contig = (P.tmin[a + T - 1] - P.tmin[a - LOOKBACK - 1]) == (T + LOOKBACK)
    a = a[contig & ~P.seam[a - 1]]
    har, ex10, ok = features(P, a)
    trail = np.sqrt(np.maximum(P.cs2[a] - P.cs2[a - 4 * T], 0.0) / 4.0)              # sd of a T-bar return from 4T bars of RV (T03's trail)
    ew = np.sqrt(np.maximum(P.ew[a - 1], 0.0))                                       # EWMA sd per minute
    ok = ok & (trail > 0) & (ew > 0)
    m = hit_labels(P.h, P.l, P.c, a, T)
    ta = (P.utc[a - 1] + pd.Timedelta(minutes=1))
    hrs = P.hr[a - 1].astype(float)
    harm = np.column_stack([f(2 * np.pi * k * hrs / 24) for k in (1, 2) for f in (np.sin, np.cos)])
    D = types.SimpleNamespace(har=har, ta=ta)
    return {"D": D, "T": T, "a": a, "ok": ok, "trail": {T: trail}, "ew": ew, "ex10": ex10, "harm": harm,
            "r": {T: np.log(P.c[a + T - 1]) - np.log(P.c[a - 1])}, "move": m, "y": {X: (m >= X * 1e-4) for X in XS},
            "close0": P.c[a - 1], "tF": P.idx[a - 1]}


def subset_before(B, t_end):
    """B restricted to anchors with ta < t_end (used by the selection run so that it cannot see a later row)."""
    keep = np.asarray(B["D"].ta < t_end)
    return take(B, keep)


def take(B, keep):
    T = B["T"]
    return {"D": types.SimpleNamespace(har=B["D"].har[keep], ta=B["D"].ta[keep]), "T": T, "a": B["a"][keep], "ok": B["ok"][keep],
            "trail": {T: B["trail"][T][keep]}, "ew": B["ew"][keep], "ex10": B["ex10"][keep], "harm": B["harm"][keep], "r": {T: B["r"][T][keep]},
            "move": B["move"][keep], "y": {X: v[keep] for X, v in B["y"].items()}, "close0": B["close0"][keep], "tF": B["tF"][keep]}


# ------------------------------------------------------------------ models
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fp_from_total_sd(sd_T, X, T):
    """first-passage probability from the sd of the T-minute return (per-minute sd = sd_T / sqrt(T))."""
    return first_passage(X * 1e-4, sd_T / np.sqrt(T), T)


def har_sd(model, B, T, idx):
    return T3.sigma(model, _bt(B, T), idx, None) * SQRT_PI2


def _bt(B, T):
    return B   # T03 helpers read B['D'].har, B['D'].ta, B['r'][H], B['trail'][H], B['ok']


def h1_prob(B, T, idx, har):
    sd = har_sd(har, B, T, idx)
    return {X: fp_from_total_sd(sd, X, T) for X in XS}, sd


def h3_design(B, idx, p1):
    return np.column_stack([logit(p1), B["ex10"][idx], B["harm"][idx]])


def fit_models(B, T, m0, days=TRAIN_DAYS):
    """HAR (T03's, plain) and the three H3 logistic regressions, fitted on the `days` days before m0 whose labels end by m0."""
    from sklearn.linear_model import LogisticRegression
    rows = T3.train_rows(B, T, m0, days)
    har = T3.fit_har(B, T, rows, None, False)
    p1, _ = h1_prob(B, T, rows, har)
    models = {"har": har, "n_train": int(len(rows)), "last_label_end": str(B["D"].ta[rows].max() + pd.Timedelta(minutes=T)) if len(rows) else None, "h3": {}}
    for X in XS:
        Z = h3_design(B, rows, p1[X]); y = B["y"][X][rows]
        mu, sd = Z.mean(0), Z.std(0) + 1e-12
        if y.min() == y.max():
            models["h3"][X] = None; continue
        lr = LogisticRegression(C=1.0, max_iter=2000).fit((Z - mu) / sd, y)
        models["h3"][X] = (mu, sd, lr.coef_[0].copy(), float(lr.intercept_[0]))
    return models


def predict_rows(models, B, T, idx, cidx):
    """All six models' probabilities on rows idx; H2's isotonic map and B0's base rate come from rows cidx (the previous calendar month).
    Returns {model: {X: p}}."""
    from sklearn.isotonic import IsotonicRegression
    har = models["har"]
    p1, _ = h1_prob(B, T, idx, har); c1, _ = h1_prob(B, T, cidx, har)
    out = {m: {} for m in MODELS}
    for X in XS:
        yc = B["y"][X][cidx].astype(float)
        out["B0"][X] = np.full(len(idx), yc.mean())
        out["B1"][X] = fp_from_total_sd(B["trail"][T][idx], X, T)
        out["B2"][X] = first_passage(X * 1e-4, B["ew"][idx], T)
        out["H1"][X] = p1[X]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(c1[X], yc)
        out["H2"][X] = iso.predict(p1[X])
        h3 = models["h3"][X]
        if h3 is None:
            out["H3"][X] = p1[X]
        else:
            mu, sd, coef, ic = h3
            z = ((h3_design(B, idx, p1[X]) - mu) / sd) @ coef + ic
            out["H3"][X] = 1 / (1 + np.exp(-z))
    return out


def walk(B, T, months, frozen=None, days=TRAIN_DAYS):
    """Monthly walk-forward over `months` (Periods). Models fitted at each month start (T03's rule), or `frozen` (one fit) for all months.
    Test rows = valid anchors of the month; calibration rows = valid anchors of the previous month whose labels end before the month starts.
    Returns the concatenated rows."""
    keys = {"month": [], "day": [], "ta": []}; Y = {X: [] for X in XS}; Pm = {m: {X: [] for X in XS} for m in MODELS}; info = []
    for month in months:
        m0 = month.start_time
        idx, n_all = T3.month_rows(B, T, month); cidx = T3.cal_rows(B, T, month)
        if len(idx) == 0 or len(cidx) == 0:
            info.append({"month": str(month), "skipped": True, "n": int(len(idx)), "n_cal": int(len(cidx))}); continue
        models = frozen if frozen is not None else fit_models(B, T, m0, days)
        pr = predict_rows(models, B, T, idx, cidx)
        ta = B["D"].ta[idx]
        keys["month"] += [str(month)] * len(idx); keys["day"].append(np.asarray(ta.normalize().values.astype("datetime64[D]").astype(np.int64))); keys["ta"].append(ta.values)
        for X in XS:
            Y[X].append(B["y"][X][idx])
            for m in MODELS:
                Pm[m][X].append(pr[m][X])
        info.append({"month": str(month), "n": int(len(idx)), "n_dropped": int(n_all - len(idx)), "n_cal": int(len(cidx)),
                     "n_train": models["n_train"], "last_label_end": models["last_label_end"]})
    cat = np.concatenate
    return {"month": np.array(keys["month"]), "day": cat(keys["day"]), "ta": cat(keys["ta"]), "y": {X: cat(Y[X]) for X in XS},
            "p": {m: {X: cat(Pm[m][X]) for X in XS} for m in MODELS}, "info": info}


# ------------------------------------------------------------------ metrics
def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    return L9.auc(y, p)


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(e / len(p))


def reliability(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); rows = []
    for k in range(bins):
        m = b == k
        rows.append({"bin": k, "n": int(m.sum()), "p_mean": float(p[m].mean()) if m.any() else None, "y_mean": float(y[m].mean()) if m.any() else None})
    return rows


def day_boot_skill(se_m, se_b, day, idx):
    """Brier skill 1 - sum(se_m) / sum(se_b) for each bootstrap draw (rows of `idx` index the unique days)."""
    _, inv = np.unique(day, return_inverse=True)
    sm = np.bincount(inv, se_m); sb = np.bincount(inv, se_b)
    return 1 - sm[idx].sum(1) / sb[idx].sum(1)


def boot_index(day, n_draw=BOOT, seed=0):
    nd = len(np.unique(day)); return np.random.default_rng(seed).integers(0, nd, (n_draw, nd))


def summarize(R, idx_boot, models=MODELS, chosen=None):
    """Per cell (X, T-run result R) and model: pooled Brier, Brier skill over B1 with the day-block bootstrap CI and one-sided p, log loss,
    AUC, ECE (10 equal-width bins), base rate, n; and monthly Brier for every model."""
    out = {}
    for X in XS:
        y = R["y"][X].astype(float); se1 = (R["p"]["B1"][X] - y) ** 2; cell = {}
        for m in models:
            p = R["p"][m][X]; se = (p - y) ** 2
            d = {"n": int(len(y)), "base_rate": float(y.mean()), "brier": float(se.mean()), "logloss": logloss(p, y), "auc": auc(y, p), "ece": ece(p, y),
                 "mean_p": float(p.mean())}
            d["skill_vs_B1"] = float(1 - se.sum() / se1.sum())
            if m != "B1":
                bs = day_boot_skill(se, se1, R["day"], idx_boot)
                d["skill_ci"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
                d["skill_p"] = float((np.sum(bs <= 0) + 1) / (len(bs) + 1))
            mo = {}
            for mm in np.unique(R["month"]):
                k = R["month"] == mm
                mo[str(mm)] = {"brier": float(se[k].mean()), "brier_b1": float(se1[k].mean()), "n": int(k.sum())}
            d["monthly"] = mo
            d["mean_monthly_brier"] = float(np.mean([v["brier"] for v in mo.values()]))
            cell[m] = d
        out[X] = cell
    return out


# ------------------------------------------------------------------ the selection-first rule
def assert_selection_committed(path, repo_dir=None):
    """Raise unless `path` is tracked by git, has no uncommitted change and is part of HEAD."""
    import subprocess
    cwd = repo_dir or os.path.dirname(os.path.abspath(path)); f = os.path.abspath(path)
    run = lambda *a: subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if not os.path.exists(f):
        raise AssertionError(f"{f} does not exist")
    if run("ls-files", "--error-unmatch", f).returncode != 0:
        raise AssertionError(f"{f} is not tracked")
    if run("status", "--porcelain", "--", f).stdout.strip():
        raise AssertionError(f"{f} has uncommitted changes")
    if run("cat-file", "-e", "HEAD:" + os.path.relpath(f, run("rev-parse", "--show-toplevel").stdout.strip()).replace("\\", "/")).returncode != 0:
        raise AssertionError(f"{f} is not in HEAD")
    return run("rev-parse", "HEAD").stdout.strip()

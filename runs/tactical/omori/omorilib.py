"""Omori aftershocks of excursions beyond HAR (hypothesis #2) and the shock-state features reused by #6 (SPEC.md, SPEC_var.md).

Everything is a causal function of the OHLCV bars read through shapelib.load_bars (hard cutoff 2025-07-20 23:59).
Anchor i = a decision at the close of bar i-1. A feature at anchor i reads bars <= i-1 only; the target window is bars i..i+T-1.
Per scored month the HAR sigma-hat, the shock stream and every coefficient are fitted on the training rows only (365 days, windows
closed by the month start), then applied to the month."""
import os, sys, math
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TAC, "shape"))
import shapelib as S  # noqa: E402  (hard-cutoff loader, clock features, roll_trailing, split semantics)

CUTOFF = S.CUTOFF
STRIDE = 5; D_MAX = 4320; TRAIN_DAYS = 365; WARM = 1500
TS = (15, 30, 60)
RV_LAGS = (1, 5, 15, 60, 240, 1440)
HAR_LAGS = (15, 60, 240, 1440)
P_GRID = tuple(round(x, 2) for x in np.arange(0.4, 1.61, 0.1)); C_GRID = (1.0, 10.0)
TAUS = (5.0, 30.0, 180.0, 1440.0)
MCAP = 3.0                                   # cap of the shock magnitude |z|/k
# k per horizon: the multiple of the HAR forecast sd * sqrt(T) at which the 2% event rate is met on the 2018 training year of the
# first scored month (calibrate_k.py; values copied here and into SPEC.md before any result)
K_T = {15: 3.882, 30: 3.845, 60: 3.867}      # results/k.json


class Bars:
    """Global causal per-bar arrays (no row after the cutoff)."""
    def __init__(self, df=None):
        df = S.load_bars() if df is None else df
        assert df.index.max() <= CUTOFF
        self.df = df; A = df[["open", "high", "low", "close", "volume"]].to_numpy(np.float64); self.A = A; self.n = len(A)
        L = np.log(np.maximum(A[:, :4], 1e-9)); self.lo, self.lh, self.ll, self.lc = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
        r = np.concatenate([[0.0], np.diff(self.lc)]); self.r = r; r2 = r * r
        self.tmin = df.index.values.astype("datetime64[m]").astype(np.int64)
        self.tod = (self.tmin % 1440).astype(float)
        self.lv = {}                                              # log mean squared 1-bar return over the k bars ending at each bar
        for k in RV_LAGS:
            s = S.roll_trailing(r2, k) / k
            self.lv[k] = np.log(s + 1e-12)
        self.clock = S.clock_features(self.tmin)                  # clock of the bar's own open time (shifted to the anchor below)
        self.t_ns = df.index.values.astype("datetime64[ns]").astype(np.int64)


# ------------------------------------------------------------------ HAR sigma-hat and the shock stream
def har_design(B, anch, T):
    """HAR + time of day for a window of T bars starting at each anchor; lags read at bar anch-1."""
    b = anch - 1; mid = (B.tod[anch] + T / 2.0) % 1440.0
    cols = [np.ones(len(anch))] + [B.lv[k][b] for k in HAR_LAGS]
    for h in (1, 2):
        cols += [np.sin(2 * np.pi * h * mid / 1440.0), np.cos(2 * np.pi * h * mid / 1440.0)]
    return np.stack(cols, 1)


def future_logvar(B, anch, T):
    cs = S.cs0(B.r * B.r); return np.log((cs[anch + T] - cs[anch]) / T + 1e-12)


def fit_har(B, tr_anch, T):
    X = har_design(B, tr_anch, T); y = future_logvar(B, tr_anch, T)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]


def sigma_hat(B, coef, anch, T):
    """HAR forecast of the per-bar sd over the T bars starting at each anchor."""
    return np.exp(0.5 * (har_design(B, anch, T) @ coef))


def target_event(B, anch, T, k, sig):
    """E_i = 1 if, within bars i..i+T-1, the price leaves C_{i-1} +- k sig sqrt(T) (high/low reach). NaN where the window does not fit."""
    n = B.n; ok = anch + T - 1 < n; a = np.where(ok, anch, 1); c0 = B.lc[a - 1]; e = a + T - 1
    hmax = pd.Series(B.lh).rolling(T).max().to_numpy()[e]; lmin = pd.Series(B.ll).rolling(T).min().to_numpy()[e]
    exc = np.maximum(hmax - c0, c0 - lmin)
    ev = (exc > k * sig[np.where(ok, np.arange(len(anch)), 0)] * math.sqrt(T)).astype(float)
    return np.where(ok, ev, np.nan)


def shock_stream(B, coef, T, k, lo, hi):
    """Per-bar |z|/k for bars in [lo, hi): the window that ended at bar j (anchor a=j-T+1, C_{j-T} to C_j) leaves the HAR band
    (endpoint excursion |lnC_j - lnC_{j-T}| / (sig_hat(a) sqrt(T)) > k). m_j = min(|z|/k, MCAP) where it does, else 0; mh_j the same
    at k/2 (foreshock). sig_hat(a) reads bars <= a-1 = j-T, so bar j's shock is known at the close of bar j."""
    j = np.arange(lo, hi); a = j - T + 1; ok = a >= WARM
    sig = np.full(len(j), np.nan); sig[ok] = sigma_hat(B, coef, a[ok], T)
    z = np.full(len(j), np.nan); z[ok] = np.abs(B.lc[j[ok]] - B.lc[j[ok] - T]) / (sig[ok] * math.sqrt(T))
    with np.errstate(invalid="ignore"):
        m = np.where(z > k, np.minimum(z / k, MCAP), 0.0); mh = np.where(z > k / 2, 1.0, 0.0)
    return np.nan_to_num(m), np.nan_to_num(mh)


def power_kernel(m, p, c, D=D_MAX):
    """K[x] = sum_{d>=1} m[x-d] (d+c)^-p (d <= D): the intensity at the position x from bars before x."""
    w = np.zeros(D + 1); d = np.arange(1, D + 1); w[1:] = (d + c) ** (-p)
    return fftconvolve(m, w)[:len(m)]


def exp_kernel(m, tau):
    """y[x] = sum_{d>=1} m[x-d] exp(-d/tau)."""
    a = math.exp(-1.0 / tau); u = lfilter([1.0], [1.0, -a], m); y = np.zeros_like(u); y[1:] = a * u[:-1]; return y


def time_since_last(m, cap=1440):
    """bars since the last shock bar strictly before x (cap)."""
    n = len(m); idx = np.arange(n); ev = m > 0
    last = np.maximum.accumulate(np.where(ev, idx, -10 ** 9)); prev = np.empty(n); prev[0] = -10 ** 9; prev[1:] = last[:-1]
    return np.minimum(idx - prev, cap)


# ------------------------------------------------------------------ features of a month
def base_features(B, anch, sig, tsl):
    """Strengthened baseline (corr.): RV at 1/5/15/60/240/1440 bars, the clock, log sigma x clock, time since last excursion."""
    b = anch - 1
    rv = np.stack([B.lv[k][b] for k in RV_LAGS], 1)
    clk = B.clock[anch]                                                   # clock of the anchor's open time
    ls = np.log(sig)[:, None]
    tod = (B.tod[anch])
    harm = np.stack([f(2 * np.pi * h * tod / 1440.0) for h in (1, 2, 3) for f in (np.sin, np.cos)], 1)
    lt = np.log1p(tsl)[:, None]
    return np.concatenate([rv, clk, ls, ls * harm, lt, lt ** 2], 1)


def standardise(Xtr, X):
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    return np.clip((np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) - mu) / sd, -8, 8)


class MonthData:
    """All arrays of one (month, T): anchors, targets, shock stream, baseline features and the kernel family. Fitted on the training rows."""
    def __init__(self, B, month, T, k, train_days=TRAIN_DAYS, stride=STRIDE, kernels=True):
        per = pd.Period(month, "M"); m0, m1 = per.start_time, (per + 1).start_time
        t_lo = m0 - pd.Timedelta(days=train_days); self.month = str(per); self.T = T; self.k = k
        i_lo = int(np.searchsorted(B.t_ns, np.datetime64(t_lo, "ns").astype(np.int64))); i_m0 = int(np.searchsorted(B.t_ns, np.datetime64(m0, "ns").astype(np.int64)))
        i_m1 = int(np.searchsorted(B.t_ns, np.datetime64(m1, "ns").astype(np.int64)))
        i_m1 = min(i_m1, B.n)
        seg_lo = max(i_lo - D_MAX - 5, WARM + T + 5); seg_hi = min(i_m1 + T, B.n)
        self.seg_lo, self.seg_hi = seg_lo, seg_hi
        anch = np.arange(max(i_lo, WARM + T + 5), i_m1 - 0, stride)
        anch = anch[anch + T - 1 < B.n]
        t_a = B.t_ns[anch - 1]
        lab_end = pd.DatetimeIndex(t_a.astype("datetime64[ns]")) + pd.Timedelta(minutes=T + 1)
        tdx = pd.DatetimeIndex(t_a.astype("datetime64[ns]"))
        self.tr = np.asarray((tdx >= t_lo) & (lab_end <= m0)); self.te = np.asarray((tdx >= m0) & (tdx < m1))
        self.anch = anch
        if self.tr.sum() < 5000 or self.te.sum() < 500:
            self.ok = False; return
        self.ok = True
        coef = fit_har(B, anch[self.tr], T); self.har = coef
        self.sig = sigma_hat(B, coef, anch, T)
        self.y = target_event(B, anch, T, k, self.sig)
        m, mh = shock_stream(B, coef, T, k, seg_lo, seg_hi)
        self.m_seg, self.mh_seg = m, mh
        pos = anch - seg_lo
        tsl = time_since_last(m)[pos]; self.tsl = tsl
        self.Xb = base_features(B, anch, self.sig, tsl).astype(np.float64)
        # foreshock: number of k/2 excursions among the last 30 bars (the larger ones have their own kernel term)
        c = np.concatenate([[0.0], np.cumsum(mh)])
        self.fs = (c[pos] - c[np.maximum(pos - 30, 0)])[:, None]
        self.kern = {}
        if kernels:
            for p in P_GRID:
                for cc in C_GRID:
                    self.kern[(p, cc)] = power_kernel(m, p, cc)[pos]
            self.ex = np.stack([exp_kernel(m, tau)[pos] for tau in TAUS], 1)
        self.rate_tr = float(np.nanmean(self.y[self.tr]))


# ------------------------------------------------------------------ logistic fits
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def fit_logit(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    mdl = LogisticRegression(C=C, max_iter=400); mdl.fit(X, y); return mdl


def offset_gain(eta, y, x, iters=8):
    """max over (a, b) of the log-likelihood of y with logit = eta + a x + b, by Newton (a profile score for one added feature)."""
    a = 0.0; b = 0.0
    for _ in range(iters):
        p = sigmoid(eta + a * x + b); w = np.maximum(p * (1 - p), 1e-9); r = y - p
        g = np.array([np.sum(r * x), np.sum(r)]); H = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g); a += step[0]; b += step[1]
    p = np.clip(sigmoid(eta + a * x + b), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), a


def score_month(B, month, T, k, train_days=TRAIN_DAYS):
    """-> dict with the test-row log-loss / Brier of: const (training rate), B, C_pow (B + Omori kernel + foreshock), C_pow_nofs,
    C_exp (B + four exponentials + foreshock), the chosen (p, c) and coefficient."""
    d = MonthData(B, month, T, k, train_days=train_days)
    if not d.ok:
        return None
    tr, te = d.tr, d.te; yv = d.y
    vtr = tr & np.isfinite(yv); vte = te & np.isfinite(yv)
    ytr, yte = yv[vtr], yv[vte]
    if yte.sum() < 5 or ytr.sum() < 50:
        return None
    Xb_tr = standardise(d.Xb[vtr], d.Xb[vtr]); Xb_te = standardise(d.Xb[vtr], d.Xb[vte])
    mb = fit_logit(Xb_tr, ytr); eta_tr = mb.decision_function(Xb_tr); p_b = mb.predict_proba(Xb_te)[:, 1]
    # profile choice of (p, c) on the training rows: one added feature on top of the baseline logit
    best = None; prof = {}
    for key, K in d.kern.items():
        kx = K[vtr]; kx = (kx - kx.mean()) / (kx.std() + 1e-12)
        ll, a = offset_gain(eta_tr, ytr, kx); prof[key] = ll
        if a > 0 and (best is None or ll > best[0]):
            best = (ll, key)
    if best is None:
        best = (max(prof.values()), max(prof, key=prof.get))
    key = best[1]
    fs_tr, fs_te = d.fs[vtr], d.fs[vte]
    def cols(Ks, fs, tr_ref):
        return np.concatenate([Ks, fs], 1) if fs is not None else Ks
    Kp = d.kern[key][:, None]
    def fit_eval(Ftr, Fte):
        Xtr = np.concatenate([Xb_tr, standardise(Ftr, Ftr)], 1); Xte = np.concatenate([Xb_te, standardise(Ftr, Fte)], 1)
        mdl = fit_logit(Xtr, ytr); return mdl, mdl.predict_proba(Xte)[:, 1]
    m_pow, p_pow = fit_eval(np.concatenate([Kp[vtr], fs_tr], 1), np.concatenate([Kp[vte], fs_te], 1))
    m_nf, p_nf = fit_eval(Kp[vtr], Kp[vte])
    m_ex, p_ex = fit_eval(np.concatenate([d.ex[vtr], fs_tr], 1), np.concatenate([d.ex[vte], fs_te], 1))
    m_fs, p_fs = fit_eval(fs_tr, fs_te)
    base_rate = float(ytr.mean())
    out = {"month": d.month, "T": T, "n": int(vte.sum()), "events": int(yte.sum()), "n_eff": float(vte.sum() * STRIDE / T),
           "rate_tr": base_rate, "p": key[0], "c": key[1], "coef_kernel": float(m_pow.coef_[0][Xb_tr.shape[1]]),
           "prof_gain": float(best[0] - prof_base(eta_tr, ytr))}
    for name, pr in (("const", np.full(len(yte), base_rate)), ("B", p_b), ("pow", p_pow), ("pow_nofs", p_nf), ("exp", p_ex), ("fs", p_fs)):
        out[f"ll_{name}"] = logloss(yte, pr); out[f"bs_{name}"] = brier(yte, pr)
    return out


def prof_base(eta, y):
    p = np.clip(sigmoid(eta), 1e-12, 1 - 1e-12); return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibrate_k(B, first_month, T, target=0.02):
    """k with a `target` event rate on the training year of `first_month` (reads nothing at or after the month start)."""
    m0 = pd.Period(first_month, "M").start_time
    i_m0 = int(np.searchsorted(B.t_ns, np.datetime64(m0, "ns").astype(np.int64)))
    i_lo = int(np.searchsorted(B.t_ns, np.datetime64(m0 - pd.Timedelta(days=TRAIN_DAYS), "ns").astype(np.int64)))
    anch = np.arange(max(i_lo, WARM + T + 5), i_m0 - T - 2, STRIDE)
    coef = fit_har(B, anch, T); sig = sigma_hat(B, coef, anch, T); lo, hi = 0.5, 12.0
    for _ in range(40):
        k = 0.5 * (lo + hi); r = float(np.nanmean(target_event(B, anch, T, k, sig)))
        lo, hi = (k, hi) if r > target else (lo, k)
    return round(0.5 * (lo + hi), 3)

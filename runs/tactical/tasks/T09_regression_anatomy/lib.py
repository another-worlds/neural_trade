"""T09 library: the anchor dataset (features, labels at three entries, condition variables, spread proxies) and the monthly walk-forward.

Protocol = H51 / H59 (runs/tactical/lab/decay.py, decay_nextopen.py): anchors every 10 one-minute bars, a 60-bar window, the 13 `tb7` features
(lab.fs_tb7), target = sign of the 15-minute return beyond a 5-bps deadband, a logistic regression (C 0.1) per calendar month trained on the
60 days before the month and scored on the month. One deliberate difference: the training anchors must have their whole label (exit bar close
included) finished before the month starts (t <= m0 - (H + 2) minutes); H51 / H59 used t < m0 - H minutes, which lets the last training label
exit on the month's first bar. The sanity run quantifies the difference against the stored H51 / H59 values.

Entries (all exit at the close of a bar, hence the three labels differ only in the entry price):
  close  enter at the close of the window's last bar a-1, exit at the close of bar a-1+h   (H51)
  nopen  enter at the open of the next bar a,           exit at the close of bar a+h       (H59; holds h+1 bars)
  mid    enter at (high+low)/2 of the next bar a,       exit at the close of bar a+h       (a spread-free mid proxy)
"""
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import guard  # noqa: E402

TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(TAC, "..", ".."))
CACHE = "D:/nt/nt_t09_cache"
L = 60; STRIDE = 10
KS = (1, 2, 3, 5, 10, 15, 30, 60)            # lags of the "fade the last k minutes" rule
HS = (1, 5, 10, 15, 30, 60)                  # target horizons in minutes
H0 = 15; DB0 = 5e-4                          # the regression's horizon and deadband
ENTRIES = ("close", "nopen", "mid")
FEAT_NAMES = ["ema20", "rsi14", "macd", "bb20", "atr14", "stoch14", "cci20", "ret1", "ret5", "ret10", "ret15", "ret20", "ret30"]
GROUPS = {"returns": ["ret1", "ret5", "ret10", "ret15", "ret20", "ret30"], "position": ["ema20", "bb20", "stoch14", "cci20"],
          "oscillator": ["rsi14", "macd"], "volatility": ["atr14"]}
CONDITIONS = ["vol", "volume", "hl", "cs", "move1bps", "move1sig", "weekend", "hour"]


SPLICE = pd.Timestamp("2024-09-13 08:00")          # file time: before it the cut file is shifted (UTC+4 in US daylight saving, UTC+5 otherwise), after it UTC
TICK_CHANGE = pd.Timestamp("2022-08-10")           # Bitstamp BTC/USD tick $0.01 -> $1
GAP_2025 = pd.Timestamp("2025-03-15")              # a 1,160-minute gap in the file


def _nth_sunday(y, m, n):
    d = pd.Timestamp(y, m, 1); return d + pd.Timedelta(days=(6 - d.dayofweek) % 7) + pd.Timedelta(weeks=n - 1)


def seam_regions(years=range(2017, 2025)):
    """File-time intervals of the cut file that hold an artificial jump: the whole day of each US daylight-saving change (second Sunday of March,
    first Sunday of November; the seam sits at a varying file hour) and 2 hours around the 2024-09-13 splice."""
    reg = []
    for y in years:
        for m, n in ((3, 2), (11, 1)):
            d = _nth_sunday(y, m, n); reg.append((d, d + pd.Timedelta(days=1)))
    reg.append((SPLICE - pd.Timedelta(hours=2), SPLICE + pd.Timedelta(hours=2)))
    return reg


def seam_mask(t_ns, which="cut", margin_min=61):
    """True for anchors whose window (60 bars back) or label (up to 60 bars ahead) can touch a seam. The forward file (UTC, Binance) has none."""
    t = pd.DatetimeIndex(t_ns.astype("datetime64[ns]")); out = np.zeros(len(t), bool)
    if which != "cut":
        return out
    for a, b in seam_regions():
        out |= np.asarray((t >= a - pd.Timedelta(minutes=margin_min)) & (t < b + pd.Timedelta(minutes=margin_min)))
    return out


def utc_hour(t_ns, which="cut"):
    """Hour of day in UTC. Cut file before the splice: file hour minus 4 during US daylight saving (second Sunday of March to first Sunday of
    November, by date) and minus 5 otherwise; after the splice, and for the forward file, the file hour is UTC."""
    t = pd.DatetimeIndex(t_ns.astype("datetime64[ns]")); h = t.hour.values.astype(int)
    if which != "cut":
        return h
    off = np.full(len(t), 5)
    for y in range(2016, 2026):
        a, b = _nth_sunday(y, 3, 2), _nth_sunday(y, 11, 1)
        off[np.asarray((t >= a) & (t < b))] = 4
    off[np.asarray(t >= SPLICE)] = 0
    return (h - off) % 24


def deadband(h):
    """5 bps at 15 minutes, scaled by sqrt(h / 15) (decay_lib.deadband)."""
    return DB0 * math.sqrt(h / 15.0)


def import_lab():
    """lab.py chdirs to the repo root on import; restore the cwd."""
    cwd = os.getcwd(); sys.path.insert(0, os.path.join(TAC, "lab"))
    try:
        import lab
    finally:
        os.chdir(cwd)
    return lab


def cs_series(h, l):
    """Corwin-Schultz (2012) spread estimate per bar from this bar and the one before, in bps (negatives set to 0)."""
    h = np.asarray(h, float); l = np.asarray(l, float)
    with np.errstate(all="ignore"):
        hl = np.log(h / l) ** 2
        beta = hl[1:] + hl[:-1]
        gamma = np.log(np.maximum(h[1:], h[:-1]) / np.minimum(l[1:], l[:-1])) ** 2
        k = 3 - 2 * math.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([[0.0], np.clip(s, 0, None)]) * 1e4


def entry_returns(o, h, l, c, anch):
    """Forward returns [N, len(HS)] per entry convention for anchors `anch` (the index of the bar after the window's last bar a-1):
    close  c[a-1+h] / c[a-1] - 1;   nopen  c[a+h] / o[a] - 1;   mid  c[a+h] / ((h[a] + l[a]) / 2) - 1.  NaN when the exit bar does not exist."""
    n = len(c); last = anch - 1; mid = (h + l) / 2
    R = {e: np.full((len(anch), len(HS)), np.nan) for e in ENTRIES}
    for j, hz in enumerate(HS):
        ok = last + hz < n; R["close"][ok, j] = c[last[ok] + hz] / c[last[ok]] - 1
        ok = anch + hz < n
        R["nopen"][ok, j] = c[anch[ok] + hz] / o[anch[ok]] - 1
        R["mid"][ok, j] = c[anch[ok] + hz] / mid[anch[ok]] - 1
    return R


def build_dataset(which, cache=True):
    """Anchor dataset of a file ('cut' or 'fwd'). Arrays of length N (anchors) unless noted."""
    path = f"{CACHE}/ds_{which}.npz"
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}
    s = guard.load_ohlcv(which); A = s.values.astype(np.float64); n = len(A)
    o, h, l, c, v = (A[:, i] for i in range(5))
    anch = np.arange(L, n - H0, STRIDE); last = anch - 1
    lab = import_lab()
    F = []
    for k in range(0, len(anch), 20000):
        a = anch[k:k + 20000]
        W = A[a[:, None] + np.arange(-L, 0)].astype(np.float32)
        F.append(np.nan_to_num(lab.fs_tb7(W)))
    F = np.concatenate(F)
    lr = np.diff(np.log(c), prepend=np.nan)
    cs = np.cumsum(np.nan_to_num(lr)); cs2 = np.cumsum(np.nan_to_num(lr) ** 2)
    # window sigma = std of the 59 one-bar log returns lr[a-L+1 : a]
    i0, i1 = anch - L + 1, anch
    m = i1 - i0; mean = (cs[i1 - 1] - cs[i0 - 1]) / m
    sig = np.sqrt(np.maximum((cs2[i1 - 1] - cs2[i0 - 1]) / m - mean ** 2, 0))
    cv = np.concatenate([[0.0], np.cumsum(v)]); vmean = (cv[anch] - cv[anch - L]) / L
    D = {"F": F, "t": s.index[last].values.astype("datetime64[ns]").astype(np.int64), "anch": anch, "sig": sig, "vmean": vmean}
    cs_bar = cs_series(h, l)
    D["seam"] = seam_mask(D["t"], which); D["hour_utc"] = utc_hour(D["t"], which)
    D["cs"] = cs_bar[last]; D["hl"] = (h[last] - l[last]) / c[last] * 1e4
    D["move1bps"] = np.abs(c[last] / c[last - 1] - 1) * 1e4
    D["move1sig"] = np.abs(F[:, FEAT_NAMES.index("ret1")])
    LR = np.full((len(anch), len(KS)), np.nan)
    for j, k in enumerate(KS):
        ok = last - k >= 0; LR[ok, j] = c[last[ok]] / c[last[ok] - k] - 1
    D["lastret"] = LR
    R = entry_returns(o, h, l, c, anch)
    for e in ENTRIES:
        D["r_" + e] = R[e]
    D["entry_open"] = o[np.minimum(anch, n - 1)]; D["close_last"] = c[last]
    # market quality of the whole series by calendar month (not only the anchors)
    D.update(market_quality(s, cs_bar))
    if cache:
        os.makedirs(CACHE, exist_ok=True); np.savez(path, **D)
    return D


def market_quality(s, cs_bar):
    """Per calendar month: median CS and high-low spread proxies (bps), mean bar volume, lag-1 autocorrelation of 1-minute log returns,
    variance ratio VR(1, 15) = var(15-minute log return, overlapping) / (15 var(1-minute log return))."""
    c = s["close"].values; lr = np.diff(np.log(c), prepend=np.nan)
    hl = ((s["high"] - s["low"]) / s["close"]).values * 1e4
    mo = s.index.to_period("M")
    rows = []
    for m in pd.period_range(mo.min(), mo.max(), freq="M"):
        ix = np.flatnonzero(mo == m)
        if len(ix) < 1000:
            continue
        x = lr[ix[0]:ix[-1] + 1]; x = x[np.isfinite(x)]
        ac1 = np.corrcoef(x[1:], x[:-1])[0, 1] if x.std() > 0 else np.nan
        cl = np.log(c[ix[0]:ix[-1] + 1]); r15 = cl[15:] - cl[:-15]
        vr = r15.var() / (15 * x.var()) if x.var() > 0 else np.nan
        cov = float(np.mean(x[1:] * x[:-1]) - x.mean() ** 2)
        roll = 2 * math.sqrt(max(-cov, 0.0)) * 1e4                     # Roll (1984) spread from the serial covariance of returns, bps
        rows.append((str(m), float(np.median(cs_bar[ix])), float(np.median(hl[ix])), float(s["volume"].values[ix].mean()), float(ac1), float(vr), roll))
    return {"mq_month": np.array([r[0] for r in rows]), "mq": np.array([r[1:] for r in rows], float)}


MQ_COLS = ("cs_median_bps", "hl_median_bps", "volume_mean", "ac1", "vr_1_15", "roll_bps")


# ------------------------------------------------------------------ the walk-forward
def month_range(t, first=None, last=None):
    t = pd.DatetimeIndex(t.astype("datetime64[ns]"))
    f = pd.Period(first, "M") if first else t[0].to_period("M") + 3
    return pd.period_range(f, pd.Period(last, "M") if last else t[-1].to_period("M"), freq="M")


def split(t, m0, m1, train_days, H=H0, first_train=None):
    """train: anchors whose label is finished before m0 (t <= m0 - (H+2) min) and at most train_days old; test: t in [m0, m1)."""
    t = pd.DatetimeIndex(t.astype("datetime64[ns]"))
    tr = (t >= m0 - pd.Timedelta(days=train_days)) & (t <= m0 - pd.Timedelta(minutes=H + 2))
    if first_train is not None:
        tr = tr & (t >= first_train)
    te = (t >= m0) & (t < m1)
    return np.asarray(tr), np.asarray(te)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(y) < 2 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def tint(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float); n = len(v)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if n < 2:
        return {"mean": float(v.mean()), "lo": None, "hi": None, "n": n}
    from scipy.stats import t as T
    h = T.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return {"mean": float(v.mean()), "lo": float(v.mean() - h), "hi": float(v.mean() + h), "n": n}


def fit_logreg(X, y, C=0.1):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=1000).fit(X, y)


def walk(D, entry="close", train_days=60, first=None, last=None, min_train=2000, min_test=500, first_train=None, extras=False,
         n_perm=5, seed=0, feature_cols=None):
    """Monthly walk-forward of the 13-feature logistic regression on label `entry`. Returns
      months: list of dicts (month, auc, n_train, n_test, coef (13), thr (top-10% |p-0.5| from training), uni (13 univariate AUCs on the test set),
              and with extras: drop (13 drop-one AUC losses), perm (13 per-feature and len(GROUPS) grouped permutation losses))
      p: array [N] of P(up) on every test anchor (NaN elsewhere; the deadband mask is applied by the analyses)
      models: {month: (scaler, model)}"""
    from sklearn.preprocessing import StandardScaler
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); F = D["F"]
    if feature_cols is not None:
        F = F[:, feature_cols]
    r = D["r_" + entry][:, HS.index(H0)]; ok = np.isfinite(r)
    y = (r > 0).astype(int); mask = ok & (np.abs(np.nan_to_num(r)) > DB0)
    rng = np.random.default_rng(seed)
    p_all = np.full(len(t), np.nan, np.float32); out = []; models = {}
    for m in month_range(D["t"], first, last):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = split(D["t"], m0, m1, train_days, first_train=first_train)
        trm, tem = tr & mask, te & mask
        if trm.sum() < min_train or tem.sum() < min_test or len(np.unique(y[tem])) < 2:
            continue
        sc = StandardScaler().fit(F[trm]); mdl = fit_logreg(sc.transform(F[trm]), y[trm])
        p = mdl.predict_proba(sc.transform(F[te]))[:, 1]; ptr = mdl.predict_proba(sc.transform(F[trm]))[:, 1]
        p_all[te] = p
        thr = float(np.quantile(np.abs(ptr - 0.5), 0.9))
        rec = {"month": str(m), "auc": auc(y[tem], p_all[tem]), "n_train": int(trm.sum()), "n_test": int(tem.sum()), "n_test_all": int(te.sum()),
               "coef": mdl.coef_[0].tolist(), "intercept": float(mdl.intercept_[0]), "thr": thr, "up_rate": float(y[tem].mean()),
               "uni": [auc(y[tem], F[tem][:, j]) for j in range(F.shape[1])]}
        if extras:
            Xtr, Xte = sc.transform(F[trm]), sc.transform(F[tem]); base = rec["auc"]; nf = F.shape[1]
            drop = []
            for j in range(nf):
                keep = [i for i in range(nf) if i != j]
                mj = fit_logreg(Xtr[:, keep], y[trm]); drop.append(base - auc(y[tem], mj.predict_proba(Xte[:, keep])[:, 1]))
            rec["drop"] = drop
            perm = []
            for j in range(nf):
                s_ = []
                for _ in range(n_perm):
                    Xp = Xte.copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                    s_.append(auc(y[tem], mdl.predict_proba(Xp)[:, 1]))
                perm.append(base - float(np.mean(s_)))
            gperm = []
            if nf == len(FEAT_NAMES):
                for g, names in GROUPS.items():
                    idx = [FEAT_NAMES.index(nm) for nm in names]; s_ = []
                    for _ in range(n_perm):
                        Xp = Xte.copy(); Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
                        s_.append(auc(y[tem], mdl.predict_proba(Xp)[:, 1]))
                    gperm.append(base - float(np.mean(s_)))
            gdrop, gonly = [], []
            if nf == len(FEAT_NAMES):
                for g, names in GROUPS.items():
                    idx = [FEAT_NAMES.index(nm) for nm in names]; keep = [i for i in range(nf) if i not in idx]
                    mg = fit_logreg(Xtr[:, keep], y[trm]); gdrop.append(base - auc(y[tem], mg.predict_proba(Xte[:, keep])[:, 1]))
                    mo = fit_logreg(Xtr[:, idx], y[trm]); gonly.append(auc(y[tem], mo.predict_proba(Xte[:, idx])[:, 1]))
            rec["perm"] = perm; rec["gperm"] = gperm; rec["gdrop"] = gdrop; rec["gonly"] = gonly
        out.append(rec); models[str(m)] = (sc, mdl)
    return {"months": out, "p": p_all, "models": models}


def by_year(rows, key, f=None, years=None):
    d = {}
    for r in rows:
        v = f(r) if f else r[key]
        d.setdefault(r["month"][:4], []).append(v)
    return {y: v for y, v in sorted(d.items())}

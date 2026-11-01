"""Causal features for the PnL study (SPEC 'data_and_features'): F-CORE (48 columns, every minute) + F-DAILY (7).

Every column at row t uses bars <= t only (EWMAs via lfilter, trailing rolling windows, completed UTC days for
F-DAILY, settled funding prints). Causality is tested by prefix invariance: features of a prefix equal the first rows
of the features of the full series (tests/test_features.py).

  python features.py        -> $PNL_WORK/features_v1.npy (dev rows x 55, float32) and features_v1.json
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import lfilter

import data as D

WORK = os.environ.get("PNL_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
RET_L = (1, 3, 5, 15, 30, 60, 120, 240, 480, 1440, 4320, 10080)
VOL_S = (15, 60, 240, 1440, 10080)
TRACK1_DROP = ("z_4320", "z_10080", "lnsig_10080", "dhi_7d", "dlo_7d", "fund_90")
WARMUP_MIN = 31 * 1440
DAILY_WARMUP_DAYS = 56


def ewm(x, s):
    """Adjusted causal EWMA with alpha = 2/(s+1): numerator and denominator both filtered (exact from the start)."""
    a = 2.0 / (s + 1.0)
    num = lfilter([a], [1.0, -(1.0 - a)], x)
    den = lfilter([a], [1.0, -(1.0 - a)], np.ones_like(x))
    return num / den


def lag(x, L):
    out = np.empty_like(x)
    out[:L] = x[0]
    out[L:] = x[:-L]
    return out


def core(b):
    """F-CORE: (n, 48) float64 and column names, for Bars b (1-min)."""
    lc = np.log(b.c)
    r = np.zeros_like(lc)
    r[1:] = np.diff(lc)
    r2 = r * r
    sig = np.maximum(np.sqrt(ewm(r2, 1440)), 0.5e-4)           # per-minute sigma, floored at 0.5 bps
    cols, names = [], []
    for L in RET_L:                                              # A. returns in sigma units
        cols.append((lc - lag(lc, L)) / (sig * np.sqrt(L)))
        names.append(f"z_{L}")
    for s in VOL_S:                                              # B. volatility levels (bps per minute)
        cols.append(np.log(np.maximum(np.sqrt(ewm(r2, s)), 1e-6) * 1e4))
        names.append(f"lnsig_{s}")
    rho = np.log(b.h / b.l)
    for s in (60, 1440):
        park = np.sqrt(np.maximum(ewm(rho * rho, s), 1e-16) / (4 * np.log(2)))
        cols.append(np.log(park / np.maximum(np.sqrt(ewm(r2, s)), 1e-8)))
        names.append(f"park_{s}")
    for s in (60, 1440):                                         # C. path shape
        cols.append(np.abs(ewm(r, s)) / (ewm(np.abs(r), s) + 1e-12))
        names.append(f"er_{s}")
    r15 = lc - lag(lc, 15)
    cols.append(np.log(np.maximum(ewm(r15 * r15, 240), 1e-16) / (15 * np.maximum(ewm(r2, 240), 1e-16))))
    names.append("vr15")
    lh, ll = np.log(b.h), np.log(b.l)
    for L, tag in ((1440, "1d"), (10080, "7d")):
        hi = pd.Series(lh).rolling(L, min_periods=1).max().to_numpy()
        lo = pd.Series(ll).rolling(L, min_periods=1).min().to_numpy()
        cols += [(lc - hi) / (sig * np.sqrt(L)), (lc - lo) / (sig * np.sqrt(L))]
        names += [f"dhi_{tag}", f"dlo_{tag}"]
    v = b.v.astype(np.float64)                                   # D. activity
    n = b.n.astype(np.float64)
    v30 = ewm(v, 43200) + 1e-9
    for s in (5, 60, 1440):
        cols.append(np.log((ewm(v, s) + 1e-9) / v30))
        names.append(f"vol_{s}")
    n30 = ewm(n, 43200) + 1e-9
    cols.append(np.log((ewm(n, 60) + 1e-9) / n30))
    names.append("trades_60")
    cols.append(np.log(((ewm(v, 60) + 1e-9) / (ewm(n, 60) + 1e-9)) / (v30 / n30)))
    names.append("tsize_60")
    imb = 2 * b.tbb.astype(np.float64) - v                       # E. taker imbalance
    for s in (5, 15, 60, 240, 1440):
        cols.append(ewm(imb, s) / (ewm(v, s) + 1e-9))
        names.append(f"imb_{s}")
    ev = np.flatnonzero(b.funding != 0)                          # F. funding (settled prints, bps per 8 h)
    rates = b.funding[ev] * 1e4
    cum = np.concatenate([[0.0], np.cumsum(rates)])
    k = np.searchsorted(ev, np.arange(len(lc)), side="right")    # number of prints settled at or before row t
    last = np.where(k > 0, rates[np.maximum(k - 1, 0)], 0.0)
    m3 = np.where(k > 0, (cum[k] - cum[np.maximum(k - 3, 0)]) / np.minimum(np.maximum(k, 1), 3), 0.0)
    m90 = np.where(k > 0, (cum[k] - cum[np.maximum(k - 90, 0)]) / np.minimum(np.maximum(k, 1), 90), 0.0)
    cols += [last, m3, m90]
    names += ["fund_last", "fund_3", "fund_90"]
    hour = ((b.t % 86400) / 3600.0)                              # G. calendar (UTC)
    for j in (1, 2, 3):
        cols += [np.sin(2 * np.pi * j * hour / 24), np.cos(2 * np.pi * j * hour / 24)]
        names += [f"hsin{j}", f"hcos{j}"]
    dow = ((b.t // 86400) + 3) % 7                               # 1970-01-01 was a Thursday -> Monday = 0
    cols += [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(np.float64)]
    names += ["dsin", "dcos", "weekend"]
    X = np.column_stack(cols)
    assert X.shape[1] == 48, X.shape
    return X, names, sig


def daily(b):
    """F-DAILY: (n, 7) from completed UTC days, forward-filled to minutes (row t uses days before t's day)."""
    day = b.t // 86400
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]
    C = np.log(b.c[ends - 1])                                    # daily log close
    rd = np.zeros_like(C)
    rd[1:] = np.diff(C)
    sd = np.maximum(np.sqrt(ewm(rd * rd, 30)), 1e-4)
    F, names = [], []
    for L in (14, 28, 56):
        F.append((C - lag(C, L)) / (sd * np.sqrt(L)))
        names.append(f"dz_{L}")
    for L in (28, 56):
        mx = pd.Series(C).rolling(L, min_periods=1).max().to_numpy()
        F.append((C - mx) / (sd * np.sqrt(L)))
        names.append(f"ddd_{L}")
    sma = np.log(pd.Series(np.exp(C)).rolling(50, min_periods=1).mean().to_numpy())
    F.append((C - sma) / (sd * np.sqrt(50)))
    names.append("dsma_50")
    F.append(np.log(0.5 / (np.sqrt(365) * sd)))
    names.append("lnvt")
    F = np.column_stack(F)
    Fd = np.zeros_like(F)                                        # value for day d uses day d-1 (completed)
    Fd[1:] = F[:-1]
    out = np.repeat(Fd, ends - starts, axis=0)
    return out, names, np.repeat(np.arange(len(starts)), ends - starts)


def build(b):
    X, names, sig = core(b)
    Fd, dn, dayidx = daily(b)
    X = np.column_stack([X, Fd])
    names = names + dn
    return X.astype(np.float32), names, sig


def track1_cols(names):
    return [i for i, n in enumerate(names[:48]) if n not in TRACK1_DROP]


def main():
    b = D.load(1)
    X, names, sig = build(b)
    os.makedirs(WORK, exist_ok=True)
    np.save(os.path.join(WORK, "features_v1.npy"), X)
    np.save(os.path.join(WORK, "sigma_v1.npy"), sig.astype(np.float32))
    src = open(os.path.abspath(__file__), "rb").read()
    meta = {"columns": names, "n": int(len(X)), "first_t": int(b.t[0]), "last_t": int(b.t[-1]),
            "track1_cols": track1_cols(names), "warmup_min": WARMUP_MIN, "daily_warmup_days": DAILY_WARMUP_DAYS,
            "features_sha256": hashlib.sha256(src).hexdigest(), "nan": int(np.isnan(X).sum()),
            "inf": int(np.isinf(X).sum())}
    json.dump(meta, open(os.path.join(WORK, "features_v1.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "columns"}))


if __name__ == "__main__":
    main()

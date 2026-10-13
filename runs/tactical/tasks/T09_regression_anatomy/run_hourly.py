"""T09 Q8: the hourly cousin. The H62 `rich` logistic (43 features, C 0.1, 730-day monthly walk-forward, 4-hour horizon, next-open entry,
5-bps-at-15-minutes deadband scaled to 4 hours) on the cut file's 1-hour bars, months 2019-01..2025-06: Q1 (coefficients, univariate AUC, drop-one,
permutation, family groups) and Q3 (fade-k impulse response, hours). Cross-checked against recent/wf.py walk_forward on the same inputs.
usage: python run_hourly.py -> results_hourly.json"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import lib
import guard
from run_minute import jsonable

RICH_NAMES = ([f"ret{k}" for k in (1, 2, 3, 5, 10, 15, 20, 30, 45, 59)] + [f"ema{p}" for p in (5, 10, 20, 40)] + [f"rsi{p}" for p in (7, 14, 28)]
              + ["macd_12_26", "macd_5_13"] + [f"bb{p}" for p in (10, 20, 40)] + [f"stoch{p}" for p in (5, 14, 30, 60)] + [f"cci{p}" for p in (10, 20, 40)]
              + ["atr7", "atr14", "mfi14", "mfi30"] + ["candle_body", "candle_upwick", "candle_lowwick", "range_over_sd", "dstd10", "dstd5", "logvol_last",
                                                       "logvol5", "obv10", "vwap20dev"])
FAMILY = {"returns": RICH_NAMES[0:10], "ema": RICH_NAMES[10:14], "rsi": RICH_NAMES[14:17], "macd": RICH_NAMES[17:19], "bollinger": RICH_NAMES[19:22],
          "stochastic": RICH_NAMES[22:26], "cci": RICH_NAMES[26:29], "atr": ["atr7", "atr14"], "mfi": ["mfi14", "mfi30"],
          "candle": RICH_NAMES[33:37], "vol_regime": ["dstd10", "dstd5"], "volume": ["logvol_last", "logvol5", "obv10"], "vwap": ["vwap20dev"]}
H = 4; L = 60; T_DAYS = 730; CLIP = 8.0; DB = 5e-4 * np.sqrt(H * 60 / 15)
KS = (1, 2, 3, 4, 6, 12, 24, 48); HS = (1, 2, 4, 8, 12, 24)
FIRST, LAST = "2019-01", "2025-06"


def build():
    s = guard.load_ohlcv("cut"); s = s[s.index < pd.Timestamp("2025-07-01")]
    b = s.resample("60min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    A = b.values.astype(np.float64); n = len(A); o, c = A[:, 0], A[:, 3]
    anch = np.arange(L, n - max(HS)); last = anch - 1
    lab = lib.import_lab(); X = []
    for k in range(0, len(anch), 20000):
        a = anch[k:k + 20000]
        X.append(np.nan_to_num(np.asarray(lab.fs_rich(A[a[:, None] + np.arange(-L, 0)].astype(np.float32)), np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    X = np.concatenate(X); assert X.shape[1] == len(RICH_NAMES), X.shape
    t = b.index[last].values.astype("datetime64[ns]")
    R = {"close": np.full((len(anch), len(HS)), np.nan), "nopen": np.full((len(anch), len(HS)), np.nan)}
    for j, h in enumerate(HS):
        ok = last + h < n; R["close"][ok, j] = c[last[ok] + h] / c[last[ok]] - 1
        ok = anch + h - 1 < n; R["nopen"][ok, j] = c[anch[ok] + h - 1] / o[anch[ok]] - 1
    LR = np.full((len(anch), len(KS)), np.nan)
    for j, k in enumerate(KS):
        LR[:, j] = c[last] / c[last - k] - 1
    return X, t, R, LR, anch, A


def walk(X, t, r, first=FIRST, last=LAST, extras=True, n_perm=5, seed=0):
    from sklearn.preprocessing import StandardScaler
    ti = pd.DatetimeIndex(t); mask = np.isfinite(r) & (np.abs(np.nan_to_num(r)) > DB); y = (r > 0).astype(int); rng = np.random.default_rng(seed); out = []
    nf = X.shape[1]
    for m in pd.period_range(first, last, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        lab_end = ti + pd.Timedelta(minutes=(H + 1) * 60)
        tr = np.asarray((ti >= m0 - pd.Timedelta(days=T_DAYS)) & (lab_end <= m0)); te = np.asarray((ti >= m0) & (ti < m1))
        trm, tem = tr & mask, te & mask
        if trm.sum() < 500 or tem.sum() < 300 or len(np.unique(y[tem])) < 2 or len(np.unique(y[trm])) < 2:
            continue
        sc = StandardScaler().fit(X[trm])

        def g(Z):
            return np.clip(sc.transform(np.asarray(Z, np.float32)), -CLIP, CLIP)
        Xtr, Xte = g(X[trm]), g(X[tem]); mdl = lib.fit_logreg(Xtr, y[trm]); base = lib.auc(y[tem], mdl.predict_proba(Xte)[:, 1])
        rec = {"month": str(m), "auc": base, "n_train": int(trm.sum()), "n_test": int(tem.sum()), "coef": mdl.coef_[0].tolist(),
               "uni": [lib.auc(y[tem], X[tem][:, j]) for j in range(nf)]}
        if extras:
            drop, perm = [], []
            for j in range(nf):
                keep = [i for i in range(nf) if i != j]; mj = lib.fit_logreg(Xtr[:, keep], y[trm])
                drop.append(base - lib.auc(y[tem], mj.predict_proba(Xte[:, keep])[:, 1]))
                s_ = []
                for _ in range(n_perm):
                    Xp = Xte.copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]; s_.append(lib.auc(y[tem], mdl.predict_proba(Xp)[:, 1]))
                perm.append(base - float(np.mean(s_)))
            gdrop, gperm, gonly = [], [], []
            for gname, names in FAMILY.items():
                idx = [RICH_NAMES.index(nm) for nm in names]; keep = [i for i in range(nf) if i not in idx]
                mg = lib.fit_logreg(Xtr[:, keep], y[trm]); gdrop.append(base - lib.auc(y[tem], mg.predict_proba(Xte[:, keep])[:, 1]))
                mo = lib.fit_logreg(Xtr[:, idx], y[trm]); gonly.append(lib.auc(y[tem], mo.predict_proba(Xte[:, idx])[:, 1]))
                s_ = []
                for _ in range(n_perm):
                    Xp = Xte.copy(); Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]; s_.append(lib.auc(y[tem], mdl.predict_proba(Xp)[:, 1]))
                gperm.append(base - float(np.mean(s_)))
            rec.update(drop=drop, perm=perm, gdrop=gdrop, gperm=gperm, gonly=gonly)
        out.append(rec)
    return {"months": out}


def agg_year(rows, f):
    d = {}
    for r in rows:
        d.setdefault(r["month"][:4], []).append(f(r))
    return {y: lib.tint(v) for y, v in sorted(d.items())}


def impulse(t, R, LR, entry, exclude_seam=True):
    ti = pd.DatetimeIndex(t); mon = np.array([str(x) for x in ti.to_period("M")]); ms = [m for m in sorted(set(mon)) if FIRST <= m <= LAST]
    seam = lib.seam_mask(np.asarray(t).astype("datetime64[ns]"), "cut", margin_min=60 * 60) if exclude_seam else np.zeros(len(ti), bool)   # window 60 h back, label up to 24 h ahead
    A = np.full((len(ms), len(KS), len(HS)), np.nan); B = np.full_like(A, np.nan)
    for i, m in enumerate(ms):
        ii = np.flatnonzero((mon == m) & ~seam)
        for a in range(len(KS)):
            lr = LR[ii, a]
            for b, h in enumerate(HS):
                r = R[entry][ii, b]; ok = np.isfinite(lr) & np.isfinite(r); db = 5e-4 * np.sqrt(h * 60 / 15); mm = ok & (np.abs(r) > db)
                if mm.sum() >= 100:
                    A[i, a, b] = lib.auc((r[mm] > 0).astype(int), -lr[mm])
                nz = ok & (lr != 0)
                if nz.sum() >= 100:
                    B[i, a, b] = float(np.mean(-np.sign(lr[nz]) * r[nz])) * 1e4
    return ms, A, B


def main():
    from scipy.stats import t as Tt
    T0 = time.time(); X, t, R, LR, anch, A = build(); print("built", X.shape, f"{time.time()-T0:.0f}s", flush=True)
    j4 = HS.index(H); W = walk(X, t, R["nopen"][:, j4]); print("walk", len(W["months"]), f"{time.time()-T0:.0f}s", flush=True)
    sys.path.insert(0, os.path.join(lib.TAC, "recent")); import wf
    ref = wf.walk_forward(X, t, R["nopen"][:, j4], "1h", "logreg", T_DAYS, FIRST, LAST, H=H, bar_min=60, db=DB)
    refd = {r["month"]: r["auc"] for r in ref}; mine = {r["month"]: r["auc"] for r in W["months"]}
    diff = [abs(mine[m] - refd[m]) for m in mine if m in refd]
    out = {"n_months": len(W["months"]), "crosscheck_vs_wf": {"n_common": len(diff), "max_abs_diff": float(max(diff)),
                                                              "months_only_mine": [m for m in mine if m not in refd], "months_only_wf": [m for m in refd if m not in mine]}}
    rows = W["months"]; q = {"features": RICH_NAMES, "groups": list(FAMILY), "years": sorted({r["month"][:4] for r in rows}), "auc": agg_year(rows, lambda r: r["auc"])}
    for key in ("coef", "uni", "drop", "perm"):
        q[key] = {nm: agg_year(rows, lambda r, j=j, key=key: r[key][j]) for j, nm in enumerate(RICH_NAMES)}
    for key in ("gdrop", "gperm", "gonly"):
        q[key] = {g: agg_year(rows, lambda r, i=i, key=key: r[key][i]) for i, g in enumerate(FAMILY)}
    q["all_months_mean"] = {k: {nm: float(np.mean([r[k][j] for r in rows])) for j, nm in enumerate(RICH_NAMES)} for k in ("drop", "perm", "coef", "uni")}
    q["all_months_mean_groups"] = {k: {g: float(np.mean([r[k][i] for r in rows])) for i, g in enumerate(FAMILY)} for k in ("gdrop", "gperm", "gonly")}
    out["q1"] = q; out["monthly_auc"] = {r["month"]: r["auc"] for r in rows}
    out["q3"] = {}
    for e in ("close", "nopen"):
        ms, Aa, Ba = impulse(t, R, LR, e)
        by = {}
        for y in sorted({m[:4] for m in ms}):
            ii = [i for i, m in enumerate(ms) if m[:4] == y]; X_ = Aa[ii]; n = np.isfinite(X_).sum(0); sd = np.nanstd(X_, 0, ddof=1)
            by[y] = {"auc": np.nanmean(X_, 0).tolist(), "bps": np.nanmean(Ba[ii], 0).tolist(),
                     "auc_halfwidth": (Tt.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(np.maximum(n, 1))).tolist()}
        out["q3"][e] = {"by_year": by, "all": {"auc": np.nanmean(Aa, 0).tolist(), "bps": np.nanmean(Ba, 0).tolist()}, "ks": KS, "hs": HS}
    ms, Aa, _ = impulse(t, R, LR, "nopen"); reg = {r["month"]: r["auc"] for r in rows}; share = {}
    for y in sorted({m[:4] for m in ms}):
        ii = [i for i, m in enumerate(ms) if m[:4] == y and m in reg]
        if not ii:
            continue
        mk = np.nanmean(Aa[ii][:, :, j4], 0); kb = int(np.nanargmax(mk)); ra = float(np.mean([reg[ms[i]] for i in ii]))
        share[y] = {"regression_auc": ra, "best_k_hours": KS[kb], "best_auc": float(mk[kb]), "share": (float(mk[kb]) - 0.5) / (ra - 0.5) if ra > 0.5 else None}
    out["q3"]["rule_share_nopen_h4"] = share
    _, A_incl, _ = impulse(t, R, LR, "nopen", exclude_seam=False)
    out["q3"]["seam"] = {"n_seam_anchors": int(lib.seam_mask(np.asarray(t).astype("datetime64[ns]"), "cut", margin_min=3600).sum()), "n_anchors": int(len(t)),
                         "auc_max_abs_diff_incl_vs_excl": float(np.nanmax(np.abs(A_incl - Aa))), "auc_mean_abs_diff": float(np.nanmean(np.abs(A_incl - Aa)))}
    json.dump(jsonable(out), open(os.path.join(HERE, "results_hourly.json"), "w"), indent=1)
    print(f"done {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

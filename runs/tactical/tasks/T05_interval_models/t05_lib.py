"""T05 (SPEC.md): better 60-minute return intervals than HAR. Six models (M0 HAR, M1 EWMA 0.94, M2 HAR+semivariances, M3 HAR+jump state,
M4 quantile regression, M5 HAR with ACI), walk-forward monthly scoring, a freeze for Look B. CPU only.
Reuses T03's HAR (fit_har, sigma, quantiles, cal_rows, month_rows, ci) and T01's guarded loaders unchanged."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.linear_model import Ridge, QuantileRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TAC, "tasks", "T03_vol_intervals"))
import t03_lib as L3  # noqa: E402  (HAR, interval-score helpers, cal rows; imports T01's loaders)
T1 = L3.T1; V = T1.V

H = 60; ALPHA = 0.10; STRIDE = T1.STRIDE; TRAIN_DAYS = 365
MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")
EWMA_LAMBDA = 0.94; ACI_GAMMA = 0.005; ACI_CLIP = (0.002, 0.5)
QR_STRIDE = 4                 # M4 is fitted on every 4th training row (LP size); recorded in the results
QR_Q = (ALPHA / 2, 1 - ALPHA / 2)
CUT_CSV = T1.TRAIN_CSV; FWD_CSV = T1.FWD_CSV
SEL_MONTHS = [str(m) for m in pd.period_range("2019-01", "2024-06", freq="M")]
LOOKA_MONTHS = [str(m) for m in pd.period_range("2017-09", "2018-12", freq="M")]
SEL_LOAD = (pd.Timestamp("2018-01-01 00:00"), pd.Timestamp("2024-07-01 02:00"))      # 365 d before 2019-01, 1440 bars of history inside, labels past 2024-06
LOOKA_LOAD = (pd.Timestamp("2017-08-17 00:00"), pd.Timestamp("2019-01-01 02:00"))
FREEZE_LOAD = (pd.Timestamp("2025-10-02 00:00"), pd.Timestamp("2026-09-30 23:59"))   # nothing from October 2026 is ever loaded
FIRST = 1441


def load(path, lo_hi, allow_late=False):
    """Guarded loader. Forward-file loads stop at 2026-09-30 23:59 unless allow_late (only score_B.py, for months that are complete)."""
    lo, hi = lo_hi
    if os.path.abspath(path) == os.path.abspath(FWD_CSV) and not allow_late:
        assert hi <= FREEZE_LOAD[1], "forward loads for the freeze stop at 2026-09-30 23:59"
    return T1.load_bars(path, lo, hi, chunk=200_000)


def make_data(s, first=FIRST, horizons=(H,)):
    """Anchors (stride 7), the 60-minute log-return label and every model's features. Features at anchor a use bars < a only."""
    D = V.Data(s.values.astype(np.float64), s.index, horizons=tuple(horizons), stride=STRIDE, first=first, with_tb7=False)
    a = D.anch; lr = D.lr; cs = D.cs2; c = D.A[:, 3]
    r = np.log(c[a + H - 1]) - np.log(c[a - 1])
    cum = lambda x: np.concatenate([[0.0], np.cumsum(x)])
    cp, cn = cum(lr ** 2 * (lr > 0)), cum(lr ** 2 * (lr < 0))
    rvk = lambda cc, k: np.log(np.maximum(cc[a] - cc[a - k], 0) / k + 1e-12)
    rv60 = cs[a] - cs[a - 60]
    ratio = np.clip((cn[a] - cn[a - 60]) / (rv60 + 1e-18), 0, 1)
    m2 = np.column_stack([D.har, rvk(cp, 60), rvk(cn, 60), rvk(cp, 1440), rvk(cn, 1440), ratio])
    mx = pd.Series(np.abs(lr)).rolling(60).max().to_numpy()[a - 1]
    sig60 = np.sqrt(np.maximum(rv60, 0) / 60 + 1e-18)
    cpp = cum(np.concatenate([[0.0], np.abs(lr[1:]) * np.abs(lr[:-1])]))
    bv = (np.pi / 2) * (60 / 59) * (cpp[a] - cpp[a - 59])
    m3 = np.column_stack([D.har, np.log1p(mx / sig60), np.clip(bv / (rv60 + 1e-18), 0, 5)])
    ew = lfilter([1 - EWMA_LAMBDA], [1, -EWMA_LAMBDA], lr ** 2)[a - 1]          # variance per minute using bars <= a-1
    sg_ewma = np.sqrt(H * ew)
    trail = np.sqrt(np.maximum(cs[a] - cs[a - 4 * H], 0.0) / 4.0)
    ok = (np.all(D.har[:, :4] > -20, 1) & (trail > 0) & (sg_ewma > 0) & np.isfinite(m2).all(1) & np.isfinite(m3).all(1))
    return {"D": D, "r": {H: r}, "trail": {H: trail}, "ok": ok, "F": {"har": D.har, "m2": m2, "m3": m3}, "ewma": sg_ewma}


# ---------------------------------------------------------------- fitting
def fit_ridge(X, r):
    """The scale model of fit_har on arbitrary features: ridge (alpha 1, standardised) of ln|r|+1e-5, times a constant fitted on the same rows."""
    y = np.log(np.abs(r) + L3.EPS_Y); mu, sd = X.mean(0), X.std(0) + 1e-12
    rr = Ridge(alpha=1.0).fit((X - mu) / sd, y)
    m = {"mu": mu.tolist(), "sd": sd.tolist(), "coef": rr.coef_.tolist(), "ic": float(rr.intercept_)}
    e = np.exp(((X - mu) / sd) @ rr.coef_ + rr.intercept_)
    m["c"] = float(np.mean(np.abs(r) / e)); return m


def ridge_sigma(m, X):
    return np.exp(((X - np.asarray(m["mu"])) / np.asarray(m["sd"])) @ np.asarray(m["coef"]) + m["ic"]) * m["c"]


def fit_qr(X, z):
    """5% and 95% conditional quantiles of z (alpha 0, highs) on standardised X, fitted on every QR_STRIDE-th row."""
    mu, sd = X.mean(0), X.std(0) + 1e-12; Xs = (X - mu) / sd
    sub = np.arange(0, len(Xs), QR_STRIDE); out = {"mu": mu.tolist(), "sd": sd.tolist(), "n_fit": int(len(sub))}
    for nm, q in zip(("lo", "hi"), QR_Q):
        m = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(Xs[sub], z[sub])
        out[nm] = {"coef": m.coef_.tolist(), "ic": float(m.intercept_)}
    return out


def qr_pred(m, X):
    Xs = (X - np.asarray(m["mu"])) / np.asarray(m["sd"])
    return Xs @ np.asarray(m["lo"]["coef"]) + m["lo"]["ic"], Xs @ np.asarray(m["hi"]["coef"]) + m["hi"]["ic"]


def fit_model(name, B, rows):
    """Parameters (plain lists) of model `name` on training rows. M1 has none."""
    r = B["r"][H][rows]
    if name == "M0":
        return L3.fit_har(B, H, rows, None, False)                                    # T03's model, unchanged
    if name == "M1":
        return {}
    if name == "M2":
        return fit_ridge(B["F"]["m2"][rows], r)
    if name == "M3":
        return fit_ridge(B["F"]["m3"][rows], r)
    if name == "M4":
        m0 = L3.fit_har(B, H, rows, None, False); sg = L3.sigma(m0, B, rows, None)
        return {"har": m0, "qr": fit_qr(B["F"]["m2"][rows], r / sg)}
    if name == "M5":
        return L3.fit_har(B, H, rows, None, False)
    raise KeyError(name)


def sigma_of(name, params, B, idx):
    if name == "M1":
        return B["ewma"][idx]
    if name in ("M0", "M5"):
        return L3.sigma(params, B, idx, None)
    if name == "M2":
        return ridge_sigma(params, B["F"]["m2"][idx])
    if name == "M3":
        return ridge_sigma(params, B["F"]["m3"][idx])
    raise KeyError(name)


# ---------------------------------------------------------------- intervals
def qfun(zs, p):
    """linear-interpolated quantile of a SORTED sample (numpy's default method)."""
    return np.interp(p * (len(zs) - 1), np.arange(len(zs)), zs)


def aci_bounds(zs, sg, r, ta, gamma=ACI_GAMMA):
    """Adaptive conformal on the normalised score z = r / sigma: the level alpha_t moves by gamma * (alpha - miss) after every anchor
    whose label has ended (ta_j + H minutes <= ta_i); the bounds are the sample quantiles zs at alpha_t / 2 and 1 - alpha_t / 2.
    Anchor i's bounds use only labels of anchors j with ta_j + H <= ta_i."""
    n = len(r); lo = np.empty(n); hi = np.empty(n); a_t = ALPHA; p = 0; miss = np.zeros(n); end = ta + pd.Timedelta(minutes=H)
    for i in range(n):
        while p < i and end[p] <= ta[i]:
            a_t = float(np.clip(a_t + gamma * (ALPHA - miss[p]), *ACI_CLIP)); p += 1
        lo[i] = qfun(zs, a_t / 2) * sg[i]; hi[i] = qfun(zs, 1 - a_t / 2) * sg[i]
        miss[i] = float((r[i] < lo[i]) or (r[i] > hi[i]))
    return lo, hi


def cal_summary(name, params, B, cidx):
    """What the calibration rows contribute: z-quantiles (M0-M3), the CQR correction (M4) or the sorted z sample (M5). Stored in the frozen file."""
    r = B["r"][H]; params = revive_model(name, params)
    if name in ("M0", "M1", "M2", "M3"):
        q = L3.quantiles(r[cidx] / sigma_of(name, params, B, cidx)); return {"lo": q["lo"], "hi": q["hi"]}
    if name == "M5":
        return {"z_sorted": np.sort(r[cidx] / sigma_of(name, params, B, cidx)).tolist()}
    sgc = L3.sigma(params["har"], B, cidx, None); z = r[cidx] / sgc; ql, qh = qr_pred(params["qr"], B["F"]["m2"][cidx]); e = np.maximum(ql - z, z - qh)   # CQR scores
    n = len(e); return {"Q": float(np.quantile(e, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher"))}


def bounds_from_cal(name, params, B, idx, cal):
    """(lo, hi) log-return bounds for rows idx from fitted params and a calibration summary."""
    params = revive_model(name, params); r = B["r"][H]
    if name in ("M0", "M1", "M2", "M3"):
        sg = sigma_of(name, params, B, idx); return cal["lo"] * sg, cal["hi"] * sg
    if name == "M5":
        return aci_bounds(np.asarray(cal["z_sorted"]), sigma_of(name, params, B, idx), r[idx], B["D"].ta[idx])
    sg = L3.sigma(params["har"], B, idx, None); ql, qh = qr_pred(params["qr"], B["F"]["m2"][idx])
    lo = (ql - cal["Q"]) * sg; hi = np.maximum((qh + cal["Q"]) * sg, lo); return lo, hi


def month_bounds(name, params, B, idx, cidx):
    """(lo, hi) for rows idx; quantiles/corrections come from calibration rows cidx (the previous month) only."""
    return bounds_from_cal(name, params, B, idx, cal_summary(name, params, B, cidx))


def score_bounds(r, lo, hi):
    a = ALPHA; isc = (hi - lo) + (2 / a) * (lo - r) * (r < lo) + (2 / a) * (r - hi) * (r > hi)
    inn = (r >= lo) & (r <= hi)
    return {"n": int(len(r)), "n_in": int(inn.sum()), "cover": float(inn.mean()), "width_bps": float(np.mean(hi - lo) * 1e4), "iscore_bps": float(np.mean(isc) * 1e4)}


def walk_forward(B, months, names, log=None):
    """Monthly walk-forward. For each month: fit on the 365 days before it (labels ended by the month start), calibrate on the previous month, score."""
    rows_out = []
    for mo in pd.PeriodIndex(months, freq="M"):
        m0 = mo.start_time; tr = L3.train_rows(B, H, m0); cidx = L3.cal_rows(B, H, mo); idx, n_all = L3.month_rows(B, H, mo)
        if len(tr) < 500 or len(cidx) < 50 or len(idx) < 50:
            raise RuntimeError(f"{mo}: train {len(tr)} cal {len(cidx)} test {len(idx)}")
        days = float((m0 - B["D"].ta[tr].min()) / pd.Timedelta(days=1))
        for nm in names:
            p = fit_model(nm, B, tr); lo, hi = month_bounds(nm, p, B, idx, cidx)
            d = score_bounds(B["r"][H][idx], lo, hi)
            d.update({"month": str(mo), "model": nm, "n_train": int(len(tr)), "n_cal": int(len(cidx)), "n_dropped": n_all - len(idx), "train_days": days})
            rows_out.append(d)
        if log:
            log(f"{mo} train {len(tr)} ({days:.0f} d) cal {len(cidx)} test {len(idx)}")
    return pd.DataFrame(rows_out)


def summarise(df):
    """per model: mean monthly interval score, pooled coverage, monthly coverage sd."""
    out = {}
    for nm, d in df.groupby("model"):
        out[nm] = {"iscore_bps": float(d.iscore_bps.mean()), "pooled_cover": float(d.n_in.sum() / d.n.sum()), "cover_sd": float(d.cover.std(ddof=1)),
                   "width_bps": float(d.width_bps.mean()), "n_months": int(len(d))}
    return out


# ---------------------------------------------------------------- freeze (Look B)
def freeze(B, names, cal_month):
    """Fit `names` on every row whose label ended by the start of the month after cal_month; store them and cal_month's calibration. JSON-ready."""
    mo = pd.Period(cal_month, "M"); m1 = (mo + 1).start_time; tr = L3.train_rows(B, H, m1, days=3650); cidx = L3.cal_rows(B, H, mo + 1)
    out = {"cal_month": cal_month, "n_train": int(len(tr)), "n_cal": int(len(cidx)), "train_first": str(B["D"].ta[tr].min()),
           "train_last_label_end": str(B["D"].ta[tr].max() + pd.Timedelta(minutes=H)), "qr_stride": QR_STRIDE, "models": {}, "cal": {}}
    for nm in names:
        p = fit_model(nm, B, tr); out["models"][nm] = _jsonable(p); out["cal"][nm] = cal_summary(nm, p, B, cidx)
    return out


def frozen_bounds(name, fm, B, idx, cidx=None):
    """Frozen parameters of `fm` (a freeze() dict). cidx None: the stored calibration (September 2026); else recalibrated on cidx (the previous month)."""
    p = fm["models"][name]
    return bounds_from_cal(name, p, B, idx, fm["cal"][name] if cidx is None else cal_summary(name, p, B, cidx))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def revive(p):
    """JSON lists back to arrays for the T03 HAR model; ridge/QR models use lists directly."""
    if isinstance(p, dict) and {"mu", "sd", "coef", "ic", "c", "plus"} <= set(p):
        q = dict(p)
        for k in ("mu", "sd", "coef"):
            q[k] = np.asarray(p[k])
        return q
    return p


def revive_model(name, p):
    if name in ("M0", "M5"):
        return revive(p)
    if name == "M4":
        return {"har": revive(p["har"]), "qr": p["qr"]}
    return p


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

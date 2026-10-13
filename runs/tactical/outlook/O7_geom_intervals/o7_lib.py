"""O7 (SPEC.md): geometry-enhanced volatility forecasts as 90% return intervals (T03's `cal` method). CPU only.
Reuses O5/O6 (frame definition, ridge, walk-forward rows), O1 (G), O3 (rows, ridge) and T03's interval score definitions. Anchor a sees bars < a only.

Forecasters (sigma_hat), fitted per month on the 365 days before it (labels ending by the month start):
  L1  = ridge of ln RV(a, a+T) on S3 + ln bank + G1 + G15      (O5's candidate = O6's L1);  sigma = exp(pred / 2)
  R0  = ridge on S3 only (same target);                                                       sigma = exp(pred / 2)
  HAR = ridge of ln(|r_T| + 1e-5) on ln RV 15/60/240/1440 + hour sin/cos (T03's form);       sigma = exp(pred)
Target of the intervals: r_T = ln c[a+T-1] - ln c[a-1], the SIGNED log return over bars a .. a+T-1, i.e. exactly the window of the RV target
(RV = sum of squared 1-minute log returns of bars a .. a+T-1; their sum is r_T). Intervals: [q05, q95] of r_T / sigma_hat on the previous
month's rows (labels ending by the scored month start) times sigma_hat, forecaster = the fit of that previous month (no row of the scored month).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OUTLOOK, "O6_geom_hourly"))
import o6_lib as P6  # noqa: E402  (imports o5_lib and removes O5's folder from sys.path again, as O6 does)
sys.path.insert(0, HERE)
P5 = P6.P5; O1 = P6.O1; O3 = P6.O3; T = P6.T
V = T.V
HORIZONS = P6.HORIZONS
SCRATCH = "D:/nt/nt_scratch/o7"
FRAME = P6.SCRATCH + "/frame.npz"                             # O6's cache: L1 == O5's L1 (O6 SPEC / test)
LOAD_FROM = P6.LOAD_FROM; LOAD_HI = T.CUTOFF                  # whole cut file; no bar after 2025-07-20 23:59
ALPHA = 0.10
EPS_Y = 1e-5
N_S3 = P5.N_S3
N_L1G = P6.N_G1G15                                            # 328 geometry columns of L1
MODELS = ("L1", "R0", "HAR")
J_SCORED = [str(m) for m in pd.period_range("2019-02", "2020-12", freq="M")]        # 23 months; 2019-01 calibrates 2019-02 only
S_SPENT1 = [str(m) for m in pd.period_range("2021-02", "2024-06", freq="M")]
S_SPENT2 = [str(m) for m in pd.period_range("2024-08", "2025-07", freq="M")]
SPANS = {"judge": J_SCORED, "spent1": S_SPENT1, "spent2": S_SPENT2}
COVER_BAND = (0.88, 0.92)
NEED_BETTER = 15


def prev_month(m):
    return str(pd.Period(m, freq="M") - 1)


def month_bounds(m):
    return P5.month_bounds(m)


# ------------------------------------------------------------------ the extra columns: signed returns and HAR features
def signed_returns(A, anchors, H):
    """r_T = ln c[a+H-1] - ln c[a-1] (A: OHLCV array with close in column 3)."""
    a = np.asarray(anchors); c = np.log(A[:, 3]); return c[a + H - 1] - c[a - 1]


def har_features(s, first):
    """V.Data HAR block [n, 6] (ln RV 15/60/240/1440, hour sin/cos) and the signed returns, over the bars s, anchors as O3.build_features (same first)."""
    D = V.Data(s.values.astype(np.float64), s.index, stride=T.STRIDE, first=first, with_tb7=False)
    return D, {"har": D.har.copy(), **{f"r{H}": signed_returns(D.A, D.anch, H) for H in HORIZONS}}


def load_extra():
    """Cache D:/nt/nt_scratch/o7/extra.npz: anch, ta, har, r15/r60/r240 for O6's frame anchors (built from the same bars; asserted equal)."""
    os.makedirs(SCRATCH, exist_ok=True); p = os.path.join(SCRATCH, "extra.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    s = T.load_bars(T.TRAIN_CSV, LOAD_FROM, LOAD_HI)
    assert s.index.max() <= LOAD_HI <= T.CUTOFF
    D, ex = har_features(s, P6.FIRST6)
    out = {"anch": D.anch, "ta": D.ta.values.astype("datetime64[ns]"), **ex}
    np.savez(p, **out); return out


def load_all():
    z = np.load(FRAME); F = {k: z[k] for k in ("ta", "anch", "XB", "G", *[f"rv{H}" for H in HORIZONS], *[f"ok{H}" for H in HORIZONS])}
    F["G"] = F["G"][:, :N_L1G]                               # L1 = XB + G1 + G15 (the first 328 columns); G60 is O6's
    ex = load_extra()
    assert np.array_equal(ex["anch"], F["anch"]) and np.array_equal(ex["ta"], F["ta"]), "extra columns not on the frame's anchors"
    F["har"] = ex["har"]
    for H in HORIZONS:
        F[f"r{H}"] = ex[f"r{H}"]
    return F


def build_frame_new(s, first):
    """Same columns for new bars (the prospective scoring): XB, G (G1 + G15), har, r_T, rv/ok per horizon."""
    F3 = O3.build_features(s, first); D, ex = har_features(s, first)
    assert np.array_equal(D.anch, F3["anch"])
    g1, g15 = O1.window_features(F3["A"], F3["anch"])
    out = {"ta": F3["ta"], "anch": F3["anch"], "XB": np.concatenate([F3["S3"], O3.log_bank(F3["bank"])], 1),
           "G": np.concatenate([g1, g15], 1).astype(np.float32), "har": ex["har"]}
    for H in HORIZONS:
        out[f"rv{H}"] = F3[f"rv{H}"]; out[f"ok{H}"] = F3[f"ok{H}"]; out[f"r{H}"] = ex[f"r{H}"]
    return out


# ------------------------------------------------------------------ design matrices
def getX(F, rows, name):
    X = F["XB"][rows]
    if name == "L1":
        return np.concatenate([X, F["G"][rows].astype(np.float64)], 1)
    if name == "L0":
        return X
    if name == "R0":
        return X[:, :N_S3]
    if name == "HAR":
        return F["har"][rows]
    raise KeyError(name)


def target(F, H, rows, name):
    return np.log(np.abs(F[f"r{H}"][rows]) + EPS_Y) if name == "HAR" else np.log(F[f"rv{H}"][rows])


def sigma_of(name, mu):
    """sigma_hat from the ridge prediction (constant scales cancel in r / sigma_hat)."""
    return np.exp(mu if name == "HAR" else 0.5 * mu)


# ------------------------------------------------------------------ rows of one fit month f (calibration month f, scored month f+1)
def month_rows(F, H, f):
    """tr: training rows of the fit for month f (365 days, labels end by f's start); cal: rows of month f with labels ending by the start of
    month f+1; te: rows of month f+1. All with the ok mask."""
    m0, m1 = month_bounds(f); n0, n1 = month_bounds(str(pd.Period(f, freq="M") + 1))
    ta = F["ta"]; ok = F[f"ok{H}"]
    tr = O3.train_rows(ta, ok, H, m0)
    inm = (ta >= np.datetime64(m0, "ns")) & (ta < np.datetime64(m1, "ns")) & (ta + np.timedelta64(H, "m") <= np.datetime64(n0, "ns")) & ok
    cal = np.where(inm)[0]; te = O3.test_rows(ta, ok, n0, n1)
    return tr, cal, te


def fit(F, H, tr, names):
    return {nm: O3.ridge_fit(getX(F, tr, nm), target(F, H, tr, nm)) for nm in names}


# ------------------------------------------------------------------ intervals and scores
def interval_bounds(z_cal):
    return float(np.quantile(z_cal, ALPHA / 2)), float(np.quantile(z_cal, 1 - ALPHA / 2))


def interval_stats(r, sg, q):
    """coverage, width and Gneiting-Raftery interval score (alpha 0.1) of [q_lo, q_hi] * sg for returns r. Widths / scores in bps."""
    lo, hi = q[0] * sg, q[1] * sg; a = ALPHA
    isc = (hi - lo) + (2 / a) * (lo - r) * (r < lo) + (2 / a) * (r - hi) * (r > hi)
    inn = (r >= lo) & (r <= hi)
    return {"n": int(len(r)), "n_in": int(inn.sum()), "cover": float(inn.mean()), "width_bps": float(np.mean(hi - lo) * 1e4),
            "iscore_bps": float(np.mean(isc) * 1e4)}


def score_fit_month(F, H, f, names=MODELS, fitted=None, rows=None):
    """Scored month f+1 for every forecaster: the fit of f predicts both the calibration rows (month f) and the scored rows."""
    tr, cal, te = rows or month_rows(F, H, f); fitted = fitted or fit(F, H, tr, names); out = {}
    for nm in names:
        sg_c = sigma_of(nm, O3.ridge_pred(fitted[nm], getX(F, cal, nm))); sg_t = sigma_of(nm, O3.ridge_pred(fitted[nm], getX(F, te, nm)))
        q = interval_bounds(F[f"r{H}"][cal] / sg_c)
        d = interval_stats(F[f"r{H}"][te], sg_t, q); d.update({"model": nm, "H": H, "n_cal": int(len(cal)), "n_tr": int(len(tr)), "q_lo": q[0], "q_hi": q[1]})
        out[nm] = d
    return out


# ------------------------------------------------------------------ statistics
def paired(df, a, b, col, months, H, min_n=3):
    """a - b per scored month: mean, 95% t interval, two-sided p, months with a better (smaller) than b, months total."""
    d0 = df[df.H == H]
    pa = d0[d0.model == a].set_index("month")[col].astype(float); pb = d0[d0.model == b].set_index("month")[col].astype(float)
    d = (pa - pb).reindex(months).dropna()
    if len(d) < min_n:
        return None
    mu, lo, hi, _ = T.ci(d.values)
    se = d.std(ddof=1) / np.sqrt(len(d)); t = mu / se if se > 0 else 0.0
    return {"mean": mu, "lo": lo, "hi": hi, "p": float(2 * stats.t.sf(abs(t), len(d) - 1)), "better": int((d < 0).sum()), "n": int(len(d)),
            "min": float(d.min()), "max": float(d.max())}


def pooled_cover(df, model, months, H):
    d = df[(df.H == H) & (df.model == model) & df.month.isin(months)]
    return float(d.n_in.sum() / d.n.sum()) if len(d) else None


def criterion(pr, cover, need=NEED_BETTER):
    """SPEC: T = 60, L1 - R0 interval score: mean < 0, 95% t interval entirely below 0, L1 better in at least `need` months; pooled L1 coverage in [0.88, 0.92]."""
    c1 = pr["mean"] < 0 and pr["hi"] < 0 and pr["better"] >= need; c2 = COVER_BAND[0] <= cover <= COVER_BAND[1]
    return {"c1_score_below_0_ci_below_0_months": bool(c1), "need_better": int(need), "better": pr["better"], "n": pr["n"],
            "c2_pooled_cover_in_band": bool(c2), "pooled_cover": cover, "pass": bool(c1 and c2)}


holm = P5.holm
sha_of = P5.sha_of

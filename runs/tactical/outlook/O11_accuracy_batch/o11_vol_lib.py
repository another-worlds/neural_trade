"""O11 parts a, b, c (SPEC.md): shared library. CPU only. Anchor a sees bars < a only.

Reuses, unchanged and by import: O7's frame loader (O6's cache of the cut file + O7's HAR / signed-return extras), O5's L1 design and ridge fit
(through o5_lib / o3_lib), T01's targets (T.build), O7's interval method (month_rows, interval_bounds).

Frames (dicts of equal-length arrays):
  cut frame  : O6's frame.npz sliced to L1 (XB 52 + G 328) + O7's extra.npz (har, r_T) + O11's extra_a.npz (T01's dA, vB, ok mask, AR regressors);
               anchors 2017-08-17 + 6145 bars .. 2025-07-20, all features from the cut file only.
  fwd frame  : the Binance file 2025-10-01 .. 2026-10-02 built on its own with O3's warm-up of 6144 bars (the bank needs it), so the first
               anchor is 2025-10-05 06:24 and 2025-10 is a partial month (the convention of O3, O5 and O7). No bar of 2025-07-21..2025-09-30 is read.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); OUTLOOK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OUTLOOK, "O7_geom_intervals"))
import o7_lib as P7  # noqa: E402
P6 = P7.P6; P5 = P7.P5; O1 = P7.O1; O3 = P7.O3; T = P7.T
SCR = "D:/nt/nt_scratch/o11"
N_XB = P5.N_B; N_S3 = P5.N_S3; N_G = P7.N_L1G                  # 52, 16, 328
N_G1 = N_G // 2
S_MONTHS = [str(m) for m in pd.period_range("2021-01", "2024-06", freq="M")]
B_MONTHS = [str(m) for m in pd.period_range("2024-07", "2025-07", freq="M")]     # 2025-07 ends at the cut (07-20)
C_MONTHS = list(T.FWD_MONTHS)                                                    # 2025-10 .. 2026-09
FORBID_LO = np.datetime64(T.FORBID_LO, "ns"); FORBID_HI = np.datetime64(T.FORBID_HI, "ns")
TRAIN_END = T.CUTOFF + pd.Timedelta(minutes=1)
GAP_DAYS = int((T.FWD_START - pd.Timestamp("2025-07-21 00:00")).total_seconds() // 86400)    # 72 days with no data between the files
HORIZONS = P7.HORIZONS


def month_bounds(m):
    return P5.month_bounds(m)


# ------------------------------------------------------------------ statistics
def series_stats(x):
    """Monthly series -> mean, 95% t interval, two-sided one-sample t p, months positive, n, min, max."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 3:
        return None
    mu, lo, hi, t = T.ci(x)
    return {"mean": mu, "lo": lo, "hi": hi, "p": float(2 * stats.t.sf(abs(t), n - 1)), "pos": int((x > 0).sum()), "neg": int((x < 0).sum()),
            "n": int(n), "min": float(x.min()), "max": float(x.max())}


def fmt(d, nd=4):
    return "n/a" if d is None else (f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}] +{d['pos']}/{d['n']} p={d['p']:.4g} "
                                    f"(min {d['min']:+.{nd}f}, max {d['max']:+.{nd}f})")


# ------------------------------------------------------------------ frames and caches
def extra_a_from_bars(s, first):
    """T01's targets on the anchors of the bar frame s: dA (next-hour minus last-hour ln RV), vB (4-hour vol of vol), T01's ok mask, AR regressors."""
    B = T.build(s, first)
    return {"anch": B["D"].anch, "ta": B["D"].ta.values.astype("datetime64[ns]"), "dA": B["T"]["A"], "vB": B["T"]["B"], "okx": B["ok"],
            "arA": B["AR"]["A"], "arB": B["AR"]["B"], "S3": B["F"]["S3"].astype(np.float64)}


def _finish(F):
    """derived columns: finite flag and the a-ok mask (T01's mask and every L1 feature finite)."""
    fin = np.isfinite(F["XB"]).all(1) & np.isfinite(F["G"]).all(1)
    F["oka"] = F["okx"] & fin
    return F


def load_cut():
    os.makedirs(SCR, exist_ok=True); p = os.path.join(SCR, "extra_a.npz")
    F = P7.load_all()
    if os.path.exists(p):
        z = np.load(p); ex = {k: z[k] for k in z.files}
    else:
        s = T.load_bars(T.TRAIN_CSV, P7.LOAD_FROM, P7.LOAD_HI)
        assert s.index.max() <= T.CUTOFF
        ex = extra_a_from_bars(s, P6.FIRST6); del s; np.savez(p, **ex)
    assert np.array_equal(ex["anch"], F["anch"]) and np.array_equal(ex["ta"], F["ta"]), "extra_a not on the frame's anchors"
    assert np.allclose(ex["S3"], F["XB"][:, :N_S3], rtol=1e-9, atol=1e-12), "T01's S3 differs from the frame's S3"
    for k in ("dA", "vB", "okx", "arA", "arB"):
        F[k] = ex[k]
    assert pd.Timestamp(F["ta"].max()) <= T.CUTOFF + pd.Timedelta(minutes=1)
    return _finish(F)


def build_fwd(s=None, first=O3.FIRST):
    """The forward frame from Binance bars (the file 2025-10-01 .. 2026-10-02 by default)."""
    s = T.load_fwd() if s is None else s
    F = P7.build_frame_new(s, first); ex = extra_a_from_bars(s, first)
    assert np.array_equal(ex["anch"], F["anch"])
    for k in ("dA", "vB", "okx", "arA", "arB"):
        F[k] = ex[k]
    return _finish(F)


def load_fwd_frame():
    os.makedirs(SCR, exist_ok=True); p = os.path.join(SCR, "fwd_frame.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    F = build_fwd(); np.savez(p, **F); return F


def combine(F1, F2):
    """Time-ordered concatenation of two frames (cut file then forward file); asserts order and that no anchor lies in the forbidden gap."""
    keys = ["ta", "XB", "G", "har", "oka", *[f"rv{H}" for H in HORIZONS], *[f"ok{H}" for H in HORIZONS], *[f"r{H}" for H in HORIZONS]]
    out = {k: np.concatenate([F1[k], F2[k]]) for k in keys}
    assert (np.diff(out["ta"].astype("int64")) > 0).all()
    assert not ((out["ta"] >= FORBID_LO) & (out["ta"] <= FORBID_HI)).any(), "an anchor inside 2025-07-21..2025-09-30"
    return out


# ------------------------------------------------------------------ rows
def train_rows_between(ta, ok, H, lo, m0):
    """rows with lo <= ta and label end (ta + H) <= m0 and ok (lo = m0 - 365 days is O3.train_rows)."""
    return np.where((ta >= np.datetime64(lo, "ns")) & (ta <= np.datetime64(m0 - pd.Timedelta(minutes=H), "ns")) & ok)[0]


def avail_lo(m0, days=365):
    """The start of the most recent `days` days of AVAILABLE data before m0: the calendar start m0 - days, moved back by the 72-day gap when it lies
    before the Binance file starts (the cut file ends 2025-07-20, the Binance file begins 2025-10-01)."""
    lo = m0 - pd.Timedelta(days=days)
    return lo if lo >= T.FWD_START else lo - pd.Timedelta(days=GAP_DAYS)


def check_rows_clean(F, rows, H, m0):
    """No training row in the forbidden gap and every label ends by m0 (the part-b guard)."""
    ta = F["ta"][rows]
    assert not ((ta >= FORBID_LO) & (ta <= FORBID_HI)).any()
    assert (ta + np.timedelta64(H, "m") <= np.datetime64(m0, "ns")).all()


# ------------------------------------------------------------------ part a: the volatility-change forecast
TG = {"A": ("dA", 60, "arA"), "B": ("vB", 240, "arB")}


def design_a(F, rows, name):
    X = F["XB"][rows]
    if name == "bank":                                        # secondary decomposition: S3 + the EWMA bank, no geometry
        return X
    return X[:, :N_S3] if name == "base" else np.concatenate([X, F["G"][rows].astype(np.float64)], 1)


def fit_a(F, tg, m0):
    """base (S3) and candidate (L1 features) ridge + the AR(1) of T01 for one target, on the 365 days before m0 (labels ending by m0)."""
    y_name, H, ar = TG[tg]; tr = O3.train_rows(F["ta"], F["oka"], H, m0)
    assert len(tr) > 1000
    check_rows_clean(F, tr, H, m0)
    y = F[y_name][tr]; st = {"tg": tg, "n_tr": int(len(tr)), "ar": T.ols1(F[ar][tr], y), "m": {}}
    for nm in ("base", "cand", "bank"):
        st["m"][nm] = O3.ridge_fit(design_a(F, tr, nm), y)
    return st


def score_a(st, F, te, span, month):
    y_name, H, ar = TG[st["tg"]]; y = F[y_name][te]; rows = []
    p_ar = st["ar"][0] + st["ar"][1] * F[ar][te]; mse_ar = float(np.mean((y - p_ar) ** 2))
    for nm in ("base", "cand", "bank"):
        p = O3.ridge_pred(st["m"][nm], design_a(F, te, nm)); mse = float(np.mean((y - p) ** 2))
        rows.append({"span": span, "month": month, "tg": st["tg"], "model": nm, "n_te": int(len(te)), "n_tr": st["n_tr"], "mse": mse, "mse_ar": mse_ar,
                     "skill": 1 - mse / mse_ar, "spearman": float(spearmanr(p, y)[0]) if np.std(p) > 0 else None})
    return rows


# ------------------------------------------------------------------ part b: ridge for ln RV, frozen against refit
def fit_ridge_h(F, H, tr, models):
    """Ridge (alpha 1) of ln RV(a, a+H) on the design of each model (o5_lib.getX names S3 / L1), refit on all of tr; the QLIKE scale c from the
    first-85% fit on the hold rows (gap ceil(H/7)); exactly o5_lib.fit_h's ridge branch for the rows tr."""
    fit, hold = O3.split_fit_hold(tr, H); k = len(fit); fi = np.arange(k); hi = np.arange(len(tr) - len(hold), len(tr))
    assert np.array_equal(tr[fi], fit) and np.array_equal(tr[hi], hold)
    y = np.log(F[f"rv{H}"][tr]); v = F[f"rv{H}"][tr]
    st = {"H": H, "n_tr": int(len(tr)), "m": {}, "c": {}, "first_ta": str(pd.Timestamp(F["ta"][tr].min())),
          "last_label_end": str(pd.Timestamp(F["ta"][tr].max()) + pd.Timedelta(minutes=H))}
    for nm in models:
        X = P5.getX(F, tr, nm); mf = O3.ridge_fit(X[fi], y[fi]); st["m"][nm] = O3.ridge_fit(X, y)
        st["c"][nm] = O3.qlike_c(v[hi], O3.ridge_pred(mf, X[hi])); del X
    return st


def score_b(st, F, te, arm, month):
    H = st["H"]; v = F[f"rv{H}"][te]; rows = []
    for nm in st["m"]:
        mu = O3.ridge_pred(st["m"][nm], P5.getX(F, te, nm))
        rows.append({"arm": arm, "month": month, "H": H, "model": nm, "n_te": int(len(te)), "n_tr": st["n_tr"],
                     "spearman": float(spearmanr(mu, v)[0]) if np.std(mu) > 0 else None, "qlike": O3.qlike(v, mu, st["c"][nm]),
                     "first_ta": st["first_ta"], "last_label_end": st["last_label_end"]})
    return rows


# ------------------------------------------------------------------ part c: tails of the 60-minute return
Q_LO, Q_HI = 0.05, 0.95
QR_ALPHA = 1e-4; QR_STRIDE = 10                       # every 10th stride-7 row = a stride-70 subsample
TOPK = 15
H_C = 60


def pinball(r, q, tau):
    d = r - q; return np.maximum(tau * d, (tau - 1) * d)


def tail_row(r, qlo, qhi):
    """counts and pinball losses (bps) of the 5% / 95% quantile forecasts."""
    p5 = float(np.mean(pinball(r, qlo, Q_LO)) * 1e4); p95 = float(np.mean(pinball(r, qhi, Q_HI)) * 1e4)
    return {"n": int(len(r)), "n_lo": int((r < qlo).sum()), "n_hi": int((r < qhi).sum()), "pin05_bps": p5, "pin95_bps": p95, "pin_bps": 0.5 * (p5 + p95)}


def month_rows_c(F, f):
    """cal: rows of month f whose label ends by the start of month f+1; te: rows of month f+1 (ok mask of H = 60)."""
    m0, m1 = month_bounds(f); n0, n1 = month_bounds(str(pd.Period(f, freq="M") + 1)); ta = F["ta"]; ok = F[f"ok{H_C}"]
    cal = np.where((ta >= np.datetime64(m0, "ns")) & (ta < np.datetime64(m1, "ns")) & (ta + np.timedelta64(H_C, "m") <= np.datetime64(n0, "ns")) & ok)[0]
    te = O3.test_rows(ta, ok, n0, n1)
    return cal, te


def l1_fit(F, tr):
    """L1 ridge of ln RV60 (O7's forecaster)."""
    return O3.ridge_fit(P7.getX(F, tr, "L1"), P7.target(F, H_C, tr, "L1"))


def sigma_l1(m, F, rows):
    return P7.sigma_of("L1", O3.ridge_pred(m, P7.getX(F, rows, "L1")))


def top_g1(m):
    """The TOPK G1 columns (indices into G) by absolute standardised ridge coefficient of the L1 fit (training data only)."""
    w = np.abs(m["w"][N_XB:N_XB + N_G1]); return np.sort(np.argsort(-w, kind="stable")[:TOPK])


def qr_features(F, rows, m, g1_idx):
    """[ln sigma_hat, G15 (164), G1 top-15, hour sin, hour cos] for the rows (sigma_hat from the L1 fit m)."""
    G = F["G"][rows].astype(np.float64)
    return np.column_stack([0.5 * O3.ridge_pred(m, P7.getX(F, rows, "L1")), G[:, N_G1:], G[:, g1_idx], F["har"][rows][:, 4:6]])


def fit_qr(F, tr, m, g1_idx=None):
    """Quantile regressions (5% and 95%) of r60 on the standardised QR features, on every 10th training row; sklearn QuantileRegressor(alpha 1e-4, highs)."""
    from sklearn.linear_model import QuantileRegressor
    g1_idx = top_g1(m) if g1_idx is None else g1_idx; sub = tr[::QR_STRIDE]
    X = qr_features(F, sub, m, g1_idx); mu = X.mean(0); sd = X.std(0) + 1e-12; Xs = (X - mu) / sd; y = F[f"r{H_C}"][sub]
    q = {tau: QuantileRegressor(quantile=tau, alpha=QR_ALPHA, solver="highs").fit(Xs, y) for tau in (Q_LO, Q_HI)}
    return {"g1_idx": g1_idx, "mu": mu, "sd": sd, "q": q, "n_sub": int(len(sub)), "nz": {tau: int((np.abs(q[tau].coef_) > 1e-12).sum()) for tau in q}}


def qr_predict(qm, F, rows, m):
    Xs = (qr_features(F, rows, m, qm["g1_idx"]) - qm["mu"]) / qm["sd"]
    return qm["q"][Q_LO].predict(Xs), qm["q"][Q_HI].predict(Xs)


def base_quantiles(F, m_f, cal, te):
    """O7's interval method: empirical 5% / 95% quantiles of r / sigma_hat on the calibration month, times sigma_hat of the scored rows."""
    z = F[f"r{H_C}"][cal] / sigma_l1(m_f, F, cal); ql, qh = np.quantile(z, Q_LO), np.quantile(z, Q_HI)
    sg = sigma_l1(m_f, F, te); return ql * sg, qh * sg

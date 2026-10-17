"""T14 candidate M4: the regime card (chart image + templated caption + numeric analogue). Run with D:/nt/nt_mm_env/python.exe.

Image (R1): the last 240 one-minute bars strictly before the anchor `ta` (UTC), drawn as a 224x224 RGB image with numpy only:
  rows 0..167  candles (green up / red down, wick = high-low, body = open-close) in LOG price, the axis scaled by the window's own range
               (min low .. max high, 4% padding), so the picture is scale free;
  rows 172..223 volume panel, bars scaled by the window's maximum volume;
  thin lines, all causal, from T13's sr_lib on the same window (pair 1 minute x 240 bars):
     S1 hull lines (yellow) through the last bar's high / low with the extreme slope to the earlier points, extended one bar to "now";
     S2 nearest unbroken fractal pivot above / below the last close (cyan), from the pivot to "now";
     S3 OLS channel of close +- 2 residual sd (lavender), centre line dim.
  Bar i of the window is drawn at column round(i * 222 / 239); "now" (one bar after the last bar) is column 223.
Caption (R2): a template over numbers known at `ta` (see caption_frame).
Numeric analogue (M4_num): the last 60 one-minute OHLCV bars, per window scale free, standardised, PCA 16.
Reduction: PCA 16, fitted ONLY on the cut-file ok anchors of 2023-07-01 .. 2024-06-30 (before Look 1; unsupervised), applied to both sources.
"""
import ctypes
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
for p in (os.path.join(TASK, "core"), os.path.join(TASK, "M6_faiss"), os.path.join(os.path.dirname(TASK), "T13_support_lines")):
    if p not in sys.path:
        sys.path.insert(0, p)
import sr_lib as S  # noqa: E402
import m6_lib as M6  # noqa: E402

W = 224
NB = 240                    # bars in the image window
PH = 168                    # price rows
VOL0 = 172                  # first volume row
NUM_BARS = 60               # window of the numeric analogue
PCA_K = 16
PCA_LO, PCA_HI = pd.Timestamp("2023-07-01 00:00"), pd.Timestamp("2024-06-30 23:59")
UP, DOWN = (38, 166, 91), (214, 69, 65)
C_S1, C_S2, C_S3, C_S3M = (250, 200, 40), (60, 200, 230), (170, 150, 255), (90, 80, 130)
VUP, VDN = (30, 100, 60), (120, 45, 45)
PCT_WIN, PCT_MIN = 30 * 24, 7 * 24


def below_normal():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


# ------------------------------------------------------------------ geometry (batch, from sr_lib)
def geometry(h, l, c, a, sigma):
    """Line geometry of the windows of NB one-minute bars ending at a-1, in log price, for the anchors positions `a` (all must fit). Returns a dict of
    arrays [n]: sR, sS (hull slopes per bar), lhm, llm (log high / low of the last bar), pivR, pivS (log level), piR, piS (window index of the pivot),
    missR, missS, b, ybar, s (OLS of close: slope per bar, mean, residual sd). Uses only bars < a."""
    a = np.asarray(a); lh, ll, lc = S._windows(np.asarray(h, float), np.asarray(l, float), np.asarray(c, float), a, 1, NB, None, None)
    m = NB - 1; x = np.arange(NB); c0 = lc[:, m]; sigma = np.asarray(sigma, float)
    dx = (m - x[:m])[None, :].astype(float)
    sR = ((lh[:, m:m + 1] - lh[:, :m]) / dx).min(1); sS = ((ll[:, m:m + 1] - ll[:, :m]) / dx).max(1)
    LR, mR, _, _, aR = S._pivot_side(lh, c0, sigma, x, m, True); LS, mS, _, _, aS = S._pivot_side(ll, c0, sigma, x, m, False)
    xm = x.mean(); xc = x - xm; sxx = float((xc ** 2).sum()); ybar = lc.mean(1); b = (lc * xc[None, :]).sum(1) / sxx
    res = lc - (ybar[:, None] + b[:, None] * xc[None, :]); s = np.sqrt((res ** 2).sum(1) / (NB - 2))
    return dict(lh=lh, ll=ll, lc=lc, sR=sR, sS=sS, lhm=lh[:, m], llm=ll[:, m], c0=c0, pivR=LR, pivS=LS, piR=np.rint(m - aR * (m + 1)).astype(int),
                piS=np.rint(m - aS * (m + 1)).astype(int), missR=mR, missS=mS, b=b, ybar=ybar, s=s, xm=xm)


def _col(i):
    return np.rint(np.asarray(i, float) * (W - 2) / (NB - 1)).astype(int)


_COLS = np.arange(W)
_T = np.where(_COLS < W - 1, _COLS * (NB - 1) / (W - 2), float(NB))          # window index (float) drawn at each column; the last column is "now"


def _put(img, y_rows, color, valid=None):
    ok = (y_rows >= 0) & (y_rows < PH)
    if valid is not None:
        ok &= valid
    cols = _COLS[ok]
    img[y_rows[ok], cols] = color


def render_one(o, hi, lo, c, v, g, k):
    """one 224x224x3 uint8 image from the raw window arrays (length NB, linear price) and row k of the geometry dict `g`."""
    img = np.zeros((W, W, 3), np.uint8)
    lh, ll = g["lh"][k], g["ll"][k]; top, bot = lh.max(), ll.min(); pad = 0.04 * (top - bot + 1e-12); top += pad; bot -= pad
    ypx = lambda lp: np.rint((PH - 1) * (top - lp) / (top - bot)).astype(int)
    lo_o, lo_c = np.log(o), np.log(c)
    r_h, r_l, r_o, r_c = ypx(lh), ypx(ll), ypx(lo_o), ypx(lo_c)
    cols = _col(np.arange(NB)); vmax = float(v.max()) + 1e-12; vh = np.rint((W - 1 - VOL0) * v / vmax).astype(int)
    for i in range(NB):
        x = cols[i]; upc = c[i] >= o[i]
        img[r_h[i]:r_l[i] + 1, x] = UP if upc else DOWN
        a_, b_ = (r_o[i], r_c[i]) if r_o[i] <= r_c[i] else (r_c[i], r_o[i])
        img[a_:b_ + 1, x] = UP if upc else DOWN
        if vh[i] > 0:
            img[W - vh[i]:W, x] = VUP if upc else VDN
    # S1 hull lines
    tt = _T; m = NB - 1
    _put(img, ypx(g["lhm"][k] - g["sR"][k] * (m - tt)), C_S1)
    _put(img, ypx(g["llm"][k] - g["sS"][k] * (m - tt)), C_S1)
    # S2 pivot levels, from the pivot to now
    if not g["missR"][k]:
        _put(img, np.full(W, ypx(g["pivR"][k])), C_S2, valid=_COLS >= _col(g["piR"][k]))
    if not g["missS"][k]:
        _put(img, np.full(W, ypx(g["pivS"][k])), C_S2, valid=_COLS >= _col(g["piS"][k]))
    # S3 channel
    f = g["ybar"][k] + g["b"][k] * (tt - g["xm"])
    _put(img, ypx(f), C_S3M); _put(img, ypx(f + 2 * g["s"][k]), C_S3); _put(img, ypx(f - 2 * g["s"][k]), C_S3)
    return img


def render_images(o, h, l, c, v, a, sigma, chunk=2000):
    """generator of (index range, uint8 [n, 224, 224, 3]) for anchors at 1-minute array positions `a` (windows [a-NB, a))."""
    a = np.asarray(a); w = a[:, None] - NB + np.arange(NB)
    for i in range(0, len(a), chunk):
        sl = slice(i, i + chunk); g = geometry(h, l, c, a[sl], sigma[sl]); ww = w[sl]
        yield sl, np.stack([render_one(o[ww[k]], h[ww[k]], l[ww[k]], c[ww[k]], v[ww[k]], g, k) for k in range(len(ww))])


# ------------------------------------------------------------------ numeric analogue
def num_window(o, h, l, c, v, a):
    """[n, 300]: the last NUM_BARS bars' log O, H, L, C minus the window's mean log close, divided by the window's range (max log high - min log low),
    and volume divided by the window's maximum. Bars < a only."""
    a = np.asarray(a); w = a[:, None] - NUM_BARS + np.arange(NUM_BARS)
    O, H, L, C, V = (np.log(o[w]), np.log(h[w]), np.log(l[w]), np.log(c[w]), v[w].astype(float))
    mu = C.mean(1, keepdims=True); rng = (H.max(1) - L.min(1))[:, None] + 1e-12
    return np.hstack([(O - mu) / rng, (H - mu) / rng, (L - mu) / rng, (C - mu) / rng, V / (V.max(1, keepdims=True) + 1e-12)])


# ------------------------------------------------------------------ captions
def _pct_series(x):
    return M6.trailing_pct(x, PCT_WIN, PCT_MIN)


def caption_frame(m1, ta, sigma, srcols):
    """DataFrame (index ta) of the numbers a caption uses, from bars strictly before each ta: vp (percentile of the last hour's realised variance among the
    preceding 30 days of hourly values, min 7 days), bbp (percentile of the 20 five-minute-bar Bollinger width of log close, same window), z20 (ln close /
    EMA20 of the 1-minute closes of the last bar, in sigma), hour, wkend, dR, dS, mR, mS (T13 S2 distances above / below close in sigma, missing flags).
    `sigma` [len(ta)] and `srcols` {'s2_dR':..., 's2_dS':..., 's2_missR':..., 's2_missS':...} [len(ta)] arrays supplied by the caller (past-only)."""
    ta = pd.DatetimeIndex(ta); h1 = pd.Timedelta(hours=1)
    _, lrv = M6.hourly_logrv(m1)
    ph = pd.Series(_pct_series(lrv.values), index=lrv.index)
    vp = ph.reindex(ta - h1).to_numpy(float)
    g5 = m1["close"].astype(float).resample("5min"); c5 = g5.last()[(g5.count() == 5).values]
    c5 = np.log(c5).reindex(pd.date_range(c5.index.min(), c5.index.max(), freq="5min"))
    bbw = 4.0 * c5.rolling(20, min_periods=20).std()
    hours = pd.date_range(ta.min().floor("h") - 400 * h1, ta.max(), freq="h")           # hourly grid long enough for the percentile window
    bb_at = pd.Series(bbw.reindex(hours - pd.Timedelta(minutes=5)).to_numpy(float), index=hours)       # the 5-minute bin that closes at the anchor
    bbp = pd.Series(_pct_series(bb_at.values), index=hours).reindex(ta).to_numpy(float)
    idx = m1.index; pos = idx.searchsorted(ta, side="left"); pl = np.maximum(pos - 1, 0)
    last_ok = (pos >= 1) & np.asarray(idx[pl] == (ta - pd.Timedelta(minutes=1)))
    cc = m1["close"].astype(float).values; e20 = pd.Series(cc).ewm(span=20, adjust=False).mean().values
    with np.errstate(all="ignore"):
        z20 = np.where(last_ok, np.log(cc[pl] / e20[pl]) / np.asarray(sigma, float), np.nan)
    df = pd.DataFrame({"vp": vp, "bbp": bbp, "z20": z20, "hour": ta.hour.values, "wkend": (ta.dayofweek.values >= 5).astype(float),
                       "dR": srcols["s2_dR"], "dS": srcols["s2_dS"], "mR": srcols["s2_missR"], "mS": srcols["s2_missS"]}, index=ta)
    return df


def _session(h):
    return "Asia session" if h < 7 else "Europe session" if h < 13 else "US open" if h < 15 else "US session" if h < 21 else "late US"


def _vol_label(p):
    return "very low" if p < .2 else "low" if p < .4 else "normal" if p < .6 else "elevated" if p < .8 else "high"


def caption(r):
    """the template, from one row of caption_frame; None when a number is missing."""
    if not all(np.isfinite([r.vp, r.bbp, r.z20])):
        return None
    sq = "squeeze" if r.bbp <= .2 else "wide bands" if r.bbp >= .8 else "normal bands"
    st = "price near EMA20" if abs(r.z20) < .5 else f"price stretched {abs(r.z20):.1f} sigma {'above' if r.z20 > 0 else 'below'} EMA20"
    res = "no pivot resistance" if r.mR else f"{r.dR:.1f} sigma below nearest resistance"
    sup = "no pivot support" if r.mS else f"{r.dS:.1f} sigma above nearest support"
    return (f"volatility {_vol_label(r.vp)} (p{int(round(r.vp * 100))}), {sq}, {st}, {int(r.hour):02d}:00 UTC {_session(int(r.hour))}, "
            f"{'weekend' if r.wkend else 'weekday'}, {res}, {sup}")


# ------------------------------------------------------------------ PCA
def fit_pca(X, ta, ok, k=PCA_K, lo=PCA_LO, hi=PCA_HI):
    """PCA fitted on the rows with ok, finite and lo <= ta <= hi only (the caller passes the CUT source). Returns (mean, components [k, d], explained ratio)."""
    ta = pd.DatetimeIndex(ta); sel = np.asarray((ta >= lo) & (ta <= hi)) & np.asarray(ok) & np.isfinite(X).all(1)
    assert sel.sum() > 1000, "too few anchors to fit the PCA"
    Z = X[sel].astype(np.float64); mu = Z.mean(0); U, s, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    sgn = np.sign(Vt[np.arange(k), np.abs(Vt[:k]).argmax(1)]); sgn[sgn == 0] = 1
    return mu, Vt[:k] * sgn[:, None], (s[:k] ** 2) / (s ** 2).sum(), int(sel.sum())


def apply_pca(X, mu, comps):
    return (X - mu) @ comps.T


# ------------------------------------------------------------------ monthly walk-forward GBM (M7's rule; image PCs -> one prediction)
POOL_DAYS, LABEL_MIN, MIN_POOL = 365, 60, 5000


def gbm_params(seed=0):
    return dict(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=200, subsample=0.8, subsample_freq=1,
                random_state=seed, n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)


def monthly_gbm(ta, X, y, ok, min_pool=MIN_POOL, log=None):
    """For each calendar month m: LightGBM regressor fitted on the pooled anchors j (ok) with ta_j >= m_start - 365 d and ta_j + 60 min <= m_start,
    predicting the anchors of month m. Honest out-of-month predictions for every served row; NaN elsewhere. Returns (pred, info)."""
    import lightgbm as lgb
    ta = pd.DatetimeIndex(ta); pred = np.full(len(ta), np.nan); info = []; per = np.asarray(ta.to_period("M")); end = ta + pd.Timedelta(minutes=LABEL_MIN)
    for mo in sorted(set(per[ok])):
        m0 = mo.start_time; te = np.where(ok & (per == mo))[0]
        tr = np.where(ok & np.asarray(ta >= m0 - pd.Timedelta(days=POOL_DAYS)) & np.asarray(end <= m0))[0]
        if len(tr) < min_pool:
            info.append({"month": str(mo), "served": False, "n_pool": int(len(tr)), "n_te": int(len(te))}); continue
        mdl = lgb.LGBMRegressor(**gbm_params()).fit(X[tr], y[tr]); pred[te] = mdl.predict(X[te])
        info.append({"month": str(mo), "served": True, "n_pool": int(len(tr)), "n_te": int(len(te))})
        if log:
            log(f"    {mo} pool {len(tr)} te {len(te)}")
    return pred, info

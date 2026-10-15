"""T13 (SPEC.md): the SR feature block, support and resistance lines built from bars strictly before the anchor, in sigma units. CPU only, numpy.

Anchor at array position a uses the window of `nb` bars of `bar` minutes that END at a-1 (the groups of a bar > 1 are aligned to the anchor, as O1's
agg15). Everything is in log price. "Now" is the position one bar after the window's last bar. Distances are divided by sigma, the sd of the T-minute
return (T = 60), so the features are scale free. The three lines (SPEC):

  S1 hull lines     upper hull of the highs, lower hull of the lows; the edge active at the LAST bar is the line through the last point with the extreme
                    slope to the earlier points (upper: the minimum slope; lower: the maximum), extended one bar to "now".
  S2 pivot levels   fractal swing highs / lows (k = 3 bars each side), confirmed only k bars later (a pivot at window index i exists only if i + k <= last
                    index). The nearest UNBROKEN pivot above and below close (no later bar's high / low went through it), with the touch count within
                    0.1 sigma, the bars since the last touch and the pivot's age.
  S3 channel        OLS line of close over the window, +- 2 residual sd.

32 columns (SR_NAMES). Clipping (fixed here, nothing tuned): distances, widths, slopes and z to [-10, 10] sigma, residual sd to [0, 10]. A missing pivot
gets the distance 10, no touch, bars-since 1, age 1 and a missing flag."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

CAP = 10.0
K_PIV = 3
TOL = 0.1                      # touch tolerance, in sigma
T_H = 60                       # the target horizon in minutes the sigma and the slopes refer to
S1_NAMES = ["s1_dR", "s1_dS", "s1_width", "s1_slopeR", "s1_slopeS", "s1_touchR", "s1_touchS", "s1_bsR", "s1_bsS", "s1_inside"]
S2_NAMES = ["s2_dR", "s2_dS", "s2_width", "s2_touchR", "s2_touchS", "s2_bsR", "s2_bsS", "s2_ageR", "s2_ageS", "s2_missR", "s2_missS"]
S3_NAMES = ["s3_dU", "s3_dL", "s3_width", "s3_slope", "s3_sres", "s3_z", "s3_touchU", "s3_touchL", "s3_bsU", "s3_bsL", "s3_inside"]
SR_NAMES = S1_NAMES + S2_NAMES + S3_NAMES
N_SR = len(SR_NAMES)
PAIRS = ((1, 60), (1, 240), (15, 60), (15, 240))       # (bar minutes, lookback bars): the four candidates of the selection


def need_bars(bar, nb):
    """one-minute bars before the anchor the window reads"""
    return bar * nb


def sigma_blend(S3):
    """sd of a 60-minute return from the first four S3 columns (log per-bar variance of the last 15 / 60 / 240 / 1440 bars): the geometric mean of the four
    variances times T. A fixed, unfitted HAR-type blend used for parts B and C (their bases have no HAR); part A uses the fitted HAR of T03."""
    return np.sqrt(T_H * np.exp(np.asarray(S3)[:, :4].mean(1)))


def clean_windows(tmin, bar, nb):
    """boolean per bar position a: the need_bars(bar, nb) bars before a exist (a >= need) and have no time gap (consecutive minutes). tmin = bar open time in
    minutes (int64) of the whole series. Returned as a function of the anchor positions."""
    need = need_bars(bar, nb); gaps = np.concatenate([[0], np.cumsum(np.diff(tmin) != 1)])        # gaps[i] = gaps between bars 0..i

    def ok(a):
        a = np.asarray(a)
        return (a >= need) & (gaps[np.maximum(a - 1, 0)] - gaps[np.maximum(a - need, 0)] == 0)
    return ok


def _windows(h, l, c, a, bar, nb, rmax, rmin):
    a = np.asarray(a)
    if bar == 1:
        w = a[:, None] - nb + np.arange(nb)
        return np.log(h[w]), np.log(l[w]), np.log(c[w])
    st = a[:, None] - bar * nb + bar * np.arange(nb)
    return np.log(rmax[st]), np.log(rmin[st]), np.log(c[st + bar - 1])


def _touch_stats(near, x, m, lo_idx=None):
    """touch count and bars since the last touch (as a fraction of the window), near [n, nb] boolean; only bars at or after lo_idx[:, None] count."""
    if lo_idx is not None:
        near = near & (x[None, :] >= lo_idx[:, None])
    cnt = near.sum(1).astype(float)
    last = np.where(near, x[None, :], -1).max(1)
    bs = np.where(last >= 0, (m - last) / (m + 1.0), 1.0)
    return cnt, bs


def _pivot_side(P, c0, sig, x, m, high):
    """nearest unbroken pivot above (high=True: pivot highs, P = log highs) or below (P = log lows) close. Returns level, miss, touches, bars since, age."""
    n, nb = P.shape; k = K_PIV
    cen = P[:, k:nb - k]
    if high:
        left = np.max(np.stack([P[:, k - s:nb - k - s] for s in range(1, k + 1)]), 0); right = np.max(np.stack([P[:, k + s:nb - k + s] for s in range(1, k + 1)]), 0)
        piv = (cen >= left) & (cen > right)
        rc = np.maximum.accumulate(P[:, ::-1], 1)[:, ::-1]; after = rc[:, k + 1:nb - k + 1]
        unb = after <= cen
        dist = np.where(piv & unb, cen - c0[:, None], np.inf)
    else:
        left = np.min(np.stack([P[:, k - s:nb - k - s] for s in range(1, k + 1)]), 0); right = np.min(np.stack([P[:, k + s:nb - k + s] for s in range(1, k + 1)]), 0)
        piv = (cen <= left) & (cen < right)
        rc = np.minimum.accumulate(P[:, ::-1], 1)[:, ::-1]; after = rc[:, k + 1:nb - k + 1]
        unb = after >= cen
        dist = np.where(piv & unb, c0[:, None] - cen, np.inf)
    j = dist.argmin(1); has = np.isfinite(dist[np.arange(n), j]); pi = j + k
    level = P[np.arange(n), pi]
    near = np.abs(P - level[:, None]) <= (TOL * sig)[:, None] + 1e-12
    cnt, bs = _touch_stats(near, x, m, pi)
    age = (m - pi) / (m + 1.0)
    return level, ~has, np.where(has, cnt, 0.0), np.where(has, bs, 1.0), np.where(has, age, 1.0)


def _sr_chunk(lh, ll, lc, sig, bar):
    n, nb = lc.shape; m = nb - 1; x = np.arange(nb); c0 = lc[:, m]; tol = TOL * sig + 1e-12; per = T_H / bar
    cl = lambda v: np.clip(v, -CAP, CAP)
    cols = {}
    # ---- S1: the hull edges active at the last bar
    dx = (m - x[:m])[None, :].astype(float)
    sR = ((lh[:, m:m + 1] - lh[:, :m]) / dx).min(1); sS = ((ll[:, m:m + 1] - ll[:, :m]) / dx).max(1)
    Rn = lh[:, m] + sR; Sn = ll[:, m] + sS
    lineR = lh[:, m:m + 1] - sR[:, None] * (m - x)[None, :]; lineS = ll[:, m:m + 1] - sS[:, None] * (m - x)[None, :]
    tR, _ = _touch_stats(np.abs(lh - lineR) <= tol[:, None] + 1e-12, x, m)
    tS, _ = _touch_stats(np.abs(ll - lineS) <= tol[:, None] + 1e-12, x, m)
    # bars since the last touch BEFORE the last bar (the last bar always touches its own line)
    _, bsR = _touch_stats((np.abs(lh - lineR) <= tol[:, None] + 1e-12)[:, :m], x[:m], m)
    _, bsS = _touch_stats((np.abs(ll - lineS) <= tol[:, None] + 1e-12)[:, :m], x[:m], m)
    cols.update(s1_dR=cl((Rn - c0) / sig), s1_dS=cl((c0 - Sn) / sig), s1_width=cl((Rn - Sn) / sig), s1_slopeR=cl(sR * per / sig), s1_slopeS=cl(sS * per / sig),
                s1_touchR=tR, s1_touchS=tS, s1_bsR=bsR, s1_bsS=bsS, s1_inside=((c0 <= Rn) & (c0 >= Sn)).astype(float))
    # ---- S2: pivots
    LR, mR, tcR, bR, aR = _pivot_side(lh, c0, sig, x, m, True); LS, mS, tcS, bS, aS = _pivot_side(ll, c0, sig, x, m, False)
    dR = np.where(mR, CAP, (LR - c0) / sig); dS = np.where(mS, CAP, (c0 - LS) / sig)
    wd = np.where(mR | mS, CAP, (LR - LS) / sig)
    cols.update(s2_dR=cl(dR), s2_dS=cl(dS), s2_width=cl(wd), s2_touchR=tcR, s2_touchS=tcS, s2_bsR=bR, s2_bsS=bS, s2_ageR=aR, s2_ageS=aS,
                s2_missR=mR.astype(float), s2_missS=mS.astype(float))
    # ---- S3: regression channel of close
    xm = x.mean(); xc = x - xm; sxx = float((xc ** 2).sum()); ybar = lc.mean(1); b = (lc * xc[None, :]).sum(1) / sxx
    f = ybar[:, None] + b[:, None] * xc[None, :]; res = lc - f; s = np.sqrt((res ** 2).sum(1) / max(nb - 2, 1))
    fn = ybar + b * (nb - xm); U = fn + 2 * s; Lw = fn - 2 * s
    tU, bU = _touch_stats(np.abs(lc - (f + 2 * s[:, None])) <= tol[:, None], x, m); tL, bL = _touch_stats(np.abs(lc - (f - 2 * s[:, None])) <= tol[:, None], x, m)
    cols.update(s3_dU=cl((U - c0) / sig), s3_dL=cl((c0 - Lw) / sig), s3_width=cl(4 * s / sig), s3_slope=cl(b * per / sig), s3_sres=np.clip(s / sig, 0, CAP),
                s3_z=cl((c0 - fn) / np.maximum(s, 1e-12)), s3_touchU=tU, s3_touchL=tL, s3_bsU=bU, s3_bsL=bL, s3_inside=((c0 <= U) & (c0 >= Lw)).astype(float))
    return np.stack([cols[nm] for nm in SR_NAMES], 1)


def sr_block(h, l, c, a, sigma, bar, nb, chunk=1500):
    """SR features [len(a), 32] for the anchors a (positions into the one-minute arrays h, l, c; bars < a only) with sigma [len(a)] (sd of the 60-minute
    return, positive). Rows whose window does not fit (a < bar * nb) are returned as zeros; the caller masks them (clean_windows)."""
    h = np.asarray(h, float); l = np.asarray(l, float); c = np.asarray(c, float); a = np.asarray(a); sigma = np.asarray(sigma, float)
    out = np.zeros((len(a), N_SR)); need = need_bars(bar, nb)
    rmax = swv(h, bar).max(1) if bar > 1 else None; rmin = swv(l, bar).min(1) if bar > 1 else None
    good = np.where((a >= need) & np.isfinite(sigma) & (sigma > 0))[0]
    for i in range(0, len(good), chunk):
        g = good[i:i + chunk]; lh, ll, lc = _windows(h, l, c, a[g], bar, nb, rmax, rmin)
        out[g] = _sr_chunk(lh, ll, lc, sigma[g], bar)
    return out

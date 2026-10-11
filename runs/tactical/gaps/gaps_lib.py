"""gaps study helpers (SPEC.md): the cutoff loader, process priority, break detection (dynamic programming, BIC, sup-Chow, block bootstrap)."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TAC, "shape")); sys.path.insert(0, os.path.join(TAC, "volnet")); sys.path.insert(0, os.path.join(TAC, "lab"))
import shapelib  # noqa: E402

CUTOFF = shapelib.CUTOFF


def nice():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def load_bars(path=None, cutoff=None):
    """OHLCV frame with NO bar after the cutoff (shapelib's chunked loader) plus a hard assert."""
    s = shapelib.load_bars(path, cutoff)
    assert s.index.max() <= (CUTOFF if cutoff is None else pd.Timestamp(cutoff))
    return s


# ------------------------------------------------------------------ breaks in the mean (Bai-Perron style, partial structural change = mean shift)
def best_partitions(y, kmax=5, h=6):
    """Globally optimal piecewise-constant fits by dynamic programming. Returns {k: (rss, [break indices])}; a break index b means a new
    segment starts at position b (0-based), so segments are [0,b1), [b1,b2), ..., each of length >= h."""
    y = np.asarray(y, float); T = len(y); c1 = np.concatenate([[0.0], np.cumsum(y)]); c2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(i, j):                                           # RSS of y[i:j] around its mean
        n = j - i; s = c1[j] - c1[i]; return (c2[j] - c2[i]) - s * s / n
    INF = np.inf
    # dp[k][j] = best RSS of y[:j] using k+1 segments
    dp = np.full((kmax + 1, T + 1), INF); arg = np.zeros((kmax + 1, T + 1), int)
    for j in range(h, T + 1):
        dp[0][j] = cost(0, j)
    for k in range(1, kmax + 1):
        for j in range((k + 1) * h, T + 1):
            lo, hi = k * h, j - h
            best, bi = INF, -1
            for i in range(lo, hi + 1):
                v = dp[k - 1][i] + cost(i, j)
                if v < best:
                    best, bi = v, i
            dp[k][j] = best; arg[k][j] = bi
    out = {}
    for k in range(kmax + 1):
        if not np.isfinite(dp[k][T]):
            continue
        br, j = [], T
        for kk in range(k, 0, -1):
            i = arg[kk][j]; br.append(int(i)); j = i
        out[k] = (float(dp[k][T]), sorted(br))
    return out


def bic(rss, T, npar):
    return T * np.log(max(rss, 1e-300) / T) + npar * np.log(T)


def trend_fit(y):
    y = np.asarray(y, float); t = np.arange(len(y)); X = np.column_stack([np.ones(len(y)), t]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    f = X @ b; return f, float(((y - f) ** 2).sum()), b


def piece_fit(y, br):
    y = np.asarray(y, float); f = np.empty(len(y)); edges = [0] + list(br) + [len(y)]
    for a, b in zip(edges[:-1], edges[1:]):
        f[a:b] = y[a:b].mean()
    return f


def select_k(y, kmax=5, h=6):
    """BIC over M_0..M_kmax (2k+1 parameters) and over M_trend (3 parameters incl. sigma is the same for all, so 2)."""
    T = len(y); parts = best_partitions(y, kmax, h)
    tab = {k: bic(r, T, 2 * k + 1) for k, (r, _) in parts.items()}
    kstar = min(tab, key=tab.get); _, rss_t, _ = trend_fit(y)
    return {"kstar": kstar, "bic_k": tab, "bic_trend": bic(rss_t, T, 2), "breaks": parts[kstar][1], "parts": parts, "rss_trend": rss_t}


def sup_chow(y, trim=0.15):
    """sup over tau of the F statistic for a break in intercept and slope (2 restrictions); returns (supF, tau index)."""
    y = np.asarray(y, float); T = len(y); t = np.arange(T); X = np.column_stack([np.ones(T), t]); best, bt = -np.inf, None
    rss_r = float(((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum())
    for tau in range(int(np.ceil(trim * T)), int(np.floor((1 - trim) * T)) + 1):
        r = 0.0
        for sl in (slice(0, tau), slice(tau, T)):
            Xi = X[sl]; yi = y[sl]; r += float(((yi - Xi @ np.linalg.lstsq(Xi, yi, rcond=None)[0]) ** 2).sum())
        F = ((rss_r - r) / 2) / (r / (T - 4))
        if F > best:
            best, bt = F, tau
    return float(best), int(bt)


def block_resample(res, block, rng):
    T = len(res); nb = int(np.ceil(T / block)); starts = rng.integers(0, T - block + 1, nb)
    return np.concatenate([res[s:s + block] for s in starts])[:T]


def chow_pvalue(y, nboot=2000, block=6, seed=0, trim=0.15):
    y = np.asarray(y, float); obs, tau = sup_chow(y, trim); f, _, _ = trend_fit(y); res = y - f; rng = np.random.default_rng(seed); cnt = 0
    for _ in range(nboot):
        yb = f + block_resample(res, block, rng)
        if sup_chow(yb, trim)[0] >= obs:
            cnt += 1
    return obs, tau, (cnt + 1) / (nboot + 1)


def break_intervals(y, kstar, breaks, nboot=1000, block=6, seed=1, kmax=None, h=6):
    """moving-block bootstrap of the residuals around the fitted M_k*; re-estimates k* breaks each time."""
    y = np.asarray(y, float); f = piece_fit(y, breaks); res = y - f; rng = np.random.default_rng(seed); draws = []
    if kstar == 0:
        return []
    for _ in range(nboot):
        yb = f + block_resample(res, block, rng)
        parts = best_partitions(yb, kstar, h)
        if kstar in parts:
            draws.append(parts[kstar][1])
    d = np.array(draws)
    return [(float(np.percentile(d[:, i], 2.5)), float(np.percentile(d[:, i], 97.5))) for i in range(kstar)]

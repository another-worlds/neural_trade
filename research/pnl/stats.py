"""Inference for the PnL study (SPEC 'objective' INFERENCE and POWER, 'selection_and_preregistration' P_SEL and DEV GATE,
'baselines' B7).

- stationary_bootstrap / paired_bootstrap: Politis-Romano stationary block bootstrap of a statistic of a daily series
  (B = 10,000; mean block 10 days for Track 1, 20 for Track 2), vectorised over replications in bounded-memory chunks.
  percentile_ci, lower_bound (G2: one-sided 95% lower bound of the mean daily net > 0), upper_bound, boot_pvalue and
  prob_positive read the distribution. Per-trade CIs (INFERENCE iii) are the same bootstrap on the per-trade net series
  with mean_block=20.
- newey_west_t: HAC t of a mean, Bartlett kernel (lag 10 or 20 days). Reported; the bootstrap decides.
- p_sel: selection-adjusted circular-shift test (B7 random-timing null, max over K candidates) on engine-F daily net
  PnL of the realised 1-min exposure; engine_f_daily is the plain float64 reference of the same PnL.
- deflated_sharpe: Bailey and Lopez de Prado (2014). Reported, not a gate.
- holm: Holm step-down adjusted p-values.
- min_detectable_mean, power_mean, min_detectable_sharpe, power_sharpe: one-sided normal power helpers (SPEC POWER).
Conventions: daily series hold one value per UTC day, in log-return units; sharpe_ann annualises with sqrt(365); every
Sharpe passed to deflated_sharpe is per period (per day), in the same units as T.
"""
import time

import numpy as np
from scipy.stats import norm

YEAR_DAYS = 365
C_IN, C_OUT = 3e-4, 5e-4              # engine F: cost per unit increase / decrease of |x|
EULER_GAMMA = 0.5772156649015329
_CHUNK = 1000                          # bootstrap replications per chunk (fixed, so a seed always gives the same draws)


# ----------------------------------------------------------------------------------------------- bootstrap statistics

def sharpe_ann(x, axis=-1):
    """Annualised Sharpe of daily log returns, mean/sd(ddof=1) * sqrt(365); 0 where sd = 0. Same as metrics.sharpe."""
    x = np.asarray(x, np.float64)
    m, sd = np.asarray(x.mean(axis)), np.asarray(x.std(axis, ddof=1))
    return np.divide(m, sd, out=np.zeros_like(m), where=sd > 0) * np.sqrt(YEAR_DAYS)


def max_drawdown(x, axis=-1):
    """Maximum drawdown (fraction of the running peak) of equity exp(cumsum x), starting at 1 before the first day.
    Vectorised along `axis`, so it can be a bootstrap statistic. Same as metrics.max_drawdown."""
    x = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    le = np.cumsum(x, -1)
    le = np.concatenate([np.zeros(le.shape[:-1] + (1,)), le], -1)
    return 1.0 - np.exp((le - np.maximum.accumulate(le, -1)).min(-1))


# --------------------------------------------------------------------------------------- stationary block bootstrap

def _sb_indices(rng, b, T, p):
    """b x T stationary-bootstrap index rows: a new block starts with probability p (always at position 0) at a
    uniform day; inside a block the index advances by one day, wrapping circularly."""
    new = rng.random((b, T)) < p
    new[:, 0] = True
    bid = np.cumsum(new.ravel()) - 1                       # block id of every cell
    start = rng.integers(0, T, size=int(bid[-1]) + 1)      # one uniform start per block
    pos = np.arange(T)
    last = np.where(new, pos, 0)
    np.maximum.accumulate(last, axis=1, out=last)           # position where the current block began
    idx = start[bid].reshape(b, T) + (pos - last)
    idx %= T
    return idx


def _apply(stat, X):
    """stat over the rows of X: stat(X, axis=1) when stat is vectorised (np.mean, sharpe_ann, ...), else row by row."""
    try:
        out = np.asarray(stat(X, axis=1), np.float64)
    except TypeError:
        out = np.asarray(np.apply_along_axis(stat, 1, X), np.float64)
    if out.shape != (len(X),):
        raise ValueError("stat must return one value per resampled series")
    return out


def _bootstrap(series, B, mean_block, seed, fn):
    T = len(series[0])
    if any(len(s) != T for s in series):
        raise ValueError("paired series must be aligned (same length)")
    if T < 2 or mean_block < 1:
        raise ValueError("need T >= 2 and mean_block >= 1")
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for a in range(0, B, _CHUNK):
        b = min(_CHUNK, B - a)
        idx = _sb_indices(rng, b, T, 1.0 / mean_block)
        out[a:a + b] = fn([s[idx] for s in series])
    return out


def stationary_bootstrap(x, B=10000, mean_block=10, seed=0, stat=np.mean):
    """Politis-Romano (1994) stationary bootstrap distribution (B values) of stat(x) for a daily series x.
    Blocks have geometric length with mean `mean_block` and wrap circularly. stat is called as stat(X, axis=1) on a
    chunk of resampled rows (falls back to one row at a time). Memory is bounded by chunks of 1,000 replications
    (about 60 MB at T = 1,800)."""
    x = np.asarray(x, np.float64)
    return _bootstrap([x], B, mean_block, seed, lambda s: _apply(stat, s[0]))


def paired_bootstrap(x, y, B=10000, mean_block=10, seed=0, stat=np.mean):
    """Bootstrap distribution of stat(x*) - stat(y*), with x and y resampled on the SAME days (PAIRED differences
    against a baseline). For stat=np.mean this is the mean of the daily difference x - y."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return _bootstrap([x, y], B, mean_block, seed, lambda s: _apply(stat, s[0]) - _apply(stat, s[1]))


def percentile_ci(dist, level=0.95):
    """Two-sided percentile interval (lo, hi) of a bootstrap distribution."""
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(dist, [a, 1.0 - a])
    return float(lo), float(hi)


def lower_bound(dist, level=0.95):
    """One-sided percentile lower bound: the (1 - level) quantile. G2 asks this to be > 0 for the mean daily net."""
    return float(np.quantile(dist, 1.0 - level))


def upper_bound(dist, level=0.95):
    """One-sided percentile upper bound: the `level` quantile (K1 closure: upper bound of net per round trip < 0)."""
    return float(np.quantile(dist, level))


def boot_pvalue(dist):
    """One-sided percentile-bootstrap p-value of H0: stat <= 0, the share of the distribution at or below 0.
    p < 1 - level is the same event as lower_bound(dist, level) > 0 (up to quantile interpolation)."""
    return float(np.mean(np.asarray(dist) <= 0))


def prob_positive(dist):
    """Share of a paired-bootstrap distribution above 0: the reported P(candidate > baseline)."""
    return float(np.mean(np.asarray(dist) > 0))


# ------------------------------------------------------------------------------------------------------- HAC t

def newey_west_se(x, lag=10, axis=-1):
    """Newey-West standard error of the mean along `axis`: sqrt(S/T), S = g0 + 2 sum_{l<=lag} (1 - l/(lag+1)) g_l,
    g_l = (1/T) sum_t e_t e_{t-l} with e the de-meaned series (Bartlett kernel, no small-sample correction)."""
    x = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    T = x.shape[-1]
    e = x - x.mean(-1, keepdims=True)
    s = np.einsum("...t,...t->...", e, e) / T
    for l in range(1, min(int(lag), T - 1) + 1):
        s = s + 2.0 * (1.0 - l / (lag + 1.0)) * np.einsum("...t,...t->...", e[..., l:], e[..., :-l]) / T
    return np.sqrt(np.maximum(s, 0.0) / T)


def _ratio(m, se):
    """m/se, with 0/0 -> 0 and m/0 -> +-inf (a constant series)."""
    m, se = np.asarray(m, np.float64), np.asarray(se, np.float64)
    out = np.where(m == 0, 0.0, np.copysign(np.inf, m))
    return np.divide(m, se, out=out, where=se > 0)


def newey_west_t(x, lag=10, axis=-1):
    """HAC t of the mean: mean / newey_west_se. Lag 10 days (Track 1) or 20 (Track 2)."""
    x = np.asarray(x, np.float64)
    return _ratio(x.mean(axis), newey_west_se(x, lag, axis))


def plain_t(x, axis=-1):
    """mean / (sd(ddof=1)/sqrt(T)) = per-period Sharpe x sqrt(T)."""
    x = np.asarray(x, np.float64)
    T = x.shape[axis]
    return _ratio(x.mean(axis), x.std(axis, ddof=1) / np.sqrt(T))


# ------------------------------------------------------------------------------------------- engine F and p_sel

def _bar_cost(x, xl, c_in, c_out):
    """Engine-F cost of moving from xl = x_{t-1} to x = x_t: c_in*inc + c_out*red, with inc = (|x|+|dx|-|xl|)/2 and
    red = (|xl|+|dx|-|x|)/2, exact including flips (a flip +1 -> -1 is red 1 + inc 1)."""
    a, al, d = np.abs(x), np.abs(xl), np.abs(x - xl)
    return c_in * 0.5 * (a + d - al) + c_out * 0.5 * (al + d - a)


def _lag(x):
    """x_{t-1} along the last axis, with x_{-1} = 0 (the run starts flat)."""
    xl = np.zeros_like(x)
    xl[..., 1:] = x[..., :-1]
    return xl


def engine_f_daily(x, r, f, day, c_in=C_IN, c_out=C_OUT, shift=0):
    """Engine-F daily net PnL (K x D, float64; D = distinct day ids in order) of exposure paths x (K x n, or n) on a
    1-min grid, with returns and funding circularly shifted by `shift` bars (np.roll semantics):
        net_t = x_{t-1} (exp(r_{t-s}) - 1) - x_{t-1} f_{t-s} - cost(x_{t-1} -> x_t),   x_{-1} = 0,
    summed per day id. Costs do not depend on the shift. Plain reference implementation (slow); p_sel uses a fast one.
    """
    x = np.atleast_2d(np.asarray(x, np.float64))
    g = np.roll(np.expm1(np.asarray(r, np.float64)) - np.asarray(f, np.float64), int(shift))
    xl = _lag(x)
    net = xl * g - _bar_cost(x, xl, c_in, c_out)
    _, inv = np.unique(np.asarray(day), return_inverse=True)
    return np.stack([np.bincount(inv, weights=row) for row in net])


class _DayGrid:
    """Maps n bars with non-decreasing day ids onto a D x M grid (M = most bars in a day; short days zero-padded)."""

    def __init__(self, day):
        day = np.asarray(day)
        n = len(day)
        if n == 0 or np.any(day[1:] < day[:-1]):
            raise ValueError("day ids must be non-empty and non-decreasing")
        self.starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
        counts = np.diff(np.r_[self.starts, n])
        self.n, self.D, self.M = n, len(self.starts), int(counts.max())
        self.ids = day[self.starts]
        self.regular = bool(np.all(counts == self.M))
        if not self.regular:
            self.flat = np.repeat(np.arange(self.D) * self.M - self.starts, counts) + np.arange(n)
            self._buf = None

    def to_grid(self, v, out):
        """Write a length-n row into a D x M (possibly strided) view."""
        if self.regular:
            out[...] = v.reshape(self.D, self.M)
        else:
            buf = np.zeros(self.D * self.M, out.dtype)
            buf[self.flat] = v
            out[...] = buf.reshape(self.D, self.M)

    def shifted(self, g_ext, s):
        """D x M grid of g rolled by s bars, from g_ext = [g, g]: a zero-copy view on a regular grid."""
        n = self.n
        v = g_ext[n - s:2 * n - s]
        if self.regular:
            return v.reshape(self.D, self.M)
        if self._buf is None:
            self._buf = np.zeros(self.D * self.M, g_ext.dtype)   # pad cells are never written, so they stay 0
        self._buf[self.flat] = v
        return self._buf.reshape(self.D, self.M)


def p_sel(X, r, f, day, c_in=C_IN, c_out=C_OUT, n_shifts=999, min_shift=30 * 1440, seed=0, tstat="hac", lag=10,
          dtype=np.float32, daily_cost=None):
    """Selection-adjusted skill test (SPEC P_SEL; B7 random-timing null), one per track.

    X: K x n realised engine-E exposure on the 1-min grid (float32 is fine); r: n 1-min log returns; f: n funding
    rates per bar (a long pays a positive rate); day: n non-decreasing day ids (e.g. t // 86400).
    Daily net PnL is engine F (see engine_f_daily). For b = 1..n_shifts, s_b ~ U{min_shift, ..., n - min_shift} bars
    (30 days in minutes by default; seeded; the same s_b for every candidate); r and f are rolled by s_b against the
    fixed positions; each candidate's daily net is recomputed and studentised; M_b = max_k t_k^b. The observed max uses
    the unshifted series. p_sel = (1 + #{M_b >= observed max}) / (n_shifts + 1), i.e. /1000 at 999 shifts.

    tstat: 'hac' (default; mean / Newey-West SE with `lag` days, the SPEC's studentised form) or 'plain'
    (mean / (sd/sqrt T) = daily Sharpe x sqrt T). A constant-zero candidate (cash) gets t = 0.
    daily_cost: optional K x D cost per day replacing the engine-F cost (e.g. engine-E's realised fees). By default
    the cost is engine F on the per-bar exposure, which also charges price drift of x between fills as trading.
    Costs are shift-invariant and computed once in float64.

    Speed: positions are laid out once as a D x K x M grid of x_{t-1}; g = exp(r) - 1 - f is stored twice end to end,
    so the rolled series is a zero-copy D x M view, and each shift is one batched gemv (D products of K x M by M),
    O(nK) multiply-adds in `dtype`. Measured single-threaded (float32, synthetic data): n = 2.54M, K = 7, 999 shifts
    in 3.5 s including setup (about 3 ms per shift), about 120 MB above the inputs; K = 8 in 4.0 s. The float32 daily
    PnL matches the float64 reference to < 0.001 bps per day.

    Returns a dict: p_sel, t_obs (K), t_max_obs, best (argmax k), p_single (K; per-candidate shift p-values, not
    selection-adjusted), null_max (n_shifts), null_t (n_shifts x K), shifts, daily_obs (K x D net per day),
    day_ids (D), seconds, and the settings.
    """
    t0 = time.perf_counter()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None]
    K, n = X.shape
    r, f = np.asarray(r, np.float64), np.asarray(f, np.float64)
    if r.shape != (n,) or f.shape != (n,) or len(day) != n:
        raise ValueError("X must be K x n and r, f, day must have length n")
    if not (0 < min_shift and n - min_shift > min_shift):
        raise ValueError("need n > 2 * min_shift")
    if tstat not in ("hac", "plain"):
        raise ValueError("tstat must be 'hac' or 'plain'")
    grid = _DayGrid(day)
    D, M = grid.D, grid.M

    # shift-invariant parts: daily cost (float64) and the D x K x M grid of lagged positions
    if daily_cost is None:
        cost = np.empty((K, D))
        for k in range(K):
            xk = X[k].astype(np.float64)
            cost[k] = np.add.reduceat(_bar_cost(xk, _lag(xk), c_in, c_out), grid.starts)
    else:
        cost = np.asarray(daily_cost, np.float64)
        if cost.shape != (K, D):
            raise ValueError("daily_cost must be K x D")
    Xg = np.zeros((D, K, M), dtype)
    for k in range(K):
        grid.to_grid(_lag(np.asarray(X[k], dtype)), Xg[:, k, :])
    g = (np.expm1(r) - f).astype(dtype)
    g_ext = np.concatenate([g, g])
    del g

    def tfun(Y):
        return newey_west_t(Y, lag) if tstat == "hac" else plain_t(Y)

    def daily(s):
        G = grid.shifted(g_ext, s)
        return np.matmul(Xg, G[:, :, None])[:, :, 0].T.astype(np.float64) - cost      # K x D

    daily_obs = daily(0)
    t_obs = tfun(daily_obs)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, n - min_shift + 1, size=n_shifts)
    null_t = np.empty((n_shifts, K))
    for b, s in enumerate(shifts):
        null_t[b] = tfun(daily(int(s)))
    null_max = null_t.max(1)
    t_max = float(t_obs.max())
    return {
        "p_sel": float((1 + np.sum(null_max >= t_max)) / (n_shifts + 1)),
        "t_obs": t_obs, "t_max_obs": t_max, "best": int(np.argmax(t_obs)),
        "p_single": (1 + (null_t >= t_obs[None, :]).sum(0)) / (n_shifts + 1),
        "null_max": null_max, "null_t": null_t, "shifts": shifts,
        "daily_obs": daily_obs, "day_ids": grid.ids,
        "seconds": time.perf_counter() - t0,
        "settings": {"n_shifts": n_shifts, "min_shift": min_shift, "seed": seed, "tstat": tstat, "lag": lag,
                     "c_in": c_in, "c_out": c_out, "dtype": np.dtype(dtype).name, "K": K, "n": n, "D": D},
    }


# --------------------------------------------------------------------------------------------- Deflated Sharpe

def expected_max_sharpe(n_trials, var_sr):
    """SR0 = sqrt(V[SR_n]) ((1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e))), g = Euler-Mascheroni: the expected maximum
    Sharpe of N independent trials with true Sharpe 0 (Bailey and Lopez de Prado 2014). SR0 = 0 for N = 1."""
    N = float(n_trials)
    if N < 1:
        raise ValueError("n_trials must be >= 1")
    if N == 1:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / N) + EULER_GAMMA * norm.ppf(1 - 1 / (N * np.e))))


def probabilistic_sharpe(sr, sr0, T, skew=0.0, kurt=3.0):
    """PSR = Phi((SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2)). SR, SR0 per period; kurt is
    Pearson kurtosis (3 for a normal, scipy.stats.kurtosis(x, fisher=False))."""
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if den <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1.0) / np.sqrt(den)))


def deflated_sharpe(sr, n_trials, var_sr, skew, kurt, T):
    """Deflated Sharpe ratio: PSR of the observed per-period Sharpe `sr` against SR0 = expected_max_sharpe(N, var_sr).
    sr and var_sr (variance of the N trial Sharpes) are per period (daily, not annualised); T = number of returns;
    skew and Pearson kurtosis of the selected strategy's returns. N = every configuration tried in the track plus
    logged reruns. Reported, not a gate."""
    return probabilistic_sharpe(sr, expected_max_sharpe(n_trials, var_sr), T, skew, kurt)


# ------------------------------------------------------------------------------------------------------- Holm

def holm(pvals):
    """Holm step-down adjusted p-values for a dict {name: p}: with p sorted ascending, adj_(i) = max_{j<=i}
    min(1, (m - j + 1) p_(j)). Returns {name: adjusted p} in the input order; reject where adjusted p <= alpha."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m, run, adj = len(names), 0.0, {}
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * float(pvals[k])))
        adj[k] = run
    return {k: adj[k] for k in pvals}


# ------------------------------------------------------------------------------------------------------- power

def _z(alpha, power):
    return norm.ppf(1.0 - alpha) + norm.ppf(power)


def min_detectable_mean(sd, n, alpha=0.05, power=0.80, deff=1.0):
    """One-sided minimum detectable mean (per trade or per day): (z_{1-alpha} + z_power) sd sqrt(deff / n).
    deff is the variance design effect (1.3 for overlapping trades in SPEC POWER)."""
    return float(_z(alpha, power) * sd * np.sqrt(deff / n))


def power_mean(delta, sd, n, alpha=0.05, deff=1.0):
    """Power of the one-sided z test when the true mean is `delta` away from the null boundary, in the alternative's
    direction: Phi(delta / (sd sqrt(deff/n)) - z_{1-alpha}). E.g. rejecting 'net >= 0' when the true net is -3 bps
    uses delta = 3."""
    return float(norm.cdf(delta / (sd * np.sqrt(deff / n)) - norm.ppf(1.0 - alpha)))


def min_detectable_sharpe(years, alpha=0.05, power=0.80):
    """Minimum detectable annualised Sharpe, with SE(annual Sharpe) ~ 1/sqrt(years): (z_{1-alpha} + z_power)/sqrt(years)."""
    return float(_z(alpha, power) / np.sqrt(years))


def power_sharpe(sr_ann, years, alpha=0.05):
    """Power to show Sharpe > 0 (one-sided) for a true annualised Sharpe: Phi(sr_ann sqrt(years) - z_{1-alpha})."""
    return float(norm.cdf(sr_ann * np.sqrt(years) - norm.ppf(1.0 - alpha)))

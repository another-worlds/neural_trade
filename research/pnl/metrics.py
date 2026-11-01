"""Performance metrics for the PnL study (SPEC 'simulator', OUTPUTS PER RUN). Pure functions on simulator outputs.

Inputs:
- r: daily (UTC) net log equity returns, one per day; equity E_d = exp(cumsum r), E = 1 before the first day;
- days: one timestamp per day: integer day numbers (t // 86400, as sim.simulate returns) or unix seconds (unit='auto'
  tells them apart: values < 1e7 are days), or datetime64 / pandas datetimes;
- x: per-bar exposure at each bar close (x[t] is held over bar t+1), as the simulator records it;
- trades: the trade log, a list of dicts (or a DataFrame) with gross_bps, net_bps, fees_bps, funding_bps, hold_minutes;
- bench: the benchmark's daily return series (BTC B&H), aligned with r.
Conventions: 365-day year. 'additive' = sum (total) or 365 x mean (annualised) of the daily log returns, the gate
statistic S; 'compound' = exp(.) - 1 of the same. Returns are in bps; drawdowns are fractions of the running peak.
Vol, Sharpe and Sortino are annualised with sqrt(365) and use no risk-free rate (a USDT futures wallet earns 0).
"""
import numpy as np
import pandas as pd

YEAR_DAYS = 365
BPS = 1e4
TRADE_KEYS = ("gross_bps", "net_bps", "fees_bps", "funding_bps", "hold_minutes")


def _f64(r):
    return np.asarray(r, np.float64)


def _day64(days, unit="auto"):
    """Day timestamps -> datetime64[D] (UTC). Integers are read in `unit`: 's' unix seconds, 'D' days since epoch, or
    'auto' (days when every |value| < 1e7, i.e. before year 29349; seconds otherwise, i.e. after 1970-04-26)."""
    a = np.asarray(days)
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating):
        if unit == "auto":
            unit = "D" if len(a) and np.abs(a).max() < 1e7 else "s"
        return a.astype(np.int64).astype(f"datetime64[{unit}]").astype("datetime64[D]")
    return pd.DatetimeIndex(pd.to_datetime(a, utc=True)).tz_convert(None).values.astype("datetime64[D]")


# --------------------------------------------------------------------------------------------- daily return metrics

def returns(r):
    """Total and annualised net return in bps: total_compound_bps = 1e4 (exp(sum r) - 1), total_additive_bps =
    1e4 sum r, ann_compound_bps = 1e4 (exp(365 mean r) - 1), ann_additive_bps = 1e4 x 365 x mean r."""
    r = _f64(r)
    s, m = float(r.sum()), float(r.mean()) if len(r) else np.nan
    return {"days": len(r), "years": len(r) / YEAR_DAYS,
            "total_compound_bps": BPS * float(np.expm1(s)), "total_additive_bps": BPS * s,
            "ann_compound_bps": BPS * float(np.expm1(YEAR_DAYS * m)), "ann_additive_bps": BPS * YEAR_DAYS * m}


def vol(r, axis=-1):
    """Annualised volatility: sd(ddof=1) x sqrt(365)."""
    return _f64(r).std(axis, ddof=1) * np.sqrt(YEAR_DAYS)


def sharpe(r, axis=-1):
    """Annualised Sharpe: mean/sd(ddof=1) x sqrt(365); 0 where sd = 0. Vectorised along `axis`."""
    r = _f64(r)
    m, sd = np.asarray(r.mean(axis)), np.asarray(r.std(axis, ddof=1))
    return np.divide(m, sd, out=np.zeros_like(m), where=sd > 0) * np.sqrt(YEAR_DAYS)


def sortino(r, axis=-1):
    """Annualised Sortino: 365 mean / (sqrt(365) DD), DD = sqrt(mean(min(r, 0)^2)) over all days (target 0)."""
    r = _f64(r)
    m, dd = np.asarray(r.mean(axis)), np.asarray(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2, axis)))
    out = np.where(m == 0, 0.0, np.copysign(np.inf, m))          # no losing day: +inf (or 0 when flat)
    return np.divide(m, dd, out=out, where=dd > 0) * np.sqrt(YEAR_DAYS)


def max_drawdown(r, axis=-1):
    """Maximum drawdown 1 - E/peak(E), as a fraction; vectorised along `axis`."""
    r = np.moveaxis(_f64(r), axis, -1)
    le = np.cumsum(r, -1)
    le = np.concatenate([np.zeros(le.shape[:-1] + (1,)), le], -1)
    return 1.0 - np.exp((le - np.maximum.accumulate(le, -1)).min(-1))


def drawdown(r):
    """Maximum drawdown and its duration. Equity points are indexed 0..T (0 = start, d = after day d), so a span
    between points is a number of days.
    - max_dd: largest 1 - E/peak; peak / trough: equity points of that drawdown;
    - duration_days: peak to recovery (first point after the trough with E >= the peak), or to the end of the series
      when it has not recovered (recovered = False);
    - longest_underwater_days: the longest span of any drawdown, peak to recovery or to the end."""
    r = _f64(r)
    le = np.r_[0.0, np.cumsum(r)]
    peak = np.maximum.accumulate(le)
    dd = 1.0 - np.exp(le - peak)
    # underwater runs: maximal runs [a, b) of points strictly below the running peak. The peak is point a - 1; a run
    # that recovers (b <= T) spans b - (a - 1) = len + 1 days, one that lasts to the end spans T - (a - 1) = len days.
    under = np.r_[False, le < peak, False].astype(np.int8)
    a, b = np.flatnonzero(np.diff(under) == 1), np.flatnonzero(np.diff(under) == -1)
    spans = (b - a) + (b < len(le))
    trough = int(np.argmax(dd))
    if dd[trough] <= 0:
        return {"max_dd": 0.0, "peak": 0, "trough": 0, "recovery": 0, "duration_days": 0, "recovered": True,
                "longest_underwater_days": 0}
    j = int(np.searchsorted(a, trough, side="right")) - 1          # the run holding the trough
    p, rec = int(a[j]) - 1, (int(b[j]) if b[j] < len(le) else None)
    return {"max_dd": float(dd[trough]), "peak": p, "trough": trough, "recovery": rec,
            "duration_days": (rec if rec is not None else len(le) - 1) - p, "recovered": rec is not None,
            "longest_underwater_days": int(spans.max())}


def calmar(r):
    """Calmar: annualised compound return / max drawdown (fractions); inf when there is no drawdown."""
    r = _f64(r)
    mdd = float(max_drawdown(r))
    ann = np.expm1(YEAR_DAYS * r.mean())
    if mdd > 0:
        return float(ann / mdd)
    return 0.0 if ann == 0 else float(np.copysign(np.inf, ann))


def positive_months(r, days, unit="auto"):
    """Share of calendar months (UTC) with a positive sum of daily log returns. Partial months at the ends count."""
    r = _f64(r)
    m = _day64(days, unit).astype("datetime64[M]")
    _, inv = np.unique(m, return_inverse=True)
    s = np.bincount(inv.ravel(), weights=r)
    return {"pct_positive_months": float(np.mean(s > 0)) if len(s) else np.nan, "months": len(s)}


def pnl_by_year(r, days, unit="auto"):
    """PnL per calendar year (UTC): {year: {additive_bps, compound_bps, days}}."""
    r = _f64(r)
    y = _day64(days, unit).astype("datetime64[Y]").astype(np.int64) + 1970
    out = {}
    for yr in np.unique(y):
        s = float(r[y == yr].sum())
        out[int(yr)] = {"additive_bps": BPS * s, "compound_bps": BPS * float(np.expm1(s)), "days": int(np.sum(y == yr))}
    return out


# --------------------------------------------------------------------------------------------------- exposure

def exposure_stats(x):
    """Mean |x|, mean x and the share of bars with |x| > 0 (time in market)."""
    x = _f64(x)
    return {"mean_abs_x": float(np.abs(x).mean()), "mean_x": float(x.mean()), "pct_in_market": float(np.mean(x != 0))}


def turnover(x, bar_minutes=1, x0=0.0):
    """Turnover sum |dx| over the bars, from x0 (flat) before the first bar, total and per 365-day year.
    On per-bar drifting exposure this also counts the price drift of x between fills."""
    x = _f64(x)
    tot = float(np.abs(np.diff(np.r_[x0, x])).sum())
    years = len(x) * bar_minutes / (1440.0 * YEAR_DAYS)
    return {"turnover": tot, "turnover_per_year": tot / years if years > 0 else np.nan}


def static_timing_split(x, r, held=False):
    """Static-versus-timing split of exposure PnL (simple returns, engine-F style): with xh the exposure held over each
    bar (x[t-1], x_{-1} = 0; or x itself if held=True) and R = exp(r) - 1, total = sum xh R = static + timing with
    static = mean(xh) sum R (B&H PnL scaled by the mean exposure) and timing = sum (xh - mean xh) R. In bps of equity."""
    x, R = _f64(x), np.expm1(_f64(r))
    xh = x if held else np.r_[0.0, x[:-1]]
    mx = xh.mean()
    return {"total_bps": BPS * float(xh @ R), "static_bps": BPS * float(mx * R.sum()),
            "timing_bps": BPS * float((xh - mx) @ R), "mean_x": float(mx)}


# ----------------------------------------------------------------------------------------------------- trades

def trade_stats(trades, years=None):
    """Per-trade statistics from the trade log: count, hit rate (net > 0), mean and median net, SD per trade (ddof=1),
    mean gross / fees / funding, mean and median hold (minutes), and trades per year when `years` is given."""
    if hasattr(trades, "to_dict"):
        trades = trades.to_dict("records")
    col = {k: np.array([float(t[k]) for t in trades]) for k in TRADE_KEYS}
    n, net = len(col["net_bps"]), col["net_bps"]
    nan = float("nan")
    return {"count": n,
            "trades_per_year": n / years if years else nan,
            "hit_rate": float(np.mean(net > 0)) if n else nan,
            "mean_net_bps": float(net.mean()) if n else nan,
            "median_net_bps": float(np.median(net)) if n else nan,
            "sd_net_bps": float(net.std(ddof=1)) if n > 1 else nan,
            "mean_gross_bps": float(col["gross_bps"].mean()) if n else nan,
            "mean_fees_bps": float(col["fees_bps"].mean()) if n else nan,
            "mean_funding_bps": float(col["funding_bps"].mean()) if n else nan,
            "mean_hold_minutes": float(col["hold_minutes"].mean()) if n else nan,
            "median_hold_minutes": float(np.median(col["hold_minutes"])) if n else nan}


# ------------------------------------------------------------------------------------------------ beta / alpha

def beta_alpha(r, bench, hac_lag=None):
    """Daily OLS r = alpha + beta bench + e. Returns beta, alpha (per day), alpha_ann_bps (1e4 x 365 x alpha), t_alpha,
    t_beta, r2, n. Classical OLS SEs; with hac_lag, Newey-West (Bartlett) sandwich SEs with that lag in days."""
    y, b = _f64(r), _f64(bench)
    if y.shape != b.shape:
        raise ValueError("r and bench must be aligned")
    n = len(y)
    X = np.column_stack([np.ones(n), b])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    e = y - X @ coef
    if hac_lag is None:
        cov = XtX_inv * (e @ e) / (n - 2)
    else:
        u = X * e[:, None]
        S = u.T @ u
        for l in range(1, min(int(hac_lag), n - 1) + 1):
            G = u[l:].T @ u[:-l]
            S += (1.0 - l / (hac_lag + 1.0)) * (G + G.T)
        cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = np.divide(coef, se, out=np.full(2, np.nan), where=se > 0)
    tss = ((y - y.mean()) ** 2).sum()
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "alpha_ann_bps": BPS * YEAR_DAYS * float(coef[0]),
            "t_alpha": float(t[0]), "t_beta": float(t[1]), "r2": float(1 - (e @ e) / tss) if tss > 0 else np.nan,
            "n": n}


# ---------------------------------------------------------------------------------------------------- summary

def summary(r, days=None, exposure=None, trades=None, bench=None, bar_minutes=1, unit="auto", hac_lag=None):
    """Every metric above that the given inputs allow, in one flat-ish dict (for results/*.json)."""
    r = _f64(r)
    out = dict(returns(r))
    dd = drawdown(r)
    out.update(vol=float(vol(r)), sharpe=float(sharpe(r)), sortino=float(sortino(r)), max_dd=dd["max_dd"],
               max_dd_duration_days=dd["duration_days"], max_dd_recovered=dd["recovered"],
               longest_underwater_days=dd["longest_underwater_days"], calmar=calmar(r))
    if days is not None:
        out.update(positive_months(r, days, unit))
        out["by_year"] = pnl_by_year(r, days, unit)
    if exposure is not None:
        out.update(exposure_stats(exposure))
        out.update(turnover(exposure, bar_minutes))
    if trades is not None:
        out["trades"] = trade_stats(trades, years=len(r) / YEAR_DAYS)
    if bench is not None:
        out["beta_alpha"] = beta_alpha(r, bench, hac_lag)
    return out

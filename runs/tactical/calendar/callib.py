"""Calendar-effect library (SPEC.md). Hourly next-open to next-open returns in bps; buckets; Newey-West; Holm."""
import numpy as np, pandas as pd
from scipy import stats as sps

SEL_START, SEL_END = "2021-01-01", "2024-06-30"
TEST_START, TEST_END = "2024-07-01", "2025-09-30"
NW_LAGS = 48


def hourly_returns(csv):
    """DataFrame indexed by hour start with column r (bps): open(t+1h)/open(t)-1. Rows with a missing hh:00 bar are dropped."""
    d = pd.read_csv(csv, usecols=["timestamp", "open"], parse_dates=["timestamp"])
    d = d[d.timestamp.dt.minute == 0].set_index("timestamp")["open"]
    d = d[~d.index.duplicated()]
    nxt = d.reindex(d.index + pd.Timedelta(hours=1)).values
    r = (nxt / d.values - 1.0) * 1e4
    return pd.DataFrame({"r": r}, index=d.index).dropna()


def buckets(idx):
    """dict name -> boolean array over the DatetimeIndex."""
    h, dow = idx.hour, idx.dayofweek
    b = {f"hour_{k:02d}": np.asarray(h == k) for k in range(24)}
    for k, n in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
        b[f"dow_{n}"] = np.asarray(dow == k)
    b["funding_pre"] = np.asarray(np.isin(h, [7, 15, 23]))
    b["funding_post"] = np.asarray(np.isin(h, [0, 8, 16]))
    b["month_last_day"] = np.asarray(idx.is_month_end)
    b["month_first_day"] = np.asarray(idx.day == 1)
    b["weekend"] = np.asarray(dow >= 5)
    return b


def nw_mean(r, mask, lags=NW_LAGS):
    """Mean of r over mask (full time index) with Newey-West (Bartlett) standard error. Returns mean, se, n."""
    D = mask.astype(float)
    n = D.sum()
    if n < 3:
        return float("nan"), float("nan"), int(n)
    m = (D * r).sum() / n
    s = D * (r - m)
    v = (s * s).sum()
    for lag in range(1, lags + 1):
        v += 2 * (1 - lag / (lags + 1)) * (s[lag:] * s[:-lag]).sum()
    return float(m), float(np.sqrt(max(v, 0.0)) / n), int(n)


def effect(r, mask):
    m, se, n = nw_mean(r, mask)
    z = m / se if se > 0 else float("nan")
    p = float(2 * sps.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {"mean_bps": m, "se": se, "n": n, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "p": p}


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni: boolean array of rejections (NaN p never rejected)."""
    p = np.array([np.inf if not np.isfinite(x) else x for x in pvals])
    k = len(p)
    rej = np.zeros(k, bool)
    for rank, i in enumerate(np.argsort(p)):
        if p[i] <= alpha / (k - rank):
            rej[i] = True
        else:
            break
    return rej


def signals(idx, signs):
    """Per-hour signal = sign of the sum of signs of the chosen buckets containing the hour. signs: name -> +1/-1."""
    b = buckets(idx)
    s = np.zeros(len(idx))
    for n, sg in signs.items():
        s += sg * b[n]
    return np.sign(s)


def trade_stats(ret_bps, times, cost):
    net = ret_bps - cost
    if len(net) == 0:
        return {"trades": 0}
    eq = np.cumprod(1 + net / 1e4)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    mon = pd.Series(net, index=times).groupby(times.to_period("M")).sum()
    k = len(mon)
    h = mon.std(ddof=1) / np.sqrt(k) * sps.t.ppf(.975, k - 1) if k > 1 else float("nan")
    return {"trades": int(len(net)), "net_bps_trade": float(net.mean()), "hit": float((net > 0).mean()),
            "net_bps_month": float(mon.mean()), "month_ci95": [float(mon.mean() - h), float(mon.mean() + h)], "months": k,
            "max_dd_pct": float(100 * (1 - eq / peak).max())}

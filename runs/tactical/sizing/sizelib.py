"""Volatility-based sizing and timing on top of a frozen direction signal (SPEC.md). Pure functions, tested in test_sizing.py.

Everything a rule uses at an entry is computed from bars up to the signal bar j (the last COMPLETED hour); the trade enters at the
open of bar j+1. The HAR volatility model is refitted every calendar month on rows whose target bars closed by the month start.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TAC, "common")); sys.path.insert(0, os.path.join(TAC, "equity"))
import cutoff as C  # noqa: E402  (H66: no bar after C.CUTOFF is read)
import eqlib as E  # noqa: E402

EPS = 1e-9
HOUR = 3600 * 10 ** 9
ROLL = 4320            # 180 days of hours: window of the vol-target median and the vol-change quantile
ROLL_MIN = 720
W_MIN, W_MAX = 0.25, 2.0
Q_SKIP = 0.80
BRAKE_DD, BRAKE_F = 0.02, 0.5
COSTS = E.COSTS
RULES = ("R0_fixed", "R1_voltarget", "R2_volchange_filter", "R3_vt_plus_filter", "R4_brake", "R5_all")
RULE_PARTS = {"R0_fixed": (0, 0, 0), "R1_voltarget": (1, 0, 0), "R2_volchange_filter": (0, 1, 0), "R3_vt_plus_filter": (1, 1, 0),
              "R4_brake": (0, 0, 1), "R5_all": (1, 1, 1)}      # (vol target, filter, brake)


# ------------------------------------------------------------------ hourly realised variance (from the cut 1-minute file)
def hourly_rv(end=None):
    """Series of the realised variance of each hour bar (sum of squared 1-minute log returns), indexed by the hour's open time.
    The hour index is the one of the 1-hour bars used everywhere (resample + dropna on the close). end: exclusive upper bound."""
    d = C.read_csv(usecols=["timestamp", "close"])
    d.index = pd.to_datetime(d["timestamp"])
    if end is not None:
        d = d[d.index < pd.Timestamp(end)]
    C.assert_cut(d.index, "minute bars")
    lr = np.log(d["close"].astype(float)).diff().fillna(0.0)
    rv = (lr * lr).resample("60min").sum()
    has = d["close"].resample("60min").last().notna()
    return rv[has]


def har_design(rv):
    """x1, x24, x168 (log of the last hour / 24-hour mean / 168-hour mean of RV) for each bar, using bars <= j only."""
    s = pd.Series(np.asarray(rv, float))
    x1 = np.log(s.values + EPS)
    x24 = np.log(s.rolling(24).mean().values + EPS)
    x168 = np.log(s.rolling(168).mean().values + EPS)
    return x1, x24, x168


def har_target(rv, h):
    """y_j = log of the mean RV over bars j+1..j+h (NaN where those bars do not exist)."""
    s = pd.Series(np.asarray(rv, float))
    m = s.rolling(h).mean().shift(-h)
    return np.log(m.values + EPS)


def har_walk_forward(rv, times, h, first, last, train_days=365):
    """Monthly OLS HAR. Returns pred[j] = fitted E[log mean RV over j+1..j+h] given bars <= j, for every bar j in [first, last] months
    (NaN elsewhere). Training rows: bar j with open time >= month start - train_days and its target bars closed by the month start
    (times[j] + (h + 1) hours <= month start). rv, times: aligned arrays (times datetime64)."""
    t = pd.DatetimeIndex(times); x1, x24, x168 = har_design(rv); y = har_target(rv, h)
    X = np.column_stack([np.ones(len(t)), x1, x24, x168]); ok = np.isfinite(X).all(1)
    pred = np.full(len(t), np.nan)
    for m in pd.period_range(first, last, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        tr = ok & np.isfinite(y) & (t >= m0 - pd.Timedelta(days=train_days)) & (t + pd.Timedelta(hours=h + 1) <= m0)
        te = ok & (t >= m0) & (t < m1)
        if tr.sum() < 500 or not te.any():
            continue
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred[te] = X[te] @ beta
    return pred, x1


def state(rv, times, h, first, last):
    """Per-bar rule inputs at the signal bar j: sigma_hat (HAR vol forecast per sqrt-hour unit), target (trailing median of sigma_hat),
    dlv (predicted log-vol change: HAR forecast minus the last hour's log RV), thr (trailing 80% quantile of dlv). Trailing windows
    use bars <= j only."""
    pred, x1 = har_walk_forward(rv, times, h, first, last)
    sig = np.exp(0.5 * pred); dlv = pred - x1
    s = pd.Series(sig); d = pd.Series(dlv)
    target = s.rolling(ROLL, min_periods=ROLL_MIN).median().values
    thr = d.rolling(ROLL, min_periods=ROLL_MIN).quantile(Q_SKIP).values
    return {"sigma": sig, "target": target, "dlv": dlv, "thr": thr, "pred": pred}


def base_weight(sig, target):
    return np.clip(target / sig, W_MIN, W_MAX)


# ------------------------------------------------------------------ rule application on a signal set
def apply_rule(rule, sig_bar, entry_ns, exit_ns, st):
    """Select the trades of a rule. sig_bar: signal bar position j per signal. Returns (order, w0): the signals kept AFTER the filter and
    the skip-while-open rule, sorted by entry, and their base weights (vol-target part; the brake acts in simulate)."""
    vt, filt, _ = RULE_PARTS[rule]
    sig_bar = np.asarray(sig_bar)
    keep0 = np.ones(len(sig_bar), bool)
    if filt:
        keep0 = ~(st["dlv"][sig_bar] > st["thr"][sig_bar])           # NaN comparisons are False: nothing skipped without a threshold
    idx = np.flatnonzero(keep0)
    k, _ = E.non_overlapping(np.asarray(entry_ns)[idx], np.asarray(exit_ns)[idx])
    idx = idx[k]; idx = idx[np.argsort(np.asarray(entry_ns)[idx], kind="stable")]
    w0 = base_weight(st["sigma"][sig_bar[idx]], st["target"][sig_bar[idx]]) if vt else np.ones(len(idx))
    return idx, w0


def simulate(w0, gross, cost_bps, brake):
    """gross [D, n] side-adjusted returns (fractions) of D side-draws over the same n trades, w0 [n] base weights.
    Per trade: w = w0 * (BRAKE_F if the brake is on), pnl = w * (gross - cost). The brake state at trade i uses the equity of trades
    < i only (non-overlapping: all closed). Returns (pnl [D,n], w [D,n])."""
    gross = np.atleast_2d(np.asarray(gross, float)); D, n = gross.shape
    net = gross - cost_bps * 1e-4
    if not brake:
        w = np.broadcast_to(np.asarray(w0, float), (D, n)).copy()
        return w * net, w
    pnl = np.empty((D, n)); w = np.empty((D, n))
    eq = np.ones(D); peak = np.ones(D); on = np.zeros(D, bool)
    for i in range(n):
        w[:, i] = w0[i] * np.where(on, BRAKE_F, 1.0)
        pnl[:, i] = w[:, i] * net[:, i]
        eq = eq + pnl[:, i]
        newpeak = eq > peak
        peak = np.maximum(peak, eq)
        dd = (peak - eq) / peak
        on = np.where(newpeak, False, np.where(dd >= BRAKE_DD, True, on))
    return pnl, w


def dd_pct(pnl):
    """max drawdown (percent of the running peak) of additive equity 1 + cumsum(pnl) along the last axis."""
    pnl = np.atleast_2d(pnl); eq = 1.0 + np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((pnl.shape[0], 1)), eq], axis=1), axis=1)[:, 1:]
    return ((peak - eq) / peak).max(1) * 100


def month_dd_share(pnl, entry_ns, limit=5.0):
    """Share of calendar months (by entry) whose within-month max drawdown (equity restarts at the month's first trade) is below limit %."""
    t = pd.DatetimeIndex(np.asarray(entry_ns, np.int64).astype("datetime64[ns]")).to_period("M")
    dds = []
    for m in pd.unique(t):
        p = np.asarray(pnl)[np.asarray(t == m)]
        dds.append(dd_pct(p)[0])
    dds = np.array(dds)
    return float((dds < limit).mean()), float(np.max(dds)), len(dds)


def rule_stats(pnl, w, entry_ns, exit_ns, start_ns, end_ns):
    """Metrics of one executed trade sequence (pnl already net of cost and weighted)."""
    s = E.equity_stats(pnl, entry_ns, exit_ns, start_ns, 0.0)
    if s["n_trades"] == 0:
        return s
    years = (end_ns - start_ns) / (365.25 * 24 * HOUR)
    s["years"] = float(years); s["avg_weight"] = float(np.mean(w))
    s["calmar"] = float(s["cum_return_pct"] / years / max(s["max_drawdown_pct"], 1e-9))
    sh, mx, nm = month_dd_share(pnl, entry_ns)
    s["share_months_dd_below_5"] = sh; s["worst_month_dd_pct"] = mx; s["n_months_with_trades"] = nm
    return s


def null_stats(w0, pl_long, pl_short, cost_bps, brake, actual_cum, draws=500, seed=0):
    rng = np.random.default_rng(seed); n = len(w0)
    flip = rng.random((draws, n)) < 0.5
    gross = np.where(flip, pl_long[None, :], pl_short[None, :])
    pnl, _ = simulate(w0, gross, cost_bps, brake)
    cum = pnl.sum(1) * 100; dd = dd_pct(pnl); hit = (pnl > 0).mean(1)
    q = lambda v, a: float(np.quantile(v, a))
    return {"draws": draws, "cum_median": q(cum, .5), "cum_p95": q(cum, .95), "dd_median": q(dd, .5), "dd_p95": q(dd, .95),
            "dd_p05": q(dd, .05), "hit_median": q(hit, .5), "hit_p95": q(hit, .95), "p_cum_ge_actual": float((cum >= actual_cum).mean())}


def evaluate(sig, st, start_ns, end_ns, costs=COSTS, rules=RULES, draws=500):
    """sig: dict of arrays per signal: bar, entry, exit, side, r_long (raw return of the long side). Returns {rule: {cost: {stats, null}}}."""
    out = {}
    for rule in rules:
        idx, w0 = apply_rule(rule, sig["bar"], sig["entry"], sig["exit"], st)
        brake = bool(RULE_PARTS[rule][2])
        rl = sig["r_long"][idx]; side = sig["side"][idx]
        gross_actual = side * rl
        ent, ex = sig["entry"][idx], sig["exit"][idx]
        res = {"n_signals": int(len(sig["bar"])), "n_trades": int(len(idx)), "levels": {}}
        for c in costs:
            pnl, w = simulate(w0, gross_actual, c, brake)
            s = rule_stats(pnl[0], w[0], ent, ex, start_ns, end_ns)
            nl = null_stats(w0, rl, -rl, c, brake, s.get("cum_return_pct", 0.0) if s["n_trades"] else 0.0, draws)
            res["levels"][str(c)] = {"stats": s, "null": nl}
        out[rule] = res
    return out

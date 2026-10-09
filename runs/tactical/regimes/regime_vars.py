"""Regime variables from minute closes, using only bars up to and including the decision bar (SPEC.md)."""
import numpy as np

N24, N1 = 1440, 60
CELLS = {"vol": ["lo", "mid", "hi"], "vtrend": ["lo", "mid", "hi"], "trend": ["lo", "mid", "hi"],
         "session": ["asia", "eu", "us"], "weekend": ["weekday", "weekend"]}


def raw_vars(close, pos):
    """close: minute closes (full array); pos: indices of decision bars. Returns dict of float arrays (NaN where < 1440 earlier minutes).
    Reads close[pos-1440 .. pos] only."""
    lc = np.log(np.asarray(close, np.float64)); pos = np.asarray(pos)
    out = {k: np.full(len(pos), np.nan) for k in ("vol", "vtrend", "trend")}
    ok = pos >= N24
    p = pos[ok]
    r = np.diff(lc, prepend=lc[0])                       # r[k] = lc[k] - lc[k-1]
    cs = np.concatenate([[0.0], np.cumsum(r * r)])      # cs[k+1] - cs[j+1] = sum r[j+1..k]
    # only prefix sums up to max(pos) matter for the values at pos; cs is causal (prefix) by construction
    v24 = np.sqrt((cs[p + 1] - cs[p + 1 - N24]) / N24); v1 = np.sqrt((cs[p + 1] - cs[p + 1 - N1]) / N1)
    out["vol"][ok] = v24; out["vtrend"][ok] = v1 / np.maximum(v24, 1e-12)
    out["trend"][ok] = np.abs(lc[p] - lc[p - N24]) / np.maximum(v24 * np.sqrt(N24), 1e-12)
    return out


def calendar(decision_time):
    """decision_time: DatetimeIndex of the decision INSTANT (bar open + bar length). Returns session code 0/1/2 and weekend 0/1."""
    h = np.asarray(decision_time.hour); d = np.asarray(decision_time.dayofweek)
    return (h // 8).astype(int), (d >= 5).astype(int)


def cuts(train_vals):
    v = np.asarray(train_vals, float); v = v[np.isfinite(v)]
    return np.quantile(v, [1 / 3, 2 / 3])


def tercile(vals, cut):
    v = np.asarray(vals, float); c = np.digitize(v, cut)
    return np.where(np.isfinite(v), c, -1)


def cell_codes(close, pos, decision_time, train_pos=None, cut_fit=None):
    """dict var -> int cell code per decision bar (-1: no regime). Tercile cuts from train_pos (positions) or from cut_fit (dict of cuts)."""
    rv = raw_vars(close, pos); ses, wk = calendar(decision_time)
    if cut_fit is None:
        tv = raw_vars(close, train_pos); cut_fit = {k: cuts(tv[k]) for k in ("vol", "vtrend", "trend")}
    codes = {k: tercile(rv[k], cut_fit[k]) for k in ("vol", "vtrend", "trend")}
    ok = np.isfinite(rv["vol"])
    codes["session"] = np.where(ok, ses, -1); codes["weekend"] = np.where(ok, wk, -1)
    return codes, cut_fit

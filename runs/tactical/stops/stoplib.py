"""HAR-set stops and take-profits on frozen side signals (SPEC.md). Pure functions, tested in test_stops.py.

A trade enters at the open of the first minute of the hour after the signal hour. Its path is read on 1-minute bars. The stop distance is
m x s_j with s_j = sigma_j x sqrt(h) taken from the HAR state at the signal bar j (information at entry only).
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TAC, "sizing"))
import sizelib as S  # noqa: E402  (also puts common/ and equity/ on the path)

C = S.C
E = S.E
MIN_NS = 60 * 10 ** 9
HOUR = S.HOUR
# id -> (stop multiple or None, take-profit multiple or None, trailing)
RULES = {"X0_time": (None, None, False), "X1_sym1": (1.0, 1.0, False), "X2_sym15": (1.5, 1.5, False), "X3_sym2": (2.0, 2.0, False),
         "X4_asym": (1.0, 2.0, False), "X5_trail1": (1.0, None, True)}
BASE_RULES = tuple(RULES)
STOP_RULES = tuple(r for r in RULES if r != "X0_time")
ALL_RULES = BASE_RULES + ("X6_best_plus_filter",)
KIND_TIME, KIND_STOP, KIND_TP = 0, 1, 2
WEIGHTS = tuple(round(0.1 * i, 1) for i in range(1, 11))


# ------------------------------------------------------------------ minute data
def load_minutes(start, end):
    """OHLC of the cut 1-minute file with start <= timestamp < end. Returns dict of arrays: t (int64 ns), o, h, l, c."""
    d = C.read_csv(usecols=["timestamp", "open", "high", "low", "close"])
    t = pd.DatetimeIndex(pd.to_datetime(d["timestamp"]))
    m = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))
    C.assert_cut(t[m], "minute bars")
    return {"t": t[m].values.astype("datetime64[ns]").astype(np.int64), **{k: d[k].values[m].astype(float) for k in ("open", "high", "low", "close")}}


# ------------------------------------------------------------------ one trade
def sim_trade(side, o, h, l, c, d_sl, d_tp, trail):
    """Path of one trade on the minutes [entry, exit) (arrays o,h,l,c of the slice). side +1 long / -1 short; d_sl, d_tp: stop and take-profit
    distances as fractions of the entry price (None = none); trail: the stop trails the best extreme through the PREVIOUS minute.
    Returns (k, price, kind): the minute index of the exit, the fill price, KIND_TIME / KIND_STOP / KIND_TP. A time exit is the last minute's
    close. Conservative rules: stop before take-profit in the same minute; a stop fills at the worse of its level and the minute's open; a
    take-profit fills at its level."""
    p0 = o[0]; n = len(o)
    if d_sl is None and d_tp is None:
        return n - 1, c[-1], KIND_TIME
    if side > 0:
        if trail:
            best = np.maximum(p0, np.concatenate([[-np.inf], np.maximum.accumulate(h)[:-1]]))
            stop = best * (1 - d_sl)
        else:
            stop = np.full(n, p0 * (1 - d_sl)) if d_sl is not None else None
        sl_hit = (l <= stop) if stop is not None else np.zeros(n, bool)
        tp = p0 * (1 + d_tp) if d_tp is not None else None
        tp_hit = (h >= tp) if tp is not None else np.zeros(n, bool)
    else:
        if trail:
            best = np.minimum(p0, np.concatenate([[np.inf], np.minimum.accumulate(l)[:-1]]))
            stop = best * (1 + d_sl)
        else:
            stop = np.full(n, p0 * (1 + d_sl)) if d_sl is not None else None
        sl_hit = (h >= stop) if stop is not None else np.zeros(n, bool)
        tp = p0 * (1 - d_tp) if d_tp is not None else None
        tp_hit = (l <= tp) if tp is not None else np.zeros(n, bool)
    hit = sl_hit | tp_hit
    if not hit.any():
        return n - 1, c[-1], KIND_TIME
    k = int(np.argmax(hit))
    if sl_hit[k]:
        price = min(o[k], stop[k]) if side > 0 else max(o[k], stop[k])
        return k, price, KIND_STOP
    return k, tp, KIND_TP


def trade_paths(mins, entry_ns, exit_ns, side, sigma_trade, rule):
    """Gross side-adjusted return, exit time (ns) and exit kind of one trade under `rule`. Returns None when the minute slice is empty."""
    i0 = np.searchsorted(mins["t"], entry_ns, "left"); i1 = np.searchsorted(mins["t"], exit_ns, "left")
    if i1 <= i0:
        return None
    o, h, l, c = (mins[k][i0:i1] for k in ("open", "high", "low", "close"))
    m_sl, m_tp, trail = RULES[rule]
    d_sl = None if m_sl is None else m_sl * sigma_trade
    d_tp = None if m_tp is None else m_tp * sigma_trade
    k, price, kind = sim_trade(side, o, h, l, c, d_sl, d_tp, trail)
    t_exit = int(exit_ns) if kind == KIND_TIME else int(mins["t"][i0 + k]) + MIN_NS
    return side * (price / o[0] - 1.0), t_exit, kind


def precompute(mins, entry, exit_, sigma_trade, rules=STOP_RULES + ("X0_time",)):
    """For every signal, side (long = 0, short = 1) and rule: gross return of that side, exit time and kind. Arrays [n, 2] per rule.
    Signals with an empty minute slice get NaN gross (never taken)."""
    n = len(entry); out = {r: {"gross": np.full((n, 2), np.nan), "exit": np.zeros((n, 2), np.int64), "kind": np.zeros((n, 2), np.int8)} for r in rules}
    for i in range(n):
        for si, side in enumerate((1, -1)):
            for r in rules:
                res = trade_paths(mins, entry[i], exit_[i], side, sigma_trade[i], r)
                if res is None:
                    continue
                out[r]["gross"][i, si], out[r]["exit"][i, si], out[r]["kind"][i, si] = res
    return out


# ------------------------------------------------------------------ sequences
def sequence(entry, exit_by_side, keep, side_idx):
    """Non-overlapping skip-while-open scan over D parallel side draws. entry [n] sorted; exit_by_side [n,2]; keep [n] bool (filter);
    side_idx [D,n] in {0,1} (0 = long). Returns taken [D,n] bool. A signal is taken if keep, its entry >= the previous actual exit (entry == exit is
    taken) and its path exists."""
    D, n = side_idx.shape
    free = np.full(D, np.iinfo(np.int64).min, np.int64); taken = np.zeros((D, n), bool)
    for i in range(n):
        if not keep[i]:
            continue
        ok = entry[i] >= free
        taken[:, i] = ok
        ex_i = exit_by_side[i][side_idx[:, i]]
        free = np.where(ok, ex_i, free)
    return taken


def pick(arr2, side_idx):
    """arr2 [n,2] -> [D,n] values of the drawn side."""
    return np.where(side_idx == 0, arr2[:, 0][None, :], arr2[:, 1][None, :])


def actual_stats(pre, rule, entry, side, keep, cost_bps, start_ns, end_ns):
    """Metrics of the real (not randomised) sequence under `rule` and `cost_bps`. Signals must have valid paths (the caller drops the rest).
    Returns (stats, details of the taken trades)."""
    si = (np.asarray(side) < 0).astype(int)[None, :]
    taken = sequence(entry, pre[rule]["exit"], keep, si)
    idx = np.flatnonzero(taken[0])
    g = pre[rule]["gross"][idx, si[0, idx]]; ex = pre[rule]["exit"][idx, si[0, idx]]; kd = pre[rule]["kind"][idx, si[0, idx]]
    pnl = g - cost_bps * 1e-4
    s = S.rule_stats(pnl, np.ones(len(idx)), entry[idx], ex, start_ns, end_ns) if len(idx) else {"n_trades": 0}
    if len(idx):
        s["share_stop"] = float((kd == KIND_STOP).mean()); s["share_tp"] = float((kd == KIND_TP).mean()); s["share_time"] = float((kd == KIND_TIME).mean())
        s["avg_hold_minutes"] = float(((ex - entry[idx]) / MIN_NS).mean())
    return s, {"idx": idx, "pnl": pnl, "entry": entry[idx], "exit": ex}


def null_stats(pre, rule, entry, keep, costs, actual_cum, draws=500, seed=0):
    """Random-side null: a fair coin per signal for the side, the whole sequence (exits, non-overlap) re-simulated per draw (once for all
    costs). actual_cum: {cost: actual cumulative return %}. Returns {str(cost): stats}."""
    rng = np.random.default_rng(seed); n = len(entry)
    si = (rng.random((draws, n)) < 0.5).astype(int)
    taken = sequence(entry, pre[rule]["exit"], keep, si)
    g = pick(pre[rule]["gross"], si); nt = taken.sum(1)
    q = lambda v, a: float(np.quantile(v, a))
    out = {}
    for c in costs:
        pnl = np.where(taken, g - c * 1e-4, 0.0)
        cum = pnl.sum(1) * 100; dd = S.dd_pct(pnl); hit = ((pnl > 0) & taken).sum(1) / np.maximum(nt, 1)
        out[str(c)] = {"draws": draws, "cum_median": q(cum, .5), "cum_p95": q(cum, .95), "dd_median": q(dd, .5), "dd_p95": q(dd, .95),
                       "dd_p05": q(dd, .05), "hit_median": q(hit, .5), "hit_p95": q(hit, .95), "n_median": float(np.median(nt)),
                       "p_cum_ge_actual": float((cum >= actual_cum[c]).mean())}
    return out


def frontier(pnl_unit, weights=WEIGHTS):
    """pnl_unit: per-trade net pnl at weight 1. Cumulative return % and max DD % at each fixed weight, and the largest weight (0.01 grid, <= 1)
    with DD < 5% and the return there."""
    pnl_unit = np.asarray(pnl_unit, float)
    rows = [{"w": w, "cum_pct": float(w * pnl_unit.sum() * 100), "dd_pct": float(S.dd_pct(w * pnl_unit)[0])} for w in weights]
    grid = np.round(np.arange(0.01, 1.0001, 0.01), 2)
    ok = [w for w in grid if S.dd_pct(w * pnl_unit)[0] < 5.0]
    wmax = float(max(ok)) if ok else 0.0
    return {"rows": rows, "w_dd5": wmax, "cum_at_w_dd5": float(wmax * pnl_unit.sum() * 100)}


# ------------------------------------------------------------------ signals with the two threshold versions
def month_thresholds_prev(preds, frac=0.10):
    """{month label: 90th percentile of |P-0.5| of the PREVIOUS calendar month's predictions (absent when the previous month is missing)}.
    preds: list of (month str, times, p, r)."""
    conf = {m: np.abs(p - 0.5) for m, _, p, _ in preds}
    out = {}
    for m in conf:
        prev = str(pd.Period(m, "M") - 1)
        if prev in conf:
            out[m] = float(np.quantile(conf[prev], 1 - frac))
    return out


def signals_v(preds, H, version, first, frac=0.10, bar_ns=HOUR):
    """Top-confidence signals of months >= first. version 'same': the month's own 90th percentile (frozen convention); 'prev': the previous month's.
    Returns arrays sorted by signal time: t_open, entry, exit, side, r_long, month."""
    thr_prev = month_thresholds_prev(preds, frac)
    to, side, rl, mon = [], [], [], []
    for m, tt, p, r in preds:
        if pd.Period(m, "M") < pd.Period(first, "M"):
            continue
        conf = np.abs(p - 0.5)
        if version == "same":
            thr = np.quantile(conf, 1 - frac)
        else:
            if m not in thr_prev:
                continue
            thr = thr_prev[m]
        k = conf >= thr
        to.append(tt[k]); side.append(np.where(p[k] > 0.5, 1, -1)); rl.append(r[k]); mon += [m] * int(k.sum())
    if not to:
        return None
    to, side, rl = (np.concatenate(a) for a in (to, side, rl))
    o = np.argsort(to, kind="stable")
    to, side, rl, mon = to[o], side[o], rl[o], np.array(mon)[o]
    return {"t_open": to, "entry": to + bar_ns, "exit": to + (H + 1) * bar_ns, "side": side, "r_long": rl, "month": mon}

"""T07: evaluation of members and combinations on named hour slices (pure functions of minute positions and a Grid)."""
import numpy as np
import pandas as pd
from scipy import stats as sps

import t07_lib as T


def member_arrays(grid, pos):
    g, u, p = [], [], []
    for i in range(pos.shape[0]):
        a, b, c = T.book(pos[i], grid.r); g.append(a); u.append(b); p.append(c)
    return np.stack(g), np.stack(u), np.stack(p)


def combos_for(grid, pos, har):
    """Minute positions of C1..C6 on the whole (history + evaluation) grid. Member own-book returns at 2 bps drive the weights."""
    g, u, p = member_arrays(grid, pos)
    nets2 = T.net(g, u, T.VERDICT_COST)
    return T.build_combos(pos, p, nets2, har), (g, u, p)


def stats_for(w, grid, sl, h0, cost):
    """Headline numbers of the minute position w on hour slice sl (a slice of hours) at one cost."""
    g, u, _ = T.book(w, grid.r)
    rh = T.net(g, u, cost)[sl]
    st = T.equity_stats(rh); years = len(rh) / T.HOURS_PER_YEAR
    m0, m1 = sl.start * 60, sl.stop * 60
    tr = T.trade_stats(np.asarray(w)[m0:m1], grid.r[m0:m1], cost)
    st.update(n_trades=tr["n_trades"], trades_per_year=tr["n_trades"] / years if years else float("nan"), hit=tr["hit"], years=years)
    return st, rh


def table_for(paths, grid, slices, costs=T.COSTS):
    """paths {name: minute position}; slices {period: slice(h0, h1)}. -> {name: {period: {cost: stats}}}, and 2-bps hourly returns."""
    out, r2 = {}, {}
    for name, w in paths.items():
        out[name] = {}
        for per, sl in slices.items():
            out[name][per] = {}
            for c in costs:
                st, rh = stats_for(w, grid, sl, sl.start, c)
                out[name][per][str(c)] = st
                if c == T.VERDICT_COST:
                    r2[(name, per)] = rh
    return out, r2


def buy_hold(grid, slices):
    out = {}
    for per, sl in slices.items():
        rh = grid.r[sl.start * 60:sl.stop * 60]
        rh = T.hsum(rh)
        st = T.equity_stats(rh); st["years"] = len(rh) / T.HOURS_PER_YEAR
        out[per] = st
    return out


def null_summary(w, grid, sl, cost, draws=1000):
    cum = T.null_cum(w, grid, cost, sl, draws=draws)
    return {"p50": float(np.quantile(cum, .5)), "p95": float(np.quantile(cum, .95)), "p05": float(np.quantile(cum, .05)), "draws": int(draws)}


def monthly_summary(rh, hour_t):
    mt = T.monthly_table(rh, hour_t)
    v = mt.values; k = len(v)
    mu = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
    tq = sps.t.ppf(0.975, k - 1) if k > 1 else float("nan")
    return {"months": {m: float(x) for m, x in mt.items()}, "mean": mu, "ci95": [mu - tq * se, mu + tq * se], "n_months": int(k),
            "share_positive": float(np.mean(v > 0))}


def correlation(r_by_member):
    """Pearson correlation matrix of the members' hourly own-book returns. r_by_member [M, H]."""
    return np.corrcoef(r_by_member)

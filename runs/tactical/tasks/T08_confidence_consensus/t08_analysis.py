"""T08: evaluation of all 27 configurations on named hour slices."""
import numpy as np
import pandas as pd

import t08_lib as L


def eval_configs(S, C, Hr, slices, names=None):
    """-> {name: {period: {cost: stats}}} and {name: (ent, side)}. slices {period: (h0, h1)}."""
    a, b = L.config_names(); names = names or (a + b); out, trades = {}, {}
    for nm in names:
        ent, side = L.simulate(L.signal_for(nm, S, C)); trades[nm] = (ent, side); out[nm] = {}
        for per, (h0, h1) in slices.items():
            out[nm][per] = {str(c): L.stats_for(ent, side, Hr, c, h0, h1)[0] for c in L.COSTS}
    return out, trades


def member_deciles(S, C, Hr, h0, h1):
    f = L.fwd4_bps(Hr); sg = np.where(np.isfinite(S), np.sign(S), 0)
    return {m: L.decile_table(C[i, h0:h1], sg[i, h0:h1], f[h0:h1]) for i, m in enumerate(L.MEMBERS)}


def consensus_deciles(S, C, Hr, h0, h1):
    f = L.fwd4_bps(Hr); cc = L.consensus_confidence(S, C)
    return {k: L.decile_table(c[h0:h1], d[h0:h1], f[h0:h1]) for k, (c, d) in cc.items()}


def finalist(stats_by_name, names, per="sel"):
    """Highest Calmar at 2 bps among names with >= 30 trades per year in the period; None if there is none."""
    ok = [n for n in names if stats_by_name[n][per]["2"]["trades_per_year"] >= L.MIN_TRADES_YEAR]
    if not ok:
        return None
    cal = lambda n: stats_by_name[n][per]["2"]["calmar"]
    return max(ok, key=lambda n: (cal(n) if np.isfinite(cal(n)) else -1e9, n))

"""T08 (SPEC.md): trade on high confidence; a conservative consensus ensemble. Library of pure functions on hourly arrays.

Conventions (reading of the SPEC):
- The decision grid is hourly. The decision of hour k uses only bars that closed before hour k and is filled at the OPEN of hour k
  (the next open after the last closed bar, as in T07). The trade holds hours k .. k+3 (4 hours) and exits at the open of hour k+4.
- Scores S[member, k] are signed (sign = direction, |S| = raw confidence). NaN = no score (no vote).
- Percentile confidence C[member, k] = rank of |S[k]| within the PREVIOUS calendar month's |S| values of the same member
  (fraction of those values <= |S[k]|). Nothing from the hour's own month enters. Fewer than MIN_PREV previous-month values: NaN.
- A trade is a round trip charged c/2 bps at entry and c/2 bps at exit EVEN WHEN it starts exactly where the previous one ends
  (conservative: no netting of back-to-back trades).
- Hourly net return = side * market return of the hour while held - the hour's trade costs; equity = cumprod(1 + hourly net).
  Market return of an hour = sum of the minute returns r[j] = open[j+1] / open[j] - 1 (T07's convention).
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T07 = os.path.abspath(os.path.join(HERE, "..", "T07_ensemble_strategy"))
if T07 not in sys.path:
    sys.path.insert(0, T07)
import t07_lib as T  # noqa: E402

HOLD = 4                                              # hours
Q_A = (0.90, 0.95, 0.98)
Q_B = (0.80, 0.90, 0.95)
KINDS = ("K1", "K2", "K3")
MEMBERS = T.MEMBERS
COSTS = T.COSTS
VERDICT_COST = T.VERDICT_COST
HOURS_PER_YEAR = T.HOURS_PER_YEAR
MIN_PREV = 30                                         # previous-month values needed to rank
MIN_TRADES_YEAR = 30
VETO_C = 0.5
CACHE = os.environ.get("T08_CACHE", "D:/nt/nt_t08_cache")


def config_names():
    a = [f"A_{m}_q{q:.2f}" for m in MEMBERS for q in Q_A]
    b = [f"B_{k}_q{q:.2f}" for k in KINDS for q in Q_B]
    return a, b


# ------------------------------------------------------------------ percentile confidence
def month_labels(hour_t):
    return np.asarray(pd.DatetimeIndex(hour_t).to_period("M").astype(str))


def prev_month_percentile(score, months):
    """c[k] = fraction of the previous calendar month's |score| values that are <= |score[k]|. `months` = 'YYYY-MM' label per hour."""
    a = np.abs(np.asarray(score, float)); months = np.asarray(months); c = np.full(len(a), np.nan)
    per = {m: pd.Period(m, "M") for m in np.unique(months)}
    by = {}
    for m in np.unique(months):
        v = a[(months == m) & np.isfinite(a)]
        by[per[m]] = np.sort(v)
    for m in np.unique(months):
        prev = by.get(per[m] - 1)
        if prev is None or len(prev) < MIN_PREV:
            continue
        k = (months == m) & np.isfinite(a)
        c[k] = np.searchsorted(prev, a[k], side="right") / len(prev)
    return c


def percentiles(S, months):
    return np.stack([prev_month_percentile(S[i], months) for i in range(S.shape[0])])


# ------------------------------------------------------------------ signals
def _sg(S):
    return np.where(np.isfinite(S), np.sign(S), 0).astype(np.int8)


def _vote(S, C, q):
    return (np.nan_to_num(C, nan=-1.0) >= q) & (_sg(S) != 0)


def signal_member(S, C, i, q):
    """Family A: member i trades when c >= q, in the direction of its sign. Returns int8 [nh] in {-1, 0, 1}."""
    return np.where(_vote(S, C, q)[i], _sg(S)[i], 0).astype(np.int8)


def signal_consensus(S, C, kind, q):
    """Family B. A member votes when c >= q (and has a non-zero sign).
    K1: M1, M2, M3 all vote with the same sign. K3: all 6 vote with the same sign.
    K2: at least 3 members vote, all voters have the same sign, and no member's sign opposes them at c >= 0.5."""
    sg = _sg(S); vote = _vote(S, C, q)
    if kind in ("K1", "K3"):
        idx = [0, 1, 2] if kind == "K1" else list(range(6))
        allv = vote[idx].all(0); same = (sg[idx] == sg[idx[0]][None, :]).all(0)
        return np.where(allv & same, sg[idx[0]], 0).astype(np.int8)
    assert kind == "K2"
    nv = vote.sum(0); pos = (vote & (sg > 0)).sum(0); neg = (vote & (sg < 0)).sum(0)
    d = np.where(pos == nv, 1, np.where(neg == nv, -1, 0))                       # voters' common sign (0: mixed)
    d = np.where(nv >= 3, d, 0)
    cc = np.nan_to_num(C, nan=-1.0)
    veto = ((sg == -d[None, :]) & (cc >= VETO_C)).any(0)
    return np.where((d != 0) & ~veto, d, 0).astype(np.int8)


def signal_for(name, S, C):
    """Signal series of a configuration name: A_M3_q0.95, B_K2_q0.90."""
    fam, who, q = name.split("_"); q = float(q[1:])
    if fam == "A":
        return signal_member(S, C, MEMBERS.index(who), q)
    return signal_consensus(S, C, who, q)


def consensus_confidence(S, C):
    """For the decile check of family B (q-free): the confidence of the consensus and its direction at each hour, or NaN.
    K1: min c of M1..M3 when their signs agree. K3: min c of all 6 when signs agree. K2: for the direction d of the highest-c member,
    the third-highest c among the members with sign d, NaN if a member with sign -d has c >= 0.5."""
    sg = _sg(S); cc = np.nan_to_num(C, nan=-1.0); nh = S.shape[1]; out = {}
    for kind, idx in (("K1", [0, 1, 2]), ("K3", list(range(6)))):
        same = (sg[idx] == sg[idx[0]][None, :]).all(0) & (sg[idx[0]] != 0) & (cc[idx] >= 0).all(0)
        out[kind] = (np.where(same, cc[idx].min(0), np.nan), sg[idx[0]])
    best = np.argmax(cc, axis=0); dd = sg[best, np.arange(nh)]
    cand = np.where(sg == dd[None, :], cc, -1.0); third = -np.sort(-cand, axis=0)[2]
    veto = ((sg == -dd[None, :]) & (cc >= VETO_C)).any(0)
    ok = (dd != 0) & (third >= 0) & ~veto
    out["K2"] = (np.where(ok, third, np.nan), dd)
    return out


# ------------------------------------------------------------------ trades
def simulate(sig, hold=HOLD):
    """Non-overlapping trades from a signal series: take the signal of hour k when no trade is open (k >= previous exit) and the exit hour
    k + hold is inside the data. Returns entry hours and sides."""
    nh = len(sig); ent, side = [], []; free = 0
    for k in np.flatnonzero(sig):
        if k >= free and k + hold <= nh:
            ent.append(k); side.append(int(sig[k])); free = k + hold
    return np.asarray(ent, np.int64), np.asarray(side, np.int64)


def hourly_returns(ent, side, Hr, cost_bps, hold=HOLD):
    """Hourly net returns of the trades. Hr = market return of each hour. Cost c/2 at the entry hour and c/2 at the exit hour."""
    nh = len(Hr); d = np.zeros(nh + 1)
    np.add.at(d, ent, side); np.add.at(d, ent + hold, -side)
    pos = np.cumsum(d)[:-1]
    cost = np.zeros(nh); h = cost_bps * 0.5e-4
    np.add.at(cost, ent, h); np.add.at(cost, np.minimum(ent + hold, nh - 1), h)
    return pos * Hr - cost, pos


def trade_net_bps(ent, side, Hr, cost_bps, hold=HOLD):
    cs = np.concatenate([[0.0], np.cumsum(Hr)])
    return side * (cs[ent + hold] - cs[ent]) * 1e4 - cost_bps


def equity_stats(rh):
    return T.equity_stats(rh)


def stats_for(ent, side, Hr, cost_bps, h0, h1):
    """Headline numbers on hour slice [h0, h1): hourly net returns, trades entering in the slice."""
    rh, _ = hourly_returns(ent, side, Hr, cost_bps)
    rh = rh[h0:h1]; st = equity_stats(rh)
    k = (ent >= h0) & (ent < h1); years = (h1 - h0) / HOURS_PER_YEAR
    nb = trade_net_bps(ent[k], side[k], Hr, cost_bps)
    st.update(n_trades=int(k.sum()), trades_per_year=float(k.sum() / years), hit=float(np.mean(nb > 0)) if k.any() else float("nan"),
              mean_bps=float(nb.mean()) if k.any() else float("nan"), years=years)
    return st, rh


def buy_hold(Hr, h0, h1):
    st = equity_stats(Hr[h0:h1]); st["years"] = (h1 - h0) / HOURS_PER_YEAR
    return st


# ------------------------------------------------------------------ null and bootstrap
def null_cum(ent, side, Hr, cost_bps, h0, h1, draws=2000, seed=0, flips=None, hold=HOLD, chunk=500):
    """Random-side null: the same trades (entries, holds, costs), a fair coin for each trade's sign. Cumulative returns over [h0, h1)."""
    from scipy import sparse
    nh = len(Hr); n = len(ent)
    cost = np.zeros(nh); h = cost_bps * 0.5e-4
    np.add.at(cost, ent, h); np.add.at(cost, np.minimum(ent + hold, nh - 1), h)
    rows = (ent[:, None] + np.arange(hold)[None, :]).ravel()
    M = sparse.csr_matrix((Hr[rows], (rows, np.repeat(np.arange(n), hold))), shape=(nh, n))      # market return of trade i in hour j
    rng = np.random.default_rng(seed); out = []
    if flips is not None:
        draws, chunk = len(flips), len(flips)
    for lo in range(0, draws, chunk):
        d = min(chunk, draws - lo)
        fl = flips if flips is not None else np.where(rng.random((d, n)) < 0.5, 1.0, -1.0)
        rh = (M @ fl.T) - cost[:, None]
        out.append(np.expm1(np.log1p(rh[h0:h1]).sum(0)))
    return np.concatenate(out) if out else np.zeros(0)


def week_bootstrap(bps, t_entry, draws=10000, seed=0):
    """Block bootstrap by week of the mean net bps per trade: weeks (of entries) are resampled with replacement.
    Returns the mean, the 95% interval (2.5 / 97.5 percentiles) and the one-sided p = P*(mean <= 0)."""
    bps = np.asarray(bps, float); wk = pd.DatetimeIndex(t_entry).to_period("W").astype(str).values
    if len(bps) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"), n_weeks=0)
    u, inv = np.unique(wk, return_inverse=True)
    s = np.bincount(inv, weights=bps, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(u), (draws, len(u)))
    m = s[ix].sum(1) / c[ix].sum(1)
    return dict(mean=float(bps.mean()), lo=float(np.quantile(m, 0.025)), hi=float(np.quantile(m, 0.975)),
                p=float((np.sum(m <= 0) + 1) / (draws + 1)), n_weeks=int(len(u)))


def holm(pvals):
    """Holm step-down at alpha 0.05 over a dict {name: p}. Returns {name: rejected}."""
    order = sorted(pvals, key=lambda k: pvals[k]); m = len(order); out = {k: False for k in order}
    for i, k in enumerate(order):
        if pvals[k] <= 0.05 / (m - i):
            out[k] = True
        else:
            break
    return out


# ------------------------------------------------------------------ hit by confidence decile
def decile_table(c, direction, fwd_bps, cost_bps=VERDICT_COST, nb=10):
    """Per confidence decile (equal-width bins of c in [0, 1]): decisions with a score, gross hit of the direction against the 4-hour
    open-to-open return, hit after cost, mean net bps. fwd_bps = 4-hour market return in bps (NaN where unknown)."""
    c = np.asarray(c, float); d = np.asarray(direction); f = np.asarray(fwd_bps, float)
    ok = np.isfinite(c) & np.isfinite(f) & (d != 0)
    b = np.minimum((c * nb).astype(int), nb - 1)
    rows = []
    for j in range(nb):
        k = ok & (b == j); n = int(k.sum())
        if n == 0:
            rows.append(dict(decile=j + 1, n=0, hit=float("nan"), hit_net=float("nan"), mean_net_bps=float("nan"))); continue
        g = d[k] * f[k]
        rows.append(dict(decile=j + 1, n=n, hit=float(np.mean(g > 0)), hit_net=float(np.mean(g - cost_bps > 0)), mean_net_bps=float(np.mean(g - cost_bps))))
    return rows


def fwd4_bps(Hr, hold=HOLD):
    cs = np.concatenate([[0.0], np.cumsum(Hr)]); nh = len(Hr); f = np.full(nh, np.nan)
    f[:nh - hold + 1] = (cs[hold:] - cs[:nh - hold + 1]) * 1e4
    return f


# ------------------------------------------------------------------ order guard (same rule as T07)
assert_selection_committed = T.assert_selection_committed

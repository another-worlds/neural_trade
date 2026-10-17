"""T14 candidate M6: regime retrieval (FAISS exact L2 over past anchors). CPU only; run with D:/nt/nt_mm_env/python.exe.

Per hourly UTC anchor `ta` (the harness grid) a regime descriptor from bars strictly before `ta`:
  vp_h, vp_d     percentile of the last hour's / last 24 hours' realised variance among the preceding 365 days' hourly values (min 14 days)
  hsin, hcos     UTC hour of `ta`;  wkend  Saturday / Sunday flag
  str20, str60   ln(close / EMA(span 20 / 60 minutes)) of the last bar before ta, in sigma units (sigma = T13's fixed blend of the S3 variances)
  sr_near, sr_asym  distance to the nearest S/R line (min of T13 S1 hull and S2 pivot distances above and below) and the resistance minus support
                 asymmetry, T13 `sr_lib` at the pair 1 minute x 240 bars, in sigma units
  lrv0, lrv1, lrv2  the last three hourly log realised variances (hours ending at ta, ta-1h, ta-2h)
Neighbours of an anchor come only from anchors j whose label window has ended by (ta - 1 day) at the start of ta's UTC day:
ta_j + 60 min <= day_start(ta) - 1 day <= ta - 1 day (a subset of the rule 'ended before ta minus 1 day'), and only ok anchors with a finite
descriptor. The index is rebuilt each UTC day from that pool, standardised with the pool's own mean / sd. k nearest by exact L2.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
T13 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "T13_support_lines")
for p in (CORE, T13):
    if p not in sys.path:
        sys.path.insert(0, p)

KEYS = ["vp_h", "vp_d", "hsin", "hcos", "wkend", "str20", "str60", "sr_near", "sr_asym", "lrv0", "lrv1", "lrv2"]
MIN_POOL = 500
LABEL_MIN = 60                 # minutes the label window of an anchor extends past ta (A: next hour; B: barrier window T = 60)
GAP = pd.Timedelta(days=1)
PCT_WIN, PCT_MIN = 365 * 24, 14 * 24
EPS = 1e-12


# ------------------------------------------------------------------ descriptors
def hourly_logrv(m1):
    """hourly log realised variance on the UTC hour grid (index = hour START; NaN unless all 60 one-minute returns exist) and the rv itself."""
    c = np.log(m1["close"].astype(float)); lr = c.diff(); lr[(m1.index.to_series().diff() != pd.Timedelta(minutes=1)).values] = np.nan
    sq = lr ** 2; g = sq.groupby(sq.index.floor("h"))
    rv = g.sum(min_count=60); cnt = g.count(); rv[cnt < 60] = np.nan
    full = pd.date_range(rv.index.min(), rv.index.max(), freq="h")
    rv = rv.reindex(full)
    return rv, np.log(rv + EPS)


def trailing_pct(x, win=PCT_WIN, min_n=PCT_MIN):
    """percentile rank (0..1) of x[t] among the finite values x[t-win+1..t] (itself included); NaN when x[t] is NaN or fewer than min_n values."""
    x = np.asarray(x, float); out = np.full(len(x), np.nan)
    for t in range(len(x)):
        if not np.isfinite(x[t]):
            continue
        w = x[max(0, t - win + 1):t + 1]; w = w[np.isfinite(w)]
        if len(w) >= min_n:
            out[t] = (np.sum(w < x[t]) + 0.5 * np.sum(w == x[t])) / len(w)
    return out


def descriptors(m1, ta, sigma, sr_block, clean_windows):
    """[len(ta), 12] float array (KEYS order; NaN where a piece is missing). `m1`: UTC one-minute OHLCV; `ta`: anchor times (hour boundaries, UTC);
    `sigma`: [len(ta)] sd of the 60-minute return. Uses only bars with open time < ta for the row of ta."""
    ta = pd.DatetimeIndex(ta); n = len(ta); out = np.full((n, len(KEYS)), np.nan)
    rv, lrv = hourly_logrv(m1)                                         # indexed by hour start
    day = rv.rolling(24, min_periods=24).sum()                          # day[h] = rv of hours h-23..h (hour starts), complete only
    ldy = np.log(day + EPS)
    ph = pd.Series(trailing_pct(lrv.values), index=lrv.index); pdy = pd.Series(trailing_pct(ldy.values), index=lrv.index)
    h1 = pd.Timedelta(hours=1)
    g = lambda s, k: s.reindex(ta - k * h1).to_numpy(float)              # the hour that started k hours before ta ends at ta - (k-1) hours
    out[:, 0] = g(ph, 1); out[:, 1] = g(pdy, 1)
    out[:, 2] = np.sin(2 * np.pi * ta.hour.values / 24); out[:, 3] = np.cos(2 * np.pi * ta.hour.values / 24)
    out[:, 4] = (ta.dayofweek.values >= 5).astype(float)
    # last bar before ta, EMAs, S/R window
    idx = m1.index; pos = idx.searchsorted(ta, side="left")
    c = m1["close"].astype(float).values; hh = m1["high"].astype(float).values; ll = m1["low"].astype(float).values
    e20 = pd.Series(c).ewm(span=20, adjust=False).mean().values; e60 = pd.Series(c).ewm(span=60, adjust=False).mean().values
    last_ok = (pos >= 1)
    pl = np.maximum(pos - 1, 0)
    last_ok &= np.asarray(idx[pl] == (ta - pd.Timedelta(minutes=1)))
    sg = np.asarray(sigma, float)
    with np.errstate(all="ignore"):
        out[:, 5] = np.where(last_ok, np.clip(np.log(c[pl] / e20[pl]) / sg, -10, 10), np.nan)
        out[:, 6] = np.where(last_ok, np.clip(np.log(c[pl] / e60[pl]) / sg, -10, 10), np.nan)
    tmin = idx.values.astype("datetime64[m]").astype(np.int64)
    clean = clean_windows(tmin, 1, 240)(pos) & last_ok & np.isfinite(sg) & (sg > 0)
    sel = np.where(clean)[0]
    if len(sel):
        X = sr_block(hh, ll, c, pos[sel], sg[sel], 1, 240)
        names = _sr_names()
        col = {nm: X[:, i] for i, nm in enumerate(names)}
        dR = np.minimum(col["s1_dR"], col["s2_dR"]); dS = np.minimum(col["s1_dS"], col["s2_dS"])
        out[sel, 7] = np.minimum(dR, dS); out[sel, 8] = dR - dS
    out[:, 9] = g(lrv, 1); out[:, 10] = g(lrv, 2); out[:, 11] = g(lrv, 3)
    return out


def _sr_names():
    import sr_lib as S
    return S.SR_NAMES


# ------------------------------------------------------------------ retrieval
def knn_pool_index(ta, ok_pool, desc, kmax=100, min_pool=MIN_POOL, shuffle_seed=None, queries=None, return_neighbours=True):
    """For every query anchor i (finite descriptor, optional mask `queries`) the kmax nearest pool anchors (ascending distance).
    Returns (nbr [N, kmax] int, -1 when the pool is smaller than min_pool or the query is not served; dist [N, kmax] sqrt L2 in standardised units; pool_n [N]).
    The pool of the UTC day d: anchors j with ok_pool[j], finite desc and ta_j + 60 min <= d - 1 day."""
    import faiss
    ta = pd.DatetimeIndex(ta); N = len(ta); fin = np.isfinite(desc).all(1)
    q = fin if queries is None else (fin & queries)
    nbr = np.full((N, kmax), -1, np.int64); dist = np.full((N, kmax), np.nan); pool_n = np.zeros(N, np.int64)
    days = ta.normalize(); ud = np.unique(days[q]); end = (ta + pd.Timedelta(minutes=LABEL_MIN)).values
    cand = np.where(ok_pool & fin)[0]; cand_end = end[cand]
    for d in ud:
        lim = np.datetime64(d - GAP)
        pj = cand[cand_end <= lim]
        pool_n_d = len(pj)
        qi = np.where(q & (days == d))[0]
        pool_n[qi] = pool_n_d
        if pool_n_d < min_pool:
            continue
        P = desc[pj]; mu = P.mean(0); sd = P.std(0) + 1e-9
        idx = faiss.IndexFlatL2(P.shape[1]); idx.add(np.ascontiguousarray((P - mu) / sd, dtype="float32"))
        D, I = idx.search(np.ascontiguousarray((desc[qi] - mu) / sd, dtype="float32"), kmax)
        nbr[qi] = pj[I]; dist[qi] = np.sqrt(np.maximum(D, 0))
    return nbr, dist, pool_n


def neighbour_stats(nbr, k, lab, ta=None, shuffle=False, seed=0):
    """mean / sd over the first k neighbours of each label column in `lab` ([N, L]); rows with nbr < 0 give NaN. With shuffle=True the labels are
    permuted among the anchors that can be neighbours (those appearing in `nbr`), the same permutation for all label columns (joint), per UTC day of
    the query, so the information carried by the neighbour identities is destroyed while the keys, distances and label marginals stay."""
    N = nbr.shape[0]; L = lab.shape[1]; sub = nbr[:, :k]; served = (sub >= 0).all(1)
    mean = np.full((N, L), np.nan); sd = np.full((N, L), np.nan)
    if not shuffle:
        v = lab[np.where(sub >= 0, sub, 0)]                           # [N, k, L]
    else:
        days = pd.DatetimeIndex(ta).normalize().values; v = np.zeros((N, k, L))
        for d in np.unique(days[served]):
            qi = np.where(served & (days == d))[0]
            pool = np.unique(sub[qi]); rng = np.random.default_rng(seed + int(d.astype("datetime64[D]").astype(np.int64)))
            perm = pool[rng.permutation(len(pool))]                      # label of anchor pool[i] becomes the label of perm[i]
            mapped = perm[np.searchsorted(pool, sub[qi])]
            v[qi] = lab[mapped]
    mean[served] = v[served].mean(1); sd[served] = v[served].std(1)
    return mean, sd, served

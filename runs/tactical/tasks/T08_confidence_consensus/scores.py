"""T08: the six members' signed hourly scores S[member, hour] on a T07 Grid (reuses T07's members, M1 caches and fitting code).
M1 reads T07's cached walk-forward outputs (D:/nt/nt_t07_cache/m1, read only; bit-identical to a recompute, checked in the report).
Importing members imports lab.py, which chdirs to the repo root: use absolute paths."""
import os
import sys

import numpy as np
import pandas as pd

import t08_lib as L
import t07_lib as T
import members as MB

T07_CACHE = "D:/nt/nt_t07_cache"


def hour_index(grid, t_entry):
    """Grid hour index of datetime64[m] entry times that fall exactly on an hour start; -1 elsewhere."""
    t = np.asarray(t_entry, dtype="datetime64[m]"); mins = (t - grid.t[0]) // np.timedelta64(1, "m")
    k = np.where((mins % 60 == 0) & (mins >= 0) & (mins < grid.n), mins // 60, -1)
    return k.astype(np.int64)


def put(S_row, grid, t_entry, p):
    k = hour_index(grid, t_entry); ok = k >= 0
    S_row[k[ok]] = np.asarray(p, float)[ok] - 0.5


def m1_scores(grid, feat, months):
    """M1: mean of the three members' P(up) at the 15-minute horizon, at the anchor that opens exactly at each hour start."""
    n = len(feat["t_sig"]); p = np.full(n, np.nan)
    for m in months:
        f = f"{T07_CACHE}/m1/{m}.npz"; z = np.load(f)
        if bool(z["ok"]):
            p[z["idx"]] = z["P"].mean(0)[:, 1]
    row = np.full(grid.nh, np.nan); have = ~np.isnan(p)
    put(row, grid, feat["t_entry"][have], p[have])
    return row


def m1_scores_frozen(grid, feat):
    P = np.load(f"{T07_CACHE}/m1/frozen.npz")["P"]
    row = np.full(grid.nh, np.nan); put(row, grid, feat["t_entry"], P.mean(0)[:, 1])
    return row


def hourly_model_scores(grid, F, preds):
    """M2 / M3 from monthly predictions {month: (rows, p)} (T07's hourly_month_preds)."""
    n = len(F["t_sig"]); p = np.full(n, np.nan)
    for m, (rows, pp) in preds.items():
        p[rows] = pp
    have = ~np.isnan(p) & F["valid"]
    t_entry = (pd.DatetimeIndex(F["t_sig"]) + pd.Timedelta(hours=1)).values.astype("datetime64[m]")
    row = np.full(grid.nh, np.nan); put(row, grid, t_entry[have], p[have])
    return row


def hourly_frozen_scores(grid, F, pred):
    p = pred(F["X"]); v = F["valid"]
    t_entry = (pd.DatetimeIndex(F["t_sig"]) + pd.Timedelta(hours=1)).values.astype("datetime64[m]")
    row = np.full(grid.nh, np.nan); put(row, grid, t_entry[v], p[v])
    return row


# ------------------------------------------------------------------ M4 - M6 scores from closed hourly bars
def hourly_hl(raw, grid):
    h = raw[["high", "low"]].resample("60min").agg({"high": "max", "low": "min"}).reindex(grid.hour_t)
    return h["high"].fillna(pd.Series(grid.hc, index=grid.hour_t)).values, h["low"].fillna(pd.Series(grid.hc, index=grid.hour_t)).values


def rule_scores_from_bars(hc, hi, lo):
    """Raw per-bar indicator values (index = the bar they describe, known at its close); the caller shifts by one hour."""
    ema = T.ema_arr
    pc = np.concatenate([[hc[0]], hc[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    atr = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean().values
    m4 = (ema(hc, 20) - ema(hc, 50)) / np.where(atr > 0, atr, np.nan)
    m5 = (50.0 - T.rsi_arr(hc)) / 50.0
    s = pd.Series(hc); mid = s.rolling(20).mean().values; sd = s.rolling(20).std(ddof=0).values
    m6 = (hc - mid) / (2.0 * np.where(sd > 0, sd, np.nan))
    out = np.stack([m4, m5, m6])
    out[:, :60] = np.nan                                                   # warm-up (as T07's M4)
    return out


def rule_scores(grid, hi, lo):
    """M4, M5, M6 scores for the decision of hour k from bars of hours < k (so a score never sees the entry hour)."""
    raw = rule_scores_from_bars(grid.hc, hi, lo)
    S = np.full_like(raw, np.nan); S[:, 1:] = raw[:, :-1]
    return S


# ------------------------------------------------------------------ stages
def build_main(end_excl, last_month, tag):
    """Scores on the main-file grid from its first full hour: monthly walk-forward M1 - M3. Months 2018-12 .. last_month."""
    raw = T.read_main(end_excl)
    if end_excl is not None:
        assert raw.index.max() < pd.Timestamp(end_excl)
    grid = T.Grid(raw)
    months = [str(m) for m in pd.period_range("2018-12", last_month, freq="M")]
    z = np.load(f"{T07_CACHE}/m1_feat_{tag}.npz"); feat = {k: z[k] for k in ("t_sig", "t_entry")}
    assert feat["t_entry"].max() < np.datetime64(pd.Timestamp(end_excl) if end_excl is not None else pd.Timestamp(T.C.CUTOFF) + pd.Timedelta(minutes=1), "m")
    S = np.full((6, grid.nh), np.nan)
    S[0] = m1_scores(grid, feat, months)
    F = MB.hourly_anchors(raw)
    S[1] = hourly_model_scores(grid, F, MB.hourly_month_preds(F, MB.H2, "logreg", 730, months))
    S[2] = hourly_model_scores(grid, F, MB.hourly_month_preds(F, MB.H3, "l1", 365, months))
    hi, lo = hourly_hl(raw, grid)
    S[3:] = rule_scores(grid, hi, lo)
    return dict(grid=grid, S=S, months=months, raw_last=raw.index.max())


def build_fwd():
    """Forward scores with the models frozen at 2025-07-20 (T07's recipe); hours 2025-10-01 .. 2026-09-30 23:00."""
    raw = T.read_fwd(); grid = T.Grid(raw)
    assert raw.index.min() >= T.FWD_START and raw.index.max() <= T.FWD_LAST
    assert not ((raw.index >= T.NEVER_LO) & (raw.index <= T.NEVER_HI)).any()
    end = pd.Timestamp(T.C.CUTOFF) + pd.Timedelta(minutes=1)
    z = np.load(f"{T07_CACHE}/m1_feat_fwd.npz"); feat = {k: z[k] for k in ("t_sig", "t_entry")}
    S = np.full((6, grid.nh), np.nan)
    S[0] = m1_scores_frozen(grid, feat)
    F_tr = MB.hourly_anchors(T.read_main(None)); F = MB.hourly_anchors(raw)
    for i, (H, kind, Td) in ((1, (MB.H2, "logreg", 730)), (2, (MB.H3, "l1", 365))):
        pred, _ntr, _last = MB.hourly_frozen(F_tr, H, kind, Td, end)
        S[i] = hourly_frozen_scores(grid, F, pred)
    hi, lo = hourly_hl(raw, grid)
    S[3:] = rule_scores(grid, hi, lo)
    return dict(grid=grid, S=S)


def save(path, b):
    g = b["grid"]
    np.savez(path, S=b["S"], hour_t=g.hour_t.values.astype("datetime64[m]"), hc=g.hc, ho=g.ho, r=g.r)


def load(path):
    z = np.load(path); return {k: z[k] for k in z.files}

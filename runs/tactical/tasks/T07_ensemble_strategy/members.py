"""T07: the six members as minute position series on a Grid, and the HAR hourly volatility forecast of C5 / C6.
Pure builders; the heavy fits (M1 networks) run in m1_core worker processes and are read from the cache."""
import os
import sys

import numpy as np
import pandas as pd

import t07_lib as T

TAC = T.TAC
sys.path.insert(0, os.path.join(TAC, "recent")); sys.path.insert(0, os.path.join(TAC, "lab"))
sys.path.insert(0, os.path.join(TAC, "tasks", "T02_vol_exposure"))

L = 60
H2, H3 = 4, 1


# ------------------------------------------------------------------ hourly features (M2 / M3)
def hourly_frame(raw):
    return raw[["open", "high", "low", "close", "volume"]].resample("60min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def hourly_anchors(raw):
    """Anchors of the hourly bars: X (rich, 43), t_sig (open time of the window's last bar), valid (bar after it is the next hour),
    r[H] = close[a+H-1] / open[a] - 1 for H = 4 and 1 (next-open)."""
    import lab
    s = hourly_frame(raw); A = s.values.astype(np.float64); n = len(A); idx = s.index
    anch = np.arange(L, n - (H2 - 1))
    W = A[anch[:, None] + np.arange(-L, 0)[None, :]].astype(np.float32)
    X = np.nan_to_num(np.asarray(lab.fs_rich(W), np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    t_sig = idx[anch - 1]; t_entry = idx[anch]
    valid = (t_entry - t_sig == pd.Timedelta(hours=1)) & (idx[anch + H2 - 1] - t_entry == pd.Timedelta(hours=H2 - 1))
    r = {H2: A[anch + H2 - 1, 3] / A[anch, 0] - 1, H3: A[anch, 3] / A[anch, 0] - 1}
    return dict(X=X, t_sig=t_sig, valid=np.asarray(valid), r=r)


def db_of(H):
    return 5e-4 * np.sqrt(H * 60 / 15.0)


def hourly_month_preds(F, H, kind, T_days, months):
    """Monthly walk-forward predictions: {month: (rows, p)}; the model of month m sees only rows whose label bar closed by the month start."""
    import wf
    from sklearn.preprocessing import StandardScaler
    ti = pd.DatetimeIndex(F["t_sig"]); r = F["r"][H]; mask = (np.abs(r) > db_of(H)) & F["valid"]; y = (r > 0).astype(int); out = {}
    for m in months:
        m = pd.Period(m, "M"); m0, m1 = m.start_time, (m + 1).start_time
        tr, te = wf.split_masks(ti, m0, m1, H, 60, T_days); trm = tr & mask
        if trm.sum() < wf.MIN_TRAIN or te.sum() < 100 or len(np.unique(y[trm])) < 2:
            continue
        Xtr = F["X"][trm]; sc = StandardScaler().fit(Xtr)
        g = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -wf.CLIP, wf.CLIP)
        mdl, _ = wf._fit(kind, g(Xtr), y[trm], gap_rows=H + 1)
        rows = np.flatnonzero(te); out[str(m)] = (rows, mdl.predict_proba(g(F["X"][rows]))[:, 1])
    return out


def hourly_frozen(F, H, kind, T_days, end):
    """One model fitted on the T_days before `end` (label bar closed by it); returns a predict(X) closure and the number of training rows."""
    import wf
    from sklearn.preprocessing import StandardScaler
    ti = pd.DatetimeIndex(F["t_sig"]); r = F["r"][H]; end = pd.Timestamp(end)
    tr = np.asarray((ti >= end - pd.Timedelta(days=T_days)) & (ti + pd.Timedelta(hours=H + 1) <= end)) & (np.abs(r) > db_of(H)) & F["valid"]
    y = (r > 0).astype(int); Xtr = F["X"][tr]; sc = StandardScaler().fit(Xtr)
    g = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -wf.CLIP, wf.CLIP)
    mdl, _ = wf._fit(kind, g(Xtr), y[tr], gap_rows=H + 1)
    return (lambda X: mdl.predict_proba(g(X))[:, 1]), int(tr.sum()), str(ti[tr].max() + pd.Timedelta(hours=H + 1))


def month_labels(t_sig):
    return np.asarray(pd.DatetimeIndex(t_sig).to_period("M").astype(str))


def conf_by_month(p, month):
    return {m: np.abs(p[month == m] - 0.5) for m in np.unique(month)}


def trades_to_position(grid, p, t_entry, month, thr, hold_min):
    ent, ex, side = T.signal_trades(t_entry.values.astype("datetime64[m]") if hasattr(t_entry, "values") else t_entry, p, month, thr, hold_min)
    ok = (ent >= grid.t[0]) & (ex <= grid.t[-1] + np.timedelta64(1, "m"))
    return T.positions_from_trades(grid, ent[ok], ex[ok], side[ok]), (ent[ok], ex[ok], side[ok])


def hourly_member_position(grid, F, preds, H, months_all):
    """M2 / M3 position from per-month predictions (`preds` {month: (rows, p)}): thresholds from the previous month's confidence."""
    n = len(F["t_sig"]); p = np.full(n, np.nan)
    for m, (rows, pp) in preds.items():
        p[rows] = pp
    month = month_labels(F["t_sig"]); have = ~np.isnan(p) & F["valid"]
    conf = {m: np.abs(p[(month == m) & have] - 0.5) for m in months_all}
    thr = T.month_thresholds(conf, list(months_all))
    t_entry = pd.DatetimeIndex(F["t_sig"]) + pd.Timedelta(hours=1)
    pos, tr = trades_to_position(grid, p[have], t_entry[have].values.astype("datetime64[m]"), month[have], thr, H * 60)
    return pos, dict(thr=thr, n_trades=len(tr[2]))


# ------------------------------------------------------------------ M1
def m1_position(grid, feat, month_files, months_all):
    """M1 from the saved worker outputs: E1 = mean of the three members' P(up); the 15-minute column; V_prev thresholds; 16-minute holds."""
    n = len(feat["t_sig"]); p = np.full(n, np.nan)
    for m, f in month_files.items():
        z = np.load(f)
        if bool(z["ok"]):
            p[z["idx"]] = z["P"].mean(0)[:, 1]
    month = month_labels(feat["t_sig"]); have = ~np.isnan(p)
    conf = {m: np.abs(p[(month == m) & have] - 0.5) for m in months_all}
    thr = T.month_thresholds(conf, list(months_all))
    pos, tr = trades_to_position(grid, p[have], feat["t_entry"][have], month[have], thr, 16)
    return pos, dict(thr=thr, n_trades=len(tr[2]), p=p)


def m1_position_frozen(grid, feat, P, months_all):
    """Forward M1: P [3, n, 3] from the frozen fit on all forward anchors."""
    p = P.mean(0)[:, 1]; month = month_labels(feat["t_sig"])
    conf = {m: np.abs(p[month == m] - 0.5) for m in months_all}
    thr = T.month_thresholds(conf, list(months_all))
    pos, tr = trades_to_position(grid, p, feat["t_entry"], month, thr, 16)
    return pos, dict(thr=thr, n_trades=len(tr[2]), p=p)


# ------------------------------------------------------------------ M4 - M6 (minute positions from hourly rules)
def rule_positions(grid):
    return [np.repeat(f(grid.hc), 60).astype(np.int8) for f in (T.pos_trend, T.pos_meanrev, T.pos_breakout)]


# ------------------------------------------------------------------ HAR volatility forecast (T02's family; HAR part only)
def _bars(grid):
    import t02lib as T2
    df = pd.DataFrame({"open": grid.o, "close": grid.c}, index=grid.index)
    return T2.Bars(df), T2


def har_fit(B, T2, span_lo, span_hi):
    """T02 Models, HAR part: ridge on log per-minute RV (1h / 4h / 24h / 7d) + hour of day, label = next-24-hour log RV per minute."""
    ah = T2.hour_anchors(B); ah = ah[ah + T2.LABEL_H < B.n]
    ok = (B.t[ah] >= span_lo) & (B.t[ah] + np.timedelta64(T2.LABEL_H, "m") <= span_hi); ah = ah[ok]
    assert len(ah) > 2000, len(ah)
    X = T2.har_feats(B, ah); y = np.log(B.rv(ah, 0, T2.LABEL_H) / T2.LABEL_H + 1e-12); good = np.isfinite(y)
    f = T2._ridge(X[good], y[good])
    return f, float(np.median(np.exp(0.5 * f(X))))


def har_monthly(grid, months):
    """sig_hat per hour of the grid (NaN where not forecast) and sig_med per hour (the month's training-span median), monthly refit on 365 days."""
    B, T2 = _bars(grid); ah_all = T2.hour_anchors(B)
    sig = np.full(grid.nh, np.nan); med = np.full(grid.nh, np.nan)
    for m in pd.period_range(months[0], months[-1], freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        sel = ah_all[(B.t[ah_all] >= np.datetime64(m0, "m")) & (B.t[ah_all] < np.datetime64(m1, "m"))]
        if len(sel) == 0:
            continue
        f, sm = har_fit(B, T2, np.datetime64(m0 - pd.Timedelta(days=T2.TRAIN_DAYS), "m"), np.datetime64(m0, "m"))
        sig[sel // 60] = np.exp(0.5 * f(T2.har_feats(B, sel))); med[sel // 60] = sm
    return sig, med


def har_frozen(grid_main, grid_fwd, end, cal_month):
    """Forward HAR: fitted once on the 365 days before `end` on the main grid; forecasts on the forward grid; sigma* = the median forecast of
    the calibration month (2025-10, from the first hour with a 7-day history)."""
    B, T2 = _bars(grid_main)
    f, _ = har_fit(B, T2, np.datetime64(pd.Timestamp(end) - pd.Timedelta(days=T2.TRAIN_DAYS), "m"), np.datetime64(pd.Timestamp(end), "m"))
    Bf, _ = _bars(grid_fwd); ah = T2.hour_anchors(Bf)
    sig = np.full(grid_fwd.nh, np.nan); sig[ah // 60] = np.exp(0.5 * f(T2.har_feats(Bf, ah)))
    cal = month_labels(grid_fwd.hour_t) == cal_month
    return sig, float(np.nanmedian(sig[cal]))

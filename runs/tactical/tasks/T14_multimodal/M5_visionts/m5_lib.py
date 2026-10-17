"""M5 VisionTS view: torch-free helpers (causal inputs, columns, debias, seasonal baseline). Series per anchor (5-minute bars, UTC):
(i) ln(rv5 + 1e-12) and (ii) ln(1 + volume); context 2016 bars (7 days) strictly before the anchor, period 288, forecast 12 bars."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import numpy as np, pandas as pd
import anchors as AN
CTX, HOR, PERIOD, MIN_FINITE = 2016, 12, 288, 1800
EPS = AN.EPS_RV


def make_frame(store):
    """5-minute frame on the rv5 grid: lrv = ln(rv5+1e-12) (NaN for incomplete bins), lvol = ln(1+volume) (ffilled, causal)."""
    r = store.rv5(); m5 = store.m5().reindex(r.index)
    f = pd.DataFrame(index=r.index)
    f["lrv"] = r["logrv5"]
    f["lvol"] = np.log1p(m5["volume"]).ffill()
    return f


def context(frame, ta):
    """(idx, X) for anchor ta: idx = the 2016 bin open times (all <= ta-5min), X float64 [2016, 2] (lrv, lvol); NaNs ffilled then bfilled.
    None when fewer than 2016 bins exist or fewer than MIN_FINITE finite values."""
    ta = pd.Timestamp(ta)
    end = int(frame.index.searchsorted(ta - pd.Timedelta(minutes=5), side="right"))
    sub = frame.iloc[max(0, end - CTX):end]
    if len(sub) < CTX or sub["lrv"].notna().sum() < MIN_FINITE or sub["lvol"].notna().sum() < MIN_FINITE:
        return None
    assert sub.index[-1] < ta, "history must end before the anchor"
    X = sub[["lrv", "lvol"]].ffill().bfill().to_numpy(float)
    if not np.isfinite(X).all():
        return None
    return sub.index, X


def hourly_from_bars(bars):
    """[n, 12] forecast ln(rv5) bars -> implied hourly log RV on T01's scale."""
    v = np.exp(bars) - EPS
    return np.log(np.clip(v, 0, None).sum(axis=-1) + EPS)


def to_columns(fc_rv, fc_vol, last_obs, debias):
    """fc_rv [n,12] forecast ln rv5, fc_vol [n,12] forecast ln(1+vol), last_obs [n] last observed ln rv5 bar, debias scalar."""
    raw = hourly_from_bars(fc_rv)
    return {"vts_logrv_mean12": fc_rv.mean(1), "vts_logrv_logsum": raw + debias, "vts_vol_mean12": fc_vol.mean(1),
            "vts_logrv_minus_last": fc_rv.mean(1) - last_obs, "point_log_rv": raw + debias}


def debias_constant(ta, implied_raw, y, ok, cap_before=None):
    """mean(target - implied) over TRAINING anchors only: the first 365 days of the source's view anchors, never past `cap_before`."""
    ta = pd.DatetimeIndex(ta); lo = ta.min(); hi = lo + pd.Timedelta(days=365)
    if cap_before is not None:
        hi = min(hi, pd.Timestamp(cap_before))
    m = np.asarray((ta >= lo) & (ta < hi)) & np.asarray(ok, bool) & np.isfinite(implied_raw) & np.isfinite(y)
    return float(np.mean(np.asarray(y)[m] - np.asarray(implied_raw)[m])), int(m.sum()), (str(lo), str(hi))


def seasonal7(store, ta_index, min_days=4):
    """7-day same-hour-of-day mean of hourly log RV per anchor: hours [ta-24h*k, ta-24h*k+1h), k=1..7; each hour's log RV = logrv_1h at the bin
    that closes it (all end before ta). NaN when fewer than `min_days` of the 7 days are finite."""
    r = store.rv5()["logrv_1h"]; ta_index = pd.DatetimeIndex(ta_index)
    M = np.stack([r.reindex(ta_index - pd.Timedelta(hours=24 * k) + pd.Timedelta(minutes=55)).to_numpy(float) for k in range(1, 8)], 1)
    n = np.isfinite(M).sum(1)
    s = np.nansum(np.where(np.isfinite(M), M, 0.0), axis=1)
    return np.where(n >= min_days, s / np.maximum(n, 1), np.nan)

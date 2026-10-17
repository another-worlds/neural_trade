"""M1 Chronos-2 inputs: causal construction of the series fed to the model for one anchor. No torch import (tests run anywhere).

Everything is built from BarStore.m5() / rv5() (complete bins only) and sliced strictly by `ta - 5 min` (bins whose last minute is before ta).
Target = ln(rv5 + 1e-12) per 5-minute bin (NaN for incomplete bins); covariates = log range and log volume of the bin (forward-filled, causal);
M1b adds HAR-style components: rolling mean of the target over 1 day (288 bins) and 1 week (2016 bins), causal per bin.
Known future covariates = sin/cos of the UTC time of day of the 12 forecast bins (also given for the history, as the API requires).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
import anchors as AN  # noqa: E402

CTX, HOR, MIN_FINITE = 512, 12, 256
EPS = AN.EPS_RV
QS = [0.1, 0.5, 0.9]


def make_frame(store, har=False):
    """Full 5-minute frame on the rv5 grid: logrv5, lrange, lvol (+ rm_day, rm_week when har). Per-bin quantities only look at bins up to themselves."""
    r = store.rv5(); m5 = store.m5().reindex(r.index)
    f = pd.DataFrame(index=r.index)
    f["target"] = r["logrv5"]
    f["lrange"] = np.log(np.log(m5["high"] / m5["low"]) + 1e-6).ffill()
    f["lvol"] = np.log1p(m5["volume"]).ffill()
    if har:
        f["rm_day"] = f["target"].rolling(288, min_periods=144).mean().ffill()
        f["rm_week"] = f["target"].rolling(2016, min_periods=1008).mean().ffill()
    return f


def tod_features(index):
    ph = 2 * np.pi * (index.hour.to_numpy() * 60 + index.minute.to_numpy()) / 1440.0
    return np.sin(ph), np.cos(ph)


def window(frame, ta):
    """The model input for anchor `ta` (UTC): dict with 'idx' (history index, all < ta), 'target', covariate arrays, and future time-of-day arrays.
    Returns None when fewer than MIN_FINITE finite target values."""
    ta = pd.Timestamp(ta)
    end = int(frame.index.searchsorted(ta - pd.Timedelta(minutes=5), side="right"))
    sub = frame.iloc[max(0, end - CTX):end]
    if sub["target"].notna().sum() < MIN_FINITE:
        return None
    assert sub.index[-1] < ta, "history must end before the anchor"
    s, c = tod_features(sub.index)
    fut = pd.date_range(ta, periods=HOR, freq="5min"); fs, fc = tod_features(fut)
    past = {"lrange": sub["lrange"].to_numpy(float), "lvol": sub["lvol"].to_numpy(float), "tod_sin": s, "tod_cos": c}
    for k in ("rm_day", "rm_week"):
        if k in sub:
            past[k] = sub[k].to_numpy(float)
    return {"idx": sub.index, "target": sub["target"].to_numpy(float), "past": past, "future": {"tod_sin": fs, "tod_cos": fc}}


def hourly_from_bars(qbars):
    """qbars [n, 12] quantile forecasts of the 5-minute log RV -> implied hourly log RV on T01's scale (the sum of the bars' variances)."""
    v = np.exp(qbars) - EPS
    return np.log(np.clip(v, 0, None).sum(axis=-1) + EPS)


def to_columns(q_hourly):
    """q_hourly [n, 3] (q10, q50, q90 implied hourly log RV) -> view columns."""
    q10, q50, q90 = q_hourly[:, 0], q_hourly[:, 1], q_hourly[:, 2]
    return {"ch_q10": q10, "ch_q50": q50, "ch_q90": q90, "ch_spread": (q90 - q50) - (q50 - q10), "point_log_rv": q50}

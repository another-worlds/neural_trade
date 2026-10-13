"""Forward test of the frozen H62 candidate on Binance klines after 2025-09-30 (AMENDMENT.md). Pure functions + loaders.
Training bars: the cut copy of the main file (rows <= 2025-07-20 23:59). Forward bars: the separate downloaded file.
load_* refuse any bar inside the protected span 2025-07-21..2025-09-30 (assert), so none can reach a computation."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
for p in ("recent", "equity", "lab"):
    sys.path.insert(0, os.path.join(TAC, p))
TRAIN_CSV = "D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv"
FWD_CSV = "D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv"
TRAIN_END = pd.Timestamp("2025-07-20 23:59")        # last training minute
PROT0, PROT1 = pd.Timestamp("2025-07-21"), pd.Timestamp("2025-10-01")
L = 60; BAR_MIN = 60; H = 4; T_DAYS = 730; TOP = 0.10; CLIP = 8.0; C_REG = 0.1
BAR_NS = BAR_MIN * 60 * 10 ** 9
VOLUME_IDX = {31: "mfi_14", 32: "mfi_30", 39: "log_vol_ratio_last", 40: "log_vol_ratio_5", 41: "obv_slope_10", 42: "vwap_dev_20"}


def feature_names():
    n = [f"ret_{k}" for k in (1, 2, 3, 5, 10, 15, 20, 30, 45, 59)] + [f"ema_{p}" for p in (5, 10, 20, 40)] + [f"rsi_{p}" for p in (7, 14, 28)]
    n += ["macd_12_26_9", "macd_5_13_5"] + [f"bb_{p}" for p in (10, 20, 40)] + [f"stoch_{p}" for p in (5, 14, 30, 60)]
    n += [f"cci_{p}" for p in (10, 20, 40)] + [f"atr_{p}" for p in (7, 14)] + [f"mfi_{p}" for p in (14, 30)]
    n += ["candle_body", "candle_upper_wick", "candle_lower_wick", "range_over_sd", "vol_regime_10", "vol_regime_5",
          "log_vol_ratio_last", "log_vol_ratio_5", "obv_slope_10", "vwap_dev_20"]
    assert len(n) == 43
    return n


def _hourly(df):
    return df.set_index("timestamp").resample("60min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def load_train_bars(csv=TRAIN_CSV):
    parts = []
    for ch in pd.read_csv(csv, usecols=["timestamp", "open", "high", "low", "close", "volume"], chunksize=500_000):
        ch["timestamp"] = pd.to_datetime(ch["timestamp"])
        ch = ch[ch["timestamp"] >= TRAIN_END - pd.Timedelta(days=T_DAYS + 5)]
        if len(ch):
            parts.append(ch)
    df = pd.concat(parts)
    assert df["timestamp"].max() <= TRAIN_END, "training bar after the training end"
    return _hourly(df)


def load_fwd_bars(csv=FWD_CSV):
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    assert not ((df["timestamp"] >= PROT0) & (df["timestamp"] < PROT1)).any(), "protected bar present"
    assert df["timestamp"].min() >= PROT1
    return _hourly(df)


def anchors(s, H=H):
    """Features X, signal time t (open of the last window bar, ns), H-bar next-open return r, anchor index. Only bars of s are used."""
    import feats as _f  # noqa  (sets sys.path for lab)
    import lab
    A = s.values.astype(np.float64); n = len(A)
    anch = np.arange(L, n - (H - 1)); t = s.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64)
    W = A[anch[:, None] + np.arange(-L, 0)].astype(np.float32)
    X = np.nan_to_num(np.asarray(lab.fs_rich(W), np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r = A[anch + H - 1, 3] / A[anch, 0] - 1
    return X, t, r, anch


def deadband(H=H):
    return 5e-4 * np.sqrt(H * BAR_MIN / 15.0)


def train_frozen():
    """Fit once on the 730 days ending TRAIN_END (label bar closed by 2025-07-21 00:00). Returns (model, scaler, n_train, span)."""
    import wf
    from sklearn.preprocessing import StandardScaler
    s = load_train_bars(); X, t, r, _ = anchors(s)
    ti = pd.DatetimeIndex(t.astype("datetime64[ns]")); end = TRAIN_END + pd.Timedelta(minutes=1)
    tr = (ti >= end - pd.Timedelta(days=T_DAYS)) & (ti + pd.Timedelta(minutes=(H + 1) * BAR_MIN) <= end) & (np.abs(r) > deadband())
    tr = np.asarray(tr); y = (r > 0).astype(int)
    sc = StandardScaler().fit(np.asarray(X[tr], np.float32))
    mdl, _ = wf._fit("logreg", np.clip(sc.transform(np.asarray(X[tr], np.float32)), -CLIP, CLIP), y[tr], gap_rows=H + 1)
    return mdl, sc, int(tr.sum()), (str(ti[tr].min()), str(ti[tr].max()))


def probs(mdl, sc, X, scaler_override=None):
    """P(up) with the frozen scaler, or with (mean, scale) override (the diagnostic arm)."""
    if scaler_override is None:
        Z = sc.transform(np.asarray(X, np.float32))
    else:
        mu, sd = scaler_override; Z = (np.asarray(X, np.float64) - mu) / sd
    return mdl.predict_proba(np.clip(Z, -CLIP, CLIP))[:, 1]


def month_threshold(conf_prev, top=TOP):
    return float(np.quantile(conf_prev, 1 - top))


def signals(p_by_month, t, thresholds):
    """p_by_month: {month: (idx, p)}. A month's signals are its anchors with |p-0.5| >= thresholds[month] (previous month's quantile)."""
    ent, ex, side, idx, mon = [], [], [], [], []
    for m, (ix, p) in p_by_month.items():
        if m not in thresholds:
            continue
        k = np.abs(p - 0.5) >= thresholds[m]
        t0 = t[ix][k]
        ent.append(t0 + BAR_NS); ex.append(t0 + (H + 1) * BAR_NS); side.append(np.where(p[k] > 0.5, 1, -1)); idx.append(ix[k]); mon += [m] * int(k.sum())
    ent, ex, side, idx = (np.concatenate(a) for a in (ent, ex, side, idx)); mon = np.array(mon)
    o = np.argsort(ent, kind="stable")
    return ent[o], ex[o], side[o], idx[o], mon[o]


def leverage_frontier(net_frac, levs=(1, 2, 3, 5, 10, 20)):
    """net_frac: per-trade net return (fraction, costs included). Additive equity scaled by leverage L (cost scales with L)."""
    out = {}
    for lv in levs:
        eq = 1.0 + np.cumsum(net_frac * lv); pk = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
        out[str(lv)] = {"cum_return_pct": float((eq[-1] - 1) * 100), "max_drawdown_pct": float(((pk - eq) / pk).max() * 100)}
    # largest leverage with max drawdown < 5%, by bisection on the (monotone) drawdown
    lo, hi = 0.0, 200.0
    for _ in range(50):
        mid = (lo + hi) / 2; eq = 1.0 + np.cumsum(net_frac * mid); pk = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
        if ((pk - eq) / pk).max() * 100 < 5.0: lo = mid
        else: hi = mid
    eq = 1.0 + np.cumsum(net_frac * lo); out["max_leverage_dd_lt_5pct"] = {"leverage": float(lo), "cum_return_pct": float((eq[-1] - 1) * 100)}
    return out

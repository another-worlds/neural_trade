"""Smoke-test data helper: last N 1-min bars of the cut file up to 2025-07-20 23:59 (never reads later data; the file ends there).
Reads only the file tail. Timestamps are file time (Bitstamp); the post-2024-09-13 08:00 part is UTC (timefix.py), so the
smoke windows are taken from the UTC part."""
import numpy as np, pandas as pd
CUT = "D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv"

def tail_1m(n=20000):
    with open(CUT, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - n * 90))
        f.readline(); lines = f.read().decode().strip().split("\n")
    df = pd.DataFrame([l.split(",") for l in lines], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in df.columns[1:]: df[c] = df[c].astype(float)
    df = df[df.timestamp <= "2025-07-20 23:59:00"]
    return df.tail(n).reset_index(drop=True)

def bars_5m(n_bars=2048):
    d = tail_1m(n_bars * 5 + 400).set_index("timestamp")
    r = d.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    r["logret"] = np.log(r.close).diff()
    r["logrv"] = np.log((np.log(r.close).diff() ** 2).rolling(12).sum() + 1e-12)
    r = r.dropna()
    return r.tail(n_bars)

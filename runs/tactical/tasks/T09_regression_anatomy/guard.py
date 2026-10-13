"""T09 data guard: the only place that names a data file. Two files are allowed, both read-only:
  CUT_CSV  the 1-minute file physically cut at 2025-07-20 23:59 (H66 recut), and
  FWD_CSV  the Binance forward file that starts at 2025-10-01 (descriptive use only).
The span 2025-07-21 00:00 .. 2025-09-30 23:59 is never read: every loader asserts it holds no bar of that span.
The original D:/nt/neural_trade/Bitcoin_BTCUSDT.csv is never opened by this study."""
import os

import numpy as np
import pandas as pd

CUT_CSV = "D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv"
FWD_CSV = "D:/nt/nt_fwd_data/BTCUSDT_1m_2025-10_onward.csv"
CUT_END = pd.Timestamp("2025-07-20 23:59")
NEVER_FROM, NEVER_TO = pd.Timestamp("2025-07-21 00:00"), pd.Timestamp("2025-09-30 23:59:59")
FWD_FIRST, FWD_LAST = pd.Timestamp("2025-10-01 00:00"), pd.Timestamp("2026-09-30 23:59")
ALLOWED = {os.path.normpath(CUT_CSV), os.path.normpath(FWD_CSV)}


def check_index(ts, name="data"):
    """Raise when any timestamp lies in the never-read span (or, for the cut file, after the cutoff)."""
    ts = pd.DatetimeIndex(ts)
    if len(ts) == 0:
        return ts
    bad = ((ts >= NEVER_FROM) & (ts <= NEVER_TO))
    if bad.any():
        raise AssertionError(f"{name}: {int(bad.sum())} bar(s) in the never-read span {NEVER_FROM}..{NEVER_TO}, first {ts[bad][0]}")
    return ts


def load_ohlcv(which):
    """which = 'cut' | 'fwd'. Returns a DataFrame (open, high, low, close, volume) indexed by the bar's open time."""
    path = {"cut": CUT_CSV, "fwd": FWD_CSV}[which]
    if os.path.normpath(path) not in ALLOWED:
        raise AssertionError(f"{path} is not an allowed file")
    s = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    if which == "cut":
        if s.index.max() > CUT_END:
            raise AssertionError(f"cut file holds a bar after {CUT_END}: {s.index.max()}")
    else:
        s = s[(s.index >= FWD_FIRST) & (s.index <= FWD_LAST)]
        if s.index.min() < FWD_FIRST:
            raise AssertionError("forward file starts before 2025-10-01")
    check_index(s.index, which)
    return s

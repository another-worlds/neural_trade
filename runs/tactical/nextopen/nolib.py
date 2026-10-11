"""Next-open entry variant (H61; descriptive re-measure, no new choice).
In the frozen studies a position opens at the CLOSE of the anchor's last bar (anchor i = window A[i-60:i], entry price A[i-1, 3]).
That fill price is the last traded price of the signal bar, so a minute edge measured against it contains bid-ask bounce (H59).
The variant fills at the OPEN of the next bar, A[i, 0], which is the first price available after the signal; the exit stays where it
was (close of the label bar A[i+H-1, 3] / first barrier touch), so the holding period is one bar shorter and the entry TIMESTAMP is
unchanged (close of bar i-1 = open of bar i).
Columns of A: open, high, low, close, volume."""
import os
import numpy as np
import pandas as pd

CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
BAR_MIN = {"1m": 1, "1h": 60}; HOR = {"1m": 15, "1h": 1}


def entry_close(A, anch):
    return A[anch - 1, 3]


def entry_next_open(A, anch):
    return A[anch, 0]


def label_return(A, anch, H, entry="close"):
    """H-bar forward return: exit at the close of bar anch+H-1; entry at the close of bar anch-1 ('close') or the open of bar anch ('nextopen')."""
    c0 = entry_close(A, anch) if entry == "close" else entry_next_open(A, anch)
    return A[anch + H - 1, 3] / c0 - 1


def load_bars(bar):
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    if bar != "1m":
        s = s.resample(f"{BAR_MIN[bar]}min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return s


def returns_for_cache(bar, t_ns, entry):
    """Forward returns for the anchors of the recent/ cache (t = open time of the anchor's last bar, ns int64)."""
    s = load_bars(bar); A = s.values.astype(np.float64)
    pos = s.index.get_indexer(pd.DatetimeIndex(np.asarray(t_ns, np.int64).astype("datetime64[ns]")))
    assert (pos >= 0).all(), "cache time not in the bars"
    return label_return(A, pos + 1, HOR[bar], entry)

"""Shared data loading and origin selection for the TSFM benchmark on neural_trade BTC 1-min data.

BENCH_DATA=repo (default): the repo CSV, holdout = last 20% of rows.
BENCH_DATA=post: Binance BTCUSDT 1-min 2026-09-20..10-30, origins from 2026-10-16 00:00 UTC on
(after Timestral's 2026-10-15 release, so targets cannot be in any model's training data).
"""
import os

import numpy as np
import pandas as pd

DATA = os.environ.get("BENCH_DATA", "repo")
_SCRATCH = os.environ.get("TSFM_EVAL_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work"))
CSV = {"repo": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "binance_btcusdt_1min_ccxt.csv"),
       "post": f"{_SCRATCH}/data/btcusdt_1m_2026-09-20_2026-10-30.csv"}[DATA]
RESULTS = f"{_SCRATCH}/results" + ("" if DATA == "repo" else f"_{DATA}")
RESULTS = os.environ.get("BENCH_RESULTS", RESULTS)   # optional override (used by the Timestral-LIG scorer)
HORIZONS = [1, 5, 15]          # neural_trade Config.HORIZON_STEPS
PRED_LEN = max(HORIZONS)
TEST_FRAC = 0.2                # repo data: chronological holdout = last 20% of rows
POST_START = "2026-10-16 00:00:00+00:00"
STRIDE = 3                     # one forecast origin every 3 minutes
QLEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
_DF = None


def load():
    global _DF
    df = pd.read_csv(CSV, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    _DF = df
    gaps = df["datetime"].diff().dt.total_seconds().iloc[1:]
    info = {
        "dataset": DATA,
        "rows": len(df),
        "start": str(df["datetime"].iloc[0]),
        "end": str(df["datetime"].iloc[-1]),
        "n_gaps_gt_60s": int((gaps > 60).sum()),
        "n_dupes": int((gaps == 0).sum()),
    }
    return df, info


def origins(n):
    """Index t = last observed bar of the context; targets are t + h."""
    if DATA == "post":
        start = int((_DF["datetime"] < pd.Timestamp(POST_START)).sum())
    else:
        start = int(n * (1 - TEST_FRAC))
    return np.arange(start, n - PRED_LEN, STRIDE)

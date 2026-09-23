"""DataFrame-level preprocessing steps (registered in neural_trade.registries.preprocessors).

``Config.PREPROCESSORS`` lists the steps applied in order; the default reproduces the
original ``DataProcessor.load_and_prepare_data`` exactly:
standardize_ohlcv -> sort_dedupe -> resample_bars -> drop_missing_close.

Scalers are deliberately NOT preprocessors: fitting a scaler on the whole frame, before the
chronological split, leaks test statistics into training. Scaling happens after the split,
on the training block only (neural_trade.data.scaling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
NUMERIC = ("Open", "High", "Low", "Close", "Volume")


def standardize_ohlcv(df: pd.DataFrame, config) -> pd.DataFrame:
    """'timestamp'/'datetime' -> parsed 'timestamp' (bad rows dropped); open..volume -> Open..Volume, numeric."""
    time_column = next((name for name in ("timestamp", "datetime") if name in df.columns), None)
    if time_column is None:
        raise ValueError("Market data must contain a 'timestamp' or 'datetime' column")
    if time_column != "timestamp":
        df = df.rename(columns={time_column: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.rename(columns=OHLCV)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sort_dedupe(df: pd.DataFrame, config) -> pd.DataFrame:
    """Sort by timestamp; keep the last row of duplicated timestamps."""
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")


def resample_bars(df: pd.DataFrame, config) -> pd.DataFrame:
    """Aggregate to Config.RESAMPLE_MINUTES bars (OHLC first/max/min/last, volume sum); add 'Date'."""
    minutes = getattr(config, "RESAMPLE_MINUTES", 1)
    if minutes:
        df = (
            df.set_index("timestamp")
              .resample(f"{minutes}min")
              .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
              .dropna(subset=["Close"])
              .reset_index()
        )
    df["Date"] = df["timestamp"]
    return df


def drop_missing_close(df: pd.DataFrame, config) -> pd.DataFrame:
    """Drop rows without a close; renumber."""
    return df.dropna(subset=["Close"]).reset_index(drop=True)


def strip_currency_symbols(df: pd.DataFrame, config) -> pd.DataFrame:
    """'$1,234.5' -> 1234.5 in the OHLCV columns (exports that format numbers as text)."""
    df = df.copy()
    for col in NUMERIC:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True).replace("", np.nan).astype(float)
    return df


def log_returns(df: pd.DataFrame, config) -> pd.DataFrame:
    """Add 'LogReturn' = log(Close_t / Close_{t-1}) (first row NaN)."""
    df = df.copy()
    df["LogReturn"] = np.log(df["Close"]).diff()
    return df


def add_time_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """Add 'minute_of_day' and 'day_of_week' from 'timestamp'."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    return df

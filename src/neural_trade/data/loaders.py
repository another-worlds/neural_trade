"""Raw market-data loaders (registered in neural_trade.registries.data_loaders).

A loader returns the RAW frame; the Preprocessors pipeline standardises it, and
:func:`validate_ohlcv_frame` checks the result.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "Close")


def load_csv(config, path: Optional[str] = None, read_csv_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Read ``path`` (default Config.CSV_PATH) with ``pandas.read_csv``."""
    return pd.read_csv(path or config.CSV_PATH, **dict(read_csv_kwargs or {}))


def load_parquet(config, path: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Read a Parquet file (needs pyarrow)."""
    return pd.read_parquet(path or config.CSV_PATH, **kwargs)


def load_dataframe(config, frame: pd.DataFrame, **_) -> pd.DataFrame:
    """Use an in-memory frame (tests, notebooks, Predictor.predict_frame)."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("load_dataframe needs a pandas DataFrame")
    return frame.copy()


def validate_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raise unless the standardised frame has timestamp + Close, sorted and non-empty."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"market data is missing columns {missing}; got {list(df.columns)}")
    if df.empty:
        raise ValueError("market data is empty after preprocessing")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("market data timestamps are not sorted ascending")
    return df

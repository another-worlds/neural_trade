"""Raw market-data loaders (registered in neural_trade.registries.data_loaders).

A loader returns the RAW frame; the Preprocessors pipeline standardises it, and
:func:`validate_ohlcv_frame` checks the result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "Close")
# The project root: committed scenarios give CSV_PATH relative to it (where the CLI runs), while a notebook runs with
# notebooks/ as its working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_data_path(path) -> Path:
    """Where to open the data file ``path``: as given when it is absolute or exists from the working directory; else
    the project root's file of that relative path when that exists; else as given (the caller reports it missing).

    Only the file opened changes, never the configured text: CSV_PATH stays relative in the Config, so a cell's
    identity (``scenario.config_identity``) is the same whether the CLI (from the root) or the control panel (from
    notebooks/) runs it, and a sweep started in one resumes in the other."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    cand = Path(PROJECT_ROOT) / p
    return cand if cand.exists() else p


def load_csv(config, path: Optional[str] = None, read_csv_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Read ``path`` (default Config.CSV_PATH, see :func:`resolve_data_path`) with ``pandas.read_csv``."""
    return pd.read_csv(resolve_data_path(path or config.CSV_PATH), **dict(read_csv_kwargs or {}))


def load_parquet(config, path: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Read a Parquet file (needs pyarrow; the path resolves as :func:`resolve_data_path` says)."""
    return pd.read_parquet(resolve_data_path(path or config.CSV_PATH), **kwargs)


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

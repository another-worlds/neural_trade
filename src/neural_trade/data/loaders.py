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


def median_bar_minutes(timestamps) -> Optional[float]:
    """The median spacing of a sorted timestamp column in minutes (None below 3 bars: no spacing to measure)."""
    ts = pd.Series(timestamps)
    if len(ts) < 3:
        return None
    return float(ts.diff().dropna().median() / pd.Timedelta(minutes=1))


def validate_ohlcv_frame(df: pd.DataFrame, bar_minutes: Optional[float] = None) -> pd.DataFrame:
    """Raise unless the standardised frame has timestamp + Close, sorted, non-empty and with a positive
    close. ``bar_minutes`` (the declared bar size, Config.RESAMPLE_MINUTES), when given, must equal the
    data's measured median bar spacing (NT-041): a 5-minute file declared as one-minute bars, or the
    reverse, is refused instead of silently annualising and reading the horizons on the wrong clock."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"market data is missing columns {missing}; got {list(df.columns)}")
    if df.empty:
        raise ValueError("market data is empty after preprocessing")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("market data timestamps are not sorted ascending")
    close = pd.to_numeric(df["Close"], errors="coerce")
    n_bad = int((close <= 0).sum())
    if n_bad:
        raise ValueError(f"market data has {n_bad} bars with a zero or negative close (first at "
                         f"{df['timestamp'][close <= 0].iloc[0]}): a price must be positive")
    if bar_minutes is not None:
        measured = median_bar_minutes(df["timestamp"])
        if measured is not None and abs(measured - float(bar_minutes)) > 1e-6 * max(1.0, float(bar_minutes)):
            raise ValueError(f"the declared bar size is {float(bar_minutes):g} minutes (RESAMPLE_MINUTES) but the "
                             f"data's median bar spacing is {measured:g} minutes: fix the bar size or the file")
    return df

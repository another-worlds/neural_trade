"""Importers into the market store (PLAN A10, DESK-0.8). Only the repo-CSV importer lives here; the ccxt backfill and
the collector come later. The CSV goes through the neural_trade preprocessors (no second cleaner), is validated in
full BEFORE the store is touched (a bad file leaves no partial write), and is written under a labelled source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from neural_trade.desk.data.store import MarketStore

__all__ = ["import_csv", "default_src", "DEFAULT_SYMBOL"]

DEFAULT_SYMBOL = "BTC/USDT"
_STEPS = ["standardize_ohlcv", "sort_dedupe"]
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def default_src(path: Union[str, Path]) -> str:
    """The source label of a repo file: ``repo:<file name>``."""
    return f"repo:{Path(path).name}"


def import_csv(store: MarketStore, path: Union[str, Path], *, symbol: str = DEFAULT_SYMBOL, tf: int = 1,
               src: Optional[str] = None, provenance: Optional[str] = None) -> Dict[str, Any]:
    """Import a ``timestamp|datetime,open,high,low,close,volume`` CSV as ``symbol`` ``tf``-minute bars labelled ``src``
    (default ``repo:<file name>``). A repeated import replaces the same rows (idempotent). ``provenance`` is stored
    beside the source; the default names the file and its span. Raises ``ValueError`` on an unusable file, writing
    nothing. Returns ``src, symbol, tf, n_bars, first_ts, last_ts, provenance``."""
    from neural_trade.data.gaps import timestamps_ns
    from neural_trade.data.loaders import validate_ohlcv_frame
    from neural_trade.data.preprocessors_registry import run_preprocessors

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: no such file")
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # parser errors are many types; one clear message
        raise ValueError(f"{path}: cannot be read as CSV: {exc}") from exc
    df = run_preprocessors(raw, None, _STEPS)             # raises ValueError without a timestamp column
    missing = [c for c in _OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; need timestamp|datetime, open, high, low, close, volume")
    df = df.dropna(subset=_OHLCV).reset_index(drop=True)
    validate_ohlcv_frame(df, float(tf))
    ts = timestamps_ns(df).astype("datetime64[ms]").astype("int64")
    label = src or default_src(path)
    first, last = pd.Timestamp(int(ts[0]), unit="ms", tz="UTC"), pd.Timestamp(int(ts[-1]), unit="ms", tz="UTC")
    text = provenance if provenance is not None else (
        f"{path.name}: {len(df)} bars, {first.isoformat()} to {last.isoformat()}")
    n = store.write_bars(symbol, tf, df, label)
    store.set_provenance(symbol, tf, label, text)
    return {"src": label, "symbol": symbol, "tf": int(tf), "n_bars": n, "first_ts": int(ts[0]), "last_ts": int(ts[-1]),
            "provenance": text}

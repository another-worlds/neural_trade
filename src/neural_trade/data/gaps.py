"""Holes in the bar series (NT-041): detection, the anchors a hole removes, and the record kept in meta.json.

A bar series from an exchange has holes (maintenance, a missing day, a file stitched from two sources), and a
forward-filled export has long flat zero-volume runs instead. Neither is visible to the window builders, which
index bars by position: a window that crosses a hole would silently join two different moments, and a target
read across it would be a price change over an unknown time.

``Config.GAP_POLICY`` decides:

* ``drop`` (default): every input window, past-delta lag and target that would span a hole is not built (the
  anchor is dropped); the number dropped is recorded. The bundled file has no hole, so nothing changes there.
* ``refuse``: a hole anywhere in the prepared data is an error.
* ``ignore``: today's behaviour before NT-041 (holes are not looked at).

A flat run (identical OHLC and zero volume, the footprint of forward filling) is counted, not dropped: the
price really was unchanged as far as the file says, and the window-free plan's rule for runs of an hour or less
is "elapsed time" (NT-041 amendment, D-037).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

FLAT_RUN_MIN_BARS = 60          # a flat zero-volume run counts when it is at least this many bars (about an hour)


def timestamps_ns(df) -> np.ndarray:
    """The frame's ``timestamp`` column as UTC ``datetime64[ns]`` (a naive column is read as UTC)."""
    ts = pd.Series(df["timestamp"])
    if getattr(ts.dtype, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.to_numpy(dtype="datetime64[ns]")


def gap_flags(df, bar_minutes: float, *, tolerance: float = 1.5) -> np.ndarray:
    """Bool array over the frame's bars: True where a bar comes more than ``tolerance`` bar sizes after the previous
    one (a hole between the two). The first bar is False."""
    ns = timestamps_ns(df)
    flags = np.zeros(len(ns), dtype=bool)
    if len(ns) > 1:
        step = np.diff(ns).astype("timedelta64[ns]").astype(np.int64)
        flags[1:] = step > tolerance * float(bar_minutes) * 60e9
    return flags


def flat_runs(df, *, min_bars: int = FLAT_RUN_MIN_BARS) -> Dict[str, int]:
    """Runs of at least ``min_bars`` bars with zero volume and open = high = low = close (forward filling):
    ``{"n_runs", "n_bars", "longest_bars"}``. All zero when the frame has no volume column."""
    out = {"n_runs": 0, "n_bars": 0, "longest_bars": 0}
    need = ("Open", "High", "Low", "Close", "Volume")
    if not all(c in df.columns for c in need) or not len(df):
        return out
    flat = ((df["Volume"].to_numpy(float) == 0) & (df["Open"].to_numpy(float) == df["Close"].to_numpy(float))
            & (df["High"].to_numpy(float) == df["Low"].to_numpy(float))
            & (df["Open"].to_numpy(float) == df["High"].to_numpy(float)))
    edges = np.diff(np.r_[False, flat, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lens = stops - starts
    keep = lens >= int(min_bars)
    out.update(n_runs=int(keep.sum()), n_bars=int(lens[keep].sum()), longest_bars=int(lens.max()) if keep.any() else 0)
    return out


def gap_report(config, df, n_anchors_all: Optional[int] = None, n_anchors_kept: Optional[int] = None,
               flags: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """What the data's holes cost, for meta.json: ``{"policy", "bar_minutes", "n_gaps", "n_missing_bars",
    "longest_gap_bars", "n_windows_dropped", "flat_runs"}``.
    ``n_windows_dropped`` is the number of anchors the policy removed (0 under ``ignore``)."""
    bar = float(config.RESAMPLE_MINUTES)
    flags = gap_flags(df, bar) if flags is None else flags
    ns = timestamps_ns(df)
    n_missing = 0
    longest = 0
    if flags.any():
        steps = np.diff(ns).astype("timedelta64[ns]").astype(np.int64) / (bar * 60e9)
        missing = np.round(steps[flags[1:]]).astype(np.int64) - 1
        n_missing, longest = int(missing.sum()), int(missing.max())
    dropped = 0 if n_anchors_all is None or n_anchors_kept is None else int(n_anchors_all - n_anchors_kept)
    return {"policy": str(config.GAP_POLICY), "bar_minutes": bar, "n_gaps": int(flags.sum()),
            "n_missing_bars": n_missing, "longest_gap_bars": longest, "n_windows_dropped": dropped,
            "flat_runs": flat_runs(df)}


__all__ = ["FLAT_RUN_MIN_BARS", "flat_runs", "gap_flags", "gap_report", "timestamps_ns"]

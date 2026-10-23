"""Market store (PLAN A6, DESK-0.7): ``market.sqlite``, one file separate from ``desk.sqlite``.

Table ``bars(symbol, tf, ts, o, h, l, c, v, src, PRIMARY KEY(symbol, tf, ts))``: ``ts`` is the bar's OPEN time in
UTC integer milliseconds, ``tf`` the bar size in minutes. Table ``sources(symbol, tf, src, first_ts, last_ts,
n_bars, updated)``. The base timeframe is 1 minute; other timeframes are resampled on read with the bin rule of
``data.preprocessors.resample_bars`` and cached per (symbol, tf) with an invalidation stamp. Gaps stay missing rows.

Gaps in A6 (minimum implemented, nothing invented; the list is in the NT-220 report): ``sources`` has no primary
key in A6 (here ``(symbol, tf, src)``); ``limit`` returns the newest bars of the range; ``start`` is inclusive and
``end`` exclusive; ``ranges`` returns the ``sources`` rows; a resampled last bin may be partial; ``gaps`` returns
the holes found by ``data.gaps.gap_flags`` (no per-source ``gap_report``).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from neural_trade.desk.store.db import BUSY_TIMEOUT_MS, DeskDb

__all__ = ["MarketStore", "open_market", "SCHEMA_VERSION", "BASE_TF"]

SCHEMA_VERSION = 1
BASE_TF = 1
_MS_PER_MIN = 60_000
_COLS = ["ts", "o", "h", "l", "c", "v"]
_SHORT = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}

_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS bars (symbol TEXT, tf INTEGER, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, "
    "src TEXT, PRIMARY KEY (symbol, tf, ts))",
    "CREATE TABLE IF NOT EXISTS sources (symbol TEXT, tf INTEGER, src TEXT, first_ts INTEGER, last_ts INTEGER, "
    "n_bars INTEGER, updated INTEGER, PRIMARY KEY (symbol, tf, src))",
]
# Provenance text per (symbol, tf, src): NOT derived from bars, so a write never rebuilds it (NT-221). It is created
# with IF NOT EXISTS outside the schema version, so stores made before it gain it on open.
_PROVENANCE = ("CREATE TABLE IF NOT EXISTS provenance (symbol TEXT, tf INTEGER, src TEXT, text TEXT, "
               "ts_written_ms INTEGER, PRIMARY KEY (symbol, tf, src))")

TimeLike = Union[int, float, str, pd.Timestamp, None]


def _ms(value: TimeLike) -> Optional[int]:
    """Epoch milliseconds (UTC) of an int/float (already ms) or anything pandas parses (naive means UTC)."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    t = pd.Timestamp(value)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // 1_000_000)


def _utc_index(ts_ms) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ts_ms, dtype=np.int64), unit="ms", utc=True))


def _now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").value // 1_000_000)


class MarketStore:
    """One connection to ``market.sqlite``. Not shared across threads; open one per thread or process.

    Every write is its own ``BEGIN IMMEDIATE`` transaction, so concurrent processes queue on the busy timeout.
    """

    def __init__(self, path: Union[str, Path], *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=busy_timeout_ms / 1000, isolation_level=None)
        try:
            self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            # Same open discipline as the desk DB: the WAL switch and the first migration of a fresh file race
            # other processes and SQLite does not always honour the busy timeout there, so retry on "locked".
            DeskDb._retry(lambda: self.conn.execute("PRAGMA journal_mode=WAL").fetchone(), busy_timeout_ms)
            self.conn.execute("PRAGMA synchronous=NORMAL")
            DeskDb._retry(self._migrate, busy_timeout_ms)
            DeskDb._retry(self._ensure_provenance, busy_timeout_ms)
        except BaseException:
            self.conn.close()
            raise
        self._gen = 0                           # bumped by this connection's own writes
        self._cache: Dict[tuple, tuple] = {}    # (symbol, tf) -> (stamp, resampled frame)

    # ---- connection ------------------------------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MarketStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        if self.version > SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} has schema version {self.version}; this code knows up to {SCHEMA_VERSION}")
        if self.version >= SCHEMA_VERSION:
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if self.version < SCHEMA_VERSION:   # another process may have migrated while we waited for the lock
                for stmt in _SCHEMA:
                    self.conn.execute(stmt)
                self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def _ensure_provenance(self) -> None:
        if self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='provenance'").fetchone():
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(_PROVENANCE)
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def set_provenance(self, symbol: str, tf: int, src: str, text: str) -> None:
        """Where the bars labelled ``src`` came from (free text); replaces an earlier text. Bar writes never touch it."""
        sql = ("INSERT INTO provenance (symbol, tf, src, text, ts_written_ms) VALUES (?,?,?,?,?) "
               "ON CONFLICT(symbol, tf, src) DO UPDATE SET text=excluded.text, ts_written_ms=excluded.ts_written_ms")
        args = (symbol, int(tf), src, str(text), _now_ms())
        DeskDb._retry(lambda: self.conn.execute(sql, args), BUSY_TIMEOUT_MS)

    def provenance(self, symbol: str, tf: int = BASE_TF) -> Dict[str, str]:
        """``{src: text}`` of the provenance texts set for (symbol, tf)."""
        cur = self.conn.execute("SELECT src, text FROM provenance WHERE symbol=? AND tf=? ORDER BY src",
                                (symbol, int(tf)))
        return {src: text for src, text in cur.fetchall()}

    def _stamp(self) -> tuple:
        # data_version changes when ANOTHER connection commits; _gen covers this connection's own writes.
        return (self._gen, self.conn.execute("PRAGMA data_version").fetchone()[0])

    # ---- writing ---------------------------------------------------------------------------------------------
    @staticmethod
    def _normalise(bars: pd.DataFrame) -> pd.DataFrame:
        """Rows ``(ts ms, o, h, l, c, v)`` from a frame with ``ts`` (ms), ``timestamp`` or ``datetime`` and open..volume
        (or o..v): the neural_trade cleaners do the parsing, sorting and last-row-wins de-duplication."""
        from neural_trade.data.gaps import timestamps_ns
        from neural_trade.data.preprocessors import sort_dedupe, standardize_ohlcv

        df = bars.copy()
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "timestamp"})
        df = df.rename(columns=_SHORT)
        if "timestamp" not in df.columns and "datetime" not in df.columns:
            raise ValueError("bars need a 'ts' (epoch ms), 'timestamp' or 'datetime' column")
        df = sort_dedupe(standardize_ohlcv(df, None), None)
        missing = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in df.columns]
        if missing:
            raise ValueError(f"bars are missing columns {missing}")
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        out = pd.DataFrame({"ts": timestamps_ns(df).astype("datetime64[ms]").astype(np.int64)})
        for short, col in zip(("o", "h", "l", "c", "v"), ("Open", "High", "Low", "Close", "Volume")):
            out[short] = df[col].to_numpy(dtype=np.float64)
        return out

    def write_bars(self, symbol: str, tf: int, bars: pd.DataFrame, src: str) -> int:
        """Upsert ``bars`` for (symbol, tf) under the label ``src``; a repeated or overlapping write replaces the row
        with the same ``ts`` (idempotent). Returns the number of rows written."""
        tf = int(tf)
        if tf < 1:
            raise ValueError(f"tf must be a positive number of minutes, got {tf}")
        rows = self._normalise(bars)
        if rows.empty:
            return 0
        data = [(symbol, tf, int(t), o, h, lo, c, v, src) for t, o, h, lo, c, v in rows[_COLS].itertuples(index=False)]
        upsert = ("INSERT INTO bars (symbol, tf, ts, o, h, l, c, v, src) VALUES (?,?,?,?,?,?,?,?,?) "
                  "ON CONFLICT(symbol, tf, ts) DO UPDATE SET o=excluded.o, h=excluded.h, l=excluded.l, "
                  "c=excluded.c, v=excluded.v, src=excluded.src")
        DeskDb._retry(lambda: self._write(upsert, data, symbol, tf), BUSY_TIMEOUT_MS)
        self._gen += 1
        return len(data)

    def _write(self, upsert: str, data: list, symbol: str, tf: int) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.executemany(upsert, data)
            # sources is derived from bars, so it can never drift from them.
            self.conn.execute("DELETE FROM sources WHERE symbol=? AND tf=?", (symbol, tf))
            self.conn.execute(
                "INSERT INTO sources (symbol, tf, src, first_ts, last_ts, n_bars, updated) "
                "SELECT symbol, tf, src, MIN(ts), MAX(ts), COUNT(*), ? FROM bars WHERE symbol=? AND tf=? "
                "GROUP BY src", (_now_ms(), symbol, tf))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    # ---- reading ---------------------------------------------------------------------------------------------
    def _read_native(self, symbol: str, tf: int, start: Optional[int], end: Optional[int],
                     limit: Optional[int]) -> pd.DataFrame:
        sql = "SELECT ts, o, h, l, c, v FROM bars WHERE symbol=? AND tf=?"
        args: List[Any] = [symbol, tf]
        if start is not None:
            sql += " AND ts>=?"
            args.append(start)
        if end is not None:
            sql += " AND ts<?"
            args.append(end)
        if limit is not None:
            sql += " ORDER BY ts DESC LIMIT ?"
            args.append(int(limit))
        else:
            sql += " ORDER BY ts"
        rows = self.conn.execute(sql, args).fetchall()
        out = pd.DataFrame(rows, columns=_COLS)
        out["ts"] = out["ts"].astype(np.int64)
        for c in _COLS[1:]:
            out[c] = out[c].astype(np.float64)
        return out.iloc[::-1].reset_index(drop=True) if limit is not None else out

    def read_bars(self, symbol: str, tf: int, start: TimeLike = None, end: TimeLike = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
        """Bars with ``start <= ts < end`` (ms or anything pandas parses) as a frame ``ts, o, h, l, c, v`` in ascending
        order; ``limit`` keeps the newest ``limit`` of them. A timeframe other than the stored ones is resampled
        from the 1-minute bars."""
        tf = int(tf)
        if tf == BASE_TF or self._has_native(symbol, tf):
            return self._read_native(symbol, tf, _ms(start), _ms(end), limit)
        return self.resample(symbol, tf, start, end, limit)

    def _has_native(self, symbol: str, tf: int) -> bool:
        row = self.conn.execute("SELECT 1 FROM sources WHERE symbol=? AND tf=? LIMIT 1", (symbol, tf)).fetchone()
        return row is not None

    def resample(self, symbol: str, tf: int, start: TimeLike = None, end: TimeLike = None,
                 limit: Optional[int] = None) -> pd.DataFrame:
        """``tf``-minute bars from the 1-minute bars: ``resample('{tf}min')`` left-labelled and left-closed (the bins
        of ``resample_bars``), open first, high max, low min, close last, volume sum, empty bins dropped. The full
        resampled series is cached per (symbol, tf) and rebuilt when any connection wrote since."""
        tf = int(tf)
        if tf < 1:
            raise ValueError(f"tf must be a positive number of minutes, got {tf}")
        if tf == BASE_TF:
            return self._read_native(symbol, BASE_TF, _ms(start), _ms(end), limit)
        key, stamp = (symbol, tf), self._stamp()
        hit = self._cache.get(key)
        if hit is None or hit[0] != stamp:
            base = self._read_native(symbol, BASE_TF, None, None, None)
            if base.empty:
                full = base
            else:
                g = base.set_index(_utc_index(base["ts"])).drop(columns="ts").resample(f"{tf}min")
                agg = g.agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna(subset=["c"])
                full = agg.reset_index(drop=True)
                full.insert(0, "ts", np.asarray(agg.index.asi8 // 1_000_000, dtype=np.int64) if agg.index.unit == "ns"
                            else np.asarray(agg.index.as_unit("ms").asi8, dtype=np.int64))
            self._cache[key] = hit = (stamp, full)
        out = hit[1]
        s, e = _ms(start), _ms(end)
        if s is not None:
            out = out[out["ts"] >= s]
        if e is not None:
            out = out[out["ts"] < e]
        if limit is not None:
            out = out.iloc[max(len(out) - int(limit), 0):]
        return out.reset_index(drop=True)

    def last_ts(self, symbol: str, tf: int = BASE_TF) -> Optional[int]:
        """Open time (ms) of the newest stored bar of (symbol, tf), or None."""
        row = self.conn.execute("SELECT MAX(ts) FROM bars WHERE symbol=? AND tf=?", (symbol, int(tf))).fetchone()
        return None if row[0] is None else int(row[0])

    def ranges(self, symbol: str, tf: int = BASE_TF) -> List[Dict[str, Any]]:
        """What is stored for (symbol, tf): one dict per source with ``src, first_ts, last_ts, n_bars, updated``."""
        cur = self.conn.execute("SELECT src, first_ts, last_ts, n_bars, updated FROM sources WHERE symbol=? AND tf=? "
                                "ORDER BY first_ts, src", (symbol, int(tf)))
        names = ("src", "first_ts", "last_ts", "n_bars", "updated")
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def gaps(self, symbol: str, tf: int = BASE_TF, start: TimeLike = None, end: TimeLike = None, *,
             tolerance: float = 1.5) -> List[Dict[str, int]]:
        """Holes in the stored bars (never filled): one dict per hole with ``start_ts`` (the last bar before it),
        ``end_ts`` (the first bar after it) and ``missing`` (bars absent). A hole is a spacing above ``tolerance``
        bar sizes, exactly ``data.gaps.gap_flags``."""
        from neural_trade.data.gaps import gap_flags

        df = self.read_bars(symbol, tf, start, end)
        if len(df) < 2:
            return []
        flags = gap_flags(pd.DataFrame({"timestamp": _utc_index(df["ts"])}), float(tf), tolerance=tolerance)
        ts = df["ts"].to_numpy()
        step = int(tf) * _MS_PER_MIN
        return [{"start_ts": int(ts[i - 1]), "end_ts": int(ts[i]), "missing": int(round((ts[i] - ts[i - 1]) / step)) - 1}
                for i in np.flatnonzero(flags)]

    # ---- export ----------------------------------------------------------------------------------------------
    def export_csv(self, symbol: str, tf: int, start: TimeLike, end: TimeLike, path: Union[str, Path]) -> Dict[str, Any]:
        """Write ``timestamp,open,high,low,close,volume`` (the neural_trade CSV contract; ISO UTC timestamps) and return
        facts shaped like ``dataset_fingerprint``: ``path, sha256, size_bytes, loader, n_rows, n_bars,
        first_timestamp, last_timestamp`` (plus the contract's names ``rows, first, last``)."""
        from neural_trade.experiments.dataset import file_sha256

        df = self.read_bars(symbol, tf, start, end)
        if df.empty:
            raise ValueError(f"no {symbol} {tf}-minute bars between {start} and {end}: nothing to export")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        idx = _utc_index(df["ts"])
        out = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                            "open": df["o"], "high": df["h"], "low": df["l"], "close": df["c"], "volume": df["v"]})
        out.to_csv(path, index=False, lineterminator="\n")
        first, last = idx[0].isoformat(), idx[-1].isoformat()
        return {"path": str(path), "sha256": file_sha256(path), "size_bytes": int(path.stat().st_size), "loader": "csv",
                "n_rows": len(df), "n_bars": len(df), "first_timestamp": first, "last_timestamp": last,
                "rows": len(df), "first": first, "last": last}


def open_market(home: Union[str, Path]) -> MarketStore:
    """The market store of a desk home: ``<home>/market.sqlite``."""
    return MarketStore(Path(home) / "market.sqlite")

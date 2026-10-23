"""Lab datasets (DESK-5.1, PLAN A8): the OHLCV CSVs a training scenario can point at, their fingerprints, and the
export of store bars into ``<home>/datasets/<name>.csv``.

A fingerprint loads and preprocesses the whole file (seconds for a 2017-2025 file), so the list never computes one
for a file over ``SYNC_LIMIT_BYTES`` (50 MB): it returns the cached value or ``fingerprint_pending: True`` and the job
runner calls :func:`compute_fingerprint` later. The cache lives in the desk ``settings`` table (or a process dict when
no repo is given), keyed by (path, size, mtime): a touched or rewritten file is a miss.

Gaps in CONTRACTS (minimum implemented): C5 names ``GET /lab/datasets`` and ``POST /lab/datasets/export`` without
fields; the row fields here (``name, path, size_bytes, mtime_ns, source, spec, fingerprint, fingerprint_pending``)
and the export arguments are this module's own. Import-light: the engine is imported inside functions.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from neural_trade.desk.config import DESK_HOME, PROJECT_ROOT

__all__ = ["SYNC_LIMIT_BYTES", "DatasetError", "list_datasets", "compute_fingerprint", "export_dataset",
           "dataset_dirs", "dataset_spec_facts", "datasets_dir"]

SYNC_LIMIT_BYTES = 50 * 1024 * 1024
_KEY = "lab.fingerprint:"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_TIME_COLS = {"timestamp", "datetime", "date", "time", "open_time"}
_OHLC = {"open", "high", "low", "close"}
_MEMORY: Dict[str, Any] = {}     # the cache when no desk settings repo is given


class DatasetError(ValueError):
    """A dataset name or request the Lab refuses."""


def datasets_dir(home: Optional[Union[str, Path]] = None) -> Path:
    return Path(home if home is not None else DESK_HOME) / "datasets"


def dataset_dirs(root: Optional[Union[str, Path]] = None, home: Optional[Union[str, Path]] = None) -> List[Path]:
    """Where the project keeps data files: the repo root (Config.CSV_PATH is relative to it; the bundled 30-day
    file and the local long file live there) and the desk's own export directory."""
    return [Path(root if root is not None else PROJECT_ROOT), datasets_dir(home)]


def _is_ohlcv(path: Path) -> bool:
    """A CSV whose header has a time column and open/high/low/close (skips e.g. indicator_params_history.csv)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            cols = {c.strip().strip('"').lower() for c in fh.readline().split(",")}
    except OSError:
        return False
    return _OHLC <= cols and bool(cols & _TIME_COLS)


def _cache_get(settings: Any, key: str) -> Any:
    return _MEMORY.get(key) if settings is None else settings.get(key)


def _cache_set(settings: Any, key: str, value: Any) -> None:
    if settings is None:
        _MEMORY[key] = value
    else:
        settings.set(key, value)


def _state(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _cached_fingerprint(settings: Any, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hit = _cache_get(settings, _KEY + state["path"])
    if hit and hit.get("size") == state["size"] and hit.get("mtime_ns") == state["mtime_ns"]:
        return dict(hit["fingerprint"])
    return None


def compute_fingerprint(path: Union[str, Path], settings: Any = None) -> Dict[str, Any]:
    """Compute (always) and cache ``dataset_fingerprint(Config(CSV_PATH=path))`` for the file's current state. Slow for
    a big file: the job runner calls this, never a request handler."""
    from neural_trade.core.config import Config
    from neural_trade.experiments.dataset import dataset_fingerprint

    path = Path(path)
    state = _state(path)
    fp = dataset_fingerprint(Config(CSV_PATH=str(path.resolve())))
    _cache_set(settings, _KEY + state["path"], {"size": state["size"], "mtime_ns": state["mtime_ns"], "fingerprint": fp})
    return dict(fp)


def dataset_spec_facts(path: Union[str, Path], fingerprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``DatasetSpec`` facts of a file under the default Config (symbol, bar minutes, window and horizons in minutes)
    plus rows / first / last when the fingerprint is known (None otherwise)."""
    from neural_trade.core.config import Config
    from neural_trade.core.dataset_spec import DatasetSpec

    spec = DatasetSpec.from_config(Config(CSV_PATH=str(path)))
    fp = fingerprint if fingerprint and "error" not in fingerprint else {}
    return {"symbol": spec.symbol, "quote_currency": spec.quote_currency, "bar_minutes": spec.bar_minutes,
            "window_minutes": spec.window_minutes, "horizon_minutes": spec.horizon_minutes,
            "rows": fp.get("n_bars"), "first": fp.get("first_timestamp"), "last": fp.get("last_timestamp")}


def list_datasets(settings: Any = None, *, root: Optional[Union[str, Path]] = None,
                  home: Optional[Union[str, Path]] = None,
                  sync_limit_bytes: int = SYNC_LIMIT_BYTES) -> List[Dict[str, Any]]:
    """The OHLCV CSVs under :func:`dataset_dirs`, one row each. ``settings`` is the desk db's ``SettingsRepo`` (or None).
    A file of at most ``sync_limit_bytes`` is fingerprinted here (and cached); a bigger one gets its cached
    fingerprint when the file is unchanged, else ``fingerprint_pending: True`` and no fingerprint."""
    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(dataset_dirs(root, home)):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.csv")):
            if not p.is_file() or not _is_ohlcv(p):
                continue
            state = _state(p)
            fp = _cached_fingerprint(settings, state)
            if fp is None and state["size"] <= sync_limit_bytes:
                fp = compute_fingerprint(p, settings)
            rows.append({"name": p.name, "path": str(p), "size_bytes": state["size"], "mtime_ns": state["mtime_ns"],
                         "source": "repo" if i == 0 else "export", "fingerprint_pending": fp is None,
                         "fingerprint": fp, "spec": dataset_spec_facts(p, fp)})
    return rows


def _target(name: str, home: Optional[Union[str, Path]]) -> Path:
    if not isinstance(name, str) or not name:
        raise DatasetError("a dataset needs a name")
    base = name[:-4] if name.lower().endswith(".csv") else name
    if not _NAME.match(base) or ".." in base:
        raise DatasetError(f"bad dataset name {name!r}: letters, digits, '.', '_' and '-' only, no path parts")
    d = datasets_dir(home).resolve()
    target = (d / (base + ".csv")).resolve()
    if target.parent != d:
        raise DatasetError(f"dataset name {name!r} leaves the datasets directory")
    return target


def export_dataset(market: Any, name: str, symbol: str, tf: int, start: Any, end: Any, *,
                   home: Optional[Union[str, Path]] = None, overwrite: bool = False) -> Dict[str, Any]:
    """Export store bars to ``<home>/datasets/<name>.csv`` (``MarketStore.export_csv``) and return its facts. The name is
    a plain file stem (a trailing ``.csv`` is allowed); path parts and absolute names are refused, and an existing
    file is kept unless ``overwrite``."""
    target = _target(name, home)
    if target.exists() and not overwrite:
        raise FileExistsError(f"dataset {target.name} already exists: pass overwrite=True to replace it")
    return market.export_csv(symbol, int(tf), start, end, target)

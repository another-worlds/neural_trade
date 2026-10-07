"""What data a run used: the dataset fingerprint, the setup and the fold layout (NT-026).

    layout = data_layout(config)          # loads and prepares the data once; no windows are built
    layout.fingerprint                    # {"path", "sha256", "size_bytes", "loader", "n_rows",
                                          #  "n_bars", "first_timestamp", "last_timestamp"}
    layout.fold(-1)                       # {"fold", "fold_id", "n_usable_folds", "role", "gap", "blocks"}

The fingerprint identifies the file (sha256 of its bytes) and the bars the model saw (after the
Preprocessors pipeline: count, first and last timestamp). ``setup_of`` records the setup as the
Config states it today: the bar size in minutes (RESAMPLE_MINUTES), LOOKBACK and HORIZON_STEPS
in bars (NT-041 adds the symbol and wall-clock units).

The fold layout is the purged split of data/splits.py (D-005) over the same sequence count the
trainer builds: per usable fold its ``fold_id`` (the TimeSeriesSplit fold, 1-based), the purge gap
and each block's sequence range and first / last decision-bar timestamp. FOLD_INDEX indexes the
usable folds as the trainer does (-1 = the latest). A fold's role is ``test`` when it is the latest
usable fold and ``dev`` otherwise (D-020: rows rank on dev folds only); a FOLD_INDEX outside the
usable folds is refused.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError

FILE_LOADERS = ("csv", "parquet")
_SHA_CACHE: Dict[tuple, str] = {}


def file_sha256(path) -> str:
    """sha256 of a file's bytes (cached per path, size and modification time)."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def setup_of(config: Config) -> Dict[str, Any]:
    """The setup of a run: bar minutes, the window and the horizons in bars (``LOOKBACK``, ``HORIZON_STEPS``,
    the keys the index reads) and, from the dataset spec (NT-041), the instrument, the quote currency, the
    window, horizons and trend lags in wall-clock minutes, and the cost profile (fee, half-spread, slippage
    per side, bps)."""
    from neural_trade.core.dataset_spec import DatasetSpec

    out = {"bar_minutes": int(config.RESAMPLE_MINUTES), "LOOKBACK": int(config.LOOKBACK),
           "HORIZON_STEPS": [int(h) for h in config.HORIZON_STEPS]}
    out.update({k: v for k, v in DatasetSpec.from_config(config).to_dict().items() if k != "bar_minutes"})
    return out


def data_key(config: Config) -> str:
    """The Config values that decide the bars, the sequences and the folds (every field of the
    'data' and 'horizons' groups except FOLD_INDEX, plus the loader and the preprocessors)."""
    from neural_trade.experiments.scenario import short_hash

    keep = {f.name: getattr(config, f.name) for f in fields(config)
            if f.metadata.get("group") in ("data", "horizons") and f.name != "FOLD_INDEX"}
    keep.update(DATA_LOADER=config.DATA_LOADER, PREPROCESSORS=list(config.PREPROCESSORS))
    return short_hash(keep)


def _iso(ts) -> str:
    try:
        return ts.isoformat()
    except AttributeError:
        return str(ts)


@dataclass
class DataLayout:
    fingerprint: Dict[str, Any]
    n_sequences: int
    folds: List[Dict[str, Any]] = field(default_factory=list)   # one per usable fold, chronological

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def fold(self, fold_index: int) -> Dict[str, Any]:
        """The fold at FOLD_INDEX ``fold_index`` (Python indexing over the usable folds, as the
        trainer selects it) with its role; InvalidConfigurationError when it does not exist."""
        n = self.n_folds
        pos = int(fold_index) if int(fold_index) >= 0 else n + int(fold_index)
        if not 0 <= pos < n:
            raise InvalidConfigurationError(
                f"FOLD_INDEX={fold_index} but the data has {n} usable folds (FOLD_INDEX -{n} .. {n - 1}): "
                "add data, raise MAX_SEQUENCE_COUNT or lower N_FOLDS / VAL_FRACTION / CAL_FRACTION")
        info = dict(self.folds[pos])
        info.update(fold=int(fold_index), position=pos, n_usable_folds=n, role="test" if pos == n - 1 else "dev")
        return info


def _prepared(config: Config):
    """``(df, n_rows, path)``: the configured data loaded, preprocessed, validated and cut at DATA_END (the bars
    the model sees; no windows built). InvalidConfigurationError for a loader without a file or a missing file."""
    from neural_trade.data.loaders import resolve_data_path
    from neural_trade.data.processor import DataProcessor, apply_data_end

    loader = str(config.DATA_LOADER)
    if loader not in FILE_LOADERS:
        raise InvalidConfigurationError(f"the experiment engine needs a file loader {FILE_LOADERS} to fingerprint "
                                        f"the data; DATA_LOADER is {loader!r}")
    path = resolve_data_path(config.CSV_PATH)   # the file; the fingerprint keeps the configured text
    if not path.is_file():
        raise InvalidConfigurationError(f"CSV_PATH {config.CSV_PATH!r} does not exist (resolved: {path.resolve()})")
    dp = DataProcessor(config)
    raw = dp.load_raw()
    df = apply_data_end(dp.preprocess(raw), config)
    return df, int(len(raw)), path


def _fingerprint(config: Config, df, n_rows: int, path) -> Dict[str, Any]:
    times = df["timestamp"]
    return {"path": str(config.CSV_PATH), "sha256": file_sha256(path), "size_bytes": int(path.stat().st_size),
            "loader": str(config.DATA_LOADER), "n_rows": n_rows, "n_bars": int(len(df)),
            "first_timestamp": _iso(times.iloc[0]), "last_timestamp": _iso(times.iloc[-1])}


_FP_CACHE: Dict[tuple, Dict[str, Any]] = {}


def dataset_fingerprint(config: Config) -> Dict[str, Any]:
    """What data a run used (NT-041): the file's sha256, the bars the model saw (count, first and last
    timestamp, after the Preprocessors and DATA_END) and the loader. Cached per file state and the Config fields
    that decide the bars, so recording it in every run's meta.json costs one load per dataset. A dataset that
    cannot be fingerprinted (an in-memory loader, a missing file) gives ``{"path", "loader", "error"}``."""
    try:
        from neural_trade.data.loaders import resolve_data_path

        path = resolve_data_path(config.CSV_PATH)
        st = path.stat()
        key = (str(path.resolve()), st.st_size, st.st_mtime_ns, str(config.DATA_LOADER), int(config.RESAMPLE_MINUTES),
               tuple(config.PREPROCESSORS), config.DATA_END, float(config.DATA_END_PROTECTED_DAYS))
    except OSError as exc:
        return {"path": str(config.CSV_PATH), "loader": str(config.DATA_LOADER), "error": f"{type(exc).__name__}: {exc}"}
    if key not in _FP_CACHE:
        try:
            df, n_rows, path = _prepared(config)
            _FP_CACHE[key] = _fingerprint(config, df, n_rows, path)
        except (ValueError, OSError, KeyError) as exc:
            return {"path": str(config.CSV_PATH), "loader": str(config.DATA_LOADER),
                    "error": f"{type(exc).__name__}: {exc}"}
    return dict(_FP_CACHE[key])


def data_layout(config: Config) -> DataLayout:
    """Load and prepare the configured data once and lay out its purged folds (no windows built)."""
    from neural_trade.data.splits import make_purged_splits
    from neural_trade.data.windowing import first_anchor, sequence_anchor_bars

    if int(max(1, config.WINDOW_STEP)) != 1:
        raise InvalidConfigurationError(f"WINDOW_STEP={config.WINDOW_STEP}: the engine's backtest fills at the "
                                        "next bar's open, which needs consecutive decision bars (WINDOW_STEP = 1)")
    df, n_rows, path = _prepared(config)
    close = df["Close"].to_numpy()
    lookback = int(config.LOOKBACK)
    start = first_anchor(lookback, config.EXTENDED_TREND_PERIODS)
    end = int(len(close) - (int(max(config.HORIZON_STEPS)) - 1))
    n_total = len(range(start, end, 1))
    cap = int(config.MAX_SEQUENCE_COUNT or 0)
    n_seq = min(n_total, cap) if cap else n_total
    if n_seq <= 0:
        raise InvalidConfigurationError(f"{config.CSV_PATH}: {len(close)} bars give no sequence for LOOKBACK "
                                        f"{lookback} and HORIZON_STEPS {list(config.HORIZON_STEPS)}")
    try:
        folds = make_purged_splits(n_seq, lookback=lookback, horizon_steps=config.HORIZON_STEPS, window_step=1,
                                   n_folds=int(config.N_FOLDS), val_fraction=float(config.VAL_FRACTION),
                                   cal_fraction=float(config.CAL_FRACTION))
    except ValueError as exc:
        raise InvalidConfigurationError(str(exc)) from exc
    times = df["timestamp"]
    fingerprint = _fingerprint(config, df, n_rows, path)
    out: List[Dict[str, Any]] = []
    for f in folds:
        blocks = {}
        for name in ("train", "val", "cal", "test"):
            idx = getattr(f, name)
            anchors = sequence_anchor_bars(config, len(close), n_total, [int(idx[0]), int(idx[-1])])
            blocks[name] = {"start": int(idx[0]), "stop": int(idx[-1]) + 1, "n": int(len(idx)),
                            "first_timestamp": _iso(times.iloc[int(anchors[0])]),
                            "last_timestamp": _iso(times.iloc[int(anchors[1])])}
        out.append({"fold_id": int(f.fold), "gap": int(f.gap), "blocks": blocks})
    return DataLayout(fingerprint, int(n_seq), out)


class LayoutCache:
    """One DataLayout per data key, so a scenario loads each distinct dataset once."""

    def __init__(self):
        self._layouts: Dict[str, DataLayout] = {}

    def get(self, config: Config) -> DataLayout:
        key = data_key(config)
        if key not in self._layouts:
            self._layouts[key] = data_layout(config)
        return self._layouts[key]


def describe_block(block: Optional[Dict[str, Any]]) -> str:
    if not block:
        return "n/a"
    return f"{block['n']} sequences, {block['first_timestamp']} .. {block['last_timestamp']}"


__all__ = ["DataLayout", "LayoutCache", "data_key", "data_layout", "dataset_fingerprint", "describe_block",
           "file_sha256", "setup_of"]

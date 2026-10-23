"""DESK-5.5: Lab leaderboard JSON, a read-only view of the experiment engine's leaderboard (D-067, CONTRACTS C5
``lab``: ``GET /lab/leaderboard?scenarios=``).

The ranking is the engine's: this module CALLS ``panel_data.build_board`` (which calls ``build_leaderboard``) and only
serialises the ``LeaderboardRow`` tree. D-020: rows are ranked on the development folds only; every test-fold column is
labelled ``split: "test"``, ``rank_eligible: false`` and never orders anything, here or in the engine. Nothing is
written to the store. A missing store or an empty index gives an empty board, not an exception.

``scenarios`` are the board's sweep ids / scenario names (``panel_data.build_board``'s ``sweep_ids``).
The ETag is the hex sha256 of ``store_fingerprint`` plus the query arguments (the API quotes it in the header).

Engine imports sit inside functions so that importing this module loads no TensorFlow.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

RANK_SPLIT = "dev"
RANK_METRIC = "sharpe_net"


def _scalar(v: Any) -> Any:
    """A JSON value: NaN / inf -> None, numpy scalars -> Python, tuples -> lists, Path -> str."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if hasattr(v, "item") and not isinstance(v, (list, tuple, dict)):      # numpy scalar / 0-d array
        try:
            v = v.item()
        except (ValueError, TypeError):
            return str(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, int):
        return v
    if isinstance(v, Path):
        return str(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _scalar(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {str(k): _scalar(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_scalar(x) for x in v]
    return str(v)


def _aggregate(agg: Any, split: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(agg):
        v = getattr(agg, f.name)
        if f.name == "fold_values":
            # int-keyed {column: {fold: value}} -> {column: [[fold, value], ...]} sorted by fold (JSON keys are text)
            out[f.name] = {str(col): [[int(k), _scalar(x)] for k, x in sorted(per.items(), key=lambda kv: int(kv[0]))]
                           for col, per in v.items()}
        else:
            out[f.name] = _scalar(v)
    out["split"] = split
    out["rank_eligible"] = split == RANK_SPLIT
    return out


def row_to_dict(row: Any) -> Dict[str, Any]:
    """One ``LeaderboardRow`` as a JSON-ready dict: every dataclass field, in field order."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(row):
        v = getattr(row, f.name)
        if f.name in ("dev", "test"):
            out[f.name] = _aggregate(v, f.name)
        else:
            out[f.name] = _scalar(v)
    return out


def board_to_dict(board: Any, *, scenarios: Sequence[str] = ()) -> Dict[str, Any]:
    """The payload of a ``panel_data.BoardData`` (``None``: an empty board), rows in the engine's dev ranking."""
    base = {"scenarios": list(scenarios), "ranked_on": RANK_SPLIT, "rank_metric": RANK_METRIC,
            "test_columns_rank": False, "test_label": "test, not used for ranking"}
    if board is None:
        return {**base, "rows": [], "guard_rails": None, "guard_rail_source": "", "n_cells": 0, "n_failed": 0,
                "n_incomplete": 0, "notes": []}
    return {**base, "rows": [row_to_dict(r) for r in board.rows], "guard_rails": _scalar(board.guard_rails),
            "guard_rail_source": board.guard_rail_source, "n_cells": int(board.n_cells),
            "n_failed": int(board.n_failed), "n_incomplete": int(board.n_incomplete),
            "notes": [str(n) for n in board.notes]}


def _store(root):
    from neural_trade.experiments.store import RunStore
    return RunStore(root)


def etag(root, scenarios: Sequence[str] = (), *, specs_dir="configs/scenarios") -> str:
    """Stable hex digest of the store fingerprint and the query arguments; changes when the index or a sweep
    summary is written."""
    from neural_trade.notebook.panel_data import store_fingerprint
    ids = [str(s) for s in scenarios]
    key = json.dumps([store_fingerprint(_store(root), ids), ids, str(specs_dir)], sort_keys=True, default=str)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def leaderboard(root="runs", scenarios: Sequence[str] = (), *, specs_dir="configs/scenarios") -> Dict[str, Any]:
    """The leaderboard JSON of ``scenarios`` in the run store at ``root`` (read-only), with its ``etag``."""
    from neural_trade.notebook.panel_data import build_board, list_sweeps
    ids = [str(s) for s in scenarios]
    store = _store(root)
    tag = etag(root, ids, specs_dir=specs_dir)
    board = None
    if ids and Path(store.index_path).is_file():
        board = build_board(store, list_sweeps(store.root), ids, specs_dir=specs_dir)
    return {**board_to_dict(board, scenarios=ids), "etag": tag}


__all__ = ["board_to_dict", "etag", "leaderboard", "row_to_dict"]

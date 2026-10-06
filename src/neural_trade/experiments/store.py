"""The run store of the experiment engine and its sqlite index (NT-026).

Layout (``root`` defaults to ``runs/``)::

    runs/
      index.sqlite                           the index (derived; rebuild_index() recreates it)
      scenarios/<scenario>/
        <run id>-<cell key>/                 one run directory per cell (experiments.run_context)
          config.yaml meta.json env.json     meta.json: "engine", "dataset", "setup", "blocks"
          status.json metrics.jsonl training_log.csv ...
          eval_report_<dev|test>.json / .md  the scorer's report (experiments.scorer)
          result.json                        status, error, scores: written once, when the cell ends

The run directories are the record: every row of the index is read back from a directory's light
JSON files (meta.json, result.json), so :meth:`RunStore.rebuild_index` gives the same index as
the one the runner kept, and a lost or stale index costs nothing. A directory without result.json
is ``incomplete`` (still running, or interrupted); ``done`` and ``failed`` come from result.json.

Tables: ``runs`` (one row per run directory: identity, provenance, status, headline scores) and
``scores`` (run_id, name, value: every score of result.json, one row each; NULL = not a number).
Engine runs live under ``scenarios/``, which notebook.runs.pick_run leaves out by default.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ENGINE_SUBTREE = "scenarios"
INDEX_NAME = "index.sqlite"
INDEX_SCHEMA_VERSION = 1
RESULT_FILE = "result.json"
STATUSES = ("done", "failed", "incomplete")

# (column, sqlite type); headline scores are copies of the named entries of the scores table
RUN_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("run_id", "TEXT PRIMARY KEY"), ("run_dir", "TEXT NOT NULL"), ("scenario", "TEXT NOT NULL"),
    ("cell_key", "TEXT NOT NULL"), ("configuration", "TEXT"), ("variant", "TEXT"), ("params", "TEXT"),
    ("fold", "INTEGER"), ("fold_id", "INTEGER"), ("role", "TEXT"), ("seed", "INTEGER"),
    ("status", "TEXT NOT NULL"), ("commit_sha", "TEXT"), ("config_hash", "TEXT"), ("settings_hash", "TEXT"),
    ("spec_hash", "TEXT"), ("dataset_sha256", "TEXT"), ("dataset_path", "TEXT"), ("dataset_first", "TEXT"),
    ("dataset_last", "TEXT"), ("dataset_n_bars", "INTEGER"), ("bar_minutes", "REAL"), ("lookback", "INTEGER"),
    ("horizon_steps", "TEXT"), ("strategy", "TEXT"), ("created_utc", "TEXT"), ("finished_utc", "TEXT"),
    ("wall_s", "REAL"), ("sec_per_step", "REAL"), ("error", "TEXT"),
    ("sharpe_net", "REAL"), ("total_return", "REAL"), ("max_drawdown", "REAL"), ("n_trades", "INTEGER"),
    ("buy_and_hold_return", "REAL"), ("random_percentile_return", "REAL"),
)
RUN_FIELDS = tuple(c for c, _ in RUN_COLUMNS)
HEADLINE_SCORES = {"sharpe_net": "backtest/sharpe_net", "total_return": "backtest/total_return",
                   "max_drawdown": "backtest/max_drawdown", "n_trades": "backtest/n_trades",
                   "buy_and_hold_return": "backtest/buy_and_hold/total_return",
                   "random_percentile_return": "backtest/random_same_freq/percentile_total_return"}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def engine_meta(run_dir) -> Optional[Dict[str, Any]]:
    """The ``engine`` section of a run directory's meta.json (None: not an engine run)."""
    try:
        meta = _read_json(Path(run_dir) / "meta.json")
    except (OSError, ValueError):
        return None
    return (meta or {}).get("engine") if isinstance(meta, dict) else None


def is_engine_run_dir(run_dir) -> bool:
    return engine_meta(run_dir) is not None


def _number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        v = float(v)
        return None if math.isnan(v) else v
    return None


STABILITY_VERDICT_FILE = "stability_verdict.json"


def _stability_scores(run_dir: Path) -> Dict[str, Optional[float]]:
    """NT-038: a stability-harness run directory holds ``stability_verdict.json`` (written after the run, by
    experiments.stability); its verdict and each check's value are indexed as ``stability/passed`` (1 or 0)
    and ``stability/<check>`` so the harness's verdicts live in the run store and its index."""
    try:
        doc = _read_json(run_dir / STABILITY_VERDICT_FILE)
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    out: Dict[str, Optional[float]] = {"stability/passed": 1.0 if doc.get("passed") else 0.0}
    for c in doc.get("checks") or []:
        out[f"stability/{c.get('name')}"] = _number(c.get("value"))
    return out


def read_run(run_dir, root) -> Tuple[Dict[str, Any], Dict[str, Optional[float]]]:
    """(runs row, scores) of one engine run directory, from its files only."""
    run_dir, root = Path(run_dir), Path(root)
    meta = _read_json(run_dir / "meta.json") or {}
    eng = meta.get("engine") or {}
    result = _read_json(run_dir / RESULT_FILE)
    ds = meta.get("dataset") or {}
    setup = meta.get("setup") or {}
    scores = {str(k): _number(v) for k, v in ((result or {}).get("scores") or {}).items()}
    scores.update(_stability_scores(run_dir))
    try:
        rel = run_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = run_dir.as_posix()
    error = (result or {}).get("error") or {}
    row = {
        "run_id": meta.get("run_id") or run_dir.name, "run_dir": rel, "scenario": eng.get("scenario"),
        "cell_key": eng.get("cell_key"), "configuration": eng.get("configuration"), "variant": eng.get("variant"),
        "params": json.dumps(eng.get("params") or {}, sort_keys=True), "fold": eng.get("fold"),
        "fold_id": eng.get("fold_id"), "role": eng.get("role"), "seed": eng.get("seed"),
        "status": (result or {}).get("status") or "incomplete", "commit_sha": eng.get("commit"),
        "config_hash": eng.get("config_hash"), "settings_hash": eng.get("settings_hash"),
        "spec_hash": eng.get("spec_hash"), "dataset_sha256": ds.get("sha256"), "dataset_path": ds.get("path"),
        "dataset_first": ds.get("first_timestamp"), "dataset_last": ds.get("last_timestamp"),
        "dataset_n_bars": ds.get("n_bars"), "bar_minutes": setup.get("bar_minutes"),
        "lookback": setup.get("LOOKBACK"),
        "horizon_steps": json.dumps(setup.get("HORIZON_STEPS")) if setup.get("HORIZON_STEPS") is not None else None,
        "strategy": (eng.get("strategy") or {}).get("name"), "created_utc": meta.get("created_utc"),
        "finished_utc": (result or {}).get("finished_utc"), "wall_s": _number((result or {}).get("wall_s")),
        "sec_per_step": _number((result or {}).get("sec_per_step")),
        "error": (f"{error.get('type', 'Error')}: {error.get('message', '')}" if error else None),
    }
    for col, key in HEADLINE_SCORES.items():
        row[col] = scores.get(key)
    return row, scores


class RunIdCollision(ValueError):
    """The same run id for two different cells (NT-182): the index never silently replaces a row."""


class RunIndex:
    """The sqlite index: a cache of the run directories' light files."""

    def __init__(self, path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.path), timeout=30)

    def ensure_schema(self) -> "RunIndex":
        cols = ", ".join(f"{c} {t}" for c, t in RUN_COLUMNS)
        with closing(self._connect()) as con, con:
            con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            con.execute(f"CREATE TABLE IF NOT EXISTS runs ({cols})")
            con.execute("CREATE INDEX IF NOT EXISTS runs_by_cell ON runs (scenario, cell_key)")
            con.execute("CREATE TABLE IF NOT EXISTS scores (run_id TEXT NOT NULL, name TEXT NOT NULL, value REAL, "
                        "PRIMARY KEY (run_id, name))")
            con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(INDEX_SCHEMA_VERSION),))
        return self

    @staticmethod
    def _write(con: sqlite3.Connection, row: Dict[str, Any], scores: Dict[str, Optional[float]]) -> None:
        old = con.execute("SELECT run_dir, scenario, cell_key FROM runs WHERE run_id = ?", (row["run_id"],)).fetchone()
        if old is not None and tuple(old) != (row["run_dir"], row["scenario"], row["cell_key"]):
            raise RunIdCollision(
                f"run id {row['run_id']} belongs to two cells: {old[1]} / {old[2]} in {old[0]} and "
                f"{row['scenario']} / {row['cell_key']} in {row['run_dir']}; the index keeps the first")
        con.execute(f"INSERT OR REPLACE INTO runs ({', '.join(RUN_FIELDS)}) VALUES "
                    f"({', '.join('?' for _ in RUN_FIELDS)})", [row.get(c) for c in RUN_FIELDS])
        con.execute("DELETE FROM scores WHERE run_id = ?", (row["run_id"],))
        con.executemany("INSERT INTO scores (run_id, name, value) VALUES (?, ?, ?)",
                        [(row["run_id"], k, v) for k, v in sorted(scores.items())])

    def add_run(self, run_dir, root) -> Dict[str, Any]:
        """(Re)index one run directory from its files; returns its row."""
        row, scores = read_run(run_dir, root)
        self.ensure_schema()
        with closing(self._connect()) as con, con:
            self._write(con, row, scores)
        return row

    def replace_scenario(self, scenario: str, run_dirs: Iterable[Path], root) -> List[Dict[str, Any]]:
        """Make the scenario's rows exactly those read from ``run_dirs`` (one transaction)."""
        read = [read_run(d, root) for d in run_dirs]
        self.ensure_schema()
        with closing(self._connect()) as con, con:
            old = [r[0] for r in con.execute("SELECT run_id FROM runs WHERE scenario = ?", (scenario,))]
            con.executemany("DELETE FROM scores WHERE run_id = ?", [(r,) for r in old])
            con.execute("DELETE FROM runs WHERE scenario = ?", (scenario,))
            for row, scores in read:
                self._write(con, row, scores)
        return [r for r, _ in read]

    def rows(self, scenario: Optional[str] = None, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Rows of the runs table (optionally of one scenario / status), ordered by scenario, cell, run id."""
        if not self.path.exists():
            return []
        where, args = [], []
        if scenario is not None:
            where.append("scenario = ?")
            args.append(scenario)
        if status is not None:
            where.append("status = ?")
            args.append(status)
        sql = f"SELECT {', '.join(RUN_FIELDS)} FROM runs" + (f" WHERE {' AND '.join(where)}" if where else "") \
            + " ORDER BY scenario, cell_key, run_id"
        with closing(self._connect()) as con:
            return [dict(zip(RUN_FIELDS, r)) for r in con.execute(sql, args)]

    def scores(self, run_id: str) -> Dict[str, Optional[float]]:
        if not self.path.exists():
            return {}
        with closing(self._connect()) as con:
            return dict(con.execute("SELECT name, value FROM scores WHERE run_id = ? ORDER BY name", (run_id,)))

    def dump(self) -> Dict[str, List[tuple]]:
        """Every row of both tables, sorted: two indexes are equal when their dumps are."""
        if not self.path.exists():
            return {"runs": [], "scores": []}
        with closing(self._connect()) as con:
            runs = sorted(con.execute(f"SELECT {', '.join(RUN_FIELDS)} FROM runs"), key=repr)
            scores = sorted(con.execute("SELECT run_id, name, value FROM scores"), key=repr)
        return {"runs": runs, "scores": scores}


class RunStore:
    """``root`` holds the engine's run directories (under ``scenarios/``) and, by default, the index."""

    def __init__(self, root="runs", index_path=None):
        self.root = Path(root)
        self.index_path = Path(index_path) if index_path is not None else self.root / INDEX_NAME

    @property
    def index(self) -> RunIndex:
        return RunIndex(self.index_path)

    def scenario_dir(self, scenario: str) -> Path:
        return self.root / ENGINE_SUBTREE / scenario

    def run_dirs(self, scenario: Optional[str] = None) -> List[Path]:
        """Engine run directories (a meta.json with an ``engine`` section), of one scenario or all."""
        base = self.root / ENGINE_SUBTREE
        parents = [base / scenario] if scenario is not None else (sorted(p for p in base.iterdir() if p.is_dir())
                                                                  if base.is_dir() else [])
        out = []
        for parent in parents:
            if parent.is_dir():
                out += [d for d in sorted(parent.iterdir()) if d.is_dir() and is_engine_run_dir(d)]
        return out

    def sync(self, scenario: str) -> List[Dict[str, Any]]:
        """Re-read the scenario's run directories into the index; returns their rows."""
        return self.index.replace_scenario(scenario, self.run_dirs(scenario), self.root)

    def rebuild_index(self, path=None) -> RunIndex:
        """A fresh index at ``path`` (default: this store's index, replaced atomically) from the run
        directories alone."""
        target = Path(path) if path is not None else self.index_path
        tmp = target.with_name(f"{target.name}.rebuild-{os.getpid()}.tmp")
        if tmp.exists():
            tmp.unlink()
        fresh = RunIndex(tmp).ensure_schema()
        by_scenario: Dict[str, List[Path]] = {}
        for d in self.run_dirs():
            by_scenario.setdefault(d.parent.name, []).append(d)
        for scenario, dirs in by_scenario.items():
            fresh.replace_scenario(scenario, dirs, self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, target)
        return RunIndex(target)


__all__ = ["ENGINE_SUBTREE", "HEADLINE_SCORES", "INDEX_NAME", "RESULT_FILE", "RUN_FIELDS", "RunIdCollision", "RunIndex",
           "RunStore",
           "STATUSES", "engine_meta", "is_engine_run_dir", "read_run"]

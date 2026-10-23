"""DESK-5.3: Lab runs, a read-only, tolerant view of the experiment engine's run store (D-067, CONTRACTS C5 ``lab``).

Serves ``GET /lab/runs?scenario``, ``GET /lab/runs/{id}`` (meta, status, result, eval report) and
``GET /lab/runs/{id}/progress``. Nothing here writes a file: the engine owns every writer (D-065). Files are read
once and closed; a missing or half-written file is treated as absent (the notebook panel's pattern, ``longrun``).

Dev and test columns (D-020): a run's fold has a ``role``. A ``test`` run's numbers are labelled ``split: test``,
``rank_eligible: false`` and are never a sort key; sorting by a score uses the ``dev_*`` columns, which are empty
for test runs (they sort last).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neural_trade.notebook import longrun

SORT_KEYS = ("created_utc", "run_id", "scenario", "state", "fold", "seed", "wall_s",
             "dev_sharpe_net", "dev_total_return", "dev_max_drawdown")
NUMBER_COLUMNS = ("sharpe_net", "total_return", "max_drawdown", "n_trades", "buy_and_hold_return",
                  "random_percentile_return")
MAX_LIMIT = 500
STATES = longrun.STATES


class RunNotFound(KeyError):
    """No run with this id in the store."""

    def __str__(self) -> str:       # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


class LabRuns:
    """Read-only access to one run store (``root``, e.g. ``runs``); ``clock`` is injectable for tests."""

    def __init__(self, root="runs", *, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.clock = clock

    # ---------------------------------------------------------------- rows
    def _rows(self, scenario: Optional[str] = None) -> List[Dict[str, Any]]:
        from neural_trade.experiments.store import RunStore, read_run

        store = RunStore(self.root)
        try:
            rows = store.index.rows(scenario)
        except Exception:       # a missing, locked or half-written index: fall back to the files
            rows = []
        if rows:
            return rows
        out = []
        for d in store.run_dirs(scenario):
            try:
                out.append(read_run(d, self.root)[0])
            except (OSError, ValueError):
                continue
        return out

    def _row(self, run_id: str) -> Dict[str, Any]:
        for r in self._rows():
            if r["run_id"] == run_id:
                return r
        raise RunNotFound(f"no run {run_id!r} in {self.root.as_posix()}")

    def _dir(self, row: Dict[str, Any]) -> Path:
        p = Path(row["run_dir"])
        return p if p.is_absolute() else self.root / p

    # ---------------------------------------------------------------- list
    def _item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        state = self._progress_of(self._dir(row), row)["state"]
        split = "test" if row.get("role") == "test" else "dev"
        other = "dev" if split == "test" else "test"
        nums = {c: row.get(c) for c in NUMBER_COLUMNS}
        item = {k: row.get(k) for k in ("run_id", "scenario", "cell_key", "configuration", "fold", "seed", "role",
                                        "status", "created_utc", "finished_utc", "wall_s", "error")}
        item.update(state=state, split=split, rank_eligible=split == "dev",
                    key_numbers={"split": split, "rank_eligible": split == "dev", **nums})
        for c in NUMBER_COLUMNS:       # explicit columns: a test run's numbers never sit in a dev_* column
            item[f"{split}_{c}"] = nums[c]
            item[f"{other}_{c}"] = None
        return item

    def list_runs(self, scenario: Optional[str] = None, *, sort: str = "created_utc", descending: bool = True,
                  offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        """``{items, total, offset, limit, sort, descending}``. ``sort`` is one of :data:`SORT_KEYS`; a test column
        is refused (D-020). Rows without a value sort last in either direction."""
        if sort not in SORT_KEYS:
            raise ValueError(f"cannot sort by {sort!r}: allowed {', '.join(SORT_KEYS)} "
                             "(test columns never rank, D-020)")
        offset, limit = max(0, int(offset)), min(max(1, int(limit)), MAX_LIMIT)
        items = [self._item(r) for r in self._rows(scenario)]
        have = [i for i in items if i.get(sort) is not None]
        none = [i for i in items if i.get(sort) is None]
        have.sort(key=lambda i: (i[sort], i["run_id"]), reverse=descending)
        ordered = have + none
        return {"items": ordered[offset:offset + limit], "total": len(ordered), "offset": offset, "limit": limit,
                "sort": sort, "descending": bool(descending)}

    # ---------------------------------------------------------------- detail
    def detail(self, run_id: str) -> Dict[str, Any]:
        """The run's light files: meta, status, the last metrics rows, result, every ``eval_report_<role>.json``,
        plus the state and the index scores."""
        from neural_trade.experiments.store import RESULT_FILE, RunStore

        row = self._row(run_id)
        d = self._dir(row)
        reports = {}
        for p in sorted(d.glob("eval_report_*.json")):
            doc = longrun._read_json(p)
            if doc is not None:
                reports[p.stem[len("eval_report_"):]] = doc
        prog = self._progress_of(d, row)
        try:
            scores = RunStore(self.root).index.scores(run_id)
        except Exception:
            scores = {}
        return {"run_id": row["run_id"], "scenario": row.get("scenario"), "run_dir": row["run_dir"],
                "state": prog["state"], "reason": prog["reason"], "meta": longrun._read_json(d / "meta.json"),
                "status": longrun._read_json(d / "status.json"),
                "metrics_tail": longrun._epoch_rows(d)[-longrun.LOG_TAIL_LINES:],
                "result": longrun._read_json(d / RESULT_FILE), "eval_report": reports, "scores": scores}

    # ---------------------------------------------------------------- progress
    def progress(self, run_id: str) -> Dict[str, Any]:
        row = self._row(run_id)
        return self._progress_of(self._dir(row), row)

    def _progress_of(self, d: Path, row: Dict[str, Any]) -> Dict[str, Any]:
        """State of one run directory: not started / running / stalled / done / failed (longrun's rules, per run)."""
        from neural_trade.experiments.store import RESULT_FILE

        now = float(self.clock())
        out: Dict[str, Any] = {"run_id": row["run_id"], "state": "not started", "reason": "", "epochs_done": 0,
                               "epochs_total": None, "elapsed_s": None, "median_epoch_s": None, "eta_s": None,
                               "last_update_s": None, "stall_after_s": None, "pid": None, "pid_alive": False,
                               "best_val_loss": None, "log_tail": []}
        rows = longrun._epoch_rows(d)
        status = longrun._read_json(d / "status.json") or {}
        result = longrun._read_json(d / RESULT_FILE)
        total = longrun._config_value(d, "EPOCHS")
        done = max(len(rows), int(longrun._num(status.get("epochs_completed")) or 0))
        med = longrun._median(r.get("epoch_seconds") for r in rows)
        updated = longrun._last_update(d)
        created = longrun._created(d)
        out.update(epochs_done=done, epochs_total=int(total) if isinstance(total, (int, float)) else None,
                   median_epoch_s=med, stall_after_s=longrun.stall_after(med),
                   last_update_s=(now - updated) if updated is not None else None)
        vals = [v for v in (longrun._num(r.get("val_loss")) for r in rows) if v is not None]
        out["best_val_loss"] = min(vals) if vals else None
        if result is not None:
            ok = result.get("status") == "done"
            err = result.get("error") or {}
            out.update(state="done" if ok else "failed", eta_s=0.0, elapsed_s=longrun._num(result.get("wall_s")),
                       reason="scored: result.json is written" if ok else
                       f"result.json says {result.get('status')}: {err.get('type', '')} "
                       f"{err.get('message', '')}".strip())
            return out
        if not rows and not status:
            out["reason"] = "no status.json or epoch rows yet"
            return out
        launches = longrun.launch_records(d.parent)
        last = launches[-1] if launches else None
        alive = bool(last) and longrun.pid_alive(last.get("pid"))
        if last:
            out.update(pid=last.get("pid"), pid_alive=alive,
                       log_tail=longrun.tail(last["log"]) if last.get("log") else [])
        out["elapsed_s"] = (now - created) if created is not None else longrun._num(status.get("elapsed_seconds"))
        out["eta_s"] = longrun.eta_seconds(done, out["epochs_total"], med)
        launched_this = bool(last) and created is not None and float(last.get("started", 0)) <= created + 60
        age, limit = out["last_update_s"], out["stall_after_s"]
        if last and launched_this and not alive:
            out.update(state="failed", reason="the launched process ended but the run has no result.json "
                                              "(interrupted or crashed)")
        elif alive or (age is not None and age <= limit):
            out.update(state="running", reason=f"epoch {done + 1} in progress")
            if alive and age is not None and age > limit:
                out.update(state="stalled", reason=f"the process is alive but nothing was written for "
                                                   f"{age / 60:.0f} min (limit {limit / 60:.0f} min)")
        else:
            out.update(state="stalled", reason=f"no update for {(age or 0) / 60:.0f} min (limit {limit / 60:.0f} min) "
                                               "and no live process recorded")
        return out


__all__ = ["LabRuns", "MAX_LIMIT", "NUMBER_COLUMNS", "RunNotFound", "SORT_KEYS", "STATES"]

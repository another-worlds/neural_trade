"""The experiment engine's resumable runner (NT-026).

    from neural_trade.experiments.runner import Runner
    report = Runner.from_spec("configs/scenarios/reference.yaml", store="runs").run()
    # or: neural-trade scenario run configs/scenarios/reference.yaml [--max-cells N]

**Plan first.** Before anything trains, :meth:`Runner.plan` checks the whole scenario: the spec
(unknown keys, every cell's Config, the strategy and cost settings), the components each Config
names (registries), and the data (the file exists and is fingerprinted; every fold is a usable
fold of the purged split; WINDOW_STEP = 1 for next-open fills). Any error stops it with nothing
written.

**One cell, one run directory.** Each pending (configuration, fold, seed) cell trains into a new
directory ``<store>/scenarios/<scenario>/<run id>-<cell key>/`` (experiments.run_context), is
scored by experiments.scorer on its fold's out-of-sample block, and ends with ``result.json``
(status, error, scores); the index is updated from those files after every cell. The runner never
writes into an existing directory, never overwrites a file and deletes nothing: a name that is
taken gets a ``-2``, ``-3`` ... suffix.

**Resume.** Re-running the same scenario skips every cell that already has a finished run
(``done``; ``failed`` too unless ``retry_failed``) with the same config hash and settings hash.
A cell whose directory has no result.json (interrupted) is kept as it is, indexed as
``incomplete``, and trained again into a new directory. Cells run one at a time in this process;
each starts from a cleared Keras session and a seeded state, so a resumed scenario reproduces
the scores of an uninterrupted one. Run one runner per scenario at a time.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neural_trade.core.config import Config
from neural_trade.experiments.dataset import LayoutCache, setup_of
from neural_trade.experiments.scenario import Cell, Scenario, ScenarioError, config_hash
from neural_trade.experiments.store import RESULT_FILE, RunStore

logger = logging.getLogger(__name__)

ENGINE_SCHEMA_VERSION = 1
FINISHED = ("done", "failed")


def train_cell(ctx, *, calibrate: bool = True, save_artifacts: bool = False):
    """The default trainer: ``train_and_evaluate`` on the cell's run context, from a cleared Keras
    session (unseeded op seeds come from per-process counters, see tests/test_reproducibility.py).
    It fits the calibration block (the scorer fits the strategy there) and never warm-starts."""
    import tensorflow as tf

    from neural_trade.training.trainer import train_and_evaluate

    tf.keras.backend.clear_session()
    return train_and_evaluate(config=ctx.config, run_context=ctx, force=True, calibrate=calibrate,
                              fit_calibration=True, save_artifacts=save_artifacts)


@dataclass
class PlannedCell:
    cell: Cell
    config: Config
    config_hash: str
    fold: Dict[str, Any]                 # DataLayout.fold(): fold, fold_id, n_usable_folds, role, gap, blocks
    dataset: Dict[str, Any]
    setup: Dict[str, Any]
    state: str = "pending"               # pending | done | failed (from the index)
    runs: List[str] = field(default_factory=list)   # run ids of this cell's earlier attempts

    @property
    def key(self) -> str:
        return self.cell.key

    @property
    def role(self) -> str:
        return self.fold["role"]


@dataclass
class RunReport:
    scenario: str
    store: str
    index: str
    n_cells: int
    skipped: List[str] = field(default_factory=list)          # finished before this call
    ran: List[Dict[str, str]] = field(default_factory=list)   # {"cell", "status", "run_dir"}
    pending: List[str] = field(default_factory=list)          # not run (max_cells reached)

    @property
    def failed(self) -> List[str]:
        return [r["cell"] for r in self.ran if r["status"] != "done"]

    def to_dict(self) -> Dict[str, Any]:
        return {"scenario": self.scenario, "store": self.store, "index": self.index, "n_cells": self.n_cells,
                "skipped": self.skipped, "ran": self.ran, "failed": self.failed, "pending": self.pending}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_new(path: Path, obj: Any) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=2, default=str))


class Runner:
    """Runs a :class:`~neural_trade.experiments.scenario.Scenario` into a :class:`RunStore`.

    ``trainer(ctx, *, calibrate, save_artifacts)`` returns a TrainResult for the run context's
    config (default :func:`train_cell`); tests inject a fast one.
    """

    def __init__(self, scenario: Scenario, store="runs", *, index_path=None,
                 trainer: Optional[Callable[..., Any]] = None, check_components: bool = True):
        self.scenario = scenario
        self.store = store if isinstance(store, RunStore) else RunStore(store, index_path)
        self.trainer = trainer if trainer is not None else train_cell
        self.check_components = check_components
        self._layouts = LayoutCache()

    @classmethod
    def from_spec(cls, path, store="runs", **kwargs) -> "Runner":
        return cls(Scenario.from_yaml(path), store, **kwargs)

    # -------------------------------------------------------------- plan
    def plan(self) -> List[PlannedCell]:
        """Validate everything and return the cells with their state; writes nothing but the index
        (re-read from the scenario's run directories when that directory exists)."""
        sc = self.scenario
        pairs = sc.validate()
        if self.check_components:
            self._check_components([cfg for _, cfg in pairs])
        planned: List[PlannedCell] = []
        seen: Dict[tuple, str] = {}
        for cell, cfg in pairs:
            try:
                layout = self._layouts.get(cfg)
                fold = layout.fold(cell.fold)
            except ScenarioError:
                raise
            except (ValueError, OSError) as exc:
                raise ScenarioError(f"{sc.where()}: cell {cell.key}: {exc}") from exc
            same = (cell.configuration.name, fold["position"], cell.seed)
            if same in seen:
                raise ScenarioError(f"{sc.where()}: folds {seen[same]} and {cell.fold} are the same fold of the data")
            seen[same] = str(cell.fold)
            planned.append(PlannedCell(cell, cfg, config_hash(cfg), fold, dict(layout.fingerprint), setup_of(cfg)))
        self._mark_states(planned)
        return planned

    def _check_components(self, configs: List[Config]) -> None:
        from neural_trade.registries import load_all, validate_config_components

        load_all(None, strict=False)
        for plugins in sorted({c.PLUGINS_DIR for c in configs if c.PLUGINS_DIR}):
            load_all(None, plugins_dir=plugins, strict=False)
        missing = sorted({m for c in configs for m in validate_config_components(c)})
        if missing:
            raise ScenarioError(f"{self.scenario.where()}: the configuration names components that are not "
                                f"registered: {missing}")

    def _mark_states(self, planned: List[PlannedCell]) -> None:
        rows = self.store.sync(self.scenario.name) if self.store.scenario_dir(self.scenario.name).is_dir() else []
        settings = self.scenario.settings_hash
        by_cell: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_cell.setdefault(r["cell_key"], []).append(r)
        for pc in planned:
            attempts = [r for r in by_cell.get(pc.key, [])
                        if r["config_hash"] == pc.config_hash and r["settings_hash"] == settings]
            pc.runs = [r["run_id"] for r in attempts]
            states = {r["status"] for r in attempts}
            pc.state = "done" if "done" in states else ("failed" if "failed" in states else "pending")

    # -------------------------------------------------------------- run
    def run(self, *, max_cells: Optional[int] = None, retry_failed: bool = False) -> RunReport:
        """Run every pending cell (at most ``max_cells`` in this call); see the module docstring."""
        planned = self.plan()
        sc = self.scenario
        report = RunReport(sc.name, str(self.store.root), str(self.store.index_path), len(planned))
        todo = []
        for pc in planned:
            if pc.state == "done" or (pc.state == "failed" and not retry_failed):
                report.skipped.append(pc.key)
            else:
                todo.append(pc)
        if todo:
            self.store.scenario_dir(sc.name).mkdir(parents=True, exist_ok=True)
            self._save_spec()
        for i, pc in enumerate(todo):
            if max_cells is not None and i >= max_cells:
                report.pending += [p.key for p in todo[i:]]
                logger.info("[scenario %s] stopped after %d cells (max_cells); %d pending: run the same command "
                            "again to resume", sc.name, i, len(todo) - i)
                break
            logger.info("[scenario %s] cell %d/%d %s (%s fold %s, seed %s)", sc.name, i + 1, len(todo), pc.key,
                        pc.role, pc.cell.fold, pc.cell.seed)
            run_dir, status = self.run_cell(pc)
            report.ran.append({"cell": pc.key, "status": status, "run_dir": str(run_dir)})
        return report

    def _save_spec(self) -> None:
        """The normalised spec as the runs used it: specs/<spec hash>.json in the scenario directory."""
        d = self.store.scenario_dir(self.scenario.name) / "specs"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.scenario.spec_hash}.json"
        if not path.exists():
            _write_new(path, self.scenario.to_dict())

    def _fold_meta(self, pc: PlannedCell) -> Dict[str, Any]:
        return {"fold": pc.cell.fold, "fold_id": pc.fold["fold_id"], "n_usable_folds": pc.fold["n_usable_folds"]}

    def _meta(self, pc: PlannedCell) -> Dict[str, Any]:
        """meta.json's engine entries: the cell's identity and provenance, the dataset fingerprint, the
        setup, and the fold's blocks (sequence ranges and anchor timestamps)."""
        from neural_trade.utils.env import git_sha

        sc = self.scenario
        engine = {"schema_version": ENGINE_SCHEMA_VERSION, "scenario": sc.name, "cell_key": pc.key,
                  "configuration": pc.cell.configuration.name, "variant": pc.cell.configuration.variant,
                  "params": pc.cell.configuration.params, **self._fold_meta(pc), "role": pc.role,
                  "seed": pc.cell.seed, "config_hash": pc.config_hash, "settings_hash": sc.settings_hash,
                  "spec_hash": sc.spec_hash, "commit": git_sha(),
                  "strategy": {"name": sc.strategy.name, "params": sc.strategy.params}, "backtest": sc.backtest,
                  "run": {"calibrate": sc.run.calibrate, "save_artifacts": sc.run.save_artifacts},
                  "spec": str(sc.source) if sc.source is not None else None}
        return {"engine": engine, "dataset": pc.dataset, "setup": pc.setup,
                "blocks": {**self._fold_meta(pc), "gap": pc.fold["gap"], **pc.fold["blocks"]}}

    def _create_context(self, pc: PlannedCell, meta: Dict[str, Any]):
        from neural_trade.experiments.run_context import RunContext

        root = self.store.scenario_dir(self.scenario.name)
        tags = ["scenario", self.scenario.name, pc.role]
        for attempt in range(1, 1000):
            name = pc.key if attempt == 1 else f"{pc.key}-{attempt}"
            try:
                return RunContext.create(pc.config, root=root, seed=pc.cell.seed, tags=tags, name=name, meta=meta)
            except FileExistsError:
                continue       # the name is taken (same second, same cell): never write into that directory
        raise RuntimeError(f"no free run directory name for {pc.key} under {root}")

    def run_cell(self, pc: PlannedCell):
        """Train, score and record one cell; returns (run directory, status). KeyboardInterrupt leaves
        the directory without result.json (``incomplete``) and propagates."""
        from neural_trade.experiments.scorer import score_result

        sc = self.scenario
        ctx = self._create_context(pc, self._meta(pc))
        self.store.index.add_run(ctx.run_dir, self.store.root)
        t0 = time.perf_counter()
        doc: Dict[str, Any] = {"schema_version": ENGINE_SCHEMA_VERSION, "run_id": ctx.run_id, "scenario": sc.name,
                               "cell_key": pc.key, "role": pc.role}
        try:
            result = self.trainer(ctx, calibrate=sc.run.calibrate, save_artifacts=sc.run.save_artifacts)
            t_train = time.perf_counter() - t0
            scored = score_result(result, role=pc.role, strategy=sc.strategy.name,
                                  strategy_params=sc.strategy.params, backtest_params=sc.backtest,
                                  run_id=ctx.run_id, out_dir=ctx.run_dir,
                                  meta={"scenario": sc.name, "cell_key": pc.key, **self._fold_meta(pc),
                                        "blocks": pc.fold["blocks"], "dataset_sha256": pc.dataset["sha256"]})
            doc.update(status="done", report=scored.paths["json"].name, train_s=t_train,
                       score_s=time.perf_counter() - t0 - t_train, scores=scored.scores)
            del result, scored
        except (KeyboardInterrupt, SystemExit):
            logger.warning("[scenario %s] %s interrupted: %s stays without %s; the next run trains the cell again",
                           sc.name, pc.key, ctx.run_dir, RESULT_FILE)
            raise
        except Exception as exc:
            logger.exception("[scenario %s] %s failed", sc.name, pc.key)
            doc.update(status="failed", scores={},
                       error={"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc(limit=20)})
        doc.update(finished_utc=_utc(), wall_s=time.perf_counter() - t0, sec_per_step=_sec_per_step(ctx.run_dir))
        _write_new(ctx.run_dir / RESULT_FILE, doc)
        row = self.store.index.add_run(ctx.run_dir, self.store.root)
        logger.info("[scenario %s] %s %s in %.1f s -> %s (net Sharpe %s, trades %s)", sc.name, pc.key, doc["status"],
                    doc["wall_s"], ctx.run_dir, row.get("sharpe_net"), row.get("n_trades"))
        return ctx.run_dir, doc["status"]


def _sec_per_step(run_dir: Path) -> Optional[float]:
    try:
        status = json.loads((Path(run_dir) / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    v = status.get("sec_per_step")
    return float(v) if isinstance(v, (int, float)) else None


def plan_table(planned: List[PlannedCell]) -> List[Dict[str, Any]]:
    """One row per cell for printing: key, configuration, fold, role, seed, state, earlier run ids."""
    return [{"cell": pc.key, "configuration": pc.cell.configuration.name, "fold": pc.cell.fold,
             "fold_id": pc.fold["fold_id"], "role": pc.role, "seed": pc.cell.seed, "state": pc.state,
             "runs": pc.runs} for pc in planned]


__all__ = ["ENGINE_SCHEMA_VERSION", "PlannedCell", "RunReport", "Runner", "plan_table", "train_cell"]

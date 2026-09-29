"""Strategy studies on stored cells (NT-076): many strategy configurations, one scenario's trained
cells, CPU only, no retraining.

    neural-trade scenario rescore configs/scenarios/reference.yaml \\
        --study configs/strategy_studies/example.yaml [--store runs] [--random-seeds N]

Every cell the experiment engine scores stores its out-of-sample and calibration predictions with
their bars (``predictions_oos.npz``, ``predictions_cal.npz``; experiments.scorer.save_predictions).
A re-score backtests each configuration of a **strategy study** on every completed cell of the
scenario that has them, exactly as the scorer does (:func:`experiments.scorer.fit_and_backtest`):
the knobs are fitted on the cell's CALIBRATION block only (``var_scale_from(cal)`` and, for a
strategy with ``from_calibration``, its entry lines from the cal SignalFrame), the out-of-sample
block is traded with next-open fills, stops on high/low and the costs of the scenario's
``backtest:`` settings with the entry's ``backtest`` on top (``bar_minutes`` from the run's
config), and the result carries buy-and-hold, always-flat and the size-matched random null. The
scenario's own strategy re-scored this way reproduces each cell's result.json scores exactly.

A study spec (YAML, ``configs/strategy_studies/``)::

    schema_version: 1
    name: example                        # the output directory's prefix
    description: one line for the reader
    entries:
      - id: cq                           # letters, digits, '.', '_', '-'
        strategy: calibrated_quantile    # Strategies registry
        params: {max_hold: 15}           # strategy knobs (unknown knobs are refused)
        backtest: {fee_bps: 5.0}         # BacktestConfig fields on top of the scenario's backtest
        grid: {entry_quantile: [0.8, 0.9]}   # optional: one configuration per combination

A grid expands into one configuration per combination, with ids ``cq[entry_quantile=0.8]``. Unknown
keys at any level, unregistered strategies, unknown strategy or backtest parameters, the knobs a
``from_calibration`` strategy fits on the calibration block, engine-owned backtest fields
(``bar_minutes``, ``minutes_per_year``) and repeated ids are refused before anything runs.

Outputs go into a NEW directory ``<store>/scenarios/<scenario>/rescore/<study>-<UTC stamp>/``, never
into a cell directory: ``cells.csv`` (one row per configuration x cell), ``leaderboard.csv`` and
``leaderboard.md`` (one row per configuration, ranked by the mean net Sharpe over the DEV cells; the
test cells' numbers are shown in their own columns and never affect the order, D-020),
``study.yaml`` (the normalised spec) and ``meta.json`` (git sha, the scenario's spec hash, the cells
used and the ones skipped with the reason, the configurations).
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.experiments.scenario import (DERIVED_STRATEGY_PARAMS, RESERVED_BACKTEST, NAME_RE, Scenario,
                                               canonical_json, config_hash, short_hash)
from neural_trade.experiments.scorer import PREDICTION_FILES, BlockSignals, fit_and_backtest, load_block
from neural_trade.experiments.store import RESULT_FILE, RunStore

logger = logging.getLogger(__name__)

STUDY_SCHEMA_VERSION = 1
STUDY_KEYS = ("schema_version", "name", "description", "entries")
ENTRY_KEYS = ("id", "strategy", "params", "backtest", "grid")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RESCORE_SUBDIR = "rescore"
BASELINES = ("buy_and_hold", "always_flat", "random_same_freq")
# the numbers of a leaderboard row, per role: (column suffix, cells.csv column, how)
LEADERBOARD_STATS = (("sharpe_net_mean", "sharpe_net", "mean"), ("sharpe_net_sd", "sharpe_net", "sd"),
                     ("total_return_mean", "total_return", "mean"), ("max_drawdown_mean", "max_drawdown", "mean"),
                     ("n_trades_mean", "n_trades", "mean"), ("beats_buy_and_hold_frac", "beats_buy_and_hold", "mean"),
                     ("random_pctile_sharpe_mean", "random_same_freq/percentile_sharpe_net", "mean"))


class StudyError(InvalidConfigurationError):
    """A strategy-study spec is malformed (a ValueError): nothing was run."""


class RescoreError(RuntimeError):
    """A re-score cannot run (for example: no dev cell of the scenario has stored predictions)."""


# ------------------------------------------------------------------ the study spec
@dataclass(frozen=True)
class StudyConfiguration:
    """One strategy configuration of a study: an entry at one grid point."""

    id: str
    entry: str
    strategy: str
    params: Dict[str, Any]
    backtest: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "entry": self.entry, "strategy": self.strategy, "params": dict(self.params),
                "backtest": dict(self.backtest)}


@dataclass
class StrategyStudy:
    name: str
    entries: List[Dict[str, Any]]            # normalised: id, strategy, params, backtest, grid
    description: str = ""
    schema_version: int = STUDY_SCHEMA_VERSION
    source: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path) -> "StrategyStudy":
        import yaml

        path = Path(path)
        if not path.is_file():
            raise StudyError(f"strategy study spec {path} does not exist")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise StudyError(f"strategy study spec {path}: not valid YAML: {exc}") from exc
        return cls.from_dict(data, source=path)

    @classmethod
    def from_dict(cls, data: Any, *, source=None) -> "StrategyStudy":
        """Parse and check everything (keys, strategies, parameters, ids); raises StudyError."""
        where = str(source) if source is not None else "strategy study spec"

        def bad(msg: str):
            raise StudyError(f"{where}: {msg}")

        if not isinstance(data, Mapping):
            bad("expected a mapping of study keys")
        _refuse_unknown(data, STUDY_KEYS, "", bad)
        version = data.get("schema_version")
        if version != STUDY_SCHEMA_VERSION:
            bad(f"schema_version must be {STUDY_SCHEMA_VERSION}, got {version!r}")
        name = data.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            bad(f"name must be 1-48 characters of letters, digits, '.', '_' or '-' (a directory name), got {name!r}")
        description = data.get("description") or ""
        if not isinstance(description, str):
            bad("description must be a string")
        raw = data.get("entries")
        if not isinstance(raw, list) or not raw:
            bad("entries must be a non-empty list of {id, strategy, params, backtest, grid}")
        entries = [_entry(e, i, bad) for i, e in enumerate(raw)]
        study = cls(name, entries, description, int(version), Path(source) if source is not None else None)
        ids = [c.id for c in study.configurations()]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            bad(f"configuration ids repeat: {dup}")
        return study

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "name": self.name, "description": self.description,
                "entries": [dict(e) for e in self.entries]}

    @property
    def spec_hash(self) -> str:
        return short_hash(self.to_dict())

    def configurations(self) -> List[StudyConfiguration]:
        """Every entry at every grid point, in spec order (entries, then the grid's product order)."""
        out = []
        for e in self.entries:
            axes = list(e["grid"])
            for point in itertools.product(*(e["grid"][a] for a in axes)) if axes else [()]:
                grid = dict(zip(axes, point))
                cid = e["id"] + ("[" + ",".join(f"{k}={_fmt(v)}" for k, v in grid.items()) + "]" if grid else "")
                out.append(StudyConfiguration(cid, e["id"], e["strategy"], {**e["params"], **grid}, dict(e["backtest"])))
        return out


def _entry(e: Any, i: int, bad) -> Dict[str, Any]:
    from neural_trade.strategy.params import _check_keys, build_backtest_config, build_strategy
    from neural_trade.strategy.strategies import Strategies

    at = f"entries[{i}]"
    if not isinstance(e, Mapping):
        bad(f"{at} must be a mapping of {list(ENTRY_KEYS)}")
    _refuse_unknown(e, ENTRY_KEYS, f"{at}.", bad)
    eid = e.get("id")
    if not isinstance(eid, str) or not ID_RE.match(eid):
        bad(f"{at}.id must be 1-64 letters, digits, '.', '_' or '-', got {eid!r}")
    at = f"entries[{i}] ({eid})"
    sname = e.get("strategy")
    if not isinstance(sname, str) or not Strategies.has(sname):
        bad(f"{at}.strategy {sname!r} is not registered; known: {Strategies.list_names()}")
    params, backtest, grid = (_mapping(e.get(k), f"{at}.{k}", bad) for k in ("params", "backtest", "grid"))
    for axis, values in grid.items():
        if not isinstance(values, list) or not values:
            bad(f"{at}.grid.{axis} must be a non-empty list of values")
        if len({canonical_json(v) for v in values}) != len(values):
            bad(f"{at}.grid.{axis} repeats a value")
    clash = sorted(set(params) & set(grid))
    if clash:
        bad(f"{at}: {clash} set in both params and grid")
    cls = Strategies.get(sname)
    try:
        _check_keys(cls, {**params, **grid}, f"strategy {sname!r}")
    except InvalidConfigurationError as exc:
        bad(f"{at}: {exc}")
    calibrated = hasattr(cls, "from_calibration")
    derived = sorted(set(params) | set(grid)) if calibrated else []
    derived = [k for k in derived if k in DERIVED_STRATEGY_PARAMS]
    if derived:
        bad(f"{at}: {derived}: {sname!r} sets these from each cell's calibration block (set entry_quantile instead)")
    reserved = {k: RESERVED_BACKTEST[k] for k in backtest if k in RESERVED_BACKTEST}
    if reserved:
        bad(f"{at}.backtest sets engine-owned fields: " + "; ".join(f"{k} ({why})" for k, why in reserved.items()))
    try:
        build_backtest_config(backtest)
    except (InvalidConfigurationError, ValueError, TypeError) as exc:
        bad(f"{at}.backtest: {exc}")
    if not calibrated:                   # a from_calibration strategy is built per cell, on its cal block
        for point in itertools.product(*grid.values()) if grid else [()]:
            try:
                build_strategy(sname, {**params, **dict(zip(grid, point))})
            except (InvalidConfigurationError, ValueError, TypeError) as exc:
                bad(f"{at}: {exc}")
    return {"id": eid, "strategy": sname, "params": params, "backtest": backtest, "grid": grid}


def _refuse_unknown(data: Mapping[str, Any], known, prefix: str, bad) -> None:
    unknown = [k for k in data if k not in known]
    if unknown:
        bad("unknown key(s) " + ", ".join(f"{prefix}{k}" for k in unknown) + f"; known: {', '.join(known)}")


def _mapping(value: Any, where: str, bad) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        bad(f"{where} must be a mapping, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (int, str)):
        return str(value)
    return canonical_json(value)


# ------------------------------------------------------------------ the stored cells
@dataclass
class StoredCell:
    run_dir: Path
    run_id: str
    cell_key: str
    fold: int
    seed: int
    role: str
    bar_minutes: float
    scores: Dict[str, Any] = field(default_factory=dict)

    def rel(self, root) -> str:
        return _rel(self.run_dir, root)

    def load(self):
        """(BlockSignals fitted on the cal block, the out-of-sample Bars, the out-of-sample frame)."""
        cal, _, _ = load_block(self.run_dir / PREDICTION_FILES["cal"])
        oos, bars, _ = load_block(self.run_dir / PREDICTION_FILES["oos"])
        if len(bars) != len(oos) or not np.allclose(bars.close, oos.last_close, rtol=1e-6):
            raise RescoreError(f"{self.run_dir}: the stored bars ({len(bars)}) do not line up with the stored "
                               f"predictions ({len(oos)})")
        return BlockSignals.build(cal, oos), bars, oos


def _rel(path: Path, root) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _bar_minutes(run_dir: Path, meta: Mapping[str, Any]) -> float:
    """RESAMPLE_MINUTES of the run's config.yaml (the scorer's bar size); meta.json's setup as a fallback."""
    import yaml

    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        return float(cfg["RESAMPLE_MINUTES"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return float((meta.get("setup") or {}).get("bar_minutes", 1))


def select_cells(scenario: Scenario, store: RunStore) -> Tuple[List[StoredCell], List[Dict[str, str]], List[str]]:
    """(cells to re-score, skipped run directories with the reason, the spec's cells with no usable run).

    A cell is used when its run directory is ``done``, belongs to a cell of this spec with the same
    config hash and calibration pass, and has both stored prediction files; when a cell has several
    such runs, the newest is used and the others are listed as skipped."""
    expected = {cell.key: config_hash(cfg) for cell, cfg in scenario.validate()}
    candidates: Dict[str, List[Tuple[str, StoredCell]]] = {}
    skipped: List[Dict[str, str]] = []

    def skip(run_dir, key, reason):
        skipped.append({"run_dir": _rel(run_dir, store.root), "cell_key": key, "reason": reason})

    for d in store.run_dirs(scenario.name):
        meta = _read_json(d / "meta.json") or {}
        eng = meta.get("engine") or {}
        key = str(eng.get("cell_key"))
        result = _read_json(d / RESULT_FILE)
        status = (result or {}).get("status") or "incomplete"
        if status != "done":
            skip(d, key, f"status {status}")
        elif key not in expected:
            skip(d, key, "not a cell of this scenario spec")
        elif eng.get("config_hash") != expected[key]:
            skip(d, key, "config hash differs from the spec's (a run of an earlier version of the scenario)")
        elif bool((eng.get("run") or {}).get("calibrate", True)) != bool(scenario.run.calibrate):
            skip(d, key, "trained with a different run.calibrate than the spec's")
        elif not all((d / f).is_file() for f in PREDICTION_FILES.values()):
            skip(d, key, "no stored predictions (" + " / ".join(PREDICTION_FILES.values())
                 + " missing: scored before NT-076, or the files were not kept on this machine)")
        else:
            cell = StoredCell(d, str(meta.get("run_id") or d.name), key, int(eng["fold"]), int(eng["seed"]),
                              str(eng["role"]), _bar_minutes(d, meta), dict((result or {}).get("scores") or {}))
            candidates.setdefault(key, []).append((str(meta.get("created_utc") or ""), cell))
    used: List[StoredCell] = []
    for key in expected:
        runs = sorted(candidates.get(key, []), key=lambda t: (t[0], t[1].run_id))
        if not runs:
            continue
        used.append(runs[-1][1])
        for _, older in runs[:-1]:
            skip(older.run_dir, key, f"an older run of the same cell ({runs[-1][1].run_id} is used)")
    missing = [k for k in expected if k not in candidates]
    return used, skipped, missing


# ------------------------------------------------------------------ re-scoring
def _num(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def cell_row(conf: StudyConfiguration, cell: StoredCell, root, res, strategy, var_scale: float) -> Dict[str, Any]:
    """One row of cells.csv: the configuration, the cell, the backtest summary and the baselines."""
    from neural_trade.experiments.scorer import _strategy_params

    row: Dict[str, Any] = {"config_id": conf.id, "entry": conf.entry, "strategy": conf.strategy,
                           "run_dir": cell.rel(root), "run_id": cell.run_id, "cell_key": cell.cell_key,
                           "fold": cell.fold, "seed": cell.seed, "role": cell.role, "var_scale": float(var_scale),
                           "fitted_params": json.dumps(_strategy_params(strategy), sort_keys=True, default=float)}
    row.update({k: _num(v) for k, v in res.summary.items()})
    for name in BASELINES:
        row.update({f"{name}/{k}": _num(v) for k, v in (res.baselines.get(name) or {}).items()})
    bh = (res.baselines.get("buy_and_hold") or {}).get("total_return")
    row["beats_buy_and_hold"] = int(bh is not None and res.summary["total_return"] > bh)
    return row


def _stat(values: Sequence[Any], how: str) -> Optional[float]:
    v = np.asarray([float(x) for x in values if x is not None and x != ""], dtype=float)
    v = v[np.isfinite(v)]
    if how == "sd":
        return float(np.std(v, ddof=1)) if len(v) >= 2 else None
    return float(np.mean(v)) if len(v) else None


def leaderboard(rows: Sequence[Mapping[str, Any]], configurations: Sequence[StudyConfiguration]) -> List[Dict[str, Any]]:
    """One row per configuration, RANKED by the mean net Sharpe over its dev cells (descending; a
    configuration without a finite value last; ties keep the spec order). The ``test_`` columns
    summarise the test cells and never affect the order (D-020)."""
    out = []
    for conf in configurations:
        mine = [r for r in rows if r["config_id"] == conf.id]
        row: Dict[str, Any] = {"config_id": conf.id, "strategy": conf.strategy,
                               "params": json.dumps(conf.params, sort_keys=True),
                               "backtest": json.dumps(conf.backtest, sort_keys=True)}
        for role in ("dev", "test"):
            cells = [r for r in mine if r["role"] == role]
            row[f"n_{role}"] = len(cells)
            for suffix, col, how in LEADERBOARD_STATS:
                row[f"{role}_{suffix}"] = _stat([r.get(col) for r in cells], how)
        out.append(row)
    key = [r["dev_sharpe_net_mean"] for r in out]
    order = sorted(range(len(out)), key=lambda i: (key[i] is None, -(key[i] if key[i] is not None else 0.0), i))
    ranked = [out[i] for i in order]
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    cols = ["rank", "config_id", "strategy", "params", "backtest", "n_dev"] \
        + [f"dev_{s}" for s, _, _ in LEADERBOARD_STATS] + ["n_test"] + [f"test_{s}" for s, _, _ in LEADERBOARD_STATS]
    return [{c: r[c] for c in cols} for r in ranked]


@dataclass
class RescoreReport:
    out_dir: Path
    scenario: str
    study: str
    rows: List[Dict[str, Any]]
    leaderboard: List[Dict[str, Any]]
    cells: List[StoredCell]
    skipped: List[Dict[str, str]]
    missing: List[str]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        top = [{k: r[k] for k in ("rank", "config_id", "n_dev", "dev_sharpe_net_mean", "dev_sharpe_net_sd",
                                  "n_test", "test_sharpe_net_mean")} for r in self.leaderboard]
        return {"out_dir": str(self.out_dir), "scenario": self.scenario, "study": self.study,
                "n_configurations": self.meta["n_configurations"], "n_cells": len(self.cells),
                "cells": [c.cell_key for c in self.cells], "skipped": self.skipped, "missing": self.missing,
                "leaderboard": top}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _new_dir(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        d = parent / (name if attempt == 1 else f"{name}-{attempt}")
        try:
            d.mkdir(exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"no free directory name for {name} under {parent}")


def rescore(scenario: Scenario, study: StrategyStudy, store="runs", *, random_seeds: Optional[int] = None,
            index_path=None) -> RescoreReport:
    """Re-score every configuration of ``study`` on every usable stored cell of ``scenario`` (see the
    module docstring); writes a new output directory and returns its report. Raises RescoreError when
    no dev cell has stored predictions (nothing is written then)."""
    from neural_trade.utils.env import git_sha

    store = store if isinstance(store, RunStore) else RunStore(store, index_path)
    if random_seeds is not None and (isinstance(random_seeds, bool) or int(random_seeds) < 0):
        raise StudyError(f"random_seeds must be >= 0, got {random_seeds!r}")
    configurations = study.configurations()
    for conf in configurations:            # the merged cost settings, checked before anything runs
        _backtest_params(scenario, conf, random_seeds)
    cells, skipped, missing = select_cells(scenario, store)
    for s in skipped:
        logger.info("[rescore %s] skipped %s: %s", scenario.name, s["run_dir"], s["reason"])
    if not any(c.role == "dev" for c in cells):
        raise RescoreError(
            f"scenario {scenario.name!r}: no dev cell has stored predictions under {store.scenario_dir(scenario.name)} "
            f"({len(cells)} usable cells, {len(skipped)} run directories skipped, {len(missing)} cells without a "
            "usable run): run the scenario first (`neural-trade scenario run`); cells scored before NT-076 stored "
            "no predictions")
    rows: List[Dict[str, Any]] = []
    for i, cell in enumerate(cells):
        signals, bars, _ = cell.load()
        logger.info("[rescore %s] cell %d/%d %s (%s): %d configurations", scenario.name, i + 1, len(cells),
                    cell.cell_key, cell.role, len(configurations))
        for conf in configurations:
            res, strat = fit_and_backtest(signals, bars, strategy=conf.strategy, strategy_params=conf.params,
                                          backtest_params=_backtest_params(scenario, conf, random_seeds),
                                          bar_minutes=cell.bar_minutes)
            rows.append(cell_row(conf, cell, store.root, res, strat, signals.var_scale))
    board = leaderboard(rows, configurations)
    out_dir = _new_dir(store.scenario_dir(scenario.name) / RESCORE_SUBDIR, f"{study.name}-{_utc()}")
    meta = {"schema_version": STUDY_SCHEMA_VERSION, "created_utc": _utc(), "git_sha": git_sha(),
            "scenario": scenario.name, "scenario_spec": str(scenario.source) if scenario.source else None,
            "scenario_spec_hash": scenario.spec_hash, "scenario_backtest": dict(scenario.backtest),
            "study": study.name, "study_spec": str(study.source) if study.source else None,
            "study_spec_hash": study.spec_hash, "random_seeds_override": random_seeds,
            "n_configurations": len(configurations), "configurations": [c.to_dict() for c in configurations],
            "ranked_by": "mean net Sharpe after costs over the dev cells (test cells shown, never ranked; D-020)",
            "cells": [{"run_dir": c.rel(store.root), "run_id": c.run_id, "cell_key": c.cell_key, "fold": c.fold,
                       "seed": c.seed, "role": c.role, "bar_minutes": c.bar_minutes} for c in cells],
            "skipped": skipped, "missing": missing}
    report = RescoreReport(out_dir, scenario.name, study.name, rows, board, cells, skipped, missing, meta)
    write_outputs(report, study)
    return report


def _backtest_params(scenario: Scenario, conf: StudyConfiguration, random_seeds: Optional[int]) -> Dict[str, Any]:
    from neural_trade.strategy.params import build_backtest_config

    params = {**dict(scenario.backtest), **dict(conf.backtest)}
    if random_seeds is not None:
        params["random_seeds"] = int(random_seeds)
    try:
        build_backtest_config(params)
    except (InvalidConfigurationError, ValueError, TypeError) as exc:
        raise StudyError(f"configuration {conf.id}: backtest {params}: {exc}") from exc
    return params


# ------------------------------------------------------------------ outputs
def _csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Optional[List[str]] = None) -> None:
    cols = list(columns or [])
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "x", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def _f(v: Any, fmt: str) -> str:
    return "n/a" if v is None or not math.isfinite(float(v)) else fmt.format(float(v))


def leaderboard_markdown(report: RescoreReport) -> str:
    m = report.meta
    n_dev = sum(c.role == "dev" for c in report.cells)
    n_test = len(report.cells) - n_dev
    costs = m["scenario_backtest"] or "the default cost profile (13 bps per side)"
    L = [f"# Strategy study `{report.study}` on scenario `{report.scenario}`", "",
         f"{m['n_configurations']} configurations x {len(report.cells)} stored cells ({n_dev} dev, {n_test} test); "
         f"{len(report.skipped)} run directories skipped, {len(report.missing)} cells without a usable run "
         "(meta.json lists them). Git " + str(m["git_sha"]) + f", scenario spec hash {m['scenario_spec_hash']}.", "",
         "Each configuration is fitted on each cell's calibration block only and backtests its out-of-sample "
         f"block: next-open fills, stops on high/low, the scenario's backtest settings ({costs}) with the "
         "entry's on top"
         + (f", {m['random_seeds_override']} random-null seeds (override)" if m["random_seeds_override"] is not None
            else "") + ".", "",
         "**Ranked by the mean net Sharpe over the dev cells.** sd is the sample standard deviation over "
         "the dev cells (n = n dev); the random-null percentile is the share of size-matched random seeds with a "
         "lower net Sharpe. The test cells are in the second table: shown, never used to rank or choose (D-020).",
         "", "## Ranking (dev cells)", "",
         "| rank | configuration | n dev | net Sharpe mean | sd | net return | max drawdown | trades | "
         "beats buy-and-hold | random-null pctile |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report.leaderboard:
        L.append(f"| {r['rank']} | `{r['config_id']}` | {r['n_dev']} | {_f(r['dev_sharpe_net_mean'], '{:+.2f}')} | "
                 f"{_f(r['dev_sharpe_net_sd'], '{:.2f}')} | {_f(r['dev_total_return_mean'], '{:+.2%}')} | "
                 f"{_f(r['dev_max_drawdown_mean'], '{:.2%}')} | {_f(r['dev_n_trades_mean'], '{:.1f}')} | "
                 f"{_f(r['dev_beats_buy_and_hold_frac'], '{:.0%}')} | "
                 f"{_f(r['dev_random_pctile_sharpe_mean'], '{:.0f}')} |")
    L += ["", "## Test cells (shown, never used to rank or choose; D-020)", "",
          "Same order as the ranking above.", "",
          "| rank | configuration | n test | net Sharpe mean | sd | net return | max drawdown | trades | "
          "beats buy-and-hold | random-null pctile |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report.leaderboard:
        L.append(f"| {r['rank']} | `{r['config_id']}` | {r['n_test']} | {_f(r['test_sharpe_net_mean'], '{:+.2f}')} | "
                 f"{_f(r['test_sharpe_net_sd'], '{:.2f}')} | {_f(r['test_total_return_mean'], '{:+.2%}')} | "
                 f"{_f(r['test_max_drawdown_mean'], '{:.2%}')} | {_f(r['test_n_trades_mean'], '{:.1f}')} | "
                 f"{_f(r['test_beats_buy_and_hold_frac'], '{:.0%}')} | "
                 f"{_f(r['test_random_pctile_sharpe_mean'], '{:.0f}')} |")
    return "\n".join(L) + "\n"


def write_outputs(report: RescoreReport, study: StrategyStudy) -> None:
    import yaml

    d = report.out_dir
    _csv(d / "cells.csv", report.rows)
    _csv(d / "leaderboard.csv", report.leaderboard)
    for name, text in (("leaderboard.md", leaderboard_markdown(report)),
                       ("study.yaml", yaml.safe_dump(study.to_dict(), sort_keys=False)),
                       ("meta.json", json.dumps(report.meta, indent=2, default=str))):
        with open(d / name, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


__all__ = ["RESCORE_SUBDIR", "RescoreError", "RescoreReport", "StoredCell", "StrategyStudy", "StudyConfiguration",
           "StudyError", "cell_row", "leaderboard", "leaderboard_markdown", "rescore", "select_cells"]

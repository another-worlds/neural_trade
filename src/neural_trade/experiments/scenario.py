"""Scenario and sweep specs of the experiment engine (NT-026).

A scenario says what to train and how to score it, in one YAML file (``configs/scenarios/``):

    schema_version: 1
    name: reference_default          # the run store subtree: <store>/scenarios/<name>/
    description: one line for the reader
    base_config: ../default.yaml     # a flat Config YAML, relative to this file (absent: Config())
    overrides: {EPOCHS: 20}          # Config fields applied to every variant
    variants:                        # named Config override sets (absent: one variant "default")
      default: {}
      no_physics: {LAMBDA_HD: 0.0}
    sweep:                           # optional grid over Config fields, crossed with every variant
      mode: grid
      axes: {LR: [0.001, 0.0003]}
    folds: [-3, -2, -1]              # FOLD_INDEX values; the latest usable fold is the test fold
    seeds: [0, 1, 2]
    strategy: {name: calibrated_quantile, params: {}}   # Strategies registry; knobs fit on cal
    backtest: {random_seeds: 100}    # BacktestConfig fields; the costs default to 13 bps per side
    run: {calibrate: true, save_artifacts: false}

A **configuration** is one variant at one grid point; a **cell** is one (configuration, fold,
seed), trained into its own run directory. Everything is checked before anything runs:
unknown keys at any level are refused, and :meth:`Scenario.validate` builds every cell's Config
(``Config.override``: unknown fields and invalid values raise). The Config fields the engine
sets itself (FOLD_INDEX and SEED from ``folds`` / ``seeds``; MODEL_PATH, SCALER_PATH and
ARTIFACTS_DIR from the run directory) are refused in overrides, variants and axes.

Hashes (the run store's resume keys): ``config_hash(config)`` of a cell's Config values;
``settings_hash`` of what else changes a cell's numbers (strategy, backtest, the calibration
pass); ``spec_hash`` of the whole normalised spec, recorded with every run.

Extension points (the schema version rises when a key changes meaning): NT-030 adds sweep modes
(quick, optuna) and a search space, NT-031 guard-rail thresholds, NT-033 rule-based scenarios
that train no network, NT-038 harness cases.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError

SCHEMA_VERSION = 1
TOP_KEYS = ("schema_version", "name", "description", "base_config", "overrides", "variants", "sweep", "folds",
            "seeds", "strategy", "backtest", "run")
SWEEP_KEYS = ("mode", "axes")
SWEEP_MODES = ("grid",)                     # NT-030 adds "quick" and "optuna"
STRATEGY_KEYS = ("name", "params")
RUN_KEYS = ("calibrate", "save_artifacts")
# Config fields the engine sets per cell or per run directory.
RESERVED_FIELDS = {"FOLD_INDEX": "set by `folds:`", "SEED": "set by `seeds:`",
                   "MODEL_PATH": "set to the run directory", "SCALER_PATH": "set to the run directory",
                   "ARTIFACTS_DIR": "set to the run directory"}
# BacktestConfig fields the engine sets: the annualisation follows the run's bar size (NT-040).
RESERVED_BACKTEST = {"bar_minutes": "set from the run's RESAMPLE_MINUTES",
                     "minutes_per_year": "fixed at 525,600 (a 24/7 market; NT-040)"}
# Strategy knobs that from_calibration strategies set from the calibration block.
DERIVED_STRATEGY_PARAMS = ("long_above", "short_below", "median")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
CONFIGURATION_NAME_MAX = 64


class ScenarioError(InvalidConfigurationError):
    """A scenario spec is malformed or names an invalid configuration (a ValueError)."""


# ------------------------------------------------------------------ hashing
def canonical_json(obj: Any) -> str:
    """Sorted-key, compact JSON: the input of every engine hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(obj: Any, n: int = 12) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def config_hash(config: Config) -> str:
    """Hash of a Config's VALUES (field docs and YAML layout do not count). A cell counts as done
    for a resume only when its finished run has the same config hash."""
    return short_hash(config.to_dict())


# ------------------------------------------------------------------ the spec
@dataclass
class StrategySpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunOptions:
    calibrate: bool = True          # the pre-training loss-weight calibration pass (train_and_evaluate)
    save_artifacts: bool = False    # the serving bundle (artifacts/); the checkpoint weights are always kept


@dataclass(frozen=True)
class Configuration:
    """One variant at one grid point: ``params`` are its variant overrides plus axis values."""

    name: str
    variant: str
    params: Dict[str, Any]
    config: Config = field(compare=False, repr=False)


@dataclass(frozen=True)
class Cell:
    configuration: Configuration
    fold: int
    seed: int

    @property
    def key(self) -> str:
        return cell_key(self.configuration.name, self.fold, self.seed)

    def config(self) -> Config:
        """The configuration's Config with this cell's FOLD_INDEX and SEED (validated)."""
        return self.configuration.config.copy(FOLD_INDEX=int(self.fold), SEED=int(self.seed))


def cell_key(configuration: str, fold: int, seed: int) -> str:
    """``<configuration>__f<fold>__s<seed>``, e.g. ``default__f-1__s0`` (safe in a directory name)."""
    return f"{configuration}__f{int(fold)}__s{int(seed)}"


@dataclass
class Scenario:
    name: str
    folds: List[int]
    seeds: List[int]
    strategy: StrategySpec
    schema_version: int = SCHEMA_VERSION
    description: str = ""
    base_config: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    variants: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"default": {}})
    sweep_mode: str = "grid"
    axes: Dict[str, List[Any]] = field(default_factory=dict)
    backtest: Dict[str, Any] = field(default_factory=dict)
    run: RunOptions = field(default_factory=RunOptions)
    source: Optional[Path] = None           # the spec file (error messages, base_config resolution)
    base_dir: Optional[Path] = None         # where a relative base_config is resolved

    # -------------------------------------------------------------- construction
    @classmethod
    def from_yaml(cls, path) -> "Scenario":
        import yaml

        path = Path(path)
        if not path.is_file():
            raise ScenarioError(f"scenario spec {path} does not exist")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ScenarioError(f"scenario spec {path}: not valid YAML: {exc}") from exc
        return cls.from_dict(data, source=path, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: Any, *, source=None, base_dir=None) -> "Scenario":
        """Parse and check the structure (keys, types, names); :meth:`validate` checks the Configs."""
        where = str(source) if source is not None else "scenario spec"

        def bad(msg: str):
            raise ScenarioError(f"{where}: {msg}")

        if not isinstance(data, Mapping):
            bad("expected a mapping of scenario keys")
        _refuse_unknown(data, TOP_KEYS, "", bad)
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            bad(f"schema_version must be {SCHEMA_VERSION} (this engine's spec schema), got {version!r}")
        name = data.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            bad(f"name must be 1-48 characters of letters, digits, '.', '_' or '-' (a directory name), got {name!r}")
        description = data.get("description") or ""
        if not isinstance(description, str):
            bad("description must be a string")
        base = data.get("base_config")
        if base is not None and not isinstance(base, str):
            bad(f"base_config must be a path string, got {base!r}")
        overrides = _mapping(data.get("overrides"), "overrides", bad)
        raw_variants = data.get("variants")
        if raw_variants is None:
            variants: Dict[str, Dict[str, Any]] = {"default": {}}
        else:
            if not isinstance(raw_variants, Mapping) or not raw_variants:
                bad("variants must be a non-empty mapping of name -> Config overrides")
            variants = {}
            for vname, vover in raw_variants.items():
                if not isinstance(vname, str) or not NAME_RE.match(vname) or "__" in vname:
                    bad(f"variant name {vname!r} must be 1-48 letters, digits, '.', '_' or '-' without '__'")
                variants[vname] = _mapping(vover, f"variants.{vname}", bad)
        sweep = _mapping(data.get("sweep"), "sweep", bad)
        _refuse_unknown(sweep, SWEEP_KEYS, "sweep.", bad)
        mode = sweep.get("mode", "grid")
        if mode not in SWEEP_MODES:
            bad(f"sweep.mode {mode!r} is not available; this engine runs {list(SWEEP_MODES)} "
                "(quick and optuna modes come with NT-030)")
        axes = _mapping(sweep.get("axes"), "sweep.axes", bad)
        for axis, values in axes.items():
            if not isinstance(values, list) or not values:
                bad(f"sweep.axes.{axis} must be a non-empty list of values")
            if len({canonical_json(v) for v in values}) != len(values):
                bad(f"sweep.axes.{axis} repeats a value")
        folds = _int_list(data.get("folds"), "folds", bad)
        seeds = _int_list(data.get("seeds"), "seeds", bad)
        if any(s < 0 for s in seeds):
            bad(f"seeds must be >= 0, got {seeds}")
        raw_strategy = _mapping(data.get("strategy"), "strategy", bad)
        _refuse_unknown(raw_strategy, STRATEGY_KEYS, "strategy.", bad)
        from neural_trade.strategy.strategies import Strategies

        sname = raw_strategy.get("name", Strategies.default)
        if not isinstance(sname, str):
            bad(f"strategy.name must be a string, got {sname!r}")
        strategy = StrategySpec(sname, _mapping(raw_strategy.get("params"), "strategy.params", bad))
        backtest = _mapping(data.get("backtest"), "backtest", bad)
        raw_run = _mapping(data.get("run"), "run", bad)
        _refuse_unknown(raw_run, RUN_KEYS, "run.", bad)
        for key, value in raw_run.items():
            if not isinstance(value, bool):
                bad(f"run.{key} must be true or false, got {value!r}")
        return cls(name=name, folds=folds, seeds=seeds, strategy=strategy, schema_version=version,
                   description=description, base_config=base, overrides=overrides, variants=variants,
                   sweep_mode=mode, axes=axes, backtest=backtest, run=RunOptions(**raw_run),
                   source=Path(source) if source is not None else None,
                   base_dir=Path(base_dir) if base_dir is not None else None)

    # -------------------------------------------------------------- views
    def where(self) -> str:
        return str(self.source) if self.source is not None else f"scenario {self.name!r}"

    def to_dict(self) -> Dict[str, Any]:
        """The normalised spec (defaults filled in), as YAML would hold it."""
        return {"schema_version": self.schema_version, "name": self.name, "description": self.description,
                "base_config": self.base_config, "overrides": self.overrides, "variants": self.variants,
                "sweep": {"mode": self.sweep_mode, "axes": self.axes}, "folds": list(self.folds),
                "seeds": list(self.seeds), "strategy": dataclasses.asdict(self.strategy),
                "backtest": dict(self.backtest), "run": dataclasses.asdict(self.run)}

    @property
    def spec_hash(self) -> str:
        return short_hash(self.to_dict())

    def settings(self) -> Dict[str, Any]:
        """What changes a cell's numbers besides its Config: strategy, costs, the calibration pass."""
        return {"strategy": dataclasses.asdict(self.strategy), "backtest": dict(sorted(self.backtest.items())),
                "calibrate": bool(self.run.calibrate)}

    @property
    def settings_hash(self) -> str:
        return short_hash(self.settings())

    def base(self) -> Config:
        """The base Config with the scenario-wide overrides (validated)."""
        if self.base_config is None:
            cfg = Config()
        else:
            path = Path(self.base_config)
            if not path.is_absolute() and self.base_dir is not None:
                path = self.base_dir / path
            if not path.is_file():
                raise ScenarioError(f"{self.where()}: base_config {self.base_config!r} not found (looked at {path})")
            cfg = _config_step(lambda: Config.from_yaml(path), f"base_config {self.base_config}", self.where())
        self._check_fields(self.overrides, "overrides")
        return _config_step(lambda: cfg.override(**self.overrides), "overrides", self.where())

    def configurations(self) -> List[Configuration]:
        """Every variant at every grid point, in spec order (variants, then the grid's product order)."""
        base = self.base()
        axis_names = list(self.axes)
        self._check_fields({a: None for a in axis_names}, "sweep.axes")
        out: List[Configuration] = []
        for vname, vover in self.variants.items():
            self._check_fields(vover, f"variants.{vname}")
            clash = sorted(set(vover) & set(axis_names))
            if clash:
                raise ScenarioError(f"{self.where()}: variants.{vname} sets {clash}, which sweep.axes also sweeps")
            for point in itertools.product(*(self.axes[a] for a in axis_names)) if axis_names else [()]:
                grid = dict(zip(axis_names, point))
                params = {**vover, **grid}
                name = _configuration_name(vname, grid)
                cfg = _config_step(lambda p=params: base.copy(**p), f"variants.{vname}"
                                   + (f" at {grid}" if grid else ""), self.where())
                out.append(Configuration(name, vname, params, cfg))
        names = [c.name for c in out]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ScenarioError(f"{self.where()}: configuration names collide: {dup}")
        return out

    def cells(self, configurations: Optional[List[Configuration]] = None) -> List[Cell]:
        """Every (configuration, fold, seed): configurations in order, then folds, then seeds."""
        confs = configurations if configurations is not None else self.configurations()
        return [Cell(c, int(f), int(s)) for c in confs for f in self.folds for s in self.seeds]

    def validate(self) -> List[Tuple[Cell, Config]]:
        """Build and validate every cell's Config and the scoring settings; returns (cell, config)
        pairs. Raises ScenarioError naming the offending key. Needs no data and trains nothing."""
        self._check_scoring()
        out = []
        for cell in self.cells():
            cfg = _config_step(cell.config, f"cell {cell.key} (FOLD_INDEX={cell.fold}, SEED={cell.seed})",
                               self.where())
            out.append((cell, cfg))
        return out

    # -------------------------------------------------------------- checks
    def _check_fields(self, overrides: Mapping[str, Any], where: str) -> None:
        reserved = {k: RESERVED_FIELDS[k] for k in overrides if k in RESERVED_FIELDS}
        if reserved:
            raise ScenarioError(f"{self.where()}: {where} sets engine-owned Config fields: "
                                + "; ".join(f"{k} ({why})" for k, why in reserved.items()))
        names = set(Config.field_names())
        unknown = [k for k in overrides if k not in names]
        if unknown:
            import difflib

            hints = {k: difflib.get_close_matches(k, names, n=3) for k in unknown}
            raise ScenarioError(f"{self.where()}: {where}: unknown Config field(s): " + "; ".join(
                k + (f" (did you mean {', '.join(v)}?)" if v else "") for k, v in hints.items()))

    def _check_scoring(self) -> None:
        from neural_trade.strategy.params import build_backtest_config
        from neural_trade.strategy.strategies import Strategies

        where = self.where()
        if not Strategies.has(self.strategy.name):
            raise ScenarioError(f"{where}: strategy.name {self.strategy.name!r} is not registered; "
                                f"known: {Strategies.list_names()}")
        cls = Strategies.get(self.strategy.name)
        known = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
        unknown = sorted(set(self.strategy.params) - known)
        if unknown:
            raise ScenarioError(f"{where}: strategy.params: unknown {self.strategy.name!r} parameter(s) {unknown}; "
                                f"known: {sorted(known)}")
        derived = sorted(set(self.strategy.params) & set(DERIVED_STRATEGY_PARAMS)) \
            if hasattr(cls, "from_calibration") else []
        if derived:
            raise ScenarioError(f"{where}: strategy.params {derived}: {self.strategy.name!r} sets these from each "
                                "fold's calibration block (set entry_quantile instead)")
        reserved = {k: RESERVED_BACKTEST[k] for k in self.backtest if k in RESERVED_BACKTEST}
        if reserved:
            raise ScenarioError(f"{where}: backtest sets engine-owned fields: "
                                + "; ".join(f"{k} ({why})" for k, why in reserved.items()))
        try:
            build_backtest_config(self.backtest)
        except (InvalidConfigurationError, ValueError, TypeError) as exc:
            raise ScenarioError(f"{where}: backtest: {exc}") from exc


# ------------------------------------------------------------------ helpers
def _refuse_unknown(data: Mapping[str, Any], known, prefix: str, bad) -> None:
    unknown = [k for k in data if k not in known]
    if unknown:
        import difflib

        hints = {k: difflib.get_close_matches(str(k), known, n=1) for k in unknown}
        bad("unknown key(s) " + ", ".join(f"{prefix}{k}" + (f" (did you mean {prefix}{h[0]}?)" if h else "")
                                          for k, h in hints.items())
            + f"; known: {', '.join(prefix + k for k in known)}")


def _mapping(value: Any, where: str, bad) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        bad(f"{where} must be a mapping, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _int_list(value: Any, where: str, bad) -> List[int]:
    if not isinstance(value, list) or not value:
        bad(f"{where} must be a non-empty list of integers, got {value!r}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        bad(f"{where} must hold integers only, got {value!r}")
    if len(set(value)) != len(value):
        bad(f"{where} repeats a value: {value!r}")
    return [int(v) for v in value]


def _config_step(fn, what: str, where: str):
    try:
        return fn()
    except InvalidConfigurationError as exc:
        raise ScenarioError(f"{where}: {what}: {exc}") from exc


def _slug(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "-".join(_slug(v) for v in value)
    if isinstance(value, str):
        return value
    return short_hash(value, 8)


def _configuration_name(variant: str, grid: Mapping[str, Any]) -> str:
    """The variant name, plus ``__<axis>-<value>`` per axis (lower-case), or a hash when too long."""
    if not grid:
        return variant
    parts = [f"{k.lower()}-{_slug(v)}" for k, v in grid.items()]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", "__".join([variant] + parts))
    if len(name) > CONFIGURATION_NAME_MAX:
        name = f"{variant}__g{short_hash(grid, 8)}"
    return name


__all__ = ["Cell", "Configuration", "RunOptions", "SCHEMA_VERSION", "Scenario", "ScenarioError", "StrategySpec",
           "canonical_json", "cell_key", "config_hash", "short_hash"]

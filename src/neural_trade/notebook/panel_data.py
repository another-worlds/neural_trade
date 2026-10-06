"""What the control panel (notebook 06, NT-034) reads and builds, without any widget: the scenarios and
sweeps it offers, the search-space choices from the Config metadata, the ``search:`` block a selection
builds, the sweeps the run store holds and the leaderboard of one or several of them.

Everything here is read-only on the run store: it opens ``index.sqlite`` and ``sweeps/*/sweep.json``
and never creates, syncs or writes either (a missing index is an empty store). The panel's widgets
(:mod:`neural_trade.notebook.control_panel`) and the headless tests call these functions directly.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.experiments.scenario import RESERVED_FIELDS, Scenario
from neural_trade.experiments.store import RunStore
from neural_trade.experiments.sweep import MODES, REFUSED_FIELDS, STRATEGY_PREFIX, SweepError, SearchSpace

logger = logging.getLogger(__name__)

SWEEP_DIR = "sweeps"
# The yardstick's three arms (VISION "The yardstick", NT-033): when the newest sweep is one of them, the
# board shows every one of them that the store holds, on one dev-fold net Sharpe column.
YARDSTICK_SCENARIOS = ("nt033_learned", "nt033_frozen_twin", "nt033_ta_ma_cross", "nt033_ta_rsi", "nt033_ta_bollinger")
# Config fields whose value is a registry key: the choices come from the registries (plugins make them dynamic).
REGISTRY_FIELDS = {"MODEL_NAME": "Models", "LOSS_NAME": "Losses", "OPTIMIZER_NAME": "Optimizers",
                   "INDICATOR_OPTIMIZER_NAME": "Optimizers", "DATA_LOADER": "DataLoaders",
                   "VISUALIZATION": "Visualizations"}


# ------------------------------------------------------------------ scenarios
@dataclass(frozen=True)
class ScenarioChoice:
    name: str
    path: Optional[Path]
    scenario: Optional[Scenario]
    problem: Optional[str] = None            # why a spec file could not be offered (shown, never silently dropped)

    @property
    def label(self) -> str:
        return self.name if self.problem is None else f"{self.name} (unusable: {self.problem[:60]})"

    @property
    def trains(self) -> bool:
        return self.scenario is not None and bool(self.scenario.run.train)


def discover_scenarios(specs_dir="configs/scenarios") -> List[ScenarioChoice]:
    """Every ``*.yaml`` of ``specs_dir`` as a :class:`ScenarioChoice`, by the spec's ``name:`` key; a file
    that does not load as a scenario is listed with its problem, not dropped."""
    specs_dir = Path(specs_dir)
    out: List[ScenarioChoice] = []
    for path in sorted(specs_dir.glob("*.yaml")) if specs_dir.is_dir() else ():
        try:
            sc = Scenario.from_yaml(path)
        except (InvalidConfigurationError, OSError, ValueError, TypeError, KeyError) as exc:
            out.append(ScenarioChoice(path.stem, path, None, f"{type(exc).__name__}: {exc}"))
            continue
        out.append(ScenarioChoice(sc.name, path, sc))
    return out


# ------------------------------------------------------------------ the sweeps the store holds
@dataclass(frozen=True)
class SweepInfo:
    sweep_id: str
    scenario: str
    mode: str
    label: str
    state: str
    stop_reason: Optional[str]
    n_trials: int
    n_done: int
    n_failed: int
    modified: float
    directory: Path
    space: Tuple[Dict[str, Any], ...] = ()
    budget: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return (f"{self.sweep_id}: {self.label} sweep of {self.scenario}, {self.state}, "
                f"{self.n_done} of {self.n_trials} trial(s) finished" + (f", {self.n_failed} failed" if self.n_failed else ""))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def list_sweeps(store_root) -> List[SweepInfo]:
    """The sweeps under ``<store>/sweeps/*/sweep.json``, newest first (by file time)."""
    base = Path(store_root) / SWEEP_DIR
    out = []
    for d in sorted(base.iterdir()) if base.is_dir() else ():
        path = d / "sweep.json"
        doc = _read_json(path)
        if doc is None:
            continue
        trials = doc.get("trials") or []
        launches = doc.get("launches") or []
        budget = (launches[-1].get("budget") if launches else None) or {}
        n_trials = int(budget.get("n_trials") or budget.get("trials_to_run") or len(trials) or 0)
        out.append(SweepInfo(
            sweep_id=str(doc.get("sweep_id") or d.name), scenario=str(doc.get("scenario") or d.name),
            mode=str(doc.get("mode") or ""), label=str(doc.get("label") or doc.get("mode") or ""),
            state=str(doc.get("state") or "unknown"), stop_reason=doc.get("stop_reason"),
            n_trials=max(n_trials, len(trials)),
            n_done=sum(t.get("state") == "COMPLETE" for t in trials), n_failed=sum(t.get("state") == "FAIL" for t in trials),
            modified=path.stat().st_mtime, directory=d, space=tuple(doc.get("space") or ()), budget=dict(budget)))
    return sorted(out, key=lambda s: (-s.modified, s.sweep_id))


def default_board_sweeps(sweeps: Sequence[SweepInfo]) -> List[str]:
    """The sweeps the board shows by default: the newest, and, when it is one of the yardstick's arms, every
    yardstick sweep of the same mode that the store holds."""
    if not sweeps:
        return []
    latest = sweeps[0]
    if latest.scenario in YARDSTICK_SCENARIOS:
        arms = [s.sweep_id for s in sweeps if s.scenario in YARDSTICK_SCENARIOS and s.mode == latest.mode]
        return arms or [latest.sweep_id]
    return [latest.sweep_id]


# ------------------------------------------------------------------ the search space
@dataclass(frozen=True)
class FieldOption:
    """One searchable Config field as the panel offers it: from ``Config.field_specs()`` (NT-029)."""
    name: str
    kind: str                                  # "int" | "float" | "choice"
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[float] = None
    choices: Tuple[Any, ...] = ()
    unit: Optional[str] = None
    doc: str = ""
    needs_bounds: bool = False                 # no finite range in the metadata: the user must give low and high
    default: Any = None

    def default_rule(self) -> Dict[str, Any]:
        """The ``search:`` rule this option starts with (None-valued keys left out). A field the metadata
        leaves without a finite, inclusive range starts at a guess around its default (a tenth to ten times
        the default, 0 to 1 for a default of 0), which the user edits: the sweep refuses a space with no bounds."""
        if self.kind == "choice":
            return {"choices": list(self.choices)}
        rule: Dict[str, Any] = {}
        low, high = self.low, self.high
        if low is None or high is None:
            d = float(self.default or 0.0)
            guess_low, guess_high = (d / 10.0, d * 10.0) if d > 0 else (0.0, 1.0)
            if self.kind == "int":
                guess_low, guess_high = max(int(guess_low), 0 if d <= 0 else 1), max(int(guess_high), 1)
            low = guess_low if low is None else low
            high = guess_high if high is None else high
        rule.update(low=low, high=high)
        if self.log:
            rule["log"] = True
        if self.step is not None and self.kind == "int":
            rule["step"] = self.step
        return rule


def registry_choices(field_name: str) -> Tuple[str, ...]:
    """The registered keys a registry-key Config field can take (empty when its registry cannot be loaded)."""
    reg = REGISTRY_FIELDS.get(field_name)
    if reg is None:
        return ()
    try:
        from neural_trade.registries import all_registries

        return tuple(sorted(all_registries()[reg].list_names()))
    except Exception as exc:  # a registry whose module needs a missing package must not take the panel down
        logger.warning("registry choices of %s unavailable: %s", field_name, exc)
        return ()


def field_options() -> List[FieldOption]:
    """The Config fields a sweep may search (metadata ``tunable``, not deprecated, not reserved by the engine,
    not refused by the sweep), in declaration order, each with the range or choices the metadata gives."""
    out = []
    for name, spec in Config.field_specs().items():
        if not spec.tunable or spec.deprecated or name in RESERVED_FIELDS or name in REFUSED_FIELDS:
            continue
        doc = spec.doc.split("\n")[0]
        if isinstance(spec.default, bool):
            out.append(FieldOption(name, "choice", choices=(False, True), unit=spec.unit, doc=doc))
        elif spec.choices is not None or name in REGISTRY_FIELDS:
            choices = tuple(spec.choices) if spec.choices is not None else registry_choices(name)
            out.append(FieldOption(name, "choice", choices=choices, unit=spec.unit, doc=doc))
        elif isinstance(spec.default, (int, float)):
            kind = "int" if isinstance(spec.default, int) else "float"
            low = spec.minimum if spec.min_inclusive else None
            high = spec.maximum if spec.max_inclusive else None
            out.append(FieldOption(name, kind, low, high, bool(spec.log), spec.step if kind == "int" else None,
                                   unit=spec.unit, doc=doc, needs_bounds=low is None or high is None,
                                   default=spec.default))
    return out


def strategy_options(strategy_name: str) -> List[FieldOption]:
    """The ``strategy.<param>`` entries a rule-only scenario searches (NT-033): the ranges its strategy declares."""
    from neural_trade.strategy.strategies import Strategies, strategy_search_space

    try:
        declared = strategy_search_space(strategy_name)
        defaults = {f.name: f.default for f in dataclasses.fields(Strategies.get(strategy_name))}
    except Exception as exc:
        logger.warning("strategy search space of %s unavailable: %s", strategy_name, exc)
        return []
    out = []
    for param, rule in declared.items():
        kind = "int" if isinstance(defaults.get(param), int) and not isinstance(defaults.get(param), bool) else "float"
        out.append(FieldOption(STRATEGY_PREFIX + param, kind, rule.get("low"), rule.get("high"), bool(rule.get("log", False)),
                               rule.get("step") if kind == "int" else None, doc=f"parameter of strategy {strategy_name}"))
    return out


def fold_choices(config: Optional[Config] = None) -> List[int]:
    """The valid ``FOLD_INDEX`` values, ``[-N_FOLDS, N_FOLDS - 1]`` (dynamic: it follows the config, not the
    metadata, as NT-029 noted); -1 is always the latest fold, the test fold."""
    n = int((config or Config()).N_FOLDS)
    return list(range(-n, n))


def search_block(rules: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The scenario's ``search:`` mapping a selection builds: ``{FIELD: {low, high, log, step | choices}}``;
    a field with no rule keys is written ``FIELD: None`` (the Config metadata's own range)."""
    return {name: (dict(rule) if rule else None) for name, rule in rules.items()}


def search_yaml(block: Mapping[str, Any]) -> str:
    """The ``search:`` block as the YAML a scenario file would hold (what the panel shows before a launch)."""
    import yaml

    return yaml.safe_dump({"search": dict(block)} if block else {"search": {}}, sort_keys=False,
                          default_flow_style=None).rstrip("\n")


def validate_search(scenario: Scenario, block: Mapping[str, Any]) -> Optional[str]:
    """None when ``block`` is a valid space for ``scenario`` (the sweep's own check), else the reason."""
    try:
        SearchSpace.from_scenario(dataclasses.replace(scenario, search=dict(block)))
    except InvalidConfigurationError as exc:
        return str(exc)
    return None


def rules_from_scenario(scenario: Scenario) -> Dict[str, Dict[str, Any]]:
    """The rules a scenario's ``search:`` block starts the panel with ({} when the scenario has none: the sweep
    then uses ``DEFAULT_SEARCH``, which the panel shows and offers as the starting point)."""
    from neural_trade.experiments.sweep import DEFAULT_SEARCH

    block = scenario.search or DEFAULT_SEARCH
    return {name: dict(rule or {}) for name, rule in block.items()}


# ------------------------------------------------------------------ the board
@dataclass
class BoardData:
    sweep_ids: Tuple[str, ...]
    rows: List[Any]                            # LeaderboardRow, ranked on the dev folds (D-020)
    guard_rails: Any
    guard_rail_source: str
    n_cells: int
    n_failed: int
    n_incomplete: int
    notes: List[str] = field(default_factory=list)


def read_board_rows(store: RunStore, sweep_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """The index rows of the given sweeps (their scenario names are the sweep ids); read-only."""
    rows: List[Dict[str, Any]] = []
    for sid in sweep_ids:
        rows += store.index.rows(sid)
    return rows


def build_board(store: RunStore, sweeps: Sequence[SweepInfo], sweep_ids: Sequence[str], *,
                specs_dir="configs/scenarios") -> Optional[BoardData]:
    """The leaderboard (NT-031) of the chosen sweeps on ONE board: guard-rail thresholds, the cost profile and the
    folds come from the first chosen sweep's scenario (its spec file, else the spec the runner stored), and the
    note says when another chosen scenario would have set different ones. None when the index holds no row of
    them yet."""
    from neural_trade.experiments.leaderboard import (
        build_leaderboard, find_scenario_spec, scenario_cost_profile, scenario_guard_rails, spec_parts)

    ids = [s for s in sweep_ids]
    rows = read_board_rows(store, ids)
    if not rows:
        return None
    by_id = {s.sweep_id: s for s in sweeps}
    specs = []
    for sid in ids:
        name = by_id[sid].scenario if sid in by_id else sid
        spec, where = find_scenario_spec(name, specs_dir, store.scenario_dir(sid) / "specs")
        specs.append((sid, spec, where))
    first_id, first_spec, first_where = specs[0]
    rails, source = scenario_guard_rails(first_spec)
    _, backtest, folds = spec_parts(first_spec)
    notes = []
    if len(ids) > 1:
        rails_of = {sid: scenario_guard_rails(spec)[0] for sid, spec, _ in specs}
        differing = [sid for sid, r in rails_of.items() if r != rails]
        notes.append(f"{len(ids)} sweeps on one board: guard-rails, cost profile and fold coverage follow {first_id}"
                     + (f" ({', '.join(differing)} would set other guard-rails)" if differing else ""))
    board = build_leaderboard(rows, guard_rails=rails, store_root=store.root,
                              board_cost=scenario_cost_profile(backtest), spec_folds=folds)
    return BoardData(tuple(ids), board, rails, f"{first_where or 'defaults'}: {source}", len(rows),
                     sum(r["status"] == "failed" for r in rows), sum(r["status"] == "incomplete" for r in rows), notes)


def store_fingerprint(store: RunStore, sweep_ids: Sequence[str] = ()) -> Tuple[Any, ...]:
    """A cheap value that changes when the run index or a sweep summary was written (the board's refresh test)."""
    def stat(path: Path):
        try:
            st = path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    return (stat(store.index_path),) + tuple(stat(store.root / SWEEP_DIR / sid / "sweep.json") for sid in sweep_ids) \
        + (len(sweep_ids),)


def spec_summary(choices: Sequence[ScenarioChoice]) -> List[Dict[str, Any]]:
    """One line per scenario spec for the 'no sweep yet' state: what a sweep of it would search."""
    out = []
    for c in choices:
        if c.scenario is None:
            out.append({"scenario": c.name, "trains": None, "folds": "", "search": c.problem or "unusable"})
            continue
        sc = c.scenario
        out.append({"scenario": sc.name, "trains": bool(sc.run.train), "folds": list(sc.folds),
                    "search": ", ".join(sc.search) if sc.search else "(the default space)"})
    return out


__all__ = ["BoardData", "FieldOption", "MODES", "REGISTRY_FIELDS", "ScenarioChoice", "SweepError", "SweepInfo",
           "YARDSTICK_SCENARIOS", "build_board", "default_board_sweeps", "discover_scenarios", "field_options",
           "fold_choices", "list_sweeps", "read_board_rows", "registry_choices", "rules_from_scenario",
           "search_block", "search_yaml", "spec_summary", "store_fingerprint", "strategy_options", "validate_search"]

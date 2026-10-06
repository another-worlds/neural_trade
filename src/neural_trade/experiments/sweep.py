"""Sweeps (NT-030): a search over Config fields, scored on the development folds through the engine.

    neural-trade sweep configs/scenarios/reference.yaml --mode quick  [--sec-per-step S]
    neural-trade sweep configs/scenarios/reference.yaml --mode optuna --n-trials 40 [--parallel 3] [--resume]

A sweep takes a scenario (data, folds, strategy, scoring, fixed overrides) and a **search space**:
the optional ``search:`` block of the scenario, ``FIELD: {low, high, log, step}`` (or ``choices:``) per
field, or ``FIELD:`` alone to take the Config metadata's range (NT-029). Only fields the metadata marks
``tunable`` can be searched; ``RESAMPLE_MINUTES`` is refused until NT-040 (the sweep and every config
in it stay at 1-minute bars); a number without a finite range needs explicit bounds. Without a
``search:`` block, :data:`DEFAULT_SEARCH` is used. ``variants:`` and ``sweep.axes:`` belong to
``scenario run`` and are refused here.

**What a trial is.** One point of the space, trained as one engine *configuration* named ``t0007`` on
every development fold with one seed, into ``<store>/scenarios/<scenario>-<mode>/``, resumable by the
engine (a finished cell is never trained again). Its value is the mean over the dev folds of the net
Sharpe after costs (D-020). The test fold is never run for a trial and never ranks. A trial with a
failed cell or a non-finite value is a FAILED trial, recorded, never dropped. The ranking function is
:func:`dev_net_sharpe`, which aggregates through NT-031's leaderboard (the mean of each dev fold's seed-mean).

**Quick mode** (no optuna needed): from a measured ``sec_per_step`` it sizes the number of trials,
the epochs and the dev folds so that the whole sweep is estimated at most ``--quick-minutes`` (5), prints the
estimate before it starts and labels every result ``quick`` (reduced epochs, one seed, no winner).

**Optuna mode**: a TPE study in sqlite (``<store>/sweeps/<id>/study.db``), resumable (``--resume``: a
finished trial is never repeated, an interrupted one is finished). Before it starts it prints and
records its GPU budget: trials x dev folds x steps x sec_per_step plus the top-5 x 3-seed re-run, and
refuses above ``--max-hours`` (12: one night; a larger budget goes to the owner). After the search the
top 5 successful trials are re-run with 3 seeds (dev folds, and the test fold for display) and ranked by the
leaderboard on the dev folds; the winner is its top row that no guard-rail disqualifies and that trained stably.

**Parallel** (``--parallel N``): trials launch in batches of N processes (``scenario run
--claim-cells``, so no two processes train a cell). N above 1 needs NT-035's record
(``runs/experiments/gpu_measurements_v1/parallel_n.json``: ``allowed_n``). The GPU-free check of
RUNBOOK "GPU rules" runs before each batch (no own trial is running then), at every N; with N above 1,
after each batch (search and re-run) the GPU's memory and utilisation are compared with the level the
record gives for N own processes, and a higher reading (someone else is on the GPU) stops the launching.
At N = 1 the record is not used at all (its levels were measured on another setup).

**Stops.** A stop (busy GPU, a reading above the level) during the search or the re-run leaves the sweep
``stopped`` with its reason and the CLI exits 1; ``--resume`` finishes it (finished cells are never trained
again). Only trials that pass the search-time guard-rails (:data:`SEARCH_TIME_RAILS`) enter the re-run.

**Rule-only scenarios (NT-033).** A scenario with ``run: {train: false}`` (the classic TA rules of
strategy/ta_rules.py, the manual-search baseline of the yardstick) is searched the same way, with the space
``strategy.<param>: {low, high, log, step}`` (or ``strategy.<param>:`` alone for the range the strategy declares,
``strategy.strategy_search_space``): a trial's ``strategy.*`` values become the strategy's params, the cells score
on the same dev folds by the same net Sharpe, no network is trained, so no ``sec_per_step`` is needed (0), the
GPU-free check is skipped and the training-health check does not apply. The per-cell ``overhead_s`` is then a
CPU estimate (pass ``--overhead-s`` accordingly); the re-run's extra seeds change nothing for a rule.

**Tunability decision (NT-029 QA note).** ``PATIENCE`` (ReduceLROnPlateau) is tunable, and the space caps
it at the scenario's ``EARLY``. ``EARLY`` (EarlyStopping patience) is NOT tunable: like ``EPOCHS`` it sets
how long a trial trains, so searching it would make trials cost different amounts and change what the
budget means.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.experiments.dataset import setup_of
from neural_trade.experiments.runner import Runner
from neural_trade.experiments.scenario import NAME_RE, RESERVED_FIELDS, Scenario, short_hash
from neural_trade.experiments.store import RunStore

logger = logging.getLogger(__name__)

SWEEP_DIR = "sweeps"
QUICK, OPTUNA = "quick", "optuna"
MODES = (QUICK, OPTUNA)
SWEEP_SCHEMA_VERSION = 1
DEFAULT_PARALLEL_RECORD = "runs/experiments/gpu_measurements_v1/parallel_n.json"
STRATEGY_PREFIX = "strategy."    # a search key that names a parameter of the scenario's strategy (NT-033)
REFUSED_FIELDS = {"RESAMPLE_MINUTES": "a sweep stays at 1-minute bars until NT-040 is done"}
# Used when the scenario has no ``search:`` block: the optimiser, the batch and the two heads' loss weights.
DEFAULT_SEARCH: Dict[str, Optional[Dict[str, Any]]] = {
    "LR": {"low": 1e-4, "high": 1e-2, "log": True},
    "BATCH_SIZE": {"low": 128, "high": 1024, "log": True},
    "LAMBDA_DIR": {"low": 0.1, "high": 3.0},
    "LAMBDA_CRPS": {"low": 0.1, "high": 3.0},
}
QUICK_EPOCHS = 3                 # quick mode trains at most this many epochs per trial
QUICK_MIN_TRIALS = 4
QUICK_MAX_TRIALS = 64
# The watch level of NT-030 (4): the record's level for N own processes plus these margins.
WATCH_SM_MARGIN_PCT = 25.0
WATCH_FB_MARGIN_MB = 1024.0
GPU_BUSY_FB_MB = 2000.0          # RUNBOOK "GPU rules": busy above this memory ...
GPU_BUSY_SM_PCT = 30.0           # ... or above this median utilisation


class SweepError(InvalidConfigurationError):
    """A sweep cannot start: bad space, refused option, budget over the limit (a ValueError)."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, obj: Any) -> None:
    """Atomic replace (a summary, rewritten after every trial)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------ search space
@dataclass(frozen=True)
class SearchParam:
    name: str
    kind: str                                   # "int" | "float" | "cat"
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "kind": self.kind}
        if self.kind == "cat":
            d["choices"] = list(self.choices or ())
        else:
            d.update(low=self.low, high=self.high, log=self.log, step=self.step)
        return d

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "cat":
            return (self.choices or ())[int(rng.integers(len(self.choices or ())))]
        if self.log:
            v = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            v = rng.uniform(self.low, self.high)
        return self._snap(v)

    def _snap(self, v: float) -> Any:
        if self.kind == "int":
            step = int(self.step or 1)
            v = int(round((v - self.low) / step)) * step + int(self.low)
            return int(min(max(v, int(self.low)), int(self.high)))
        return float(min(max(v, self.low), self.high))

    def suggest(self, trial) -> Any:
        if self.kind == "cat":
            return trial.suggest_categorical(self.name, list(self.choices or ()))
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), step=int(self.step or 1),
                                     log=bool(self.log and not self.step))
        return trial.suggest_float(self.name, float(self.low), float(self.high), log=self.log)


@dataclass(frozen=True)
class SearchSpace:
    params: Tuple[SearchParam, ...]

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.params]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.params]

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        return {p.name: p.sample(rng) for p in self.params}

    def suggest(self, trial) -> Dict[str, Any]:
        return {p.name: p.suggest(trial) for p in self.params}

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "SearchSpace":
        search = scenario.search or DEFAULT_SEARCH
        where = scenario.where()
        specs = Config.field_specs()
        base = scenario.base()
        params = []
        for name, rule in search.items():
            rule = dict(rule or {})
            if name.startswith(STRATEGY_PREFIX):
                params.append(cls._strategy_param(scenario, name, rule, where))
                continue
            spec = specs.get(name)
            if spec is None:
                raise SweepError(f"{where}: search.{name}: unknown Config field")
            if name in REFUSED_FIELDS:
                raise SweepError(f"{where}: search.{name} is refused: {REFUSED_FIELDS[name]}")
            if name in RESERVED_FIELDS:
                raise SweepError(f"{where}: search.{name} is set by the engine ({RESERVED_FIELDS[name]})")
            if not spec.tunable or spec.deprecated:
                raise SweepError(f"{where}: search.{name} is not tunable (Config metadata, NT-029); tunable fields: "
                                 f"{sorted(n for n, s in specs.items() if s.tunable and not s.deprecated)}")
            unknown = sorted(set(rule) - {"low", "high", "log", "step", "choices"})
            if unknown:
                raise SweepError(f"{where}: search.{name}: unknown key(s) {unknown}")
            params.append(cls._param(spec, rule, base, where))
        return cls(tuple(params))

    @staticmethod
    def _strategy_param(scenario: Scenario, name: str, rule: Mapping[str, Any], where: str) -> SearchParam:
        """``strategy.<param>``: a parameter the scenario's strategy declares searchable (NT-033); the rule's
        keys override the declared range."""
        from neural_trade.strategy.strategies import Strategies, strategy_search_space

        sname, param = scenario.strategy.name, name[len(STRATEGY_PREFIX):]
        declared = strategy_search_space(sname)
        if param not in declared:
            raise SweepError(f"{where}: search.{name}: strategy {sname!r} declares no searchable parameter "
                             f"{param!r}; it declares {sorted(declared)}")
        unknown = sorted(set(rule) - {"low", "high", "log", "step"})
        if unknown:
            raise SweepError(f"{where}: search.{name}: unknown key(s) {unknown}")
        merged = {**declared[param], **rule}
        default = next(f.default for f in dataclasses.fields(Strategies.get(sname)) if f.name == param)
        kind = "int" if isinstance(default, int) and not isinstance(default, bool) else "float"
        low, high = merged["low"], merged["high"]
        if not low < high:
            raise SweepError(f"{where}: search.{name}: low must be below high, got {low} .. {high}")
        log = bool(merged.get("log", False))
        if log and low <= 0:
            raise SweepError(f"{where}: search.{name}: log sampling needs low > 0, got {low}")
        return SearchParam(name, kind, float(low), float(high), log, merged.get("step") if kind == "int" else None)

    @staticmethod
    def _param(spec, rule: Mapping[str, Any], base: Config, where: str) -> SearchParam:
        name, default = spec.name, spec.default
        if isinstance(default, bool) or spec.choices is not None or "choices" in rule:
            choices = tuple(rule.get("choices") or (spec.choices if spec.choices is not None else (False, True)))
            if not choices:
                raise SweepError(f"{where}: search.{name}: empty choices")
            for c in choices:
                why = spec.check(c)
                if why:
                    raise SweepError(f"{where}: search.{name}: {why}")
            return SearchParam(name, "cat", choices=choices)
        if not isinstance(default, (int, float)):
            raise SweepError(f"{where}: search.{name}: only numbers, flags and named choices can be searched "
                             f"(the field holds {type(default).__name__})")
        kind = "int" if isinstance(default, int) else "float"
        low = rule.get("low", spec.minimum if spec.min_inclusive else None)
        high = rule.get("high", spec.maximum if spec.max_inclusive else None)
        if low is None or high is None:
            raise SweepError(f"{where}: search.{name} has no finite, inclusive range in the Config metadata "
                             f"({spec.range_text() or 'unbounded'}): give both `low` and `high`")
        if name == "PATIENCE":
            high = min(high, int(base.EARLY))            # ReduceLROnPlateau patience above EARLY never acts
        if not low < high:
            raise SweepError(f"{where}: search.{name}: low must be below high, got {low} .. {high}")
        for bound in (low, high):
            why = spec.check(bound)
            if why:
                raise SweepError(f"{where}: search.{name}: {why}")
        log = bool(rule.get("log", spec.log))
        if log and low <= 0:
            raise SweepError(f"{where}: search.{name}: log sampling needs low > 0, got {low}")
        step = rule.get("step", spec.step if kind == "int" else None)
        return SearchParam(name, kind, float(low), float(high), log, step)


# ------------------------------------------------------------------ scoring (the leaderboard's place)
@dataclass
class TrialScore:
    """A trial's ranking number: the dev-fold mean of the net Sharpe (per fold: the mean over seeds)."""

    value: Optional[float]
    per_fold: Dict[int, float] = field(default_factory=dict)
    n_cells: int = 0
    spread: Optional[float] = None              # std of the per-fold values (folds are the unit, D-046)
    mean_trades: Optional[float] = None
    reason: Optional[str] = None                # why value is None
    ineligible: Optional[str] = None            # the search-time guard-rails it fails (see SEARCH_TIME_RAILS)

    @property
    def eligible(self) -> bool:
        """A finite value and every search-time guard-rail passed: may enter the re-run, may lead."""
        return self.value is not None and self.ineligible is None

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "per_fold": {str(k): v for k, v in sorted(self.per_fold.items())},
                "n_cells": self.n_cells, "spread": self.spread, "mean_trades": self.mean_trades,
                "reason": self.reason, "eligible": self.eligible, "ineligible": self.ineligible}


# The leaderboard's guard-rails a trial's single-seed dev cells can already settle at search time: the rank value
# is finite, the trial trades on every dev fold, its net Sharpe was computed at the board's cost profile, and it
# has a scored cell on every dev fold. The return-based rails (drawdown, buy-and-hold, random null) are judged on
# the re-run's seed mean only. A trial failing one of these never enters the re-run, and Optuna is told
# INELIGIBLE_VALUE for it instead of its (idle or cost-mismatched) Sharpe.
SEARCH_TIME_RAILS = ("status", "dev_data", "ranking_value", "min_trades", "cost_profile", "fold_coverage")
INELIGIBLE_VALUE = -1.0e6                       # below any real annualised net Sharpe


def dev_net_sharpe(rows: Sequence[Mapping[str, Any]], dev_folds: Sequence[int], *, store_root=None,
                   guard_rails=None, board_cost=None) -> TrialScore:
    """Dev-fold net Sharpe after costs of one configuration's index rows: NT-031's aggregation
    (:func:`~neural_trade.experiments.leaderboard.build_leaderboard`: the mean of each dev fold's seed-mean,
    the spread between fold means). A dev fold without a finished cell, or with a non-finite value, makes
    the trial failed. ``guard_rails`` / ``board_cost`` (the scenario's; default the leaderboard's defaults) set
    ``ineligible`` from the :data:`SEARCH_TIME_RAILS` that fail. Test-fold rows are never read here."""
    from neural_trade.experiments.leaderboard import RANK_METRIC, build_leaderboard

    problems = []
    done_rows = []
    for fold in dev_folds:
        mine = [r for r in rows if r.get("role") == "dev" and r.get("fold") == fold]
        done = [r for r in mine if r.get("status") == "done"]
        if not done:
            err = next((r.get("error") for r in mine if r.get("status") == "failed"), None)
            problems.append(f"fold {fold}: " + (f"failed ({err})" if err else "no finished cell"))
            continue
        if any(r.get("sharpe_net") is None or not math.isfinite(float(r["sharpe_net"])) for r in done):
            problems.append(f"fold {fold}: non-finite net Sharpe")
            continue
        done_rows += done
    n_cells = len(done_rows)
    if problems:
        return TrialScore(None, {}, n_cells, reason="; ".join(problems))
    row = build_leaderboard(done_rows, spec_folds=list(dev_folds), store_root=store_root, guard_rails=guard_rails,
                            board_cost=board_cost)[0]
    failed = [f"{g.name}: {g.detail}" for g in row.guard_rails if g.name in SEARCH_TIME_RAILS and not g.passed]
    return TrialScore(row.dev.values.get(RANK_METRIC), {int(k): v for k, v in row.dev.fold_values[RANK_METRIC].items()},
                      n_cells, row.dev.spread.get(RANK_METRIC), row.dev.values.get("n_trades"),
                      ineligible="; ".join(failed) or None)


HEALTH_MAX_NONFINITE_GRAD_STEPS = 0


def _loss_problem(doc: Mapping[str, Any], key: str) -> Optional[str]:
    """'missing' when ``key`` is absent, 'non-finite' when it is null (the trainer's writers turn NaN and inf
    into null: ``telemetry.epoch_logger._plain``, ``trainer._finite_or_none``) or not a finite number."""
    if key not in doc:
        return "missing"
    v = doc[key]
    try:
        return None if v is not None and math.isfinite(float(v)) else "non-finite"
    except (TypeError, ValueError):
        return "non-finite"


def run_health(run_dir, *, max_nonfinite_grad_steps: int = HEALTH_MAX_NONFINITE_GRAD_STEPS) -> Optional[str]:
    """Why a finished cell's training is unstable or unverifiable, or None. Read from what the trainer writes
    (``JsonlEpochLogger`` and the served-epoch record): a ``loss`` or ``val_loss`` in any epoch of
    ``metrics.jsonl``, or ``weights_val_loss`` / ``val_loss`` in ``status.json``, that is null (the writers store
    NaN and inf as null) or non-finite; any of these keys missing; ``metrics.jsonl`` or ``status.json``
    missing or without an epoch (the engine's trainer always writes both); more non-finite-gradient steps than
    the limit. A trial with such a cell is FAILED however good its Sharpe looks."""
    d = Path(run_dir)
    reasons = []
    nonfinite_steps = 0.0
    try:
        lines = [ln for ln in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        lines = []
        reasons.append("metrics.jsonl missing (no epoch logged)")
    else:
        if not lines:
            reasons.append("metrics.jsonl has no epoch")
    for line in lines:
        try:
            m = json.loads(line)
        except ValueError:
            reasons.append("metrics.jsonl has an unreadable line")
            continue
        for key in ("loss", "val_loss"):
            why = _loss_problem(m, key)
            if why:
                reasons.append(f"{why} {key} in epoch {m.get('epoch')}")
        try:
            nonfinite_steps += float(m.get("nonfinite_grad_steps") or 0.0)
        except (TypeError, ValueError):
            reasons.append(f"unreadable nonfinite_grad_steps in epoch {m.get('epoch')}")
    try:
        status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        reasons.append("status.json missing or unreadable")
    else:
        for key, what in (("weights_val_loss", "the served epoch"), ("val_loss", "the last epoch")):
            why = _loss_problem(status, key)
            if why:
                reasons.append(f"{why} {key} in status.json ({what})")
    if nonfinite_steps > max_nonfinite_grad_steps:
        reasons.append(f"nonfinite_grad_steps {nonfinite_steps:g} > {max_nonfinite_grad_steps}")
    return "; ".join(dict.fromkeys(reasons)) or None


def held_out_columns(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """The test fold's numbers, shown beside a ranking and never used by it."""
    test = [r for r in rows if r.get("role") == "test" and r.get("status") == "done"]

    def mean(col):
        v = [r[col] for r in test if r.get(col) is not None]
        return float(np.mean(v)) if v else None

    return {"test_sharpe_net": mean("sharpe_net"), "test_max_drawdown": mean("max_drawdown"),
            "test_n_trades": mean("n_trades"), "n_test_cells": len(test)}


# ------------------------------------------------------------------ GPU: free check, watch, record
@dataclass
class GpuStatus:
    free: bool
    detail: Dict[str, Any] = field(default_factory=dict)


def _cpu_only() -> bool:
    return os.environ.get("CUDA_VISIBLE_DEVICES", None) == "-1"


def parse_dmon(text: str) -> List[Dict[str, float]]:
    """Samples of ``nvidia-smi dmon -s um`` by column NAME (the header ``# gpu sm mem enc dec jpg ofa fb bar1
    ccpm``: fb is the 8th column, not the 4th), one dict per sample row."""
    names: List[str] = []
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if line.lstrip().startswith("#"):
            if not names and len(parts) > 2 and "gpu" in parts[1:] and "sm" in parts[1:]:
                names = parts[1:]
            continue
        if not names or len(parts) < len(names):
            continue
        row = {}
        for n, v in zip(names, parts):
            try:
                row[n] = float(v)
            except ValueError:
                row[n] = float("nan")
        out.append(row)
    return out


def gpu_status_from_dmon(text: str) -> GpuStatus:
    """RUNBOOK "GPU rules": busy when the median fb is above 2000 MB or the median sm above 30%."""
    samples = [r for r in parse_dmon(text) if "sm" in r and "fb" in r]
    if not samples:
        return GpuStatus(False, {"error": "nvidia-smi dmon gave no samples (or no sm / fb columns)"})
    med_sm = float(np.median([r["sm"] for r in samples]))
    med_fb = float(np.median([r["fb"] for r in samples]))
    return GpuStatus(med_fb <= GPU_BUSY_FB_MB and med_sm <= GPU_BUSY_SM_PCT,
                     {"median_sm_pct": med_sm, "median_fb_mb": med_fb, "samples": len(samples)})


def nvidia_smi_gpu_check(samples: int = 10, *, run_dmon: Optional[Callable[[int], str]] = None) -> GpuStatus:
    """The GPU-free check of RUNBOOK "GPU rules" (10 one-second ``dmon`` samples)."""
    if _cpu_only():
        return GpuStatus(True, {"note": "CUDA_VISIBLE_DEVICES=-1: a CPU run, no GPU check"})
    try:
        if run_dmon is None:
            text = subprocess.run(["nvidia-smi", "dmon", "-s", "um", "-c", str(samples)], capture_output=True,
                                  text=True, timeout=samples + 30).stdout
        else:
            text = run_dmon(samples)
    except (OSError, subprocess.SubprocessError) as exc:
        return GpuStatus(False, {"error": f"nvidia-smi failed: {exc}"})
    return gpu_status_from_dmon(text)


class NvidiaSmiMonitor:
    """Samples GPU utilisation and memory once a second while a batch runs (a thread)."""

    def __init__(self):
        self._stop = None
        self._thread = None
        self._sm: List[float] = []
        self._fb: List[float] = []

    def start(self) -> None:
        import threading

        self._stop = threading.Event()
        self._sm, self._fb = [], []

        def loop():
            while not self._stop.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                         timeout=10).stdout.strip().splitlines()[0]
                    a, b = (float(x) for x in out.split(","))
                    self._sm.append(a)
                    self._fb.append(b)
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    pass
                self._stop.wait(1.0)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=15)
        return {"mean_sm_pct": float(np.mean(self._sm)) if self._sm else None,
                "mean_fb_mb": float(np.mean(self._fb)) if self._fb else None,
                "peak_fb_mb": float(np.max(self._fb)) if self._fb else None}


def record_setup_warnings(record: Mapping[str, Any], config: Config) -> List[str]:
    """Where NT-035's measured setup (its ``setup`` text) differs from the setup being swept. The record was
    measured before D-047 (close-only input), so a record that does not name the OHLCV input is flagged."""
    import re

    text = str(record.get("setup") or "")
    if not text:
        return ["the parallel record states no measured setup"]
    out = []
    for name in ("LOOKBACK", "BATCH_SIZE"):
        m = re.search(name + r"\s+(\d+)", text)
        if m and int(m.group(1)) != int(getattr(config, name)):
            out.append(f"{name} {m.group(1)} measured, {getattr(config, name)} swept")
    m = re.search(r"HORIZON_STEPS\s+(\[[^\]]*\])", text)
    if m and [int(x) for x in json.loads(m.group(1))] != [int(h) for h in config.HORIZON_STEPS]:
        out.append(f"HORIZON_STEPS {m.group(1)} measured, {list(config.HORIZON_STEPS)} swept")
    if list(config.INPUT_SERIES) != ["close"] and "OHLCV" not in text and "INPUT_SERIES" not in text:
        out.append("the record does not name its input layout (measured before D-047's OHLCV default); the swept "
                   f"input is {list(config.INPUT_SERIES)}")
    return out


def load_parallel_record(path) -> Dict[str, Any]:
    """NT-035's result: ``allowed_n`` and the per-N ``utilization``; no file means N = 1."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return {"allowed_n": 1, "utilization": {}, "path": str(p) if p else None, "found": False}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {"allowed_n": int(doc.get("allowed_n", 1)), "utilization": doc.get("utilization") or {},
            "setup": doc.get("setup"), "measured_utc": doc.get("measured_utc"), "path": str(p), "found": True}


def exceeds_watch_level(reading: Mapping[str, Optional[float]], level: Optional[Mapping[str, float]]) -> Optional[str]:
    """The reason when a batch's GPU reading is above the recorded level for its N (someone else is on the
    GPU), else None. The record's note says memory is the reliable signal (the desktop alone shows sm ~40%)."""
    if not level or reading.get("peak_fb_mb") is None:
        return None
    if reading["peak_fb_mb"] > level["peak_fb_mb"] + WATCH_FB_MARGIN_MB:
        return f"peak memory {reading['peak_fb_mb']:.0f} MB above the recorded {level['peak_fb_mb']:.0f} MB"
    if reading.get("mean_sm_pct") is not None and reading["mean_sm_pct"] > level["mean_sm_pct"] + WATCH_SM_MARGIN_PCT:
        return f"mean utilisation {reading['mean_sm_pct']:.0f}% above the recorded {level['mean_sm_pct']:.0f}%"
    return None


# ------------------------------------------------------------------ sizing and budget
def steps_per_epoch(train_n: int, batch_size: int) -> int:
    return int(math.ceil(train_n / max(1, int(batch_size))))


def cell_seconds(steps: int, epochs: int, sec_per_step: float, overhead_s: float) -> float:
    """One cell: epochs x steps x sec_per_step plus the fixed cost (data prep, calibration, scoring)."""
    return float(epochs) * float(steps) * float(sec_per_step) + float(overhead_s)


@dataclass
class QuickPlan:
    n_trials: int
    epochs: int
    folds: List[int]
    seconds_per_trial: float
    estimated_s: float
    budget_s: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def size_quick(*, dev_steps: Mapping[int, int], epochs_cap: int, sec_per_step: float, overhead_s: float,
               budget_s: float, min_trials: int = QUICK_MIN_TRIALS, max_trials: int = QUICK_MAX_TRIALS) -> QuickPlan:
    """The largest quick sweep whose estimate stays within ``budget_s``: epochs from min(cap, 3) down to 1, all
    dev folds then only the newest one, and at least ``min_trials`` trials when any option allows it."""
    dev = sorted(dev_steps)
    options = []
    for epochs in range(max(1, min(int(epochs_cap), QUICK_EPOCHS)), 0, -1):
        for folds in (dev, dev[-1:]):
            per_trial = sum(cell_seconds(dev_steps[f], epochs, sec_per_step, overhead_s) for f in folds)
            n = min(int(budget_s // per_trial), max_trials)
            options.append((n, epochs, folds, per_trial))
            if len(dev) == 1:
                break
    good = [o for o in options if o[0] >= min_trials]
    n, epochs, folds, per_trial = good[0] if good else max(options, key=lambda o: o[0])
    if n < 1:
        raise SweepError(f"even one trial (1 epoch, fold {folds[-1]}) is estimated at {per_trial:.0f} s, over the "
                         f"{budget_s:.0f} s quick budget (sec_per_step {sec_per_step}): use a smaller layout "
                         "(for example the 6-hour screen layout) or optuna mode")
    return QuickPlan(n, epochs, [int(f) for f in folds], per_trial, n * per_trial, float(budget_s))


# The Config fields that change a training step's cost: a stored run's sec_per_step stands for this setup only
# when every one of them is in its config.yaml with the same value.
SETUP_FIELDS = ("BATCH_SIZE", "INPUT_SERIES", "INDICATOR_FAMILIES", "LOOKBACK", "HORIZON_STEPS", "RESAMPLE_MINUTES",
                "MAX_SEQUENCE_COUNT", "MODEL_NAME", "ATTENTION_MODE", "DETERMINISTIC_GRU", "PROBE_GRADIENTS")


def _plain_value(v: Any) -> Any:
    """Tuples as lists, recursively: a YAML-loaded value and a Config value compare equal when they hold the same."""
    if isinstance(v, (list, tuple)):
        return [_plain_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain_value(x) for k, x in v.items()}
    return v


def current_device() -> str:
    """"cpu" or "gpu": what a run started now would train on."""
    if _cpu_only():
        return "cpu"
    try:
        import neural_trade  # noqa: F401  (puts the CUDA DLLs on PATH before tensorflow)
        import tensorflow as tf

        return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    except Exception:  # noqa: BLE001 - no TensorFlow: the run cannot be a GPU run
        return "cpu"


def run_device(run_dir) -> Optional[str]:
    """The device a stored run trained on, from its env.json (its ``gpus`` list); None when unknown."""
    try:
        env = json.loads((Path(run_dir) / "env.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return "gpu" if env.get("gpus") else "cpu"


def setup_mismatch(row: Mapping[str, Any], run_dir, config: Config, dataset_sha: Optional[str],
                   device: str) -> Optional[str]:
    """Why a stored run is NOT the same setup as ``config`` (dataset fingerprint, the Config fields of
    :data:`SETUP_FIELDS`, the device), or None when it is. The fields are compared with the RAW keys of the run's
    own config.yaml, never through ``Config.from_yaml`` (which would fill today's defaults into a file written
    before a field existed, e.g. a pre-D-047 close-only run without INPUT_SERIES): a field missing from the
    stored file is unknown, and unknown is a mismatch. The device comes from the run's env.json."""
    import yaml

    if not dataset_sha or row.get("dataset_sha256") != dataset_sha:
        return "dataset fingerprint differs or is unknown"
    try:
        theirs = yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable config.yaml is an unknown setup
        return "config.yaml unreadable"
    if not isinstance(theirs, dict):
        return "config.yaml unreadable"
    for name in SETUP_FIELDS:
        if name not in theirs:
            return f"{name} missing from the stored config.yaml (unknown: written before the field existed)"
        mine = _plain_value(getattr(config, name))
        if _plain_value(theirs[name]) != mine:
            return f"{name} differs ({theirs[name]!r} against {mine!r})"
    dev = run_device(run_dir)
    if dev != device:
        return f"device differs ({dev} against {device})"
    return None


def latest_sec_per_step(store: RunStore, config: Config, dataset_sha: Optional[str], device: str
                        ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """(sec_per_step of the newest finished run of exactly the same setup, the reasons other runs were
    refused). Same setup: the dataset fingerprint, every field of :data:`SETUP_FIELDS` present in the run's
    config.yaml with the same value, and the device. Never falls back to another setup."""
    best, refused = None, []
    for r in store.index.rows(status="done"):
        if r.get("sec_per_step") is None:
            continue
        why = setup_mismatch(r, store.root / r["run_dir"], config, dataset_sha, device)
        if why:
            refused.append(f"{r['run_id']}: {why}")
            continue
        key = r.get("created_utc") or ""
        if best is None or key > best[0]:
            best = (key, {"sec_per_step": float(r["sec_per_step"]), "run_id": r["run_id"], "run_dir": r["run_dir"]})
    return (best[1] if best else None), refused


def code_source_dir() -> Path:
    """The directory that holds the imported ``neural_trade`` package (put first on a trial process's PYTHONPATH)."""
    import neural_trade

    return Path(neural_trade.__file__).resolve().parent.parent


def code_info() -> Dict[str, str]:
    """Which code this sweep runs (and launches its trials with): the package's source directory and git sha."""
    from neural_trade.utils.env import git_sha

    return {"source_dir": str(code_source_dir()), "git_sha": git_sha()}


# ------------------------------------------------------------------ the sweep
@dataclass
class SweepOptions:
    mode: str = OPTUNA
    n_trials: int = 30
    stop_after: Optional[int] = None            # at most this many NEW trials in this call, then stop (no re-run)
    max_hours: float = 12.0
    parallel: int = 1
    parallel_record: Optional[str] = DEFAULT_PARALLEL_RECORD
    sec_per_step: Optional[float] = None        # else the latest run of the same setup in the index
    quick_minutes: float = 5.0
    overhead_s: float = 30.0                    # per cell: data prep, calibration pass, scoring (an estimate)
    top_k: int = 5
    rerun_seeds: int = 3
    max_nonfinite_grad_steps: int = 0           # a trial with more non-finite-gradient steps is FAILED
    device: Optional[str] = None                # "cpu" | "gpu" for the sec_per_step match (default: detected)
    sampler_seed: int = 0
    resume: bool = False
    when_busy: str = "stop"                     # "stop" | "wait" when the GPU-free check fails
    wait_poll_s: float = 300.0
    wait_max_s: float = 7200.0
    dry_run: bool = False                       # print the estimate / budget and stop


@dataclass
class SweepResult:
    sweep_id: str
    mode: str
    label: str
    directory: str
    budget: Dict[str, Any]
    trials: List[Dict[str, Any]]
    state: str                                  # complete | stopped | dry_run | quick_complete
    stop_reason: Optional[str] = None
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class Sweep:
    """One sweep of one scenario into a run store (see the module docstring).

    ``trainer`` goes to the engine's runner (tests inject a fast one); ``runner_factory(scenario)``,
    ``gpu_check()``, ``monitor_factory()``, ``launcher(spec_paths, store)`` and ``sleep`` are the
    injection points the stubbed tests use.
    """

    def __init__(self, scenario: Scenario, store="runs", options: Optional[SweepOptions] = None, *, index_path=None,
                 trainer: Optional[Callable[..., Any]] = None, runner_factory: Optional[Callable[..., Runner]] = None,
                 gpu_check: Optional[Callable[[], GpuStatus]] = None, monitor_factory=None,
                 launcher: Optional[Callable[..., List[int]]] = None, sleep: Callable[[float], None] = time.sleep,
                 announce: Optional[Callable[[str], None]] = None, check_components: bool = True):
        self.scenario = scenario
        self.store = store if isinstance(store, RunStore) else RunStore(store, index_path)
        self.options = options or SweepOptions()
        if self.options.mode not in MODES:
            raise SweepError(f"--mode must be one of {list(MODES)}, got {self.options.mode!r}")
        self.sweep_id = f"{scenario.name}-{self.options.mode}"[:48].rstrip("._-")
        if not NAME_RE.match(self.sweep_id):
            raise SweepError(f"scenario name {scenario.name!r} gives an invalid sweep id {self.sweep_id!r}")
        self.directory = self.store.root / SWEEP_DIR / self.sweep_id
        self.trainer = trainer
        self.check_components = check_components
        self._runner_factory = runner_factory
        self.gpu_check = gpu_check or nvidia_smi_gpu_check
        self.monitor_factory = monitor_factory or NvidiaSmiMonitor
        self.launcher = launcher or self._subprocess_launcher
        self.sleep = sleep
        self.announce = announce or (lambda text: logger.info("%s", text))
        from neural_trade.experiments.dataset import LayoutCache

        self._layouts = LayoutCache()
        self.label = QUICK if self.options.mode == QUICK else OPTUNA
        self.space: Optional[SearchSpace] = None
        self.dev_folds: List[int] = []
        self.test_folds: List[int] = []
        self.dev_steps: Dict[int, int] = {}
        self.record: Dict[str, Any] = {}
        self.trial_log: Dict[int, Dict[str, Any]] = {}
        self._extra_overrides: Dict[str, Any] = {}

    # -------------------------------------------------------------- setup
    @property
    def summary_path(self) -> Path:
        return self.directory / "sweep.json"

    def _check_scenario(self) -> None:
        sc, o = self.scenario, self.options
        where = sc.where()
        if set(sc.variants) - {"default"} or any(sc.variants.values()):
            raise SweepError(f"{where}: a sweep searches the `search:` space; `variants:` belong to `scenario run` "
                             "(put fixed settings in `overrides:`)")
        if sc.axes:
            raise SweepError(f"{where}: `sweep.axes` belong to `scenario run`; a sweep takes `search:` instead")
        if not sc.run.train:
            config_keys = sorted(k for k in sc.search if not k.startswith(STRATEGY_PREFIX))
            if not sc.search or config_keys:
                raise SweepError(f"{where}: run.train is false (a rule-only scenario): its `search:` block must name "
                                 f"the strategy's parameters as `strategy.<param>` (a Config field changes nothing "
                                 f"here), got {config_keys or 'no search block'}")
        for fname in REFUSED_FIELDS:
            if fname in sc.overrides and sc.overrides[fname] != Config.field_specs()[fname].default:
                raise SweepError(f"{where}: {fname} is refused in a sweep: {REFUSED_FIELDS[fname]}")
        if int(sc.base().RESAMPLE_MINUTES) != 1:
            raise SweepError(f"{where}: RESAMPLE_MINUTES is {sc.base().RESAMPLE_MINUTES}: {REFUSED_FIELDS['RESAMPLE_MINUTES']} "
                             "(NT-040 lifts this)")
        if o.n_trials < 1 or o.top_k < 1 or o.rerun_seeds < 1 or o.parallel < 1 or o.max_hours <= 0:
            raise SweepError("n_trials, top_k, rerun_seeds and parallel must be >= 1 and max_hours > 0")
        if o.when_busy not in ("stop", "wait"):
            raise SweepError("when_busy must be 'stop' or 'wait'")
        if self.directory.exists() and not o.resume and not o.dry_run:
            raise SweepError(f"{self.directory} exists: pass resume (--resume) to continue that sweep; a sweep never "
                             "overwrites an earlier one")
        rec = load_parallel_record(o.parallel_record)
        self.record = rec
        if o.parallel > 1 and o.parallel > rec["allowed_n"]:
            raise SweepError(f"--parallel {o.parallel} is refused: the GPU measurement record "
                             f"({rec['path']}, found: {rec['found']}) allows N = {rec['allowed_n']}; "
                             "no record means N = 1 (NT-035)")

    def _probe(self) -> None:
        """Plan the scenario once (validates everything, loads the data layout): the dev and test folds and
        each dev fold's training size."""
        sc = self.scenario
        probe = dataclasses.replace(sc, name=self.sweep_id, variants={"probe": {}}, axes={}, search={},
                                    seeds=[int(sc.seeds[0])])
        planned = self._factory(probe).plan()
        self.dev_folds = [pc.cell.fold for pc in planned if pc.role == "dev"]
        self.test_folds = [pc.cell.fold for pc in planned if pc.role == "test"]
        if not self.dev_folds:
            raise SweepError(f"{sc.where()}: the scenario's folds {sc.folds} contain no development fold "
                             "(the latest usable fold is the test fold; a sweep ranks on dev folds only, D-020)")
        self.base_config = planned[0].config
        self.dataset_sha = planned[0].dataset.get("sha256")
        self.train_n = {pc.cell.fold: int(pc.fold["blocks"]["train"]["n"]) for pc in planned}
        base_batch = int(self.base_config.BATCH_SIZE)
        batch_param = next((p for p in (self.space.params if self.space else ()) if p.name == "BATCH_SIZE"), None)
        low = int(batch_param.low) if batch_param is not None else base_batch
        # steps per epoch of each dev fold: the upper bound (the space's lowest batch, for the refusal) and the
        # expected value (each trial's own batch drawn from the space, for the figure printed beside it)
        self.dev_steps = {f: steps_per_epoch(self.train_n[f], low) for f in self.dev_folds}
        rng = np.random.default_rng(0)
        batches = ([int(self.space.sample(rng)["BATCH_SIZE"]) for _ in range(256)] if batch_param is not None
                   else [base_batch])
        self.expected_steps = {f: float(np.mean([steps_per_epoch(self.train_n[f], b) for b in batches]))
                               for f in self.train_n}
        self.upper_steps = {f: steps_per_epoch(self.train_n[f], low) for f in self.train_n}
        self.setup = setup_of(self.base_config)
        self.epochs = int(self.base_config.EPOCHS)

    def _factory(self, scenario: Scenario) -> Runner:
        if self._runner_factory is not None:
            return self._runner_factory(scenario)
        r = Runner(scenario, self.store, trainer=self.trainer, check_components=self.check_components, claim_cells=True)
        r._layouts = self._layouts
        return r

    def _sec_per_step(self) -> Tuple[float, Dict[str, Any]]:
        if not self.scenario.run.train:
            return 0.0, {"source": "none: the scenario trains no network (run.train false, NT-033)"}
        if self.options.sec_per_step is not None:
            return float(self.options.sec_per_step), {"source": "given (--sec-per-step)"}
        device = self.options.device or current_device()
        found, refused = latest_sec_per_step(self.store, self.base_config, self.dataset_sha, device)
        if found is None:
            raise SweepError("no measured sec_per_step for THIS setup: no finished run with the same dataset, "
                             f"BATCH_SIZE {self.base_config.BATCH_SIZE}, input layout, LOOKBACK {self.base_config.LOOKBACK}, "
                             f"HORIZON_STEPS {list(self.base_config.HORIZON_STEPS)}, bar size, MAX_SEQUENCE_COUNT, "
                             f"model ({self.base_config.MODEL_NAME}, ATTENTION_MODE, DETERMINISTIC_GRU, "
                             f"PROBE_GRADIENTS), all stored in its config.yaml, and device {device} is in the run index {self.store.index_path} "
                             f"({len(refused)} run(s) of other setups were not used"
                             + (f", for example {refused[0]}" if refused else "")
                             + "); pass --sec-per-step, or run one cell of the setup first")
        return found["sec_per_step"], {"source": "latest run of the same setup", "device": device,
                                       **{k: v for k, v in found.items() if k != "sec_per_step"}}

    # -------------------------------------------------------------- budget
    def _rerun_seconds(self, k: int, steps: Mapping[int, float], sec_per_step: float) -> float:
        """The top-``k`` re-run: every seed on the test fold, and the seeds after the first on each dev fold
        (the first seed's dev cells are the search's own and are not trained again)."""
        o = self.options
        e = self.epochs

        def cell(f):
            return cell_seconds(steps[f], e, sec_per_step, o.overhead_s)

        return k * (sum(cell(f) * (o.rerun_seeds - 1) for f in self.dev_folds)
                    + sum(cell(f) * o.rerun_seeds for f in self.test_folds))

    def estimate_optuna(self, sec_per_step: float, n_new_trials: int, n_total_trials: Optional[int] = None) -> Dict[str, Any]:
        """The GPU budget. ``gpu_hours`` is the upper bound used for the refusal (every trial at the space's
        lowest batch size, EPOCHS epochs, no early stopping); ``expected_gpu_hours`` uses each trial's expected
        batch size. ``trials_that_fit`` is the most new trials that keep the upper bound within ``max_hours``."""
        o = self.options
        e = self.epochs
        k = min(o.top_k, n_total_trials if n_total_trials is not None else max(o.n_trials, 1))

        def per_trial(steps):
            return sum(cell_seconds(steps[f], e, sec_per_step, o.overhead_s) for f in self.dev_folds)

        up_trial, ex_trial = per_trial(self.upper_steps), per_trial(self.expected_steps)
        up_rerun = self._rerun_seconds(k, self.upper_steps, sec_per_step)
        ex_rerun = self._rerun_seconds(k, self.expected_steps, sec_per_step)
        full_rerun = self._rerun_seconds(o.top_k, self.upper_steps, sec_per_step)
        fit = int(max((o.max_hours * 3600.0 - full_rerun) // up_trial, 0)) if up_trial > 0 else 0
        return {"trials_to_run": n_new_trials, "dev_folds": self.dev_folds, "test_folds": self.test_folds,
                "epochs_upper_bound": e, "sec_per_step": sec_per_step, "overhead_s_per_cell": o.overhead_s,
                "steps_per_epoch_upper_per_fold": {str(f): v for f, v in self.upper_steps.items()},
                "steps_per_epoch_expected_per_fold": {str(f): v for f, v in self.expected_steps.items()},
                "search_gpu_hours": n_new_trials * up_trial / 3600.0,
                "rerun": {"top_k": k, "seeds": o.rerun_seeds, "dev_folds_after_first_seed": o.rerun_seeds - 1,
                          "test_folds": self.test_folds},
                "rerun_gpu_hours": up_rerun / 3600.0,
                "gpu_hours": (n_new_trials * up_trial + up_rerun) / 3600.0,
                "expected_gpu_hours": (n_new_trials * ex_trial + ex_rerun) / 3600.0, "max_hours": o.max_hours,
                "trials_that_fit": fit,
                "note": "gpu_hours is an upper bound (the space's lowest batch size, all EPOCHS, no early stopping); "
                        "expected_gpu_hours uses each trial's expected batch size, still without early stopping"}

    # -------------------------------------------------------------- the run
    def run(self) -> SweepResult:
        o = self.options
        self._check_scenario()
        self.space = SearchSpace.from_scenario(self.scenario)
        self._probe()
        if self.record.get("found") and o.parallel > 1:      # the record is used only for N > 1 (allowed N, levels)
            self.record["warnings"] = record_setup_warnings(self.record, self.base_config)
            for w in self.record["warnings"]:
                logger.warning("parallel record %s (measured %s) may not fit this sweep: %s", self.record["path"],
                               self.record.get("measured_utc"), w)
        sps, sps_info = self._sec_per_step()
        if o.mode == QUICK:
            return self._run_quick(sps, sps_info)
        return self._run_optuna(sps, sps_info)

    # ---- quick
    def _run_quick(self, sps: float, sps_info: Dict[str, Any]) -> SweepResult:
        o = self.options
        saved = self._read_summary() if o.resume else {}
        if saved.get("quick_plan"):
            plan = QuickPlan(**saved["quick_plan"])
            points = saved["points"]
        else:
            plan = size_quick(dev_steps=self.dev_steps, epochs_cap=self.epochs, sec_per_step=sps,
                              overhead_s=o.overhead_s, budget_s=o.quick_minutes * 60.0)
            rng = np.random.default_rng(o.sampler_seed)
            points = [self.space.sample(rng) for _ in range(plan.n_trials)]
        budget = {"mode": QUICK, "label": QUICK, "sec_per_step": sps, "sec_per_step_source": sps_info,
                  "overhead_s_per_cell": o.overhead_s, **plan.to_dict(), "estimated_minutes": plan.estimated_s / 60.0}
        self.announce(f"quick sweep {self.sweep_id}: {plan.n_trials} trials x dev folds {plan.folds} x {plan.epochs} "
                      f"epoch(s), sec_per_step {sps:.4g} ({sps_info['source']}): estimated {plan.estimated_s / 60.0:.1f} "
                      f"minutes of the {o.quick_minutes:g} allowed (an estimate; results are labelled quick)")
        if o.dry_run:
            return SweepResult(self.sweep_id, QUICK, QUICK, str(self.directory), budget, [], "dry_run")
        self.dev_folds = list(plan.folds)
        self._extra_overrides = {"EPOCHS": int(plan.epochs)}
        self._init_summary(budget, {"quick_plan": plan.to_dict(), "points": points})
        todo = [(i, p) for i, p in enumerate(points) if i not in self.trial_log or self.trial_log[i].get("state") == "RUNNING"]
        if o.stop_after is not None:
            todo = todo[:o.stop_after]
        stop = self._execute(todo)
        ranking = self._ranking()
        state = "stopped" if stop or len(self.trial_log) < len(points) else "quick_complete"
        res = SweepResult(self.sweep_id, QUICK, QUICK, str(self.directory), budget, self._trials(), state, stop,
                          ranking, None)
        self._write_summary(res, extra={"quick_plan": plan.to_dict(), "points": points,
                                        "leader": next((r for r in ranking if r["rank"] is not None), None),
                                        "note": "quick results: reduced epochs, one seed, dev folds "
                                                f"{plan.folds}; a leader, not a winner"})
        return res

    # ---- optuna
    def _run_optuna(self, sps: float, sps_info: Dict[str, Any]) -> SweepResult:
        try:
            import optuna
        except ImportError as exc:
            raise SweepError("optuna mode needs optuna (pip install -r requirements.txt; "
                             "the pin is in pyproject's `sweep` extra)") from exc
        o = self.options
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_path = self.directory / "study.db"
        storage = f"sqlite:///{study_path.as_posix()}"
        sampler = optuna.samplers.TPESampler(seed=o.sampler_seed)
        states = optuna.trial.TrialState
        # the budget is checked BEFORE a study is created: an existing study is only read, a new one not yet made
        old = optuna.load_study(study_name=self.sweep_id, storage=storage, sampler=sampler) if study_path.is_file() else None
        n_done = sum(t.state.is_finished() for t in old.trials) if old else 0
        n_new = max(o.n_trials - n_done, 0)
        budget = {"mode": OPTUNA, "label": OPTUNA, "n_trials": o.n_trials, "finished_before": n_done,
                  **self.estimate_optuna(sps, n_new), "sec_per_step_source": sps_info}
        self.announce(f"optuna sweep {self.sweep_id}: GPU budget upper bound {budget['gpu_hours']:.2f} h, expected "
                      f"{budget['expected_gpu_hours']:.2f} h (search {budget['search_gpu_hours']:.2f} h for {n_new} "
                      f"trial(s) x {len(self.dev_folds)} dev fold(s), re-run {budget['rerun_gpu_hours']:.2f} h; "
                      f"sec_per_step {sps:.4g}, {sps_info['source']}); limit {o.max_hours:g} h; "
                      f"{budget['trials_that_fit']} trial(s) fit")
        if budget["gpu_hours"] > o.max_hours:
            raise SweepError(f"the GPU budget {budget['gpu_hours']:.2f} h (upper bound; expected "
                             f"{budget['expected_gpu_hours']:.2f} h) is over --max-hours {o.max_hours:g} "
                             f"(one night; a larger budget goes to the owner): nothing was started; "
                             f"{budget['trials_that_fit']} new trial(s) fit within the limit")
        if o.dry_run:
            return SweepResult(self.sweep_id, OPTUNA, OPTUNA, str(self.directory), budget, [], "dry_run")
        self.directory.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(study_name=self.sweep_id, storage=storage, direction="maximize", sampler=sampler,
                                    load_if_exists=True)
        running = [t for t in study.trials if t.state == states.RUNNING]
        self._init_summary(budget, {"study": str(study_path)})
        # an interrupted trial (RUNNING in the study) is finished first, with its own parameters
        todo: List[Tuple[int, Dict[str, Any]]] = [(t.number, dict(t.params)) for t in running]
        n_ask = max(n_new - len(todo), 0)
        if o.stop_after is not None:
            n_ask = max(min(n_ask, o.stop_after - len(todo)), 0)
        stop = None
        pending_ask = n_ask
        while True:
            batch: List[Tuple[int, Dict[str, Any]]] = todo[:o.parallel]
            todo = todo[o.parallel:]
            while len(batch) < o.parallel and pending_ask > 0:
                trial = study.ask()
                params = self.space.suggest(trial)
                batch.append((trial.number, params))
                pending_ask -= 1
            if not batch:
                break
            stop = self._execute(batch, on_scored=lambda n, s: self._tell(study, n, s))
            if stop and (todo or pending_ask > 0):
                break
            if stop:
                logger.warning("%s (nothing left to launch in this call)", stop)
                stop = None
        n_finished = sum(t.state.is_finished() for t in study.trials)
        complete = n_finished >= o.n_trials and not stop
        ranking = self._ranking()
        winner = None
        rerun_table: List[Dict[str, Any]] = []
        ok = [r for r in ranking if r["rank"] is not None]          # failed and ineligible trials are never re-run
        if complete and ok:
            rerun_table, winner, rerun_stop = self._rerun(ok[: o.top_k])
            if rerun_stop:
                complete, stop = False, rerun_stop + " (the search is finished; resume to finish the re-run)"
        res = SweepResult(self.sweep_id, OPTUNA, OPTUNA, str(self.directory), budget, self._trials(),
                          "complete" if complete else "stopped", stop or (None if complete else "stopped after "
                          f"{n_finished} of {o.n_trials} trials (resume to continue)"), ranking, winner)
        self._write_summary(res, extra={"rerun": rerun_table, "study": str(study_path)})
        return res

    @staticmethod
    def _tell(study, number: int, score: TrialScore) -> None:
        """FAIL for a failed trial; INELIGIBLE_VALUE (never its idle or cost-mismatched Sharpe) for a trial that
        fails a search-time guard-rail; else its dev value."""
        import optuna

        if score.value is None:
            study.tell(number, state=optuna.trial.TrialState.FAIL)
        elif not score.eligible:
            study.tell(number, INELIGIBLE_VALUE)
        else:
            study.tell(number, score.value)

    # ---- shared execution
    def _variant(self, number: int) -> str:
        return f"t{number:04d}"

    def _trial_scenario(self, number: int, params: Dict[str, Any], *, folds: Sequence[int], seeds: Sequence[int]) -> Scenario:
        sc = self.scenario
        desc = (f"{self.label} sweep of {sc.name}: trial {number}" +
                (" (quick: reduced epochs, one seed)" if self.label == QUICK else ""))
        config_params = {k: v for k, v in params.items() if not k.startswith(STRATEGY_PREFIX)}
        strategy = dataclasses.replace(sc.strategy, params={
            **sc.strategy.params, **{k[len(STRATEGY_PREFIX):]: v for k, v in params.items()
                                     if k.startswith(STRATEGY_PREFIX)}})
        return dataclasses.replace(sc, name=self.sweep_id, description=desc,
                                   variants={self._variant(number): config_params}, strategy=strategy,
                                   axes={}, search={}, folds=[int(f) for f in folds], seeds=[int(s) for s in seeds],
                                   overrides={**sc.overrides, **self._extra_overrides})

    def _execute(self, batch: List[Tuple[int, Dict[str, Any]]], *, on_scored=None) -> Optional[str]:
        """Run the trials in batches of ``parallel``; returns a stop reason, or None when all ran."""
        o = self.options
        level = self._watch_level()
        for start in range(0, len(batch), o.parallel):
            chunk = batch[start:start + o.parallel]
            reason = self._wait_for_gpu()
            if reason:
                return reason
            seed = int(self.scenario.seeds[0])
            for number, params in chunk:
                self.trial_log[number] = {"number": number, "variant": self._variant(number), "params": params,
                                          "state": "RUNNING", "label": self.label}
            self._save_progress()
            monitor = self.monitor_factory()
            monitor.start()
            try:
                scenarios = [self._trial_scenario(n, p, folds=self.dev_folds, seeds=[seed]) for n, p in chunk]
                if o.parallel > 1:
                    self._launch(scenarios)
                else:
                    for sc in scenarios:
                        self._factory(sc).run()
            finally:
                reading = monitor.stop()
            self.store.sync(self.sweep_id)
            rows = self.store.index.rows(self.sweep_id)
            for number, _params in chunk:
                mine = [r for r in rows if r["variant"] == self._variant(number)]
                score = self._score_trial(mine)
                if on_scored is not None:
                    on_scored(number, score)
                if score.value is None:
                    logger.warning("trial %d FAILED (recorded, not dropped): %s", number, score.reason)
                elif not score.eligible:
                    logger.warning("trial %d is not eligible for the re-run (a search-time guard-rail): %s", number,
                                   score.ineligible)
                self.trial_log[number].update(score=score.to_dict(), value=score.value, reason=score.reason,
                                              eligible=score.eligible, ineligible=score.ineligible,
                                              state="COMPLETE" if score.value is not None else "FAIL")
            self.trial_log[chunk[0][0]]["gpu_reading"] = reading
            self._save_progress()
            why = exceeds_watch_level(reading, level)
            if why:
                return f"stopped launching: GPU {why} (someone else may be on the GPU)"
        return None

    def _watch_level(self) -> Optional[Mapping[str, float]]:
        """The record's GPU level for N own processes, watched after each batch, only when N > 1: the record is
        then the one that allowed N. At N = 1 no record level applies (it was measured on another setup, and a
        single process of today's setup may well use more memory); the GPU-free check before each batch does."""
        if self.options.parallel <= 1:
            return None
        return (self.record.get("utilization") or {}).get(str(self.options.parallel))

    def _unstable(self, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
        """The first stability problem among a trial's finished cells (see :func:`run_health`), else None."""
        if not self.scenario.run.train:
            return None                          # no training, nothing to be unstable
        for r in rows:
            if r.get("status") == "done":
                why = run_health(self.store.root / r["run_dir"],
                                 max_nonfinite_grad_steps=self.options.max_nonfinite_grad_steps)
                if why:
                    return f"unstable training in {r['cell_key']}: {why}"
        return None

    def _score_trial(self, rows: Sequence[Mapping[str, Any]]) -> TrialScore:
        """The trial's dev score through the leaderboard's aggregation; a failed fold or an unstable cell
        (non-finite loss, non-finite-gradient steps above the limit) makes it FAILED with the reason."""
        from neural_trade.experiments.leaderboard import scenario_cost_profile, scenario_guard_rails

        score = dev_net_sharpe(rows, self.dev_folds, store_root=self.store.root,
                               guard_rails=scenario_guard_rails(self.scenario)[0],
                               board_cost=scenario_cost_profile(self.scenario.backtest))
        bad = self._unstable(rows)
        if bad and score.value is not None:
            return dataclasses.replace(score, value=None, reason=bad)
        return score

    def _wait_for_gpu(self) -> Optional[str]:
        """The GPU-free check (no own trial runs now); waits or stops as configured."""
        o = self.options
        if not self.scenario.run.train:
            return None                          # a rule-only trial uses no GPU
        waited = 0.0
        while True:
            status = self.gpu_check()
            self.record_check(status)
            if status.free:
                return None
            if o.when_busy == "stop" or waited >= o.wait_max_s:
                return f"the GPU is busy ({status.detail}); stopped before the batch (resume later)"
            logger.warning("GPU busy %s: waiting %.0f s", status.detail, o.wait_poll_s)
            self.sleep(o.wait_poll_s)
            waited += o.wait_poll_s

    def record_check(self, status: GpuStatus) -> None:
        self.record.setdefault("checks", []).append({"utc": _utc(), "free": status.free, **status.detail})

    def _launch(self, scenarios: List[Scenario]) -> None:
        paths = []
        for sc in scenarios:
            d = sc.to_dict()
            if d["base_config"] and not Path(d["base_config"]).is_absolute() and sc.base_dir is not None:
                d["base_config"] = str((sc.base_dir / d["base_config"]).resolve())
            import yaml

            path = self.directory / "trials" / f"{next(iter(sc.variants))}-{short_hash(d)}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8", newline="\n")
            paths.append(path)
        codes = self.launcher(paths, self.store)
        if any(c not in (0, 1) for c in codes):                  # 1 = a failed cell (recorded); others: a crash
            logger.warning("a trial process exited with %s (cells it did not finish stay incomplete)", codes)

    def _launch_env(self) -> Dict[str, str]:
        """The environment of a trial process: the CALLER's code first on PYTHONPATH (the source directory
        of the imported ``neural_trade``), so the trials run the code this sweep runs, whatever is installed."""
        env = dict(os.environ)
        src = str(code_source_dir())
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def _subprocess_launcher(self, paths: List[Path], store: RunStore) -> List[int]:
        procs = []
        env = self._launch_env()
        for p in paths:
            cmd = [sys.executable, "-m", "neural_trade.cli", "scenario", "run", str(p), "--store", str(store.root),
                   "--index", str(store.index_path), "--claim-cells"]
            log = open(p.with_suffix(".log"), "ab")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        codes = []
        for proc, log in procs:
            codes.append(proc.wait())
            log.close()
        return codes

    # ---- ranking, re-run
    def _trials(self) -> List[Dict[str, Any]]:
        return [self.trial_log[k] for k in sorted(self.trial_log)]

    @staticmethod
    def _eligible(t: Mapping[str, Any]) -> bool:
        return t.get("value") is not None and not t.get("ineligible")

    def _ranking(self) -> List[Dict[str, Any]]:
        """Eligible trials (a value and every search-time guard-rail passed), best dev value first, ranked; then
        the completed trials a search-time guard-rail rules out, then the failed ones, both unranked and flagged.
        Only ranked trials enter the re-run or lead a quick sweep."""
        trials = list(self.trial_log.values())
        ok = sorted((t for t in trials if self._eligible(t)), key=lambda t: -t["value"])
        out_ = sorted((t for t in trials if t.get("value") is not None and not self._eligible(t)),
                      key=lambda t: -t["value"])
        bad = [t for t in trials if t.get("value") is None]
        keys = ("number", "variant", "params", "value", "state", "label")
        return ([{"rank": i + 1, **{k: t.get(k) for k in keys}} for i, t in enumerate(ok)]
                + [{"rank": None, **{k: t.get(k) for k in keys}, "ineligible": t.get("ineligible")} for t in out_]
                + [{"rank": None, **{k: t.get(k) for k in keys}, "reason": t.get("reason")} for t in bad])

    def _rerun(self, top: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
        """The eligible top trials again with ``rerun_seeds`` seeds on every fold, ranked by NT-031's leaderboard
        on the dev folds (the mean of each fold's seed-mean; the test fold is shown, never ranks). The winner is
        the top row that is not disqualified by the scenario's guard-rails and trained stably. Returns (table,
        winner, stop reason): a busy GPU before a batch, or a reading above the watch level after one, stops the
        re-run with no table and no winner (resume finishes it; finished cells are not trained again)."""
        from neural_trade.experiments.leaderboard import (
            RANK_METRIC, build_leaderboard, scenario_cost_profile, scenario_guard_rails)
        from neural_trade.experiments.leaderboard import winner as leaderboard_winner

        o = self.options
        base_seed = int(self.scenario.seeds[0])
        seeds = [base_seed + i for i in range(o.rerun_seeds)]
        folds = self.dev_folds + [f for f in self.test_folds if f not in self.dev_folds]
        level = self._watch_level()
        for start in range(0, len(top), o.parallel):
            busy = self._wait_for_gpu()
            if busy:
                return [], None, f"re-run stopped: {busy}"
            scs = [self._trial_scenario(t["number"], t["params"], folds=folds, seeds=seeds)
                   for t in top[start:start + o.parallel]]
            monitor = self.monitor_factory()
            monitor.start()
            try:
                if o.parallel > 1:
                    self._launch(scs)
                else:
                    for sc in scs:
                        self._factory(sc).run()
            finally:
                reading = monitor.stop()
            why = exceeds_watch_level(reading, level)
            if why and start + o.parallel < len(top):
                return [], None, f"re-run stopped launching: GPU {why} (someone else may be on the GPU; resume later)"
        self.store.sync(self.sweep_id)
        variants = {t["variant"]: t for t in top}
        rows = [r for r in self.store.index.rows(self.sweep_id) if r["variant"] in variants]
        unstable = {v: self._unstable([r for r in rows if r["variant"] == v]) for v in variants}
        board = build_leaderboard(rows, guard_rails=scenario_guard_rails(self.scenario)[0], store_root=self.store.root,
                                  board_cost=scenario_cost_profile(self.scenario.backtest),
                                  spec_folds=self.scenario.folds)
        table = []
        for lb in board:
            t = variants[lb.configuration]
            guards = [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in lb.guard_rails]
            table.append({"rank": lb.rank, "number": t["number"], "variant": lb.configuration, "params": t["params"],
                          "dev_seed_mean_net_sharpe": lb.dev.values.get(RANK_METRIC),
                          "dev_per_fold": {str(k): v for k, v in lb.dev.fold_values[RANK_METRIC].items()},
                          "dev_fold_spread": lb.dev.spread.get(RANK_METRIC), "n_dev_cells": lb.dev.n_rows,
                          "n_seeds": lb.dev.n_seeds, "mean_trades": lb.dev.values.get("n_trades"),
                          "single_seed_value": t["value"], "status": lb.status, "disqualified": lb.disqualified,
                          "guard_rails": guards, "unstable": unstable[lb.configuration],
                          "test_sharpe_net": lb.test.values.get(RANK_METRIC),
                          "test_max_drawdown": lb.test.values.get("max_drawdown"),
                          "test_n_trades": lb.test.values.get("n_trades"), "n_test_cells": lb.test.n_rows,
                          "test_note": "test fold: shown, never used to rank"})
        top_row = leaderboard_winner([lb for lb in board if not unstable[lb.configuration]])
        winner = next((r for r in table if top_row is not None and r["variant"] == top_row.configuration), None)
        return table, winner, None

    # ---- the summary file
    def _read_summary(self) -> Dict[str, Any]:
        try:
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _init_summary(self, budget: Dict[str, Any], extra: Dict[str, Any]) -> None:
        saved = self._read_summary()
        for t in saved.get("trials", []):
            self.trial_log[int(t["number"])] = t
        launches = list(saved.get("launches", []))
        launches.append({"utc": _utc(), "budget": budget, "options": dataclasses.asdict(self.options),
                         "code": code_info(),
                         "parallel_record": {k: v for k, v in self.record.items() if k != "checks"}})
        self._launches = launches
        self._extra = extra
        # the budget is on disk BEFORE the first trial starts
        _write_json(self.summary_path, {"schema_version": SWEEP_SCHEMA_VERSION, "sweep_id": self.sweep_id,
                                        "scenario": self.scenario.name, "mode": self.options.mode, "label": self.label,
                                        "space": self.space.to_dict(), "launches": launches,
                                        "trials": self._trials(), "state": "started", **extra})

    def _save_progress(self) -> None:
        doc = self._read_summary()
        doc.update(trials=self._trials(), gpu_checks=self.record.get("checks", [])[-20:], state="running")
        _write_json(self.summary_path, doc)

    def _write_summary(self, res: SweepResult, extra: Dict[str, Any]) -> None:
        doc = self._read_summary()
        doc.update(trials=res.trials, state=res.state, stop_reason=res.stop_reason, ranking=res.ranking,
                   winner=res.winner, ranking_column="dev-fold mean net Sharpe after costs (test never ranks)",
                   finished_utc=_utc(), gpu_checks=self.record.get("checks", [])[-20:], **extra)
        _write_json(self.summary_path, doc)
        if res.winner is not None:
            _write_json(self.directory / "winner.json", {"sweep_id": self.sweep_id, **res.winner,
                                                         "config_overrides": res.winner["params"]})


__all__ = ["DEFAULT_SEARCH", "INELIGIBLE_VALUE", "SEARCH_TIME_RAILS", "SETUP_FIELDS", "parse_dmon", "gpu_status_from_dmon", "run_health", "code_info", "code_source_dir",
           "record_setup_warnings", "setup_mismatch", "current_device", "GpuStatus", "MODES", "QuickPlan", "SearchParam", "SearchSpace", "Sweep",
           "SweepError", "SweepOptions", "SweepResult", "TrialScore", "cell_seconds", "dev_net_sharpe",
           "exceeds_watch_level", "latest_sec_per_step", "load_parallel_record", "nvidia_smi_gpu_check",
           "size_quick", "steps_per_epoch", "held_out_columns"]

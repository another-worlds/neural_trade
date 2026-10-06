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
:func:`dev_net_sharpe`: the one place the leaderboard (NT-031) replaces.

**Quick mode** (no optuna needed): from a measured ``sec_per_step`` it sizes the number of trials,
the epochs and the dev folds so that the whole sweep is estimated at most ``--quick-minutes`` (5), prints the
estimate before it starts and labels every result ``quick`` (reduced epochs, one seed, no winner).

**Optuna mode**: a TPE study in sqlite (``<store>/sweeps/<id>/study.db``), resumable (``--resume``: a
finished trial is never repeated, an interrupted one is finished). Before it starts it prints and
records its GPU budget: trials x dev folds x steps x sec_per_step plus the top-5 x 3-seed re-run, and
refuses above ``--max-hours`` (12: one night; a larger budget goes to the owner). After the search the
top 5 are re-run with 3 seeds (dev folds, and the test fold for display) and the winner is the best
seed mean on the dev folds.

**Parallel** (``--parallel N``): trials launch in batches of N processes (``scenario run
--claim-cells``, so no two processes train a cell). N above 1 needs NT-035's record
(``runs/experiments/gpu_measurements_v1/parallel_n.json``: ``allowed_n``). The GPU-free check of
RUNBOOK "GPU rules" runs before each batch (no own trial is running then); after each batch the
GPU's memory and utilisation are compared with the level the record gives for N own processes, and a
higher reading (someone else is on the GPU) stops the launching.

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
REFUSED_FIELDS = {"RESAMPLE_MINUTES": "a sweep stays at 1-minute bars until NT-040 is done"}
# Used when the scenario has no ``search:`` block: the optimiser, the batch and the two heads' loss weights.
DEFAULT_SEARCH: Dict[str, Optional[Dict[str, Any]]] = {
    "LR": {"low": 1e-4, "high": 1e-2, "log": True},
    "BATCH_SIZE": {"low": 64, "high": 1024, "log": True},
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

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "per_fold": {str(k): v for k, v in sorted(self.per_fold.items())},
                "n_cells": self.n_cells, "spread": self.spread, "mean_trades": self.mean_trades,
                "reason": self.reason}


def dev_net_sharpe(rows: Sequence[Mapping[str, Any]], dev_folds: Sequence[int]) -> TrialScore:
    """Dev-fold net Sharpe after costs of one configuration's index rows. The only ranking number of a
    sweep; test-fold rows are never read here. Swap point for NT-031's leaderboard function."""
    per_fold: Dict[int, List[float]] = {}
    trades: List[float] = []
    problems = []
    for fold in dev_folds:
        mine = [r for r in rows if r.get("role") == "dev" and r.get("fold") == fold]
        done = [r for r in mine if r.get("status") == "done"]
        if not done:
            err = next((r.get("error") for r in mine if r.get("status") == "failed"), None)
            problems.append(f"fold {fold}: " + (f"failed ({err})" if err else "no finished cell"))
            continue
        vals = [r.get("sharpe_net") for r in done]
        if any(v is None or not math.isfinite(float(v)) for v in vals):
            problems.append(f"fold {fold}: non-finite net Sharpe")
            continue
        per_fold[int(fold)] = float(np.mean(vals))
        trades += [float(r["n_trades"]) for r in done if r.get("n_trades") is not None]
    n_cells = sum(1 for r in rows if r.get("role") == "dev" and r.get("status") == "done")
    if problems:
        return TrialScore(None, per_fold, n_cells, reason="; ".join(problems))
    vals = list(per_fold.values())
    return TrialScore(float(np.mean(vals)), per_fold, n_cells,
                      float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                      float(np.mean(trades)) if trades else None)


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


def nvidia_smi_gpu_check(samples: int = 10) -> GpuStatus:
    """RUNBOOK "GPU rules": busy when fb > 2000 MB or the median sm > 30% over 10 one-second samples."""
    if _cpu_only():
        return GpuStatus(True, {"note": "CUDA_VISIBLE_DEVICES=-1: a CPU run, no GPU check"})
    try:
        out = subprocess.run(["nvidia-smi", "dmon", "-s", "um", "-c", str(samples)],
                             capture_output=True, text=True, timeout=samples + 30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return GpuStatus(False, {"error": f"nvidia-smi failed: {exc}"})
    sm, fb = [], []
    for line in out.splitlines():
        parts = line.split()
        if parts and not line.startswith("#") and len(parts) >= 5:
            try:
                sm.append(float(parts[1]))
                fb.append(float(parts[3]))
            except ValueError:
                continue
    if not sm:
        return GpuStatus(False, {"error": "nvidia-smi dmon gave no samples"})
    med_sm, med_fb = float(np.median(sm)), float(np.median(fb))
    return GpuStatus(med_fb <= GPU_BUSY_FB_MB and med_sm <= GPU_BUSY_SM_PCT,
                     {"median_sm_pct": med_sm, "median_fb_mb": med_fb, "samples": len(sm)})


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


def load_parallel_record(path) -> Dict[str, Any]:
    """NT-035's result: ``allowed_n`` and the per-N ``utilization``; no file means N = 1."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return {"allowed_n": 1, "utilization": {}, "path": str(p) if p else None, "found": False}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {"allowed_n": int(doc.get("allowed_n", 1)), "utilization": doc.get("utilization") or {},
            "path": str(p), "found": True}


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


def latest_sec_per_step(store: RunStore, setup: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """sec_per_step of the newest finished run of the same setup in the index (None: no such run)."""
    best = None
    for r in store.index.rows(status="done"):
        if r.get("sec_per_step") is None:
            continue
        if (r.get("bar_minutes") == setup.get("bar_minutes") and r.get("lookback") == setup.get("LOOKBACK")
                and r.get("horizon_steps") == json.dumps(setup.get("HORIZON_STEPS"))):
            key = r.get("created_utc") or ""
            if best is None or key > best[0]:
                best = (key, {"sec_per_step": float(r["sec_per_step"]), "run_id": r["run_id"], "run_dir": r["run_dir"]})
    return best[1] if best else None


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
    min_trades: float = 1.0                     # a winner trades at least this often on average (a 0-trade row is not one)
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
        batch = int(planned[0].config.BATCH_SIZE)
        if self.space is not None and "BATCH_SIZE" in self.space.names:      # the estimate takes the most steps
            batch = min(batch, int(next(p for p in self.space.params if p.name == "BATCH_SIZE").low or batch))
        self.dev_steps = {pc.cell.fold: steps_per_epoch(pc.fold["blocks"]["train"]["n"], batch)
                          for pc in planned if pc.role == "dev"}
        self.setup = setup_of(planned[0].config)
        self.epochs = int(planned[0].config.EPOCHS)

    def _factory(self, scenario: Scenario) -> Runner:
        if self._runner_factory is not None:
            return self._runner_factory(scenario)
        r = Runner(scenario, self.store, trainer=self.trainer, check_components=self.check_components, claim_cells=True)
        r._layouts = self._layouts
        return r

    def _sec_per_step(self) -> Tuple[float, Dict[str, Any]]:
        if self.options.sec_per_step is not None:
            return float(self.options.sec_per_step), {"source": "given (--sec-per-step)"}
        found = latest_sec_per_step(self.store, self.setup)
        if found is None:
            raise SweepError("no measured sec_per_step: no finished run of this setup "
                             f"({self.setup.get('bar_minutes')}-minute bars, LOOKBACK {self.setup.get('LOOKBACK')}, "
                             f"HORIZON_STEPS {self.setup.get('HORIZON_STEPS')}) in the run index {self.store.index_path}; "
                             "pass --sec-per-step, or run one cell of the setup first")
        return found["sec_per_step"], {"source": "latest run of the same setup", **{k: v for k, v in found.items()
                                                                                     if k != "sec_per_step"}}

    # -------------------------------------------------------------- budget
    def estimate_optuna(self, sec_per_step: float, n_new_trials: int) -> Dict[str, Any]:
        o = self.options
        epochs = self.epochs
        per_trial = sum(cell_seconds(s, epochs, sec_per_step, o.overhead_s) for s in self.dev_steps.values())
        n_top = min(o.top_k, max(o.n_trials, 1))
        n_folds_rerun = len(self.dev_folds) + len(self.test_folds)
        mean_steps = float(np.mean(list(self.dev_steps.values())))
        rerun = n_top * o.rerun_seeds * n_folds_rerun * cell_seconds(mean_steps, epochs, sec_per_step, o.overhead_s)
        search_s = n_new_trials * per_trial
        total_h = (search_s + rerun) / 3600.0
        return {"trials_to_run": n_new_trials, "dev_folds": self.dev_folds, "epochs_upper_bound": epochs,
                "steps_per_epoch_per_dev_fold": {str(k): v for k, v in self.dev_steps.items()},
                "sec_per_step": sec_per_step, "overhead_s_per_cell": o.overhead_s,
                "search_gpu_hours": search_s / 3600.0, "rerun": {"top_k": n_top, "seeds": o.rerun_seeds,
                                                                   "folds": self.dev_folds + self.test_folds},
                "rerun_gpu_hours": rerun / 3600.0, "gpu_hours": total_h, "max_hours": o.max_hours,
                "note": "an upper bound: early stopping may end a trial before EPOCHS; the re-run counts every seed "
                        "on every fold although the first seed's dev cells already exist"}

    # -------------------------------------------------------------- the run
    def run(self) -> SweepResult:
        o = self.options
        self._check_scenario()
        self.space = SearchSpace.from_scenario(self.scenario)
        self._probe()
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
                                        "leader": ranking[0] if ranking else None,
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
        if not o.dry_run:
            self.directory.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(study_name=self.sweep_id, storage=f"sqlite:///{study_path.as_posix()}",
                                    direction="maximize", sampler=optuna.samplers.TPESampler(seed=o.sampler_seed),
                                    load_if_exists=True) if not o.dry_run else None
        states = optuna.trial.TrialState
        n_done = sum(t.state.is_finished() for t in study.trials) if study else 0
        running = [t for t in study.trials if t.state == states.RUNNING] if study else []
        n_new = max(o.n_trials - n_done, 0)
        budget = {"mode": OPTUNA, "label": OPTUNA, "n_trials": o.n_trials, "finished_before": n_done,
                  **self.estimate_optuna(sps, n_new), "sec_per_step_source": sps_info}
        self.announce(f"optuna sweep {self.sweep_id}: GPU budget {budget['gpu_hours']:.2f} h "
                      f"(search {budget['search_gpu_hours']:.2f} h for {n_new} trial(s) x {len(self.dev_folds)} dev "
                      f"fold(s), re-run {budget['rerun_gpu_hours']:.2f} h; sec_per_step {sps:.4g}, "
                      f"{sps_info['source']}); limit {o.max_hours:g} h")
        if budget["gpu_hours"] > o.max_hours:
            fit = o.max_hours * 3600.0 - budget["rerun_gpu_hours"] * 3600.0
            raise SweepError(f"the GPU budget {budget['gpu_hours']:.2f} h is over --max-hours {o.max_hours:g} "
                             f"(one night; a larger budget goes to the owner): nothing was started; about "
                             f"{max(int(fit // max(budget['search_gpu_hours'] * 3600.0 / max(n_new, 1), 1)), 0)} trials fit")
        if o.dry_run:
            return SweepResult(self.sweep_id, OPTUNA, OPTUNA, str(self.directory), budget, [], "dry_run")
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
        if complete and ranking:
            rerun_table, winner = self._rerun(ranking[: o.top_k])
        res = SweepResult(self.sweep_id, OPTUNA, OPTUNA, str(self.directory), budget, self._trials(),
                          "complete" if complete else "stopped", stop or (None if complete else "stopped after "
                          f"{n_finished} of {o.n_trials} trials (resume to continue)"), ranking, winner)
        self._write_summary(res, extra={"rerun": rerun_table, "study": str(study_path)})
        return res

    @staticmethod
    def _tell(study, number: int, score: TrialScore) -> None:
        import optuna

        if score.value is None:
            study.tell(number, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(number, score.value)

    # ---- shared execution
    def _variant(self, number: int) -> str:
        return f"t{number:04d}"

    def _trial_scenario(self, number: int, params: Dict[str, Any], *, folds: Sequence[int], seeds: Sequence[int]) -> Scenario:
        sc = self.scenario
        desc = (f"{self.label} sweep of {sc.name}: trial {number}" +
                (" (quick: reduced epochs, one seed)" if self.label == QUICK else ""))
        return dataclasses.replace(sc, name=self.sweep_id, description=desc, variants={self._variant(number): params},
                                   axes={}, search={}, folds=[int(f) for f in folds], seeds=[int(s) for s in seeds],
                                   overrides={**sc.overrides, **self._extra_overrides})

    def _execute(self, batch: List[Tuple[int, Dict[str, Any]]], *, on_scored=None) -> Optional[str]:
        """Run the trials in batches of ``parallel``; returns a stop reason, or None when all ran."""
        o = self.options
        level = (self.record.get("utilization") or {}).get(str(o.parallel))
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
                score = dev_net_sharpe(mine, self.dev_folds)
                if on_scored is not None:
                    on_scored(number, score)
                if score.value is None:
                    logger.warning("trial %d FAILED (recorded, not dropped): %s", number, score.reason)
                self.trial_log[number].update(score=score.to_dict(), value=score.value, reason=score.reason,
                                              state="COMPLETE" if score.value is not None else "FAIL")
            self.trial_log[chunk[0][0]]["gpu_reading"] = reading
            self._save_progress()
            why = exceeds_watch_level(reading, level)
            if why:
                return f"stopped launching: GPU {why} (someone else may be on the GPU)"
        return None

    def _wait_for_gpu(self) -> Optional[str]:
        """The GPU-free check (no own trial runs now); waits or stops as configured."""
        o = self.options
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

    def _subprocess_launcher(self, paths: List[Path], store: RunStore) -> List[int]:
        procs = []
        for p in paths:
            cmd = [sys.executable, "-m", "neural_trade.cli", "scenario", "run", str(p), "--store", str(store.root),
                   "--index", str(store.index_path), "--claim-cells"]
            log = open(p.with_suffix(".log"), "ab")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        codes = []
        for proc, log in procs:
            codes.append(proc.wait())
            log.close()
        return codes

    # ---- ranking, re-run
    def _trials(self) -> List[Dict[str, Any]]:
        return [self.trial_log[k] for k in sorted(self.trial_log)]

    def _ranking(self) -> List[Dict[str, Any]]:
        """Completed trials, best dev value first; failed trials follow, flagged."""
        ok = sorted((t for t in self.trial_log.values() if t.get("value") is not None), key=lambda t: -t["value"])
        bad = [t for t in self.trial_log.values() if t.get("value") is None]
        return [{"rank": i + 1, **{k: t.get(k) for k in ("number", "variant", "params", "value", "state", "label")}}
                for i, t in enumerate(ok)] + [{"rank": None, **{k: t.get(k) for k in ("number", "variant", "params",
                                                                                      "value", "state", "reason")}}
                                              for t in bad]

    def _rerun(self, top: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Top trials again with ``rerun_seeds`` seeds on every fold; ranked by the dev-fold seed mean (the
        test fold is shown, never ranks); the winner needs an average of ``min_trades`` trades or more."""
        o = self.options
        base_seed = int(self.scenario.seeds[0])
        seeds = [base_seed + i for i in range(o.rerun_seeds)]
        folds = self.dev_folds + [f for f in self.test_folds if f not in self.dev_folds]
        table = []
        for start in range(0, len(top), o.parallel):
            if self._wait_for_gpu():
                return table, None
            scs = [self._trial_scenario(t["number"], t["params"], folds=folds, seeds=seeds)
                   for t in top[start:start + o.parallel]]
            if o.parallel > 1:
                self._launch(scs)
            else:
                for sc in scs:
                    self._factory(sc).run()
        self.store.sync(self.sweep_id)
        rows = self.store.index.rows(self.sweep_id)
        for t in top:
            mine = [r for r in rows if r["variant"] == t["variant"]]
            score = dev_net_sharpe(mine, self.dev_folds)
            n_seeds = len({r["seed"] for r in mine if r["role"] == "dev" and r["status"] == "done"})
            table.append({"number": t["number"], "variant": t["variant"], "params": t["params"],
                          "dev_seed_mean_net_sharpe": score.value, "dev_per_fold": score.to_dict()["per_fold"],
                          "dev_fold_spread": score.spread, "n_dev_cells": score.n_cells, "n_seeds": n_seeds,
                          "mean_trades": score.mean_trades, "single_seed_value": t["value"],
                          "reason": score.reason, **held_out_columns(mine),
                          "test_note": "test fold: shown, never used to rank"})
        eligible = [r for r in table if r["dev_seed_mean_net_sharpe"] is not None
                    and (r["mean_trades"] or 0.0) >= o.min_trades]
        eligible.sort(key=lambda r: -r["dev_seed_mean_net_sharpe"])
        winner = eligible[0] if eligible else None
        table.sort(key=lambda r: (r["dev_seed_mean_net_sharpe"] is None, -(r["dev_seed_mean_net_sharpe"] or 0.0)))
        return table, winner

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


__all__ = ["DEFAULT_SEARCH", "GpuStatus", "MODES", "QuickPlan", "SearchParam", "SearchSpace", "Sweep",
           "SweepError", "SweepOptions", "SweepResult", "TrialScore", "cell_seconds", "dev_net_sharpe",
           "exceeds_watch_level", "latest_sec_per_step", "load_parallel_record", "nvidia_smi_gpu_check",
           "size_quick", "steps_per_epoch", "held_out_columns"]

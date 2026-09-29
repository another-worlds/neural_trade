"""Screen mode (NT-088): mass, sub-30-second CPU/GPU trials over a grid/sample of Config fields.

Where ``scenario run`` (runner.py) trains one Config per (fold, seed) into its own run directory
with the full evaluation machinery (baselines, backtest, random null, stored predictions, a serving
bundle), a screen has none of that: it trains hundreds of tiny configurations (a 6-hour training
block is the reference size, ``docs/research/2026-09-29-screen-plan.md``) to find broken or unstable
math and hyperparameter regions, not to rank quality (level 1 of the plan; level 2 is the existing
``scenario run`` on survivors). A screen never writes a per-trial run directory by default: one JSON
line per trial in ``<store>/screens/<name>/results.jsonl`` is the whole record.

    neural-trade screen configs/screens/example_6h.yaml [--shard 0/3] [--store runs]

A screen spec (schema_version: 1)::

    name: example_6h
    base_config: ../default.yaml        # optional, like a scenario's base_config
    overrides: {MAX_SEQUENCE_COUNT: 4500, N_FOLDS: 2, VAL_FRACTION: 0.1, CAL_FRACTION: 0.1}
    grid:
      axes: {LR: [0.0001, 0.001], BATCH_SIZE: [64, 256]}
    sample:                             # optional, in addition to (not crossed with) the grid
      n: 20
      method: random                    # or lhs
      seed: 0
      space: {LAMBDA_HD: {low: 0.0, high: 1.0}, LR: {low: 0.0001, high: 0.01, log: true}}
    slices: ["2024-01-15T00:00:00", "2024-06-01T00:00:00"]   # DATA_END values (K volatility regimes)
    seeds: [0, 1]
    run: {calibrate: false, epochs: 3}
    rules: {finite: true, max_nonfinite_grad_steps: 0, max_clipped_share: 0.5,
            min_train_loss_drop: 0.01, max_term_share: 0.9}

Every ``(grid point or sample point) x slice x seed`` is one trial. Trials are trained through a
light internal path (not ``train_and_evaluate``, which also fits the post-hoc calibration pipeline
and, for a run context, writes a serving bundle): data loaded and prepared, AND its sliding windows
built (``DataProcessor.build_windows``, the part that loops every bar), ONCE per data key per
process (the cache is keyed by :func:`neural_trade.experiments.dataset.data_key`, which already
covers every Config field that can change the prepared bars or the windows, DATA_END included), a
real ``CustomTrainModel`` trained for the spec's ``run.epochs``, no checkpoints, no baselines, no
backtest, no random null, no stored predictions, no serving bundle. Only the fold split, target
scaling and window normalisation (``DataProcessor.prepare_datasets_from_windows``, cheap: no per-bar
loop) run per trial, since fields outside the data key (N_FOLDS, VAL_FRACTION, CAL_FRACTION,
WINDOW_NORMALIZER, FOLD_INDEX, BATCH_SIZE, ...) may still vary trial to trial. ``run.calibrate``
switches the pre-training loss-weight calibration pass (``training.lambda_calibration.calibrate_loss_weights``)
on or off, same as a scenario's ``run.calibrate``.

Health numbers (all finite, ``nonfinite_grad_steps``, the max and mean of the per-logged-step
``grad_global_norm``, the share of logged steps at or above ``GRAD_CLIP_NORM``, the training loss's
first-to-last-epoch drop, the final validation loss, and each loss term's share of the final total)
come from the same per-epoch aggregates ``CustomTrainModel`` already exposes in
``history.history`` (``training/custom_model.py``'s ``_EpochTrainLogs``), plus a small per-batch
sampler (:class:`_GradNormSampler`) for the max / share, which epoch aggregates alone cannot give.
Pre-registered ``rules:`` in the spec turn the health numbers into a pass/fail with reasons.

Resumable: a trial's key (a hash of its exact Config values,
:func:`neural_trade.experiments.scenario.config_hash`) already present in its results file skips it.
``--shard i/N`` runs trial index ``j`` only when ``j % N == i``: the shards are disjoint and their
union is every trial, so ``N`` processes (NT-035's 3-process ceiling) split a screen without locking
any cell. Each shard writes its OWN file, ``results.shard-{i}-of-{N}.jsonl`` (0-indexed), so
concurrent processes never append to the same file; :func:`merge_results` reads every shard file
(and a plain ``results.jsonl`` from an unsharded run, if present) back together for resumability
checks and for reporting total progress across shards.

A non-finite health number (an extreme LAMBDA_* blowing up training) is written to the JSONL row as
``null`` (never as a raw NaN/Infinity, which ``json.dumps(allow_nan=False)`` would refuse to
serialise and so leave the trial unrecorded and unresumable): see :func:`_sanitize_nonfinite`. Such a
trial is always recorded ``passed: false``, regardless of the spec's ``rules:``, with a reason naming
the non-finite field(s).
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.experiments.dataset import data_key
from neural_trade.experiments.scenario import config_hash, short_hash

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
RESULTS_FILE = "results.jsonl"
HORIZONS = ("h0", "h1", "h2")

TOP_KEYS = ("schema_version", "name", "description", "base_config", "overrides", "grid", "sample",
           "slices", "seeds", "run", "rules")
GRID_KEYS = ("axes",)
SAMPLE_KEYS = ("n", "method", "seed", "space")
SAMPLE_SPACE_KEYS = ("low", "high", "log")
SAMPLE_METHODS = ("random", "lhs")
RUN_KEYS = ("calibrate", "epochs", "reuse_graph")
RULE_KEYS = ("finite", "max_nonfinite_grad_steps", "max_clipped_share", "min_train_loss_drop", "max_term_share",
            "clip_skip_epochs")
DEFAULT_CLIP_SKIP_EPOCHS = 1

# Continuous Config fields (NT-092 phase 2): read from a live tf.Variable (or a Keras optimizer
# hyperparameter, which becomes one after the optimizer's first use) at every training step, so
# changing one between trials never needs a new trace. Trials in one screen that differ ONLY in these
# fields (plus SEED, DATA_END and EPOCHS, which never affect the graph - see _STRUCTURAL_IGNORE) share
# one structural group and reuse the group's compiled train/test step (_TrialGroup). Every other
# Config field is structural: changing it retraces (a new group).
#
#   field                  | read from, at run time
#   -----------------------|----------------------------------------------------------------------
#   LR                     | Keras Optimizer hyper `learning_rate` (a backing tf.Variable created on
#                          | first use, training/optim.py:build_optimizers; set via `optimizer.
#                          | learning_rate = ...`, which assigns the variable, docs/RUNBOOK.md).
#   ADAM_BETA1/2           | same Keras hyper mechanism, `optimizer.beta_1` / `beta_2`.
#   INDICATOR_LR_MULT      | the indicator optimizer's own `learning_rate` hyper (LR * this,
#                          | training/optim.py:build_indicator_optimizer); reset directly, not by
#                          | rebuilding the optimizer.
#   GRAD_CLIP_NORM         | `CustomTrainModel.grad_clip_norm` (training/custom_model.py), a
#                          | tf.Variable; train_step's clip-or-not branch is a graph-safe `tf.cond`
#                          | on this Variable's value, not a Python `if`, so 0 <-> nonzero is also
#                          | continuous.
#   LAMBDA_SHORT/POINT/LONG/EXTENDED_TREND/DIR/VAR/VOL/CRPS/SOFT_ECE/T_PERP/CASIMIR/HD/IFE/
#   VAC_OVERFLOW/PNL       | `model.lambda_<key>` properties (training/lambdas.py), tf.Variables.
#   LAMBDA_TREND_OUTER/DIR_OUTER/NLL_OUTER/COHERENCE
#                          | the same lambdas.py properties, added by NT-092 (outer multipliers that
#                          | were plain Python floats before).
#
# NOT continuous, despite being tunable weights or hyperparameters (each is read as a plain Python
# value baked into the graph at trace time, in a file outside this item's area, so changing it forces
# a retrace / a new structural group):
#   LAMBDA_VAC, LAMBDA_DIR_ALIGN, LAMBDA_INTER  - read via `model.config.LAMBDA_*` inside
#       losses/functions.py (tf.constant(...) at trace time); LAMBDA_DIR_ALIGN_OUTER additionally
#       gates a Python `if` that skips the op entirely when it is 0 (functions.py:675).
#   INDICATOR_GRAD_MULT    - read at layer-build time, models/layers/learnable_indicators.py.
#   ABLATE_LAMBDAS         - applied once via `lambdas.ablate(...)` outside train_step; a different
#       list of ablated names is a different loss-term on/off structure, kept structural for
#       simplicity even where every named field happens to be variable-backed.
CONTINUOUS_FIELDS = (
    "LR", "ADAM_BETA1", "ADAM_BETA2", "GRAD_CLIP_NORM", "INDICATOR_LR_MULT",
    "LAMBDA_SHORT", "LAMBDA_POINT", "LAMBDA_LONG", "LAMBDA_EXTENDED_TREND", "LAMBDA_DIR", "LAMBDA_VAR",
    "LAMBDA_VOL", "LAMBDA_CRPS", "LAMBDA_SOFT_ECE", "LAMBDA_T_PERP", "LAMBDA_CASIMIR", "LAMBDA_HD",
    "LAMBDA_IFE", "LAMBDA_VAC_OVERFLOW", "LAMBDA_PNL",
    "LAMBDA_TREND_OUTER", "LAMBDA_DIR_OUTER", "LAMBDA_NLL_OUTER", "LAMBDA_COHERENCE",
)
# Fields that never affect the traced graph at all (not because they are variables, but because they
# are consumed entirely outside train_step/test_step): the seed only picks initial weights and RNG
# streams (reset per trial, see _reset_group_trial), DATA_END only picks which rows are loaded
# (handled by the data/window cache, keyed separately by data_key()), and EPOCHS only controls how
# many times `fit()` is called.
_STRUCTURAL_IGNORE = frozenset(CONTINUOUS_FIELDS) | {"SEED", "DATA_END", "EPOCHS"}


def structural_key(cfg: Config) -> str:
    """The hash of every Config field EXCEPT the continuous ones and SEED/DATA_END/EPOCHS (see
    CONTINUOUS_FIELDS and _STRUCTURAL_IGNORE above). Two trials with the same structural_key build the
    exact same graph (same shapes, same Python-level branches) and so can share one _TrialGroup."""
    return short_hash({k: v for k, v in cfg.to_dict().items() if k not in _STRUCTURAL_IGNORE})
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")

# The final-epoch loss-term keys CustomTrainModel logs (training/custom_model.py:_update_diagnostics),
# read from history.history to compute each term's share of the final total loss.
LOSS_TERM_KEYS = ("point_loss", "trend_loss", "dir_loss", "nll_loss", "crps_loss", "soft_ece_loss",
                  "reg_loss", "inter_reg", "vol_loss", "t_perp_loss", "casimir_loss", "vac_loss",
                  "hd_loss", "ife_loss", "vac_overflow_loss", "pnl_val")  # pnl_val: pnl_utility (NT-087), logged lambda-weighted


class ScreenError(InvalidConfigurationError):
    """A screen spec is malformed or names an invalid configuration (a ValueError)."""


# ------------------------------------------------------------------ the spec
@dataclass
class RunOptions:
    calibrate: bool = True
    epochs: Optional[int] = None
    # NT-092 phase 2: group trials by structural_key() and reuse one traced graph (_TrialGroup) per
    # group, resetting weights/optimizer state/continuous hyperparameters between trials. `false` runs
    # every trial through the original fresh-model-per-trial path (kept as the reference: a spec can
    # ask for it to reproduce phase-1 numbers exactly, or for a direct fresh-vs-reused comparison).
    reuse_graph: bool = True


@dataclass
class ScreenSpec:
    name: str
    schema_version: int = SCHEMA_VERSION
    description: str = ""
    base_config: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    axes: Dict[str, List[Any]] = field(default_factory=dict)
    sample: Optional[Dict[str, Any]] = None
    slices: List[Optional[str]] = field(default_factory=lambda: [None])
    seeds: List[int] = field(default_factory=lambda: [0])
    run: RunOptions = field(default_factory=RunOptions)
    rules: Dict[str, Any] = field(default_factory=dict)
    source: Optional[Path] = None
    base_dir: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path) -> "ScreenSpec":
        import yaml

        path = Path(path)
        if not path.is_file():
            raise ScreenError(f"screen spec {path} does not exist")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ScreenError(f"screen spec {path}: not valid YAML: {exc}") from exc
        return cls.from_dict(data, source=path, base_dir=path.parent)

    @classmethod
    def from_dict(cls, data: Any, *, source=None, base_dir=None) -> "ScreenSpec":
        where = str(source) if source is not None else "screen spec"

        def bad(msg: str):
            raise ScreenError(f"{where}: {msg}")

        if not isinstance(data, Mapping):
            bad("expected a mapping of screen keys")
        _refuse_unknown(data, TOP_KEYS, "", bad)
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            bad(f"schema_version must be {SCHEMA_VERSION}, got {version!r}")
        name = data.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            bad(f"name must be 1-48 characters of letters, digits, '.', '_' or '-' (a directory name), got {name!r}")
        description = data.get("description") or ""
        base = data.get("base_config")
        if base is not None and not isinstance(base, str):
            bad(f"base_config must be a path string, got {base!r}")
        overrides = _mapping(data.get("overrides"), "overrides", bad)
        grid = _mapping(data.get("grid"), "grid", bad)
        _refuse_unknown(grid, GRID_KEYS, "grid.", bad)
        axes = _mapping(grid.get("axes"), "grid.axes", bad)
        for axis, values in axes.items():
            if not isinstance(values, list) or not values:
                bad(f"grid.axes.{axis} must be a non-empty list of values")
        sample = data.get("sample")
        if sample is not None:
            sample = _mapping(sample, "sample", bad)
            _refuse_unknown(sample, SAMPLE_KEYS, "sample.", bad)
            if not isinstance(sample.get("n"), int) or sample.get("n") <= 0:
                bad(f"sample.n must be a positive integer, got {sample.get('n')!r}")
            method = sample.get("method", "random")
            if method not in SAMPLE_METHODS:
                bad(f"sample.method must be one of {SAMPLE_METHODS}, got {method!r}")
            space = _mapping(sample.get("space"), "sample.space", bad)
            if not space:
                bad("sample.space must be a non-empty mapping of field -> {low, high[, log]}")
            for fname, s in space.items():
                s = _mapping(s, f"sample.space.{fname}", bad)
                _refuse_unknown(s, SAMPLE_SPACE_KEYS, f"sample.space.{fname}.", bad)
                if "low" not in s or "high" not in s:
                    bad(f"sample.space.{fname} needs explicit 'low' and 'high' bounds")
                if float(s["low"]) >= float(s["high"]):
                    bad(f"sample.space.{fname}: low must be < high, got {s}")
            sample = {**sample, "space": space, "method": method, "seed": int(sample.get("seed", 0))}
        slices_raw = data.get("slices")
        if slices_raw is None:
            slices: List[Optional[str]] = [None]
        else:
            if not isinstance(slices_raw, list) or not slices_raw:
                bad(f"slices must be a non-empty list of timestamps (or null), got {slices_raw!r}")
            slices = [None if v is None else str(v) for v in slices_raw]
        seeds = data.get("seeds", [0])
        if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int)
                                                            for s in seeds):
            bad(f"seeds must be a non-empty list of integers, got {seeds!r}")
        raw_run = _mapping(data.get("run"), "run", bad)
        _refuse_unknown(raw_run, RUN_KEYS, "run.", bad)
        if "calibrate" in raw_run and not isinstance(raw_run["calibrate"], bool):
            bad(f"run.calibrate must be true or false, got {raw_run['calibrate']!r}")
        if "epochs" in raw_run and raw_run["epochs"] is not None and (
                isinstance(raw_run["epochs"], bool) or not isinstance(raw_run["epochs"], int) or raw_run["epochs"] < 1):
            bad(f"run.epochs must be a positive integer or null, got {raw_run['epochs']!r}")
        if "reuse_graph" in raw_run and not isinstance(raw_run["reuse_graph"], bool):
            bad(f"run.reuse_graph must be true or false, got {raw_run['reuse_graph']!r}")
        rules = _mapping(data.get("rules"), "rules", bad)
        _refuse_unknown(rules, RULE_KEYS, "rules.", bad)
        if "clip_skip_epochs" in rules and (isinstance(rules["clip_skip_epochs"], bool)
                                            or not isinstance(rules["clip_skip_epochs"], int)
                                            or rules["clip_skip_epochs"] < 0):
            bad(f"rules.clip_skip_epochs must be a non-negative integer, got {rules['clip_skip_epochs']!r}")
        return cls(name=name, schema_version=version, description=description, base_config=base,
                   overrides=overrides, axes=axes, sample=sample, slices=slices, seeds=[int(s) for s in seeds],
                   run=RunOptions(**raw_run), rules=rules,
                   source=Path(source) if source is not None else None,
                   base_dir=Path(base_dir) if base_dir is not None else None)

    # -------------------------------------------------------------- views
    def where(self) -> str:
        return str(self.source) if self.source is not None else f"screen {self.name!r}"

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "name": self.name, "description": self.description,
                "base_config": self.base_config, "overrides": self.overrides, "grid": {"axes": self.axes},
                "sample": self.sample, "slices": list(self.slices), "seeds": list(self.seeds),
                "run": {"calibrate": self.run.calibrate, "epochs": self.run.epochs}, "rules": dict(self.rules)}

    @property
    def spec_hash(self) -> str:
        return short_hash(self.to_dict())

    @property
    def clip_skip_epochs(self) -> int:
        """The configured ``rules.clip_skip_epochs``, or the default (1) when unset - BEFORE the
        ``EPOCHS`` guard in :meth:`base`. Use :meth:`effective_clip_skip_epochs` for the value
        actually passed to a trial."""
        return int(self.rules.get("clip_skip_epochs", DEFAULT_CLIP_SKIP_EPOCHS))

    def effective_clip_skip_epochs(self, epochs: int) -> int:
        """``clip_skip_epochs``, clamped to ``epochs - 1`` (never negative) when the spec did not
        set it explicitly - so a quick, default-everything 1-epoch smoke screen still scores its one
        epoch (clip_skip_epochs 0) instead of tripping :meth:`base`'s min_epochs guard, which fires
        only when the SPEC explicitly asks for a ``clip_skip_epochs`` that leaves no epoch to score."""
        if "clip_skip_epochs" in self.rules:
            return self.clip_skip_epochs
        return max(0, min(self.clip_skip_epochs, int(epochs) - 1))

    def base(self) -> Config:
        """The base Config with the spec-wide overrides and ``run.epochs`` applied (validated)."""
        if self.base_config is None:
            cfg = Config()
        else:
            path = Path(self.base_config)
            if not path.is_absolute() and self.base_dir is not None:
                path = self.base_dir / path
            if not path.is_file():
                raise ScreenError(f"{self.where()}: base_config {self.base_config!r} not found (looked at {path})")
            cfg = _step(lambda: Config.from_yaml(path), f"base_config {self.base_config}", self.where())
        overrides = dict(self.overrides)
        if self.run.epochs is not None:
            overrides["EPOCHS"] = int(self.run.epochs)
        self._check_fields(overrides, "overrides")
        self._check_fields({a: None for a in self.axes}, "grid.axes")
        if self.sample:
            self._check_fields({f: None for f in self.sample["space"]}, "sample.space")
            self._check_sample_bounds()
        base_cfg = _step(lambda: cfg.override(**overrides), "overrides", self.where())
        # min_epochs guard (NT-092 acceptance 5): clip_skip_epochs excludes an entire epoch's steps
        # from clipped_share; with clip_skip_epochs >= EPOCHS every trial's clipped_share would be
        # computed over ZERO steps (None, never a real pass/fail signal). Only an EXPLICIT
        # rules.clip_skip_epochs is refused this way - the unset default is clamped down instead
        # (effective_clip_skip_epochs), so an ordinary 1-epoch smoke screen still scores its epoch.
        if "clip_skip_epochs" in self.rules and self.clip_skip_epochs >= int(base_cfg.EPOCHS):
            raise ScreenError(f"{self.where()}: rules.clip_skip_epochs={self.clip_skip_epochs} must be less than "
                              f"EPOCHS={int(base_cfg.EPOCHS)} (min_epochs guard: otherwise no epoch is ever scored)")
        return base_cfg

    def _check_sample_bounds(self) -> None:
        """(P3, D-020-adjacent robustness) ``sample.space`` bounds must be sane BEFORE any drawing
        happens: ``log: true`` needs ``low > 0`` (``math.log`` of a non-positive number is a cryptic
        ``ValueError`` deep inside ``_draw``/``_lhs``, not a clear spec error), and ``low``/``high``
        must fall inside the Config field's own declared range (:meth:`Config.field_specs`), so a
        sample never draws a value ``Config.override`` would refuse mid-run."""
        specs = Config.field_specs()
        for fname, s in self.sample["space"].items():
            spec = specs.get(fname)
            if spec is None:
                continue  # unknown field: already refused by _check_fields above
            low, high = float(s["low"]), float(s["high"])
            if s.get("log") and low <= 0:
                raise ScreenError(f"{self.where()}: sample.space.{fname}: log sampling needs low > 0, "
                                  f"got low={low!r}")
            for bound_name, bound in (("low", low), ("high", high)):
                msg = spec.check(bound)
                if msg is not None:
                    raise ScreenError(f"{self.where()}: sample.space.{fname}.{bound_name}={bound!r} is "
                                      f"outside {fname}'s valid range {spec.range_text()}")

    def _check_fields(self, overrides: Mapping[str, Any], where: str) -> None:
        names = set(Config.field_names())
        unknown = [k for k in overrides if k not in names]
        if unknown:
            import difflib

            hints = {k: difflib.get_close_matches(k, names, n=3) for k in unknown}
            raise ScreenError(f"{self.where()}: {where}: unknown Config field(s): " + "; ".join(
                k + (f" (did you mean {', '.join(v)}?)" if v else "") for k, v in hints.items()))


def _refuse_unknown(data: Mapping[str, Any], known, prefix: str, bad) -> None:
    unknown = [k for k in data if k not in known]
    if unknown:
        bad(f"unknown key(s) {', '.join(prefix + str(k) for k in unknown)}; known: "
            f"{', '.join(prefix + k for k in known)}")


def _mapping(value: Any, where: str, bad) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        bad(f"{where} must be a mapping, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _step(fn, what: str, where: str):
    try:
        return fn()
    except InvalidConfigurationError as exc:
        raise ScreenError(f"{where}: {what}: {exc}") from exc


# ------------------------------------------------------------------ trial generation
def _grid_points(axes: Mapping[str, List[Any]]) -> List[Dict[str, Any]]:
    names = list(axes)
    if not names:
        return [{}]
    return [dict(zip(names, pt)) for pt in itertools.product(*(axes[n] for n in names))]


def _draw(rng: np.random.Generator, low: float, high: float, log: bool, n: int) -> np.ndarray:
    if log:
        return np.exp(rng.uniform(math.log(low), math.log(high), size=n))
    return rng.uniform(low, high, size=n)


def _lhs(rng: np.random.Generator, low: float, high: float, log: bool, n: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n + 1)
    u = edges[:-1] + rng.uniform(0.0, 1.0, size=n) * (edges[1:] - edges[:-1])
    rng.shuffle(u)
    lo, hi = (math.log(low), math.log(high)) if log else (low, high)
    v = lo + u * (hi - lo)
    return np.exp(v) if log else v


def _sample_points(sample: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not sample:
        return []
    n = int(sample["n"])
    method = sample.get("method", "random")
    rng = np.random.default_rng(int(sample.get("seed", 0)))
    space = sample["space"]
    names = list(space)
    draw = _lhs if method == "lhs" else _draw
    cols = {name: draw(rng, float(s["low"]), float(s["high"]), bool(s.get("log", False)), n)
           for name, s in space.items()}
    return [{name: float(cols[name][i]) for name in names} for i in range(n)]


def _snap(name: str, value: Any, specs: Mapping[str, Any]) -> Any:
    """Round a sampled float to an int when the Config field is integer-typed."""
    spec = specs.get(name)
    if spec is not None and spec.type == "int" and isinstance(value, float):
        return int(round(value))
    return value


@dataclass(frozen=True)
class Trial:
    index: int
    source: str                 # "grid" or "sample"
    params: Dict[str, Any]      # the grid point or sample point values
    data_end: Optional[str]
    seed: int
    config: Config = field(compare=False, repr=False)

    @property
    def key(self) -> str:
        return config_hash(self.config)


def build_trials(spec: ScreenSpec) -> List[Trial]:
    """Every ``(grid point or sample point) x slice x seed``, in spec order; validates the whole
    spec (raises :class:`ScreenError` naming the offending key), including, for every trial, its
    effective ``DATA_END`` against the protected dev/test span (D-020, :func:`_preflight_data_end`).
    That preflight check reads each distinct data file's raw bars once (to resolve the implicit
    "use the newest data" case and the true protected-span boundary) but builds no windows and
    trains nothing."""
    base = spec.base()
    field_specs = Config.field_specs()
    points: List[Tuple[str, Dict[str, Any]]] = [("grid", p) for p in _grid_points(spec.axes)]
    points += [("sample", p) for p in _sample_points(spec.sample)]
    if not points:
        points = [("grid", {})]
    trials: List[Trial] = []
    idx = 0
    for source, point in points:
        overrides = {k: _snap(k, v, field_specs) for k, v in point.items()}
        for data_end in spec.slices:
            for seed in spec.seeds:
                point_overrides = dict(overrides)
                point_overrides["SEED"] = int(seed)
                if data_end is not None:
                    point_overrides["DATA_END"] = data_end
                cfg = _step(lambda p=point_overrides: base.copy(**p),
                           f"trial {idx} ({source} {point}, slice {data_end!r}, seed {seed})", spec.where())
                trials.append(Trial(idx, source, dict(point), data_end, int(seed), cfg))
                idx += 1
    _preflight_data_end(trials, spec)
    return trials


def _preflight_data_end(trials: Sequence["Trial"], spec: ScreenSpec) -> None:
    """D-020: validate every trial's ``DATA_END`` - explicit, or the implicit "use the newest data"
    case (``DATA_END`` unset, which today's ``apply_data_end`` never checks at all, since that is the
    ordinary, correct behaviour everywhere else) - against the protected dev/test span BEFORE any
    trial trains. A spec with any violating slice fails immediately here, with no training for ANY
    trial, not even the ones that would have been fine. Reads each distinct data file's raw,
    preprocessed bars once (cheap: no windows built), cached by (CSV_PATH, DATA_LOADER,
    PREPROCESSORS) across trials that share a file.

    Also closes the other half of the D-020 gap: a screen spec could otherwise lower
    ``DATA_END_PROTECTED_DAYS`` below the Config default (64 days) to sneak a ``DATA_END`` close to
    the true tail past the guard. Forbidden here ONLY against a file that actually spans at least the
    default protected span - the bundled 30-day CSV and the synthetic test fixtures are shorter than
    that to begin with, so the default would refuse every ``DATA_END`` outright; RUNBOOK "Screen mode"
    and ``configs/screens/example_6h.yaml`` document lowering it for that reason, and a real campaign
    against the long 2017-2025 file (which does span more than 64 days) keeps the floor."""
    from neural_trade.data.processor import DataProcessor, apply_data_end

    default_days = float(Config.field_specs()["DATA_END_PROTECTED_DAYS"].default)
    file_cache: Dict[Tuple[str, str, Tuple[str, ...]], Tuple[Any, float]] = {}
    for trial in trials:
        cfg = trial.config
        fkey = (str(cfg.CSV_PATH), str(cfg.DATA_LOADER), tuple(cfg.PREPROCESSORS))
        cached = file_cache.get(fkey)
        if cached is None:
            dp = DataProcessor(cfg)
            df = dp.preprocess(dp.load_raw())
            ts = df["timestamp"]
            span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
            cached = file_cache[fkey] = (df, span_days)
        df, span_days = cached
        if span_days >= default_days and float(cfg.DATA_END_PROTECTED_DAYS) < default_days:
            raise ScreenError(
                f"{spec.where()}: trial {trial.index}: DATA_END_PROTECTED_DAYS="
                f"{float(cfg.DATA_END_PROTECTED_DAYS)!r} is below the Config default ({default_days!r}) "
                f"against a file spanning {span_days:.1f} days; only a file shorter than the default "
                "protected span may lower it (D-020)")
        probe_cfg = cfg if cfg.DATA_END is not None else cfg.copy(DATA_END=str(df["timestamp"].max()))
        try:
            apply_data_end(df, probe_cfg)
        except ValueError as exc:
            raise ScreenError(f"{spec.where()}: trial {trial.index} (slice {trial.data_end!r}): {exc}") from exc


def shard_of(index: int, shard: Optional[Tuple[int, int]]) -> bool:
    """True when trial ``index`` belongs to ``shard`` (``(i, N)``): ``index % N == i``. ``None``
    (no ``--shard``): every trial belongs to the single implicit shard."""
    if shard is None:
        return True
    i, n = shard
    return index % n == i


def parse_shard(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if text is None:
        return None
    i_s, sep, n_s = text.partition("/")
    if not sep:
        raise ScreenError(f"--shard must be i/N, got {text!r}")
    try:
        i, n = int(i_s), int(n_s)
    except ValueError as exc:
        raise ScreenError(f"--shard must be i/N with integers, got {text!r}") from exc
    if n <= 0 or not (0 <= i < n):
        raise ScreenError(f"--shard must be i/N with 0 <= i < N, got {text!r}")
    return i, n


# ------------------------------------------------------------------ the light training path
class _GradNormSampler(tf.keras.callbacks.Callback):
    """Per-logged-step ``grad_global_norm`` (max, mean, share at/above ``clip_norm``), from the
    exact epoch-Mean accumulator ``CustomTrainModel`` already keeps (``_step_means['grad_global_norm']``,
    updated only on logged steps per ``Config.TRAIN_METRICS_EVERY``, training/custom_model.py). A
    ``tf.keras.metrics.Mean`` accumulates an exact running sum/count, reset at each epoch start, so
    the delta between two observations equals the value of the single update between them.

    ``clipped_share`` is an APPROXIMATION, documented here because ``training/custom_model.py``'s
    ``train_step`` (outside this item's files) actually clips gradients in TWO SEPARATE groups by
    ``tf.clip_by_global_norm`` - the network weights and the indicator logit variables - each against
    its own group norm, and it is the pre-clip norm of the COMBINED (both groups') gradients that is
    sampled here (``grad_global_norm``, computed once in ``train_step`` before the split). So
    ``clipped_share`` is the share of logged steps where the COMBINED norm was at or above
    ``GRAD_CLIP_NORM``, not "one particular group actually got clipped this step": the combined norm
    can exceed the clip norm while neither group's own (smaller) norm does, and either group's own
    norm can occasionally exceed it while the combined norm (dominated by the other, larger group)
    does not, without this sampler seeing it separately. True per-group tracking would need a second
    epoch accumulator inside ``CustomTrainModel``, outside NT-088's files."""

    def __init__(self, clip_norm: float, *, clip_skip_epochs: int = 0):
        super().__init__()
        self.clip_norm = float(clip_norm or 0.0)
        # clip_skip_epochs (NT-092, default 1 via the spec's rules): the first epoch's logged steps
        # never count toward clipped_share (or the norm max/mean) - only toward n_steps. At LR 1e-4
        # every 2-epoch trial failed max_clipped_share because the initial, pre-any-update gradient
        # norm is routinely far above the clip on epoch 0's first few steps, before training has done
        # anything; that transient is not the "is this config unstable" signal the rule is for.
        self.clip_skip_epochs = int(clip_skip_epochs or 0)
        self._epoch = -1
        self._prev_total = 0.0
        self._prev_count = 0.0
        self.n_steps = 0
        self.n_valid_steps = 0  # logged steps whose sampled norm was finite (see clipped_share)
        self.max_norm: Optional[float] = None
        self._sum_norm = 0.0
        self.n_clipped = 0

    @property
    def mean_norm(self) -> Optional[float]:
        return self._sum_norm / self.n_valid_steps if self.n_valid_steps else None

    @property
    def clipped_share(self) -> Optional[float]:
        """The share of steps with a finite sampled norm that were at/above ``clip_norm``, or
        ``None`` (JSON ``null``) when every logged step's norm was non-finite: with no valid step
        there is no clip/no-clip signal at all, and ``0.0`` would misread as "nothing was clipped"."""
        return self.n_clipped / self.n_valid_steps if self.n_valid_steps else None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._prev_total = 0.0
        self._prev_count = 0.0

    def on_train_batch_end(self, batch, logs=None):
        m = getattr(self.model, "_step_means", {}).get("grad_global_norm")
        if m is None:
            return
        try:
            total, count = float(m.total.numpy()), float(m.count.numpy())
        except Exception:  # telemetry only: never break training
            return
        if count > self._prev_count:
            value = (total - self._prev_total) / (count - self._prev_count)
            self._prev_total, self._prev_count = total, count
            self.n_steps += 1
            if self._epoch < self.clip_skip_epochs:  # excluded from clipped_share (see __init__)
                return
            if not math.isfinite(value):  # no valid clip/no-clip signal from this step
                return
            self.n_valid_steps += 1
            self._sum_norm += value
            self.max_norm = value if self.max_norm is None else max(self.max_norm, value)
            if self.clip_norm > 0 and value >= self.clip_norm:
                self.n_clipped += 1


def _last_finite(series: Optional[Sequence[Any]]) -> Optional[float]:
    for v in reversed(list(series or [])):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            return v
    return None


def _term_multiplier(key: str, cfg: Config, model: Any = None) -> float:
    """The extra factor needed to turn the value ``CustomTrainModel`` logs for ``key``
    (``training/custom_model.py``'s ``_update_diagnostics`` scalars, read from ``history.history``)
    into its true contribution to the total loss (``losses/functions.py:custom_loss``'s ``total``),
    so that ``loss_term_shares`` (below) reflects what a trial's LAMBDA_* values actually did, not
    their raw, un-weighted magnitude.

    Traced against ``custom_loss`` line by line (round 2 re-check, NT-088 QA finding 5): the
    per-term LAMBDA_* is applied INSIDE ``custom_loss`` before the value is returned in
    ``LossComponents`` for most of the physics/regularisation terms, so most of them are already
    fully weighted when logged. Only direction, NLL, CRPS and soft-ECE are logged as the RAW,
    un-weighted per-example loss (``LossComponents`` carries ``dir_loss_h*``/``nll_h*_val``/
    ``crps_h*_val``/``soft_ece_h*_val`` straight from their computation, never ``total_dir_loss``
    etc.). Four groups:

    - already fully weighted per horizon inside ``custom_loss`` before being logged, multiplier
      1.0: ``point_loss`` (LAMBDA_SHORT/POINT/LONG applied per horizon), ``t_perp_loss``
      (``c.t_perp_total`` IS ``lambda_t_perp * (...)``, not the raw per-horizon sum -- round 1's
      table put this term in the RAW group by mistake, which double-counted LAMBDA_T_PERP: a
      LAMBDA_T_PERP of 100 reported a share of ~78 instead of the true ~0.78), ``casimir_loss``
      (``casimir_val = lambda_casimir * casimir_interference_loss(...)``), ``hd_loss``
      (``hd_val = lambda_hd * hyper_decoherence_coupling_loss(...)``), ``ife_loss``
      (``ife_val = lambda_ife * information_flow_entropy_loss(...)``), ``vac_overflow_loss``
      (``vac_overflow_val = lambda_vac_overflow * vacuum_overflow_t_perp_loss(...)``), ``vac_loss``
      (LAMBDA_VAC is a threshold inside ``vacuum_bandwidth_loss``'s relu clamp, not a multiplicative
      weight, and ``total`` adds ``vac_val`` with an implicit weight of 1);
    - logged already-weighted but missing ONE further OUTER multiplier ``total`` applies on top:
      ``trend_loss`` needs LAMBDA_TREND_OUTER; ``inter_reg`` (pre-weighted by LAMBDA_INTER, a
      Config-only field never touched by calibration or ablation) and ``vol_loss`` (pre-weighted by
      ``model.lambda_vol``, which calibration/ablation DO update, so it is already correct at the
      point it is logged) each need the further fixed 0.1 outer weight ``custom_loss`` gives them;
    - logged RAW / un-weighted: ``dir_loss``, ``nll_loss``, ``crps_loss``, ``soft_ece_loss``.
      ``total`` only ever sees them multiplied by their LAMBDA_* (and, for direction and NLL, an
      outer multiplier too). ``reg_loss`` is logged but ``total`` uses ``0 * reg_loss`` (never
      affects it): multiplier 0.0.

    When ``model`` (the trained ``CustomTrainModel``) is given, the multiplier for the four raw
    terms and ``trend_loss`` is read from the model's OWN lambda attributes -- ``model.lambda_dir``,
    ``model.lambda_var``, ``model.lambda_crps``, ``model.lambda_soft_ece`` and (since NT-092)
    ``model.lambda_dir_outer``/``lambda_nll_outer``/``lambda_trend_outer`` -- all live ``tf.Variable``
    properties (``training/lambdas.py``) -- rather than the pre-run ``cfg.LAMBDA_*``. This matters whenever ``run.calibrate`` (default True) rescaled them
    after training, or ``ABLATE_LAMBDAS`` zeroed one: reading ``cfg`` alone made shares sum to
    1.232 in a calibrated repro instead of ~1 (QA finding 5). ``model=None`` (e.g. a bare
    ``history``/``cfg`` reconstruction in a test) falls back to ``cfg.LAMBDA_*``, unchanged from
    round 1.

    This is an approximation of ``total`` itself: two further terms ``custom_loss`` can add
    (``dir_align_loss`` via LAMBDA_DIR_ALIGN_OUTER, ``coherence_penalty`` via
    LAMBDA_COHERENCE_OUTER) are not logged as separate history keys at all, so they are not in
    ``loss_term_shares``; both default to an outer weight of 0 (LAMBDA_DIR_ALIGN_OUTER) or a small
    one, so shares sum close to but not always exactly 1.0."""
    def _live(attr: str, cfg_name: str, default: float = 1.0) -> float:
        if model is not None:
            return float(getattr(model, attr, getattr(cfg, cfg_name, default)))
        return float(getattr(cfg, cfg_name, default))

    if key == "trend_loss":
        return _live("lambda_trend_outer", "LAMBDA_TREND_OUTER")
    if key == "dir_loss":
        return _live("lambda_dir_outer", "LAMBDA_DIR_OUTER") * _live("lambda_dir", "LAMBDA_DIR")
    if key == "nll_loss":
        return _live("lambda_nll_outer", "LAMBDA_NLL_OUTER") * _live("lambda_var", "LAMBDA_VAR")
    if key == "crps_loss":
        return _live("lambda_crps", "LAMBDA_CRPS")
    if key == "soft_ece_loss":
        return _live("lambda_soft_ece", "LAMBDA_SOFT_ECE")
    if key == "reg_loss":
        return 0.0
    if key in ("inter_reg", "vol_loss"):
        return 0.1
    # already fully weighted inside custom_loss before being logged: point_loss, t_perp_loss,
    # casimir_loss, vac_loss, hd_loss, ife_loss, vac_overflow_loss.
    return 1.0


def _health_from_history(history, sampler: _GradNormSampler, cfg: Config, model: Any = None) -> Dict[str, Any]:
    """The health numbers of one trial's ``history`` (``CustomTrainModel.fit``'s return). ``model``
    is the trained ``CustomTrainModel``, when available: passed through to :func:`_term_multiplier`
    so ``loss_term_shares`` reflects the lambdas actually applied during training -- post-calibration
    (``run.calibrate``) and post-``ABLATE_LAMBDAS`` -- rather than the pre-run ``cfg.LAMBDA_*``
    (NT-088 QA finding 5). ``model=None`` (tests that only have a fake ``history`` and a ``cfg``)
    falls back to ``cfg.LAMBDA_*``, unchanged from round 1."""
    hist: Dict[str, List[Any]] = dict(getattr(history, "history", None) or {})
    all_values = [v for series in hist.values() for v in series]
    finite = all(math.isfinite(float(v)) for v in all_values) if all_values else False
    loss_series = [float(v) for v in hist.get("loss", [])]
    nonfinite_total = sum(int(round(float(v))) for v in hist.get("nonfinite_grad_steps", [])
                          if math.isfinite(float(v)))
    train_loss_drop = None
    if len(loss_series) >= 1 and math.isfinite(loss_series[0]) and loss_series[0] != 0 \
            and math.isfinite(loss_series[-1]):
        train_loss_drop = (loss_series[0] - loss_series[-1]) / abs(loss_series[0])
    final_total = _last_finite(hist.get("loss"))
    term_shares = {}
    if final_total not in (None, 0.0):
        for key in LOSS_TERM_KEYS:
            v = _last_finite(hist.get(key))
            if v is not None:
                term_shares[key] = (v * _term_multiplier(key, cfg, model)) / final_total
    return {
        "finite": bool(finite),
        "nonfinite_grad_steps": nonfinite_total,
        "grad_global_norm_max": sampler.max_norm,
        "grad_global_norm_mean": sampler.mean_norm,
        "clipped_share": sampler.clipped_share,
        "n_logged_steps": sampler.n_steps,
        "train_loss_drop": train_loss_drop,
        "final_train_loss": final_total,
        "final_val_loss": _last_finite(hist.get("val_loss")),
        "loss_term_shares": term_shares,
        "max_term_share": max(term_shares.values()) if term_shares else None,
        "epochs_run": len(loss_series),
    }


def _direction_auc(model, val_block: Mapping[str, Any], cfg: Config, target_scaler) -> Dict[str, Any]:
    """Per-horizon direction AUC on the validation block (labelled with its noise level, D-012:
    ``n`` non-deadband rows and ``n_eff = n // horizon bars``, the block being one contiguous
    stretch)."""
    from neural_trade.core.postprocess import heads_to_predictions
    from neural_trade.metrics.statistics import auc_score

    X = np.asarray(val_block["X"], dtype="float32")
    n = int(len(val_block["y_raw"]))
    ds = tf.data.Dataset.from_tensor_slices(X).batch(int(cfg.BATCH_SIZE))
    heads = model.predict(ds, verbose=0)
    preds = heads_to_predictions(heads, n, float(target_scaler.scale_[0]), float(target_scaler.mean_[0]), cfg)
    y_raw = np.asarray(val_block["y_raw"], dtype=float)
    last_close = np.asarray(val_block["last_close"], dtype=float).reshape(-1)
    deadband = float(cfg.DIR_DEADBAND_BPS) / 10000.0
    horizons = list(cfg.HORIZON_STEPS)
    out: Dict[str, Any] = {}
    for i, h in enumerate(HORIZONS):
        if i >= y_raw.shape[1]:
            continue
        ret = y_raw[:, i] / np.where(np.abs(last_close) > 1e-9, last_close, np.nan)
        mask = np.isfinite(ret) & (np.abs(ret) > deadband)
        n_masked = int(mask.sum())
        n_eff = max(1, n_masked // int(horizons[i])) if i < len(horizons) and n_masked else None
        row: Dict[str, Any] = {"auc": None, "n": n_masked, "n_eff": n_eff}
        y_true = (ret[mask] > 0).astype(int)
        if n_masked >= 2 and len(np.unique(y_true)) >= 2:
            prob = np.asarray(preds["direction_prob"][h], dtype=float)[mask]
            auc = auc_score(y_true, prob)
            row["auc"] = auc if math.isfinite(auc) else None
        out[h] = row
    return out


def _load_cached(cfg: Config, cache: Dict[str, Any]) -> Tuple[Any, Any]:
    """``(df, close)`` of ``cfg``'s prepared data (:meth:`DataProcessor.load_and_prepare_data`,
    already sliced to ``DATA_END``), computed once per :func:`data_key` and cached in ``cache``
    across trials that share it (tested directly: tests/test_screen.py)."""
    from neural_trade.data.processor import DataProcessor

    key = data_key(cfg)
    cached = cache.get(key)
    if cached is None:
        cached = DataProcessor(cfg).load_and_prepare_data()
        cache[key] = cached
    return cached


def _windowed_cached(cfg: Config, cache: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """``(X_seq, y_seq, last_close_seq, extended_trends)`` of ``cfg``'s sliding windows
    (:meth:`DataProcessor.build_windows`, trimmed to ``MAX_SEQUENCE_COUNT``), computed once per
    :func:`data_key` and cached in ``cache`` across trials that share it. Windowing loops every bar
    (``data/windowing.py:make_sequences_with_extended_trends``) and used to re-run on every trial even
    though every field it reads is already part of ``data_key`` (LOOKBACK, HORIZON_STEPS,
    EXTENDED_TREND_PERIODS, WINDOW_STEP, MAX_SEQUENCE_COUNT): a 10-18 second cost per trial on the
    long file that this cache removes for every trial after the first sharing a data key. Kept under
    a distinct cache key (``"win:" + data_key``) so it never collides with :func:`_load_cached`'s
    ``(df, close)`` entry for the same key."""
    from neural_trade.data.processor import DataProcessor

    key = "win:" + data_key(cfg)
    cached = cache.get(key)
    if cached is None:
        df, close = _load_cached(cfg, cache)
        cached = DataProcessor(cfg).build_windows(close)
        cache[key] = cached
    return cached


class _EpochTimer(tf.keras.callbacks.Callback):
    """Wall-clock seconds per epoch, for the phase-2 tracing decision (RUNBOOK "Screen mode"):
    ``trace_time ~= epoch_s[0] - median(epoch_s[1:])`` estimates the one-off tf.function tracing cost
    folded into the first epoch's wall time."""

    def __init__(self):
        super().__init__()
        self.epoch_s: List[float] = []
        self._t0: Optional[float] = None

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_s.append(time.perf_counter() - (self._t0 if self._t0 is not None else time.perf_counter()))


@dataclass
class _PreparedTrial:
    """One trial's data, ready for ``fit``: everything :func:`_run_trial_light` and
    :meth:`_TrialGroup.run_one` both need, built by the shared :func:`_prepare_trial_data` (NT-092:
    factored out so the reuse path does not duplicate the fresh path's data handling)."""
    train_ds: Any
    val_ds: Any
    val_block: Mapping[str, Any]
    target_scaler: Any
    y_train: Any
    n_train: int
    load_s: float
    prep_s: float


def _prepare_trial_data(cfg: Config, cache: Dict[str, Any]) -> _PreparedTrial:
    """Load (cached by :func:`data_key`), window (cached) and split/scale/normalise (per trial,
    cheap) ``cfg``'s data, and build its train/val ``tf.data.Dataset``s. Shared by the fresh
    (:func:`_run_trial_light`) and reused-graph (:class:`_TrialGroup`) paths."""
    from neural_trade.data.processor import DataProcessor
    from neural_trade.data.datasets import create_datasets

    t0 = time.perf_counter()
    _load_cached(cfg, cache)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_seq, y_seq, last_close_seq, extended_trends = _windowed_cached(cfg, cache)
    dp = DataProcessor(cfg)
    (X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
     X_test_seq, y_test_scaled, last_close_test, extended_trends_test,
     y_train, y_test, target_scaler) = dp.prepare_datasets_from_windows(X_seq, y_seq, last_close_seq,
                                                                        extended_trends)
    val_block = dp.val_block
    train_ds, val_ds = create_datasets(cfg, X_train_seq, y_train_scaled, last_close_train,
                                       extended_trends_train, val_block["X"], val_block["y_scaled"],
                                       val_block["last_close"], val_block["extended_trends"])
    t_prep = time.perf_counter() - t1
    return _PreparedTrial(train_ds, val_ds, val_block, target_scaler, y_train, int(X_train_seq.shape[0]),
                          t_load, t_prep)


def _run_trial_light(cfg: Config, cache: Dict[str, Any], *, calibrate: bool,
                     clip_skip_epochs: int = DEFAULT_CLIP_SKIP_EPOCHS
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train ``cfg`` on the light, FRESH-MODEL-PER-TRIAL path and return
    ``(health, direction_auc, timings)`` (the phase-1 path, NT-088; still used when
    ``run.reuse_graph: false`` and, for tests, whenever ``run_trial``/``run_screen`` is given an
    explicit ``trainer``). Data (load + preprocess + the DATA_END slice) AND its windows
    (:func:`_windowed_cached`) are cached in ``cache`` by :func:`data_key`, so many trials that share a
    data key pay both costs once per process; only the fold split / scaling / normalisation (cheap, no
    per-bar loop) and the model itself are built fresh per trial. ``timings`` has ``load_s`` (the
    cached-or-not data load), ``prep_s`` (windowing, cached-or-not, plus the per-trial
    split/scale/normalise), ``build_s`` (``Models.build`` + optimizers + compile only), ``train_s``
    (``fit``), ``score_s`` (health + direction AUC) and ``epoch_s`` (a list of per-epoch wall-clock
    seconds, :class:`_EpochTimer`)."""
    from neural_trade.registries.models import Models
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.training.lambda_calibration import calibrate_loss_weights
    from neural_trade.training.lambdas import ablate
    from neural_trade.training.optim import build_optimizers
    from neural_trade.training.reset import reset_stateful_rngs
    from neural_trade.utils.seeding import seed_everything

    seed_everything(int(cfg.SEED))
    prepared = _prepare_trial_data(cfg, cache)
    train_ds, val_ds, val_block, target_scaler, y_train = (prepared.train_ds, prepared.val_ds,
                                                           prepared.val_block, prepared.target_scaler,
                                                           prepared.y_train)

    t2 = time.perf_counter()
    base_model = Models.build(cfg.MODEL_NAME, cfg)
    optimizer_pair = build_optimizers(cfg)
    pred_scale = np.std(y_train) if np.std(y_train) > 0 else 1.0
    pred_mean = np.mean(y_train)
    custom_model = CustomTrainModel(base_model=base_model, pred_scale=pred_scale, pred_mean=pred_mean,
                                    lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
                                    lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND,
                                    lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND, lambda_dir=cfg.LAMBDA_DIR,
                                    config=cfg, indicator_optimizer=optimizer_pair.indicator,
                                    inputs=base_model.inputs, outputs=base_model.outputs)
    # Explicit, controlled dropout-generator reset (NT-092): see training/reset.py's module doc.
    # Called here too (not just from the reuse path) so a fresh trial and a later reused trial of the
    # exact same seed draw the identical dropout stream (acceptance 2), rather than relying on
    # whatever construction-order seed Keras assigned this fresh model's Dropout layers internally.
    reset_stateful_rngs(custom_model, int(cfg.SEED))
    if calibrate:
        calibrate_loss_weights(custom_model, train_ds, cfg, prepared.n_train)
    if cfg.ABLATE_LAMBDAS:
        ablate(custom_model, cfg.ABLATE_LAMBDAS)
    custom_model.compile(optimizer=optimizer_pair.main)
    t_build = time.perf_counter() - t2

    t3 = time.perf_counter()
    sampler = _GradNormSampler(cfg.GRAD_CLIP_NORM, clip_skip_epochs=clip_skip_epochs)
    epoch_timer = _EpochTimer()
    history = custom_model.fit(train_ds, validation_data=val_ds, epochs=int(cfg.EPOCHS),
                               callbacks=[sampler, epoch_timer], verbose=0)
    t_train = time.perf_counter() - t3

    t4 = time.perf_counter()
    health = _health_from_history(history, sampler, cfg, custom_model)
    auc = _direction_auc(custom_model, val_block, cfg, target_scaler)
    t_score = time.perf_counter() - t4
    tf.keras.backend.clear_session()
    return health, auc, {"load_s": prepared.load_s, "prep_s": prepared.prep_s, "build_s": t_build,
                         "train_s": t_train, "score_s": t_score, "epoch_s": list(epoch_timer.epoch_s)}


# ------------------------------------------------------------------ phase 2: reused-graph trials
class _TrialGroup:
    """One structural group's persistent apparatus (NT-092 phase 2): the model, both optimizers and
    the compiled ``train_function``/``test_function`` are built ONCE, from the group's first trial,
    and reused by every later trial that shares its :func:`structural_key`. A trial only resets:

    - the model's WEIGHTS (a throwaway model built fresh with ``Models.build`` at the trial's seed,
      copied in with ``set_weights`` - cheap, no tracing: tracing happens inside ``fit()``'s first
      call, not at construction);
    - every stochastic layer's dropout generator (:func:`reset_stateful_rngs`, from the trial's seed);
    - both optimizers' internal state (moments, ``iterations``): every optimizer variable assigned to
      zero;
    - the continuous hyperparameters (:data:`CONTINUOUS_FIELDS`): LR and the Adam betas via the
      Keras optimizer hyper properties, INDICATOR_LR_MULT via the indicator optimizer's own
      ``learning_rate``, GRAD_CLIP_NORM via ``custom_model.grad_clip_norm``, every LAMBDA_* via
      ``custom_model.set_lambda_values``;
    - calibration (``run.calibrate``) and ``ABLATE_LAMBDAS``, exactly as the fresh path applies them
      (structural: every trial in a group shares the same ``ABLATE_LAMBDAS``, see CONTINUOUS_FIELDS's
      docstring, but calibration still re-samples per trial since the data/lambdas differ).

    Never rebuilt or re-``compile``d between trials, so ``custom_model.fit()``'s first call after the
    group's first trial reuses the already-traced graph instead of retracing it."""

    def __init__(self, first_cfg: Config):
        from neural_trade.registries.models import Models
        from neural_trade.training.custom_model import CustomTrainModel
        from neural_trade.training.optim import build_optimizers

        self.key = structural_key(first_cfg)
        base_model = Models.build(first_cfg.MODEL_NAME, first_cfg)
        optimizer_pair = build_optimizers(first_cfg)
        self.optimizer_pair = optimizer_pair
        self.custom_model = CustomTrainModel(
            base_model=base_model, pred_scale=1.0, pred_mean=0.0,
            lambda_point=first_cfg.LAMBDA_POINT, lambda_local_trend=first_cfg.LAMBDA_LOCAL_TREND,
            lambda_global_trend=first_cfg.LAMBDA_GLOBAL_TREND,
            lambda_extended_trend=first_cfg.LAMBDA_EXTENDED_TREND, lambda_dir=first_cfg.LAMBDA_DIR,
            config=first_cfg, indicator_optimizer=optimizer_pair.indicator,
            inputs=base_model.inputs, outputs=base_model.outputs)
        self.custom_model.compile(optimizer=optimizer_pair.main)

    def _reset_optimizer(self, optimizer) -> None:
        for var in optimizer.variables():
            var.assign(tf.zeros_like(var))

    def _reset_continuous(self, cfg: Config) -> None:
        from neural_trade.training.lambdas import CONFIG_NAME_OF_KEY

        main, indicator = self.optimizer_pair.main, self.optimizer_pair.indicator
        main.learning_rate = float(cfg.LR)
        if hasattr(main, "beta_1"):
            main.beta_1 = float(cfg.ADAM_BETA1)
        if hasattr(main, "beta_2"):
            main.beta_2 = float(cfg.ADAM_BETA2)
        indicator.learning_rate = float(cfg.LR) * float(cfg.INDICATOR_LR_MULT)
        if hasattr(indicator, "beta_1"):
            indicator.beta_1 = float(cfg.ADAM_BETA1)
        if hasattr(indicator, "beta_2"):
            indicator.beta_2 = float(cfg.ADAM_BETA2)
        self.custom_model.grad_clip_norm = float(getattr(cfg, "GRAD_CLIP_NORM", 0.0) or 0.0)
        self.custom_model.set_lambda_values(**{f"lambda_{key}": float(getattr(cfg, name))
                                               for key, name in CONFIG_NAME_OF_KEY.items()})

    def run_one(self, cfg: Config, cache: Dict[str, Any], *, calibrate: bool,
               clip_skip_epochs: int = DEFAULT_CLIP_SKIP_EPOCHS
               ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Reset this group's model/optimizers to ``cfg``'s seed and continuous values, train and
        score - same return shape as :func:`_run_trial_light`. ``build_s`` is 0 (nothing is built:
        the model already exists); the wall this saves versus the fresh path IS the group's amortised
        trace cost."""
        from neural_trade.registries.models import Models
        from neural_trade.training.lambda_calibration import calibrate_loss_weights
        from neural_trade.training.lambdas import ablate
        from neural_trade.training.reset import reset_stateful_rngs
        from neural_trade.utils.seeding import seed_everything

        assert structural_key(cfg) == self.key, "run_one called with a config outside this group"
        seed_everything(int(cfg.SEED))
        prepared = _prepare_trial_data(cfg, cache)

        t2 = time.perf_counter()
        fresh = Models.build(cfg.MODEL_NAME, cfg)  # cheap: initial weights only, never traced/fit
        self.custom_model.base_model.set_weights(fresh.get_weights())
        reset_stateful_rngs(self.custom_model, int(cfg.SEED))
        self._reset_optimizer(self.optimizer_pair.main)
        self._reset_optimizer(self.optimizer_pair.indicator)
        self._reset_continuous(cfg)
        pred_scale = np.std(prepared.y_train) if np.std(prepared.y_train) > 0 else 1.0
        pred_mean = np.mean(prepared.y_train)
        self.custom_model.pred_scale.assign(float(pred_scale))
        self.custom_model.pred_mean.assign(float(pred_mean))
        if calibrate:
            calibrate_loss_weights(self.custom_model, prepared.train_ds, cfg, prepared.n_train)
        if cfg.ABLATE_LAMBDAS:
            ablate(self.custom_model, cfg.ABLATE_LAMBDAS)
        t_build = time.perf_counter() - t2

        t3 = time.perf_counter()
        sampler = _GradNormSampler(cfg.GRAD_CLIP_NORM, clip_skip_epochs=clip_skip_epochs)
        epoch_timer = _EpochTimer()
        history = self.custom_model.fit(prepared.train_ds, validation_data=prepared.val_ds,
                                        epochs=int(cfg.EPOCHS), callbacks=[sampler, epoch_timer], verbose=0)
        t_train = time.perf_counter() - t3

        t4 = time.perf_counter()
        health = _health_from_history(history, sampler, cfg, self.custom_model)
        auc = _direction_auc(self.custom_model, prepared.val_block, cfg, prepared.target_scaler)
        t_score = time.perf_counter() - t4
        return health, auc, {"load_s": prepared.load_s, "prep_s": prepared.prep_s, "build_s": t_build,
                             "train_s": t_train, "score_s": t_score, "epoch_s": list(epoch_timer.epoch_s)}


# ------------------------------------------------------------------ rules
def apply_rules(health: Mapping[str, Any], rules: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """(passed, reasons) of the pre-registered ``rules:`` block against one trial's health numbers."""
    reasons: List[str] = []
    if rules.get("finite", True) and not health.get("finite", False):
        reasons.append("non-finite value in the training history")
    if "max_nonfinite_grad_steps" in rules and health.get("nonfinite_grad_steps", 0) > rules["max_nonfinite_grad_steps"]:
        reasons.append(f"nonfinite_grad_steps {health['nonfinite_grad_steps']} > {rules['max_nonfinite_grad_steps']}")
    share = health.get("clipped_share")
    if "max_clipped_share" in rules and share is not None and share > rules["max_clipped_share"]:
        reasons.append(f"clipped_share {share:.4f} > {rules['max_clipped_share']}")
    if "min_train_loss_drop" in rules:
        drop = health.get("train_loss_drop")
        if drop is None or drop < rules["min_train_loss_drop"]:
            reasons.append(f"train_loss_drop {drop} < {rules['min_train_loss_drop']}")
    term_share = health.get("max_term_share")
    if "max_term_share" in rules and term_share is not None and term_share > rules["max_term_share"]:
        reasons.append(f"max_term_share {term_share:.4f} > {rules['max_term_share']}")
    return (not reasons, reasons)


# ------------------------------------------------------------------ running a screen
@dataclass
class ScreenReport:
    name: str
    results_path: str
    n_trials: int
    ran: int = 0
    skipped: int = 0
    passed: int = 0
    failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "results_path": self.results_path, "n_trials": self.n_trials,
                "ran": self.ran, "skipped": self.skipped, "passed": self.passed, "failed": self.failed}


def _existing_keys(path: Path) -> set:
    keys = set()
    for row in _read_jsonl(path):
        key = row.get("trial_key")
        if key is not None:
            keys.add(key)
    return keys


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _shard_result_path(out_dir: Path, shard: Optional[Tuple[int, int]]) -> Path:
    """The file THIS process appends to: the plain, unsharded ``results.jsonl`` when ``shard`` is
    ``None``, or its own ``results.shard-{i}-of-{N}.jsonl`` (0-indexed) when running as one of ``N``
    ``--shard`` processes, so concurrent shards never append to the same file (previously they all
    wrote ``results.jsonl``, risking interleaved/corrupted lines)."""
    if shard is None:
        return out_dir / RESULTS_FILE
    i, n = shard
    return out_dir / f"results.shard-{i}-of-{n}.jsonl"


def _shard_glob(out_dir: Path, n: int) -> List[Path]:
    return sorted(out_dir.glob(f"results.shard-*-of-{n}.jsonl"))


def merge_results(store="runs", name: Optional[str] = None, *, n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every row of screen ``name``'s results, merged across its shard files
    (``results.shard-{i}-of-{N}.jsonl``) and a plain ``results.jsonl`` (an unsharded run), de-duplicated
    by ``trial_key`` (first file wins). Used for resumability (a shard skips a trial any shard already
    finished) and for reporting total progress across concurrently running ``--shard`` processes. ``n``
    restricts the shard files read to one shard count (glob ``results.shard-*-of-{n}.jsonl``); omitted,
    every shard file present is read (``results.shard-*-of-*.jsonl``)."""
    out_dir = Path(store) / "screens" / str(name)
    paths: List[Path] = []
    plain = out_dir / RESULTS_FILE
    if plain.is_file():
        paths.append(plain)
    paths += _shard_glob(out_dir, n) if n is not None else sorted(out_dir.glob("results.shard-*-of-*.jsonl"))
    rows: Dict[Any, Dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            rows.setdefault(row.get("trial_key"), row)
    return list(rows.values())


def _merged_existing_keys(out_dir: Path, shard: Optional[Tuple[int, int]]) -> set:
    """``trial_key``s already recorded by ANY shard of this run (or the plain file), so resuming one
    shard also skips a trial some other shard already finished, not only its own file -- but ONLY
    shard files for the SAME total shard count ``N`` (``shard[1]``, via ``_shard_glob``'s
    ``results.shard-*-of-{n}.jsonl`` glob): shard files left over from a run with a DIFFERENT ``N``
    are never merged or considered here, since a different ``N`` partitions the trial grid
    differently and its shard files are not comparable to this run's."""
    if shard is None:
        return _existing_keys(out_dir / RESULTS_FILE)
    keys = _existing_keys(out_dir / RESULTS_FILE)   # a prior unsharded run, if any
    for path in _shard_glob(out_dir, shard[1]):
        keys |= _existing_keys(path)
    return keys


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str, allow_nan=False) + "\n")


def _sanitize_nonfinite(obj: Any, path: str = "") -> Tuple[Any, List[str]]:
    """Recursively replace a non-finite float (NaN, +Inf, -Inf) anywhere inside ``obj`` with ``None``
    (JSON ``null``), returning ``(sanitized, dotted_paths_that_were_non_finite)``. An extreme LAMBDA_*
    can drive a health number (or the direction AUC) to NaN/Inf; ``json.dumps(..., allow_nan=False)``
    (:func:`_append_jsonl`, D-012/D-020-adjacent: every JSONL row must stay strictly valid JSON) would
    otherwise raise on that ONE row and leave the trial permanently unrecorded - a resume just re-runs
    and crashes again. Writing ``null`` instead (never a raw NaN/Infinity token, which the standard
    does not allow) keeps the row, and the trial resumable, at the cost of losing that one number."""
    bad: List[str] = []

    def walk(o: Any, p: str) -> Any:
        if isinstance(o, float):
            if not math.isfinite(o):
                bad.append(p or "<root>")
                return None
            return o
        if isinstance(o, dict):
            return {k: walk(v, f"{p}.{k}" if p else str(k)) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{p}[{i}]") for i, v in enumerate(o)]
        return o

    return walk(obj, path), bad


def run_trial(trial: Trial, spec: ScreenSpec, cache: Dict[str, Any],
             trainer: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Train and score one trial; returns the JSONL row (schema_version, trial identity, config
    diff, health, direction AUC, timings, pass/fail with reasons). ``trainer`` (tests only) replaces
    the real light path: ``trainer(cfg, cache, calibrate=...) -> (health, auc, timings)``.

    Any non-finite number anywhere in the row (see :func:`_sanitize_nonfinite`) is written as ``null``
    and forces ``passed: false`` with a reason naming the field(s), regardless of the spec's
    ``rules:`` - a non-finite health number means training itself broke, not a threshold decision."""
    t0 = time.perf_counter()
    if trainer is not None:
        health, auc, timings = trainer(trial.config, cache, calibrate=spec.run.calibrate)
    else:
        clip_skip_epochs = spec.effective_clip_skip_epochs(int(trial.config.EPOCHS))
        health, auc, timings = _run_trial_light(trial.config, cache, calibrate=spec.run.calibrate,
                                                clip_skip_epochs=clip_skip_epochs)
    passed, reasons = apply_rules(health, spec.rules)
    row = {"schema_version": SCHEMA_VERSION, "screen": spec.name, "trial_key": trial.key,
          "trial_index": trial.index, "source": trial.source, "config_diff": trial.params,
          "data_end": trial.data_end, "seed": trial.seed, "health": health, "direction_auc": auc,
          "timings": timings, "passed": passed, "reasons": reasons, "wall_s": time.perf_counter() - t0}
    row, nonfinite_paths = _sanitize_nonfinite(row)
    if nonfinite_paths:
        row["passed"] = False
        row["reasons"] = list(row["reasons"]) + [f"non-finite value(s) written as null: {', '.join(nonfinite_paths)}"]
    return row


def _group_order(trials: Sequence[Trial]) -> List[Trial]:
    """``trials``, stably sorted by :func:`structural_key` (trials of one group become contiguous,
    keeping their original relative order within the group and between groups' first appearances), so
    :func:`run_screen`'s reuse path builds each group's :class:`_TrialGroup` once and never has to hold
    more than one group's model in memory at a time. Trial IDENTITY (``trial.index``, ``trial.key``)
    and the written row are unaffected: only the ORDER trials run in changes, which resumability does
    not depend on (rows are merged by ``trial_key``, not position)."""
    first_seen: Dict[str, int] = {}
    for t in trials:
        first_seen.setdefault(structural_key(t.config), len(first_seen))
    return sorted(trials, key=lambda t: (first_seen[structural_key(t.config)], t.index))


def run_screen(spec: ScreenSpec, *, store="runs", shard: Optional[Tuple[int, int]] = None,
               max_trials: Optional[int] = None,
               trainer: Optional[Callable[..., Any]] = None) -> ScreenReport:
    """Run every not-yet-resumed trial of ``spec`` that belongs to ``shard`` (default: all of them),
    appending one line per trial to this shard's results file (:func:`_shard_result_path`). At most
    ``max_trials`` trials run in this call (the same command resumes the rest).

    When ``trainer`` is given (tests only), every trial runs through it independently: grouping and
    graph reuse are meaningless for a fake, single-call objective. Otherwise, when
    ``spec.run.reuse_graph`` (default true), trials are processed in :func:`_group_order` and each
    contiguous run of one :func:`structural_key` shares one :class:`_TrialGroup`; ``reuse_graph:
    false`` keeps every trial on the original fresh-model-per-trial path (:func:`_run_trial_light`)."""
    trials = [t for t in build_trials(spec) if shard_of(t.index, shard)]
    out_dir = Path(store) / "screens" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / f"spec-{spec.spec_hash}.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2, default=str), encoding="utf-8")
    results_path = _shard_result_path(out_dir, shard)
    done_keys = _merged_existing_keys(out_dir, shard)
    cache: Dict[str, Any] = {}
    report = ScreenReport(spec.name, str(results_path), len(trials))
    clip_skip_epochs = spec.effective_clip_skip_epochs(int(spec.base().EPOCHS))
    use_groups = trainer is None and spec.run.reuse_graph
    ordered = _group_order(trials) if use_groups else trials
    group: Optional[_TrialGroup] = None
    for position, trial in enumerate(ordered, start=1):
        if trial.key in done_keys:
            report.skipped += 1
            continue
        if max_trials is not None and report.ran >= max_trials:
            break
        logger.info("[screen %s] trial %d/%d (%s %s, slice %s, seed %s)", spec.name, position,
                    len(trials), trial.source, trial.params, trial.data_end, trial.seed)
        if use_groups:
            key = structural_key(trial.config)
            if group is None or group.key != key:
                if group is not None:
                    tf.keras.backend.clear_session()
                group = _TrialGroup(trial.config)
            row = run_trial(trial, spec, cache,
                            trainer=lambda cfg, c, *, calibrate, _g=group: _g.run_one(
                                cfg, c, calibrate=calibrate, clip_skip_epochs=clip_skip_epochs))
        else:
            # trainer=None here (use_groups is False and it can still be an explicit fake): run_trial's
            # own default branch computes clip_skip_epochs from spec, identically to the group branch.
            row = run_trial(trial, spec, cache, trainer=trainer)
        _append_jsonl(results_path, row)
        done_keys.add(trial.key)
        report.ran += 1
        report.passed += int(row["passed"])
        report.failed += int(not row["passed"])
    if group is not None:
        tf.keras.backend.clear_session()
    return report


__all__ = ["CONTINUOUS_FIELDS", "DEFAULT_CLIP_SKIP_EPOCHS", "RESULTS_FILE", "RunOptions", "ScreenError",
          "ScreenReport", "ScreenSpec", "SCHEMA_VERSION", "Trial", "apply_rules", "build_trials",
          "merge_results", "parse_shard", "run_screen", "run_trial", "shard_of", "structural_key"]

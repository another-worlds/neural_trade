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
and, for a run context, writes a serving bundle): data loaded and prepared ONCE per data key per
process (the cache is keyed by :func:`neural_trade.experiments.dataset.data_key`, which already
covers every Config field that can change the prepared bars, DATA_END included), a real
``CustomTrainModel`` trained for the spec's ``run.epochs``, no checkpoints, no baselines, no
backtest, no random null, no stored predictions, no serving bundle. ``run.calibrate`` switches the
pre-training loss-weight calibration pass (``training.lambda_calibration.calibrate_loss_weights``)
on or off, same as a scenario's ``run.calibrate``.

Health numbers (all finite, ``nonfinite_grad_steps``, the max and mean of the per-logged-step
``grad_global_norm``, the share of logged steps at or above ``GRAD_CLIP_NORM``, the training loss's
first-to-last-epoch drop, the final validation loss, and each loss term's share of the final total)
come from the same per-epoch aggregates ``CustomTrainModel`` already exposes in
``history.history`` (``training/custom_model.py``'s ``_EpochTrainLogs``), plus a small per-batch
sampler (:class:`_GradNormSampler`) for the max / share, which epoch aggregates alone cannot give.
Pre-registered ``rules:`` in the spec turn the health numbers into a pass/fail with reasons.

Resumable: ``results.jsonl`` already holding a trial's key (a hash of its exact Config values,
:func:`neural_trade.experiments.scenario.config_hash`) skips it. ``--shard i/N`` runs trial index
``j`` only when ``j % N == i``: the shards are disjoint and their union is every trial, so ``N``
processes (NT-035's 3-process ceiling) split a screen without locking any cell.
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
RUN_KEYS = ("calibrate", "epochs")
RULE_KEYS = ("finite", "max_nonfinite_grad_steps", "max_clipped_share", "min_train_loss_drop", "max_term_share")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")

# The final-epoch loss-term keys CustomTrainModel logs (training/custom_model.py:_update_diagnostics),
# read from history.history to compute each term's share of the final total loss.
LOSS_TERM_KEYS = ("point_loss", "trend_loss", "dir_loss", "nll_loss", "crps_loss", "soft_ece_loss",
                  "reg_loss", "inter_reg", "vol_loss", "t_perp_loss", "casimir_loss", "vac_loss",
                  "hd_loss", "ife_loss", "vac_overflow_loss")


class ScreenError(InvalidConfigurationError):
    """A screen spec is malformed or names an invalid configuration (a ValueError)."""


# ------------------------------------------------------------------ the spec
@dataclass
class RunOptions:
    calibrate: bool = True
    epochs: Optional[int] = None


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
        rules = _mapping(data.get("rules"), "rules", bad)
        _refuse_unknown(rules, RULE_KEYS, "rules.", bad)
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
        return _step(lambda: cfg.override(**overrides), "overrides", self.where())

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
    spec (raises :class:`ScreenError` naming the offending key). Needs no data and trains nothing."""
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
    return trials


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
    the delta between two observations equals the value of the single update between them."""

    def __init__(self, clip_norm: float):
        super().__init__()
        self.clip_norm = float(clip_norm or 0.0)
        self._prev_total = 0.0
        self._prev_count = 0.0
        self.n_steps = 0
        self.max_norm: Optional[float] = None
        self._sum_norm = 0.0
        self.n_clipped = 0

    @property
    def mean_norm(self) -> Optional[float]:
        return self._sum_norm / self.n_steps if self.n_steps else None

    @property
    def clipped_share(self) -> Optional[float]:
        return self.n_clipped / self.n_steps if self.n_steps else None

    def on_epoch_begin(self, epoch, logs=None):
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


def _health_from_history(history, sampler: _GradNormSampler) -> Dict[str, Any]:
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
                term_shares[key] = v / final_total
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


def _run_trial_light(cfg: Config, cache: Dict[str, Any], *, calibrate: bool
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """Train ``cfg`` on the light path and return ``(health, direction_auc, timings)``. Data (load +
    preprocess + the DATA_END slice) is cached in ``cache`` by :func:`data_key`, so many trials that
    share a data key pay that cost once per process."""
    from neural_trade.data.processor import DataProcessor
    from neural_trade.data.datasets import create_datasets
    from neural_trade.registries.models import Models
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.training.lambda_calibration import calibrate_loss_weights
    from neural_trade.training.lambdas import ablate
    from neural_trade.training.optim import build_optimizers
    from neural_trade.utils.seeding import seed_everything

    seed_everything(int(cfg.SEED))
    t0 = time.perf_counter()
    df, close = _load_cached(cfg, cache)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    dp = DataProcessor(cfg)
    (X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
     X_test_seq, y_test_scaled, last_close_test, extended_trends_test,
     y_train, y_test, target_scaler) = dp.prepare_datasets(df, close)
    val_block = dp.val_block
    train_ds, val_ds = create_datasets(cfg, X_train_seq, y_train_scaled, last_close_train,
                                       extended_trends_train, val_block["X"], val_block["y_scaled"],
                                       val_block["last_close"], val_block["extended_trends"])
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
    if calibrate:
        calibrate_loss_weights(custom_model, train_ds, cfg, X_train_seq.shape[0])
    if cfg.ABLATE_LAMBDAS:
        ablate(custom_model, cfg.ABLATE_LAMBDAS)
    custom_model.compile(optimizer=optimizer_pair.main)
    t_build = time.perf_counter() - t1

    t2 = time.perf_counter()
    sampler = _GradNormSampler(cfg.GRAD_CLIP_NORM)
    history = custom_model.fit(train_ds, validation_data=val_ds, epochs=int(cfg.EPOCHS),
                               callbacks=[sampler], verbose=0)
    t_train = time.perf_counter() - t2

    t3 = time.perf_counter()
    health = _health_from_history(history, sampler)
    auc = _direction_auc(custom_model, val_block, cfg, target_scaler)
    t_score = time.perf_counter() - t3
    tf.keras.backend.clear_session()
    return health, auc, {"load_s": t_load, "build_s": t_build, "train_s": t_train, "score_s": t_score}


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
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            keys.add(json.loads(line)["trial_key"])
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str, allow_nan=False) + "\n")


def run_trial(trial: Trial, spec: ScreenSpec, cache: Dict[str, Any],
             trainer: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Train and score one trial; returns the JSONL row (schema_version, trial identity, config
    diff, health, direction AUC, timings, pass/fail with reasons). ``trainer`` (tests only) replaces
    the real light path: ``trainer(cfg, cache, calibrate=...) -> (health, auc, timings)``."""
    t0 = time.perf_counter()
    fn = trainer if trainer is not None else _run_trial_light
    health, auc, timings = fn(trial.config, cache, calibrate=spec.run.calibrate)
    passed, reasons = apply_rules(health, spec.rules)
    return {"schema_version": SCHEMA_VERSION, "screen": spec.name, "trial_key": trial.key,
            "trial_index": trial.index, "source": trial.source, "config_diff": trial.params,
            "data_end": trial.data_end, "seed": trial.seed, "health": health, "direction_auc": auc,
            "timings": timings, "passed": passed, "reasons": reasons, "wall_s": time.perf_counter() - t0}


def run_screen(spec: ScreenSpec, *, store="runs", shard: Optional[Tuple[int, int]] = None,
               max_trials: Optional[int] = None,
               trainer: Optional[Callable[..., Any]] = None) -> ScreenReport:
    """Run every not-yet-resumed trial of ``spec`` that belongs to ``shard`` (default: all of them),
    appending one line per trial to ``<store>/screens/<name>/results.jsonl``. At most ``max_trials``
    trials run in this call (the same command resumes the rest)."""
    trials = [t for t in build_trials(spec) if shard_of(t.index, shard)]
    out_dir = Path(store) / "screens" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / f"spec-{spec.spec_hash}.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2, default=str), encoding="utf-8")
    results_path = out_dir / RESULTS_FILE
    done_keys = _existing_keys(results_path)
    cache: Dict[str, Any] = {}
    report = ScreenReport(spec.name, str(results_path), len(trials))
    for trial in trials:
        if trial.key in done_keys:
            report.skipped += 1
            continue
        if max_trials is not None and report.ran >= max_trials:
            break
        logger.info("[screen %s] trial %d/%d (%s %s, slice %s, seed %s)", spec.name, trial.index + 1,
                    len(trials), trial.source, trial.params, trial.data_end, trial.seed)
        row = run_trial(trial, spec, cache, trainer=trainer)
        _append_jsonl(results_path, row)
        done_keys.add(trial.key)
        report.ran += 1
        report.passed += int(row["passed"])
        report.failed += int(not row["passed"])
    return report


__all__ = ["RESULTS_FILE", "RunOptions", "ScreenError", "ScreenReport", "ScreenSpec", "SCHEMA_VERSION", "Trial",
          "apply_rules", "build_trials", "parse_shard", "run_screen", "run_trial", "shard_of"]

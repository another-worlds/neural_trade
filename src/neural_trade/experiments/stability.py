"""The on-demand stability harness (NT-038, D-026).

    neural-trade stability --profile tiny|reference [--csv FILE] [--store runs] [--cases a,b] [--seeds 0,1,2]

Every setup must pass this before it is trusted (VISION MVP 3). The harness stresses one setup with

* **scale and volatility**: the price level and the log-return volatility of the data x0.1 and x10;
* **extreme inputs**: a long constant block (flat windows), one-bar spikes and a permanent level jump, prices
  x1e4 and x1e-4 (the data are rewritten into ``<report dir>/data/<case>.csv``; the dataset fingerprint of the
  run records which file it trained on);
* **fault injection**: a NaN in the input windows, in one loss term (``crps_loss``) and in one gradient
  (``point_loss``), each of which must STOP the run with :class:`~neural_trade.training.stability_guard
  .UnstableTrainingError` naming the term;
* **named configurations** (BACKLOG NT-038 notes): the wide-span horizons 5/60/240, the slow-period start with
  INDICATOR_LR_MULT 5 and 1. Cases a CPU cannot hold (periods of 1,440 and 10,080 bars; the per-channel scale
  variant, which has no Config switch) are defined and marked ``GPU, NT-051``: they are listed in the report,
  never run here;

each with 3 seeds (the thresholds file's ``seeds``), in **strict mode** (``STRICT_LOSS_MASKS``: the loss masks
off, so a non-finite term reaches the total instead of being hidden).

**The per-term gradient probe** (``--probe off|on|failed``, NT-191) costs about 12x a reference cell on CPU (777 s
against 58 s, identical numbers: the probe never changes training). ``failed`` (the reference profile's default):
every cell runs with the probe OFF; a cell that fails a verdict check is re-run ONCE with the probe on
(PROBE_EVERY 1) and the REPORT blames the loss term of that re-run's probe sample. ``on``: every cell probed
(the earlier behaviour). ``off``: never (the tiny profile's default). A cell whose run ended in a resource error
(``ResourceExhaustedError``, ``MemoryError``, ``OSError``, a worker crash) is **not a verdict**: it is reported as
such, left out of the pass and fail counts and re-run by ``--retry-non-verdict <harness id>`` as a new launch.

**Cases are engine scenarios.** :func:`build_scenario` makes one variant per case; :class:`~neural_trade
.experiments.runner.Runner` trains every (case, seed) cell into the run store and its index (the harness
trainer applies the case's fault). After the runs, :func:`evaluate_run` judges each cell against the thresholds
file (``configs/stability_thresholds.yaml`` = v1, the default; ``--thresholds v2`` names the v2 file of NT-187: n_eff
gates, scaled NLL, a degenerate-baseline guard; pre-registered; the sha256 of the file used is in every report), writes
``stability_verdict.json`` into the cell's run directory (the store indexes it as ``stability/*`` scores) and
:func:`write_report` writes ``runs/stability/<id>/REPORT.md`` (pass or fail per case, the loss term blamed for
a failure), ``verdicts.json`` and ``failing_regions.json`` (machine-readable: ``core.guard`` refuses a
configuration inside one and the sweep search spaces exclude them).

**Failing regions** come only from configuration cases (the case's own overrides become the region's
conditions, an exact point: the harness does not extrapolate). A data or fault case that fails is reported,
with no region (the data are not a Config field).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
THRESHOLDS_FILE = CONFIGS_DIR / "stability_thresholds.yaml"            # v1: frozen, the default until a SPEC names v2
THRESHOLDS_V2_FILE = CONFIGS_DIR / "stability_thresholds_v2.yaml"      # NT-187
# The thresholds file a profile uses when none is named (None: THRESHOLDS_FILE, v1). NT-051's SPEC names v2 here or on
# the command line (`--thresholds v2`); the report carries the sha256 of the file actually used.
PROFILE_THRESHOLDS: Dict[str, Optional[Path]] = {"tiny": None, "reference": None}
FUZZ_CONSTANT_BARS = 100     # NT-187: 60-120 bars; flat TRAINING windows stay a minority (thresholds v2 `case_design`)
VERDICT_FILE = "stability_verdict.json"
STABILITY_DIR = "stability"
GPU_NOTE = "GPU, NT-051"

# Config overrides every case shares: strict mode (masks off). The per-term gradient probe is a profile's choice:
# it costs about 10% of a GPU step but, on CPU, a tiny run's trace of 17 terms x 3 groups x 136 pairs took 200 s
# against 30 s without it (measured 2026-10-06), so the tiny profile leaves it off (that check is then
# reported "not evaluated"). PROFILES holds the probe's `on` setting; PROFILE_PROBE says when it is used.
COMMON_OVERRIDES: Dict[str, Any] = {"STRICT_LOSS_MASKS": True}
# The probe (NT-191): `off` never, `on` every cell (the profile's PROBE_EVERY), `failed` off first and a probe-on re-run
# (PROBE_EVERY 1) of each cell that fails a verdict check. A profile's default when none is named:
PROBE_MODES = ("off", "on", "failed")
PROFILE_PROBE: Dict[str, str] = {"tiny": "off", "reference": "failed"}
PROBE_SOURCE = "probe sample, one batch"      # what the REPORT calls a blame taken from the probe
# Error types that say the machine, not the setup, failed: not a verdict (NT-191). OSError's subclasses count too.
NON_VERDICT_ERRORS = ("ResourceExhaustedError", "MemoryError", "OSError", "BrokenProcessPool", "WorkerCrash")
NOT_A_VERDICT = "NOT A VERDICT"
# `failed` mode: at most this many probe re-runs per launch. 777 s per reference cell on CPU with the probe (measured at
# PROBE_EVERY 5, the reference profile's cadence) against the 3 h cap of OPERATING_MODEL:
# floor((10800 - sum of the probe-off times) / 777) = 10. A re-run probes at PROBE_EVERY 1, which on the tiny profile
# cost about 1.25x the PROBE_EVERY-5 run (236.7 against 189.9 s, CPU), so the cap may be about 8 at the re-run's real
# cost: NT-051's SPEC states its measured per-re-run cost and the cap it uses.
MAX_PROBE_RERUNS = 10
EXIT_REFUSED = 64            # the harness refused its arguments and ran nothing (1 and 2 are results)
NOT_RERUN_CAP = "not re-run (cap)"
# Profiles: the data layout and training length of a harness run. `tiny` is the CPU test size.
PROFILES: Dict[str, Dict[str, Any]] = {
    # tiny: one close-only instance per family (the tests' NT-109 size: the OHLCV catalogue costs wall time)
    "tiny": {"MAX_SEQUENCE_COUNT": 700, "EPOCHS": 1, "BATCH_SIZE": 32, "PROBE_GRADIENTS": False, "TRAIN_METRICS_EVERY": 1,
             "INPUT_SERIES": ["close"], "INDICATOR_FAMILIES": {}, "MA_SPANS": [5],
             "MACD_SETTINGS": [{"fast": 12, "slow": 26, "signal": 9}], "RSI_PERIODS": [14], "BB_PERIODS": [20]},
    "reference": {"N_FOLDS": 2, "MAX_SEQUENCE_COUNT": 4500, "EPOCHS": 3, "PROBE_GRADIENTS": True, "PROBE_EVERY": 5,
                  "TRAIN_METRICS_EVERY": 1,
                  "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1},
}
PROFILE_FOLDS = [-2]
FAULT_TERMS = {"crps_loss": "crps_gaussian_loss"}      # masked counter name -> the losses.functions function


def probe_overrides(profile: str, mode: str) -> Dict[str, Any]:
    """The Config overrides of a probe mode: off (and failed's first run) no probe; on the profile's cadence; rerun
    the probe on every step (the blame re-run of a failed cell)."""
    if mode == "rerun":
        return {"PROBE_GRADIENTS": True, "PROBE_EVERY": 1}
    if mode == "on":
        return {"PROBE_GRADIENTS": True, "PROBE_EVERY": int(PROFILES[profile].get("PROBE_EVERY", 5))}
    if mode in ("off", "failed"):
        return {"PROBE_GRADIENTS": False}
    raise ValueError(f"probe must be one of {list(PROBE_MODES)}, got {mode!r}")


# OSError subclasses that say the setup is wrong (a missing file, a bad path, no permission), not the machine.
SETUP_OS_ERRORS = ("FileNotFoundError", "FileExistsError", "NotADirectoryError", "IsADirectoryError", "PermissionError")
# Windows' transient lock failures (access denied, sharing and lock violations: another process holds the file; NT-185,
# D-065): a PermissionError (or other OSError) with one of these winerror codes is the machine, not the setup.
TRANSIENT_WINERRORS = (5, 32, 33)
_WINERROR = re.compile(r"\[WinError (\d+)\]")


def _winerror(message: str, winerror=None) -> Optional[int]:
    """The Windows error code: ``winerror`` when given, else the first ``[WinError N]`` found anywhere in the message."""
    if winerror is not None:
        try:
            return int(winerror)
        except (TypeError, ValueError):
            return None
    m = _WINERROR.search(message or "")
    return int(m.group(1)) if m else None
# TensorFlow's InternalError and UnknownError are a verdict-free machine failure only with one of these in the message.
RESOURCE_MESSAGE = re.compile(r"out of memory|oom|alloc|cudnn|cuda_error|cublas|cusolver|resource exhausted|"
                              r"paging file|no space left", re.IGNORECASE)
RESOURCE_TF_ERRORS = ("InternalError", "UnknownError")


def is_non_verdict_error(name, message: str = "", *, winerror=None) -> bool:
    """True for an error type that is a resource or machine failure, not a result of the setup. A deterministic setup
    error (FileNotFoundError and the other path errors) is a verdict-side error, except a PermissionError whose Windows
    code (``winerror``, or ``[WinError N]`` in ``message``) is a transient lock (:data:`TRANSIENT_WINERRORS`);
    TensorFlow's InternalError and UnknownError count only when ``message`` names memory, an allocation or
    cuDNN/CUDA."""
    import builtins

    if not name:
        return False
    name = str(name)
    if name == "PermissionError" and _winerror(message, winerror) in TRANSIENT_WINERRORS:
        return True
    if name in SETUP_OS_ERRORS:
        return False
    if name in RESOURCE_TF_ERRORS:
        return bool(RESOURCE_MESSAGE.search(message or ""))
    if name in NON_VERDICT_ERRORS:
        return True
    cls = getattr(builtins, name, None)
    return isinstance(cls, type) and issubclass(cls, OSError)


# ------------------------------------------------------------------ thresholds
@dataclass(frozen=True)
class Thresholds:
    path: Path
    sha256: str
    name: str
    checks: Mapping[str, Any]
    fault_detection: Mapping[str, Any]
    seeds: int
    report_only: Sequence[str] = ()
    schema_version: int = 1
    case_design: Mapping[str, Any] = field(default_factory=dict)       # v2: pre-registered case designs
    expected_n_eff: Mapping[str, Any] = field(default_factory=dict)    # v2: expected n_eff per profile and case

    def check(self, key: str) -> Any:
        return self.checks[key]

    def has(self, key: str) -> bool:
        return key in self.checks


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_thresholds(path=None) -> Thresholds:
    """The pre-registered thresholds with the file's sha256 (any edit, a comment included, changes it)."""
    import yaml

    p = resolve_thresholds_path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if doc.get("schema_version") not in (1, 2):
        raise ValueError(f"{p}: schema_version must be 1 or 2")
    return Thresholds(p, file_sha256(p), str(doc["name"]), dict(doc["checks"]), dict(doc["fault_detection"]),
                      int(doc.get("seeds", 3)), tuple(doc.get("report_only") or ()), int(doc["schema_version"]),
                      dict(doc.get("case_design") or {}), dict(doc.get("expected_n_eff") or {}))


def resolve_thresholds_path(spec=None, profile: Optional[str] = None) -> Path:
    """The thresholds file: a path, or a name of a file in configs/ (``v2``, ``stability_thresholds_v2``, with or
    without ``.yaml``); None means the profile's default (:data:`PROFILE_THRESHOLDS`), else v1."""
    if spec is None:
        return Path(PROFILE_THRESHOLDS.get(profile or "") or THRESHOLDS_FILE)
    p = Path(spec)
    if p.is_file():
        return p
    name = str(spec)
    for cand in (name, f"{name}.yaml", f"stability_thresholds_{name}.yaml", f"stability_thresholds_{name}"):
        if (CONFIGS_DIR / cand).is_file() and not Path(cand).is_absolute():
            return CONFIGS_DIR / cand
    raise FileNotFoundError(f"no thresholds file {spec!r} (a path, or a name in {CONFIGS_DIR})")


# ------------------------------------------------------------------ cases
@dataclass(frozen=True)
class Case:
    id: str
    group: str                       # scale | volatility | extreme_input | fault | configuration
    description: str
    overrides: Mapping[str, Any] = field(default_factory=dict)      # Config overrides (a configuration case's region)
    data: Optional[Mapping[str, Any]] = None                        # a transform of the bars, see transform_bars
    fault: Optional[Mapping[str, Any]] = None                       # {"kind": nan_input|nan_term|nan_gradient, ...}
    expect: str = "pass"             # pass: the run is healthy | detect: the guard must stop it, naming the term
    runnable: bool = True            # False: defined, not run here (reason in `note`)
    note: str = ""
    region: bool = False             # a configuration case: a failure writes a failing region
    layout: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)   # profile -> data-layout overrides the
                                                                            # case needs (never part of a region)

    @property
    def expects_term(self) -> Optional[str]:
        return (self.fault or {}).get("term")


def default_cases() -> List[Case]:
    cs: List[Case] = [Case("control", "scale", "the setup unchanged: the harness's own sanity check")]
    for k in (0.1, 10):
        cs.append(Case(f"scale_x{k:g}", "scale", f"price level x{k:g} (OHLC; returns unchanged)",
                       data={"kind": "scale", "k": float(k)}))
    for k in (0.1, 10):
        cs.append(Case(f"vol_x{k:g}", "volatility", f"log-return volatility x{k:g} (price path rescaled)",
                       data={"kind": "vol", "k": float(k)}))
    cs += [
        Case("fuzz_constant", "extreme_input", f"a {FUZZ_CONSTANT_BARS}-bar block of one constant price (flat windows; a minority of "
                                                  "the training windows)",
             data={"kind": "constant", "bars": FUZZ_CONSTANT_BARS}),
        Case("fuzz_jumps", "extreme_input", "five one-bar spikes (x4, x0.25) and a permanent x2 level jump",
             data={"kind": "jumps"}),
        Case("fuzz_large_price", "extreme_input", "prices x1e4 (about 1e9)", data={"kind": "scale", "k": 1e4}),
        Case("fuzz_small_price", "extreme_input", "prices x1e-4 (about 11)", data={"kind": "scale", "k": 1e-4}),
        Case("fault_nan_input", "fault", "a NaN in one training window", fault={"kind": "nan_input"}, expect="detect"),
        Case("fault_nan_term", "fault", "a NaN in one loss term (crps_loss)", overrides={"LAMBDA_CRPS": 1.0},
             fault={"kind": "nan_term", "term": "crps_loss"}, expect="detect"),
        Case("fault_nan_gradient", "fault", "a NaN in one gradient (point_loss, finite value)",
             fault={"kind": "nan_gradient"}, expect="detect"),
        Case("horizons_5_60_240", "configuration",
             "wide-span horizons 5/60/240 bars (a pooled scaler gives scaled variances 0.054/0.586/2.359 against "
             "v_ref 1 and VAR_FLOOR 1e-4, NT-141 review 2026-10-06)",
             overrides={"HORIZON_STEPS": [5, 60, 240], "EXTENDED_TREND_PERIODS": [5, 60, 240]}, region=True,
             layout={"tiny": {"MAX_SEQUENCE_COUNT": 2700, "N_FOLDS": 4},      # the purge gap is 480 bars here
                     "reference": {"MAX_SEQUENCE_COUNT": 9000, "N_FOLDS": 3}}),
    ]
    for lr in (5.0, 1.0):
        cs.append(Case(f"slow_periods_proxy_lr{lr:g}", "configuration",
                       f"slow periods starting at the window length (CPU proxy of the long-memory case), "
                       f"INDICATOR_LR_MULT {lr:g}",
                       overrides={"MA_SPANS": [50], "RSI_PERIODS": [50], "BB_PERIODS": [50], "INDICATOR_LR_MULT": lr}, region=True,
                       note="a proxy: the real case needs periods of 1,440 and 10,080 bars"))
    for n in (1440, 10080):
        for lr in (5.0, 1.0):
            cs.append(Case(f"long_memory_{n}_lr{lr:g}", "configuration",
                           f"slow indicator periods starting at {n} bars, INDICATOR_LR_MULT {lr:g}",
                           overrides={"INDICATOR_LR_MULT": lr}, runnable=False,
                           note=f"{GPU_NOTE}: the window model caps LOOKBACK at 1,440 and learned periods at "
                                "LOOKBACK; periods this long need a window of that many bars (GPU memory, "
                                "BATCH_SIZE x LOOKBACK^2) or the series engine (research track R6)"))
    cs.append(Case("long_memory_scale_norm", "configuration",
                   "the long-memory case with per-channel scale normalisation of the indicator inputs",
                   runnable=False, note=f"{GPU_NOTE}: no Config switch for a per-channel scale normalisation exists "
                                        "yet (B_model_indicators.md item 5)"))
    return cs


def case_by_id(cases: Sequence[Case]) -> Dict[str, Case]:
    return {c.id: c for c in cases}


# ------------------------------------------------------------------ data transforms (extreme inputs)
def _columns(df) -> Dict[str, str]:
    lower = {str(c).lower(): c for c in df.columns}
    cols = {k: lower[k] for k in ("open", "high", "low", "close") if k in lower}
    if "close" not in cols:
        raise ValueError(f"the bars have no close column: {list(df.columns)}")
    return cols


def used_bar_start(n_bars: int, profile: str, case: Optional["Case"] = None) -> int:
    """The first bar of the file that the run's windows use. The trainer builds windows only over the newest
    MAX_SEQUENCE_COUNT sequences (NT-177), so the first used bar is ``n - lookback - max horizon + 1 - MAX``
    (Config defaults for the lookback and the horizons: no data case changes them)."""
    from neural_trade.core.config import Config

    cfg = Config()
    mx = {**PROFILES[profile], **((case.layout.get(profile, {})) if case else {})}.get("MAX_SEQUENCE_COUNT",
                                                                                   cfg.MAX_SEQUENCE_COUNT)
    return max(0, int(n_bars) - int(cfg.LOOKBACK) - int(max(cfg.HORIZON_STEPS)) + 1 - int(mx))


def transform_bars(df, spec: Mapping[str, Any]):
    """A copy of the OHLCV frame ``df`` with the transform ``spec`` applied to the price columns (the volume and
    the timestamps stay); ``kind``: scale (``k``), vol (``k``), constant (``bars``), jumps. ``start`` (default 0)
    is the first bar the run's windows use (:func:`used_bar_start`): constant blocks, spikes and level jumps are
    placed at offsets from it, inside the first ~150 bars, which every profile's TRAINING block covers, so they
    are seen by the training windows and the loss (a transform elsewhere in a 43,500-bar file is invisible to a
    run that trains on the newest windows)."""
    out = df.copy()
    cols = _columns(out)
    kind = spec["kind"]
    close = out[cols["close"]].to_numpy(dtype=np.float64)
    n = len(out)
    t0 = int(spec.get("start", 0))
    flat = None
    if kind == "scale":
        ratio = np.full(n, float(spec["k"]))
    elif kind == "vol":
        new = close[0] * np.exp(float(spec["k"]) * (np.log(close) - np.log(close[0])))
        ratio = new / close
    elif kind == "constant":
        a, bars = t0 + 10, min(int(spec.get("bars", 100)), n - t0 - 10)
        ratio = np.ones(n)
        flat = (a, a + bars)
    elif kind == "jumps":
        ratio = np.ones(n)
        ratio[t0 + 60:] *= 2.0
        for off, f in zip((15, 45, 75, 105, 125), (4.0, 0.25, 4.0, 0.25, 4.0)):
            if t0 + off < n:
                ratio[t0 + off] *= f
    else:
        raise ValueError(f"unknown data transform {kind!r}")
    for col in cols.values():
        out[col] = out[col].to_numpy(dtype=np.float64) * ratio
    if flat is not None:                        # a constant block: open = high = low = close
        for col in cols.values():
            out.loc[out.index[flat[0]:flat[1]], col] = float(close[flat[0]])
    return out


# ------------------------------------------------------------------ fault injection
@contextlib.contextmanager
def fault_context(fault: Optional[Mapping[str, Any]]) -> Iterator[None]:
    """Patch the code so the case's fault happens in training (nothing is changed for ``fault=None``)."""
    if not fault:
        yield
        return
    from unittest import mock

    import tensorflow as tf

    import neural_trade.losses.functions as lf
    from neural_trade.data.processor import DataProcessor

    kind = fault["kind"]
    if kind == "nan_term":
        fn = FAULT_TERMS[fault["term"]]
        patcher = mock.patch.object(lf, fn, lambda *a, **k: tf.constant(float("nan"), dtype=tf.float32))
    elif kind == "nan_gradient":
        @tf.custom_gradient
        def _nan_gradient_identity(x):
            def grad(dy):
                return dy * float("nan")
            return x, grad

        orig = lf.point_huber

        def wrapped(model, y_true_scaled, y_pred_scaled, last_close_scaled=None, delta=None):
            return _nan_gradient_identity(orig(model, y_true_scaled, y_pred_scaled,
                                               last_close_scaled=last_close_scaled, delta=delta))

        patcher = mock.patch.object(lf, "point_huber", wrapped)
    elif kind == "nan_input":
        orig_prepare = DataProcessor.prepare_datasets

        def prepare(self, *a, **k):
            out = list(orig_prepare(self, *a, **k))
            x = np.array(out[0], copy=True)
            x[0, 0] = np.nan
            out[0] = x
            return tuple(out)

        patcher = mock.patch.object(DataProcessor, "prepare_datasets", prepare)
    else:
        raise ValueError(f"unknown fault {kind!r}")
    with patcher:
        yield


class HarnessTrainer:
    """The engine trainer of the harness: applies the case's fault, then the engine's own ``train_cell`` (which
    adds the stability guard in strict mode). The case is found from the cell's configuration name."""

    def __init__(self, cases: Sequence[Case]):
        self.cases = case_by_id(cases)

    def __call__(self, ctx, *, calibrate, save_artifacts):
        from neural_trade.experiments.runner import train_cell

        meta = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))
        case = self.cases[meta["engine"]["configuration"]]
        with fault_context(case.fault):
            return train_cell(ctx, calibrate=calibrate, save_artifacts=save_artifacts)


# ------------------------------------------------------------------ the scenario
def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_scenario(cases: Sequence[Case], *, name: str, profile: str, seeds: Sequence[int],
                   case_csv: Mapping[str, str], folds: Sequence[int] = tuple(PROFILE_FOLDS),
                   probe: Optional[str] = None) -> Dict[str, Any]:
    """The engine scenario spec (a dict for ``Scenario.from_dict``): one variant per runnable case. ``probe`` is
    a probe mode (:data:`PROBE_MODES`; None: the profile's default)."""
    mode = probe or PROFILE_PROBE[profile]
    variants = {}
    for c in cases:
        if not c.runnable:
            continue
        v = dict(c.overrides)
        v.update(c.layout.get(profile, {}))
        if c.id in case_csv:
            v["CSV_PATH"] = case_csv[c.id]
        variants[c.id] = v
    return {"schema_version": 1, "name": name,
            "description": f"stability harness ({profile} profile, NT-038): strict mode, probe {mode}, "
                           f"{len(variants)} cases x {len(seeds)} seeds",
            "overrides": {**COMMON_OVERRIDES, **PROFILES[profile], **probe_overrides(profile, mode)}, "variants": variants, "folds": list(folds),
            "seeds": [int(s) for s in seeds], "strategy": {"name": "calibrated_quantile", "params": {}},
            "backtest": {"random_seeds": 5}, "run": {"calibrate": True, "save_artifacts": False}}


def write_case_data(cases: Sequence[Case], csv, out_dir, profile: str = "tiny") -> Dict[str, str]:
    """``<out_dir>/data/<case>.csv`` for every runnable case with a data transform; {case id: path}."""
    import pandas as pd

    todo = [c for c in cases if c.runnable and c.data]
    if not todo:
        return {}
    from neural_trade.data.loaders import resolve_data_path
    base = pd.read_csv(resolve_data_path(csv))          # like the loader: a relative path also resolves from the project root
    d = Path(out_dir) / "data"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in todo:
        path = d / f"{c.id}.csv"
        spec = {**c.data, "start": used_bar_start(len(base), profile, c)}
        transform_bars(base, spec).to_csv(path, index=False)
        out[c.id] = str(path)
    return out


# ------------------------------------------------------------------ verdicts
@dataclass
class Check:
    name: str
    value: Optional[float]
    limit: Any
    passed: bool
    detail: str = ""
    evaluated: bool = True
    report_only: bool = False        # computed and shown, never fails the cell (thresholds file `report_only`)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "limit": self.limit, "passed": self.passed,
                "detail": self.detail, "evaluated": self.evaluated, "report_only": self.report_only}


@dataclass
class Verdict:
    case: str
    run_id: str
    cell_key: str
    seed: int
    passed: bool
    checks: List[Check]
    blamed: List[str] = field(default_factory=list)
    status: str = "done"
    error: str = ""
    thresholds_sha256: str = ""
    non_verdict: bool = False        # a resource error or a crash: no verdict, excluded from the counts (NT-191)
    kind: str = "primary"            # primary | probe_rerun | retry
    probe: bool = False              # the per-term probe was on in this run
    rerun_of: str = ""               # probe_rerun: the run id of the failed first run
    retry_of: str = ""               # retry: the run id of the non-verdict run it replaces
    blame_source: str = ""           # where `blamed` came from: the error text, masked-term counters, the probe sample
    blame_reason: str = ""           # when nothing is blamed: why

    def to_dict(self) -> Dict[str, Any]:
        return {"case": self.case, "run_id": self.run_id, "cell_key": self.cell_key, "seed": self.seed,
                "passed": self.passed, "blamed": self.blamed, "status": self.status, "error": self.error,
                "thresholds_sha256": self.thresholds_sha256, "non_verdict": self.non_verdict, "kind": self.kind,
                "probe": self.probe, "rerun_of": self.rerun_of, "retry_of": self.retry_of,
                "blame_source": self.blame_source, "blame_reason": self.blame_reason,
                "checks": [c.to_dict() for c in self.checks]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Verdict":
        checks = [Check(c["name"], c.get("value"), c.get("limit"), bool(c["passed"]), c.get("detail", ""),
                        bool(c.get("evaluated", True)), bool(c.get("report_only", False))) for c in d.get("checks", [])]
        return cls(d["case"], d["run_id"], d.get("cell_key", ""), int(d.get("seed", -1)), bool(d["passed"]), checks,
                   list(d.get("blamed") or []), d.get("status", "done"), d.get("error", ""),
                   d.get("thresholds_sha256", ""), bool(d.get("non_verdict", False)), d.get("kind", "primary"),
                   bool(d.get("probe", False)), d.get("rerun_of", ""), d.get("retry_of", ""),
                   d.get("blame_source", ""), d.get("blame_reason", ""))

    @property
    def failed_checks(self) -> List[Check]:
        return [c for c in self.checks if not c.passed and not c.report_only]


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _blamed_from_message(message: str) -> List[str]:
    m = re.search(r"blamed: (.*?);", message or "")
    return re.findall(r"([a-z][a-z_0-9]*) \(", m.group(1)) if m else []


def _probe_shares(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """{term_group: mean share over the epochs} from the probe_grad_share_* keys; None when any is non-finite."""
    series: Dict[str, List[Any]] = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith("probe_grad_share_"):
                series.setdefault(k[len("probe_grad_share_"):], []).append(v)
    return {k: (float(np.mean([float(x) for x in vs])) if all(_finite(x) for x in vs) else None)
            for k, vs in series.items()}


def probe_blame(rows: Sequence[Mapping[str, Any]]):
    """(term, group, share, epoch) of the largest probe share in the FIRST epoch whose finite shares sum to 1 in a
    variable group (an epoch the probe did not sample logs 0 everywhere); None when no epoch qualifies. One batch's
    gradient split, not an epoch average: the REPORT labels it :data:`PROBE_SOURCE`."""
    for r in rows:
        groups: Dict[str, Dict[str, float]] = {}
        broken = set()                      # a group with a non-finite share has no usable sample
        for k, v in r.items():
            m = re.fullmatch(r"probe_grad_share_(.+)_(trunk|head|indicator)", k)
            if m and _finite(v):
                groups.setdefault(m.group(2), {})[m.group(1)] = float(v)
            elif m:
                broken.add(m.group(2))
        valid = {g: sh for g, sh in groups.items() if g not in broken and abs(sum(sh.values()) - 1.0) < 1e-3}
        if valid:
            share, group, term = max((v, g, t) for g, sh in valid.items() for t, v in sh.items())
            return term, group, share, r.get("epoch")
    return None


def _verdict_passed(checks: Sequence[Check]) -> bool:
    return all(c.passed or c.report_only for c in checks)


def evaluate_run(run_dir, case: Case, thresholds: Thresholds) -> Verdict:
    """Judge one finished cell against the pre-registered thresholds (nothing is read but the run directory)."""
    from neural_trade.core.config import Config
    from neural_trade.core.guard import regions_disabled
    from neural_trade.evaluation.report import health_block
    from neural_trade.experiments.sweep import run_health
    from neural_trade.telemetry.epoch_logger import read_metrics

    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    eng = meta.get("engine") or {}
    result = json.loads((d / "result.json").read_text(encoding="utf-8")) if (d / "result.json").is_file() else {}
    status = result.get("status") or "incomplete"
    err = result.get("error") or {}
    message = f"{err.get('type', '')}: {err.get('message', '')}" if err else ""
    blamed = _blamed_from_message(err.get("message", ""))
    checks: List[Check] = []
    T = thresholds
    only = set(T.report_only)

    def add(name, value, limit, passed, detail="", evaluated=True):
        checks.append(Check(name, value, limit, bool(passed), detail, evaluated, name in only))

    source = {"v": "the error text" if blamed else ""}

    def verdict():
        return Verdict(case.id, meta.get("run_id", d.name), eng.get("cell_key", ""), int(meta.get("seed", -1)),
                       _verdict_passed(checks), checks, blamed, status, message, T.sha256, blame_source=source["v"])

    if status == "incomplete" or (status == "failed" and is_non_verdict_error(err.get("type"), err.get("message", ""),
                                                                            winerror=err.get("winerror"))):
        crash = "worker crash: the run directory has no result.json" if status == "incomplete" else message
        return Verdict(case.id, meta.get("run_id", d.name), eng.get("cell_key", ""), int(meta.get("seed", -1)), False,
                       [], [], status, crash, T.sha256, non_verdict=True,
                       blame_reason="not a verdict: the run did not finish for a resource reason")

    if case.expect == "detect":
        fd = T.fault_detection
        stopped = status == "failed" and err.get("type") == fd.get("error_type")
        add("fault_stopped_run", 1.0 if stopped else 0.0, fd.get("error_type"), stopped,
            "" if stopped else f"the run ended {status}" + (f" with {message}" if message else ""))
        if fd.get("names_the_term") and case.expects_term:
            named = case.expects_term in (err.get("message") or "")
            add("fault_names_term", 1.0 if named else 0.0, case.expects_term, named,
                "" if named else f"the error does not name {case.expects_term}: {message}")
        return verdict()

    completed = status == "done"
    add("run_completed", 1.0 if completed else 0.0, "done", completed, "" if completed else f"{status}: {message}")
    rows = read_metrics(d / "metrics.jsonl") if (d / "metrics.jsonl").is_file() else []
    if rows:
        with regions_disabled():     # a re-test of a configuration inside a known failing region
            cfg = Config.from_yaml(d / "config.yaml")
        why = run_health(d, max_nonfinite_grad_steps=10 ** 9)
        add("loss_finite", 0.0 if why else 1.0, True, not why, why or "")
        losses = [float(r[k]) for r in rows for k in ("loss", "val_loss") if _finite(r.get(k))]
        lim = float(T.check("max_abs_loss"))
        top = max((abs(v) for v in losses), default=None)
        add("max_abs_loss", top, lim, top is None or top <= lim, "" if top is not None else "no finite loss logged",
            top is not None)
        train = [float(r["loss"]) for r in rows if _finite(r.get("loss"))]
        lim = float(T.check("max_loss_over_first"))
        if len(train) >= 2 and train[0] != 0:
            ratio = max(train) / abs(train[0])
            add("loss_over_first", ratio, lim, ratio <= lim)
        else:
            add("loss_over_first", None, lim, True, "fewer than 2 epochs (or a zero first loss)", False)
        n_steps = sum(float(r.get("n_steps") or 0) for r in rows)
        nonfinite = sum(float(r.get("nonfinite_grad_steps") or 0) for r in rows)
        lim = float(T.check("max_nonfinite_step_rate"))
        if n_steps:
            rate = nonfinite / n_steps
            add("nonfinite_step_rate", rate, lim, rate <= lim)
        else:
            add("nonfinite_step_rate", None, lim, True, "n_steps was not logged (a run before NT-037)", False)
        h = health_block(rows, cfg)
        lim = float(T.check("max_clipped_share"))
        for grp in ("main", "indicator"):
            clipped = h.get(f"grad_clip_steps_{grp}_total")
            share = (float(clipped) / n_steps) if clipped is not None and n_steps else None
            add(f"clipped_share_{grp}", share, lim, share is None or share <= lim,
                "" if share is not None else "not logged", share is not None)
        shares = _probe_shares(rows)
        lim = float(T.check("max_term_gradient_share"))
        if shares:
            bad = sorted(k for k, v in shares.items() if v is None)
            top_k = max((k for k, v in shares.items() if v is not None), key=lambda k: shares[k], default=None)
            top_v = shares[top_k] if top_k else None
            ok = not bad and (top_v is None or top_v <= lim)
            detail = (f"non-finite probe share(s): {bad[:4]}" if bad else
                      (f"largest: {top_k} = {top_v:.3f}" if top_k else ""))
            add("term_gradient_share", top_v, lim, ok, detail)
            if not ok and top_k and not blamed:
                blamed, source["v"] = [top_k.rsplit("_", 1)[0]], PROBE_SOURCE
        else:
            add("term_gradient_share", None, lim, True, "the probe was off", False)
        lim = float(T.check("max_var_at_floor_share"))
        worst = None
        for r in rows:
            for k, at_floor in r.items():
                m = re.fullmatch(r"var_at_floor_(h\d+)", k)
                n_dir = r.get(f"dir_n_{m.group(1)}") if m else None
                if n_dir and at_floor is not None:
                    worst = max(worst or 0.0, float(at_floor) / float(n_dir))
        add("var_at_floor_share", worst, lim, worst is None or worst <= lim,
            "" if worst is not None else "not logged", worst is not None)
        at_bound = h.get("periods_at_bound") or {}
        lim = int(T.check("max_periods_at_bound"))
        add("periods_at_bound", float(len(at_bound)), lim, len(at_bound) <= lim, ", ".join(sorted(at_bound)[:4]))
        masked = h.get("masked_terms_total") or {}
        total_masked = float(sum(masked.values()))
        lim = float(T.check("max_masked_term_steps"))
        add("masked_term_steps", total_masked, lim, total_masked <= lim,
            ", ".join(f"{k[len('masked_'):]}={v:g}" for k, v in sorted(masked.items())[:4]))
        if masked and not blamed:
            from neural_trade.training.stability_guard import blame

            blamed, source["v"] = blame({k[len("masked_"):]: v for k, v in masked.items()}), "masked-term counters"
    _score_checks(result.get("scores") or {}, T, add)
    return verdict()


def _score_checks(scores: Mapping[str, Any], T: Thresholds, add) -> None:
    """The variance-head and coverage checks, from the scored block's numbers (result.json ``scores``)."""
    if T.schema_version >= 2:
        _variance_checks_v2(scores, T, add)
        _coverage_check(scores, T, add)
        return
    hs = sorted({m.group(1) for k in scores if (m := re.match(r"(h\d+)/variance/", k))})
    limits = {"nll": float(T.check("max_variance_nll")), "nll_c": float(T.check("max_variance_nll_over_const")),
              "crps_c": float(T.check("max_variance_crps_over_const"))}

    def val(k):
        v = scores.get(k)
        return float(v) if _finite(v) else None

    worst: Dict[str, Optional[float]] = {"nll": None, "nll_c": None, "crps_c": None}
    where = {"nll": "", "nll_c": "", "crps_c": ""}
    for h in hs:
        nll, crps = val(f"{h}/variance/nll"), val(f"{h}/variance/crps")
        nll0, crps0 = val(f"baseline/const_var/{h}/variance/nll"), val(f"baseline/const_var/{h}/variance/crps")
        cand = {"nll": nll, "nll_c": None if nll is None or nll0 is None else nll - nll0,
                "crps_c": None if not crps or not crps0 else crps / crps0}
        for k, raw in (("nll", scores.get(f"{h}/variance/nll")), ("crps_c", scores.get(f"{h}/variance/crps"))):
            if raw is not None and not _finite(raw):          # a non-finite head number is a broken head
                cand[k] = math.inf
        for k, v in cand.items():
            if v is not None and (worst[k] is None or v > worst[k]):
                worst[k], where[k] = v, h
    for name, key in (("variance_nll", "nll"), ("variance_nll_over_const", "nll_c"),
                      ("variance_crps_over_const", "crps_c")):
        v = worst[key]
        add(name, v, limits[key], v is None or v <= limits[key],
            f"worst horizon: {where[key]}" if v is not None else "the run was not scored", v is not None)
    _coverage_check(scores, T, add)


def _coverage_check(scores: Mapping[str, Any], T: Thresholds, add) -> None:
    # coverage: only on a horizon with enough effective samples (D-012)
    def val(k):
        v = scores.get(k)
        return float(v) if _finite(v) else None

    min_n = float(T.check("min_n_eff"))
    lim = float(T.check("min_coverage90"))
    cov: Dict[str, float] = {}
    skipped: List[str] = []
    for k, v in scores.items():
        m = re.fullmatch(r"(h\d+)/variance/coverage90", k)
        if not m or not _finite(v):
            continue
        n_eff = val(f"{m.group(1)}/n_eff")
        if n_eff is None or n_eff < min_n:
            skipped.append(f"{m.group(1)} (n_eff {n_eff:g})" if n_eff is not None else m.group(1))
            continue
        cov[k] = float(v)
    if cov:
        lo = min(cov, key=lambda k: cov[k])
        add("coverage90", cov[lo], lim, cov[lo] >= lim,
            f"lowest: {lo}" + (f"; not evaluated: {skipped}" if skipped else ""))
    else:
        add("coverage90", None, lim, True,
            f"not evaluated: n_eff below {min_n:g} on {skipped}" if skipped else "the run was not scored", False)


def _variance_checks_v2(scores: Mapping[str, Any], T: Thresholds, add) -> None:
    """Thresholds v2 (NT-187): the variance-head checks with n_eff gates, a degenerate-baseline guard and the NLL in
    scaled units. A horizon contributes to a check only when its n_eff reaches the check's threshold; a check no
    horizon contributes to is "not evaluated" and says why."""
    hs = sorted({m.group(1) for k in scores if (m := re.match(r"(h\d+)/variance/", k))})
    lim_nll_s, lim_nll_c, lim_crps_c = (float(T.check(k)) for k in
                                         ("max_variance_nll_scaled", "max_variance_nll_over_const",
                                          "max_variance_crps_over_const"))
    min_excess, min_scale = float(T.check("min_n_eff_variance_excess")), float(T.check("min_n_eff_variance"))
    max_base = float(T.check("max_const_baseline_nll_scaled"))

    def val(k):
        v = scores.get(k)
        return float(v) if _finite(v) else None

    worst: Dict[str, Optional[float]] = {"nll_s": None, "nll_c": None, "crps_c": None}
    where = {k: "" for k in worst}
    skipped: Dict[str, List[str]] = {k: [] for k in worst}
    for h in hs:
        n_eff = val(f"{h}/n_eff")
        nll, crps = scores.get(f"{h}/variance/nll"), scores.get(f"{h}/variance/crps")
        nll0, crps0 = val(f"baseline/const_var/{h}/variance/nll"), val(f"baseline/const_var/{h}/variance/crps")
        rms = val(f"{h}/delta/rmse_zero")
        scale = math.log(rms) if rms is not None and rms > 0 else None
        base_ok = nll0 is not None and crps0 is not None and crps0 > 0
        base_why = "" if base_ok else f"{h}: the constant baseline is not finite"
        if base_ok and scale is not None and nll0 - scale > max_base:
            base_ok, base_why = False, f"{h}: the constant baseline NLL is absurd ({nll0 - scale:.4g} scaled, bound {max_base:g})"
        gates = {"nll_s": min_scale, "nll_c": min_excess, "crps_c": min_scale}
        why_not = {"nll_s": f"{h}: no price-change scale (rmse_zero)", "nll_c": base_why, "crps_c": base_why}
        cand: Dict[str, Optional[float]] = {}
        if scale is not None and nll is not None:
            cand["nll_s"] = (float(nll) - scale) if _finite(nll) else math.inf   # a non-finite head is a broken head
        if base_ok:
            if nll is not None:
                cand["nll_c"] = (float(nll) - nll0) if _finite(nll) else math.inf
            if crps is not None:
                cand["crps_c"] = (float(crps) / crps0) if _finite(crps) else math.inf
        for k, min_n in gates.items():
            if k not in cand:
                if nll is not None or crps is not None:
                    skipped[k].append(why_not[k] or f"{h}: not scored")
                continue
            if n_eff is None or n_eff < min_n:
                skipped[k].append(f"{h} (n_eff {'missing' if n_eff is None else f'{n_eff:g}'} < {min_n:g})")
                continue
            if worst[k] is None or cand[k] > worst[k]:
                worst[k], where[k] = cand[k], h
    for name, key, lim in (("variance_nll", "nll_s", lim_nll_s), ("variance_nll_over_const", "nll_c", lim_nll_c),
                           ("variance_crps_over_const", "crps_c", lim_crps_c)):
        v = worst[key]
        note = ("; not evaluated: " + ", ".join(skipped[key])) if skipped[key] else ""
        if v is not None:
            add(name, v, lim, v <= lim, f"worst horizon: {where[key]}" + note)
        else:
            add(name, None, lim, True, ("not evaluated: " + ", ".join(skipped[key])) if skipped[key]
                else "the run was not scored", False)


def write_verdict(run_dir, verdict: Verdict) -> Path:
    p = Path(run_dir) / VERDICT_FILE
    p.write_text(json.dumps(verdict.to_dict(), indent=2, default=str), encoding="utf-8", newline="\n")
    return p


# ------------------------------------------------------------------ failing regions
def region_for_case(case: Case, report: str, reason: str):
    """The failing region of a configuration case: its overrides as exact conditions (an exact point)."""
    from neural_trade.core.guard import Region

    conds: Dict[str, Dict[str, Any]] = {}
    for k, v in case.overrides.items():
        if isinstance(v, (bool, list, tuple, str)):
            conds[k] = {"values": [list(v) if isinstance(v, tuple) else v]}
        elif isinstance(v, (int, float)):
            conds[k] = {"min": v, "max": v}
    return Region(f"stability:{case.id}", conds, case.id, report, reason)


# ------------------------------------------------------------------ the run
@dataclass
class HarnessResult:
    harness_id: str
    out_dir: Path
    scenario: str
    thresholds_sha256: str
    verdicts: List[Verdict]                  # the verdict of every (case, seed): the first run, or the retry of a crash
    not_run: List[Case]
    case_passed: Dict[str, bool]             # True only for PASS (a FAIL and a NOT A VERDICT are both False)
    regions: List[Any]
    report: Path
    n_eff: Dict[str, Dict[str, float]] = field(default_factory=dict)    # case -> {horizon: n_eff} of its scored block
    case_status: Dict[str, str] = field(default_factory=dict)           # case -> PASS | FAIL | NOT A VERDICT
    reruns: List[Verdict] = field(default_factory=list)                 # the probe re-runs of failed cells (NT-191)
    superseded: List[Verdict] = field(default_factory=list)             # non-verdict runs a retry replaced
    not_rerun: List[Verdict] = field(default_factory=list)              # failed cells left without a probe re-run (cap)
    probe_mode: str = "off"
    retry_of: str = ""                       # the launch whose non-verdict cells this launch re-ran

    @property
    def passed(self) -> bool:
        return all(self.case_passed.values())

    @property
    def verdict_failed(self) -> bool:
        return any(s == "FAIL" for s in self.case_status.values())

    @property
    def non_verdict_cells(self) -> int:
        return sum(v.non_verdict for v in self.verdicts)

    @property
    def exit_code(self) -> int:
        """1 for a verdict failure, 2 when cells that are not a verdict are left, else 0 (NT-191)."""
        return 1 if self.verdict_failed else (2 if self.non_verdict_cells else 0)


def plan_cases(*, profile: str = "tiny", csv=None, case_ids: Optional[Sequence[str]] = None,
               seeds: Sequence[int] = (0,), work_dir=None, probe: Optional[str] = None):
    """A dry plan: every runnable case's cells through the engine's planner (the spec, every Config, the components
    and the data layout of the profile), nothing trained and no run directory written. ``work_dir`` receives the
    case data files. Returns the planned cells; an unplannable case raises ScenarioError naming its cell."""
    import tempfile

    from neural_trade.core.config import Config
    from neural_trade.core.guard import regions_disabled
    from neural_trade.experiments.runner import Runner
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.store import RunStore

    cases = [c for c in default_cases() if c.runnable and (not case_ids or c.id in set(case_ids))]
    csv = csv if csv is not None else Config().CSV_PATH
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(work_dir) if work_dir is not None else Path(tmp)
        case_csv = write_case_data(cases, csv, out, profile)
        spec = build_scenario(cases, name="stab-plan", profile=profile, seeds=list(seeds),
                              case_csv={**{c.id: str(csv) for c in cases}, **case_csv}, probe=probe)
        with regions_disabled():
            return Runner(Scenario.from_dict(spec), RunStore(Path(tmp) / "store")).plan()


def expected_n_eff_for(T: Thresholds, profile: str, case_id: str) -> Optional[List[int]]:
    """The thresholds file's expected n_eff per horizon for a case (v2's table); None when the file has none."""
    table = (T.expected_n_eff or {}).get(profile)
    if not table:
        return None
    if isinstance(table.get(case_id), Mapping):
        return list(table[case_id]["n_eff"])
    return list(table["default_cases"]) if "default_cases" in table else None


def dry_run(*, profile: Optional[str] = None, csv=None, case_ids: Optional[Sequence[str]] = None,
            seeds: Optional[Sequence[int]] = None, thresholds_path=None, probe: Optional[str] = None) -> Dict[str, Any]:
    """What a real run would do, per cell, without training (NT-191): the case, the seed, n_eff per horizon (the
    planner's, beside the thresholds file's expected table), the probe mode, the steps per epoch and the epochs."""
    profile = profile or "tiny"
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {profile!r}")
    mode = probe or PROFILE_PROBE[profile]
    probe_overrides(profile, mode)                      # validates the mode
    T = load_thresholds(resolve_thresholds_path(thresholds_path, profile))
    unknown = sorted(set(case_ids or ()) - {c.id for c in default_cases()})
    if unknown:
        raise ValueError(f"unknown case(s) {unknown}; known: {[c.id for c in default_cases()]}")
    seed_list = [int(s) for s in seeds] if seeds else list(range(T.seeds))
    planned = plan_cases(profile=profile, csv=csv, case_ids=case_ids, seeds=seed_list, probe=mode)
    cells = []
    for pc in planned:
        cfg, case = pc.config, pc.cell.configuration.name
        horizons = [int(h) for h in cfg.HORIZON_STEPS]
        n_test, n_train = int(pc.fold["blocks"]["test"]["n"]), int(pc.fold["blocks"]["train"]["n"])
        n_eff = {f"h{i}": n_test // h for i, h in enumerate(horizons)}
        expected = expected_n_eff_for(T, profile, case)
        steps = -(-n_train // int(cfg.BATCH_SIZE))
        cells.append({"case": case, "seed": int(pc.cell.seed), "cell": pc.key, "fold": pc.cell.fold,
                      "n_eff": n_eff, "expected_n_eff": expected,
                      "n_eff_matches_expected": None if expected is None else list(n_eff.values()) == expected,
                      "probe": mode, "probe_rerun": "on a failing cell, PROBE_EVERY 1" if mode == "failed" else None,
                      "train_windows": n_train, "batch_size": int(cfg.BATCH_SIZE), "epochs": int(cfg.EPOCHS),
                      "steps_per_epoch": steps, "steps": steps * int(cfg.EPOCHS)})
    return {"profile": profile, "probe": mode, "thresholds": T.path.name, "thresholds_sha256": T.sha256,
            "seeds": seed_list, "n_cells": len(cells), "cells": cells}


# ------------------------------------------------------------------ the run: helpers
def _run_spec(spec: Mapping[str, Any], st, trainer, names: List[str]):
    """Run one scenario spec into the store; returns its index rows."""
    from neural_trade.core.guard import regions_disabled
    from neural_trade.experiments.runner import Runner
    from neural_trade.experiments.scenario import Scenario

    scenario = Scenario.from_dict(dict(spec))
    names.append(scenario.name)
    with regions_disabled():     # the harness re-tests configurations inside known failing regions
        Runner(scenario, st, trainer=trainer).run()
    return st.sync(scenario.name)


def _cell_spec(base: Mapping[str, Any], case_id: str, seed: int, name: str, extra: Mapping[str, Any]) -> Dict[str, Any]:
    """The base scenario narrowed to one (case, seed) cell, with extra overrides (a probe re-run, a retry)."""
    spec = json.loads(json.dumps(base))
    spec["name"] = name
    spec["variants"] = {case_id: spec["variants"][case_id]}
    spec["seeds"] = [int(seed)]
    spec["overrides"] = {**spec["overrides"], **extra}
    return spec


def _judge(rows, by_id: Mapping[str, Case], st, T: Thresholds, *, kind: str, probe: bool,
           n_eff: Dict[str, Dict[str, float]], dirs: Dict[str, Path]) -> List[Verdict]:
    """Judge every row of a launch (a run directory without a result is a worker crash: not a verdict) and write
    each verdict into its run directory."""
    out: List[Verdict] = []
    for r in rows:
        case = by_id.get(r["configuration"])
        if case is None:
            continue
        run_dir = st.root / r["run_dir"]
        if (run_dir / "result.json").is_file() and case.id not in n_eff:
            sc = (json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("scores") or {})
            found = {k.split("/")[0]: float(v) for k, v in sc.items() if re.fullmatch(r"h\d+/n_eff", k) and _finite(v)}
            if found:
                n_eff[case.id] = dict(sorted(found.items()))
        v = evaluate_run(run_dir, case, T)
        v.kind, v.probe = kind, probe
        write_verdict(run_dir, v)
        dirs[v.run_id] = run_dir
        out.append(v)
    return out


def _attribute(v: Verdict, run_dir: Optional[Path], *, mode: str, reran: bool, rerun_error: str = "") -> None:
    """Fill the blame of a failed cell that the run itself did not name: from the probe sample of ``run_dir`` (the
    probe re-run, or the run itself in mode ``on``); else the reason there is none."""
    from neural_trade.telemetry.epoch_logger import read_metrics

    if v.blamed:
        return
    sample = None
    if run_dir is not None and (Path(run_dir) / "metrics.jsonl").is_file():
        sample = probe_blame(read_metrics(Path(run_dir) / "metrics.jsonl"))
    if sample:
        term, group, share, epoch = sample
        v.blamed, v.blame_source = [term], PROBE_SOURCE
        v.blame_reason = f"{group} group, {share:.2f} of its gradient norm, epoch {epoch}"
    elif mode == "capped":
        v.blame_reason = v.blame_reason or NOT_RERUN_CAP
    elif reran and rerun_error:
        v.blame_reason = f"the probe re-run crashed ({rerun_error}), so there is no probe sample"
    elif mode == "off":
        v.blame_reason = "probe off in this run (use --probe failed or on for a probe sample)"
    elif reran:
        v.blame_reason = "the probe re-run logged no epoch whose probe shares sum to 1"
    else:
        v.blame_reason = "the probe logged no epoch whose probe shares sum to 1"


def _case_status(vs: Sequence[Verdict], n_seeds: int) -> str:
    if any(not v.passed and not v.non_verdict for v in vs) or len(vs) != n_seeds:
        return "FAIL"
    return NOT_A_VERDICT if any(v.non_verdict for v in vs) else "PASS"


def _load_origin(base: Path, ident: str) -> Dict[str, Any]:
    """The verdicts.json and scenario.json of an earlier launch (``latest``: the newest one)."""
    if ident == "latest":
        found = sorted(d.name for d in base.iterdir() if (d / "verdicts.json").is_file()) if base.is_dir() else []
        ident = found[-1] if found else ident
    d = base / ident
    if not (d / "verdicts.json").is_file() or not (d / "scenario.json").is_file():
        raise ValueError(f"no such harness launch {ident!r} under {base}")
    doc = json.loads((d / "verdicts.json").read_text(encoding="utf-8"))
    if "profile" not in doc or "probe_mode" not in doc:
        raise ValueError(f"{d}: written before NT-191 (no profile or probe mode); run the harness again")
    doc["spec"] = json.loads((d / "scenario.json").read_text(encoding="utf-8"))
    doc["id"] = ident
    return doc


def run_harness(*, profile: Optional[str] = "tiny", csv=None, store="runs", case_ids: Optional[Sequence[str]] = None,
                seeds: Optional[Sequence[int]] = None, thresholds_path=None, harness_id: Optional[str] = None,
                trainer=None, out_root=None, probe: Optional[str] = None,
                retry_non_verdict_of: Optional[str] = None, max_probe_reruns: int = MAX_PROBE_RERUNS) -> HarnessResult:
    """Run the harness: write the data, run the cases as an engine scenario into ``store``, judge every cell,
    write the report. ``trainer`` replaces the engine trainer (tests). ``probe``: off | on | failed (None: the
    profile's default, :data:`PROFILE_PROBE`). ``retry_non_verdict_of``: the id of an earlier launch (or ``latest``)
    whose not-a-verdict cells are re-run as a new launch; its other verdicts are carried over unchanged.
    ``max_probe_reruns``: in mode ``failed`` at most this many failed cells are re-run with the probe in one launch;
    the others are listed in the report as not re-run (cap)."""
    if max_probe_reruns < 0:
        raise ValueError(f"max_probe_reruns must be >= 0, got {max_probe_reruns}")
    from neural_trade.core.config import Config
    from neural_trade.experiments.store import RunStore

    st = store if isinstance(store, RunStore) else RunStore(store)
    base_out = Path(out_root if out_root is not None else st.root) / STABILITY_DIR
    origin = _load_origin(base_out, retry_non_verdict_of) if retry_non_verdict_of else None
    if origin:
        if profile not in (None, origin["profile"]):
            raise ValueError(f"launch {origin['id']} ran the {origin['profile']!r} profile, not {profile!r}")
        if case_ids or seeds:
            raise ValueError("a retry takes the cases and the seeds of the launch it retries")
        if probe not in (None, origin["probe_mode"]):
            raise ValueError(f"launch {origin['id']} used --probe {origin['probe_mode']}; a retry keeps it")
        profile, mode = origin["profile"], origin["probe_mode"]
        if thresholds_path is None:
            thresholds_path = origin.get("thresholds_file")
    else:
        profile = profile or "tiny"
        mode = probe or PROFILE_PROBE.get(profile, "off")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {profile!r}")
    probe_overrides(profile, mode)                           # validates the mode
    T = load_thresholds(resolve_thresholds_path(thresholds_path, profile))
    if origin and T.sha256 != origin["thresholds_sha256"]:
        raise ValueError(f"launch {origin['id']} was judged against thresholds sha256 {origin['thresholds_sha256']}; "
                         f"{T.path.name} is {T.sha256}: a retry keeps the pre-registered thresholds")
    all_cases = default_cases()
    if origin:
        case_ids = list(origin["case_ids"]) + list(origin.get("not_run_ids") or [])
    if case_ids:
        unknown = sorted(set(case_ids) - {c.id for c in all_cases})
        if unknown:
            raise ValueError(f"unknown case(s) {unknown}; known: {[c.id for c in all_cases]}")
        all_cases = [c for c in all_cases if c.id in set(case_ids)]
    seed_list = [int(s) for s in origin["seeds"]] if origin else (
        [int(s) for s in seeds] if seeds else list(range(T.seeds)))
    hid = harness_id or f"{_stamp()}-{profile}"
    if harness_id is None:       # two launches in one second (a retry right after a run) get distinct ids
        n = 1
        while (base_out / hid).exists():
            n += 1
            hid = f"{_stamp()}-{profile}-{n}"
    if origin and not any(d.get("non_verdict") for d in origin["verdicts"]):
        raise ValueError(f"launch {origin['id']} has nothing to retry: no cell is 'not a verdict'")
    if origin is None:
        from neural_trade.data.loaders import resolve_data_path

        csv = csv if csv is not None else Config().CSV_PATH
        if not resolve_data_path(csv).is_file():             # refused before any run directory exists
            raise ValueError(f"the bars file {csv} does not exist")
    out_dir = base_out / hid
    out_dir.mkdir(parents=True, exist_ok=False)
    runnable = [c for c in all_cases if c.runnable]
    not_run = [c for c in all_cases if not c.runnable]
    by_id = case_by_id(runnable)
    name = f"stab-{hid}"[:48]
    short = name[:40]
    n_eff: Dict[str, Dict[str, float]] = {c: dict(h) for c, h in (origin or {}).get("n_eff", {}).items()}
    dirs: Dict[str, Path] = {}
    ran_names: List[str] = []
    reruns: List[Verdict] = [Verdict.from_dict(d) for d in (origin or {}).get("reruns", [])]
    superseded: List[Verdict] = [Verdict.from_dict(d) for d in (origin or {}).get("superseded", [])]
    runner_trainer = trainer if trainer is not None else HarnessTrainer(runnable)
    if origin is None:
        case_csv = write_case_data(runnable, csv, out_dir, profile)
        base_csv = {c.id: str(csv) for c in runnable if c.id not in case_csv}
        spec = build_scenario(runnable, name=name, profile=profile, seeds=seed_list,
                              case_csv={**base_csv, **case_csv}, probe=mode)
        (out_dir / "scenario.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8", newline="\n")
        scenario_name = spec["name"]
        new = _judge(_run_spec(spec, st, runner_trainer, ran_names), by_id, st, T, kind="primary", probe=(mode == "on"),
                     n_eff=n_eff, dirs=dirs)
        verdicts = list(new)
    else:
        spec = origin["spec"]
        (out_dir / "scenario.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8", newline="\n")
        scenario_name = name
        carried = [Verdict.from_dict(d) for d in origin["verdicts"]]
        todo = [v for v in carried if v.non_verdict]
        if not todo:
            raise ValueError(f"launch {origin['id']} has nothing to retry: no cell is 'not a verdict'")
        verdicts, new = [v for v in carried if not v.non_verdict], []
        for i, old in enumerate(todo):
            sub = _cell_spec(spec, old.case, old.seed, f"{short}-r{i}", {})
            got = _judge(_run_spec(sub, st, runner_trainer, ran_names), by_id, st, T, kind="retry", probe=(mode == "on"),
                         n_eff=n_eff, dirs=dirs)
            for g in got:
                g.retry_of = old.run_id
                write_verdict(dirs[g.run_id], g)
            superseded.append(old)
            new += got
        verdicts += new
    failing = [v for v in new if not v.passed and not v.non_verdict]
    not_rerun: List[Verdict] = [Verdict.from_dict(d) for d in (origin or {}).get("not_rerun", [])]
    n_rerun = 0
    for v in failing:
        if mode == "failed" and n_rerun >= max_probe_reruns:
            v.blame_reason = f"{NOT_RERUN_CAP}: {max_probe_reruns} probe re-runs already in this launch"
            not_rerun.append(v)
            _attribute(v, None, mode="capped", reran=False)
        elif mode == "failed":                    # once, with the probe on every step, for the blame
            n_rerun += 1
            sub = _cell_spec(spec, v.case, v.seed, f"{short}-p{len(reruns)}", probe_overrides(profile, "rerun"))
            got = _judge(_run_spec(sub, st, runner_trainer, ran_names), by_id, st, T, kind="probe_rerun", probe=True, n_eff={},
                         dirs=dirs)
            for g in got:
                g.rerun_of = v.run_id
                write_verdict(dirs[g.run_id], g)
            reruns += got
            crash = got[0].error if got and got[0].non_verdict else ("" if got else "no run was produced")
            _attribute(v, dirs[got[0].run_id] if got and not got[0].non_verdict else None, mode=mode, reran=True,
                       rerun_error=crash)
        else:
            _attribute(v, dirs.get(v.run_id) if mode == "on" else None, mode=mode, reran=False)
        write_verdict(dirs[v.run_id], v)
    for sub_name in sorted(set(ran_names) | {scenario_name}):
        st.sync(sub_name)        # the index now carries the verdicts (stability/* scores)
    verdicts.sort(key=lambda v: (v.case, v.seed))
    case_status = {c.id: _case_status([v for v in verdicts if v.case == c.id], len(seed_list)) for c in runnable}
    case_passed = {c: s == "PASS" for c, s in case_status.items()}
    report_path = out_dir / "REPORT.md"
    regions = []
    for c in runnable:
        if c.region and case_status[c.id] == "FAIL":
            fails = [v for v in verdicts if v.case == c.id and not v.passed and not v.non_verdict]
            why = "; ".join(sorted({f"{k.name}" for v in fails for k in v.failed_checks})) or "no verdict"
            regions.append(region_for_case(c, str(report_path).replace("\\", "/"),
                                           f"the stability harness's case {c.id} failed: {why}"))
    from neural_trade.core.guard import write_regions

    write_regions(out_dir / "failing_regions.json", regions)
    (out_dir / "verdicts.json").write_text(json.dumps(
        {"harness_id": hid, "profile": profile, "probe_mode": mode, "seeds": seed_list, "retry_of": origin["id"] if origin else "",
         "thresholds_sha256": T.sha256, "thresholds_file": str(T.path),
         "case_ids": [c.id for c in runnable], "not_run_ids": [c.id for c in not_run], "n_eff": n_eff,
         "verdicts": [v.to_dict() for v in verdicts], "reruns": [v.to_dict() for v in reruns],
         "superseded": [v.to_dict() for v in superseded], "not_rerun": [v.to_dict() for v in not_rerun],
         "max_probe_reruns": max_probe_reruns, "case_passed": case_passed, "case_status": case_status},
        indent=2, default=str), encoding="utf-8", newline="\n")
    result = HarnessResult(hid, out_dir, scenario_name, T.sha256, verdicts, not_run, case_passed, regions, report_path,
                           n_eff, case_status, reruns, superseded, not_rerun, mode, origin["id"] if origin else "")
    report_path.write_text(render_report(result, T, runnable, profile, seed_list, st), encoding="utf-8", newline="\n")
    return result


def _blame_text(v: Verdict) -> str:
    if v.blamed:
        return f"{', '.join(v.blamed)} ({v.blame_source})"
    return f"- ({v.blame_reason})" if v.blame_reason else "-"


def render_report(res: HarnessResult, T: Thresholds, cases: Sequence[Case], profile: str, seeds: Sequence[int],
                  st) -> str:
    from neural_trade.utils.env import git_sha

    n_pass = sum(s == "PASS" for s in res.case_status.values())
    n_fail = sum(s == "FAIL" for s in res.case_status.values())
    n_nv = sum(s == NOT_A_VERDICT for s in res.case_status.values())
    if res.verdict_failed:
        overall = "FAIL"
    elif res.non_verdict_cells:
        overall = (f"INCOMPLETE ({res.non_verdict_cells} cell(s) are not a verdict: "
                   f"`neural-trade stability --retry-non-verdict {res.harness_id}`)")
    else:
        overall = "PASS" if not res.not_run else "PASS (cases run)"
    probe_text = {"off": "off", "on": "on in every cell",
                  "failed": "off, then on (PROBE_EVERY 1) in one re-run of each failed cell"}[res.probe_mode]
    L: List[str] = [f"# Stability harness report {res.harness_id}", ""]
    L += [f"- thresholds: `{T.path.name}` ({T.name}), sha256 `{T.sha256}`",
          f"- profile: {profile}; seeds: {list(seeds)}; strict mode (STRICT_LOSS_MASKS); per-term probe: {probe_text}",
          f"- commit: `{git_sha()}`; scenario `{res.scenario}`; run store `{st.root}`, index `{st.index_path}`"]
    if res.retry_of:
        L.append(f"- a retry launch: re-ran the not-a-verdict cells of `{res.retry_of}`; every other verdict is carried "
                 "over from it unchanged")
    L += [f"- overall: **{overall}**"
          f" ({n_pass} of {len(res.case_passed)} cases passed; {n_fail} failed; {n_nv} case(s) not a verdict; "
          f"{len(res.not_run)} case(s) defined, not run)", ""]
    L += ["## Cases", "", "| case | group | expect | verdict | seeds passed | blamed loss term | failed checks |",
          "|---|---|---|---|---|---|---|"]
    for c in cases:
        vs = [v for v in res.verdicts if v.case == c.id]
        ok = sum(v.passed for v in vs)
        nv = sum(v.non_verdict for v in vs)
        blamed = sorted({_blame_text(v) for v in vs if not v.passed and not v.non_verdict})
        failed = sorted({k.name for v in vs for k in v.failed_checks})
        L.append(f"| {c.id} | {c.group} | {c.expect} | {res.case_status.get(c.id, 'FAIL')} | "
                 f"{ok}/{len(seeds)}{f' ({nv} not a verdict)' if nv else ''} | {'; '.join(blamed) or '-'} | "
                 f"{', '.join(failed) or '-'} |")
    L += ["", f"Blame comes from the run's own error text (a guarded run), the masked-term counters, or the probe "
              f"sample of the cell's probe run ({PROBE_SOURCE}: the largest share of the first epoch whose shares sum "
              "to 1, not an epoch average); a '-' says why. A failure with a probe sample names the term, the "
              "variable group and the share in the failure list below. Data and fault cases get no failing region "
              "by design (the data are not a Config field); a cell that is not a verdict gets neither a blame nor "
              "a region.", ""]
    if res.n_eff:
        mins = [float(T.checks[k]) for k in ("min_n_eff_variance_excess", "min_n_eff_variance") if k in T.checks]
        L += ["## n_eff of the scored block per case", "",
              "n_eff = bars scored // horizon (D-012). A variance-head check is judged on a horizon only when its n_eff "
              "reaches the thresholds file's gate" + (f" (here {', '.join(f'{m:g}' for m in sorted(mins))})" if mins
                                                       else " (this file has none)") + "; otherwise it is reported "
              "\"not evaluated\" in the cell's checks.", "", "| case | n_eff by horizon |", "|---|---|"]
        L += [f"| {c.id} | {', '.join(f'{h} {v:g}' for h, v in res.n_eff[c.id].items())} |"
              for c in cases if c.id in res.n_eff]
        L.append("")
    failing = [v for v in res.verdicts if not v.passed and not v.non_verdict]
    if failing:
        L += ["## Failures", ""]
        for v in failing:
            L.append(f"- `{v.cell_key}` (run `{v.run_id}`): " + "; ".join(
                f"{c.name} = {c.value} (limit {c.limit}){': ' + c.detail if c.detail else ''}"
                for c in v.failed_checks) + f". Blamed: {_blame_text(v)}"
                + (f" [{v.blame_reason}]" if v.blamed and v.blame_reason else "") + ".")
        L.append("")
    if res.not_rerun:
        L += ["## Failed cells not re-run (cap)", "",
              f"`--max-probe-reruns` was reached: these failed cells have no probe re-run and no blame "
              f"({NOT_RERUN_CAP}). Raise the cap or run them alone (`--cases`, `--probe on`).", "",
              "| case | seed | run |", "|---|---|---|"]
        L += [f"| {v.case} | {v.seed} | `{v.run_id}` |" for v in res.not_rerun]
        L.append("")
    if res.reruns:
        first = {v.run_id: v for v in res.verdicts}
        L += ["## Probe re-runs", "",
              "Each failed cell ran once with the probe off and once with it on (PROBE_EVERY 1). The probe never "
              "changes training: the re-run's verdict is shown beside the first run's, and the first run's verdict "
              "is the one that counts.", "",
              "| case | seed | first run | probe re-run | first verdict | re-run verdict | blamed |", "|---|---|---|---|---|---|---|"]
        for r in res.reruns:
            f = first.get(r.rerun_of)
            L.append(f"| {r.case} | {r.seed} | `{r.rerun_of}` | `{r.run_id}` | {'PASS' if f and f.passed else 'FAIL'} | "
                     f"{NOT_A_VERDICT if r.non_verdict else 'PASS' if r.passed else 'FAIL'} | "
                     f"{_blame_text(f) if f else '-'} |")
        L.append("")
    nv_now = [v for v in res.verdicts if v.non_verdict]
    if res.superseded or nv_now:
        L += ["## Cells that were not a verdict", "",
              "A resource error (ResourceExhaustedError, MemoryError, OSError) or a crashed worker is not a verdict: "
              "the cell is left out of the pass and fail counts until a retry gives it one. Both runs are listed.", "",
              "| case | seed | run | error | replaced by |", "|---|---|---|---|---|"]
        later = {v.retry_of: v for v in res.verdicts if v.retry_of}
        for v in [*res.superseded, *nv_now]:
            by = later.get(v.run_id)
            L.append(f"| {v.case} | {v.seed} | `{v.run_id}` | {v.error or '-'} | {f'`{by.run_id}`' if by else 'still open'} |")
        L.append("")
    L += ["## Every cell", "", "| run | case | seed | verdict | checks (value / limit) |", "|---|---|---|---|---|"]
    for v in res.verdicts:
        cs = "; ".join(f"{c.name} {'n/a' if c.value is None else f'{c.value:.4g}'}/{c.limit}"
                       f"{'' if c.evaluated else ' (not evaluated)'}" for c in v.checks)
        word = NOT_A_VERDICT if v.non_verdict else ("PASS" if v.passed else "FAIL")
        note = f" (retry of `{v.retry_of}`)" if v.retry_of else ""
        L.append(f"| `{v.run_id}` | {v.case} | {v.seed} | {word}{note} | {cs or v.error or '-'} |")
    L.append("")
    if res.not_run:
        L += ["## Defined, not run here", ""]
        L += [f"- {c.id}: {c.description}. {c.note}" for c in res.not_run]
        L.append("")
    L += ["## Failing regions", ""]
    if res.regions:
        L += [f"- `{r.id}`: {r.describe()} ({r.reason})" for r in res.regions]
        L += ["", f"Machine-readable: `{res.out_dir.as_posix()}/failing_regions.json` (copy the regions into "
                  "`configs/stability_failing_regions.json` to make `Config.validate` refuse them)."]
    else:
        L.append("None: no configuration case failed.")
    L.append("")
    return "\n".join(L)


__all__ = ["COMMON_OVERRIDES", "NOT_A_VERDICT", "PROBE_MODES", "PROBE_SOURCE", "PROFILE_PROBE", "dry_run",
           "EXIT_REFUSED", "MAX_PROBE_RERUNS", "NOT_RERUN_CAP", "expected_n_eff_for", "is_non_verdict_error", "plan_cases", "probe_blame", "probe_overrides", "Case", "Check", "HarnessResult", "HarnessTrainer", "PROFILES", "THRESHOLDS_FILE",
           "THRESHOLDS_V2_FILE", "Thresholds", "Verdict", "VERDICT_FILE", "FUZZ_CONSTANT_BARS", "PROFILE_THRESHOLDS",
           "resolve_thresholds_path", "build_scenario", "default_cases", "evaluate_run",
           "fault_context", "file_sha256", "load_thresholds", "region_for_case", "render_report", "run_harness",
           "transform_bars", "write_case_data", "write_verdict"]

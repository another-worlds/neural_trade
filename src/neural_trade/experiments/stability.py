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
off, so a non-finite term reaches the total instead of being hidden); the ``reference`` profile also runs the
per-term gradient probe (the ``tiny`` CPU profile does not: see :data:`COMMON_OVERRIDES`).

**Cases are engine scenarios.** :func:`build_scenario` makes one variant per case; :class:`~neural_trade
.experiments.runner.Runner` trains every (case, seed) cell into the run store and its index (the harness
trainer applies the case's fault). After the runs, :func:`evaluate_run` judges each cell against the thresholds
file (``configs/stability_thresholds.yaml``, pre-registered; its sha256 is in every report), writes
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

THRESHOLDS_FILE = Path(__file__).resolve().parents[3] / "configs" / "stability_thresholds.yaml"
VERDICT_FILE = "stability_verdict.json"
STABILITY_DIR = "stability"
GPU_NOTE = "GPU, NT-051"

# Config overrides every case shares: strict mode (masks off). The per-term gradient probe is a profile's choice:
# it costs about 10% of a GPU step but, on CPU, a tiny run's trace of 17 terms x 3 groups x 136 pairs took 200 s
# against 30 s without it (measured 2026-10-06), so the tiny profile leaves it off (that check is then
# reported "not evaluated") and the reference profile turns it on.
COMMON_OVERRIDES: Dict[str, Any] = {"STRICT_LOSS_MASKS": True}
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


# ------------------------------------------------------------------ thresholds
@dataclass(frozen=True)
class Thresholds:
    path: Path
    sha256: str
    name: str
    checks: Mapping[str, Any]
    fault_detection: Mapping[str, Any]
    seeds: int

    def check(self, key: str) -> Any:
        return self.checks[key]


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_thresholds(path=None) -> Thresholds:
    """The pre-registered thresholds with the file's sha256 (any edit, a comment included, changes it)."""
    import yaml

    p = Path(path) if path is not None else THRESHOLDS_FILE
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1:
        raise ValueError(f"{p}: schema_version must be 1")
    return Thresholds(p, file_sha256(p), str(doc["name"]), dict(doc["checks"]), dict(doc["fault_detection"]),
                      int(doc.get("seeds", 3)))


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
    tiny_layout: Mapping[str, Any] = field(default_factory=dict)    # data-layout overrides the tiny profile needs
                                                                    # (never part of a failing region)

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
        Case("fuzz_constant", "extreme_input", "a 400-bar block of one constant price (flat windows)",
             data={"kind": "constant", "bars": 400}),
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
             tiny_layout={"MAX_SEQUENCE_COUNT": 2700, "N_FOLDS": 4}),     # the purge gap is 480 bars here
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


def transform_bars(df, spec: Mapping[str, Any]):
    """A copy of the OHLCV frame ``df`` with the transform ``spec`` applied to the price columns (the volume and
    the timestamps stay); ``kind``: scale (``k``), vol (``k``), constant (``bars``), jumps."""
    out = df.copy()
    cols = _columns(out)
    kind = spec["kind"]
    close = out[cols["close"]].to_numpy(dtype=np.float64)
    n = len(out)
    if kind == "scale":
        ratio = np.full(n, float(spec["k"]))
    elif kind == "vol":
        new = close[0] * np.exp(float(spec["k"]) * (np.log(close) - np.log(close[0])))
        ratio = new / close
    elif kind == "constant":
        start, bars = n // 2, min(int(spec["bars"]), n // 2)
        ratio = np.ones(n)
        ratio[start:start + bars] = close[start] / close[start:start + bars]
    elif kind == "jumps":
        ratio = np.ones(n)
        ratio[n // 3:] *= 2.0
        for i, f in zip(np.linspace(n // 6, n - n // 6, 5).astype(int), (4.0, 0.25, 4.0, 0.25, 4.0)):
            ratio[i] *= f
    else:
        raise ValueError(f"unknown data transform {kind!r}")
    for col in cols.values():
        out[col] = out[col].to_numpy(dtype=np.float64) * ratio
    if kind == "constant":                      # a constant block: open = high = low = close
        start, bars = n // 2, min(int(spec["bars"]), n // 2)
        for col in cols.values():
            out.loc[out.index[start:start + bars], col] = float(close[start])
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
                   case_csv: Mapping[str, str], folds: Sequence[int] = tuple(PROFILE_FOLDS)) -> Dict[str, Any]:
    """The engine scenario spec (a dict for ``Scenario.from_dict``): one variant per runnable case."""
    variants = {}
    for c in cases:
        if not c.runnable:
            continue
        v = dict(c.overrides)
        if profile == "tiny":
            v.update(c.tiny_layout)
        if c.id in case_csv:
            v["CSV_PATH"] = case_csv[c.id]
        variants[c.id] = v
    return {"schema_version": 1, "name": name,
            "description": f"stability harness ({profile} profile, NT-038): strict mode, per-term probe, "
                           f"{len(variants)} cases x {len(seeds)} seeds",
            "overrides": {**COMMON_OVERRIDES, **PROFILES[profile]}, "variants": variants, "folds": list(folds),
            "seeds": [int(s) for s in seeds], "strategy": {"name": "calibrated_quantile", "params": {}},
            "backtest": {"random_seeds": 5}, "run": {"calibrate": False, "save_artifacts": False}}


def write_case_data(cases: Sequence[Case], csv, out_dir) -> Dict[str, str]:
    """``<out_dir>/data/<case>.csv`` for every runnable case with a data transform; {case id: path}."""
    import pandas as pd

    todo = [c for c in cases if c.runnable and c.data]
    if not todo:
        return {}
    base = pd.read_csv(csv)
    d = Path(out_dir) / "data"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in todo:
        path = d / f"{c.id}.csv"
        transform_bars(base, c.data).to_csv(path, index=False)
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

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "limit": self.limit, "passed": self.passed,
                "detail": self.detail, "evaluated": self.evaluated}


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

    def to_dict(self) -> Dict[str, Any]:
        return {"case": self.case, "run_id": self.run_id, "cell_key": self.cell_key, "seed": self.seed,
                "passed": self.passed, "blamed": self.blamed, "status": self.status, "error": self.error,
                "thresholds_sha256": self.thresholds_sha256, "checks": [c.to_dict() for c in self.checks]}

    @property
    def failed_checks(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]


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


def evaluate_run(run_dir, case: Case, thresholds: Thresholds) -> Verdict:
    """Judge one finished cell against the pre-registered thresholds (nothing is read but the run directory)."""
    from neural_trade.core.config import Config
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

    if case.expect == "detect":
        fd = T.fault_detection
        stopped = status == "failed" and err.get("type") == fd.get("error_type")
        checks.append(Check("fault_stopped_run", 1.0 if stopped else 0.0, fd.get("error_type"), stopped,
                            "" if stopped else f"the run ended {status}" + (f" with {message}" if message else "")))
        if fd.get("names_the_term") and case.expects_term:
            named = case.expects_term in (err.get("message") or "")
            checks.append(Check("fault_names_term", 1.0 if named else 0.0, case.expects_term, named,
                                "" if named else f"the error does not name {case.expects_term}: {message}"))
        return Verdict(case.id, meta.get("run_id", d.name), eng.get("cell_key", ""), int(meta.get("seed", -1)),
                       all(c.passed for c in checks), checks, blamed, status, message, T.sha256)

    completed = status == "done"
    checks.append(Check("run_completed", 1.0 if completed else 0.0, "done", completed,
                        "" if completed else f"{status}: {message}"))
    rows = read_metrics(d / "metrics.jsonl") if (d / "metrics.jsonl").is_file() else []
    if rows:
        cfg = Config.from_yaml(d / "config.yaml")
        why = run_health(d, max_nonfinite_grad_steps=10 ** 9)
        checks.append(Check("loss_finite", 0.0 if why else 1.0, True, not why, why or ""))
        n_steps = sum(float(r.get("n_steps") or 0) for r in rows)
        nonfinite = sum(float(r.get("nonfinite_grad_steps") or 0) for r in rows)
        rate = nonfinite / n_steps if n_steps else None
        lim = float(T.check("max_nonfinite_step_rate"))
        checks.append(Check("nonfinite_step_rate", rate, lim, rate is not None and rate <= lim,
                            "" if rate is not None else "no steps counted"))
        h = health_block(rows, cfg)
        lim = float(T.check("max_clipped_share"))
        for grp in ("main", "indicator"):
            clipped = h.get(f"grad_clip_steps_{grp}_total")
            share = (float(clipped) / n_steps) if clipped is not None and n_steps else None
            checks.append(Check(f"clipped_share_{grp}", share, lim, share is None or share <= lim,
                                "" if share is not None else "not logged", share is not None))
        shares = _probe_shares(rows)
        lim = float(T.check("max_term_gradient_share"))
        if shares:
            bad = sorted(k for k, v in shares.items() if v is None)
            top = max((k for k, v in shares.items() if v is not None), key=lambda k: shares[k], default=None)
            top_v = shares[top] if top else None
            ok = not bad and (top_v is None or top_v <= lim)
            detail = (f"non-finite probe share(s): {bad[:4]}" if bad else
                      (f"largest: {top} = {top_v:.3f}" if top else ""))
            checks.append(Check("term_gradient_share", top_v, lim, ok, detail))
            if not ok and top and not blamed:
                blamed = [top.rsplit("_", 1)[0]]
        else:
            checks.append(Check("term_gradient_share", None, lim, True, "the probe was off", False))
        lim = float(T.check("max_var_at_floor_share"))
        worst = None
        for r in rows:
            for hz in ("h0", "h1", "h2"):
                n_dir, at_floor = r.get(f"dir_n_{hz}"), r.get(f"var_at_floor_{hz}")
                if n_dir and at_floor is not None:
                    worst = max(worst or 0.0, float(at_floor) / float(n_dir))
        checks.append(Check("var_at_floor_share", worst, lim, worst is None or worst <= lim,
                            "" if worst is not None else "not logged", worst is not None))
        at_bound = h.get("periods_at_bound") or {}
        lim = int(T.check("max_periods_at_bound"))
        checks.append(Check("periods_at_bound", float(len(at_bound)), lim, len(at_bound) <= lim,
                            ", ".join(sorted(at_bound)[:4])))
        masked = h.get("masked_terms_total") or {}
        total_masked = float(sum(masked.values()))
        lim = float(T.check("max_masked_term_steps"))
        checks.append(Check("masked_term_steps", total_masked, lim, total_masked <= lim,
                            ", ".join(f"{k[len('masked_'):]}={v:g}" for k, v in sorted(masked.items())[:4])))
        if masked and not blamed:
            from neural_trade.training.stability_guard import blame

            blamed = blame({k[len("masked_"):]: v for k, v in masked.items()})
    scores = result.get("scores") or {}
    cov = {k: v for k, v in scores.items() if re.fullmatch(r"h\d+/variance/coverage90", k) and _finite(v)}
    lim = float(T.check("min_coverage90"))
    if cov:
        lo = min(cov, key=lambda k: cov[k])
        checks.append(Check("coverage90", float(cov[lo]), lim, float(cov[lo]) >= lim, f"lowest: {lo}"))
    else:
        checks.append(Check("coverage90", None, lim, True, "the run was not scored", False))
    return Verdict(case.id, meta.get("run_id", d.name), eng.get("cell_key", ""), int(meta.get("seed", -1)),
                   all(c.passed for c in checks), checks, blamed, status, message, T.sha256)


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
    verdicts: List[Verdict]
    not_run: List[Case]
    case_passed: Dict[str, bool]
    regions: List[Any]
    report: Path

    @property
    def passed(self) -> bool:
        return all(self.case_passed.values())


def run_harness(*, profile: str = "tiny", csv=None, store="runs", case_ids: Optional[Sequence[str]] = None,
                seeds: Optional[Sequence[int]] = None, thresholds_path=None, harness_id: Optional[str] = None,
                trainer=None, out_root=None) -> HarnessResult:
    """Run the harness: write the data, run the cases as an engine scenario into ``store``, judge every cell,
    write the report. ``trainer`` replaces the engine trainer (tests)."""
    from neural_trade.core.config import Config
    from neural_trade.core.guard import regions_disabled
    from neural_trade.experiments.runner import Runner
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.store import RunStore

    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}, got {profile!r}")
    T = load_thresholds(thresholds_path)
    all_cases = default_cases()
    if case_ids:
        unknown = sorted(set(case_ids) - {c.id for c in all_cases})
        if unknown:
            raise ValueError(f"unknown case(s) {unknown}; known: {[c.id for c in all_cases]}")
        all_cases = [c for c in all_cases if c.id in set(case_ids)]
    seed_list = [int(s) for s in seeds] if seeds else list(range(T.seeds))
    hid = harness_id or f"{_stamp()}-{profile}"
    st = store if isinstance(store, RunStore) else RunStore(store)
    out_dir = Path(out_root if out_root is not None else st.root) / STABILITY_DIR / hid
    out_dir.mkdir(parents=True, exist_ok=False)
    csv = csv if csv is not None else Config().CSV_PATH
    runnable = [c for c in all_cases if c.runnable]
    not_run = [c for c in all_cases if not c.runnable]
    case_csv = write_case_data(runnable, csv, out_dir)
    base_csv = {c.id: str(csv) for c in runnable if c.id not in case_csv}
    spec = build_scenario(runnable, name=f"stab-{hid}"[:48], profile=profile, seeds=seed_list,
                          case_csv={**base_csv, **case_csv})
    (out_dir / "scenario.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8", newline="\n")
    scenario = Scenario.from_dict(spec)
    by_id = case_by_id(runnable)
    with regions_disabled():     # the harness re-tests configurations inside known failing regions
        runner = Runner(scenario, st, trainer=trainer if trainer is not None else HarnessTrainer(runnable))
        runner.run()
    rows = st.sync(scenario.name)
    verdicts: List[Verdict] = []
    for r in rows:
        case = by_id.get(r["configuration"])
        if case is None or r["status"] == "incomplete":
            continue
        run_dir = st.root / r["run_dir"]
        v = evaluate_run(run_dir, case, T)
        write_verdict(run_dir, v)
        verdicts.append(v)
    st.sync(scenario.name)       # the index now carries the verdicts (stability/* scores)
    verdicts.sort(key=lambda v: (v.case, v.seed))
    case_passed = {c.id: bool([v for v in verdicts if v.case == c.id]) and
                   all(v.passed for v in verdicts if v.case == c.id) and
                   len([v for v in verdicts if v.case == c.id]) == len(seed_list) for c in runnable}
    report_path = out_dir / "REPORT.md"
    regions = []
    for c in runnable:
        if c.region and not case_passed[c.id]:
            fails = [v for v in verdicts if v.case == c.id and not v.passed]
            why = "; ".join(sorted({f"{k.name}" for v in fails for k in v.failed_checks})) or "no verdict"
            regions.append(region_for_case(c, str(report_path).replace("\\", "/"),
                                           f"the stability harness's case {c.id} failed: {why}"))
    from neural_trade.core.guard import write_regions

    write_regions(out_dir / "failing_regions.json", regions)
    (out_dir / "verdicts.json").write_text(json.dumps(
        {"harness_id": hid, "thresholds_sha256": T.sha256, "verdicts": [v.to_dict() for v in verdicts],
         "case_passed": case_passed}, indent=2, default=str), encoding="utf-8", newline="\n")
    result = HarnessResult(hid, out_dir, scenario.name, T.sha256, verdicts, not_run, case_passed, regions, report_path)
    report_path.write_text(render_report(result, T, runnable, profile, seed_list, st), encoding="utf-8", newline="\n")
    return result


def render_report(res: HarnessResult, T: Thresholds, cases: Sequence[Case], profile: str, seeds: Sequence[int],
                  st) -> str:
    from neural_trade.utils.env import git_sha

    L: List[str] = [f"# Stability harness report {res.harness_id}", ""]
    L += [f"- thresholds: `{T.path.name}` ({T.name}), sha256 `{T.sha256}`",
          f"- profile: {profile}; seeds: {list(seeds)}; strict mode (STRICT_LOSS_MASKS); per-term probe {'on' if PROFILES[profile].get('PROBE_GRADIENTS') else 'off'}",
          f"- commit: `{git_sha()}`; scenario `{res.scenario}`; run store `{st.root}`, index `{st.index_path}`",
          f"- overall: **{'PASS' if res.passed and not res.not_run else 'PASS (cases run)' if res.passed else 'FAIL'}**"
          f" ({sum(res.case_passed.values())} of {len(res.case_passed)} cases passed; "
          f"{len(res.not_run)} case(s) defined, not run)", ""]
    L += ["## Cases", "", "| case | group | expect | verdict | seeds passed | blamed loss term | failed checks |",
          "|---|---|---|---|---|---|---|"]
    for c in cases:
        vs = [v for v in res.verdicts if v.case == c.id]
        ok = sum(v.passed for v in vs)
        blamed = sorted({t for v in vs if not v.passed for t in v.blamed})
        failed = sorted({k.name for v in vs for k in v.failed_checks})
        L.append(f"| {c.id} | {c.group} | {c.expect} | {'PASS' if res.case_passed.get(c.id) else 'FAIL'} | "
                 f"{ok}/{len(seeds)} | {', '.join(blamed) or '-'} | {', '.join(failed) or '-'} |")
    L.append("")
    failing = [v for v in res.verdicts if not v.passed]
    if failing:
        L += ["## Failures", ""]
        for v in failing:
            L.append(f"- `{v.cell_key}` (run `{v.run_id}`): " + "; ".join(
                f"{c.name} = {c.value} (limit {c.limit}){': ' + c.detail if c.detail else ''}"
                for c in v.failed_checks))
        L.append("")
    L += ["## Every cell", "", "| run | case | seed | verdict | checks (value / limit) |", "|---|---|---|---|---|"]
    for v in res.verdicts:
        cs = "; ".join(f"{c.name} {'n/a' if c.value is None else f'{c.value:.4g}'}/{c.limit}"
                       f"{'' if c.evaluated else ' (not evaluated)'}" for c in v.checks)
        L.append(f"| `{v.run_id}` | {v.case} | {v.seed} | {'PASS' if v.passed else 'FAIL'} | {cs} |")
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


__all__ = ["COMMON_OVERRIDES", "Case", "Check", "HarnessResult", "HarnessTrainer", "PROFILES", "THRESHOLDS_FILE",
           "Thresholds", "Verdict", "VERDICT_FILE", "build_scenario", "default_cases", "evaluate_run",
           "fault_context", "file_sha256", "load_thresholds", "region_for_case", "render_report", "run_harness",
           "transform_bars", "write_case_data", "write_verdict"]

"""Ablation harness for the physics loss terms (plan C4).

A spec (``configs/ablation_physics.yaml``) names the terms and their "on" values, the modes,
seeds and periods (walk-forward folds). The grid is

    all_on, all_off                         the family
    only:<term>   (leave_one_in)            the term alone vs all_off
    without:<term> (leave_one_out)          all_on vs all_on minus the term

x seeds x periods; with six terms that is 14 conditions x 3 seeds x 2 periods = 84 runs.

Each cell trains in its OWN process (``python -m neural_trade.experiments.ablation cell ...``):
TF op seeds depend on per-process counters and GPU memory is released on exit. A cell writes
``cells/<key>.json`` and is skipped when that file exists (resume). The per-term lambda
calibration pass is run once (on all_on) and its weights are frozen for every cell - the pass
rescales every lambda by a shared reference, so calibrating per cell would let toggling one
term silently change the others.

Verdicts are pre-registered in ``configs/ablation_criteria.yaml`` and computed by ``analyze``:
paired deltas over (seed, period), oriented so positive = the term helps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

MODES = ("all_on", "all_off", "leave_one_in", "leave_one_out")
VERDICTS = ("VALUE", "HARMFUL", "NEUTRAL", "INCONCLUSIVE")


# ------------------------------------------------------------------ spec
@dataclass
class Condition:
    name: str
    mode: str
    term: Optional[str]
    overrides: Dict[str, float]


@dataclass
class Cell:
    condition: Condition
    seed: int
    period: str
    fold_index: int

    @property
    def key(self) -> str:
        return f"{self.condition.name.replace(':', '-')}__s{self.seed}__{self.period}"


@dataclass
class AblationSpec:
    name: str
    terms: Dict[str, float]                       # LAMBDA_* -> value when the term is on
    modes: List[str] = field(default_factory=lambda: list(MODES))
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    periods: Dict[str, int] = field(default_factory=lambda: {"P1": -2, "P2": -1})  # name -> FOLD_INDEX
    base_overrides: Dict[str, Any] = field(default_factory=dict)
    scales: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    calibrate: str = "once"                       # "once" (all_on, then frozen) or "none"
    strategy: str = "enhanced_multi_horizon"

    def __post_init__(self):
        bad = [m for m in self.modes if m not in MODES]
        if bad:
            raise ValueError(f"unknown ablation modes {bad}; known: {MODES}")
        if self.calibrate not in ("once", "none"):
            raise ValueError(f"calibrate must be 'once' or 'none', got {self.calibrate!r}")
        if not self.terms:
            raise ValueError("an ablation needs at least one term")

    @classmethod
    def from_yaml(cls, path) -> "AblationSpec":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def conditions(self) -> List[Condition]:
        on = {t: float(v) for t, v in self.terms.items()}
        off = {t: 0.0 for t in self.terms}
        out: List[Condition] = []
        if "all_on" in self.modes or "leave_one_out" in self.modes:
            out.append(Condition("all_on", "all_on", None, dict(on)))
        if "all_off" in self.modes or "leave_one_in" in self.modes:
            out.append(Condition("all_off", "all_off", None, dict(off)))
        if "leave_one_in" in self.modes:
            out += [Condition(f"only:{t}", "leave_one_in", t, {**off, t: on[t]}) for t in self.terms]
        if "leave_one_out" in self.modes:
            out += [Condition(f"without:{t}", "leave_one_out", t, {**on, t: 0.0}) for t in self.terms]
        return out

    def cells(self) -> List[Cell]:
        return [Cell(c, int(s), p, int(f)) for c in self.conditions() for p, f in self.periods.items()
                for s in self.seeds]

    def overrides_for(self, cell: Cell, scale: str, frozen: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if scale not in self.scales:
            raise ValueError(f"unknown scale {scale!r}; spec defines {sorted(self.scales)}")
        return {**self.base_overrides, **self.scales[scale], **(frozen or {}), **cell.condition.overrides,
                "SEED": cell.seed, "FOLD_INDEX": cell.fold_index}


# ------------------------------------------------------------------ one cell (runs in its own process)
def _numeric(d: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = float(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
    return out


def execute_cell(spec: AblationSpec, cell: Cell, scale: str, out_dir, frozen: Optional[Dict[str, float]] = None,
                 csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Train, evaluate (with baselines) and backtest one cell; returns the results row."""
    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.evaluation.baselines import BaselineSet
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.evaluation.report import evaluate
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.strategy import Bars, SignalFrame, build_strategy, run_backtest, var_scale_from
    from neural_trade.training.trainer import train_and_evaluate

    out_dir = Path(out_dir)
    overrides = spec.overrides_for(cell, scale, frozen)
    epochs = int(overrides.pop("EPOCHS", 20))
    cfg = Config().override(**({"CSV_PATH": csv_path} if csv_path else {}), **overrides)
    ctx = RunContext.create(cfg, root=out_dir / "runs", seed=cell.seed, name=cell.key,
                            tags=[spec.name, cell.condition.name, cell.period, scale])
    t0 = time.time()
    result = train_and_evaluate(config=ctx.config, run_context=ctx, epochs=epochs, force=True, calibrate=False,
                                fit_calibration=True, save_artifacts=False)
    wall = time.time() - t0

    blocks = split_arrays(ctx.config)
    test = PredictionFrame.from_result(result, "test")
    cal = PredictionFrame.from_result(result, "cal") if result.predictions_cal is not None else None
    base = BaselineSet.fit(blocks["train"]["X"], blocks["train"]["y"], blocks["train"]["last_close"],
                           float(ctx.config.DIR_DEADBAND_BPS))
    bars = Bars.from_frame(blocks["df"], blocks["test"]["anchor_bar"])
    var_scale = var_scale_from(cal) if cal is not None else var_scale_from(test)
    bt = run_backtest(SignalFrame.build(test, var_scale), bars, build_strategy(spec.strategy))
    report = evaluate(test, ctx.config, baselines=base, cal_frame=cal, run_id=ctx.run_id,
                      backtest={"strategy": spec.strategy, "summary": bt.summary})
    report.to_json(ctx.path("eval_report_test.json"))
    report.to_markdown(ctx.path("eval_report_test.md"))

    hist = getattr(result.history, "history", {}) or {}
    row = {"key": cell.key, "condition": cell.condition.name, "mode": cell.condition.mode,
           "term": cell.condition.term, "seed": cell.seed, "period": cell.period, "scale": scale,
           "run_id": ctx.run_id, "wall_s": wall, "epochs_run": len(hist.get("loss", [])),
           **_numeric(report.flat())}
    return row


def _cell_path(out_dir, cell: Cell) -> Path:
    return Path(out_dir) / "cells" / f"{cell.key}.json"


def load_rows(out_dir) -> List[Dict[str, Any]]:
    d = Path(out_dir) / "cells"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.exists() else []


# ------------------------------------------------------------------ calibration once, then frozen
def calibrate_once(spec: AblationSpec, scale: str, out_dir, csv_path: Optional[str] = None) -> Dict[str, float]:
    """Run the lambda calibration pass on all_on (first period, first seed) and freeze the non-ablated
    lambdas it chooses. Cached in ``frozen_lambdas.json``."""
    path = Path(out_dir) / "frozen_lambdas.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if spec.calibrate == "none":
        return {}
    from neural_trade.core.config import Config
    from neural_trade.training.lambdas import CONFIG_NAME_OF_KEY
    from neural_trade.training.trainer import train_and_evaluate

    cond = next(c for c in spec.conditions() if c.name == "all_on") if "all_on" in spec.modes else \
        Condition("all_on", "all_on", None, {t: float(v) for t, v in spec.terms.items()})
    first_period, fold = next(iter(spec.periods.items()))
    cell = Cell(cond, int(spec.seeds[0]), first_period, int(fold))
    overrides = spec.overrides_for(cell, scale)
    overrides.pop("EPOCHS", None)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    overrides.update(MODEL_PATH=str(Path(out_dir) / "calibration_weights.h5"),
                     SCALER_PATH=str(Path(out_dir) / "calibration_scaler.joblib"))
    cfg = Config().override(**({"CSV_PATH": csv_path} if csv_path else {}), **overrides)
    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False,
                                save_artifacts=False)
    lambdas = result.calibration_lambdas or {}
    frozen = {}
    for key, value in lambdas.items():
        name = CONFIG_NAME_OF_KEY.get(key[len("lambda_"):]) if key.startswith("lambda_") else None
        if name and name not in spec.terms:
            frozen[name] = float(value)
    path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


# ------------------------------------------------------------------ the grid
def run_grid(spec: AblationSpec, scale: str, out_dir, *, runner: Optional[Callable[..., Dict[str, Any]]] = None,
             resume: bool = True, limit: Optional[int] = None, csv_path: Optional[str] = None,
             frozen: Optional[Dict[str, float]] = None, log: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    """Run every pending cell. ``runner(spec, cell, scale, out_dir, frozen)`` returns the row; the
    default launches one subprocess per cell."""
    out_dir = Path(out_dir)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    if frozen is None:
        frozen = calibrate_once(spec, scale, out_dir, csv_path) if runner is None else {}
    runner = runner or (lambda s, c, sc, o, f: _run_cell_subprocess(s, c, sc, o, csv_path))
    done = 0
    for cell in spec.cells():
        path = _cell_path(out_dir, cell)
        if resume and path.exists():
            continue
        if limit is not None and done >= limit:
            break
        log(f"[ablation] {cell.key} ...")
        t0 = time.time()
        row = runner(spec, cell, scale, out_dir, frozen)
        row.setdefault("wall_s", time.time() - t0)
        path.write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
        log(f"[ablation] {cell.key} done in {row['wall_s'] / 60:.1f} min")
        done += 1
    rows = load_rows(out_dir)
    write_results(rows, out_dir)
    return rows


def _run_cell_subprocess(spec: AblationSpec, cell: Cell, scale: str, out_dir, csv_path=None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    spec_path = out_dir / "spec.json"
    tmp = out_dir / "cells" / f"{cell.key}.partial.json"
    cmd = [sys.executable, "-m", "neural_trade.experiments.ablation", "cell", "--spec-json", str(spec_path),
           "--scale", scale, "--out", str(out_dir), "--key", cell.key, "--row-out", str(tmp)]
    if csv_path:
        cmd += ["--csv", csv_path]
    logf = out_dir / "logs" / f"{cell.key}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED", "0"))
    with open(logf, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if rc != 0 or not tmp.exists():
        raise RuntimeError(f"ablation cell {cell.key} failed (exit {rc}); see {logf}")
    row = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink()
    return row


def dry_run(spec: AblationSpec, out_dir, sec_per_run: Optional[float] = None) -> Dict[str, Any]:
    """How many cells remain and how long they should take (from completed cells' wall time)."""
    rows = load_rows(out_dir)
    done = {r["key"] for r in rows}
    cells = spec.cells()
    pending = [c.key for c in cells if c.key not in done]
    walls = [r["wall_s"] for r in rows if r.get("wall_s")]
    per = float(np.mean(walls)) if walls else sec_per_run
    return {"total": len(cells), "done": len(cells) - len(pending), "pending": len(pending),
            "sec_per_run": per, "projected_hours": (per * len(pending) / 3600.0) if per else None,
            "pending_keys": pending}


# ------------------------------------------------------------------ analysis
@dataclass
class MetricCriterion:
    metric: str
    higher_is_better: bool = True
    mde: float = 0.0

    def orient(self, x: float) -> float:
        return x if self.higher_is_better else -x


@dataclass
class Criteria:
    primary: Dict[str, List[MetricCriterion]]          # term (or "family") -> primaries
    guardrails: List[Dict[str, Any]]                    # {metric, higher_is_better, tolerance}
    min_agree_frac: float = 5 / 6

    @classmethod
    def from_yaml(cls, path) -> "Criteria":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data) -> "Criteria":
        prim = {k: [MetricCriterion(**m) for m in v] for k, v in (data.get("primary") or {}).items()}
        return cls(prim, list(data.get("guardrails") or []), float(data.get("min_agree_frac", 5 / 6)))

    def for_term(self, term: Optional[str]) -> List[MetricCriterion]:
        if term is None:
            return self.primary.get("family", [])
        if term in self.primary:
            return self.primary[term]
        return self.primary.get("default", [])


def _index(rows):
    return {(r["condition"], r["seed"], r["period"]): r for r in rows}


def _pairs(idx, treat: str, control: str, metric: str):
    out = []
    for (cond, seed, period), r in idx.items():
        if cond != treat:
            continue
        c = idx.get((control, seed, period))
        if c is None:
            continue
        a, b = r.get(metric), c.get(metric)
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            continue
        out.append(((seed, period), a, b))
    return out


def _sigma_seed(rows, condition: str, metric: str) -> float:
    """Pooled across-seed standard deviation of ``metric`` within ``condition`` (per period)."""
    by_period: Dict[str, List[float]] = {}
    for r in rows:
        v = r.get(metric)
        if r["condition"] == condition and v is not None and math.isfinite(v):
            by_period.setdefault(r["period"], []).append(v)
    var = [np.var(v, ddof=1) for v in by_period.values() if len(v) >= 2]
    return float(math.sqrt(np.mean(var))) if var else float("nan")


def compare(rows, treat: str, control: str, crit: MetricCriterion, min_agree_frac: float) -> Dict[str, Any]:
    idx = _index(rows)
    pairs = _pairs(idx, treat, control, crit.metric)
    n = len(pairs)
    deltas = np.array([crit.orient(a - b) for _, a, b in pairs], dtype=float)
    sigma = _sigma_seed(rows, control, crit.metric)
    out = {"metric": crit.metric, "treatment": treat, "control": control, "n_pairs": n,
           "mean_delta": float(deltas.mean()) if n else float("nan"),
           "sd_delta": float(deltas.std(ddof=1)) if n > 1 else float("nan"),
           "sigma_seed": sigma, "mde": crit.mde,
           "n_positive": int((deltas > 0).sum()), "n_negative": int((deltas < 0).sum())}
    if n < 2:
        out["verdict"] = "INCONCLUSIVE"
        return out
    thr = max(sigma if math.isfinite(sigma) else 0.0, crit.mde)
    need = math.ceil(min_agree_frac * n - 1e-9)
    m = out["mean_delta"]
    if m > thr and out["n_positive"] >= need:
        out["verdict"] = "VALUE"
    elif m < -thr and out["n_negative"] >= need:
        out["verdict"] = "HARMFUL"
    elif abs(m) <= crit.mde:
        out["verdict"] = "NEUTRAL"
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def _combine_primaries(verdicts: Iterable[str]) -> str:
    v = list(verdicts)
    if not v:
        return "INCONCLUSIVE"
    if "HARMFUL" in v:
        return "HARMFUL"
    if "VALUE" in v:
        return "VALUE"
    if all(x == "NEUTRAL" for x in v):
        return "NEUTRAL"
    return "INCONCLUSIVE"


def _combine_modes(verdicts: Iterable[str]) -> str:
    s = set(verdicts)
    if not s:
        return "INCONCLUSIVE"
    for good in ("VALUE", "HARMFUL"):
        if s <= {good, "NEUTRAL"} and good in s:
            return good
    if s == {"NEUTRAL"}:
        return "NEUTRAL"
    return "INCONCLUSIVE"


def _guardrail_breaches(rows, treat, control, guardrails):
    breaches = []
    idx = _index(rows)
    for g in guardrails:
        crit = MetricCriterion(g["metric"], bool(g.get("higher_is_better", True)))
        pairs = _pairs(idx, treat, control, crit.metric)
        if len(pairs) < 2:
            continue
        mean = float(np.mean([crit.orient(a - b) for _, a, b in pairs]))
        if mean < -float(g.get("tolerance", 0.0)):
            breaches.append({"metric": crit.metric, "mean_delta": mean, "tolerance": float(g.get("tolerance", 0.0))})
    return breaches


def analyze(rows: List[Dict[str, Any]], spec: AblationSpec, criteria: Criteria) -> Dict[str, Any]:
    """Per-term verdicts from the completed rows (see the module docstring)."""
    conds = {c.name for c in spec.conditions()}
    out: Dict[str, Any] = {"terms": {}, "family": None}
    for term in spec.terms:
        modes = {}
        for mode, treat, control in (("leave_one_in", f"only:{term}", "all_off"),
                                     ("leave_one_out", "all_on", f"without:{term}")):
            if treat not in conds or control not in conds:
                continue
            comps = [compare(rows, treat, control, c, criteria.min_agree_frac) for c in criteria.for_term(term)]
            breaches = _guardrail_breaches(rows, treat, control, criteria.guardrails)
            verdict = _combine_primaries(c["verdict"] for c in comps)
            if verdict == "VALUE" and breaches:
                verdict = "INCONCLUSIVE"
            modes[mode] = {"verdict": verdict, "metrics": comps, "guardrail_breaches": breaches}
        out["terms"][term] = {"verdict": _combine_modes(m["verdict"] for m in modes.values()), "modes": modes}
    if {"all_on", "all_off"} <= conds:
        comps = [compare(rows, "all_on", "all_off", c, criteria.min_agree_frac) for c in criteria.for_term(None)]
        breaches = _guardrail_breaches(rows, "all_on", "all_off", criteria.guardrails)
        verdict = _combine_primaries(c["verdict"] for c in comps)
        if verdict == "VALUE" and breaches:
            verdict = "INCONCLUSIVE"
        out["family"] = {"verdict": verdict, "metrics": comps, "guardrail_breaches": breaches}
    return out


# ------------------------------------------------------------------ outputs
def write_results(rows: List[Dict[str, Any]], out_dir) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "results.csv"
    if not rows:
        return path
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_summary(analysis: Dict[str, Any], out_dir) -> Path:
    path = Path(out_dir) / "summary.csv"
    fields = ["term", "mode", "metric", "treatment", "control", "n_pairs", "mean_delta", "sd_delta", "sigma_seed",
              "mde", "n_positive", "n_negative", "verdict", "mode_verdict", "term_verdict"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for term, t in analysis["terms"].items():
            for mode, m in t["modes"].items():
                for c in m["metrics"]:
                    w.writerow({"term": term, "mode": mode, **{k: c[k] for k in fields if k in c},
                                "mode_verdict": m["verdict"], "term_verdict": t["verdict"]})
        fam = analysis.get("family")
        if fam:
            for c in fam["metrics"]:
                w.writerow({"term": "family", "mode": "all_on_vs_all_off", **{k: c[k] for k in fields if k in c},
                            "mode_verdict": fam["verdict"], "term_verdict": fam["verdict"]})
    return path


def _fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def write_report(analysis: Dict[str, Any], rows: List[Dict[str, Any]], spec: AblationSpec, criteria: Criteria,
                 out_dir, scale: str) -> Path:
    L = [f"# Ablation `{spec.name}` - scale `{scale}`", "",
         f"{len(rows)} completed runs; terms {', '.join(spec.terms)}; seeds {spec.seeds}; periods "
         f"{', '.join(f'{k} (fold {v})' for k, v in spec.periods.items())}; lambda calibration: {spec.calibrate}.",
         "", "Deltas are paired over (seed, period) and oriented so that **positive = the term helps**. "
         f"A verdict of VALUE needs mean delta > max(seed sigma, MDE) with at least "
         f"{criteria.min_agree_frac:.0%} of pairs agreeing and no guard-rail breach (criteria pre-registered in "
         "`configs/ablation_criteria.yaml`).", "", "## Verdicts", "",
         "| term | leave-one-in | leave-one-out | verdict |", "|---|---|---|---|"]
    for term, t in analysis["terms"].items():
        li = t["modes"].get("leave_one_in", {}).get("verdict", "-")
        lo = t["modes"].get("leave_one_out", {}).get("verdict", "-")
        L.append(f"| `{term}` | {li} | {lo} | **{t['verdict']}** |")
    if analysis.get("family"):
        L.append(f"| family (all_on vs all_off) | | | **{analysis['family']['verdict']}** |")
    L += ["", "## Per-metric deltas", "",
          "| term | mode | metric | pairs | mean delta | sd | seed sigma | MDE | +/- | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for term, t in analysis["terms"].items():
        for mode, m in t["modes"].items():
            for c in m["metrics"]:
                L.append(f"| `{term}` | {mode} | `{c['metric']}` | {c['n_pairs']} | {_fmt(c['mean_delta'])} | "
                         f"{_fmt(c['sd_delta'])} | {_fmt(c['sigma_seed'])} | {c['mde']:g} | "
                         f"{c['n_positive']}/{c['n_negative']} | {c['verdict']} |")
            for b in m["guardrail_breaches"]:
                L.append(f"| `{term}` | {mode} | guard-rail `{b['metric']}` | | {_fmt(b['mean_delta'])} | | | "
                         f"tol {b['tolerance']:g} | | BREACH |")
    fam = analysis.get("family")
    if fam:
        for c in fam["metrics"]:
            L.append(f"| family | all_on vs all_off | `{c['metric']}` | {c['n_pairs']} | {_fmt(c['mean_delta'])} | "
                     f"{_fmt(c['sd_delta'])} | {_fmt(c['sigma_seed'])} | {c['mde']:g} | "
                     f"{c['n_positive']}/{c['n_negative']} | {c['verdict']} |")
    L += ["", "## Condition means", ""]
    metrics = sorted({c.metric for cs in criteria.primary.values() for c in cs})
    L.append("| condition | n | " + " | ".join(f"`{m}`" for m in metrics) + " |")
    L.append("|---|---|" + "---|" * len(metrics))
    for cond in [c.name for c in spec.conditions()]:
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        vals = []
        for m in metrics:
            v = [r[m] for r in rs if r.get(m) is not None and math.isfinite(r[m])]
            vals.append(f"{np.mean(v):.4f}" if v else "nan")
        L.append(f"| `{cond}` | {len(rs)} | " + " | ".join(vals) + " |")
    L += ["", "## Runs", "", "| key | run id | epochs | minutes |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["key"]):
        L.append(f"| {r['key']} | `{r.get('run_id', '')}` | {r.get('epochs_run', '')} | "
                 f"{r.get('wall_s', 0) / 60:.1f} |")
    path = Path(out_dir) / "report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def summarize_dir(spec: AblationSpec, criteria: Criteria, out_dir, scale: str) -> Dict[str, Any]:
    rows = load_rows(out_dir)
    write_results(rows, out_dir)
    analysis = analyze(rows, spec, criteria)
    (Path(out_dir) / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float), encoding="utf-8")
    write_summary(analysis, out_dir)
    write_report(analysis, rows, spec, criteria, out_dir, scale)
    return analysis


# ------------------------------------------------------------------ cell entry point
def _cell_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m neural_trade.experiments.ablation cell")
    ap.add_argument("cmd", choices=["cell"])
    ap.add_argument("--spec-json", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--row-out", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)
    spec = AblationSpec(**json.loads(Path(args.spec_json).read_text(encoding="utf-8")))
    cell = next(c for c in spec.cells() if c.key == args.key)
    frozen_path = Path(args.out) / "frozen_lambdas.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    row = execute_cell(spec, cell, args.scale, args.out, frozen, args.csv)
    Path(args.row_out).write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_cell_main())

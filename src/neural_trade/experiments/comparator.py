"""The paired comparator for "A beats B" verdicts (D-025, NT-032).

D-025: a verdict over two scenarios (learned against frozen, a loss term on against off, any two
engine scenarios) rests on a paired test over ``(seed, fold)`` pairs on the same blocks, plus a
minimum practical effect fixed *before* the compared runs start, not a point tolerance on one run
(the v1 physics ablation withdrew a family on exactly that: h1 AUC -0.0119 against a 0.01 tolerance,
D-003, while identical GPU runs differ by 0.01-0.05 AUC, NT-003).

    from neural_trade.experiments.comparator import CompareSpec, compare
    spec = CompareSpec.from_yaml("configs/compares/learned_vs_frozen.yaml")
    result = compare(spec)
    print(result.to_markdown())

A :class:`CompareSpec` is a pre-registered YAML: the metric, its direction, the minimum effect, the
judgement folds (D-025: folds no earlier choice used) and the guard-rails, all fixed before the
runs it will compare start (``registered_utc``). :func:`compare` reads the engine's run store
(NT-026, ``experiments.store``), pairs the two scenarios' runs by ``(seed, fold)``, refuses a pair
whose dataset or setup fingerprint differs between A and B or whose fold is not a judgement fold,
refuses the whole comparison when fewer than ``min_pairs`` pairs survive or when the spec's
registration postdates a compared run's start (the spec "changed after the first run started"), and
returns the paired difference, its interval, the verdict and the guard-rail verdicts, all by the
same paired test (D-025).

D-037 (the window-free plan) asks for several refinements beyond D-025's original text; the ones a
general-purpose comparator can offer are implemented as building blocks a spec opts into
(``metric_kind: log_ratio``, ``estimator: hodges_lehmann``, ``non_inferiority_margin``, ``looks``),
plus the standalone helpers :func:`per_fold_retention` and :func:`intersection_union_verdict`. What
is deferred, and why, is in the NT-032 backlog entry and this module's ``DEFERRED`` docstring below.

DEFERRED (D-037 amendment; not implemented here):
    - **Contention and re-time metadata.** The plan's A/B-1 wants each run's GPU-contention state
      (how many concurrent training processes were sharing the card, NT-035) recorded and used to
      flag or re-time a pair. No run directory carries that field yet (the runner's per-run meta.json
      does not record concurrency): writing it is a runner/experimenter change, out of this item's
      files (``training/``, ``models/``, the runner's own instrumentation, none of them touched here).
      A follow-up item can add ``meta.json["engine"]["contention"]`` and extend :func:`pair_runs` to
      surface or refuse on it; nothing here requires or simulates it.
    - **The literal A/B-1 primary metric** ``r_f = d_f - E_A,f / 3`` (a CRPS-edge-specific retention
      formula with a baseline term ``E_A,f`` computed from A/B-1's own scoring code). The generic
      ``per_fold_retention(diff_by_fold, baseline_by_fold, denom)`` below is the building block; the
      A/B-1 study's own spec supplies ``baseline_by_fold`` and ``denom=3`` when it is written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.experiments.scenario import short_hash
from neural_trade.experiments.store import RunStore
from neural_trade.metrics.statistics import hodges_lehmann, paired_t_ci, pocock_alpha, wilcoxon_hl_ci

DIRECTIONS = ("higher_better", "lower_better")
METRIC_KINDS = ("diff", "log_ratio")
ESTIMATORS = ("mean", "hodges_lehmann")
TOP_KEYS = ("name", "description", "scenario_a", "scenario_b", "metric", "direction", "min_effect", "alpha",
           "registered_utc", "judgment_folds", "min_pairs", "pairs_planned", "looks", "look_index", "metric_kind",
           "estimator", "non_inferiority_margin", "guard_rails", "route_owner_metrics", "noise_sd", "seed_sd",
           "block_sd", "root")
GUARD_RAIL_KEYS = ("metric", "direction", "max_degradation", "metric_kind")


class CompareError(InvalidConfigurationError):
    """A compare spec is malformed, or the comparator refuses to compare (too few pairs, a fold that
    is not a judgement fold, a mismatched fingerprint, or a spec registered after a run started)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------- spec
@dataclass
class GuardRail:
    metric: str
    direction: str = "higher_better"
    max_degradation: float = 0.0            # A may be worse than B by at most this much (metric units)
    metric_kind: str = "diff"

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise CompareError(f"guard_rails: direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.metric_kind not in METRIC_KINDS:
            raise CompareError(f"guard_rails: metric_kind must be one of {METRIC_KINDS}, got {self.metric_kind!r}")
        if self.max_degradation < 0:
            raise CompareError("guard_rails: max_degradation must be >= 0")


@dataclass
class CompareSpec:
    """A pre-registered comparison of two engine scenarios (NT-026 run store), A and B.

    ``registered_utc`` is the pre-registration timestamp: :func:`compare` refuses a comparison where
    a paired run started (``meta.json["created_utc"]``) before this timestamp, i.e. a spec written or
    edited after GPU time had already begun (D-025's "fixed before the run"). Set it explicitly to
    the moment the spec was committed; leaving it unset stamps "now", which is only correct when the
    spec is loaded and used to compare in the same act of registration (tests, and specs compared
    immediately after being written).
    """
    name: str
    scenario_a: str
    scenario_b: str
    metric: str
    min_effect: float
    judgment_folds: Tuple[int, ...]
    description: str = ""
    direction: str = "higher_better"
    alpha: float = 0.05
    registered_utc: str = field(default_factory=_utcnow)
    min_pairs: int = 5
    pairs_planned: Optional[int] = None
    looks: int = 1
    look_index: int = 1
    metric_kind: str = "diff"
    estimator: str = "mean"
    non_inferiority_margin: Optional[float] = None
    guard_rails: Tuple[GuardRail, ...] = ()
    route_owner_metrics: Tuple[str, ...] = ()
    noise_sd: Optional[float] = None
    seed_sd: Optional[float] = None
    block_sd: Optional[float] = None
    root: str = "runs"

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise CompareError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.metric_kind not in METRIC_KINDS:
            raise CompareError(f"metric_kind must be one of {METRIC_KINDS}, got {self.metric_kind!r}")
        if self.estimator not in ESTIMATORS:
            raise CompareError(f"estimator must be one of {ESTIMATORS}, got {self.estimator!r}")
        if self.min_pairs < 5:
            raise CompareError("min_pairs must be >= 5 (D-025's lead reading: at least 5 (seed, fold) pairs)")
        if not self.judgment_folds:
            raise CompareError("judgment_folds must name at least one fold (D-025: folds no earlier choice used)")
        if self.looks < 1:
            raise CompareError("looks must be >= 1")
        if not (1 <= self.look_index <= self.looks):
            raise CompareError("look_index must be between 1 and looks")
        self.judgment_folds = tuple(int(f) for f in self.judgment_folds)
        self.guard_rails = tuple(
            g if isinstance(g, GuardRail) else GuardRail(**g) for g in self.guard_rails)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "CompareSpec":
        unknown = sorted(set(d) - set(TOP_KEYS))
        if unknown:
            raise CompareError(f"compare spec: unknown key(s) {unknown}, expected a subset of {TOP_KEYS}")
        d = dict(d)
        for req in ("name", "scenario_a", "scenario_b", "metric", "min_effect", "judgment_folds"):
            if req not in d:
                raise CompareError(f"compare spec: missing required key {req!r}")
        guard_rails = d.get("guard_rails") or []
        for g in guard_rails:
            unknown_g = sorted(set(g) - set(GUARD_RAIL_KEYS))
            if unknown_g:
                raise CompareError(f"guard_rails: unknown key(s) {unknown_g}, expected a subset of {GUARD_RAIL_KEYS}")
        d["guard_rails"] = guard_rails
        return CompareSpec(**d)

    @staticmethod
    def from_yaml(path) -> "CompareSpec":
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise CompareError(f"{path}: a compare spec must be a YAML mapping")
        return CompareSpec.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in TOP_KEYS}
        d["judgment_folds"] = list(d["judgment_folds"])
        d["guard_rails"] = [vars(g) for g in d["guard_rails"]]
        return d

    @property
    def spec_hash(self) -> str:
        """The pre-registration hash: everything but nothing (``registered_utc`` is part of the
        registration, so a spec edited and re-registered after runs already started hashes
        differently, and :func:`compare` catches the timing directly via ``registered_utc``)."""
        return short_hash(self.to_dict())


# ------------------------------------------------------------------------------------------- pairing
@dataclass
class Pair:
    seed: int
    fold: int
    a_run_id: str
    b_run_id: str
    a_value: float
    b_value: float
    a_created_utc: Optional[str]
    b_created_utc: Optional[str]
    diff: float                     # signed so that positive = A better (direction already applied)


@dataclass
class Excluded:
    seed: Optional[int]
    fold: Optional[int]
    run_id: Optional[str]
    reason: str


FINGERPRINT_FIELDS = ("dataset_sha256", "bar_minutes", "horizon_steps")


def _rows_by_seed_fold(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """The latest (by created_utc) done row per (seed, fold); everything else lists as a duplicate."""
    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in rows:
        if r.get("status") != "done" or r.get("seed") is None or r.get("fold") is None:
            continue
        key = (int(r["seed"]), int(r["fold"]))
        cur = best.get(key)
        if cur is None or (r.get("created_utc") or "") > (cur.get("created_utc") or ""):
            best[key] = r
    return best


def _signed_diff(a: float, b: float, direction: str, metric_kind: str) -> Optional[float]:
    if metric_kind == "log_ratio":
        if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
            return None
        raw = float(np.log(a / b))
    else:
        raw = float(a - b)
    return raw if direction == "higher_better" else -raw


def pair_runs(spec: CompareSpec, *, metric: Optional[str] = None, direction: Optional[str] = None,
             metric_kind: Optional[str] = None) -> Tuple[List[Pair], List[Excluded]]:
    """Pair scenario A's and B's done runs by ``(seed, fold)``, restricted to ``spec.judgment_folds``,
    refusing (excluding, with a reason) a pair whose dataset/setup fingerprint differs between A and
    B or whose metric is missing. ``metric``/``direction``/``metric_kind`` default to the spec's own
    (a guard-rail passes its own)."""
    metric = metric if metric is not None else spec.metric
    direction = direction if direction is not None else spec.direction
    metric_kind = metric_kind if metric_kind is not None else spec.metric_kind
    store = RunStore(spec.root)
    rows_a = store.sync(spec.scenario_a)
    rows_b = store.sync(spec.scenario_b)
    by_a, by_b = _rows_by_seed_fold(rows_a), _rows_by_seed_fold(rows_b)
    pairs: List[Pair] = []
    excluded: List[Excluded] = []
    for key in sorted(set(by_a) & set(by_b)):
        seed, fold = key
        a, b = by_a[key], by_b[key]
        if fold not in spec.judgment_folds:
            excluded.append(Excluded(seed, fold, a["run_id"], f"fold {fold} is not a judgement fold "
                                     f"{list(spec.judgment_folds)}"))
            continue
        mismatched = [f for f in FINGERPRINT_FIELDS if a.get(f) != b.get(f)]
        if mismatched:
            excluded.append(Excluded(seed, fold, a["run_id"],
                                     f"fingerprint mismatch on {mismatched}: A={[a.get(f) for f in mismatched]} "
                                     f"B={[b.get(f) for f in mismatched]}"))
            continue
        av = store.index.scores(a["run_id"]).get(metric)
        bv = store.index.scores(b["run_id"]).get(metric)
        if av is None or bv is None:
            excluded.append(Excluded(seed, fold, a["run_id"], f"metric {metric!r} missing (A={av}, B={bv})"))
            continue
        diff = _signed_diff(av, bv, direction, metric_kind)
        if diff is None or not np.isfinite(diff):
            excluded.append(Excluded(seed, fold, a["run_id"], f"non-finite {metric_kind} of A={av}, B={bv}"))
            continue
        pairs.append(Pair(seed, fold, a["run_id"], b["run_id"], float(av), float(bv),
                          a.get("created_utc"), b.get("created_utc"), diff))
    only_a = sorted(set(by_a) - set(by_b))
    only_b = sorted(set(by_b) - set(by_a))
    for seed, fold in only_a:
        excluded.append(Excluded(seed, fold, by_a[(seed, fold)]["run_id"], "no matching (seed, fold) run in B"))
    for seed, fold in only_b:
        excluded.append(Excluded(seed, fold, by_b[(seed, fold)]["run_id"], "no matching (seed, fold) run in A"))
    return pairs, excluded


# ------------------------------------------------------------------------------------- paired verdict
def _estimate(diffs: np.ndarray, *, estimator: str, alpha: float) -> Dict[str, float]:
    if estimator == "hodges_lehmann":
        est = hodges_lehmann(diffs)
        lo, hi = wilcoxon_hl_ci(diffs, alpha=alpha)
        return {"estimate": est, "ci_lo": lo, "ci_hi": hi, "statistic": float("nan")}
    m, lo, hi, t = paired_t_ci(diffs, alpha=alpha)
    return {"estimate": m, "ci_lo": lo, "ci_hi": hi, "statistic": t}


def _verdict_from_ci(lo: float, hi: float, min_effect: float, a_name: str, b_name: str) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "inconclusive"
    if lo >= min_effect:
        return f"{a_name} beats {b_name}"
    if hi <= -min_effect:
        return f"{b_name} beats {a_name}"
    return "inconclusive"


def non_inferiority_verdict(lo: float, hi: float, margin: float) -> str:
    """"pass" (A is not inferior to B by more than ``margin``), "breach", or "undecided", judged by
    the same paired interval (D-025) rather than a point check against the margin."""
    if np.isnan(lo) or np.isnan(hi) or margin < 0:
        return "undecided"
    if lo > -margin:
        return "pass"
    if hi < -margin:
        return "breach"
    return "undecided"


def per_fold_retention(diff_by_fold: Mapping[int, Sequence[float]], baseline_by_fold: Mapping[int, float],
                       *, denom: float = 1.0, alpha: float = 0.05) -> Dict[str, Any]:
    """The D-037 per-fold retention building block: ``r_f = mean(diff_by_fold[f]) - baseline_by_fold[f]
    / denom`` for every fold present in both mappings, then the one-sided (upper-bound) paired
    interval of ``r_f`` across folds (non-inferior iff the upper bound is below 0). Generic: a study's
    own spec supplies ``baseline_by_fold`` (e.g. A/B-1's per-fold CRPS edge ``E_A,f``) and ``denom``
    (A/B-1 uses 3, its number of horizons)."""
    folds = sorted(set(diff_by_fold) & set(baseline_by_fold))
    if not folds:
        return {"folds": [], "r_f": [], "mean_r": float("nan"), "upper_bound": float("nan"), "verdict": "undecided"}
    r_f = [float(np.mean(diff_by_fold[f])) - float(baseline_by_fold[f]) / denom for f in folds]
    if len(r_f) < 2:
        mean_r = r_f[0] if r_f else float("nan")
        lower = upper = mean_r
    else:
        from scipy import stats as _st
        arr = np.asarray(r_f, float)
        mean_r = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        tcrit = float(_st.t.ppf(1.0 - alpha, len(arr) - 1))    # one-sided: only the bound that matters
        lower, upper = mean_r - tcrit * se, mean_r + tcrit * se
    # non-inferior iff the upper bound is below 0 (r_f cannot be positive with any credibility);
    # breach iff the lower bound is above 0 (r_f cannot be zero-or-negative with any credibility).
    verdict = "non_inferior" if upper < 0 else ("breach" if lower > 0 else "undecided")
    return {"folds": folds, "r_f": r_f, "mean_r": mean_r, "lower_bound": lower, "upper_bound": upper,
           "verdict": verdict}


def intersection_union_verdict(component_verdicts: Mapping[str, str], *,
                               owner_route: Sequence[str] = ()) -> Dict[str, Any]:
    """D-037's intersection-union combination: ADOPT only when every named component is "pass" (or a
    winning "A beats B" / "non_inferior"); a breach on a name in ``owner_route`` (D-018: a speed
    regression the lead cannot accept on its own) routes to the owner instead of rejecting outright."""
    ok = {"pass", "non_inferior"}
    breaches = [k for k, v in component_verdicts.items() if v == "breach"]
    routed = [k for k in breaches if k in owner_route]
    hard_breaches = [k for k in breaches if k not in owner_route]
    if hard_breaches:
        overall = "reject"
    elif routed:
        overall = "needs_owner"
    elif all(v in ok for v in component_verdicts.values()):
        overall = "adopt"
    else:
        overall = "undecided"
    return {"overall": overall, "breaches": breaches, "routed_to_owner": routed,
           "components": dict(component_verdicts)}


# -------------------------------------------------------------------------------------- the result
@dataclass
class CompareResult:
    spec: CompareSpec
    pairs: List[Pair]
    excluded: List[Excluded]
    estimate: Dict[str, float]
    verdict: str
    refusal_reason: Optional[str]
    guard_rail_results: List[Dict[str, Any]]
    non_inferiority: Optional[Dict[str, Any]]
    alpha_used: float
    generated_utc: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(), "spec_hash": self.spec.spec_hash,
            "n_pairs": len(self.pairs),
            "pairs": [vars(p) for p in self.pairs],
            "excluded_pairs": [vars(e) for e in self.excluded],
            "estimator": self.spec.estimator, "alpha_used": self.alpha_used,
            "estimate": self.estimate, "verdict": self.verdict, "refusal_reason": self.refusal_reason,
            "guard_rails": self.guard_rail_results, "non_inferiority": self.non_inferiority,
            "generated_utc": self.generated_utc,
        }

    def to_markdown(self) -> str:
        s = self.spec
        lines = [f"# {s.name}", "", s.description, "",
                f"**A** = `{s.scenario_a}`, **B** = `{s.scenario_b}` · metric `{s.metric}` "
                f"({s.direction}, {s.metric_kind}) · minimum effect {s.min_effect:g} · "
                f"judgement folds {list(s.judgment_folds)} · alpha {s.alpha:g}", "",
                f"spec hash `{s.spec_hash}` · registered {s.registered_utc}", ""]
        if self.refusal_reason:
            lines += [f"**Refused: {self.refusal_reason}**", ""]
            return "\n".join(lines)
        est = self.estimate
        lines += [f"**Verdict: {self.verdict}** ({s.estimator}: {est['estimate']:.4g}, "
                 f"95% CI [{est['ci_lo']:.4g}, {est['ci_hi']:.4g}], n = {len(self.pairs)} pairs)", "",
                 "| seed | fold | A | B | diff |", "|---|---|---|---|---|"]
        for p in self.pairs:
            lines.append(f"| {p.seed} | {p.fold} | {p.a_value:.4g} | {p.b_value:.4g} | {p.diff:.4g} |")
        if self.excluded:
            lines += ["", f"{len(self.excluded)} pair(s) excluded:", ""]
            for e in self.excluded:
                lines.append(f"- seed {e.seed}, fold {e.fold}: {e.reason}")
        if self.guard_rail_results:
            lines += ["", "## Guard-rails", "", "| metric | verdict | estimate | CI |", "|---|---|---|---|"]
            for g in self.guard_rail_results:
                lines.append(f"| {g['metric']} | {g['verdict']} | {g['estimate']['estimate']:.4g} | "
                             f"[{g['estimate']['ci_lo']:.4g}, {g['estimate']['ci_hi']:.4g}] |")
        if self.non_inferiority is not None:
            lines += ["", f"Non-inferiority (margin {s.non_inferiority_margin:g}): "
                          f"**{self.non_inferiority['verdict']}**"]
        return "\n".join(lines)


def _refuse(spec: CompareSpec, reason: str) -> CompareResult:
    return CompareResult(spec, [], [], {"estimate": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                                        "statistic": float("nan")}, "refused", reason, [], None, spec.alpha)


def compare(spec: CompareSpec) -> CompareResult:
    """Run the comparison a pre-registered :class:`CompareSpec` describes: pair, check pre-registration
    timing and the pair count, compute the paired estimate and verdict, and judge every guard-rail by
    the same paired test (D-025)."""
    pairs, excluded = pair_runs(spec)
    if len(pairs) < spec.min_pairs:
        return _refuse(spec, f"only {len(pairs)} pair(s) survived (need >= {spec.min_pairs}); "
                             f"{len(excluded)} excluded, see excluded_pairs")
    starts = [p.a_created_utc for p in pairs if p.a_created_utc] + [p.b_created_utc for p in pairs if p.b_created_utc]
    if starts and min(starts) < spec.registered_utc:
        return _refuse(spec, f"a compared run started at {min(starts)}, before the spec's registration "
                             f"{spec.registered_utc}: the spec was written or changed after GPU time began")
    if spec.pairs_planned is not None and spec.looks <= 1 and len(pairs) != spec.pairs_planned:
        return _refuse(spec, f"{len(pairs)} pairs found but the spec pre-registered exactly "
                             f"{spec.pairs_planned} (no peeking): re-register before adding runs")
    alpha_used = pocock_alpha(spec.alpha, spec.looks) if spec.looks > 1 else spec.alpha
    diffs = np.array([p.diff for p in pairs], float)
    est = _estimate(diffs, estimator=spec.estimator, alpha=alpha_used)
    verdict = _verdict_from_ci(est["ci_lo"], est["ci_hi"], spec.min_effect, "A", "B")

    guard_results = []
    for g in spec.guard_rails:
        gp, gexcluded = pair_runs(spec, metric=g.metric, direction=g.direction, metric_kind=g.metric_kind)
        gp = [p for p in gp if any(pp.seed == p.seed and pp.fold == p.fold for pp in pairs)]
        if len(gp) < spec.min_pairs:
            guard_results.append({"metric": g.metric, "verdict": "undecided", "n_pairs": len(gp),
                                  "estimate": {"estimate": float("nan"), "ci_lo": float("nan"),
                                              "ci_hi": float("nan"), "statistic": float("nan")}})
            continue
        gdiffs = np.array([p.diff for p in gp], float)
        gest = _estimate(gdiffs, estimator=spec.estimator, alpha=alpha_used)
        gv = ("pass" if gest["ci_lo"] > -g.max_degradation else
             "breach" if gest["ci_hi"] < -g.max_degradation else "undecided")
        guard_results.append({"metric": g.metric, "verdict": gv, "n_pairs": len(gp), "estimate": gest})

    ni = None
    if spec.non_inferiority_margin is not None:
        ni = {"margin": spec.non_inferiority_margin,
             "verdict": non_inferiority_verdict(est["ci_lo"], est["ci_hi"], spec.non_inferiority_margin)}

    return CompareResult(spec, pairs, excluded, est, verdict, None, guard_results, ni, alpha_used)


# ----------------------------------------------------------------------------------- error simulation
def simulate_error_rates(spec: CompareSpec, *, n_pairs: Optional[int] = None, n_sim: int = 1000,
                         seed: int = 0) -> Dict[str, Any]:
    """Calibrated null/power simulation (D-025's acceptance (4), D-037's "calibrated to the measured
    variance components"): ``n_sim`` (>= 1000) synthetic comparisons of ``n_pairs`` pairs each, drawn
    from Normal(0, noise_sd) for the false-'beats'-rate check and Normal(2 x min_effect, noise_sd) for
    the power check.

    ``noise_sd`` is ``spec.noise_sd`` if given, else ``sqrt(seed_sd^2 + block_sd^2)`` from the spec's
    two variance components (D-037: seed noise between runs, 80-bar block noise within a run), else a
    ValueError naming what to set. Vectorised (fast: 1000 x 20 pairs runs in well under a second), so
    the fast suite runs the full simulation rather than a reduced one.
    """
    if spec.noise_sd is not None:
        noise_sd = spec.noise_sd
    elif spec.seed_sd is not None and spec.block_sd is not None:
        noise_sd = float(np.sqrt(spec.seed_sd ** 2 + spec.block_sd ** 2))
    else:
        raise CompareError("simulate_error_rates needs spec.noise_sd, or both spec.seed_sd and spec.block_sd")
    n = n_pairs if n_pairs is not None else (spec.pairs_planned or spec.min_pairs)
    n_sim = max(1000, int(n_sim))
    rng = np.random.default_rng(seed)
    alpha_used = pocock_alpha(spec.alpha, spec.looks) if spec.looks > 1 else spec.alpha

    from scipy import stats as _st
    tcrit = float(_st.t.ppf(1.0 - alpha_used / 2.0, n - 1)) if n > 1 else float("nan")

    def _rates(effect: float) -> Tuple[np.ndarray, np.ndarray]:
        d = rng.normal(effect, noise_sd, size=(n_sim, n))
        mean = d.mean(axis=1)
        se = d.std(axis=1, ddof=1) / np.sqrt(n)
        lo, hi = mean - tcrit * se, mean + tcrit * se
        return lo >= spec.min_effect, hi <= -spec.min_effect

    beats_a0, beats_b0 = _rates(0.0)
    false_beats = beats_a0 | beats_b0
    false_rate = float(false_beats.mean())
    mc_error = float(np.sqrt(false_rate * (1 - false_rate) / n_sim))

    beats_a2, _ = _rates(2.0 * spec.min_effect)
    power = float(beats_a2.mean())
    power_mc_error = float(np.sqrt(power * (1 - power) / n_sim))

    return {"n_sim": n_sim, "n_pairs": n, "noise_sd": noise_sd, "alpha_used": alpha_used, "seed": seed,
           "false_beats_rate": false_rate, "false_beats_mc_error": mc_error,
           "power_at_2x_min_effect": power, "power_mc_error": power_mc_error}


__all__ = ["CompareError", "CompareResult", "CompareSpec", "Excluded", "GuardRail", "Pair", "compare",
          "intersection_union_verdict", "non_inferiority_verdict", "pair_runs", "per_fold_retention",
          "simulate_error_rates"]

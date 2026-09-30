"""The paired comparator for "A beats B" verdicts (D-025, NT-032; D-046 supersedes D-025's "at
least 5 (seed, fold) pairs" reading with "at least 5 judgement folds").

D-025: a verdict over two scenarios (learned against frozen, a loss term on against off, any two
engine scenarios) rests on a paired test plus a minimum practical effect fixed *before* the compared
runs start, not a point tolerance on one run (the v1 physics ablation withdrew a family on exactly
that: h1 AUC -0.0119 against a 0.01 tolerance, D-003, while identical GPU runs differ by 0.01-0.05
AUC, NT-003).

D-046 (QA repair round 1 on NT-032): the unit of inference is the **judgement fold**, not the
(seed, fold) pair. Seeds trained on the same fold's blocks share that fold's 80-bar block noise (the
same bars, the same realised path), so treating them as independent inflates the false-"beats" rate
well past 5% (measured: 0.069 / 0.146 / 0.314 at min_effect 0.01 / 0.005 / 0 for one fold x 5 seeds).
A verdict therefore averages each fold's paired differences over its seeds first, and runs the paired
test over folds (``min_folds``, at least 5, never lower).

    from neural_trade.experiments.comparator import CompareSpec, compare
    spec = CompareSpec.from_yaml("configs/compares/learned_vs_frozen.yaml")
    result = compare(spec)
    print(result.to_markdown())

A :class:`CompareSpec` is a pre-registered YAML: the metric, its direction, the minimum effect, the
judgement folds (D-025: folds no earlier choice used) and the guard-rails, all fixed before the
runs it will compare start. Registration is two separate, both-enforced things:

1. ``registered_utc`` (required; there is no "now" default: a spec that does not declare it is not
   pre-registered) must predate every compared run's ``created_utc`` (:func:`compare` refuses
   otherwise). When the spec file has a git history, :func:`CompareSpec.from_yaml` prefers the
   file's last commit time over the self-declared string (an objective record beats a string that
   could be edited without updating), recorded as ``registered_utc_source``.
2. **The spec's content** is locked by a sidecar (``<root>/compares/<name>/registration.json``)
   written the first time a name is compared: a later call whose ``spec_hash`` differs (the min
   effect, a guard-rail, anything) is refused, even if ``registered_utc`` is untouched. This is what
   catches "edited the spec, kept the timestamp".

:func:`compare` reads the engine's run store (NT-026, ``experiments.store``) **read-only** (it never
touches ``index.sqlite``: it walks the run directories and reads their light JSON files directly, the
same files the index is built from), pairs the two scenarios' runs by ``(seed, fold)`` restricted to
one named configuration per side, refuses a pair whose dataset, setup or block fingerprint differs
between A and B or whose fold is not a judgement fold, refuses the whole comparison when fewer than
``min_folds`` judgement folds survive or the pre-registration checks above fail, and returns the
per-fold paired difference, its interval, the verdict and the guard-rail verdicts, all by the same
paired test over folds (D-025, D-046).

D-037 (the window-free plan) asks for several refinements beyond D-025's original text; the ones a
general-purpose comparator can offer are implemented as building blocks a spec opts into
(``metric_kind: log_ratio``, ``estimator: hodges_lehmann``, ``non_inferiority_margin``, ``looks``),
plus the standalone helpers :func:`per_fold_retention` and :func:`intersection_union_verdict`. Paired
coverage (a conformal-coverage metric compared like any other) needs no special code: any
``result.json`` score key works, including a coverage metric. What is deferred, and why, is below.

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
    - **Infinite pairs kept in the rank statistic.** A log-ratio or coverage pair that comes out
      non-finite (a zero denominator, a 0%/100% coverage edge) is excluded with a reason, not folded
      into the Hodges-Lehmann/Wilcoxon ranking as a censored extreme value. Doing that properly needs
      a rank statistic that treats +-inf as "beyond every finite value" rather than undefined, which
      is a distinct estimator, not a flag on this one; excluding is conservative (it only ever
      shrinks the fold count, never manufactures a spurious rank) but is not what D-037 envisions.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.experiments.scenario import short_hash
from neural_trade.experiments.store import RunStore, read_run
from neural_trade.metrics.statistics import hodges_lehmann, paired_t_ci, pocock_alpha, wilcoxon_hl_ci

DIRECTIONS = ("higher_better", "lower_better")
METRIC_KINDS = ("diff", "log_ratio")
ESTIMATORS = ("mean", "hodges_lehmann")
DUPLICATE_POLICIES = ("refuse", "latest", "average")
TOP_KEYS = ("name", "description", "scenario_a", "scenario_b", "configuration_a", "configuration_b",
           "duplicate_policy", "metric", "direction", "min_effect", "alpha", "registered_utc", "judgment_folds",
           "min_pairs", "min_folds", "pairs_planned", "looks", "look_index", "metric_kind", "estimator",
           "non_inferiority_margin", "guard_rails", "route_owner_metrics", "noise_sd", "seed_sd", "block_sd",
           "root")
GUARD_RAIL_KEYS = ("metric", "direction", "max_degradation", "metric_kind")
_TS_FORMATS = ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ")


class CompareError(InvalidConfigurationError):
    """A compare spec is malformed, or the comparator refuses to compare (too few judgement folds, a
    fold that is not a judgement fold, a mismatched fingerprint, an ambiguous configuration, or a
    spec that is not, or is no longer, honestly pre-registered)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(s: str) -> datetime:
    """A UTC timestamp in the run store's compact form (``created_utc``, ``20260930T081536Z``) or
    ISO 8601 (a spec's ``registered_utc``, ``2026-09-30T08:15:36Z`` or with a numeric offset)."""
    s = str(s).strip()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CompareError(f"cannot parse timestamp {s!r}: expected {_TS_FORMATS[0]!r} or ISO 8601") from exc


def _git_commit_time(path) -> Optional[str]:
    """The last git commit time of ``path`` (ISO 8601 with offset), or None (no git, not committed,
    git unavailable). :func:`CompareSpec.from_yaml` prefers this over a spec's self-declared
    ``registered_utc`` when it exists (RUNBOOK "Paired comparator")."""
    path = Path(path)
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", path.name], cwd=str(path.resolve().parent),
                             capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    ts = (out.stdout or "").strip()
    return ts if out.returncode == 0 and ts else None


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

    ``registered_utc`` is required (D-046): a spec that does not declare when it was pre-registered
    is not pre-registered. :func:`compare` refuses a comparison where a paired run started
    (``meta.json["created_utc"]``) before the *effective* registration time (:attr:`effective_registered_utc`:
    the git commit time of the spec file when :func:`from_yaml` found one, else ``registered_utc``
    itself), i.e. a spec written or edited after GPU time had already begun (D-025's "fixed before
    the run"). A second, independent check locks the spec's *content*: the first ``compare()`` call
    for a given ``name`` records its ``spec_hash`` in ``<root>/compares/<name>/registration.json``;
    a later call under the same name whose hash differs (anything edited, even with the same
    ``registered_utc``) is refused.
    """
    name: str
    scenario_a: str
    scenario_b: str
    metric: str
    min_effect: float
    judgment_folds: Tuple[int, ...]
    registered_utc: str
    description: str = ""
    configuration_a: Optional[str] = None
    configuration_b: Optional[str] = None
    duplicate_policy: str = "refuse"
    direction: str = "higher_better"
    alpha: float = 0.05
    min_pairs: int = 5
    min_folds: int = 5
    pairs_planned: Optional[Union[int, Sequence[int]]] = None
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
    # not spec keys: filled in by from_yaml (git commit time) or left as the declared value
    registered_utc_effective: Optional[str] = field(default=None, repr=False)
    registered_utc_source: str = field(default="declared", repr=False)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise CompareError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.metric_kind not in METRIC_KINDS:
            raise CompareError(f"metric_kind must be one of {METRIC_KINDS}, got {self.metric_kind!r}")
        if self.estimator not in ESTIMATORS:
            raise CompareError(f"estimator must be one of {ESTIMATORS}, got {self.estimator!r}")
        if self.duplicate_policy not in DUPLICATE_POLICIES:
            raise CompareError(f"duplicate_policy must be one of {DUPLICATE_POLICIES}, got "
                              f"{self.duplicate_policy!r}")
        if self.min_pairs < 5:
            raise CompareError("min_pairs must be >= 5")
        if self.min_folds < 5:
            raise CompareError("min_folds must be >= 5 (D-046: at least 5 judgement folds, never fewer)")
        if not self.judgment_folds:
            raise CompareError("judgment_folds must name at least one fold (D-025: folds no earlier choice used)")
        if not self.registered_utc:
            raise CompareError("registered_utc is required: a spec that does not declare when it was "
                              "pre-registered is not pre-registered (D-046)")
        _parse_utc(self.registered_utc)                            # raises CompareError if unparsable
        if self.looks < 1:
            raise CompareError("looks must be >= 1")
        if not (1 <= self.look_index <= self.looks):
            raise CompareError("look_index must be between 1 and looks")
        if isinstance(self.pairs_planned, (list, tuple)):
            if len(self.pairs_planned) != self.looks:
                raise CompareError(f"pairs_planned as a list must have one entry per look ({self.looks}), "
                                  f"got {len(self.pairs_planned)}")
            self.pairs_planned = tuple(int(p) for p in self.pairs_planned)
        elif self.pairs_planned is not None:
            self.pairs_planned = int(self.pairs_planned)
        self.judgment_folds = tuple(int(f) for f in self.judgment_folds)
        self.guard_rails = tuple(
            g if isinstance(g, GuardRail) else GuardRail(**g) for g in self.guard_rails)
        if self.registered_utc_effective is None:
            self.registered_utc_effective = self.registered_utc

    @property
    def effective_registered_utc(self) -> str:
        return self.registered_utc_effective or self.registered_utc

    @property
    def pairs_planned_for_this_look(self) -> Optional[int]:
        if self.pairs_planned is None:
            return None
        if isinstance(self.pairs_planned, tuple):
            return self.pairs_planned[self.look_index - 1]
        return self.pairs_planned

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "CompareSpec":
        unknown = sorted(set(d) - set(TOP_KEYS))
        if unknown:
            raise CompareError(f"compare spec: unknown key(s) {unknown}, expected a subset of {TOP_KEYS}")
        d = dict(d)
        for req in ("name", "scenario_a", "scenario_b", "metric", "min_effect", "judgment_folds", "registered_utc"):
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
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise CompareError(f"{path}: a compare spec must be a YAML mapping")
        spec = CompareSpec.from_dict(data)
        git_time = _git_commit_time(path)
        if git_time:
            spec.registered_utc_effective = git_time
            spec.registered_utc_source = "git_commit_time"
        return spec

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in TOP_KEYS}
        d["judgment_folds"] = list(d["judgment_folds"])
        d["guard_rails"] = [vars(g) for g in d["guard_rails"]]
        if isinstance(d["pairs_planned"], tuple):
            d["pairs_planned"] = list(d["pairs_planned"])
        return d

    @property
    def spec_hash(self) -> str:
        """The pre-registration hash of the spec's declared content (not the derived
        ``registered_utc_effective`` / ``_source``): :func:`compare` locks this in a sidecar the
        first time a ``name`` is compared and refuses any later call under the same name whose
        content hashes differently (D-046)."""
        return short_hash(self.to_dict())

    def registration_info(self) -> Dict[str, Any]:
        return {"declared": self.registered_utc, "effective": self.effective_registered_utc,
               "source": self.registered_utc_source}


def _registration_path(spec: CompareSpec) -> Path:
    return Path(spec.root) / "compares" / spec.name / "registration.json"


def _check_and_record_registration(spec: CompareSpec) -> Optional[str]:
    """None on success; a refusal reason when a prior registration's hash disagrees. The first call
    for ``spec.name`` writes the sidecar (the act of registration)."""
    path = _registration_path(spec)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if existing.get("spec_hash") != spec.spec_hash:
            return (f"the spec {spec.name!r} changed since it was registered ({path}): recorded hash "
                    f"{existing.get('spec_hash')!r}, current {spec.spec_hash!r}. Write a new name instead of "
                    f"editing a registered comparison")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": spec.name, "spec_hash": spec.spec_hash,
                                "registration": spec.registration_info(), "recorded_utc": _utcnow()},
                               indent=2), encoding="utf-8")
    return None


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


# Anchor fingerprint (D-037's "anchor hashes per block"): the run-store columns plus the judged
# out-of-sample block's own bounds (meta.json's "blocks" section, not carried into the store's row).
FINGERPRINT_FIELDS = ("dataset_sha256", "bar_minutes", "horizon_steps", "lookback")


def _load_scenario_rows(scenario: str, root) -> List[Dict[str, Any]]:
    """Every engine run directory's row + scores + block bounds, read directly from meta.json /
    result.json (never through the sqlite index: :func:`compare` must not write it, F below)."""
    store = RunStore(root)
    out = []
    for d in store.run_dirs(scenario):
        row, scores = read_run(d, store.root)
        try:
            meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        row = dict(row)
        row["_scores"] = scores
        row["_blocks"] = meta.get("blocks") or {}
        out.append(row)
    return out


def _block_fingerprint(row: Mapping[str, Any]):
    block = (row.get("_blocks") or {}).get("test") or {}
    if "start" in block and "stop" in block:
        return ("start_stop", block.get("start"), block.get("stop"))
    return ("timestamps", block.get("first_timestamp"), block.get("last_timestamp"))


def _group_rows(rows: Sequence[Dict[str, Any]], configuration: Optional[str]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """(seed, fold) -> {"kind": "ok", "row": ...} | {"kind": "duplicate", "rows": [...]} |
    {"kind": "ambiguous", "configurations": [...]}. "ambiguous": more than one configuration shares
    this (seed, fold) and none was named (``configuration`` is None): never silently averaged or
    latest-picked (that would silently mix configurations, D-046 point E)."""
    by_key: Dict[Tuple[int, int], Dict[Optional[str], List[Dict[str, Any]]]] = {}
    for r in rows:
        if r.get("status") != "done" or r.get("seed") is None or r.get("fold") is None:
            continue
        key = (int(r["seed"]), int(r["fold"]))
        by_key.setdefault(key, {}).setdefault(r.get("configuration"), []).append(r)
    resolved: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, cfgs in by_key.items():
        if configuration is not None:
            cfgs = {c: v for c, v in cfgs.items() if c == configuration}
            if not cfgs:
                continue
        if len(cfgs) > 1:
            resolved[key] = {"kind": "ambiguous", "configurations": sorted(c for c in cfgs if c is not None)}
            continue
        (_, cfg_rows), = cfgs.items()
        if len(cfg_rows) > 1:
            resolved[key] = {"kind": "duplicate",
                             "rows": sorted(cfg_rows, key=lambda r: r.get("created_utc") or "")}
        else:
            resolved[key] = {"kind": "ok", "row": cfg_rows[0]}
    return resolved


def _resolve_side(entry: Optional[Dict[str, Any]], policy: str) -> Tuple[Any, Optional[str]]:
    """(row-or-average-marker, reason). ``reason`` is None on success. Never drops a duplicate or an
    ambiguous configuration silently (D-046 point E): both are excluded with the names involved
    unless the spec's ``duplicate_policy`` resolves the duplicate case explicitly."""
    if entry is None:
        return None, "no matching (seed, fold) run"
    if entry["kind"] == "ambiguous":
        return None, f"multiple configurations share this (seed, fold) and none was named: " \
                     f"{entry['configurations']} (set configuration_a / configuration_b)"
    if entry["kind"] == "duplicate":
        rows = entry["rows"]
        ids = [r["run_id"] for r in rows]
        if policy == "latest":
            return rows[-1], None
        if policy == "average":
            return {"__average__": rows}, None
        return None, (f"{len(rows)} done runs for the same (seed, fold, configuration): {ids} "
                      f"(set duplicate_policy: latest or average)")
    return entry["row"], None


def _representative(entry: Any) -> Dict[str, Any]:
    return entry["__average__"][-1] if isinstance(entry, dict) and "__average__" in entry else entry


def _metric_value(entry: Any, metric: str) -> Optional[float]:
    if isinstance(entry, dict) and "__average__" in entry:
        vals = [r["_scores"].get(metric) for r in entry["__average__"]]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    return entry["_scores"].get(metric)


def _run_id_label(entry: Any) -> str:
    if isinstance(entry, dict) and "__average__" in entry:
        return "+".join(r["run_id"] for r in entry["__average__"])
    return entry["run_id"]


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
    """Pair scenario A's and B's done runs by ``(seed, fold)``, restricted to ``spec.judgment_folds``
    and each side's named ``configuration_a`` / ``configuration_b`` (unset: refuses an ambiguous
    (seed, fold) rather than guessing). Refuses (excludes, with a reason) a pair whose dataset,
    setup or judged-block fingerprint differs between A and B, whose metric is missing, or a
    non-finite ``log_ratio``. Read-only: never touches ``index.sqlite`` (F below).
    ``metric``/``direction``/``metric_kind`` default to the spec's own (a guard-rail passes its own).
    """
    metric = metric if metric is not None else spec.metric
    direction = direction if direction is not None else spec.direction
    metric_kind = metric_kind if metric_kind is not None else spec.metric_kind
    rows_a = _load_scenario_rows(spec.scenario_a, spec.root)
    rows_b = _load_scenario_rows(spec.scenario_b, spec.root)
    by_a = _group_rows(rows_a, spec.configuration_a)
    by_b = _group_rows(rows_b, spec.configuration_b)
    pairs: List[Pair] = []
    excluded: List[Excluded] = []
    for key in sorted(set(by_a) | set(by_b)):
        seed, fold = key
        entry_a, reason_a = _resolve_side(by_a.get(key), spec.duplicate_policy)
        entry_b, reason_b = _resolve_side(by_b.get(key), spec.duplicate_policy)
        if reason_a or reason_b:
            side, reason = ("A", reason_a) if reason_a else ("B", reason_b)
            excluded.append(Excluded(seed, fold, None, f"side {side}: {reason}"))
            continue
        a, b = _representative(entry_a), _representative(entry_b)
        if fold not in spec.judgment_folds:
            excluded.append(Excluded(seed, fold, a["run_id"], f"fold {fold} is not a judgement fold "
                                     f"{list(spec.judgment_folds)}"))
            continue
        mismatched = [f for f in FINGERPRINT_FIELDS if a.get(f) != b.get(f)]
        if _block_fingerprint(a) != _block_fingerprint(b):
            mismatched = mismatched + ["judged_block"]
        if mismatched:
            excluded.append(Excluded(seed, fold, a["run_id"],
                                     f"fingerprint mismatch on {mismatched}: A={[a.get(f) for f in mismatched if f != 'judged_block']} "
                                     f"B={[b.get(f) for f in mismatched if f != 'judged_block']}"))
            continue
        av, bv = _metric_value(entry_a, metric), _metric_value(entry_b, metric)
        if av is None or bv is None:
            excluded.append(Excluded(seed, fold, a["run_id"], f"metric {metric!r} missing (A={av}, B={bv})"))
            continue
        diff = _signed_diff(av, bv, direction, metric_kind)
        if diff is None or not np.isfinite(diff):
            excluded.append(Excluded(seed, fold, a["run_id"], f"non-finite {metric_kind} of A={av}, B={bv}"))
            continue
        pairs.append(Pair(seed, fold, _run_id_label(entry_a), _run_id_label(entry_b), float(av), float(bv),
                          a.get("created_utc"), b.get("created_utc"), diff))
    return pairs, excluded


# ------------------------------------------------------------------------------------------ folding
@dataclass
class FoldRow:
    fold: int
    seeds: Tuple[int, ...]
    mean_diff: float


def _fold_rows(pairs: Sequence[Pair]) -> List[FoldRow]:
    """One row per judgement fold present in ``pairs``: the mean of that fold's paired differences
    over its seeds (D-046: seeds on one fold share the fold's block noise, so they are averaged, not
    treated as independent units, before the paired test runs over folds)."""
    by_fold: Dict[int, List[Pair]] = {}
    for p in pairs:
        by_fold.setdefault(p.fold, []).append(p)
    return [FoldRow(f, tuple(sorted(p.seed for p in ps)), float(np.mean([p.diff for p in ps])))
           for f, ps in sorted(by_fold.items())]


# ------------------------------------------------------------------------------------- paired verdict
def _estimate(diffs: np.ndarray, *, estimator: str, alpha: float) -> Dict[str, Any]:
    if estimator == "hodges_lehmann":
        est = hodges_lehmann(diffs)
        lo, hi = wilcoxon_hl_ci(diffs, alpha=alpha)
        if np.isnan(lo) or np.isnan(hi):
            m, tlo, thi, t = paired_t_ci(diffs, alpha=alpha)
            return {"estimate": est, "ci_lo": tlo, "ci_hi": thi, "statistic": t,
                    "note": f"no exact Wilcoxon interval exists at n={len(diffs)}, alpha={alpha:g} "
                            f"(needs n >= ~6): fell back to the t interval around the mean; the "
                            f"Hodges-Lehmann point estimate is kept"}
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
    fold_rows: List[FoldRow]
    estimate: Dict[str, Any]
    verdict: str
    refusal_reason: Optional[str]
    guard_rail_results: List[Dict[str, Any]]
    non_inferiority: Optional[Dict[str, Any]]
    alpha_used: float
    generated_utc: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(), "spec_hash": self.spec.spec_hash,
            "registration": self.spec.registration_info(),
            "n_pairs": len(self.pairs), "n_folds": len(self.fold_rows),
            "pairs": [vars(p) for p in self.pairs],
            "fold_rows": [vars(f) for f in self.fold_rows],
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
                f"judgement folds {list(s.judgment_folds)} (min_folds {s.min_folds}) · alpha {s.alpha:g}", "",
                f"spec hash `{s.spec_hash}` · registered {s.registered_utc} "
                f"(effective {s.effective_registered_utc}, source {s.registered_utc_source})", ""]
        if self.refusal_reason:
            lines += [f"**Refused: {self.refusal_reason}**", ""]
            if self.excluded:
                lines += [f"{len(self.excluded)} pair(s) excluded:", ""]
                for e in self.excluded:
                    lines.append(f"- seed {e.seed}, fold {e.fold}: {e.reason}")
            return "\n".join(lines)
        est = self.estimate
        lines += [f"**Verdict: {self.verdict}** ({s.estimator}: {est['estimate']:.4g}, "
                 f"95% CI [{est['ci_lo']:.4g}, {est['ci_hi']:.4g}], n = {len(self.fold_rows)} folds, "
                 f"{len(self.pairs)} pairs)", ""]
        if "note" in est:
            lines += [f"_{est['note']}_", ""]
        lines += ["| fold | seeds | mean diff |", "|---|---|---|"]
        for fr in self.fold_rows:
            lines.append(f"| {fr.fold} | {list(fr.seeds)} | {fr.mean_diff:.4g} |")
        lines += ["", "| seed | fold | A | B | diff |", "|---|---|---|---|---|"]
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


def _refuse(spec: CompareSpec, reason: str, *, pairs=(), excluded=()) -> CompareResult:
    return CompareResult(spec, list(pairs), list(excluded), [], {"estimate": float("nan"), "ci_lo": float("nan"),
                                                                  "ci_hi": float("nan"), "statistic": float("nan")},
                        "refused", reason, [], None, spec.alpha)


def compare(spec: CompareSpec) -> CompareResult:
    """Run the comparison a pre-registered :class:`CompareSpec` describes: lock/verify its
    registration, pair, aggregate to one row per judgement fold, compute the paired estimate and
    verdict over folds, and judge every guard-rail by the same paired test (D-025, D-046)."""
    reg_reason = _check_and_record_registration(spec)
    if reg_reason:
        return _refuse(spec, reg_reason)

    pairs, excluded = pair_runs(spec)
    fold_rows = _fold_rows(pairs)
    # the fold count is checked first (D-046: the unit of inference is the fold): a design with too
    # few judgement folds is refused by naming the fold count, even when it also has too few pairs.
    if len(fold_rows) < spec.min_folds:
        return _refuse(spec, f"only {len(fold_rows)} judgement fold(s) had a usable pair "
                             f"({sorted(fr.fold for fr in fold_rows)}), need >= {spec.min_folds} (D-046)",
                        pairs=pairs, excluded=excluded)
    if len(pairs) < spec.min_pairs:
        return _refuse(spec, f"only {len(pairs)} pair(s) survived (need >= {spec.min_pairs}); "
                             f"{len(excluded)} excluded, see excluded_pairs", pairs=pairs, excluded=excluded)

    effective = _parse_utc(spec.effective_registered_utc)
    starts = [p.a_created_utc for p in pairs if p.a_created_utc] + [p.b_created_utc for p in pairs if p.b_created_utc]
    if starts:
        earliest = min(starts, key=_parse_utc)
        if _parse_utc(earliest) < effective:
            return _refuse(spec, f"a compared run started at {earliest}, before the spec's effective "
                                 f"registration {spec.effective_registered_utc} ({spec.registered_utc_source}): "
                                 f"the spec was written or changed after GPU time began", pairs=pairs,
                            excluded=excluded)
    planned = spec.pairs_planned_for_this_look
    if planned is not None and len(pairs) != planned:
        return _refuse(spec, f"{len(pairs)} pairs found but the spec pre-registered exactly {planned} for "
                             f"look {spec.look_index}/{spec.looks} (no peeking): re-register before adding runs",
                        pairs=pairs, excluded=excluded)

    alpha_used = pocock_alpha(spec.alpha, spec.looks, one_sided=False) if spec.looks > 1 else spec.alpha
    fold_diffs = np.array([fr.mean_diff for fr in fold_rows], float)
    est = _estimate(fold_diffs, estimator=spec.estimator, alpha=alpha_used)
    verdict = _verdict_from_ci(est["ci_lo"], est["ci_hi"], spec.min_effect, "A", "B")

    guard_results = []
    for g in spec.guard_rails:
        gp, _gexcluded = pair_runs(spec, metric=g.metric, direction=g.direction, metric_kind=g.metric_kind)
        gp = [p for p in gp if any(pp.seed == p.seed and pp.fold == p.fold for pp in pairs)]
        gfold_rows = _fold_rows(gp)
        if len(gfold_rows) < spec.min_folds:
            guard_results.append({"metric": g.metric, "verdict": "undecided", "n_folds": len(gfold_rows),
                                  "estimate": {"estimate": float("nan"), "ci_lo": float("nan"),
                                              "ci_hi": float("nan"), "statistic": float("nan")}})
            continue
        gdiffs = np.array([fr.mean_diff for fr in gfold_rows], float)
        gest = _estimate(gdiffs, estimator=spec.estimator, alpha=alpha_used)
        gv = ("pass" if gest["ci_lo"] > -g.max_degradation else
             "breach" if gest["ci_hi"] < -g.max_degradation else "undecided")
        guard_results.append({"metric": g.metric, "verdict": gv, "n_folds": len(gfold_rows), "estimate": gest})

    ni = None
    if spec.non_inferiority_margin is not None:
        ni = {"margin": spec.non_inferiority_margin,
             "verdict": non_inferiority_verdict(est["ci_lo"], est["ci_hi"], spec.non_inferiority_margin)}

    return CompareResult(spec, pairs, excluded, fold_rows, est, verdict, None, guard_results, ni, alpha_used)


# ----------------------------------------------------------------------------------- error simulation
def simulate_error_rates(spec: CompareSpec, *, n_folds: Optional[int] = None, seeds_per_fold: int = 1,
                         n_sim: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """Calibrated null/power simulation matching the fold x seed structure :func:`compare` actually
    uses (D-046): each simulated judgement fold gets one shared ``block_sd`` (or ``noise_sd``, treated
    as the fold's own noise when it is the only component given) draw, and each of its
    ``seeds_per_fold`` seeds adds independent ``seed_sd`` noise on top; the fold's mean over its
    seeds is what the paired test runs over (:func:`_estimate`, ``spec.estimator`` included), exactly
    as :func:`compare` computes it -- not a per-pair statistic, which understates the truly-null false
    "beats" rate whenever more than one seed shares a fold's block noise (D-025's acceptance
    criterion; D-037's "calibrated to the measured variance components").

    ``n_folds`` defaults to ``spec.min_folds``; refuses fewer than that (a design with too few
    judgement folds is invalid regardless of how many seeds it has, D-046). Needs ``spec.noise_sd``
    (then treated as the fold-level component, ``seed_sd`` implicitly 0), or both ``spec.seed_sd``
    and ``spec.block_sd``.
    """
    if spec.block_sd is not None:
        block_sd = spec.block_sd
        seed_sd = spec.seed_sd if spec.seed_sd is not None else 0.0
    elif spec.noise_sd is not None:
        block_sd, seed_sd = spec.noise_sd, 0.0
    else:
        raise CompareError("simulate_error_rates needs spec.noise_sd, or spec.block_sd (with an optional "
                          "spec.seed_sd)")
    n_folds = n_folds if n_folds is not None else spec.min_folds
    if n_folds < spec.min_folds:
        raise CompareError(f"n_folds={n_folds} is below spec.min_folds={spec.min_folds}: not a valid design")
    n_sim = max(1000, int(n_sim))
    rng = np.random.default_rng(seed)
    alpha_used = pocock_alpha(spec.alpha, spec.looks, one_sided=False) if spec.looks > 1 else spec.alpha
    # the noise_sd this call is "calibrated to" for reporting: the resulting per-fold-mean sd
    effective_noise_sd = float(np.sqrt(block_sd ** 2 + (seed_sd ** 2) / max(1, seeds_per_fold)))

    def _fold_means(effect: float) -> np.ndarray:
        block = rng.normal(0.0, block_sd, size=(n_sim, n_folds, 1)) if block_sd > 0 else \
            np.zeros((n_sim, n_folds, 1))
        seeds = rng.normal(effect, seed_sd, size=(n_sim, n_folds, seeds_per_fold)) if seed_sd > 0 else \
            np.full((n_sim, n_folds, seeds_per_fold), effect)
        return (block + seeds).mean(axis=2)                        # [n_sim, n_folds]: the fold means compare() uses

    def _verdicts(fold_means: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        beats_a = np.zeros(n_sim, dtype=bool)
        beats_b = np.zeros(n_sim, dtype=bool)
        for i in range(n_sim):
            est = _estimate(fold_means[i], estimator=spec.estimator, alpha=alpha_used)
            v = _verdict_from_ci(est["ci_lo"], est["ci_hi"], spec.min_effect, "A", "B")
            beats_a[i] = v == "A beats B"
            beats_b[i] = v == "B beats A"
        return beats_a, beats_b

    beats_a0, beats_b0 = _verdicts(_fold_means(0.0))
    false_beats = beats_a0 | beats_b0
    false_rate = float(false_beats.mean())
    mc_error = float(np.sqrt(false_rate * (1 - false_rate) / n_sim))

    beats_a2, _ = _verdicts(_fold_means(2.0 * spec.min_effect))
    power = float(beats_a2.mean())
    power_mc_error = float(np.sqrt(power * (1 - power) / n_sim))

    return {"n_sim": n_sim, "n_folds": n_folds, "seeds_per_fold": seeds_per_fold, "block_sd": block_sd,
           "seed_sd": seed_sd, "noise_sd": effective_noise_sd, "alpha_used": alpha_used, "seed": seed,
           "estimator": spec.estimator, "false_beats_rate": false_rate, "false_beats_mc_error": mc_error,
           "power_at_2x_min_effect": power, "power_mc_error": power_mc_error}


__all__ = ["CompareError", "CompareResult", "CompareSpec", "Excluded", "FoldRow", "GuardRail", "Pair", "compare",
          "intersection_union_verdict", "non_inferiority_verdict", "pair_runs", "per_fold_retention",
          "simulate_error_rates"]

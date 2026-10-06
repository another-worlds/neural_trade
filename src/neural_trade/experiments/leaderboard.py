"""The leaderboard (NT-031): one row per configuration, ranked by the dev-fold net Sharpe after
costs (D-020), with guard-rails beside it that can disqualify a row from the winner, and the
test-fold numbers shown on every row but never used to rank or choose (D-020, D-044: the cost
profile defaults to 0 per side).

Reads the run store's index (:mod:`neural_trade.experiments.store`): every row is one trained cell
(configuration, fold, seed), scored by :mod:`neural_trade.experiments.scorer` onto its fold's role,
``dev`` (an earlier fold the ranking uses) or ``test`` (the latest fold). This module groups a
scenario's rows by ``configuration`` and aggregates fold x seed with D-046's unit of inference: the
mean of each dev fold's seed-mean is the number ranked on (``RoleAggregate.values``), and its spread
is the sample standard deviation BETWEEN fold means (``RoleAggregate.spread``) -- never pretending
seeds that share one fold's sampling noise are independent draws. The same aggregation is applied to
the test-role rows, shown but excluded from the sort.

Guard-rails (VISION "The yardstick": maximum drawdown, the number of trades, beating buy-and-hold,
beating the random null at the same frequency) are evaluated on the dev aggregate. A scenario spec
does not yet carry guard-rail thresholds of its own (``experiments/scenario.py``'s module docstring
flags this as a NT-031 extension point for a later schema version); until that exists,
:func:`build_leaderboard` takes a :class:`GuardRailSpec` explicitly and falls back to
``DEFAULT_GUARD_RAILS`` (minimum trades 1, so a 0-trade row is disqualified per the NT-076 QA note;
must beat buy-and-hold; must beat the random null's median) when the caller passes none.

The dataset fingerprint shown per row is the run's recorded ``dataset_sha256`` (meta.json's
``dataset.sha256``, carried into the index by :mod:`neural_trade.experiments.store`); a run written
before that field existed reports "n/a" (NT-041 is the follow-up that back-fills it everywhere).

A row's ``status`` is "done" when at least one of its cells scored, "failed" when every cell of the
configuration failed, and "incomplete" otherwise (still running, or interrupted, with no failures
yet). A failed or data-less row is never the winner: it sorts to the bottom of the dev ranking.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from neural_trade.experiments.store import RunStore

# the headline metrics every row aggregates, in display order; (store column, label, format)
METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("sharpe_net", "net Sharpe", "{:+.3f}"),
    ("total_return", "net return", "{:+.2%}"),
    ("max_drawdown", "max drawdown", "{:.2%}"),
    ("n_trades", "trades", "{:.1f}"),
    ("buy_and_hold_return", "buy & hold return", "{:+.2%}"),
    ("random_percentile_return", "random-null percentile", "{:.0f}"),
)
RANK_METRIC = "sharpe_net"              # the ranking column: dev-fold net Sharpe after costs (D-020)
ROLES = ("dev", "test")


# ------------------------------------------------------------------ guard-rails
@dataclass(frozen=True)
class GuardRailSpec:
    """Thresholds a row's DEV aggregate must clear to be eligible as the winner (VISION "The
    yardstick"). ``None`` disables a threshold check. Passed explicitly by the caller (a scenario
    does not carry these yet, see the module docstring); ``DEFAULT_GUARD_RAILS`` applies otherwise.
    """
    max_drawdown_max: Optional[float] = None        # e.g. 0.25: dev mean max_drawdown must be <=
    min_trades: Optional[float] = 1.0                # dev mean n_trades must be >= (0-trade rows fail)
    require_beat_buy_and_hold: bool = True           # dev mean total_return > dev mean buy_and_hold_return
    require_beat_random_null: bool = True            # dev mean random-null percentile >= the line below
    random_null_percentile_min: float = 50.0


DEFAULT_GUARD_RAILS = GuardRailSpec()

_BLOCK_FIELDS = {"max_drawdown": "max_drawdown_max", "min_trades": "min_trades",
                 "random_null_percentile": "random_null_percentile_min",
                 "beat_buy_and_hold": "require_beat_buy_and_hold", "beat_random_null": "require_beat_random_null"}


def guard_rails_from_block(block: Optional[Mapping[str, Any]], **overrides: Any) -> Tuple[GuardRailSpec, str]:
    """The thresholds of a scenario's ``leaderboard:`` block (``Scenario.leaderboard``; keys missing
    from it keep today's defaults), then keyword ``overrides`` in the block's own key names (the CLI
    flags; ``None`` means not given). Returns the spec and a one-line description naming where each
    non-default value came from, for the table header."""
    values: Dict[str, Any] = {}
    note: List[str] = []
    for key, value in (block or {}).items():
        values[_BLOCK_FIELDS[key]] = value
        note.append(f"{key}={value} (scenario)")
    for key, value in overrides.items():
        if value is not None:
            values[_BLOCK_FIELDS[key]] = value
            note = [n for n in note if not n.startswith(f"{key}=")] + [f"{key}={value} (command-line override)"]
    return dataclasses.replace(DEFAULT_GUARD_RAILS, **values), ("; ".join(note) or "defaults")


def describe_guard_rails(spec: GuardRailSpec, source: str = "defaults") -> str:
    parts = [f"max drawdown <= {spec.max_drawdown_max:.0%}" if spec.max_drawdown_max is not None
             else "max drawdown not checked",
             f"trades >= {spec.min_trades:g} on the mean and on every dev fold" if spec.min_trades is not None
             else "trades not checked",
             "beat buy-and-hold" if spec.require_beat_buy_and_hold else "buy-and-hold not checked",
             f"random-null percentile >= {spec.random_null_percentile_min:g}" if spec.require_beat_random_null
             else "random null not checked"]
    return "; ".join(parts) + f" [thresholds: {source}]"


@dataclass(frozen=True)
class GuardRail:
    name: str
    description: str
    passed: bool
    detail: str


def _num(v: Optional[float]) -> Tuple[bool, str]:
    """(usable, reason): a missing or non-finite guard-rail value never passes, and says why."""
    if v is None:
        return False, "missing"
    if not math.isfinite(v):
        return False, f"non-finite ({v})"
    return True, ""


def _check_guard_rails(dev_values: Mapping[str, Optional[float]], spec: GuardRailSpec, n_dev_rows: int,
                        fold_trades: Optional[Mapping[int, float]] = None) -> List[GuardRail]:
    rails: List[GuardRail] = []
    if n_dev_rows == 0:
        rails.append(GuardRail("dev_data", "has at least one scored dev-fold cell", False,
                                "no scored dev-fold cell"))
        return rails
    # the ranking number itself must be a finite number, or the row has no rank to defend
    ok, why = _num(dev_values.get(RANK_METRIC))
    rails.append(GuardRail("ranking_value", "dev net Sharpe is a finite number", ok, "ok" if ok else why))

    def rail(name: str, desc: str, keys: Sequence[str], test, show) -> None:
        vals = [dev_values.get(k) for k in keys]
        bad = [f"{k} {why}" for k, (okv, why) in zip(keys, map(_num, vals)) if not okv]
        if bad:
            rails.append(GuardRail(name, desc, False, "; ".join(bad)))
        else:
            rails.append(GuardRail(name, desc, bool(test(*vals)), show(*vals)))

    if spec.max_drawdown_max is not None:
        rail("max_drawdown", f"dev max drawdown <= {spec.max_drawdown_max:.0%}", ["max_drawdown"],
             lambda dd: dd <= spec.max_drawdown_max, lambda dd: f"{dd:.2%}")
    if spec.min_trades is not None:
        rail("min_trades", f"dev trades >= {spec.min_trades:g} on the mean and on every dev fold", ["n_trades"],
             lambda nt: nt >= spec.min_trades, lambda nt: f"{nt:.1f}")
        # an idle fold is not averaged away: every dev fold must clear the activity threshold
        idle = [f"fold {f}: {v:.1f}" if math.isfinite(v) else f"fold {f}: non-finite ({v})"
                for f, v in sorted((fold_trades or {}).items())
                if not (math.isfinite(v) and v >= spec.min_trades)]
        if idle:
            rails[-1] = GuardRail("min_trades", rails[-1].description, False,
                                  f"{rails[-1].detail}; below {spec.min_trades:g} trades on " + ", ".join(idle))
    if spec.require_beat_buy_and_hold:
        rail("beat_buy_and_hold", "dev net return > dev buy-and-hold return",
             ["total_return", "buy_and_hold_return"], lambda tr, bh: tr > bh,
             lambda tr, bh: f"{tr:+.2%} vs {bh:+.2%}")
    if spec.require_beat_random_null:
        rail("beat_random_null", f"dev random-null percentile >= {spec.random_null_percentile_min:g}",
             ["random_percentile_return"], lambda pr: pr >= spec.random_null_percentile_min,
             lambda pr: f"{pr:.0f}")
    return rails


# ------------------------------------------------------------------ aggregation
def _mean(v: Sequence[float]) -> float:
    """Mean that tolerates nan / inf (``statistics.mean`` raises on nan): a non-finite value stays
    visible in the aggregate so a guard-rail can name it."""
    return sum(v) / len(v)


def _stdev(v: Sequence[float]) -> float:
    if not all(math.isfinite(x) for x in v):
        return float("nan")
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


@dataclass(frozen=True)
class RoleAggregate:
    """One role's (dev or test) aggregate over a configuration's cells: the mean of each fold's
    seed-mean (``values``), the sample standard deviation across fold means (``spread``, None with
    fewer than two folds with data) -- D-046's unit of inference is the fold -- and, for re-runs,
    the number of seeds per fold (``seeds_per_fold``) and the mean over folds of the standard
    deviation across a fold's seeds (``seed_spread``, None when no fold has two seeds)."""
    n_folds: int
    n_rows: int
    folds: Tuple[int, ...]
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    spread: Dict[str, Optional[float]] = field(default_factory=dict)
    seeds_per_fold: Tuple[int, ...] = ()
    seed_spread: Dict[str, Optional[float]] = field(default_factory=dict)
    fold_values: Dict[str, Dict[int, float]] = field(default_factory=dict)   # column -> fold -> seed-mean

    @property
    def n_seeds(self) -> int:
        """Seeds per fold (the largest count over folds; a re-run with 3 seeds says 3)."""
        return max(self.seeds_per_fold, default=0)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> RoleAggregate:
    folds = sorted({int(r["fold"]) for r in rows if r.get("fold") is not None})
    values: Dict[str, Optional[float]] = {}
    spread: Dict[str, Optional[float]] = {}
    seed_spread: Dict[str, Optional[float]] = {}
    fold_values: Dict[str, Dict[int, float]] = {}
    for col, _, _ in METRICS:
        fold_means, seed_sds = [], []
        fold_values[col] = {}
        for f in folds:
            seed_vals = [r[col] for r in rows if r.get("fold") == f and r.get(col) is not None]
            if seed_vals:
                fold_means.append(_mean(seed_vals))
                fold_values[col][f] = fold_means[-1]
            if len(seed_vals) >= 2:
                seed_sds.append(_stdev(seed_vals))
        values[col] = _mean(fold_means) if fold_means else None
        spread[col] = _stdev(fold_means) if len(fold_means) >= 2 else None
        seed_spread[col] = _mean(seed_sds) if seed_sds else None
    seeds_per_fold = tuple(len({r.get("seed") for r in rows if r.get("fold") == f}) for f in folds)
    return RoleAggregate(n_folds=len(folds), n_rows=len(rows), folds=tuple(folds), values=values, spread=spread,
                         seeds_per_fold=seeds_per_fold, seed_spread=seed_spread,
                         fold_values=fold_values)


# ------------------------------------------------------------------ rows
@dataclass(frozen=True)
class LeaderboardRow:
    scenario: str
    configuration: str
    status: str                              # "done" | "failed" | "incomplete"
    dev: RoleAggregate
    test: RoleAggregate
    guard_rails: Tuple[GuardRail, ...]
    disqualified: bool
    dataset_fingerprint: Optional[str]       # meta.json dataset.sha256, or None ("n/a"; NT-041)
    bar_minutes: Optional[float]
    horizon_steps: Optional[Tuple[int, ...]]
    strategy: Optional[str]
    n_cells: int
    n_failed: int
    errors: Tuple[str, ...] = ()
    rank: Optional[int] = None               # filled in by build_leaderboard / set_ranks


def _first(rows: Sequence[Mapping[str, Any]], key: str):
    for r in rows:
        v = r.get(key)
        if v is not None:
            return v
    return None


def _horizon_steps(rows: Sequence[Mapping[str, Any]]) -> Optional[Tuple[int, ...]]:
    raw = _first(rows, "horizon_steps")
    if raw is None:
        return None
    try:
        return tuple(int(x) for x in json.loads(raw))
    except (ValueError, TypeError):
        return None


def _configuration_row(scenario: str, configuration: str, rows: Sequence[Mapping[str, Any]],
                        guard_rails: GuardRailSpec) -> LeaderboardRow:
    done = [r for r in rows if r.get("status") == "done"]
    failed = [r for r in rows if r.get("status") == "failed"]
    if rows and len(failed) == len(rows):
        status = "failed"
    elif done:
        status = "done"
    else:
        status = "incomplete"
    dev_agg = _aggregate([r for r in done if r.get("role") == "dev"])
    test_agg = _aggregate([r for r in done if r.get("role") == "test"])
    rails = _check_guard_rails(dev_agg.values, guard_rails, dev_agg.n_rows,
                          dev_agg.fold_values.get("n_trades"))
    if status != "done":
        errs = "; ".join(sorted({r["error"] for r in failed if r.get("error")}))
        rails.insert(0, GuardRail("status", "the configuration has a scored cell", False,
                                   status + (f": {errs}" if errs else "")))
    disqualified = status != "done" or any(not g.passed for g in rails)
    errors = tuple(sorted({r["error"] for r in failed if r.get("error")}))
    return LeaderboardRow(
        scenario=scenario, configuration=configuration, status=status, dev=dev_agg, test=test_agg,
        guard_rails=tuple(rails), disqualified=disqualified,
        dataset_fingerprint=_first(rows, "dataset_sha256"), bar_minutes=_first(rows, "bar_minutes"),
        horizon_steps=_horizon_steps(rows), strategy=_first(rows, "strategy"),
        n_cells=len(rows), n_failed=len(failed), errors=errors)


def build_leaderboard(rows: Sequence[Mapping[str, Any]], *, guard_rails: Optional[GuardRailSpec] = None
                       ) -> List[LeaderboardRow]:
    """One :class:`LeaderboardRow` per (scenario, configuration) in ``rows`` (the run index's rows,
    e.g. ``RunIndex.rows()`` / ``RunStore.index.rows()``), sorted by the dev aggregate's net Sharpe
    only (descending; a configuration with no dev value sorts last; ties keep first-seen order).
    ``guard_rails`` defaults to :data:`DEFAULT_GUARD_RAILS`.
    """
    spec = guard_rails if guard_rails is not None else DEFAULT_GUARD_RAILS
    order: List[Tuple[str, str]] = []
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("scenario")), str(r.get("configuration")))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    built = [_configuration_row(scenario, configuration, groups[(scenario, configuration)], spec)
             for scenario, configuration in order]

    def sort_key(i: int):
        v = built[i].dev.values.get(RANK_METRIC)
        finite = v is not None and math.isfinite(v)
        return (not finite, -(v if finite else 0.0), i)

    ranked = [built[i] for i in sorted(range(len(built)), key=sort_key)]
    return [dataclasses.replace(row, rank=rank) for rank, row in enumerate(ranked, 1)]


def scenario_guard_rails(scenario: Any = None, **overrides: Any) -> Tuple[GuardRailSpec, str]:
    """Guard-rail thresholds from a :class:`~neural_trade.experiments.scenario.Scenario` (or its
    ``leaderboard`` block as a mapping; None gives the defaults), with keyword overrides (see
    :func:`guard_rails_from_block`)."""
    block = getattr(scenario, "leaderboard", scenario)
    return guard_rails_from_block(block, **overrides)


def leaderboard_for_scenario(store: RunStore, scenario: str, *, guard_rails: Optional[GuardRailSpec] = None,
                              spec: Any = None, sync: bool = True) -> List[LeaderboardRow]:
    """``build_leaderboard`` of one scenario's rows, read from ``store``'s index. The guard-rail
    thresholds come from ``guard_rails`` if given, else from ``spec`` (the scenario's spec: a
    ``Scenario``), else the defaults. ``sync=True`` (default) re-reads the scenario's run directories
    first (``RunStore.sync``); pass ``False`` to use the index as it stands."""
    rows = store.sync(scenario) if sync else store.index.rows(scenario)
    if guard_rails is None and spec is not None:
        guard_rails = scenario_guard_rails(spec)[0]
    return build_leaderboard(rows, guard_rails=guard_rails)


def winner(rows: Sequence[LeaderboardRow]) -> Optional[LeaderboardRow]:
    """The top row that is not disqualified (None if every row is disqualified or there are none)."""
    return next((r for r in rows if not r.disqualified), None)


# ------------------------------------------------------------------ text table
def _fmt_metric(agg: RoleAggregate, col: str, fmt: str, *, counts: bool = True) -> str:
    v = agg.values.get(col)
    if v is None:
        return "n/a"
    base = fmt.format(v)
    if not math.isfinite(v):
        return base
    sd, ssd = agg.spread.get(col), agg.seed_spread.get(col)
    parts = []
    if sd is not None:
        parts.append(f"fold sd {fmt.format(sd).lstrip('+')}")
    if ssd is not None:
        parts.append(f"seed sd {fmt.format(ssd).lstrip('+')}")
    if counts:
        parts.append(f"{agg.n_folds} fold{'s' * (agg.n_folds != 1)} x {agg.n_seeds} seed{'s' * (agg.n_seeds != 1)}, "
                     f"{agg.n_rows} cell{'s' * (agg.n_rows != 1)}")
    return base + (f" ({', '.join(parts)})" if parts else "")


def _guard_text(r: LeaderboardRow) -> str:
    gr = "; ".join(f"{g.name} {'OK' if g.passed else 'FAIL'} ({g.detail})" for g in r.guard_rails
                   if not (g.name == "ranking_value" and g.passed)) or "n/a"
    return ("DISQUALIFIED: " if r.disqualified else "") + gr


def _status_text(r: LeaderboardRow) -> str:
    return r.status + (f" ({r.n_failed} of {r.n_cells} cells failed)" if r.n_failed and r.status != "failed" else "")


TABLE_HEADER = ("rank", "configuration", "status", "ranking: dev net Sharpe (spread, counts)", "dev net return",
                "dev max drawdown", "dev trades", "dev buy & hold", "dev random-null percentile", "guard-rails",
                "test net Sharpe (test, not used for ranking)", "test net return (test, not used for ranking)",
                "test max drawdown (test, not used for ranking)", "test trades (test, not used for ranking)",
                "dataset fingerprint", "bar (min)", "horizons (bars)", "strategy")


def table_cells(r: LeaderboardRow) -> List[str]:
    """The text of every column of one row, in ``TABLE_HEADER`` order (the Markdown table and the
    figure's table share it, so they cannot drift apart)."""
    fp = r.dataset_fingerprint
    return [str(r.rank), r.configuration, _status_text(r),
            _fmt_metric(r.dev, "sharpe_net", "{:+.3f}"),
            _fmt_metric(r.dev, "total_return", "{:+.2%}", counts=False),
            _fmt_metric(r.dev, "max_drawdown", "{:.2%}", counts=False),
            _fmt_metric(r.dev, "n_trades", "{:.1f}", counts=False),
            _fmt_metric(r.dev, "buy_and_hold_return", "{:+.2%}", counts=False),
            _fmt_metric(r.dev, "random_percentile_return", "{:.0f}", counts=False), _guard_text(r),
            _fmt_metric(r.test, "sharpe_net", "{:+.3f}"),
            _fmt_metric(r.test, "total_return", "{:+.2%}", counts=False),
            _fmt_metric(r.test, "max_drawdown", "{:.2%}", counts=False),
            _fmt_metric(r.test, "n_trades", "{:.1f}", counts=False),
            "n/a" if not fp else f"{fp[:12]}...", "n/a" if r.bar_minutes is None else f"{r.bar_minutes:g}",
            "n/a" if r.horizon_steps is None else ", ".join(str(h) for h in r.horizon_steps), r.strategy or "n/a"]


def leaderboard_markdown(rows: Sequence[LeaderboardRow], *, guard_rails: Optional[GuardRailSpec] = None,
                         guard_rail_source: str = "defaults") -> str:
    """A Markdown table: the ranking column labelled as such, the test-fold columns labelled
    'test, not used for ranking' (criterion 5), guard-rails and disqualification, the dataset
    fingerprint, bar size, horizons and strategy on every row (criterion 4)."""
    scenario = rows[0].scenario if rows else "(empty)"
    lines = [f"# Leaderboard: `{scenario}`", "",
             "One row per configuration. **Ranking column: dev-fold net Sharpe after costs** (mean over "
             "the dev folds and their seeds, D-020, D-046). Guard-rails beside it can disqualify a row from "
             "the winner (VISION \"The yardstick\"). The **test-fold columns are test, not used for ranking** "
             "(D-020): shown for every row, never used to rank or choose.", "",
             "Guard-rails: " + describe_guard_rails(guard_rails or DEFAULT_GUARD_RAILS, guard_rail_source) + ".", "",
             "| " + " | ".join(TABLE_HEADER) + " |", "|" + "---|" * len(TABLE_HEADER)]
    for r in rows:
        cells = table_cells(r)
        cells[1] = f"`{cells[1]}`"
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    w = winner(rows)
    lines += ["", f"**Winner:** `{w.configuration}` (rank {w.rank})" if w is not None
              else "**Winner:** none (every row disqualified or no scored dev data)."]
    return "\n".join(lines) + "\n"


__all__ = ["DEFAULT_GUARD_RAILS", "GuardRail", "GuardRailSpec", "LeaderboardRow", "METRICS", "RANK_METRIC",
           "ROLES", "RoleAggregate", "describe_guard_rails", "guard_rails_from_block",
           "scenario_guard_rails", "TABLE_HEADER", "table_cells", "build_leaderboard", "leaderboard_for_scenario", "leaderboard_markdown",
           "winner"]

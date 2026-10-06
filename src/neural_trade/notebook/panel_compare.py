"""Comparing configurations in the control panel (notebook 06, NT-034): every logged metric of two or more
leaderboard rows, per horizon, with its spread over folds and seeds, and the paired verdict of NT-032.

**What is drawn (D-014).** For each selected configuration (a leaderboard row: ``scenario / configuration``)
the cells of the chosen role (``dev``: the folds the ranking uses; ``test``: shown, never ranks, D-020) are
aggregated as the leaderboard does (D-046: the unit of inference is the fold): the value is the mean over
folds of each fold's seed-mean, the *fold sd* is the sample sd between those fold means, the *seed sd* is the
mean over folds of the sd across a fold's seeds. One figure per metric group (direction, Gaussian direction,
served delta, raw delta, variance, confidence gap, sample sizes, coherence, backtest); one panel per metric;
one dot per configuration and horizon in the horizon's colour (h0 blue, h1 orange, h2 green, more horizons
extend the palette), a whisker for the fold sd and a thinner grey line below it for the seed sd, the metric's
no-skill reference where it has one. A metric nobody logged has no panel (the subtitle names how many). The
table under the figures lists **every** score key of the cells (the per-horizon metrics, the backtest and its
baselines, the fitted baselines, the training facts) with value, fold sd and seed sd per configuration.

**The verdict.** A paired verdict exists only for a pre-registered :class:`CompareSpec`
(``configs/compares/*.yaml``, written before the runs; D-025): when one names the two selected configurations
(either order) its :func:`~neural_trade.experiments.comparator.compare` result is shown as it is. Otherwise
the panel says so and shows the per-fold paired differences of one metric as **exploratory**: a table, not a
verdict.
"""
from __future__ import annotations

import dataclasses
import html
import json
import logging
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from neural_trade.evaluation.report import GAUSS_CONST_NA_KEYS, SERVED_DELTA_NA_KEYS
from neural_trade.experiments.store import RunStore
from neural_trade.visualization import theme as T

logger = logging.getLogger(__name__)

HORIZON_KEY = re.compile(r"^h(\d+)/(?P<group>[A-Za-z_0-9]+)/(?P<metric>.+)$")
SAMPLE_KEY = re.compile(r"^h(\d+)/(?P<metric>n|n_eff)$")
BACKTEST_KEY = re.compile(r"^backtest/(?P<metric>[A-Za-z_0-9]+)$")
COHERENCE_KEY = re.compile(r"^coherence/(?P<metric>.+)$")
# figure groups, in display order: (group id, title)
GROUPS: Tuple[Tuple[str, str], ...] = (
    ("direction", "Direction head (P(up))"), ("gauss_direction", "Direction from the Gaussian price head"),
    ("delta", "Price change, served delta"), ("delta_raw", "Price change, raw heads"),
    ("variance", "Variance head and intervals"), ("confidence_gap", "Confidence gap"),
    ("sample", "Sample sizes"), ("coherence", "Coherence across horizons"), ("backtest", "Backtest summary"))
GROUP_TITLES = dict(GROUPS)
NCOLS = 4
PANEL_PX = 250


@dataclass(frozen=True)
class MetricAgg:
    """One configuration's aggregate of one score key over a role's cells."""
    value: Optional[float]
    fold_sd: Optional[float]
    seed_sd: Optional[float]
    n_folds: int
    n_seeds: int
    fold_values: Tuple[Tuple[int, float], ...] = ()
    n_na: int = 0                    # cells whose value is undefined (served delta 0, D-007) and left out


@dataclass
class ConfigScores:
    label: str                                  # "scenario / configuration"
    scenario: str
    configuration: str
    role: str
    n_cells: int
    folds: Tuple[int, ...]
    aggs: Dict[str, MetricAgg] = field(default_factory=dict)
    na: Dict[str, int] = field(default_factory=dict)   # score key -> cells where it is n/a (beta = 0), none has a value


def _finite(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _sd(v: Sequence[float]) -> Optional[float]:
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def key_parts(key: str) -> Optional[Tuple[str, Optional[int], str]]:
    """(group, horizon index or None, metric name) of a score key a figure draws; None for a key the figures
    leave to the table (backtest baselines, fitted baselines, training facts)."""
    m = HORIZON_KEY.match(key)
    if m:
        return m.group("group"), int(m.group(1)), m.group("metric")
    m = SAMPLE_KEY.match(key)
    if m:
        return "sample", int(m.group(1)), m.group("metric")
    m = COHERENCE_KEY.match(key)
    if m:
        return "coherence", None, m.group("metric")
    m = BACKTEST_KEY.match(key)
    if m:
        return "backtest", None, m.group("metric")
    return None


# Statistics of the served delta (and of the Gaussian readout built on it) that a constant 0 does not have: with a
# shrink beta of 0 (D-007) they are stored as fixed values by the scorer (0.0, 0.5, ...), which are not measurements.
# They are left out here (drawn and tabled as n/a); the raw heads' own panels carry the information. The rule is the
# report's (evaluation.report SERVED_DELTA_NA_KEYS / GAUSS_CONST_NA_KEYS, the markdown's ZERO_BETA_NA and
# GAUSS_CONST_NA cells), plus what the index stores besides the markdown rows: the served delta's mean (the
# constant's own 0, NT-180) and the constant readout's confusion counts and rates (it calls up on no bar).
CONST_READOUT_KEYS = ("acc", "bal_acc", "precision", "recall", "specificity", "f1", "tp", "fp", "tn", "fn")
SERVED_NA_METRICS = {"delta": tuple(SERVED_DELTA_NA_KEYS) + ("mean_pred",),
                     "gauss_direction": tuple(GAUSS_CONST_NA_KEYS) + CONST_READOUT_KEYS}
NA_TEXT = "n/a (beta = 0)"


def zero_beta_horizons(run_dir, role: str) -> set:
    """Horizon names whose served delta is the constant 0 in this cell: a recorded shrink beta of 0, or a served delta
    found 0 on every sample (``meta.served_delta_zero``), read from the cell's own ``eval_report_<role>.json``."""
    try:
        meta = json.loads((Path(run_dir) / f"eval_report_{role}.json").read_text(encoding="utf-8")).get("meta") or {}
    except (OSError, ValueError):
        return set()
    zero = {h for h, b in (meta.get("delta_scale") or {}).items() if b is not None and b <= 0}
    return zero | set(meta.get("served_delta_zero") or ())


def _is_served_na(key: str, zero: set) -> bool:
    parts = key_parts(key)
    return bool(parts and parts[1] is not None and f"h{parts[1]}" in zero
                and parts[2] in SERVED_NA_METRICS.get(parts[0], ()))


def aggregate_configuration(store: RunStore, rows: Sequence[Mapping[str, Any]], *, role: str = "dev") -> ConfigScores:
    """:class:`ConfigScores` of one configuration from its index ``rows`` (all one scenario and configuration):
    the cells of ``role`` that finished, every score key read from the index's ``scores`` table."""
    rows = [r for r in rows if r.get("role") == role and r.get("status") == "done"]
    scenario = str(rows[0]["scenario"]) if rows else ""
    configuration = str(rows[0]["configuration"]) if rows else ""
    folds = sorted({int(r["fold"]) for r in rows if r.get("fold") is not None})
    per_cell = []
    zero_cells: Dict[str, int] = {}
    for r in rows:
        scores = store.index.scores(r["run_id"])
        zero = zero_beta_horizons(store.root / str(r["run_dir"]), role)
        for k in list(scores):
            if _is_served_na(k, zero):
                scores[k] = None
                zero_cells[k] = zero_cells.get(k, 0) + 1
        per_cell.append((r, scores))
    keys = sorted({k for _, s in per_cell for k, v in s.items() if _finite(v) is not None})
    out = ConfigScores(f"{scenario} / {configuration}", scenario, configuration, role, len(rows), tuple(folds))
    for key in keys:
        fold_means, seed_sds, seeds = [], [], 0
        for f in folds:
            vals = [_finite(s.get(key)) for r, s in per_cell if r.get("fold") == f]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            fold_means.append((f, sum(vals) / len(vals)))
            seeds = max(seeds, len(vals))
            sd = _sd(vals)
            if sd is not None:
                seed_sds.append(sd)
        if fold_means:
            means = [m for _, m in fold_means]
            out.aggs[key] = MetricAgg(sum(means) / len(means), _sd(means),
                                      sum(seed_sds) / len(seed_sds) if seed_sds else None, len(fold_means), seeds,
                                      tuple(fold_means), n_na=zero_cells.get(key, 0))
    out.na = {k: n for k, n in zero_cells.items() if k not in out.aggs}
    return out


def aggregate_selection(store: RunStore, selection: Sequence[Tuple[str, str]], *, role: str = "dev") -> List[ConfigScores]:
    """:class:`ConfigScores` for each (scenario, configuration) of ``selection``, in the order given."""
    out = []
    for scenario, configuration in selection:
        rows = [r for r in store.index.rows(scenario) if r.get("configuration") == configuration]
        cs = aggregate_configuration(store, rows, role=role)
        if not rows:
            cs = ConfigScores(f"{scenario} / {configuration}", scenario, configuration, role, 0, ())
        else:
            cs = dataclasses.replace(cs, label=f"{scenario} / {configuration}", scenario=scenario, configuration=configuration)
        out.append(cs)
    return out


def metric_table(configs: Sequence[ConfigScores]):
    """Every score key of the configurations, one row each: value, fold sd and seed sd per configuration
    (a pandas DataFrame; ``NA_TEXT`` where the value is n/a at beta = 0 on every cell, ``None`` where a configuration
    has no value)."""
    import pandas as pd

    keys = sorted({k for c in configs for k in list(c.aggs) + list(c.na)})
    cols: Dict[Tuple[str, str], List[Any]] = {}
    for c in configs:
        for part in ("value", "fold sd", "seed sd"):
            cols[(c.label, part)] = []
    for k in keys:
        for c in configs:
            a = c.aggs.get(k)
            cols[(c.label, "value")].append(a.value if a else NA_TEXT if k in c.na else None)
            cols[(c.label, "fold sd")].append(a.fold_sd if a else None)
            cols[(c.label, "seed sd")].append(a.seed_sd if a else None)
    df = pd.DataFrame(cols, index=pd.Index(keys, name="score"))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _horizon_color(i: int) -> str:
    return T.SERIES[i % len(T.SERIES)]


def _config_color(i: int) -> str:
    return T.OTHER_SERIES[i % len(T.OTHER_SERIES)]


def _reference(metric: str):
    from neural_trade.visualization.comparison import REFERENCE

    return REFERENCE.get(metric.rsplit("/", 1)[-1])


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:+.4g}"


def comparison_figures(configs: Sequence[ConfigScores], *, horizon_steps: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """{group id: figure}: one figure per metric group that has data, from the configurations' aggregates
    (see the module docstring). ``horizon_steps`` (bars per horizon) labels the horizons in the legend."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if len(configs) < 2:
        raise ValueError("a comparison needs at least two configurations")
    labels = [c.label for c in configs]
    role = configs[0].role
    # (group) -> metric -> horizon (or None) -> {config index: MetricAgg}
    tree: Dict[str, Dict[str, Dict[Optional[int], Dict[int, MetricAgg]]]] = {}
    for ci, c in enumerate(configs):
        for key, agg in c.aggs.items():
            parts = key_parts(key)
            if parts is None:
                continue
            group, h, metric = parts
            tree.setdefault(group, {}).setdefault(metric, {}).setdefault(h, {})[ci] = agg
    na_marks: Dict[str, Dict[str, Dict[Optional[int], List[int]]]] = {}
    for ci, c in enumerate(configs):
        for key in c.na:
            parts = key_parts(key)
            if parts is None:
                continue
            group, h, metric = parts
            tree.setdefault(group, {}).setdefault(metric, {}).setdefault(h, {})
            na_marks.setdefault(group, {}).setdefault(metric, {}).setdefault(h, []).append(ci)
    figs: Dict[str, Any] = {}
    n_cells = [c.n_cells for c in configs]
    for group, title in GROUPS:
        metrics = sorted(tree.get(group, {}))
        if not metrics:
            continue
        nrows = math.ceil(len(metrics) / NCOLS)
        cells = [[{} if r * NCOLS + c < len(metrics) else None for c in range(NCOLS)] for r in range(nrows)]
        fig = make_subplots(rows=nrows, cols=NCOLS, specs=cells, subplot_titles=metrics, horizontal_spacing=0.07,
                            vertical_spacing=min(0.12, 0.5 / max(nrows, 1)))
        seen_horizon = set()
        for pi, metric in enumerate(metrics):
            row, col = pi // NCOLS + 1, pi % NCOLS + 1
            by_h = tree[group][metric]
            horizons = sorted(h for h in by_h if h is not None)
            slots = horizons if horizons else [None]
            fig.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=row, col=col)   # the panel exists
            ax = "" if pi == 0 else str(pi + 1)
            spread = 0.62 / max(len(slots), 1)
            for si, h in enumerate(slots):
                members = by_h[h] if h is not None else by_h.get(None, {})
                for ci in na_marks.get(group, {}).get(metric, {}).get(h, []):
                    fig.add_annotation(x=0.5, xref=f"x{ax} domain", y=ci + (si - (len(slots) - 1) / 2) * spread, yref=f"y{ax}",
                                       text=NA_TEXT, showarrow=False, font=dict(size=10, color=_horizon_color(h) if h is not None
                                                                                else T.MUTED))
                offset = (si - (len(slots) - 1) / 2) * spread
                xs, ys, fold_sd, seed_sd, hover, colors = [], [], [], [], [], []
                for ci in sorted(members):
                    a = members[ci]
                    xs.append(a.value)
                    ys.append(ci + offset)
                    fold_sd.append(a.fold_sd or 0.0)
                    seed_sd.append(a.seed_sd or 0.0)
                    colors.append(_horizon_color(h) if h is not None else _config_color(ci))
                    hover.append(f"{labels[ci]}<br>{metric}" + (f" h{h}" if h is not None else "")
                                 + (f"<br>n/a on {a.n_na} cell(s) (beta = 0: served delta is 0), left out" if a.n_na else "")
                                 + f"<br>value {_fmt(a.value)}<br>fold sd {_fmt(a.fold_sd)} ({a.n_folds} fold(s))"
                                   f"<br>seed sd {_fmt(a.seed_sd)} ({a.n_seeds} seed(s))")
                if not xs:
                    continue
                name = (T.horizon_label(f"h{h}") if h is not None else "per configuration")
                if h is not None and horizon_steps is not None and h < len(horizon_steps):
                    name = f"h{h} ({horizon_steps[h]} bars)"
                legend = h is not None and h not in seen_horizon
                seen_horizon.add(h)
                fig.add_trace(go.Scatter(
                    x=xs, y=[y - 0.07 for y in ys], mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(size=1, color=T.MUTED),
                    error_x=dict(type="data", array=seed_sd, visible=True, thickness=1, width=0, color=T.MUTED)),
                    row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers", name=name, legendgroup=name, showlegend=legend,
                    marker=dict(size=9, color=colors, line=dict(width=1, color=T.PAPER)),
                    error_x=dict(type="data", array=fold_sd, visible=True, thickness=2, width=4, color=colors[0]
                                 if h is not None else T.INK_2),
                    text=hover, hovertemplate="%{text}<extra></extra>"), row=row, col=col)
            ref = _reference(metric)
            if ref is not None:
                fig.add_vline(x=ref[0], line=dict(color=T.NEUTRAL, width=1, dash="dash"), row=row, col=col)
            fig.update_yaxes(tickvals=list(range(len(configs))) if col == 1 else [], ticktext=labels if col == 1 else [],
                             range=[len(configs) - 0.4, -0.6], zeroline=False, row=row, col=col)
        height = max(360, PANEL_PX * nrows + 140)
        left = int(min(max(len(x) for x in labels) * 6.2, 360)) + 50
        subtitle = (f"{role} cells; dot = mean over folds of each fold's seed-mean, whisker = sd between folds, thin grey line "
                    f"below = sd across a fold's seeds; dashed = the metric's no-skill reference where it has one; cells per "
                    f"configuration: {', '.join(str(n) for n in n_cells)}"
                    + ("; the test fold is shown, never used to rank (D-020)" if role == "test" else "")
                    + ("; 'n/a (beta = 0)' = the served delta is the constant 0 there (D-007), not a measured 0: see the raw-head "
                       "figure" if any(c.na or any(a.n_na for a in c.aggs.values()) for c in configs) else ""))
        width = max(900, left + 230 * NCOLS)
        subtitle = "<br>".join(textwrap.wrap(subtitle, max(60, int(width / 6.4))))
        top = 90 + 16 * (subtitle.count("<br>") + 1)
        T.apply(fig, title=f"{title}: {len(configs)} configurations", height=height + top - 110, subtitle=subtitle)
        fig.update_layout(margin=dict(l=left, r=24, t=top + 40, b=40), width=width,
                          legend=dict(orientation="h", yref="container", y=1 - (top - 6) / (height + top - 110),
                                      yanchor="top", x=0.0, xanchor="left"))
        figs[group] = fig
    return figs


def paired_by_fold(a: ConfigScores, b: ConfigScores, metric: str) -> List[Dict[str, Any]]:
    """Exploratory pairs: for each fold both configurations have, A's and B's seed-mean of ``metric`` and A - B."""
    fa = dict(a.aggs[metric].fold_values) if metric in a.aggs else {}
    fb = dict(b.aggs[metric].fold_values) if metric in b.aggs else {}
    return [{"fold": f, "a": fa[f], "b": fb[f], "diff": fa[f] - fb[f]} for f in sorted(set(fa) & set(fb))]


def exploratory_html(a: ConfigScores, b: ConfigScores, metric: str, *,
                     why: str = "no pre-registered comparison names this pair") -> str:
    rows = paired_by_fold(a, b, metric)
    head = (f"<b>Exploratory paired differences of <code>{html.escape(metric)}</code> (A - B)</b>, per fold, seeds averaged "
            f"first (D-046): <b>not a verdict</b> ({html.escape(why)}; D-025).<br>"
            f"A = {html.escape(a.label)}, B = {html.escape(b.label)}")
    if not rows:
        return head + "<br>no fold has a value for both."
    diffs = [r["diff"] for r in rows]
    mean, sd = sum(diffs) / len(diffs), _sd(diffs)
    body = "".join(f"<tr><td>{r['fold']}</td><td>{r['a']:+.4f}</td><td>{r['b']:+.4f}</td><td>{r['diff']:+.4f}</td></tr>"
                   for r in rows)
    tail = (f"mean difference {mean:+.4f}" + (f", sd between folds {sd:.4f}" if sd is not None else "")
            + f", A ahead on {sum(d > 0 for d in diffs)} of {len(diffs)} fold(s)"
            + ("; fewer than 5 folds (D-046 asks at least 5 judgement folds for a verdict)" if len(diffs) < 5 else ""))
    return (f"{head}<table style='margin:6px 0'><tr><th>fold</th><th>A</th><th>B</th><th>A - B</th></tr>{body}</table>{tail}")


def find_compare_spec(a: ConfigScores, b: ConfigScores, compares_dir="configs/compares") -> Tuple[Optional[Path], bool]:
    """(path, flipped) of the first pre-registered spec file in ``compares_dir`` that names the pair (A, B) as
    (scenario_a, scenario_b) or, flipped, as (B, A); (None, False) when none does. The path, not only the spec:
    ``neural-trade compare <path> --out runs/compares/<file stem>`` stores its result under the file's stem."""
    from neural_trade.experiments.comparator import CompareError, CompareSpec

    compares_dir = Path(compares_dir)
    for path in sorted(compares_dir.glob("*.yaml")) if compares_dir.is_dir() else ():
        try:
            spec = CompareSpec.from_yaml(path)
        except (CompareError, OSError, ValueError, TypeError, KeyError):
            continue

        def names(x: ConfigScores, scenario: str, configuration: Optional[str]) -> bool:
            return x.scenario == scenario and (configuration is None or x.configuration == configuration)

        if names(a, spec.scenario_a, spec.configuration_a) and names(b, spec.scenario_b, spec.configuration_b):
            return path, False
        if names(b, spec.scenario_a, spec.configuration_a) and names(a, spec.scenario_b, spec.configuration_b):
            return path, True
    return None, False


def result_dirs(spec_path, spec, store: RunStore) -> List[Path]:
    """Where a stored result of this spec may be, first match wins: ``<store>/compares/<file stem>`` (RUNBOOK
    "Paired comparator": ``--out runs/compares/<name>`` with the file's name, run from the repo root, whose ``runs``
    is the store), then ``<store>/compares/<spec name>``."""
    base = Path(store.root) / "compares"
    out = [base / Path(spec_path).stem]
    if spec.name != Path(spec_path).stem:
        out.append(base / spec.name)
    return out


def stored_verdict(spec_path, spec, store: RunStore) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """(result.json, its directory) that ``neural-trade compare <spec_path> --out <dir>`` stored for this spec, or
    (None, None). Read-only: the panel never calls ``compare()``, which registers the spec (writes into the store)."""
    for d in result_dirs(spec_path, spec, store):
        try:
            doc = json.loads((d / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            return doc, d
    return None, None


def _num(v: Any, fmt: str = ".4g") -> str:
    return format(v, fmt) if isinstance(v, (int, float)) and math.isfinite(v) else "n/a"


def _ci(est: Mapping[str, Any]) -> str:
    return f"{_num(est.get('estimate'))}, 95% CI [{_num(est.get('ci_lo'))}, {_num(est.get('ci_hi'))}]"


def _html_table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table style='margin:4px 0'><tr>{th}</tr>{body}</table>"


def result_html(doc: Mapping[str, Any], spec, result_dir: Optional[Path] = None) -> str:
    """Everything NT-032's stored result says (D-014), read-only: the verdict with its interval, the non-inferiority
    result, the guard-rails, the per-fold and per-pair tables (run ids included), the excluded pairs, the registration,
    and the stored ``report.md`` itself."""
    est = doc.get("estimate") or {}
    parts = []
    if doc.get("refusal_reason"):
        parts.append(f"<b>Refused</b>: {html.escape(str(doc['refusal_reason']))}")
    else:
        parts.append(f"<b>{html.escape(str(doc.get('verdict')))}</b>: {html.escape(spec.metric)} "
                     f"{html.escape(str(doc.get('estimator') or spec.estimator))} {_ci(est)} over {doc.get('n_folds')} "
                     f"judgement folds ({doc.get('n_pairs')} pairs), minimum effect {spec.min_effect:g}, alpha "
                     f"{_num(doc.get('alpha_used', spec.alpha), 'g')}; stored {html.escape(str(doc.get('generated_utc')))}")
    ni = doc.get("non_inferiority")
    if isinstance(ni, Mapping) and ni:
        parts.append(f"<b>Non-inferiority</b> (margin {_num(ni.get('margin'), 'g')}): "
                     f"<b>{html.escape(str(ni.get('verdict')))}</b>")
    guards = doc.get("guard_rails") or []
    if guards:
        parts.append("<b>Guard-rails</b> (the same paired test)" + _html_table(
            ("metric", "verdict", "folds", "estimate, 95% CI"),
            [(g.get("metric"), g.get("verdict"), g.get("n_folds"), _ci(g.get("estimate") or {})) for g in guards]))
    folds = doc.get("fold_rows") or []
    if folds:
        parts.append("<b>Per judgement fold</b> (seeds averaged first, D-046)" + _html_table(
            ("fold", "seeds", "mean A - B"), [(r.get("fold"), r.get("seeds"), _num(r.get("mean_diff"))) for r in folds]))
    pairs = doc.get("pairs") or []
    if pairs:
        parts.append("<b>Per (seed, fold) pair</b>" + _html_table(
            ("seed", "fold", "A", "B", "A - B", "A run", "B run"),
            [(p.get("seed"), p.get("fold"), _num(p.get("a_value")), _num(p.get("b_value")), _num(p.get("diff")),
              p.get("a_run_id"), p.get("b_run_id")) for p in pairs]))
    excluded = doc.get("excluded_pairs") or []
    parts.append(f"<b>Excluded pairs</b>: {len(excluded)}"
                 + (_html_table(("pair",), [(json.dumps(e, default=str),) for e in excluded]) if excluded else ""))
    reg = doc.get("registration") or {}
    if reg:
        parts.append(f"registered {html.escape(str(reg.get('declared')))} (effective {html.escape(str(reg.get('effective')))}, "
                     f"source {html.escape(str(reg.get('source')))}); spec hash {html.escape(str(doc.get('spec_hash')))}")
    if result_dir is not None:
        report = Path(result_dir) / "report.md"
        try:
            text: Optional[str] = report.read_text(encoding="utf-8")
        except OSError:
            text = None
        where = html.escape(str(report))
        parts.append(f"<details><summary>The stored <code>report.md</code> ({where})</summary>"
                     f"<pre style='white-space:pre-wrap'>{html.escape(text)}</pre></details>" if text is not None
                     else f"no <code>report.md</code> next to the result ({where})")
    return "<br>".join(parts)


def verdict_html(a: ConfigScores, b: ConfigScores, store: RunStore, *, metric: str = "backtest/sharpe_net",
                 compares_dir="configs/compares") -> str:
    """The panel's verdict block for the selected pair, read-only on the store: NT-032's stored result in full when a
    pre-registered spec names the pair and the store holds its result (a result of an edited spec is refused as stale);
    "no verdict yet" with the command to run when it does not; the explicit statement plus the exploratory table when
    no spec names the pair."""
    from neural_trade.experiments.comparator import CompareSpec

    path, flipped = find_compare_spec(a, b, compares_dir)
    if path is None:
        return exploratory_html(a, b, metric)
    spec = CompareSpec.from_yaml(path)
    head = (f"<b>Paired verdict (NT-032, pre-registered spec <code>{html.escape(str(path))}</code>, "
            f"name {html.escape(spec.name)})</b>: A = {html.escape(spec.scenario_a)}"
            + (f" / {html.escape(spec.configuration_a)}" if spec.configuration_a else "")
            + f", B = {html.escape(spec.scenario_b)}"
            + (f" / {html.escape(spec.configuration_b)}" if spec.configuration_b else "")
            + (" (the selection's rows are in the reverse order)" if flipped else ""))
    doc, where = stored_verdict(path, spec, store)
    dirs = result_dirs(path, spec, store)
    cmd = html.escape(f"neural-trade compare {path} --out {dirs[0]}")
    if doc is None:
        return (f"{head}<br><b>no verdict yet</b>: the store holds no result for this spec (looked in "
                f"{html.escape(', '.join(str(d) for d in dirs))}; viewing never runs or registers a comparison). "
                f"Run <code>{cmd}</code> after the runs it names exist.<br>"
                + exploratory_html(a, b, metric, why="the pre-registered comparison has no stored result yet"))
    if doc.get("spec_hash") != spec.spec_hash:
        return (f"{head}<br><b>stored result is stale</b> ({html.escape(str(where))}): it was written for another content "
                f"of this spec (hash {html.escape(str(doc.get('spec_hash')))}, now {spec.spec_hash}). Run "
                f"<code>{cmd}</code> under a new name.")
    return f"{head}<br>stored in {html.escape(str(where))}<br>{result_html(doc, spec, where)}"


def table_html(configs: Sequence[ConfigScores]) -> str:
    """Every score key of the configurations as an HTML table."""
    df = metric_table(configs).astype(object)

    def cell(v: Any) -> str:
        if isinstance(v, str):
            return v
        return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.6g}"

    return df.applymap(cell).to_html(max_rows=None, classes="nt-scores")


__all__ = ["ConfigScores", "GROUPS", "MetricAgg", "aggregate_configuration", "aggregate_selection",
           "comparison_figures", "exploratory_html", "find_compare_spec", "key_parts", "metric_table",
           "paired_by_fold", "result_dirs", "result_html", "stored_verdict", "table_html", "verdict_html"]

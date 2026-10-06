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
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


@dataclass
class ConfigScores:
    label: str                                  # "scenario / configuration"
    scenario: str
    configuration: str
    role: str
    n_cells: int
    folds: Tuple[int, ...]
    aggs: Dict[str, MetricAgg] = field(default_factory=dict)


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


def aggregate_configuration(store: RunStore, rows: Sequence[Mapping[str, Any]], *, role: str = "dev") -> ConfigScores:
    """:class:`ConfigScores` of one configuration from its index ``rows`` (all one scenario and configuration):
    the cells of ``role`` that finished, every score key read from the index's ``scores`` table."""
    rows = [r for r in rows if r.get("role") == role and r.get("status") == "done"]
    scenario = str(rows[0]["scenario"]) if rows else ""
    configuration = str(rows[0]["configuration"]) if rows else ""
    folds = sorted({int(r["fold"]) for r in rows if r.get("fold") is not None})
    per_cell = [(r, store.index.scores(r["run_id"])) for r in rows]
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
                                      tuple(fold_means))
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
    (a pandas DataFrame; ``None`` where a configuration has no value)."""
    import pandas as pd

    keys = sorted({k for c in configs for k in c.aggs})
    cols: Dict[Tuple[str, str], List[Optional[float]]] = {}
    for c in configs:
        for part in ("value", "fold sd", "seed sd"):
            cols[(c.label, part)] = []
    for k in keys:
        for c in configs:
            a = c.aggs.get(k)
            cols[(c.label, "value")].append(a.value if a else None)
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
            spread = 0.62 / max(len(slots), 1)
            for si, h in enumerate(slots):
                members = by_h[h] if h is not None else by_h.get(None, {})
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
                    + ("; the test fold is shown, never used to rank (D-020)" if role == "test" else ""))
        T.apply(fig, title=f"{title}: {len(configs)} configurations", height=height, subtitle=subtitle)
        fig.update_layout(margin=dict(l=left, r=24, t=110, b=40), width=max(900, left + 230 * NCOLS))
        figs[group] = fig
    return figs


def paired_by_fold(a: ConfigScores, b: ConfigScores, metric: str) -> List[Dict[str, Any]]:
    """Exploratory pairs: for each fold both configurations have, A's and B's seed-mean of ``metric`` and A - B."""
    fa = dict(a.aggs[metric].fold_values) if metric in a.aggs else {}
    fb = dict(b.aggs[metric].fold_values) if metric in b.aggs else {}
    return [{"fold": f, "a": fa[f], "b": fb[f], "diff": fa[f] - fb[f]} for f in sorted(set(fa) & set(fb))]


def exploratory_html(a: ConfigScores, b: ConfigScores, metric: str) -> str:
    rows = paired_by_fold(a, b, metric)
    head = (f"<b>Exploratory paired differences of <code>{html.escape(metric)}</code> (A - B)</b>, per fold, seeds averaged "
            f"first (D-046): <b>not a verdict</b> (no pre-registered comparison names this pair; D-025).<br>"
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


def find_compare_spec(a: ConfigScores, b: ConfigScores, compares_dir="configs/compares"):
    """(CompareSpec, flipped) of the first pre-registered spec in ``compares_dir`` that names the pair
    (A, B) as (scenario_a, scenario_b) or, flipped, as (B, A); (None, False) when none does."""
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
            return spec, False
        if names(b, spec.scenario_a, spec.configuration_a) and names(a, spec.scenario_b, spec.configuration_b):
            return spec, True
    return None, False


def verdict_html(a: ConfigScores, b: ConfigScores, store: RunStore, *, metric: str = "backtest/sharpe_net",
                 compares_dir="configs/compares") -> str:
    """The panel's verdict block for the selected pair: NT-032's result when a pre-registered spec names it,
    else the explicit statement plus the exploratory table."""
    spec, flipped = find_compare_spec(a, b, compares_dir)
    if spec is None:
        return exploratory_html(a, b, metric)
    from neural_trade.experiments.comparator import compare

    result = compare(dataclasses.replace(spec, root=str(store.root)))
    est = result.estimate
    lines = [f"<b>Paired verdict (NT-032, pre-registered spec <code>{html.escape(spec.name)}</code>)</b>: "
             f"A = {html.escape(spec.scenario_a)}, B = {html.escape(spec.scenario_b)}"
             + (" (the selection's rows are in the reverse order)" if flipped else "")]
    if result.refusal_reason:
        lines.append(f"<b>Refused</b>: {html.escape(result.refusal_reason)}")
    else:
        lines.append(f"<b>{html.escape(result.verdict)}</b>: {html.escape(spec.metric)} {html.escape(spec.estimator)} "
                     f"{est['estimate']:.4g}, 95% CI [{est['ci_lo']:.4g}, {est['ci_hi']:.4g}] over {len(result.fold_rows)} "
                     f"judgement folds ({len(result.pairs)} pairs), minimum effect {spec.min_effect:g}")
    lines.append(f"<pre style='white-space:pre-wrap;max-height:260px;overflow:auto'>{html.escape(result.to_markdown())}</pre>")
    return "<br>".join(lines)


def table_html(configs: Sequence[ConfigScores]) -> str:
    """Every score key of the configurations as an HTML table."""
    df = metric_table(configs)
    return df.to_html(float_format=lambda v: f"{v:.6g}", na_rep="n/a", max_rows=None, classes="nt-scores")


__all__ = ["ConfigScores", "GROUPS", "MetricAgg", "aggregate_configuration", "aggregate_selection",
           "comparison_figures", "exploratory_html", "find_compare_spec", "key_parts", "metric_table",
           "paired_by_fold", "table_html", "verdict_html"]

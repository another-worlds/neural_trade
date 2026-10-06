"""One panel: permutation importance of each indicator family instance, with its noise band (NT-048).

The first panel is the loss importance from :class:`neural_trade.evaluation.permutation_importance.GroupImportance`
(neutral colour: an indicator never takes a horizon colour). Positive means shuffling that instance raised
the per-window loss. Then one panel per horizon, in that horizon's colour, with the drop of the direction
AUC; the hover also gives the hit-rate drop. Whiskers are the 2.5 and 97.5 percentiles of the
block-bootstrap resamples.
"""
from __future__ import annotations

from typing import Mapping

from neural_trade.visualization import theme as T


def _field(row, key):
    if isinstance(row, Mapping):
        return row[key]
    return getattr(row, key)


def _rows(data):
    if isinstance(data, Mapping):
        data = data["groups"]
    rows = list(data)
    if not rows:
        raise ValueError("permutation importance has no groups to draw")
    return rows


def _bars(go, names, vals, los, his, *, name, color, hover):
    plus = [max(h - v, 0.0) for v, h in zip(vals, his)]
    minus = [max(v - lo, 0.0) for v, lo in zip(vals, los)]
    return go.Bar(
        y=names, x=vals, orientation="h", name=name, showlegend=False,
        marker=dict(color=color, line=dict(width=0)),
        error_x=dict(type="data", array=plus, arrayminus=minus, color=T.MUTED, thickness=1.2, width=3),
        hovertext=hover, hoverinfo="text")


def permutation_importance(data, config=None, **_):
    """One panel of loss importance, then one panel per horizon of the direction AUC drop.

    ``data`` is a sequence of group results, or ``{"groups": ...}``. Every panel has the same rows
    (one per family instance) and the block-bootstrap 2.5 and 97.5 percentile whiskers.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = _rows(data)
    names = [str(_field(row, "name")) for row in rows]
    horizons = list(_field(rows[0], "auc"))
    loss = [float(_field(row, "loss")) for row in rows]
    lo = [float(_field(row, "loss_lo")) for row in rows]
    hi = [float(_field(row, "loss_hi")) for row in rows]

    def hlabel(h):
        return T.horizon_label(h, config) if h in T.HORIZONS else str(h)

    hover = []
    for k, (row, name) in enumerate(zip(rows, names)):
        lines = [f"{name}", f"loss importance {loss[k]:.4g} [{lo[k]:.4g}, {hi[k]:.4g}]"]
        for h in horizons:
            lines.append(f"{hlabel(h)} AUC drop {float(_field(row, 'auc')[h]):.4g} "
                         f"[{float(_field(row, 'auc_lo')[h]):.4g}, {float(_field(row, 'auc_hi')[h]):.4g}]")
            if hasattr(row, "hit_drop") or (isinstance(row, Mapping) and "hit_drop" in row):
                lines.append(f"{hlabel(h)} hit-rate drop {float(_field(row, 'hit_drop')[h]):.4g} "
                             f"[{float(_field(row, 'hit_lo')[h]):.4g}, {float(_field(row, 'hit_hi')[h]):.4g}]")
        hover.append("<br>".join(lines))
    titles = ["loss importance"] + [f"{hlabel(h)} AUC drop" for h in horizons]
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles, shared_yaxes=True,
                        horizontal_spacing=0.03)
    fig.add_trace(_bars(go, names, loss, lo, hi,
                        name="loss importance", color=T.NEUTRAL, hover=hover), row=1, col=1)
    for c, h in enumerate(horizons, start=2):
        vals = [float(_field(row, "auc")[h]) for row in rows]
        los = [float(_field(row, "auc_lo")[h]) for row in rows]
        his = [float(_field(row, "auc_hi")[h]) for row in rows]
        fig.add_trace(_bars(go, names, vals, los, his, name=f"{hlabel(h)} AUC drop",
                            color=T.HORIZON_COLORS.get(h, T.NEUTRAL), hover=hover), row=1, col=c)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(names)), automargin=True)
    fig.update_xaxes(zeroline=True)
    T.apply(
        fig,
        title="Indicator permutation importance",
        subtitle="One bar per family instance. Positive means shuffling it raised the per-window loss "
                 "or lowered the direction AUC. Whiskers are the block-bootstrap 2.5 and 97.5 percentiles.",
        height=max(320, 26 * len(rows) + 150),
        legend_top=False,
    )
    return fig

"""One panel: permutation importance of each indicator family instance, with its noise band (NT-048).

The bar is the loss importance from :class:`neural_trade.evaluation.permutation_importance.GroupImportance`.
Positive means shuffling that instance raised the per-window loss. Whiskers are the 2.5 and 97.5
percentiles of the block-bootstrap mean. Direction importance is named in the hover, not drawn:
horizon colours belong to horizons, and this figure uses none of them and none of the copy colours.
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


def permutation_importance(data, config=None, **_):
    """Bar per group. ``data`` is a sequence of group results, or ``{"groups": ...}``."""
    import plotly.graph_objects as go

    rows = _rows(data)
    names = [str(_field(row, "name")) for row in rows]
    loss = [float(_field(row, "loss")) for row in rows]
    lo = [float(_field(row, "loss_lo")) for row in rows]
    hi = [float(_field(row, "loss_hi")) for row in rows]
    hover = []
    for row, name, point, left, right in zip(rows, names, loss, lo, hi):
        lines = [f"{name}", f"loss importance {point:.4g} [{left:.4g}, {right:.4g}]"]
        auc = _field(row, "auc")
        for h in auc:
            label = T.horizon_label(h, config) if h in T.HORIZONS else str(h)
            lines.append(
                f"{label} direction {float(auc[h]):.4g} "
                f"[{float(_field(row, 'auc_lo')[h]):.4g}, {float(_field(row, 'auc_hi')[h]):.4g}]"
            )
        hover.append("<br>".join(lines))
    plus = [max(right - point, 0.0) for point, right in zip(loss, hi)]
    minus = [max(point - left, 0.0) for point, left in zip(loss, lo)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=loss, orientation="h", name="loss importance", showlegend=False,
        marker=dict(color=T.NEUTRAL, line=dict(width=0)),
        error_x=dict(type="data", array=plus, arrayminus=minus, color=T.MUTED, thickness=1.2, width=3),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(names)), automargin=True)
    fig.update_xaxes(title_text="loss importance", zeroline=True)
    T.apply(
        fig,
        title="Indicator permutation importance",
        subtitle="One bar per family instance. Positive means shuffling it raised the per-window loss. "
                 "Whiskers are the block-bootstrap 2.5 and 97.5 percentiles.",
        height=max(320, 26 * len(rows) + 150),
        legend_top=False,
    )
    return fig

"""One visual system for every neural_trade figure: a dark Plotly template and fixed colour roles.

Colours come from a validated categorical palette (dark steps, checked for colour-blind
separation on the chart surface) and are assigned by role, never cycled:

* horizons: h0 / h1 / h2 always take slots 1-3 (blue, orange, aqua) in every figure;
* train vs validation: the same colour, validation solid and training dotted;
* outcomes (win / loss, good / bad) use the status colours, always with a marker shape too.

    from neural_trade.visualization.theme import apply, HORIZON_COLORS
    fig = apply(go.Figure(...), title="...", height=500)
"""
from __future__ import annotations

from typing import Optional

# ------------------------------------------------------------------ surfaces and ink
PAPER = "#121211"
SURFACE = "#1a1a19"
INK = "#f2f1ec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
NEUTRAL = "#9a9890"          # reference lines: "perfect", zero, 50%

# ------------------------------------------------------------------ categorical slots (dark steps)
SERIES = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
HORIZONS = ("h0", "h1", "h2")
HORIZON_COLORS = dict(zip(HORIZONS, SERIES[:3]))

# ------------------------------------------------------------------ status (never used for a series)
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# Up / long and down / short: violet and yellow (never a horizon colour), always with a shape
# (triangle up / down). The realised class in the direction figures uses the same pair.
LONG_COLOR = UP_COLOR = SERIES[6]
SHORT_COLOR = DOWN_COLOR = SERIES[3]
# Categorical series that are not horizons (strategies, loss components, runs): the slots after
# the three horizon colours, then neutral inks. Assign in this order, never cycle into h0-h2.
OTHER_SERIES = (SERIES[3], SERIES[4], SERIES[5], SERIES[6], SERIES[7], "#c3c2b7", "#8a8984")

VAL_DASH = "solid"
TRAIN_DASH = "dot"
ALT_DASH = "dashdot"         # a third reading of the same quantity (e.g. the Gaussian direction readout)

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
TEMPLATE_NAME = "neural_trade"


def _template():
    import plotly.graph_objects as go

    axis = dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS, zerolinewidth=1,
                tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=INK_2, size=12)),
                showspikes=True, spikecolor=MUTED, spikethickness=1, spikedash="dot", spikemode="across",
                spikesnap="cursor", automargin=True)
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT, color=INK_2, size=12),
            title=dict(font=dict(color=INK, size=15), x=0.01, xanchor="left", yref="container", y=1.0,
                       yanchor="top", pad=dict(t=14)),
            paper_bgcolor=PAPER, plot_bgcolor=SURFACE, colorway=list(SERIES),
            xaxis=axis, yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=11), itemsizing="constant", itemwidth=30,
                        groupclick="togglegroup"),
            hoverlabel=dict(bgcolor="#262624", bordercolor=AXIS, font=dict(color=INK, family=FONT, size=12)),
            margin=dict(l=64, r=24, t=72, b=48),
            annotationdefaults=dict(font=dict(color=INK_2, size=12)),
            shapedefaults=dict(line=dict(color=NEUTRAL)),
            bargap=0.15,
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2), marker=dict(size=7))],
            scattergl=[go.Scattergl(line=dict(width=1.5), marker=dict(size=5))],
            histogram=[go.Histogram(marker=dict(line=dict(width=0)))],
        ),
    )


def register() -> str:
    """Register the template with plotly (idempotent); returns its name."""
    import plotly.io as pio

    if TEMPLATE_NAME not in pio.templates:
        pio.templates[TEMPLATE_NAME] = _template()
    return TEMPLATE_NAME


def apply(fig, *, title: Optional[str] = None, height: Optional[int] = None, subtitle: Optional[str] = None,
          legend_top: bool = True):
    """Style ``fig`` with the template; ``subtitle`` goes on a second, smaller title line."""
    register()
    fig.update_layout(template=TEMPLATE_NAME)
    if title is not None:
        text = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
                                    if subtitle else "")
        fig.update_layout(title_text=text)
    if height is not None:
        fig.update_layout(height=height)
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0))
    # subplot titles are annotations: give them the secondary ink, a size below the figure title
    for a in fig.layout.annotations or ():
        if a.font is None or a.font.size is None:
            a.update(font=dict(color=INK_2, size=12))
    return fig


def horizon_label(h: str, config=None) -> str:
    """'h1 (15 bars)' when the config knows the horizon steps, else 'h1'."""
    steps = getattr(config, "HORIZON_STEPS", None) if config is not None else None
    if steps is not None:
        i = HORIZONS.index(h)
        if i < len(steps):
            return f"{h} ({steps[i]} bars)"
    return h


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def empty_panels(fig) -> list:
    """Subplot axes that hold no trace (a panel drawn with nothing in it). Used by the tests and by
    the dashboards to put a 'not logged' note instead of a blank frame."""
    used = {(getattr(t, "xaxis", None) or "x", getattr(t, "yaxis", None) or "y") for t in fig.data}
    used_y = {y for _, y in used}
    out = []
    for name in fig.layout:
        if name.startswith("yaxis"):
            ref = "y" + name[5:]
            if ref not in used_y:
                out.append(ref)
    return out


def note_on_empty(fig, text: str = "not logged in this run"):
    """Write ``text`` in the middle of every empty panel, so a missing metric is visible as such."""
    for yref in empty_panels(fig):
        ax = fig.layout["yaxis" + yref[1:]]
        xref = ax.anchor or "x"
        xd = fig.layout["xaxis" + xref[1:]].domain or (0, 1)
        yd = ax.domain or (0, 1)
        fig.add_annotation(x=(xd[0] + xd[1]) / 2, y=(yd[0] + yd[1]) / 2, xref="paper", yref="paper",
                           text=text, showarrow=False, font=dict(color=MUTED, size=12))
    return fig


def panel_legend(fig, legend_id: str, row: int, col: int, title: str):
    """Give one subplot its own legend, drawn as the panel's heading (title, then the keys in a row).

    The panel's subplot title (if it has the same text) is removed so the two do not overlap.
    Traces join it with ``legend=legend_id``.
    """
    sp = fig.get_subplot(row, col)
    xd, yd = sp.xaxis.domain, sp.yaxis.domain
    fig.update_layout({legend_id: dict(
        title=dict(text=f"<b>{title}</b>   ", side="left", font=dict(color=INK_2, size=12)),
        orientation="h", x=xd[0], xanchor="left", y=yd[1] + 0.002, yanchor="bottom",
        font=dict(size=10, color=INK_2), bgcolor="rgba(0,0,0,0)", itemsizing="constant", itemwidth=30,
        groupclick="togglegroup", tracegroupgap=0)})
    keep = [a for a in (fig.layout.annotations or ()) if a.text != title]
    fig.layout.annotations = keep
    return fig


def positive(y):
    """Values for a log axis: non-positive entries become gaps instead of vertical drops to -inf."""
    import numpy as np

    if y is None:
        return None
    y = np.asarray(y, dtype=float).copy()
    y[~(y > 0)] = np.nan
    return y


def legend_once():
    """Closure: True the first time a legend group name is seen (show each group's legend entry once
    in a multi-panel figure where the same series appears in several panels)."""
    seen = set()

    def first(name: str) -> bool:
        if name in seen:
            return False
        seen.add(name)
        return True

    return first

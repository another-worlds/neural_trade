"""The leaderboard figure (NT-031, D-014): configurations ranked by the dev-fold net Sharpe after
costs, with the table of every row below.

Bars (one lane per configuration, rank 1 at the top):

* the bar is the dev net Sharpe (the ranking column). Colour says eligibility, never a horizon
  (theme.py: SERIES[:3] are reserved for h0-h2): the winner in the status colour GOOD **with a star
  marker** and "winner" in its label; an eligible row in a neutral ink; a disqualified or not-comparable
  row dimmed **and hatched** ("/"), with "DISQUALIFIED" or "not comparable" in its label;
* both spreads where they exist, as separate whiskers: the fold sd (between dev-fold means, D-046's
  unit of inference) on the bar, in ink; the seed sd (mean over folds of the sd across a fold's seeds)
  just above it, in its own colour. A row with neither (one cell) says "no spread (1 cell)"; a row
  with no scored dev cell says so;
* the test-fold net Sharpe as an open diamond just below, labelled "test, not used for ranking" (D-020).

Every row label carries the cost profile its stored net Sharpe was computed with, and the subtitle
states the board's profile (the header line of :func:`neural_trade.experiments.leaderboard.cost_header`).
The table keeps every column the leaderboard computes (D-014: no simplified tier); every cell is
wrapped to its column's character budget (``COLUMN_CHARS``) and the column widths, the row height and
the figure size follow from those budgets, so no text is cut off.

:func:`write_png` writes a PNG (kaleido when installed, else headless Microsoft Edge, as
``scripts/notebooks/render.py`` does); ``neural-trade leaderboard --out`` uses it.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

from neural_trade.visualization import theme as T
from neural_trade.visualization.theme import apply

from neural_trade.experiments.leaderboard import TABLE_HEADER, LeaderboardRow, cost_header, table_cells

logger = logging.getLogger(__name__)

# characters per line of each table column, in TABLE_HEADER order
COLUMN_CHARS = (10, 22, 24, 28, 30, 14, 14, 14, 14, 12, 44, 18, 14, 14, 14, 14, 8, 10, 20)
CHAR_PX, CELL_PAD_PX, LINE_PX = 7.0, 16, 15     # generous for 11 px system-ui text
LANE_PX = 64                                   # one configuration's lane in the bar panel
MARGIN_B, MARGIN_R, GAP_PX = 30, 24, 95

ELIGIBLE = T.NEUTRAL
WINNER = T.GOOD
DISQUALIFIED = T.MUTED
FOLD_SD = T.INK
SEED_SD = T.OTHER_SERIES[1]
TEST_MARK = T.OTHER_SERIES[0]


def _fmt(v, fmt: str = "{:+.3f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def wrap_cell(text: str, width: int) -> List[str]:
    """The lines of one cell: split after '; ' (one guard-rail per line), then wrapped to ``width``
    characters, breaking a word longer than the line."""
    chunks = [c + (";" if i < len(text.split("; ")) - 1 else "") for i, c in enumerate(text.split("; "))]
    lines: List[str] = []
    for c in chunks:
        lines += textwrap.wrap(c, width=width, break_long_words=True, break_on_hyphens=False) or [""]
    return lines


def _escape(text: str) -> str:
    """'<' and '>' as entities (plotly reads a '<' as the start of a tag); nothing else changes."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _cell_html(lines: Sequence[str]) -> str:
    return "<br>".join(_escape(x) for x in lines)


def _label(r: LeaderboardRow) -> str:
    tags = []
    if r.is_winner:
        tags.append("winner")
    if r.disqualified:
        tags.append("DISQUALIFIED")
    if r.status != "failed" and not r.cost_comparable:
        tags.append("not comparable")
    cost = r.cost_profile.short() if r.cost_profile is not None else r.cost_text.split(":")[0]
    return f"#{r.rank} {r.configuration} · {cost}" + (f" [{', '.join(tags)}]" if tags else "")


def leaderboard_figure(rows: Sequence[LeaderboardRow], *, title: Optional[str] = None, height: Optional[int] = None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scenario = rows[0].scenario if rows else "(no runs)"
    n = max(1, len(rows))
    # ---- table geometry from the character budgets
    header_lines = [wrap_cell(h, w) for h, w in zip(TABLE_HEADER, COLUMN_CHARS)]
    body = [[wrap_cell(v, w) for v, w in zip(table_cells(r), COLUMN_CHARS)] for r in rows]
    col_px = [w * CHAR_PX + CELL_PAD_PX for w in COLUMN_CHARS]
    header_h = max(len(x) for x in header_lines) * LINE_PX + 12
    cell_h = max([len(c) for row in body for c in row] or [1]) * LINE_PX + 12
    table_px = header_h + cell_h * len(rows) + 10
    labels_by_rank = [_label(r) for r in rows]
    margin_l = int(max([len(x) for x in labels_by_rank] or [10]) * CHAR_PX) + 24
    width = int(margin_l + sum(col_px) + MARGIN_R)
    bars_px = 70 + LANE_PX * n
    plot_h = bars_px + GAP_PX + table_px
    # title, the wrapped cost header, the legend, then the panel title: each on its own band
    sub_lines = textwrap.wrap(cost_header(rows), width=max(60, int((width - 40) / CHAR_PX)))
    subtitle = "<br>".join(_escape(x) for x in sub_lines)
    title_px = 40 + 16 * len(sub_lines)
    margin_t = title_px + 70
    fig = make_subplots(rows=2, cols=1, row_heights=[bars_px, table_px], vertical_spacing=GAP_PX / plot_h,
                        specs=[[{"type": "xy"}], [{"type": "table"}]],
                        subplot_titles=("Dev-fold net Sharpe after costs (ranking column) with fold sd and seed sd "
                                        "-- test-fold Sharpe shown, not used for ranking",
                                        "Every row (guard-rails, cost profile, both roles' numbers, provenance)"))
    if not rows:
        T.note_on_empty(fig, "no runs in this scenario's index yet")
        return apply(fig, title=title or f"Leaderboard: {scenario}", height=height or 520)

    ordered = list(reversed(rows))          # rank 1 drawn at the top
    ys = list(range(len(ordered)))
    labels = [_label(r) for r in ordered]

    def dev_v(r):
        v = r.dev.values.get("sharpe_net")
        return v if v is not None and v == v and abs(v) != float("inf") else None

    def hover(r):
        gr = "; ".join(f"{g.name} {'OK' if g.passed else 'FAIL'} ({g.detail})" for g in r.guard_rails) or "n/a"
        return (f"{r.configuration} (rank {r.rank}, {r.status})<br>dev net Sharpe {_fmt(r.dev.values.get('sharpe_net'))} "
                f"(fold sd {_fmt(r.dev.spread.get('sharpe_net'), '{:.3f}')}, seed sd "
                f"{_fmt(r.dev.seed_spread.get('sharpe_net'), '{:.3f}')}, {r.dev.n_folds} folds x {r.dev.n_seeds} seeds, "
                f"{r.dev.n_rows} cells)<br>cost profile: {_escape(r.cost_text)}<br>guard-rails: {_escape(gr)}")

    classes = (("winner (top row passing every guard-rail)", lambda r: r.is_winner, WINNER, 0.9, ""),
               ("eligible", lambda r: not r.is_winner and not r.disqualified, ELIGIBLE, 0.75, ""),
               ("disqualified or not comparable (hatched)", lambda r: r.disqualified, DISQUALIFIED, 0.45, "/"))
    for name, pick, color, alpha, shape in classes:
        idx = [i for i, r in enumerate(ordered) if pick(r)]
        if not idx:
            continue
        fig.add_trace(go.Bar(
            y=[ys[i] for i in idx], x=[dev_v(ordered[i]) or 0.0 for i in idx], orientation="h", width=0.5,
            name=f"dev net Sharpe: {name}",
            marker=dict(color=T.rgba(color, alpha), line=dict(color=color, width=1.2),
                        pattern=dict(shape=shape, fgcolor=T.INK_2, size=7, solidity=0.25)),
            customdata=[hover(ordered[i]) for i in idx], hovertemplate="%{customdata}<extra></extra>"), 1, 1)

    def whisker(name, key, offset, color, symbol):
        idx = [i for i, r in enumerate(ordered) if dev_v(r) is not None and getattr(r.dev, key).get("sharpe_net")
               is not None]
        if not idx:
            return
        fig.add_trace(go.Scatter(
            y=[ys[i] + offset for i in idx], x=[dev_v(ordered[i]) for i in idx], mode="markers", name=name,
            marker=dict(symbol=symbol, size=12, color=color, line=dict(color=color, width=2)),
            error_x=dict(type="data", array=[getattr(ordered[i].dev, key)["sharpe_net"] for i in idx],
                         color=color, thickness=2, width=6, visible=True),
            customdata=[f"{ordered[i].configuration}: {name} {getattr(ordered[i].dev, key)['sharpe_net']:.3f}"
                        for i in idx], hovertemplate="%{customdata}<extra></extra>"), 1, 1)

    whisker("fold sd (between dev-fold means, D-046)", "spread", 0.0, FOLD_SD, "line-ns")
    whisker("seed sd (mean over folds of the sd across a fold's seeds)", "seed_spread", 0.3, SEED_SD, "line-ns")

    notes = [(i, "no spread (1 cell)" if r.dev.n_rows == 1 else f"no spread ({r.dev.n_rows} cells)")
             for i, r in enumerate(ordered) if dev_v(r) is not None and r.dev.spread.get("sharpe_net") is None
             and r.dev.seed_spread.get("sharpe_net") is None]
    notes += [(i, f"no scored dev cell ({r.status})") for i, r in enumerate(ordered) if dev_v(r) is None]
    if notes:
        fig.add_trace(go.Scatter(
            # in the (empty) seed-sd lane above the bar, written from the bar's end toward zero, so it
            # never sits on the hatching and never runs off the axis
            y=[ys[i] + 0.3 for i, _ in notes], x=[dev_v(ordered[i]) or 0.0 for i, _ in notes], mode="text",
            text=[f"  {t}  " for _, t in notes],
            textposition=["middle right" if (dev_v(ordered[i]) or 0.0) < 0 else "middle left" for i, _ in notes],
            textfont=dict(color=T.INK, size=11),
            name="spread note", showlegend=False, hoverinfo="skip"), 1, 1)

    win = [i for i, r in enumerate(ordered) if r.is_winner]
    if win:
        fig.add_trace(go.Scatter(
            y=[ys[i] for i in win], x=[dev_v(ordered[i]) or 0.0 for i in win], mode="markers",
            name="winner marker", marker=dict(symbol="star", size=18, color=WINNER, line=dict(color=T.INK, width=1)),
            hovertemplate="winner<extra></extra>"), 1, 1)

    test_idx = [i for i, r in enumerate(ordered) if r.test.values.get("sharpe_net") is not None]
    if test_idx:
        fig.add_trace(go.Scatter(
            y=[ys[i] - 0.3 for i in test_idx], x=[ordered[i].test.values["sharpe_net"] for i in test_idx],
            mode="markers", name="test net Sharpe (test, not used for ranking)",
            marker=dict(symbol="diamond-open", size=11, color=TEST_MARK, line=dict(width=2)),
            customdata=[f"{ordered[i].configuration}: test net Sharpe {_fmt(ordered[i].test.values['sharpe_net'])} "
                        f"({ordered[i].test.n_folds} folds / {ordered[i].test.n_rows} cells) -- test, not used for "
                        "ranking" for i in test_idx], hovertemplate="%{customdata}<extra></extra>"), 1, 1)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, width=1, dash="6px,3px"), row=1, col=1)   # dotted = training (D-014)
    fig.update_yaxes(tickvals=ys, ticktext=labels, range=[-0.6, len(ordered) - 0.4], automargin=False, row=1, col=1)
    fig.update_xaxes(title_text="dev net Sharpe after costs (annualised)", row=1, col=1)
    fig.update_layout(barmode="overlay")

    row_colors = [T.rgba(T.MUTED, 0.18) if r.disqualified else (T.rgba(WINNER, 0.12) if r.is_winner
                  else "rgba(0,0,0,0)") for r in rows]
    fig.add_trace(go.Table(
        columnwidth=col_px,
        header=dict(values=[_cell_html(h) for h in header_lines], fill_color=T.SURFACE,
                    font=dict(color=T.INK, size=11), align="left", height=header_h),
        cells=dict(values=[[_cell_html(body[r][c]) for r in range(len(rows))] for c in range(len(TABLE_HEADER))],
                   fill_color=[row_colors] * len(TABLE_HEADER), font=dict(color=T.INK_2, size=11),
                   align="left", height=cell_h)), 2, 1)

    fig = apply(fig, title=title or f"Leaderboard: {scenario}", subtitle=subtitle,
                height=height or int(plot_h + margin_t + MARGIN_B))
    total_h = fig.layout.height
    fig.update_layout(width=width, margin=dict(l=margin_l, r=MARGIN_R, t=margin_t, b=MARGIN_B),
                      legend=dict(orientation="h", yref="container", y=1 - (title_px + 4) / total_h,
                                  yanchor="top", x=0.0, xanchor="left"))
    return fig


def write_png(fig, png, *, tries: int = 3) -> bool:
    """Write ``fig`` to ``png``: kaleido when installed, else headless Microsoft Edge (Windows). True
    when the PNG exists; False (and a warning) when neither is available."""
    png = Path(png).resolve()
    try:
        import kaleido  # noqa: F401
    except ImportError:
        pass
    else:
        fig.write_image(str(png))
        return png.is_file()
    edge = shutil.which("msedge") or shutil.which("msedge.exe") or next(
        (p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).is_file()), None)
    if edge is None:
        logger.warning("leaderboard PNG not written (%s): neither kaleido nor Microsoft Edge is available", png)
        return False
    width, height = int(fig.layout.width or 1500), int(fig.layout.height or 600)
    html_path = png.with_suffix(".png.html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True, default_width=f"{width}px",
                   config={"displayModeBar": False})
    try:
        for _ in range(tries):
            png.unlink(missing_ok=True)
            profile = Path(tempfile.mkdtemp(prefix="edge_profile_", dir=png.parent))
            try:
                subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                                "--no-default-browser-check", "--disable-extensions", f"--user-data-dir={profile}",
                                f"--window-size={width + 20},{height + 20}", "--virtual-time-budget=8000",
                                f"--screenshot={png}", html_path.as_uri()], capture_output=True, timeout=120)
            except subprocess.TimeoutExpired:
                pass
            finally:
                shutil.rmtree(profile, ignore_errors=True)
            if png.is_file() and png.stat().st_size > 0:
                return True
        logger.warning("leaderboard PNG not written (%s): Edge produced no screenshot", png)
        return False
    finally:
        html_path.unlink(missing_ok=True)


def leaderboard(data, config=None, **kw):
    """``data``: the list of ``experiments.leaderboard.LeaderboardRow`` (Visualizations registry)."""
    return leaderboard_figure(data, **kw)


__all__ = ["COLUMN_CHARS", "leaderboard", "leaderboard_figure", "wrap_cell", "write_png"]

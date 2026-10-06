"""The leaderboard figure (NT-031, D-014): configurations ranked by the dev-fold net Sharpe after
costs, each with its spread over folds and seeds; the test-fold Sharpe sits beside it, labelled
"test, not used for ranking" (D-020); a guard-rail breach is drawn as a disqualified (hatched,
dimmed) bar, named in its hover text and in the table below. The table keeps every column the
leaderboard computes (D-014: no simplified tier) -- status, both roles' return / drawdown / trade
count, the guard-rail detail, the dataset fingerprint, the bar size, the horizons and the strategy.
"""
from __future__ import annotations

from typing import Optional, Sequence

from neural_trade.visualization import theme as T
from neural_trade.visualization.theme import apply

from neural_trade.experiments.leaderboard import TABLE_HEADER, LeaderboardRow, table_cells, winner


def _fmt(v, fmt: str = "{:+.3f}") -> str:
    return "n/a" if v is None else fmt.format(v)


BAR_PX, TABLE_ROW_PX = 40, 175      # a wrapped guard-rail cell needs about 175 px of table row


def _height(n: int) -> int:
    return 230 + BAR_PX * n + 90 + TABLE_ROW_PX * n


def _row_heights(n: int):
    bars, table = 100 + BAR_PX * n, 90 + TABLE_ROW_PX * n
    return [bars / (bars + table), table / (bars + table)]


def leaderboard_figure(rows: Sequence[LeaderboardRow], *, title: Optional[str] = None, height: Optional[int] = None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scenario = rows[0].scenario if rows else "(no runs)"
    fig = make_subplots(rows=2, cols=1, row_heights=_row_heights(max(1, len(rows))), vertical_spacing=0.12,
                        specs=[[{"type": "xy"}], [{"type": "table"}]],
                        subplot_titles=("Dev-fold net Sharpe after costs (ranking column) -- "
                                        "test-fold Sharpe shown, not used for ranking",
                                        "Every row (guard-rails, both roles' numbers, provenance)"))
    if not rows:
        T.note_on_empty(fig, "no runs in this scenario's index yet")
        return apply(fig, title=title or f"Leaderboard: {scenario}", height=height or 520)

    ordered = list(reversed(rows))          # rank 1 drawn at the top of a horizontal bar chart
    labels = [f"#{r.rank} {r.configuration}" + ("  [DISQUALIFIED]" if r.disqualified else "") for r in ordered]
    dev_v = [r.dev.values.get("sharpe_net") for r in ordered]
    dev_e = [r.dev.spread.get("sharpe_net") or r.dev.seed_spread.get("sharpe_net") or 0.0 for r in ordered]
    top = winner(rows)
    dev_x = [v if v is not None else 0.0 for v in dev_v]
    colors = [T.MUTED if r.disqualified else (T.GOOD if r is top else T.SERIES[0]) for r in ordered]
    dev_hover = []
    for r in ordered:
        gr = "; ".join(f"{g.name} {'OK' if g.passed else 'FAIL'} ({g.detail})" for g in r.guard_rails) or "n/a"
        dev_hover.append(f"{r.configuration} (rank {r.rank}, {r.status})<br>dev net Sharpe "
                         f"{_fmt(r.dev.values.get('sharpe_net'))} (fold sd {_fmt(r.dev.spread.get('sharpe_net'), '{:.3f}')}, "
                         f"seed sd {_fmt(r.dev.seed_spread.get('sharpe_net'), '{:.3f}')}, {r.dev.n_folds} folds x "
                         f"{r.dev.n_seeds} seeds, {r.dev.n_rows} cells)<br>guard-rails: {gr}")
    fig.add_trace(go.Bar(y=labels, x=dev_x, orientation="h", name="dev net Sharpe (ranking)",
                         marker=dict(color=colors), error_x=dict(type="data", array=dev_e, color=T.MUTED,
                                                                  thickness=1.3, visible=True),
                         customdata=dev_hover, hovertemplate="%{customdata}<extra></extra>"), 1, 1)
    test_v = [r.test.values.get("sharpe_net") for r in ordered]
    test_x = [v for v in test_v if v is not None]
    test_y = [lab for lab, v in zip(labels, test_v) if v is not None]
    test_hover = [f"{r.configuration}: test net Sharpe {_fmt(r.test.values.get('sharpe_net'))} "
                 f"({r.test.n_folds} folds / {r.test.n_rows} cells) -- test, not used for ranking"
                 for r, v in zip(ordered, test_v) if v is not None]
    fig.add_trace(go.Scatter(y=test_y, x=test_x, mode="markers", name="test net Sharpe (not used for ranking)",
                             marker=dict(symbol="diamond-open", size=11, color=T.OTHER_SERIES[0], line=dict(width=2)),
                             customdata=test_hover, hovertemplate="%{customdata}<extra></extra>"), 1, 1)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, width=1, dash="dot"), row=1, col=1)

    header = list(TABLE_HEADER)
    cols = [[] for _ in header]
    for r in rows:
        for c, v in zip(cols, table_cells(r)):
            c.append(v)
    row_colors = [T.rgba(T.MUTED, 0.18) if r.disqualified else "rgba(0,0,0,0)" for r in rows]
    fig.add_trace(go.Table(
        header=dict(values=header, fill_color=T.SURFACE, font=dict(color=T.INK, size=11), align="left"),
        cells=dict(values=cols, fill_color=[row_colors] * len(header), font=dict(color=T.INK_2, size=11),
                  align="left", height=TABLE_ROW_PX - 20)), 2, 1)

    fig = apply(fig, title=title or f"Leaderboard: {scenario}", height=height or _height(len(rows)))
    return fig


def leaderboard(data, config=None, **kw):
    """``data``: the list of ``experiments.leaderboard.LeaderboardRow`` (Visualizations registry)."""
    return leaderboard_figure(data, **kw)


__all__ = ["leaderboard", "leaderboard_figure"]

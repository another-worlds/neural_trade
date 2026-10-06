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

from neural_trade.experiments.leaderboard import LeaderboardRow, winner


def _fmt(v, fmt: str = "{:+.3f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def leaderboard_figure(rows: Sequence[LeaderboardRow], *, title: Optional[str] = None, height: Optional[int] = None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scenario = rows[0].scenario if rows else "(no runs)"
    fig = make_subplots(rows=2, cols=1, row_heights=[0.4, 0.6], vertical_spacing=0.16,
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
    dev_e = [r.dev.spread.get("sharpe_net") or 0.0 for r in ordered]
    dev_x = [v if v is not None else 0.0 for v in dev_v]
    colors = [T.MUTED if r.disqualified else (T.GOOD if r is winner(rows) else T.SERIES[0]) for r in ordered]
    dev_hover = []
    for r in ordered:
        gr = "; ".join(f"{g.name} {'OK' if g.passed else 'FAIL'} ({g.detail})" for g in r.guard_rails) or "n/a"
        dev_hover.append(f"{r.configuration} (rank {r.rank}, {r.status})<br>dev net Sharpe "
                         f"{_fmt(r.dev.values.get('sharpe_net'))} (sd {_fmt(r.dev.spread.get('sharpe_net'), '{:.3f}')}"
                         f", {r.dev.n_folds} folds / {r.dev.n_rows} cells)<br>guard-rails: {gr}")
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

    header = ["rank", "configuration", "status", "ranking: dev net Sharpe (sd, n folds/cells)", "dev return",
             "dev max dd", "dev trades", "guard-rails", "test: net Sharpe (test, not used for ranking)",
             "dataset fingerprint", "bar (min)", "horizons", "strategy"]
    cols = [[] for _ in header]
    for r in rows:
        gr = "; ".join(f"{g.name} {'OK' if g.passed else 'FAIL'}" for g in r.guard_rails) or "n/a"
        if r.disqualified:
            gr = "DISQUALIFIED: " + gr
        fp = r.dataset_fingerprint
        fp = "n/a" if not fp else f"{fp[:12]}..."
        vals = [r.rank, r.configuration, r.status,
               f"{_fmt(r.dev.values.get('sharpe_net'))} (sd {_fmt(r.dev.spread.get('sharpe_net'), '{:.3f}')}, "
               f"{r.dev.n_folds}f/{r.dev.n_rows}c)",
               _fmt(r.dev.values.get("total_return"), "{:+.2%}"), _fmt(r.dev.values.get("max_drawdown"), "{:.2%}"),
               _fmt(r.dev.values.get("n_trades"), "{:.1f}"), gr,
               f"{_fmt(r.test.values.get('sharpe_net'))} ({r.test.n_folds}f/{r.test.n_rows}c)",
               fp, "n/a" if r.bar_minutes is None else f"{r.bar_minutes:g}",
               "n/a" if r.horizon_steps is None else ", ".join(str(h) for h in r.horizon_steps),
               r.strategy or "n/a"]
        for c, v in zip(cols, vals):
            c.append(v)
    row_colors = [T.rgba(T.MUTED, 0.18) if r.disqualified else "rgba(0,0,0,0)" for r in rows]
    fig.add_trace(go.Table(
        header=dict(values=header, fill_color=T.SURFACE, font=dict(color=T.INK, size=11), align="left"),
        cells=dict(values=cols, fill_color=[row_colors] * len(header), font=dict(color=T.INK_2, size=11),
                  align="left", height=24)), 2, 1)

    n = max(1, len(rows))
    fig = apply(fig, title=title or f"Leaderboard: {scenario}",
               height=height or max(560, 110 + 36 * n + 30 * n + 280))
    return fig


def leaderboard(data, config=None, **kw):
    """``data``: the list of ``experiments.leaderboard.LeaderboardRow`` (Visualizations registry)."""
    return leaderboard_figure(data, **kw)


__all__ = ["leaderboard", "leaderboard_figure"]

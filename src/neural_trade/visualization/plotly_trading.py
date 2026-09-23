"""Backtest figure: price with entries/exits and the stop/target levels, equity (net and gross),
and the position held (moved from the notebooks' trade plots, drawn from a BacktestResult)."""
from __future__ import annotations

import numpy as np


def plotly_trading(data, config=None, *, bars=None, title=None, **_):
    """``data``: a BacktestResult; ``bars``: the Bars it ran on (for the price panel)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    res = data
    n = len(res.position)
    x = np.arange(n)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.04,
                        subplot_titles=("Price and trades", "Equity", "Position"))
    if bars is not None:
        fig.add_trace(go.Scatter(x=x, y=bars.close, name="close", line=dict(width=1, color="#6b7280")), 1, 1)
    for side, color, symbol in (("LONG", "#15803d", "triangle-up"), ("SHORT", "#b91c1c", "triangle-down")):
        tr = [t for t in res.trades if t.side == side]
        if tr:
            fig.add_trace(go.Scatter(
                x=[t.entry_bar for t in tr], y=[t.entry_price for t in tr], mode="markers", name=f"{side} entry",
                marker=dict(symbol=symbol, size=9, color=color),
                hovertext=[f"{t.entry_reason} size {t.notional:,.0f}" for t in tr]), 1, 1)
            fig.add_trace(go.Scatter(
                x=[t.exit_bar for t in tr], y=[t.exit_price for t in tr], mode="markers", name=f"{side} exit",
                marker=dict(symbol="x", size=8, color=color),
                hovertext=[f"{t.exit_reason} net {t.net_pnl:+,.2f}" for t in tr]), 1, 1)
    fig.add_trace(go.Scatter(x=np.arange(-1, n), y=res.equity, name="equity (net)", line=dict(width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=np.arange(-1, n), y=res.equity_gross, name="equity (gross)",
                             line=dict(width=1, dash="dot")), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=res.position, name="position", line=dict(width=1), fill="tozeroy"), 3, 1)
    s = res.summary
    fig.update_layout(
        title=title or (f"{res.strategy}: {s['n_trades']} trades, net return {100 * s['total_return']:+.2f}%, "
                        f"Sharpe {s['sharpe_net']:.2f}, max DD {100 * s['max_drawdown']:.2f}%"),
        height=760, hovermode="x unified")
    fig.update_xaxes(title_text="bar", row=3, col=1)
    return fig

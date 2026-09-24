"""Trading dashboard, per-trade analytics, strategy comparison."""
from __future__ import annotations

import numpy as np

from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")


def test_trading_dashboard_shares_one_x_axis_and_marks_every_trade(viz_backtest):
    from neural_trade.visualization.trading_dashboard import trading_dashboard_figure

    res, bars, sig, strat = viz_backtest
    fig = trading_dashboard_figure(res, bars, sig, strat)
    assert {t.xaxis for t in fig.data} == {"x"} and fig.layout.hoversubplots == "axis"
    assert T.empty_panels(fig) == []
    entries = sum(len(t.x) for t in fig.data if t.name in ("long entry", "short entry"))
    exits = sum(len(t.x) for t in fig.data if t.name in ("exit, win", "exit, loss"))
    assert entries == exits == res.summary["n_trades"]
    assert any(t.name.startswith("long entry ") for t in fig.data)            # the strategy's entry line
    window = trading_dashboard_figure(res, bars, sig, strat, start=100, end=400)
    assert len(window.data[0].x) == 300

def test_trade_analytics_and_excursions(viz_backtest):
    from neural_trade.visualization.trading_dashboard import excursions, trade_analytics_figure

    res, bars, _, _ = viz_backtest
    assert T.empty_panels(trade_analytics_figure(res, bars)) == []
    mfe, mae = excursions(res.trades, bars)
    assert len(mfe) == len(res.trades) and np.all(mfe >= mae)

def test_strategy_comparison(viz_backtest):
    from neural_trade.visualization.trading_dashboard import strategy_comparison_figure

    res, bars, _, _ = viz_backtest
    fig = strategy_comparison_figure({"a": res, "b": res}, bars)
    assert T.empty_panels(fig) == [] and len(fig.data) == 4

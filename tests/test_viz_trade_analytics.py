"""Per-trade analytics and the strategy comparison."""
from __future__ import annotations

import numpy as np

from neural_trade.visualization import theme as T


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

"""Signals, strategies and the backtest engine (plan C5).

    from neural_trade.strategy import SignalFrame, Bars, backtest, build_strategy, var_scale_from
    signals = SignalFrame.build(test_frame, var_scale_from(cal_frame))
    result = backtest(signals, bars, build_strategy("enhanced_multi_horizon"))
"""
from neural_trade.strategy.backtest import (BacktestConfig, BacktestResult, Bars, assert_no_lookahead, backtest,
                                            backtest_frame, random_same_frequency, run_backtest)
from neural_trade.strategy.params import build_backtest_config, build_strategy, from_file, load_params
from neural_trade.strategy.performance import max_drawdown, profit_factor, sharpe, sortino, summarize
from neural_trade.strategy.signals import SignalFrame, var_scale_from
from neural_trade.strategy.strategies import (AlwaysFlat, BuyAndHold, EnhancedMultiHorizonStrategy, LiberalStrategy,
                                              QuantileSignalStrategy, RandomSignal, Strategies, Strategy,
                                              ThresholdSpikeStrategy)
from neural_trade.strategy.trades import Order, Trade

__all__ = [
    "AlwaysFlat", "BacktestConfig", "BacktestResult", "Bars", "BuyAndHold", "EnhancedMultiHorizonStrategy",
    "LiberalStrategy", "Order", "QuantileSignalStrategy", "RandomSignal", "SignalFrame", "Strategies", "Strategy", "ThresholdSpikeStrategy",
    "Trade", "assert_no_lookahead", "backtest", "backtest_frame", "build_backtest_config", "build_strategy",
    "from_file", "load_params", "max_drawdown", "profit_factor", "random_same_frequency", "run_backtest", "sharpe",
    "sortino", "summarize", "var_scale_from",
]

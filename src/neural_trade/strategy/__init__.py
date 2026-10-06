"""Signals, strategies and the backtest engine (plan C5).

    from neural_trade.strategy import SignalFrame, Bars, backtest, build_strategy, var_scale_from
    signals = SignalFrame.build(test_frame, var_scale_from(cal_frame))
    result = backtest(signals, bars, build_strategy("enhanced_multi_horizon"))

Discrete strategies hold one all-or-nothing position; ``ExposureStrategy`` subclasses (the exposure
mode, ``run_exposure_backtest``) hold a target exposure. The variance-driven strategies of NT-077 are in
``variance_strategies`` (docs/research/2026-09-29-strategy-architectures/README.md).
"""
from neural_trade.strategy.backtest import (BacktestConfig, BacktestResult, Bars, assert_no_lookahead, backtest,
                                            backtest_frame, circular_shift_null, random_same_frequency,
                                            run_backtest, run_exposure_backtest)
from neural_trade.strategy.params import build_backtest_config, build_strategy, from_file, load_params
from neural_trade.strategy.performance import max_drawdown, profit_factor, sharpe, sortino, summarize
from neural_trade.strategy.signals import EWMA_HALFLIFE, EWMA_WARMUP, SignalFrame, ewma_sigma, var_scale_from
from neural_trade.strategy.strategies import (AlwaysFlat, BuyAndHold, EnhancedMultiHorizonStrategy, ExposureStrategy,
                                              FittedOnCalibration, LiberalStrategy, QuantileSignalStrategy,
                                              RandomSignal, Strategies, Strategy, ThresholdSpikeStrategy)
from neural_trade.strategy.ta_rules import BollingerBreakoutStrategy, MACrossStrategy, RSIThresholdStrategy
from neural_trade.strategy.trades import Order, Trade
from neural_trade.strategy.variance_strategies import (EdgeOverCostStrategy, GatedTAStrategy, NetEdgeKellyStrategy,
                                                       VolRegimeLongStrategy, VolTargetStrategy)

__all__ = [
    "AlwaysFlat", "BacktestConfig", "BacktestResult", "Bars", "BollingerBreakoutStrategy", "BuyAndHold",
    "EWMA_HALFLIFE", "EWMA_WARMUP",
    "EdgeOverCostStrategy", "EnhancedMultiHorizonStrategy", "ExposureStrategy", "FittedOnCalibration",
    "GatedTAStrategy", "LiberalStrategy", "MACrossStrategy", "NetEdgeKellyStrategy", "Order",
    "QuantileSignalStrategy", "RSIThresholdStrategy", "RandomSignal",
    "SignalFrame", "Strategies", "Strategy", "ThresholdSpikeStrategy", "Trade", "VolRegimeLongStrategy",
    "VolTargetStrategy", "assert_no_lookahead", "backtest", "backtest_frame", "build_backtest_config",
    "build_strategy", "circular_shift_null", "ewma_sigma", "from_file", "load_params", "max_drawdown",
    "profit_factor", "random_same_frequency", "run_backtest", "run_exposure_backtest", "sharpe", "sortino",
    "summarize", "var_scale_from",
]

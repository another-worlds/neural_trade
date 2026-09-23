"""Compatibility shim: moved to ``neural_trade.metrics.numpy_metrics`` (B5; removed in B18)."""
from neural_trade._shim import alias_module

alias_module(__name__, "neural_trade.metrics.numpy_metrics")

"""Compatibility shim: moved to ``neural_trade.utils.math`` (B5; removed in B18)."""
from neural_trade._shim import alias_module

alias_module(__name__, "neural_trade.utils.math")

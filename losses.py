"""Compatibility shim: the loss functions moved to ``neural_trade.losses.functions`` (B3; removed in B18)."""
from neural_trade._shim import alias_module

alias_module(__name__, "neural_trade.losses.functions")

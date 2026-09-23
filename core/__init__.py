"""Compatibility shim: ``core`` moved to ``neural_trade.core`` (Phase B1; removed in B18)."""
from neural_trade._shim import alias_package

alias_package(__name__, "neural_trade.core", ("registry", "exceptions"))

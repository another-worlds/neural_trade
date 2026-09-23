"""Compatibility shim: the monolith's API now lives in the ``neural_trade`` package.

``import model`` returns :mod:`neural_trade.compat` (Phase B12; removed in B18 once the
notebooks import from ``neural_trade``). See that module for where each name moved.
"""
from neural_trade._shim import alias_module

alias_module(__name__, "neural_trade.compat")

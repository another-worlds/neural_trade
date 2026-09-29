"""DataLoaders registry (registry 5 of 9): where raw market data comes from.

The class moved to :mod:`neural_trade.data.loaders_registry` in NT-027 (the layering fix): the
registry and the implementations it wraps now live together in ``data/``, so ``data/processor.py``
no longer needs to import ``neural_trade.registries``. This module re-exports :class:`DataLoaders`
so every existing ``from neural_trade.registries.data_loaders import DataLoaders`` keeps working.
"""
from __future__ import annotations

from neural_trade.data.loaders_registry import DataLoaders

__all__ = ["DataLoaders"]

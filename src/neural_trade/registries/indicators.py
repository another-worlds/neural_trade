"""Indicators registry (registry 10, D-027): learnable indicator families by name.

The class lives in :mod:`neural_trade.indicators.registry`, next to the family contract and
today's four families, so ``models/`` reaches it without importing ``neural_trade.registries``
(the NT-027 layering; same pattern as :mod:`neural_trade.registries.layers`). This module
re-exports :class:`Indicators` next to the nine registries of D-002.
"""
from __future__ import annotations

from neural_trade.indicators.registry import Indicators

__all__ = ["Indicators"]

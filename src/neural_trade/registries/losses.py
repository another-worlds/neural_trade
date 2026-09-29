"""Losses registry (registry 9 of 9): component losses and training objectives.

The class moved to :mod:`neural_trade.losses.registry` in NT-027 (the layering fix):
:mod:`neural_trade.losses.functions` decorates itself with ``Losses.register``, so the registry now
lives in the same package as the module it discovers. This module re-exports :class:`Losses` and
the objective-tier constants so every existing ``from neural_trade.registries.losses import ...``
keeps working.
"""
from __future__ import annotations

from neural_trade.losses.registry import OBJECTIVE_PARAMS, OBJECTIVE_TAG, Losses

__all__ = ["Losses", "OBJECTIVE_TAG", "OBJECTIVE_PARAMS"]

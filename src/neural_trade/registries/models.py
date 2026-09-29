"""Models registry (registry 1 of 9): architecture builders.

The class moved to :mod:`neural_trade.models.registry` in NT-027 (the layering fix):
``models/facade.py`` dispatches through it (``Models.build(...)``), so the registry now lives in
the same package as its client and ``models/`` no longer needs to import ``neural_trade
.registries``. This module re-exports :class:`Models` and :func:`ensure_predictive_outputs` so
every existing ``from neural_trade.registries.models import ...`` keeps working.
"""
from __future__ import annotations

from neural_trade.models.registry import Models, ensure_predictive_outputs

__all__ = ["Models", "ensure_predictive_outputs"]

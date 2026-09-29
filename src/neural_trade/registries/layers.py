"""Layers registry (registry 7 of 9): custom Keras layers by name.

The class moved to :mod:`neural_trade.models.layers_registry` in NT-027 (the layering fix):
``models/gru_attention.py`` dispatches through it on every forward pass (``Layers.for_role(...)``),
so the registry now lives in the same package as its client and ``models/`` no longer needs to
import ``neural_trade.registries``. This module re-exports :class:`Layers` so every existing
``from neural_trade.registries.layers import Layers`` keeps working.
"""
from __future__ import annotations

from neural_trade.models.layers_registry import Layers

__all__ = ["Layers"]

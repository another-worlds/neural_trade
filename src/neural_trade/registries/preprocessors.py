"""Preprocessors registry (registry 8 of 9): DataFrame -> DataFrame steps.

The class moved to :mod:`neural_trade.data.preprocessors_registry` in NT-027 (the layering fix):
the registry and the steps it wraps now live together in ``data/``, so ``data/processor.py`` no
longer needs to import ``neural_trade.registries``. This module re-exports :class:`Preprocessors`
and :func:`run_preprocessors` so every existing
``from neural_trade.registries.preprocessors import ...`` keeps working.
"""
from __future__ import annotations

from neural_trade.data.preprocessors_registry import Preprocessors, run_preprocessors

__all__ = ["Preprocessors", "run_preprocessors"]

"""Metrics registry (registry 3 of 9), two tiers in one registry.

The class moved to :mod:`neural_trade.metrics.registry` in NT-027 (the layering fix):
``metrics/evaluate.py`` dispatches through it (``Metrics.numpy_functions(...)``), so the registry
now lives in the same package as its client and ``metrics/`` no longer needs to import
``neural_trade.registries``. This module re-exports :class:`Metrics` and the metric-name tuples so
every existing ``from neural_trade.registries.metrics import ...`` keeps working.
"""
from __future__ import annotations

from neural_trade.metrics.registry import (DIRECTION_METRICS, DISTRIBUTION_METRICS,
                                           REGRESSION_METRICS, STEP_METRICS, TF_TAG, Metrics)

__all__ = ["Metrics", "TF_TAG", "REGRESSION_METRICS", "DIRECTION_METRICS", "DISTRIBUTION_METRICS",
          "STEP_METRICS"]

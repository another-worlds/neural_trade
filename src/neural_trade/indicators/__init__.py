"""Learnable indicator families and their registry (D-027, D-031, NT-046).

Importing this package registers today's four families (``families``); the model layer
(``models/layers/learnable_indicators.py``) assembles its channels from the entries the
config lists. ``instances`` is import-light (no TensorFlow) for callers that only read the
configuration.
"""
from __future__ import annotations

from .base import (
    META_SCALE,
    ChannelSpec,
    FamilyContext,
    IndicatorFamily,
    ParamSpec,
    compute_reference,
    m_single_ewma,
)
from .instances import indicator_instances, num_learnable_logits
from .registry import Indicators
from . import families  # noqa: F401  (registers the four families)

__all__ = ["ChannelSpec", "FamilyContext", "IndicatorFamily", "Indicators", "META_SCALE",
           "ParamSpec", "compute_reference", "indicator_instances", "m_single_ewma",
           "num_learnable_logits"]

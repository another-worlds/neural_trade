"""Learnable indicator families and their registry (D-027, D-031, NT-046).

Importing this package registers today's four families (``families``); the model layer
(``models/layers/learnable_indicators.py``) assembles its channels from the entries the
config lists. ``instances`` is import-light (no TensorFlow) for callers that only read the
configuration.
"""
from __future__ import annotations

from .base import (
    BASE_SERIES,
    DERIVED_SERIES,
    META_SCALE,
    ChannelSpec,
    FamilyContext,
    IndicatorFamily,
    ParamSpec,
    compute_reference,
    m_single_ewma,
    m_soft_extremum,
)
from .instances import indicator_instances, num_learnable_logits
from .registry import Indicators
from . import families  # noqa: F401  (registers the four NT-046 families)
from . import families_ohlcv  # noqa: F401  (registers the ten OHLCV families, NT-047)

Indicators._initialized = True

__all__ = ["BASE_SERIES", "ChannelSpec", "DERIVED_SERIES", "FamilyContext",
           "IndicatorFamily", "Indicators", "META_SCALE", "ParamSpec", "compute_reference",
           "indicator_instances", "m_single_ewma", "m_soft_extremum",
           "num_learnable_logits"]

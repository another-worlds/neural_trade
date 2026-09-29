"""Model heads -> the predictions dict (shared by training evaluation and serving).

Moved to :mod:`neural_trade.core.postprocess` in NT-027 (the layering fix): both
``training/trainer.py`` and ``serving/predictor.py`` used to reach across into each other's
package for this (``training -> serving`` here, ``serving -> training`` for ``ArtifactBundle``),
which the layering test forbids. This module re-exports everything so
``from neural_trade.serving.postprocess import ...`` keeps working.
"""
from __future__ import annotations

from neural_trade.core.postprocess import (HORIZONS, heads_to_predictions, inverse_scale,
                                           sanitize_prob, sanitize_var)

__all__ = ["HORIZONS", "inverse_scale", "sanitize_prob", "sanitize_var", "heads_to_predictions"]

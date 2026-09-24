"""Model analytics: what each of the 9 heads does on a held-out block, per horizon.

Every figure has one column per horizon (h0 / h1 / h2) and takes a PredictionFrame (the served
predictions: calibrated P(up), shrunk delta, conformal intervals) plus the config (deadband):

* :func:`direction_analytics_figure` - P(up) distribution by realised class, ROC curves (direction
  head and the price head's Gaussian readout), reliability of raw vs calibrated P(up).
* :func:`delta_analytics_figure` - realised vs predicted move (raw price head), the binned
  calibration curve of the price head, and the rolling correlation along the block.
* :func:`variance_analytics_figure` - predicted sigma vs realised error (binned), PIT histogram,
  rolling interval coverage.
* :func:`confidence_analytics_figure` - accuracy by confidence decile, selective accuracy (trade
  only the most confident x%), confusion matrices.
* :func:`coherence_analytics_figure` - how the horizons agree: P(up) correlation, vote counts,
  accuracy when the horizons agree vs disagree.

Direction metrics use the deadband mask (moves inside ``DIR_DEADBAND_BPS`` have no label).
"""
from __future__ import annotations

from neural_trade.visualization.analytics_common import roc_curve  # noqa: F401
from neural_trade.visualization.analytics_confidence import (  # noqa: F401
    coherence_analytics_figure, confidence_analytics_figure,
)
from neural_trade.visualization.analytics_delta import delta_analytics_figure  # noqa: F401
from neural_trade.visualization.analytics_direction import direction_analytics_figure  # noqa: F401
from neural_trade.visualization.analytics_variance import variance_analytics_figure  # noqa: F401


# ------------------------------------------------------------------ registry entries (data, config)
def direction_analytics(data, config=None, **kw):
    return direction_analytics_figure(data, config, **kw)


def delta_analytics(data, config=None, **kw):
    return delta_analytics_figure(data, config, **kw)


def variance_analytics(data, config=None, **kw):
    return variance_analytics_figure(data, config, **kw)


def confidence_analytics(data, config=None, **kw):
    return confidence_analytics_figure(data, config, **kw)


def coherence_analytics(data, config=None, **kw):
    return coherence_analytics_figure(data, config, **kw)

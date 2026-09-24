"""Model analytics: what each of the 9 heads does on a held-out block, per horizon.

Every figure has one column per horizon (h0 / h1 / h2) and takes a PredictionFrame (the served
predictions: calibrated P(up), shrunk delta, conformal intervals) plus the config (deadband):

* :func:`direction_analytics_figure` - P(up) distribution by realised class; ROC drawn as lift
  over chance (TPR - FPR, area = AUC - 0.5) for the direction head and the price head's Gaussian
  readout inside a no-skill 95% band; reliability of raw vs calibrated P(up) (10 equal-count bins,
  block-clustered CIs); a scorecard against a constant 0.5 with the evaluation report's numbers.
* :func:`delta_analytics_figure` - a per-horizon error table (raw vs served vs predicting 0, with
  HAC skill intervals); every-sample scatter with shaded clip margins and the served beta line;
  binned calibration with overlap-adjusted CIs; rolling correlation and rolling served skill with
  no-skill bands.
* :func:`variance_analytics_figure` - RMS error per predicted-sigma bin (block-bootstrap CIs)
  against a trailing realised-vol baseline; PIT; tail rates relative to a Gaussian; rolling 90%
  coverage with a noise band; rolling interval width.
* :func:`confidence_analytics_figure` - accuracy by decile of |P(up) - 0.5| and of the strategies'
  confidence exp(-var / var_scale); selective accuracy; confusion matrices with recall, precision,
  balanced accuracy and MCC.
* :func:`coherence_analytics_figure` - P(up) correlation, vote patterns, realised up-rate by the
  number of up votes, direction-head vs price-head sign agreement, |delta| ordering (raw vs served)
  and the strategies' vote agreement at SignalFrame's vote lines.

Direction metrics use the deadband mask (moves inside ``DIR_DEADBAND_BPS`` have no label). Every
interval accounts for overlapping targets (consecutive samples share h - 1 of their h bars).
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

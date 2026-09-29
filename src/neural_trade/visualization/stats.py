"""Uncertainty for the figures: how much of what a chart shows is noise.

Moved to :mod:`neural_trade.metrics.statistics` in NT-027 (the shared metrics and statistics
module): evaluation/report.py needs the same effective-sample helpers as the figures, so they now
live below both. This module re-exports everything so ``import ... as S; S.n_eff(...)`` (every
figure module's usual pattern) keeps working unchanged.

* :func:`n_eff` - effective sample size.
* :func:`wilson` - binomial proportion interval on the effective count.
* :func:`mean_ci` - normal interval of a mean on the effective count.
* :func:`corr_null_r` - the +/- band a correlation r stays inside by chance (no relationship), in r units.
* :func:`corr_null` - the same band on the Fisher-z scale (atanh r), for intervals built on that scale.
* :func:`auc_ci` - Hanley-McNeil interval for a ROC AUC on effective class counts.
* :func:`thin` - indices that keep at most ``max_points`` of a long, ordered series (for plotting).
"""
from __future__ import annotations

from neural_trade.metrics.statistics import (Z95, auc_ci, corr_null, corr_null_r, horizon_steps,
                                             mean_ci, n_eff, thin, wilson)

__all__ = ["Z95", "n_eff", "wilson", "mean_ci", "corr_null", "corr_null_r", "auc_ci", "thin",
          "horizon_steps"]

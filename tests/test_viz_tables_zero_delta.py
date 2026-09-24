"""classification_table(readout="gaussian") where the served delta is 0 (beta = 0): the readout is the constant 0.5."""
from __future__ import annotations

import copy
import math

import numpy as np

from neural_trade.evaluation.frame import HORIZONS
from neural_trade.evaluation.report import GAUSS_CONST_NA
from neural_trade.visualization import analytics_tables as AT

RATES = ("calls up (predicted up-rate)", "accuracy", "accuracy 95% CI (n_eff)", "balanced accuracy", "MCC", "AUC",
         "AUC 95% CI (n_eff)", "Brier", "ECE (positive class)", "TP (called up, went up)")


def _scaled(frame, betas):
    fr = copy.deepcopy(frame)
    fr.delta = {h: betas[h] * np.asarray(frame.delta[h], float) for h in HORIZONS}
    return fr


def test_gaussian_readout_of_a_served_delta_of_zero_is_na_not_a_measured_chance_level(viz_frame, viz_config):
    """The constant 0.5 never calls up and has AUC 0.5 and Brier 0.25 in every sample: the table printed them
    (with an AUC interval) as if measured (final review, round 2)."""
    df = AT.classification_table(_scaled(viz_frame, {h: 0.0 for h in HORIZONS}), viz_config, readout="gaussian",
                                 digits=None)
    for h in HORIZONS:
        for row in RATES:
            assert df.loc[row, h] == GAUSS_CONST_NA, (row, h)
        assert df.loc["mean P(up), all samples", h] == 0.5 and df.loc["std P(up), all samples", h] == 0.0
        assert 0 < df.loc["true up-rate", h] < 1 and math.isfinite(df.loc["ECE of a constant 0.5", h])
    assert "beta = 0 for h0, h1, h2" in df.attrs["caption"] and "constant 0.5" in df.attrs["caption"]
    # the direction head itself is measured whatever the price head serves
    head = AT.classification_table(_scaled(viz_frame, {h: 0.0 for h in HORIZONS}), viz_config, digits=None)
    assert GAUSS_CONST_NA not in head.to_numpy() and "caption" not in head.attrs


def test_gaussian_readout_is_measured_where_the_served_delta_varies(viz_frame, viz_config):
    df = AT.classification_table(_scaled(viz_frame, {"h0": 0.21, "h1": 0.0, "h2": 0.25}), viz_config,
                                 readout="gaussian", digits=None)
    assert df.loc["AUC", "h1"] == GAUSS_CONST_NA and "beta = 0 for h1:" in df.attrs["caption"]
    for h in ("h0", "h2"):
        assert isinstance(df.loc["AUC", h], float) and 0 < df.loc["AUC", h] < 1
    full = AT.classification_table(viz_frame, viz_config, readout="gaussian", digits=None)
    assert GAUSS_CONST_NA not in full.to_numpy() and "caption" not in full.attrs

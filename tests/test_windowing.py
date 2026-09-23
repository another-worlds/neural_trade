"""Inference windows reach the newest bar and agree with the training windows."""
from __future__ import annotations

import numpy as np

from neural_trade.core.config import Config
from neural_trade.data.windowing import make_inference_windows, make_sequences_with_extended_trends


def test_inference_windows_end_at_the_last_bar_and_match_training_windows(synthetic_close):
    cfg = Config()
    X, y, lc, ext = make_sequences_with_extended_trends(cfg, synthetic_close, cfg.LOOKBACK)
    Xi, lci, exti = make_inference_windows(synthetic_close, cfg.LOOKBACK,
                                           extended_trend_periods=cfg.EXTENDED_TREND_PERIODS)
    np.testing.assert_array_equal(Xi[-1], synthetic_close[-cfg.LOOKBACK:])
    assert lci[-1] == synthetic_close[-1]
    # the training set stops max(H) - 1 bars early (it needs future targets); overlap is identical
    n = len(X)
    np.testing.assert_array_equal(Xi[:n], X)
    np.testing.assert_array_equal(lci[:n], lc)
    np.testing.assert_array_equal(exti[:n], ext)
    assert len(Xi) - n == max(cfg.HORIZON_STEPS) - 1 + 1

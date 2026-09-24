"""Direction-head analytics."""
from __future__ import annotations

import pytest


H = ("h0", "h1", "h2")


def test_roc_curve_matches_sklearn(viz_frame):
    from sklearn.metrics import roc_auc_score

    from neural_trade.visualization.model_analytics import roc_curve

    lab = (viz_frame.y[:, 1] > 0).astype(float)
    assert roc_curve(lab, viz_frame.direction_prob["h1"])[2] == pytest.approx(roc_auc_score(lab, viz_frame.direction_prob["h1"]),
                                                                          abs=1e-9)

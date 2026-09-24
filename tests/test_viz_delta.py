"""Price-head (delta) analytics."""
from __future__ import annotations

import numpy as np


H = ("h0", "h1", "h2")


def test_delta_analytics_uses_the_raw_heads_when_given(viz_frame, viz_config):
    from neural_trade.visualization.model_analytics import delta_analytics_figure

    raw = {h: 2 * viz_frame.delta[h] for h in H}
    fig = delta_analytics_figure(viz_frame, viz_config, raw_delta=raw)
    assert np.allclose(fig.data[0].x, raw["h0"])

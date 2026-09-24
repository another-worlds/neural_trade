"""Every model-analytics figure draws every panel it has data for."""
from __future__ import annotations

import pytest

from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")


@pytest.mark.parametrize("name", ["direction_analytics", "delta_analytics", "variance_analytics",
                                  "confidence_analytics", "coherence_analytics"])
def test_model_analytics_figures_have_no_empty_panel(viz_frame, viz_config, name):
    from neural_trade.registries.visualizations import Visualizations

    fig = Visualizations.build(name, viz_frame, viz_config)
    assert T.empty_panels(fig) == [], name

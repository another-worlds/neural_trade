"""Confidence and cross-horizon coherence analytics."""
from __future__ import annotations



H = ("h0", "h1", "h2")


def test_categorical_axes_keep_every_category(viz_frame, viz_config):
    """Plotly turns '1', '2' into a numeric axis unless told otherwise, silently dropping text categories."""
    from neural_trade.visualization.model_analytics import coherence_analytics_figure, confidence_analytics_figure

    coh = coherence_analytics_figure(viz_frame, viz_config)
    assert coh.layout.xaxis2.type == "category" and len(coh.data[1].x) == 4
    conf = confidence_analytics_figure(viz_frame, viz_config)
    assert conf.layout.xaxis.type == "category" and list(conf.data[0].x) == [str(k) for k in range(1, 11)]

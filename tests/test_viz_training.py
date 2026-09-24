"""Training dashboard, health tiles and the live batch strip."""
from __future__ import annotations


from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")


def test_training_dashboard_draws_every_logged_metric(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    fig = training_dashboard_figure(viz_history(), viz_config)
    assert T.empty_panels(fig) == []
    names = {t.name for t in fig.data}
    assert {"h0 (10 bars)", "h1 (15 bars)", "h2 (20 bars)", "best", "network"} <= names

def test_training_dashboard_marks_metrics_that_were_not_logged(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_dashboard_figure

    fig = training_dashboard_figure(viz_history(full=False), viz_config)
    notes = [a.text for a in fig.layout.annotations if a.text == "not logged in this run"]
    assert len(notes) == len(T.empty_panels(fig)) > 0

def test_default_training_visualization_is_the_dashboard(viz_config, viz_history):
    """plotly_interactive (Config.VISUALIZATION) used to draw an empty direction panel."""
    from neural_trade.registries.visualizations import Visualizations

    fig = Visualizations.build("plotly_interactive", viz_history(), viz_config)
    assert T.empty_panels(fig) == []
    assert any(t.name == "h1 (15 bars)" for t in fig.data)

def test_training_health_flags_collapse_patience_and_nonfinite_gradients(viz_config, viz_history):
    from neural_trade.visualization.training_dashboard import training_health, training_health_html

    rows = viz_history()
    rows[-1]["val_pred_up_rate_h1"] = 0.995                      # predicts one class
    rows[-1]["nonfinite_grad_steps"] = 3.0
    for r in rows[2:]:
        r["val_loss"] = 10.0                                     # best epoch stays at 1-2
    checks = training_health(rows, viz_config)
    assert checks["direction h1"]["status"] == "critical"
    assert checks["gradients"]["status"] == "critical"
    assert checks["patience"]["status"] == "critical" and checks["patience"]["value"].endswith("/ 3")
    assert "learning rate" in checks and "1 reductions" in checks["learning rate"]["value"]
    assert "direction h1" in training_health_html(rows, viz_config)

def test_batch_loss_figure():
    from neural_trade.visualization.training_dashboard import batch_loss_figure

    fig = batch_loss_figure([(0.1, 7.0), (0.5, 6.8)], [(1, 6.0)])
    assert [len(t.x) for t in fig.data] == [2, 1]

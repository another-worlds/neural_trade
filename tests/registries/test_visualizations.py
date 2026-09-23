"""Visualizations registry and the model.py compatibility shim."""
from __future__ import annotations

import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.registries.visualizations import Visualizations


def _history(n=4):
    return {"loss": [3.0 - 0.1 * i for i in range(n)], "val_loss": [3.2 - 0.1 * i for i in range(n)],
            "dir_acc_h1": [0.5 + 0.01 * i for i in range(n)], "val_dir_acc_h1": [0.49 + 0.01 * i for i in range(n)]}


def test_registered_components():
    assert {"plotly_interactive", "qbox_dashboard_html", "matplotlib_splits",
            "indicator_evolution"} <= set(Visualizations.list_names())


def test_training_curves_from_a_history_dict_and_from_jsonl_rows():
    pytest.importorskip("plotly")
    fig = Visualizations.build("plotly_interactive", _history(), Config())
    names = {t.name for t in fig.data}
    assert {"loss", "val_loss"} <= names
    rows = [{k: v[i] for k, v in _history().items()} for i in range(4)]
    assert len(Visualizations.build(None, rows, Config()).data) >= 2


def test_indicator_evolution_and_qbox_html():
    pytest.importorskip("plotly")
    df = pd.DataFrame({"epoch": [0, 1], "ma_period_0": [5.0, 5.2], "change_ma_period_0": [None, 4.0]})
    fig = Visualizations.build("indicator_evolution", df, Config())
    assert [t.name for t in fig.data] == ["ma_period_0"]
    html = Visualizations.build("qbox_dashboard_html", {"hd_loss": 0.5, "val_hd_loss": 0.4}, Config())
    assert "Hyper-Dec" in html
    assert Visualizations.build("qbox_dashboard_html", {"hd_loss": 0.0}, Config()) == ""


def test_model_shim_is_the_compat_module():
    import model
    import neural_trade.compat as compat
    from neural_trade.training.trainer import train_and_evaluate

    assert model is compat and model.train_and_evaluate is train_and_evaluate
    assert {"Config", "CustomTrainModel", "DataProcessor", "train_and_evaluate"} <= set(compat.__all__)

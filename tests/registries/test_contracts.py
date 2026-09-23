"""Every registry on the training path is actually queried during training; load_all and plugins."""
from __future__ import annotations

from pathlib import Path

import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.core.registry import BaseRegistry

TRAINING_REGISTRIES = {"Models", "Optimizers", "Metrics", "Callbacks", "DataLoaders", "Layers",
                       "Preprocessors", "Losses"}
ALL_NINE = TRAINING_REGISTRIES | {"Visualizations"}
REPO = Path(__file__).resolve().parents[2]


def test_training_and_reporting_query_all_nine_registries(tf, tiny_config, tmp_path, synthetic_bars, monkeypatch):
    from neural_trade.training.trainer import train_and_evaluate

    hits = set()
    original = BaseRegistry.__dict__["_checked_component"].__func__

    def spy(cls, name):
        hits.add(cls.__name__)
        return original(cls, name)

    monkeypatch.setattr(BaseRegistry, "_checked_component", classmethod(spy))
    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH, cfg.MODEL_PATH, cfg.SCALER_PATH = (str(tmp_path / "bars.csv"), str(tmp_path / "w.h5"),
                                                     str(tmp_path / "s.joblib"))
    result = train_and_evaluate(config=cfg, epochs=1, force=True, calibrate=False, fit_calibration=False)
    assert TRAINING_REGISTRIES <= hits, f"registries never queried: {TRAINING_REGISTRIES - hits}"

    # the reporting path reaches the ninth registry (Visualizations renders the evaluation figure)
    from neural_trade.evaluation.walk_forward import evaluate_result

    report = evaluate_result(result, out_dir=tmp_path / "report", figure=True)
    assert ALL_NINE <= hits, f"registries never queried: {ALL_NINE - hits}"
    assert (tmp_path / "report" / "eval_report_test.md").exists() and report.n == len(result.y_test)


def test_load_all_resolves_the_default_config_and_rejects_unknown_components():
    from neural_trade.registries import load_all, registry_summary, validate_config_components

    regs = load_all(Config())
    assert len(regs) == 9 and not validate_config_components(Config())
    assert "total" in registry_summary()
    with pytest.raises(InvalidConfigurationError, match="Models:nope"):
        load_all(Config(MODEL_NAME="nope"))


@pytest.fixture
def clean_plugins():
    """Plugins register into the process-wide registries; undo that after the test."""
    import sys

    from neural_trade.registries.metrics import Metrics

    yield
    Metrics.remove("n_samples")
    for name in [m for m in sys.modules if m.startswith("neural_trade_plugins")]:
        del sys.modules[name]


def test_plugins_register_components_and_templates_are_skipped(clean_plugins):
    from neural_trade.core.plugin_loader import load_plugins
    from neural_trade.registries.metrics import Metrics

    loaded = load_plugins(REPO / "plugins", strict=True)
    assert any(name.endswith("examples.echo_metric") for name in loaded)
    assert not any("templates" in name for name in loaded)
    assert Metrics.build("n_samples", [1, 2, 3], [0, 0, 0]) == 3.0
    assert load_plugins(REPO / "plugins") == loaded  # idempotent

"""NT-048 indicator report: offline HTML, the CLI command, the notebook hook, and the engine flag.

The fast tests stub the writer. The slow test trains one epoch on CPU and checks the file that
``write_indicator_report`` writes from that run's artifacts and metrics.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_SCRIPT_SRC = re.compile(r"<script\b[^>]*\bsrc=[\"']https?://", re.IGNORECASE)


def _spec(**extra):
    data = {"schema_version": 1, "name": "rep", "folds": [-1], "seeds": [0]}
    data.update(extra)
    return data


def test_the_flag_defaults_off_and_rejects_a_non_bool(tmp_path):
    from neural_trade.experiments.scenario import Scenario, ScenarioError

    assert Scenario.from_dict(_spec()).run.indicator_report is False
    on = Scenario.from_dict(_spec(run={"indicator_report": True}))
    assert on.run.indicator_report is True
    assert "indicator_report" not in on.settings()
    assert on.settings_hash == Scenario.from_dict(_spec()).settings_hash
    assert on.spec_hash != Scenario.from_dict(_spec()).spec_hash
    with pytest.raises(ScenarioError, match="true or false"):
        Scenario.from_dict(_spec(run={"indicator_report": 1}))

    path = tmp_path / "s.yaml"
    path.write_text("schema_version: 1\nname: rep\nfolds: [-1]\nseeds: [0]\nrun:\n  indicator_report: true\n",
                    encoding="utf-8")
    assert Scenario.from_yaml(path).run.indicator_report is True


def test_indicator_report_requires_save_artifacts():
    from neural_trade.experiments.scenario import Scenario, ScenarioError

    sc = Scenario.from_dict(_spec(run={"indicator_report": True, "save_artifacts": False}))
    with pytest.raises(ScenarioError, match="save_artifacts"):
        sc.validate()


def test_the_html_embeds_plotly_and_refuses_an_empty_panel(tmp_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from neural_trade.serving.indicator_report import write_indicator_report

    filled = go.Figure(go.Scatter(y=[1.0, 2.0]))
    path = write_indicator_report(tmp_path, figures=(filled, filled, filled))
    html = path.read_text(encoding="utf-8")
    assert _SCRIPT_SRC.search(html) is None
    assert html.count("plotly-graph-div") >= 3
    assert path.name == "indicator_report.html"

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    empty = make_subplots(rows=1, cols=1)
    with pytest.raises(RuntimeError, match="empty panel"):
        write_indicator_report(fresh, figures=(filled, filled, empty))
    assert not (fresh / "indicator_report.html").exists()
    with pytest.raises(ValueError, match="three figures"):
        write_indicator_report(fresh, figures=(filled,))


def test_missing_bundle_or_log_is_refused_before_a_figure_is_built(tmp_path):
    from neural_trade.serving.indicator_report import indicator_figures

    with pytest.raises(FileNotFoundError, match="artifacts"):
        indicator_figures(tmp_path)
    (tmp_path / "artifacts").mkdir()
    with pytest.raises(FileNotFoundError, match="metrics.jsonl"):
        indicator_figures(tmp_path)


def test_training_does_not_import_the_indicator_report():
    root = Path(__file__).resolve().parents[1] / "src" / "neural_trade" / "training"
    offenders = [str(path.relative_to(root)) for path in sorted(root.rglob("*.py"))
                 if "indicator_report" in path.read_text(encoding="utf-8")]
    assert offenders == []


def test_the_engine_writes_the_report_only_when_the_scenario_asks(tmp_path, synthetic_bars, monkeypatch):
    from neural_trade.experiments.runner import Runner
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.store import RunStore
    from tests.test_experiment_engine import FakeTrainer, spec

    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    calls = []

    def fake_write(run_dir):
        calls.append(Path(run_dir))
        return Path(run_dir) / "indicator_report.html"

    monkeypatch.setattr("neural_trade.serving.indicator_report.write_indicator_report", fake_write)

    off = Runner(Scenario.from_dict(spec(csv, folds=[-1], seeds=[0])), RunStore(tmp_path / "off"),
                 trainer=FakeTrainer()).run()
    assert off.ran[0]["status"] == "done" and calls == []
    off_meta = json.loads((Path(off.ran[0]["run_dir"]) / "meta.json").read_text(encoding="utf-8"))
    assert off_meta["engine"]["run"]["indicator_report"] is False

    on_spec = spec(csv, folds=[-1], seeds=[0],
                   run={"calibrate": False, "save_artifacts": True, "indicator_report": True})
    on = Runner(Scenario.from_dict(on_spec), RunStore(tmp_path / "on"), trainer=FakeTrainer()).run()
    assert on.ran[0]["status"] == "done" and not on.failed
    assert calls == [Path(on.ran[0]["run_dir"])]
    on_meta = json.loads((calls[0] / "meta.json").read_text(encoding="utf-8"))
    assert on_meta["engine"]["run"]["indicator_report"] is True

    def boom(run_dir):
        raise RuntimeError("report broke")

    monkeypatch.setattr("neural_trade.serving.indicator_report.write_indicator_report", boom)
    bad = Runner(Scenario.from_dict(on_spec), RunStore(tmp_path / "bad"), trainer=FakeTrainer()).run()
    assert bad.failed == [bad.ran[0]["cell"]]
    assert (Path(bad.ran[0]["run_dir"]) / "result.json").is_file()


def test_a_notebook_session_writes_the_report_when_it_has_a_run_directory(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    from neural_trade.core.config import Config
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.notebook.session import TrainingSession

    calls = []

    def fake_train(*_args, **_kwargs):
        return "stub-result"

    def fake_write(run_dir):
        calls.append(Path(run_dir))
        return Path(run_dir) / "indicator_report.html"

    monkeypatch.setattr("neural_trade.training.trainer.train_and_evaluate", fake_train)
    monkeypatch.setattr("neural_trade.serving.indicator_report.write_indicator_report", fake_write)

    cfg = Config(EPOCHS=1)
    ctx = RunContext.create(cfg, root=tmp_path, seed=0, write_env=False)
    session = TrainingSession(cfg, run_context=ctx, epochs=1)
    session._callback()  # the Keras hook still imports TensorFlow; the stub trainer is not what loads it
    session._run()
    assert session.error is None and session.result == "stub-result"
    assert session.status == "finished"
    assert calls == [ctx.run_dir]

    calls.clear()
    bare = TrainingSession(cfg, epochs=1)
    bare._run()
    assert bare.error is None and bare.result == "stub-result" and calls == []

    def boom(run_dir):
        raise RuntimeError("report broke")

    monkeypatch.setattr("neural_trade.serving.indicator_report.write_indicator_report", boom)
    broken = TrainingSession(cfg, run_context=ctx, epochs=1)
    broken._run()
    assert isinstance(broken.error, RuntimeError) and "report broke" in str(broken.error)
    assert broken.status.startswith("failed")


def test_indicators_command_prints_the_report_path(tmp_path, monkeypatch, capsys):
    from neural_trade.cli import main

    run = tmp_path / "run"
    run.mkdir()

    def fake_write(run_dir):
        path = Path(run_dir) / "indicator_report.html"
        path.write_text("ok", encoding="utf-8")
        return path

    monkeypatch.setattr("neural_trade.serving.indicator_report.write_indicator_report", fake_write)
    assert main(["indicators", str(run)]) == 0
    assert capsys.readouterr().out.strip() == str(run / "indicator_report.html")
    assert (run / "indicator_report.html").read_text(encoding="utf-8") == "ok"


@pytest.mark.slow
def test_a_trained_run_writes_an_offline_report_with_three_figures(tmp_path, synthetic_bars):
    tf = pytest.importorskip("tensorflow")
    tf.keras.utils.set_random_seed(0)

    from neural_trade.core.config import Config
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.serving.indicator_report import indicator_figures, write_indicator_report
    from neural_trade.training.trainer import train_and_evaluate
    from neural_trade.visualization import theme as T

    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    cfg = Config(EPOCHS=1, BATCH_SIZE=16, MAX_SEQUENCE_COUNT=1200, PATIENCE=1, EARLY=1,
                 CSV_PATH=str(csv.resolve()))
    ctx = RunContext.create(cfg, root=tmp_path / "runs", seed=0, write_env=False)
    train_and_evaluate(config=ctx.config, run_context=ctx, epochs=1, force=True, calibrate=False,
                       save_artifacts=True)
    assert (ctx.run_dir / "artifacts").is_dir() and (ctx.run_dir / "metrics.jsonl").is_file()

    figs = indicator_figures(ctx.run_dir)
    assert len(figs) == 3
    for fig in figs:
        assert T.empty_panels(fig) == []
    path = write_indicator_report(ctx.run_dir, figures=figs)
    html = path.read_text(encoding="utf-8")
    assert path == ctx.run_dir / "indicator_report.html"
    assert _SCRIPT_SRC.search(html) is None
    assert html.count("plotly-graph-div") >= 3


def _default_family_figures():
    import numpy as np
    import pandas as pd

    from neural_trade.core.config import Config
    from neural_trade.core.indicator_periods import configured_periods
    from neural_trade.visualization import discovered_indicators as DI
    from neural_trade.visualization import indicator_evolution as IE

    cfg = Config()
    length = int(cfg.LOOKBACK)
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 0.4, length))
    window = np.stack([close + rng.normal(0, 0.05, length), close + rng.uniform(0.05, 0.8, length),
                       close - rng.uniform(0.05, 0.8, length), close, rng.uniform(0.2, 2.0, length)],
                      axis=-1).astype(np.float32)
    periods = configured_periods(cfg)
    app = pd.DataFrame([periods])
    app.attrs["base"] = dict(periods)
    app.attrs["block"] = "test"
    discovered = DI.discovered_indicators(close[None, :], cfg, applied=app, ohlcv=window[None], window=0)
    rows = [{"epoch": e + 1, **{f"period/{k}": v + 0.1 * e for k, v in periods.items()}} for e in range(4)]
    return cfg, periods, discovered, IE.indicator_family_periods(rows, cfg, applied=app)


def test_the_report_figures_hold_every_default_family_and_instance():
    """NT-048 (3): the default 14 families x 3 instances are in the price figure, the 54 learned periods are in
    the periods figure, and the importance figure has one row per instance."""
    import re

    from neural_trade.evaluation.permutation_importance import GroupImportance, indicator_channel_groups
    from neural_trade.indicators import indicator_instances
    from neural_trade.visualization import discovered_indicators as DI
    from neural_trade.visualization import indicator_evolution as IE
    from neural_trade.visualization import theme as T
    from neural_trade.visualization.permutation_importance import permutation_importance

    cfg, periods, discovered, family = _default_family_figures()
    fams = list(indicator_instances(cfg))
    assert len(fams) == 14 and len(periods) == 54

    def headings(fig):
        return {re.sub(r"<[^>]+>", "", fig.layout[k].title.text or "").strip()
                for k in fig.layout if str(k).startswith("legend")}

    heads = headings(discovered)
    for fam in fams:
        for i in range(3):
            assert f"{DI._display_name(fam)} #{i}" in heads, (fam, i)
    assert T.empty_panels(discovered) == [] and T.empty_panels(family) == []
    solid = [t for t in family.data if t.mode == "lines+markers"]
    assert len(solid) == len(periods) == 54                      # every learned period has its trace
    for fam in fams:
        assert any(h.startswith(IE._family_title(fam).replace(" periods", "")) for h in headings(family)), fam
    rows = [GroupImportance(name, 0.1, 0.0, 0.2, {"h0": 0.01}, {"h0": -0.01}, {"h0": 0.02},
                            {"h0": 0.0}, {"h0": -0.01}, {"h0": 0.01})
            for name, _sl in indicator_channel_groups(cfg)]
    fig = permutation_importance(rows, cfg)
    assert len(rows) == 42 and len(fig.data[0].y) == 42 and T.empty_panels(fig) == []

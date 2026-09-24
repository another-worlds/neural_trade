"""Notebook front-ends (neural_trade.notebook) and their figures, driven headlessly."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from neural_trade.visualization.calibration_plots import reliability_figure, reliability_table


def test_reliability_table_is_flat_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    t = reliability_table(y, p, 10)
    assert t.shape == (10, 5) and np.all(np.abs(t[:, 0] - t[:, 1]) < 0.03) and t[:, 2].sum() == p.size
    assert len(reliability_figure(y, p, np.clip(p * 1.1, 0, 1)).data) == 3


def test_comparison_and_ablation_figures():
    from neural_trade.visualization.comparison import ablation_deltas_figure, runs_comparison_figure

    df = pd.DataFrame({"h1/direction/auc": [0.51, 0.53], "backtest/sharpe_net": [-3.0, -1.0], "seed": [0, 1]},
                      index=["run-a", "run-b"])
    assert len(runs_comparison_figure(df).data) == 2
    analysis = {"terms": {"LAMBDA_HD": {"verdict": "NEUTRAL", "modes": {"leave_one_in": {"verdict": "NEUTRAL", "metrics": [
        {"metric": "h1/variance/crpss", "mean_delta": 0.001, "sd_delta": 0.002, "verdict": "NEUTRAL"}]}}}},
                "family": {"verdict": "VALUE", "metrics": [{"metric": "h1/direction/mcc", "mean_delta": 0.02,
                                                            "sd_delta": 0.01, "verdict": "VALUE"}]}}
    fig = ablation_deltas_figure(analysis)
    assert {t.name for t in fig.data} == {"NEUTRAL", "VALUE"}


def test_split_table_and_overview(synthetic_bars, tmp_path):
    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.visualization.data_overview import split_overview_figure, split_table

    synthetic_bars.to_csv(tmp_path / "b.csv", index=False)
    cfg = Config(CSV_PATH=str(tmp_path / "b.csv"), MAX_SEQUENCE_COUNT=2000)
    blocks = split_arrays(cfg)
    t = split_table(blocks, cfg)
    assert list(t.index) == ["train", "val", "cal", "test"] and (t["from"] <= t["to"]).all()
    assert t.loc["val", "from"] > t.loc["train", "to"] and t.loc["test", "from"] > t.loc["cal", "to"]  # purged, ordered
    assert 0 < t.loc["test", "up-rate h1"] < 1
    assert len(split_overview_figure(blocks, cfg).layout.shapes) == 4


def test_session_pause_blocks_the_training_thread_until_resume():
    from neural_trade.core.config import Config
    from neural_trade.notebook import TrainingSession

    s = TrainingSession(Config(EPOCHS=1))
    s.status = "training"
    cb = s._callback()

    class _M:
        stop_training = False

    cb.model = _M()
    s.pause()
    assert s.status == "paused"
    t = threading.Thread(target=cb.on_train_batch_end, args=(1,))
    t.start()
    time.sleep(0.6)
    assert t.is_alive(), "a paused session must hold the training thread"
    s.resume()
    t.join(2)
    assert not t.is_alive() and not cb.model.stop_training
    s.stop()
    cb.on_train_batch_end(2)
    assert cb.model.stop_training


# ---------------------------------------------------------------------------- one real run, three front-ends
@pytest.fixture(scope="module")
def session_run(tmp_path_factory, synthetic_bars):
    from neural_trade.core.config import Config
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.notebook import TrainingSession

    tmp = tmp_path_factory.mktemp("ui")
    csv = tmp / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    cfg = Config(EPOCHS=2, BATCH_SIZE=32, MAX_SEQUENCE_COUNT=1500, CSV_PATH=str(csv),
                 CALLBACKS=["early_stopping", "tqdm_progress"])
    ctx = RunContext.create(cfg, root=tmp / "runs")
    session = TrainingSession(cfg, run_context=ctx, epochs=2, calibrate=False)
    session.widget()
    session.start()
    result = session.wait(timeout=900)
    return session, result, ctx, csv


@pytest.mark.slow
def test_training_session_trains_in_a_thread_with_live_widgets(session_run):
    session, result, ctx, _ = session_run
    assert session.status == "finished" and result is not None and len(session.history) == 2
    assert {"loss", "val_loss", "seconds"} <= set(session.history[-1])
    assert session._w["epochs"].value == 2 and session._w["curves"].outputs  # curves were drawn
    assert any("Training on" in o.get("text", "") or "Epoch" in o.get("text", "") or o.get("text")
               for o in session._w["log"].outputs)               # the thread's log went to the panel
    assert (ctx.run_dir / "artifacts" / "weights.h5").exists()
    assert "tqdm_progress" not in ctx.config.CALLBACKS            # no console bars from a thread


@pytest.mark.slow
def test_stop_ends_training_early_but_still_evaluates(session_run, tmp_path):
    from neural_trade.notebook import TrainingSession

    session, _, ctx, csv = session_run
    s2 = TrainingSession(ctx.config.copy().override(MODEL_PATH=str(tmp_path / "w.h5"),
                                                    SCALER_PATH=str(tmp_path / "s.joblib")),
                         epochs=5, calibrate=False, save_artifacts=False)
    s2.start()
    s2.stop()
    res = s2.wait(timeout=900)
    assert res is not None and s2.status == "stopped early - evaluated" and len(s2.history) <= 1
    assert "delta" in res.predictions


@pytest.mark.slow
def test_backtest_explorer_runs_every_strategy_from_the_widget(session_run):
    from neural_trade.notebook import BacktestExplorer
    from neural_trade.strategy import Strategies

    _, _, ctx, csv = session_run
    ex = BacktestExplorer.from_run(ctx.run_dir, csv_path=str(csv))
    box = ex.widget()
    for name in Strategies.list_names():
        ex._w["strategy"].value = name                       # rebuilds the knob widgets
        assert all(hasattr(k, "_field") for k in ex._w["knobs"].children)
        ex._w["costs"]["random_seeds"].value = 2
        ex.click_run()
        assert ex.last is not None and ex.last.strategy == name, ex._w["status"].value
    table = ex.summary_frame()
    assert {"buy_and_hold", "always_flat", "random_same_freq"} <= set(table.index)
    res = ex.run("calibrated_quantile", {"entry_quantile": 0.8}, {"fee_bps": 0.0, "random_seeds": 0})
    assert res.config.fee_bps == 0.0 and box is ex.widget()


@pytest.mark.slow
def test_calibration_explorer_refits_and_scores_on_test(session_run):
    from neural_trade.notebook import CalibrationExplorer

    _, _, ctx, csv = session_run
    ex = CalibrationExplorer.from_run(ctx.run_dir, csv_path=str(csv))
    for scale, shrink in (("realized_vol", True), ("none", False)):
        t = ex.refit(scale, shrink_delta=shrink, alpha=0.2)
        assert list(t.index) == ["h0", "h1", "h2"] and t["coverage"].between(0, 1).all()
        if not shrink:
            assert (t["delta beta"] == 1.0).all() and np.allclose(t["EV raw delta"], t["EV served delta"])
    rel, cov = ex.figures("h2")
    assert len(rel.data) == 3 and len(cov.data) == 2
    ex.click_refit()


def test_pick_run_finds_the_newest_servable_run_or_explains(tmp_path):
    import os

    from neural_trade.notebook import pick_run, servable_runs

    with pytest.raises(FileNotFoundError, match="notebook 01_train_and_monitor"):
        pick_run(None, tmp_path)
    for i, name in enumerate(("gates/m6", "20260101-a", "20260102-b")):
        (tmp_path / name / "artifacts").mkdir(parents=True)
        (tmp_path / name / "artifacts" / "weights.h5").write_bytes(b"")
        os.utime(tmp_path / name, (1_000 + i, 1_000 + i))
    (tmp_path / "no_bundle").mkdir()
    assert [p.name for p in servable_runs(tmp_path)] == ["20260102-b", "20260101-a", "m6"]
    assert pick_run(None, tmp_path).name == "20260102-b"
    assert pick_run(tmp_path / "gates" / "m6", tmp_path).name == "m6"
    with pytest.raises(FileNotFoundError, match="no serving bundle"):
        pick_run(tmp_path / "no_bundle", tmp_path)


def test_show_puts_figures_and_tables_into_the_widget_itself():
    import ipywidgets as w
    import plotly.graph_objects as go

    from neural_trade.notebook._display import show

    out = w.Output()
    show(out, go.Figure(go.Scatter(y=[1, 2, 3]), layout_title_text="curves"))
    (o,) = out.outputs
    assert "application/vnd.plotly.v1+json" in o["data"] and "text/html" in o["data"]
    show(out, pd.DataFrame({"a": [1]}))                      # replaces, never appends
    (o,) = out.outputs
    assert "<table" in o["data"]["text/html"]

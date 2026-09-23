"""JSONL epoch telemetry never raises; convergence score NaN stays NaN; RunContext layout."""
from __future__ import annotations

import json

import numpy as np

from neural_trade.core.config import Config
from neural_trade.experiments.run_context import RunContext
from neural_trade.telemetry.epoch_logger import JsonlEpochLogger, read_metrics
from neural_trade.training.callbacks import ParamsLogger


def test_jsonl_logger_writes_one_line_per_epoch_with_weights_and_periods(make_loss_model, tmp_path):
    m = make_loss_model(261.0, 3.2)
    m.compile(optimizer="adam")
    cb = JsonlEpochLogger(tmp_path, run_id="r1")
    cb.set_model(m)
    cb.on_train_begin()
    for e in range(3):
        cb.on_epoch_begin(e)
        cb.on_train_batch_end(0)
        cb.on_epoch_end(e, {"loss": 1.0 + e, "val_loss": float("nan")})
    cb.on_train_end()
    rows = read_metrics(tmp_path / "metrics.jsonl")
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert rows[1]["loss"] == 2.0 and rows[0]["val_loss"] is None  # NaN -> null, file stays valid JSON
    assert rows[0]["lambda_hd"] > 0 and rows[0]["run_id"] == "r1" and rows[0]["lr"] is not None
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["done"] and status["epochs_completed"] == 3 and status["n_errors"] == 0


def test_jsonl_logger_never_raises_when_it_cannot_write(make_loss_model, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    cb = JsonlEpochLogger(blocker / "sub")  # parent is a file: every write fails
    cb.set_model(make_loss_model(261.0, 3.2))
    cb.on_train_begin()
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, {"loss": 1.0})
    cb.on_train_end()
    assert cb.n_errors >= 2


def test_convergence_score_nan_is_nan():
    info = ParamsLogger(layer=None)._detect_convergence([{"change_ma_period_0": float("nan")},
                                                         {"change_ma_period_0": float("nan")}])
    assert info is not None and np.isnan(info["convergence_score"])


def test_run_context_layout(tmp_path):
    ctx = RunContext.create(Config(EPOCHS=3), root=tmp_path, tags=["t"], name="demo", write_env=False)
    assert ctx.run_dir.is_dir() and ctx.run_id.endswith("-demo")
    assert (ctx.run_dir / "config.yaml").exists() and (ctx.run_dir / "meta.json").exists()
    assert ctx.config.MODEL_PATH == str(ctx.run_dir / "weights.h5")
    back = RunContext.load(ctx.run_dir)
    assert back.config.EPOCHS == 3 and back.tags == ["t"]


def test_compare_runs_tabulates_scored_runs_and_refuses_unscored(tmp_path):
    import json

    import pytest

    from neural_trade.experiments.compare import compare_runs

    def run(name, auc, scored=True, seed=0):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"seed": seed, "tags": ["t"]}), encoding="utf-8")
        if scored:
            rep = {"run_id": name, "model": {"horizons": {"h1": {"direction": {"auc": auc}, "delta": {},
                                                                  "variance": {"crpss": 0.01},
                                                                  "gauss_direction": {}}},
                                             "coherence": {"coherence_primary": 0.6}},
                   "backtest": {"summary": {"sharpe_net": -1.0}}}
            (d / "eval_report_test.json").write_text(json.dumps(rep), encoding="utf-8")

    run("a", 0.51)
    run("b", 0.53, seed=1)
    df = compare_runs(tmp_path / "*", metrics=["h1/direction/auc", "backtest/sharpe_net"])
    assert list(df.index) == ["a", "b"] and df.loc["b", "h1/direction/auc"] == 0.53 and df.loc["a", "seed"] == 0
    assert "coherence/coherence_primary" in compare_runs(tmp_path / "a").columns
    run("c", 0.0, scored=False)
    with pytest.raises(FileNotFoundError, match="no eval_report_test.json"):
        compare_runs(tmp_path / "*")
    assert compare_runs(tmp_path / "*", skip_unscored=True).attrs["unscored"] == ["c"]

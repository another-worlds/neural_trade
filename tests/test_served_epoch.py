"""The served model is the best-validation epoch, whether or not EarlyStopping stopped the run.

Keras 2.10 ``EarlyStopping(restore_best_weights=True)`` puts the best weights back only when it
stops training itself. A run that reached EPOCHS used to evaluate, calibrate and bundle the LAST
epoch while the dashboard starred the best one. These tests fail on that code.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import tensorflow as tf


class _ScriptedValLoss(tf.keras.callbacks.Callback):
    """Overwrite val_loss in the epoch logs with a fixed sequence, before the stoppers read it."""

    def __init__(self, values):
        super().__init__()
        self.values = list(values)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["val_loss"] = float(self.values[epoch])


class _CaptureWeights(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([np.array(w, copy=True) for w in self.model.get_weights()])


def _same(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_a_run_that_reaches_the_epoch_cap_serves_the_best_validation_epoch(tf, tiny_config, tmp_path,
                                                                          synthetic_bars, monkeypatch):
    import neural_trade.training.trainer as trainer_mod
    from neural_trade.experiments.run_context import RunContext

    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(csv)
    cfg.MAX_SEQUENCE_COUNT, cfg.BATCH_SIZE = 400, 32   # a few steps per epoch: this test is about which weights are kept
    cfg.EARLY = 50            # early stopping can never fire in 3 epochs
    cfg.PATIENCE = 50
    cfg.CALLBACKS = ["early_stopping", "model_checkpoint"]
    ctx = RunContext.create(cfg, root=tmp_path / "runs")

    scripted = _ScriptedValLoss([3.0, 1.0, 2.0])   # best is epoch 2 of 3 (1-based)
    real_build = trainer_mod.build_callbacks
    monkeypatch.setattr(trainer_mod, "build_callbacks",
                        lambda c, context, names: [scripted] + real_build(c, context, names))
    capture = _CaptureWeights()
    result = trainer_mod.train_and_evaluate(config=ctx.config, run_context=ctx, epochs=3, force=True,
                                            calibrate=False, fit_calibration=False, extra_callbacks=[capture])

    assert result.history.history["val_loss"] == [3.0, 1.0, 2.0]
    assert len(capture.weights) == 3 and not _same(capture.weights[1], capture.weights[2])
    served = result.model.get_weights()
    assert _same(served, capture.weights[1]), "the served weights must be the best-validation epoch's"
    assert not _same(served, capture.weights[2]), "the last epoch's weights were served"
    assert result.weights_epoch == 2 and result.weights_val_loss == 1.0
    assert "restored after training" in result.weights_source

    meta = json.loads((ctx.run_dir / "artifacts" / "meta.json").read_text(encoding="utf-8"))
    assert meta["weights_epoch"] == 2 and meta["weights_val_loss"] == 1.0 and meta["epochs_run"] == 3
    status = json.loads((ctx.run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["weights_epoch"] == 2 and status["weights_val_loss"] == 1.0 and status["done"] is True
    # every epoch row carries the learning rate it trained with, read at the epoch's start
    from neural_trade.telemetry.epoch_logger import read_metrics

    logged = read_metrics(ctx.run_dir / "metrics.jsonl")
    assert len(logged) == 3 and all(math.isclose(r["lr_used"], cfg.LR, rel_tol=1e-6) for r in logged)
    assert "lr_used" in result.history.history


class _StubModel:
    def __init__(self):
        self.set_to = None

    def set_weights(self, w):
        self.set_to = w


def _stopper(best_epoch, stopped_epoch, best=1.0, restore=True, monitor="val_loss"):
    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, restore_best_weights=restore)
    es.best_weights, es.best_epoch, es.stopped_epoch, es.best = ["best"], best_epoch, stopped_epoch, best
    return es


def test_serve_best_weights_covers_every_way_a_run_ends():
    from neural_trade.training.trainer import _serve_best_weights

    hist = SimpleNamespace(history={"val_loss": [3.0, 1.0, 2.0, 2.5]})
    m = _StubModel()   # ran to the cap: restore now
    assert _serve_best_weights(m, [_stopper(1, 0)], hist)[:2] == (2, 1.0) and m.set_to == ["best"]
    m = _StubModel()   # the stopper fired: Keras already restored, nothing to do
    epoch, vl, source = _serve_best_weights(m, [_stopper(1, 3)], hist)
    assert (epoch, vl) == (2, 1.0) and m.set_to is None and "restored by EarlyStopping" in source
    m = _StubModel()   # no restoring stopper on val_loss: the last epoch is served
    for cbs in ([], [_stopper(1, 0, restore=False)], [_stopper(1, 0, monitor="val_dir_mcc_h1")]):
        epoch, vl, source = _serve_best_weights(m, cbs, hist)
        assert (epoch, vl) == (4, 2.5) and m.set_to is None and "last epoch" in source
    assert _serve_best_weights(_StubModel(), [], None)[0] is None


def test_the_learning_rate_of_each_epoch_is_read_at_its_start_whatever_the_callback_order(tmp_path):
    """Finding 100 (robust fix): with ReduceLROnPlateau BEFORE the logger (a user's CALLBACKS order),
    the logger's 'lr' is the next epoch's rate; 'lr_used' is still the rate the epoch trained with."""
    from neural_trade.telemetry.epoch_logger import JsonlEpochLogger, read_metrics
    from neural_trade.training.trainer import LearningRateInEffect

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 3)).astype("float32")
    y = x.sum(axis=1, keepdims=True)
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(3,))])
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss="mse")
    # min_delta so large that no epoch after the first counts as an improvement: a cut after epochs 2, 3, 4
    cut = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=0, min_delta=1e9, verbose=0)
    model.fit(x, y, epochs=4, batch_size=16, verbose=0,
              callbacks=[LearningRateInEffect(), cut, JsonlEpochLogger(tmp_path, run_id="t")])
    rows = read_metrics(tmp_path / "metrics.jsonl")
    assert np.allclose([r["lr_used"] for r in rows], [0.01, 0.01, 0.005, 0.0025])
    assert np.allclose([r["lr"] for r in rows], [0.01, 0.005, 0.0025, 0.00125])      # one epoch late


def test_the_epoch_logger_is_placed_before_the_lr_scheduler():
    from neural_trade.training.trainer import _with_epoch_logger

    assert _with_epoch_logger(["early_stopping", "reduce_lr_on_plateau", "csv_logger"]) == [
        "early_stopping", "jsonl_epoch_logger", "reduce_lr_on_plateau", "csv_logger"]
    assert _with_epoch_logger(["early_stopping"]) == ["early_stopping", "jsonl_epoch_logger"]
    kept = ["reduce_lr_on_plateau", "jsonl_epoch_logger"]
    assert _with_epoch_logger(kept) == kept      # an explicit position is respected

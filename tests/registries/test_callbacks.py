"""Callbacks registry: default order reproduces the old hard-coded list; schedule and threshold."""
from __future__ import annotations

import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.registries.callbacks import Callbacks, build_callbacks
from neural_trade.training.callbacks import (LambdaScheduleCallback, MetricThresholdStop, ParamsLogger,
                                             TqdmCallback, TrainContext)


def test_twelve_callbacks_registered():
    assert len(Callbacks.list_names()) == 12
    assert {"csv_logger", "early_stopping", "model_checkpoint", "tqdm_progress", "params_logger",
            "reduce_lr_on_plateau", "jsonl_epoch_logger", "lambda_schedule", "tensorboard"} <= set(Callbacks.list_names())


def test_default_list_matches_the_previous_trainer_order(tmp_path):
    cfg = Config(MODEL_PATH=str(tmp_path / "w.h5"))
    cbs = build_callbacks(cfg, TrainContext(run_dir=tmp_path))
    types = [type(c) for c in cbs]
    assert types == [tf.keras.callbacks.CSVLogger, tf.keras.callbacks.EarlyStopping,
                     tf.keras.callbacks.ModelCheckpoint, TqdmCallback, ParamsLogger,
                     tf.keras.callbacks.ReduceLROnPlateau]
    es = cbs[1]
    assert es.monitor == "val_loss" and es.patience == cfg.EARLY and es.restore_best_weights
    assert cbs[0].filename == str(tmp_path / "training_log.csv")


def test_lambda_schedule_is_piecewise_constant(make_loss_model):
    m = make_loss_model(261.0, 3.2)
    cb = LambdaScheduleCallback({"lambda_hd": {0: 0.0, 2: 0.3}})
    cb.set_model(m)
    seen = []
    for epoch in range(4):
        cb.on_epoch_begin(epoch)
        seen.append(round(float(m.lambda_hd), 6))
    assert seen == [0.0, 0.0, 0.3, 0.3]


def test_metric_threshold_stops_on_non_finite(make_loss_model):
    m = make_loss_model(261.0, 3.2)
    cb = MetricThresholdStop(monitor="loss", threshold=100.0)
    cb.set_model(m)
    m.stop_training = False
    cb.on_epoch_end(0, {"loss": 5.0})
    assert not m.stop_training
    cb.on_epoch_end(1, {"loss": float("nan")})
    assert m.stop_training and cb.stopped_epoch == 1

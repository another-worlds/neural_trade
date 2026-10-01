"""Callbacks registry: default order reproduces the old hard-coded list; schedule and threshold."""
from __future__ import annotations

import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.registries.callbacks import Callbacks, build_callbacks
from neural_trade.training.callbacks import (LambdaScheduleCallback, MetricThresholdStop, ParamsLogger,
                                             ReduceLRBothOptimizers, TqdmCallback, TrainContext)


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


def test_reduce_lr_on_plateau_default_is_plain_and_untouched(tmp_path):
    """NT-097: LR_SCHEDULE_BOTH_OPTIMIZERS off (default) keeps the exact pre-NT-097 callback."""
    cfg = Config(MODEL_PATH=str(tmp_path / "w.h5"))
    assert cfg.LR_SCHEDULE_BOTH_OPTIMIZERS is False
    cbs = build_callbacks(cfg, TrainContext(run_dir=tmp_path))
    reduce_lr = [c for c in cbs if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau)][0]
    assert type(reduce_lr) is tf.keras.callbacks.ReduceLROnPlateau  # not the ReduceLRBothOptimizers subclass


def test_reduce_lr_on_plateau_both_optimizers_switch_selects_the_subclass(tmp_path):
    cfg = Config(MODEL_PATH=str(tmp_path / "w.h5"), LR_SCHEDULE_BOTH_OPTIMIZERS=True)
    cbs = build_callbacks(cfg, TrainContext(run_dir=tmp_path))
    reduce_lr = [c for c in cbs if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau)][0]
    assert isinstance(reduce_lr, ReduceLRBothOptimizers)


def test_reduce_lr_both_optimizers_scales_indicator_lr_by_the_same_factor(make_loss_model):
    """NT-097 point 10: the indicator LR never decayed on its own (B_model_indicators.md 2.2/7.10)."""
    m = make_loss_model(261.0, 3.2)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    m.indicator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)  # ratio 5, as in production
    cb = ReduceLRBothOptimizers(monitor="val_loss", factor=0.5, patience=0, cooldown=0, min_delta=1e-8)
    cb.set_model(m)
    cb.on_train_begin()
    for epoch, val in enumerate([1.0, 1.0, 1.0]):  # no improvement -> reduces once patience is exceeded
        cb.on_epoch_end(epoch, {"val_loss": val})
    main_lr = float(tf.keras.backend.get_value(m.optimizer.lr))
    ind_lr = float(tf.keras.backend.get_value(m.indicator_optimizer.lr))
    assert main_lr < 0.001, "the main optimizer's rate should have been reduced"
    assert abs(ind_lr / main_lr - 5.0) < 1e-6, "the indicator/main LR ratio must be preserved"

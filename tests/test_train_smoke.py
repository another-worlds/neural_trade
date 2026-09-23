"""Three real training steps on synthetic bars.

Fails on the pre-fix code: the NaN gradient from the trend loss reached
tf.clip_by_global_norm, every weight became NaN on step 1, the learnable
indicator periods went NaN and validation loss froze.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _build(cfg, tmp_path, synthetic_bars):
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.data.processor import DataProcessor
    from neural_trade.models.facade import PricePredictor

    csv_path = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv_path, index=False)
    cfg.CSV_PATH = str(csv_path)
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")

    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    (X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te, y_tr, _y_te, _scaler) = dp.prepare_datasets(df, close)

    predictor = PricePredictor(cfg)
    base = predictor.build_model()
    std = float(np.std(y_tr))
    pred_scale = std if std > 0 else 1.0
    pred_mean = float(np.mean(y_tr))
    model = CustomTrainModel(
        base_model=base, pred_scale=pred_scale, pred_mean=pred_mean,
        lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
        lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND, lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND,
        lambda_dir=cfg.LAMBDA_DIR, config=cfg, inputs=base.inputs, outputs=base.outputs,
    )
    train_ds, val_ds = predictor.create_datasets(X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LR))
    return model, train_ds, val_ds


def test_three_steps_keep_weights_finite_and_val_loss_moves(tf, tiny_config, tmp_path, synthetic_bars, monkeypatch):
    monkeypatch.chdir(tmp_path)  # CSVLogger / ParamsLogger write relative paths
    tf.keras.utils.set_random_seed(0)
    model, train_ds, val_ds = _build(tiny_config, tmp_path, synthetic_bars)

    layer = model._indicator_layer
    assert layer is not None, "indicator layer not located; gradient routing would silently degrade"
    before_params = layer.get_learned_parameters()
    assert len(before_params) == 18 and all(np.isfinite(v) for v in before_params.values())

    before = model.evaluate(val_ds, verbose=0, return_dict=True)["loss"]
    assert np.isfinite(before)

    hist = model.fit(train_ds, epochs=1, steps_per_epoch=3, verbose=0)

    assert all(np.all(np.isfinite(w)) for w in model.get_weights()), "non-finite weights after 3 steps"
    assert hist.history["nonfinite_grad_steps"][-1] == 0
    assert np.isfinite(hist.history["grad_global_norm"][-1])

    after_params = layer.get_learned_parameters()
    assert all(np.isfinite(v) for v in after_params.values()), f"indicator periods went non-finite: {after_params}"
    assert any(abs(after_params[k] - before_params[k]) > 1e-6 for k in before_params), "no indicator period moved"
    assert all(v >= tiny_config.MOMENTUM_CLIP_MIN - 1e-3 for v in after_params.values())

    after = model.evaluate(val_ds, verbose=0, return_dict=True)["loss"]
    assert np.isfinite(after)
    assert after != before, "validation loss did not change after an update (frozen evaluation path)"


class _FreezeLearning(tf.keras.callbacks.Callback):
    """Set both optimizers' learning rates to 0 so validation loss is exactly flat."""

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.0)
        tf.keras.backend.set_value(self.model.indicator_optimizer.learning_rate, 0.0)


def test_early_stopping_fires_on_a_plateau(tf, tiny_config, tmp_path, synthetic_bars, monkeypatch):
    """M4: 'early stopping fires on a plateaued run'.

    Before Phase A, EARLY and PATIENCE both equalled EPOCHS and were read from the class, not
    the instance, so no stopper could ever fire. With a frozen model val_loss is exactly flat
    and EarlyStopping(patience=EARLY) must stop after EARLY + 1 epochs, restoring the best.
    """
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.EARLY = 2
    cfg.PATIENCE = 1

    result = train_and_evaluate(config=cfg, epochs=8, force=True, calibrate=False,
                                fit_calibration=False, extra_callbacks=[_FreezeLearning()])
    val = result.history.history["val_loss"]
    assert len(set(np.round(val, 9))) == 1, f"frozen model should have a flat val_loss, got {val}"
    assert len(val) == cfg.EARLY + 1, f"expected to stop after {cfg.EARLY + 1} epochs, ran {len(val)}"

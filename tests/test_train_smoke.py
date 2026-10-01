"""Real training on synthetic bars (and on the bundled CSV for the default-Config run).

- Three training steps keep weights and indicator periods finite and move validation loss. Fails on
  the pre-fix code: the NaN gradient from the trend loss reached tf.clip_by_global_norm, every
  weight became NaN on step 1, the learnable indicator periods went NaN and validation loss froze.
- EarlyStopping stops a frozen model after EARLY + 1 epochs.
- TRAIN_METRICS_EVERY subsamples the step diagnostics but keeps the loss and epoch logs complete.
- A default-Config run without a RunContext leaves only the MODEL_PATH weights in its working
  directory (NT-028; the warm start reads them, NT-049).
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf


def _build(cfg, tmp_path, synthetic_bars):
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.data.processor import DataProcessor
    from neural_trade.data.datasets import create_datasets
    from neural_trade.models.registry import Models

    csv_path = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv_path, index=False)
    cfg.CSV_PATH = str(csv_path)
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")

    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    (X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te, y_tr, _y_te, _scaler) = dp.prepare_datasets(df, close)

    # NT-028: PricePredictor (models/facade.py) was a thin, uncalled wrapper over these two; the
    # current API is Models.build(...) and data.datasets.create_datasets(...) directly.
    base = Models.build(getattr(cfg, 'MODEL_NAME', None), cfg)
    std = float(np.std(y_tr))
    pred_scale = std if std > 0 else 1.0
    pred_mean = float(np.mean(y_tr))
    model = CustomTrainModel(
        base_model=base, pred_scale=pred_scale, pred_mean=pred_mean,
        lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
        lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND, lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND,
        lambda_dir=cfg.LAMBDA_DIR, config=cfg, inputs=base.inputs, outputs=base.outputs,
    )
    train_ds, val_ds = create_datasets(cfg, X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LR))
    return model, train_ds, val_ds


@pytest.mark.slow
def test_three_steps_keep_weights_finite_and_val_loss_moves(tf, tiny_config, tmp_path, synthetic_bars, monkeypatch):
    """NT-109: kept on the full OHLCV/14-family default (``tiny_config``, unshrunk families) and
    marked slow - the assertion below is exactly about that family count (D-047's gradient-routing
    completeness), so it is one of the tests this item's acceptance (2) keeps training the default."""
    monkeypatch.chdir(tmp_path)  # CSVLogger / ParamsLogger write relative paths
    tf.keras.utils.set_random_seed(0)
    model, train_ds, val_ds = _build(tiny_config, tmp_path, synthetic_bars)

    layer = model._indicator_layer
    assert layer is not None, "indicator layer not located; gradient routing would silently degrade"
    before_params = layer.get_learned_parameters()
    assert len(before_params) == 54 and all(np.isfinite(v) for v in before_params.values())  # 14 families x 3 (NT-047)

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


def test_early_stopping_fires_on_a_plateau(tf, tiny_close_only_config, tmp_path, synthetic_bars, monkeypatch,
                                           run_eagerly):
    """M4: 'early stopping fires on a plateaued run'.

    Before Phase A, EARLY and PATIENCE both equalled EPOCHS and were read from the class, not
    the instance, so no stopper could ever fire. With a frozen model val_loss is exactly flat
    and EarlyStopping(patience=EARLY) must stop after EARLY + 1 epochs, restoring the best.

    NT-109: whether the stopper fires does not depend on the indicator family count, so this runs on
    ``tiny_close_only_config``, and eagerly (``run_eagerly``) to skip the graph-trace cost.
    """
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_close_only_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.EARLY = 2
    cfg.PATIENCE = 1
    cfg.BATCH_SIZE = 64   # fewer, bigger steps per epoch (NT-109, eager mode): irrelevant to which epoch wins
    cfg.MAX_SEQUENCE_COUNT = 450   # smaller than tiny_config's 600, still enough for the purged split

    result = train_and_evaluate(config=cfg, epochs=8, force=True, calibrate=False,
                                fit_calibration=False, extra_callbacks=[_FreezeLearning()])
    val = result.history.history["val_loss"]
    assert len(set(np.round(val, 9))) == 1, f"frozen model should have a flat val_loss, got {val}"
    assert len(val) == cfg.EARLY + 1, f"expected to stop after {cfg.EARLY + 1} epochs, ran {len(val)}"


def test_training_diagnostics_are_subsampled_but_the_loss_and_epoch_logs_are_complete(
        tf, tiny_close_only_config, tmp_path, synthetic_bars, monkeypatch, run_eagerly):
    """TRAIN_METRICS_EVERY=4: the loss accumulates every step, the diagnostics every 4th step
    (steps 1, 5, 9 of 10), and the epoch logs still carry the full train_*/val_* keys (added once per
    epoch by _EpochTrainLogs, before validation resets the accumulators).

    NT-109: subsampling counts and log keys do not depend on the indicator family count, so this runs
    on ``tiny_close_only_config``, and eagerly (``run_eagerly``) to skip the graph-trace cost."""
    monkeypatch.chdir(tmp_path)
    tiny_close_only_config.TRAIN_METRICS_EVERY = 4
    model, train_ds, val_ds = _build(tiny_close_only_config, tmp_path, synthetic_bars)
    seen = {}

    class Probe(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            seen["loss"] = float(model._step_means["loss"].count)
            seen["diag"] = float(model._step_means["point_h1"].count)
            seen["logs"] = sorted(logs or {})

    hist = model.fit(train_ds, epochs=1, steps_per_epoch=10, validation_data=val_ds, verbose=0, callbacks=[Probe()])
    b = tiny_close_only_config.BATCH_SIZE
    assert seen["loss"] == 10 * b and seen["diag"] == 3 * b
    assert seen["logs"] == ["loss", "nonfinite_grad_steps"]  # lean per-step logs
    h = hist.history
    for key in ("loss", "val_loss", "grad_global_norm", "point_h1", "train_dir_mcc_h1", "pit_ks_h1",
                "val_dir_mcc_h1", "val_pit_ks_h1", "nonfinite_grad_steps"):
        assert key in h and np.isfinite(h[key][-1]), key


@pytest.mark.slow
def test_a_default_config_run_without_a_run_context_leaves_only_the_weights(tf, tmp_path, monkeypatch):
    """NT-028 acceptance (3): a 1-epoch CPU train_and_evaluate() with the default Config and no
    RunContext creates no file in its working directory that nothing reads. training_log.csv and
    indicator_params_history.csv (csv_logger / params_logger skip without a run directory) and
    SCALER_PATH (nothing loads it) are gone; the MODEL_PATH weights stay, because the warm start
    reads them on a later run in the same directory (NT-049). Only the data is shrunk
    (MAX_SEQUENCE_COUNT) and read from the bundled CSV by absolute path.

    NT-109: this is the item's "at least one test still trains the full default config", so it keeps
    the unmodified default ``Config()`` (all 14 OHLCV families) and moves to the slow suite."""
    from pathlib import Path

    import pytest

    from neural_trade.core.config import Config
    from neural_trade.training.trainer import train_and_evaluate

    csv = Path(__file__).resolve().parent.parent / "binance_btcusdt_1min_ccxt.csv"
    if not csv.exists():
        pytest.skip(f"{csv.name} is not present")
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    train_and_evaluate(config=cfg, csv_path=str(csv), config_overrides={"MAX_SEQUENCE_COUNT": 3000}, epochs=1)

    created = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert created == [Config().MODEL_PATH], created

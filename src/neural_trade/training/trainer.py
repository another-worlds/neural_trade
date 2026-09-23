"""train_and_evaluate: the end-to-end training entry point (moved from model.py in B10).

data (DataProcessor) -> model (Models registry) -> objective (Losses registry) -> optimizers
(Optimizers registry) -> optional loss-weight calibration -> ablations -> fit with early
stopping on the VALIDATION block -> predictions on test -> CalibrationPipeline fit on the
CALIBRATION block and applied to test -> TrainResult.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from neural_trade.core.config import Config
from neural_trade.data.datasets import create_datasets
from neural_trade.data.processor import DataProcessor
from neural_trade.utils.seeding import seed_everything
from neural_trade.metrics.evaluate import _compute_all_horizon_metrics
from neural_trade.registries.models import Models
from neural_trade.serving.postprocess import heads_to_predictions
from neural_trade.registries.callbacks import build_callbacks
from neural_trade.training.callbacks import TrainContext
from neural_trade.training.custom_model import CustomTrainModel
from neural_trade.training.lambda_calibration import calibrate_loss_weights
from neural_trade.training.lambdas import ablate
from neural_trade.training.optim import build_optimizers

try:
    from neural_trade.calibration import CalibrationPipeline as _CalibrationPipeline
except Exception:  # pragma: no cover
    _CalibrationPipeline = None

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Single-source-of-truth training + inference output bundle."""

    config: 'Config'
    model: 'CustomTrainModel'
    target_scaler: StandardScaler
    input_scaler: Optional[StandardScaler]

    X_test_seq: np.ndarray
    y_test: np.ndarray  # raw deltas [N,3]
    last_close_test: np.ndarray
    extended_trends_test: np.ndarray
    history: Any

    # Predictions are raw (inverse-scaled) deltas and head outputs.
    predictions: Dict[str, Dict[str, np.ndarray]]
    metrics: Dict[str, Any]
    calibration_pipeline: Optional[Any] = None  # CalibrationPipeline, None if not fitted
    calibration_lambdas: Optional[Dict[str, float]] = None  # Lambdas after pre-training calibration
    # Calibration-split artefacts (None when fit_calibration=False or the pipeline failed).
    predictions_cal: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    predictions_calibrated: Optional[Dict[str, Any]] = None  # CalibrationPipeline.apply(predictions) on TEST
    calibration_report: Optional[Dict[str, Dict[str, float]]] = None  # conformal coverage on TEST, per horizon
    y_cal: Optional[np.ndarray] = None
    last_close_cal: Optional[np.ndarray] = None
    fold: Optional[Any] = None  # FoldIndices used for the split
    normalizer: Optional[Any] = None  # data.scaling.WindowNormalizer fitted on train
    windows_test: Optional[np.ndarray] = None  # RAW test windows [N, LOOKBACK]
    windows_cal: Optional[np.ndarray] = None   # RAW calibration windows
    artifacts_dir: Optional[str] = None  # where the serving bundle was written, if any


def _apply_config_overrides(config: 'Config', overrides: Optional[dict]) -> 'Config':
    """Validated update: unknown names raise InvalidConfigurationError with suggestions
    (the old setattr loop silently created misspelled attributes that nothing read)."""
    if not overrides:
        return config
    return config.override(**dict(overrides))


def _temperature_of(pipeline, h):
    """Fitted temperature for horizon *h* from a CalibrationPipeline, NaN if unavailable."""
    ts = getattr(pipeline, 'temperature_scaler', None)
    for attr in ('temperatures', 'temperature', 'T', 'temps', '_temperatures'):
        v = getattr(ts, attr, None)
        if isinstance(v, dict) and h in v:
            try:
                return float(v[h])
            except (TypeError, ValueError):
                return float('nan')
    return float('nan')


def _calibration_coverage_report(calibrated, y_true_raw, pipeline, alpha=0.1):
    """Empirical coverage / mean width of the conformal intervals on a held-out split, per horizon."""
    report = {}
    y = np.asarray(y_true_raw, dtype=float)
    for i, h in enumerate(("h0", "h1", "h2")):
        if y.ndim < 2 or i >= y.shape[1] or h not in calibrated.get("intervals", {}):
            continue
        lo, hi = calibrated["intervals"][h]
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        m = min(len(lo), y.shape[0])
        inside = (y[:m, i] >= lo[:m]) & (y[:m, i] <= hi[:m])
        report[h] = {
            "coverage90": float(np.mean(inside)) if m else float('nan'),
            "width90": float(np.mean(hi[:m] - lo[:m])) if m else float('nan'),
            "temperature": _temperature_of(pipeline, h),
            "target": 1.0 - alpha,
            "n": int(m),
        }
    return report


def _predict_heads(model, X, n, target_scaler, cfg, batch_size=None):
    """Run the model on scaled windows and return the raw-unit predictions dict.

    {"delta": {h: raw $ deltas}, "direction_prob": {h: P(up) in [0, 1]}, "variance": {h: scaled var}}
    Shared with serving (neural_trade.serving.postprocess), so served and reported predictions match.
    """
    bs = int(batch_size or getattr(cfg, 'BATCH_SIZE', 64))
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(X, dtype='float32')).batch(bs)
    heads = model.predict(ds, verbose=0)
    return heads_to_predictions(heads, n, float(target_scaler.scale_[0]), float(target_scaler.mean_[0]), cfg)


def train_and_evaluate(
    *,
    config: Optional['Config'] = None,
    config_overrides: Optional[dict] = None,
    csv_path: Optional[str] = None,
    read_csv_kwargs: Optional[dict] = None,
    epochs: Optional[int] = None,
    force: bool = False,
    calibrate: bool = True,
    fit_calibration: bool = True,
    extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    run_context=None,
    save_artifacts: Optional[bool] = None,
) -> TrainResult:
    """Train (optionally) and evaluate, returning a rich result bundle.

    This is intended to be the notebook's single source of truth for:
    - data prep + scaling
    - model heads and extraction
    - metrics and evaluation semantics
    """

    cfg = config or Config()
    if csv_path is not None:
        cfg.CSV_PATH = csv_path
    cfg = _apply_config_overrides(cfg, config_overrides)
    cfg.validate()  # P0-3 / P1-4: early enforcement (added in Config refactor)
    # Seed AFTER the overrides: seeding first ignored a SEED given in config_overrides.
    seed_everything(int(getattr(cfg, 'SEED', 42)))

    logger.info("Starting enhanced model training with extended trend features...")
    data_processor = DataProcessor(cfg)
    df, close_values = data_processor.load_and_prepare_data(read_csv_kwargs=read_csv_kwargs)

    (X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
     X_test_seq, y_test_scaled, last_close_test, extended_trends_test,
     y_train, y_test, target_scaler) = data_processor.prepare_datasets(df, close_values)

    input_scaler = getattr(data_processor, 'input_scaler', None)
    base_model = Models.build(getattr(cfg, 'MODEL_NAME', None), cfg)
    optimizer_pair = build_optimizers(cfg)  # Optimizers registry: OPTIMIZER_NAME, INDICATOR_OPTIMIZER_NAME
    pred_scale = np.std(y_train) if np.std(y_train) > 0 else 1.0
    pred_mean = np.mean(y_train)
    custom_model = CustomTrainModel(
        base_model=base_model,
        pred_scale=pred_scale,
        pred_mean=pred_mean,
        lambda_point=cfg.LAMBDA_POINT,
        lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
        lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND,
        lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND,
        lambda_dir=cfg.LAMBDA_DIR,
        config=cfg,
        indicator_optimizer=optimizer_pair.indicator,
        inputs=base_model.inputs,
        outputs=base_model.outputs,
    )

    _vb = data_processor.val_block  # early stopping / checkpoint / LR select on the VALIDATION block, never on test
    train_ds, val_ds = create_datasets(cfg,
        X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
        _vb["X"], _vb["y_scaled"], _vb["last_close"], _vb["extended_trends"],
    )

    # Pre-training loss-weight calibration (training.lambda_calibration); restores on failure.
    _calib_lambdas: Optional[Dict[str, float]] = None
    if calibrate is True:
        _calib_lambdas = calibrate_loss_weights(custom_model, train_ds, cfg, X_train_seq.shape[0])
    # Ablations (Config.ABLATE_LAMBDAS) apply AFTER calibration so toggling one term never
    # rescales the others through the calibration reference.
    if getattr(cfg, 'ABLATE_LAMBDAS', None):
        ablate(custom_model, cfg.ABLATE_LAMBDAS)

    custom_model.compile(optimizer=optimizer_pair.main)

    learnable_layer = None
    for layer in custom_model.layers:
        if getattr(layer, 'name', '').startswith('learnable_indicators'):
            learnable_layer = layer
            break
    # Callbacks registry, Config.CALLBACKS in order (the default is the previously hard-coded list).
    # S21: exactly one EarlyStopping (val_loss, restore best); the MCC stopper is opt-in only.
    context = TrainContext(model=custom_model, indicator_layer=learnable_layer,
                           run_dir=getattr(run_context, 'run_dir', None), run_id=getattr(run_context, 'run_id', None))
    names = list(getattr(cfg, 'CALLBACKS', []) or [])
    if run_context is not None and 'jsonl_epoch_logger' not in names:
        names.append('jsonl_epoch_logger')
    if getattr(cfg, 'LOSS_WEIGHT_SCHEDULE', None) and 'lambda_schedule' not in names:
        names.append('lambda_schedule')
    callbacks_list = build_callbacks(cfg, context, names)
    if extra_callbacks:
        callbacks_list += list(extra_callbacks)

    actual_epochs = int(epochs) if epochs is not None else int(cfg.EPOCHS)
    history = None
    if os.path.exists(cfg.MODEL_PATH) and not force:
        logger.info(f"Loading existing model weights from {cfg.MODEL_PATH}...")
        try:
            custom_model.load_weights(cfg.MODEL_PATH)
        except Exception as e:
            logger.warning(f"Warning: failed to load existing weights but continuing: {e}")
    else:
        if os.path.exists(cfg.MODEL_PATH) and force:
            try:
                custom_model.load_weights(cfg.MODEL_PATH)  # force=True retrains, warm-started from these
                logger.info("Warm start from existing weights %s", cfg.MODEL_PATH)
            except Exception:
                logger.warning("existing weights %s not loaded; training from scratch", cfg.MODEL_PATH,
                               exc_info=True)
        logger.info(f"Training for {actual_epochs} epochs...")
        history = custom_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=actual_epochs,
            callbacks=callbacks_list,
            verbose=0,
        )
        logger.info(f"Enhanced model weights saved to {cfg.MODEL_PATH}")
        try:
            joblib.dump(target_scaler, cfg.SCALER_PATH)
            if input_scaler is not None:
                joblib.dump(input_scaler, cfg.SCALER_PATH.replace('.joblib', '_input.joblib'))
        except Exception:
            logger.exception("could not save the scalers to %s", cfg.SCALER_PATH)

    logger.info("Evaluating enhanced model...")
    predictions = _predict_heads(custom_model, X_test_seq, y_test.shape[0], target_scaler, cfg)
    y_pred_h0_raw, y_pred_h1_raw, y_pred_h2_raw = (predictions["delta"][h] for h in ("h0", "h1", "h2"))
    dir_pred_h0, dir_pred_h1, dir_pred_h2 = (predictions["direction_prob"][h] for h in ("h0", "h1", "h2"))
    var_pred_h0, var_pred_h1, var_pred_h2 = (predictions["variance"][h] for h in ("h0", "h1", "h2"))

    # === DIAGNOSTIC: Check prediction quality ===
    # Print statistics to help diagnose issues
    # Diagnostic: construct horizon labels from config.HORIZON_STEPS and RESAMPLE_MINUTES
    def _format_tf_local(minutes: int) -> str:
        if minutes % 1440 == 0:
            days = minutes // 1440
            return f"{days}d" if days > 1 else "1d"
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"{hours}h"
        return f"{minutes}min"

    resample = int(getattr(cfg, 'RESAMPLE_MINUTES', 1))
    horizon_steps = list(getattr(cfg, 'HORIZON_STEPS', [1, 5, 15]))
    horizon_labels = [f"{k}(" + _format_tf_local(int(k_step * resample)) + ")" for k, k_step in zip(['h0','h1','h2'], horizon_steps)]

    logger.info("\n[Diagnostic: Prediction Statistics]")
    for h_idx, (h_name, y_pred_raw) in enumerate(zip(horizon_labels, [y_pred_h0_raw, y_pred_h1_raw, y_pred_h2_raw])):
        y_true_raw = y_test[:, h_idx]
        pred_mean = np.mean(y_pred_raw)
        pred_std = np.std(y_pred_raw)
        true_mean = np.mean(y_true_raw)
        true_std = np.std(y_true_raw)
        pred_min = np.min(y_pred_raw)
        pred_max = np.max(y_pred_raw)
        true_min = np.min(y_true_raw)
        true_max = np.max(y_true_raw)
        logger.info(f"  {h_name}: pred_mean={pred_mean:.6f}, true_mean={true_mean:.6f} | pred_std={pred_std:.6f}, true_std={true_std:.6f}")
        logger.info(f"         pred_range=[{pred_min:.6f}, {pred_max:.6f}], true_range=[{true_min:.6f}, {true_max:.6f}]")


    metrics = _compute_all_horizon_metrics(
        config=cfg,
        y_true_deltas=np.asarray(y_test),
        y_pred_deltas=predictions["delta"],
        last_close=np.asarray(last_close_test),
        dir_probs=predictions["direction_prob"],
    )

    # Attach a back-compat attribute
    try:
        custom_model.predictions_dict = predictions
    except Exception:
        logger.debug("could not attach predictions_dict to the model", exc_info=True)

    # Post-hoc calibration: fit on the CAL block, apply to the TEST predictions.
    # Previously it was fit on the test split itself (voiding the conformal guarantee and
    # contaminating every reported test metric) and nothing ever consumed the fit.
    cal_pipeline = None
    predictions_cal = None
    predictions_calibrated = None
    calibration_report: Optional[Dict[str, Any]] = None
    _cb = getattr(data_processor, 'cal_block', None)
    if fit_calibration and _CalibrationPipeline is not None and _cb is not None:
        try:
            logger.info("\nFitting CalibrationPipeline on the calibration split...")
            predictions_cal = _predict_heads(custom_model, _cb['X'], _cb['y_raw'].shape[0], target_scaler, cfg)
            cal_pipeline = _CalibrationPipeline(conformal_scale=getattr(cfg, 'CONFORMAL_SCALE', 'none'),
                                                shrink_delta=bool(getattr(cfg, 'DELTA_SHRINKAGE', False)))
            cal_pipeline.fit_from_arrays(
                predictions_dict=predictions_cal,
                y_true_delta_raw=np.asarray(_cb['y_raw'], dtype=float),
                last_close=np.asarray(_cb['last_close'], dtype=float),
                deadband_bps=float(getattr(cfg, 'DIR_DEADBAND_BPS', 0.0)),
                windows=_cb.get('X_raw'),
                pred_scale=float(target_scaler.scale_[0]),
                horizon_steps=tuple(cfg.HORIZON_STEPS),
            )
            predictions_calibrated = cal_pipeline.apply(predictions, alpha=0.1,
                                                        windows=getattr(data_processor, 'test_windows_raw', None))
            calibration_report = _calibration_coverage_report(predictions_calibrated, np.asarray(y_test), cal_pipeline)
            for _h, _row in calibration_report.items():
                logger.info(f"  [test] {_h}: conformal coverage@90 = {_row['coverage90']:.3f} (target >= 0.90), "
                      f"mean width = {_row['width90']:.2f} raw units, T = {_row['temperature']:.3f}")
        except Exception as _cal_err:
            import traceback
            logger.warning(f"CalibrationPipeline fit FAILED (continuing without calibration): {_cal_err}")
            traceback.print_exc()
            cal_pipeline = None
            predictions_calibrated = None

    result = TrainResult(
        config=cfg,
        model=custom_model,
        target_scaler=target_scaler,
        input_scaler=input_scaler,
        X_test_seq=X_test_seq,
        y_test=np.asarray(y_test),
        last_close_test=np.asarray(last_close_test),
        extended_trends_test=np.asarray(extended_trends_test),
        history=history,
        predictions=predictions,
        metrics=metrics,
        calibration_pipeline=cal_pipeline,
        calibration_lambdas=_calib_lambdas,
        predictions_cal=predictions_cal,
        predictions_calibrated=predictions_calibrated,
        calibration_report=calibration_report,
        y_cal=(np.asarray(_cb['y_raw']) if _cb is not None else None),
        last_close_cal=(np.asarray(_cb['last_close']) if _cb is not None else None),
        fold=getattr(data_processor, 'fold', None),
        windows_test=getattr(data_processor, 'test_windows_raw', None),
        windows_cal=(_cb.get('X_raw') if _cb is not None else None),
        normalizer=getattr(data_processor, 'normalizer', None),
    )

    # Serving bundle (training.artifacts): weights, config, target scale, normaliser, calibration.
    # Written for run-context runs by default, or whenever save_artifacts=True.
    if save_artifacts or (save_artifacts is None and run_context is not None):
        from neural_trade.training.artifacts import ArtifactBundle

        out_dir = cfg.ARTIFACTS_DIR
        ArtifactBundle.from_result(result).save(out_dir)
        result.artifacts_dir = str(out_dir)
        logger.info(f"Artifact bundle written to {out_dir}")
    return result


def train_model(extra_callbacks=None, epochs=None, force=False, calibrate=True):
    # Backward-compatible wrapper; prefer `train_and_evaluate()` for new code.
    result = train_and_evaluate(
        config=Config(),
        config_overrides=None,
        csv_path=None,
        read_csv_kwargs=None,
        epochs=epochs,
        force=force,
        calibrate=calibrate,
        extra_callbacks=list(extra_callbacks) if extra_callbacks else None,
    )

    custom_model = result.model
    target_scaler = result.target_scaler
    X_test_seq = result.X_test_seq
    y_test = result.y_test
    last_close_test = result.last_close_test
    history = result.history
    extended_trends_test = result.extended_trends_test

    # For legacy callers, keep `y_pred` as the 5-min horizon delta series.
    # The full set of head outputs is exposed via `predictions_dict`.
    y_pred = np.asarray(result.predictions["delta"]["h1"], dtype=float).reshape(-1)
    predictions_dict = result.predictions

    # Provide a horizon-wide summary (no "primary horizon" framing).
    try:
        m = result.metrics
        if isinstance(m, dict) and 'delta' in m:
            logger.info("\n[Summary: Per-Horizon Delta Metrics]")
            for h_key, label in zip(m.get('meta', {}).get('horizon_keys', ['h0','h1','h2']), m.get('meta', {}).get('horizon_labels', ['1min','5min','15min'])):
                hm = m['delta'].get(h_key, {})
                pm = m['price'].get(h_key, {})
                logger.info(f"  {label}: MSE={hm.get('mse'):.6f}, RMSE={hm.get('rmse'):.6f}, R2={pm.get('r2', hm.get('r2')):.6f}")
    except Exception:
        logger.debug("metric summary not printable", exc_info=True)

    return (
        custom_model,
        target_scaler,
        X_test_seq,
        y_test,
        y_pred,
        last_close_test,
        history,
        extended_trends_test,
        predictions_dict,
    )

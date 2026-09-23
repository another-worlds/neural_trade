"""Training callbacks (moved from model.py in B10; the Callbacks registry arrives in B11)."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from tqdm import tqdm


class TqdmCallback(callbacks.Callback):
    """Custom callback to show tqdm progress bar during training."""

    def __init__(self):
        super().__init__()
        self.epoch_bar = None
        self.batch_bar = None
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_bar = tqdm(total=self.params['epochs'], desc='Training Progress', unit='epoch')

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_bar = tqdm(total=self.params['steps'], desc=f'Epoch {epoch+1}', unit='batch', leave=False)

    def on_batch_end(self, batch, logs=None):
        if self.batch_bar:
            self.batch_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_bar:
            self.batch_bar.close()
        if self.epoch_bar:
            # Update with current metrics
            elapsed_time = time.time() - self.start_time
            logs_str = ""
            if logs:
                metrics = ['loss', 'val_loss', 'val_f1', 'val_dir_acc']
                log_items = [f"{k}={v:.4f}" for k, v in logs.items() if k in metrics and v is not None]
                logs_str = " | " + " ".join(log_items) if log_items else ""

            self.epoch_bar.set_postfix_str(f"Time: {elapsed_time:.1f}s{logs_str}")
            self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        if self.epoch_bar:
            total_time = time.time() - self.start_time
            self.epoch_bar.set_postfix_str(f"Completed in {total_time:.1f}s")
            self.epoch_bar.close()


class ParamsLogger(tf.keras.callbacks.Callback):
    """
    Enhanced ParamsLogger for tracking learnable indicator parameters per epoch.

    Features:
    - Logs all 30+ learnable indicator parameters at each epoch
    - Writes CSV after each epoch (immediate feedback)
    - Tracks parameter change rates for convergence detection
    - Detects drift vs convergence patterns
    """
    def __init__(self, layer, out_csv='indicator_params_history.csv'):
        super().__init__()
        self.layer = layer
        self.out_csv = out_csv
        self.rows = []
        self.prev_params = None
        self.prev_epoch = -1
        self.convergence_window = 5  # epochs for convergence detection

    # Prefixes that identify actual indicator parameters vs. Keras log scalars
    _INDICATOR_PREFIXES = ('ma_period_', 'macd_', 'rsi_period_', 'bb_period_')

    def _is_indicator_key(self, key):
        return any(key.startswith(p) for p in self._INDICATOR_PREFIXES)

    def _calculate_param_changes(self, current_params):
        """Calculate per-parameter change rates for convergence detection.

        Only computes changes for actual learnable indicator parameters
        (ma_period_*, macd_*, rsi_period_*, bb_period_*).  Keras log scalars
        like log_loss / log_val_loss are intentionally excluded to avoid them
        contaminating the convergence signal.
        """
        if self.prev_params is None:
            return None

        changes = {}
        for key in current_params:
            if not self._is_indicator_key(key):
                continue  # skip log_*, epoch, timestamp, convergence_*, etc.
            if key in self.prev_params:
                try:
                    prev_val = float(self.prev_params[key])
                    curr_val = float(current_params[key])
                    if abs(prev_val) > 1e-6:
                        change_pct = abs(curr_val - prev_val) / abs(prev_val) * 100.0
                    else:
                        change_pct = abs(curr_val - prev_val) * 100.0
                    changes[f'change_{key}'] = float(change_pct)
                except (ValueError, TypeError):
                    pass
        return changes if changes else {}

    def _detect_convergence(self, rows_window):
        """
        Detect convergence vs drift patterns over recent epochs.

        Computes a per-epoch mean-change series across indicator params only,
        then fits a linear slope.  The slope direction distinguishes true
        convergence (slope < 0, params decelerating) from a plateau (slope ~0,
        params already small) and drift (slope > 0, params accelerating).

        Returns dict with:
          convergence_score     — 0-1, grounded at 3%/epoch = 0 (fully active)
          mean_param_change_pct — mean indicator-param % change in latest epoch
          std_param_change_pct  — std of individual param changes in latest epoch
          slope_pct_per_epoch   — linear trend of mean_change over the window
                                  (negative = converging, positive = drifting)
        """
        if len(rows_window) < 2:
            return None

        # Only look at indicator-parameter change keys (change_ma_period_*, etc.)
        indicator_change_keys = [
            k for k in rows_window[0].keys()
            if k.startswith('change_') and self._is_indicator_key(k[len('change_'):])
        ]
        if not indicator_change_keys:
            return None

        # Build per-epoch mean-change series  [epoch_i_mean, epoch_i+1_mean, ...]
        epoch_means = []
        for row in rows_window:
            vals = [row[k] for k in indicator_change_keys if k in row and row[k] is not None]
            if vals:
                epoch_means.append(np.mean(vals))

        if not epoch_means:
            return None

        current_mean = float(epoch_means[-1])
        current_std  = float(np.std(
            [rows_window[-1].get(k, 0.0) for k in indicator_change_keys
             if rows_window[-1].get(k) is not None]
        ))

        # Linear slope over the window (units: %/epoch)
        if len(epoch_means) >= 2:
            n = len(epoch_means)
            xs = np.arange(n, dtype=float)
            slope = float(np.polyfit(xs, epoch_means, 1)[0])
        else:
            slope = 0.0

        # Score: grounded so that 3% mean change = score 0 (fully active),
        # < 0.5% = score >= 0.83 (converged territory)
        # NaN must not read as "converged": Python's min(1.0, nan) returns 1.0.
        convergence_score = (float('nan') if not np.isfinite(current_mean)
                             else max(0.0, min(1.0, 1.0 - (current_mean / 3.0))))

        return {
            'convergence_score':     float(convergence_score),
            'mean_param_change_pct': current_mean,
            'std_param_change_pct':  current_std,
            'slope_pct_per_epoch':   slope,
        }

    def on_epoch_end(self, epoch, logs=None):
        """Enhanced to include immediate CSV writes and convergence tracking."""
        try:
            params = self.layer.get_learned_parameters()
        except Exception:
            params = {}
            try:
                getp = getattr(self.layer, 'get_learned_parameters', None)
                if callable(getp):
                    params = getp()
            except Exception:
                params = {}

        # Ensure all values are floats
        params = {k: (float(v) if v is not None else None) for k, v in (params or {}).items()}

        # Add epoch and timestamp
        params['epoch'] = int(epoch)
        import datetime
        params['timestamp'] = datetime.datetime.now().isoformat()

        # Calculate parameter changes if we have previous data
        changes = self._calculate_param_changes(params)
        if changes:
            params.update(changes)

        # Detect convergence if we have enough window
        if len(self.rows) >= self.convergence_window:
            window = self.rows[-(self.convergence_window-1):] + [params]
            convergence_info = self._detect_convergence(window)
            if convergence_info:
                params.update(convergence_info)

        # Add training metrics if available
        if logs:
            for k, v in logs.items():
                try:
                    params[f'log_{k}'] = float(v)
                except Exception:
                    params[f'log_{k}'] = v

        self.rows.append(params)
        self.prev_params = params.copy()

        # Write CSV immediately after each epoch (per-epoch tracking)
        if self.rows:
            try:
                pd.DataFrame(self.rows).to_csv(self.out_csv, index=False)
            except OSError as exc:  # e.g. the CSV is open in Excel: never abort training over telemetry
                warnings.warn(f"ParamsLogger: could not write {self.out_csv}: {exc}")

            # Log convergence status periodically (every 5 epochs)
            if epoch % 5 == 0 or epoch < 3:
                if 'convergence_score' in params:
                    conv_score  = params['convergence_score']
                    mean_change = params['mean_param_change_pct']
                    slope       = params.get('slope_pct_per_epoch', 0.0)
                    # Status derived from both magnitude AND trend direction
                    if mean_change < 0.5:
                        status = "converged"
                    elif slope < -0.3:
                        status = "converging"
                    elif slope > 0.3:
                        status = "drifting"
                    else:
                        status = "plateau"
                    print(f"Epoch {epoch}: Params {status} "
                          f"(score={conv_score:.3f}, mean={mean_change:.2f}%, "
                          f"slope={slope:+.2f}%/ep)")
                elif epoch < 3:
                    print(f"Epoch {epoch}: Indicator params logged to {self.out_csv}")

    def on_train_end(self, logs=None):
        """Final summary and stats."""
        if self.rows:
            print("\n=== Indicator Learning Summary ===")
            print(f"Total epochs tracked: {len(self.rows)}")
            print(f"Parameters logged per epoch: ~{len(self.rows[0])}")
            print(f"CSV saved to: {self.out_csv}")

            # Calculate final convergence metrics
            if len(self.rows) > 1:
                recent_window = self.rows[-min(10, len(self.rows)):]
                convergence_info = self._detect_convergence(recent_window)
                if convergence_info:
                    slope = convergence_info.get('slope_pct_per_epoch', 0.0)
                    print(f"Final Convergence Score: {convergence_info['convergence_score']:.3f}")
                    print(f"Final Mean Change:    {convergence_info['mean_param_change_pct']:.2f}%")
                    print(f"Final Std Change:     {convergence_info['std_param_change_pct']:.2f}%")
                    print(f"Final Slope:          {slope:+.2f}%/ep  "
                          f"({'decelerating' if slope < 0 else 'accelerating' if slope > 0 else 'flat'})")


# ============================================================================ B11: registry-built callbacks
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402


@dataclass
class TrainContext:
    """What a callback builder may need besides the config."""

    model: Any = None
    indicator_layer: Any = None
    run_dir: Optional[Path] = None
    run_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def path(self, name: str) -> str:
        """``name`` inside the run directory, or relative to the working directory (legacy)."""
        return str(Path(self.run_dir) / name) if self.run_dir else name


class LambdaScheduleCallback(callbacks.Callback):
    """Piecewise-constant loss-weight schedule: ``{"lambda_hd": {0: 0.0, 3: 0.1}, ...}``.

    At the start of each epoch every scheduled weight takes the value of the latest key <= epoch.
    The weights are tf.Variables, so the change applies without retracing.
    """

    def __init__(self, schedule: Dict[str, Dict[Any, float]]):
        super().__init__()
        self.schedule = {name: {int(k): float(v) for k, v in steps.items()}
                         for name, steps in (schedule or {}).items()}

    def on_epoch_begin(self, epoch, logs=None):
        for name, steps in self.schedule.items():
            due = [e for e in steps if e <= epoch]
            if due:
                setattr(self.model, name, steps[max(due)])


class MetricThresholdStop(callbacks.Callback):
    """Stop training when ``monitor`` crosses ``threshold`` (or is not finite) at an epoch end."""

    def __init__(self, monitor="loss", threshold=float("inf"), mode="max"):
        super().__init__()
        self.monitor, self.threshold, self.mode = monitor, float(threshold), mode
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get(self.monitor)
        if v is None:
            return
        crossed = (not np.isfinite(v)) or (v > self.threshold if self.mode == "max" else v < self.threshold)
        if crossed:
            self.stopped_epoch = epoch
            self.model.stop_training = True


# ---- builders registered in neural_trade.registries.callbacks: f(config, context) ----------
def build_csv_logger(config, context):
    """Keras CSVLogger to training_log.csv."""
    return callbacks.CSVLogger(context.path("training_log.csv"), append=True)


def build_early_stopping(config, context):
    """EarlyStopping on val_loss, patience EARLY, restoring the best weights."""
    return callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY, restore_best_weights=True)


def build_model_checkpoint(config, context):
    """Best-on-validation weights to MODEL_PATH."""
    return callbacks.ModelCheckpoint(config.MODEL_PATH, save_best_only=True, monitor="val_loss",
                                     save_weights_only=True)


def build_mcc_early_stopping(config, context):
    """EarlyStopping on val_dir_mcc_h1 (max); opt-in, without weight restore."""
    return callbacks.EarlyStopping(monitor="val_dir_mcc_h1", mode="max", patience=config.EARLY,
                                   restore_best_weights=False)


def build_reduce_lr_on_plateau(config, context):
    """Halve the learning rate after PATIENCE epochs without val_loss improvement."""
    return callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config.PATIENCE)


def build_tqdm_progress(config, context):
    """Console progress bars."""
    return TqdmCallback()


def build_params_logger(config, context):
    """Learned indicator periods per epoch to indicator_params_history.csv (legacy CSV)."""
    return ParamsLogger(layer=context.indicator_layer, out_csv=context.path("indicator_params_history.csv"))


def build_jsonl_epoch_logger(config, context):
    """Append-only metrics.jsonl + status.json (never raises)."""
    from neural_trade.telemetry.epoch_logger import JsonlEpochLogger

    return JsonlEpochLogger(context.run_dir or ".", context.indicator_layer, context.run_id)


def build_lambda_schedule(config, context):
    """Config.LOSS_WEIGHT_SCHEDULE applied at each epoch start."""
    return LambdaScheduleCallback(getattr(config, "LOSS_WEIGHT_SCHEDULE", None) or {})


def build_metric_threshold(config, context):
    """Stop on a non-finite or out-of-range metric (settings in context.extra["metric_threshold"])."""
    return MetricThresholdStop(**dict(context.extra.get("metric_threshold", {})))


def build_tensorboard(config, context):
    """TensorBoard scalars under <run>/tb."""
    return callbacks.TensorBoard(log_dir=context.path("tb"), write_graph=False, profile_batch=0)


def build_interactive_plot(config, context):
    """The notebook's live Plotly dashboard (widgets passed in context.extra["interactive_plot"])."""
    from neural_trade.visualization.plotly_training import make_interactive_plot_callback

    return make_interactive_plot_callback(config=config, **dict(context.extra.get("interactive_plot", {})))

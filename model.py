csv="binance_btcusdt_1min_ccxt.csv"

import os
import sys
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")  # must be set before TensorFlow is imported

# Console encoding guard. This module's progress output contains non-ASCII glyphs
# (lambda, T-perp, arrows, ~ - 137 characters over 25 distinct code points). On a Windows
# console whose code page is not UTF-8 (cp1251 on the development box) `print` raises
# UnicodeEncodeError, and because one such print sits inside the pre-training lambda
# calibration pass the whole pass aborted and fell back to the configured lambdas - a
# numerical feature silently disabled by a log line. Replacing unencodable characters
# costs a "?" in the log instead of an aborted training stage.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):  # not a TextIOWrapper, or already detached
        pass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, losses, initializers, regularizers
import math
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error, f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from losses import Losses
import losses as _losses
import neural_trade.utils.math as mh
from neural_trade.metrics.tf_direction import direction_labels_tf
from neural_trade.core.outputs import PredictiveOutputs
from neural_trade.registries.models import Models
from neural_trade.registries.metrics import Metrics
from neural_trade.metrics.evaluate import _compute_all_horizon_metrics  # noqa: F401  (moved in B8)
from neural_trade.metrics.tf_direction import (DirectionAccumulator, PITAccumulator, STEP_MEAN_KEYS,
                                               TRAIN_ONLY_MEAN_KEYS, direction_counts, direction_stats,
                                               direction_metrics_from_stats)
from neural_trade.models.layers import (EnergyGate, LearnableIndicators,  # noqa: F401  (moved in B6)
                                        PositionalEncodingLayer, VacuumSaturationNoise)
# Loss functions (custom) are implemented centrally in `losses.py` to
# maintain a single authoritative source and avoid duplication.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import time
from tqdm import tqdm
import plotly.io as pio
#pio.renderers.default = 'colab'

try:
    # Optional local utilities (kept lightweight). If missing, fall back to sklearn MAPE only.
    from neural_trade.metrics.numpy_metrics import (safe_mape, smape, wape, reconstruct_prices,
                                                    mask_by_min_abs_y, pit_uniformity,
                                                    compute_direction_labels_np)
except Exception:
    safe_mape = None
    smape = None
    wape = None
    reconstruct_prices = None
    mask_by_min_abs_y = None
    compute_direction_labels_np = None

try:
    from calibration import CalibrationPipeline as _CalibrationPipeline
except Exception:  # calibration package missing — non-fatal
    _CalibrationPipeline = None

# Config moved to neural_trade.core.config (Phase B4): a real typed dataclass with every
# previous setting and default, validation, override() and YAML round-trip.
from neural_trade.core.config import Config  # noqa: E402


@dataclass(frozen=True)
class FoldIndices:
    """Index blocks of one purged chronological fold (sequence indices, not bar indices)."""
    fold: int
    gap: int
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray
    test: np.ndarray


def make_purged_splits(n_seq, *, lookback, horizon_steps, window_step=1, n_folds=5,
                       val_fraction=0.066, cal_fraction=0.066, gap=None):
    """Chronological train | gap | val | gap | cal | gap | test blocks per TimeSeriesSplit fold.

    A training sequence anchored at bar i carries labels up to bar i + max(H) - 1; a later
    sequence anchored at bar j reads inputs from bar j - LOOKBACK. They share a bar iff
    j - i <= LOOKBACK + max(H) - 1, so a gap of LOOKBACK + max(H) sequences (79 + 1 for the
    default config) guarantees no bar is both a training label and an evaluation input.
    The gap is applied between every adjacent pair of blocks. With WINDOW_STEP > 1 the gap
    is expressed in sequences (ceil of the bar gap / step).

    Returns folds in chronological order; folds whose train block would be empty are omitted.
    """
    horizon_steps = [int(h) for h in horizon_steps]
    bar_gap = int(gap) if gap is not None else int(lookback) + int(max(horizon_steps))
    seq_gap = int(math.ceil(bar_gap / max(1, int(window_step))))
    val_len = max(1, int(round(val_fraction * n_seq)))
    cal_len = max(1, int(round(cal_fraction * n_seq)))

    folds = []
    tscv = TimeSeriesSplit(n_splits=int(n_folds))
    for k, (_, test_idx) in enumerate(tscv.split(np.arange(n_seq)), start=1):
        t0, t1 = int(test_idx[0]), int(test_idx[-1]) + 1
        cal1 = t0 - seq_gap
        cal0 = cal1 - cal_len
        val1 = cal0 - seq_gap
        val0 = val1 - val_len
        train1 = val0 - seq_gap
        if train1 <= 0:
            continue
        folds.append(FoldIndices(
            fold=k, gap=seq_gap,
            train=np.arange(0, train1), val=np.arange(val0, val1),
            cal=np.arange(cal0, cal1), test=np.arange(t0, t1),
        ))
    if not folds:
        raise ValueError(
            f"Not enough sequences ({n_seq}) for a purged split with gap={seq_gap}, "
            f"val_len={val_len}, cal_len={cal_len}, n_folds={n_folds}: add data, raise "
            f"MAX_SEQUENCE_COUNT, or lower VAL_FRACTION / CAL_FRACTION."
        )
    return folds


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.target_scaler = None
        self.input_scaler = None

    def clean_numeric(self, series):
        return series.astype(str).str.replace(r'[\$,]', '', regex=True).replace('', np.nan).astype(float)

    def load_and_prepare_data(self, read_csv_kwargs: Optional[dict] = None):
        """Load and prep minute-level Bitcoin data with optional resampling.

        `read_csv_kwargs` is passed through to `pd.read_csv`.
        This makes ingestion robust in notebooks where the CSV may require
        non-default parsing settings.
        """
        read_csv_kwargs = dict(read_csv_kwargs or {})
        df = pd.read_csv(self.config.CSV_PATH, **read_csv_kwargs)

        # Accept the repository dataset's `datetime` name as well as the
        # legacy `timestamp` name used by older exports.
        time_column = next(
            (name for name in ('timestamp', 'datetime') if name in df.columns),
            None,
        )
        if time_column is None:
            raise ValueError(
                "Market data must contain a 'timestamp' or 'datetime' column"
            )
        if time_column != 'timestamp':
            df = df.rename(columns={time_column: 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).copy()
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')

        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if self.config.RESAMPLE_MINUTES:
            df = (
                df.set_index('timestamp')
                  .resample(f"{self.config.RESAMPLE_MINUTES}min")
                  .agg({
                      'Open': 'first',
                      'High': 'max',
                      'Low': 'min',
                      'Close': 'last',
                      'Volume': 'sum'
                  })
                  .dropna(subset=['Close'])
                  .reset_index()
            )
            df['Date'] = df['timestamp']
        else:
            df['Date'] = df['timestamp']

        df = df.dropna(subset=['Close']).reset_index(drop=True)
        print(f"Dataset length after cleaning: {len(df)}")
        print("Date range after cleaning:", df['Date'].min(), "to", df['Date'].max())

        if len(df) < self.config.LOOKBACK + 2:
            raise ValueError(f"Not enough rows ({len(df)}) for lookback={self.config.LOOKBACK}")

        return df, df['Close'].values.astype('float32')

    def compute_extended_trend_features(self, close_values, index, periods):
        """Compute extended trend features as ABSOLUTE DELTAS (not percent-changes).

        CRITICAL: Extended trends must be in the same units as prediction targets (absolute deltas in $).
        Note: `periods` are expressed in numbers of resampled bars (i.e., multiples of `config.RESAMPLE_MINUTES` minutes).
        This ensures semantic consistency in the trend loss function.
        
        Previously computed as percent-changes (returns), which caused:
        - Semantic mismatch with targets (deltas in dollars)
        - Apples-to-oranges comparison in trend_loss
        - Weak/confused supervision signal
        
        Now computes: delta[t, t-period] = price[t] - price[t-period] (in dollars)
        This matches the target semantics exactly.

        Ensures that any period-based indexing uses integer offsets and
        guards against negative indices or out-of-bounds access.
        """
        features = []
        # Ensure we have a 1-D numpy array for safe integer indexing
        close_values = np.asarray(close_values).reshape(-1)
        n = close_values.shape[0]
        idx = int(index)
        # Clamp idx to valid range just in case
        if idx < 0:
            idx = 0
        elif idx >= n:
            idx = n - 1

        current_price = close_values[int(idx)]
        for period in periods:
            # Convert possible float periods (e.g., 60/4 -> 15.0) to int steps
            p = int(period)
            if p <= 0:
                features.append(0.0)
                continue

            ref_idx = int(idx - p)
            if ref_idx >= 0:
                past_price = close_values[ref_idx]
                # FIXED: Compute absolute delta (in dollars), not percent-change
                # This matches the semantics of targets: delta = future_price - current_price
                delta = current_price - past_price
                features.append(float(delta))
            else:
                features.append(0.0)

        return np.array(features, dtype='float32')

    def make_sequences_with_extended_trends(self, close_array, lookback):
        X, y, last_close, extended_trends = [], [], [], []
        # Ensure start index is an integer even if periods are provided as floats
        max_extended_period = int(max(self.config.EXTENDED_TREND_PERIODS))
        start_idx = int(max(lookback, max_extended_period))
        step = int(max(1, getattr(self.config, 'WINDOW_STEP', 1)))

        horizon_steps = [int(h) for h in getattr(self.config, 'HORIZON_STEPS', [1, 5, 15])]
        if not horizon_steps:
            raise ValueError("Config.HORIZON_STEPS must be a non-empty list of positive integers")
        if any(h <= 0 for h in horizon_steps):
            raise ValueError(f"Invalid horizon steps: {horizon_steps}")
        max_h = int(max(horizon_steps))

        # Ensure targets are within bounds for all horizons
        end_idx = int(len(close_array) - (max_h - 1))

        for i in range(start_idx, end_idx, step):
            window = close_array[i-lookback:i]
            # Targets (Option A): predict DELTAS relative to last_close at time t.
            #   delta_h = close[t+h] - last_close[t]
            # This is more stationary than absolute price and aligns with trading semantics.
            lc = float(close_array[i - 1])
            target = np.array([float(close_array[i + (h - 1)]) - lc for h in horizon_steps], dtype='float32')
            ext_features = self.compute_extended_trend_features(close_array, int(i-1), self.config.EXTENDED_TREND_PERIODS)
            X.append(window)
            y.append(target)
            last_close.append(close_array[i-1])
            extended_trends.append(ext_features)

        return (
            np.array(X, dtype='float32'),
            np.array(y, dtype='float32'),
            np.array(last_close, dtype='float32'),
            np.array(extended_trends, dtype='float32')
        )

    def plot_splits(self, df, start_idx, tscv, X_seq_len):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df['Date'], df['Close'], label='BTC Close Price', alpha=0.8)
        split_boundaries = [0]
        for train_idx, test_idx in tscv.split(np.arange(X_seq_len)):
            split_boundaries.append(test_idx[0])
        split_boundaries.append(X_seq_len)
        colors = ['#fff8b0', '#d2f8d2']
        labels = ['Train', 'Test']
        used = set()
        for i in range(len(split_boundaries)-1):
            s = start_idx + split_boundaries[i]
            e = start_idx + split_boundaries[i+1]
            color = colors[i % 2]
            label = labels[i % 2] if labels[i % 2] not in used else ""
            used.add(labels[i % 2])
            ax.axvspan(df['Date'].iloc[s], df['Date'].iloc[e-1], color=color, alpha=0.2, label=label)
        ax.set_title('BTC Price with Walk-Forward Validation (Train=Yellow, Test=Green)')
        ax.set_xlabel('Date')
        ax.set_ylabel('BTC Price (USD)')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def prepare_datasets(self, df, close_values):
        X_seq, y_seq, last_close_seq, extended_trends = self.make_sequences_with_extended_trends(
            close_values, self.config.LOOKBACK
        )
        print(f"Sequences with extended trends: {X_seq.shape}, {y_seq.shape}, Extended: {extended_trends.shape}")

        max_sequences = getattr(self.config, 'MAX_SEQUENCE_COUNT', None)
        if max_sequences and X_seq.shape[0] > max_sequences:
            original_count = X_seq.shape[0]
            take_from = original_count - max_sequences
            X_seq = X_seq[take_from:]
            y_seq = y_seq[take_from:]
            last_close_seq = last_close_seq[take_from:]
            extended_trends = extended_trends[take_from:]
            print(f"[OK] Limited sequence set from {original_count} to {max_sequences} (most recent window)")

        print("[INFO] Dataset Statistics:")
        print(f"   Total sequences: {X_seq.shape[0]}")

        # Four-way PURGED chronological split: train | gap | val | gap | cal | gap | test.
        #   train -> gradients and the target scaler;  val -> early stopping / checkpoint / LR;
        #   cal   -> post-hoc calibration (temperature, conformal);  test -> reported once.
        # Previously the last TimeSeriesSplit fold was BOTH validation and test with no gap:
        # 79 "validation" windows contained bars that were training labels, model selection
        # happened on the test set, and calibration was fit on the test set too.
        fold = make_purged_splits(
            X_seq.shape[0],
            lookback=self.config.LOOKBACK,
            horizon_steps=self.config.HORIZON_STEPS,
            window_step=int(max(1, getattr(self.config, 'WINDOW_STEP', 1))),
            n_folds=int(getattr(self.config, 'N_FOLDS', 5)),
            val_fraction=float(getattr(self.config, 'VAL_FRACTION', 0.066)),
            cal_fraction=float(getattr(self.config, 'CAL_FRACTION', 0.066)),
        )[-1]
        self.fold = fold

        def _take(idx):
            return X_seq[idx], y_seq[idx], last_close_seq[idx], extended_trends[idx]

        X_train_seq, y_train, last_close_train, extended_trends_train = _take(fold.train)
        X_val_seq, y_val, last_close_val, extended_trends_val = _take(fold.val)
        X_cal_seq, y_cal, last_close_cal, extended_trends_cal = _take(fold.cal)
        X_test_seq, y_test, last_close_test, extended_trends_test = _take(fold.test)

        train_batches = math.ceil(X_train_seq.shape[0] / self.config.BATCH_SIZE)
        test_batches = math.ceil(X_test_seq.shape[0] / self.config.BATCH_SIZE)
        print(f"   Train sequences: {X_train_seq.shape[0]} (batches/epoch: {train_batches})")
        print(f"   Val sequences:   {X_val_seq.shape[0]}   Cal sequences: {X_cal_seq.shape[0]}   "
              f"(purge gap: {fold.gap} sequences, fold {fold.fold})")
        print(f"   Test sequences:  {X_test_seq.shape[0]} (batches: {test_batches})")

        # Targets are multi-horizon [N, 3]: one scaler, fit on TRAIN only, shared across horizons.
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
        y_cal_scaled = target_scaler.transform(y_cal.reshape(-1, 1)).reshape(y_cal.shape)
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        # Window-relative input in TARGET-scaler units: (x - last_close) / target_scale.
        # The previous per-lag-position StandardScaler z-scored the absolute price LEVEL over
        # the whole training span, so within one hour the 60 inputs barely differed and the
        # model's dominant signal was "where is BTC vs. its multi-week mean" - noise w.r.t. a
        # delta target, and a second reason it learned to repeat the last close. This puts the
        # window, last_close, the extended-trend features, the price heads and sigma in one
        # unit system. Parameter-free: there is no input scaler to persist.
        input_scale = float(target_scaler.scale_[0]) if float(target_scaler.scale_[0]) > 0 else 1.0

        def _normalise(X, lc):
            return ((X - lc[:, None]) / input_scale).astype('float32')

        X_train_seq_scaled = _normalise(X_train_seq, last_close_train)
        X_test_seq_scaled = _normalise(X_test_seq, last_close_test)
        input_scaler = None

        joblib.dump(target_scaler, self.config.SCALER_PATH)

        # Keep references for programmatic use without changing the return signature.
        self.target_scaler = target_scaler
        self.input_scaler = input_scaler
        self.input_scale = input_scale
        # Validation and calibration blocks, consumed by train_and_evaluate.
        self.val_block = dict(X=_normalise(X_val_seq, last_close_val), y_scaled=y_val_scaled, y_raw=y_val,
                              last_close=last_close_val, extended_trends=extended_trends_val)
        self.cal_block = dict(X=_normalise(X_cal_seq, last_close_cal), y_scaled=y_cal_scaled, y_raw=y_cal,
                              last_close=last_close_cal, extended_trends=extended_trends_cal)

        return (X_train_seq_scaled, y_train_scaled, last_close_train, extended_trends_train,
                X_test_seq_scaled, y_test_scaled, last_close_test, extended_trends_test,
                y_train, y_test, target_scaler)


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


def _apply_config_overrides(config: 'Config', overrides: Optional[dict]) -> 'Config':
    """Validated update: unknown names raise InvalidConfigurationError with suggestions
    (the old setattr loop silently created misspelled attributes that nothing read)."""
    if not overrides:
        return config
    return config.override(**dict(overrides))


# _compute_all_horizon_metrics moved to neural_trade.metrics.evaluate (B8); re-imported above.


def make_interactive_plot_callback(
    *,
    config: 'Config',
    loss_output,
    metrics_output,
    progress_widget,
    total_epochs: int,
    batch_metrics_output=None,
    primary_horizon: str = "h1",
    prefer_gauss: bool = True,
    should_pause=None,
    should_stop=None,
    batch_update_interval: int = 1,
    epoch_info_widget=None,
):
    """Notebook-friendly interactive Plotly callback.

    This is a rewrite/encapsulation of the notebook's Cell 3 callback so notebooks can
    depend on `model.py` as the single source of truth.
    """
    from IPython.display import clear_output, display
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import ipywidgets as widgets

    import time

    def _bool_call(maybe_callable) -> bool:
        try:
            return bool(maybe_callable()) if callable(maybe_callable) else bool(maybe_callable)
        except Exception:
            return False

    class _InteractivePlotCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = {}
            self.epoch_count = 0
            self.batch_history = {}
            self.total_batches = None
            self.batch_count = 0
            self.batch_update_interval = max(1, int(batch_update_interval))

        def on_epoch_begin(self, epoch, logs=None):
            self.batch_history.clear()
            self.batch_count = 0
            self.total_batches = self.params.get('steps') if self.params is not None else None
            if batch_metrics_output is not None:
                with batch_metrics_output:
                    clear_output(wait=True)

        def on_train_batch_end(self, batch, logs=None):
            logs = logs or {}
            logs = add_plot_aliases(logs, primary_horizon=primary_horizon, prefer_gauss=prefer_gauss)
            batch_idx = (batch or 0) + 1
            self.batch_history.setdefault('batch', []).append(batch_idx)
            for key, value in logs.items():
                if value is None:
                    continue
                try:
                    self.batch_history.setdefault(key, []).append(float(value))
                except Exception:
                    pass

            # Batch plot (loss + a couple key metrics)
            if batch_metrics_output is not None and self.batch_count % self.batch_update_interval == 0:
                with batch_metrics_output:
                    clear_output(wait=True)
                    batches = self.batch_history.get('batch', [])
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        subplot_titles=("Batch Loss", "Batch Direction Metrics"),
                                        vertical_spacing=0.12)
                    if 'loss' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['loss'], mode='lines', name='loss', line=dict(color='#64B5F6')), row=1, col=1)
                    if 'dir_acc' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['dir_acc'], mode='lines', name='dir_acc', line=dict(color='#81C784')), row=2, col=1)
                    if 'f1' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['f1'], mode='lines', name='f1', line=dict(color='#FFB74D')), row=2, col=1)
                    
                    # Add 50% dotted lines to metrics subplot (row 2)
                    if batches:
                        fig.add_hline(y=0.5, line_dash="dot", line_color="#888888", row=2, col=1, annotation_text="50%", annotation_position="right")
                    
                    # Calculate axis range with padding to prevent data touching borders
                    if batches:
                        x_min, x_max = min(batches), max(batches)
                        x_padding = max(1, (x_max - x_min) * 0.03)  # 3% padding
                    else:
                        x_min, x_max, x_padding = 0, 1, 0.1
                    
                    # Dark theme styling with proper margins
                    fig.update_layout(
                        height=450,
                        showlegend=True,
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#0d0d0d',
                        font=dict(color='#e0e0e0'),
                        margin=dict(l=60, r=40, t=40, b=40),
                        xaxis_showgrid=True,
                        xaxis_gridwidth=1,
                        xaxis_gridcolor='#333333',
                        yaxis_showgrid=True,
                        yaxis_gridwidth=1,
                        yaxis_gridcolor='#333333',
                        xaxis2_showgrid=True,
                        xaxis2_gridwidth=1,
                        xaxis2_gridcolor='#333333',
                        yaxis2_showgrid=True,
                        yaxis2_gridwidth=1,
                        yaxis2_gridcolor='#333333',
                    )
                    
                    # Set x-axis range with padding (shared x-axis, only xaxis2 controls both)
                    fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
                    
                    # Set y-axis range for metrics subplot with padding
                    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
                    
                    # Update axes styling
                    fig.update_xaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)
                    fig.update_yaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)
                    
                    # Update subplot titles color
                    for annotation in fig['layout']['annotations']:
                        annotation['font'] = dict(color='#e0e0e0', size=12)
                    
                    display(fig)

            self.batch_count += 1

        def on_epoch_end(self, epoch, logs=None):
            # Optional notebook controls.
            # Keep this inside the callback so the notebook can remain a thin UI wrapper.
            if _bool_call(should_stop):
                try:
                    self.model.stop_training = True
                except Exception:
                    pass
                return

            # Cooperative pause loop (safe no-op if not provided)
            while _bool_call(should_pause) and not _bool_call(should_stop):
                time.sleep(0.1)
            if _bool_call(should_stop):
                try:
                    self.model.stop_training = True
                except Exception:
                    pass
                return

            logs = add_plot_aliases(logs or {}, primary_horizon=primary_horizon, prefer_gauss=prefer_gauss)
            self.epoch_count += 1
            for k, v in (logs or {}).items():
                if v is None:
                    continue
                try:
                    self.history.setdefault(k, []).append(float(v))
                except Exception:
                    pass

            try:
                progress_widget.value = min(total_epochs, epoch + 1)
                progress_widget.description = f'Epoch ({epoch + 1}/{total_epochs}):'
            except Exception:
                pass

            # Update epoch info widget if provided
            try:
                if epoch_info_widget is not None:
                    # Compute patience
                    patience_used_info = 0
                    if 'val_loss' in self.history and len(self.history['val_loss']) > 1:
                        best_idx = np.argmin(self.history['val_loss'])
                        patience_used_info = len(self.history['val_loss']) - 1 - best_idx
                    patience_max_info = getattr(config, 'PATIENCE', total_epochs)
                    
                    # Get current metrics
                    curr_loss = logs.get('loss', 0)
                    curr_val_loss = logs.get('val_loss', 0)
                    curr_dir_acc = logs.get('val_dir_acc_avg', 0)
                    curr_f1 = logs.get('val_f1_avg', 0)
                    
                    # Determine status color
                    if patience_used_info > patience_max_info * 0.8:
                        patience_color = "#EF5350"  # Red - close to stopping
                    elif patience_used_info > patience_max_info * 0.5:
                        patience_color = "#FFB74D"  # Orange - warning
                    else:
                        patience_color = "#81C784"  # Green - good
                    
                    epoch_info_widget.value = f"""
                    <div style="font-family: monospace; color: #e0e0e0; background-color: #1a1a1a; 
                                padding: 12px 20px; border-radius: 5px; text-align: center; 
                                border: 1px solid #333; margin-bottom: 10px;">
                        <span style="font-size: 18px; font-weight: bold; color: #64B5F6;">
                            🔄 Epoch {self.epoch_count}/{total_epochs}
                        </span>
                        <span style="color: {patience_color}; margin-left: 20px;">
                            ⏳ Patience: {patience_used_info}/{patience_max_info}
                        </span>
                        <span style="color: #64B5F6; margin-left: 20px;">
                            📉 Loss: {curr_loss:.4f}
                        </span>
                        <span style="color: #42A5F5; margin-left: 15px;">
                            Val: {curr_val_loss:.4f}
                        </span>
                        <span style="color: #81C784; margin-left: 20px;">
                            🎯 Acc: {curr_dir_acc:.1%}
                        </span>
                        <span style="color: #FFB74D; margin-left: 15px;">
                            F1: {curr_f1:.3f}
                        </span>
                    </div>
                    """
            except Exception:
                pass

            # Epoch plots
            with loss_output:
                clear_output(wait=True)
                
                # Compute patience estimation (epochs since best val_loss)
                patience_used = 0
                if 'val_loss' in self.history and len(self.history['val_loss']) > 1:
                    best_val_loss_idx = np.argmin(self.history['val_loss'])
                    patience_used = len(self.history['val_loss']) - 1 - best_val_loss_idx
                patience_max = getattr(config, 'PATIENCE', total_epochs)
                
                # Compute key metrics for header
                current_loss = self.history.get('loss', [0])[-1] if 'loss' in self.history else 0
                current_val_loss = self.history.get('val_loss', [0])[-1] if 'val_loss' in self.history else 0
                current_dir_acc = self.history.get('val_dir_acc_avg', [0])[-1] if 'val_dir_acc_avg' in self.history else 0
                
                # Build title with epoch progress and metrics
                title_text = (f"<b>Epoch {self.epoch_count}/{total_epochs}</b> │ "
                             f"Patience: {patience_used}/{patience_max} │ "
                             f"Loss: {current_loss:.4f} │ Val Loss: {current_val_loss:.4f} │ "
                             f"Val Dir Acc: {current_dir_acc:.1%}")
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=("Epoch Loss", "Epoch Direction Metrics"),
                                    vertical_spacing=0.12)
                epochs = list(range(1, self.epoch_count + 1))
                if 'loss' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['loss'], mode='lines+markers', name='loss', line=dict(color='#64B5F6')), row=1, col=1)
                if 'val_loss' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['val_loss'], mode='lines+markers', name='val_loss', line=dict(color='#42A5F5')), row=1, col=1)
                if 'dir_acc_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['dir_acc_avg'], mode='lines+markers', name='dir_acc_avg', line=dict(color='#81C784')), row=2, col=1)
                if 'val_dir_acc_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['val_dir_acc_avg'], mode='lines+markers', name='val_dir_acc_avg', line=dict(color='#66BB6A')), row=2, col=1)
                # Balanced Accuracy: (Sensitivity + Specificity) / 2
                # Range [0, 1], 50% = random, class-imbalance robust (unlike accuracy)
                if 'bal_acc_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['bal_acc_avg'], mode='lines+markers', name='bal_acc_avg', line=dict(color='#FFB74D')), row=2, col=1)
                if 'val_bal_acc_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['val_bal_acc_avg'], mode='lines+markers', name='val_bal_acc_avg', line=dict(color='#FFA726')), row=2, col=1)
                
                # Add 50% dotted lines to metrics subplot (row 2)
                if epochs:
                    fig.add_hline(y=0.5, line_dash="dot", line_color="#888888", row=2, col=1, annotation_text="50%", annotation_position="right")
                
                # Calculate axis range with padding
                if epochs:
                    x_min, x_max = min(epochs), max(epochs)
                    x_padding = max(0.5, (x_max - x_min) * 0.05)  # 5% padding
                else:
                    x_min, x_max, x_padding = 0, 1, 0.1
                
                # Dark theme styling with proper margins
                fig.update_layout(
                    title=dict(text=title_text, font=dict(size=14, color='#e0e0e0'), x=0.5, xanchor='center'),
                    height=560,
                    showlegend=True,
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#0d0d0d',
                    font=dict(color='#e0e0e0'),
                    margin=dict(l=60, r=40, t=70, b=40),
                    xaxis_showgrid=True,
                    xaxis_gridwidth=1,
                    xaxis_gridcolor='#333333',
                    yaxis_showgrid=True,
                    yaxis_gridwidth=1,
                    yaxis_gridcolor='#333333',
                    xaxis2_showgrid=True,
                    xaxis2_gridwidth=1,
                    xaxis2_gridcolor='#333333',
                    yaxis2_showgrid=True,
                    yaxis2_gridwidth=1,
                    yaxis2_gridcolor='#333333',
                )
                
                # Set x-axis range with padding
                fig.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
                
                # Set y-axis range for metrics subplot with padding
                fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
                
                # Update axes styling
                fig.update_xaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)
                fig.update_yaxes(showline=True, linewidth=1, linecolor='#444444', mirror=False, zeroline=False)
                
                # Update subplot titles color
                for annotation in fig['layout']['annotations']:
                    annotation['font'] = dict(color='#e0e0e0', size=12)
                
                display(fig)

            with metrics_output:
                clear_output(wait=True)
                
                # Compute convergence metrics
                loss_hist = self.history.get('loss', [])
                val_loss_hist = self.history.get('val_loss', [])
                
                # Convergence: rate of loss decrease (last 3 epochs)
                convergence_rate = 0.0
                if len(loss_hist) >= 3:
                    recent_losses = loss_hist[-3:]
                    convergence_rate = (recent_losses[0] - recent_losses[-1]) / (len(recent_losses) - 1) if len(recent_losses) > 1 else 0
                
                # Stability: std of recent validation losses
                stability = 0.0
                if len(val_loss_hist) >= 3:
                    stability = 1.0 - min(1.0, np.std(val_loss_hist[-5:]) * 10)  # Higher = more stable
                
                # Generalization gap: difference between train and val loss
                gen_gap = 0.0
                if loss_hist and val_loss_hist:
                    gen_gap = val_loss_hist[-1] - loss_hist[-1]
                
                # Coherence: How well train/val losses track each other (moving in same direction)
                # High coherence (>0.8) = model generalizes well, losses move together
                # Low/negative coherence = overfitting (train improves, val doesn't) or noise
                # Note: This measures train/val alignment, not cross-horizon consistency
                coherence = 0.0
                if len(loss_hist) >= 3 and len(val_loss_hist) >= 3:
                    try:
                        # Use direction agreement instead of correlation for robustness
                        # Direction: did loss increase or decrease between epochs?
                        train_diffs = np.diff(loss_hist[-10:])
                        val_diffs = np.diff(val_loss_hist[-10:])
                        if len(train_diffs) > 0 and len(val_diffs) > 0:
                            # Direction agreement: both increasing or both decreasing
                            train_dirs = np.sign(train_diffs)
                            val_dirs = np.sign(val_diffs)
                            agreement = np.mean(train_dirs == val_dirs)
                            coherence = agreement  # Range [0, 1], 1 = perfect agreement
                        else:
                            coherence = 0.5  # Neutral if not enough data
                    except Exception:
                        coherence = 0.0
                
                # Learning progress: improvement from initial
                progress = 0.0
                if len(val_loss_hist) >= 2:
                    progress = (val_loss_hist[0] - val_loss_hist[-1]) / val_loss_hist[0] if val_loss_hist[0] > 0 else 0
                
                # Build HTML output for dark theme visibility
                from IPython.display import HTML
                
                conv_status = "↓ converging" if convergence_rate > 0.001 else ("→ plateau" if abs(convergence_rate) < 0.001 else "↑ diverging")
                conv_color = "#81C784" if convergence_rate > 0.001 else ("#FFB74D" if abs(convergence_rate) < 0.001 else "#EF5350")
                
                stab_status = "stable" if stability > 0.8 else ("moderate" if stability > 0.5 else "unstable")
                stab_color = "#81C784" if stability > 0.8 else ("#FFB74D" if stability > 0.5 else "#EF5350")
                
                gap_status = "good" if gen_gap < 0.1 else ("warning" if gen_gap < 0.3 else "overfitting")
                gap_color = "#81C784" if gen_gap < 0.1 else ("#FFB74D" if gen_gap < 0.3 else "#EF5350")
                
                coh_status = "aligned" if coherence > 0.8 else ("moderate" if coherence > 0.5 else "misaligned")
                coh_color = "#81C784" if coherence > 0.8 else ("#FFB74D" if coherence > 0.5 else "#EF5350")
                
                html_content = f"""
                <div style="font-family: monospace; color: #e0e0e0; background-color: #0d0d0d; padding: 15px; border-radius: 5px;">
                    <div style="text-align: center; font-size: 16px; font-weight: bold; border-bottom: 2px solid #444; padding-bottom: 10px; margin-bottom: 15px;">
                        📊 EPOCH METRICS DASHBOARD
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <div style="color: #64B5F6; font-weight: bold; margin-bottom: 8px;">📉 LOSSES</div>
                        <div style="margin-left: 15px;">
                            <span style="display: inline-block; width: 180px;">loss:</span> <span style="color: #64B5F6;">{logs.get('loss', 0):.6f}</span><br>
                            <span style="display: inline-block; width: 180px;">val_loss:</span> <span style="color: #42A5F5;">{logs.get('val_loss', 0):.6f}</span><br>
                            {'<span style="display: inline-block; width: 180px;">point_loss:</span> <span style="color: #90CAF9;">' + f"{logs.get('point_loss', 0):.6f}" + '</span><br>' if 'point_loss' in logs else ''}
                            {'<span style="display: inline-block; width: 180px;">dir_loss:</span> <span style="color: #90CAF9;">' + f"{logs.get('dir_loss', 0):.6f}" + '</span><br>' if 'dir_loss' in logs else ''}
                            {'<span style="display: inline-block; width: 180px;">nll_loss:</span> <span style="color: #90CAF9;">' + f"{logs.get('nll_loss', 0):.6f}" + '</span><br>' if 'nll_loss' in logs else ''}
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <div style="color: #81C784; font-weight: bold; margin-bottom: 8px;">🎯 DIRECTION METRICS</div>
                        <div style="margin-left: 15px;">
                            <span style="display: inline-block; width: 180px;">dir_acc_avg:</span> <span style="color: #81C784;">{logs.get('dir_acc_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">val_dir_acc_avg:</span> <span style="color: #66BB6A;">{logs.get('val_dir_acc_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">bal_acc_avg:</span> <span style="color: #FFB74D;">{logs.get('bal_acc_avg', 0):.4f}</span> <span style="color: #888;">(50%=random)</span><br>
                            <span style="display: inline-block; width: 180px;">val_bal_acc_avg:</span> <span style="color: #FFA726;">{logs.get('val_bal_acc_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">brier_avg:</span> <span style="color: #CE93D8;">{logs.get('brier_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">ece_avg:</span> <span style="color: #BA68C8;">{logs.get('ece_avg', 0):.4f}</span><br>
                        </div>
                    </div>
                    
                    {_qbox_dashboard_html(logs)}
                    
                    <div style="margin-bottom: 10px;">
                        <div style="color: #CE93D8; font-weight: bold; margin-bottom: 8px;">📈 TRAINING HEALTH</div>
                        <div style="margin-left: 15px;">
                            <span style="display: inline-block; width: 180px;">Convergence:</span> <span style="color: {conv_color};">{convergence_rate:+.6f} ({conv_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Stability:</span> <span style="color: {stab_color};">{stability:.4f} ({stab_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Gen. Gap:</span> <span style="color: {gap_color};">{gen_gap:+.6f} ({gap_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Coherence:</span> <span style="color: {coh_color};">{coherence:.4f} ({coh_status})</span><br>
                            <span style="display: inline-block; width: 180px;">Progress:</span> <span style="color: {'#81C784' if progress > 0 else '#EF5350'};">{progress*100:+.2f}%</span><br>
                        </div>
                    </div>
                </div>
                """
                display(HTML(html_content))

    return _InteractivePlotCallback()


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
    Price heads are inverse-transformed with the target scaler; direction and variance heads are
    sanitised (NaN/Inf -> neutral) and clipped. Shared by the test and calibration paths.
    """
    bs = int(batch_size or getattr(cfg, 'BATCH_SIZE', 64))
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(X, dtype='float32')).batch(bs)
    heads = PredictiveOutputs(*model.predict(ds, verbose=0))
    n = int(n)

    def _delta(head):
        scaled = np.asarray(head).reshape(-1)[:n]
        return target_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()

    def _prob(head):
        p = np.asarray(head, dtype=float).reshape(-1)[:n]
        return np.nan_to_num(p, nan=0.5, posinf=0.5, neginf=0.5).clip(0.0, 1.0)

    def _var(head):
        v = np.asarray(head, dtype=float).reshape(-1)[:n]
        return np.nan_to_num(v, nan=1.0, posinf=1.0, neginf=1.0).clip(float(cfg.VAR_FLOOR), float(cfg.VAR_CAP))

    return {
        "delta": {"h0": _delta(heads.price_h0), "h1": _delta(heads.price_h1), "h2": _delta(heads.price_h2)},
        "direction_prob": {"h0": _prob(heads.direction_h0), "h1": _prob(heads.direction_h1),
                           "h2": _prob(heads.direction_h2)},
        "variance": {"h0": _var(heads.variance_h0), "h1": _var(heads.variance_h1), "h2": _var(heads.variance_h2)},
    }


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
) -> TrainResult:
    """Train (optionally) and evaluate, returning a rich result bundle.

    This is intended to be the notebook's single source of truth for:
    - data prep + scaling
    - model heads and extraction
    - metrics and evaluation semantics
    """

    cfg = config or Config()
    tf.keras.utils.set_random_seed(int(getattr(cfg, 'SEED', 42)))
    if csv_path is not None:
        cfg.CSV_PATH = csv_path
    cfg = _apply_config_overrides(cfg, config_overrides)
    cfg.validate()  # P0-3 / P1-4: early enforcement (added in Config refactor)

    print("Starting enhanced model training with extended trend features...")
    data_processor = DataProcessor(cfg)
    df, close_values = data_processor.load_and_prepare_data(read_csv_kwargs=read_csv_kwargs)

    (X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
     X_test_seq, y_test_scaled, last_close_test, extended_trends_test,
     y_train, y_test, target_scaler) = data_processor.prepare_datasets(df, close_values)

    input_scaler = getattr(data_processor, 'input_scaler', None)
    predictor = PricePredictor(cfg)
    base_model = predictor.build_model()
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
        inputs=base_model.inputs,
        outputs=base_model.outputs,
    )

    _vb = data_processor.val_block  # early stopping / checkpoint / LR select on the VALIDATION block, never on test
    train_ds, val_ds = predictor.create_datasets(
        X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
        _vb["X"], _vb["y_scaled"], _vb["last_close"], _vb["extended_trends"],
    )

    # Optional calibration (kept identical to train_model behavior)
    _calib_lambdas: Optional[Dict[str, float]] = None  # set below if calibrate=True
    _CALIB_LAMBDA_NAMES = (
        'lambda_short', 'lambda_point', 'lambda_long', 'lambda_extended_trend', 'lambda_dir',
        'lambda_var', 'lambda_vol', 'lambda_crps', 'lambda_soft_ece',
        'lambda_t_perp', 'lambda_casimir', 'lambda_hd', 'lambda_ife',
    )
    _calib_saved: Dict[str, float] = {}  # populated after the originals are read; restored on failure
    if calibrate is True:
        try:
            # ----------------------------------------------------------------
            # Read calibration knobs from config (with safe fallbacks)
            # ----------------------------------------------------------------
            # Derive batch counts from actual training dataset size so the
            # calibration overhead scales with the training set, not a hardcoded number.
            train_batches  = math.ceil(X_train_seq.shape[0] / cfg.BATCH_SIZE)
            warmup_frac    = float(getattr(cfg, 'CALIB_WARMUP_FRACTION', 0.15))
            sample_frac    = float(getattr(cfg, 'CALIB_SAMPLE_FRACTION', 0.35))
            n_warmup       = max(1, round(train_batches * warmup_frac))
            n_sample       = max(1, round(train_batches * sample_frac))
            lam_min    = float(getattr(cfg, 'CALIB_LAMBDA_MIN', 0.1))
            lam_max    = float(getattr(cfg, 'CALIB_LAMBDA_MAX', 20.0))
            d_global   = float(getattr(cfg, 'CALIB_DAMPING', getattr(cfg, 'DAMPING', 0.5)))
            if 'DAMPING' in dir(cfg) and not hasattr(cfg, 'CALIB_DAMPING'):
                logging.getLogger(__name__).warning("Config.DAMPING is legacy (P1-1); prefer CALIB_DAMPING")
            calib_outer = bool(getattr(cfg, 'CALIB_OUTER', False))

            def _d(attr):
                """Resolve per-component damping, falling back to global."""
                v = getattr(cfg, attr, None)
                return float(v) if v is not None else d_global

            d_point = _d('CALIB_DAMPING_POINT')
            d_trend = _d('CALIB_DAMPING_TREND')
            d_dir   = _d('CALIB_DAMPING_DIR')
            d_var   = _d('CALIB_DAMPING_VAR')
            d_crps  = _d('CALIB_DAMPING_CRPS')
            d_ece   = _d('CALIB_DAMPING_ECE')
            d_vol   = _d('CALIB_DAMPING_VOL')
            d_physics = _d('CALIB_DAMPING_PHYSICS')  # 0.0 by default: bounded regularisers are excluded from equalisation

            # ----------------------------------------------------------------
            # Save originals and reset all per-component lambdas to 1.0
            # so that natural magnitudes are measured without existing weights.
            # ----------------------------------------------------------------
            orig_short = float(custom_model.lambda_short)
            orig_point = float(custom_model.lambda_point)
            orig_long  = float(custom_model.lambda_long)
            orig_ext   = float(custom_model.lambda_extended_trend)
            orig_dir   = float(custom_model.lambda_dir)
            orig_var   = float(custom_model.lambda_var)
            orig_vol   = float(custom_model.lambda_vol)
            orig_crps  = float(custom_model.lambda_crps)
            orig_ece   = float(custom_model.lambda_soft_ece)
            orig_t_perp   = float(custom_model.lambda_t_perp)
            orig_casimir  = float(custom_model.lambda_casimir)
            orig_hd       = float(custom_model.lambda_hd)
            orig_ife      = float(custom_model.lambda_ife)
            _calib_saved.update({_n: float(getattr(custom_model, _n)) for _n in _CALIB_LAMBDA_NAMES})

            custom_model.lambda_short            = 1.0
            custom_model.lambda_point            = 1.0
            custom_model.lambda_long             = 1.0
            custom_model.lambda_extended_trend   = 1.0
            custom_model.lambda_dir              = 1.0
            custom_model.lambda_var              = 1.0
            custom_model.lambda_vol              = 1.0
            custom_model.lambda_crps             = 1.0
            custom_model.lambda_soft_ece         = 1.0
            custom_model.lambda_t_perp           = 1.0
            custom_model.lambda_casimir          = 1.0
            custom_model.lambda_hd               = 1.0
            custom_model.lambda_ife              = 1.0

            # ----------------------------------------------------------------
            # Phase 1 — warm-up forward passes (no sampling, no gradient). There is no BatchNorm in the graph; this builds the graph and model.losses before sampling.
            # ----------------------------------------------------------------
            print(f"[calib] Warm-up forward passes over {n_warmup}/{train_batches} batches ({warmup_frac:.0%} of epoch) to build the graph and layer losses before sampling...")
            for batch in train_ds.take(n_warmup):
                x_batch, _, _, _ = batch
                _ = custom_model(x_batch, training=True)

            # ----------------------------------------------------------------
            # Phase 2 — Sample loss magnitudes
            # ----------------------------------------------------------------
            print(f"[calib] Sampling loss magnitudes over {n_sample}/{train_batches} batches ({sample_frac:.0%} of epoch)...")
            short_buf, point_buf, long_buf = [], [], []
            ext_buf, dir_buf, var_buf, vol_buf = [], [], [], []
            crps_buf, ece_buf = [], []
            t_perp_buf, casimir_buf, vac_buf, hd_buf, ife_buf, vac_overflow_buf = [], [], [], [], [], []

            for batch in train_ds.take(n_sample):
                x_batch, y_batch, last_batch, ext_batch = batch
                _y_pred_raw = custom_model(x_batch, training=True)
                # Strip 10th output (vacuum_overflow) before passing to custom_loss
                (*y_pred_batch, _vac_overflow_batch) = _y_pred_raw
                (total,
                 point_h0, point_h1, point_h2,
                 local_h0, global_h0, ext_h0,
                 local_h1, global_h1, ext_h1,
                 local_h2, global_h2, ext_h2,
                 dir_h0, dir_h1, dir_h2,
                 nll_h0, nll_h1, nll_h2,
                 reg_val, inter_reg, vol_loss_val,
                 crps_h0_c, crps_h1_c, crps_h2_c,
                 soft_ece_h0_c, soft_ece_h1_c, soft_ece_h2_c,
                 t_perp_c, casimir_c, vac_c, hd_c, ife_c,
                 vac_overflow_c) = custom_model.custom_loss(
                    x_batch, y_batch, y_pred_batch, last_batch, ext_batch,
                    vacuum_overflow=_vac_overflow_batch
                )

                short_buf.append(float(point_h0))
                point_buf.append(float(point_h1))
                long_buf.append(float(point_h2))
                ext_buf.append(float((ext_h0 + ext_h1 + ext_h2) / 3.0))
                dir_buf.append(float((dir_h0 + dir_h1 + dir_h2) / 3.0))
                var_buf.append(float((nll_h0 + nll_h1 + nll_h2) / 3.0))
                vol_buf.append(float(vol_loss_val))
                crps_buf.append(float((crps_h0_c + crps_h1_c + crps_h2_c) / 3.0))
                ece_buf.append(float((soft_ece_h0_c + soft_ece_h1_c + soft_ece_h2_c) / 3.0))
                t_perp_buf.append(float(t_perp_c))
                casimir_buf.append(float(casimir_c))
                vac_buf.append(float(vac_c))
                hd_buf.append(float(hd_c))
                ife_buf.append(float(ife_c))
                vac_overflow_buf.append(float(vac_overflow_c))

            def _med(buf):
                return float(np.median(np.array(buf))) if buf else 0.0

            med_short = _med(short_buf)
            med_point = _med(point_buf)
            med_long  = _med(long_buf)
            med_ext   = _med(ext_buf)
            med_dir   = _med(dir_buf)
            med_var   = _med(var_buf)
            med_vol   = _med(vol_buf)
            med_crps  = _med(crps_buf)
            med_ece   = _med(ece_buf)
            med_t_perp  = _med(t_perp_buf)
            med_casimir = _med(casimir_buf)
            med_vac     = _med(vac_buf)
            med_hd      = _med(hd_buf)
            med_ife     = _med(ife_buf)
            med_vac_overflow = _med(vac_overflow_buf)

            # Reference = mean of all active (non-zero) component medians.
            # CRPS and ECE are included only when their config lambda is active.
            crps_active = float(getattr(cfg, 'LAMBDA_CRPS', 0.0)) > 0.0
            ece_active  = float(getattr(cfg, 'LAMBDA_SOFT_ECE', 0.0)) > 0.0
            t_perp_active  = float(getattr(cfg, 'LAMBDA_T_PERP',  0.0)) > 0.0
            casimir_active = float(getattr(cfg, 'LAMBDA_CASIMIR', 0.0)) > 0.0
            hd_active      = float(getattr(cfg, 'LAMBDA_HD',      0.0)) > 0.0
            ife_active     = float(getattr(cfg, 'LAMBDA_IFE',     0.0)) > 0.0
            candidate_meds = [med_short, med_point, med_long, med_ext, med_dir, med_var, med_vol]
            if crps_active:
                candidate_meds.append(med_crps)
            if ece_active:
                candidate_meds.append(med_ece)
            if t_perp_active:
                candidate_meds.append(med_t_perp)
            if casimir_active:
                candidate_meds.append(med_casimir)
            if hd_active:
                candidate_meds.append(med_hd)
            if ife_active:
                candidate_meds.append(med_ife)
            vac_overflow_active = float(getattr(cfg, 'LAMBDA_VAC_OVERFLOW', 0.0)) > 0.0
            if vac_overflow_active and med_vac_overflow > 1e-8:
                candidate_meds.append(med_vac_overflow)
            # vac is always added (vacuum bandwidth self-limiting is always active)
            if med_vac > 1e-8:
                candidate_meds.append(med_vac)
            non_zero = [m for m in candidate_meds if m > 1e-8]
            ref_loss = float(np.mean(non_zero)) if non_zero else 1.0

            # ----------------------------------------------------------------
            # Phase 3 — Damped rescaling and clamping
            # ----------------------------------------------------------------
            eps = 1e-8

            def _rescale(orig, med, damping):
                if med > eps:
                    return float(np.clip(orig * (ref_loss / (med + eps)) ** damping, lam_min, lam_max))
                return orig  # component inactive — keep original

            new_short = _rescale(orig_short, med_short, d_point)
            new_point = _rescale(orig_point, med_point, d_point)
            new_long  = _rescale(orig_long,  med_long,  d_point)
            new_ext   = _rescale(orig_ext,   med_ext,   d_trend)
            new_dir   = _rescale(orig_dir,   med_dir,   d_dir)
            new_var   = _rescale(orig_var,   med_var,   d_var)
            new_vol   = _rescale(orig_vol,   med_vol,   d_vol)
            new_crps  = _rescale(orig_crps,  med_crps,  d_crps) if crps_active else orig_crps
            new_ece   = _rescale(orig_ece,   med_ece,   d_ece)  if ece_active  else orig_ece
            new_t_perp  = _rescale(orig_t_perp,  med_t_perp,  d_physics) if t_perp_active  else orig_t_perp
            new_casimir = _rescale(orig_casimir, med_casimir, d_physics) if casimir_active else orig_casimir
            new_hd      = _rescale(orig_hd,      med_hd,      d_physics) if hd_active      else orig_hd
            new_ife     = _rescale(orig_ife,     med_ife,     d_physics) if ife_active     else orig_ife

            custom_model.lambda_short          = new_short
            custom_model.lambda_point          = new_point
            custom_model.lambda_long           = new_long
            custom_model.lambda_extended_trend = new_ext
            custom_model.lambda_dir            = new_dir
            custom_model.lambda_var            = new_var
            custom_model.lambda_vol            = new_vol
            custom_model.lambda_crps           = new_crps
            custom_model.lambda_soft_ece       = new_ece
            custom_model.lambda_t_perp         = new_t_perp
            custom_model.lambda_casimir        = new_casimir
            custom_model.lambda_hd             = new_hd
            custom_model.lambda_ife            = new_ife

            # ----------------------------------------------------------------
            # Phase 4 — Optional outer-multiplier calibration (CALIB_OUTER)
            # Calibrates lambda_trend_outer, lambda_dir_outer, lambda_nll_outer
            # so that the already-rescaled per-component group sums are equalized.
            # Uses same damping logic (d_global) and same clamp bounds.
            # ----------------------------------------------------------------
            if calib_outer:
                med_trend_group = new_ext * med_ext          # post-rescale magnitude proxy
                med_dir_group   = new_dir * med_dir
                med_nll_group   = new_var * med_var
                outer_meds = [m for m in [med_trend_group, med_dir_group, med_nll_group] if m > eps]
                ref_outer = float(np.mean(outer_meds)) if outer_meds else 1.0

                def _rescale_outer(orig_outer, med_g):
                    if med_g > eps:
                        return float(np.clip(orig_outer * (ref_outer / (med_g + eps)) ** d_global, lam_min, lam_max))
                    return orig_outer

                custom_model.lambda_trend_outer = _rescale_outer(custom_model.lambda_trend_outer, med_trend_group)
                custom_model.lambda_dir_outer   = _rescale_outer(custom_model.lambda_dir_outer,   med_dir_group)
                custom_model.lambda_nll_outer   = _rescale_outer(custom_model.lambda_nll_outer,   med_nll_group)

            # ----------------------------------------------------------------
            # Print report
            # ----------------------------------------------------------------
            def _fmt_row(name, orig, med, new, active=True):
                skip = "" if active else " [skipped — inactive]"
                arrow = f"{orig:.4f} → {new:.4f}"
                return f"  {name:<14} med={med:.6f}  {arrow}{skip}"

            print("[calib] Sampled medians and updated lambdas:")
            print(_fmt_row("λ_short",  orig_short, med_short, new_short))
            print(_fmt_row("λ_point",  orig_point, med_point, new_point))
            print(_fmt_row("λ_long",   orig_long,  med_long,  new_long))
            print(_fmt_row("λ_trend",  orig_ext,   med_ext,   new_ext))
            print(_fmt_row("λ_dir",    orig_dir,   med_dir,   new_dir))
            print(_fmt_row("λ_var",    orig_var,   med_var,   new_var))
            print(_fmt_row("λ_vol",    orig_vol,   med_vol,   new_vol))
            print(_fmt_row("λ_crps",   orig_crps,  med_crps,  new_crps,  active=crps_active))
            print(_fmt_row("λ_ece",    orig_ece,   med_ece,   new_ece,   active=ece_active))
            print(_fmt_row("λ_t_perp", orig_t_perp,  med_t_perp,  new_t_perp,  active=t_perp_active))
            print(_fmt_row("λ_casimir",orig_casimir, med_casimir, new_casimir, active=casimir_active))
            print(_fmt_row("λ_hd",     orig_hd,      med_hd,      new_hd,      active=hd_active))
            print(_fmt_row("λ_ife",    orig_ife,     med_ife,     new_ife,     active=ife_active))
            lambda_vac_orig = float(getattr(cfg, 'LAMBDA_VAC', 0.0))
            print(_fmt_row("Λ_vac(thr)", lambda_vac_orig, med_vac, lambda_vac_orig, active=True) + "  (threshold, not rescaled)  # P0-2: default now 0 (opt-in)")
            if calib_outer:
                print(f"  [outer] λ_trend_outer={custom_model.lambda_trend_outer:.4f}  "
                      f"λ_dir_outer={custom_model.lambda_dir_outer:.4f}  "
                      f"λ_nll_outer={custom_model.lambda_nll_outer:.4f}")
            print(f"[calib] ref_loss={ref_loss:.6f}  d_global={d_global}  "
                  f"warmup={n_warmup}/{train_batches}  sample={n_sample}/{train_batches}  clamp=[{lam_min}, {lam_max}]")

            _calib_lambdas = {
                'lambda_short':          new_short,
                'lambda_point':          new_point,
                'lambda_long':           new_long,
                'lambda_extended_trend': new_ext,
                'lambda_dir':            new_dir,
                'lambda_var':            new_var,
                'lambda_vol':            new_vol,
                'lambda_crps':           new_crps,
                'lambda_soft_ece':       new_ece,
                'lambda_t_perp':         new_t_perp,
                'lambda_casimir':        new_casimir,
                'lambda_hd':             new_hd,
                'lambda_ife':            new_ife,
                'ref_loss':              ref_loss,
            }
            if calib_outer:
                _calib_lambdas.update({
                    'lambda_trend_outer': custom_model.lambda_trend_outer,
                    'lambda_dir_outer':   custom_model.lambda_dir_outer,
                    'lambda_nll_outer':   custom_model.lambda_nll_outer,
                })

        except Exception as e:
            import traceback
            # Restore the lambdas that were reset to 1.0 for sampling. Without this, a
            # failure after the reset silently trained with every lambda at 1.0 while
            # the message claimed "default lambdas".
            for _name, _value in _calib_saved.items():
                setattr(custom_model, _name, _value)
            print(f"[calib] Calibration pass failed — restored configured lambdas and continuing: {e}")
            traceback.print_exc()

    opt = optimizers.Adam(learning_rate=cfg.LR)
    custom_model.compile(optimizer=opt)

    csv_logger = callbacks.CSVLogger("training_log.csv", append=True)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=cfg.EARLY, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(cfg.MODEL_PATH, save_best_only=True, monitor='val_loss', save_weights_only=True)
    tqdm_callback = TqdmCallback()
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=cfg.PATIENCE)

    learnable_layer = None
    for layer in custom_model.layers:
        if getattr(layer, 'name', '').startswith('learnable_indicators'):
            learnable_layer = layer
            break
    params_logger = ParamsLogger(layer=learnable_layer, out_csv='indicator_params_history.csv')

    # S21: there is exactly one EarlyStopping (on val_loss, restoring best weights). A second
    # one on val_dir_mcc_h1 without restore used to race it; it has been removed.
    callbacks_list = [csv_logger, es, ckpt, tqdm_callback, params_logger, lr_scheduler]
    if extra_callbacks:
        callbacks_list += list(extra_callbacks)

    actual_epochs = int(epochs) if epochs is not None else int(cfg.EPOCHS)
    history = None
    if os.path.exists(cfg.MODEL_PATH) and not force:
        print(f"Loading existing model weights from {cfg.MODEL_PATH}...")
        try:
            custom_model.load_weights(cfg.MODEL_PATH)
        except Exception as e:
            print(f"Warning: failed to load existing weights but continuing: {e}")
    else:
        if os.path.exists(cfg.MODEL_PATH) and force:
            try:
                custom_model.load_weights(cfg.MODEL_PATH)
            except Exception:
                pass
        print(f"Training for {actual_epochs} epochs...")
        history = custom_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=actual_epochs,
            callbacks=callbacks_list,
            verbose=0,
        )
        print(f"Enhanced model weights saved to {cfg.MODEL_PATH}")
        try:
            joblib.dump(target_scaler, cfg.SCALER_PATH)
            if input_scaler is not None:
                joblib.dump(input_scaler, cfg.SCALER_PATH.replace('.joblib', '_input.joblib'))
        except Exception:
            pass

    print("Evaluating enhanced model...")
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

    print("\n[Diagnostic: Prediction Statistics]")
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
        print(f"  {h_name}: pred_mean={pred_mean:.6f}, true_mean={true_mean:.6f} | pred_std={pred_std:.6f}, true_std={true_std:.6f}")
        print(f"         pred_range=[{pred_min:.6f}, {pred_max:.6f}], true_range=[{true_min:.6f}, {true_max:.6f}]")


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
        pass

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
            print("\nFitting CalibrationPipeline on the calibration split...")
            predictions_cal = _predict_heads(custom_model, _cb['X'], _cb['y_raw'].shape[0], target_scaler, cfg)
            cal_pipeline = _CalibrationPipeline()
            cal_pipeline.fit_from_arrays(
                predictions_dict=predictions_cal,
                y_true_delta_raw=np.asarray(_cb['y_raw'], dtype=float),
                last_close=np.asarray(_cb['last_close'], dtype=float),
                deadband_bps=float(getattr(cfg, 'DIR_DEADBAND_BPS', 0.0)),
            )
            predictions_calibrated = cal_pipeline.apply(predictions, alpha=0.1)
            calibration_report = _calibration_coverage_report(predictions_calibrated, np.asarray(y_test), cal_pipeline)
            for _h, _row in calibration_report.items():
                print(f"  [test] {_h}: conformal coverage@90 = {_row['coverage90']:.3f} (target >= 0.90), "
                      f"mean width = {_row['width90']:.2f} raw units, T = {_row['temperature']:.3f}")
        except Exception as _cal_err:
            import traceback
            print(f"CalibrationPipeline fit FAILED (continuing without calibration): {_cal_err}")
            traceback.print_exc()
            cal_pipeline = None
            predictions_calibrated = None

    return TrainResult(
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
    )

# -----------------------------



# PredictiveOutputs (named view over the 10 model outputs) lives in neural_trade.core.outputs.


class PricePredictor:
    def __init__(self, config: Config):
        self.config = config

    def build_model(self):
        """Build the configured architecture (Models registry, Config.MODEL_NAME)."""
        return Models.build(getattr(self.config, 'MODEL_NAME', None), self.config)

    def create_datasets(self, X_train, y_train, last_close_train, extended_trends_train,
                        X_test, y_test, last_close_test, extended_trends_test):
        def make_tf_dataset(Xseq, yseq, last_close, extended_trends, batch_size, shuffle=False):
            ds = tf.data.Dataset.from_tensor_slices((
                Xseq, yseq, last_close.reshape(-1,1), extended_trends
            ))
            if shuffle:
                ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = make_tf_dataset(X_train, y_train, last_close_train, extended_trends_train,
                                   self.config.BATCH_SIZE, shuffle=True)
        val_ds = make_tf_dataset(X_test, y_test, last_close_test, extended_trends_test,
                                 self.config.BATCH_SIZE, shuffle=False)
        return train_ds, val_ds

# -----------------------------





# Step-metric machinery (epoch accumulators, direction statistics) lives in
# neural_trade.metrics.tf_direction; the metric functions are the Metrics registry's TF tier.

class CustomTrainModel(models.Model):
    def __init__(self, base_model, pred_scale, pred_mean,
                 lambda_point=1.0, lambda_local_trend=1.0, lambda_global_trend=0.2,
                 lambda_extended_trend=0.16, lambda_dir=1.0, config=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.epsilon = 1e-8
        # Per-term loss weights live in non-trainable tf.Variables behind properties (see
        # _make_lambda_property after this class). Reads in losses.py (`model.lambda_x`) and
        # writes in the calibration pass (`model.lambda_x = v`) are unchanged, but a write is
        # now an in-place .assign() that takes effect on the next step without retracing.
        # Created with attribute tracking off so Keras does not add them to the weights file.
        self._setattr_tracking = False
        self._lambda_vars = {}
        self._setattr_tracking = True

        # Cast important scalars to float32 early
        self.pred_scale = tf.cast(pred_scale, tf.float32)
        self.pred_mean = tf.cast(pred_mean, tf.float32)

        # Basic numeric guard
        if tf.keras.backend.get_value(self.pred_scale) < 1e-6:
            raise ValueError("pred_scale is too small, which may cause numerical instability.")

        self.lambda_point = float(lambda_point)
        self.lambda_local_trend = float(lambda_local_trend)
        self.lambda_global_trend = float(lambda_global_trend)
        self.lambda_extended_trend = float(lambda_extended_trend)
        self.lambda_dir = float(lambda_dir)  # New
        self.lambda_vol = config.LAMBDA_VOL
        self.lambda_short = config.LAMBDA_SHORT
        self.lambda_long = config.LAMBDA_LONG
        self.lambda_var = config.LAMBDA_VAR
        self.lambda_trend_outer = float(getattr(config, 'LAMBDA_TREND_OUTER', 0.5))
        self.lambda_dir_outer = float(getattr(config, 'LAMBDA_DIR_OUTER', 0.5))
        self.lambda_dir_align_outer = float(getattr(config, 'LAMBDA_DIR_ALIGN_OUTER', 0.0))
        self.lambda_coherence_outer = float(getattr(config, 'LAMBDA_COHERENCE', 1.0))
        self.lambda_nll_outer = float(getattr(config, 'LAMBDA_NLL_OUTER', 1.0))
        self.lambda_crps = float(getattr(config, 'LAMBDA_CRPS', 0.0))
        self.lambda_soft_ece = float(getattr(config, 'LAMBDA_SOFT_ECE', 0.0))
        # === T_⊥ / QBOX lambdas (default 0.0 → backward compatible; enable explicitly) ===
        self.lambda_t_perp       = float(getattr(config, 'LAMBDA_T_PERP',       0.0))
        self.lambda_casimir      = float(getattr(config, 'LAMBDA_CASIMIR',      0.0))
        self.lambda_hd           = float(getattr(config, 'LAMBDA_HD',           0.0))
        self.lambda_ife          = float(getattr(config, 'LAMBDA_IFE',          0.0))
        self.lambda_vac_overflow = float(getattr(config, 'LAMBDA_VAC_OVERFLOW', 0.0))
        self.config = config or Config()

        # Dedicated optimizer for indicator logit vars (LR = main LR * INDICATOR_LR_MULT).
        # Adam normalizes gradient magnitudes, so scaling grads is insufficient — a higher LR
        # is the only way to give indicator params a genuinely larger step size.
        ind_lr = float(self.config.LR) * float(getattr(self.config, 'INDICATOR_LR_MULT', 10.0))
        self.indicator_optimizer = optimizers.Adam(learning_rate=ind_lr)

        # Single source-of-truth for Huber delta (in *scaled* units)
        self.huber_delta = float(self.config.HUBER_DELTA)

        # Numerical epsilon used in denominators
        self.eps = tf.constant(1e-8, dtype=tf.float32)
        # Counts training steps whose update was zeroed by the finite-gradient guard (reset each epoch by Keras).
        self.nonfinite_grad_steps = tf.keras.metrics.Sum(name='nonfinite_grad_steps')
        # Epoch accumulators behind every logged step metric (see neural_trade.metrics.tf_direction.STEP_MEAN_KEYS).
        # Created with attribute tracking off; the `metrics` property below hands them to
        # Keras so they are reset at every epoch and before every evaluation.
        self._setattr_tracking = False
        self._step_means = {k: tf.keras.metrics.Mean(name=k) for k in STEP_MEAN_KEYS}
        # Step metrics resolved ONCE from the Metrics registry (TF tier), never inside tf.function.
        self._step_metric_fns = Metrics.tf_functions(getattr(self.config, 'STEP_METRICS', None))
        self._dir_head_acc = DirectionAccumulator(name='dir_head_accumulator')
        self._dir_gauss_acc = DirectionAccumulator(name='dir_gauss_accumulator')
        self._pit_acc = PITAccumulator(var_floor=float(getattr(self.config, 'VAR_FLOOR', 1e-4)),
                                        var_cap=float(getattr(self.config, 'VAR_CAP', 1e3)),
                                        name='pit_accumulator')
        self._setattr_tracking = True

        # Robust (non-string) collection of indicator vars for gradient routing
        # (to indicator_optimizer) and post-step period clipping.
        # Falls back gracefully if the layer is not present (e.g. during some tests).
        self._indicator_var_ids = set()
        self._indicator_layer = None
        try:
            if base_model is not None:
                for layer in base_model.layers:
                    if getattr(layer, 'name', '').startswith('learnable_indicators'):
                        ind_vars = getattr(layer, 'get_indicator_trainable_variables', lambda: [])()
                        self._indicator_var_ids = {id(v) for v in ind_vars}
                        self._indicator_layer = layer
                        break
        except Exception:
            pass

        # NOTE: We no longer use tf.keras.losses.Huber in the primary point supervision path;
        # point loss delegates to the registered "point_huber" which implements log(cosh).
        # A separate piecewise Huber lives in CustomTrainModel.huber (unused for the main loss).
        # Config.USE_HUBER is legacy and not consulted by the active custom_loss.
    def _logit_from_alpha(self, alpha): return mh.logit_from_alpha(alpha, self.epsilon)
    def _alpha_from_logit(self, logit): return mh.alpha_from_logit(logit)
    def _logit_from_period(self, period): return mh.logit_from_period(period, self.epsilon)
    def _period_from_logit(self, logit): return mh.period_from_logit(logit, self.epsilon)
    # -------------------------
    # Unified element-wise Huber
    # -------------------------
    def huber(self, x, delta=None):
        """Element-wise Huber (returns same-shape tensor). Works on scaled differences."""
        if delta is None:
            delta = tf.cast(self.huber_delta, tf.float32)
        else:
            delta = tf.cast(delta, tf.float32)

        x = tf.cast(x, tf.float32)
        abs_x = tf.abs(x)
        quadratic = 0.5 * tf.square(x)
        linear = delta * (abs_x - 0.5 * delta)
        return tf.where(abs_x <= delta, quadratic, linear)

    @property
    def metrics(self):
        """Metrics Keras resets at each epoch / evaluation: the built-in ones plus the epoch
        accumulators behind the step logs."""
        base = list(super().metrics)
        extra = list(getattr(self, '_step_means', {}).values())
        extra += [m for m in (getattr(self, '_dir_head_acc', None), getattr(self, '_dir_gauss_acc', None),
                              getattr(self, '_pit_acc', None)) if m is not None]
        seen = {id(m) for m in base}
        return base + [m for m in extra if id(m) not in seen]

    def _epoch_logs(self, loss_components, y_true, y_pred_9, true_dirs, head_probs, gauss_probs, masks,
                    head_prefix, gauss_prefix, training, grad_global_norm=None):
        """Update the epoch accumulators with this batch and return their running aggregates."""
        c = loss_components
        batch = tf.cast(tf.shape(y_true)[0], tf.float32)
        scalars = {
            'loss': c.total,
            'point_loss': c.point_h0 + c.point_h1 + c.point_h2,
            'point_h0': c.point_h0, 'point_h1': c.point_h1, 'point_h2': c.point_h2,
            'trend_h0': c.local_h0 + c.global_h0 + c.extended_h0,
            'trend_h1': c.local_h1 + c.global_h1 + c.extended_h1,
            'trend_h2': c.local_h2 + c.global_h2 + c.extended_h2,
            'local_h0': c.local_h0, 'global_h0': c.global_h0, 'extended_h0': c.extended_h0,
            'local_h1': c.local_h1, 'global_h1': c.global_h1, 'extended_h1': c.extended_h1,
            'local_h2': c.local_h2, 'global_h2': c.global_h2, 'extended_h2': c.extended_h2,
            'dir_loss': c.dir_h0 + c.dir_h1 + c.dir_h2,
            'dir_loss_h0': c.dir_h0, 'dir_loss_h1': c.dir_h1, 'dir_loss_h2': c.dir_h2,
            'nll_loss': c.nll_h0 + c.nll_h1 + c.nll_h2,
            'nll_h0': c.nll_h0, 'nll_h1': c.nll_h1, 'nll_h2': c.nll_h2,
            'crps_loss': c.crps_h0 + c.crps_h1 + c.crps_h2,
            'crps_h0': c.crps_h0, 'crps_h1': c.crps_h1, 'crps_h2': c.crps_h2,
            'soft_ece_loss': c.soft_ece_h0 + c.soft_ece_h1 + c.soft_ece_h2,
            'soft_ece_h0': c.soft_ece_h0, 'soft_ece_h1': c.soft_ece_h1, 'soft_ece_h2': c.soft_ece_h2,
            'reg_loss': c.reg_loss, 'inter_reg': c.inter_reg, 'vol_loss': c.vol_loss,
            't_perp_loss': c.t_perp_total, 'casimir_loss': c.casimir_val, 'vac_loss': c.vac_val,
            'hd_loss': c.hd_val, 'ife_loss': c.ife_val, 'vac_overflow_loss': c.vac_overflow_val,
        }
        scalars['trend_loss'] = scalars['trend_h0'] + scalars['trend_h1'] + scalars['trend_h2']
        if training and grad_global_norm is not None:
            scalars['grad_global_norm'] = grad_global_norm
        for k, v in scalars.items():
            self._step_means[k].update_state(tf.cast(v, tf.float32), sample_weight=batch)
        self._dir_head_acc.update_state(true_dirs, head_probs, masks)
        self._dir_gauss_acc.update_state(true_dirs, gauss_probs, masks)
        self._pit_acc.update_state([y_true[:, 0], y_true[:, 1], y_true[:, 2]],
                                   [y_pred_9[0], y_pred_9[3], y_pred_9[6]],
                                   [y_pred_9[2], y_pred_9[5], y_pred_9[8]])
        logs = {k: m.result() for k, m in self._step_means.items()
                if training or k not in TRAIN_ONLY_MEAN_KEYS}
        logs.update(self._pit_acc.logs())
        logs.update(self._dir_head_acc.logs(head_prefix, self._step_metric_fns))
        logs.update(self._dir_gauss_acc.logs(gauss_prefix, self._step_metric_fns))
        return logs


    # Small utility: reduce-mean with safe casting
    def _reduce_mean(self, x):
        return tf.reduce_mean(tf.cast(x, tf.float32))

    @staticmethod
    def _normal_cdf(z):
        """Standard Normal CDF using erf; z can be any float tensor."""
        z = tf.cast(z, tf.float32)
        return 0.5 * (1.0 + tf.math.erf(z / tf.constant(np.sqrt(2.0), dtype=tf.float32)))

    # -------------------------
    # Utility / transforms (moved outside class to avoid tracing issues)
    # -------------------------
    def _pit_ks(self, y, mu, var):
        """Kolmogorov-Smirnov distance between the PIT values Phi((y - mu) / sigma) and U[0, 1].

        Computed in-graph. The previous implementation called .numpy() on symbolic tensors
        inside the traced train step, was swallowed by a bare except, and logged NaN in
        every epoch; test_step did not compute it at all.
        """
        y = tf.cast(tf.reshape(y, [-1]), tf.float32)
        mu = tf.cast(tf.reshape(mu, [-1]), tf.float32)
        var = tf.cast(tf.reshape(var, [-1]), tf.float32)
        var = tf.clip_by_value(var, float(getattr(self.config, 'VAR_FLOOR', 1e-4)),
                               float(getattr(self.config, 'VAR_CAP', 1e3)))
        u = tf.sort(self._normal_cdf((y - mu) / (tf.sqrt(var) + self.eps)))
        n = tf.cast(tf.shape(u)[0], tf.float32)
        i = tf.range(1.0, n + 1.0, dtype=tf.float32)
        d_plus = tf.reduce_max(i / n - u)
        d_minus = tf.reduce_max(u - (i - 1.0) / n)
        return tf.maximum(d_plus, d_minus)

    @staticmethod
    def _to_scaled_static(raw, pred_mean, pred_scale, eps=1e-8):
        """Convert raw prices to scaled units (same domain as dataset scaling)."""
        raw = tf.cast(raw, tf.float32)
        return (raw - pred_mean) / (pred_scale + eps)

    def _to_scaled(self, raw):
        """Instance helper that uses the stored scaling parameters."""
        return self._to_scaled_static(raw, self.pred_mean, self.pred_scale, self.eps)

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    # -------------------------
    # Focal Loss for imbalanced binary classification
    # -------------------------
    def focal_loss(self, true_labels, logits, alpha=None, gamma=None, reduce=True):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.focal_loss(self, true_labels, logits, alpha=alpha, gamma=gamma, reduce=reduce)

    # -------------------------
    # Dice Loss for F1-like optimization (differentiable)
    # -------------------------
    def dice_loss(self, true_labels, logits, smooth=1.0, reduce=True):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.dice_loss(self, true_labels, logits, smooth=smooth, reduce=reduce)

    # -------------------------
    # Combined Focal + Dice Loss for balanced optimization
    # -------------------------
    def combined_direction_loss(self, true_labels, logits, alpha=None, gamma=None, 
                                 focal_weight=0.5, dice_weight=0.5, reduce=True):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.combined_direction_loss(self, true_labels, logits, alpha=alpha, gamma=gamma,
                                               focal_weight=focal_weight, dice_weight=dice_weight, reduce=reduce)

    # -------------------------
    # Dynamic Alpha Computation for Class Balancing
    # -------------------------
    def compute_dynamic_alpha(self, true_labels, min_alpha=0.3, max_alpha=0.7):
        """
        Compute dynamic focal alpha based on actual class distribution in batch.
        
        Alpha weights the DOWN class (label=0), so:
        - If batch has more UP (label=1), alpha should be higher (weight DOWN more)
        - If batch has more DOWN (label=0), alpha should be lower (weight UP more)
        
        Args:
            true_labels: Binary labels [B]
            min_alpha: Minimum alpha (clips to prevent instability)
            max_alpha: Maximum alpha (clips to prevent instability)
        
        Returns:
            Dynamic alpha value clipped to [min_alpha, max_alpha]
        """
        true_labels = tf.cast(true_labels, tf.float32)
        
        # Compute proportion of UP class (label=1)
        up_ratio = tf.reduce_mean(true_labels)
        
        # Alpha = up_ratio means: weight DOWN inversely to its frequency
        # If up_ratio=0.6 (60% UP), alpha=0.6 → DOWN gets 0.6 weight, UP gets 0.4
        # This balances the classes
        alpha = up_ratio
        
        # Clip for stability
        alpha = tf.clip_by_value(alpha, min_alpha, max_alpha)
        
        return alpha

    # -------------------------
    # Point loss (log-cosh)
    # -------------------------
    def point_huber(self, y_true_scaled, y_pred_scaled, last_close_scaled=None, delta=None):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.point_huber(self, y_true_scaled, y_pred_scaled, last_close_scaled=last_close_scaled, delta=delta)


    # -------------------------
    # Local trend loss
    # -------------------------
    def local_trend_loss(self, x_window, y_true_raw, y_pred_raw, last_close_raw):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.local_trend_loss(self, x_window, y_true_raw, y_pred_raw, last_close_raw)


    # -------------------------
    # Extended & global trends
    # -------------------------
    def extended_trend_loss(self, x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.extended_trend_loss(self, x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw)

    # -------------------------
    # Combined custom loss (NEW: Per-horizon outputs with focal loss)
    # -------------------------
    def custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends,
                    vacuum_overflow=None):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends,
                                   vacuum_overflow=vacuum_overflow)

    def train_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        with tf.GradientTape() as tape:
            y_pred_list = self(x_window, training=True)
            # Named view for the 10 outputs (robust to future aux heads / reordering).
            heads = PredictiveOutputs(*y_pred_list)
            # Still provide the 9-tuple expected by current custom_loss signature + the vacuum separately.
            y_pred_9 = y_pred_list[:9]
            vac_overflow_pred = heads.vacuum_overflow
            loss_components = self.custom_loss(x_window, y_true, y_pred_9, last_close,
                                               extended_trends,
                                               vacuum_overflow=vac_overflow_pred)

        # loss_components is now a LossComponents NamedTuple (see registries/losses.py).
        # Positional unpack is preserved for compatibility; attribute access is also available.
        (total_loss_val,
         point_h0, point_h1, point_h2,
         local_h0, global_h0, extended_h0,
         local_h1, global_h1, extended_h1,
         local_h2, global_h2, extended_h2,
         dir_h0, dir_h1, dir_h2,
         nll_h0, nll_h1, nll_h2,
         reg_val, inter_reg, vol_loss,
         crps_h0, crps_h1, crps_h2,
         soft_ece_h0, soft_ece_h1, soft_ece_h2,
         t_perp_total, casimir_val, vac_val, hd_val, ife_val,
         vac_overflow_val) = loss_components

        grads = tape.gradient(total_loss_val, self.trainable_variables)

        # ---- Finite-gradient guard --------------------------------------------
        # One non-finite gradient anywhere used to poison EVERY weight in a single
        # step: tf.clip_by_global_norm computed a NaN global norm and rescaled every
        # gradient in the group by it. Zero the whole update instead, and count it.
        _present = [g for g in grads if g is not None]
        grad_global_norm = tf.linalg.global_norm(_present) if _present else tf.constant(0.0, dtype=tf.float32)
        step_finite = tf.math.is_finite(total_loss_val)
        if _present:
            step_finite = tf.logical_and(
                step_finite,
                tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in _present])),
            )
        self.nonfinite_grad_steps.update_state(tf.cast(tf.logical_not(step_finite), tf.float32))
        # tf.where, not `g * mask`: NaN * 0 is still NaN.
        grads = [None if g is None else tf.where(step_finite, g, tf.zeros_like(g)) for g in grads]

        # Split gradients into NN weights vs. indicator logit vars using id() set
        # (populated in __init__ from the layer's get_indicator_trainable_variables).
        # This replaces fragile substring matching on variable names.
        nn_gvs, ind_gvs = [], []
        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue
            (ind_gvs if id(v) in self._indicator_var_ids else nn_gvs).append((g, v))

        # Clip NN grads by global norm only (indicator grads are small scalars; Adam handles scale)
        # Also clip indicator grads for stability (high INDICATOR_LR_MULT + STE can produce large updates
        # on the scalar logit vars, leading to extreme alphas/periods and NaN cascade in features/preds).
        clip_norm = float(getattr(self.config, 'GRAD_CLIP_NORM', 0.0) or 0.0)
        if clip_norm > 0.0:
            if nn_gvs:
                nn_gs_clipped, _ = tf.clip_by_global_norm(
                    [g for g, v in nn_gvs], clip_norm)
                nn_gvs = list(zip(nn_gs_clipped, [v for g, v in nn_gvs]))
            if ind_gvs:
                ind_gs_clipped, _ = tf.clip_by_global_norm(
                    [g for g, v in ind_gvs], clip_norm)
                ind_gvs = list(zip(ind_gs_clipped, [v for g, v in ind_gvs]))

        # Apply gradients with separate optimizers
        self.optimizer.apply_gradients(nn_gvs)
        self.indicator_optimizer.apply_gradients(ind_gvs)

        # Clip learned indicator periods by delegating to the layer that owns them.
        # This encapsulates the period <-> logit conversion and removes duplicated
        # name-based string checks that used to live in train_step.
        min_p = self.config.MOMENTUM_CLIP_MIN
        max_p = self.config.MOMENTUM_CLIP_MAX
        if self._indicator_layer is not None:
            self._indicator_layer.clip_learned_periods(min_p, max_p)
        # Fallback for any legacy 'momentum_raw' style vars that might still be
        # attached directly to the base model (rare).
        for var in self.base_model.trainable_variables:
            if 'momentum_raw' in getattr(var, 'name', '').lower():
                p = tf.nn.softplus(var) + 1.0
                clipped = tf.clip_by_value(p, min_p, max_p)
                raw = tf.math.asinh((clipped - 1.0) / 2.0)
                var.assign(raw)

        # === COMPUTE DIRECTION METRICS FOR ALL 3 HORIZONS ===
        y_true = tf.cast(y_true, tf.float32)
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, 3] (delta_raw)
        last_close_squeeze = tf.squeeze(last_close, axis=1)
        # Match training direction labeling (including deadband if enabled)
        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        # One labelling rule for every path (neural_trade.metrics.tf_direction).
        mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2 = direction_labels_tf(
            y_true_raw, last_close_squeeze, deadband_bps, self.eps)

        # Extract direction predictions for all 3 horizons
        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred_9
        dir_pred_h0 = tf.squeeze(dir_pred_h0, axis=1)
        dir_pred_h1 = tf.squeeze(dir_pred_h1, axis=1)
        dir_pred_h2 = tf.squeeze(dir_pred_h2, axis=1)

        # Gaussian-implied P(up) from (mu, var): interpretable and consistent with regression.
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e3), tf.float32)
        var_h0_c = tf.clip_by_value(tf.squeeze(var_h0, axis=1), var_floor, var_cap)
        var_h1_c = tf.clip_by_value(tf.squeeze(var_h1, axis=1), var_floor, var_cap)
        var_h2_c = tf.clip_by_value(tf.squeeze(var_h2, axis=1), var_floor, var_cap)
        mu_h0 = tf.squeeze(price_h0, axis=1)
        mu_h1 = tf.squeeze(price_h1, axis=1)
        mu_h2 = tf.squeeze(price_h2, axis=1)
        # P(up | the move left the deadband): matches the masked labels (see losses.gaussian_up_prob_given_move).
        _lc = tf.squeeze(last_close, axis=1)
        gauss_p_up_h0 = _losses.gaussian_up_prob_given_move(mu_h0, var_h0_c, _lc, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h1 = _losses.gaussian_up_prob_given_move(mu_h1, var_h1_c, _lc, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h2 = _losses.gaussian_up_prob_given_move(mu_h2, var_h2_c, _lc, deadband, self.pred_mean, self.pred_scale, self.eps)

        logs = self._epoch_logs(
            loss_components, y_true, y_pred_9,
            (true_dir_h0, true_dir_h1, true_dir_h2),
            (dir_pred_h0, dir_pred_h1, dir_pred_h2),
            (gauss_p_up_h0, gauss_p_up_h1, gauss_p_up_h2),
            (mask_h0, mask_h1, mask_h2),
            head_prefix="train_", gauss_prefix="train_gauss_", training=True,
            grad_global_norm=grad_global_norm,
        )
        logs["nonfinite_grad_steps"] = self.nonfinite_grad_steps.result()
        return logs

    def _compute_direction_metrics(self, true_dir_h0, true_dir_h1, true_dir_h2, dir_pred_h0, dir_pred_h1, dir_pred_h2, mask_h0=None, mask_h1=None, mask_h2=None, prefix=""):
        """Direction metrics (acc, sensitivity, specificity, balanced acc, F1, MCC, Brier,
        positive-class ECE, predicted/true up rates, mean prob) over the given arrays, per horizon.

        Same formulas as the epoch accumulators (tf_direction.direction_metrics_from_stats); this form
        evaluates one set of arrays in full.
        """
        metrics = {}
        for h_name, true_dir, dir_pred, mask in (("h0", true_dir_h0, dir_pred_h0, mask_h0),
                                                 ("h1", true_dir_h1, dir_pred_h1, mask_h1),
                                                 ("h2", true_dir_h2, dir_pred_h2, mask_h2)):
            m = tf.ones_like(tf.cast(true_dir, tf.float32)) if mask is None else mask
            stats = direction_stats(*direction_counts(true_dir, dir_pred, m))
            metrics.update(direction_metrics_from_stats(stats, prefix, h_name,
                                                        getattr(self, '_step_metric_fns', None)))
        return metrics

    def test_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        y_pred_list = self(x_window, training=False)
        heads = PredictiveOutputs(*y_pred_list)
        y_pred_9 = y_pred_list[:9]
        vac_overflow_pred = heads.vacuum_overflow
        loss_components = self.custom_loss(x_window, y_true, y_pred_9, last_close,
                                           extended_trends,
                                           vacuum_overflow=None)  # identically 0 at eval (tanh^2 < E_max): the term would be a constant lambda in every val_loss

        # Unpack 34-component tuple (LossComponents NamedTuple; positional ok)
        (total_loss_val,
         point_h0, point_h1, point_h2,
         local_h0, global_h0, extended_h0,
         local_h1, global_h1, extended_h1,
         local_h2, global_h2, extended_h2,
         dir_h0, dir_h1, dir_h2,
         nll_h0, nll_h1, nll_h2,
         reg_val, inter_reg, vol_loss,
         crps_h0, crps_h1, crps_h2,
         soft_ece_h0, soft_ece_h1, soft_ece_h2,
         t_perp_total, casimir_val, vac_val, hd_val, ife_val,
         vac_overflow_val) = loss_components

        # Compute direction labels with the same trade-aware deadband used in training loss.
        y_true = tf.cast(y_true, tf.float32)
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, 3]
        last_close_squeeze = tf.squeeze(last_close, axis=1)

        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        # One labelling rule for every path (neural_trade.metrics.tf_direction).
        mask_h0, mask_h1, mask_h2, true_dir_h0, true_dir_h1, true_dir_h2 = direction_labels_tf(
            y_true_raw, last_close_squeeze, deadband_bps, self.eps)

        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred_9
        dir_pred_h0 = tf.squeeze(dir_pred_h0, axis=1)
        dir_pred_h1 = tf.squeeze(dir_pred_h1, axis=1)
        dir_pred_h2 = tf.squeeze(dir_pred_h2, axis=1)

        # Gaussian-implied P(up) from (mu, var)
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e3), tf.float32)
        var_h0_c = tf.clip_by_value(tf.squeeze(var_h0, axis=1), var_floor, var_cap)
        var_h1_c = tf.clip_by_value(tf.squeeze(var_h1, axis=1), var_floor, var_cap)
        var_h2_c = tf.clip_by_value(tf.squeeze(var_h2, axis=1), var_floor, var_cap)
        mu_h0 = tf.squeeze(price_h0, axis=1)
        mu_h1 = tf.squeeze(price_h1, axis=1)
        mu_h2 = tf.squeeze(price_h2, axis=1)
        # P(up | the move left the deadband): matches the masked labels (see losses.gaussian_up_prob_given_move).
        gauss_p_up_h0 = _losses.gaussian_up_prob_given_move(mu_h0, var_h0_c, last_close_squeeze, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h1 = _losses.gaussian_up_prob_given_move(mu_h1, var_h1_c, last_close_squeeze, deadband, self.pred_mean, self.pred_scale, self.eps)
        gauss_p_up_h2 = _losses.gaussian_up_prob_given_move(mu_h2, var_h2_c, last_close_squeeze, deadband, self.pred_mean, self.pred_scale, self.eps)

        # IMPORTANT: do NOT prefix with "val_" here. Keras automatically prefixes
        # validation metrics with "val_"; adding it ourselves creates "val_val_*" keys.
        return self._epoch_logs(
            loss_components, y_true, y_pred_9,
            (true_dir_h0, true_dir_h1, true_dir_h2),
            (dir_pred_h0, dir_pred_h1, dir_pred_h2),
            (gauss_p_up_h0, gauss_p_up_h1, gauss_p_up_h2),
            (mask_h0, mask_h1, mask_h2),
            head_prefix="", gauss_prefix="gauss_", training=False,
        )


    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'pred_scale': float(self.pred_scale.numpy()) if isinstance(self.pred_scale, tf.Tensor) else float(self.pred_scale),
            'pred_mean': float(self.pred_mean.numpy()) if isinstance(self.pred_mean, tf.Tensor) else float(self.pred_mean),
            'lambda_point': float(self.lambda_point),
            'lambda_local_trend': float(self.lambda_local_trend),
            'lambda_global_trend': float(self.lambda_global_trend),
            'lambda_extended_trend': float(self.lambda_extended_trend),
            'lambda_dir': float(self.lambda_dir)
        })
        return cfg

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config_instance = Config()
        predictor = PricePredictor(config_instance)
        base_model = predictor.build_model()
        pred_scale = config.pop('pred_scale', 1.0)
        pred_mean = config.pop('pred_mean', 0.0)
        lambda_point = config.pop('lambda_point', 1.0)
        lambda_local_trend = config.pop('lambda_local_trend', 1.0)
        lambda_global_trend = config.pop('lambda_global_trend', 0.2)
        lambda_extended_trend = config.pop('lambda_extended_trend', 0.16)
        lambda_dir = config.pop('lambda_dir', 1.0)
        instance = cls(base_model=base_model,
                       pred_scale=pred_scale,
                       pred_mean=pred_mean,
                       lambda_point=lambda_point,
                       lambda_local_trend=lambda_local_trend,
                       lambda_global_trend=lambda_global_trend,
                       lambda_extended_trend=lambda_extended_trend,
                       lambda_dir=lambda_dir,
                       config=config_instance,
                       **config)
        return instance

# ---- Loss-weight properties for CustomTrainModel ---------------------------------------
# Each `lambda_<key>` is a non-trainable tf.Variable kept in model._lambda_vars.
# `model.lambda_x` returns the Variable (usable directly inside the traced train step);
# `model.lambda_x = v` assigns in place, so schedules and ablations work after compile().
_LAMBDA_VARIABLE_KEYS = ('short', 'point', 'long', 'extended_trend', 'dir', 'var', 'vol',
                         'crps', 'soft_ece', 't_perp', 'casimir', 'hd', 'ife', 'vac_overflow')


def _make_lambda_property(key):
    name = f'lambda_{key}'

    def _get(self):
        return self._lambda_vars[key]

    def _set(self, value):
        var = self._lambda_vars.get(key)
        if var is None:
            self._lambda_vars[key] = tf.Variable(float(value), trainable=False, dtype=tf.float32, name=name)
        else:
            var.assign(float(value))

    return property(_get, _set, doc=f"Non-trainable tf.Variable weight for the '{key}' loss term.")


for _key in _LAMBDA_VARIABLE_KEYS:
    setattr(CustomTrainModel, f'lambda_{_key}', _make_lambda_property(_key))


def _get_lambda_values(self):
    """Current per-term loss weights as plain floats (for logging, ablation and export)."""
    return {f'lambda_{k}': float(v.numpy()) for k, v in self._lambda_vars.items()}


def _set_lambda_values(self, **weights):
    """Assign per-term loss weights in place, e.g. model.set_lambda_values(lambda_hd=0.0)."""
    for name, value in weights.items():
        if not name.startswith('lambda_') or name[len('lambda_'):] not in _LAMBDA_VARIABLE_KEYS:
            raise KeyError(f"unknown loss weight {name!r}; known: {[f'lambda_{k}' for k in _LAMBDA_VARIABLE_KEYS]}")
        setattr(self, name, value)


CustomTrainModel.get_lambda_values = _get_lambda_values
CustomTrainModel.set_lambda_values = _set_lambda_values


def _first_present(mapping, keys):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None

def _sum_present(mapping, keys):
    total = None
    for k in keys:
        if k not in mapping or mapping[k] is None:
            continue
        total = mapping[k] if total is None else (total + mapping[k])
    return total

def _mean_present(mapping, keys):
    total = None
    count = 0
    for k in keys:
        if k not in mapping or mapping[k] is None:
            continue
        total = mapping[k] if total is None else (total + mapping[k])
        count += 1
    if total is None or count == 0:
        return None
    return total / float(count)

def _qbox_dashboard_html(logs):
    """Return an HTML string for the T_⊥ / QBOX section of the epoch dashboard.

    Only renders if at least one QBOX component has a non-trivial value (> 1e-6),
    so the section stays hidden when all QBOX lambdas are 0.
    """
    components = [
        ('t_perp_loss',       'T⊥ calib',   'val_t_perp_loss'),
        ('casimir_loss',      'Casimir',     'val_casimir_loss'),
        ('vac_loss',          'Vac BW',      'val_vac_loss'),
        ('hd_loss',           'Hyper-Dec',   'val_hd_loss'),
        ('ife_loss',          'Info-Flow',   'val_ife_loss'),
        ('vac_overflow_loss', 'T⊥ Overflow', 'val_vac_overflow_loss'),
    ]
    rows = []
    for train_key, label, val_key in components:
        train_val = float(logs.get(train_key, 0.0))
        val_val   = float(logs.get(val_key,   0.0))
        if train_val > 1e-6 or val_val > 1e-6:
            rows.append(
                f'<span style="display: inline-block; width: 180px;">{label}:</span>'
                f' <span style="color: #CE93D8;">{train_val:.6f}</span>'
                f' <span style="color: #888; font-size: 11px;">val: {val_val:.6f}</span><br>'
            )
    if not rows:
        return ''
    inner = '\n                            '.join(rows)
    return f"""
                    <div style="margin-bottom: 15px;">
                        <div style="color: #CE93D8; font-weight: bold; margin-bottom: 8px;">⚛ T&#x22A5; / QBOX LOSSES</div>
                        <div style="margin-left: 15px;">
                            {inner}
                        </div>
                    </div>"""


def add_plot_aliases(logs, primary_horizon="h1", prefer_gauss=True):
    """Add plotting-friendly aliases into a Keras `logs` dict.

    The training code emits per-horizon metrics (e.g., `val_dir_f1_h1`, `train_dir_acc_h1`)
    and uses `nll_loss` for variance NLL. The notebook historically plotted legacy keys
    like `val_f1`, `val_dir_acc`, `var_nll`, and aggregated `*_trend_loss` fields.

    This helper keeps plotting code stable by:
    - Mapping per-horizon direction metrics to horizon-agnostic keys.
    - Computing legacy aggregate trend-loss keys from available components.
    - Aliasing `var_nll` -> `nll_loss`.

    It is safe to call on batch logs or epoch logs.
    """
    if logs is None:
        return {}
    out = dict(logs)

    def set_if_missing(key, value):
        if key not in out and value is not None:
            out[key] = value

    def set_agg(key, value):
        """Set aggregate-style aliases robustly.
        If value is None, remove any stale key from output to avoid carrying old values forward.
        Converts tensors/arrays to scalar floats by taking mean when needed.
        """
        if value is None:
            # Remove stale aggregate if no data present
            out.pop(key, None)
            return
        # Normalize to a scalar float when possible
        try:
            # TensorFlow tensors
            if hasattr(value, "numpy"):
                v = value.numpy()
            else:
                v = value
            v = np.asarray(v)
            if v.size == 1:
                out[key] = float(v.item())
            else:
                out[key] = float(np.mean(v))
        except Exception:
            try:
                out[key] = float(value)
            except Exception:
                out[key] = value

    # --- Loss aliases (legacy plotting names) ---
    set_if_missing("var_nll", _first_present(out, ["nll_loss"]))
    set_if_missing("val_var_nll", _first_present(out, ["val_nll_loss"]))

    set_if_missing("local_trend_loss", _sum_present(out, ["local_h0", "local_h1", "local_h2"]))
    set_if_missing("val_local_trend_loss", _sum_present(out, ["val_local_h0", "val_local_h1", "val_local_h2"]))

    set_if_missing("global_trend_loss", _sum_present(out, ["global_h0", "global_h1", "global_h2"]))
    set_if_missing("val_global_trend_loss", _sum_present(out, ["val_global_h0", "val_global_h1", "val_global_h2"]))

    set_if_missing("extended_trend_loss", _sum_present(out, ["extended_h0", "extended_h1", "extended_h2"]))
    set_if_missing("val_extended_trend_loss", _sum_present(out, ["val_extended_h0", "val_extended_h1", "val_extended_h2"]))

    # --- Direction metric aliases (primary horizon, head vs gauss preference) ---
    train_pref = "train_gauss_" if prefer_gauss else "train_"
    train_fallback = "train_" if prefer_gauss else "train_gauss_"

    val_pref = "val_gauss_" if prefer_gauss else "val_"
    val_fallback = "val_" if prefer_gauss else "val_gauss_"

    # Average across horizons (h0/h1/h2)
    horizons = ("h0", "h1", "h2")

    train_acc_keys = [f"{train_pref}dir_acc_{h}" for h in horizons]
    train_f1_keys = [f"{train_pref}dir_f1_{h}" for h in horizons]
    train_sens_keys = [f"{train_pref}dir_sensitivity_{h}" for h in horizons]
    train_spec_keys = [f"{train_pref}dir_specificity_{h}" for h in horizons]

    train_acc_fb = [f"{train_fallback}dir_acc_{h}" for h in horizons]
    train_f1_fb = [f"{train_fallback}dir_f1_{h}" for h in horizons]
    train_sens_fb = [f"{train_fallback}dir_sensitivity_{h}" for h in horizons]
    train_spec_fb = [f"{train_fallback}dir_specificity_{h}" for h in horizons]

    val_acc_keys = [f"{val_pref}dir_acc_{h}" for h in horizons]
    val_f1_keys = [f"{val_pref}dir_f1_{h}" for h in horizons]
    val_sens_keys = [f"{val_pref}dir_sensitivity_{h}" for h in horizons]
    val_spec_keys = [f"{val_pref}dir_specificity_{h}" for h in horizons]

    val_acc_fb = [f"{val_fallback}dir_acc_{h}" for h in horizons]
    val_f1_fb = [f"{val_fallback}dir_f1_{h}" for h in horizons]
    val_sens_fb = [f"{val_fallback}dir_sensitivity_{h}" for h in horizons]
    val_spec_fb = [f"{val_fallback}dir_specificity_{h}" for h in horizons]

    # MCC, Brier, ECE keys for averaging
    train_mcc_keys = [f"{train_pref}dir_mcc_{h}" for h in horizons]
    train_mcc_fb = [f"{train_fallback}dir_mcc_{h}" for h in horizons]
    train_brier_keys = [f"{train_pref}dir_brier_{h}" for h in horizons]
    train_brier_fb = [f"{train_fallback}dir_brier_{h}" for h in horizons]
    train_ece_keys = [f"{train_pref}dir_ece_{h}" for h in horizons]
    train_ece_fb = [f"{train_fallback}dir_ece_{h}" for h in horizons]
    # Balanced Accuracy keys for averaging
    train_bal_acc_keys = [f"{train_pref}dir_bal_acc_{h}" for h in horizons]
    train_bal_acc_fb = [f"{train_fallback}dir_bal_acc_{h}" for h in horizons]

    val_mcc_keys = [f"{val_pref}dir_mcc_{h}" for h in horizons]
    val_mcc_fb = [f"{val_fallback}dir_mcc_{h}" for h in horizons]
    val_brier_keys = [f"{val_pref}dir_brier_{h}" for h in horizons]
    val_brier_fb = [f"{val_fallback}dir_brier_{h}" for h in horizons]
    val_ece_keys = [f"{val_pref}dir_ece_{h}" for h in horizons]
    val_ece_fb = [f"{val_fallback}dir_ece_{h}" for h in horizons]
    # Balanced Accuracy keys for validation
    val_bal_acc_keys = [f"{val_pref}dir_bal_acc_{h}" for h in horizons]
    val_bal_acc_fb = [f"{val_fallback}dir_bal_acc_{h}" for h in horizons]

    set_agg("dir_acc_avg", _first_present({"v": _mean_present(out, train_acc_keys), "v2": _mean_present(out, train_acc_fb)}, ["v", "v2"]))
    set_agg("f1_avg", _first_present({"v": _mean_present(out, train_f1_keys), "v2": _mean_present(out, train_f1_fb)}, ["v", "v2"]))
    set_agg("dir_sensitivity_avg", _first_present({"v": _mean_present(out, train_sens_keys), "v2": _mean_present(out, train_sens_fb)}, ["v", "v2"]))
    set_agg("dir_specificity_avg", _first_present({"v": _mean_present(out, train_spec_keys), "v2": _mean_present(out, train_spec_fb)}, ["v", "v2"]))
    # MCC, Brier, ECE averages (class-imbalance robust metrics)
    set_agg("mcc_avg", _first_present({"v": _mean_present(out, train_mcc_keys), "v2": _mean_present(out, train_mcc_fb)}, ["v", "v2"]))
    set_agg("brier_avg", _first_present({"v": _mean_present(out, train_brier_keys), "v2": _mean_present(out, train_brier_fb)}, ["v", "v2"]))
    set_agg("ece_avg", _first_present({"v": _mean_present(out, train_ece_keys), "v2": _mean_present(out, train_ece_fb)}, ["v", "v2"]))
    # Balanced Accuracy average (class-imbalance robust, 50% = random, range [0,1])
    set_agg("bal_acc_avg", _first_present({"v": _mean_present(out, train_bal_acc_keys), "v2": _mean_present(out, train_bal_acc_fb)}, ["v", "v2"]))

    set_agg("val_dir_acc_avg", _first_present({"v": _mean_present(out, val_acc_keys), "v2": _mean_present(out, val_acc_fb)}, ["v", "v2"]))
    set_agg("val_f1_avg", _first_present({"v": _mean_present(out, val_f1_keys), "v2": _mean_present(out, val_f1_fb)}, ["v", "v2"]))
    set_agg("val_dir_sensitivity_avg", _first_present({"v": _mean_present(out, val_sens_keys), "v2": _mean_present(out, val_sens_fb)}, ["v", "v2"]))
    set_agg("val_dir_specificity_avg", _first_present({"v": _mean_present(out, val_spec_keys), "v2": _mean_present(out, val_spec_fb)}, ["v", "v2"]))
    # Validation MCC, Brier, ECE averages
    set_agg("val_mcc_avg", _first_present({"v": _mean_present(out, val_mcc_keys), "v2": _mean_present(out, val_mcc_fb)}, ["v", "v2"]))
    set_agg("val_brier_avg", _first_present({"v": _mean_present(out, val_brier_keys), "v2": _mean_present(out, val_brier_fb)}, ["v", "v2"]))
    set_agg("val_ece_avg", _first_present({"v": _mean_present(out, val_ece_keys), "v2": _mean_present(out, val_ece_fb)}, ["v", "v2"]))
    # Validation Balanced Accuracy average
    set_agg("val_bal_acc_avg", _first_present({"v": _mean_present(out, val_bal_acc_keys), "v2": _mean_present(out, val_bal_acc_fb)}, ["v", "v2"]))

    # Batch-level aliases used by batch plot
    set_if_missing(
        "dir_acc",
        _first_present(
            out,
            [
                f"{train_pref}dir_acc_{primary_horizon}",
                f"{train_fallback}dir_acc_{primary_horizon}",
                f"dir_acc_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "f1",
        _first_present(
            out,
            [
                f"{train_pref}dir_f1_{primary_horizon}",
                f"{train_fallback}dir_f1_{primary_horizon}",
                f"dir_f1_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "dir_mcc",
        _first_present(
            out,
            [
                f"{train_pref}dir_mcc_{primary_horizon}",
                f"{train_fallback}dir_mcc_{primary_horizon}",
                f"dir_mcc_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "dir_sensitivity",
        _first_present(out, [f"{train_pref}dir_sensitivity_{primary_horizon}", f"{train_fallback}dir_sensitivity_{primary_horizon}"]),
    )
    set_if_missing(
        "dir_specificity",
        _first_present(out, [f"{train_pref}dir_specificity_{primary_horizon}", f"{train_fallback}dir_specificity_{primary_horizon}"]),
    )

    # Epoch-level aliases used by validation metrics plot
    set_if_missing(
        "val_dir_acc",
        _first_present(
            out,
            [
                f"{val_pref}dir_acc_{primary_horizon}",
                f"{val_fallback}dir_acc_{primary_horizon}",
                f"dir_acc_{primary_horizon}",
                "val_dir_acc",
            ],
        ),
    )
    set_if_missing(
        "val_f1",
        _first_present(
            out,
            [
                f"{val_pref}dir_f1_{primary_horizon}",
                f"{val_fallback}dir_f1_{primary_horizon}",
                f"dir_f1_{primary_horizon}",
                "val_f1",
            ],
        ),
    )
    set_if_missing(
        "val_dir_mcc",
        _first_present(
            out,
            [
                f"{val_pref}dir_mcc_{primary_horizon}",
                f"{val_fallback}dir_mcc_{primary_horizon}",
                f"dir_mcc_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "val_dir_sensitivity",
        _first_present(
            out,
            [
                f"{val_pref}dir_sensitivity_{primary_horizon}",
                f"{val_fallback}dir_sensitivity_{primary_horizon}",
                f"dir_sensitivity_{primary_horizon}",
            ],
        ),
    )
    set_if_missing(
        "val_dir_specificity",
        _first_present(
            out,
            [
                f"{val_pref}dir_specificity_{primary_horizon}",
                f"{val_fallback}dir_specificity_{primary_horizon}",
                f"dir_specificity_{primary_horizon}",
            ],
        ),
    )

    # Back-compat: treat recall as sensitivity for UP class
    set_if_missing("val_recall", out.get("val_dir_sensitivity"))

    # Prefer avg metrics for legacy val_* keys if present
    set_if_missing("val_f1", out.get("val_f1_avg"))
    set_if_missing("val_dir_acc", out.get("val_dir_acc_avg"))
    set_if_missing("val_dir_sensitivity", out.get("val_dir_sensitivity_avg"))
    set_if_missing("val_dir_specificity", out.get("val_dir_specificity_avg"))

    # --- QBOX / T_⊥ aggregate alias ---
    # Sum all active T_⊥ loss components so the dashboard can plot a single trend line.
    # Components with lambda=0 contribute 0, so this is safe even when losses are off.
    _qbox_train_keys = ['t_perp_loss', 'casimir_loss', 'vac_loss', 'hd_loss', 'ife_loss', 'vac_overflow_loss']
    _qbox_val_keys   = [f'val_{k}' for k in _qbox_train_keys]
    _qbox_train_sum  = sum(float(out[k]) for k in _qbox_train_keys if k in out)
    _qbox_val_sum    = sum(float(out[k]) for k in _qbox_val_keys   if k in out)
    set_if_missing('qbox_loss',     _qbox_train_sum if _qbox_train_sum > 0 else None)
    set_if_missing('val_qbox_loss', _qbox_val_sum   if _qbox_val_sum   > 0 else None)

    return out


# -----------------------------
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


class SimpleLoggingCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch+1}: local_trend={logs.get('local_trend', 0):.6f}, loss={logs.get('loss', 0):.6f}, point_loss={logs.get('point_loss', 0):.6f}, reg_loss={logs.get('reg_loss', 0):.6f}, trend_loss={logs.get('trend_loss', 0):.6f}, val_local_trend={logs.get('val_local_trend', 0):.6f}, val_loss={logs.get('val_loss', 0):.6f}, val_point_loss={logs.get('val_point_loss', 0):.6f}, val_reg_loss={logs.get('val_reg_loss', 0):.6f}, val_trend_loss={logs.get('val_trend_loss', 0):.6f}")

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
            print("\n[Summary: Per-Horizon Delta Metrics]")
            for h_key, label in zip(m.get('meta', {}).get('horizon_keys', ['h0','h1','h2']), m.get('meta', {}).get('horizon_labels', ['1min','5min','15min'])):
                hm = m['delta'].get(h_key, {})
                pm = m['price'].get(h_key, {})
                print(f"  {label}: MSE={hm.get('mse'):.6f}, RMSE={hm.get('rmse'):.6f}, R2={pm.get('r2', hm.get('r2')):.6f}")
    except Exception:
        pass

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
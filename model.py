csv="Bitcoin_BTCUSDT.csv"

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, losses, initializers, regularizers
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import time
from tqdm import tqdm

try:
    # Optional local utilities (kept lightweight). If missing, fall back to sklearn MAPE only.
    from metrics_utils import safe_mape, smape, wape, reconstruct_prices, mask_by_min_abs_y
except Exception:
    safe_mape = None
    smape = None
    wape = None
    reconstruct_prices = None
    mask_by_min_abs_y = None

# -----------------------------
class Config:

    HOUR = 60
    DAY = HOUR * 24

    # Data Configuration tuned for minute-level data
    CSV_PATH = csv
    LOOKBACK = HOUR   # Reduced to 1 hour of minute data
    WINDOW_STEP = 1  # Generate a training sample every minute for true minute-level modeling
    RESAMPLE_MINUTES = None  # Optionally aggregate to coarser bars (e.g., set to 5 for 5-minute bars)
    BATCH_SIZE = 360
    EPOCHS = 80
    LR = 1e-4  # Fixed from critically low 1e-10; reasonable for Adam optimizer
    PATIENCE = EPOCHS
    MAX_SEQUENCE_COUNT = 1440  # Limit most recent sequences to bound training size



    # Use integer periods to avoid float indexing issues
    # Extended trend features are computed as percent-change over these lags (in minutes)
    EXTENDED_TREND_PERIODS = [1, 5, 15]  # 1m, 5m, 15m

    # Supervision horizons (in minutes ahead from last_close). These define the 3 output towers.
    # h0=1m, h1=5m, h2=15m.
    HORIZON_STEPS = [1, 5, 15]

# Loss Function Weights
    DAMPING = 0.5
    LAMBDA_POINT = 1.0  
    LAMBDA_LOCAL_TREND  = 1.0
    LAMBDA_GLOBAL_TREND =  1.0
    LAMBDA_EXTENDED_TREND = 1.0
    LAMBDA_QUANTILE = 1.0
    REG_MOMENTUM_L2 = 1e-4
    MOMENTUM_CLIP_MIN = 1.0
    MOMENTUM_CLIP_MAX = LOOKBACK
    USE_HUBER = True
    
    # === HORIZON-SPECIFIC LOSS WEIGHTS ===
    # Per-horizon lambda weights for point loss (delta prediction accuracy)
    # Different horizons may have different importance/difficulty:
    # - h0 (1-min): Short-term noise, harder to predict, may need lower weight to avoid overfitting noise
    # - h1 (5-min): Primary horizon, balanced signal/noise, standard weight
    # - h2 (15-min): Long-term trend, more stable, higher weight to enforce consistency
    LAMBDA_SHORT = 0.8   # h0 (1-min):  Reduced from 1.0 to avoid noise overfitting
    LAMBDA_POINT = 1.0   # h1 (5-min):  Primary horizon baseline
    LAMBDA_LONG = 1.2    # h2 (15-min): Increased from 1.0 to enforce long-term consistency
    
    # Auxiliary loss weights
    LAMBDA_DIR = 1.0  # Direction classification (focal loss)
    LAMBDA_INTER = 1.0  # Interconnection regularization between horizons
    LAMBDA_VOL = 1.0  # Volatility penalty (weak constraint)
    LAMBDA_VAR = 1.0  # Variance NLL (confidence estimation)

# paths
    # MODEL_PATH v2: Major architectural refactor for multi-horizon direction classification
    # - 3 independent output towers (h0_1min, h1_5min, h2_15min) instead of shared heads
    # - 9 outputs (3 price + 3 direction + 3 variance) vs 3 outputs (price, direction, variance)
    # - Focal loss for direction heads with α=0.7 focusing on minority class (DOWN moves)
    # - Per-horizon direction metrics: accuracy, F1, sensitivity, specificity, MCC
    # - MCC-based early stopping monitors val_dir_mcc_h1 (primary horizon) for optimal trade-off
    # v3: true multi-horizon supervision (separate targets per tower)
    MODEL_PATH = "nn_learnable_indicators_v3.weights.h5"
    SCALER_PATH = "scaler_v3.joblib"

# TA initial params
    MA_SPANS = [5, 15, 30] 
    MACD_SETTINGS = [
        {'fast': 15, 'slow': 30, 'signal': 5},
        {'fast': 45, 'slow': 60, 'signal': 30},
        {'fast': 30, 'slow': 45, 'signal': 15}
    ]
    CUSTOM_MACD_PAIRS = [(30, 60), (15, 30), (25, 45)] 
    MOMENTUM_PERIODS = [1, 2, 3]  
    RSI_PERIODS = [9, 21, 30]  
    BB_PERIODS = [10, 20, 50]  

# Activation function settings
    TANH_SCALE = 1.0
    HUBER_DELTA = 1.0
    SIGMOID_SCALE = 1.0

# Training stability controls
    INDICATOR_GRAD_MULT = 10.0
    GRAD_CLIP_NORM = 5.0

# Focal loss hyperparameters for direction classification
    # NOTE: alpha weights DOWN class (label=0), (1-alpha) weights UP (label=1)
    # Class weighting was NOT the cause of DOWN bias - root causes were:
    # 1. Weak direction loss weight (fixed: 0.2 → 0.5)
    # 2. Zero deadband creating label noise (fixed: 5 bps)
    FOCAL_ALPHA = 0.5  # Balanced class weights (was 0.7, now neutral)
    FOCAL_GAMMA = 2.0  # Focus parameter for hard examples

    # Trade-aware direction labeling deadband.
    # If > 0, direction loss/metrics treat returns within +/- deadband as neutral.
    # Units: basis points (bps). Example: 10 bps = 0.10%.
    # CRITICAL FIX: Non-zero deadband filters label noise from tiny price moves
    DIR_DEADBAND_BPS = 5.0  # 5 bps = 0.05% minimum move for UP classification

    # Stabilize NLL and prevent variance head from dominating early.
    # Variance is in SCALED units^2.
    VAR_FLOOR = 1e-4
    VAR_CAP = 1e4

    # Align direction head with distribution-implied P(up) from (mu, var).
    # Setting this > 0 helps avoid degenerate constant direction probabilities.
    LAMBDA_DIR_ALIGN = 0.1
# -----------------------------
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

        # Parse timestamp column (minute-level data format)
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

        tscv = TimeSeriesSplit(n_splits=5)
        # self.plot_splits(df, start_idx=max(self.config.LOOKBACK, max(self.config.EXTENDED_TREND_PERIODS)),
        #                 tscv=tscv, X_seq_len=len(X_seq))

        train_indices, test_indices = list(tscv.split(X_seq))[-1]

        X_train_seq, X_test_seq = X_seq[train_indices], X_seq[test_indices]
        y_train, y_test = y_seq[train_indices], y_seq[test_indices]
        last_close_train, last_close_test = last_close_seq[train_indices], last_close_seq[test_indices]
        extended_trends_train, extended_trends_test = extended_trends[train_indices], extended_trends[test_indices]

        train_batches = math.ceil(X_train_seq.shape[0] / self.config.BATCH_SIZE)
        test_batches = math.ceil(X_test_seq.shape[0] / self.config.BATCH_SIZE)
        print(f"   Train sequences: {X_train_seq.shape[0]} (batches/epoch: {train_batches})")
        print(f"   Test sequences: {X_test_seq.shape[0]} (batches: {test_batches})")

        # Targets are multi-horizon: shape [N, 3].
        # Use a single scaler fit on ALL horizons (flattened) to keep consistent scaling.
        target_scaler = StandardScaler()
        y_train_flat = y_train.reshape(-1, 1)
        y_test_flat = y_test.reshape(-1, 1)
        y_train_scaled = target_scaler.fit_transform(y_train_flat).reshape(y_train.shape)
        y_test_scaled = target_scaler.transform(y_test_flat).reshape(y_test.shape)

        # Scale input sequences for better normalization
        input_scaler = StandardScaler()
        X_train_seq_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[-1])
        X_train_seq_scaled = input_scaler.fit_transform(X_train_seq_reshaped).reshape(X_train_seq.shape)
        X_test_seq_reshaped = X_test_seq.reshape(-1, X_test_seq.shape[-1])
        X_test_seq_scaled = input_scaler.transform(X_test_seq_reshaped).reshape(X_test_seq.shape)

        joblib.dump(target_scaler, self.config.SCALER_PATH)
        joblib.dump(input_scaler, self.config.SCALER_PATH.replace('.joblib', '_input.joblib'))

        # Keep references for programmatic use without changing the return signature.
        self.target_scaler = target_scaler
        self.input_scaler = input_scaler

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


def _apply_config_overrides(config: 'Config', overrides: Optional[dict]) -> 'Config':
    if not overrides:
        return config
    for k, v in dict(overrides).items():
        setattr(config, k, v)
    return config


def _compute_all_horizon_metrics(
    *,
    config: 'Config',
    y_true_deltas: np.ndarray,
    y_pred_deltas: Dict[str, np.ndarray],
    last_close: np.ndarray,
    dir_probs: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute consistent metrics for all horizons.

    Returns a dict with per-horizon delta-space metrics, price-space metrics, and direction metrics.
    """

    horizons = ("h0", "h1", "h2")
    horizon_names = ("1-min", "5-min", "15-min")
    y_true_deltas = np.asarray(y_true_deltas)
    if y_true_deltas.ndim != 2 or y_true_deltas.shape[1] != 3:
        raise ValueError(f"Expected y_true_deltas shape (N,3), got {y_true_deltas.shape}")
    lc = np.asarray(last_close, dtype=float).reshape(-1)

    out: Dict[str, Any] = {
        "delta": {},
        "price": {},
        "direction": {},
    }

    deadband_bps = float(getattr(config, 'DIR_DEADBAND_BPS', 0.0))
    deadband = deadband_bps / 10000.0
    threshold_delta = deadband * (lc + 1e-12)
    min_abs_delta_for_mape = float(getattr(config, 'DELTA_MAPE_MIN_ABS', 1.0))

    for idx, (h_key, h_label) in enumerate(zip(horizons, horizon_names)):
        y_t = np.asarray(y_true_deltas[:, idx], dtype=float).reshape(-1)
        y_p = np.asarray(y_pred_deltas[h_key], dtype=float).reshape(-1)
        thr = np.asarray(threshold_delta, dtype=float).reshape(-1)
        n = min(len(y_t), len(y_p), len(lc), len(thr))
        y_t = y_t[:n]
        y_p = y_p[:n]
        lc_h = lc[:n]
        thr = thr[:n]

        # Delta-space metrics (raw price differences)
        mse_delta = mean_squared_error(y_t, y_p)
        rmse_delta = float(np.sqrt(mse_delta))
        # Note: R2 in delta space can be negative (worse than predicting mean).
        # This is expected when predictions are poor relative to delta variance.
        r2_delta = r2_score(y_t, y_p)

        delta_metrics = {
            "mse": float(mse_delta),
            "rmse": float(rmse_delta),
            "r2": float(r2_delta),
        }
        if safe_mape is not None and smape is not None and wape is not None and reconstruct_prices is not None:
            delta_metrics.update({
                "mape_delta": float(mean_absolute_percentage_error(y_t, y_p)),
                "safe_mape_delta": float(safe_mape(y_t, y_p, min_abs_y=min_abs_delta_for_mape)),
                "smape_delta": float(smape(y_t, y_p)),
                "wape_delta": float(wape(y_t, y_p)),
            })

        out["delta"][h_key] = delta_metrics

        # CRITICAL: Price-space R2 is the most interpretable metric for price prediction.
        # Reconstruct prices: price[t+h] = last_close[t] + delta[t, t+h]
        # R2 in price space measures how well cumulative predictions track actual future prices.
        y_true_price = lc_h + y_t  # Simple reconstruction: last_close + delta
        y_pred_price = lc_h + y_p

        # In price space, R2 is more stable because:
        # 1. Price levels have larger variance than deltas
        # 2. R2 measures the fraction of price-level variance explained
        # 3. This aligns with trading objectives (predicting future prices, not just changes)
        r2_price_simple = r2_score(y_true_price, y_pred_price)
        
        price_metrics = {
            "r2": float(r2_price_simple),  # R2 in price space (primary metric)
            "mse": float(mean_squared_error(y_true_price, y_pred_price)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        }
        
        if safe_mape is not None and smape is not None and wape is not None and reconstruct_prices is not None:
            # Use the more sophisticated reconstruction if available for additional metrics
            y_true_price_soph = reconstruct_prices(lc_h, y_t)
            y_pred_price_soph = reconstruct_prices(lc_h, y_p)
            price_metrics.update({
                "r2_soph": float(r2_score(y_true_price_soph, y_pred_price_soph)),
                "mape": float(safe_mape(y_true_price_soph, y_pred_price_soph)),
                "smape": float(smape(y_true_price_soph, y_pred_price_soph)),
                "wape": float(wape(y_true_price_soph, y_pred_price_soph)),
            })

        out["price"][h_key] = price_metrics

        true_dir = (y_t > thr)
        if dir_probs is not None and h_key in dir_probs and dir_probs[h_key] is not None:
            p = np.asarray(dir_probs[h_key], dtype=float).reshape(-1)[:n]
            pred_dir = (p >= 0.5)
        else:
            pred_dir = (y_p > thr)

        out["direction"][h_key] = {
            "acc": float(accuracy_score(true_dir.astype(int), pred_dir.astype(int))),
            "f1": float(f1_score(true_dir.astype(int), pred_dir.astype(int), zero_division=0)),
        }

    out["meta"] = {
        "horizon_keys": list(horizons),
        "horizon_labels": list(horizon_names),
        "deadband_bps": float(deadband_bps),
        "delta_safe_mape_min_abs": float(min_abs_delta_for_mape),
    }
    return out


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
                if 'f1_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['f1_avg'], mode='lines+markers', name='f1_avg', line=dict(color='#FFB74D')), row=2, col=1)
                if 'val_f1_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['val_f1_avg'], mode='lines+markers', name='val_f1_avg', line=dict(color='#FFA726')), row=2, col=1)
                
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
                
                # Coherence: correlation between train and val metrics
                coherence = 0.0
                if len(loss_hist) >= 3 and len(val_loss_hist) >= 3:
                    try:
                        coherence = np.corrcoef(loss_hist[-10:], val_loss_hist[-10:])[0, 1]
                        coherence = coherence if not np.isnan(coherence) else 0.0
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
                            <span style="display: inline-block; width: 180px;">f1_avg:</span> <span style="color: #FFB74D;">{logs.get('f1_avg', 0):.4f}</span><br>
                            <span style="display: inline-block; width: 180px;">val_f1_avg:</span> <span style="color: #FFA726;">{logs.get('val_f1_avg', 0):.4f}</span><br>
                        </div>
                    </div>
                    
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


def train_and_evaluate(
    *,
    config: Optional['Config'] = None,
    config_overrides: Optional[dict] = None,
    csv_path: Optional[str] = None,
    read_csv_kwargs: Optional[dict] = None,
    epochs: Optional[int] = None,
    force: bool = False,
    calibrate: bool = True,
    extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> TrainResult:
    """Train (optionally) and evaluate, returning a rich result bundle.

    This is intended to be the notebook's single source of truth for:
    - data prep + scaling
    - model heads and extraction
    - metrics and evaluation semantics
    """

    tf.keras.utils.set_random_seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    cfg = config or Config()
    if csv_path is not None:
        cfg.CSV_PATH = csv_path
    cfg = _apply_config_overrides(cfg, config_overrides)

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

    train_ds, val_ds = predictor.create_datasets(
        X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
        X_test_seq, y_test_scaled, last_close_test, extended_trends_test,
    )

    # Optional calibration (kept identical to train_model behavior)
    if calibrate is True:
        try:
            orig_lambda_point = custom_model.lambda_point
            orig_lambda_local = custom_model.lambda_local_trend
            orig_lambda_global = custom_model.lambda_global_trend
            orig_lambda_ext = custom_model.lambda_extended_trend
            orig_lambda_dir = custom_model.lambda_dir
            orig_lambda_var = custom_model.lambda_var

            custom_model.lambda_point = 1.0
            custom_model.lambda_local_trend = 1.0
            custom_model.lambda_global_trend = 1.0
            custom_model.lambda_extended_trend = 1.0
            custom_model.lambda_dir = 1.0
            custom_model.lambda_var = 1.0

            n_calib_batches = Config.BATCH_SIZE
            print("Warming up BatchNorm statistics for accurate calibration...")
            for batch in train_ds.take(n_calib_batches):
                x_batch, _, _, _ = batch
                _ = custom_model(x_batch, training=True)

            point_losses, local_losses, global_losses, ext_losses, dir_losses, var_losses = [], [], [], [], [], []
            for batch in train_ds.take(n_calib_batches):
                x_batch, y_batch, last_batch, ext_batch = batch
                y_pred_batch = custom_model(x_batch, training=False)
                (total,
                 point_h0, point_h1, point_h2,
                 local_h0, global_h0, ext_h0,
                 local_h1, global_h1, ext_h1,
                 local_h2, global_h2, ext_h2,
                 dir_h0, dir_h1, dir_h2,
                 nll_h0, nll_h1, nll_h2,
                 reg_val, inter_reg, vol_loss) = custom_model.custom_loss(
                    x_batch, y_batch, y_pred_batch, last_batch, ext_batch
                )

                point_losses.append(float((point_h0 + point_h1 + point_h2) / 3.0))
                local_losses.append(float((local_h0 + local_h1 + local_h2) / 3.0))
                global_losses.append(float((global_h0 + global_h1 + global_h2) / 3.0))
                ext_losses.append(float((ext_h0 + ext_h1 + ext_h2) / 3.0))
                dir_losses.append(float((dir_h0 + dir_h1 + dir_h2) / 3.0))
                var_losses.append(float((nll_h0 + nll_h1 + nll_h2) / 3.0))

            med_point = float(np.median(np.array(point_losses))) if point_losses else 0.0
            med_local = float(np.median(np.array(local_losses))) if local_losses else 0.0
            med_global = float(np.median(np.array(global_losses))) if global_losses else 0.0
            med_ext = float(np.median(np.array(ext_losses))) if ext_losses else 0.0
            med_dir = float(np.median(np.array(dir_losses))) if dir_losses else 0.0
            med_var = float(np.median(np.array(var_losses))) if var_losses else 0.0

            non_zero_medians = [m for m in [med_point, med_local, med_global, med_ext, med_dir, med_var] if m > 1e-8]
            ref_loss = float(np.mean(non_zero_medians)) if non_zero_medians else 1.0

            eps = 1e-8
            damping = Config.DAMPING
            new_point = orig_lambda_point * (ref_loss / (med_point + eps)) ** damping if med_point > eps else orig_lambda_point
            new_local = orig_lambda_local * (ref_loss / (med_local + eps)) ** damping if med_local > eps else orig_lambda_local
            new_global = orig_lambda_global * (ref_loss / (med_global + eps)) ** damping if med_global > eps else orig_lambda_global
            new_ext = orig_lambda_ext * (ref_loss / (med_ext + eps)) ** damping if med_ext > eps else orig_lambda_ext
            new_dir = orig_lambda_dir * (ref_loss / (med_dir + eps)) ** damping if med_dir > eps else orig_lambda_dir
            new_var = orig_lambda_var * (ref_loss / (med_var + eps)) ** damping if med_var > eps else orig_lambda_var

            min_lambda = 0.1
            max_lambda = 20
            custom_model.lambda_point = float(np.clip(new_point, min_lambda, max_lambda))
            custom_model.lambda_local_trend = float(np.clip(new_local, min_lambda, max_lambda))
            custom_model.lambda_global_trend = float(np.clip(new_global, min_lambda, max_lambda))
            custom_model.lambda_extended_trend = float(np.clip(new_ext, min_lambda, max_lambda))
            custom_model.lambda_dir = float(np.clip(new_dir, min_lambda, max_lambda))
            custom_model.lambda_var = float(np.clip(new_var, min_lambda, max_lambda))

            print(f"Calibration (multi-batch median with actual losses): "
                  f"med_point={med_point:.6f}, med_local={med_local:.6f}, "
                  f"med_global={med_global:.6f}, med_ext={med_ext:.6f}, med_dir={med_dir:.6f}, med_var={med_var:.6f}, ref_loss={ref_loss:.6f}")
            print(f"New lambdas (damped+clamped): "
                  f"point={custom_model.lambda_point:.6f}, local={custom_model.lambda_local_trend:.6f}, "
                  f"global={custom_model.lambda_global_trend:.6f}, ext={custom_model.lambda_extended_trend:.6f}, dir={custom_model.lambda_dir:.6f}, var={custom_model.lambda_var:.6f}")
        except Exception as e:
            print("Calibration pass failed, proceeding with default lambdas:", e)

    opt = optimizers.Adam(learning_rate=cfg.LR)
    custom_model.compile(optimizer=opt)

    csv_logger = callbacks.CSVLogger("training_log.csv", append=True)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=Config.PATIENCE, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(cfg.MODEL_PATH, save_best_only=True, monitor='val_loss', save_weights_only=True)
    es_dir = callbacks.EarlyStopping(
        monitor='val_dir_acc_h1',
        patience=Config.PATIENCE,
        mode='max',
        restore_best_weights=False
    )
    tqdm_callback = TqdmCallback()
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=Config.PATIENCE)

    learnable_layer = None
    for layer in custom_model.layers:
        if getattr(layer, 'name', '').startswith('learnable_indicators'):
            learnable_layer = layer
            break
    params_logger = ParamsLogger(layer=learnable_layer, out_csv='indicator_params_history.csv')

    callbacks_list = [csv_logger, es, ckpt, es_dir, tqdm_callback, params_logger, lr_scheduler]
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
    X_test_simple = tf.data.Dataset.from_tensor_slices(X_test_seq).batch(cfg.BATCH_SIZE)
    y_pred_all = custom_model.predict(X_test_simple)

    # Extract 3 price heads (scaled deltas from model outputs)
    # CRITICAL: These are SCALED predictions (trained in scaled delta space)
    y_pred_price_scaled = np.column_stack([
        y_pred_all[0][:, 0],
        y_pred_all[3][:, 0],
        y_pred_all[6][:, 0],
    ])
    y_pred_price_scaled = y_pred_price_scaled[:len(y_test)]

    # Inverse-transform from scaled space back to raw delta space
    # This ensures predictions have the same statistical properties as the original deltas
    y_pred_h0_raw = target_scaler.inverse_transform(y_pred_price_scaled[:, 0].reshape(-1, 1)).ravel()
    y_pred_h1_raw = target_scaler.inverse_transform(y_pred_price_scaled[:, 1].reshape(-1, 1)).ravel()
    y_pred_h2_raw = target_scaler.inverse_transform(y_pred_price_scaled[:, 2].reshape(-1, 1)).ravel()

    # === DIAGNOSTIC: Check prediction quality ===
    # Print statistics to help diagnose issues
    print("\n[Diagnostic: Prediction Statistics]")
    for h_idx, (h_name, y_pred_raw) in enumerate([('h0(1m)', y_pred_h0_raw), ('h1(5m)', y_pred_h1_raw), ('h2(15m)', y_pred_h2_raw)]):
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

    dir_pred_h0 = np.asarray(y_pred_all[1]).reshape(-1)[:len(y_test)]
    dir_pred_h1 = np.asarray(y_pred_all[4]).reshape(-1)[:len(y_test)]
    dir_pred_h2 = np.asarray(y_pred_all[7]).reshape(-1)[:len(y_test)]

    var_pred_h0 = np.asarray(y_pred_all[2]).reshape(-1)[:len(y_test)]
    var_pred_h1 = np.asarray(y_pred_all[5]).reshape(-1)[:len(y_test)]
    var_pred_h2 = np.asarray(y_pred_all[8]).reshape(-1)[:len(y_test)]

    predictions = {
        "delta": {"h0": y_pred_h0_raw, "h1": y_pred_h1_raw, "h2": y_pred_h2_raw},
        "direction_prob": {"h0": dir_pred_h0, "h1": dir_pred_h1, "h2": dir_pred_h2},
        "variance": {"h0": var_pred_h0, "h1": var_pred_h1, "h2": var_pred_h2},
    }

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
    )

# -----------------------------
def differentiable_past_sample(x_seq, fractional_lag, sigma=1.0):
    batch = tf.shape(x_seq)[0]
    time = tf.shape(x_seq)[1]
    positions = tf.cast(tf.range(time), tf.float32)
    T_minus_1 = tf.cast(time - 1, tf.float32)
    pos_target = T_minus_1 - fractional_lag
    pos_target_exp = tf.expand_dims(pos_target, axis=1)
    positions_exp = tf.reshape(positions, (1, -1))
    dists = positions_exp - pos_target_exp
    weights = tf.exp(-0.5 * tf.square(dists / sigma))
    weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-12)
    sampled = tf.reduce_sum(weights * x_seq, axis=1)
    return sampled

# -----------------------------
class LearnableIndicators(layers.Layer):
    def __init__(self, config: Config, gaussian_sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.epsilon = 1e-8
        self.gaussian_sigma = gaussian_sigma
        self.alpha_vars_ma = []
        self.macd_alpha_vars = {}
        self.rsi_alpha_vars = []
        self.bb_alpha_vars = []
        self.momentum_raw = []
        self.all_logit_vars = []  # New: collect all for metacalibration
        self.meta_scale = 0.5  # Increased from 0.1 for stronger adjustments
        self.grad_multiplier = config.INDICATOR_GRAD_MULT  # Apply gradient boost

    def _logit_from_alpha(self, alpha):
        return tf.math.log(alpha + self.epsilon) - tf.math.log(1.0 - alpha + self.epsilon)

    def _alpha_from_logit(self, logit):
        return tf.sigmoid(logit)

    def _logit_from_period(self, period):
        alpha = 2.0 / (period + 1.0)
        return self._logit_from_alpha(alpha)

    def _period_from_logit(self, logit):
        alpha = self._alpha_from_logit(logit)
        period = (2.0 / (alpha + self.epsilon)) - 1.0
        return tf.maximum(period, 0.0)

    def build(self, input_shape):
        # input_shape[0] is close_seq, [1] is meta_adjust [B, num_logits]
        for i, s in enumerate(self.config.MA_SPANS):
            init_logit = self._logit_from_period(s)
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(init_logit),
                                trainable=True,
                                name=f'alpha_ma_{i}',
                                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            self.alpha_vars_ma.append(v)
            self.all_logit_vars.append(v)

        for i, settings in enumerate(self.config.MACD_SETTINGS):
            v_fast = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['fast'])),
                trainable=True,
                name=f'macd_{i}_fast',
                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            v_slow = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['slow'])),
                trainable=True,
                name=f'macd_{i}_slow',
                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            v_signal = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['signal'])),
                trainable=True,
                name=f'macd_{i}_signal',
                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            self.macd_alpha_vars[f'macd_{i}_fast'] = v_fast
            self.macd_alpha_vars[f'macd_{i}_slow'] = v_slow
            self.macd_alpha_vars[f'macd_{i}_signal'] = v_signal
            self.all_logit_vars.extend([v_fast, v_slow, v_signal])

        for i, (s, l) in enumerate(self.config.CUSTOM_MACD_PAIRS):
            v_short = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(s)),
                trainable=True,
                name=f'pair_{i}_short',
                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            v_long = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(l)),
                trainable=True,
                name=f'pair_{i}_long',
                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            v_sig = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(9)),
                trainable=True,
                name=f'pair_{i}_sig',
                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            self.macd_alpha_vars[f'pair_{i}_short'] = v_short
            self.macd_alpha_vars[f'pair_{i}_long'] = v_long
            self.macd_alpha_vars[f'pair_{i}_sig'] = v_sig
            self.all_logit_vars.extend([v_short, v_long, v_sig])

        for i, p in enumerate(self.config.RSI_PERIODS):
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(self._logit_from_period(p)),
                                trainable=True,
                                name=f'rsi_alpha_{i}',
                                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            self.rsi_alpha_vars.append(v)
            self.all_logit_vars.append(v)

        for i, p in enumerate(self.config.BB_PERIODS):
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(self._logit_from_period(p)),
                                trainable=True,
                                name=f'bb_alpha_{i}',
                                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            self.bb_alpha_vars.append(v)
            self.all_logit_vars.append(v)

        for i, p in enumerate(self.config.MOMENTUM_PERIODS):
            init_raw = np.log(min(p, 200.0))
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(init_raw),
                                trainable=True,
                                name=f'momentum_raw_{i}',
                                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            self.momentum_raw.append(v)
            self.all_logit_vars.append(v)

        super().build(input_shape)

    def ewma_seq(self, x_seq, alpha_scalar):
        x_seq = tf.cast(x_seq, tf.float32)
        def step(prev, cur):
            return alpha_scalar * cur + (1.0 - alpha_scalar) * prev
        first = x_seq[:, 0]
        rest = x_seq[:, 1:]
        ema_rest = tf.scan(
            fn=lambda prev, cur: step(prev, cur),
            elems=tf.transpose(rest, perm=[1, 0]),
            initializer=first,
            parallel_iterations=1
        )
        ema_rest = tf.transpose(ema_rest, perm=[1, 0])
        ema_full = tf.concat([tf.expand_dims(first, axis=1), ema_rest], axis=1)
        return ema_full

    def call(self, inputs, training=None):
        x, meta_adjust = inputs
        x = tf.cast(x, tf.float32)
        features = []
        idx = 0  # Index for slicing meta_adjust

        for logit in self.alpha_vars_ma:
            # Apply gradient multiplier for better learning
            logit_boosted = logit + tf.stop_gradient(logit) * (self.grad_multiplier - 1.0)
            adjusted_logit = logit_boosted + meta_adjust[:, idx] * self.meta_scale
            alpha = self._alpha_from_logit(adjusted_logit)
            ema_seq = self.ewma_seq(x, alpha)
            features.append(ema_seq)
            idx += 1

        for i in range(len(self.config.MACD_SETTINGS)):
            fast_var = self.macd_alpha_vars[f'macd_{i}_fast']
            slow_var = self.macd_alpha_vars[f'macd_{i}_slow']
            sig_var = self.macd_alpha_vars[f'macd_{i}_signal']
            # Boost gradients
            fast_boosted = fast_var + tf.stop_gradient(fast_var) * (self.grad_multiplier - 1.0)
            slow_boosted = slow_var + tf.stop_gradient(slow_var) * (self.grad_multiplier - 1.0)
            sig_boosted = sig_var + tf.stop_gradient(sig_var) * (self.grad_multiplier - 1.0)
            fast_logit = fast_boosted + meta_adjust[:, idx] * self.meta_scale
            slow_logit = slow_boosted + meta_adjust[:, idx+1] * self.meta_scale
            sig_logit = sig_boosted + meta_adjust[:, idx+2] * self.meta_scale
            fast = self._alpha_from_logit(fast_logit)
            slow = self._alpha_from_logit(slow_logit)
            sig = self._alpha_from_logit(sig_logit)
            ema_f = self.ewma_seq(x, fast)
            ema_s = self.ewma_seq(x, slow)
            macd_line = ema_f - ema_s
            macd_sig = self.ewma_seq(macd_line, sig)
            macd_hist = macd_line - macd_sig
            features.extend([macd_line, macd_sig, macd_hist])
            # Add binary cross for signals
            macd_cross = tf.sign(macd_hist)
            features.append(macd_cross)
            idx += 3

        for i in range(len(self.config.CUSTOM_MACD_PAIRS)):
            short_var = self.macd_alpha_vars[f'pair_{i}_short']
            long_var = self.macd_alpha_vars[f'pair_{i}_long']
            sig_var = self.macd_alpha_vars[f'pair_{i}_sig']
            # Boost gradients
            short_boosted = short_var + tf.stop_gradient(short_var) * (self.grad_multiplier - 1.0)
            long_boosted = long_var + tf.stop_gradient(long_var) * (self.grad_multiplier - 1.0)
            sig_boosted = sig_var + tf.stop_gradient(sig_var) * (self.grad_multiplier - 1.0)
            short_logit = short_boosted + meta_adjust[:, idx] * self.meta_scale
            long_logit = long_boosted + meta_adjust[:, idx+1] * self.meta_scale
            sig_logit = sig_boosted + meta_adjust[:, idx+2] * self.meta_scale
            a_s = self._alpha_from_logit(short_logit)
            a_l = self._alpha_from_logit(long_logit)
            a_sig = self._alpha_from_logit(sig_logit)
            ema_s = self.ewma_seq(x, a_s)
            ema_l = self.ewma_seq(x, a_l)
            macd_line_pair = ema_s - ema_l
            macd_sig_pair = self.ewma_seq(macd_line_pair, a_sig)
            macd_hist_pair = macd_line_pair - macd_sig_pair
            features.extend([macd_line_pair, macd_sig_pair, macd_hist_pair])
            idx += 3

        diffs = x[:, 1:] - x[:, :-1]
        gains = tf.where(diffs > 0, diffs, tf.zeros_like(diffs))
        losses = tf.where(diffs < 0, -diffs, tf.zeros_like(diffs))
        gains_padded = tf.concat([tf.zeros((tf.shape(gains)[0], 1), dtype=gains.dtype), gains], axis=1)
        losses_padded = tf.concat([tf.zeros((tf.shape(losses)[0], 1), dtype=losses.dtype), losses], axis=1)

        for logit in self.rsi_alpha_vars:
            # Boost gradients
            logit_boosted = logit + tf.stop_gradient(logit) * (self.grad_multiplier - 1.0)
            adjusted_logit = logit_boosted + meta_adjust[:, idx] * self.meta_scale
            rsi_alpha = self._alpha_from_logit(adjusted_logit)
            gains_ema = self.ewma_seq(gains_padded, rsi_alpha)
            losses_ema = self.ewma_seq(losses_padded, rsi_alpha)
            rs = gains_ema / (losses_ema + 1e-8)
            rsi_seq = 100.0 - (100.0 / (1.0 + rs))
            features.append(rsi_seq)
            idx += 1

        for logit in self.bb_alpha_vars:
            # Boost gradients
            logit_boosted = logit + tf.stop_gradient(logit) * (self.grad_multiplier - 1.0)
            adjusted_logit = logit_boosted + meta_adjust[:, idx] * self.meta_scale
            bb_alpha = self._alpha_from_logit(adjusted_logit)
            ema_mean = self.ewma_seq(x, bb_alpha)
            sq_dev = tf.square(x - ema_mean)
            ema_var = self.ewma_seq(sq_dev, bb_alpha)
            ema_std = tf.sqrt(ema_var + 1e-8)
            features.extend([ema_mean, ema_mean + 2.0 * ema_std, ema_mean - 2.0 * ema_std])
            # Add Bollinger %B
            bb_percent = (x - (ema_mean - 2.0 * ema_std)) / (4.0 * ema_std + 1e-8)
            features.append(bb_percent)
            idx += 1

        for raw in self.momentum_raw:
            adjusted_raw = raw + meta_adjust[:, idx] * self.meta_scale
            period = tf.nn.softplus(adjusted_raw) + 1.0
            batch_period = tf.broadcast_to(period, [tf.shape(x)[0]])
            sampled_past = differentiable_past_sample(x, batch_period, sigma=self.gaussian_sigma)
            momentum_val = x - tf.expand_dims(sampled_past, axis=1)  # Full seq approximation by repeating
            features.append(momentum_val)
            idx += 1

        features.append(x)  # Add raw close as a "indicator" sequence

        output = tf.stack(features, axis=-1)  # [B, LOOKBACK, num_features]
        n_features = len(features)
        tf.ensure_shape(output, [None, self.config.LOOKBACK, n_features])
        output.set_shape([None, self.config.LOOKBACK, n_features])
        return output

    def get_learned_parameters(self):
        learned = {}
        for i, v in enumerate(self.alpha_vars_ma):
            period = self._period_from_logit(v).numpy()
            learned[f'ma_period_{i}'] = float(period)
        for k, v in self.macd_alpha_vars.items():
            period = self._period_from_logit(v).numpy()
            learned[f'{k}'] = float(period)
        for i, v in enumerate(self.rsi_alpha_vars):
            period = self._period_from_logit(v).numpy()
            learned[f'rsi_period_{i}'] = float(period)
        for i, v in enumerate(self.bb_alpha_vars):
            period = self._period_from_logit(v).numpy()
            learned[f'bb_period_{i}'] = float(period)
        for i, raw in enumerate(self.momentum_raw):
            p = (tf.nn.softplus(raw) + 1.0).numpy()
            learned[f'momentum_period_{i}'] = float(p)
        return learned

class PositionalEncodingLayer(layers.Layer):
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        positions = tf.range(0, tf.cast(seq_len, tf.float32), dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [batch_size, 1])
        positions = tf.expand_dims(positions, axis=-1)
        div_term = tf.exp(2.0 * tf.range(0, tf.cast(d_model // 2, tf.float32), dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
        div_term = tf.expand_dims(div_term, axis=0)
        div_term = tf.expand_dims(div_term, axis=0)  # [1,1,d//2]
        angles = positions * div_term
        even = tf.sin(angles)
        odd = tf.cos(angles)
        pe = tf.concat([even, odd], axis=-1)
        return pe

class PricePredictor:
    def __init__(self, config: Config):
        self.config = config

    def build_model(self):
        inp = layers.Input(shape=(self.config.LOOKBACK,), name='close_sequence')

        # Compute meta_adjust from raw input stats
        inp_resh = layers.Reshape((self.config.LOOKBACK, 1))(inp)  # [B, LOOKBACK, 1] for pooling
        meta_inp = layers.Concatenate()([
            layers.GlobalAveragePooling1D()(inp_resh),
            layers.GlobalMaxPooling1D()(inp_resh)
        ])
        num_logits = (len(self.config.MA_SPANS) +
                      len(self.config.MACD_SETTINGS) * 3 +
                      len(self.config.CUSTOM_MACD_PAIRS) * 3 +
                      len(self.config.RSI_PERIODS) +
                      len(self.config.BB_PERIODS) +
                      len(self.config.MOMENTUM_PERIODS))
        meta_adjust = layers.Dense(num_logits, activation='tanh')(meta_inp)

        # Enhanced Learnable Indicators: Now takes [inp, meta_adjust], outputs sequences [B, LOOKBACK, num_ind]
        ind_seq = LearnableIndicators(self.config, name='learnable_indicators')([inp, meta_adjust])

        # Memory-Supplemented Layers: Capture temporal interconnections
        memory = layers.Bidirectional(layers.GRU(64, return_sequences=True))(ind_seq)
        memory = layers.Dropout(0.1)(memory)

        # Interconnection Attention: Model relations between indicators
        att_key_dim = 32
        att = layers.MultiHeadAttention(num_heads=8, key_dim=att_key_dim)(memory, memory)
        x = layers.Add()([memory, att])
        x = layers.LayerNormalization()(x)

        # Graph-like view: Attend across indicators
        x_perm = layers.Permute((2, 1))(x)  # [B, num_ind, LOOKBACK]
        inter_att = layers.MultiHeadAttention(num_heads=4, key_dim=att_key_dim)(x_perm, x_perm)
        x_perm = layers.Add()([x_perm, inter_att])
        x = layers.Permute((2, 1))(layers.LayerNormalization()(x_perm))  # Back to [B, LOOKBACK, num_ind]

        # Multi-scale Conv feature extractor
        x_short = layers.Conv1D(16, 3, padding='same', activation='gelu')(x)
        x_med = layers.Conv1D(16, 7, padding='same', activation='gelu')(x)
        x_long = layers.Conv1D(16, 15, padding='same', activation='gelu')(x)
        x = layers.Concatenate()([x_short, x_med, x_long])

        # Positional encoding
        x = layers.Add()([x, PositionalEncodingLayer()(x)])

        # Transformer-style blocks (reduced to 2 for speed)
        for _ in range(2):
            att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
            x = layers.Add()([x, att])
            x = layers.LayerNormalization()(x)
            ff = layers.Dense(32, activation='gelu')(x)
            ff = layers.Dropout(0.1)(ff)
            ff = layers.Dense(x.shape[-1])(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)

        # Global context vector
        context = layers.GlobalAveragePooling1D()(x)

        # Sequence summary for regression
        seq_flat = layers.Flatten()(x)

        # Shared dense layer for all output heads
        shared_dense = layers.Dense(32, activation='gelu',
                                   kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(seq_flat)
        shared_dense = layers.Concatenate()([shared_dense, context])

        # === THREE INDEPENDENT OUTPUT TOWERS (h0, h1, h2) ===
        # Each horizon has its own price, direction, and confidence (variance) head
        
        # Variance bias initialization: softplus(x) ≈ x for x > 0
        # Initialize bias so initial variance ≈ 1.0 (unit variance in scaled space)
        # softplus(0) ≈ 0.693, softplus(0.5) ≈ 0.97, softplus(1.0) ≈ 1.31
        var_bias_init = tf.keras.initializers.Constant(0.5)  # Initial variance ≈ 0.97
        
        # Direction bias initialization: sigmoid(0) = 0.5 (unbiased)
        # Keep at 0 for balanced initial predictions
        dir_bias_init = tf.keras.initializers.Zeros()
        
        # ---- TOWER 0 (1-minute horizon) ----
        tower_h0 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h0 = layers.Dense(1, name='price_h0')(tower_h0)
        direction_h0 = layers.Dense(1, activation='sigmoid', name='direction_h0',
                                   bias_initializer=dir_bias_init)(tower_h0)
        variance_h0 = layers.Dense(1, activation='softplus', name='variance_h0',
                                  bias_initializer=var_bias_init)(tower_h0)
        # Variance initialized to ~1.0 via softplus(0.5) for stable NLL training

        # ---- TOWER 1 (5-minute horizon - PRIMARY) ----
        tower_h1 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h1 = layers.Dense(1, name='price_h1')(tower_h1)
        direction_h1 = layers.Dense(1, activation='sigmoid', name='direction_h1',
                                   bias_initializer=dir_bias_init)(tower_h1)
        variance_h1 = layers.Dense(1, activation='softplus', name='variance_h1',
                                  bias_initializer=var_bias_init)(tower_h1)
        # Variance initialized to ~1.0 via softplus(0.5) for stable NLL training

        # ---- TOWER 2 (15-minute horizon) ----
        tower_h2 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h2 = layers.Dense(1, name='price_h2')(tower_h2)
        direction_h2 = layers.Dense(1, activation='sigmoid', name='direction_h2',
                                   bias_initializer=dir_bias_init)(tower_h2)
        variance_h2 = layers.Dense(1, activation='softplus', name='variance_h2',
                                  bias_initializer=var_bias_init)(tower_h2)
        # Variance initialized to ~1.0 via softplus(0.5) for stable NLL training

        # === FINAL MODEL: 9 outputs (3 horizons × 3 heads each) ===
        return models.Model(
            inputs=inp,
            outputs=[
                price_h0, direction_h0, variance_h0,
                price_h1, direction_h1, variance_h1,
                price_h2, direction_h2, variance_h2
            ]
        )

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


class CustomTrainModel(models.Model):
    def __init__(self, base_model, pred_scale, pred_mean,
                 lambda_point=1.0, lambda_local_trend=1.0, lambda_global_trend=0.2,
                 lambda_extended_trend=0.16, lambda_dir=1.0, config=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.epsilon = 1e-8

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
        self.config = config or Config()

        # Single source-of-truth for Huber delta (in *scaled* units)
        self.huber_delta = float(self.config.HUBER_DELTA)

        # Numerical epsilon used in denominators
        self.eps = tf.constant(1e-8, dtype=tf.float32)

        # NOTE: We no longer use tf.keras.losses.Huber; point loss will call self.point_huber.
        # This yields identical math but keeps a single implementation.
    def _logit_from_alpha(self, alpha): return tf.math.log(alpha + self.epsilon) - tf.math.log(1.0 - alpha + self.epsilon)
    def _alpha_from_logit(self, logit): return tf.sigmoid(logit)
    def _logit_from_period(self, period):
        alpha = 2.0 / (period + 1.0)
        return self._logit_from_alpha(alpha)
    def _period_from_logit(self, logit):
        alpha = self._alpha_from_logit(logit)
        period = (2.0 / (alpha + self.epsilon)) - 1.0
        return tf.maximum(period, 0.0)
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
        """
        Focal Loss: -α(1-p_t)^γ * log(p_t)
        Addresses class imbalance by down-weighting easy examples and focusing on hard ones.
        
        Args:
            true_labels: Binary labels [B] (0 or 1)
            logits: Predicted probabilities [B] from sigmoid (0 to 1)
            alpha: Class weight for minority class (default 0.7 weights DOWN class)
            gamma: Focusing parameter (default 2.0; higher = more focus on hard examples)
        
        Returns:
            If reduce=True: scalar mean focal loss.
            If reduce=False: per-example focal loss vector.
        """
        if alpha is None:
            alpha = self.config.FOCAL_ALPHA
        if gamma is None:
            gamma = self.config.FOCAL_GAMMA
        
        alpha = tf.cast(alpha, tf.float32)
        gamma = tf.cast(gamma, tf.float32)
        
        # Ensure inputs are float32
        true_labels = tf.cast(true_labels, tf.float32)
        logits = tf.cast(logits, tf.float32)
        
        # Clamp logits to prevent log(0)
        logits = tf.clip_by_value(logits, 1e-7, 1.0 - 1e-7)
        
        # Compute focal weight: (1 - p_t)^γ where p_t is the probability of true class
        p_t = true_labels * logits + (1.0 - true_labels) * (1.0 - logits)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Compute base cross-entropy
        bce = -true_labels * tf.math.log(logits) - (1.0 - true_labels) * tf.math.log(1.0 - logits)
        
        # Apply focal weighting and class weighting.
        # alpha weights DOWN class (label=0), (1-alpha) weights UP (label=1)
        # With alpha=0.5, both classes are weighted equally
        class_weight = alpha * (1.0 - true_labels) + (1.0 - alpha) * true_labels
        
        focal = class_weight * focal_weight * bce

        if reduce:
            return tf.reduce_mean(focal)
        return focal

    # -------------------------
    # Point loss (log-cosh)
    # -------------------------
    def point_huber(self, y_true_scaled, y_pred_scaled, last_close_scaled=None, delta=None):
        """Point loss in scaled space.

        For delta-target training, the natural pivot is 0 (not last_close).
        `last_close_scaled` is kept optional for API compatibility.
        """
        # Squeeze explicitly on axis=1 to be shape-safe
        y_true = tf.squeeze(y_true_scaled, axis=1)
        y_pred = tf.squeeze(y_pred_scaled, axis=1)

        # Compute differences in scaled domain
        diffs = y_true - y_pred

        # Delta-target loss: symmetric log-cosh in scaled domain.
        per_elem = tf.math.log(tf.cosh(diffs))
        # Clip log-cosh to prevent unbounded growth when |diffs| is large
        per_elem = tf.clip_by_value(per_elem, -10.0, 10.0)

        result = self._reduce_mean(per_elem)
        # Guard against NaN/Inf contamination
        result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
        return result


    # -------------------------
    # Local trend loss
    # -------------------------
    def local_trend_loss(self, x_window, y_true_raw, y_pred_raw, last_close_raw):
        """
        Local trend = difference (scaled) between y_true and last_close vs
                      y_pred and last_close.
        Now applies Huber on diffs, then tanh bounding on the loss.
        """
        last_close = tf.squeeze(last_close_raw, axis=1)
        last_close_scaled = self._to_scaled_static(last_close, self.pred_mean, self.pred_scale, self.eps)

        y_true_scaled = self._to_scaled_static(y_true_raw, self.pred_mean, self.pred_scale, self.eps)
        y_pred_scaled = self._to_scaled_static(y_pred_raw, self.pred_mean, self.pred_scale, self.eps)

        actual_trend = y_true_scaled - last_close_scaled
        pred_trend = y_pred_scaled - last_close_scaled

        # Difference of trends
        trend_diffs = actual_trend - pred_trend

        # Apply log-cosh for soft quadratic loss with clipping
        per_elem = tf.math.log(tf.cosh(trend_diffs))
        per_elem = tf.clip_by_value(per_elem, -10.0, 10.0)

        result = self._reduce_mean(per_elem)
        # Guard against NaN/Inf contamination
        result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
        return result


    # -------------------------
    # Extended & global trends
    # -------------------------
    def extended_trend_loss(self, x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw):
        """
        Returns tuple: (global_loss_scalar, extended_loss_scalar).
        Now huber -> then tanh bounding.
        """
        start_of_window = tf.squeeze(x_window[:, 0:1], axis=1)
        start_scaled = self._to_scaled_static(start_of_window, self.pred_mean, self.pred_scale, self.eps)

        y_true_scaled = self._to_scaled_static(y_true_raw, self.pred_mean, self.pred_scale, self.eps)
        y_pred_scaled = self._to_scaled_static(y_pred_raw, self.pred_mean, self.pred_scale, self.eps)

        # ---- Global trend (target vs start of window) ----
        global_diffs = (y_true_scaled - start_scaled) - (y_pred_scaled - start_scaled)
        global_logcosh = tf.math.log(tf.cosh(global_diffs))
        # Clip to prevent unbounded growth
        global_logcosh = tf.clip_by_value(global_logcosh, -10.0, 10.0)
        global_loss = self._reduce_mean(global_logcosh)
        # Guard against NaN/Inf
        global_loss = tf.where(tf.math.is_finite(global_loss), global_loss, tf.constant(0.0, dtype=tf.float32))

        # ---- Extended / multi-scale trend ----
        last_close = tf.squeeze(last_close_raw, axis=1)
        last_close_scaled = self._to_scaled(last_close)
        pred_trend_scaled = y_pred_scaled - last_close_scaled

        n_trend_features = tf.shape(extended_trends)[1]

        def compute_extended():
            eps = tf.cast(1e-8, tf.float32)

            # --- Long-term trend ---
            long_term_trend = tf.cast(extended_trends[:, -1], tf.float32)
            long_term_trend = tf.clip_by_value(long_term_trend, -0.999, 1e6)
            past_price_long = last_close / (1.0 + long_term_trend + eps)
            long_price_diff_raw = last_close - past_price_long
            long_price_diff_scaled = long_price_diff_raw / (self.pred_scale + self.eps)

            long_diffs = pred_trend_scaled - long_price_diff_scaled
            long_logcosh = tf.math.log(tf.cosh(long_diffs))
            # Clip to prevent unbounded growth
            long_logcosh = tf.clip_by_value(long_logcosh, -10.0, 10.0)
            extended_loss_long = self._reduce_mean(long_logcosh)
            # Guard against NaN/Inf
            extended_loss_long = tf.where(tf.math.is_finite(extended_loss_long), extended_loss_long, tf.constant(0.0, dtype=tf.float32))

            # --- Multi-scale short-term trends ---
            def compute_multi():
                short_trends = tf.cast(extended_trends[:, :-1], tf.float32)
                short_trends = tf.clip_by_value(short_trends, -0.999, 1e6)

                past_prices = last_close[:, None] / (1.0 + short_trends + eps)
                short_price_diff_raw = last_close[:, None] - past_prices
                short_price_diff_scaled = short_price_diff_raw / (self.pred_scale + self.eps)

                short_diffs = tf.expand_dims(pred_trend_scaled, 1) - short_price_diff_scaled
                logcosh_losses = tf.math.log(tf.cosh(short_diffs))
                # Clip to prevent unbounded growth
                logcosh_losses = tf.clip_by_value(logcosh_losses, -10.0, 10.0)

                # Normalize across scales
                per_scale_mean = tf.reduce_mean(logcosh_losses, axis=0)
                denom = tf.reduce_mean(per_scale_mean) + self.eps
                normalized_per_scale = per_scale_mean / denom
                result = tf.reduce_mean(normalized_per_scale)
                # Guard against NaN/Inf
                result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
                return result

            def no_multi():
                return tf.constant(0.0, dtype=tf.float32)

            multi_scale_loss = tf.cond(tf.greater(tf.shape(extended_trends)[1], 1), compute_multi, no_multi)
            return extended_loss_long + multi_scale_loss

        def no_extended():
            return tf.constant(0.0, dtype=tf.float32)

        extended_loss = tf.cond(tf.greater(n_trend_features, 0), compute_extended, no_extended)

        return global_loss, extended_loss

    # -------------------------
    # Combined custom loss (NEW: Per-horizon outputs with focal loss)
    # -------------------------
    def custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends):
        """
        Multi-horizon, multi-task loss computation.
        
        y_pred now has 9 outputs: [price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2]
        Each horizon (h0=1min, h1=5min, h2=15min) has independent price, direction, and confidence outputs.
        
        Losses computed:
        - 3 point losses (one per horizon)
        - 3×3 trend losses (local, global, extended for each horizon)
        - 3 focal direction losses (per horizon, replacing BCE)
        - 3 variance NLL losses (per horizon)
        - Regularization and volatility penalties
        
        Returns: 29-component tuple for comprehensive loss tracking
        """
        
        # Unpack 9 outputs
        price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2 = y_pred
        
        # Prepare targets (multi-horizon)
        # Option A: y_true is DELTA in scaled space, so y_true_raw is DELTA in raw (price units).
        y_true = tf.cast(y_true, tf.float32)  # [B, 3]
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, 3]  (delta_raw)
        last_close_squeeze = tf.squeeze(last_close, axis=1)  # [B] (raw price)

        y_true_h0 = y_true[:, 0:1]
        y_true_h1 = y_true[:, 1:2]
        y_true_h2 = y_true[:, 2:3]

        y_true_raw_h0 = y_true_raw[:, 0]
        y_true_raw_h1 = y_true_raw[:, 1]
        y_true_raw_h2 = y_true_raw[:, 2]

        # Trade-aware direction labeling with optional deadband.
        # Use returns for labeling/masking even though regression target is delta:
        #   ret = delta / last_close
        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        ret_h0 = (y_true_raw_h0) / (last_close_squeeze + self.eps)
        ret_h1 = (y_true_raw_h1) / (last_close_squeeze + self.eps)
        ret_h2 = (y_true_raw_h2) / (last_close_squeeze + self.eps)

        mask_h0 = tf.cast(tf.abs(ret_h0) > deadband, tf.float32)
        mask_h1 = tf.cast(tf.abs(ret_h1) > deadband, tf.float32)
        mask_h2 = tf.cast(tf.abs(ret_h2) > deadband, tf.float32)

        true_dir_h0 = tf.cast(ret_h0 > deadband, tf.float32)
        true_dir_h1 = tf.cast(ret_h1 > deadband, tf.float32)
        true_dir_h2 = tf.cast(ret_h2 > deadband, tf.float32)

        # === POINT LOSSES (3 horizons × 1 = 3 components) ===
        # CRITICAL: Training operates in SCALED delta space.
        # y_true_h* are SCALED deltas: (raw_delta - mean) / std
        # price_h* are SCALED predictions (unbounded, optimized in scaled space)
        # Compute Huber loss in scaled space where the model was trained.
        point_loss_h0_val = self.lambda_short * self.point_huber(y_true_h0, price_h0)
        point_loss_h1_val = self.lambda_point * self.point_huber(y_true_h1, price_h1)
        point_loss_h2_val = self.lambda_long * self.point_huber(y_true_h2, price_h2)
        point_loss_val = point_loss_h0_val + point_loss_h1_val + point_loss_h2_val

        # === TREND LOSSES ===
        # CRITICAL REWRITE: Trend loss now directly supervises PREDICTIONS to respect historical trends.
        # 
        # OLD (INCORRECT): Penalized targets for not aligning with extended_trends
        #   trend_diff = (y_true_raw - extended_trends × last_close) / pred_scale
        #   This imposed regularization on the TARGETS, not predictions—semantically backwards.
        #
        # NEW (CORRECT): Extended trends act as baseline predictions; penalize prediction deviation from baseline.
        # 
        # Semantic: Extended trends represent "what a simple historical-trend model would predict"
        # We penalize the learned model for diverging too far from this strong baseline.
        # If the learned model can't beat the trend baseline, its prediction should be close to it.
        #
        # Extended trends are now absolute deltas (dollars), matching prediction targets exactly.
        
        pred_scale = tf.cast(self.pred_scale + self.eps, tf.float32)
        
        # Convert extended trends from raw deltas to scaled space (same space as predictions)
        extended_trends_scaled_h0 = extended_trends[:, 0:1] / pred_scale  # [B, 1] scaled
        extended_trends_scaled_h1 = extended_trends[:, 1:2] / pred_scale  # [B, 1] scaled
        extended_trends_scaled_h2 = extended_trends[:, 2:3] / pred_scale  # [B, 1] scaled
        
        # Trend loss: Penalize predictions for deviating from trend baseline
        # If model predictions are unreasonably far from trends, this acts as regularization
        # If model predictions match/beat trends, this loss is near zero
        trend_loss_h0 = tf.reduce_mean(tf.square(price_h0 - extended_trends_scaled_h0))
        trend_loss_h1 = tf.reduce_mean(tf.square(price_h1 - extended_trends_scaled_h1))
        trend_loss_h2 = tf.reduce_mean(tf.square(price_h2 - extended_trends_scaled_h2))
        
        # === CROSS-HORIZON COHERENCE CONSTRAINTS ===
        # STRENGTHENED: Enforce consistency across horizons (h0 → h1 → h2)
        # Multiple constraints ensure multi-horizon predictions form a coherent picture:
        # 1. Direction consistency: All three horizons should agree on UP/DOWN
        # 2. Magnitude consistency: |pred_h0| ≤ |pred_h1| ≤ |pred_h2| (longer horizons = larger moves)
        # 3. Smoothness: No abrupt sign changes across consecutive horizons
        
        # Constraint 1: Direction consistency (all signs should match)
        sign_pred_h0 = tf.sign(price_h0)
        sign_pred_h1 = tf.sign(price_h1)
        sign_pred_h2 = tf.sign(price_h2)
        
        # Penalize disagreement in direction predictions
        dir_agree_h01 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h0, sign_pred_h1), tf.float32))
        dir_agree_h12 = tf.reduce_mean(tf.cast(tf.equal(sign_pred_h1, sign_pred_h2), tf.float32))
        dir_disagree_loss = 1.0 - (dir_agree_h01 + dir_agree_h12) / 2.0  # Loss when disagreement occurs
        
        # Constraint 2: Magnitude monotonicity: |h0| ≤ |h1| ≤ |h2| (longer horizons = larger absolute moves)
        # This enforces that 5-min predictions are at least as large as 1-min, etc.
        abs_pred_h0 = tf.abs(price_h0)
        abs_pred_h1 = tf.abs(price_h1)
        abs_pred_h2 = tf.abs(price_h2)
        
        # Penalize violations of monotonic magnitude increase
        magnitude_h01_violation = tf.nn.relu(abs_pred_h0 - abs_pred_h1)  # Penalize if h0 > h1
        magnitude_h12_violation = tf.nn.relu(abs_pred_h1 - abs_pred_h2)  # Penalize if h1 > h2
        magnitude_loss = tf.reduce_mean(magnitude_h01_violation + magnitude_h12_violation)
        
        # Constraint 3: Smoothness penalty on target signs (should be consistent across horizons)
        sign_target_h0 = tf.sign(y_true_raw_h0)
        sign_target_h1 = tf.sign(y_true_raw_h1)
        sign_target_h2 = tf.sign(y_true_raw_h2)
        target_smoothness_loss = tf.reduce_mean(
            tf.cast(tf.math.logical_xor(sign_target_h1 == sign_target_h0, 
                                         sign_target_h1 == sign_target_h2), tf.float32)
        )
        
        # Combine all coherence constraints (STRENGTHENED: increased weight from 0.01 to 0.1)
        # This makes cross-horizon consistency a significant part of the optimization
        coherence_penalty = (dir_disagree_loss + magnitude_loss + target_smoothness_loss) / 3.0
        
        local_trend_h0 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h0 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h0 = self.lambda_extended_trend * trend_loss_h0
        local_trend_h1 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h1 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h1 = self.lambda_extended_trend * trend_loss_h1
        local_trend_h2 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h2 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h2 = self.lambda_extended_trend * trend_loss_h2
        
        trend_loss_val = extended_trend_h0 + extended_trend_h1 + extended_trend_h2 + coherence_penalty * 0.01  # small weight for coherence

        # === DIRECTION LOSSES (3 horizons × focal loss = 3 components) ===
        # Apply deadband masks: ignore neutral samples (mask=0). Guard against empty masks.
        dir_pred_h0 = tf.squeeze(dir_h0, axis=1)
        dir_pred_h1 = tf.squeeze(dir_h1, axis=1)
        dir_pred_h2 = tf.squeeze(dir_h2, axis=1)

        per_ex_h0 = self.focal_loss(true_dir_h0, dir_pred_h0, reduce=False)
        per_ex_h1 = self.focal_loss(true_dir_h1, dir_pred_h1, reduce=False)
        per_ex_h2 = self.focal_loss(true_dir_h2, dir_pred_h2, reduce=False)

        dir_loss_h0 = tf.reduce_sum(per_ex_h0 * mask_h0) / (tf.reduce_sum(mask_h0) + self.eps)
        dir_loss_h1 = tf.reduce_sum(per_ex_h1 * mask_h1) / (tf.reduce_sum(mask_h1) + self.eps)
        dir_loss_h2 = tf.reduce_sum(per_ex_h2 * mask_h2) / (tf.reduce_sum(mask_h2) + self.eps)
        total_dir_loss = self.lambda_dir * (dir_loss_h0 + dir_loss_h1 + dir_loss_h2)

        # === VARIANCE NLL LOSSES (3 horizons × 1 = 3 components) ===
        # Clip variance to prevent log/div blow-ups and stabilize gradients.
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e4), tf.float32)
        var_h0_c = tf.clip_by_value(var_h0, var_floor, var_cap)
        var_h1_c = tf.clip_by_value(var_h1, var_floor, var_cap)
        var_h2_c = tf.clip_by_value(var_h2, var_floor, var_cap)

        nll_h0 = 0.5 * tf.math.log(var_h0_c + self.eps) + 0.5 * tf.square(y_true_h0 - price_h0) / (var_h0_c + self.eps)
        nll_h0_val = tf.reduce_mean(nll_h0)
        
        nll_h1 = 0.5 * tf.math.log(var_h1_c + self.eps) + 0.5 * tf.square(y_true_h1 - price_h1) / (var_h1_c + self.eps)
        nll_h1_val = tf.reduce_mean(nll_h1)
        
        nll_h2 = 0.5 * tf.math.log(var_h2_c + self.eps) + 0.5 * tf.square(y_true_h2 - price_h2) / (var_h2_c + self.eps)
        nll_h2_val = tf.reduce_mean(nll_h2)
        
        total_nll = self.lambda_var * (nll_h0_val + nll_h1_val + nll_h2_val)

        # === GAUSSIAN-IMPLIED DIRECTION PROBABILITIES (from mu/var) ===
        # Delta-target: interpret mu as expected delta. Define P(up) consistent with deadband:
        #   P(ret > deadband)  <=>  P(delta > deadband * last_close)
        mu_h0 = tf.squeeze(price_h0, axis=1)
        mu_h1 = tf.squeeze(price_h1, axis=1)
        mu_h2 = tf.squeeze(price_h2, axis=1)
        sigma_h0 = tf.sqrt(tf.squeeze(var_h0_c, axis=1) + self.eps)
        sigma_h1 = tf.sqrt(tf.squeeze(var_h1_c, axis=1) + self.eps)
        sigma_h2 = tf.sqrt(tf.squeeze(var_h2_c, axis=1) + self.eps)
        deadband_delta_scaled = (deadband * last_close_squeeze) / (self.pred_scale + self.eps)
        z_up_h0 = (mu_h0 - deadband_delta_scaled) / (sigma_h0 + self.eps)
        z_up_h1 = (mu_h1 - deadband_delta_scaled) / (sigma_h1 + self.eps)
        z_up_h2 = (mu_h2 - deadband_delta_scaled) / (sigma_h2 + self.eps)
        gauss_p_up_h0 = self._normal_cdf(z_up_h0)
        gauss_p_up_h1 = self._normal_cdf(z_up_h1)
        gauss_p_up_h2 = self._normal_cdf(z_up_h2)

        # Optional alignment: encourage direction head to match distribution-implied P(up)
        # IMPORTANT: keep this graph-safe (no Python `if` on tensors).
        lambda_dir_align = tf.constant(float(getattr(self.config, 'LAMBDA_DIR_ALIGN', 0.0)), dtype=tf.float32)

        align_h0 = tf.keras.losses.binary_crossentropy(gauss_p_up_h0, dir_pred_h0)
        align_h1 = tf.keras.losses.binary_crossentropy(gauss_p_up_h1, dir_pred_h1)
        align_h2 = tf.keras.losses.binary_crossentropy(gauss_p_up_h2, dir_pred_h2)
        # Apply deadband mask to alignment too (avoid pushing on neutral/noise moves)
        align_h0 = tf.reduce_sum(align_h0 * mask_h0) / (tf.reduce_sum(mask_h0) + self.eps)
        align_h1 = tf.reduce_sum(align_h1 * mask_h1) / (tf.reduce_sum(mask_h1) + self.eps)
        align_h2 = tf.reduce_sum(align_h2 * mask_h2) / (tf.reduce_sum(mask_h2) + self.eps)
        dir_align_loss = lambda_dir_align * (align_h0 + align_h1 + align_h2)

        # === REGULARIZATION (unchanged) ===
        reg_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)
        inter_reg = self.config.LAMBDA_INTER * reg_loss

        # === VOLATILITY PENALTY (use primary horizon h1) ===
        # Delta-target: compare dispersion of predicted vs true deltas (scaled space).
        actual_trend = y_true[:, 1]
        pred_trend_scaled = tf.squeeze(price_h1, axis=1)
        actual_std = tf.math.reduce_std(actual_trend)
        pred_std = tf.math.reduce_std(pred_trend_scaled)
        vol_diff = tf.abs(pred_std - actual_std)
        vol_diff_clipped = tf.minimum(vol_diff, 10.0)
        vol_loss = vol_diff_clipped * self.lambda_vol
        vol_loss = tf.where(tf.math.is_finite(vol_loss), vol_loss, tf.constant(0.0, dtype=tf.float32))

        # === TOTAL LOSS ===
        # Principle: Point loss (delta prediction) is the PRIMARY task.
        # Other losses (direction, trend, variance, coherence) are auxiliary constraints.
        # Weight them such that point loss drives ~60-70% of optimization, with strong cross-horizon constraints.
        #
        # UPDATED WEIGHTING (Post-audit):
        # - Point loss: DOMINANT (weight=1.0) - Delta prediction is the core task
        # - Trend loss: MODERATE (weight=0.3) - Penalizes predictions deviating from trend baseline
        # - Direction loss: AUXILIARY (weight=0.2) - Secondary task, classification-based
        # - Coherence loss: STRENGTHENED (weight=0.1) - Cross-horizon consistency now significant
        # - Other losses: WEAK (weight≤0.1) - Regularization and constraints
        #
        # Point loss: log-cosh in scaled space; typical magnitude ~0.1-1.0 per epoch
        # Trend loss: Now penalizes predictions vs trend baseline; typical ~0.01-0.1
        # Coherence loss: STRENGTHENED from 0.01 to 0.1; combines direction/magnitude/smoothness
        # Direction loss: Focal loss on classification; typical ~0.1-0.5
        # Variance NLL: Log-likelihood penalty; can be negative or large; typical ~-1 to +2
        # Regularization: L2 on indicators; small, typically ~0.001-0.01
        #
        # Strategy: Use relative scaling to ensure point_loss drives optimization while strengthening
        # cross-horizon coherence to prevent contradictory predictions across horizons.
        total = (
            point_loss_val +
            0.3 * trend_loss_val +      # Trend baseline consistency (auxiliary)
            0.5 * total_dir_loss +      # Direction classification (INCREASED from 0.2)
            0.15 * dir_align_loss +     # Distribution alignment (increased for better calibration)
            reg_loss +
            0.1 * inter_reg +           # Indicator correlation (weak regularization)
            0.01 * vol_loss +           # Volatility penalty (very weak)
            0.1 * coherence_penalty +   # STRENGTHENED: Cross-horizon coherence (0.01 → 0.1)
            0.5 * total_nll             # Variance NLL (moderate contribution)
        )

        # === RETURN 22-COMPONENT TUPLE for comprehensive logging ===
        # CORRECTED: Actually 22 components, not 29 (fixed documentation)
        # 
        # Format breakdown (22 components total):
        # [0] total_loss (single scalar)
        # [1-3] point_h0, point_h1, point_h2 (3 point loss components)
        # [4-12] local_h0, global_h0, extended_h0, local_h1, global_h1, extended_h1, 
        #        local_h2, global_h2, extended_h2 (9 trend loss components; local/global are 0)
        # [13-15] dir_h0, dir_h1, dir_h2 (3 direction loss components)
        # [16-18] nll_h0, nll_h1, nll_h2 (3 variance NLL components)
        # [19-21] reg_loss, inter_reg, vol_loss (3 regularization components)
        return (
            total,
            point_loss_h0_val, point_loss_h1_val, point_loss_h2_val,
            local_trend_h0, global_trend_h0, extended_trend_h0,
            local_trend_h1, global_trend_h1, extended_trend_h1,
            local_trend_h2, global_trend_h2, extended_trend_h2,
            dir_loss_h0, dir_loss_h1, dir_loss_h2,
            nll_h0_val, nll_h1_val, nll_h2_val,
            reg_loss, inter_reg, vol_loss
        )

    def train_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        with tf.GradientTape() as tape:
            y_pred = self(x_window, training=True)
            loss_components = self.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)

        # Unpack 22-component tuple from custom_loss
        (total_loss_val,
         point_h0, point_h1, point_h2,
         local_h0, global_h0, extended_h0,
         local_h1, global_h1, extended_h1,
         local_h2, global_h2, extended_h2,
         dir_h0, dir_h1, dir_h2,
         nll_h0, nll_h1, nll_h2,
         reg_val, inter_reg, vol_loss) = loss_components

        grads = tape.gradient(total_loss_val, self.trainable_variables)

        # Build list of (grad, var) excluding None grads
        grads_and_vars = []
        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue
            name = v.name.lower()
            # Scale indicator-related variable grads
            if ('alpha_ma' in name or 'macd_' in name or 'pair_' in name or
                'rsi_alpha' in name or 'bb_alpha' in name or 'momentum_raw' in name):
                g = g * self.config.INDICATOR_GRAD_MULT
            grads_and_vars.append((g, v))

        # Optional global-norm clipping
        if getattr(self.config, 'GRAD_CLIP_NORM', 0.0) and self.config.GRAD_CLIP_NORM > 0.0:
            grads_only = [gv[0] for gv in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads_only, self.config.GRAD_CLIP_NORM)
            grads_and_vars = list(zip(clipped_grads, [gv[1] for gv in grads_and_vars]))

        # Apply gradients
        self.optimizer.apply_gradients(grads_and_vars)

        # Keep indicator periods within sensible bounds
        min_p = self.config.MOMENTUM_CLIP_MIN
        max_p = self.config.MOMENTUM_CLIP_MAX
        for var in self.base_model.trainable_variables:
            name = var.name.lower()
            if ('alpha_ma' in name or 'macd_' in name or 'pair_' in name or
                'rsi_alpha' in name or 'bb_alpha' in name):
                period = self._period_from_logit(var)
                clipped = tf.clip_by_value(period, min_p, max_p)
                logit = self._logit_from_period(clipped)
                var.assign(logit)
            elif 'momentum_raw' in name:
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

        ret_h0 = (y_true_raw[:, 0]) / (last_close_squeeze + self.eps)
        ret_h1 = (y_true_raw[:, 1]) / (last_close_squeeze + self.eps)
        ret_h2 = (y_true_raw[:, 2]) / (last_close_squeeze + self.eps)

        mask_h0 = tf.cast(tf.abs(ret_h0) > deadband, tf.float32)
        mask_h1 = tf.cast(tf.abs(ret_h1) > deadband, tf.float32)
        mask_h2 = tf.cast(tf.abs(ret_h2) > deadband, tf.float32)

        true_dir_h0 = tf.cast(ret_h0 > deadband, tf.float32)
        true_dir_h1 = tf.cast(ret_h1 > deadband, tf.float32)
        true_dir_h2 = tf.cast(ret_h2 > deadband, tf.float32)

        # Extract direction predictions for all 3 horizons
        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred
        dir_pred_h0 = tf.squeeze(dir_pred_h0, axis=1)
        dir_pred_h1 = tf.squeeze(dir_pred_h1, axis=1)
        dir_pred_h2 = tf.squeeze(dir_pred_h2, axis=1)

        # Gaussian-implied P(up) from (mu, var): interpretable and consistent with regression.
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e4), tf.float32)
        var_h0_c = tf.clip_by_value(tf.squeeze(var_h0, axis=1), var_floor, var_cap)
        var_h1_c = tf.clip_by_value(tf.squeeze(var_h1, axis=1), var_floor, var_cap)
        var_h2_c = tf.clip_by_value(tf.squeeze(var_h2, axis=1), var_floor, var_cap)
        mu_h0 = tf.squeeze(price_h0, axis=1)
        mu_h1 = tf.squeeze(price_h1, axis=1)
        mu_h2 = tf.squeeze(price_h2, axis=1)
        deadband_delta_scaled = (deadband * tf.squeeze(last_close, axis=1)) / (self.pred_scale + self.eps)
        gauss_p_up_h0 = self._normal_cdf((mu_h0 - deadband_delta_scaled) / (tf.sqrt(var_h0_c) + self.eps))
        gauss_p_up_h1 = self._normal_cdf((mu_h1 - deadband_delta_scaled) / (tf.sqrt(var_h1_c) + self.eps))
        gauss_p_up_h2 = self._normal_cdf((mu_h2 - deadband_delta_scaled) / (tf.sqrt(var_h2_c) + self.eps))

        # Compute per-horizon metrics (masked if deadband is set)
        metrics_head = self._compute_direction_metrics(
            true_dir_h0, true_dir_h1, true_dir_h2,
            dir_pred_h0, dir_pred_h1, dir_pred_h2,
            mask_h0=mask_h0, mask_h1=mask_h1, mask_h2=mask_h2,
            prefix="train_"
        )
        metrics_gauss = self._compute_direction_metrics(
            true_dir_h0, true_dir_h1, true_dir_h2,
            gauss_p_up_h0, gauss_p_up_h1, gauss_p_up_h2,
            mask_h0=mask_h0, mask_h1=mask_h1, mask_h2=mask_h2,
            prefix="train_gauss_"
        )

        # Trend metrics: margins (bps), agreement rates, magnitudes (bps)
        trend_margin_h0 = tf.reduce_mean(tf.abs(y_true_raw[:, 0] - extended_trends[:, 0] * last_close_squeeze)) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        trend_margin_h1 = tf.reduce_mean(tf.abs(y_true_raw[:, 1] - extended_trends[:, 1] * last_close_squeeze)) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        trend_margin_h2 = tf.reduce_mean(tf.abs(y_true_raw[:, 2] - extended_trends[:, 2] * last_close_squeeze)) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        agreement_rate_h0 = tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true_raw[:, 0]), tf.sign(extended_trends[:, 0])), tf.float32))
        agreement_rate_h1 = tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true_raw[:, 1]), tf.sign(extended_trends[:, 1])), tf.float32))
        agreement_rate_h2 = tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true_raw[:, 2]), tf.sign(extended_trends[:, 2])), tf.float32))
        magnitude_bps_h0 = tf.reduce_mean(tf.abs(y_true_raw[:, 0])) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        magnitude_bps_h1 = tf.reduce_mean(tf.abs(y_true_raw[:, 1])) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        magnitude_bps_h2 = tf.reduce_mean(tf.abs(y_true_raw[:, 2])) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)

        # Add all loss components to metrics
        point_loss_total = point_h0 + point_h1 + point_h2
        trend_loss_h0 = local_h0 + global_h0 + extended_h0
        trend_loss_h1 = local_h1 + global_h1 + extended_h1
        trend_loss_h2 = local_h2 + global_h2 + extended_h2
        trend_loss_total = trend_loss_h0 + trend_loss_h1 + trend_loss_h2
        dir_loss_total = dir_h0 + dir_h1 + dir_h2
        nll_total = nll_h0 + nll_h1 + nll_h2

        return {
            "loss": total_loss_val,
            "point_loss": point_loss_total,
            "point_h0": point_h0,
            "point_h1": point_h1,
            "point_h2": point_h2,
            "trend_loss": trend_loss_total,
            "trend_h0": trend_loss_h0,
            "trend_h1": trend_loss_h1,
            "trend_h2": trend_loss_h2,
            "local_h0": local_h0,
            "global_h0": global_h0,
            "extended_h0": extended_h0,
            "local_h1": local_h1,
            "global_h1": global_h1,
            "extended_h1": extended_h1,
            "local_h2": local_h2,
            "global_h2": global_h2,
            "extended_h2": extended_h2,
            "dir_loss": dir_loss_total,
            "dir_loss_h0": dir_h0,
            "dir_loss_h1": dir_h1,
            "dir_loss_h2": dir_h2,
            "nll_loss": nll_total,
            "nll_h0": nll_h0,
            "nll_h1": nll_h1,
            "nll_h2": nll_h2,
            "reg_loss": reg_val,
            "inter_reg": inter_reg,
            "vol_loss": vol_loss,
            **metrics_head,
            **metrics_gauss
        }

    def _compute_direction_metrics(self, true_dir_h0, true_dir_h1, true_dir_h2, dir_pred_h0, dir_pred_h1, dir_pred_h2, mask_h0=None, mask_h1=None, mask_h2=None, prefix=""):
        """
        Compute per-horizon direction classification metrics.
        Returns dict with accuracy, F1, sensitivity, specificity, MCC for each horizon.
        """
        metrics = {}

        masks = {
            "h0": tf.ones_like(true_dir_h0) if mask_h0 is None else tf.cast(mask_h0, tf.float32),
            "h1": tf.ones_like(true_dir_h1) if mask_h1 is None else tf.cast(mask_h1, tf.float32),
            "h2": tf.ones_like(true_dir_h2) if mask_h2 is None else tf.cast(mask_h2, tf.float32),
        }

        for horizon_idx, (h_name, true_dir, dir_pred) in enumerate([
            ("h0", true_dir_h0, dir_pred_h0),
            ("h1", true_dir_h1, dir_pred_h1),
            ("h2", true_dir_h2, dir_pred_h2)
        ]):
            # Binary predictions
            pred_dir_binary = tf.cast(dir_pred > 0.5, tf.float32)

            m = masks[h_name]

            # Confusion matrix elements
            TP = tf.reduce_sum(pred_dir_binary * true_dir * m)
            TN = tf.reduce_sum((1.0 - pred_dir_binary) * (1.0 - true_dir) * m)
            FP = tf.reduce_sum(pred_dir_binary * (1.0 - true_dir) * m)
            FN = tf.reduce_sum((1.0 - pred_dir_binary) * true_dir * m)

            # Accuracy
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            metrics[f"{prefix}dir_acc_{h_name}"] = accuracy

            # Sensitivity (True Positive Rate / Recall for UP class)
            sensitivity = TP / (TP + FN + 1e-8)
            metrics[f"{prefix}dir_sensitivity_{h_name}"] = sensitivity

            # Specificity (True Negative Rate)
            specificity = TN / (TN + FP + 1e-8)
            metrics[f"{prefix}dir_specificity_{h_name}"] = specificity

            # F1 Score (harmonic mean of precision and recall)
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-8)
            metrics[f"{prefix}dir_f1_{h_name}"] = f1

            # Matthews Correlation Coefficient (balanced metric for binary classification)
            mcc_numerator = (TP * TN) - (FP * FN)
            mcc_denominator = tf.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-8)
            mcc = mcc_numerator / mcc_denominator
            metrics[f"{prefix}dir_mcc_{h_name}"] = mcc

            # CALIBRATION METRICS: Per-class prediction rates to detect bias
            # pred_up_rate: % of predictions that are UP (should be ~50% for balanced predictions)
            total_samples = TP + TN + FP + FN + 1e-8
            pred_up_rate = (TP + FP) / total_samples  # Predictions that are UP
            true_up_rate = (TP + FN) / total_samples  # Ground truth that is UP
            metrics[f"{prefix}pred_up_rate_{h_name}"] = pred_up_rate
            metrics[f"{prefix}true_up_rate_{h_name}"] = true_up_rate
            
            # Mean predicted probability (should be ~0.5 for calibrated model)
            mean_prob = tf.reduce_sum(dir_pred * m) / (tf.reduce_sum(m) + 1e-8)
            metrics[f"{prefix}mean_dir_prob_{h_name}"] = mean_prob

        return metrics

    def test_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        y_pred = self(x_window, training=False)
        loss_components = self.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)

        # Unpack 22-component tuple
        (total_loss_val,
         point_h0, point_h1, point_h2,
         local_h0, global_h0, extended_h0,
         local_h1, global_h1, extended_h1,
         local_h2, global_h2, extended_h2,
         dir_h0, dir_h1, dir_h2,
         nll_h0, nll_h1, nll_h2,
         reg_val, inter_reg, vol_loss) = loss_components

        # Compute direction labels with the same trade-aware deadband used in training loss.
        y_true = tf.cast(y_true, tf.float32)
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, 3]
        last_close_squeeze = tf.squeeze(last_close, axis=1)

        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        # Targets are deltas; compute returns as delta / last_close (matches train_step)
        ret_h0 = (y_true_raw[:, 0]) / (last_close_squeeze + self.eps)
        ret_h1 = (y_true_raw[:, 1]) / (last_close_squeeze + self.eps)
        ret_h2 = (y_true_raw[:, 2]) / (last_close_squeeze + self.eps)

        mask_h0 = tf.cast(tf.abs(ret_h0) > deadband, tf.float32)
        mask_h1 = tf.cast(tf.abs(ret_h1) > deadband, tf.float32)
        mask_h2 = tf.cast(tf.abs(ret_h2) > deadband, tf.float32)

        true_dir_h0 = tf.cast(ret_h0 > deadband, tf.float32)
        true_dir_h1 = tf.cast(ret_h1 > deadband, tf.float32)
        true_dir_h2 = tf.cast(ret_h2 > deadband, tf.float32)

        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred
        dir_pred_h0 = tf.squeeze(dir_pred_h0, axis=1)
        dir_pred_h1 = tf.squeeze(dir_pred_h1, axis=1)
        dir_pred_h2 = tf.squeeze(dir_pred_h2, axis=1)

        # Gaussian-implied P(up) from (mu, var)
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e4), tf.float32)
        var_h0_c = tf.clip_by_value(tf.squeeze(var_h0, axis=1), var_floor, var_cap)
        var_h1_c = tf.clip_by_value(tf.squeeze(var_h1, axis=1), var_floor, var_cap)
        var_h2_c = tf.clip_by_value(tf.squeeze(var_h2, axis=1), var_floor, var_cap)
        mu_h0 = tf.squeeze(price_h0, axis=1)
        mu_h1 = tf.squeeze(price_h1, axis=1)
        mu_h2 = tf.squeeze(price_h2, axis=1)
        # Threshold for "UP" in scaled-delta space, consistent with direction labeling
        deadband_delta_scaled = (deadband * last_close_squeeze) / (self.pred_scale + self.eps)
        gauss_p_up_h0 = self._normal_cdf((mu_h0 - deadband_delta_scaled) / (tf.sqrt(var_h0_c) + self.eps))
        gauss_p_up_h1 = self._normal_cdf((mu_h1 - deadband_delta_scaled) / (tf.sqrt(var_h1_c) + self.eps))
        gauss_p_up_h2 = self._normal_cdf((mu_h2 - deadband_delta_scaled) / (tf.sqrt(var_h2_c) + self.eps))

        # IMPORTANT: do NOT prefix with "val_" here. Keras automatically prefixes
        # validation metrics with "val_"; adding it ourselves creates "val_val_*" keys.
        metrics_head = self._compute_direction_metrics(
            true_dir_h0, true_dir_h1, true_dir_h2,
            dir_pred_h0, dir_pred_h1, dir_pred_h2,
            mask_h0=mask_h0, mask_h1=mask_h1, mask_h2=mask_h2,
            prefix=""
        )
        metrics_gauss = self._compute_direction_metrics(
            true_dir_h0, true_dir_h1, true_dir_h2,
            gauss_p_up_h0, gauss_p_up_h1, gauss_p_up_h2,
            mask_h0=mask_h0, mask_h1=mask_h1, mask_h2=mask_h2,
            prefix="gauss_"
        )

        # Trend metrics: margins (bps), agreement rates, magnitudes (bps)
        trend_margin_h0 = tf.reduce_mean(tf.abs(y_true_raw[:, 0] - extended_trends[:, 0] * last_close_squeeze)) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        trend_margin_h1 = tf.reduce_mean(tf.abs(y_true_raw[:, 1] - extended_trends[:, 1] * last_close_squeeze)) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        trend_margin_h2 = tf.reduce_mean(tf.abs(y_true_raw[:, 2] - extended_trends[:, 2] * last_close_squeeze)) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        agreement_rate_h0 = tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true_raw[:, 0]), tf.sign(extended_trends[:, 0])), tf.float32))
        agreement_rate_h1 = tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true_raw[:, 1]), tf.sign(extended_trends[:, 1])), tf.float32))
        agreement_rate_h2 = tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true_raw[:, 2]), tf.sign(extended_trends[:, 2])), tf.float32))
        magnitude_bps_h0 = tf.reduce_mean(tf.abs(y_true_raw[:, 0])) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        magnitude_bps_h1 = tf.reduce_mean(tf.abs(y_true_raw[:, 1])) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)
        magnitude_bps_h2 = tf.reduce_mean(tf.abs(y_true_raw[:, 2])) * 10000 / (tf.reduce_mean(last_close_squeeze) + self.eps)

        # Test step
        # Loss components
        point_loss_total = point_h0 + point_h1 + point_h2
        trend_loss_h0 = local_h0 + global_h0 + extended_h0
        trend_loss_h1 = local_h1 + global_h1 + extended_h1
        trend_loss_h2 = local_h2 + global_h2 + extended_h2
        trend_loss_total = trend_loss_h0 + trend_loss_h1 + trend_loss_h2
        dir_loss_total = dir_h0 + dir_h1 + dir_h2
        nll_total = nll_h0 + nll_h1 + nll_h2

        return {
            "loss": total_loss_val,
            "point_loss": point_loss_total,
            "point_h0": point_h0,
            "point_h1": point_h1,
            "point_h2": point_h2,
            "trend_loss": trend_loss_total,
            "trend_h0": trend_loss_h0,
            "trend_h1": trend_loss_h1,
            "trend_h2": trend_loss_h2,
            "local_h0": local_h0,
            "global_h0": global_h0,
            "extended_h0": extended_h0,
            "local_h1": local_h1,
            "global_h1": global_h1,
            "extended_h1": extended_h1,
            "local_h2": local_h2,
            "global_h2": global_h2,
            "extended_h2": extended_h2,
            "dir_loss": dir_loss_total,
            "dir_loss_h0": dir_h0,
            "dir_loss_h1": dir_h1,
            "dir_loss_h2": dir_h2,
            "nll_loss": nll_total,
            "nll_h0": nll_h0,
            "nll_h1": nll_h1,
            "nll_h2": nll_h2,
            "reg_loss": reg_val,
            "inter_reg": inter_reg,
            "vol_loss": vol_loss,
            **metrics_head,
            **metrics_gauss
        }

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

    set_if_missing("dir_acc_avg", _first_present({"v": _mean_present(out, train_acc_keys), "v2": _mean_present(out, train_acc_fb)}, ["v", "v2"]))
    set_if_missing("f1_avg", _first_present({"v": _mean_present(out, train_f1_keys), "v2": _mean_present(out, train_f1_fb)}, ["v", "v2"]))
    set_if_missing("dir_sensitivity_avg", _first_present({"v": _mean_present(out, train_sens_keys), "v2": _mean_present(out, train_sens_fb)}, ["v", "v2"]))
    set_if_missing("dir_specificity_avg", _first_present({"v": _mean_present(out, train_spec_keys), "v2": _mean_present(out, train_spec_fb)}, ["v", "v2"]))

    set_if_missing("val_dir_acc_avg", _first_present({"v": _mean_present(out, val_acc_keys), "v2": _mean_present(out, val_acc_fb)}, ["v", "v2"]))
    set_if_missing("val_f1_avg", _first_present({"v": _mean_present(out, val_f1_keys), "v2": _mean_present(out, val_f1_fb)}, ["v", "v2"]))
    set_if_missing("val_dir_sensitivity_avg", _first_present({"v": _mean_present(out, val_sens_keys), "v2": _mean_present(out, val_sens_fb)}, ["v", "v2"]))
    set_if_missing("val_dir_specificity_avg", _first_present({"v": _mean_present(out, val_spec_keys), "v2": _mean_present(out, val_spec_fb)}, ["v", "v2"]))

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

    def _calculate_param_changes(self, current_params):
        """Calculate per-parameter change rates for convergence detection."""
        if self.prev_params is None:
            return None

        changes = {}
        for key in current_params:
            if key in self.prev_params:
                try:
                    prev_val = float(self.prev_params[key])
                    curr_val = float(current_params[key])
                    if prev_val is not None and curr_val is not None:
                        # Calculate relative change (avoid division by zero)
                        if abs(prev_val) > 1e-6:
                            change_pct = abs(curr_val - prev_val) / abs(prev_val) * 100.0
                        else:
                            change_pct = abs(curr_val - prev_val) * 100.0
                        changes[f'change_{key}'] = float(change_pct)
                except (ValueError, TypeError):
                    # Skip non-numeric columns like 'timestamp', 'epoch'
                    pass
        return changes if changes else {}

    def _detect_convergence(self, rows_window):
        """
        Detect convergence vs drift patterns over recent epochs.
        Returns convergence score (0-1, higher = more converged).
        """
        if len(rows_window) < 2:
            return None

        # Extract change rates
        change_keys = [k for k in rows_window[0].keys() if k.startswith('change_')]
        if not change_keys:
            return None

        # Calculate mean and std of changes across window
        changes_over_window = []
        for row in rows_window:
            for key in change_keys:
                if key in row:
                    changes_over_window.append(row[key])

        if not changes_over_window:
            return None

        mean_change = np.mean(changes_over_window)
        std_change = np.std(changes_over_window)

        # Convergence score: lower mean change = more converged (0-1 scale)
        # Normalize to [0, 1] where convergence means mean_change < 1%
        convergence_score = max(0.0, min(1.0, 1.0 - (mean_change / 10.0)))

        return {
            'convergence_score': float(convergence_score),
            'mean_param_change_pct': float(mean_change),
            'std_param_change_pct': float(std_change)
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
            df = pd.DataFrame(self.rows)
            df.to_csv(self.out_csv, index=False)

            # Log convergence status periodically (every 5 epochs)
            if epoch % 5 == 0 or epoch < 3:
                if 'convergence_score' in params:
                    conv_score = params['convergence_score']
                    mean_change = params['mean_param_change_pct']
                    status = "converging" if conv_score > 0.7 else "active" if conv_score > 0.3 else "drifting"
                    print(f"Epoch {epoch}: Params {status} (convergence={conv_score:.3f}, "
                          f"mean_change={mean_change:.2f}%)")
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
                    print(f"Final Convergence Score: {convergence_info['convergence_score']:.3f}")
                    print(f"Final Mean Change: {convergence_info['mean_param_change_pct']:.2f}%")
                    print(f"Final Std Change: {convergence_info['std_param_change_pct']:.2f}%")

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
            for h_key, label in zip(m.get('meta', {}).get('horizon_keys', ['h0','h1','h2']), m.get('meta', {}).get('horizon_labels', ['1-min','5-min','15-min'])):
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
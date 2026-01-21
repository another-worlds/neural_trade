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
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error, f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from losses import Losses
import losses as _losses
# Loss functions (custom) are implemented centrally in `losses.py` to
# maintain a single authoritative source and avoid duplication.
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
    RESAMPLE_MINUTES = 1  # Optionally aggregate to coarser bars (e.g., set to 5 for 5-minute bars)
    BATCH_SIZE = 540
    EPOCHS = 20
    LR = 1e-3  # Fixed from critically low 1e-10; reasonable for Adam optimizer
    PATIENCE = EPOCHS
    MAX_SEQUENCE_COUNT = 1440 * 2  # Limit most recent sequences to bound training size



    # Use integer periods to avoid float indexing issues
    # Extended trend features are computed as percent-change over these lags (in minutes)
    EXTENDED_TREND_PERIODS = [1, 5, 15]  # 1m, 5m, 15m

    # Supervision horizons (in minutes ahead from last_close).
    # These define the output towers dynamically (one tower per horizon).
    # Example: [1, 5, 15] creates 3 towers for 1min, 5min, 15min horizons
    # The model architecture scales automatically to len(HORIZON_STEPS) towers.
    HORIZON_STEPS = [1, 5, 15]

# Loss Function Weights
    DAMPING = 0.5
    LAMBDA_LOCAL_TREND  = 0.2
    LAMBDA_GLOBAL_TREND =  0.2
    LAMBDA_EXTENDED_TREND = 1.0
    LAMBDA_QUANTILE = 0.1
    REG_MOMENTUM_L2 = 0
    MOMENTUM_CLIP_MIN = 1.0
    MOMENTUM_CLIP_MAX = LOOKBACK
    USE_HUBER = True
    
    # === HORIZON-SPECIFIC LOSS WEIGHTS ===
    # Per-horizon lambda weights for point loss (delta prediction accuracy)
    # Different horizons may have different importance/difficulty:
    # - h0 (1-min): Short-term noise, harder to predict, may need lower weight to avoid overfitting noise
    # - h1 (5-min): Primary horizon, balanced signal/noise, standard weight
    # - h2 (15-min): Long-term trend, more stable, higher weight to enforce consistency
    LAMBDA_SHORT = 0.5   # h0 (1-min):  Reduced from 1.0 to avoid noise overfitting
    LAMBDA_POINT = 1.0   # h1 (5-min):  Primary horizon baseline
    LAMBDA_LONG = 0.5   # h2 (15-min): Increased from 1.0 to enforce long-term consistency
    
    # Auxiliary loss weights
    LAMBDA_DIR = 5.0  # Direction classification (focal loss)
    LAMBDA_INTER = 0.1  # Interconnection regularization between horizons
    LAMBDA_VOL = 0.1  # Volatility penalty (weak constraint)
    LAMBDA_VAR = 5.0  # Variance NLL (confidence estimation)

    # Variance calibration bounds (in scaled space)
    VAR_FLOOR = 0.1  # Minimum variance = 0.1 (prevents overconfidence, std ≈ 0.316)
    VAR_CAP = 1e4   # Maximum variance = 10000 (allows high uncertainty)
    
    # Calibration setup
    MAX_LAMBDA = 1.0
    MIN_LAMBDA = 0.1

# paths
    # MODEL_PATH v3: Modular multi-horizon architecture with dynamic tower generation
    # - N independent output towers (one per horizon in HORIZON_STEPS)
    # - 3N outputs total (price + direction + variance per horizon)
    # - Example: [1, 5, 15] → 9 outputs (3 horizons × 3 heads each)
    # - Focal loss for direction heads with dynamic alpha tuning
    # - Per-horizon direction metrics: accuracy, F1, sensitivity, specificity, MCC
    # - True multi-horizon supervision with validated extended trend alignment
    MODEL_PATH = "nn_learnable_indicators_v3.weights.h5"
    SCALER_PATH = "scaler_v3.joblib"

# TA initial params
    MA_SPANS = [5, 15, 30] 
    MACD_SETTINGS = [
        {'fast': 15, 'slow': 30, 'signal': 5},
        {'fast': 21, 'slow': 30, 'signal': 14},
        {'fast': 30, 'slow': 45, 'signal': 21}
    ]
    RSI_PERIODS = [9, 21, 45]  
    BB_PERIODS = [10, 20, 45]  

# Activation function settings
    TANH_SCALE = 3.0
    HUBER_DELTA = 3.0
    SIGMOID_SCALE = 3.0

# Training stability controls
    INDICATOR_GRAD_MULT = 1.0
    GRAD_CLIP_NORM = 10.0

# Focal loss hyperparameters for direction classification
    # NOTE: alpha weights DOWN class (label=0), (1-alpha) weights UP (label=1)
    # Class weighting was NOT the cause of DOWN bias - root causes were:
    # 1. Weak direction loss weight (fixed: 0.2 → 0.5)
    # 2. Zero deadband creating label noise (fixed: 5 bps)
    FOCAL_ALPHA = 0.5  # Balanced class weights (was 0.7, now neutral)
    FOCAL_GAMMA = 2  # Focus parameter for hard examples

    # Trade-aware direction labeling deadband.
    # If > 0, direction loss/metrics treat returns within +/- deadband as neutral.
    # Units: basis points (bps). Example: 10 bps = 0.10%.
    # CRITICAL FIX: Non-zero deadband filters label noise from tiny price moves
    DIR_DEADBAND_BPS = 0.05  # 5 bps = 0.05% minimum move for UP classification

    # Stabilize NLL and prevent variance head from dominating early.
    # Variance is in SCALED units^2.
    VAR_FLOOR = 1e-4
    VAR_CAP = 1e4

    # Align direction head with distribution-implied P(up) from (mu, var).
    # Setting this > 0 helps avoid degenerate constant direction probabilities.
    LAMBDA_DIR_ALIGN = 0.7

    # === HORIZON MODULARITY: Dynamic Properties ===
    def __init__(self):
        """Initialize and validate configuration."""
        self.validate()

    @property
    def num_horizons(self):
        """Number of prediction horizons (dynamically computed from HORIZON_STEPS)."""
        return len(self.HORIZON_STEPS)

    @property
    def horizon_keys(self):
        """Generate horizon keys dynamically: ('h0', 'h1', 'h2', ..., 'hN').

        These keys are used throughout the codebase to reference specific horizons
        in a modular way, avoiding hardcoded literals like "h0", "h1", "h2".
        """
        return tuple(f"h{i}" for i in range(self.num_horizons))

    @property
    def horizon_lambda_weights(self):
        """Get per-horizon loss weights as a list [lambda_h0, lambda_h1, lambda_h2, ...].

        For backward compatibility, maps the first 3 horizons to LAMBDA_SHORT, LAMBDA_POINT, LAMBDA_LONG.
        For additional horizons beyond 3, defaults to LAMBDA_POINT (1.0).

        This allows the model to work with any number of horizons while preserving
        the existing 3-horizon weight configuration.
        """
        weights = []
        for i in range(self.num_horizons):
            if i == 0:
                weights.append(self.LAMBDA_SHORT)
            elif i == 1:
                weights.append(self.LAMBDA_POINT)
            elif i == 2:
                weights.append(self.LAMBDA_LONG)
            else:
                # Default to LAMBDA_POINT for any additional horizons
                weights.append(self.LAMBDA_POINT)
        return weights

    def validate(self):
        """Validate horizon configuration for consistency and correctness.

        Raises:
            ValueError: If configuration is invalid
        """
        # 1. HORIZON_STEPS must be non-empty
        if not self.HORIZON_STEPS:
            raise ValueError("Config.HORIZON_STEPS must be a non-empty list")

        # 2. All horizon steps must be positive integers
        if not all(isinstance(h, int) and h > 0 for h in self.HORIZON_STEPS):
            raise ValueError(
                f"All HORIZON_STEPS must be positive integers, got {self.HORIZON_STEPS}"
            )

        # 3. Horizons must be in ascending order (for coherence constraints to make sense)
        if self.HORIZON_STEPS != sorted(self.HORIZON_STEPS):
            raise ValueError(
                f"HORIZON_STEPS must be in ascending order, got {self.HORIZON_STEPS}"
            )

        # 4. CRITICAL: EXTENDED_TREND_PERIODS must exactly match HORIZON_STEPS
        # This is required because the loss function assumes extended_trends[:, i]
        # corresponds to HORIZON_STEPS[i]. Any mismatch will cause semantic errors.
        if self.EXTENDED_TREND_PERIODS != self.HORIZON_STEPS:
            raise ValueError(
                f"EXTENDED_TREND_PERIODS {self.EXTENDED_TREND_PERIODS} must exactly match "
                f"HORIZON_STEPS {self.HORIZON_STEPS} for target-feature alignment in loss computation. "
                f"These two lists define the same time scales for extended trend features and prediction targets."
            )

        # 5. Horizon steps must fit within the lookback window
        max_horizon = max(self.HORIZON_STEPS)
        if max_horizon >= self.LOOKBACK:
            raise ValueError(
                f"Maximum horizon step ({max_horizon}) must be less than "
                f"LOOKBACK ({self.LOOKBACK}) to ensure valid sequences"
            )

        # 6. All lambda weights must be non-negative
        lambda_attrs = [
            'LAMBDA_SHORT', 'LAMBDA_POINT', 'LAMBDA_LONG', 'LAMBDA_DIR',
            'LAMBDA_INTER', 'LAMBDA_VOL', 'LAMBDA_VAR', 'LAMBDA_EXTENDED_TREND',
            'LAMBDA_LOCAL_TREND', 'LAMBDA_GLOBAL_TREND'
        ]
        for attr in lambda_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value < 0:
                    raise ValueError(f"{attr} must be non-negative, got {value}")

# -----------------------------
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.target_scaler = None
        self.input_scaler = None

        # Validate horizon configuration (critical for data processing correctness)
        if hasattr(config, 'EXTENDED_TREND_PERIODS') and hasattr(config, 'HORIZON_STEPS'):
            if config.EXTENDED_TREND_PERIODS != config.HORIZON_STEPS:
                raise ValueError(
                    f"DataProcessor requires EXTENDED_TREND_PERIODS {config.EXTENDED_TREND_PERIODS} "
                    f"to match HORIZON_STEPS {config.HORIZON_STEPS} for target-feature alignment. "
                    f"These two lists define the same time scales for extended trend features and prediction targets."
                )

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

    horizon_keys = config.horizon_keys
    num_horizons = config.num_horizons
    # Compute human-readable horizon labels based on HORIZON_STEPS and RESAMPLE_MINUTES.
    def _format_tf(minutes: int) -> str:
        # Prefer days/hours when evenly divisible, otherwise show minutes
        if minutes % 1440 == 0:
            days = minutes // 1440
            return f"{days}d" if days > 1 else "1d"
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"{hours}h"
        return f"{minutes}min"

    try:
        horizon_steps = list(getattr(config, 'HORIZON_STEPS', [1, 5, 15]))
    except Exception:
        horizon_steps = [1, 5, 15]

    horizon_names = tuple(_format_tf(int(step * getattr(config, 'RESAMPLE_MINUTES', 1))) for step in horizon_steps)
    y_true_deltas = np.asarray(y_true_deltas)
    if y_true_deltas.ndim != 2 or y_true_deltas.shape[1] != num_horizons:
        raise ValueError(f"Expected y_true_deltas shape (N,{num_horizons}), got {y_true_deltas.shape}")
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

    for idx, (h_key, h_label) in enumerate(zip(horizon_keys, horizon_names)):
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
        mae_delta = float(np.mean(np.abs(y_t - y_p)))
        # Note: Explained Variance CAN be negative when predictions are poor (like R²)
        # EV < 0 means predictions are worse than predicting the mean
        ev_delta = explained_variance_score(y_t, y_p)
        # Correlation coefficient is bounded [-1, 1] and measures linear relationship
        # More robust than EV for evaluating prediction quality
        corr_delta = float(np.corrcoef(y_t, y_p)[0, 1]) if len(y_t) > 1 else 0.0
        corr_delta = 0.0 if np.isnan(corr_delta) else corr_delta

        delta_metrics = {
            "mse": float(mse_delta),
            "rmse": float(rmse_delta),
            "mae": float(mae_delta),
            "ev": float(ev_delta),  # Can be negative if predictions are poor
            "corr": corr_delta,  # Pearson correlation [-1, 1]
        }
        if safe_mape is not None and smape is not None and wape is not None and reconstruct_prices is not None:
            delta_metrics.update({
                "mape_delta": float(mean_absolute_percentage_error(y_t, y_p)),
                "safe_mape_delta": float(safe_mape(y_t, y_p, min_abs_y=min_abs_delta_for_mape)),
                "smape_delta": float(smape(y_t, y_p)),
                "wape_delta": float(wape(y_t, y_p)),
            })

        out["delta"][h_key] = delta_metrics

        # CRITICAL: Price-space EV is the most interpretable metric for price prediction.
        # Reconstruct prices: price[t+h] = last_close[t] + delta[t, t+h]
        # EV in price space measures how well cumulative predictions track actual future prices.
        y_true_price = lc_h + y_t  # Simple reconstruction: last_close + delta
        y_pred_price = lc_h + y_p

        # In price space, EV is more stable because:
        # 1. Price levels have larger variance than deltas
        # 2. EV measures the fraction of price-level variance explained
        # 3. This aligns with trading objectives (predicting future prices, not just changes)
        ev_price_simple = explained_variance_score(y_true_price, y_pred_price)
        corr_price = float(np.corrcoef(y_true_price, y_pred_price)[0, 1]) if len(y_true_price) > 1 else 0.0
        corr_price = 0.0 if np.isnan(corr_price) else corr_price
        
        price_metrics = {
            "ev": float(ev_price_simple),  # Explained variance in price space
            "mse": float(mean_squared_error(y_true_price, y_pred_price)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
            "corr": corr_price,  # Pearson correlation in price space
        }
        
        if safe_mape is not None and smape is not None and wape is not None and reconstruct_prices is not None:
            # Use the more sophisticated reconstruction if available for additional metrics
            y_true_price_soph = reconstruct_prices(lc_h, y_t)
            y_pred_price_soph = reconstruct_prices(lc_h, y_p)
            price_metrics.update({
                "ev_soph": float(explained_variance_score(y_true_price_soph, y_pred_price_soph)),
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

    # --- Metric classification helpers (module-level so they can be tested) ---
    def classify_by_threshold(val, good, stable, higher_better=True):
        try:
            v = float(val)
        except Exception:
            return 'bad'
        if higher_better:
            if v >= good:
                return 'good'
            elif v >= stable:
                return 'stable'
            else:
                return 'bad'
        else:
            if v <= good:
                return 'good'
            elif v <= stable:
                return 'stable'
            else:
                return 'bad'

    def classify_dir_acc(acc):
        return classify_by_threshold(acc, good=0.6, stable=0.55, higher_better=True)

    def classify_ece(ece):
        return classify_by_threshold(ece, good=0.05, stable=0.1, higher_better=False)

    def classify_brier(brier):
        return classify_by_threshold(brier, good=0.05, stable=0.1, higher_better=False)

    def classify_stability_metric(stab):
        try:
            s = float(stab)
        except Exception:
            return 'bad'
        return 'good' if s > 0.8 else ('stable' if s > 0.5 else 'bad')

    def classify_convergence_metric(conv):
        try:
            c = float(conv)
        except Exception:
            return 'bad'
        return 'good' if c > 0.001 else ('stable' if abs(c) < 0.001 else 'bad')

    def classify_gen_gap_metric(gap):
        try:
            g = float(gap)
        except Exception:
            return 'bad'
        return 'good' if g < 0.1 else ('stable' if g < 0.3 else 'bad')

    def classify_coherence_metric(coh):
        try:
            c = float(coh)
        except Exception:
            return 'bad'
        return 'good' if c > 0.8 else ('stable' if c > 0.5 else 'bad')

    def compute_metrics_y_range(history: dict, keys=None, center: float = 0.5, min_half: float = 0.02, pad_frac: float = 0.1):
        """Compute a y-axis (min,max) range for direction-like metrics so small oscillations around center are visible.

        - history: dict of lists (same structure as callback.history)
        - keys: list of metric keys to consider; defaults to direction & balanced-accuracy metrics
        - center: baseline center (0.5)
        - min_half: minimum half-range around center
        - pad_frac: fractional padding applied to half-range
        """
        if keys is None:
            keys = ['dir_acc_avg', 'val_dir_acc_avg', 'bal_acc_avg', 'val_bal_acc_avg']
        metric_values = []
        for k in keys:
            vals = history.get(k)
            if vals:
                try:
                    metric_values.extend([float(v) for v in vals if v is not None])
                except Exception:
                    pass
        if metric_values:
            min_v = min(metric_values)
            max_v = max(metric_values)
            half_range = max(abs(center - min_v), abs(max_v - center), float(min_half))
            padding = max(0.01, half_range * float(pad_frac))
            y_min = max(0.0, center - half_range - padding)
            y_max = min(1.0, center + half_range + padding)
            return y_min, y_max
        return -0.05, 1.05

    def compute_loss_y_range(loss_sequence, pad_frac: float = 0.1, min_pad: float = 1e-6):
        """Compute a y-axis range for loss sequences with proportional padding.

        - loss_sequence: iterable of loss values (numbers)
        - pad_frac: fractional padding applied to the span
        - min_pad: minimum absolute padding
        Returns (y_min, y_max)
        """
        try:
            arr = np.array([float(x) for x in loss_sequence if x is not None])
            if arr.size == 0:
                return None
            min_v = float(np.min(arr))
            max_v = float(np.max(arr))
            span = max_v - min_v
            if span <= 0.0:
                # Single-valued series; provide a small symmetric padding
                pad = max(min_pad, max(abs(min_v) * 0.01, 1e-3))
            else:
                pad = max(min_pad, span * float(pad_frac))
            y_min = max(0.0, min_v - pad)
            y_max = max_v + pad
            return y_min, y_max
        except Exception:
            return None

    def classify_epoch_loss(curr_val_loss, val_history=None, improvement_threshold=0.01, stable_threshold=0.002):
        """Classify epoch loss trend based on the current validation loss and history.

        Returns: ('good'|'stable'|'bad', explanation_dict)
        Explanation dict contains: prev_val, delta_pct, best_val, best_idx
        """
        try:
            curr = float(curr_val_loss)
        except Exception:
            return 'bad', {'reason': 'invalid_current_loss'}

        hist = list(val_history) if val_history is not None else []
        prev = None
        if len(hist) >= 2:
            prev = float(hist[-2])
        elif len(hist) == 1:
            prev = float(hist[0])

        best_val = min(hist) if hist else curr
        best_idx = int(hist.index(best_val)) if hist else 0

        delta_pct = None
        if prev is None or prev == 0:
            # Not enough history — mark as stable
            status = 'stable'
            delta_pct = 0.0
        else:
            delta_pct = (prev - curr) / prev
            if delta_pct >= improvement_threshold:
                status = 'good'
            elif abs(delta_pct) < stable_threshold:
                status = 'stable'
            else:
                status = 'bad'

        explanation = {
            'prev_val': prev,
            'delta_pct': delta_pct,
            'best_val': best_val,
            'best_idx': best_idx,
        }
        return status, explanation

    def metric_limits_text(name: str) -> str:
        """Return a short human-readable string describing the Good/Stable boundaries for a named metric.

        This is used by the dashboard legend to always show the numerical limits next to each tracked metric.
        """
        if not isinstance(name, str):
            return ''
        n = name.strip().lower()
        if 'dir' in n and 'acc' in n:
            return 'Good: ≥0.60; Stable: ≥0.55'
        if 'ece' in n:
            return 'Good: ≤0.05; Stable: ≤0.10'
        if 'brier' in n:
            return 'Good: ≤0.05; Stable: ≤0.10'
        if 'stability' in n:
            return 'Good: >0.80; Stable: >0.50'
        if 'gen gap' in n or ('gap' in n and 'gen' in n) or (n == 'gen gap'):
            return 'Good: <0.10; Stable: <0.30'
        if 'coherence' in n:
            return 'Good: >0.80; Stable: >0.50'
        if 'epoch loss' in n or ('epoch' in n and 'loss' in n) or n == 'loss':
            # Uses classify_epoch_loss thresholds: improvement_threshold=1%, stable_threshold=0.2%
            return 'Good: ≥1.00% improvement; Stable: |Δ| <0.20%'
        return ''

    def summarize_indicator_params(csv_path: str = 'indicator_params_history.csv', last_n: int = 3):
        """Read indicator params CSV and return a compact summary useful for the dashboard.

        Returns a dict with keys:
            - 'convergence_score' (float or None)
            - 'mean_param_change_pct' (float or None)
            - 'top_changes' : list of (param_name, change_pct, current_value)
            - 'group_stats' : dict mapping group -> {avg, min, max, count}
            - 'csv_path'
        The function is resilient to missing files and non-numeric columns.
        """
        try:
            import pandas as pd
            if not os.path.exists(csv_path):
                return {'convergence_score': None, 'mean_param_change_pct': None, 'top_changes': [], 'group_stats': {}, 'csv_path': csv_path}
            df = pd.read_csv(csv_path)
            if df.empty:
                return {'convergence_score': None, 'mean_param_change_pct': None, 'top_changes': [], 'group_stats': {}, 'csv_path': csv_path}

            # Consider last row as most recent values
            last = df.iloc[-1]
            # Collect change_* columns
            change_cols = [c for c in df.columns if c.startswith('change_')]
            top_changes = []
            changes = []
            if change_cols:
                for c in change_cols:
                    try:
                        val = float(last.get(c, 0.0))
                    except Exception:
                        val = 0.0
                    param_name = c[len('change_'):]
                    curr_val = None
                    if param_name in df.columns:
                        try:
                            curr_val = float(last.get(param_name))
                        except Exception:
                            curr_val = None
                    changes.append((param_name, float(val), curr_val))
                # Sort by absolute change descending
                changes.sort(key=lambda x: x[1], reverse=True)
                top_changes = changes[:min(len(changes), 5)]

            # Group statistics: group by prefix before first underscore (e.g., 'ma_period_0' -> 'ma')
            group_stats = {}
            for pname, change_pct, curr_val in changes:
                grp = pname.split('_')[0] if pname and isinstance(pname, str) else 'other'
                lst = group_stats.setdefault(grp, [])
                lst.append(float(change_pct))

            # compute avg/min/max per group
            for grp, lst in list(group_stats.items()):
                if not lst:
                    group_stats[grp] = {'avg': None, 'min': None, 'max': None, 'count': 0}
                else:
                    arr = np.array(lst, dtype=float)
                    group_stats[grp] = {
                        'avg': float(np.mean(arr)),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                        'count': int(len(arr)),
                    }

            conv_score = None
            mean_change = None
            if 'convergence_score' in df.columns:
                try:
                    conv_score = float(last.get('convergence_score'))
                except Exception:
                    conv_score = None
            if 'mean_param_change_pct' in df.columns:
                try:
                    mean_change = float(last.get('mean_param_change_pct'))
                except Exception:
                    mean_change = None

            return {
                'convergence_score': conv_score,
                'mean_param_change_pct': mean_change,
                'top_changes': top_changes,
                'group_stats': group_stats,
                'csv_path': csv_path,
            }
        except Exception:
            return {'convergence_score': None, 'mean_param_change_pct': None, 'top_changes': [], 'group_stats': {}, 'csv_path': csv_path}

    def load_indicator_params_df(csv_path: str = 'indicator_params_history.csv'):
        """Load the full indicator params CSV as a pandas DataFrame and normalize change columns.

        Returns an empty DataFrame if CSV is missing or empty.
        """
        try:
            import pandas as pd
            if not os.path.exists(csv_path):
                return pd.DataFrame()
            df = pd.read_csv(csv_path)
            if df.empty:
                return pd.DataFrame()
            # Normalize change_* columns to floats and rename them to parameter names
            change_cols = [c for c in df.columns if c.startswith('change_')]
            # Create a df_changes with columns param_name -> change_pct
            df_changes = pd.DataFrame()
            for c in change_cols:
                pname = c[len('change_'):]
                try:
                    df_changes[pname] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
                except Exception:
                    df_changes[pname] = 0.0

            # Also include parameter value columns (if present in CSV) as 'val_<param>' so plots
            # can show the actual indicator values over time. We keep original names for change
            # columns and add 'val_' prefixed columns for values to avoid collisions.
            for pname in list(df_changes.columns):
                if pname in df.columns:
                    try:
                        df_changes['val_' + pname] = pd.to_numeric(df[pname], errors='coerce')
                    except Exception:
                        df_changes['val_' + pname] = np.nan

            # Attach epoch index if present
            if 'epoch' in df.columns:
                df_changes['epoch'] = df['epoch'].astype(int)
            else:
                df_changes['epoch'] = list(range(len(df_changes)))
            return df_changes
        except Exception:
            import pandas as pd
            return pd.DataFrame()

    def plot_indicator_movements(df_changes, max_params_per_group: int = 6):
        """Return a plotly Figure with group mean change and per-parameter movement lines.

        Changes are interpreted as signed deltas (not absolute). If the original parameter
        values are present in the DataFrame they are expected as columns named 'val_<param>'
        and will be plotted under the corresponding movements subplot for each group.

        df_changes: DataFrame with columns for params and 'epoch'
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            if df_changes is None or df_changes.empty:
                return None

            epoch = df_changes['epoch'] if 'epoch' in df_changes else df_changes.index
            # identify groups (ignore 'val_' prefixed columns when determining param groups)
            param_cols = [c for c in df_changes.columns if c != 'epoch' and not c.startswith('val_')]
            groups = {}
            for p in param_cols:
                grp = p.split('_')[0] if isinstance(p, str) and '_' in p else 'other'
                groups.setdefault(grp, []).append(p)

            # Build subplots: first row = overall group mean change, then for each group two rows:
            # (a) parameter movements and (b) parameter values (if available)
            nrows = 1 + len(groups) * 2
            subplot_titles = ['Group mean change']
            for g in groups.keys():
                subplot_titles.append(f"{g} movements")
                subplot_titles.append(f"{g} values")

            fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                subplot_titles=subplot_titles)

            # Group mean change (signed mean across parameters in group)
            for gidx, (g, cols) in enumerate(groups.items()):
                arr = df_changes[cols].mean(axis=1)
                fig.add_trace(go.Scatter(x=epoch, y=arr, mode='lines+markers', name=f"{g} mean change", line=dict(width=2)), row=1, col=1)

            # Per-group parameter movements and values
            row = 2
            for g, cols in groups.items():
                # rank params by signed max change (largest positive first)
                ranks = sorted(cols, key=lambda c: df_changes[c].max(), reverse=True)
                # parameter movement lines
                for c in ranks[:max_params_per_group]:
                    fig.add_trace(go.Scatter(x=epoch, y=df_changes[c], mode='lines', name=f"{g}/{c}"), row=row, col=1)
                row += 1

                # parameter value lines (if 'val_<param>' columns exist)
                for c in ranks[:max_params_per_group]:
                    val_col = 'val_' + c
                    if val_col in df_changes.columns:
                        fig.add_trace(go.Scatter(x=epoch, y=df_changes[val_col], mode='lines', name=f"{g}/{c} value"), row=row, col=1)
                row += 1

            fig.update_layout(height=220 * nrows, showlegend=True, template='plotly_dark', title_text='Indicator Parameter Movements')
            fig.update_xaxes(title_text='Epoch', row=nrows, col=1)
            return fig
        except Exception:
            return None

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
                    
                    # Autoscale loss y-range if available
                    loss_range = None
                    if 'loss' in self.batch_history and self.batch_history.get('loss'):
                        loss_range = compute_loss_y_range(self.batch_history.get('loss', []))
                    if loss_range:
                        fig.update_yaxes(range=list(loss_range), row=1, col=1)
                    else:
                        # fallback to automatic padding
                        pass

                    # Autoscale metrics subplot (centered around 0.5)
                    y_min, y_max = compute_metrics_y_range(self.batch_history, keys=['dir_acc', 'f1'])
                    fig.update_yaxes(range=[y_min, y_max], row=2, col=1)
                    
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
                
                # Compute y-axis range using helper so small oscillations around 50% are visible
                y_min, y_max = compute_metrics_y_range(self.history)
                fig.update_yaxes(range=[y_min, y_max], row=2, col=1)
                
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
                
                # --- Evaluate metrics and prepare suggestions / legend highlighting
                # Direction quality
                dir_acc = logs.get('val_dir_acc_avg', logs.get('dir_acc_avg', 0.0))
                dir_status = classify_dir_acc(dir_acc)

                # Calibration / confidence metrics
                ece = logs.get('ece_avg', 0.0)
                ece_status = classify_ece(ece)

                brier = logs.get('brier_avg', 0.0)
                brier_status = classify_brier(brier)

                # Training health
                stab_status_calc = classify_stability_metric(stability)
                conv_status_calc = classify_convergence_metric(convergence_rate)
                gap_status_calc = classify_gen_gap_metric(gen_gap)
                coh_status_calc = classify_coherence_metric(coherence)

                # Compose suggestions (simple heuristic rules)
                suggestions = []
                if dir_status == 'bad':
                    suggestions.append('Improve direction calibration: increase training data, adjust class weights, or tune thresholds.')
                elif dir_status == 'stable':
                    suggestions.append('Direction metrics are acceptable; consider minor calibration improvements (ECE / Brier).')
                else:
                    suggestions.append('Direction predictions look healthy. Consider collecting more validation cases for robustness checks.')

                if gap_status_calc == 'bad':
                    suggestions.append('Large generalization gap → consider stronger regularization or early stopping.')
                elif gap_status_calc == 'stable':
                    suggestions.append('Moderate generalization gap → watch for overfitting and try small reg increases.')

                if stab_status_calc == 'bad':
                    suggestions.append('Unstable training → reduce learning rate or increase batch size.')

                if conv_status_calc == 'bad':
                    suggestions.append('Diverging training → try reducing learning rate / check data preprocessing.')

                if ece_status == 'bad' or brier_status == 'bad':
                    suggestions.append('Calibration is poor → consider temperature scaling or calibration layer.')

                # Epoch loss evaluation (fourth column)
                val_history = self.history.get('val_loss', [])
                curr_val_loss = logs.get('val_loss', 0.0)
                loss_status, loss_expl = classify_epoch_loss(curr_val_loss, val_history=val_history)
                loss_prev = loss_expl.get('prev_val')
                loss_delta = loss_expl.get('delta_pct')
                loss_best = loss_expl.get('best_val')
                best_epoch = loss_expl.get('best_idx') + 1 if loss_expl.get('best_idx') is not None else None

                loss_suggestions = []
                if loss_status == 'good':
                    loss_suggestions.append('Validation loss improving — consider longer training or reduced LR decay for more gains.')
                elif loss_status == 'stable':
                    loss_suggestions.append('Loss plateau — try reducing LR schedule, increase capacity, or train longer.')
                else:
                    loss_suggestions.append('Validation loss increasing — check learning rate, data pipeline, and overfitting signs (regularize, augment).')

                # Try to read recent indicator params summary (CSV produced by ParamsLogger)
                try:
                    params_summary = summarize_indicator_params(csv_path='indicator_params_history.csv')
                except Exception:
                    params_summary = {'convergence_score': None, 'mean_param_change_pct': None, 'top_changes': [], 'csv_path': 'indicator_params_history.csv'}

                # Compute a fallback progress color variable to avoid placing '#' inside f-string expressions
                progress_color = '#81C784' if progress > 0 else '#EF5350'

                # Prepare small HTML pieces that can be safely embedded into the main f-string
                point_loss_html = f"<span style='display: inline-block; width: 180px;'>point_loss:</span> <span style='color: #90CAF9;'>{logs.get('point_loss', 0):.6f}</span><br>" if 'point_loss' in logs else ''
                dir_loss_html = f"<span style='display: inline-block; width: 180px;'>dir_loss:</span> <span style='color: #90CAF9;'>{logs.get('dir_loss', 0):.6f}</span><br>" if 'dir_loss' in logs else ''
                nll_loss_html = f"<span style='display: inline-block; width: 180px;'>nll_loss:</span> <span style='color: #90CAF9;'>{logs.get('nll_loss', 0):.6f}</span><br>" if 'nll_loss' in logs else ''

                # Build a small HTML snippet summarizing params CSV (keep it out of the big f-string to avoid nested f-string parsing issues)
                ps = params_summary or {}
                conv_val = ps.get('convergence_score')
                conv_text = f"{conv_val:.3f}" if conv_val is not None else 'n/a'
                mean_val = ps.get('mean_param_change_pct')
                mean_text = f"{mean_val:.2f}%" if mean_val is not None else 'n/a'
                top = ps.get('top_changes', []) or []
                if top:
                    top_text = ', '.join([f"{p}:{c:.2f}% (val={v if v is not None else 'n/a'})" for p, c, v in top])
                else:
                    top_text = 'n/a'
                group_stats = ps.get('group_stats', {}) or {}
                if group_stats:
                    group_text = ', '.join([f"{g}:{v['avg']:.2f}/{v['min']:.2f}/{v['max']:.2f}" for g, v in group_stats.items()])
                else:
                    group_text = 'n/a'

                params_html = (
                    f"<div style='margin-bottom:6px;'><b>Convergence:</b> {conv_text}; <b>Mean Δ:</b> {mean_text}</div>"
                    f"<div style='font-size:12px; color:#bdbdbd; margin-bottom:6px;'><b>Top changes:</b> {top_text}</div>"
                    f"<div style='font-size:12px; color:#bdbdbd; margin-bottom:6px;'><b>Group stats (avg/min/max %):</b> {group_text}</div>"
                )

                # Helper to make legend row with highlighting when currently in that status
                def _legend_row(name, status, label_color, bounds_text=""):
                    """Return an HTML row containing the name, status and the human-readable bounds_text.
                    The row is visually highlighted when the metric is currently in the given status.
                    """
                    is_active = 'box-shadow: 0 0 8px ' + label_color + '; background-color: rgba(255,255,255,0.02);' if status == 'good' else ('box-shadow: 0 0 8px ' + label_color + '; background-color: rgba(255,255,255,0.01);' if status == 'stable' else '')
                    text_color = label_color if status == 'good' else ('#FFB74D' if status == 'stable' else '#EF5350')
                    return (f"<div style='padding:6px; margin-bottom:6px; border-radius:4px; {is_active}'>"
                            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                            f"<span style='display:inline-block; width:140px;'>{name}</span>"
                            f"<span style='font-weight:bold; color:{text_color};'>{status.upper()}</span>"
                            f"</div>"
                            f"<div style='font-size:11px; color:#9e9e9e; margin-top:3px;'>{bounds_text}</div>"
                            f"</div>")

                # Build 4-column layout: existing left + evaluation middle + legend + epoch-loss column
                html_content = f"""
                <div style="font-family: monospace; color: #e0e0e0; background-color: #0d0d0d; padding: 15px; border-radius: 5px; display:flex; gap:20px;">
                    <div style="flex:0 0 420px;">
                        <div style="text-align: center; font-size: 16px; font-weight: bold; border-bottom: 2px solid #444; padding-bottom: 10px; margin-bottom: 15px;">
                            📊 EPOCH METRICS DASHBOARD
                        </div>
                        
                        <div style="margin-bottom: 15px;">
                            <div style="color: #64B5F6; font-weight: bold; margin-bottom: 8px;">📉 LOSSES</div>
                            <div style="margin-left: 15px;">
                                <span style="display: inline-block; width: 180px;">loss:</span> <span style="color: #64B5F6;">{logs.get('loss', 0):.6f}</span><br>
                                <span style="display: inline-block; width: 180px;">val_loss:</span> <span style="color: #42A5F5;">{logs.get('val_loss', 0):.6f}</span><br>
                                {point_loss_html}
                                {dir_loss_html}
                                {nll_loss_html}
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
                        
                        <div style="margin-bottom: 10px;">
                            <div style="color: #CE93D8; font-weight: bold; margin-bottom: 8px;">📈 TRAINING HEALTH</div>
                            <div style="margin-left: 15px;">
                                <span style="display: inline-block; width: 180px;">Convergence:</span> <span style="color: {conv_color};">{convergence_rate:+.6f} ({conv_status})</span><br>
                                <span style="display: inline-block; width: 180px;">Stability:</span> <span style="color: {stab_color};">{stability:.4f} ({stab_status})</span><br>
                                <span style="display: inline-block; width: 180px;">Gen. Gap:</span> <span style="color: {gap_color};">{gen_gap:+.6f} ({gap_status})</span><br>
                                <span style="display: inline-block; width: 180px;">Coherence:</span> <span style="color: {coh_color};">{coherence:.4f} ({coh_status})</span><br>
                                <span style="display: inline-block; width: 180px;">Progress:</span> <span style="color: {progress_color};">{progress*100:+.2f}%</span><br>
                            </div>
                        </div>
                    </div>

                    <div style="flex:0 0 320px; background-color:#0b0b0b; border-radius:6px; padding:10px;">
                        <div style="font-weight:bold; color:#FFD54F; margin-bottom:8px;">💡 Epoch Metrics Evaluation</div>
                        <div style="color:#e0e0e0; font-size:13px; line-height:1.35;">
                            <div style="margin-bottom:8px;"><b>Direction health:</b> {dir_status.upper()} (val_dir_acc={dir_acc:.3f}, ECE={ece:.3f})</div>
                            <div style="margin-bottom:8px;"><b>Training health:</b> Convergence={conv_status_calc.upper()}, Stability={stab_status_calc.upper()}, Gap={gap_status_calc.upper()}</div>
                            <div style="margin-bottom:6px;"><b>Coherence:</b> {coh_status_calc.upper()} ({coherence:.3f})</div>
                            <div style="margin-top:8px; font-weight:bold; color:#BDBDBD;">Suggestions:</div>
                            <ul style="margin:6px 0 0 16px; color:#bdbdbd;">
                                {''.join([f'<li>{s}</li>' for s in suggestions])}
                            </ul>
                        </div>
                    </div>

                    <div style="flex:0 0 280px; background-color:#0b0b0b; border-radius:6px; padding:10px;">
                        <div style="font-weight:bold; color:#90CAF9; margin-bottom:8px;">🔖 Dashboard Legend (оценкой)</div>
                        <div style="color:#e0e0e0; font-size:13px; line-height:1.35;">
                            <div style="margin-bottom:8px; color:#9e9e9e;">Entries are highlighted when metric is currently in the corresponding assessment boundaries (границы оценки).</div>
                            {_legend_row('Val Dir Acc', dir_status, '#66BB6A', metric_limits_text('Val Dir Acc'))}
                            {_legend_row('ECE', ece_status, '#BA68C8', metric_limits_text('ECE'))}
                            {_legend_row('Brier', brier_status, '#CE93D8', metric_limits_text('Brier'))}
                            {_legend_row('Stability', stab_status_calc, '#81C784', metric_limits_text('Stability'))}
                            {_legend_row('Gen Gap', gap_status_calc, '#64B5F6', metric_limits_text('Gen Gap'))}
                            {_legend_row('Coherence', coh_status_calc, '#FFA726', metric_limits_text('Coherence'))}
                            {_legend_row('Epoch Loss', loss_status, '#FF7043', metric_limits_text('Epoch Loss'))}
                        </div>
                    </div>

                    <div style="flex:0 0 320px; background-color:#0b0b0b; border-radius:6px; padding:10px;">
                        <div style="font-weight:bold; color:#FF7043; margin-bottom:8px;">🔬 Epoch Loss & Indicator Params</div>
                        <div style="color:#e0e0e0; font-size:13px; line-height:1.35;">
                            <div style="margin-bottom:6px;"><b>Current Val Loss:</b> {curr_val_loss:.6f}</div>
                            <div style="margin-bottom:6px;"><b>Previous Val Loss:</b> {'n/a' if loss_prev is None else f'{loss_prev:.6f}'}</div>
                            <div style="margin-bottom:6px;"><b>Delta:</b> {'n/a' if loss_delta is None else f'{loss_delta*100:+.2f}%'} </div>
                            <div style="margin-bottom:6px;"><b>Best Val Loss:</b> {loss_best:.6f} (epoch: {best_epoch})</div>

                            <hr style="border:0; height:1px; background:#222; margin:8px 0;">
                            <div style="font-weight:bold; color:#FFD54F; margin-bottom:6px;">🔧 Indicator Params (recent)</div>
                            {params_html}
                            <div style="font-size:11px; color:#9e9e9e; margin-top:6px;">Detailed CSV: {params_summary.get('csv_path')}</div>

                            <div style="margin-top:8px; font-weight:bold; color:#BDBDBD;">Status & Suggestions:</div>
                            <div style="margin-top:6px; margin-bottom:6px;">{_legend_row('Epoch Loss', loss_status, '#FF7043', metric_limits_text('Epoch Loss'))}</div>
                            <ul style="margin:6px 0 0 16px; color:#bdbdbd;">
                                {''.join([f'<li>{s}</li>' for s in loss_suggestions])}
                            </ul>
                        </div>
                    </div>
                </div>
                """
                display(HTML(html_content))

                # --- Indicator movement plots (below the dashboard) ---
                try:
                    df_changes = load_indicator_params_df(csv_path='indicator_params_history.csv')
                    fig = plot_indicator_movements(df_changes)
                    if fig is not None:
                        display(fig)
                except Exception:
                    # Non-fatal, dashboard should not break if plotting fails
                    pass

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
            num_horizons = cfg.num_horizons
            horizon_keys = cfg.horizon_keys

            for batch in train_ds.take(n_calib_batches):
                x_batch, y_batch, last_batch, ext_batch = batch
                y_pred_batch = custom_model(x_batch, training=False)
                loss_dict = custom_model.custom_loss(
                    x_batch, y_batch, y_pred_batch, last_batch, ext_batch
                )

                # Extract losses dynamically for all horizons
                point_vals = [float(loss_dict[f'point_loss_{h}']) for h in horizon_keys]
                local_vals = [float(loss_dict[f'local_trend_{h}']) for h in horizon_keys]
                global_vals = [float(loss_dict[f'global_trend_{h}']) for h in horizon_keys]
                ext_vals = [float(loss_dict[f'extended_trend_{h}']) for h in horizon_keys]
                dir_vals = [float(loss_dict[f'dir_loss_{h}']) for h in horizon_keys]
                nll_vals = [float(loss_dict[f'nll_{h}']) for h in horizon_keys]

                point_losses.append(sum(point_vals) / num_horizons)
                local_losses.append(sum(local_vals) / num_horizons)
                global_losses.append(sum(global_vals) / num_horizons)
                ext_losses.append(sum(ext_vals) / num_horizons)
                dir_losses.append(sum(dir_vals) / num_horizons)
                var_losses.append(sum(nll_vals) / num_horizons)

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

            min_lambda = Config.MIN_LAMBDA 
            max_lambda = Config.MAX_LAMBDA
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
    # MCC-based early stopping for direction head (class-imbalance robust)
    # MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # Range: [-1, 1], where 1 = perfect, 0 = random, -1 = inverse
    # Unlike accuracy, MCC is balanced even with severe class imbalance
    # Use primary horizon (middle if available, otherwise first)
    primary_horizon_idx = cfg.num_horizons // 2 if cfg.num_horizons > 1 else 0
    primary_horizon_key = cfg.horizon_keys[primary_horizon_idx]
    es_dir = callbacks.EarlyStopping(
        monitor=f'val_dir_mcc_{primary_horizon_key}',
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

    # Extract price heads dynamically (scaled deltas from model outputs)
    # CRITICAL: These are SCALED predictions (trained in scaled delta space)
    # Model outputs: [price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, ...]
    num_horizons = cfg.num_horizons
    horizon_keys = cfg.horizon_keys

    y_pred_price_scaled_list = []
    for i in range(num_horizons):
        base_idx = i * 3  # Each horizon has 3 outputs: price, direction, variance
        price_pred = y_pred_all[base_idx][:, 0]
        y_pred_price_scaled_list.append(price_pred)

    y_pred_price_scaled = np.column_stack(y_pred_price_scaled_list)
    y_pred_price_scaled = y_pred_price_scaled[:len(y_test)]

    # Inverse-transform from scaled space back to raw delta space dynamically
    # This ensures predictions have the same statistical properties as the original deltas
    y_pred_raw_list = []
    for i in range(num_horizons):
        y_pred_raw = target_scaler.inverse_transform(y_pred_price_scaled[:, i].reshape(-1, 1)).ravel()
        y_pred_raw_list.append(y_pred_raw)

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
    horizon_steps = cfg.HORIZON_STEPS
    horizon_labels = [f"{h_key}(" + _format_tf_local(int(h_step * resample)) + ")"
                      for h_key, h_step in zip(horizon_keys, horizon_steps)]

    print("\n[Diagnostic: Prediction Statistics]")
    for h_idx, (h_name, y_pred_raw) in enumerate(zip(horizon_labels, y_pred_raw_list)):
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

    # Dynamically build predictions dict for all horizons
    # y_pred_all contains [price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, ...]
    num_horizons = cfg.num_horizons
    horizon_keys = cfg.horizon_keys

    delta_preds = {}
    dir_preds = {}
    var_preds = {}

    for i, h_key in enumerate(horizon_keys):
        base_idx = i * 3
        # Price (delta) predictions were already extracted above as y_pred_hX_raw
        # We need to access them from the locals() dict or rebuild
        # Since we already have y_pred_h0_raw, y_pred_h1_raw, y_pred_h2_raw from earlier loop,
        # let's rebuild here for consistency
        price_pred = np.asarray(y_pred_all[base_idx]).reshape(-1)[:len(y_test)]
        dir_pred = np.asarray(y_pred_all[base_idx + 1]).reshape(-1)[:len(y_test)]
        var_pred = np.asarray(y_pred_all[base_idx + 2]).reshape(-1)[:len(y_test)]

        # Unscale price predictions (same as earlier loop)
        y_pred_raw = price_pred * target_scaler_std + target_scaler_mean

        delta_preds[h_key] = y_pred_raw
        dir_preds[h_key] = dir_pred
        var_preds[h_key] = var_pred

    predictions = {
        "delta": delta_preds,
        "direction_prob": dir_preds,
        "variance": var_preds,
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
class LearnableIndicators(layers.Layer):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.epsilon = 1e-8
        self.alpha_vars_ma = []
        self.macd_alpha_vars = {}
        self.rsi_alpha_vars = []
        self.bb_alpha_vars = []
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
                      len(self.config.RSI_PERIODS) +
                      len(self.config.BB_PERIODS))
        meta_adjust = layers.Dense(num_logits, activation='tanh')(meta_inp)

        # Enhanced Learnable Indicators: Now takes [inp, meta_adjust], outputs sequences [B, LOOKBACK, num_ind]
        ind_seq = LearnableIndicators(self.config, name='learnable_indicators')([inp, meta_adjust])

        # Memory-Supplemented Layers: Capture temporal interconnections
        memory = layers.Bidirectional(layers.GRU(64, return_sequences=True))(ind_seq)
        memory = layers.Dropout(0.5)(memory)

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
            att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.5)(x, x)
            x = layers.Add()([x, att])
            x = layers.LayerNormalization()(x)
            ff = layers.Dense(32, activation='gelu')(x)
            ff = layers.Dropout(0.5)(ff)
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

        # === DYNAMIC OUTPUT TOWER GENERATION ===
        # Generate N independent towers (one per horizon in HORIZON_STEPS)
        # Each horizon has its own price, direction, and confidence (variance) head

        # Variance bias initialization: softplus(x) ≈ x for x > 0
        # Initialize bias so initial variance ≈ 1.3 (higher than unit variance for calibration learning)
        # softplus(0) ≈ 0.693, softplus(0.5) ≈ 0.97, softplus(1.0) ≈ 1.31
        var_bias_init = tf.keras.initializers.Constant(1.0)  # Initial variance ≈ 1.31

        # Direction bias initialization: sigmoid(0) = 0.5 (unbiased)
        # Keep at 0 for balanced initial predictions
        dir_bias_init = tf.keras.initializers.Zeros()

        outputs = []
        for i, horizon_step in enumerate(self.config.HORIZON_STEPS):
            h_key = f"h{i}"
            # Comment indicating what this horizon represents in minutes
            horizon_minutes = horizon_step * self.config.RESAMPLE_MINUTES

            # Create tower for this horizon
            tower = layers.Dense(
                16,
                activation='gelu',
                kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2),
                name=f'tower_{h_key}'
            )(shared_dense)

            # Price prediction head (scaled delta from last_close)
            price = layers.Dense(1, name=f'price_{h_key}')(tower)

            # Direction classification head (P(up) given features)
            direction = layers.Dense(
                1,
                activation='sigmoid',
                name=f'direction_{h_key}',
                bias_initializer=dir_bias_init
            )(tower)

            # Variance estimation head (predictive uncertainty)
            variance = layers.Dense(
                1,
                activation='softplus',
                name=f'variance_{h_key}',
                bias_initializer=var_bias_init
            )(tower)

            # Add this horizon's outputs (order matters: price, direction, variance)
            outputs.extend([price, direction, variance])

        # === FINAL MODEL: 3N outputs (N horizons × 3 heads each) ===
        # Example: N=3 → 9 outputs (price_h0, direction_h0, variance_h0, ...)
        return models.Model(inputs=inp, outputs=outputs)

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
        # Accumulator for per-batch gradient stats (appended each training batch)
        # Each entry is a dict: {'avg_pre': float, 'avg_post': float, 'frac_zero_pre': float, 'frac_zero_post': float}
        self._grad_batch_stats = []

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
    def custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends)

    def train_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        with tf.GradientTape() as tape:
            y_pred = self(x_window, training=True)
            loss_dict = self.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)

        # Extract total loss for gradient computation
        total_loss_val = loss_dict['total']

        grads = tape.gradient(total_loss_val, self.trainable_variables)

        # --- Compute gradient diagnostics for indicator-related variables (pre-scaling) ---
        def _is_indicator_var(name: str) -> bool:
            name = (name or '').lower()
            return ('alpha_ma' in name or 'macd_' in name or 'pair_' in name or
                    'rsi_alpha' in name or 'bb_alpha' in name or 'momentum_raw' in name)

        pre_norms = []
        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue
            if _is_indicator_var(v.name):
                try:
                    # Safe eager check
                    if tf.executing_eagerly():
                        pre_norms.append(float(tf.norm(g).numpy()))
                    else:
                        pre_norms.append(float(tf.norm(g)))
                except Exception:
                    pre_norms.append(0.0)

        # Build list of (grad, var) excluding None grads and apply indicator grad multiplier
        grads_and_vars = []
        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue
            name = v.name.lower()
            # Scale indicator-related variable grads
            if _is_indicator_var(name):
                g = g * self.config.INDICATOR_GRAD_MULT
            grads_and_vars.append((g, v))

        # --- Compute post-scaling norms for diagnostics ---
        post_norms = []
        for g, v in grads_and_vars:
            if _is_indicator_var(v.name):
                try:
                    if tf.executing_eagerly():
                        post_norms.append(float(tf.norm(g).numpy()))
                    else:
                        post_norms.append(float(tf.norm(g)))
                except Exception:
                    post_norms.append(0.0)

        # Aggregate per-batch stats and append (safe, minimal memory)
        try:
            avg_pre = float(np.mean(pre_norms)) if pre_norms else 0.0
        except Exception:
            avg_pre = 0.0
        try:
            avg_post = float(np.mean(post_norms)) if post_norms else 0.0
        except Exception:
            avg_post = 0.0
        frac_zero_pre = float(sum(1 for x in pre_norms if x <= 1e-12) / len(pre_norms)) if pre_norms else 1.0
        frac_zero_post = float(sum(1 for x in post_norms if x <= 1e-12) / len(post_norms)) if post_norms else 1.0
        try:
            self._grad_batch_stats.append({'avg_pre': avg_pre, 'avg_post': avg_post, 'frac_zero_pre': frac_zero_pre, 'frac_zero_post': frac_zero_post})
        except Exception:
            # be defensive: ignore if model removed/disabled
            pass

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

        # === COMPUTE DIRECTION METRICS FOR ALL N HORIZONS (DYNAMIC) ===
        num_horizons = self.config.num_horizons
        y_true = tf.cast(y_true, tf.float32)
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, N] (delta_raw)
        last_close_squeeze = tf.squeeze(last_close, axis=1)

        # Match training direction labeling (including deadband if enabled)
        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        # Compute returns, masks, and direction labels for all horizons dynamically
        rets = []
        masks = []
        true_dirs = []
        dir_preds_head = []
        gauss_p_ups = []

        # Extract predictions dynamically (3 outputs per horizon: price, direction, variance)
        for i in range(num_horizons):
            base_idx = i * 3
            price_h = y_pred[base_idx]
            dir_h = y_pred[base_idx + 1]
            var_h = y_pred[base_idx + 2]

            # Compute return for this horizon
            ret_h = (y_true_raw[:, i]) / (last_close_squeeze + self.eps)
            rets.append(ret_h)

            # Compute mask (deadband filtering)
            mask_h = tf.cast(tf.abs(ret_h) > deadband, tf.float32)
            masks.append(mask_h)

            # Compute true direction label
            true_dir_h = tf.cast(ret_h > deadband, tf.float32)
            true_dirs.append(true_dir_h)

            # Extract direction prediction (squeeze to 1D)
            dir_pred_h = tf.squeeze(dir_h, axis=1)
            dir_preds_head.append(dir_pred_h)

            # Gaussian-implied P(up) from (mu, var)
            var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
            var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e4), tf.float32)
            var_h_clipped = tf.clip_by_value(tf.squeeze(var_h, axis=1), var_floor, var_cap)
            mu_h = tf.squeeze(price_h, axis=1)
            deadband_delta_scaled = (deadband * last_close_squeeze) / (self.pred_scale + self.eps)
            z_up = (mu_h - deadband_delta_scaled) / (tf.sqrt(var_h_clipped) + self.eps)
            gauss_p_up_h = self._normal_cdf(z_up)
            gauss_p_ups.append(gauss_p_up_h)

        # Compute per-horizon metrics (masked if deadband is set)
        metrics_head = self._compute_direction_metrics(
            true_dirs=true_dirs,
            dir_preds=dir_preds_head,
            masks=masks,
            prefix="train_"
        )
        metrics_gauss = self._compute_direction_metrics(
            true_dirs=true_dirs,
            dir_preds=gauss_p_ups,
            masks=masks,
            prefix="train_gauss_"
        )

        # Trend metrics: margins (bps), agreement rates, magnitudes (bps) - computed dynamically
        # These metrics are diagnostic only - not used in loss computation

        # Extract loss components from dict for metrics
        # Build aggregates dynamically based on number of horizons
        horizon_keys = self.config.horizon_keys

        point_losses_list = [loss_dict.get(f'point_loss_{h}', 0.0) for h in horizon_keys]
        dir_losses_list = [loss_dict.get(f'dir_loss_{h}', 0.0) for h in horizon_keys]
        nll_list = [loss_dict.get(f'nll_{h}', 0.0) for h in horizon_keys]
        extended_trend_list = [loss_dict.get(f'extended_trend_{h}', 0.0) for h in horizon_keys]
        local_trend_list = [loss_dict.get(f'local_trend_{h}', 0.0) for h in horizon_keys]
        global_trend_list = [loss_dict.get(f'global_trend_{h}', 0.0) for h in horizon_keys]

        point_loss_total = tf.add_n(point_losses_list) if point_losses_list else tf.constant(0.0, dtype=tf.float32)
        dir_loss_total = tf.add_n(dir_losses_list) if dir_losses_list else tf.constant(0.0, dtype=tf.float32)
        nll_total = tf.add_n(nll_list) if nll_list else tf.constant(0.0, dtype=tf.float32)

        # Build per-horizon trend totals
        trend_totals_list = []
        for i in range(len(horizon_keys)):
            h = horizon_keys[i]
            trend_h = local_trend_list[i] + global_trend_list[i] + extended_trend_list[i]
            trend_totals_list.append(trend_h)
        trend_loss_total = tf.add_n(trend_totals_list) if trend_totals_list else tf.constant(0.0, dtype=tf.float32)

        # Build metrics dict with dynamic horizon support
        metrics_dict = {
            "loss": total_loss_val,
            "point_loss": point_loss_total,
            "trend_loss": trend_loss_total,
            "dir_loss": dir_loss_total,
            "nll_loss": nll_total,
            "reg_loss": loss_dict.get('reg_loss', 0.0),
            "inter_reg": loss_dict.get('inter_reg', 0.0),
            "vol_loss": loss_dict.get('vol_loss', 0.0),
        }

        # Add per-horizon components
        for i, h in enumerate(horizon_keys):
            metrics_dict[f'point_{h}'] = point_losses_list[i]
            metrics_dict[f'trend_{h}'] = trend_totals_list[i]
            metrics_dict[f'local_{h}'] = local_trend_list[i]
            metrics_dict[f'global_{h}'] = global_trend_list[i]
            metrics_dict[f'extended_{h}'] = extended_trend_list[i]
            metrics_dict[f'dir_loss_{h}'] = dir_losses_list[i]
            metrics_dict[f'nll_{h}'] = nll_list[i]

        # Add direction metrics
        metrics_dict.update(metrics_head)
        metrics_dict.update(metrics_gauss)

        return metrics_dict

    def _compute_direction_metrics(self, true_dirs, dir_preds, masks=None, prefix=""):
        """
        Compute per-horizon direction classification metrics for N horizons.

        Args:
            true_dirs: List of N tensors containing true direction labels (0/1)
            dir_preds: List of N tensors containing predicted direction probabilities
            masks: Optional list of N tensors for masking (e.g., deadband filtering)
            prefix: Metric name prefix (e.g., "train_", "val_")

        Returns:
            Dict with accuracy, F1, sensitivity, specificity, MCC, Brier, ECE for each horizon.
        """
        metrics = {}
        horizon_keys = self.config.horizon_keys
        num_horizons = self.config.num_horizons

        # Validate inputs
        if len(true_dirs) != num_horizons or len(dir_preds) != num_horizons:
            raise ValueError(
                f"Expected {num_horizons} horizons, got true_dirs={len(true_dirs)}, "
                f"dir_preds={len(dir_preds)}"
            )

        # Default masks to all ones if not provided
        if masks is None:
            masks = [tf.ones_like(td) for td in true_dirs]
        elif len(masks) != num_horizons:
            raise ValueError(f"Expected {num_horizons} masks, got {len(masks)}")

        # Cast masks to float32
        masks = [tf.cast(m, tf.float32) for m in masks]

        for horizon_idx, (h_name, true_dir, dir_pred, m) in enumerate(
            zip(horizon_keys, true_dirs, dir_preds, masks)
        ):
            # Binary predictions
            pred_dir_binary = tf.cast(dir_pred > 0.5, tf.float32)
            mask_sum = tf.reduce_sum(m)
            no_samples = mask_sum < 1e-8
            nan = tf.constant(np.nan, dtype=tf.float32)

            # Confusion matrix elements
            TP = tf.reduce_sum(pred_dir_binary * true_dir * m)
            TN = tf.reduce_sum((1.0 - pred_dir_binary) * (1.0 - true_dir) * m)
            FP = tf.reduce_sum(pred_dir_binary * (1.0 - true_dir) * m)
            FN = tf.reduce_sum((1.0 - pred_dir_binary) * true_dir * m)

            # Accuracy
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            accuracy = tf.where(no_samples, nan, accuracy)
            metrics[f"{prefix}dir_acc_{h_name}"] = accuracy

            # Sensitivity (True Positive Rate / Recall for UP class)
            sensitivity = TP / (TP + FN + 1e-8)
            sensitivity = tf.where(no_samples, nan, sensitivity)
            metrics[f"{prefix}dir_sensitivity_{h_name}"] = sensitivity

            # Specificity (True Negative Rate)
            specificity = TN / (TN + FP + 1e-8)
            specificity = tf.where(no_samples, nan, specificity)
            metrics[f"{prefix}dir_specificity_{h_name}"] = specificity

            # Balanced Accuracy: (Sensitivity + Specificity) / 2
            balanced_acc = (sensitivity + specificity) / 2.0
            balanced_acc = tf.where(no_samples, nan, balanced_acc)
            metrics[f"{prefix}dir_bal_acc_{h_name}"] = balanced_acc

            # F1 Score (harmonic mean of precision and recall)
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-8)
            f1 = tf.where(no_samples, nan, f1)
            metrics[f"{prefix}dir_f1_{h_name}"] = f1

            # Matthews Correlation Coefficient (balanced metric for binary classification)
            mcc_numerator = (TP * TN) - (FP * FN)
            marginal_product = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
            mcc_denominator = tf.sqrt(marginal_product + 1e-8)
            mcc_raw = tf.where(
                marginal_product > 1e-8,
                mcc_numerator / mcc_denominator,
                tf.constant(0.0, dtype=tf.float32)
            )
            mcc = tf.where(no_samples, nan, mcc_raw)
            metrics[f"{prefix}dir_mcc_{h_name}"] = mcc

            # ========== CALIBRATION METRICS ==========
            # Brier Score: Mean squared error between predicted probability and actual outcome
            brier_per_sample = tf.square(dir_pred - true_dir)
            brier_score = tf.reduce_sum(brier_per_sample * m) / (mask_sum + 1e-8)
            brier_score = tf.where(no_samples, nan, brier_score)
            metrics[f"{prefix}dir_brier_{h_name}"] = brier_score

            # Expected Calibration Error (ECE): partition [0,1] into inclusive bins
            n_bins = 10
            ece_sum = tf.constant(0.0, dtype=tf.float32)
            total_masked = mask_sum + 1e-8
            dir_pred_clipped = tf.clip_by_value(dir_pred, 0.0, 1.0)
            for bin_idx in range(n_bins):
                bin_lower = tf.cast(bin_idx, tf.float32) / n_bins
                bin_upper = tf.cast(bin_idx + 1, tf.float32) / n_bins
                if bin_idx == n_bins - 1:
                    in_bin = tf.cast((dir_pred_clipped >= bin_lower) & (dir_pred_clipped <= bin_upper), tf.float32) * m
                else:
                    in_bin = tf.cast((dir_pred_clipped >= bin_lower) & (dir_pred_clipped < bin_upper), tf.float32) * m
                bin_count = tf.reduce_sum(in_bin)
                bin_correct = tf.reduce_sum(tf.cast(pred_dir_binary == true_dir, tf.float32) * in_bin)
                bin_acc = bin_correct / (bin_count + 1e-8)
                bin_conf = tf.reduce_sum(dir_pred_clipped * in_bin) / (bin_count + 1e-8)
                ece_sum = ece_sum + (bin_count / total_masked) * tf.abs(bin_acc - bin_conf)
            ece_sum = tf.where(no_samples, nan, ece_sum)
            metrics[f"{prefix}dir_ece_{h_name}"] = ece_sum

            # CALIBRATION METRICS: Per-class prediction rates to detect bias
            total_samples = TP + TN + FP + FN + 1e-8
            pred_up_rate = (TP + FP) / total_samples
            true_up_rate = (TP + FN) / total_samples
            pred_up_rate = tf.where(no_samples, nan, pred_up_rate)
            true_up_rate = tf.where(no_samples, nan, true_up_rate)
            metrics[f"{prefix}pred_up_rate_{h_name}"] = pred_up_rate
            metrics[f"{prefix}true_up_rate_{h_name}"] = true_up_rate

            # Mean predicted probability (should be ~0.5 for calibrated model)
            mean_prob = tf.reduce_sum(dir_pred_clipped * m) / (mask_sum + 1e-8)
            mean_prob = tf.where(no_samples, nan, mean_prob)
            metrics[f"{prefix}mean_dir_prob_{h_name}"] = mean_prob

        return metrics

    def test_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        y_pred = self(x_window, training=False)
        loss_dict = self.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)

        # Extract total loss
        total_loss_val = loss_dict['total']

        # === COMPUTE DIRECTION METRICS FOR ALL N HORIZONS (DYNAMIC) ===
        num_horizons = self.config.num_horizons
        horizon_keys = self.config.horizon_keys

        y_true = tf.cast(y_true, tf.float32)
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, N]
        last_close_squeeze = tf.squeeze(last_close, axis=1)

        deadband_bps = tf.cast(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0), tf.float32)
        deadband = deadband_bps / tf.constant(10000.0, dtype=tf.float32)

        # Compute returns, masks, and direction labels for all horizons dynamically
        true_dirs = []
        dir_preds_head = []
        gauss_p_ups = []
        masks = []

        # Extract predictions dynamically (3 outputs per horizon)
        for i in range(num_horizons):
            base_idx = i * 3
            price_h = y_pred[base_idx]
            dir_h = y_pred[base_idx + 1]
            var_h = y_pred[base_idx + 2]

            # Compute return for this horizon
            ret_h = (y_true_raw[:, i]) / (last_close_squeeze + self.eps)

            # Compute mask (deadband filtering)
            mask_h = tf.cast(tf.abs(ret_h) > deadband, tf.float32)
            masks.append(mask_h)

            # Compute true direction label
            true_dir_h = tf.cast(ret_h > deadband, tf.float32)
            true_dirs.append(true_dir_h)

            # Extract direction prediction (squeeze to 1D)
            dir_pred_h = tf.squeeze(dir_h, axis=1)
            dir_preds_head.append(dir_pred_h)

            # Gaussian-implied P(up) from (mu, var)
            var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
            var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e4), tf.float32)
            var_h_clipped = tf.clip_by_value(tf.squeeze(var_h, axis=1), var_floor, var_cap)
            mu_h = tf.squeeze(price_h, axis=1)
            deadband_delta_scaled = (deadband * last_close_squeeze) / (self.pred_scale + self.eps)
            z_up = (mu_h - deadband_delta_scaled) / (tf.sqrt(var_h_clipped) + self.eps)
            gauss_p_up_h = self._normal_cdf(z_up)
            gauss_p_ups.append(gauss_p_up_h)

        # IMPORTANT: do NOT prefix with "val_" here. Keras automatically prefixes
        # validation metrics with "val_"; adding it ourselves creates "val_val_*" keys.
        metrics_head = self._compute_direction_metrics(
            true_dirs=true_dirs,
            dir_preds=dir_preds_head,
            masks=masks,
            prefix=""
        )
        metrics_gauss = self._compute_direction_metrics(
            true_dirs=true_dirs,
            dir_preds=gauss_p_ups,
            masks=masks,
            prefix="gauss_"
        )

        # Extract loss components from dict for metrics
        point_losses_list = [loss_dict.get(f'point_loss_{h}', 0.0) for h in horizon_keys]
        dir_losses_list = [loss_dict.get(f'dir_loss_{h}', 0.0) for h in horizon_keys]
        nll_list = [loss_dict.get(f'nll_{h}', 0.0) for h in horizon_keys]
        extended_trend_list = [loss_dict.get(f'extended_trend_{h}', 0.0) for h in horizon_keys]
        local_trend_list = [loss_dict.get(f'local_trend_{h}', 0.0) for h in horizon_keys]
        global_trend_list = [loss_dict.get(f'global_trend_{h}', 0.0) for h in horizon_keys]

        point_loss_total = tf.add_n(point_losses_list) if point_losses_list else tf.constant(0.0, dtype=tf.float32)
        dir_loss_total = tf.add_n(dir_losses_list) if dir_losses_list else tf.constant(0.0, dtype=tf.float32)
        nll_total = tf.add_n(nll_list) if nll_list else tf.constant(0.0, dtype=tf.float32)

        # Build per-horizon trend totals
        trend_totals_list = []
        for i in range(len(horizon_keys)):
            h = horizon_keys[i]
            trend_h = local_trend_list[i] + global_trend_list[i] + extended_trend_list[i]
            trend_totals_list.append(trend_h)
        trend_loss_total = tf.add_n(trend_totals_list) if trend_totals_list else tf.constant(0.0, dtype=tf.float32)

        # Build metrics dict with dynamic horizon support
        metrics_dict = {
            "loss": total_loss_val,
            "point_loss": point_loss_total,
            "trend_loss": trend_loss_total,
            "dir_loss": dir_loss_total,
            "nll_loss": nll_total,
            "reg_loss": loss_dict.get('reg_loss', 0.0),
            "inter_reg": loss_dict.get('inter_reg', 0.0),
            "vol_loss": loss_dict.get('vol_loss', 0.0),
        }

        # Add per-horizon components
        for i, h in enumerate(horizon_keys):
            metrics_dict[f'point_{h}'] = point_losses_list[i]
            metrics_dict[f'trend_{h}'] = trend_totals_list[i]
            metrics_dict[f'local_{h}'] = local_trend_list[i]
            metrics_dict[f'global_{h}'] = global_trend_list[i]
            metrics_dict[f'extended_{h}'] = extended_trend_list[i]
            metrics_dict[f'dir_loss_{h}'] = dir_losses_list[i]
            metrics_dict[f'nll_{h}'] = nll_list[i]

        # Add direction metrics
        metrics_dict.update(metrics_head)
        metrics_dict.update(metrics_gauss)

        return metrics_dict

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

    # Detect horizons from available keys (dynamic)
    available_horizons = set()
    for key in out.keys():
        for h_key in ['local_', 'global_', 'extended_']:
            if h_key in key:
                parts = key.replace(h_key, '').replace('val_', '').split('_')
                for part in parts:
                    if part.startswith('h') and len(part) > 1 and part[1:].isdigit():
                        available_horizons.add(part)

    if not available_horizons:
        # Fallback to default h0, h1, h2
        available_horizons = {'h0', 'h1', 'h2'}

    horizons_for_agg = sorted(available_horizons, key=lambda h: int(h[1:]))

    set_if_missing("local_trend_loss", _sum_present(out, [f"local_{h}" for h in horizons_for_agg]))
    set_if_missing("val_local_trend_loss", _sum_present(out, [f"val_local_{h}" for h in horizons_for_agg]))

    set_if_missing("global_trend_loss", _sum_present(out, [f"global_{h}" for h in horizons_for_agg]))
    set_if_missing("val_global_trend_loss", _sum_present(out, [f"val_global_{h}" for h in horizons_for_agg]))

    set_if_missing("extended_trend_loss", _sum_present(out, [f"extended_{h}" for h in horizons_for_agg]))
    set_if_missing("val_extended_trend_loss", _sum_present(out, [f"val_extended_{h}" for h in horizons_for_agg]))

    # --- Direction metric aliases (primary horizon, head vs gauss preference) ---
    train_pref = "train_gauss_" if prefer_gauss else "train_"
    train_fallback = "train_" if prefer_gauss else "train_gauss_"

    val_pref = "val_gauss_" if prefer_gauss else "val_"
    val_fallback = "val_" if prefer_gauss else "val_gauss_"

    # Detect available horizons from logs (dynamic discovery)
    # Look for keys like "dir_acc_h0", "dir_acc_h1", etc.
    detected_horizons = set()
    for key in out.keys():
        if 'dir_acc_' in key:
            parts = key.split('_')
            for part in parts:
                if part.startswith('h') and part[1:].isdigit():
                    detected_horizons.add(part)

    # Fallback to default if no horizons detected
    if not detected_horizons:
        horizons = ("h0", "h1", "h2")
    else:
        # Sort by horizon index (h0, h1, h2, ...)
        horizons = tuple(sorted(detected_horizons, key=lambda h: int(h[1:])))

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

        # Aggregate gradient diagnostics if model accumulated them during training
        try:
            grad_stats = getattr(self.model, '_grad_batch_stats', None)
            if grad_stats:
                avg_pre = float(np.mean([s.get('avg_pre', 0.0) for s in grad_stats]))
                avg_post = float(np.mean([s.get('avg_post', 0.0) for s in grad_stats]))
                frac_zero_pre = float(np.mean([s.get('frac_zero_pre', 1.0) for s in grad_stats]))
                frac_zero_post = float(np.mean([s.get('frac_zero_post', 1.0) for s in grad_stats]))
                params['grad_indicator_avg_pre'] = avg_pre
                params['grad_indicator_avg_post'] = avg_post
                params['grad_indicator_frac_zero_pre'] = frac_zero_pre
                params['grad_indicator_frac_zero_post'] = frac_zero_post
        except Exception:
            pass

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
    cfg = Config()
    result = train_and_evaluate(
        config=cfg,
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

    # For legacy callers, keep `y_pred` as the primary horizon delta series.
    # The full set of head outputs is exposed via `predictions_dict`.
    # Use primary horizon (middle if available, otherwise first)
    primary_horizon_idx = cfg.num_horizons // 2 if cfg.num_horizons > 1 else 0
    primary_horizon_key = cfg.horizon_keys[primary_horizon_idx]
    y_pred = np.asarray(result.predictions["delta"][primary_horizon_key], dtype=float).reshape(-1)
    predictions_dict = result.predictions

    # Provide a horizon-wide summary (no "primary horizon" framing).
    try:
        m = result.metrics
        if isinstance(m, dict) and 'delta' in m:
            print("\n[Summary: Per-Horizon Delta Metrics]")
            # Use config horizon keys as primary source, with fallback
            horizon_keys_list = m.get('meta', {}).get('horizon_keys', list(cfg.horizon_keys))
            horizon_labels_list = m.get('meta', {}).get('horizon_labels')
            if horizon_labels_list is None:
                # Generate fallback labels from config
                resample = 1  # default
                horizon_labels_list = [f"{h_key}({h_step}min)" for h_key, h_step in zip(cfg.horizon_keys, cfg.HORIZON_STEPS)]

            for h_key, label in zip(horizon_keys_list, horizon_labels_list):
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
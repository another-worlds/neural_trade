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

# Import mathematical helper functions
import math_helpers as mh

# -----------------------------
class Config:

    HOUR = 60
    DAY = HOUR * 24

    # Data Configuration tuned for minute-level data
    CSV_PATH = csv
    LOOKBACK = HOUR   # Reduced to 1 hour of minute data
    WINDOW_STEP = 1  # Generate a training sample every minute for true minute-level modeling
    RESAMPLE_MINUTES = None  # Optionally aggregate to coarser bars (e.g., set to 5 for 5-minute bars)
    BATCH_SIZE = 144 * 4
    EPOCHS = 10
    LR = 1e-3  # Fixed from critically low 1e-10; reasonable for Adam optimizer
    PATIENCE = 3  # EarlyStopping on val_loss (M8); must be < EPOCHS to be able to fire
    MAX_SEQUENCE_COUNT = None  # M9: no cap; the harness sets WINDOW_STEP (e.g. 20) to bound training size
    SEED = 42  # M8: replaces the hard-coded set_random_seed(42)

    # M8: validation = the last VAL_FRACTION of the training range, with a purge of VAL_PURGE_BARS bars
    # between the last training label and the first validation origin. The test fold lives only in the harness.
    VAL_FRACTION = 0.15
    VAL_PURGE_BARS = 15

    # M4: unit-free inputs. Channel 0 = 1e4*ln(close/last_close) (bps); EXTRA_CHANNELS adds 5 causal channels:
    # log range (bps), ln(v / EWMA_1440 v), taker imbalance, UTC hour sin, hour cos. Input shape (LOOKBACK, 6).
    EXTRA_CHANNELS = True
    VOLUME_EWMA_SPAN = 1440
    # Leading bars of a frame used only to warm up causal features (never a sequence origin).
    FEATURE_WARMUP_BARS = 0



    # Use integer periods to avoid float indexing issues
    # Extended trend features are computed as percent-change over these lags (in minutes)
    EXTENDED_TREND_PERIODS = [1, 5, 15]  # 1m, 5m, 15m

    # Supervision horizons (in minutes ahead from last_close). These define the 3 output towers.
    # h0=1m, h1=5m, h2=15m.
    HORIZON_STEPS = [1, 5, 15]

# Loss Function Weights
    DAMPING = 1
    LAMBDA_POINT = 1  
    LAMBDA_LOCAL_TREND  = 0.1
    LAMBDA_GLOBAL_TREND =  0.1
    LAMBDA_EXTENDED_TREND = 0.1
    LAMBDA_QUANTILE = 1
    REG_MOMENTUM_L2 = 1e-4
    MOMENTUM_CLIP_MIN = 1.0
    MOMENTUM_CLIP_MAX = LOOKBACK
    USE_HUBER = True
    LAMBDA_DIR = 1  # New for direction loss
    LAMBDA_INTER = 0  # M7: L2 is already in reg_loss; a second copy would count it twice
    LAMBDA_VOL = 0  # M7: the SD-matching term rewards noisy forecasts
    LAMBDA_SHORT = 1  # For multi-horizon short-term
    LAMBDA_LONG = 1  # For multi-horizon long-term
    LAMBDA_VAR = 1  # For variance NLL
    DF_VAR = 3.0  # Degrees of freedom for Student's t in variance NLL
    TAU_PRICE = 0.5  # M6: median (was 0.4, a built-in DOWN bias)

# paths
    # MODEL_PATH v2: Major architectural refactor for multi-horizon direction classification
    # - 3 independent output towers (h0_1min, h1_5min, h2_15min) instead of shared heads
    # - 9 outputs (3 price + 3 direction + 3 variance) vs 3 outputs (price, direction, variance)
    # - Per-horizon direction metrics: accuracy, F1, sensitivity, specificity, MCC
    # v3: true multi-horizon supervision (separate targets per tower)
    # v4: model_py_fixes M1-M11 (unit-free 6-channel input, unbiased losses, purged validation, bundles).
    # M10: train_and_evaluate() rewrites both paths into a bundle directory keyed by FOLD_ID + config hash:
    #   <ARTIFACT_DIR>/bundle_<FOLD_ID>_<hash>/{model.weights.h5, target_scaler.joblib, input_scaler.joblib, config.json}
    MODEL_VERSION = "v4"
    MODEL_PATH = "nn_learnable_indicators_v4.weights.h5"
    SCALER_PATH = "scaler_v4.joblib"
    FOLD_ID = "default"
    ARTIFACT_DIR = "."
    LOG_DIR = "."  # training_log.csv and indicator_params_history.csv

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
    INDICATOR_GRAD_MULT = 1.0  # M1: was 20.0; the forward-pass copy of it killed the indicators (D1)
    GRAD_CLIP_NORM = 5.0

# Focal loss hyperparameters for direction classification
    FOCAL_ALPHA = 0.5  # M6: symmetric (0.7 put the loss-optimal P(up) at 0.38 on balanced labels)
    FOCAL_GAMMA = 1.0  # Focus parameter: higher = focus more on hard examples

    # Trade-aware direction labeling deadband.
    # If > 0, direction loss/metrics treat returns within +/- deadband as neutral.
    # Units: basis points (bps). Example: 10 bps = 0.10%.
    DIR_DEADBAND_BPS = 7.0  # M6: the cost level

    # Stabilize NLL and prevent variance head from dominating early.
    # Variance is in SCALED units^2.
    VAR_FLOOR = 1e-4
    VAR_CAP = 1e4

    # Align direction head with distribution-implied P(up) from (mu, var).
    # Setting this > 0 helps avoid degenerate constant direction probabilities.
    LAMBDA_DIR_ALIGN = 0.1

# -----------------------------
@dataclass
class LossComponents:
    """Structured container for all loss components, replacing the 29-tuple return."""
    total: tf.Tensor

    # Point losses (3)
    point_h0: tf.Tensor
    point_h1: tf.Tensor
    point_h2: tf.Tensor

    # Trend losses (9: 3 types × 3 horizons)
    local_trend_h0: tf.Tensor
    global_trend_h0: tf.Tensor
    extended_trend_h0: tf.Tensor
    local_trend_h1: tf.Tensor
    global_trend_h1: tf.Tensor
    extended_trend_h1: tf.Tensor
    local_trend_h2: tf.Tensor
    global_trend_h2: tf.Tensor
    extended_trend_h2: tf.Tensor

    # Direction losses (3)
    dir_h0: tf.Tensor
    dir_h1: tf.Tensor
    dir_h2: tf.Tensor

    # Variance NLL losses (3)
    nll_h0: tf.Tensor
    nll_h1: tf.Tensor
    nll_h2: tf.Tensor

    # Regularization and penalties (3)
    reg_loss: tf.Tensor
    inter_reg: tf.Tensor
    vol_loss: tf.Tensor

    def as_tuple(self):
        """Convert to tuple format for backward compatibility."""
        return (
            self.total,
            self.point_h0, self.point_h1, self.point_h2,
            self.local_trend_h0, self.global_trend_h0, self.extended_trend_h0,
            self.local_trend_h1, self.global_trend_h1, self.extended_trend_h1,
            self.local_trend_h2, self.global_trend_h2, self.extended_trend_h2,
            self.dir_h0, self.dir_h1, self.dir_h2,
            self.nll_h0, self.nll_h1, self.nll_h2,
            self.reg_loss, self.inter_reg, self.vol_loss
        )

# -----------------------------
def build_input_windows(close_array, extra, origins, lookback):
    """M4 model inputs for decision bars `origins` (row indices): float32 [N, lookback, 1 + n_extra].

    Channel 0 = 1e4*ln(close[o-L+1..o] / close[o]); then the rows o-L+1..o of `extra` ([n_rows, n_extra]).
    Only bars <= o are read, so the input of an origin cannot change when later bars change.
    The same function builds training sequences and predict_at() inputs.
    """
    origins = np.asarray(origins, dtype=np.int64).reshape(-1)
    if origins.size and origins.min() < lookback - 1:
        raise ValueError(f"origin {origins.min()} has fewer than lookback={lookback} bars of history")
    lc = np.log(np.asarray(close_array, dtype=np.float64).reshape(-1))
    idx = origins[:, None] + np.arange(-lookback + 1, 1, dtype=np.int64)[None, :]
    chans = [(1e4 * (lc[idx] - lc[origins][:, None])).astype('float32')[..., None]]
    if extra is not None:
        chans.append(np.asarray(extra, dtype='float32')[idx])
    return np.concatenate(chans, axis=-1)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.target_scaler = None
        self.input_scaler = None

    def clean_numeric(self, series):
        return series.astype(str).str.replace(r'[\$,]', '', regex=True).replace('', np.nan).astype(float)

    def load_and_prepare_data(self, read_csv_kwargs: Optional[dict] = None, frame: Optional[pd.DataFrame] = None):
        """Load and prep minute-level Bitcoin data with optional resampling.

        `read_csv_kwargs` is passed through to `pd.read_csv`.
        This makes ingestion robust in notebooks where the CSV may require
        non-default parsing settings.

        M5: `frame` is an in-memory DataFrame (e.g. built from research/pnl/data.py bars) used instead of the CSV.
        The time column may be called `timestamp` or `datetime` (datetime-like, or numeric unix seconds).
        `taker_buy_base` and `trades` are kept when present.
        """
        if frame is not None:
            df = frame.copy()
        else:
            read_csv_kwargs = dict(read_csv_kwargs or {})
            df = pd.read_csv(self.config.CSV_PATH, **read_csv_kwargs)

        # Parse timestamp column (minute-level data format); the repo CSV calls it `datetime` (D6)
        if 'timestamp' not in df.columns and 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        if 'timestamp' not in df.columns:
            raise ValueError("Expected a 'timestamp' or 'datetime' column")
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        else:
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

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'taker_buy_base', 'trades']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if self.config.RESAMPLE_MINUTES:
            agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            for col in ('taker_buy_base', 'trades'):
                if col in df.columns:
                    agg[col] = 'sum'
            df = (
                df.set_index('timestamp')
                  .resample(f"{self.config.RESAMPLE_MINUTES}min")
                  .agg(agg)
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
        """Compute extended trend features using safe integer indices.

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
                trend = (current_price - past_price) / (past_price + 1e-8)
                features.append(float(trend))
            else:
                features.append(0.0)

        return np.array(features, dtype='float32')

    def compute_extra_channels(self, df):
        """M4: 5 causal per-bar channels, [N, 5] float32. Row t uses bars <= t only.

        0: log range 1e4*ln(High/Low) (bps); 1: ln(Volume / EWMA_span(Volume)), adjusted EWMA through bar t,
        clipped to +-10 (zero-volume bars); 2: taker imbalance (2*taker_buy_base - Volume)/max(Volume, 1e-9);
        3, 4: sin/cos of the UTC time of day. A missing source column gives a zero channel.
        """
        n = len(df)
        out = np.zeros((n, 5), dtype=np.float64)
        if 'High' in df.columns and 'Low' in df.columns:
            hi = df['High'].to_numpy(np.float64)
            lo = df['Low'].to_numpy(np.float64)
            out[:, 0] = 1e4 * np.log(hi / lo)
        if 'Volume' in df.columns:
            v = df['Volume'].to_numpy(np.float64)
            span = int(getattr(self.config, 'VOLUME_EWMA_SPAN', 1440))
            ev = pd.Series(v).ewm(span=span, adjust=True).mean().to_numpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                out[:, 1] = np.clip(np.log(np.maximum(v, 1e-12) / np.maximum(ev, 1e-12)), -10.0, 10.0)
            if 'taker_buy_base' in df.columns:
                tbb = df['taker_buy_base'].to_numpy(np.float64)
                out[:, 2] = (2.0 * tbb - v) / np.maximum(v, 1e-9)
        ts = pd.to_datetime(df['timestamp'])
        if getattr(ts.dt, 'tz', None) is not None:
            ts = ts.dt.tz_convert('UTC')
        day_frac = (ts.dt.hour.to_numpy() * 3600 + ts.dt.minute.to_numpy() * 60 + ts.dt.second.to_numpy()) / 86400.0
        out[:, 3] = np.sin(2.0 * np.pi * day_frac)
        out[:, 4] = np.cos(2.0 * np.pi * day_frac)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype('float32')

    def make_sequences_with_extended_trends(self, close_array, lookback, extra=None):
        """Build training sequences. M4: unit-free inputs and targets.

        X[k, :, 0] = 1e4*ln(close[o-L+1..o] / close[o]) (bps relative to the last close o = i-1), plus
        extra[o-L+1..o] channels when given -> X shape [N, L, 1 + n_extra].
        y[k, j] = 1e4*ln(close[o+h_j] / close[o]) (bps). last_close is the constant 1e4, so every internal
        ret = delta / last_close conversion (labels, deadband, alignment) returns the log return.
        Origins o are kept in self.seq_origin_idx.
        """
        X, y, last_close, extended_trends = [], [], [], []
        # Ensure start index is an integer even if periods are provided as floats
        max_extended_period = int(max(self.config.EXTENDED_TREND_PERIODS))
        start_idx = int(max(lookback, max_extended_period, int(getattr(self.config, 'FEATURE_WARMUP_BARS', 0) or 0)))
        step = int(max(1, getattr(self.config, 'WINDOW_STEP', 1)))

        horizon_steps = [int(h) for h in getattr(self.config, 'HORIZON_STEPS', [1, 5, 15])]
        if not horizon_steps:
            raise ValueError("Config.HORIZON_STEPS must be a non-empty list of positive integers")
        if any(h <= 0 for h in horizon_steps):
            raise ValueError(f"Invalid horizon steps: {horizon_steps}")
        max_h = int(max(horizon_steps))

        # Ensure targets are within bounds for all horizons
        end_idx = int(len(close_array) - (max_h - 1))

        origins = np.arange(start_idx, end_idx, step, dtype=np.int64) - 1
        self.seq_origin_idx = origins
        lc64 = np.log(np.asarray(close_array, dtype=np.float64).reshape(-1))
        for o in origins:
            # Targets: log returns in bps relative to the last close (M4). Stationary and unit-free.
            target = np.array([1e4 * (lc64[o + h] - lc64[o]) for h in horizon_steps], dtype='float32')
            ext_features = self.compute_extended_trend_features(close_array, int(o), self.config.EXTENDED_TREND_PERIODS)
            y.append(target)
            extended_trends.append(ext_features)
        X = build_input_windows(close_array, extra, origins, lookback)
        last_close = np.full(len(origins), 1e4, dtype='float32')

        return (
            X,
            np.array(y, dtype='float32').reshape(-1, len(horizon_steps)),
            last_close,
            np.array(extended_trends, dtype='float32').reshape(-1, len(self.config.EXTENDED_TREND_PERIODS))
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
        extra = self.compute_extra_channels(df) if getattr(self.config, 'EXTRA_CHANNELS', False) else None
        X_seq, y_seq, last_close_seq, extended_trends = self.make_sequences_with_extended_trends(
            close_values, self.config.LOOKBACK, extra=extra
        )
        origins = self.seq_origin_idx
        print(f"Sequences with extended trends: {X_seq.shape}, {y_seq.shape}, Extended: {extended_trends.shape}")

        max_sequences = getattr(self.config, 'MAX_SEQUENCE_COUNT', None)
        if max_sequences and X_seq.shape[0] > max_sequences:
            original_count = X_seq.shape[0]
            take_from = original_count - max_sequences
            X_seq = X_seq[take_from:]
            y_seq = y_seq[take_from:]
            last_close_seq = last_close_seq[take_from:]
            extended_trends = extended_trends[take_from:]
            origins = origins[take_from:]
            print(f"[OK] Limited sequence set from {original_count} to {max_sequences} (most recent window)")

        print("[INFO] Dataset Statistics:")
        print(f"   Total sequences: {X_seq.shape[0]}")

        # M8: validation = the last VAL_FRACTION of the range. Training rows must have their last label bar
        # (origin + max horizon) at least VAL_PURGE_BARS bars before the first validation origin.
        # The test fold is never seen here; it exists only in the harness.
        n_seq = X_seq.shape[0]
        n_val = int(math.ceil(float(self.config.VAL_FRACTION) * n_seq))
        if n_val < 1 or n_val >= n_seq:
            raise ValueError(f"Cannot split {n_seq} sequences with VAL_FRACTION={self.config.VAL_FRACTION}")
        test_indices = np.arange(n_seq - n_val, n_seq)
        first_val_origin = int(origins[test_indices[0]])
        max_h = int(max(self.config.HORIZON_STEPS))
        purge = int(getattr(self.config, 'VAL_PURGE_BARS', max_h))
        train_indices = np.flatnonzero(origins[:n_seq - n_val] + max(purge, max_h) < first_val_origin)
        if len(train_indices) == 0:
            raise ValueError("No training sequences left after the validation purge")
        self.train_indices, self.val_indices, self.seq_origins = train_indices, test_indices, origins

        # Both direction classes (beyond the deadband) must be present in validation at every horizon.
        deadband_bps = float(getattr(self.config, 'DIR_DEADBAND_BPS', 0.0))
        for j, h in enumerate(self.config.HORIZON_STEPS):
            r = y_seq[test_indices, j]
            n_up, n_down = int((r > deadband_bps).sum()), int((r < -deadband_bps).sum())
            if n_up == 0 or n_down == 0:
                raise ValueError(f"Validation block lacks a direction class at h={h}: up={n_up}, down={n_down}")

        X_train_seq, X_test_seq = X_seq[train_indices], X_seq[test_indices]
        y_train, y_test = y_seq[train_indices], y_seq[test_indices]
        last_close_train, last_close_test = last_close_seq[train_indices], last_close_seq[test_indices]
        extended_trends_train, extended_trends_test = extended_trends[train_indices], extended_trends[test_indices]

        train_batches = math.ceil(X_train_seq.shape[0] / self.config.BATCH_SIZE)
        test_batches = math.ceil(X_test_seq.shape[0] / self.config.BATCH_SIZE)
        print(f"   Train sequences: {X_train_seq.shape[0]} (batches/epoch: {train_batches})")
        print(f"   Validation sequences: {X_test_seq.shape[0]} (batches: {test_batches}), "
              f"purge {first_val_origin - int(origins[train_indices[-1]])} bars between last train and first val origin")

        # Targets are multi-horizon: shape [N, 3].
        # Use a single scaler fit on ALL horizons (flattened) to keep consistent scaling.
        # M4: with_mean=False, so a scaled 0 is a raw 0 (P(up) thresholds and pred_mean = 0 stay consistent).
        target_scaler = StandardScaler(with_mean=False)
        y_train_flat = y_train.reshape(-1, 1)
        y_test_flat = y_test.reshape(-1, 1)
        y_train_scaled = target_scaler.fit_transform(y_train_flat).reshape(y_train.shape)
        y_test_scaled = target_scaler.transform(y_test_flat).reshape(y_test.shape)

        # M4: one global input scaler (one scale per channel, shared by every lag), fitted on training rows only.
        input_scaler = StandardScaler(with_mean=False)
        X_train_seq_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[-1])
        X_train_seq_scaled = input_scaler.fit_transform(X_train_seq_reshaped).reshape(X_train_seq.shape)
        X_test_seq_reshaped = X_test_seq.reshape(-1, X_test_seq.shape[-1])
        X_test_seq_scaled = input_scaler.transform(X_test_seq_reshaped).reshape(X_test_seq.shape)
        # Scalers are written only into the M10 bundle by train_and_evaluate(), not on every call.

        # Keep references for programmatic use without changing the return signature.
        self.target_scaler = target_scaler
        self.input_scaler = input_scaler

        return (X_train_seq_scaled, y_train_scaled, last_close_train, extended_trends_train,
                X_test_seq_scaled, y_test_scaled, last_close_test, extended_trends_test,
                y_train, y_test, target_scaler)


@dataclass
class TrainResult:
    """Single-source-of-truth training + inference output bundle.

    M8: the *_test fields hold the internal VALIDATION block (last VAL_FRACTION of the training range).
    A real test fold exists only in the calling harness.
    """

    config: 'Config'
    model: 'CustomTrainModel'
    target_scaler: StandardScaler
    input_scaler: Optional[StandardScaler]

    X_test_seq: np.ndarray
    y_test: np.ndarray  # raw targets [N,3], bps log returns (M4)
    last_close_test: np.ndarray
    extended_trends_test: np.ndarray
    history: Any

    # Predictions are raw (inverse-scaled) deltas and head outputs.
    predictions: Dict[str, Dict[str, np.ndarray]]
    metrics: Dict[str, Any]
    bundle_dir: Optional[str] = None  # M10


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

        mse = mean_squared_error(y_t, y_p)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_t, y_p)

        delta_metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
        }
        if safe_mape is not None and smape is not None and wape is not None and reconstruct_prices is not None:
            delta_metrics.update({
                "mape_delta": float(mean_absolute_percentage_error(y_t, y_p)),
                "safe_mape_delta": float(safe_mape(y_t, y_p, min_abs_y=min_abs_delta_for_mape)),
                "smape_delta": float(smape(y_t, y_p)),
                "wape_delta": float(wape(y_t, y_p)),
            })

        out["delta"][h_key] = delta_metrics

        if safe_mape is not None and smape is not None and wape is not None and reconstruct_prices is not None:
            y_true_price = reconstruct_prices(lc_h, y_t)
            y_pred_price = reconstruct_prices(lc_h, y_p)
            out["price"][h_key] = {
                "mse": float(mean_squared_error(y_true_price, y_pred_price)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
                "r2": float(r2_score(y_true_price, y_pred_price)),
                "mape": float(safe_mape(y_true_price, y_pred_price)),
                "smape": float(smape(y_true_price, y_pred_price)),
                "wape": float(wape(y_true_price, y_pred_price)),
            }

        true_dir = (y_t > thr)
        p = None
        if dir_probs is not None and h_key in dir_probs and dir_probs[h_key] is not None:
            p = np.asarray(dir_probs[h_key], dtype=float).reshape(-1)[:n]
            pred_dir = (p >= 0.5)
        else:
            pred_dir = (y_p > thr)

        out["direction"][h_key] = {
            "acc": float(accuracy_score(true_dir.astype(int), pred_dir.astype(int))),
            "f1": float(f1_score(true_dir.astype(int), pred_dir.astype(int), zero_division=0)),
        }
        # M6: AUC of P(up), masked (|move| > deadband) and unmasked (every non-zero move)
        if p is not None:
            from sklearn.metrics import roc_auc_score
            for name, m in (("auc", np.abs(y_t) > thr), ("auc_unmasked", y_t != 0)):
                lab = (y_t[m] > 0).astype(int)
                out["direction"][h_key][name] = (float(roc_auc_score(lab, p[m]))
                                                 if 0 < lab.sum() < len(lab) else float('nan'))
            out["direction"][h_key]["mean_p_up"] = float(np.mean(p))

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
            if batch_metrics_output is not None and self.batch_count % 10 == 0:
                with batch_metrics_output:
                    clear_output(wait=True)
                    batches = self.batch_history.get('batch', [])
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        subplot_titles=("Batch Loss", "Batch Direction Metrics"))
                    if 'loss' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['loss'], mode='lines', name='loss'), row=1, col=1)
                    if 'dir_acc' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['dir_acc'], mode='lines', name='dir_acc'), row=2, col=1)
                    if 'f1' in self.batch_history:
                        fig.add_trace(go.Scatter(x=batches, y=self.batch_history['f1'], mode='lines', name='f1'), row=2, col=1)
                    fig.update_layout(height=450, showlegend=True)
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
            except Exception:
                pass

            # Epoch plots
            with loss_output:
                clear_output(wait=True)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=("Epoch Loss", "Epoch Direction Metrics"))
                epochs = list(range(1, self.epoch_count + 1))
                if 'loss' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['loss'], mode='lines+markers', name='loss'), row=1, col=1)
                if 'val_loss' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['val_loss'], mode='lines+markers', name='val_loss'), row=1, col=1)
                if 'dir_acc_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['dir_acc_avg'], mode='lines+markers', name='dir_acc_avg'), row=2, col=1)
                if 'val_dir_acc_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['val_dir_acc_avg'], mode='lines+markers', name='val_dir_acc_avg'), row=2, col=1)
                if 'f1_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['f1_avg'], mode='lines+markers', name='f1_avg'), row=2, col=1)
                if 'val_f1_avg' in self.history:
                    fig.add_trace(go.Scatter(x=epochs, y=self.history['val_f1_avg'], mode='lines+markers', name='val_f1_avg'), row=2, col=1)
                fig.update_layout(height=520, showlegend=True)
                display(fig)

            with metrics_output:
                clear_output(wait=True)
                keys = [
                    'loss', 'val_loss',
                    'point_loss', 'trend_loss', 'dir_loss', 'nll_loss',
                    'dir_acc_avg', 'val_dir_acc_avg', 'f1_avg', 'val_f1_avg'
                ]
                for k in keys:
                    if k in logs:
                        print(f"{k}: {logs[k]}")

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
    frame: Optional[pd.DataFrame] = None,
) -> TrainResult:
    """Train (optionally) and evaluate, returning a rich result bundle.

    This is intended to be the notebook's single source of truth for:
    - data prep + scaling
    - model heads and extraction
    - metrics and evaluation semantics

    M8: the data passed in (CSV or `frame`, M5) is the TRAINING RANGE only. Its last VAL_FRACTION is the
    validation block (purged), used for early stopping and the LR schedule. Test data never enters here.
    M10: artifacts go to a bundle keyed by FOLD_ID + config hash. force=True always trains (never loads);
    force=False loads a bundle only if its config hash and data fingerprint match, and raises if they do not.
    `calibrate` is accepted for backward compatibility and ignored: the loss-weight calibration was removed (M7).
    """

    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    cfg = config or Config()
    if csv_path is not None:
        cfg.CSV_PATH = csv_path
    cfg = _apply_config_overrides(cfg, config_overrides)
    tf.keras.utils.set_random_seed(int(cfg.SEED))

    print("Starting enhanced model training with extended trend features...")
    data_processor = DataProcessor(cfg)
    df, close_values = data_processor.load_and_prepare_data(read_csv_kwargs=read_csv_kwargs, frame=frame)

    (X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
     X_test_seq, y_test_scaled, last_close_test, extended_trends_test,
     y_train, y_test, target_scaler) = data_processor.prepare_datasets(df, close_values)

    input_scaler = getattr(data_processor, 'input_scaler', None)
    predictor = PricePredictor(cfg)
    base_model = predictor.build_model()
    # M4/M7: unit-free targets, target scaler without centring -> pred_mean = 0, pred_scale = the scaler's scale.
    pred_scale = float(target_scaler.scale_[0])
    pred_mean = 0.0
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

    # M7: no loss-weight calibration. lambda_point = lambda_dir = lambda_var = 1 on unit-free targets.

    opt = optimizers.Adam(learning_rate=cfg.LR)
    custom_model.compile(optimizer=opt)

    # M10: one bundle per (fold, config hash) holds the weights, both scalers and the config.
    bundle_dir = bundle_dir_for(cfg)
    cfg.MODEL_PATH = os.path.join(bundle_dir, "model.weights.h5")
    cfg.SCALER_PATH = os.path.join(bundle_dir, "target_scaler.joblib")
    fingerprint = data_fingerprint(df, close_values)
    log_dir = getattr(cfg, 'LOG_DIR', '.') or '.'
    os.makedirs(log_dir, exist_ok=True)

    csv_logger = callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv"), append=True)
    # M8: early stopping on the purged validation block; restore_best_weights restores the best epoch at train end.
    es = callbacks.EarlyStopping(monitor='val_loss', patience=cfg.PATIENCE, restore_best_weights=True)
    tqdm_callback = TqdmCallback()
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)

    learnable_layer = None
    for layer in custom_model.base_model.layers:
        if getattr(layer, 'name', '').startswith('learnable_indicators'):
            learnable_layer = layer
            break
    params_logger = ParamsLogger(layer=learnable_layer, out_csv=os.path.join(log_dir, 'indicator_params_history.csv'))

    callbacks_list = [csv_logger, es, tqdm_callback, params_logger, lr_scheduler]
    if extra_callbacks:
        callbacks_list += list(extra_callbacks)

    actual_epochs = int(epochs) if epochs is not None else int(cfg.EPOCHS)
    history = None
    if os.path.exists(cfg.MODEL_PATH) and not force:
        # Never load silently: the bundle must match this config hash and this exact training data.
        meta = _read_bundle_meta(bundle_dir)
        if meta.get('config_hash') != config_hash(cfg) or meta.get('data_fingerprint') != fingerprint:
            raise ValueError(f"Bundle {bundle_dir} does not match this config/data "
                             f"(config {meta.get('config_hash')} vs {config_hash(cfg)}, "
                             f"data {meta.get('data_fingerprint')} vs {fingerprint}); pass force=True to retrain.")
        print(f"Loading bundle {bundle_dir} (config hash and data fingerprint match)...")
        custom_model.base_model.load_weights(cfg.MODEL_PATH)
        target_scaler = joblib.load(cfg.SCALER_PATH)
        input_scaler = joblib.load(os.path.join(bundle_dir, "input_scaler.joblib"))
    else:
        print(f"Training for {actual_epochs} epochs...")
        history = custom_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=actual_epochs,
            callbacks=callbacks_list,
            verbose=0,
        )
        org, tr_i, va_i = data_processor.seq_origins, data_processor.train_indices, data_processor.val_indices
        split = {name: str(df['timestamp'].iloc[int(org[i])]) for name, i in
                 (('train_first_origin', tr_i[0]), ('train_last_origin', tr_i[-1]),
                  ('val_first_origin', va_i[0]), ('val_last_origin', va_i[-1]))}
        split.update({'n_train': int(len(tr_i)), 'n_val': int(len(va_i)), 'n_rows': int(len(df))})
        save_bundle(bundle_dir, cfg, custom_model.base_model, target_scaler, input_scaler,
                    extra_meta={'data_fingerprint': fingerprint, 'split': split, 'pred_scale': float(pred_scale),
                                'pred_mean': float(pred_mean),
                                'best_epoch': int(getattr(es, 'best_epoch', -1)),
                                'history': {k: [float(x) for x in v] for k, v in history.history.items()
                                            if k in ('loss', 'val_loss', 'learning_rate', 'lr')}})
        print(f"Bundle saved to {bundle_dir}")

    print("Evaluating enhanced model...")
    X_test_simple = tf.data.Dataset.from_tensor_slices(X_test_seq).batch(cfg.BATCH_SIZE)
    y_pred_all = custom_model.predict(X_test_simple, verbose=0)

    # Extract 3 price heads (scaled deltas)
    y_pred_price_all = np.column_stack([
        y_pred_all[0][:, 0],
        y_pred_all[3][:, 0],
        y_pred_all[6][:, 0],
    ])
    y_pred_price_all = y_pred_price_all[:len(y_test)]

    y_pred_h0_raw = target_scaler.inverse_transform(y_pred_price_all[:, 0].reshape(-1, 1)).ravel()
    y_pred_h1_raw = target_scaler.inverse_transform(y_pred_price_all[:, 1].reshape(-1, 1)).ravel()
    y_pred_h2_raw = target_scaler.inverse_transform(y_pred_price_all[:, 2].reshape(-1, 1)).ravel()

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

    if history is not None:
        _update_bundle_meta(bundle_dir, {'val_metrics': metrics})

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
        bundle_dir=bundle_dir,
    )

# -----------------------------
# M10: artifacts (one bundle per fold id + config hash)
# -----------------------------
_NON_HASHED_CONFIG_KEYS = {'CSV_PATH', 'MODEL_PATH', 'SCALER_PATH', 'ARTIFACT_DIR', 'LOG_DIR', 'FOLD_ID'}


def config_dict(cfg) -> Dict[str, Any]:
    """All upper-case Config attributes (class defaults plus instance overrides), JSON-ready."""
    import json
    out = {}
    for k in dir(cfg):
        if not k.isupper():
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue
        out[k] = json.loads(json.dumps(v, default=str))
    return out


def config_hash(cfg) -> str:
    """Hash of every hyperparameter that changes the model (paths and the fold id excluded)."""
    import hashlib
    import json
    d = {k: v for k, v in config_dict(cfg).items() if k not in _NON_HASHED_CONFIG_KEYS}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def bundle_dir_for(cfg) -> str:
    return os.path.join(getattr(cfg, 'ARTIFACT_DIR', '.') or '.', f"bundle_{cfg.FOLD_ID}_{config_hash(cfg)}")


def data_fingerprint(df, close_values) -> str:
    import hashlib
    h = hashlib.sha256(np.ascontiguousarray(np.asarray(close_values, dtype=np.float32)).tobytes())
    h.update(f"{len(df)}|{df['timestamp'].iloc[0]}|{df['timestamp'].iloc[-1]}".encode())
    return h.hexdigest()[:16]


def _read_bundle_meta(bundle_dir) -> Dict[str, Any]:
    import json
    p = os.path.join(bundle_dir, "config.json")
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


def _write_bundle_meta(bundle_dir, meta):
    import json
    p = os.path.join(bundle_dir, "config.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    os.replace(p + ".tmp", p)


def _update_bundle_meta(bundle_dir, extra):
    meta = _read_bundle_meta(bundle_dir)
    meta.update(extra)
    _write_bundle_meta(bundle_dir, meta)


def save_bundle(bundle_dir, cfg, base_model, target_scaler, input_scaler, extra_meta=None):
    """Write <bundle_dir>/{model.weights.h5, target_scaler.joblib, input_scaler.joblib, config.json}."""
    import datetime
    os.makedirs(bundle_dir, exist_ok=True)
    base_model.save_weights(os.path.join(bundle_dir, "model.weights.h5"))
    joblib.dump(target_scaler, os.path.join(bundle_dir, "target_scaler.joblib"))
    joblib.dump(input_scaler, os.path.join(bundle_dir, "input_scaler.joblib"))
    meta = {
        'model_version': getattr(cfg, 'MODEL_VERSION', None),
        'fold_id': cfg.FOLD_ID,
        'config_hash': config_hash(cfg),
        'config': config_dict(cfg),
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'tf_version': tf.__version__,
    }
    meta.update(extra_meta or {})
    _write_bundle_meta(bundle_dir, meta)


@dataclass
class Bundle:
    """A trained model ready for inference: base Keras model, both scalers and the config (M10)."""
    config: 'Config'
    model: Any
    target_scaler: StandardScaler
    input_scaler: StandardScaler
    path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    _forward: Any = None


def load_bundle(bundle_dir) -> Bundle:
    """Rebuild the model from config.json and load its weights and scalers (works in a fresh process)."""
    meta = _read_bundle_meta(bundle_dir)
    if not meta:
        raise FileNotFoundError(f"No bundle at {bundle_dir}")
    cfg = Config()
    for k, v in meta['config'].items():
        setattr(cfg, k, v)
    if config_hash(cfg) != meta['config_hash']:
        raise ValueError(f"Config hash mismatch in {bundle_dir}: {config_hash(cfg)} vs {meta['config_hash']}")
    base_model = PricePredictor(cfg).build_model()
    base_model.load_weights(os.path.join(bundle_dir, "model.weights.h5"))
    return Bundle(config=cfg, model=base_model,
                  target_scaler=joblib.load(os.path.join(bundle_dir, "target_scaler.joblib")),
                  input_scaler=joblib.load(os.path.join(bundle_dir, "input_scaler.joblib")),
                  path=bundle_dir, meta=meta)


def bundle_from_result(result: TrainResult) -> Bundle:
    return Bundle(config=result.config, model=result.model.base_model, target_scaler=result.target_scaler,
                  input_scaler=result.input_scaler, path=result.bundle_dir,
                  meta=_read_bundle_meta(result.bundle_dir) if result.bundle_dir else None)


# -----------------------------
# M11: causal, batched inference at arbitrary decision bars
# -----------------------------
def predict_at(bundle, frame, idx, batch_size=4096, chunk=65536) -> Dict[str, np.ndarray]:
    """Forecasts at decision bars `idx` (row positions of `frame`, a clean M5 frame).

    Each origin o reads bars o-LOOKBACK+1..o only (inputs and causal extra channels), so bars after o
    cannot change its forecast. Returns float32 arrays [len(idx), 3] for h in HORIZON_STEPS:
      'mu'    : the tau=0.5 price head, bps of log return;
      'pup'   : the direction head, P(up) beyond the deadband;
      'scale' : the Student-t scale (DF_VAR dof), bps. The predictive variance is DF/(DF-2) * scale^2.
    """
    if isinstance(bundle, str):
        bundle = load_bundle(bundle)
    cfg = bundle.config
    proc = DataProcessor(cfg)
    df, close_values = proc.load_and_prepare_data(frame=frame)
    if len(df) != len(frame):
        raise ValueError(f"frame must be sorted, unique and NaN-free: {len(frame)} rows -> {len(df)} after cleaning")
    extra = proc.compute_extra_channels(df) if getattr(cfg, 'EXTRA_CHANNELS', False) else None
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    n_h = len(cfg.HORIZON_STEPS)
    out = {k: np.empty((len(idx), n_h), dtype=np.float32) for k in ('mu', 'pup', 'scale')}
    if bundle._forward is None:
        model = bundle.model
        bundle._forward = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    t_scale = float(bundle.target_scaler.scale_[0])
    var_floor, var_cap = float(cfg.VAR_FLOOR), float(cfg.VAR_CAP)
    for a in range(0, len(idx), chunk):
        X = build_input_windows(close_values, extra, idx[a:a + chunk], cfg.LOOKBACK)
        X = bundle.input_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
        for b in range(0, len(X), batch_size):
            ys = [np.asarray(y) for y in bundle._forward(tf.constant(X[b:b + batch_size]))]
            sl = slice(a + b, a + b + len(ys[0]))
            for j in range(n_h):
                out['mu'][sl, j] = bundle.target_scaler.inverse_transform(ys[3 * j][:, :1]).ravel()
                out['pup'][sl, j] = ys[3 * j + 1].ravel()
                out['scale'][sl, j] = np.sqrt(np.clip(ys[3 * j + 2].ravel(), var_floor, var_cap)) * t_scale
    return out

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
        self.grad_multiplier = config.INDICATOR_GRAD_MULT  # M1: kept for reference; not used in the forward pass

    def _logit_from_alpha(self, alpha):
        return mh.logit_from_alpha(alpha, self.epsilon)

    def _alpha_from_logit(self, logit):
        return mh.alpha_from_logit(logit)

    def _logit_from_period(self, period):
        return mh.logit_from_period(period, self.epsilon)

    def _period_from_logit(self, logit):
        return mh.period_from_logit(logit, self.epsilon)

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
            # M2: exact inverse of the forward period softplus(raw) + 1 (was log(p), read back as softplus+1)
            init_raw = float(mh.momentum_raw_from_period(
                float(p), self.config.MOMENTUM_CLIP_MIN, self.config.MOMENTUM_CLIP_MAX))
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(init_raw),
                                trainable=True,
                                name=f'momentum_raw_{i}',
                                regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))
            self.momentum_raw.append(v)
            self.all_logit_vars.append(v)

        super().build(input_shape)

    def ewma_seq(self, x_seq, alpha_scalar):
        return mh.ewma_sequence(x_seq, alpha_scalar)

    def call(self, inputs, training=None):
        x, meta_adjust = inputs
        x = tf.cast(x, tf.float32)
        features = []
        idx = 0  # Index for slicing meta_adjust

        for logit in self.alpha_vars_ma:
            # M1: the forward pass uses the logit itself (the old `logit + stop_gradient(logit)*(m-1)` was 20*logit)
            adjusted_logit = logit + meta_adjust[:, idx] * self.meta_scale
            alpha = self._alpha_from_logit(adjusted_logit)
            ema_seq = self.ewma_seq(x, alpha)
            features.append(ema_seq)
            idx += 1

        for i in range(len(self.config.MACD_SETTINGS)):
            fast_var = self.macd_alpha_vars[f'macd_{i}_fast']
            slow_var = self.macd_alpha_vars[f'macd_{i}_slow']
            sig_var = self.macd_alpha_vars[f'macd_{i}_signal']
            fast_logit = fast_var + meta_adjust[:, idx] * self.meta_scale  # M1
            slow_logit = slow_var + meta_adjust[:, idx+1] * self.meta_scale
            sig_logit = sig_var + meta_adjust[:, idx+2] * self.meta_scale
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
            short_logit = short_var + meta_adjust[:, idx] * self.meta_scale  # M1
            long_logit = long_var + meta_adjust[:, idx+1] * self.meta_scale
            sig_logit = sig_var + meta_adjust[:, idx+2] * self.meta_scale
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
            adjusted_logit = logit + meta_adjust[:, idx] * self.meta_scale  # M1
            rsi_alpha = self._alpha_from_logit(adjusted_logit)
            gains_ema = self.ewma_seq(gains_padded, rsi_alpha)
            losses_ema = self.ewma_seq(losses_padded, rsi_alpha)
            rs = gains_ema / (losses_ema + 1e-8)
            # M3: RSI on a centred unit scale, (RSI - 50)/100 in [-0.5, 0.5]. On the 0-100 scale its three channels
            # set the LayerNorm mean and spread (about 13x every other indicator) and drown the rest.
            rsi_seq = 0.5 - (1.0 / (1.0 + rs))
            features.append(rsi_seq)
            idx += 1

        for logit in self.bb_alpha_vars:
            adjusted_logit = logit + meta_adjust[:, idx] * self.meta_scale  # M1
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
            period = mh.momentum_period_from_raw(adjusted_raw)
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
        # M1: effective periods through the forward mapping (alpha = sigmoid(logit), P = 2/alpha - 1;
        # momentum P = softplus(raw) + 1), before the per-sample meta_adjust offset.
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
            p = mh.momentum_period_from_raw(raw).numpy()
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
        # M4: [B, LOOKBACK, 1 + 5 extra channels]; channel 0 = bps relative to the last close.
        n_channels = 1 + (5 if getattr(self.config, 'EXTRA_CHANNELS', False) else 0)
        inp = layers.Input(shape=(self.config.LOOKBACK, n_channels), name='close_sequence')
        close_seq = inp[:, :, 0]  # indicators and meta_adjust read channel 0 only

        # Compute meta_adjust from raw input stats
        inp_resh = layers.Reshape((self.config.LOOKBACK, 1))(close_seq)  # [B, LOOKBACK, 1] for pooling
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
        ind_seq = LearnableIndicators(self.config, name='learnable_indicators')([close_seq, meta_adjust])
        # M3: normalise the indicator vector per time step so %B-like channels cannot saturate the GRU gates
        ind_seq = layers.LayerNormalization(name='indicator_norm')(ind_seq)
        if n_channels > 1:
            # M4: the extra causal channels join after the LayerNorm
            ind_seq = layers.Concatenate(axis=-1, name='indicators_and_extras')([ind_seq, inp[:, :, 1:]])

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
        
        # ---- TOWER 0 (1-minute horizon) ----
        tower_h0 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h0 = layers.Dense(1, name='price_h0')(tower_h0)
        direction_h0 = layers.Dense(1, activation='sigmoid', name='direction_h0')(tower_h0)
        variance_h0 = layers.Dense(1, activation='softplus', name='variance_h0')(tower_h0)
        # Keep strictly-positive variance via softplus; avoid forcing an additional +1 offset.

        # ---- TOWER 1 (5-minute horizon - PRIMARY) ----
        tower_h1 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h1 = layers.Dense(1, name='price_h1')(tower_h1)
        direction_h1 = layers.Dense(1, activation='sigmoid', name='direction_h1')(tower_h1)
        variance_h1 = layers.Dense(1, activation='softplus', name='variance_h1')(tower_h1)
        # Keep strictly-positive variance via softplus; avoid forcing an additional +1 offset.

        # ---- TOWER 2 (15-minute horizon) ----
        tower_h2 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h2 = layers.Dense(1, name='price_h2')(tower_h2)
        direction_h2 = layers.Dense(1, activation='sigmoid', name='direction_h2')(tower_h2)
        variance_h2 = layers.Dense(1, activation='softplus', name='variance_h2')(tower_h2)
        # Keep strictly-positive variance via softplus; avoid forcing an additional +1 offset.

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
    def _logit_from_alpha(self, alpha):
        return mh.logit_from_alpha(alpha, self.epsilon)

    def _alpha_from_logit(self, logit):
        return mh.alpha_from_logit(logit)

    def _logit_from_period(self, period):
        return mh.logit_from_period(period, self.epsilon)

    def _period_from_logit(self, logit):
        return mh.period_from_logit(logit, self.epsilon)
    # -------------------------
    # Unified element-wise Huber
    # -------------------------
    def huber(self, x, delta=None):
        """Element-wise Huber (returns same-shape tensor). Works on scaled differences."""
        if delta is None:
            delta = self.huber_delta
        return mh.huber_loss(x, delta)

    # Small utility: reduce-mean with safe casting
    def _reduce_mean(self, x):
        return mh.safe_reduce_mean(x)

    @staticmethod
    def _normal_cdf(z):
        """Standard Normal CDF using erf; z can be any float tensor."""
        return mh.normal_cdf(z)

    # -------------------------
    # Utility / transforms (moved outside class to avoid tracing issues)
    # -------------------------
    @staticmethod
    def _to_scaled_static(raw, pred_mean, pred_scale, eps=1e-8):
        """Convert raw prices to scaled units (same domain as dataset scaling)."""
        return mh.to_scaled(raw, pred_mean, pred_scale, eps)

    def _to_scaled(self, raw):
        """Instance helper that uses the stored scaling parameters."""
        return mh.to_scaled(raw, self.pred_mean, self.pred_scale, self.eps)

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

        return mh.focal_loss(true_labels, logits, alpha, gamma, reduce)

    # -------------------------
    # Point loss (pinball/quantile)
    # -------------------------
    def point_huber(self, y_true_scaled, y_pred_scaled, last_close_scaled=None, delta=None):
        """Point loss in scaled space using pinball (quantile) loss for asymmetry.

        For delta-target training, the natural pivot is 0 (not last_close).
        `last_close_scaled` is kept optional for API compatibility.
        """
        # Squeeze explicitly on axis=1 to be shape-safe
        y_true = tf.squeeze(y_true_scaled, axis=1)
        y_pred = tf.squeeze(y_pred_scaled, axis=1)

        # Use pinball loss from math_helpers
        return mh.pinball_loss(y_true, y_pred, tau=self.config.TAU_PRICE)


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
    # Loss computation helper methods for refactored custom_loss
    # -------------------------
    def _prepare_targets_and_directions(self, y_true, y_pred, last_close):
        """Extract and prepare targets and direction labels for all horizons.

        Returns: Dict containing prepared targets, masks, and direction labels.
        """
        # Unpack 9 outputs
        price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2 = y_pred

        # Prepare targets (multi-horizon)
        y_true = tf.cast(y_true, tf.float32)  # [B, 3]
        y_true_raw = y_true * self.pred_scale + self.pred_mean  # [B, 3]  (delta_raw)
        last_close_squeeze = tf.squeeze(last_close, axis=1)  # [B] (raw price)

        y_true_h0 = y_true[:, 0:1]
        y_true_h1 = y_true[:, 1:2]
        y_true_h2 = y_true[:, 2:3]

        y_true_raw_h0 = y_true_raw[:, 0]
        y_true_raw_h1 = y_true_raw[:, 1]
        y_true_raw_h2 = y_true_raw[:, 2]

        # Trade-aware direction labeling with optional deadband
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

        return {
            'predictions': {'price_h0': price_h0, 'dir_h0': dir_h0, 'var_h0': var_h0,
                          'price_h1': price_h1, 'dir_h1': dir_h1, 'var_h1': var_h1,
                          'price_h2': price_h2, 'dir_h2': dir_h2, 'var_h2': var_h2},
            'targets': {'y_true_h0': y_true_h0, 'y_true_h1': y_true_h1, 'y_true_h2': y_true_h2},
            'targets_raw': {'y_true_raw_h0': y_true_raw_h0, 'y_true_raw_h1': y_true_raw_h1,
                           'y_true_raw_h2': y_true_raw_h2},
            'directions': {'true_dir_h0': true_dir_h0, 'true_dir_h1': true_dir_h1, 'true_dir_h2': true_dir_h2},
            'masks': {'mask_h0': mask_h0, 'mask_h1': mask_h1, 'mask_h2': mask_h2},
            'last_close': last_close_squeeze,
            'deadband': deadband
        }

    def _compute_point_losses(self, predictions, targets):
        """Compute point losses for all three horizons."""
        point_loss_h0_val = self.lambda_short * self.point_huber(targets['y_true_h0'], predictions['price_h0'])
        point_loss_h1_val = self.lambda_point * self.point_huber(targets['y_true_h1'], predictions['price_h1'])
        point_loss_h2_val = self.lambda_long * self.point_huber(targets['y_true_h2'], predictions['price_h2'])

        return {
            'point_h0': point_loss_h0_val,
            'point_h1': point_loss_h1_val,
            'point_h2': point_loss_h2_val,
            'total': point_loss_h0_val + point_loss_h1_val + point_loss_h2_val
        }

    def _compute_trend_losses(self, targets_raw, extended_trends, last_close):
        """Compute trend losses including extended trends and cross-horizon coherence."""
        # Input-based prior: align predicted deltas with extended_trends (pct_change * last_close)
        trend_prior_h0 = tf.reduce_mean(tf.square(targets_raw['y_true_raw_h0'] - extended_trends[:, 0] * last_close))
        trend_prior_h1 = tf.reduce_mean(tf.square(targets_raw['y_true_raw_h1'] - extended_trends[:, 1] * last_close))
        trend_prior_h2 = tf.reduce_mean(tf.square(targets_raw['y_true_raw_h2'] - extended_trends[:, 2] * last_close))

        # Cross-horizon coherence: penalize inconsistent curvature
        sign_h0 = tf.sign(targets_raw['y_true_raw_h0'])
        sign_h1 = tf.sign(targets_raw['y_true_raw_h1'])
        sign_h2 = tf.sign(targets_raw['y_true_raw_h2'])
        coherence_penalty = tf.reduce_mean(tf.cast(tf.math.logical_xor(sign_h1 == sign_h0, sign_h1 == sign_h2), tf.float32))

        local_trend_h0 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h0 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h0 = self.lambda_extended_trend * trend_prior_h0

        local_trend_h1 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h1 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h1 = self.lambda_extended_trend * trend_prior_h1

        local_trend_h2 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h2 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h2 = self.lambda_extended_trend * trend_prior_h2

        total = extended_trend_h0 + extended_trend_h1 + extended_trend_h2 + coherence_penalty * 0.01

        return {
            'local_h0': local_trend_h0, 'global_h0': global_trend_h0, 'extended_h0': extended_trend_h0,
            'local_h1': local_trend_h1, 'global_h1': global_trend_h1, 'extended_h1': extended_trend_h1,
            'local_h2': local_trend_h2, 'global_h2': global_trend_h2, 'extended_h2': extended_trend_h2,
            'total': total
        }

    def _compute_direction_losses(self, predictions, directions, masks):
        """Compute focal direction losses with deadband masking."""
        dir_pred_h0 = tf.squeeze(predictions['dir_h0'], axis=1)
        dir_pred_h1 = tf.squeeze(predictions['dir_h1'], axis=1)
        dir_pred_h2 = tf.squeeze(predictions['dir_h2'], axis=1)

        per_ex_h0 = self.focal_loss(directions['true_dir_h0'], dir_pred_h0, reduce=False)
        per_ex_h1 = self.focal_loss(directions['true_dir_h1'], dir_pred_h1, reduce=False)
        per_ex_h2 = self.focal_loss(directions['true_dir_h2'], dir_pred_h2, reduce=False)

        dir_loss_h0 = tf.reduce_sum(per_ex_h0 * masks['mask_h0']) / (tf.reduce_sum(masks['mask_h0']) + self.eps)
        dir_loss_h1 = tf.reduce_sum(per_ex_h1 * masks['mask_h1']) / (tf.reduce_sum(masks['mask_h1']) + self.eps)
        dir_loss_h2 = tf.reduce_sum(per_ex_h2 * masks['mask_h2']) / (tf.reduce_sum(masks['mask_h2']) + self.eps)

        return {
            'dir_h0': dir_loss_h0,
            'dir_h1': dir_loss_h1,
            'dir_h2': dir_loss_h2,
            'dir_preds': {'dir_pred_h0': dir_pred_h0, 'dir_pred_h1': dir_pred_h1, 'dir_pred_h2': dir_pred_h2},
            'total': self.lambda_dir * (dir_loss_h0 + dir_loss_h1 + dir_loss_h2)
        }

    def _compute_variance_nll_losses(self, predictions, targets):
        """Compute variance NLL losses using Student's t distribution."""
        var_floor = tf.cast(getattr(self.config, 'VAR_FLOOR', 1e-4), tf.float32)
        var_cap = tf.cast(getattr(self.config, 'VAR_CAP', 1e4), tf.float32)
        var_h0_c = tf.clip_by_value(predictions['var_h0'], var_floor, var_cap)
        var_h1_c = tf.clip_by_value(predictions['var_h1'], var_floor, var_cap)
        var_h2_c = tf.clip_by_value(predictions['var_h2'], var_floor, var_cap)

        # Student's t NLL for robustness to heavy tails
        nu = tf.constant(self.config.DF_VAR, dtype=tf.float32)
        error_h0 = targets['y_true_h0'] - predictions['price_h0']
        error_h1 = targets['y_true_h1'] - predictions['price_h1']
        error_h2 = targets['y_true_h2'] - predictions['price_h2']

        term1 = tf.math.lgamma((nu + 1.0)/2.0) - tf.math.lgamma(nu/2.0)
        term2_h0 = -0.5 * tf.math.log(tf.constant(math.pi) * nu * var_h0_c)
        term2_h1 = -0.5 * tf.math.log(tf.constant(math.pi) * nu * var_h1_c)
        term2_h2 = -0.5 * tf.math.log(tf.constant(math.pi) * nu * var_h2_c)

        term3_h0 = - (nu + 1.0)/2.0 * tf.math.log(1.0 + tf.square(error_h0) / (nu * var_h0_c + self.eps))
        term3_h1 = - (nu + 1.0)/2.0 * tf.math.log(1.0 + tf.square(error_h1) / (nu * var_h1_c + self.eps))
        term3_h2 = - (nu + 1.0)/2.0 * tf.math.log(1.0 + tf.square(error_h2) / (nu * var_h2_c + self.eps))

        nll_h0_val = tf.reduce_mean(-(term1 + term2_h0 + term3_h0))
        nll_h1_val = tf.reduce_mean(-(term1 + term2_h1 + term3_h1))
        nll_h2_val = tf.reduce_mean(-(term1 + term2_h2 + term3_h2))

        return {
            'nll_h0': nll_h0_val,
            'nll_h1': nll_h1_val,
            'nll_h2': nll_h2_val,
            'var_clipped': {'var_h0_c': var_h0_c, 'var_h1_c': var_h1_c, 'var_h2_c': var_h2_c},
            'total': self.lambda_var * (nll_h0_val + nll_h1_val + nll_h2_val)
        }

    def _compute_direction_alignment(self, predictions, dir_preds, var_clipped, masks, last_close, deadband):
        """Compute alignment loss between direction head and distribution-implied probabilities."""
        # Gaussian-implied direction probabilities from (mu, var)
        mu_h0 = tf.squeeze(predictions['price_h0'], axis=1)
        mu_h1 = tf.squeeze(predictions['price_h1'], axis=1)
        mu_h2 = tf.squeeze(predictions['price_h2'], axis=1)

        sigma_h0 = tf.sqrt(tf.squeeze(var_clipped['var_h0_c'], axis=1) + self.eps)
        sigma_h1 = tf.sqrt(tf.squeeze(var_clipped['var_h1_c'], axis=1) + self.eps)
        sigma_h2 = tf.sqrt(tf.squeeze(var_clipped['var_h2_c'], axis=1) + self.eps)

        deadband_delta_scaled = (deadband * last_close) / (self.pred_scale + self.eps)

        # M6: with a deadband the head estimates P(r > d | |r| > d); align it with the same Gaussian quantity
        gauss_p_up_h0 = mh.gauss_p_up_beyond_deadband(mu_h0, sigma_h0, deadband_delta_scaled, self.eps)
        gauss_p_up_h1 = mh.gauss_p_up_beyond_deadband(mu_h1, sigma_h1, deadband_delta_scaled, self.eps)
        gauss_p_up_h2 = mh.gauss_p_up_beyond_deadband(mu_h2, sigma_h2, deadband_delta_scaled, self.eps)

        lambda_dir_align = tf.constant(float(getattr(self.config, 'LAMBDA_DIR_ALIGN', 0.0)), dtype=tf.float32)

        align_h0 = tf.keras.losses.binary_crossentropy(gauss_p_up_h0, dir_preds['dir_pred_h0'])
        align_h1 = tf.keras.losses.binary_crossentropy(gauss_p_up_h1, dir_preds['dir_pred_h1'])
        align_h2 = tf.keras.losses.binary_crossentropy(gauss_p_up_h2, dir_preds['dir_pred_h2'])

        # Apply deadband mask to alignment
        align_h0 = tf.reduce_sum(align_h0 * masks['mask_h0']) / (tf.reduce_sum(masks['mask_h0']) + self.eps)
        align_h1 = tf.reduce_sum(align_h1 * masks['mask_h1']) / (tf.reduce_sum(masks['mask_h1']) + self.eps)
        align_h2 = tf.reduce_sum(align_h2 * masks['mask_h2']) / (tf.reduce_sum(masks['mask_h2']) + self.eps)

        return lambda_dir_align * (align_h0 + align_h1 + align_h2)

    def _compute_regularization_and_volatility(self, predictions, targets, y_true):
        """Compute regularization and volatility penalty losses."""
        # Regularization
        reg_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)
        inter_reg = self.config.LAMBDA_INTER * reg_loss

        # Volatility penalty (use primary horizon h1)
        actual_trend = y_true[:, 1]
        pred_trend_scaled = tf.squeeze(predictions['price_h1'], axis=1)
        actual_std = tf.math.reduce_std(actual_trend)
        pred_std = tf.math.reduce_std(pred_trend_scaled)
        vol_diff = tf.abs(pred_std - actual_std)
        vol_diff_clipped = tf.minimum(vol_diff, 10.0)
        vol_loss = vol_diff_clipped * self.lambda_vol
        vol_loss = tf.where(tf.math.is_finite(vol_loss), vol_loss, tf.constant(0.0, dtype=tf.float32))

        return {'reg_loss': reg_loss, 'inter_reg': inter_reg, 'vol_loss': vol_loss}

    # -------------------------
    # Combined custom loss (REFACTORED for clarity)
    # -------------------------
    def custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends):
        """
        Multi-horizon, multi-task loss computation (REFACTORED).

        y_pred has 9 outputs: [price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2]
        Each horizon (h0=1min, h1=5min, h2=15min) has independent price, direction, and confidence outputs.

        Losses computed:
        - 3 point losses (one per horizon)
        - 3×3 trend losses (local, global, extended for each horizon)
        - 3 focal direction losses (per horizon)
        - 3 variance NLL losses (per horizon)
        - Direction alignment loss
        - Regularization and volatility penalties

        Returns: 29-component tuple for backward compatibility (via LossComponents.as_tuple())
        """
        # Step 1: Prepare all targets and direction labels
        prepared = self._prepare_targets_and_directions(y_true, y_pred, last_close)

        # Step 2: Compute point losses
        point_losses = self._compute_point_losses(prepared['predictions'], prepared['targets'])

        # Step 3: Compute trend losses
        trend_losses = self._compute_trend_losses(
            prepared['targets_raw'], extended_trends, prepared['last_close']
        )

        # Step 4: Compute direction losses
        dir_losses = self._compute_direction_losses(
            prepared['predictions'], prepared['directions'], prepared['masks']
        )

        # Step 5: Compute variance NLL losses
        nll_losses = self._compute_variance_nll_losses(prepared['predictions'], prepared['targets'])

        # Step 6: Compute direction alignment loss
        dir_align_loss = self._compute_direction_alignment(
            prepared['predictions'], dir_losses['dir_preds'], nll_losses['var_clipped'],
            prepared['masks'], prepared['last_close'], prepared['deadband']
        )

        # Step 7: Compute regularization and volatility penalties
        reg_and_vol = self._compute_regularization_and_volatility(
            prepared['predictions'], prepared['targets'], y_true
        )

        # Step 8: Sum all losses.
        # M7: the trend terms compare TRUE targets with past trends; no prediction enters them, so their
        # gradient is zero. They are still computed and logged, with zero weight in the total.
        total_loss = (
            point_losses['total'] +
            dir_losses['total'] +
            nll_losses['total'] +
            dir_align_loss +
            reg_and_vol['reg_loss'] +
            reg_and_vol['inter_reg'] +
            reg_and_vol['vol_loss']
        )

        # Step 9: Package results in structured format
        loss_components = LossComponents(
            total=total_loss,
            point_h0=point_losses['point_h0'],
            point_h1=point_losses['point_h1'],
            point_h2=point_losses['point_h2'],
            local_trend_h0=trend_losses['local_h0'],
            global_trend_h0=trend_losses['global_h0'],
            extended_trend_h0=trend_losses['extended_h0'],
            local_trend_h1=trend_losses['local_h1'],
            global_trend_h1=trend_losses['global_h1'],
            extended_trend_h1=trend_losses['extended_h1'],
            local_trend_h2=trend_losses['local_h2'],
            global_trend_h2=trend_losses['global_h2'],
            extended_trend_h2=trend_losses['extended_h2'],
            dir_h0=dir_losses['dir_h0'],
            dir_h1=dir_losses['dir_h1'],
            dir_h2=dir_losses['dir_h2'],
            nll_h0=nll_losses['nll_h0'],
            nll_h1=nll_losses['nll_h1'],
            nll_h2=nll_losses['nll_h2'],
            reg_loss=reg_and_vol['reg_loss'],
            inter_reg=reg_and_vol['inter_reg'],
            vol_loss=reg_and_vol['vol_loss']
        )

        # Return as tuple for backward compatibility with train_step
        return loss_components.as_tuple()

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
                # M2: clip the period, then write back through the exact inverse of softplus(raw) + 1
                var.assign(mh.momentum_writeback(var, min_p, max_p))

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
        gauss_p_up_h0 = mh.gauss_p_up_beyond_deadband(mu_h0, tf.sqrt(var_h0_c), deadband_delta_scaled, self.eps)
        gauss_p_up_h1 = mh.gauss_p_up_beyond_deadband(mu_h1, tf.sqrt(var_h1_c), deadband_delta_scaled, self.eps)
        gauss_p_up_h2 = mh.gauss_p_up_beyond_deadband(mu_h2, tf.sqrt(var_h2_c), deadband_delta_scaled, self.eps)

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
        gauss_p_up_h0 = mh.gauss_p_up_beyond_deadband(mu_h0, tf.sqrt(var_h0_c), deadband_delta_scaled, self.eps)
        gauss_p_up_h1 = mh.gauss_p_up_beyond_deadband(mu_h1, tf.sqrt(var_h1_c), deadband_delta_scaled, self.eps)
        gauss_p_up_h2 = mh.gauss_p_up_beyond_deadband(mu_h2, tf.sqrt(var_h2_c), deadband_delta_scaled, self.eps)

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
        _first_present(out, [f"{train_pref}dir_acc_{primary_horizon}", f"{train_fallback}dir_acc_{primary_horizon}"]),
    )
    set_if_missing(
        "f1",
        _first_present(out, [f"{train_pref}dir_f1_{primary_horizon}", f"{train_fallback}dir_f1_{primary_horizon}"]),
    )
    set_if_missing(
        "dir_mcc",
        _first_present(out, [f"{train_pref}dir_mcc_{primary_horizon}", f"{train_fallback}dir_mcc_{primary_horizon}"]),
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
        _first_present(out, [f"{val_pref}dir_acc_{primary_horizon}", f"{val_fallback}dir_acc_{primary_horizon}", "val_dir_acc"]),
    )
    set_if_missing(
        "val_f1",
        _first_present(out, [f"{val_pref}dir_f1_{primary_horizon}", f"{val_fallback}dir_f1_{primary_horizon}", "val_f1"]),
    )
    set_if_missing(
        "val_dir_mcc",
        _first_present(out, [f"{val_pref}dir_mcc_{primary_horizon}", f"{val_fallback}dir_mcc_{primary_horizon}"]),
    )
    set_if_missing(
        "val_dir_sensitivity",
        _first_present(out, [f"{val_pref}dir_sensitivity_{primary_horizon}", f"{val_fallback}dir_sensitivity_{primary_horizon}"]),
    )
    set_if_missing(
        "val_dir_specificity",
        _first_present(out, [f"{val_pref}dir_specificity_{primary_horizon}", f"{val_fallback}dir_specificity_{primary_horizon}"]),
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
                print(f"  {label}: MSE={hm.get('mse'):.6f}, RMSE={hm.get('rmse'):.6f}, R2={hm.get('r2'):.6f}")
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
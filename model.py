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
import plotly.io as pio
#pio.renderers.default = 'colab'

try:
    # Optional local utilities (kept lightweight). If missing, fall back to sklearn MAPE only.
    from metrics_utils import safe_mape, smape, wape, reconstruct_prices, mask_by_min_abs_y, pit_uniformity
except Exception:
    safe_mape = None
    smape = None
    wape = None
    reconstruct_prices = None
    mask_by_min_abs_y = None

try:
    from calibration import CalibrationPipeline as _CalibrationPipeline
except Exception:  # calibration package missing — non-fatal
    _CalibrationPipeline = None

# -----------------------------
class Config:

    HOUR = 60
    DAY = HOUR * 24

    # Data Configuration tuned for minute-level data
    CSV_PATH = csv
    LOOKBACK = HOUR   # Reduced to 1 hour of minute data
    WINDOW_STEP = 1  # Generate a training sample every minute for true minute-level modeling
    RESAMPLE_MINUTES = 1  # Optionally aggregate to coarser bars (e.g., set to 5 for 5-minute bars)
    BATCH_SIZE = 1440 #int(2160)#0 / 10
    EPOCHS = 2 * 10
    LR = 2e-4  # Fixed from critically low 1e-10; reasonable for Adam optimizer
    PATIENCE = EPOCHS# //2  # lr scheduler patience (set to half of total epochs for gradual decay, or equal to epochs for no decay)
    EARLY=EPOCHS # Early stopping patience (set to total epochs for no early stopping, or a smaller value for actual early stopping)
    MAX_SEQUENCE_COUNT = 1440 *  (31 +6 ) #int(1440 * 60 + 60 * 0.2)#(31 +6 ) #1440 * 364## / 10  # Limit most recent sequences to bound training size
    

    

    # Use integer periods to avoid float indexing issues
    # Extended trend features are computed as percent-change over these lags (in minutes)
    EXTENDED_TREND_PERIODS = [10, 15, 20]  # 1m, 5m, 15m

    # Supervision horizons (in minutes ahead from last_close). These define the 3 output towers.
    # h0=1m, h1=5m, h2=15m.
    HORIZON_STEPS = [10, 15, 20]


#1 - 15
# Loss Function Weights
    DAMPING = 0.5  # Legacy alias — use CALIB_DAMPING for new code

    # === LAMBDA CALIBRATION CONTROLS ===
    # Pre-training pass that measures natural loss magnitudes and rescales lambdas so all
    # components start at the same scale, preventing any one term from dominating by virtue
    # of units alone. Formula: new_λ = clip(orig_λ × (ref/med)^d, λ_min, λ_max)
    # where ref = mean of non-zero medians, med = per-component median, d = damping.
    CALIB_WARMUP_FRACTION = 0.05  # BN warmup: fraction of training batches (e.g. 0.15 × 25 ≈ 4 batches)
    CALIB_SAMPLE_FRACTION = 0.1  # Loss sampling: fraction of training batches (e.g. 0.35 × 25 ≈ 9 batches)
    CALIB_LAMBDA_MIN = 0.1       # Lower clamp bound on all calibrated lambdas
    CALIB_LAMBDA_MAX = 20.0      # Upper clamp bound on all calibrated lambdas
    CALIB_DAMPING = 1          # Global damping d ∈ [0, 1]. 0=no change, 1=full equalization
    # Per-component damping overrides (None → fall back to CALIB_DAMPING).
    # Set lower values (e.g. 0.2) for noisy components where you want gentler adjustment.
    CALIB_DAMPING_POINT = None   # Applies to lambda_short, lambda_point, lambda_long
    CALIB_DAMPING_TREND = None   # Applies to lambda_extended_trend
    CALIB_DAMPING_DIR   = None   # Applies to lambda_dir (focal+dice)
    CALIB_DAMPING_VAR   = None   # Applies to lambda_var (NLL)
    CALIB_DAMPING_CRPS  = None   # Applies to lambda_crps
    CALIB_DAMPING_ECE   = None   # Applies to lambda_soft_ece
    CALIB_DAMPING_VOL   = None   # Applies to lambda_vol
    # Outer-multiplier calibration (default OFF — preserves existing behavior).
    # When True, calibrates lambda_trend_outer, lambda_dir_outer, lambda_nll_outer
    # using the already-calibrated per-component group sums.
    CALIB_OUTER = False

    LAMBDA_LOCAL_TREND  = 1.0
    LAMBDA_GLOBAL_TREND =  1.0
    LAMBDA_EXTENDED_TREND = 1.0
    LAMBDA_QUANTILE = 1.0
    REG_MOMENTUM_L2 = 0
    INDICATOR_L2 = 0     # Dedicated L2 for indicator logit vars (separate from NN Dense weights)
    INDICATOR_LR_MULT = 5.0  # Insdicator optimizer LR = LR * INDICATOR_LR_MULT
    MOMENTUM_CLIP_MIN = 1.0
    MOMENTUM_CLIP_MAX = LOOKBACK
    USE_HUBER = True
    
    # === HORIZON-SPECIFIC LOSS WEIGHTS ===
    # Per-horizon lambda weights for point loss (delta prediction accuracy)
    # Different horizons may have different importance/difficulty:
    # - h0 (1-min): Short-term noise, harder to predict, may need lower weight to avoid overfitting noise
    # - h1 (5-min): Primary horizon, balanced signal/noise, standard weight
    # - h2 (15-min): Long-term trend, more stable, higher weight to enforce consistency
    LAMBDA_SHORT = 1.0  # h0 (1-min):  Reduced from 1.0 to avoid noise overfitting
    LAMBDA_POINT = 1.0   # h1 (5-min):  Primary horizon baseline
    LAMBDA_LONG = 1.0   # h2 (15-min): Increased from 1.0 to enforce long-term consistency
    
    # Auxiliary loss weights
    LAMBDA_DIR = 1.0  # Direction classification (focal loss)
    LAMBDA_INTER =1.0  # Interconnection regularization between horizons
    LAMBDA_VOL = 1.0  # Volatility penalty (weak constraint)
    LAMBDA_VAR = 1.0  # Variance NLL (confidence estimation)

    # Outer multipliers for combined loss terms in total= expression.
    # These are separate from the per-component lambda weights above (which scale inside custom_loss),
    # and allow calibration or manual tuning of each grouped term independently.
    LAMBDA_TREND_OUTER = 1.0      # Weight for trend_loss_val in total
    LAMBDA_DIR_OUTER = 1.0        # Weight for total_dir_loss in total
    LAMBDA_DIR_ALIGN_OUTER = 1.0  # Weight for dir_align_loss in total
    LAMBDA_COHERENCE = 1.0        # Weight for coherence_penalty in total
    LAMBDA_NLL_OUTER = 1.0        # Weight for total_nll in total
    LAMBDA_CRPS = 1.0             # CRPS calibration loss (0 = off; try 0.1–1.0)
    LAMBDA_SOFT_ECE = 1.0       # Soft-ECE direction calibration loss (0 = off; try 0.05–0.5)

    # === T_⊥ / QBOX ONTOLOGY CONTROLS ===
    # These implement the perpendicular-tensor (T_⊥) physical model, semantically
    # transposed to time-series trading:
    #   T_⊥  → unexplained variance / hidden order-flow
    #   k(E) → volatility-adaptive kernel size (high vol = short kernel dominates)
    #   Λ_vac → max allowed cross-horizon prediction spread
    #   Casimir → destructive interference between horizons requires high σ
    #   Hyper-decoherence → volatility is a resource; high vol should → high σ
    #   Information flow → each horizon must reveal NEW information
    T_PERP_DIM = 16             # Dimension of perpendicular projection subspace
    LAMBDA_T_PERP = 0.8      # T_⊥ calibration loss (0=off; try 0.5 when enabling)
    LAMBDA_CASIMIR =  0.8       # Casimir inter-scale interference loss (0=off; try 0.5)
    LAMBDA_VAC = 1.0            # Vacuum bandwidth threshold Λ_vac (cross-horizon spread limit)
    LAMBDA_HD =  0.8             # Hyper-decoherence coupling loss (0=off; try 0.3)
    LAMBDA_IFE = 0.8         # Information flow entropy loss (0=off; try 0.3)
    RHO_MAX = 0.95              # Max allowed cross-horizon Pearson correlation
    #   Vacuum saturation: natural + artificial noise fills each t_perp_proj kernel to
    #   E_max capacity at all times.  Excess energy above the ceiling = T_⊥ intensity.
    VACUUM_E_MAX = 1.0          # Per-dim energy ceiling (tanh² max = 1.0; tune 0.5–1.0)
    LAMBDA_VAC_OVERFLOW = 0.5   # Weight for vacuum overflow T_⊥ precision loss

    # Variance calibration bounds (in scaled space)
    VAR_FLOOR = 0.1  # Minimum variance = 0.1 (prevents overconfidence, std ≈ 0.316)
    VAR_CAP = 1e3   # Maximum variance = 10000 (allows high uncertainty)

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
    MA_SPANS = [5, 10, 30]
    MACD_SETTINGS = [
      {'fast': 12, 'slow': 26, 'signal': 9},
      {'fast': 5, 'slow': 35, 'signal': 5},
      {'fast': 8, 'slow': 17, 'signal': 9}
    ]
    RSI_PERIODS = [9, 14, 21]
    BB_PERIODS = [10, 20, 25]

# Activation function settings
    TANH_SCALE = 1.0
    HUBER_DELTA = 1.0
    SIGMOID_SCALE = 1.0

# Training stability controls
    INDICATOR_GRAD_MULT = 5.0    # STE gradient scale in call(); 500 was compensating the broken STE idiom
    GRAD_CLIP_NORM = 20.0        # Applied to NN weights only; indicator grads use dedicated optimizer

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
    DIR_DEADBAND_BPS = 0.0  # 5 bps = 0.05% minimum move for UP classification

    # Stabilize NLL and prevent variance head from dominating early.
    # Variance is in SCALED units^2.
    VAR_FLOOR = 1e-4
    VAR_CAP = 1e3

    # Align direction head with distribution-implied P(up) from (mu, var).
    # Setting this > 0 helps avoid degenerate constant direction probabilities.
    LAMBDA_DIR_ALIGN = 0.7


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
    calibration_pipeline: Optional[Any] = None  # CalibrationPipeline, None if not fitted
    calibration_lambdas: Optional[Dict[str, float]] = None  # Lambdas after pre-training calibration


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
    _calib_lambdas: Optional[Dict[str, float]] = None  # set below if calibrate=True
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

            # ----------------------------------------------------------------
            # Save originals and reset all per-component lambdas to 1.0
            # so that natural magnitudes are measured without existing weights.
            # ----------------------------------------------------------------
            orig_short = custom_model.lambda_short
            orig_point = custom_model.lambda_point
            orig_long  = custom_model.lambda_long
            orig_ext   = custom_model.lambda_extended_trend
            orig_dir   = custom_model.lambda_dir
            orig_var   = custom_model.lambda_var
            orig_vol   = custom_model.lambda_vol
            orig_crps  = custom_model.lambda_crps
            orig_ece   = custom_model.lambda_soft_ece
            orig_t_perp   = custom_model.lambda_t_perp
            orig_casimir  = custom_model.lambda_casimir
            orig_hd       = custom_model.lambda_hd
            orig_ife      = custom_model.lambda_ife

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
            # Phase 1 — BatchNorm warmup (no sampling, no gradient)
            # ----------------------------------------------------------------
            print(f"[calib] Warming up BatchNorm over {n_warmup}/{train_batches} batches ({warmup_frac:.0%} of epoch)...")
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
            new_t_perp  = _rescale(orig_t_perp,  med_t_perp,  d_global) if t_perp_active  else orig_t_perp
            new_casimir = _rescale(orig_casimir, med_casimir, d_global) if casimir_active else orig_casimir
            new_hd      = _rescale(orig_hd,      med_hd,      d_global) if hd_active      else orig_hd
            new_ife     = _rescale(orig_ife,     med_ife,     d_global) if ife_active     else orig_ife

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
            lambda_vac_orig = float(getattr(cfg, 'LAMBDA_VAC', 1.0))
            print(_fmt_row("Λ_vac(thr)", lambda_vac_orig, med_vac, lambda_vac_orig, active=True) + "  (threshold, not rescaled)")
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
            print(f"[calib] Calibration pass failed — proceeding with default lambdas: {e}")
            traceback.print_exc()

    opt = optimizers.Adam(learning_rate=cfg.LR)
    custom_model.compile(optimizer=opt)

    csv_logger = callbacks.CSVLogger("training_log.csv", append=True)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=Config.EARLY, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(cfg.MODEL_PATH, save_best_only=True, monitor='val_loss', save_weights_only=True)
    # MCC-based early stopping for direction head (class-imbalance robust)
    # MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # Range: [-1, 1], where 1 = perfect, 0 = random, -1 = inverse
    # Unlike accuracy, MCC is balanced even with severe class imbalance
    es_dir = callbacks.EarlyStopping(
        monitor='val_dir_mcc_h1',
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

    # Post-hoc calibration pipeline (temperature scaling + conformal intervals)
    cal_pipeline = None
    if fit_calibration and _CalibrationPipeline is not None:
        try:
            print("\nFitting CalibrationPipeline on test split...")
            _draft = TrainResult(
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
            cal_pipeline = _CalibrationPipeline()
            cal_pipeline.fit(_draft)
        except Exception as _cal_err:
            print(f"CalibrationPipeline fit skipped: {_cal_err}")
            cal_pipeline = None

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
                                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            self.alpha_vars_ma.append(v)
            self.all_logit_vars.append(v)

        for i, settings in enumerate(self.config.MACD_SETTINGS):
            v_fast = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['fast'])),
                trainable=True,
                name=f'macd_{i}_fast',
                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            v_slow = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['slow'])),
                trainable=True,
                name=f'macd_{i}_slow',
                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            v_signal = self.add_weight(
                shape=(),
                initializer=initializers.Constant(self._logit_from_period(settings['signal'])),
                trainable=True,
                name=f'macd_{i}_signal',
                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            self.macd_alpha_vars[f'macd_{i}_fast'] = v_fast
            self.macd_alpha_vars[f'macd_{i}_slow'] = v_slow
            self.macd_alpha_vars[f'macd_{i}_signal'] = v_signal
            self.all_logit_vars.extend([v_fast, v_slow, v_signal])

        for i, p in enumerate(self.config.RSI_PERIODS):
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(self._logit_from_period(p)),
                                trainable=True,
                                name=f'rsi_alpha_{i}',
                                regularizer=regularizers.L2(self.config.INDICATOR_L2))
            self.rsi_alpha_vars.append(v)
            self.all_logit_vars.append(v)

        for i, p in enumerate(self.config.BB_PERIODS):
            v = self.add_weight(shape=(),
                                initializer=initializers.Constant(self._logit_from_period(p)),
                                trainable=True,
                                name=f'bb_alpha_{i}',
                                regularizer=regularizers.L2(self.config.INDICATOR_L2))
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
            parallel_iterations=10
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
            # Correct STE gradient trick: forward=logit (unchanged), backward=logit * grad_multiplier
            # Old (broken): logit + stop_grad(logit)*(k-1) => forward=logit*k (saturates sigmoid!)
            # New (correct): k*logit - stop_grad((k-1)*logit) => forward=logit, backward=k
            logit_for_alpha = (self.grad_multiplier * logit
                               - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
            adjusted_logit = logit_for_alpha + meta_adjust[:, idx] * self.meta_scale
            alpha = self._alpha_from_logit(adjusted_logit)
            ema_seq = self.ewma_seq(x, alpha)
            features.append(ema_seq)
            idx += 1

        for i in range(len(self.config.MACD_SETTINGS)):
            fast_var = self.macd_alpha_vars[f'macd_{i}_fast']
            slow_var = self.macd_alpha_vars[f'macd_{i}_slow']
            sig_var = self.macd_alpha_vars[f'macd_{i}_signal']
            # Correct STE gradient trick (see MA block above for explanation)
            fast_for_alpha = (self.grad_multiplier * fast_var
                              - tf.stop_gradient((self.grad_multiplier - 1.0) * fast_var))
            slow_for_alpha = (self.grad_multiplier * slow_var
                              - tf.stop_gradient((self.grad_multiplier - 1.0) * slow_var))
            sig_for_alpha  = (self.grad_multiplier * sig_var
                              - tf.stop_gradient((self.grad_multiplier - 1.0) * sig_var))
            fast_logit = fast_for_alpha + meta_adjust[:, idx] * self.meta_scale
            slow_logit = slow_for_alpha + meta_adjust[:, idx+1] * self.meta_scale
            sig_logit  = sig_for_alpha  + meta_adjust[:, idx+2] * self.meta_scale
            fast = self._alpha_from_logit(fast_logit)
            slow = self._alpha_from_logit(slow_logit)
            sig = self._alpha_from_logit(sig_logit)
            ema_f = self.ewma_seq(x, fast)
            ema_s = self.ewma_seq(x, slow)
            macd_line = ema_f - ema_s
            macd_sig = self.ewma_seq(macd_line, sig)
            macd_hist = macd_line - macd_sig
            features.extend([macd_line, macd_sig, macd_hist])
            # Soft sign: tf.sign has zero gradient; tanh*10 is a differentiable approximation
            macd_cross = tf.tanh(macd_hist * 10.0)
            features.append(macd_cross)
            idx += 3

        diffs = x[:, 1:] - x[:, :-1]
        gains = tf.where(diffs > 0, diffs, tf.zeros_like(diffs))
        losses = tf.where(diffs < 0, -diffs, tf.zeros_like(diffs))
        gains_padded = tf.concat([tf.zeros((tf.shape(gains)[0], 1), dtype=gains.dtype), gains], axis=1)
        losses_padded = tf.concat([tf.zeros((tf.shape(losses)[0], 1), dtype=losses.dtype), losses], axis=1)

        for logit in self.rsi_alpha_vars:
            # Correct STE gradient trick (see MA block above for explanation)
            logit_for_alpha = (self.grad_multiplier * logit
                               - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
            adjusted_logit = logit_for_alpha + meta_adjust[:, idx] * self.meta_scale
            rsi_alpha = self._alpha_from_logit(adjusted_logit)
            gains_ema = self.ewma_seq(gains_padded, rsi_alpha)
            losses_ema = self.ewma_seq(losses_padded, rsi_alpha)
            rs = gains_ema / (losses_ema + 1e-8)
            rsi_seq = 100.0 - (100.0 / (1.0 + rs))
            features.append(rsi_seq)
            idx += 1

        for logit in self.bb_alpha_vars:
            # Correct STE gradient trick (see MA block above for explanation)
            logit_for_alpha = (self.grad_multiplier * logit
                               - tf.stop_gradient((self.grad_multiplier - 1.0) * logit))
            adjusted_logit = logit_for_alpha + meta_adjust[:, idx] * self.meta_scale
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
        memory = layers.Dropout(0.8)(memory)

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

        # === ENERGY GATE — k(E) adaptive kernel weighting ===
        # High local volatility (energy) → short kernel dominates (fine-grained view).
        # Low volatility → long kernel dominates (coarse trend view).
        # Implements: x = Σ_k gate_k(E) · x_k   (energy-weighted sum, not concat)
        # This is the semantic transpose of k(E) from QBOX: the convolution kernel size
        # adapts to local market energy rather than being fixed.
        _inp_resh_2d = layers.Reshape((self.config.LOOKBACK, 1))(inp)
        _local_mean = layers.GlobalAveragePooling1D()(_inp_resh_2d)           # [B, 1]
        _inp_center = layers.Subtract()([
            layers.Reshape((self.config.LOOKBACK, 1))(inp),
            layers.RepeatVector(self.config.LOOKBACK)(
                layers.Reshape((1,))(_local_mean))
        ])
        _local_var = layers.GlobalAveragePooling1D()(
            layers.Lambda(lambda t: tf.square(t))(_inp_center)
        )  # [B, 1]
        _local_max = layers.GlobalMaxPooling1D()(_inp_resh_2d)                # [B, 1]
        _energy_feats = layers.Concatenate()([_local_var, _local_max])        # [B, 2]
        _energy_gate = layers.Dense(3, activation='softmax',
                                    name='energy_gate')(_energy_feats)        # [B, 3]

        # Expand gate to [B, 1, 1] for broadcast multiply with [B, LOOKBACK, 16]
        _gate_s = layers.Lambda(
            lambda g: tf.expand_dims(tf.expand_dims(g[:, 0], 1), 2))(_energy_gate)
        _gate_m = layers.Lambda(
            lambda g: tf.expand_dims(tf.expand_dims(g[:, 1], 1), 2))(_energy_gate)
        _gate_l = layers.Lambda(
            lambda g: tf.expand_dims(tf.expand_dims(g[:, 2], 1), 2))(_energy_gate)

        x_short_g = layers.Multiply()([x_short, _gate_s])
        x_med_g   = layers.Multiply()([x_med,   _gate_m])
        x_long_g  = layers.Multiply()([x_long,  _gate_l])
        x = layers.Add()([x_short_g, x_med_g, x_long_g])        # [B, LOOKBACK, 16]
        x = layers.LayerNormalization()(x)

        # Positional encoding
        x = layers.Add()([x, PositionalEncodingLayer()(x)])

        # Transformer-style blocks (reduced to 2 for speed)
        for _ in range(2):
            att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.8)(x, x)
            x = layers.Add()([x, att])
            x = layers.LayerNormalization()(x)
            ff = layers.Dense(32, activation='gelu')(x)
            ff = layers.Dropout(0.8)(ff)
            ff = layers.Dense(x.shape[-1])(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)

        # Global context vector
        context = layers.GlobalAveragePooling1D()(x)

        # === T_⊥ PERPENDICULAR PROJECTION ===
        # T_⊥ encodes the energy/information that escaped the observable projection.
        # In trading: unexplained residual variance = hidden order-flow / regime change.
        # perp_magnitude → conditions ALL variance heads: high T_⊥ → high predicted σ.
        # This prevents variance heads from driving uncertainty to zero when T_⊥ is large.
        _t_perp_dim = int(getattr(self.config, 'T_PERP_DIM', 16))
        h_perp = layers.Dense(_t_perp_dim, activation='tanh',
                               name='t_perp_proj')(context)               # [B, T_PERP_DIM]

        # === VACUUM SATURATION ===
        # Fill each kernel dimension to VACUUM_E_MAX with calibrated Gaussian noise.
        # natural_noise + artificial_noise = E_max at all times during training.
        # training=False: pure pass-through (deterministic inference).
        _e_max = float(getattr(self.config, 'VACUUM_E_MAX', 1.0))
        h_perp_sat = VacuumSaturationNoise(
            e_max=_e_max, name='vacuum_saturation')(h_perp)               # [B, T_PERP_DIM]

        # Overflow = per-sample mean energy above E_max in the saturated subspace.
        # This is the observable T_⊥ intensity: energy that could not be absorbed by
        # the vacuum kernels — proportional to unexplained prediction residual.
        vacuum_overflow = layers.Lambda(
            lambda h: tf.nn.relu(
                tf.reduce_mean(tf.square(h), axis=1, keepdims=True)
                - tf.constant(_e_max, dtype=tf.float32)
            ),
            name='vacuum_overflow'
        )(h_perp_sat)                                                     # [B, 1]

        perp_magnitude = layers.Dense(1, activation='softplus',
                                      name='t_perp_magnitude')(h_perp_sat)  # [B, 1]

        # === REGIME GATE (White-hole / T_⊥^up detector) ===
        # Regime gate ≈ 1.0 when market is in a "white hole" state:
        #   new information is flowing IN from outside (regime breaks, flash crashes,
        #   macro news shocks), making the current visible projection insufficient.
        # Regime gate ≈ 0.0 = "black hole" state: coherent trend, info is observable.
        # Computed from local price std (volatility level) fused with global context.
        _inp_for_gate = layers.Reshape((self.config.LOOKBACK, 1))(inp)
        _gate_vol = layers.GlobalAveragePooling1D()(
            layers.Lambda(lambda t: tf.abs(t - tf.reduce_mean(t, axis=1, keepdims=True)))(
                _inp_for_gate)
        )                                                                    # [B, 1]
        regime_gate = layers.Dense(1, activation='sigmoid',
                                   name='regime_gate')(
            layers.Concatenate()([_gate_vol, context]))                  # [B, 1]

        # Sequence summary for regression
        seq_flat = layers.Flatten()(x)

        # Shared dense layer for all output heads
        shared_dense = layers.Dense(32, activation='gelu',
                                   kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(seq_flat)
        shared_dense = layers.Concatenate()([shared_dense, context])

        # === THREE INDEPENDENT OUTPUT TOWERS (h0, h1, h2) ===
        # Each horizon has its own price, direction, and confidence (variance) head
        
        # Variance bias initialization: softplus(x) ≈ x for x > 0
        # Initialize bias so initial variance ≈ 1.3 (higher than unit variance for calibration learning)
        # softplus(0) ≈ 0.693, softplus(0.5) ≈ 0.97, softplus(1.0) ≈ 1.31
        var_bias_init = tf.keras.initializers.Constant(1.0)  # Initial variance ≈ 1.31
        
        # Direction bias initialization: sigmoid(0) = 0.5 (unbiased)
        # Keep at 0 for balanced initial predictions
        dir_bias_init = tf.keras.initializers.Zeros()
        
        # ---- TOWER 0 (1-minute horizon) ----
        tower_h0 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h0 = layers.Dense(1, name='price_h0')(tower_h0)
        direction_h0 = layers.Dense(1, activation='sigmoid', name='direction_h0',
                                   bias_initializer=dir_bias_init)(tower_h0)
        # Variance head conditioned on T_⊥ and regime gate:
        #   high perp_magnitude → more energy in hidden dims → higher σ²
        #   high regime_gate → white-hole / regime-break → higher σ²
        tower_h0_var_input = layers.Concatenate()([tower_h0, perp_magnitude, regime_gate])
        variance_h0 = layers.Dense(1, activation='softplus', name='variance_h0',
                                  bias_initializer=var_bias_init)(tower_h0_var_input)

        # ---- TOWER 1 (5-minute horizon - PRIMARY) ----
        tower_h1 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h1 = layers.Dense(1, name='price_h1')(tower_h1)
        direction_h1 = layers.Dense(1, activation='sigmoid', name='direction_h1',
                                   bias_initializer=dir_bias_init)(tower_h1)
        tower_h1_var_input = layers.Concatenate()([tower_h1, perp_magnitude, regime_gate])
        variance_h1 = layers.Dense(1, activation='softplus', name='variance_h1',
                                  bias_initializer=var_bias_init)(tower_h1_var_input)

        # ---- TOWER 2 (15-minute horizon) ----
        tower_h2 = layers.Dense(16, activation='gelu',
                               kernel_regularizer=regularizers.L2(self.config.REG_MOMENTUM_L2))(shared_dense)
        price_h2 = layers.Dense(1, name='price_h2')(tower_h2)
        direction_h2 = layers.Dense(1, activation='sigmoid', name='direction_h2',
                                   bias_initializer=dir_bias_init)(tower_h2)
        tower_h2_var_input = layers.Concatenate()([tower_h2, perp_magnitude, regime_gate])
        variance_h2 = layers.Dense(1, activation='softplus', name='variance_h2',
                                  bias_initializer=var_bias_init)(tower_h2_var_input)

        # === FINAL MODEL: 10 outputs (3 horizons × 3 heads + vacuum_overflow) ===
        # Output index layout:
        #   0: price_h0    1: direction_h0    2: variance_h0
        #   3: price_h1    4: direction_h1    5: variance_h1
        #   6: price_h2    7: direction_h2    8: variance_h2
        #   9: vacuum_overflow  [B, 1]  (T_⊥ overflow intensity; 0 at inference)
        return models.Model(
            inputs=inp,
            outputs=[
                price_h0, direction_h0, variance_h0,
                price_h1, direction_h1, variance_h1,
                price_h2, direction_h2, variance_h2,
                vacuum_overflow,
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


class VacuumSaturationNoise(layers.Layer):
    """Vacuum Saturation Noise layer for the T_⊥ perpendicular subspace.

    Maintains maximum vacuum kernel energy density at all times:
        natural_noise + artificial_noise = VACUUM_E_MAX (per dimension)

    During training:
    1. Measure actual per-dimension energy:  energy[d] = mean(h²[:, d])   [T_PERP_DIM]
    2. Compute deficit:                      deficit[d] = relu(E_max - energy[d])
    3. Inject calibrated Gaussian noise:     noise ~ N(0, stop_grad(sqrt(deficit + ε)))
    4. Return h + noise  (saturated subspace)

    The noise is stop-gradient w.r.t. the deficit measurement so the network learns to
    fill the vacuum with real signal rather than chasing the artificial noise level.

    During inference (training=False): pass-through (deterministic predictions).

    The per-sample mean energy above E_max (computed downstream as a Lambda layer)
    is the observable T_⊥ overflow intensity — proportional to the fraction of the
    prediction residual that lives in the hidden perpendicular subspace.
    """

    def __init__(self, e_max=1.0, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.e_max = float(e_max)
        self.eps   = float(eps)

    def call(self, h_perp, training=None):
        if not training:
            return h_perp

        # h_perp: [B, T_PERP_DIM], values in (-1, 1) due to upstream tanh
        h = tf.cast(h_perp, tf.float32)

        # Batch-level per-dimension energy measurement
        energy_per_dim = tf.reduce_mean(tf.square(h), axis=0)          # [T_PERP_DIM]

        # Deficit = how much energy is missing to reach E_max per dim
        deficit = tf.nn.relu(
            tf.constant(self.e_max, dtype=tf.float32) - energy_per_dim
        )                                                                # [T_PERP_DIM]

        # Stop-gradient: noise level is treated as a constant forcing signal,
        # not a target the network learns to game by minimising deficit.
        noise_std = tf.stop_gradient(
            tf.sqrt(deficit + tf.constant(self.eps, dtype=tf.float32))
        )                                                                # [T_PERP_DIM]

        noise = tf.random.normal(shape=tf.shape(h), dtype=tf.float32)  # [B, T_PERP_DIM]
        return h + noise * noise_std                                    # broadcast over B

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'e_max': self.e_max, 'eps': self.eps})
        return cfg


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
        self.lambda_trend_outer = float(getattr(config, 'LAMBDA_TREND_OUTER', 0.5))
        self.lambda_dir_outer = float(getattr(config, 'LAMBDA_DIR_OUTER', 0.5))
        self.lambda_dir_align_outer = float(getattr(config, 'LAMBDA_DIR_ALIGN_OUTER', 0.5))
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
    def custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends,
                    vacuum_overflow=None):
        """Delegate to centralized implementation in `losses.py`."""
        return _losses.custom_loss(self, x_window, y_true, y_pred, last_close, extended_trends,
                                   vacuum_overflow=vacuum_overflow)

    def train_step(self, data):
        x_window, y_true, last_close, extended_trends = data
        with tf.GradientTape() as tape:
            y_pred = self(x_window, training=True)
            # Unpack 10th output (vacuum overflow) before passing to custom_loss
            (*y_pred_9, vac_overflow_pred) = y_pred
            loss_components = self.custom_loss(x_window, y_true, y_pred_9, last_close,
                                               extended_trends,
                                               vacuum_overflow=vac_overflow_pred)

        # Unpack 34-component tuple from custom_loss
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

        # Split gradients into NN weights vs. indicator logit vars.
        # They use separate Adam optimizers because Adam normalizes gradient magnitude —
        # scaling grads before apply cancels out. Higher LR on indicator optimizer is the
        # correct way to give indicator params a larger effective step size.
        _is_ind = lambda name: any(k in name for k in
            ('alpha_ma', 'macd_', 'rsi_alpha', 'bb_alpha', 'pair_', 'momentum_raw'))

        nn_gvs, ind_gvs = [], []
        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue
            (ind_gvs if _is_ind(v.name.lower()) else nn_gvs).append((g, v))

        # Clip NN grads by global norm only (indicator grads are small scalars; Adam handles scale)
        if getattr(self.config, 'GRAD_CLIP_NORM', 0.0) and self.config.GRAD_CLIP_NORM > 0.0:
            nn_gs_clipped, _ = tf.clip_by_global_norm(
                [g for g, v in nn_gvs], self.config.GRAD_CLIP_NORM)
            nn_gvs = list(zip(nn_gs_clipped, [v for g, v in nn_gvs]))

        # Apply gradients with separate optimizers
        self.optimizer.apply_gradients(nn_gvs)
        self.indicator_optimizer.apply_gradients(ind_gvs)

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
        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred_9
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
        crps_total = crps_h0 + crps_h1 + crps_h2
        soft_ece_total = soft_ece_h0 + soft_ece_h1 + soft_ece_h2

        # PIT uniformity (KS statistic) — computed in numpy from the batch tensors.
        # Smaller KS → more uniform PIT → better calibrated variance head.
        pit_ks_h0 = tf.constant(float('nan'), dtype=tf.float32)
        pit_ks_h1 = tf.constant(float('nan'), dtype=tf.float32)
        pit_ks_h2 = tf.constant(float('nan'), dtype=tf.float32)
        if pit_uniformity is not None:
            try:
                sigma_h0_np = tf.sqrt(tf.clip_by_value(tf.squeeze(y_pred_9[2], axis=1),
                                                        float(getattr(self.config, 'VAR_FLOOR', 1e-4)),
                                                        float(getattr(self.config, 'VAR_CAP', 1e3)))).numpy()
                sigma_h1_np = tf.sqrt(tf.clip_by_value(tf.squeeze(y_pred_9[5], axis=1),
                                                        float(getattr(self.config, 'VAR_FLOOR', 1e-4)),
                                                        float(getattr(self.config, 'VAR_CAP', 1e3)))).numpy()
                sigma_h2_np = tf.sqrt(tf.clip_by_value(tf.squeeze(y_pred_9[8], axis=1),
                                                        float(getattr(self.config, 'VAR_FLOOR', 1e-4)),
                                                        float(getattr(self.config, 'VAR_CAP', 1e3)))).numpy()
                y_true_h0_np = y_true[:, 0].numpy()
                y_true_h1_np = y_true[:, 1].numpy()
                y_true_h2_np = y_true[:, 2].numpy()
                mu_h0_np = tf.squeeze(y_pred_9[0], axis=1).numpy()
                mu_h1_np = tf.squeeze(y_pred_9[3], axis=1).numpy()
                mu_h2_np = tf.squeeze(y_pred_9[6], axis=1).numpy()
                pit_ks_h0 = tf.constant(pit_uniformity(y_true_h0_np, mu_h0_np, sigma_h0_np), dtype=tf.float32)
                pit_ks_h1 = tf.constant(pit_uniformity(y_true_h1_np, mu_h1_np, sigma_h1_np), dtype=tf.float32)
                pit_ks_h2 = tf.constant(pit_uniformity(y_true_h2_np, mu_h2_np, sigma_h2_np), dtype=tf.float32)
            except Exception:
                pass

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
            "crps_loss": crps_total,
            "crps_h0": crps_h0,
            "crps_h1": crps_h1,
            "crps_h2": crps_h2,
            "soft_ece_loss": soft_ece_total,
            "soft_ece_h0": soft_ece_h0,
            "soft_ece_h1": soft_ece_h1,
            "soft_ece_h2": soft_ece_h2,
            "pit_ks_h0": pit_ks_h0,
            "pit_ks_h1": pit_ks_h1,
            "pit_ks_h2": pit_ks_h2,
            "reg_loss": reg_val,
            "inter_reg": inter_reg,
            "vol_loss": vol_loss,
            # === T_⊥ / QBOX metrics ===
            "t_perp_loss": t_perp_total,
            "casimir_loss": casimir_val,
            "vac_loss": vac_val,
            "hd_loss": hd_val,
            "ife_loss": ife_val,
            "vac_overflow_loss": vac_overflow_val,
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
        # Unpack 10th output (vacuum overflow; near-zero at inference)
        (*y_pred_9, vac_overflow_pred) = y_pred
        loss_components = self.custom_loss(x_window, y_true, y_pred_9, last_close,
                                           extended_trends,
                                           vacuum_overflow=vac_overflow_pred)

        # Unpack 34-component tuple
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

        price_h0, dir_pred_h0, var_h0, price_h1, dir_pred_h1, var_h1, price_h2, dir_pred_h2, var_h2 = y_pred_9
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
        crps_total = crps_h0 + crps_h1 + crps_h2
        soft_ece_total = soft_ece_h0 + soft_ece_h1 + soft_ece_h2

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
            "crps_loss": crps_total,
            "crps_h0": crps_h0,
            "crps_h1": crps_h1,
            "crps_h2": crps_h2,
            "soft_ece_loss": soft_ece_total,
            "soft_ece_h0": soft_ece_h0,
            "soft_ece_h1": soft_ece_h1,
            "soft_ece_h2": soft_ece_h2,
            "reg_loss": reg_val,
            "inter_reg": inter_reg,
            "vol_loss": vol_loss,
            # === T_⊥ / QBOX metrics ===
            "t_perp_loss": t_perp_total,
            "casimir_loss": casimir_val,
            "vac_loss": vac_val,
            "hd_loss": hd_val,
            "ife_loss": ife_val,
            "vac_overflow_loss": vac_overflow_val,
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
        convergence_score = max(0.0, min(1.0, 1.0 - (current_mean / 3.0)))

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
            df = pd.DataFrame(self.rows)
            df.to_csv(self.out_csv, index=False)

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
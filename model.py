csv="Bitcoin_BTCUSDT.csv"

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, losses, initializers, regularizers
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import time
from tqdm import tqdm

# -----------------------------
class Config:

    HOUR = 60
    DAY = HOUR * 24

    # Data Configuration tuned for minute-level data
    CSV_PATH = csv
    LOOKBACK = HOUR   # Reduced to 1 hour of minute data
    WINDOW_STEP = 1  # Generate a training sample every minute for true minute-level modeling
    RESAMPLE_MINUTES = None  # Optionally aggregate to coarser bars (e.g., set to 5 for 5-minute bars)
    BATCH_SIZE = 144
    EPOCHS = 40
    LR = 1e-3  # Fixed from critically low 1e-10; reasonable for Adam optimizer
    PATIENCE = EPOCHS
    MAX_SEQUENCE_COUNT = 144 * 20  # Limit most recent sequences to bound training size



    # Use integer periods to avoid float indexing issues
    # Extended trend features are computed as percent-change over these lags (in minutes)
    EXTENDED_TREND_PERIODS = [1, 5, 15]  # 1m, 5m, 15m

    # Supervision horizons (in minutes ahead from last_close). These define the 3 output towers.
    # h0=1m, h1=5m, h2=15m.
    HORIZON_STEPS = [1, 5, 15]

# Loss Function Weights
    DAMPING = 1
    LAMBDA_POINT = 1  
    LAMBDA_LOCAL_TREND  = 1
    LAMBDA_GLOBAL_TREND =  1
    LAMBDA_EXTENDED_TREND = 0.1
    LAMBDA_QUANTILE = 1
    REG_MOMENTUM_L2 = 1e-4
    MOMENTUM_CLIP_MIN = 1.0
    MOMENTUM_CLIP_MAX = LOOKBACK
    USE_HUBER = True
    LAMBDA_DIR = 1  # New for direction loss
    LAMBDA_INTER = 1  # New for interconnection reg
    LAMBDA_VOL = 1  # New for volatility penalty
    LAMBDA_SHORT = 1  # For multi-horizon short-term
    LAMBDA_LONG = 1  # For multi-horizon long-term
    LAMBDA_VAR = 1  # For variance NLL

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
    INDICATOR_GRAD_MULT = 20.0
    GRAD_CLIP_NORM = 5.0

# Focal loss hyperparameters for direction classification
    FOCAL_ALPHA = 0.7  # Class weight: favor minority class (DOWN)
    FOCAL_GAMMA = 1.0  # Focus parameter: higher = focus more on hard examples

    # Trade-aware direction labeling deadband.
    # If > 0, direction loss/metrics treat returns within +/- deadband as neutral.
    # Units: basis points (bps). Example: 10 bps = 0.10%.
    DIR_DEADBAND_BPS = 0.0

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

    def clean_numeric(self, series):
        return series.astype(str).str.replace(r'[\$,]', '', regex=True).replace('', np.nan).astype(float)

    def load_and_prepare_data(self):
        """Load and prep minute-level Bitcoin data with optional resampling."""
        df = pd.read_csv(self.config.CSV_PATH)

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

        return (X_train_seq_scaled, y_train_scaled, last_close_train, extended_trends_train,
                X_test_seq_scaled, y_test_scaled, last_close_test, extended_trends_test,
                y_train, y_test, target_scaler)

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
        # Here alpha is the weight for the DOWN class (label=0), and (1-alpha) weights UP (label=1).
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
        # Delta-target: compare scaled deltas directly (pivot at 0).
        point_loss_h0_val = self.lambda_short * self.point_huber(y_true_h0, price_h0)
        point_loss_h1_val = self.lambda_point * self.point_huber(y_true_h1, price_h1)
        point_loss_h2_val = self.lambda_long * self.point_huber(y_true_h2, price_h2)
        point_loss_val = point_loss_h0_val + point_loss_h1_val + point_loss_h2_val

        # === TREND LOSSES ===
        # Delta-target training: implement delta-consistent trend losses.
        # Input-based prior: align predicted deltas with extended_trends (pct_change * last_close)
        trend_prior_h0 = tf.reduce_mean(tf.square(y_true_raw_h0 - extended_trends[:, 0] * last_close_squeeze))
        trend_prior_h1 = tf.reduce_mean(tf.square(y_true_raw_h1 - extended_trends[:, 1] * last_close_squeeze))
        trend_prior_h2 = tf.reduce_mean(tf.square(y_true_raw_h2 - extended_trends[:, 2] * last_close_squeeze))
        
        # Cross-horizon coherence: penalize inconsistent curvature (simple version: penalize if signs don't align monotonically)
        sign_h0 = tf.sign(y_true_raw_h0)
        sign_h1 = tf.sign(y_true_raw_h1)
        sign_h2 = tf.sign(y_true_raw_h2)
        # Penalize if h1 doesn't match the trend from h0 to h2
        coherence_penalty = tf.reduce_mean(tf.cast(tf.logical_xor(sign_h1 == sign_h0, sign_h1 == sign_h2), tf.float32))
        
        local_trend_h0 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h0 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h0 = self.lambda_extended_trend * trend_prior_h0
        local_trend_h1 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h1 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h1 = self.lambda_extended_trend * trend_prior_h1
        local_trend_h2 = tf.constant(0.0, dtype=tf.float32)
        global_trend_h2 = tf.constant(0.0, dtype=tf.float32)
        extended_trend_h2 = self.lambda_extended_trend * trend_prior_h2
        
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
        total = point_loss_val + trend_loss_val + total_dir_loss + dir_align_loss + reg_loss + inter_reg + vol_loss + total_nll

        # === RETURN 29-COMPONENT TUPLE for comprehensive logging ===
        # Format: (total, 
        #   point_h0, point_h1, point_h2,
        #   local_h0, global_h0, extended_h0,
        #   local_h1, global_h1, extended_h1,
        #   local_h2, global_h2, extended_h2,
        #   dir_h0, dir_h1, dir_h2,
        #   nll_h0, nll_h1, nll_h2,
        #   reg_loss, inter_reg, vol_loss)
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

        ret_h0 = (y_true_raw[:, 0] - last_close_squeeze) / (last_close_squeeze + self.eps)
        ret_h1 = (y_true_raw[:, 1] - last_close_squeeze) / (last_close_squeeze + self.eps)
        ret_h2 = (y_true_raw[:, 2] - last_close_squeeze) / (last_close_squeeze + self.eps)

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
        last_close_scaled = self._to_scaled(last_close_squeeze)
        gauss_p_up_h0 = self._normal_cdf((mu_h0 - last_close_scaled) / (tf.sqrt(var_h0_c) + self.eps))
        gauss_p_up_h1 = self._normal_cdf((mu_h1 - last_close_scaled) / (tf.sqrt(var_h1_c) + self.eps))
        gauss_p_up_h2 = self._normal_cdf((mu_h2 - last_close_scaled) / (tf.sqrt(var_h2_c) + self.eps))

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
    tf.keras.utils.set_random_seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print("Starting enhanced model training with extended trend features...")
    config = Config()
    data_processor = DataProcessor(config)
    df, close_values = data_processor.load_and_prepare_data()

    (X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
     X_test_seq, y_test_scaled, last_close_test, extended_trends_test,
     y_train, y_test, target_scaler) = data_processor.prepare_datasets(df, close_values)

    # Calculate training statistics for time estimation
    train_batches = math.ceil(X_train_seq.shape[0] / config.BATCH_SIZE)
    actual_epochs = int(epochs) if epochs is not None else int(config.EPOCHS)

    # Estimate training time based on hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        time_per_batch = 0.15  # seconds per batch on GPU (conservative estimate)
        device_type = "GPU"
    else:
        time_per_batch = 2.0   # seconds per batch on CPU (conservative estimate)
        device_type = "CPU"

    estimated_time_per_epoch = train_batches * time_per_batch
    estimated_total_time = estimated_time_per_epoch * actual_epochs

    print(f"\n⏱️  Training Time Estimation ({device_type}):")
    print(f"   Batches per epoch: {train_batches}")
    print(f"   Time per batch: ~{time_per_batch:.2f}s")
    print(f"   Time per epoch: ~{estimated_time_per_epoch/60:.1f} minutes")
    print(f"   Total time ({actual_epochs} epochs): ~{estimated_total_time/3600:.1f} hours")

    predictor = PricePredictor(config)
    base_model = predictor.build_model()
    pred_scale = np.std(y_train) if np.std(y_train) > 0 else 1.0
    pred_mean = np.mean(y_train)
    custom_model = CustomTrainModel(
        base_model=base_model,
        pred_scale=pred_scale,
        pred_mean=pred_mean,
        lambda_point=config.LAMBDA_POINT,
        lambda_local_trend=config.LAMBDA_LOCAL_TREND,
        lambda_global_trend=config.LAMBDA_GLOBAL_TREND,
        lambda_extended_trend=config.LAMBDA_EXTENDED_TREND,
        lambda_dir=config.LAMBDA_DIR,
        config=config,
        inputs=base_model.inputs,
        outputs=base_model.outputs
    )

    # Create datasets before compilation so we can calibrate lambdas on multiple batches
    predictor = PricePredictor(config)
    train_ds, val_ds = predictor.create_datasets(
        X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
        X_test_seq, y_test_scaled, last_close_test, extended_trends_test
    )

    if calibrate == True:
        # Robust multi-batch median calibration to scale all lambdas relative to their loss magnitudes
        try:
            # Save original lambdas
            orig_lambda_point = custom_model.lambda_point
            orig_lambda_local = custom_model.lambda_local_trend
            orig_lambda_global = custom_model.lambda_global_trend
            orig_lambda_ext = custom_model.lambda_extended_trend
            orig_lambda_dir = custom_model.lambda_dir
            orig_lambda_var = custom_model.lambda_var

            # Temporarily set all lambdas to 1.0 for unweighted loss computation
            custom_model.lambda_point = 1.0
            custom_model.lambda_local_trend = 1.0
            custom_model.lambda_global_trend = 1.0
            custom_model.lambda_extended_trend = 1.0
            custom_model.lambda_dir = 1.0
            custom_model.lambda_var = 1.0

            n_calib_batches = 256  # Number of batches for robust sampling

            # Warm-up pass to update BatchNorm statistics
            print("Warming up BatchNorm statistics for accurate calibration...")
            for batch in train_ds.take(n_calib_batches):
                x_batch, _, _, _ = batch
                _ = custom_model(x_batch, training=True)  # Forward pass to update BN running stats

            point_losses, local_losses, global_losses, ext_losses, dir_losses, var_losses = [], [], [], [], [], []

            for i, batch in enumerate(train_ds.take(n_calib_batches)):
                x_batch, y_batch, last_batch, ext_batch = batch
                y_pred_batch = custom_model(x_batch, training=False)
                # === Updated unpacking for 22-component tuple (per-horizon losses) ===
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
                # Aggregate across horizons
                point_val = point_h0 + point_h1 + point_h2
                local_val = local_h0 + local_h1 + local_h2
                global_val = global_h0 + global_h1 + global_h2
                ext_val = ext_h0 + ext_h1 + ext_h2
                dir_val = dir_h0 + dir_h1 + dir_h2
                nll_val = nll_h0 + nll_h1 + nll_h2
                
                point_losses.append(float(point_val))
                local_losses.append(float(local_val))
                global_losses.append(float(global_val))
                ext_losses.append(float(ext_val))
                dir_losses.append(float(dir_val))
                var_losses.append(float(nll_val))

            # Use median to reduce sensitivity to outliers
            med_point = float(np.median(np.array(point_losses))) if point_losses else 0.0
            med_local = float(np.median(np.array(local_losses))) if local_losses else 0.0
            med_global = float(np.median(np.array(global_losses))) if global_losses else 0.0
            med_ext = float(np.median(np.array(ext_losses))) if ext_losses else 0.0
            med_dir = float(np.median(np.array(dir_losses))) if dir_losses else 0.0
            med_var = float(np.median(np.array(var_losses))) if var_losses else 0.0

            # Compute reference loss as the mean of non-zero median losses
            non_zero_medians = [m for m in [med_point, med_local, med_global, med_ext, med_dir, med_var] if m > 1e-8]
            ref_loss = float(np.mean(non_zero_medians)) if non_zero_medians else 1.0

            eps = 1e-8
            damping = Config.DAMPING  # Square root adjustment to avoid overreaction

            # Adjust all lambdas to balance contributions relative to reference loss
            new_point = orig_lambda_point * (ref_loss / (med_point + eps)) ** damping if med_point > eps else orig_lambda_point
            new_local = orig_lambda_local * (ref_loss / (med_local + eps)) ** damping if med_local > eps else orig_lambda_local
            new_global = orig_lambda_global * (ref_loss / (med_global + eps)) ** damping if med_global > eps else orig_lambda_global
            new_ext = orig_lambda_ext * (ref_loss / (med_ext + eps)) ** damping if med_ext > eps else orig_lambda_ext
            new_dir = orig_lambda_dir * (ref_loss / (med_dir + eps)) ** damping if med_dir > eps else orig_lambda_dir
            new_var = orig_lambda_var * (ref_loss / (med_var + eps)) ** damping if med_var > eps else orig_lambda_var

            # Clamp lambdas for stability
            min_lambda = 1e-3
            max_lambda = 1000.0
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


    opt = optimizers.Adam(learning_rate=config.LR)
    custom_model.compile(optimizer=opt)
    csv_logger = callbacks.CSVLogger("training_log.csv", append=True)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=config.PATIENCE, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(config.MODEL_PATH, save_best_only=True, monitor='val_loss', save_weights_only=True)
    # === STEP 7: Add per-horizon MCC-based early stopping (primary horizon h1) ===
    # Monitor the Matthews Correlation Coefficient for the primary 5-min horizon (h1)
    # MCC provides balanced metric for imbalanced classification; maximized not minimized
    es_mcc = callbacks.EarlyStopping(
        monitor='val_gauss_dir_mcc_h1',  # Primary horizon (5-min) MCC from (mu,var)
        patience=15,
        mode='max',  # MCC is maximized (higher is better)
        restore_best_weights=False  # Rely on ckpt checkpoint, not this callback
    )
    simple_logger = SimpleLoggingCallback()
    tqdm_callback = TqdmCallback()
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=Config.EPOCHS//2)
    learnable_layer = None
    for layer in custom_model.layers:
        if getattr(layer, 'name', '').startswith('learnable_indicators'):
            learnable_layer = layer
            break
    params_logger = ParamsLogger(layer=learnable_layer, out_csv='indicator_params_history.csv')

    # Decide how many epochs to run (allow caller to override)
    actual_epochs = int(epochs) if epochs is not None else int(config.EPOCHS)

    if os.path.exists(config.MODEL_PATH):
        print(f"Loading existing model weights from {config.MODEL_PATH}...")
        try:
            custom_model.load_weights(config.MODEL_PATH)
        except Exception as e:
            print(f"Warning: failed to load existing weights but continuing: {e}")
        if os.path.exists(config.SCALER_PATH):
            try:
                target_scaler = joblib.load(config.SCALER_PATH)
                print(f"Loaded scaler from {config.SCALER_PATH}")
            except Exception as e:
                print(f"Warning: failed to load scaler: {e}")
        else:
            print(f"Scaler file not found at {config.SCALER_PATH}.")
            # proceed — scaler will be saved after training

        if not force:
            print("Model weights loaded. Skipping training (pass force=True to override).")
            history = None
        else:
            print("Force flag set: continuing to train despite existing weights.")
            callbacks_list = [csv_logger, es, ckpt, es_mcc, tqdm_callback, params_logger, lr_scheduler]
            if extra_callbacks:
                callbacks_list += list(extra_callbacks)
            print(f"Training for {actual_epochs} epochs (force mode)...")
            history = custom_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=actual_epochs,
                callbacks=callbacks_list,
                verbose=1,  # Set to 0 to let tqdm handle progress display
            )
            print(f"Enhanced model weights saved to {config.MODEL_PATH}")
            try:
                joblib.dump(target_scaler, config.SCALER_PATH)
                print(f"Scaler saved to {config.SCALER_PATH}")
            except Exception:
                pass
    else:
        print("No saved weights found. Training a new model with extended trend features...")
        callbacks_list = [csv_logger, es, ckpt, es_mcc, tqdm_callback, params_logger, lr_scheduler]
        if extra_callbacks:
            callbacks_list += list(extra_callbacks)
        print(f"Training for {actual_epochs} epochs...")
        history = custom_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=actual_epochs,
            callbacks=callbacks_list,
            verbose=0,  # Set to 0 to let tqdm handle progress display
        )
        print(f"Enhanced model weights saved to {config.MODEL_PATH}")
        try:
            joblib.dump(target_scaler, config.SCALER_PATH)
            print(f"Scaler saved to {config.SCALER_PATH}")
        except Exception:
            pass

    print("Evaluating enhanced model...")
    train_ds, val_ds = predictor.create_datasets(
        X_train_seq, y_train_scaled, last_close_train, extended_trends_train,
        X_test_seq, y_test_scaled, last_close_test, extended_trends_test
    )
    X_test_simple = tf.data.Dataset.from_tensor_slices(X_test_seq).batch(config.BATCH_SIZE)
    y_pred_all = custom_model.predict(X_test_simple)
    # Model returns 9 outputs (interleaved): price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2
    y_pred_price_all = np.column_stack([
        y_pred_all[0][:, 0],  # price_h0
        y_pred_all[3][:, 0],  # price_h1
        y_pred_all[6][:, 0]   # price_h2
    ])  # Shape: [test_len, 3] for (h0=1min, h1=5min, h2=15min)

    # Only keep predictions for actual test samples (remove padding from last batch if any)
    y_pred_price_all = y_pred_price_all[:len(y_test)]  # [test_len, 3]

    # Extract per-horizon predictions
    y_pred_h0 = y_pred_price_all[:, 0]  # 1-min horizon
    y_pred_h1 = y_pred_price_all[:, 1]  # 5-min horizon (primary)
    y_pred_h2 = y_pred_price_all[:, 2]  # 15-min horizon

    # Inverse transform each horizon (target_scaler expects [n, 1] shape)
    y_pred_h0_raw = target_scaler.inverse_transform(y_pred_h0.reshape(-1, 1)).ravel()
    y_pred_h1_raw = target_scaler.inverse_transform(y_pred_h1.reshape(-1, 1)).ravel()  # Primary
    y_pred_h2_raw = target_scaler.inverse_transform(y_pred_h2.reshape(-1, 1)).ravel()

    # Targets are multi-horizon deltas: y_test[:,0]=h0, y_test[:,1]=h1, y_test[:,2]=h2
    y_test = np.asarray(y_test)
    if y_test.ndim != 2 or y_test.shape[1] != 3:
        raise ValueError(f"Expected y_test shape (N,3) for delta targets, got {y_test.shape}")

    # Use h1 (5-min primary) for main evaluation metrics
    y_pred = y_pred_h1_raw
    y_true_primary = y_test[:, 1]

    # Compute regression metrics for all horizons (delta space)
    print("\n[Per-Horizon Delta Evaluation]")
    y_true_h0 = y_test[:, 0]
    y_true_h1 = y_test[:, 1]
    y_true_h2 = y_test[:, 2]
    for h_name, h_pred, h_true in [
        ("h0_1min", y_pred_h0_raw, y_true_h0),
        ("h1_5min", y_pred_h1_raw, y_true_h1),
        ("h2_15min", y_pred_h2_raw, y_true_h2),
    ]:
        mse_h = mean_squared_error(h_true, h_pred)
        r2_h = r2_score(h_true, h_pred)
        mape_h = mean_absolute_percentage_error(h_true, h_pred)
        print(f"  {h_name}: MSE={mse_h:.6f}, R2={r2_h:.6f}, MAPE={mape_h:.4f}")

    # Extract direction predictions from 9-output list
    dir_pred_h0 = np.asarray(y_pred_all[1]).reshape(-1)
    dir_pred_h1 = np.asarray(y_pred_all[4]).reshape(-1)
    dir_pred_h2 = np.asarray(y_pred_all[7]).reshape(-1)
    dir_pred_h0 = dir_pred_h0[:len(y_true_h0)]
    dir_pred_h1 = dir_pred_h1[:len(y_true_h1)]
    dir_pred_h2 = dir_pred_h2[:len(y_true_h2)]

    # Compute regression metrics for primary horizon (h1)
    mse_val = mean_squared_error(y_true_primary, y_pred)
    r2_val = r2_score(y_true_primary, y_pred)
    mape_val = mean_absolute_percentage_error(y_true_primary, y_pred)

    # Compute direction metrics for all 3 horizons
    print("\n[Per-Horizon Direction Metrics]")
    print(f"{'Horizon':<12} {'Accuracy':<10} {'F1':<10} {'Sensitivity':<12} {'Specificity':<12} {'MCC':<10}")
    print("-" * 66)
    
    direction_results = {}
    deadband_bps = float(getattr(config, 'DIR_DEADBAND_BPS', 0.0))
    deadband = deadband_bps / 10000.0
    threshold_delta = deadband * (np.asarray(last_close_test, dtype=float) + 1e-12)

    for h_idx, (h_name, h_delta_pred, h_delta_true, h_dir_prob) in enumerate([
        ("h0_1min", y_pred_h0_raw, y_true_h0, dir_pred_h0),
        ("h1_5min", y_pred_h1_raw, y_true_h1, dir_pred_h1),
        ("h2_15min", y_pred_h2_raw, y_true_h2, dir_pred_h2),
    ]):
        # Binary direction: UP if delta > deadband * last_close, DOWN otherwise
        pred_dir = (h_delta_pred > threshold_delta)  # bool
        true_dir = (h_delta_true > threshold_delta)  # bool
        
        # Confusion matrix components
        tp = np.sum((pred_dir == 1) & (true_dir == 1))
        tn = np.sum((pred_dir == 0) & (true_dir == 0))
        fp = np.sum((pred_dir == 1) & (true_dir == 0))
        fn = np.sum((pred_dir == 0) & (true_dir == 1))
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1_score_val = f1_score(true_dir, pred_dir, zero_division=0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        
        # Matthews Correlation Coefficient
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
        
        print(f"{h_name:<12} {accuracy:<10.4f} {f1_score_val:<10.4f} {sensitivity:<12.4f} {specificity:<12.4f} {mcc:<10.4f}")
        
        direction_results[h_name] = {
            "accuracy": float(accuracy),
            "f1": float(f1_score_val),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "mcc": float(mcc),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        }

    print(f"\n[Primary Horizon (h1_5min)] MSE: {mse_val:.6f}, R2: {r2_val:.6f}, MAPE: {mape_val:.4f}")

    return (custom_model, target_scaler, X_test_seq, y_test, y_pred,
            last_close_test, history, extended_trends_test,
            {"h0": y_pred_h0_raw, "h1": y_pred_h1_raw, "h2": y_pred_h2_raw})
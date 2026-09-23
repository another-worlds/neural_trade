"""DataProcessor: the data pipeline facade (moved from model.py in Phase B9).

load (DataLoaders registry, Config.DATA_LOADER) -> preprocess (Preprocessors registry,
Config.PREPROCESSORS, in order) -> validate -> windows + forward targets -> purged
train | val | cal | test split -> target scaler and window normaliser fit on TRAIN only.
The public methods and return values are those of the original class.
"""
from __future__ import annotations

import math
from typing import Optional

import joblib
import numpy as np

from neural_trade.data.loaders import validate_ohlcv_frame
from neural_trade.data.scaling import WindowNormalizer, fit_target_scaler, transform_targets
from neural_trade.data.splits import make_purged_splits
from neural_trade.data.windowing import compute_extended_trend_features, make_sequences_with_extended_trends


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.target_scaler = None
        self.input_scaler = None

    def clean_numeric(self, series):
        return series.astype(str).str.replace(r'[\$,]', '', regex=True).replace('', np.nan).astype(float)

    def load_raw(self, read_csv_kwargs: Optional[dict] = None, **loader_kwargs):
        """Raw frame from the configured loader (Config.DATA_LOADER)."""
        from neural_trade.registries.data_loaders import DataLoaders

        name = getattr(self.config, 'DATA_LOADER', 'csv')
        if name == 'csv':
            loader_kwargs['read_csv_kwargs'] = read_csv_kwargs
        return DataLoaders.build(name, self.config, **loader_kwargs)

    def preprocess(self, df):
        """Apply Config.PREPROCESSORS in order, then validate the standardised frame."""
        from neural_trade.registries.preprocessors import run_preprocessors

        return validate_ohlcv_frame(run_preprocessors(df, self.config))

    def load_and_prepare_data(self, read_csv_kwargs: Optional[dict] = None, **loader_kwargs):
        """Load and prepare minute-level data; returns ``(df, close_values float32)``.

        ``read_csv_kwargs`` is passed through to ``pd.read_csv`` for the csv loader.
        """
        df = self.preprocess(self.load_raw(read_csv_kwargs, **loader_kwargs))
        print(f"Dataset length after cleaning: {len(df)}")
        print("Date range after cleaning:", df['Date'].min(), "to", df['Date'].max())

        if len(df) < self.config.LOOKBACK + 2:
            raise ValueError(f"Not enough rows ({len(df)}) for lookback={self.config.LOOKBACK}")

        return df, df['Close'].values.astype('float32')

    def compute_extended_trend_features(self, close_values, index, periods):
        return compute_extended_trend_features(close_values, index, periods)

    def make_sequences_with_extended_trends(self, close_array, lookback):
        return make_sequences_with_extended_trends(self.config, close_array, lookback)

    def plot_splits(self, df, start_idx, tscv, X_seq_len):
        from neural_trade.visualization.matplotlib_splits import plot_splits

        return plot_splits(df, start_idx, tscv, X_seq_len)

    def prepare_datasets(self, df, close_values):
        X_seq, y_seq, last_close_seq, extended_trends = make_sequences_with_extended_trends(
            self.config, close_values, self.config.LOOKBACK
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
        target_scaler = fit_target_scaler(y_train)
        y_train_scaled = transform_targets(target_scaler, y_train)
        y_val_scaled = transform_targets(target_scaler, y_val)
        y_cal_scaled = transform_targets(target_scaler, y_cal)
        y_test_scaled = transform_targets(target_scaler, y_test)

        # Window normalisation (Config.WINDOW_NORMALIZER, default window_relative:
        # (x - last_close) / target_scale). The previous per-lag-position StandardScaler z-scored the
        # absolute price LEVEL, so the model's dominant signal was "where is BTC vs. its multi-week
        # mean" - noise w.r.t. a delta target. See neural_trade.data.scaling.
        normalizer = WindowNormalizer.fit(getattr(self.config, 'WINDOW_NORMALIZER', 'window_relative'),
                                          X_train_seq, target_scaler)
        input_scale = normalizer.scale
        _normalise = normalizer.transform

        X_train_seq_scaled = _normalise(X_train_seq, last_close_train)
        X_test_seq_scaled = _normalise(X_test_seq, last_close_test)
        input_scaler = normalizer.per_lag  # None for window_relative (nothing to persist)

        joblib.dump(target_scaler, self.config.SCALER_PATH)

        # Keep references for programmatic use without changing the return signature.
        self.target_scaler = target_scaler
        self.input_scaler = input_scaler
        self.input_scale = input_scale
        self.normalizer = normalizer
        # Validation and calibration blocks, consumed by train_and_evaluate.
        self.val_block = dict(X=_normalise(X_val_seq, last_close_val), y_scaled=y_val_scaled, y_raw=y_val,
                              last_close=last_close_val, extended_trends=extended_trends_val)
        self.cal_block = dict(X=_normalise(X_cal_seq, last_close_cal), y_scaled=y_cal_scaled, y_raw=y_cal,
                              last_close=last_close_cal, extended_trends=extended_trends_cal)

        return (X_train_seq_scaled, y_train_scaled, last_close_train, extended_trends_train,
                X_test_seq_scaled, y_test_scaled, last_close_test, extended_trends_test,
                y_train, y_test, target_scaler)

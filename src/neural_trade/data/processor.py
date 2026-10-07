"""DataProcessor: the data pipeline facade (moved from model.py in Phase B9).

load (DataLoaders registry, Config.DATA_LOADER) -> preprocess (Preprocessors registry,
Config.PREPROCESSORS, in order) -> validate -> windows + forward targets -> purged
train | val | cal | test split -> target scaler and window normaliser fit on TRAIN only.
The public methods and return values are those of the original class.
"""
from __future__ import annotations

import logging

import math
from typing import Optional

import numpy as np
import pandas as pd

from neural_trade.data.loaders import validate_ohlcv_frame
from neural_trade.data.scaling import WindowNormalizer, fit_target_scaler, transform_targets
from neural_trade.data.plan import make_plan
from neural_trade.data.splits import FoldIndices, make_purged_splits
from neural_trade.data.windowing import (compute_extended_trend_features, frame_series, make_multichannel_windows,
                                         make_sequences_with_extended_trends, sequence_counts)

logger = logging.getLogger(__name__)


def apply_data_end(df, config):
    """Slice ``df`` (already preprocessed, sorted by timestamp) to end at ``Config.DATA_END``, taken
    BEFORE ``MAX_SEQUENCE_COUNT`` trims from the end of the slice (NT-088 screen mode: trials pick a
    volatility regime anywhere in history, not only the file's newest bars). Refuses a ``DATA_END``
    that falls within the protected dev/test span (the file's last ``DATA_END_PROTECTED_DAYS`` days),
    so a screen trial cannot leak the long file's held-out period into training (D-020). ``None``
    (the default): no slicing, today's behaviour."""
    data_end = getattr(config, "DATA_END", None)
    if data_end is None:
        return df
    try:
        end = pd.Timestamp(data_end)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"DATA_END={data_end!r} is not a valid timestamp: {exc}") from exc
    ts = df["timestamp"]
    tz = getattr(ts.dtype, "tz", None)
    if end.tzinfo is None and tz is not None:
        end = end.tz_localize(tz)
    elif end.tzinfo is not None and tz is None:
        end = end.tz_localize(None)
    last = ts.max()
    protected_days = float(getattr(config, "DATA_END_PROTECTED_DAYS", 64.0))
    protected_start = last - pd.Timedelta(days=protected_days)
    if end >= protected_start:
        raise ValueError(
            f"DATA_END={data_end} falls within the protected dev/test span: the file's last "
            f"{protected_days:g} days start at {protected_start} (last bar {last}); choose an earlier "
            "DATA_END, or the run's dev/test period could leak into training (D-020)")
    sliced = df[ts <= end]
    if sliced.empty:
        raise ValueError(f"DATA_END={data_end} is before the data starts ({ts.min()})")
    return sliced.reset_index(drop=True)


def _select_fold(folds, index):
    """The fold at ``index`` of make_purged_splits' list (folds with an empty train block are omitted,
    so index -1 is always the latest; an out-of-range positive index is an error)."""
    try:
        return folds[index]
    except IndexError:
        raise ValueError(f"FOLD_INDEX={index} but only {len(folds)} usable folds exist") from None


def split_arrays(config, read_csv_kwargs=None):
    """RAW (unscaled) CLOSE windows, targets, last closes and trend features of every block of
    the configured fold: ``{"train": {...}, "val": {...}, "cal": {...}, "test": {...}, "fold": FoldIndices}``.
    Every consumer of these blocks (baselines, realized vol, backtests) reads close windows;
    the model's multi-series input (Config.INPUT_SERIES, NT-047) is built by ``prepare_datasets``.

    Deterministic given the config, so any saved run can rebuild its splits (baselines,
    backtests, re-evaluation) without having stored them.
    """
    dp = DataProcessor(config)
    df, close = dp.load_and_prepare_data(read_csv_kwargs=read_csv_kwargs)
    # NT-041: one plan decides the anchors (the hole policy, the MAX_SEQUENCE_COUNT cap) and the folds; NT-177:
    # only the sequences the fold reads are windowed
    plan = make_plan(config, df)
    fold = plan.fold(int(getattr(config, "FOLD_INDEX", -1)))
    lo, hi = plan.read(fold)
    anchors = plan.anchors[lo:hi]
    fold = _shift_fold(fold, lo)
    X, y, lc, ext = make_sequences_with_extended_trends(config, close, config.LOOKBACK, anchors=anchors)
    # model-input windows (NT-047): identical to X in close-only mode, [N, L, C] otherwise
    series_names = list(getattr(config, "INPUT_SERIES", None) or ["close"])
    Xm = X if series_names == ["close"] else make_multichannel_windows(
        config, frame_series(config, df), config.LOOKBACK, anchors=anchors)
    out = {"fold": fold, "close": close, "df": df, "gaps": plan.gaps}
    for name in ("train", "val", "cal", "test"):
        idx = getattr(fold, name)
        out[name] = {"X": X[idx], "X_model": Xm[idx], "y": y[idx], "last_close": lc[idx],
                     "extended_trends": ext[idx], "index": idx,
                     "anchor_bar": (anchors[idx] - 1).astype(np.int32)}   # bar positions: int32 as before
    return out


def _shift_fold(fold: FoldIndices, lo: int) -> FoldIndices:
    """``fold`` with its indices counted from sequence ``lo`` (the first one its windows hold)."""
    if not lo:
        return fold
    return FoldIndices(fold.fold, fold.gap, fold.train - lo, fold.val - lo, fold.cal - lo, fold.test - lo)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.target_scaler = None
        self.input_scaler = None

    def clean_numeric(self, series):
        return series.astype(str).str.replace(r'[\$,]', '', regex=True).replace('', np.nan).astype(float)

    def load_raw(self, read_csv_kwargs: Optional[dict] = None, **loader_kwargs):
        """Raw frame from the configured loader (Config.DATA_LOADER)."""
        from neural_trade.data.loaders_registry import DataLoaders

        name = getattr(self.config, 'DATA_LOADER', 'csv')
        if name == 'csv':
            loader_kwargs['read_csv_kwargs'] = read_csv_kwargs
        return DataLoaders.build(name, self.config, **loader_kwargs)

    def preprocess(self, df):
        """Apply Config.PREPROCESSORS in order, then validate the standardised frame."""
        from neural_trade.data.preprocessors_registry import run_preprocessors

        return validate_ohlcv_frame(run_preprocessors(df, self.config),
                                    bar_minutes=getattr(self.config, 'RESAMPLE_MINUTES', None))

    def load_and_prepare_data(self, read_csv_kwargs: Optional[dict] = None, **loader_kwargs):
        """Load and prepare minute-level data; returns ``(df, close_values float32)``.

        ``read_csv_kwargs`` is passed through to ``pd.read_csv`` for the csv loader.
        """
        df = self.preprocess(self.load_raw(read_csv_kwargs, **loader_kwargs))
        df = apply_data_end(df, self.config)
        logger.info(f"Dataset length after cleaning: {len(df)}")
        logger.info('%s %s %s %s', "Date range after cleaning:", df['Date'].min(), "to", df['Date'].max())

        if len(df) < self.config.LOOKBACK + 2:
            raise ValueError(f"Not enough rows ({len(df)}) for lookback={self.config.LOOKBACK}")

        return df, df['Close'].values.astype('float32')

    def compute_extended_trend_features(self, close_values, index, periods):
        return compute_extended_trend_features(close_values, index, periods)

    def make_sequences_with_extended_trends(self, close_array, lookback):
        return make_sequences_with_extended_trends(self.config, close_array, lookback)

    def build_windows(self, close_values, df=None):
        """Sliding windows/targets (:func:`make_sequences_with_extended_trends`) trimmed to
        ``MAX_SEQUENCE_COUNT`` (most recent kept): the part of :meth:`prepare_datasets` that loops
        every bar and is therefore worth caching. A pure function of ``self.config`` and
        ``close_values`` (every field it reads - LOOKBACK, HORIZON_STEPS, EXTENDED_TREND_PERIODS,
        WINDOW_STEP, MAX_SEQUENCE_COUNT - is part of :func:`neural_trade.experiments.dataset.data_key`),
        so a caller that trains many configurations sharing a data key (NT-088 screen mode) can build
        it once and reuse it across all of them, only redoing the split/scale/normalise step below.

        Returns ``(X_seq, y_seq, last_close_seq, extended_trends, X_model)``: ``X_model`` is the model
        input (NT-047), ``X_seq`` itself in close-only mode, otherwise the [N, LOOKBACK, C] windows over
        ``Config.INPUT_SERIES`` built from ``df`` (required then; INPUT_SERIES is part of the data key).
        """
        # NT-177: only the newest MAX_SEQUENCE_COUNT sequences are built (the oldest are never windowed).
        # NT-041: with the bar frame the plan also applies the hole policy and, for the timed layout, windows only
        # the span the chosen fold reads (remembered for prepare_datasets_from_windows)
        self._plan = self._lo = None
        if df is not None:
            plan = make_plan(self.config, df)
            lo, hi = plan.read(plan.fold(int(getattr(self.config, "FOLD_INDEX", -1))))
            self._plan, self._lo = plan, lo
            anchors, n_total, dropped = plan.anchors[lo:hi], plan.n_total, 0
            X_seq, y_seq, last_close_seq, extended_trends = make_sequences_with_extended_trends(
                self.config, close_values, self.config.LOOKBACK, anchors=anchors
            )
        else:
            anchors = None
            n_total, dropped = sequence_counts(self.config, len(close_values))
            X_seq, y_seq, last_close_seq, extended_trends = make_sequences_with_extended_trends(
                self.config, close_values, self.config.LOOKBACK, first_seq=dropped
            )
        logger.info(f"Sequences with extended trends: {X_seq.shape}, {y_seq.shape}, Extended: {extended_trends.shape}")

        # Model input windows (NT-047): the close windows themselves in close-only mode
        # (Config.INPUT_SERIES == ['close']: the pre-NT-047 path, bit-for-bit), otherwise
        # [N, LOOKBACK, C] over the configured series, on the SAME anchors. The raw close
        # windows (X_seq) stay what every downstream consumer of "raw windows" reads
        # (conformal realized vol, baselines, backtests).
        series_names = list(getattr(self.config, 'INPUT_SERIES', None) or ['close'])
        if series_names == ['close']:
            X_model = X_seq
        else:
            if df is None:
                raise ValueError(f"INPUT_SERIES={series_names} needs the bar frame: call build_windows(close, df)")
            X_model = make_multichannel_windows(self.config, frame_series(self.config, df),
                                                self.config.LOOKBACK, first_seq=dropped, anchors=anchors)
            if X_model.shape[0] != X_seq.shape[0]:
                raise RuntimeError(f"model windows ({X_model.shape[0]}) and close windows "
                                   f"({X_seq.shape[0]}) disagree - a windowing bug")

        if n_total > len(X_seq):
            logger.info(f"[OK] Limited sequence set from {n_total} to {len(X_seq)} (most recent window)")
        return X_seq, y_seq, last_close_seq, extended_trends, X_model

    def prepare_datasets(self, df, close_values):
        X_seq, y_seq, last_close_seq, extended_trends, X_model = self.build_windows(close_values, df)
        return self.prepare_datasets_from_windows(X_seq, y_seq, last_close_seq, extended_trends, X_model=X_model)

    def prepare_datasets_from_windows(self, X_seq, y_seq, last_close_seq, extended_trends, X_model=None):
        """The fold split, target scaling and window normalisation of :meth:`prepare_datasets`, given
        already-built (and, for MAX_SEQUENCE_COUNT, already-trimmed) windows. Splitting an array by
        index and fitting a scaler on it is cheap next to building the windows themselves (no
        per-bar Python loop), so a caller that caches :meth:`build_windows` per data key (NT-088)
        still pays this part once per trial - the fields it reads beyond the window/data_key ones
        (N_FOLDS, VAL_FRACTION, CAL_FRACTION, WINDOW_STEP, BATCH_SIZE, WINDOW_NORMALIZER, FOLD_INDEX)
        may differ per trial even when the data key does not.
        """
        series_names = list(getattr(self.config, 'INPUT_SERIES', None) or ['close'])
        if X_model is None:
            if series_names != ['close']:
                raise ValueError(f"INPUT_SERIES={series_names}: pass X_model from build_windows(close, df)")
            X_model = X_seq
        logger.info("[INFO] Dataset Statistics:")
        logger.info(f"   Total sequences: {X_seq.shape[0]}")

        # Four-way PURGED chronological split: train | gap | val | gap | cal | gap | test.
        #   train -> gradients and the target scaler;  val -> early stopping / checkpoint / LR;
        #   cal   -> post-hoc calibration (temperature, conformal);  test -> reported once.
        # Previously the last TimeSeriesSplit fold was BOTH validation and test with no gap:
        # 79 "validation" windows contained bars that were training labels, model selection
        # happened on the test set, and calibration was fit on the test set too.
        if getattr(self.config, 'FOLD_LAYOUT', 'tscv') == 'timed':
            plan = getattr(self, '_plan', None)
            if plan is None:
                raise ValueError("FOLD_LAYOUT 'timed' places the folds by timestamps: build the windows with "
                                 "this DataProcessor's build_windows(close, df) (or prepare_datasets), not a "
                                 "cache made elsewhere")
            fold = _shift_fold(plan.fold(int(getattr(self.config, 'FOLD_INDEX', -1))), self._lo)
        else:
            fold = make_purged_splits(
                X_seq.shape[0],
                lookback=self.config.LOOKBACK,
                horizon_steps=self.config.HORIZON_STEPS,
                window_step=int(max(1, getattr(self.config, 'WINDOW_STEP', 1))),
                n_folds=int(getattr(self.config, 'N_FOLDS', 5)),
                val_fraction=float(getattr(self.config, 'VAL_FRACTION', 0.066)),
                cal_fraction=float(getattr(self.config, 'CAL_FRACTION', 0.066)),
            )
            fold = _select_fold(fold, int(getattr(self.config, 'FOLD_INDEX', -1)))
        self.fold = fold

        def _take(idx):
            return X_model[idx], y_seq[idx], last_close_seq[idx], extended_trends[idx]

        X_train_seq, y_train, last_close_train, extended_trends_train = _take(fold.train)
        X_val_seq, y_val, last_close_val, extended_trends_val = _take(fold.val)
        X_cal_seq, y_cal, last_close_cal, extended_trends_cal = _take(fold.cal)
        X_test_seq, y_test, last_close_test, extended_trends_test = _take(fold.test)

        train_batches = math.ceil(X_train_seq.shape[0] / self.config.BATCH_SIZE)
        test_batches = math.ceil(X_test_seq.shape[0] / self.config.BATCH_SIZE)
        logger.info(f"   Train sequences: {X_train_seq.shape[0]} (batches/epoch: {train_batches})")
        logger.info(f"   Val sequences:   {X_val_seq.shape[0]}   Cal sequences: {X_cal_seq.shape[0]}   "
              f"(purge gap: {fold.gap} sequences, fold {fold.fold})")
        logger.info(f"   Test sequences:  {X_test_seq.shape[0]} (batches: {test_batches})")

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
                                          X_train_seq, target_scaler,
                                          input_series=series_names if X_train_seq.ndim == 3 else None)
        input_scale = normalizer.scale
        _normalise = normalizer.transform

        X_train_seq_scaled = _normalise(X_train_seq, last_close_train)
        X_test_seq_scaled = _normalise(X_test_seq, last_close_test)
        input_scaler = normalizer.per_lag  # None for window_relative (nothing to persist)

        # NT-028: the target scaler is no longer dumped here. Nothing loads SCALER_PATH (the serving
        # bundle carries the target scale in meta.json); train_and_evaluate writes it into a run
        # directory only, so a run without a RunContext leaves no unread file behind.

        # Keep references for programmatic use without changing the return signature.
        self.target_scaler = target_scaler
        self.input_scaler = input_scaler
        self.input_scale = input_scale
        self.normalizer = normalizer
        # Validation and calibration blocks, consumed by train_and_evaluate.
        # RAW close windows for the consumers of "raw windows" (conformal realized vol,
        # baselines, backtests): in multi-series mode (NT-047) that is the close CHANNEL of
        # the model windows (identical to the close windows: same anchors).
        def _raw_close(Xb):
            return Xb if Xb.ndim == 2 else np.ascontiguousarray(Xb[..., series_names.index('close')])

        self.val_block = dict(X=_normalise(X_val_seq, last_close_val), y_scaled=y_val_scaled, y_raw=y_val,
                              last_close=last_close_val, extended_trends=extended_trends_val)
        self.cal_block = dict(X=_normalise(X_cal_seq, last_close_cal), y_scaled=y_cal_scaled, y_raw=y_cal,
                              last_close=last_close_cal, extended_trends=extended_trends_cal,
                              X_raw=_raw_close(X_cal_seq))
        # RAW test CLOSE windows (conformal realized-vol scales, baselines, backtests).
        self.test_windows_raw = _raw_close(X_test_seq)

        return (X_train_seq_scaled, y_train_scaled, last_close_train, extended_trends_train,
                X_test_seq_scaled, y_test_scaled, last_close_test, extended_trends_test,
                y_train, y_test, target_scaler)

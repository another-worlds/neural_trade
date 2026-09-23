"""Sliding windows, forward targets and past-delta features (moved from DataProcessor in B9).

For an anchor bar i (the first bar AFTER the window):
    window         = close[i - LOOKBACK : i]
    last_close     = close[i - 1]
    target[k]      = close[i + h_k - 1] - last_close           (raw dollar delta, horizon h_k)
    extended[k]    = last_close - close[i - 1 - p_k]           (past delta over EXTENDED_TREND_PERIODS[k])
Targets are strictly after every input bar; see tests/test_data_processor.py.
"""
from __future__ import annotations

import numpy as np


def compute_extended_trend_features(close_values, index, periods):
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


def make_sequences_with_extended_trends(config, close_array, lookback):
    X, y, last_close, extended_trends = [], [], [], []
    # Ensure start index is an integer even if periods are provided as floats
    max_extended_period = int(max(config.EXTENDED_TREND_PERIODS))
    start_idx = int(max(lookback, max_extended_period))
    step = int(max(1, getattr(config, 'WINDOW_STEP', 1)))

    horizon_steps = [int(h) for h in getattr(config, 'HORIZON_STEPS', [1, 5, 15])]
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
        ext_features = compute_extended_trend_features(close_array, int(i-1), config.EXTENDED_TREND_PERIODS)
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


def make_inference_windows(close_array, lookback, *, extended_trend_periods=None):
    """Windows for PREDICTION: every complete window, including the latest, with no targets.

    ``make_sequences_with_extended_trends`` needs max(HORIZON_STEPS) future bars per window,
    so it can never produce the window ending at the newest bar - the one a live prediction
    needs. Returns ``(X [N, lookback], last_close [N], extended [N, len(periods)])`` for the
    anchors ``lookback .. len(close)`` (the last window ends at the final bar).
    """
    close = np.asarray(close_array, dtype="float32").reshape(-1)
    periods = [int(p) for p in (extended_trend_periods or [])]
    start = int(max([lookback] + periods))
    if len(close) < start:
        raise ValueError(f"need at least {start} bars, got {len(close)}")
    X, lc, ext = [], [], []
    for i in range(start, len(close) + 1):
        X.append(close[i - lookback:i])
        lc.append(close[i - 1])
        ext.append(compute_extended_trend_features(close, i - 1, periods) if periods else np.zeros(0, "float32"))
    return (np.array(X, dtype="float32"), np.array(lc, dtype="float32"),
            np.array(ext, dtype="float32").reshape(len(X), len(periods)))

"""Evaluation metrics of a set of predictions (moved from model.py in Phase B8).

* :func:`_compute_all_horizon_metrics` - the legacy nested report (delta / price /
  direction per horizon) that ``TrainResult.metrics`` has always carried.
* :func:`registry_metrics` - every metric named in ``Config.METRICS``, computed through the
  Metrics registry, per horizon; direction metrics use the neutral mask.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, explained_variance_score, f1_score,
                             mean_absolute_percentage_error, mean_squared_error)

from neural_trade.metrics.direction_labels import compute_direction_labels_np
from neural_trade.metrics.numpy_metrics import reconstruct_prices, safe_mape, smape, wape

HORIZONS = ("h0", "h1", "h2")


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
        # Sanitize predictions: if NaN/Inf slipped through (e.g. before full stability fixes),
        # replace so sklearn metrics don't raise ValueError. Diagnostics will still surface the nans.
        y_p = np.nan_to_num(y_p, nan=0.0, posinf=0.0, neginf=0.0)
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

        # Direction labels with the same NEUTRAL MASK the train/validation metrics use:
        # |return| <= deadband is neither UP nor DOWN and is excluded. The previous
        # delta-space threshold had no mask, so this accuracy never matched val_dir_acc.
        # S22: one labelling rule for every path (metrics_utils.compute_direction_labels_np).
        if compute_direction_labels_np is not None:
            _lab, dir_mask = compute_direction_labels_np(y_t, lc_h, deadband_bps)["h0"]
            true_dir = _lab.astype(bool)
        else:  # metrics_utils unavailable: same rule inline
            ret = y_t / (lc_h + 1e-12)
            dir_mask = np.abs(ret) > deadband
            true_dir = (ret > deadband)
        if dir_probs is not None and h_key in dir_probs and dir_probs[h_key] is not None:
            p = np.asarray(dir_probs[h_key], dtype=float).reshape(-1)[:n]
            pred_dir = (p >= 0.5)
        else:
            pred_dir = (y_p > thr)
        if int(dir_mask.sum()) > 0:
            _td, _pd = true_dir[dir_mask].astype(int), pred_dir[dir_mask].astype(int)
            _acc, _f1 = float(accuracy_score(_td, _pd)), float(f1_score(_td, _pd, zero_division=0))
        else:
            _acc, _f1 = float('nan'), float('nan')
        out["direction"][h_key] = {"acc": _acc, "f1": _f1, "n_masked": int(dir_mask.sum())}

    out["meta"] = {
        "horizon_keys": list(horizons),
        "horizon_labels": list(horizon_names),
        "deadband_bps": float(deadband_bps),
        "delta_safe_mape_min_abs": float(min_abs_delta_for_mape),
    }
    return out


def registry_metrics(config, y_true_deltas, predictions, last_close, *, pred_scale=None,
                     intervals=None, names=None) -> Dict[str, Dict[str, float]]:
    """``{horizon: {metric: value}}`` for the Config.METRICS names (or ``names``).

    ``predictions`` is the TrainResult layout: {"delta": {h}, "direction_prob": {h},
    "variance": {h}} with variance in SCALED units; ``intervals`` optionally maps each
    horizon to (lower, upper) raw-dollar bounds (the calibrated conformal intervals).
    ``pit_ks`` needs ``pred_scale`` (the target scaler's std) to put the variance in dollars
    and ``coverage`` needs ``intervals``; each is skipped when its input is missing.
    """
    from neural_trade.metrics import numpy_metrics as npm
    from neural_trade.registries.metrics import Metrics

    names = list(names or getattr(config, "METRICS", None) or Metrics.numpy_names())
    fns = Metrics.numpy_functions(names)
    y = np.asarray(y_true_deltas, dtype=float)
    lc = np.asarray(last_close, dtype=float).reshape(-1)
    labels_by_h = compute_direction_labels_np(y, lc, float(getattr(config, "DIR_DEADBAND_BPS", 0.0)))
    out: Dict[str, Dict[str, float]] = {}
    for i, h in enumerate(HORIZONS):
        yt = y[:, i]
        delta = np.asarray(predictions["delta"][h], dtype=float)
        prob = np.asarray(predictions["direction_prob"][h], dtype=float)
        labels, mask = labels_by_h[h]
        row = {}
        for name, fn in fns.items():
            if name in npm.DIRECTION_METRICS:
                row[name] = fn(labels, prob, mask=mask)
            elif name == "pit_ks":
                if not pred_scale:
                    continue
                var = np.asarray(predictions["variance"][h], dtype=float)
                row[name] = fn(yt, delta, variance=var * float(pred_scale) ** 2)
            elif name == "coverage":
                if intervals and h in intervals:
                    lo, hi = intervals[h]
                    row[name] = fn(yt, np.stack([np.asarray(lo), np.asarray(hi)], axis=1))
            else:
                row[name] = fn(yt, delta)
        out[h] = row
    return out

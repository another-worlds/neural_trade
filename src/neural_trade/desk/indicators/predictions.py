"""DESK-5.6: prediction overlay arrays for the chart (D-067, PLAN A8, CONTRACTS C5 ``GET /lab/predictions``).

:func:`overlay` runs a promoted model's ``serving.Predictor`` over the last ``n`` closed bars and returns JSON-safe
arrays with ONE ROW PER BAR (``t`` has ``n`` entries and so has every series; a bar with no complete window yet, or a
non-finite value, is ``null``). Per horizon: the price path (``price`` at ``target_t = t + horizon_bars`` bars), the
``lo90`` / ``hi90`` band, ``sigma`` and the P(up) series with its ``direction_signal``.

Honest numbers (README rule 5):

* ``direction_signal == "none"`` (D-066, the temperature fit ended at a bound) is "no usable direction signal": the P(up)
  series is ``flagged: true``, ``draw_as_forecast: false`` and carries NO values (``values`` is ``null``); the front end
  draws a notice, never a line. ``"ok"`` draws the calibrated P(up); ``"n/a"`` (no calibration pipeline in the bundle)
  draws the raw head and says ``calibrated: false``.
* A served delta of exactly zero because the delta shrinkage beta is 0 (D-007) is reported ``delta_available: false``:
  ``delta`` and ``price`` are ``null`` (never a flat "forecast" at the last close), with the raw heads alongside as
  ``delta_raw`` and ``price_raw``. The band and ``sigma`` stay (they are not the shrunk delta).

Causality: window ``i`` ends at bar ``i`` (``Predictor.predict_windows_frame``), so the overlay at bar ``t`` reads bars up
to ``t`` only; the frame is cut to the newest ``n`` + warm-up bars before predicting. The caller passes closed bars only.
Input: a raw OHLCV frame as the Predictor reads it (``datetime`` or ``timestamp`` plus ``open..volume``) or the desk
store's ``ts, o, h, l, c, v`` frame (:func:`raw_frame`). TensorFlow is never imported here (the Predictor is passed in).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["overlay", "raw_frame", "warmup_bars", "NO_SIGNAL_REASON"]

NO_SIGNAL_REASON = "no usable direction signal (the temperature fit ended at a bound, D-066)"
_NO_DELTA_REASON = "served delta n/a: the delta shrinkage beta is 0 (D-007); raw heads shown alongside"
_STORE_COLUMNS = {"ts": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}


def raw_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """The Predictor's raw frame from the desk store's ``ts, o, h, l, c, v`` bars (``ts`` in epoch ms, UTC)."""
    out = bars.rename(columns=_STORE_COLUMNS).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    return out


def warmup_bars(config: Any) -> int:
    """Bars before the first complete window (the first anchor), so ``n + warmup_bars`` bars give ``n`` rows."""
    from neural_trade.data.windowing import first_anchor

    return int(first_anchor(config.LOOKBACK, config.EXTENDED_TREND_PERIODS))


def _clean(values: Any) -> List[Optional[float]]:
    return [None if not math.isfinite(float(v)) else float(v) for v in np.asarray(values, dtype=float)]


def _epoch_ms(ts: pd.Series) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(ts))
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return np.asarray((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1), dtype="int64")


def _beta_zero(predictor: Any, key: str) -> bool:
    pipeline = getattr(getattr(predictor, "bundle", None), "calibration_pipeline", None)
    if pipeline is None or not getattr(pipeline, "shrink_delta", False):
        return False
    return float(getattr(pipeline, "delta_scale", {}).get(key, 1.0)) == 0.0


def overlay(predictor: Any, bars: pd.DataFrame, n: int, *, alpha: float = 0.1) -> Dict[str, Any]:
    """Overlay arrays over the newest ``n`` bars of ``bars`` (see the module docstring for the payload)."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    frame = raw_frame(bars) if {"ts", "c"} <= set(bars.columns) else bars
    config = predictor.config
    frame = frame.iloc[-(n + warmup_bars(config)):]
    batch, df, anchors = predictor.predict_windows_frame(frame, alpha=alpha)
    rows = len(df)
    take = min(n, rows)
    t = _epoch_ms(df["timestamp"])[-take:]
    bar_ms = int(round(float(config.RESAMPLE_MINUTES) * 60_000))
    anchors = np.asarray(anchors, dtype=int)
    cols = batch.to_frame()

    def spread(name: str) -> np.ndarray:
        """A per-window column placed on its bar (NaN where the bar has no window), newest ``take`` bars."""
        full = np.full(rows, np.nan)
        full[anchors] = np.asarray(cols[name], dtype=float)
        return full[-take:]

    last_close = spread("last_close")
    signal = batch.direction_signal
    horizons = []
    for key, steps in zip(batch.horizons, batch.horizon_steps):
        sig = str(signal.get(key, "ok")) if signal else "n/a"
        no_delta = _beta_zero(predictor, key)
        delta_raw = None if batch.delta_raw is None else spread_delta(batch.delta_raw[key], anchors, rows)[-take:]
        price_raw = None if delta_raw is None else last_close + delta_raw
        if sig == "none":
            p_up = {"flagged": True, "draw_as_forecast": False, "reason": NO_SIGNAL_REASON, "values": None,
                    "calibrated": True}
        else:
            name = f"{key}_p_up_calibrated" if sig == "ok" else f"{key}_p_up"
            p_up = {"flagged": False, "draw_as_forecast": True, "reason": None, "values": _clean(spread(name)),
                    "calibrated": sig == "ok"}
        horizons.append({
            "key": key, "horizon_bars": int(steps), "direction_signal": sig,
            "target_t": [int(x) + int(steps) * bar_ms for x in t],
            "delta_available": not no_delta, "delta_note": _NO_DELTA_REASON if no_delta else None,
            "delta": None if no_delta else _clean(spread(f"{key}_delta")),
            "price": None if no_delta else _clean(spread(f"{key}_price")),
            "delta_raw": None if delta_raw is None else _clean(delta_raw),
            "price_raw": None if price_raw is None else _clean(price_raw),
            "lo90": _clean(spread(f"{key}_lo90")), "hi90": _clean(spread(f"{key}_hi90")),
            "sigma": _clean(spread(f"{key}_sigma")), "p_up": p_up})
    return {"n": int(take), "t": [int(x) for x in t], "bar_ms": bar_ms, "alpha": float(alpha),
            "last_close": _clean(last_close), "has_window": [bool(np.isfinite(v)) for v in last_close],
            "horizons": horizons}


def spread_delta(values: Any, anchors: np.ndarray, rows: int) -> np.ndarray:
    full = np.full(rows, np.nan)
    full[anchors] = np.asarray(values, dtype=float).reshape(-1)
    return full

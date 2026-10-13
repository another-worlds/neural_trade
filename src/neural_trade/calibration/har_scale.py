"""HAR forecast of the absolute return, used as an opt-in conformal interval scale (``CONFORMAL_SCALE: har``).

Tactical finding H85 / H87 (runs/tactical on branch nt-tactical): intervals scaled by a HAR forecast of
``ln |r_H|`` are sharper at the same coverage than intervals scaled by the window's trailing realised
volatility (forward Binance 2025-11..2026-09, T = 60: interval score -9.4 bps against trailing RV).

Model, per horizon of H bars. At a decision bar t (the last input bar; only bars <= t are read):

* features: ``ln RV_w(t)`` for w in 15 / 60 / 240 / 1440 wall-clock minutes (converted to bars by the bar
  size), ``RV_w = mean of squared 1-bar log returns over the last w bars``, plus ``sin`` / ``cos`` of the
  time of day of bar t (24 h period, fractional hours);
* target: ``ln(|ln c[t+H] - ln c[t]| + 1e-5)``;
* fit: ridge (alpha 1, unpenalised intercept) on the standardised features (clipped at +-5), on the TRAINING block's
  decision bars only (never cal / test), rows with the full longest window of history;
* scale: ``exp(prediction) * k * c[t]`` with ``k = mean(|r_H| / exp(prediction))`` on the training rows (so the
  scale is in raw price-delta units and calibrated to the mean absolute return; the conformal quantile of the
  calibration block rescales it anyway).

The fit refuses a training block whose decision bars span less than two longest windows (the long RV and the
time of day would not vary in training and every later block would saturate the clips).

A decision bar with less than the longest window of history (only possible at the start of the data) uses
the history it has. Pure numpy; no state besides the fitted coefficients.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np

logger = logging.getLogger(__name__)

HAR_WINDOWS_MINUTES = (15, 60, 240, 1440)
RIDGE_ALPHA = 1.0
EPS_Y = 1e-5
RV_FLOOR = 1e-12
Y_MARGIN = 1.0  # the forecast of ln|r| is clipped to the training targets' range +- this (a factor e)
Z_CLIP = 5.0   # standardised features are clipped here: a block far outside the training span cannot extrapolate
_MIN_FIT_ROWS = 30
MIN_SPAN_WINDOWS = 2   # the training decision bars must span at least this many longest windows (2 days at 1440 min)
_DAY_NS = 86_400 * 10**9


def _cum_sq_returns(close: np.ndarray) -> np.ndarray:
    """cs[k] = sum of squared 1-bar log returns of bars 1..k (cs[0] = 0)."""
    lc = np.log(np.asarray(close, dtype=float))
    r2 = np.diff(lc) ** 2
    return np.concatenate([[0.0], np.cumsum(r2)])


class HarScale:
    """Per-horizon HAR forecast of the absolute return, fitted on training decision bars only."""

    def __init__(self, bar_minutes: float = 1.0, windows_minutes: Sequence[float] = HAR_WINDOWS_MINUTES) -> None:
        if not bar_minutes or float(bar_minutes) <= 0:
            raise ValueError(f"bar_minutes must be positive, got {bar_minutes!r}")
        self.bar_minutes = float(bar_minutes)
        self.windows_bars = tuple(max(1, int(round(float(w) / self.bar_minutes))) for w in windows_minutes)
        self.models: Dict[int, dict] = {}
        self.clip_fractions: Dict[int, dict] = {}   # last sigma() call per horizon: {'z': share z-clipped, 'y': share y-clipped}

    # ------------------------------------------------------------------ features
    def features(self, close: np.ndarray, times_ns: np.ndarray, decision: np.ndarray) -> np.ndarray:
        """[n, len(windows) + 2] features at the decision bars; reads bars <= each decision bar only."""
        close = np.asarray(close, dtype=float)
        t = np.asarray(decision, dtype=np.int64).reshape(-1)
        if len(t) and (t.min() < 1 or t.max() >= len(close)):
            raise ValueError("decision bars must lie in [1, len(close) - 1]")
        cs = _cum_sq_returns(close)
        cols = []
        for w in self.windows_bars:
            lo = np.maximum(t - w, 0)
            rv = (cs[t] - cs[lo]) / (t - lo)
            cols.append(np.log(np.maximum(rv, RV_FLOOR)))
        tod = (np.asarray(times_ns, dtype="datetime64[ns]").astype("int64")[t] % _DAY_NS) / _DAY_NS * 2.0 * np.pi
        cols += [np.sin(tod), np.cos(tod)]
        return np.column_stack(cols)

    # ------------------------------------------------------------------ fit
    def fit(self, close: np.ndarray, times_ns: np.ndarray, train_decision: np.ndarray,
            horizon_bars: Sequence[int]) -> "HarScale":
        """Fit one ridge per horizon on the TRAINING block's decision bars (the caller passes only those)."""
        close = np.asarray(close, dtype=float)
        t_all = np.asarray(train_decision, dtype=np.int64).reshape(-1)
        full = t_all >= max(self.windows_bars)
        span = int(t_all.max() - t_all.min()) if len(t_all) else 0
        if span < MIN_SPAN_WINDOWS * max(self.windows_bars):
            raise ValueError(
                f"HAR scale: the training block's decision bars span {span} bars "
                f"({span * self.bar_minutes / 1440:.2f} days); the {max(self.windows_bars)}-bar RV and the time-of-day "
                f"features need at least {MIN_SPAN_WINDOWS * max(self.windows_bars)} bars ({MIN_SPAN_WINDOWS} longest windows) "
                f"to vary inside training. Use a longer training block or CONFORMAL_SCALE='realized_vol'")
        lc = np.log(close)
        self.models = {}
        for H in (int(h) for h in horizon_bars):
            t = t_all[full & (t_all + H <= len(close) - 1)]
            if len(t) < _MIN_FIT_ROWS:
                raise ValueError(f"HAR scale: only {len(t)} training rows with {max(self.windows_bars)} bars of "
                                 f"history for horizon {H} (need {_MIN_FIT_ROWS}); use CONFORMAL_SCALE='realized_vol'")
            X = self.features(close, times_ns, t)
            y = np.log(np.abs(lc[t + H] - lc[t]) + EPS_Y)
            mu, sd = X.mean(0), X.std(0) + 1e-12
            Z = np.clip((X - mu) / sd, -Z_CLIP, Z_CLIP)
            ybar = float(y.mean())
            coef = np.linalg.solve(Z.T @ Z + RIDGE_ALPHA * np.eye(Z.shape[1]), Z.T @ (y - ybar))
            lo_y, hi_y = float(y.min()) - Y_MARGIN, float(y.max()) + Y_MARGIN
            e = np.exp(np.clip(Z @ coef + ybar, lo_y, hi_y))
            c = float(np.mean(np.abs(lc[t + H] - lc[t]) / e))
            self.models[H] = {"mu": mu, "sd": sd, "coef": coef, "ic": ybar, "c": c, "y_lo": lo_y, "y_hi": hi_y, "n_train": int(len(t))}
        logger.info("HarScale fitted: " + ", ".join(f"H={H} n={m['n_train']}" for H, m in self.models.items()))
        return self

    # ------------------------------------------------------------------ apply
    def sigma(self, close: np.ndarray, times_ns: np.ndarray, decision: np.ndarray, horizon: int) -> np.ndarray:
        """Scale in raw price-delta units at the decision bars: finite and positive."""
        if int(horizon) not in self.models:
            raise RuntimeError(f"HarScale was not fitted for horizon {horizon}")
        m = self.models[int(horizon)]
        t = np.asarray(decision, dtype=np.int64).reshape(-1)
        Z0 = (self.features(close, times_ns, t) - m["mu"]) / m["sd"]
        Z = np.clip(Z0, -Z_CLIP, Z_CLIP)
        lin = Z @ m["coef"] + m["ic"]
        # diagnostics: the share of rows whose features / forecast hit a clip (high = the block is outside training)
        self.clip_fractions[int(horizon)] = {"z": float(np.mean(np.any(np.abs(Z0) > Z_CLIP, axis=1))) if len(t) else 0.0,
                                             "y": float(np.mean((lin < m["y_lo"]) | (lin > m["y_hi"]))) if len(t) else 0.0}
        s = np.exp(np.clip(lin, m["y_lo"], m["y_hi"])) * m["c"] * np.asarray(close, dtype=float)[t]
        return np.maximum(s, np.finfo(float).tiny)

    def sigma_by_horizon(self, close, times_ns, decision, horizon_steps: Sequence[int],
                         names: Sequence[str] = ("h0", "h1", "h2")) -> Dict[str, np.ndarray]:
        return {n: self.sigma(close, times_ns, decision, int(k)) for n, k in zip(names, horizon_steps)}

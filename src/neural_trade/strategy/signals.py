"""Multi-horizon signals for strategies, vectorised and causal.

``SignalFrame.build(frame, var_scale)`` turns the nine heads of every bar into the features the
notebook strategies used (weighted direction, weighted move, confidence, strength, agreement,
coherence), with three corrections to the notebook versions:

* ``var_scale`` must come from the CALIBRATION split (``var_scale_from(cal_frame)``); the
  notebook took the median variance of the TEST split, using the future to size today's trades.
* The variance-spike detector uses a TRAILING mean (the notebook's ``np.convolve(mode='same')``
  and ``rolling(center=True)`` averaged future bars in).
* Volatility is reported in DOLLARS (sigma = sqrt(var_scaled) * pred_scale); the notebook
  multiplied a scaled sigma by the price, placing stops at about -50% of the price.

Every feature at bar t depends only on predictions at bars <= t (see backtest.assert_no_lookahead).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame

DEFAULT_LAMBDAS = {"h0": 1.0, "h1": 1.0, "h2": 1.0}


def var_scale_from(frame: PredictionFrame) -> float:
    """Median positive predicted variance (scaled units) over a frame - use the CAL frame."""
    v = np.concatenate([frame.variance_scaled[h] for h in HORIZONS])
    v = v[v > 1e-8]
    return float(np.median(v)) if len(v) else 1.0


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    return (c[idx] - c[lo]) / (idx - lo)


@dataclass
class SignalFrame:
    close: np.ndarray            # decision price (the last close) per bar
    p: np.ndarray                # [N, 3] P(up) (calibrated when available)
    delta: np.ndarray            # [N, 3] predicted $ change
    sigma: np.ndarray            # [N, 3] predicted $ sigma
    var_scaled: np.ndarray       # [N, 3]
    confidence: np.ndarray       # [N, 3] exp(-var / var_scale)
    weighted_direction: np.ndarray
    weighted_move: np.ndarray
    volatility: np.ndarray       # confidence-weighted $ sigma
    strength: np.ndarray
    avg_confidence: np.ndarray
    agreement: np.ndarray        # max(votes up, votes down) / 3, votes at 0.55 / 0.45
    consensus: np.ndarray        # +1 UP, -1 DOWN, 0 NEUTRAL
    magnitude_coherent: np.ndarray
    direction_aligned: np.ndarray
    var_spike: np.ndarray        # h1 variance above spike_thresh x its trailing mean
    var_scale: float

    def __len__(self):
        return len(self.close)

    @classmethod
    def build(cls, frame: PredictionFrame, var_scale: float, *, lambdas: Optional[Dict[str, float]] = None,
              calibrated: bool = True, spike_window: int = 20, spike_thresh: float = 2.0) -> "SignalFrame":
        lam = np.array([(lambdas or DEFAULT_LAMBDAS)[h] for h in HORIZONS], dtype=float)
        p = np.stack([frame.prob(h, calibrated) for h in HORIZONS], 1)
        d = np.stack([frame.delta[h] for h in HORIZONS], 1)
        v = np.stack([frame.variance_scaled[h] for h in HORIZONS], 1)
        sig = np.stack([frame.sigma(h) for h in HORIZONS], 1)
        conf = np.exp(-np.clip(v, 0, 1e4) / var_scale) if var_scale > 1e-8 else np.full_like(v, 0.5)
        w = lam[None, :] * conf
        wsum = w.sum(1)
        safe = np.where(wsum < 1e-8, 1.0, wsum)
        wdir = np.where(wsum < 1e-8, 0.5, (w * p).sum(1) / safe)
        wmove = np.where(wsum < 1e-8, 0.0, (w * d).sum(1) / safe)
        vol = (sig * conf).sum(1) / (conf.sum(1) + 1e-8)
        strength = np.clip((w * np.abs(p - 0.5) * 2).sum(1) / (wsum + 1e-8), 0, 1)
        up = (p > 0.55).sum(1)
        down = (p < 0.45).sum(1)
        agreement = np.where(up + down == 0, 1 / 3, np.maximum(up, down) / 3.0)
        consensus = np.sign(up - down).astype(int)
        mag = (np.abs(d[:, 0]) <= np.abs(d[:, 1]) + 1e-6) & (np.abs(d[:, 1]) <= np.abs(d[:, 2]) + 1e-6)
        aligned = ((p > 0.5) == (d > 0)).all(1)
        spike = v[:, 1] > spike_thresh * (_trailing_mean(v[:, 1], spike_window) + 1e-7)
        return cls(frame.last_close.astype(float), p, d, sig, v, conf, wdir, wmove, vol, strength,
                   conf.mean(1), agreement, consensus, mag, aligned, spike, float(var_scale))

    def agreeing_horizons(self, side: int) -> np.ndarray:
        """How many horizons' predicted deltas point to ``side`` (+1 up, -1 down), per bar."""
        return (np.sign(self.delta) == side).sum(1)

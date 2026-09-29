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

Read-outs for the variance-driven strategies (NT-077; docs/research/2026-09-29-strategy-architectures/
README.md sections 2.0 and 5.2), all causal:

* ``sigma_ret`` [N, 3]: the model's $ sigma over the close, a return-unit sigma per horizon.
* ``mu_gauss`` [N, 3]: sigma x Phi^-1(P(up)), the expected $ move a Gaussian with that sigma and
  that P(up) implies (P(up) clipped to [1e-6, 1 - 1e-6]).
* ``sigma_ewma`` [N, 3]: the model-free twin of ``sigma``: an EWMA (half-life ``EWMA_HALFLIFE`` bars)
  of squared one-bar log returns of ``close`` (the anchors are consecutive bars), square-rooted,
  scaled by sqrt(horizon bars) and by the close, in $ like ``sigma``. It is built from the frame
  alone, so the first ``EWMA_WARMUP`` bars of every frame are NaN (every variance strategy waits
  that long, so model and EWMA rows decide on the same bars).

``sigma_for(source, h)`` returns horizon h's $ sigma from ``"model"`` or ``"ewma"``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame

DEFAULT_LAMBDAS = {"h0": 1.0, "h1": 1.0, "h2": 1.0}

# The model-free sigma (``SignalFrame.sigma_ewma``): half-life of the EWMA of squared one-bar log
# returns, and the bars at the start of a frame where it is NaN (4 half-lives: the first bar's weight
# has decayed to 1/16).
EWMA_HALFLIFE = 60
EWMA_WARMUP = 240
SIGMA_SOURCES = ("model", "ewma")
_P_CLIP = 1e-6


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


def ewma_sigma(close: np.ndarray, horizon_steps, *, halflife: int = EWMA_HALFLIFE,
               warmup: int = EWMA_WARMUP) -> np.ndarray:
    """[N, H] $ sigma of each horizon's move from the closes alone (the model-free twin).

    v_t = EWMA (half-life ``halflife``, normalised weights) of r_1^2..r_t with r_i = log(c_i / c_{i-1});
    sigma[t, h] = sqrt(v_t) x sqrt(horizon_steps[h]) x c_t. Trailing only: bar t reads closes <= t.
    NaN for t < ``warmup`` (and at t = 0, which has no return).
    """
    import pandas as pd

    c = np.asarray(close, dtype=float).reshape(-1)
    n = len(c)
    out = np.full((n, len(horizon_steps)), np.nan)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.log(c[1:] / c[:-1]) ** 2
    v = np.empty(n)
    v[0] = np.nan
    v[1:] = pd.Series(r2).ewm(halflife=halflife, adjust=True).mean().to_numpy()
    sd = np.sqrt(v)[:, None] * np.sqrt(np.asarray(horizon_steps, dtype=float))[None, :] * c[:, None]
    sd[: max(int(warmup), 1)] = np.nan
    return sd


# A horizon votes up / down when its P(up) is beyond these lines (SignalFrame.agreement and
# consensus; the coherence figure reads them from here).
VOTE_UP = 0.55
VOTE_DOWN = 0.45

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
    # NT-077 read-outs (module docstring); derived from the fields above in __post_init__ when not given.
    horizon_steps: Tuple[int, ...] = (10, 15, 20)
    sigma_ret: Optional[np.ndarray] = None     # [N, 3] sigma / close
    mu_gauss: Optional[np.ndarray] = None      # [N, 3] sigma x Phi^-1(p), in $
    sigma_ewma: Optional[np.ndarray] = None    # [N, 3] the model-free $ sigma; NaN for the first EWMA_WARMUP bars

    def __post_init__(self):
        from scipy.special import ndtri

        self.horizon_steps = tuple(int(x) for x in self.horizon_steps)
        close = np.asarray(self.close, dtype=float)
        if self.sigma_ret is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                self.sigma_ret = np.asarray(self.sigma, dtype=float) / close[:, None]
        if self.mu_gauss is None:
            p = np.clip(np.asarray(self.p, dtype=float), _P_CLIP, 1.0 - _P_CLIP)
            self.mu_gauss = np.asarray(self.sigma, dtype=float) * ndtri(p)
        if self.sigma_ewma is None:
            self.sigma_ewma = ewma_sigma(close, self.horizon_steps)

    def __len__(self):
        return len(self.close)

    def sigma_for(self, source: str, h: int) -> np.ndarray:
        """Horizon ``h``'s $ sigma per bar: the model's (``"model"``) or the EWMA twin's (``"ewma"``)."""
        if source == "model":
            return self.sigma[:, h]
        if source == "ewma":
            return self.sigma_ewma[:, h]
        raise ValueError(f"sigma source must be one of {SIGMA_SOURCES}, got {source!r}")

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
        up = (p > VOTE_UP).sum(1)
        down = (p < VOTE_DOWN).sum(1)
        agreement = np.where(up + down == 0, 1 / 3, np.maximum(up, down) / 3.0)
        consensus = np.sign(up - down).astype(int)
        mag = (np.abs(d[:, 0]) <= np.abs(d[:, 1]) + 1e-6) & (np.abs(d[:, 1]) <= np.abs(d[:, 2]) + 1e-6)
        aligned = ((p > 0.5) == (d > 0)).all(1)
        spike = v[:, 1] > spike_thresh * (_trailing_mean(v[:, 1], spike_window) + 1e-7)
        return cls(frame.last_close.astype(float), p, d, sig, v, conf, wdir, wmove, vol, strength,
                   conf.mean(1), agreement, consensus, mag, aligned, spike, float(var_scale),
                   tuple(frame.horizon_steps))

    def agreeing_horizons(self, side: int) -> np.ndarray:
        """How many horizons' predicted deltas point to ``side`` (+1 up, -1 down), per bar."""
        return (np.sign(self.delta) == side).sum(1)

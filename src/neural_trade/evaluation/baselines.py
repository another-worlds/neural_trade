"""Baselines every evaluation report compares against (fit on the TRAINING block only).

    zero_delta    persistence: the price does not move (delta = 0)
    mean_delta    the training mean delta per horizon
    class_prior   P(up) = the training up-rate among non-neutral moves
    logreg_lags   logistic regression on trailing log returns over {1,5,10,15,20,30,59} bars and
                  the window's 1-bar volatility - the simplest model that can find the same signal
    const_var     Gaussian with the training mean and a constant training variance (for CRPS,
                  NLL, PIT and interval coverage)

A model earns a claim only where it beats these. zero_delta in particular exposes the
"price-space EV" trap: predicting no change scores an EV near 0.999 on price LEVELS, so price
EV is never reported - only EV on the deltas the model actually predicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.metrics.direction_labels import direction_labels_np

LAGS = (1, 5, 10, 15, 20, 30, 59)
RELEVANT = {
    "zero_delta": ("delta",),
    "mean_delta": ("delta",),
    "class_prior": ("direction",),
    "logreg_lags": ("direction",),
    "const_var": ("variance",),
}


def lag_features(X_raw: np.ndarray, lags=LAGS) -> np.ndarray:
    """Trailing log returns to the last close over each lag, plus 1-bar log-return volatility."""
    X = np.asarray(X_raw, dtype=float)
    logx = np.log(np.maximum(X, 1e-12))
    last = logx[:, -1]
    feats = [last - logx[:, -1 - min(k, X.shape[1] - 1)] for k in lags]
    feats.append(np.std(np.diff(logx, axis=1), axis=1))
    return np.stack(feats, axis=1)


@dataclass
class BaselineSet:
    """Fitted baselines; ``predict(frame)`` returns a PredictionFrame per baseline."""

    mean_delta: np.ndarray
    rms_delta: np.ndarray
    std_delta: np.ndarray
    up_prior: np.ndarray
    logreg: Dict[str, object] = field(default_factory=dict)
    scaler: object = None

    @classmethod
    def fit(cls, X_train_raw, y_train, last_close_train, deadband_bps: float) -> "BaselineSet":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(y_train, dtype=float)
        labels = direction_labels_np(y, last_close_train, deadband_bps)
        prior = np.array([labels[h][0][labels[h][1]].mean() if labels[h][1].any() else 0.5 for h in HORIZONS])
        out = cls(y.mean(0), np.sqrt((y ** 2).mean(0)), y.std(0), prior)
        if X_train_raw is not None:
            F = lag_features(X_train_raw)
            out.scaler = StandardScaler().fit(F)
            Fs = out.scaler.transform(F)
            for h in HORIZONS:
                lab, mask = labels[h]
                if mask.sum() > 50 and 0 < lab[mask].mean() < 1:
                    out.logreg[h] = LogisticRegression(C=1.0, max_iter=1000).fit(Fs[mask], lab[mask])
        return out

    def predict(self, frame: PredictionFrame) -> Dict[str, PredictionFrame]:
        n = len(frame)
        scale = float(frame.pred_scale)
        ones = np.ones(n)

        def pf(delta, prob, sigma, name):
            var = {h: (sigma[i] / scale) ** 2 * ones for i, h in enumerate(HORIZONS)}
            return PredictionFrame(frame.y, frame.last_close, delta, prob, var, scale, frame.pred_mean,
                                   frame.horizon_steps, frame.split, None, None, frame.X_raw,
                                   {"baseline": name})

        half = {h: 0.5 * ones for h in HORIZONS}
        prior = {h: self.up_prior[i] * ones for i, h in enumerate(HORIZONS)}
        zeros = {h: 0.0 * ones for h in HORIZONS}
        means = {h: self.mean_delta[i] * ones for i, h in enumerate(HORIZONS)}
        out = {
            "zero_delta": pf(zeros, half, self.rms_delta, "zero_delta"),
            "mean_delta": pf(means, prior, self.std_delta, "mean_delta"),
            "class_prior": pf(zeros, prior, self.rms_delta, "class_prior"),
            "const_var": pf(means, prior, self.std_delta, "const_var"),
        }
        if self.logreg and frame.X_raw is not None:
            Fs = self.scaler.transform(lag_features(frame.X_raw))
            probs = {h: (self.logreg[h].predict_proba(Fs)[:, 1] if h in self.logreg else prior[h]) for h in HORIZONS}
            out["logreg_lags"] = pf(zeros, probs, self.rms_delta, "logreg_lags")
        return out

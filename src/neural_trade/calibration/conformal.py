"""Distribution-free conformal prediction intervals for price heads.

Conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2023) gives a
finite-sample, model-agnostic guarantee:

    P(y_true in [y_pred - q, y_pred + q]) >= 1 - alpha

for any calibration set of size N, regardless of model quality or distributional
assumptions.  This is strictly stronger than the Gaussian interval
[mu +/- z*sigma], which relies on a correctly specified variance head.

The nonconformity score is the absolute residual in raw (unscaled) price-delta space,
optionally NORMALISED by a per-sample scale u_i known at prediction time (locally adaptive
conformal, Lei et al. 2018):

    s_i = |y_true_raw_i - y_pred_raw_i| / u_i          interval: y_pred +/- q_hat * u

The conformal quantile is:

    q_hat = quantile(s_{1..N}, ceil((1-alpha)*(N+1)/N))

With u = 1 the interval has one width everywhere; after a volatility regime change between
the calibration and test blocks it over- or under-covers everywhere (the M4 gate saw 0.965
coverage at a 0.90 target). ``interval_scale`` provides u: the window's realised volatility
(default, Config.CONFORMAL_SCALE="realized_vol"), the model's own sigma, or none.

Fit one `ConformalRegressor` per output horizon; save/load with joblib.

Usage
-----
    from neural_trade.calibration.conformal import ConformalRegressor

    # On the calibration split (raw price delta, NOT scaled):
    cr_h1 = ConformalRegressor()
    cr_h1.fit(y_true_raw_h1_cal, y_pred_raw_h1_cal)
    cr_h1.save("calibration/conformal_h1.joblib")

    # At inference time:
    cr_h1 = ConformalRegressor.load("calibration/conformal_h1.joblib")
    lo, hi = cr_h1.predict_interval(y_pred_raw_h1_test, alpha=0.1)
    # P(y_true in [lo, hi]) >= 90% by construction
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

HORIZONS = ("h0", "h1", "h2")
SCALE_MODES = ("none", "sigma", "realized_vol")


def interval_scale(mode: str, *, windows=None, variance_scaled=None, pred_scale: Optional[float] = None,
                   horizon_steps: Sequence[int] = (10, 15, 20), n: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Per-sample conformal scale u per horizon (raw price-delta units), known at prediction time.

    * ``"none"``: 1 everywhere (plain split conformal).
    * ``"sigma"``: the model's predicted sigma, ``sqrt(variance_scaled) * pred_scale``.
    * ``"realized_vol"``: std of the input window's 1-bar close changes times sqrt(horizon bars)
      (``windows``: RAW close windows [N, LOOKBACK]).
    """
    if mode not in SCALE_MODES:
        raise ValueError(f"conformal scale must be one of {SCALE_MODES}, got {mode!r}")
    if mode == "none":
        if n is None:
            if windows is not None:
                n = len(windows)
            elif variance_scaled is not None:
                n = len(np.asarray(variance_scaled[HORIZONS[0]]).reshape(-1))
            else:
                raise ValueError("interval_scale('none') needs n, windows or variance_scaled")
        return {h: np.ones(int(n)) for h in HORIZONS}
    if mode == "sigma":
        if variance_scaled is None or pred_scale is None:
            raise ValueError("conformal scale 'sigma' needs variance_scaled and pred_scale")
        return {h: np.sqrt(np.maximum(np.asarray(variance_scaled[h], float).reshape(-1), 0.0)) * float(pred_scale)
                for h in HORIZONS}
    if windows is None:
        raise ValueError("conformal scale 'realized_vol' needs the raw input windows")
    X = np.asarray(windows, dtype=float)
    step_sd = np.diff(X, axis=1).std(axis=1)
    return {h: step_sd * np.sqrt(float(k)) for h, k in zip(HORIZONS, horizon_steps)}


class ConformalRegressor:
    """Conformal prediction interval for a single output horizon.

    Parameters
    ----------
    scores : sorted nonconformity scores from the calibration set (set by fit).
    """

    def __init__(self) -> None:
        self._scores: np.ndarray = np.array([])
        self.normalized = False
        self.scale_floor = 0.0

    def _scale(self, scale, n):
        if not self.normalized:
            if scale is not None:
                raise ValueError("this regressor was fit without a scale; do not pass one")
            return np.ones(n)
        if scale is None:
            raise ValueError("this regressor was fit with a per-sample scale; pass scale= to predict_interval")
        u = np.asarray(scale, dtype=float).reshape(-1)
        if len(u) != n:
            raise ValueError(f"scale has {len(u)} entries for {n} predictions")
        return np.maximum(u, self.scale_floor)

    def fit(
        self,
        y_true_raw: np.ndarray,
        y_pred_raw: np.ndarray,
        scale: Optional[np.ndarray] = None,
    ) -> "ConformalRegressor":
        """Compute and store nonconformity scores from a calibration split.

        Both arrays must be in the same (unscaled) price-delta units produced
        by `target_scaler.inverse_transform(...)`.

        Parameters
        ----------
        y_true_raw : realized price deltas, shape [N]
        y_pred_raw : predicted price deltas, shape [N]
        scale      : optional per-sample scale u_i > 0 (normalised conformal); its floor is set
                     to 1% of the calibration median so a flat window cannot produce a zero width
        """
        y_true = np.asarray(y_true_raw, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred_raw, dtype=float).reshape(-1)
        n = min(len(y_true), len(y_pred))
        self.normalized = scale is not None
        if self.normalized:
            u = np.asarray(scale, dtype=float).reshape(-1)[:n]
            self.scale_floor = max(1e-12, 0.01 * float(np.median(u)))
            u = np.maximum(u, self.scale_floor)
        else:
            u = np.ones(n)
        self._scores = np.sort(np.abs(y_true[:n] - y_pred[:n]) / u)
        print(f"ConformalRegressor: fitted on {n} samples, "
              f"median score = {np.median(self._scores):.4f}")
        return self

    def predict_interval(
        self,
        y_pred_raw: np.ndarray,
        alpha: float = 0.1,
        scale: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) conformal prediction intervals.

        Guaranteed marginal coverage: P(y_true in [lo, hi]) >= 1 - alpha.

        Parameters
        ----------
        y_pred_raw : predicted price deltas, shape [M]
        alpha      : miscoverage level (0.1 => 90% coverage guarantee)

        Returns
        -------
        lo, hi : arrays of shape [M] in the same units as y_pred_raw
        """
        if len(self._scores) == 0:
            raise RuntimeError("Call fit() before predict_interval().")

        N = len(self._scores)
        # Adjusted quantile level: ceil((1-alpha)*(N+1)) / N
        idx = int(np.ceil((1.0 - alpha) * (N + 1)))
        idx = min(idx, N) - 1  # clamp to valid index, convert to 0-based
        q_hat = float(self._scores[idx])

        y_pred = np.asarray(y_pred_raw, dtype=float).reshape(-1)
        half = q_hat * self._scale(scale, len(y_pred))
        return y_pred - half, y_pred + half

    @property
    def n_calibration(self) -> int:
        return len(self._scores)

    def empirical_quantile(self, alpha: float = 0.1) -> float:
        """Return the conformal quantile for a given alpha."""
        if len(self._scores) == 0:
            raise RuntimeError("Call fit() first.")
        N = len(self._scores)
        idx = min(int(np.ceil((1.0 - alpha) * (N + 1))), N) - 1
        return float(self._scores[idx])

    def save(self, path: str) -> None:
        """Serialise to a joblib file."""
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"scores": self._scores, "normalized": self.normalized, "scale_floor": self.scale_floor}, path)
        print(f"ConformalRegressor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ConformalRegressor":
        """Load from a joblib file."""
        import joblib
        obj = cls()
        data = joblib.load(path)
        obj._scores = np.sort(data["scores"])
        obj.normalized = bool(data.get("normalized", False))
        obj.scale_floor = float(data.get("scale_floor", 0.0))
        return obj

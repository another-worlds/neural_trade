"""Distribution-free conformal prediction intervals for price heads.

Conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2023) gives a
finite-sample, model-agnostic guarantee:

    P(y_true in [y_pred - q, y_pred + q]) >= 1 - alpha

for any calibration set of size N, regardless of model quality or distributional
assumptions.  This is strictly stronger than the Gaussian interval
[mu ± z*sigma], which relies on a correctly specified variance head.

The nonconformity score used here is the absolute residual in raw (unscaled)
price-delta space:

    s_i = |y_true_raw_i - y_pred_raw_i|

The conformal quantile is:

    q_hat = quantile(s_{1..N}, ceil((1-alpha)*(N+1)/N))

Fit one `ConformalRegressor` per output horizon; save/load with joblib.

Usage
-----
    from calibration.conformal import ConformalRegressor

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
from typing import Tuple

import numpy as np


class ConformalRegressor:
    """Conformal prediction interval for a single output horizon.

    Parameters
    ----------
    scores : sorted nonconformity scores from the calibration set (set by fit).
    """

    def __init__(self) -> None:
        self._scores: np.ndarray = np.array([])

    def fit(
        self,
        y_true_raw: np.ndarray,
        y_pred_raw: np.ndarray,
    ) -> "ConformalRegressor":
        """Compute and store nonconformity scores from a calibration split.

        Both arrays must be in the same (unscaled) price-delta units produced
        by `target_scaler.inverse_transform(...)`.

        Parameters
        ----------
        y_true_raw : realized price deltas, shape [N]
        y_pred_raw : predicted price deltas, shape [N]
        """
        y_true = np.asarray(y_true_raw, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred_raw, dtype=float).reshape(-1)
        n = min(len(y_true), len(y_pred))
        self._scores = np.sort(np.abs(y_true[:n] - y_pred[:n]))
        print(f"ConformalRegressor: fitted on {n} samples, "
              f"median score = {np.median(self._scores):.4f}")
        return self

    def predict_interval(
        self,
        y_pred_raw: np.ndarray,
        alpha: float = 0.1,
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
        lo = y_pred - q_hat
        hi = y_pred + q_hat
        return lo, hi

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
        joblib.dump({"scores": self._scores}, path)
        print(f"ConformalRegressor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ConformalRegressor":
        """Load from a joblib file."""
        import joblib
        obj = cls()
        data = joblib.load(path)
        obj._scores = np.sort(data["scores"])
        return obj

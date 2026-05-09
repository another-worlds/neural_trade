"""Per-horizon temperature scaling for direction heads.

Temperature scaling (Guo et al., 2017) is a single-parameter post-hoc
calibration method.  For a sigmoid head that outputs p = sigmoid(z), the
calibrated probability is:

    p_cal = sigmoid(z / T)

where T is a learnable scalar fit by minimising the negative log-likelihood
on a held-out calibration set.  T > 1 softens an overconfident head;
T < 1 sharpens an underconfident one.

One T is fit independently per output horizon (h0, h1, h2), so the scaler
does not interfere with cross-horizon coherence.

Usage
-----
    from calibration.temperature_scaling import TemperatureScaler

    scaler = TemperatureScaler()
    # probs_h* : np.ndarray of shape [N] with values in (0, 1)
    # labels_h* : np.ndarray of shape [N] with binary values {0, 1}
    scaler.fit(probs_h0, labels_h0, probs_h1, labels_h1, probs_h2, labels_h2)
    scaler.save("calibration/temperature_params.json")

    # At inference time:
    scaler = TemperatureScaler.load("calibration/temperature_params.json")
    p_cal_h1 = scaler.calibrate(probs_h1, horizon="h1")
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Inverse sigmoid (logit), numerically safe."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary negative log-likelihood (mean over samples)."""
    p = _sigmoid(logits)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def _fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 500,
    lr: float = 0.05,
) -> float:
    """Fit a temperature scalar T by gradient descent on NLL.

    Parameters
    ----------
    probs  : predicted probabilities in (0, 1), shape [N]
    labels : binary ground-truth labels {0, 1}, shape [N]
    n_steps: number of gradient-descent iterations
    lr     : step size

    Returns
    -------
    T : fitted temperature scalar (float, > 0)
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if len(probs) == 0 or len(labels) == 0:
        return 1.0

    base_logits = _logit(probs)  # z = logit(p)
    T = 1.0
    best_T, best_nll = T, float("inf")

    for _ in range(n_steps):
        scaled_logits = base_logits / T
        p_cal = _sigmoid(scaled_logits)
        p_cal = np.clip(p_cal, 1e-7, 1.0 - 1e-7)

        # Gradient of NLL w.r.t. T:
        # NLL = -mean[y*log(p) + (1-y)*log(1-p)]  with p = sigmoid(z/T)
        # dNLL/dT = mean[ (p - y) * (-z / T^2) ]
        grad = np.mean((p_cal - labels) * (-base_logits / (T ** 2)))
        T = T - lr * grad
        T = max(T, 1e-3)  # prevent collapse to zero or negative

        nll = _nll(scaled_logits, labels)
        if nll < best_nll:
            best_nll, best_T = nll, T

    return float(best_T)


class TemperatureScaler:
    """Post-hoc per-horizon temperature scaler for direction heads.

    Parameters
    ----------
    temperatures : dict mapping "h0"/"h1"/"h2" to float T values.
                   Defaults to T=1.0 (identity) for all horizons.
    """

    HORIZONS = ("h0", "h1", "h2")

    def __init__(self, temperatures: Optional[Dict[str, float]] = None):
        self.temperatures: Dict[str, float] = dict(temperatures or {h: 1.0 for h in self.HORIZONS})

    def fit(
        self,
        probs_h0: np.ndarray,
        labels_h0: np.ndarray,
        probs_h1: np.ndarray,
        labels_h1: np.ndarray,
        probs_h2: np.ndarray,
        labels_h2: np.ndarray,
        n_steps: int = 500,
        lr: float = 0.05,
    ) -> "TemperatureScaler":
        """Fit one temperature per horizon on a calibration set.

        The calibration set should be held out from both training and test data
        (e.g. the last N days of the training window before the test split).
        With BATCH_SIZE=1440 a 30-day window (≈43 200 samples) is sufficient.

        Parameters
        ----------
        probs_h* : predicted direction probabilities in (0,1) from dir_h* head
        labels_h*: binary realized direction labels {0,1}
        """
        data = [
            ("h0", probs_h0, labels_h0),
            ("h1", probs_h1, labels_h1),
            ("h2", probs_h2, labels_h2),
        ]
        for h, probs, labels in data:
            T = _fit_temperature(probs, labels, n_steps=n_steps, lr=lr)
            self.temperatures[h] = T
            print(f"  TemperatureScaler [{h}]: T = {T:.4f}")
        return self

    def calibrate(self, probs: np.ndarray, horizon: str = "h1") -> np.ndarray:
        """Apply temperature scaling to raw direction probabilities.

        Parameters
        ----------
        probs   : raw sigmoid probabilities in (0, 1), shape [N]
        horizon : one of "h0", "h1", "h2"

        Returns
        -------
        np.ndarray of calibrated probabilities in (0, 1), shape [N]
        """
        T = self.temperatures.get(horizon, 1.0)
        if abs(T - 1.0) < 1e-6:
            return np.asarray(probs, dtype=float)
        logits = _logit(np.asarray(probs, dtype=float))
        return _sigmoid(logits / T)

    def save(self, path: str) -> None:
        """Serialise temperatures to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"temperatures": self.temperatures}, f, indent=2)
        print(f"TemperatureScaler saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        """Load temperatures from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(temperatures=data["temperatures"])

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
    from neural_trade.calibration.temperature_scaling import TemperatureScaler

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

import logging

import json
import os
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


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


# Search bounds for T. A fit within _BOUND_TOL (in log T) of either bound did not find an interior
# minimum: the head is flat (T at the upper bound: no usable signal) or wildly under-confident
# (lower bound), and the result is flagged rather than reported as calibrated.
T_MIN = 1e-2
T_MAX = 1e3
_BOUND_TOL = 1e-3

STATUS_OK = "ok"
STATUS_LOWER = "lower_bound"
STATUS_UPPER = "upper_bound"


def _fit_temperature_status(probs: np.ndarray, labels: np.ndarray) -> tuple:
    """(T, status): T minimises the NLL of sigmoid(logit(p) / T) over [T_MIN, T_MAX].

    The NLL is convex in 1 / T, so it is unimodal in log T and a bounded scalar search on log T
    reaches the minimum. Deterministic. status is "ok", "lower_bound" or "upper_bound".
    """
    from scipy.optimize import minimize_scalar

    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if len(probs) == 0 or len(labels) == 0:
        return 1.0, STATUS_OK
    z = _logit(probs)
    lo, hi = float(np.log(T_MIN)), float(np.log(T_MAX))
    res = minimize_scalar(lambda lt: _nll(z / np.exp(lt), labels), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-10, "maxiter": 500})
    lt = float(res.x)
    # Brent's bounded method never evaluates the end points; compare them explicitly.
    for cand in (lo, hi):
        if _nll(z / np.exp(cand), labels) < _nll(z / np.exp(lt), labels):
            lt = cand
    status = STATUS_OK
    if lt <= lo + _BOUND_TOL:
        status = STATUS_LOWER
    elif lt >= hi - _BOUND_TOL:
        status = STATUS_UPPER
    return float(np.exp(lt)), status


def _fit_temperature(probs: np.ndarray, labels: np.ndarray, n_steps: int = 0, lr: float = 0.0) -> float:
    """Fit a temperature scalar T by minimising the NLL (n_steps and lr are ignored, kept for callers)."""
    return _fit_temperature_status(probs, labels)[0]


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
        # Per horizon: "ok", or "lower_bound" / "upper_bound" when the fit sits at a search bound.
        self.fit_status: Dict[str, str] = {h: STATUS_OK for h in self.temperatures}

    def at_bound(self) -> Dict[str, str]:
        """Horizons whose fitted T sits at a search bound, with which bound."""
        return {h: st for h, st in self.fit_status.items() if st != STATUS_OK}

    def fit(
        self,
        probs_h0: np.ndarray,
        labels_h0: np.ndarray,
        probs_h1: np.ndarray,
        labels_h1: np.ndarray,
        probs_h2: np.ndarray,
        labels_h2: np.ndarray,
        n_steps: int = 0,
        lr: float = 0.0,
    ) -> "TemperatureScaler":
        """Fit one temperature per horizon on a calibration set.

        The calibration set should be held out from both training and test data
        (e.g. the last N days of the training window before the test split).
        With BATCH_SIZE=1440 a 30-day window (≈43 200 samples) is sufficient.

        Parameters
        ----------
        probs_h* : predicted direction probabilities in (0,1) from dir_h* head
        labels_h*: binary realized direction labels {0,1}
        n_steps, lr: ignored (the fit is a bounded scalar search, not gradient descent)
        """
        data = [
            ("h0", probs_h0, labels_h0),
            ("h1", probs_h1, labels_h1),
            ("h2", probs_h2, labels_h2),
        ]
        for h, probs, labels in data:
            T, status = _fit_temperature_status(probs, labels)
            self.temperatures[h] = T
            self.fit_status[h] = status
            logger.info(f"  TemperatureScaler [{h}]: T = {T:.4f}")
            if status != STATUS_OK:
                logger.warning(f"  TemperatureScaler [{h}]: T = {T:.4g} is at the {status.replace('_', ' ')} "
                               f"[{T_MIN:g}, {T_MAX:g}]: the NLL has no interior minimum on this block")
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
            json.dump({"temperatures": self.temperatures, "fit_status": self.fit_status}, f, indent=2)
        logger.info(f"TemperatureScaler saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        """Load temperatures from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        obj = cls(temperatures=data["temperatures"])
        obj.fit_status.update(data.get("fit_status", {}))
        return obj

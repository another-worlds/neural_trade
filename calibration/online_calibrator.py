"""Online adaptive temperature scaling for live inference.

A static temperature fitted on historical calibration data drifts as market
volatility regimes change.  This module updates T_h incrementally after each
realized trade outcome, tracking the current calibration state without
requiring a full re-fit.

Algorithm
---------
For each new (predicted probability p, realized binary label y) pair:

1. Compute the NLL gradient w.r.t. T:
       g = (sigmoid(logit(p) / T) - y) * (-logit(p) / T^2)
2. Take a gradient step: T ← T - lr * g
3. Apply EMA smoothing: T_ema ← decay * T_ema + (1 - decay) * T
4. Clamp T to [T_min, T_max] to prevent degenerate values.

The calibrated probability used for trading is always sigmoid(logit(p) / T_ema).

Usage
-----
    from calibration.online_calibrator import OnlineTemperatureCalibrator

    # Initialise from saved offline temperatures (or with defaults):
    calib = OnlineTemperatureCalibrator.from_file("calibration/temperature_params.json")

    # At each new bar, before signal fusion:
    p_cal_h1 = calib.calibrate(raw_p_h1, horizon="h1")

    # After a trade closes with realized label y in {0, 1}:
    calib.update(raw_p_h1_at_entry, realized_label_y, horizon="h1")
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np


def _logit(p: float, eps: float = 1e-7) -> float:
    p = max(min(float(p), 1.0 - eps), eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


class OnlineTemperatureCalibrator:
    """Per-horizon online temperature scaler.

    Parameters
    ----------
    temperatures : initial T per horizon ("h0", "h1", "h2")
    lr           : gradient step size for temperature update
    ema_decay    : EMA smoothing factor (higher = slower adaptation)
    T_min / T_max: clamp range to prevent degenerate temperatures
    """

    HORIZONS = ("h0", "h1", "h2")

    def __init__(
        self,
        temperatures: Optional[Dict[str, float]] = None,
        lr: float = 0.05,
        ema_decay: float = 0.98,
        T_min: float = 0.1,
        T_max: float = 10.0,
    ) -> None:
        self.lr = lr
        self.ema_decay = ema_decay
        self.T_min = T_min
        self.T_max = T_max

        init = temperatures or {h: 1.0 for h in self.HORIZONS}
        # Maintain two copies: raw (updated each step) and EMA-smoothed (used for inference)
        self._T: Dict[str, float] = {h: float(init.get(h, 1.0)) for h in self.HORIZONS}
        self._T_ema: Dict[str, float] = dict(self._T)
        self._n_updates: Dict[str, int] = {h: 0 for h in self.HORIZONS}

    def update(self, prob: float, label: float, horizon: str = "h1") -> None:
        """Incorporate one new (prediction, outcome) pair.

        Parameters
        ----------
        prob    : raw sigmoid probability from dir_h* head (before calibration)
        label   : realized binary outcome {0, 1}
        horizon : "h0", "h1", or "h2"
        """
        T = self._T[horizon]
        z = _logit(prob)
        # Calibrated sigmoid at current T
        p_cal = _sigmoid(z / T)
        # dNLL/dT = (p_cal - y) * (-z / T^2)
        grad = (p_cal - float(label)) * (-z / (T ** 2))
        T_new = T - self.lr * grad
        T_new = float(np.clip(T_new, self.T_min, self.T_max))
        self._T[horizon] = T_new
        # EMA smoothing prevents rapid drift from single noisy outcomes
        self._T_ema[horizon] = (
            self.ema_decay * self._T_ema[horizon] + (1.0 - self.ema_decay) * T_new
        )
        self._n_updates[horizon] += 1

    def calibrate(self, prob: float, horizon: str = "h1") -> float:
        """Return the calibrated probability using the current EMA temperature.

        Parameters
        ----------
        prob    : raw sigmoid probability in (0, 1)
        horizon : "h0", "h1", or "h2"

        Returns
        -------
        Calibrated probability in (0, 1)
        """
        T = self._T_ema[horizon]
        if abs(T - 1.0) < 1e-6:
            return float(prob)
        return _sigmoid(_logit(prob) / T)

    def calibrate_array(self, probs: np.ndarray, horizon: str = "h1") -> np.ndarray:
        """Vectorised calibration over a batch of probabilities."""
        T = self._T_ema[horizon]
        if abs(T - 1.0) < 1e-6:
            return np.asarray(probs, dtype=float)
        eps = 1e-7
        p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))
        return 1.0 / (1.0 + np.exp(-logits / T))

    @property
    def state(self) -> Dict[str, Dict[str, float]]:
        return {
            h: {"T": self._T[h], "T_ema": self._T_ema[h], "n_updates": self._n_updates[h]}
            for h in self.HORIZONS
        }

    def save(self, path: str) -> None:
        """Save current state to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "temperatures": self._T_ema,   # save EMA for continuity
            "lr": self.lr,
            "ema_decay": self.ema_decay,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "n_updates": self._n_updates,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "OnlineTemperatureCalibrator":
        """Initialise from a saved temperature file (offline or online format)."""
        with open(path) as f:
            data = json.load(f)
        temperatures = data.get("temperatures", {h: 1.0 for h in cls.HORIZONS})
        lr = kwargs.pop("lr", data.get("lr", 0.05))
        ema_decay = kwargs.pop("ema_decay", data.get("ema_decay", 0.98))
        T_min = kwargs.pop("T_min", data.get("T_min", 0.1))
        T_max = kwargs.pop("T_max", data.get("T_max", 10.0))
        obj = cls(temperatures=temperatures, lr=lr, ema_decay=ema_decay, T_min=T_min, T_max=T_max)
        # Restore update counts if available
        for h, n in data.get("n_updates", {}).items():
            obj._n_updates[h] = int(n)
        return obj

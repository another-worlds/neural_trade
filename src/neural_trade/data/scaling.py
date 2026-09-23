"""Target scaling and window normalisation (fit on TRAIN only).

* :func:`fit_target_scaler` - one StandardScaler over all horizons' raw deltas.
* :class:`WindowNormalizer` - how input windows are put on a common scale:
    ``window_relative`` (default, S18): ``(x - last_close) / target_scale``. Parameter-free
      apart from the target scale; puts window, last close, trends, price heads and sigma in
      one unit system and is invariant to the price level.
    ``per_lag_standard`` (legacy, pre-S18): a StandardScaler per lag position fit on the
      training windows' absolute price LEVEL; kept only to reproduce old runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

WINDOW_NORMALIZERS = ("window_relative", "per_lag_standard")


def fit_target_scaler(y_train: np.ndarray) -> StandardScaler:
    """StandardScaler fit on the pooled training deltas of every horizon."""
    y = np.asarray(y_train)
    return StandardScaler().fit(y.reshape(-1, 1))


def transform_targets(scaler: StandardScaler, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    return scaler.transform(y.reshape(-1, 1)).reshape(y.shape)


@dataclass
class WindowNormalizer:
    kind: str = "window_relative"
    scale: float = 1.0
    per_lag: Optional[StandardScaler] = None

    @classmethod
    def fit(cls, kind: str, X_train: np.ndarray, target_scaler: StandardScaler) -> "WindowNormalizer":
        if kind not in WINDOW_NORMALIZERS:
            raise ValueError(f"unknown window normaliser {kind!r}; choose from {WINDOW_NORMALIZERS}")
        scale = float(target_scaler.scale_[0]) if float(target_scaler.scale_[0]) > 0 else 1.0
        if kind == "per_lag_standard":
            return cls(kind, scale, StandardScaler().fit(np.asarray(X_train)))
        return cls(kind, scale)

    def transform(self, X: np.ndarray, last_close: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.kind == "per_lag_standard":
            return self.per_lag.transform(X).astype("float32")
        return ((X - np.asarray(last_close)[:, None]) / self.scale).astype("float32")

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "scale": self.scale}
        if self.per_lag is not None:
            d["per_lag_mean"] = self.per_lag.mean_.tolist()
            d["per_lag_scale"] = self.per_lag.scale_.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WindowNormalizer":
        per_lag = None
        if d.get("per_lag_mean") is not None:
            per_lag = StandardScaler()
            per_lag.mean_ = np.asarray(d["per_lag_mean"])
            per_lag.scale_ = np.asarray(d["per_lag_scale"])
            per_lag.var_ = per_lag.scale_ ** 2
            per_lag.n_features_in_ = len(per_lag.mean_)
        return cls(d["kind"], float(d["scale"]), per_lag)

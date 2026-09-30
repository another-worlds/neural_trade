"""Target scaling and window normalisation (fit on TRAIN only).

* :func:`fit_target_scaler` - one StandardScaler over all horizons' raw deltas.
* :class:`WindowNormalizer` - how input windows are put on a common scale:
    ``window_relative`` (default, S18): ``(x - last_close) / target_scale``. Parameter-free
      apart from the target scale; puts window, last close, trends, price heads and sigma in
      one unit system and is invariant to the price level.
    ``per_lag_standard`` (legacy, pre-S18): a StandardScaler per lag position fit on the
      training windows' absolute price LEVEL; kept only to reproduce old runs (close-only).

Multi-series windows (NT-047, ``Config.INPUT_SERIES``): the open / high / low / close
channels are window-relative like the close-only input ((x - last_close) / target_scale,
so price differences across channels stay in target-scaler units); the VOLUME channel is
divided by the train-set mean volume (``vol_scale``, fit on the TRAIN windows only). A
pure positive scale keeps volume non-negative and keeps volume RATIOS exact - the
volume-weighted indicator families (OBV, VWAP, MFI) read the volume only through signs and
ratios, which a shifted z-score (negative "volumes") would corrupt; heavy-tailed spikes
reach the model only through those bounded ratios and the meta pooling.
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
    # multi-series windows (NT-047): the channel names, and the train-mean volume scale
    input_series: Optional[tuple] = None
    vol_scale: float = 1.0

    @classmethod
    def fit(cls, kind: str, X_train: np.ndarray, target_scaler: StandardScaler,
            *, input_series=None) -> "WindowNormalizer":
        """``X_train`` is ``[N, L]`` (close-only) or ``[N, L, C]`` with ``input_series``
        naming the C channels; the volume scale is its train-mean (1.0 when degenerate)."""
        if kind not in WINDOW_NORMALIZERS:
            raise ValueError(f"unknown window normaliser {kind!r}; choose from {WINDOW_NORMALIZERS}")
        scale = float(target_scaler.scale_[0]) if float(target_scaler.scale_[0]) > 0 else 1.0
        X_train = np.asarray(X_train)
        if X_train.ndim == 3:
            if kind == "per_lag_standard":
                raise ValueError("per_lag_standard is the legacy close-only normaliser; "
                                 "it does not support multi-series windows (INPUT_SERIES)")
            series = tuple(input_series or ())
            if len(series) != X_train.shape[-1] or "close" not in series:
                raise ValueError(f"input_series {series!r} does not match the window's "
                                 f"{X_train.shape[-1]} channels (or lacks 'close')")
            vol_scale = 1.0
            if "volume" in series:
                mean_vol = float(np.mean(X_train[..., series.index("volume")]))
                vol_scale = mean_vol if np.isfinite(mean_vol) and mean_vol > 0 else 1.0
            return cls(kind, scale, None, series, vol_scale)
        if kind == "per_lag_standard":
            return cls(kind, scale, StandardScaler().fit(X_train))
        return cls(kind, scale)

    def transform(self, X: np.ndarray, last_close: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 3:
            if not self.input_series:
                raise ValueError("this normaliser was fit on close-only windows; it cannot "
                                 "transform multi-series windows")
            lc = np.asarray(last_close)[:, None]  # float32, like the close-only path
            out = np.empty(X.shape, dtype="float32")
            for j, name in enumerate(self.input_series):
                if name == "volume":
                    out[..., j] = X[..., j] / self.vol_scale
                else:
                    out[..., j] = (X[..., j] - lc) / self.scale
            return out
        if self.kind == "per_lag_standard":
            return self.per_lag.transform(X).astype("float32")
        return ((X - np.asarray(last_close)[:, None]) / self.scale).astype("float32")

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "scale": self.scale}
        if self.per_lag is not None:
            d["per_lag_mean"] = self.per_lag.mean_.tolist()
            d["per_lag_scale"] = self.per_lag.scale_.tolist()
        if self.input_series is not None:
            d["input_series"] = list(self.input_series)
            d["vol_scale"] = self.vol_scale
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
        series = tuple(d["input_series"]) if d.get("input_series") else None
        return cls(d["kind"], float(d["scale"]), per_lag, series, float(d.get("vol_scale", 1.0)))

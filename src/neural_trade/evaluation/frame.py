"""PredictionFrame: one split's aligned predictions and outcomes, the input to every evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

HORIZONS = ("h0", "h1", "h2")
NPZ_FORMAT_VERSION = 1          # PredictionFrame.save_npz / load_npz
NPZ_EXTRA_PREFIX = "extra__"


@dataclass
class PredictionFrame:
    """Per-sample arrays for one split, time-ordered.

    ``y``: realised raw deltas [N, 3]; ``delta``/``direction_prob``/``variance_scaled``: model
    heads per horizon (variance in SCALED units, sigma_$ = sqrt(var) * pred_scale);
    ``direction_prob_calibrated`` and ``intervals`` come from the CalibrationPipeline when fitted.
    ``X_raw``: the raw input windows (baselines and backtests use them).
    """

    y: np.ndarray
    last_close: np.ndarray
    delta: Dict[str, np.ndarray]
    direction_prob: Dict[str, np.ndarray]
    variance_scaled: Dict[str, np.ndarray]
    pred_scale: float
    pred_mean: float = 0.0
    horizon_steps: Tuple[int, int, int] = (10, 15, 20)
    split: str = "test"
    direction_prob_calibrated: Optional[Dict[str, np.ndarray]] = None
    intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    X_raw: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float)
        self.last_close = np.asarray(self.last_close, dtype=float).reshape(-1)
        n = len(self.last_close)
        for name in ("delta", "direction_prob", "variance_scaled"):
            d = getattr(self, name)
            setattr(self, name, {h: np.asarray(d[h], dtype=float).reshape(-1)[:n] for h in HORIZONS})
        if self.direction_prob_calibrated is not None:
            self.direction_prob_calibrated = {h: np.asarray(v, float).reshape(-1)[:n]
                                              for h, v in self.direction_prob_calibrated.items()}

    def __len__(self):
        return len(self.last_close)

    def sigma(self, h: str) -> np.ndarray:
        """Predicted standard deviation of the delta, in dollars."""
        return np.sqrt(np.maximum(self.variance_scaled[h], 0.0)) * float(self.pred_scale)

    def gauss_prob(self, h: str, deadband_bps: float) -> np.ndarray:
        from neural_trade.metrics.direction_labels import gaussian_up_prob_given_move_np

        return gaussian_up_prob_given_move_np(self.delta[h], self.variance_scaled[h], self.last_close,
                                              deadband_bps, self.pred_scale)

    def prob(self, h: str, calibrated: bool = True) -> np.ndarray:
        if calibrated and self.direction_prob_calibrated is not None:
            return self.direction_prob_calibrated[h]
        return self.direction_prob[h]

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_result(cls, result, split: str = "test", X_raw: Optional[np.ndarray] = None) -> "PredictionFrame":
        """From a TrainResult: the test split (default) or the calibration split."""
        scale = float(result.target_scaler.scale_[0])
        mean = float(result.target_scaler.mean_[0])
        if split == "test":
            preds, y, lc = result.predictions, result.y_test, result.last_close_test
            cal = result.predictions_calibrated or {}
            if X_raw is None:
                X_raw = getattr(result, "windows_test", None)
        elif split == "cal":
            if result.predictions_cal is None:
                raise ValueError("this TrainResult has no calibration-split predictions")
            preds, y, lc = result.predictions_cal, result.y_cal, result.last_close_cal
            cal = (result.calibration_pipeline.apply(preds, windows=getattr(result, "windows_cal", None))
                   if result.calibration_pipeline is not None else {})
            if X_raw is None:
                X_raw = getattr(result, "windows_cal", None)
        else:
            raise ValueError(f"split must be 'test' or 'cal', got {split!r}")
        # The served delta: shrunk by the calibration pipeline when it fitted a delta scale.
        delta = cal.get("delta") or preds["delta"]
        frame = cls(y, lc, delta, preds["direction_prob"], preds["variance"], scale, mean,
                    tuple(result.config.HORIZON_STEPS), split, cal.get("direction_prob"), cal.get("intervals"),
                    X_raw)
        # evaluate() scores the raw price heads and records the betas from these (report.delta_raw group)
        frame.meta["delta_raw"] = {h: np.asarray(preds["delta"][h], float).reshape(-1)[:len(frame)] for h in HORIZONS}
        betas = getattr(result.calibration_pipeline, "delta_scale", None)
        if cal.get("delta") is not None and betas:
            frame.meta["delta_scale"] = {h: float(betas[h]) for h in HORIZONS if h in betas}
        return frame

    @classmethod
    def from_npz(cls, path, pred_scale: float, pred_mean: float = 0.0, horizon_steps=(10, 15, 20),
                 split: str = "test", X_raw=None) -> "PredictionFrame":
        """From the predictions_*.npz that scripts/gate_run.py writes."""
        z = np.load(Path(path))
        get = lambda kind: {h: z[f"{kind}_{h}"] for h in HORIZONS}  # noqa: E731
        cal = {h: z[f"calibrated_direction_prob_{h}"] for h in HORIZONS
               if f"calibrated_direction_prob_{h}" in z.files} or None
        iv = {h: (z[f"interval90_lo_{h}"], z[f"interval90_hi_{h}"]) for h in HORIZONS
              if f"interval90_lo_{h}" in z.files} or None
        return cls(z["y"], z["last_close"], get("delta"), get("direction_prob"), get("variance"),
                   pred_scale, pred_mean, tuple(horizon_steps), split, cal, iv, X_raw)

    # ------------------------------------------------------------------ stored frames (NT-076)
    def save_npz(self, path, *, extra: Optional[Mapping[str, Any]] = None, overwrite: bool = False) -> Path:
        """Store this frame in a compressed ``.npz`` that :meth:`load_npz` rebuilds exactly.

        Stored: ``y``, ``last_close``, ``pred_scale``, ``pred_mean``, ``horizon_steps``, ``split``
        and, per horizon, the served ``delta``, ``direction_prob``, ``variance_scaled``, the calibrated
        probability and the interval bounds when present, the RAW head delta (``meta["delta_raw"]``,
        before the delta shrinkage) and the shrinkage beta (``meta["delta_scale"]``) when present.
        ``X_raw`` (the input windows) is not stored. ``extra``: further named arrays (for example the
        block's bars), returned by ``load_npz`` in ``meta["extra"]``. The file must not exist yet unless
        ``overwrite`` (the experiment engine never overwrites a run's files).
        """
        horizons = list(self.delta)
        arrays: Dict[str, np.ndarray] = {
            "format_version": np.asarray(NPZ_FORMAT_VERSION), "horizons": np.asarray(horizons, dtype=str),
            "y": self.y, "last_close": self.last_close, "pred_scale": np.asarray(float(self.pred_scale)),
            "pred_mean": np.asarray(float(self.pred_mean)),
            "horizon_steps": np.asarray([int(s) for s in self.horizon_steps], dtype=np.int64),
            "split": np.asarray(str(self.split))}
        raw = self.meta.get("delta_raw") or {}
        betas = self.meta.get("delta_scale") or {}
        for h in horizons:
            arrays[f"delta__{h}"] = self.delta[h]
            arrays[f"direction_prob__{h}"] = self.direction_prob[h]
            arrays[f"variance_scaled__{h}"] = self.variance_scaled[h]
            if self.direction_prob_calibrated is not None and h in self.direction_prob_calibrated:
                arrays[f"direction_prob_calibrated__{h}"] = self.direction_prob_calibrated[h]
            if self.intervals is not None and h in self.intervals:
                lo, hi = self.intervals[h]
                arrays[f"interval_lo__{h}"], arrays[f"interval_hi__{h}"] = np.asarray(lo), np.asarray(hi)
            if h in raw:
                arrays[f"delta_raw__{h}"] = np.asarray(raw[h])
            if h in betas:
                arrays[f"delta_scale__{h}"] = np.asarray(float(betas[h]))
        for name, value in dict(extra or {}).items():
            arrays[f"{NPZ_EXTRA_PREFIX}{name}"] = np.asarray(value)
        path = Path(path)
        with open(path, "wb" if overwrite else "xb") as fh:
            np.savez_compressed(fh, **arrays)
        return path

    @classmethod
    def load_npz(cls, path) -> "PredictionFrame":
        """The frame :meth:`save_npz` stored (``X_raw`` is None; the ``extra`` arrays are in
        ``meta["extra"]``). Not for the ``predictions_*.npz`` of scripts/gate_run.py: see from_npz."""
        with np.load(Path(path), allow_pickle=False) as z:
            files = set(z.files)
            if "format_version" not in files:
                raise ValueError(f"{path}: not a PredictionFrame.save_npz file (no format_version)")
            version = int(z["format_version"])
            if version != NPZ_FORMAT_VERSION:
                raise ValueError(f"{path}: format_version {version}, this code reads {NPZ_FORMAT_VERSION}")
            horizons = [str(h) for h in z["horizons"]]
            per = lambda kind: {h: z[f"{kind}__{h}"] for h in horizons if f"{kind}__{h}" in files}  # noqa: E731
            cal = per("direction_prob_calibrated") or None
            lo, hi = per("interval_lo"), per("interval_hi")
            intervals = {h: (lo[h], hi[h]) for h in lo} or None
            frame = cls(z["y"], z["last_close"], per("delta"), per("direction_prob"), per("variance_scaled"),
                        float(z["pred_scale"]), float(z["pred_mean"]), tuple(int(s) for s in z["horizon_steps"]),
                        str(z["split"]), cal, intervals)
            raw = per("delta_raw")
            if raw:
                frame.meta["delta_raw"] = {h: np.asarray(v, float) for h, v in raw.items()}
            betas = per("delta_scale")
            if betas:
                frame.meta["delta_scale"] = {h: float(v) for h, v in betas.items()}
            frame.meta["extra"] = {k[len(NPZ_EXTRA_PREFIX):]: z[k] for k in z.files if k.startswith(NPZ_EXTRA_PREFIX)}
        return frame

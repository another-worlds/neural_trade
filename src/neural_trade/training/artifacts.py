"""ArtifactBundle: everything needed to serve a trained model, in one directory (plan section B5).

    artifacts/
      weights.h5          base-model weights (the 10-head functional model)
      config.yaml         the exact training configuration
      meta.json           target scale/mean, window normaliser, loss weights (configured,
                          calibrated, final), fold sizes, versions, git sha
      calibration/        the fitted CalibrationPipeline (temperature + conformal), if any

Before this, training persisted only the weights and the target scaler: pred_scale and
pred_mean were never saved, the fitted calibration pipeline was never saved, and there was
no ``joblib.load`` anywhere in the repository - nothing could be served.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from neural_trade import __version__
from neural_trade.core.config import Config
from neural_trade.data.scaling import WindowNormalizer

FORMAT_VERSION = 1


@dataclass
class ArtifactBundle:
    config: Config
    pred_scale: float
    pred_mean: float
    normalizer: WindowNormalizer
    weights_path: Optional[Path] = None
    calibration_pipeline: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ build from a run
    @classmethod
    def from_result(cls, result) -> "ArtifactBundle":
        dp_norm = getattr(result, "normalizer", None)
        scale = float(result.target_scaler.scale_[0])
        mean = float(result.target_scaler.mean_[0])
        normalizer = dp_norm or WindowNormalizer("window_relative", scale if scale > 0 else 1.0)
        meta = {
            "lambda_values_final": result.model.get_lambda_values() if hasattr(result.model, "get_lambda_values") else {},
            "calibration_lambdas": result.calibration_lambdas,
            "calibration_report": result.calibration_report,
            "fold": ({"train": len(result.fold.train), "val": len(result.fold.val), "cal": len(result.fold.cal),
                      "test": len(result.fold.test), "gap": result.fold.gap} if result.fold is not None else None),
            "epochs_run": len(getattr(result.history, "history", {}).get("loss", [])) if result.history else 0,
            # Strategy confidence scale from the CALIBRATION block (never the data being traded).
            "var_scale": _var_scale(result.predictions_cal),
            "weighted_direction_quantiles": _direction_quantiles(result),
        }
        bundle = cls(result.config, scale, mean, normalizer, calibration_pipeline=result.calibration_pipeline,
                     meta=meta)
        bundle._model = result.model
        return bundle

    # ------------------------------------------------------------------ persistence
    def save(self, directory) -> Path:
        from neural_trade.utils.env import fingerprint

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        model = getattr(self, "_model", None)
        if model is not None:
            base = getattr(model, "base_model", None) or model
            base.save_weights(str(d / "weights.h5"))
            self.weights_path = d / "weights.h5"
        self.config.to_yaml(d / "config.yaml")
        meta = dict(self.meta)
        meta.update({
            "format_version": FORMAT_VERSION,
            "neural_trade_version": __version__,
            "pred_scale": self.pred_scale,
            "pred_mean": self.pred_mean,
            "normalizer": self.normalizer.to_dict(),
            "has_calibration": self.calibration_pipeline is not None,
            "env": fingerprint(include_devices=False),
        })
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
        if self.calibration_pipeline is not None:
            self.calibration_pipeline.save(str(d / "calibration"))
        return d

    @classmethod
    def load(cls, directory) -> "ArtifactBundle":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(f"artifact format {meta['format_version']} is newer than this code ({FORMAT_VERSION})")
        config = Config.from_yaml(d / "config.yaml")
        calib = None
        if meta.get("has_calibration") and (d / "calibration").is_dir():
            from neural_trade.calibration import CalibrationPipeline

            calib = CalibrationPipeline.load(str(d / "calibration"))
        weights = d / "weights.h5"
        return cls(config, float(meta["pred_scale"]), float(meta["pred_mean"]),
                   WindowNormalizer.from_dict(meta["normalizer"]), weights if weights.exists() else None, calib, meta)

    def build_model(self):
        """The base functional model with the saved weights (inference only)."""
        from neural_trade.registries.models import Models

        model = Models.build(self.config.MODEL_NAME, self.config)
        if self.weights_path is None:
            raise FileNotFoundError("this bundle has no weights.h5")
        model.load_weights(str(self.weights_path))
        return model


def _var_scale(predictions) -> Optional[float]:
    if not predictions:
        return None
    import numpy as np

    v = np.concatenate([np.asarray(predictions["variance"][h], float).reshape(-1) for h in ("h0", "h1", "h2")])
    v = v[v > 1e-8]
    return float(np.median(v)) if len(v) else None


def _direction_quantiles(result) -> Optional[Dict[str, float]]:
    """Quantiles of the calibration block's confidence-weighted P(up) (strategy thresholds)."""
    if not getattr(result, "predictions_cal", None) or getattr(result, "y_cal", None) is None:
        return None
    import numpy as np

    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.strategy.signals import SignalFrame, var_scale_from

    cal = PredictionFrame.from_result(result, "cal")
    w = SignalFrame.build(cal, var_scale_from(cal)).weighted_direction
    return {str(q): float(np.quantile(w, q)) for q in (0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95)}


def _json_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

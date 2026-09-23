"""Unified CalibrationPipeline — one-stop calibration interface.

Quick start (notebook)
-----------------------
::

    from neural_trade.calibration import CalibrationPipeline

    # After training:
    pipeline = CalibrationPipeline()
    pipeline.fit(result)               # result = TrainResult from train_and_evaluate()
    pipeline.summary()

    # Apply to any predictions dict:
    cal = pipeline.apply(result.predictions)
    # cal['direction_prob']['h1']  — temperature-scaled P(UP)
    # cal['delta']['h1']           — unchanged
    # cal['variance']['h1']        — unchanged
    # cal['intervals']['h1']       — (lo, hi) conformal price-delta bounds (90% coverage)

    # Save / reload:
    pipeline.save("calibration/")
    pipeline = CalibrationPipeline.load("calibration/")

    # Live trading — after each closed trade:
    pipeline.update_online(raw_prob_at_entry, realized_label, horizon='h1')
    p_cal = pipeline.calibrate_online(current_raw_prob, horizon='h1')
"""

from __future__ import annotations

import logging

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from neural_trade.calibration.temperature_scaling import TemperatureScaler
from neural_trade.calibration.conformal import SCALE_MODES, ConformalRegressor, interval_scale
from neural_trade.calibration.online_calibrator import OnlineTemperatureCalibrator

try:
    from neural_trade.metrics.direction_labels import direction_labels_np as _compute_direction_labels_np
except Exception:  # fallback if metrics_utils not on path in some test contexts
    _compute_direction_labels_np = None

logger = logging.getLogger(__name__)

HORIZONS = ("h0", "h1", "h2")
_DEFAULT_DIR = "calibration"


# ---------------------------------------------------------------------------
# Internal helper — mirrors exact label construction used during training
# ---------------------------------------------------------------------------

def _direction_labels(
    y_true_delta_raw: np.ndarray,
    last_close: np.ndarray,
    deadband_bps: float = 0.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Convert raw price-delta matrix to per-horizon (labels, mask) pairs.

    Delegates to the single shared rule (neural_trade.metrics.direction_labels); the
    inline fallback copy was removed.
    """
    if _compute_direction_labels_np is None:  # pragma: no cover - package always importable
        raise ImportError("neural_trade.metrics.direction_labels is required")
    return _compute_direction_labels_np(y_true_delta_raw, last_close, deadband_bps)


# ---------------------------------------------------------------------------
# CalibrationPipeline
# ---------------------------------------------------------------------------

class CalibrationPipeline:
    """Unified post-hoc calibration for the neural_trade direction and price heads.

    Wraps three independent calibrators:

    * **TemperatureScaler** — per-horizon temperature scaling for direction sigmoid heads
    * **ConformalRegressor** (×3) — distribution-free price-delta prediction intervals
    * **OnlineTemperatureCalibrator** — adaptive online temperature for live trading

    All methods accept / return the ``predictions_dict`` format produced by
    ``train_and_evaluate``:
    ``{'delta': {'h0':…,'h1':…,'h2':…}, 'direction_prob': {…}, 'variance': {…}}``
    """

    def __init__(self, conformal_scale: str = "none") -> None:
        if conformal_scale not in SCALE_MODES:
            raise ValueError(f"conformal_scale must be one of {SCALE_MODES}, got {conformal_scale!r}")
        # Per-sample interval scale (see calibration.conformal.interval_scale); "realized_vol"
        # needs the raw input windows at fit and apply time.
        self.conformal_scale = conformal_scale
        self.pred_scale: Optional[float] = None
        self.horizon_steps: Tuple[int, ...] = (10, 15, 20)
        self.temperature_scaler = TemperatureScaler()
        self.conformal: Dict[str, ConformalRegressor] = {
            h: ConformalRegressor() for h in HORIZONS
        }
        self.online: Optional[OnlineTemperatureCalibrator] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        result,
        *,
        split: str = "cal",
        deadband_bps: Optional[float] = None,
        conformal_alpha: float = 0.1,
    ) -> "CalibrationPipeline":
        """Fit all calibrators from a ``TrainResult`` object.

        ``split`` must be ``"cal"``: the dedicated calibration block that ``train_and_evaluate``
        carves out between train and test (``result.predictions_cal`` / ``y_cal`` /
        ``last_close_cal``). Fitting on ``"test"`` is refused - split conformal's finite-sample
        guarantee needs calibration and evaluation samples to be disjoint, and the temperatures
        would otherwise be tuned on the very data used to report performance.
        """
        if split == "test":
            raise ValueError(
                "CalibrationPipeline.fit(split='test') is refused: calibration must be fit on the "
                "calibration block (split='cal'), never on the data used to report metrics."
            )
        if split != "cal":
            raise ValueError(f"unknown split {split!r}; expected 'cal'")
        preds = getattr(result, "predictions_cal", None)
        y_cal = getattr(result, "y_cal", None)
        lc_cal = getattr(result, "last_close_cal", None)
        if preds is None or y_cal is None or lc_cal is None:
            raise ValueError(
                "TrainResult carries no calibration split (predictions_cal / y_cal / last_close_cal). "
                "Re-run train_and_evaluate(fit_calibration=True)."
            )
        if deadband_bps is None:
            deadband_bps = float(getattr(result.config, "DIR_DEADBAND_BPS", 0.0))

        return self.fit_from_arrays(
            predictions_dict=preds,
            y_true_delta_raw=np.asarray(y_cal, dtype=float),
            last_close=np.asarray(lc_cal, dtype=float),
            deadband_bps=deadband_bps,
            conformal_alpha=conformal_alpha,
            windows=getattr(result, "windows_cal", None),
            pred_scale=float(result.target_scaler.scale_[0]),
            horizon_steps=tuple(getattr(result.config, "HORIZON_STEPS", self.horizon_steps)),
        )

    def fit_from_arrays(
        self,
        predictions_dict: Dict,
        y_true_delta_raw: np.ndarray,
        last_close: np.ndarray,
        deadband_bps: float = 0.0,
        conformal_alpha: float = 0.1,
        windows: Optional[np.ndarray] = None,
        pred_scale: Optional[float] = None,
        horizon_steps: Optional[Sequence[int]] = None,
    ) -> "CalibrationPipeline":
        """Fit from raw arrays when a TrainResult is not available.

        Parameters
        ----------
        predictions_dict  : ``result.predictions`` format
        y_true_delta_raw  : array [N, 3] of realized price deltas (raw / inverse-scaled units)
        last_close        : array [N] of last close prices
        deadband_bps      : deadband used during training (match Config.DIR_DEADBAND_BPS)
        conformal_alpha   : miscoverage level for coverage reporting
        windows           : RAW input windows [N, LOOKBACK] (conformal_scale="realized_vol")
        pred_scale        : target-scaler scale (conformal_scale="sigma")
        horizon_steps     : bars per horizon (conformal_scale="realized_vol")
        """
        if pred_scale is not None:
            self.pred_scale = float(pred_scale)
        if horizon_steps is not None:
            self.horizon_steps = tuple(int(k) for k in horizon_steps)
        y = np.asarray(y_true_delta_raw, dtype=float)
        N = len(y)
        logger.info(f"CalibrationPipeline: fitting on {N} samples")

        labels_map = _direction_labels(y, last_close, deadband_bps)

        # ------------------------------------------------------------------
        # 1. Temperature scaling — fit on deadband-filtered samples
        # ------------------------------------------------------------------
        logger.info("\n[1/2] Temperature scaling (direction heads)...")
        h0_lab, h0_mask = labels_map["h0"]
        h1_lab, h1_mask = labels_map["h1"]
        h2_lab, h2_mask = labels_map["h2"]

        p_h0 = np.asarray(predictions_dict["direction_prob"]["h0"], dtype=float)
        p_h1 = np.asarray(predictions_dict["direction_prob"]["h1"], dtype=float)
        p_h2 = np.asarray(predictions_dict["direction_prob"]["h2"], dtype=float)

        # Apply mask — fit only where |return| > deadband
        self.temperature_scaler.fit(
            p_h0[h0_mask], h0_lab[h0_mask],
            p_h1[h1_mask], h1_lab[h1_mask],
            p_h2[h2_mask], h2_lab[h2_mask],
        )

        # ------------------------------------------------------------------
        # 2. Conformal regressors — fit on all samples (no deadband filter)
        # ------------------------------------------------------------------
        logger.info(f"\n[2/2] Conformal regressors (price delta intervals, scale={self.conformal_scale})...")
        scales = self._scales(predictions_dict, windows, N)
        for i, h in enumerate(HORIZONS):
            if i >= y.shape[1]:
                break
            y_true_h = y[:, i]
            y_pred_h = np.asarray(predictions_dict["delta"][h], dtype=float)
            u = scales[h] if scales is not None else None
            self.conformal[h].fit(y_true_h, y_pred_h, scale=u)
            lo, hi = self.conformal[h].predict_interval(y_pred_h, alpha=conformal_alpha, scale=u)
            cov = float(np.mean((y_true_h >= lo) & (y_true_h <= hi)))
            logger.info(f"  [{h}] coverage @ alpha={conformal_alpha:.2f}: {cov:.3f} "
                  f"(target >= {1 - conformal_alpha:.2f}),  mean half-width {np.mean(hi - lo) / 2:.2f} raw units")

        # ------------------------------------------------------------------
        # 3. Online calibrator — warm-start from offline temperatures
        # ------------------------------------------------------------------
        self.online = OnlineTemperatureCalibrator(
            temperatures=self.temperature_scaler.temperatures,
        )

        self._fitted = True
        logger.info("\nCalibrationPipeline: fitting complete.")
        return self

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def _scales(self, predictions_dict, windows, n):
        """Per-horizon conformal scales, or None for plain (unnormalised) conformal."""
        if self.conformal_scale == "none":
            return None
        return interval_scale(self.conformal_scale, windows=windows,
                              variance_scaled=predictions_dict.get("variance"), pred_scale=self.pred_scale,
                              horizon_steps=self.horizon_steps, n=n)

    def apply(
        self,
        predictions_dict: Dict,
        alpha: float = 0.1,
        windows: Optional[np.ndarray] = None,
    ) -> Dict:
        """Apply all calibrators and return a drop-in replacement predictions dict.

        The returned dict has the same structure as the input plus an
        ``'intervals'`` key:

        * ``direction_prob[h]`` — temperature-scaled P(UP), replaces raw values
        * ``delta[h]``          — unchanged (pass-through)
        * ``variance[h]``       — unchanged (pass-through)
        * ``intervals[h]``      — ``(lo, hi)`` conformal price-delta bounds

        Parameters
        ----------
        predictions_dict : raw ``result.predictions``-format dict
        alpha            : conformal miscoverage level (default 0.1 → 90% coverage)
        windows          : RAW input windows, required when conformal_scale="realized_vol"
        """
        self._require_fitted()
        n = len(np.asarray(predictions_dict["delta"][HORIZONS[0]]).reshape(-1))
        scales = self._scales(predictions_dict, windows, n)

        cal: Dict = {
            "delta": {h: np.asarray(predictions_dict["delta"][h]) for h in HORIZONS},
            "variance": {h: np.asarray(predictions_dict["variance"][h]) for h in HORIZONS},
            "direction_prob": {},
            "intervals": {},
        }

        for h in HORIZONS:
            raw_probs = np.asarray(predictions_dict["direction_prob"][h], dtype=float)
            cal["direction_prob"][h] = self.temperature_scaler.calibrate(raw_probs, horizon=h)

            y_pred = np.asarray(predictions_dict["delta"][h], dtype=float)
            cal["intervals"][h] = self.conformal[h].predict_interval(
                y_pred, alpha=alpha, scale=scales[h] if scales is not None else None)

        return cal

    def predict_intervals(
        self,
        predictions_dict: Dict,
        alpha: float = 0.1,
        windows: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return only conformal intervals without scaling direction probs.

        Returns
        -------
        dict mapping 'h0'/'h1'/'h2' to ``(lo, hi)`` arrays in raw price-delta units
        """
        self._require_fitted()
        n = len(np.asarray(predictions_dict["delta"][HORIZONS[0]]).reshape(-1))
        scales = self._scales(predictions_dict, windows, n)
        return {
            h: self.conformal[h].predict_interval(
                np.asarray(predictions_dict["delta"][h], dtype=float),
                alpha=alpha, scale=scales[h] if scales is not None else None,
            )
            for h in HORIZONS
        }

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update_online(self, prob: float, label: float, horizon: str = "h1") -> None:
        """Update the online temperature after a realized trade outcome.

        Call after each closed trade with the direction probability predicted
        at entry and the realized outcome.

        Parameters
        ----------
        prob    : raw direction probability at trade entry, in (0, 1)
        label   : realized binary outcome — 1 for UP, 0 for DOWN
        horizon : 'h0', 'h1', or 'h2'
        """
        self._require_fitted()
        self.online.update(prob, label, horizon)

    def calibrate_online(self, prob: float, horizon: str = "h1") -> float:
        """Apply current online temperature to a single raw probability.

        Lightweight scalar operation — use this in the live trading loop.

        Parameters
        ----------
        prob    : raw direction probability in (0, 1)
        horizon : 'h0', 'h1', or 'h2'

        Returns
        -------
        Calibrated probability in (0, 1)
        """
        self._require_fitted()
        return self.online.calibrate(prob, horizon)

    def calibrate_online_array(self, probs: np.ndarray, horizon: str = "h1") -> np.ndarray:
        """Vectorised version of ``calibrate_online`` for a batch."""
        self._require_fitted()
        return self.online.calibrate_array(probs, horizon)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, directory: str = _DEFAULT_DIR) -> None:
        """Save the full pipeline to *directory*.

        Files written:
        - ``temperature_params.json``  — offline temperature scalers
        - ``conformal_h0.joblib``
        - ``conformal_h1.joblib``
        - ``conformal_h2.joblib``
        - ``online_calibrator.json``   — online temperatures + state
        - ``pipeline_meta.json``       — fitted flag
        """
        os.makedirs(directory, exist_ok=True)
        self.temperature_scaler.save(os.path.join(directory, "temperature_params.json"))
        for h in HORIZONS:
            self.conformal[h].save(os.path.join(directory, f"conformal_{h}.joblib"))
        if self.online is not None:
            self.online.save(os.path.join(directory, "online_calibrator.json"))
        with open(os.path.join(directory, "pipeline_meta.json"), "w") as fh:
            json.dump({"fitted": self._fitted, "conformal_scale": self.conformal_scale,
                       "pred_scale": self.pred_scale, "horizon_steps": list(self.horizon_steps)}, fh, indent=2)
        logger.info(f"CalibrationPipeline saved to '{directory}/'")

    @classmethod
    def load(cls, directory: str = _DEFAULT_DIR) -> "CalibrationPipeline":
        """Load a previously saved pipeline.

        Parameters
        ----------
        directory : path written by ``save()``

        Returns
        -------
        A fully fitted CalibrationPipeline ready for ``apply()``
        """
        obj = cls()
        obj.temperature_scaler = TemperatureScaler.load(
            os.path.join(directory, "temperature_params.json")
        )
        for h in HORIZONS:
            obj.conformal[h] = ConformalRegressor.load(
                os.path.join(directory, f"conformal_{h}.joblib")
            )
        online_path = os.path.join(directory, "online_calibrator.json")
        if os.path.exists(online_path):
            obj.online = OnlineTemperatureCalibrator.from_file(online_path)
        else:
            obj.online = OnlineTemperatureCalibrator(
                temperatures=obj.temperature_scaler.temperatures
            )
        meta_path = os.path.join(directory, "pipeline_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
            obj._fitted = bool(meta.get("fitted", True))
            obj.conformal_scale = meta.get("conformal_scale", "none")
            obj.pred_scale = meta.get("pred_scale")
            obj.horizon_steps = tuple(meta.get("horizon_steps", obj.horizon_steps))
        else:
            obj._fitted = True
        logger.info(f"CalibrationPipeline loaded from '{directory}/'")
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise summary of all fitted calibration parameters."""
        if not self._fitted:
            logger.info("Pipeline is not fitted. Call fit(result) first.")
            return

        logger.info('%s', "=" * 55)
        logger.info("CalibrationPipeline Summary")
        logger.info('%s', "=" * 55)

        logger.info("\nTemperature scaling (direction heads):")
        for h in HORIZONS:
            T = self.temperature_scaler.temperatures.get(h, 1.0)
            if T > 1.05:
                note = "overconfident — softened"
            elif T < 0.95:
                note = "underconfident — sharpened"
            else:
                note = "well-calibrated (T ≈ 1)"
            logger.info(f"  {h}: T = {T:.4f}  ({note})")

        logger.info("\nConformal regressors (price delta intervals):")
        for h in HORIZONS:
            n = self.conformal[h].n_calibration
            if n > 0:
                q90 = self.conformal[h].empirical_quantile(0.1)
                unit = "raw price units" if self.conformal_scale == "none" else f"x {self.conformal_scale} scale"
                logger.info(f"  {h}: N = {n},  90%-quantile = +/-{q90:.4f} {unit}")
            else:
                logger.info(f"  {h}: not fitted")

        if self.online is not None:
            logger.info("\nOnline calibrator (adaptive temperatures):")
            for h, st in self.online.state.items():
                logger.info(f"  {h}: T_ema = {st['T_ema']:.4f},  updates = {st['n_updates']}")

        logger.info('%s', "=" * 55)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Pipeline is not fitted. "
                "Call pipeline.fit(result) or CalibrationPipeline.load(directory) first."
            )

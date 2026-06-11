"""Unified CalibrationPipeline — one-stop calibration interface.

Quick start (notebook)
-----------------------
::

    from calibration import CalibrationPipeline

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

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from calibration.temperature_scaling import TemperatureScaler
from calibration.conformal import ConformalRegressor
from calibration.online_calibrator import OnlineTemperatureCalibrator

try:
    from metrics_utils import compute_direction_labels_np as _compute_direction_labels_np
except Exception:  # fallback if metrics_utils not on path in some test contexts
    _compute_direction_labels_np = None

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

    Delegates to the single shared implementation in metrics_utils to avoid
    duplication with the training loss paths and model.py metrics.
    """
    if _compute_direction_labels_np is not None:
        return _compute_direction_labels_np(y_true_delta_raw, last_close, deadband_bps)
    # Fallback (should rarely be hit): original inline logic
    lc = np.asarray(last_close, dtype=float).reshape(-1)
    y = np.asarray(y_true_delta_raw, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    deadband = deadband_bps / 10_000.0
    result = {}
    for i, h in enumerate(HORIZONS):
        if i >= y.shape[1]:
            break
        ret = y[:, i] / (lc + 1e-12)
        mask = (np.abs(ret) > deadband)
        labels = (ret > deadband).astype(float)
        result[h] = (labels, mask)
    return result


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

    def __init__(self) -> None:
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
        deadband_bps: Optional[float] = None,
        conformal_alpha: float = 0.1,
    ) -> "CalibrationPipeline":
        """Fit all calibrators from a ``TrainResult`` object.

        Uses the held-out test split stored in *result* as the calibration set.
        Call this once, immediately after ``train_and_evaluate``.

        Parameters
        ----------
        result          : TrainResult returned by ``train_and_evaluate``
        deadband_bps    : override config deadband (default: read from result.config)
        conformal_alpha : miscoverage target for coverage reporting (default 0.1 → 90%)
        """
        if deadband_bps is None:
            deadband_bps = float(getattr(result.config, "DIR_DEADBAND_BPS", 0.0))

        return self.fit_from_arrays(
            predictions_dict=result.predictions,
            y_true_delta_raw=np.asarray(result.y_test, dtype=float),
            last_close=np.asarray(result.last_close_test, dtype=float),
            deadband_bps=deadband_bps,
            conformal_alpha=conformal_alpha,
        )

    def fit_from_arrays(
        self,
        predictions_dict: Dict,
        y_true_delta_raw: np.ndarray,
        last_close: np.ndarray,
        deadband_bps: float = 0.0,
        conformal_alpha: float = 0.1,
    ) -> "CalibrationPipeline":
        """Fit from raw arrays when a TrainResult is not available.

        Parameters
        ----------
        predictions_dict  : ``result.predictions`` format
        y_true_delta_raw  : array [N, 3] of realized price deltas (raw / inverse-scaled units)
        last_close        : array [N] of last close prices
        deadband_bps      : deadband used during training (match Config.DIR_DEADBAND_BPS)
        conformal_alpha   : miscoverage level for coverage reporting
        """
        y = np.asarray(y_true_delta_raw, dtype=float)
        N = len(y)
        print(f"CalibrationPipeline: fitting on {N} samples")

        labels_map = _direction_labels(y, last_close, deadband_bps)

        # ------------------------------------------------------------------
        # 1. Temperature scaling — fit on deadband-filtered samples
        # ------------------------------------------------------------------
        print("\n[1/2] Temperature scaling (direction heads)...")
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
        print("\n[2/2] Conformal regressors (price delta intervals)...")
        for i, h in enumerate(HORIZONS):
            if i >= y.shape[1]:
                break
            y_true_h = y[:, i]
            y_pred_h = np.asarray(predictions_dict["delta"][h], dtype=float)
            self.conformal[h].fit(y_true_h, y_pred_h)
            lo, hi = self.conformal[h].predict_interval(y_pred_h, alpha=conformal_alpha)
            cov = float(np.mean((y_true_h >= lo) & (y_true_h <= hi)))
            q = self.conformal[h].empirical_quantile(conformal_alpha)
            print(f"  [{h}] coverage @ α={conformal_alpha:.2f}: {cov:.3f} "
                  f"(target ≥ {1 - conformal_alpha:.2f}),  ±{q:.4f} raw units")

        # ------------------------------------------------------------------
        # 3. Online calibrator — warm-start from offline temperatures
        # ------------------------------------------------------------------
        self.online = OnlineTemperatureCalibrator(
            temperatures=self.temperature_scaler.temperatures,
        )

        self._fitted = True
        print("\nCalibrationPipeline: fitting complete.")
        return self

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def apply(
        self,
        predictions_dict: Dict,
        alpha: float = 0.1,
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
        """
        self._require_fitted()

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
            cal["intervals"][h] = self.conformal[h].predict_interval(y_pred, alpha=alpha)

        return cal

    def predict_intervals(
        self,
        predictions_dict: Dict,
        alpha: float = 0.1,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return only conformal intervals without scaling direction probs.

        Returns
        -------
        dict mapping 'h0'/'h1'/'h2' to ``(lo, hi)`` arrays in raw price-delta units
        """
        self._require_fitted()
        return {
            h: self.conformal[h].predict_interval(
                np.asarray(predictions_dict["delta"][h], dtype=float),
                alpha=alpha,
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
            json.dump({"fitted": self._fitted}, fh, indent=2)
        print(f"CalibrationPipeline saved to '{directory}/'")

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
                obj._fitted = bool(json.load(fh).get("fitted", True))
        else:
            obj._fitted = True
        print(f"CalibrationPipeline loaded from '{directory}/'")
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise summary of all fitted calibration parameters."""
        if not self._fitted:
            print("Pipeline is not fitted. Call fit(result) first.")
            return

        print("=" * 55)
        print("CalibrationPipeline Summary")
        print("=" * 55)

        print("\nTemperature scaling (direction heads):")
        for h in HORIZONS:
            T = self.temperature_scaler.temperatures.get(h, 1.0)
            if T > 1.05:
                note = "overconfident — softened"
            elif T < 0.95:
                note = "underconfident — sharpened"
            else:
                note = "well-calibrated (T ≈ 1)"
            print(f"  {h}: T = {T:.4f}  ({note})")

        print("\nConformal regressors (price delta intervals):")
        for h in HORIZONS:
            n = self.conformal[h].n_calibration
            if n > 0:
                q90 = self.conformal[h].empirical_quantile(0.1)
                print(f"  {h}: N = {n},  90%-quantile = ±{q90:.4f} raw price units")
            else:
                print(f"  {h}: not fitted")

        if self.online is not None:
            print("\nOnline calibrator (adaptive temperatures):")
            for h, st in self.online.state.items():
                print(f"  {h}: T_ema = {st['T_ema']:.4f},  updates = {st['n_updates']}")

        print("=" * 55)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Pipeline is not fitted. "
                "Call pipeline.fit(result) or CalibrationPipeline.load(directory) first."
            )

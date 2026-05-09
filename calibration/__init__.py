"""Post-hoc calibration utilities for the neural_trade model.

Modules
-------
temperature_scaling
    Per-horizon temperature scaling for direction heads.  Fit a scalar T_h
    on a held-out calibration set; apply before signal fusion at inference time.

conformal
    Distribution-free conformal prediction intervals for price heads.  Guarantees
    empirical coverage >= 1-alpha regardless of model quality.

online_calibrator
    Online adaptive temperature scaling for live inference.  Updates T_h after
    each realized trade outcome to track non-stationary market regimes.
"""

from calibration.temperature_scaling import TemperatureScaler
from calibration.conformal import ConformalRegressor
from calibration.online_calibrator import OnlineTemperatureCalibrator
from calibration.pipeline import CalibrationPipeline

__all__ = [
    "TemperatureScaler",
    "ConformalRegressor",
    "OnlineTemperatureCalibrator",
    "CalibrationPipeline",
]

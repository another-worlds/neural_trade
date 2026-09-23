"""Compatibility shim: ``calibration`` moved to ``neural_trade.calibration`` (B10/B13; removed in B18)."""
from neural_trade._shim import alias_package

alias_package(__name__, "neural_trade.calibration",
              ("conformal", "online_calibrator", "pipeline", "temperature_scaling"))

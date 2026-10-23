"""Notebook front-ends: interactive controls whose logic lives here, so the notebooks stay thin.

    from neural_trade.notebook import TrainingSession, BacktestExplorer, CalibrationExplorer

* :class:`TrainingSession` - trains in a background thread with Pause / Resume / Stop buttons,
  live epoch curves and a log panel (buttons only work while the notebook cell is idle, which is
  why training does not block the cell).
* :class:`BacktestExplorer` - strategy, knobs and costs as widgets; Run re-backtests the run's test block.
* :class:`CalibrationExplorer` - refit the calibration pipeline on the calibration block with other
  options and compare test coverage, interval width, reliability and delta shrinkage.

Requires ipywidgets (the ``notebooks`` extra). Nothing here imports TensorFlow at import time.
"""
import importlib
from typing import Any

# PEP 562: the exported names resolve on first access, so `import neural_trade.notebook.<module>` (the desk
# server reuses `longrun`) does not drag in pandas, ipywidgets or the training stack through the interactive
# classes. Submodules (`panel_data`, `run_report`, ...) are not in this map: the import system handles them.
_EXPORTS = {
    "BacktestExplorer": "backtest_ui",
    "load_run_blocks": "backtest_ui",
    "CalibrationExplorer": "calibration_ui",
    "pick_run": "runs",
    "servable_runs": "runs",
    "TrainingSession": "session",
}

__all__ = ["BacktestExplorer", "CalibrationExplorer", "TrainingSession", "load_run_blocks", "pick_run",
           "servable_runs"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # cache: later accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# neural_trade.notebook.run_report is imported by its own name (`from neural_trade.notebook import run_report`),
# like the other per-notebook modules; it is not re-exported here to keep this import cheap (no pandas at
# `import neural_trade.notebook` time for the interactive classes above).

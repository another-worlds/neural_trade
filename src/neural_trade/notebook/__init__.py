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
from neural_trade.notebook.backtest_ui import BacktestExplorer, load_run_blocks
from neural_trade.notebook.calibration_ui import CalibrationExplorer
from neural_trade.notebook.runs import pick_run, servable_runs
from neural_trade.notebook.session import TrainingSession

__all__ = ["BacktestExplorer", "CalibrationExplorer", "TrainingSession", "load_run_blocks", "pick_run",
           "servable_runs"]

# neural_trade.notebook.run_report is imported by its own name (`from neural_trade.notebook import run_report`),
# like the other per-notebook modules; it is not re-exported here to keep this import cheap (no pandas at
# `import neural_trade.notebook` time for the interactive classes above).

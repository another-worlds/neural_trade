"""neural_trade: multi-horizon BTC forecasting with learnable indicators.

Importing the package is cheap and never imports TensorFlow; heavy modules
(training, models, serving) are imported on demand by the caller:

    from neural_trade.core.config import Config
    from neural_trade.training.trainer import train_and_evaluate

Two process-level defaults are set here because they must precede any TensorFlow import
that a submodule triggers:

* ``TF_DETERMINISTIC_OPS=1`` (unless already set): deterministic kernels (plan S24).
* stdout/stderr replace unencodable characters instead of raising: on a Windows console
  with a non-UTF-8 code page a single non-ASCII glyph in a progress line used to raise
  UnicodeEncodeError and silently abort the lambda-calibration pass.
"""
import os as _os
import sys as _sys

_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):  # not a TextIOWrapper (e.g. captured by a test runner)
        pass

from neural_trade.core.logging import configure_logging as _configure_logging  # noqa: E402

_configure_logging()

__version__ = "0.3.0"

__all__ = ["__version__"]

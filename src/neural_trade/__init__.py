"""neural_trade: multi-horizon BTC forecasting with learnable indicators.

Importing the package is cheap and never imports TensorFlow; heavy modules
(training, models, serving) are imported on demand by the caller:

    from neural_trade.core.config import Config
    from neural_trade.training.trainer import train_and_evaluate

Process-level defaults are set here because they must precede any TensorFlow import
that a submodule triggers:

* ``TF_DETERMINISTIC_OPS=1`` (unless already set): deterministic kernels (plan S24).
* stdout/stderr replace unencodable characters instead of raising: on a Windows console
  with a non-UTF-8 code page a single non-ASCII glyph in a progress line used to raise
  UnicodeEncodeError and silently abort the lambda-calibration pass.
* On Windows, the conda environment's CUDA/cuDNN DLL folders are put on PATH (see
  ``_windows_cuda_dll_path``): TensorFlow 2.10 finds them only through PATH, so a Python or
  Jupyter kernel started without ``conda activate`` otherwise trains on the CPU with nothing
  but a log line to say so. Set ``NEURAL_TRADE_NO_DLL_PATH=1`` to opt out.
"""
import os as _os
import sys as _sys

_os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):  # not a TextIOWrapper (e.g. captured by a test runner)
        pass


def _windows_cuda_dll_path():
    """Prepend the environment's DLL folders to PATH when its CUDA runtime is installed there.

    Returns the folders added (empty when nothing was needed or this is not Windows). Only takes
    effect if TensorFlow has not been imported yet - import neural_trade first.
    """
    if _os.name != "nt" or _os.environ.get("NEURAL_TRADE_NO_DLL_PATH") == "1":
        return []
    prefix = _sys.prefix
    if not _os.path.isfile(_os.path.join(prefix, "Library", "bin", "cudart64_110.dll")):
        return []
    folders = [_os.path.join(prefix, *p) for p in (("Library", "bin"), ("Library", "mingw-w64", "bin"),
                                                   ("Library", "usr", "bin"), ("Scripts",), ("bin",), ())]
    folders = [f for f in folders if _os.path.isdir(f)]
    current = _os.environ.get("PATH", "").split(_os.pathsep)
    normalized = {_os.path.normcase(_os.path.normpath(p)) for p in current if p}
    missing = [f for f in folders if _os.path.normcase(_os.path.normpath(f)) not in normalized]
    if missing:
        _os.environ["PATH"] = _os.pathsep.join(missing + current)
    for f in folders:
        try:
            _os.add_dll_directory(f)
        except (AttributeError, OSError):
            pass
    return missing


_CUDA_DLL_FOLDERS_ADDED = _windows_cuda_dll_path()

from neural_trade.core.logging import configure_logging as _configure_logging  # noqa: E402

_configure_logging()

__version__ = "0.3.0"

__all__ = ["__version__"]

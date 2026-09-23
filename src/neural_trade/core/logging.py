"""Package logging: every module logs to ``logging.getLogger(__name__)`` (no ``print`` in ``src/``).

``configure_logging`` (called by ``import neural_trade``) gives the ``neural_trade`` logger one
handler that writes plain messages to the CURRENT ``sys.stdout`` - resolved at each emit, so
notebook cells and captured streams see training progress exactly as the old prints did.
Level: ``NEURAL_TRADE_LOG_LEVEL`` (default INFO); ``configure_logging(level)`` changes it, and
an application that configures logging itself can call ``configure_logging(handler=False)``
to drop the package handler and let records propagate to the root logger.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Union

ROOT = "neural_trade"


class _CurrentStdoutHandler(logging.StreamHandler):
    """A StreamHandler bound to whatever ``sys.stdout`` (or ``sys.stderr``) is when a record is emitted."""

    def __init__(self, to_stderr: bool = False):
        self.to_stderr = to_stderr
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stderr if self.to_stderr else sys.stdout

    @stream.setter
    def stream(self, _value):  # StreamHandler.__init__ assigns it; the property always wins
        pass


def configure_logging(level: Optional[Union[int, str]] = None, *, handler: bool = True,
                      stream: Optional[str] = None) -> logging.Logger:
    """``stream``: "stdout" (default at import) or "stderr" (the CLI: stdout carries command output)."""
    log = logging.getLogger(ROOT)
    level = level if level is not None else os.environ.get("NEURAL_TRADE_LOG_LEVEL", "INFO")
    log.setLevel(level.upper() if isinstance(level, str) else level)
    ours = [h for h in log.handlers if isinstance(h, _CurrentStdoutHandler)]
    if stream is not None:
        for h in ours:
            h.to_stderr = stream == "stderr"
    if handler and not ours:
        h = _CurrentStdoutHandler(to_stderr=stream == "stderr")
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
        log.propagate = False
    elif not handler:
        for h in ours:
            log.removeHandler(h)
        log.propagate = True
    return log


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

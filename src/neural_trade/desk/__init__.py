"""The trading desk (D-067): the top layer of ``neural_trade``. Nothing else in the package imports it.

Importing this package must stay cheap: no TensorFlow, fastapi, nicegui or ccxt (``tests/desk/test_import_guard.py``).
"""
from __future__ import annotations

from neural_trade import __version__
from neural_trade.desk.config import DESK_HOME, DeskSettings

__all__ = ["DESK_HOME", "DeskSettings", "__version__"]

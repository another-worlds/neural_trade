"""Desk built-in strategies (DESK-1.6). Importing this package registers them in ``DeskStrategies``.

``ma_cross``, ``rsi_threshold`` and ``bollinger_breakout`` copy the entry and exit rules of
``neural_trade.strategy.ta_rules`` (ta_ma_cross, ta_rsi, ta_bollinger) and call its ``sma``, ``rolling_std`` and
``wilder_rsi``; they place the same entry bars as those rules at the same parameters. ``buy_and_hold`` and
``always_flat`` are the baselines of the report, ``random_entries`` is the matched-frequency null.

``neural_trade.strategy.ta_rules`` is imported inside ``on_bar``: importing the ``neural_trade.strategy`` package
loads TensorFlow, and ``import neural_trade.desk`` must not (tests/desk/test_import_guard.py).
"""
from neural_trade.desk.strategies.always_flat import AlwaysFlat
from neural_trade.desk.strategies.bollinger_breakout import BollingerBreakout
from neural_trade.desk.strategies.buy_and_hold import BuyAndHold
from neural_trade.desk.strategies.ma_cross import MACross
from neural_trade.desk.strategies.random_entries import RandomEntries
from neural_trade.desk.strategies.rsi_threshold import RSIThreshold

__all__ = ["AlwaysFlat", "BollingerBreakout", "BuyAndHold", "MACross", "RSIThreshold", "RandomEntries"]

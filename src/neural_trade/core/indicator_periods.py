"""Names and configured starts of the learned indicator periods (plain Python: no TensorFlow, no plotting).

``LearnableIndicators.build`` initialises one logit per period from the config, in this order,
and ``get_learned_parameters()`` reports the periods under these names: ``ma_period_i``,
``macd_i_{fast,slow,signal}``, ``rsi_period_i``, ``bb_period_i``. The JSONL epoch logger records
the periods when training begins (``period_init.json``) and the indicator figures measure change
from that start; both use this module, so telemetry does not import the plotting code.
"""
from __future__ import annotations

from typing import Dict

PERIOD_INIT_FILE = "period_init.json"
MACD_ROLES = ("fast", "slow", "signal")


def configured_periods(config) -> Dict[str, float]:
    """The period each learned copy is initialised to, by metrics name (``ma_period_0`` ...).

    Empty when ``config`` is None or lacks the indicator settings."""
    if config is None:
        return {}
    try:
        out = {f"ma_period_{i}": float(s) for i, s in enumerate(config.MA_SPANS)}
        for i, m in enumerate(config.MACD_SETTINGS):
            out.update({f"macd_{i}_{r}": float(m[r]) for r in MACD_ROLES})
        out.update({f"rsi_period_{i}": float(p) for i, p in enumerate(config.RSI_PERIODS)})
        out.update({f"bb_period_{i}": float(p) for i, p in enumerate(config.BB_PERIODS)})
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}
    return out

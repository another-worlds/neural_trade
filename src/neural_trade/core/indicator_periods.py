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
        extras = dict(getattr(config, "INDICATOR_FAMILIES", None) or {})
        out = {}
        if "ma" not in extras:
            out.update({f"ma_period_{i}": float(s) for i, s in enumerate(config.MA_SPANS)})
        if "macd" not in extras:
            for i, m in enumerate(config.MACD_SETTINGS):
                out.update({f"macd_{i}_{r}": float(m[r]) for r in MACD_ROLES})
        if "rsi" not in extras:
            out.update({f"rsi_period_{i}": float(p) for i, p in enumerate(config.RSI_PERIODS)})
        if "bb" not in extras:
            out.update({f"bb_period_{i}": float(p) for i, p in enumerate(config.BB_PERIODS)})
        # Families from Config.INDICATOR_FAMILIES (NT-046), named as they are reported: a
        # scalar instance -> '<family>_period_<i>', a dict -> '<family>_<i>_<param>'. This
        # module stays registry-free, so a dict parameter left at its textbook default (the
        # family fills it in) is not listed here.
        for fam, insts in extras.items():
            for i, inst in enumerate(insts or []):
                if isinstance(inst, dict):
                    out.update({f"{fam}_{i}_{k}": float(v) for k, v in inst.items()})
                else:
                    out[f"{fam}_period_{i}"] = float(inst)
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}
    return out

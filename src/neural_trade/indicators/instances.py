"""Which family instances a config selects (NT-046). Plain Python: no TensorFlow.

The four historical fields (``MA_SPANS``, ``MACD_SETTINGS``, ``RSI_PERIODS``, ``BB_PERIODS``)
stay the configuration of today's four families, so existing config files load unchanged;
``Config.INDICATOR_FAMILIES`` adds further families (or overrides one of the four) by name.
The mapping preserves the historical order (ma, macd, rsi, bb, then the extras), which fixes
the meta-adjust columns, the weight order and the channel order of the model.
"""
from __future__ import annotations

from typing import Dict, List


def indicator_instances(config) -> "Dict[str, List]":
    """Ordered ``{family name: [instance, ...]}``; an instance is a starting period (scalar)
    or a mapping of parameter periods, in bars. Families with no instances are left out."""
    out: Dict[str, List] = {
        "ma": list(getattr(config, "MA_SPANS", None) or []),
        "macd": [dict(m) for m in (getattr(config, "MACD_SETTINGS", None) or [])],
        "rsi": list(getattr(config, "RSI_PERIODS", None) or []),
        "bb": list(getattr(config, "BB_PERIODS", None) or []),
    }
    for name, insts in (getattr(config, "INDICATOR_FAMILIES", None) or {}).items():
        out[name] = list(insts or [])
    return {name: insts for name, insts in out.items() if insts}


def num_learnable_logits(config) -> int:
    """Total learnable periods the configured instances declare (the meta-adjust width)."""
    from .registry import Indicators

    return sum(len(Indicators.get(name).params) * len(insts)
               for name, insts in indicator_instances(config).items())

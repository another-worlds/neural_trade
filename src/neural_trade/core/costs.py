"""The per-instrument cost profile of a setup (NT-041; D-044: every default is 0).

Fee, half-spread and slippage per side, in basis points of the notional, are Config fields
(FEE_BPS, HALF_SPREAD_BPS, SLIPPAGE_BPS) so a run's setup carries them; the backtest engine's
``BacktestConfig`` fields have the same names in lower case. ``cost_profile_of`` is the one place
that maps the one onto the other.
"""
from __future__ import annotations

from typing import Dict

COST_FIELDS = ("fee_bps", "half_spread_bps", "slippage_bps")


def cost_profile_of(config) -> Dict[str, float]:
    """``{"fee_bps", "half_spread_bps", "slippage_bps"}`` of a Config (0.0 each by default, and for a config object
    without the fields)."""
    return {"fee_bps": float(getattr(config, "FEE_BPS", 0.0)),
            "half_spread_bps": float(getattr(config, "HALF_SPREAD_BPS", 0.0)),
            "slippage_bps": float(getattr(config, "SLIPPAGE_BPS", 0.0))}


__all__ = ["COST_FIELDS", "cost_profile_of"]

"""Cost profiles (PLAN A5.1): per-side costs identical to ``BacktestConfig.slip_rate`` / ``fee_rate``.

Field names match the keys of ``DeskSettings.cost_profiles`` (``desk/config.py``). Pure data, no TensorFlow.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from neural_trade.desk.core.model import Liquidity


@dataclass(frozen=True)
class CostProfile:
    """Basis points per side. ``slip_rate`` and ``fee_rate`` are the oracle's expressions, operand for operand."""

    name: str = "custom"
    fee_bps_taker: float = 0.0
    fee_bps_maker: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        for f in ("fee_bps_taker", "fee_bps_maker", "half_spread_bps", "slippage_bps"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0, got {getattr(self, f)}")

    @property
    def slip_rate(self) -> float:
        return (self.half_spread_bps + self.slippage_bps) / 1e4

    def fee_rate(self, liquidity: Liquidity = Liquidity.TAKER) -> float:
        bps = self.fee_bps_maker if Liquidity(liquidity) is Liquidity.MAKER else self.fee_bps_taker
        return bps / 1e4

    def side_costs(self, qty: float, mid: float, liquidity: Liquidity = Liquidity.TAKER) -> tuple[float, float]:
        """(slip cost, fee) of one side: ``qty x mid x slip_rate`` and ``fee_rate x qty x mid``."""
        return qty * mid * self.slip_rate, self.fee_rate(liquidity) * qty * mid

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "CostProfile":
        known = {f.name for f in fields(cls)} - {"name"}
        extra = set(d) - known
        if extra:
            raise ValueError(f"cost profile {name!r}: unknown keys {sorted(extra)}")
        return cls(name=name, **{k: float(v) for k, v in d.items()})

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


RESEARCH_ZERO = CostProfile(name="research_zero")  # D-044
# A stated assumption (owner question 3 open): 0.1% fee per side, no spread or slippage.
BINANCE_SPOT_TAKER = CostProfile(name="binance_spot_taker", fee_bps_taker=10.0, fee_bps_maker=10.0)

PRESETS: dict[str, CostProfile] = {p.name: p for p in (RESEARCH_ZERO, BINANCE_SPOT_TAKER)}

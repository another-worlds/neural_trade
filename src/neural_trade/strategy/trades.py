"""Orders and trades (the notebook's Trade and EnhancedTrade in one record)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Order:
    """A strategy's decision at a bar's close; the engine fills it at the next bar's open."""

    side: str                       # "LONG" or "SHORT"
    size_frac: float = 1.0          # fraction of equity committed (notional)
    tp: Optional[float] = None      # take-profit price (checked against the bar's high/low)
    sl: Optional[float] = None      # stop-loss price
    tp_is_offset: bool = False      # tp/sl given as $ offsets from the fill price
    reason: str = ""
    max_hold: Optional[int] = None  # bars; None -> BacktestConfig.max_hold
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    side: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    notional: float
    gross_pnl: float
    costs: float
    exit_reason: str
    tp: Optional[float] = None
    sl: Optional[float] = None
    entry_reason: str = ""
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def bars_held(self) -> int:
        return self.exit_bar - self.entry_bar

    @property
    def return_pct(self) -> float:
        return 100.0 * self.net_pnl / self.notional if self.notional else 0.0

    def is_win(self) -> bool:
        return self.net_pnl > 0

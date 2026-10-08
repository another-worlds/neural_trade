"""The dataset spec: what series a run is about and its lengths in wall-clock time (NT-041, D-022).

    spec = DatasetSpec.from_config(config)
    spec.symbol, spec.quote_currency, spec.bar_minutes, spec.data_file
    spec.window_minutes, spec.horizon_minutes          # always derived: bars x bar size
    spec.window_bars, spec.horizon_bars                # the bars the pipeline reads
    spec.cost_profile                                  # fee, half-spread, slippage per side (bps)

The setting itself is the Config: a length is configured either in bars (LOOKBACK, HORIZON_STEPS,
EXTENDED_TREND_PERIODS, today's fields) or in wall-clock minutes (WINDOW_MINUTES, HORIZON_MINUTES,
EXTENDED_TREND_MINUTES), converted to bars by the bar size (RESAMPLE_MINUTES). A length that is not a
whole number of bars is refused, never rounded. The spec is the read-only view of both, so figures,
backtests, run metadata and the leaderboard all take the instrument, the bar size and the units from
one place. The reference setup (BTC/USDT, one-minute bars, 60-minute window, horizons of 10, 15 and 20
minutes) gives today's 60 bars and 10 / 15 / 20 bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .exceptions import InvalidConfigurationError

_EXACT_TOL = 1e-9


def minutes_to_bars(minutes: float, bar_minutes: float, what: str = "length") -> int:
    """``minutes`` / ``bar_minutes`` as a whole number of bars; InvalidConfigurationError when the bar size is
    not positive, or the length is not a positive whole multiple of the bar size (never rounded)."""
    if not bar_minutes or float(bar_minutes) <= 0:
        raise InvalidConfigurationError(f"{what}: the bar size must be positive, got {bar_minutes!r} minutes")
    ratio = float(minutes) / float(bar_minutes)
    bars = int(round(ratio))
    if bars < 1 or abs(ratio - bars) > _EXACT_TOL * max(1.0, abs(ratio)):
        raise InvalidConfigurationError(
            f"{what}={minutes!r} minutes is not a whole number of {bar_minutes:g}-minute bars "
            f"(it is {ratio:.6g} bars): choose a multiple of the bar size")
    return bars


def _minutes(bars: int, bar_minutes: float) -> float:
    m = float(bars) * float(bar_minutes)
    return int(m) if float(m).is_integer() else m


@dataclass(frozen=True)
class DatasetSpec:
    """The instrument, the bar size, the data file and the lengths of a setup (read-only view of a Config)."""
    symbol: str
    quote_currency: str
    bar_minutes: float
    data_file: str
    window_bars: int
    horizon_bars: List[int]
    trend_bars: List[int]
    cost_profile: Dict[str, float]

    @classmethod
    def from_config(cls, config) -> "DatasetSpec":
        from .costs import cost_profile_of

        return cls(symbol=str(config.SYMBOL), quote_currency=str(config.QUOTE_CURRENCY),
                   bar_minutes=float(config.RESAMPLE_MINUTES), data_file=str(config.CSV_PATH),
                   window_bars=int(config.LOOKBACK), horizon_bars=[int(h) for h in config.HORIZON_STEPS],
                   trend_bars=[int(p) for p in config.EXTENDED_TREND_PERIODS],
                   cost_profile=cost_profile_of(config))

    @property
    def window_minutes(self) -> float:
        return _minutes(self.window_bars, self.bar_minutes)

    @property
    def horizon_minutes(self) -> List[float]:
        return [_minutes(h, self.bar_minutes) for h in self.horizon_bars]

    @property
    def trend_minutes(self) -> List[float]:
        return [_minutes(p, self.bar_minutes) for p in self.trend_bars]

    @property
    def bar_label(self) -> str:
        """The bar size as text: ``1-minute``, ``5-minute``, ``1-hour``, ``1-day`` ..."""
        return bar_label(self.bar_minutes)

    @property
    def title_tag(self) -> str:
        """``BTC/USDT 1-minute``: the instrument and the bar size, for figure titles."""
        return f"{self.symbol} {self.bar_label}"

    def to_dict(self) -> Dict[str, Any]:
        """The recorded setup (run meta.json "setup" and the leaderboard): the wall-clock units beside the bars."""
        return {"symbol": self.symbol, "quote_currency": self.quote_currency, "bar_minutes": self.bar_minutes,
                "window_minutes": self.window_minutes, "horizon_minutes": self.horizon_minutes,
                "trend_minutes": self.trend_minutes, "cost_profile": dict(self.cost_profile)}


def bar_label(bar_minutes: Optional[float]) -> str:
    """``1-minute`` / ``15-minute`` / ``1-hour`` / ``1-day`` for a bar size in minutes."""
    if bar_minutes is None or not math.isfinite(float(bar_minutes)) or float(bar_minutes) <= 0:
        return "n/a"
    m = float(bar_minutes)
    for size, name in ((1440.0, "day"), (60.0, "hour")):
        if m >= size and abs(m / size - round(m / size)) < 1e-9:
            return f"{int(round(m / size))}-{name}"
    return f"{m:g}-minute"


__all__ = ["DatasetSpec", "bar_label", "minutes_to_bars"]

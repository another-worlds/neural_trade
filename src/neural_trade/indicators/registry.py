"""The Indicators registry (D-027, D-031): learnable indicator families by name.

It extends the nine registries of D-002 (``neural_trade.registries.indicators`` re-exports it
next to them). A component is an :class:`~neural_trade.indicators.base.IndicatorFamily`
*instance* declaring its inputs, learnable parameters with textbook defaults and bounds,
output channels, drawing spec and M(eps); ``neural_trade.indicators.families`` registers
today's four (MA/EMA, MACD, RSI, Bollinger). The model builds its indicator channels from
the entries the config lists (``instances.indicator_instances``), so a new family is one
registry entry plus one config line (``Config.INDICATOR_FAMILIES``).
"""
from __future__ import annotations

from typing import Any, ClassVar, Tuple

from neural_trade.core.registry import BaseRegistry

from .base import DRAW_TARGETS, ChannelSpec, IndicatorFamily, ParamSpec


class Indicators(BaseRegistry):
    registry = {}
    strict = True
    default = None  # a config always names its families; there is no 'default family'
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.indicators.families",
                                                    "neural_trade.indicators.families_ohlcv")

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        return (isinstance(component, IndicatorFamily)
                and bool(component.name)
                and bool(component.inputs)
                and len(component.params) > 0
                and all(isinstance(p, ParamSpec) and p.default >= p.minimum for p in component.params)
                and len(component.channels) > 0
                and all(isinstance(c, ChannelSpec) and c.draw in DRAW_TARGETS for c in component.channels)
                and component.draw in DRAW_TARGETS)

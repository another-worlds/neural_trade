"""always_flat: never orders (the zero line every strategy must beat; oracle ``AlwaysFlat``)."""
from __future__ import annotations

from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy


@DeskStrategies.register(name="always_flat", tags=["baseline"])
class AlwaysFlat(DeskStrategy):
    """Never trades."""

    name = "always_flat"

    def on_bar(self, ctx) -> None:
        return None

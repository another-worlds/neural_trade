"""buy_and_hold: one long entry at the first bar it is asked, held to the end (oracle ``BuyAndHold``)."""
from __future__ import annotations

from neural_trade.desk.core.model import PositionSide
from neural_trade.desk.core.params import ParamSpec
from neural_trade.desk.core.strategy import DeskStrategies, DeskStrategy

HOLD_FOREVER = 10 ** 9   # the oracle's max_hold: the engine's TIME exit never fires


@DeskStrategies.register(name="buy_and_hold", tags=["baseline"])
class BuyAndHold(DeskStrategy):
    """Enters once (BUY, ``size`` of equity at the next open) and never exits; a backtest closes it as EOW."""

    name = "buy_and_hold"
    PARAMS = (ParamSpec("size", "float", 1.0, 0.01, 1.0, 0.01, doc="fraction of equity"),)

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._entered = False

    def on_start(self, ctx) -> None:
        self._entered = False

    def on_bar(self, ctx) -> None:
        if self._entered or ctx.position.side is not PositionSide.FLAT:
            return
        self._entered = True
        ctx.submit_market("BUY", size_frac=self.params["size"], max_hold=HOLD_FOREVER, tag="buy_and_hold")

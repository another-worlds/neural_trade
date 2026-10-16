"""Pure fill functions (PLAN A5, CONTRACTS C1). ``side`` is +1 buy / -1 sell; all prices are floats.

The returned ``Fill`` has ``mid`` = the reference price before spread and slippage, ``price = mid x (1 + side x slip)``
and ``fee`` on the mid notional, exactly as ``strategy/backtest.py``. Fields a pure function cannot know
(``trade_id``, ``client_id``, ``fee_currency``) are empty strings; the caller sets them with ``dataclasses.replace``.
"""
from __future__ import annotations

from typing import Optional

from neural_trade.desk.core.costs import CostProfile
from neural_trade.desk.core.model import Bar, Fill, Liquidity, Side


def _fill(instrument_id: str, ts: int, mid: float, side: int, cost: CostProfile, qty: float,
          liquidity: Liquidity) -> Fill:
    if side not in (1, -1):
        raise ValueError(f"side must be +1 or -1, got {side!r}")
    _, fee = cost.side_costs(qty, mid, liquidity)
    return Fill(trade_id="", client_id="", instrument_id=instrument_id, side=Side.BUY if side > 0 else Side.SELL,
                qty=qty, price=mid * (1 + side * cost.slip_rate), mid=mid, fee=fee, fee_currency="",
                liquidity=liquidity, ts=ts)


def market_fill(open_: float, side: int, cost: CostProfile, qty: float) -> Fill:
    """A market order fills at the bar's open (A5.1); taker fee. Instrument and ts are the caller's."""
    return _fill("", 0, open_, side, cost, qty, Liquidity.TAKER)


def limit_fill(bar: Bar, side: int, limit: float, cost: CostProfile, qty: float) -> Optional[Fill]:
    """Buy fills when low <= limit at min(limit, open); sell when high >= limit at max(limit, open) (A5.2).
    A touch counts as a fill. Maker fee, same slip arithmetic as every fill (A5.1)."""
    if side > 0:
        mid = min(limit, bar.open) if bar.low <= limit else None
    else:
        mid = max(limit, bar.open) if bar.high >= limit else None
    if mid is None:
        return None
    return _fill(bar.instrument_id, bar.ts_open_ms, mid, side, cost, qty, Liquidity.MAKER)


def stop_fill(bar: Bar, side: int, trigger: float, cost: CostProfile, qty: float) -> Optional[Fill]:
    """STOP_MARKET: buy triggers when high >= trigger, fills at max(trigger, open); sell when low <= trigger at
    min(trigger, open) (A5.2), plus slippage; taker fee."""
    if side > 0:
        mid = max(trigger, bar.open) if bar.high >= trigger else None
    else:
        mid = min(trigger, bar.open) if bar.low <= trigger else None
    if mid is None:
        return None
    return _fill(bar.instrument_id, bar.ts_open_ms, mid, side, cost, qty, Liquidity.TAKER)


def bracket_exit(bar: Bar, side: int, tp: Optional[float], sl: Optional[float],
                 tiebreak: str) -> Optional[tuple[str, float]]:
    """Exit of a position of ``side`` (+1 long) on this bar's high/low (A5.3): ('TP'|'SL', mid price).
    Both hit -> ``tiebreak`` ('sl_first' or 'tp_first'); a gap through the level fills at the open. The price is
    the mid; price the exit with ``market_fill``-style costs at that mid."""
    if tiebreak not in ("sl_first", "tp_first"):
        raise ValueError(f"tiebreak must be 'sl_first' or 'tp_first', got {tiebreak!r}")
    if side not in (1, -1):
        raise ValueError(f"side must be +1 or -1, got {side!r}")
    if side > 0:
        sl_hit = sl is not None and bar.low <= sl
        tp_hit = tp is not None and bar.high >= tp
    else:
        sl_hit = sl is not None and bar.high >= sl
        tp_hit = tp is not None and bar.low <= tp
    if sl_hit and tp_hit:
        tp_hit = tiebreak == "tp_first"
        sl_hit = not tp_hit
    if sl_hit:
        return "SL", (min(sl, bar.open) if side > 0 else max(sl, bar.open))
    if tp_hit:
        return "TP", (max(tp, bar.open) if side > 0 else min(tp, bar.open))
    return None

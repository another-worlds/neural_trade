"""Equity-curve statistics (not (y_true, y_pred) metrics, so not in the Metrics registry)."""
from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np

MINUTES_PER_YEAR = 525_600


def sharpe(returns, periods_per_year: int = MINUTES_PER_YEAR) -> float:
    """Annualised mean/std of per-bar returns (0 when flat)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def sortino(returns, periods_per_year: int = MINUTES_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fall as a fraction of the peak (a positive number)."""
    e = np.asarray(equity, dtype=float)
    if not len(e):
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.max((peak - e) / peak))


def profit_factor(pnls: Iterable[float]) -> float:
    p = np.asarray(list(pnls), dtype=float)
    gains, losses = p[p > 0].sum(), -p[p < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def traded_notional(trades) -> float:
    """$ notional traded over both sides of every trade: the entry notional (qty x entry mid) plus
    the exit notional (qty x exit mid = entry notional + side x gross P&L, since gross P&L is on mids)."""
    total = 0.0
    for t in trades:
        sign = 1.0 if t.side == "LONG" else -1.0
        total += 2.0 * t.notional + sign * t.gross_pnl
    return float(total)


def breakeven_cost_bps(gross_pnl: float, notional: float) -> float:
    """The round-trip cost (bps) at which net P&L is zero: gross P&L / traded notional x 1e4 x 2
    (costs are charged per side on the traded notional). NaN when nothing traded."""
    return float(gross_pnl / notional * 1e4 * 2.0) if notional > 0 else float("nan")


def summarize(equity, bar_returns_net, bar_returns_gross, trades, fees_paid: float, exposure: float,
              periods_per_year: int = MINUTES_PER_YEAR) -> Dict[str, float]:
    """Equity and trade statistics of one backtest.

    Besides the classic keys: ``traded_notional`` ($ over both sides, ``traded_notional()``),
    ``breakeven_cost_bps`` (the round-trip cost that would zero the net P&L) and
    ``gross_edge_per_trade_bps`` (the mean over trades of gross P&L / entry notional, x 1e4; NaN
    without trades). The exposure engine overrides the trade-based ones (``backtest.run_exposure_backtest``).
    """
    net = [t.net_pnl for t in trades]
    notional = traded_notional(trades)
    gross = float(sum(t.gross_pnl for t in trades))
    edges = [t.gross_pnl / t.notional for t in trades if t.notional > 0]
    return {
        "n_trades": len(trades),
        "total_return": float(equity[-1] / equity[0] - 1.0) if len(equity) else 0.0,
        "sharpe_net": sharpe(bar_returns_net, periods_per_year),
        "sharpe_gross": sharpe(bar_returns_gross, periods_per_year),
        "sortino": sortino(bar_returns_net, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float(np.mean([p > 0 for p in net])) if net else float("nan"),
        "hit_rate_gross": float(np.mean([t.gross_pnl > 0 for t in trades])) if trades else float("nan"),  # before costs
        "profit_factor": profit_factor(net),
        "avg_hold_bars": float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        "exposure": float(exposure),
        "turnover": float(sum(2 * t.notional for t in trades) / equity[0]) if len(equity) else 0.0,
        "fees_paid": float(fees_paid),
        "traded_notional": notional,
        "breakeven_cost_bps": breakeven_cost_bps(gross, notional),
        "gross_edge_per_trade_bps": float(np.mean(edges) * 1e4) if edges else float("nan"),
    }

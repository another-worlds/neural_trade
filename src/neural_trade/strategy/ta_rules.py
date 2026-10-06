"""Classic technical-analysis rules: the manual-search baseline of the yardstick (NT-033, D-020).

VISION "The yardstick": the learned indicators must beat, under the same search budget and the same
dev-fold net Sharpe, (b) classic TA rules whose parameters the same search tunes. Three rules, each a
registered Strategy that reads only the close up to the decision bar (causal; they pass
``assert_no_lookahead``) and takes the price series from the SignalFrame, so they run on any
SignalFrame, including the price-only one of a scenario that trains no network
(:func:`price_only_frame`, ``experiments.scorer.score_strategy_only``).

========  ================================================  =====================  ==================
rule      entry (flat, at the close of bar t)               exit                   searchable (range)
========  ================================================  =====================  ==================
ma_cross  LONG while SMA(fast) > SMA(slow), SHORT while     the SMAs cross back    fast 2..30,
          below (state, so a reversal re-enters on the                             slow 30..240
          next bar); long only with allow_short = False
rsi       LONG when RSI < lower, SHORT when RSI > upper     RSI back through 50    period 2..60,
          (mean reversion)                                  (``exit_level``)       lower 10..40,
                                                                                   upper 60..90
bollinger LONG when close > SMA + k sd (breakout),          close back through    period 5..120,
          SHORT when close < SMA - k sd                     the middle band        k 1.0..3.0
========  ================================================  =====================  ==================

Textbook starting points (the defaults; what a person would try first): MA cross 10 / 30, RSI period
14 with 30 / 70, Bollinger 20 bars and 2 standard deviations (population sd, as the network's Bollinger).
The ranges are declared on the dataclass fields (:func:`~neural_trade.strategy.strategies.search_field`)
in the keys of a scenario's ``search:`` block, and :func:`strategy_search_space` reads them, so the
same search machinery that tunes the network's Config fields tunes these. fast <= slow holds over the
whole range (fast <= 30 <= slow); fast == slow is valid and never trades (the SMAs coincide).

**RSI definition (decision, NT-033 note of 2026-10-06).** The rule's RSI is Wilder's: gains and losses
smoothed with alpha = 1/n, seeded by the plain mean of the first n changes (RSI is defined from bar n
on), RSI = 100 - 100 / (1 + avg gain / avg loss). The network's RSI is different: its periods 9 / 14 /
21 are EMA spans, alpha = 2/(p+1) (indicators/families.py). Wilder's n equals an EMA span of 2n - 1, so
the textbook "RSI 14" of this rule (alpha 1/14) corresponds to span 27 in the network, and the network's
span 14 is Wilder's 7.5. The frozen twin keeps the network's own configured spans (the same maths,
frozen); the rule's ``period`` is a searchable Wilder n. A result that compares the two names which one
it means.

Each position is held at most ``max_hold`` bars (default 1,440, a day of 1-minute bars). The
indicators are computed once per SignalFrame, from its closes in time order (bar t reads closes <= t);
a rule waits ``warmup()`` bars at the start of a block until its indicator has history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np

from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.strategy.signals import SignalFrame
from neural_trade.strategy.strategies import Strategies, Strategy, search_field
from neural_trade.strategy.trades import Order

MAX_HOLD = 1440


def sma(close: np.ndarray, n: int) -> np.ndarray:
    """Trailing simple moving average of ``n`` bars; NaN before bar n - 1."""
    import pandas as pd

    return pd.Series(np.asarray(close, dtype=float)).rolling(int(n)).mean().to_numpy()


def rolling_std(close: np.ndarray, n: int) -> np.ndarray:
    """Trailing population standard deviation (ddof 0) of ``n`` bars; NaN before bar n - 1."""
    import pandas as pd

    return pd.Series(np.asarray(close, dtype=float)).rolling(int(n)).std(ddof=0).to_numpy()


def wilder_rsi(close: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RSI(n) of ``close`` (module docstring); NaN before bar n; trailing only."""
    c = np.asarray(close, dtype=float)
    out = np.full(len(c), np.nan)
    n = int(n)
    if len(c) <= n:
        return out
    d = np.diff(c)
    up, dn = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    gain, loss = float(up[:n].mean()), float(dn[:n].mean())
    for t in range(n, len(c)):
        if t > n:
            gain = (gain * (n - 1) + up[t - 1]) / n
            loss = (loss * (n - 1) + dn[t - 1]) / n
        if loss == 0.0:
            out[t] = 50.0 if gain == 0.0 else 100.0
        else:
            out[t] = 100.0 - 100.0 / (1.0 + gain / loss)
    return out


def price_only_frame(close, y=None, horizon_steps=(10, 15, 20), split: str = "test"):
    """A PredictionFrame that carries only prices: the heads are neutral (delta 0, P(up) 0.5, variance 1,
    no calibration), for a rule that reads the close alone. ``close``: the decision price of every bar
    (``last_close``); ``y``: realised deltas [N, 3] (zeros when absent)."""
    from neural_trade.evaluation.frame import HORIZONS, PredictionFrame

    c = np.asarray(close, dtype=float).reshape(-1)
    n = len(c)
    zeros = {h: np.zeros(n) for h in HORIZONS}
    frame = PredictionFrame(np.zeros((n, len(HORIZONS))) if y is None else np.asarray(y, dtype=float), c,
                            zeros, {h: np.full(n, 0.5) for h in HORIZONS}, {h: np.ones(n) for h in HORIZONS},
                            1.0, 0.0, tuple(horizon_steps), split)
    frame.meta["delta_raw"] = {h: np.zeros(n) for h in HORIZONS}      # no raw heads: coherence is not used
    return frame


class _TARule(Strategy):
    """The shared plumbing: indicator arrays computed once per SignalFrame, kept with the frame they
    belong to (the reference stops a recycled ``id`` from matching a different frame)."""

    price_only: ClassVar[bool] = True

    def _arrays(self, s: SignalFrame):
        cached = getattr(self, "_cache", None)
        if cached is None or cached[0] is not s:
            self._cache = (s, self._compute(np.asarray(s.close, dtype=float)))
        return self._cache[1]

    def _compute(self, close: np.ndarray):
        raise NotImplementedError

    def _order(self, sign: int, reason: str) -> Optional[Order]:
        if sign < 0 and not self.allow_short:
            return None
        return Order("LONG" if sign > 0 else "SHORT", self.size, reason=reason, max_hold=self.max_hold)


@Strategies.register(name="ta_ma_cross", tags=["baseline", "ta", "nt-033"])
@dataclass
class MACrossStrategy(_TARule):
    """Fast / slow SMA crossover, in the market on the side of the faster average (module docstring)."""

    name: ClassVar[str] = "ta_ma_cross"
    fast: int = search_field(10, 2, 30)
    slow: int = search_field(30, 30, 240)
    allow_short: bool = True
    size: float = 1.0
    max_hold: int = MAX_HOLD

    def __post_init__(self):
        if not 1 <= self.fast <= self.slow:
            raise InvalidConfigurationError(f"need 1 <= fast <= slow, got fast={self.fast}, slow={self.slow}")

    def warmup(self) -> int:
        return int(self.slow)

    def _compute(self, close):
        return sma(close, self.fast) - sma(close, self.slow)

    def decide(self, s, t):
        diff = self._arrays(s)[t]
        if not np.isfinite(diff) or diff == 0.0:
            return None
        return self._order(1 if diff > 0 else -1, "ma_cross")

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        diff = self._arrays(s)[t]
        sign = 1 if side == "LONG" else -1
        return "CROSS" if np.isfinite(diff) and sign * diff < 0 else None


@Strategies.register(name="ta_rsi", tags=["baseline", "ta", "nt-033"])
@dataclass
class RSIThresholdStrategy(_TARule):
    """Wilder-RSI mean reversion: buy oversold, sell overbought, out when the RSI is back at 50."""

    name: ClassVar[str] = "ta_rsi"
    period: int = search_field(14, 2, 60)          # Wilder's n (not the network's EMA span; module docstring)
    lower: float = search_field(30.0, 10.0, 40.0)
    upper: float = search_field(70.0, 60.0, 90.0)
    exit_level: float = 50.0
    allow_short: bool = True
    size: float = 1.0
    max_hold: int = MAX_HOLD

    def __post_init__(self):
        if self.period < 1:
            raise InvalidConfigurationError(f"period must be at least 1, got {self.period}")
        if not 0.0 < self.lower < self.exit_level < self.upper < 100.0:
            raise InvalidConfigurationError(
                f"need 0 < lower < exit_level < upper < 100, got {self.lower}, {self.exit_level}, {self.upper}")

    def warmup(self) -> int:
        return int(self.period)

    def _compute(self, close):
        return wilder_rsi(close, self.period)

    def decide(self, s, t):
        r = self._arrays(s)[t]
        if not np.isfinite(r):
            return None
        if r < self.lower:
            return self._order(1, "rsi_oversold")
        if r > self.upper:
            return self._order(-1, "rsi_overbought")
        return None

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        r = self._arrays(s)[t]
        if not np.isfinite(r):
            return None
        if (side == "LONG" and r >= self.exit_level) or (side == "SHORT" and r <= self.exit_level):
            return "RSI_MID"
        return None


@Strategies.register(name="ta_bollinger", tags=["baseline", "ta", "nt-033"])
@dataclass
class BollingerBreakoutStrategy(_TARule):
    """Bollinger breakout: with the break out of SMA +- k sd, out when the close is back at the SMA."""

    name: ClassVar[str] = "ta_bollinger"
    period: int = search_field(20, 5, 120)
    k: float = search_field(2.0, 1.0, 3.0)
    allow_short: bool = True
    size: float = 1.0
    max_hold: int = MAX_HOLD

    def __post_init__(self):
        if self.period < 2:
            raise InvalidConfigurationError(f"period must be at least 2, got {self.period}")
        if not self.k > 0:
            raise InvalidConfigurationError(f"k must be positive, got {self.k}")

    def warmup(self) -> int:
        return int(self.period)

    def _compute(self, close):
        mid = sma(close, self.period)
        sd = rolling_std(close, self.period)
        return mid, mid + self.k * sd, mid - self.k * sd

    def decide(self, s, t):
        mid, upper, lower = self._arrays(s)
        if not np.isfinite(mid[t]):
            return None
        c = s.close[t]
        if c > upper[t]:
            return self._order(1, "bb_breakout_up")
        if c < lower[t]:
            return self._order(-1, "bb_breakout_down")
        return None

    def exit_signal(self, s, t, side, bars_held, entry_price, order):
        mid = self._arrays(s)[0][t]
        sign = 1 if side == "LONG" else -1
        return "BB_MID" if np.isfinite(mid) and sign * (s.close[t] - mid) < 0 else None


TA_RULES: Tuple[str, ...] = ("ta_ma_cross", "ta_rsi", "ta_bollinger")

__all__ = ["BollingerBreakoutStrategy", "MACrossStrategy", "RSIThresholdStrategy", "TA_RULES", "price_only_frame",
           "rolling_std", "sma", "wilder_rsi"]

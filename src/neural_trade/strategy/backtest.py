"""An honest bar-by-bar backtest engine.

Timing: a strategy decides at bar t's CLOSE from signals that use predictions <= t; the order
fills at bar t+1's OPEN. The notebooks filled at the signal bar's own close, which is a price
nobody could trade at once the signal is known.

Per bar t the engine
1. at the open: fills a pending exit, then a pending entry (a gap through a stop fills at the
   open, not at the stop);
2. intrabar: checks stop-loss and take-profit against the bar's LOW/HIGH (the notebooks only saw
   closes); when both are touched in one bar the stop is assumed first (``sl_first``);
3. at the close: a position held ``max_hold`` bars or flagged by ``exit_signal`` gets a pending
   exit; a flat book asks ``decide`` for a pending entry; equity is marked to the close.

Costs per side, on the mid notional: fee ``fee_bps`` + ``half_spread_bps`` + ``slippage_bps``
(10 + 1 + 2 = 13 bps by default). Fills are the mid moved adversely by spread + slippage;
gross P&L is on mids, net = gross - costs. An open position is closed at the last close
(``EOW``) when ``mark_to_market_at_end``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy.performance import MINUTES_PER_YEAR, summarize
from neural_trade.strategy.signals import SignalFrame
from neural_trade.strategy.strategies import RandomSignal, Strategies, Strategy
from neural_trade.strategy.trades import Order, Trade


@dataclass
class BacktestConfig:
    fill: str = "next_open"
    fee_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    tp_sl_on: str = "high_low"          # or "close"
    same_bar_tiebreak: str = "sl_first"  # or "tp_first"
    max_hold: int = 30
    mark_to_market_at_end: bool = True
    initial_equity: float = 10_000.0
    bar_minutes: float = 1.0
    minutes_per_year: int = MINUTES_PER_YEAR
    random_seeds: int = 100

    def __post_init__(self):
        if self.fill != "next_open":
            raise ValueError("only fill='next_open' is supported (a same-bar close fill is look-ahead)")
        if self.tp_sl_on not in ("high_low", "close"):
            raise ValueError(f"tp_sl_on must be 'high_low' or 'close', got {self.tp_sl_on!r}")
        if self.same_bar_tiebreak not in ("sl_first", "tp_first"):
            raise ValueError(f"same_bar_tiebreak must be 'sl_first' or 'tp_first', got {self.same_bar_tiebreak!r}")

    @property
    def slip_rate(self) -> float:
        return (self.half_spread_bps + self.slippage_bps) / 1e4

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 1e4

    @property
    def periods_per_year(self) -> int:
        return int(round(self.minutes_per_year / self.bar_minutes))


@dataclass
class Bars:
    """OHLC of each sample's decision bar (bar t = the last input bar of sample t)."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __post_init__(self):
        for k in ("open", "high", "low", "close"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=float).reshape(-1))
        if not (len(self.open) == len(self.high) == len(self.low) == len(self.close)):
            raise ValueError("open/high/low/close must have equal lengths")

    def __len__(self):
        return len(self.close)

    @classmethod
    def from_close(cls, close) -> "Bars":
        """Close-only bars: open = previous close, high/low = max/min(open, close)."""
        c = np.asarray(close, dtype=float).reshape(-1)
        o = np.concatenate([c[:1], c[:-1]])
        return cls(o, np.maximum(o, c), np.minimum(o, c), c)

    @classmethod
    def from_frame(cls, df, anchor_bars) -> "Bars":
        """Bars of a standardised OHLCV frame (Open/High/Low/Close columns) at the anchor rows.

        Anchors must be consecutive bars (WINDOW_STEP = 1): the engine fills at bar t+1's open.
        """
        idx = np.asarray(anchor_bars, dtype=int)
        if len(idx) > 1 and not np.all(np.diff(idx) == 1):
            raise ValueError("anchor bars must be consecutive (WINDOW_STEP=1) for next-open fills")
        cols = {c.lower(): c for c in df.columns}
        close = df[cols["close"]].to_numpy(float)[idx]
        get = lambda name: df[cols[name]].to_numpy(float)[idx] if name in cols else close  # noqa: E731
        return cls(get("open"), get("high"), get("low"), close)

    def slice(self, stop: int) -> "Bars":
        return Bars(self.open[:stop], self.high[:stop], self.low[:stop], self.close[:stop])


@dataclass
class _Position:
    sign: int
    qty: float
    entry_mid: float
    entry_fill: float
    entry_bar: int
    tp: Optional[float]
    sl: Optional[float]
    max_hold: int
    order: Order
    entry_costs: float
    entry_fee: float


@dataclass
class BacktestResult:
    strategy: str
    equity: np.ndarray                  # [N + 1]: initial, then marked at each bar's close
    equity_gross: np.ndarray
    trades: List[Trade]
    decisions: List[Dict[str, Any]]     # every order placed: bar, side, size, reason
    position: np.ndarray                # [N] signed exposure (units of equity) held over each bar
    summary: Dict[str, float]
    config: BacktestConfig
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def trades_frame(self):
        import pandas as pd

        rows = []
        for t in self.trades:
            row = asdict(t)
            row.pop("info", None)
            row.update(net_pnl=t.net_pnl, bars_held=t.bars_held, return_pct=t.return_pct)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "summary": self.summary, "baselines": self.baselines,
                "config": asdict(self.config)}


def _level(value, is_offset, base):
    if value is None:
        return None
    return float(base + value) if is_offset else float(value)


def run_backtest(signals: SignalFrame, bars: Bars, strategy: Strategy,
                 config: Optional[BacktestConfig] = None) -> BacktestResult:
    """Simulate ``strategy`` over the aligned ``signals`` and ``bars`` (same length)."""
    cfg = config or BacktestConfig()
    n = len(bars)
    if len(signals) != n:
        raise ValueError(f"signals ({len(signals)}) and bars ({n}) must be aligned")
    slip, fee = cfg.slip_rate, cfg.fee_rate
    equity_cash = cfg.initial_equity         # realised equity (net of all costs paid so far)
    gross_cash = cfg.initial_equity
    equity = np.empty(n + 1)
    equity_gross = np.empty(n + 1)
    equity[0] = equity_gross[0] = cfg.initial_equity
    position = np.zeros(n)
    trades: List[Trade] = []
    decisions: List[Dict[str, Any]] = []
    pos: Optional[_Position] = None
    pending_entry: Optional[Order] = None
    pending_exit: Optional[str] = None
    warmup = int(strategy.warmup())
    fees_paid = 0.0

    def close_position(t, mid, reason):
        nonlocal pos, equity_cash, gross_cash, fees_paid
        exit_fill = mid * (1 - pos.sign * slip)
        exit_fee = fee * pos.qty * mid
        exit_costs = pos.qty * mid * slip + exit_fee
        gross = pos.sign * pos.qty * (mid - pos.entry_mid)
        equity_cash += gross - exit_costs
        gross_cash += gross
        fees_paid += exit_fee
        trades.append(Trade(side="LONG" if pos.sign > 0 else "SHORT", entry_bar=pos.entry_bar, exit_bar=t,
                            entry_price=pos.entry_fill, exit_price=exit_fill, notional=pos.qty * pos.entry_mid,
                            gross_pnl=gross, costs=pos.entry_costs + exit_costs, exit_reason=reason,
                            tp=pos.tp, sl=pos.sl, entry_reason=pos.order.reason, info=dict(pos.order.info)))
        pos = None

    for t in range(n):
        o, h, lo, c = bars.open[t], bars.high[t], bars.low[t], bars.close[t]
        # 1. at the open
        if pending_exit is not None and pos is not None:
            close_position(t, o, pending_exit)
        pending_exit = None
        if pending_entry is not None and pos is None:
            order = pending_entry
            sign = 1 if order.side == "LONG" else -1
            notional = float(np.clip(order.size_frac, 0.0, 1.0)) * equity_cash
            if notional > 0:
                qty = notional / o
                entry_fee = fee * notional
                costs = notional * slip + entry_fee
                equity_cash -= costs
                fees_paid += entry_fee
                pos = _Position(sign, qty, o, o * (1 + sign * slip), t,
                                _level(order.tp, order.tp_is_offset, o), _level(order.sl, order.tp_is_offset, o),
                                int(order.max_hold if order.max_hold is not None else cfg.max_hold),
                                order, costs, entry_fee)
        pending_entry = None
        # 2. intrabar stop-loss / take-profit
        if pos is not None:
            hi_px, lo_px = (h, lo) if cfg.tp_sl_on == "high_low" else (c, c)
            if pos.sign > 0:
                sl_hit = pos.sl is not None and lo_px <= pos.sl
                tp_hit = pos.tp is not None and hi_px >= pos.tp
            else:
                sl_hit = pos.sl is not None and hi_px >= pos.sl
                tp_hit = pos.tp is not None and lo_px <= pos.tp
            if sl_hit and tp_hit:
                tp_hit = cfg.same_bar_tiebreak == "tp_first"
                sl_hit = not tp_hit
            if sl_hit:
                # A gap through the stop fills at the open, which is worse than the stop.
                px = min(pos.sl, o) if pos.sign > 0 else max(pos.sl, o)
                close_position(t, px, "SL")
            elif tp_hit:
                px = max(pos.tp, o) if pos.sign > 0 else min(pos.tp, o)
                close_position(t, px, "TP")
        # 3. at the close
        if pos is not None:
            position[t] = pos.sign * pos.qty * pos.entry_mid / max(equity_cash, 1e-12)
            held = t - pos.entry_bar + 1
            if held >= pos.max_hold:
                pending_exit = "TIME"
            else:
                pending_exit = strategy.exit_signal(signals, t, "LONG" if pos.sign > 0 else "SHORT", held,
                                                    pos.entry_mid, pos.order)
        elif t >= warmup and t < n - 1:
            pending_entry = strategy.decide(signals, t)
            if pending_entry is not None:
                decisions.append({"bar": t, "side": pending_entry.side, "size": pending_entry.size_frac,
                                  "reason": pending_entry.reason})
        if t == n - 1 and pos is not None and cfg.mark_to_market_at_end:
            close_position(t, c, "EOW")
        unreal = pos.sign * pos.qty * (c - pos.entry_mid) if pos is not None else 0.0
        equity[t + 1] = equity_cash + unreal
        equity_gross[t + 1] = gross_cash + unreal

    net_ret = np.diff(equity) / equity[:-1]
    gross_ret = np.diff(equity_gross) / equity_gross[:-1]
    summary = summarize(equity, net_ret, gross_ret, trades, fees_paid, float(np.mean(position != 0)) if n else 0.0,
                        cfg.periods_per_year)
    summary["costs_paid"] = float(sum(t.costs for t in trades))
    summary["gross_pnl"] = float(sum(t.gross_pnl for t in trades))
    summary["net_pnl"] = float(sum(t.net_pnl for t in trades))
    return BacktestResult(getattr(strategy, "name", type(strategy).__name__), equity, equity_gross, trades,
                          decisions, position, summary, cfg)


# ------------------------------------------------------------------ baselines
def random_same_frequency(signals: SignalFrame, bars: Bars, result: BacktestResult,
                          config: Optional[BacktestConfig] = None, seeds: Optional[int] = None) -> Dict[str, float]:
    """Where the strategy ranks among random strategies with its trade rate and holding time."""
    cfg = config or result.config
    k = int(seeds if seeds is not None else cfg.random_seeds)
    n_tr = result.summary["n_trades"]
    if n_tr == 0 or k <= 0:
        return {"n_seeds": 0, "percentile_total_return": float("nan"), "percentile_sharpe_net": float("nan")}
    hold = max(1, int(round(result.summary["avg_hold_bars"])) or 1)
    flat_bars = max(1, len(bars) - int(result.summary["exposure"] * len(bars)))
    rate = min(1.0, n_tr / flat_bars)
    rets, sharpes = [], []
    for seed in range(k):
        r = run_backtest(signals, bars, RandomSignal(trade_rate=rate, hold_bars=hold, seed=seed), cfg)
        rets.append(r.summary["total_return"])
        sharpes.append(r.summary["sharpe_net"])
    rets, sharpes = np.array(rets), np.array(sharpes)
    return {
        "n_seeds": k, "trade_rate": rate, "hold_bars": hold,
        "random_mean_total_return": float(rets.mean()), "random_mean_sharpe_net": float(sharpes.mean()),
        "percentile_total_return": float(100.0 * np.mean(rets < result.summary["total_return"])),
        "percentile_sharpe_net": float(100.0 * np.mean(sharpes < result.summary["sharpe_net"])),
    }


def backtest(signals: SignalFrame, bars: Bars, strategy: Strategy, config: Optional[BacktestConfig] = None,
             baselines: bool = True) -> BacktestResult:
    """``run_backtest`` plus the baselines every report carries: buy-and-hold, always-flat, random."""
    cfg = config or BacktestConfig()
    res = run_backtest(signals, bars, strategy, cfg)
    if baselines:
        for name in ("buy_and_hold", "always_flat"):
            res.baselines[name] = run_backtest(signals, bars, Strategies.build(name), cfg).summary
        res.baselines["random_same_freq"] = random_same_frequency(signals, bars, res, cfg)
    return res


def backtest_frame(frame: PredictionFrame, bars: Bars, strategy: Strategy, *, var_scale: float,
                   config: Optional[BacktestConfig] = None, calibrated: bool = True, baselines: bool = True,
                   lambdas=None) -> BacktestResult:
    """Build signals from a PredictionFrame (``var_scale`` from the CAL split) and backtest."""
    signals = SignalFrame.build(frame, var_scale, lambdas=lambdas, calibrated=calibrated)
    return backtest(signals, bars, strategy, config, baselines)


# ------------------------------------------------------------------ look-ahead self-test
def _perturb_after(frame: PredictionFrame, bars: Bars, t: int, rng) -> tuple:
    n = len(frame)
    tail = slice(t + 1, n)

    def jitter(a, scale):
        a = np.array(a, dtype=float, copy=True)
        a[tail] = a[tail] + rng.normal(0, scale, a[tail].shape)
        return a

    delta = {h: jitter(frame.delta[h], 500.0) for h in HORIZONS}
    prob = {h: np.clip(jitter(frame.direction_prob[h], 0.3), 0, 1) for h in HORIZONS}
    var = {h: np.abs(jitter(frame.variance_scaled[h], 1.0)) for h in HORIZONS}
    cal = None
    if frame.direction_prob_calibrated is not None:
        cal = {h: np.clip(jitter(frame.direction_prob_calibrated[h], 0.3), 0, 1) for h in HORIZONS}
    lc = jitter(frame.last_close, 300.0)
    f2 = PredictionFrame(frame.y, lc, delta, prob, var, frame.pred_scale, frame.pred_mean, frame.horizon_steps,
                         frame.split, cal)
    b2 = Bars(jitter(bars.open, 300.0), jitter(bars.high, 300.0), jitter(bars.low, 300.0), jitter(bars.close, 300.0))
    return f2, b2


def assert_no_lookahead(frame: PredictionFrame, bars: Bars, make_strategy: Callable[[], Strategy], *,
                        var_scale: float, probes: Sequence[int] = (), config: Optional[BacktestConfig] = None,
                        seed: int = 0) -> None:
    """Perturb every prediction and bar AFTER t; the equity curve and every decision up to t
    must be unchanged. Raises AssertionError naming the first differing bar."""
    cfg = config or BacktestConfig()
    rng = np.random.default_rng(seed)
    n = len(frame)
    probes = list(probes) or [n // 4, n // 2, (3 * n) // 4]
    base = run_backtest(SignalFrame.build(frame, var_scale), bars, make_strategy(), cfg)
    for t in probes:
        f2, b2 = _perturb_after(frame, bars, t, rng)
        other = run_backtest(SignalFrame.build(f2, var_scale), b2, make_strategy(), cfg)
        before = [d for d in base.decisions if d["bar"] <= t]
        after = [d for d in other.decisions if d["bar"] <= t]
        if before != after:
            raise AssertionError(f"look-ahead: decisions up to bar {t} changed when data after {t} changed")
        # equity[t + 1] is the mark at bar t's close
        if not np.allclose(base.equity[: t + 2], other.equity[: t + 2], rtol=0, atol=1e-9):
            first = int(np.argmax(~np.isclose(base.equity[: t + 2], other.equity[: t + 2], rtol=0, atol=1e-9)))
            raise AssertionError(f"look-ahead: equity at bar {first - 1} changed when data after {t} changed")


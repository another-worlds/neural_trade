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
(0 by default, D-044: no trading costs assumed; set them explicitly to backtest at a cost). Fills
are the mid moved adversely by spread + slippage;
gross P&L is on mids, net = gross - costs. An open position is closed at the last close
(``EOW``) when ``mark_to_market_at_end``.

Exposure mode (NT-077; docs/research/2026-09-29-strategy-architectures/README.md section 5.1):
``run_backtest`` hands an ``ExposureStrategy`` to ``run_exposure_backtest``, which holds a signed
QUANTITY (so the exposure drifts with the price), decides a target exposure at the close every
``decide_every`` bars and rebalances at the next open only when the target leaves the no-trade band.
Same timing, same cost fields, no stops. Its random baseline is a circular-shift timing null
(``circular_shift_null``) instead of the size-matched random entries of discrete strategies.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

import numpy as np

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy.performance import breakeven_cost_bps, periods_per_year, summarize
from neural_trade.strategy.signals import SignalFrame
from neural_trade.strategy.strategies import ExposureStrategy, RandomSignal, Strategies, Strategy
from neural_trade.strategy.trades import Order, Trade


@dataclass
class BacktestConfig:
    fill: str = "next_open"
    fee_bps: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    tp_sl_on: str = "high_low"          # or "close"
    same_bar_tiebreak: str = "sl_first"  # or "tp_first"
    max_hold: int = 30
    mark_to_market_at_end: bool = True
    initial_equity: float = 10_000.0
    bar_minutes: float = 1.0
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
        """Bars per year at ``bar_minutes``, from ``performance.periods_per_year`` (the one function
        of the bar size and a named calendar; the 24/7 default gives ``minutes_per_year``)."""
        return periods_per_year(self.bar_minutes)


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
    # Exposure mode only (run_exposure_backtest): mode "exposure"; the target exposure in force over
    # each bar (after that bar's open fill); every target evaluated at a decision bar.
    mode: str = "discrete"
    target_path: Optional[np.ndarray] = None
    targets: List[Dict[str, Any]] = field(default_factory=list)
    # Private: what the strategy was asked and answered at each bar (exit requests, each order's
    # tp/sl/max_hold); only ``assert_no_lookahead`` reads it. Not part of equality or repr.
    _trace: List[Dict[str, Any]] = field(default_factory=list, repr=False, compare=False)

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
    """Simulate ``strategy`` over the aligned ``signals`` and ``bars`` (same length). An
    ``ExposureStrategy`` goes to ``run_exposure_backtest``."""
    if isinstance(strategy, ExposureStrategy):
        return run_exposure_backtest(signals, bars, strategy, config)
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
    trace: List[Dict[str, Any]] = []
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
                trace.append({"bar": t, "kind": "exit", "request": pending_exit})
        elif t >= warmup and t < n - 1:
            pending_entry = strategy.decide(signals, t)
            if pending_entry is not None:
                decisions.append({"bar": t, "side": pending_entry.side, "size": pending_entry.size_frac,
                                  "reason": pending_entry.reason})
                o_ = pending_entry
                trace.append({"bar": t, "kind": "order", "tp": o_.tp, "sl": o_.sl, "tp_is_offset": o_.tp_is_offset,
                              "max_hold": o_.max_hold})
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
                          decisions, position, summary, cfg, _trace=trace)


# ------------------------------------------------------------------ exposure mode
def run_exposure_backtest(signals: SignalFrame, bars: Bars, strategy: ExposureStrategy,
                          config: Optional[BacktestConfig] = None) -> BacktestResult:
    """Simulate a target-exposure strategy over the aligned ``signals`` and ``bars``.

    Per bar t:

    1. at the open: a queued target is filled. The traded notional is |target - held exposure| x the
       equity at that open; the fill is the mid moved adversely by half-spread + slippage and the fee
       is charged on the notional (``BacktestConfig``'s cost fields, as in the discrete engine). The
       book then holds a signed QUANTITY = target x equity / open, so its exposure drifts with the price;
    2. at the close: equity is marked (cash + quantity x close) and ``position[t]`` is the drifted
       exposure quantity x close / equity. When t % decide_every == 0, t >= warmup and t < n - 1, the
       strategy's ``target(s, t, current)`` (current = ``position[t]``) is clipped to
       +-max_abs_exposure; if |target - current| > band it is queued for bar t + 1's open (with
       ``trade_to_band_edge``: current + sign(d) x (|d| - band), d = target - current). No stops;
    3. at the last close, with ``mark_to_market_at_end``, the book is closed out (costed like a fill).

    Returns a ``BacktestResult`` with ``mode`` = "exposure", ``trades`` = [], ``decisions`` = one
    record per fill {bar (the decision bar, as in the discrete engine), fill_bar (bar + 1; the last bar
    for the close-out), from, to, notional, costs, reason ("target", "band_edge", or "EOW" for the
    close-out), side ("LONG" for a buy, "SHORT" for a sell, so order markers work), fill (the price)}, ``targets`` = every target
    evaluated {bar, current, target (None when not finite: no decision), queued (the queued exposure
    or None)}, and ``target_path`` (the target in force over each bar). The summary has every key
    ``summarize`` gives, with ``n_trades`` = ``n_rebalances`` (the strategy's fills, the close-out
    excluded), so an activity floor reads one column for both modes; plus ``mean_abs_exposure``,
    ``traded_notional`` ($, close-out included), ``turnover`` (traded_notional / initial equity),
    ``cost_drag`` (costs / initial equity), ``breakeven_cost_bps`` (gross P&L / traded notional x 1e4
    x 2, a round trip) and ``gross_edge_per_trade_bps`` (a round trip of notional counts as one trade,
    so it equals ``breakeven_cost_bps``). ``avg_hold_bars`` is the mean number of bars from a fill to
    the next fill or the block's end; ``exposure`` the share of bars with a nonzero position.
    """
    cfg = config or BacktestConfig()
    n = len(bars)
    if len(signals) != n:
        raise ValueError(f"signals ({len(signals)}) and bars ({n}) must be aligned")
    slip, fee = cfg.slip_rate, cfg.fee_rate
    init = float(cfg.initial_equity)
    every = max(1, int(strategy.decide_every))
    band, cap = float(strategy.band), abs(float(strategy.max_abs_exposure))
    to_edge = bool(strategy.trade_to_band_edge)
    warmup = int(strategy.warmup())
    cash = gross_cash = init
    qty = 0.0
    held_target = 0.0
    equity = np.empty(n + 1)
    equity_gross = np.empty(n + 1)
    equity[0] = equity_gross[0] = init
    position = np.zeros(n)
    target_path = np.zeros(n)
    decisions: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []
    pending: Optional[tuple] = None
    tot = {"notional": 0.0, "costs": 0.0, "fees": 0.0}

    def fill(t, mid, to, decided_at, reason) -> bool:
        nonlocal cash, gross_cash, qty
        eq = cash + qty * mid
        if not (eq > 0 and mid > 0):
            return False
        new_qty = to * eq / mid
        dq = new_qty - qty
        notional = abs(dq) * mid
        fee_paid = fee * notional
        costs = notional * slip + fee_paid
        frm = qty * mid / eq
        cash -= dq * mid + costs
        gross_cash -= dq * mid
        tot["notional"] += notional
        tot["costs"] += costs
        tot["fees"] += fee_paid
        decisions.append({"bar": decided_at, "fill_bar": t, "from": float(frm), "to": float(to),
                          "notional": float(notional), "costs": float(costs), "reason": reason,
                          "side": "LONG" if dq > 0 else "SHORT", "fill": float(mid * (1 + np.sign(dq) * slip))})
        qty = new_qty
        return True

    for t in range(n):
        o, c = bars.open[t], bars.close[t]
        if pending is not None:
            to, decided_at, reason = pending
            if fill(t, o, to, decided_at, reason):
                held_target = to
            pending = None
        target_path[t] = held_target
        eq_close = cash + qty * c
        current = qty * c / eq_close if eq_close > 0 else 0.0
        position[t] = current
        if t >= warmup and t < n - 1 and t % every == 0:
            raw = strategy.target(signals, t, current)
            rec = {"bar": t, "current": float(current), "target": None, "queued": None}
            if raw is not None and math.isfinite(raw):
                tgt = min(cap, max(-cap, float(raw)))
                rec["target"] = tgt
                d = tgt - current
                if abs(d) > band:
                    to = current + math.copysign(abs(d) - band, d) if to_edge else tgt
                    pending = (float(to), t, "band_edge" if to_edge else "target")
                    rec["queued"] = float(to)
            targets.append(rec)
        if t == n - 1 and cfg.mark_to_market_at_end and qty != 0.0:
            fill(t, c, 0.0, t, "EOW")
        equity[t + 1] = cash + qty * c
        equity_gross[t + 1] = gross_cash + qty * c

    net_ret = np.diff(equity) / equity[:-1]
    gross_ret = np.diff(equity_gross) / equity_gross[:-1]
    summary = summarize(equity, net_ret, gross_ret, [], tot["fees"], float(np.mean(position != 0)) if n else 0.0,
                        cfg.periods_per_year)
    fills = [d["fill_bar"] for d in decisions if d["reason"] != "EOW"]
    gross_pnl = float(equity_gross[-1] - init)
    be = breakeven_cost_bps(gross_pnl, tot["notional"])
    summary.update(
        n_trades=len(fills), n_rebalances=len(fills),
        avg_hold_bars=float(np.mean(np.diff(fills + [n]))) if fills else 0.0,
        mean_abs_exposure=float(np.mean(np.abs(position))) if n else 0.0,
        traded_notional=float(tot["notional"]), turnover=float(tot["notional"] / init),
        cost_drag=float(tot["costs"] / init), breakeven_cost_bps=be, gross_edge_per_trade_bps=be,
        costs_paid=float(tot["costs"]), gross_pnl=gross_pnl, net_pnl=float(equity[-1] - init))
    return BacktestResult(getattr(strategy, "name", type(strategy).__name__), equity, equity_gross, [], decisions,
                          position, summary, cfg, mode="exposure", target_path=target_path, targets=targets)


def fill_events(result: BacktestResult) -> np.ndarray:
    """[N] the exposure each of the strategy's fills traded to, at its fill bar; NaN on bars without a
    fill (the end-of-block close-out is not the strategy's and is left out)."""
    events = np.full(len(result.position), np.nan)
    for d in result.decisions:
        if d.get("reason") != "EOW":
            events[d["fill_bar"]] = d["to"]
    return events


@dataclass
class _ReplayFills(ExposureStrategy):
    """Replays a schedule of fills (the timing null): at decision bar t it queues ``events[t + 1]``
    when that is finite, so every fill lands on its scheduled bar (a fill scheduled at bar 0 cannot be
    decided and is dropped)."""

    name: ClassVar[str] = "timing_null"
    events: Optional[np.ndarray] = None
    decide_every: int = 1
    band: float = 0.0

    def target(self, s, t, current):
        want = float(self.events[t + 1])
        return want if math.isfinite(want) else current


TIMING_NULL_SEED = 0
TIMING_NULL_MARGIN = 720          # shifts stay at least min(720, n // 4) bars away from 0 and n


def circular_shift_null(signals: SignalFrame, bars: Bars, result: BacktestResult,
                        config: Optional[BacktestConfig] = None, seeds: Optional[int] = None) -> Dict[str, float]:
    """Where an exposure strategy ranks among copies of its own timing shifted in time.

    The strategy's own exposure path, as its schedule of fills (``fill_events``: the exposure each
    rebalance traded to, at its fill bar), is shifted circularly by ``seeds`` offsets (default: the
    config's ``random_seeds``) drawn by ``np.random.default_rng(TIMING_NULL_SEED)`` from [m, n - m],
    m = min(720, n // 4). Each shifted schedule is re-traded and re-costed by ``run_exposure_backtest``
    on the shifted path's own exposure changes (the notional at each shifted fill is |to - the drifted
    exposure then| x the equity then); the strategy itself is never called. The same targets, the same
    number of rebalances, random timing. Unshifted, the replay reproduces the strategy's equity.

    Returns the keys of ``random_same_frequency`` (``n_seeds``, ``random_mean_total_return``,
    ``random_mean_sharpe_net``, ``random_p05_total_return``, ``random_p95_total_return``,
    ``random_mean_gross_return``, ``percentile_total_return``, ``percentile_sharpe_net``,
    ``percentile_gross_return``) plus ``null`` = "circular_shift", ``shift_min`` and ``shift_max``.
    Without rebalances, seeds or room to shift: ``n_seeds`` = 0 and NaN percentiles.
    """
    cfg = config or result.config
    k = int(seeds if seeds is not None else cfg.random_seeds)
    n = len(bars)
    m = min(TIMING_NULL_MARGIN, n // 4)
    if result.mode != "exposure" or k <= 0 or n < 2 or not result.summary.get("n_rebalances", 0):
        return {"n_seeds": 0, "null": "circular_shift", "percentile_total_return": float("nan"),
                "percentile_sharpe_net": float("nan")}
    shifts = np.random.default_rng(TIMING_NULL_SEED).integers(m, n - m + 1, size=k)
    events = fill_events(result)
    init = float(cfg.initial_equity)
    rets, sharpes, grosses = [], [], []
    for shift in shifts:
        r = run_exposure_backtest(signals, bars, _ReplayFills(events=np.roll(events, int(shift))), cfg)
        rets.append(r.summary["total_return"])
        sharpes.append(r.summary["sharpe_net"])
        grosses.append(r.summary["gross_pnl"] / init)
    rets, sharpes, grosses = np.array(rets), np.array(sharpes), np.array(grosses)
    gross = result.summary.get("gross_pnl", 0.0) / init
    return {
        "n_seeds": k, "null": "circular_shift", "shift_min": m, "shift_max": n - m,
        "random_mean_total_return": float(rets.mean()), "random_mean_sharpe_net": float(sharpes.mean()),
        "random_p05_total_return": float(np.percentile(rets, 5)), "random_p95_total_return": float(np.percentile(rets, 95)),
        "random_mean_gross_return": float(grosses.mean()),
        "percentile_total_return": float(100.0 * np.mean(rets < result.summary["total_return"])),
        "percentile_sharpe_net": float(100.0 * np.mean(sharpes < result.summary["sharpe_net"])),
        "percentile_gross_return": float(100.0 * np.mean(grosses < gross)),
    }


# ------------------------------------------------------------------ baselines
def _mean_fill_size(result: BacktestResult) -> float:
    """The mean size of the positions ``result`` opened (defined in ``random_same_frequency``);
    1.0 when no order opened one."""
    sizes = np.clip([float(d.get("size", 1.0)) for d in result.decisions], 0.0, 1.0)
    sizes = sizes[sizes > 0]
    return float(sizes.mean()) if len(sizes) else 1.0


def random_same_frequency(signals: SignalFrame, bars: Bars, result: BacktestResult,
                          config: Optional[BacktestConfig] = None, seeds: Optional[int] = None) -> Dict[str, float]:
    """Where the strategy ranks among random entries with its trade rate, holding time AND mean
    position size: ``RandomSignal`` with seeds 0..seeds-1 (default: the config's ``random_seeds``),
    so the same call gives the same numbers.

    Matched to ``result``: the entry rate is its trade count over its flat bars, the holding time its
    mean holding time (rounded, at least 1 bar), and ``size_frac`` its mean position size: the
    arithmetic mean, over the orders the strategy placed (``result.decisions``, one per entry; bars
    where ``decide`` returned None do not count), of each order's ``size_frac`` clipped to [0, 1] as
    the engine clips it at the fill, leaving out orders whose clipped size is 0 (they open no
    position). Every order counts once, whatever its holding time. After costs a return is mostly
    cost x size x trade count, so a null trading at full size while the strategy sizes down pays more
    costs and would make the strategy look skilled when it is not.

    Returns ``n_seeds``, ``trade_rate``, ``hold_bars``, ``size_frac``; over the seeds the mean, 5th and
    95th percentile of the net total return, the mean net Sharpe and the mean gross return (gross P&L
    before costs over the initial equity); and the strategy's percentile among the seeds (the share of
    seeds strictly below it, in %) by net return, net Sharpe and gross return. Without trades or
    seeds only ``n_seeds`` = 0 and NaN percentiles of net return and net Sharpe.

    An exposure-mode ``result`` (``result.mode == "exposure"``) gets the circular-shift timing null
    (``circular_shift_null``), with the same keys, instead.
    """
    if result.mode == "exposure":
        return circular_shift_null(signals, bars, result, config, seeds)
    cfg = config or result.config
    k = int(seeds if seeds is not None else cfg.random_seeds)
    n_tr = result.summary["n_trades"]
    if n_tr == 0 or k <= 0:
        return {"n_seeds": 0, "percentile_total_return": float("nan"), "percentile_sharpe_net": float("nan")}
    hold = max(1, int(round(result.summary["avg_hold_bars"])) or 1)
    flat_bars = max(1, len(bars) - int(result.summary["exposure"] * len(bars)))
    rate = min(1.0, n_tr / flat_bars)
    size = _mean_fill_size(result)
    init = float(cfg.initial_equity)
    rets, sharpes, grosses = [], [], []
    for seed in range(k):
        r = run_backtest(signals, bars, RandomSignal(trade_rate=rate, hold_bars=hold, seed=seed, size_frac=size), cfg)
        rets.append(r.summary["total_return"])
        sharpes.append(r.summary["sharpe_net"])
        grosses.append(r.summary["gross_pnl"] / init)
    rets, sharpes, grosses = np.array(rets), np.array(sharpes), np.array(grosses)
    gross = result.summary.get("gross_pnl", 0.0) / init
    return {
        "n_seeds": k, "trade_rate": rate, "hold_bars": hold, "size_frac": size,
        "random_mean_total_return": float(rets.mean()), "random_mean_sharpe_net": float(sharpes.mean()),
        "random_p05_total_return": float(np.percentile(rets, 5)), "random_p95_total_return": float(np.percentile(rets, 95)),
        "random_mean_gross_return": float(grosses.mean()),
        "percentile_total_return": float(100.0 * np.mean(rets < result.summary["total_return"])),
        "percentile_sharpe_net": float(100.0 * np.mean(sharpes < result.summary["sharpe_net"])),
        "percentile_gross_return": float(100.0 * np.mean(grosses < gross)),
    }


def backtest(signals: SignalFrame, bars: Bars, strategy: Strategy, config: Optional[BacktestConfig] = None,
             baselines: bool = True) -> BacktestResult:
    """``run_backtest`` plus the baselines every report carries: buy-and-hold, always-flat, and
    random entries with the strategy's trade rate, holding time and mean size
    (``random_same_frequency``, ``config.random_seeds`` seeds); for an ``ExposureStrategy`` the
    random baseline under the same key is the circular-shift timing null (``circular_shift_null``)."""
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


def _default_probes(base: BacktestResult, n: int, count: int) -> List[int]:
    """Probe bars for the self-test. A one-bar peek at bar t' reads a bar that differs only when the
    probe is t' itself, so the probes are spread over the bars where the strategy was actually asked
    something (decisions, exit requests, evaluated targets) plus a uniform grid."""
    asked = sorted({int(d["bar"]) for d in base.decisions} | {int(r["bar"]) for r in base._trace}
                   | {int(d["bar"]) for d in base.targets})
    asked = [t for t in asked if t < n - 1]
    picks = [asked[int(i)] for i in np.linspace(0, len(asked) - 1, min(count, len(asked)))] if asked else []
    grid = [int(i) for i in np.linspace(n // 8, n - 2, 4)] if n > 8 else []
    return sorted(set(picks) | set(grid))


def assert_no_lookahead(frame: PredictionFrame, bars: Bars, make_strategy: Callable[[], Strategy], *,
                        var_scale: float, probes: Sequence[int] = (), config: Optional[BacktestConfig] = None,
                        seed: int = 0, n_probes: int = 24) -> None:
    """Perturb every prediction and bar AFTER t; everything the strategy was asked and answered up to
    t must be unchanged. Raises AssertionError naming the first differing bar. Compared, per probe t:

    * the ``decisions`` (bar, side, size, reason) with bar <= t;
    * the trace of the engine's questions to the strategy at bars <= t: every ``exit_signal`` answer,
      and each order's ``tp``, ``sl``, ``tp_is_offset`` and ``max_hold`` (the ``decisions`` dicts carry
      none of these);
    * the equity up to the mark at bar t's close.

    Exposure strategies: their fills up to bar t (``decisions`` with ``fill_bar`` <= t) and every
    target evaluated at a decision bar <= t (``targets``) must be unchanged; ``SignalFrame.build``
    rebuilds the EWMA sigma from the perturbed closes.

    Without ``probes``, up to ``n_probes`` bars are taken from those the base run asked the strategy
    about, plus a grid of 4 (see ``_default_probes``); a one-bar peek is caught only at its own bar."""
    cfg = config or BacktestConfig()
    rng = np.random.default_rng(seed)
    n = len(frame)
    base = run_backtest(SignalFrame.build(frame, var_scale), bars, make_strategy(), cfg)
    probes = list(probes) or _default_probes(base, n, n_probes)
    for t in probes:
        f2, b2 = _perturb_after(frame, bars, t, rng)
        other = run_backtest(SignalFrame.build(f2, var_scale), b2, make_strategy(), cfg)
        at = "fill_bar" if base.mode == "exposure" else "bar"   # an exposure record carries its fill (open t + 1)
        before = [d for d in base.decisions if d[at] <= t]
        after = [d for d in other.decisions if d[at] <= t]
        if before != after:
            raise AssertionError(f"look-ahead: decisions up to bar {t} changed when data after {t} changed")
        if [r for r in base._trace if r["bar"] <= t] != [r for r in other._trace if r["bar"] <= t]:
            raise AssertionError(f"look-ahead: exit requests or order levels up to bar {t} changed when data "
                                 f"after {t} changed")
        if [d for d in base.targets if d["bar"] <= t] != [d for d in other.targets if d["bar"] <= t]:
            raise AssertionError(f"look-ahead: targets up to bar {t} changed when data after {t} changed")
        # equity[t + 1] is the mark at bar t's close
        if not np.allclose(base.equity[: t + 2], other.equity[: t + 2], rtol=0, atol=1e-9):
            first = int(np.argmax(~np.isclose(base.equity[: t + 2], other.equity[: t + 2], rtol=0, atol=1e-9)))
            raise AssertionError(f"look-ahead: equity at bar {first - 1} changed when data after {t} changed")


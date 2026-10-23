"""NT-227 (DESK-1.6): the Strategy API and the six built-ins.

The real engine loop (DESK-1.7) does not exist yet, so ``drive`` is its A5.9 stand-in: a HistoricalSource over a
MarketStore, SimBroker + ExecutionEngine + Portfolio on a margin-like instrument (A5.13), per closed bar: exits, entries,
intrabar, then the close step (EOW > TIME > ``on_bar`` when ``t >= warmup_bars``, not the last bar, not a break).
The three TA rules are compared with the oracle (``strategy.backtest.run_backtest`` on ``ta_ma_cross`` / ``ta_rsi`` /
``ta_bollinger``) bar index by bar index: entry bar, side, exit bar and exit reason.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.exceptions import ComponentNotFoundError, ComponentValidationError, DuplicateRegistrationError
from neural_trade.desk.broker.sim import SimBroker
from neural_trade.desk.core.bus import EventBus
from neural_trade.desk.core.clock import SimClock
from neural_trade.desk.core.costs import RESEARCH_ZERO
from neural_trade.desk.core.execution import ExecutionEngine
from neural_trade.desk.core.model import (AccountKind, Bar, Fill, Instrument, Order, OrderType, Position,
                                          PositionSide, Side)
from neural_trade.desk.core.portfolio import Portfolio
from neural_trade.desk.core.risk import RiskGuard, RiskLimits
from neural_trade.desk.core.strategy import (BAR_COLUMNS, DeskStrategies, DeskStrategy, IndicatorsUnavailable,
                                             StrategyContext)
from neural_trade.desk.data.sources import HistoricalSource
from neural_trade.desk.data.store import MarketStore
from neural_trade.desk.strategies import (AlwaysFlat, BollingerBreakout, BuyAndHold, MACross, RandomEntries,
                                          RSIThreshold)

SYM = "BTC/USDT"
INS_ID = "BINANCE:BTCUSDT"
INS = Instrument(id=INS_ID, symbol=SYM, base="BTC", quote="USDT", price_increment=0.0, size_increment=0.0,
                 min_qty=0.0, min_notional=0.0, is_spot=False)
BUILTINS = ("always_flat", "bollinger_breakout", "buy_and_hold", "ma_cross", "random_entries", "rsi_threshold")


@dataclass
class Run:
    trades: list = field(default_factory=list)   # (entry_bar, exit_bar, side, entry tag, exit tag)
    calls: list = field(default_factory=list)    # (t, flat before the call)
    violations: list = field(default_factory=list)
    pf: Portfolio = None
    eng: ExecutionEngine = None
    ctx: StrategyContext = None

    @property
    def entries(self):
        return [(t[0], t[2]) for t in self.trades]


def to_store(df: pd.DataFrame, path) -> MarketStore:
    ts = pd.DatetimeIndex(df["datetime"]).asi8 // 10 ** 6
    store = MarketStore(path)
    store.write_bars(SYM, 1, pd.DataFrame({"ts": ts, "o": df["open"], "h": df["high"], "l": df["low"],
                                           "c": df["close"], "v": df["volume"]}), "t")
    return store


def drive(strategy: DeskStrategy, df, tmp_path, *, breaks=None, max_hold_default=30, warmup=None, source=None,
          setup=None, ctx_kwargs=None) -> Run:
    """A5.9 in miniature (see the module docstring); ``warmup`` overrides ``strategy.warmup_bars`` (to ask early);
    ``setup(run)`` runs before the first bar (a test's hook to the Portfolio)."""
    store = to_store(df, tmp_path / "m.sqlite")
    src = source(store) if source else HistoricalSource(store, SYM, 1)
    clock = SimClock(0)
    pf = Portfolio("acc", kind=AccountKind.BACKTEST, initial_cash=10_000.0, clock=clock)
    broker = SimBroker(INS, RESEARCH_ZERO, pf, clock)
    eng = ExecutionEngine(broker, pf, RiskGuard(RiskLimits.unlimited(), INS, pf, clock), EventBus(), clock,
                          lambda k, p: None)
    sid = strategy.name
    ctx = StrategyContext(execution=eng, source=src, strategy_id=sid, clock=clock, **(ctx_kwargs or {}))
    run = Run(pf=pf, eng=eng, ctx=ctx)
    pid, n = f"{INS_ID}-{sid}", len(df)
    warm = strategy.warmup_bars if warmup is None else warmup
    st = {"t": 0, "entry": None, "hold": max_hold_default, "open": None}

    def apply(evs):
        for e in evs:
            was_flat = pf.positions.get(pid) is None or pf.positions[pid].qty == 0
            eng.apply(e)
            if isinstance(e, Fill):
                tag = eng.order(e.client_id).tags.get("tag")
                if was_flat:
                    o = eng.order(e.client_id)
                    st["entry"], st["hold"] = st["t"], (o.bracket or {}).get("max_hold") or max_hold_default
                    st["open"] = (st["t"], "LONG" if e.side is Side.BUY else "SHORT", tag)
                else:
                    entry_bar, side, entry_tag = st["open"]
                    run.trades.append((entry_bar, st["t"], side, entry_tag, tag))

    def ask(t):
        before = set(eng.orders)
        flat = pf.positions.get(pid) is None or pf.positions[pid].qty == 0
        strategy.on_bar(ctx)
        run.calls.append((t, flat))
        for cid in set(eng.orders) - before:
            if not flat and not eng.orders[cid].reduce_only:
                run.violations.append((t, cid))

    if setup:
        setup(run)
    strategy.on_start(ctx)
    for t in range(n):
        bar = src.next_closed_bar()
        st["t"] = t
        ctx._advance(bar)
        clock.set(bar.ts_open_ms)
        apply(broker.fill_exits(bar))
        apply(broker.fill_entries(bar, pf.realised_equity()))
        apply(broker.on_bar_intrabar(bar))
        eng.set_marks({INS_ID: bar.close})
        clock.set(bar.ts_close_ms)
        broken = breaks is not None and bool(breaks[t])
        pos = pf.positions.get(pid)
        if pos is not None and pos.qty > 0:
            if broken or t == n - 1:
                eng.submit(Order(client_id=f"eow:{t}", instrument_id=INS_ID, account_id="acc", strategy_id=sid,
                                 side=Side.SELL if pos.side is PositionSide.LONG else Side.BUY,
                                 type=OrderType.MARKET, qty=pos.qty, reduce_only=True,
                                 tags={"fill_at": "close", "reason": "EOW", "tag": "EOW"}))
                apply(broker.fill_at_close(bar))
            elif t - st["entry"] + 1 >= st["hold"]:
                eng.close_position(pid, tag="TIME")
            else:
                ask(t)
        elif t >= warm and t < n - 1 and not broken:
            ask(t)
        pf.mark({INS_ID: bar.close}, bar.ts_close_ms)
    strategy.on_stop(ctx)
    return run


def oracle(name, params, df, breaks=None):
    from neural_trade.strategy import BacktestConfig, Bars, SignalFrame, build_strategy, run_backtest
    from neural_trade.strategy.ta_rules import price_only_frame

    close = df["close"].to_numpy(float)
    bars = Bars(df["open"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float), close, breaks)
    sig = SignalFrame.build(price_only_frame(close), 1.0)
    return run_backtest(sig, bars, build_strategy(name, params), BacktestConfig())


def n_breaks(n):
    b = np.zeros(n, dtype=bool)
    b[[400, 401, 1200, 2500]] = True
    return b


# ------------------------------------------------------------------ (1) parameters
@pytest.mark.parametrize("name", BUILTINS)
def test_params_schema_lists_every_param_spec_with_bounds(name):
    cls = DeskStrategies.get(name)
    schema = cls.params_schema()
    assert [s["name"] for s in schema] == [p.name for p in cls.PARAMS]
    for s, p in zip(schema, cls.PARAMS):
        assert (s["kind"], s["default"], s["minimum"], s["maximum"]) == (p.kind, p.default, p.minimum, p.maximum)
        if p.kind in ("int", "float"):
            assert s["minimum"] is not None and s["maximum"] is not None, f"{name}.{p.name} has no bounds"
    assert cls().params == {p.name: p.default for p in cls.PARAMS}   # defaults construct


@pytest.mark.parametrize("cls,bad,msg", [
    (MACross, {"fast": 1}, "fast: 1 is below the minimum 2"),
    (MACross, {"slow": 241}, "slow: 241 is above the maximum 240"),
    (MACross, {"fast": 30, "slow": 30, "max_hold": 0}, "max_hold: 0 is below the minimum 1"),
    (MACross, {"size": 1.5}, "size: 1.5 is above the maximum 1.0"),
    (RSIThreshold, {"period": 61}, "period: 61 is above the maximum 60"),
    (RSIThreshold, {"lower": 9.0}, "lower: 9.0 is below the minimum 10.0"),
    (RSIThreshold, {"upper": 95.0}, "upper: 95.0 is above the maximum 90.0"),
    (BollingerBreakout, {"period": 4}, "period: 4 is below the minimum 5"),
    (BollingerBreakout, {"k": 3.5}, "k: 3.5 is above the maximum 3.0"),
    (RandomEntries, {"trade_rate": 1.5}, "trade_rate: 1.5 is above the maximum 1.0"),
    (RandomEntries, {"seed": -1}, "seed: -1 is below the minimum 0"),
    (BuyAndHold, {"size": 0.0}, "size: 0.0 is below the minimum 0.01"),
])
def test_out_of_range_param_raises_naming_the_bound(cls, bad, msg):
    with pytest.raises(ValueError, match=msg):
        cls(**bad)


def test_cross_parameter_rules_and_unknown_keys_raise():
    with pytest.raises(ValueError, match="lower < exit_level < upper"):
        RSIThreshold(lower=40.0, exit_level=40.0)
    with pytest.raises(ValueError, match="lower < exit_level < upper"):
        RSIThreshold(upper=60.0, exit_level=60.0)
    with pytest.raises(ValueError, match="unknown parameter"):
        AlwaysFlat(speed=3)


# ------------------------------------------------------------------ (2) the TA rules against the oracle
MA_SETS = [dict(fast=10, slow=30), dict(fast=3, slow=45, max_hold=40), dict(fast=8, slow=35, allow_short=False)]
RSI_SETS = [dict(period=14, lower=40.0, upper=60.0), dict(period=7, lower=35.0, upper=65.0, exit_level=45.0, max_hold=60),
            dict(period=21, lower=40.0, upper=60.0, allow_short=False)]
BB_SETS = [dict(period=20, k=2.0), dict(period=10, k=1.0, max_hold=50), dict(period=40, k=1.2, allow_short=False)]
TA = [("ta_ma_cross", MACross, MA_SETS), ("ta_rsi", RSIThreshold, RSI_SETS), ("ta_bollinger", BollingerBreakout, BB_SETS)]
HEAD = 700   # bars of synthetic_bars used by the parity tests (time budget)


def compare(oname, cls, params, df, tmp_path, breaks=None):
    res = oracle(oname, params, df, breaks)
    run = drive(cls(**params), df, tmp_path, breaks=breaks)
    ref = [(t.entry_bar, t.exit_bar, t.side, t.entry_reason, t.exit_reason) for t in res.trades]
    assert run.trades == ref, f"{oname} {params}"
    assert not run.violations
    np.testing.assert_allclose([e for _, e in run.pf.equity_curve], res.equity[1:], rtol=0, atol=1e-6)
    return run


@pytest.mark.parametrize("oname,cls,sets", TA, ids=[t[0] for t in TA])
def test_ta_rules_place_the_same_entry_bars_as_the_oracle(oname, cls, sets, synthetic_bars, tmp_path):
    sides = set()
    for i, params in enumerate(sets):
        sub = tmp_path / str(i)
        sub.mkdir()
        run = compare(oname, cls, params, synthetic_bars.iloc[:HEAD], sub)
        assert len(run.trades) >= 6, f"{params}: only {len(run.trades)} trades, the comparison would be vacuous"
        sides |= {s for _, s in run.entries}
        if params.get("allow_short") is False:
            assert {s for _, s in run.entries} == {"LONG"}
    assert sides == {"LONG", "SHORT"}


@pytest.mark.parametrize("oname,cls,sets", TA, ids=[t[0] for t in TA])
def test_ta_rules_agree_with_the_oracle_across_data_breaks(oname, cls, sets, synthetic_bars, tmp_path):
    brk = np.zeros(len(synthetic_bars), dtype=bool)
    brk[100::150] = True
    run = compare(oname, cls, sets[0], synthetic_bars, tmp_path, breaks=brk)
    at = set(np.flatnonzero(brk))
    assert {t for t, _ in run.calls}.isdisjoint(at)    # never asked on a break bar
    assert any(tr[4] == "EOW" and tr[1] in at for tr in run.trades)   # a position open at a hole closes there


def frame_from_close(c):
    c = np.asarray(c, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"datetime": pd.date_range("2025-10-11 02:30:00", periods=len(c), freq="1min", tz="UTC"),
                         "open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0})


def flat_series(seed, n=1500):
    """Integer-tick prices with flat stretches of 3-200 bars, an exact repeated block (MA ties) and a fall that ends in
    a long flat stretch (Wilder's average loss decays towards 0: RSI flips between 0, 50 and 100)."""
    rng = np.random.default_rng(seed)
    c = np.round(100 + np.cumsum(rng.normal(0, 0.8, n)))
    for a, ln in ((150, 3), (300, 40), (700, 200), (1000, 17)):
        c[a:a + ln] = c[a]
    c[1100:1160] = c[1040:1100]
    c[1200:1230] = c[1199] - np.arange(1, 31)    # a fall ...
    c[1230:1400] = c[1229]                       # ... then flat
    return c


FLAT_SETS = {"ta_ma_cross": [dict(fast=3, slow=30)], "ta_rsi": [dict(period=5), dict(period=14, lower=40.0, upper=60.0)],
             "ta_bollinger": [dict(period=10, k=1.0)]}


@pytest.mark.parametrize("oname,cls,sets", TA, ids=[t[0] for t in TA])
def test_ta_rules_agree_with_the_oracle_on_flat_stretches_and_ties(oname, cls, sets, tmp_path):
    for seed in (1,):
        df = frame_from_close(flat_series(seed))
        for i, params in enumerate(FLAT_SETS[oname]):
            sub = tmp_path / f"{seed}_{i}"
            sub.mkdir()
            run = compare(oname, cls, params, df, sub)
            assert run.trades, (oname, params)


def test_rsi_flat_after_a_fall_period_5_matches_the_oracle_trade_for_trade(tmp_path):
    """QA: a fall followed by a flat stretch; the truncated-window RSI traded 15 times, the full-history oracle 27."""
    c = np.r_[np.full(30, 100.0), 100 - np.arange(1, 21) * 1.0, np.full(150, 80.0)]
    rng = np.random.default_rng(3)
    c = np.round(np.r_[c, c[-1] + np.cumsum(rng.normal(0, 1, 700))] + 0.0)
    run = compare("ta_rsi", RSIThreshold, dict(period=5), frame_from_close(c), tmp_path)
    assert len(run.trades) >= 20


def probe_values(cls, params, c, tmp_path, *, every=3, ctx_kwargs=None):
    """Drive ``cls`` over ``c`` asking for its decision value only on every ``every``-th bar (the bars in between are
    caught up by the strategy's state); returns {bar index: value}."""
    df = frame_from_close(c)
    index = {int(t): i for i, t in enumerate(pd.DatetimeIndex(df["datetime"]).asi8 // 10 ** 6)}
    seen = {}
    read = {"rsi_threshold": "_update", "ma_cross": "_diff", "bollinger_breakout": "_bands"}[cls.name]

    class Probe(cls):
        def on_bar(self, ctx):
            t = index[ctx.bar.ts_open_ms]
            if t % every == 0:
                seen[t] = getattr(self, read)(ctx)

    drive(Probe(**params), df, tmp_path, warmup=0, ctx_kwargs=ctx_kwargs)
    return seen


def test_incremental_rsi_state_equals_wilder_rsi_bit_for_bit_and_catches_up_skipped_bars(tmp_path):
    from neural_trade.strategy.ta_rules import wilder_rsi
    c = flat_series(4)
    seen = probe_values(RSIThreshold, dict(period=7), c, tmp_path)
    ref = wilder_rsi(c, 7)
    ts = np.array(sorted(t for t, r in seen.items() if r is not None))
    assert len(ts) > 450 and set(seen) == set(range(0, len(c) - 1, 3))     # two of every three bars skipped
    assert np.array_equal([seen[t] for t in ts], ref[ts])


def hard_close(seed, n=1500):
    """flat_series scaled, with a 1000x jump and a tick-rounded stretch."""
    c = flat_series(seed, n) * 37.13
    c[800:] *= 1e3
    c[1300:1450] = np.round(c[1300:1450] / 1e-3) * 1e-3
    return c


@pytest.mark.parametrize("params", [dict(fast=2, slow=30), dict(fast=7, slow=45), dict(fast=30, slow=240)])
def test_ma_cross_decision_value_is_the_oracles_full_array_value_bit_for_bit(params, tmp_path):
    from neural_trade.strategy.ta_rules import sma
    for seed in (4, 6):
        c = hard_close(seed)
        sub = tmp_path / str(seed)
        sub.mkdir()
        seen = probe_values(MACross, params, c, sub)
        ts = sorted(seen)
        full = sma(c, params["fast"]) - sma(c, params["slow"])
        got = np.array([seen[t] for t in ts])
        assert np.array_equal(got, full[ts], equal_nan=True)
        assert np.isfinite(got).sum() > 350


@pytest.mark.parametrize("params", [dict(period=5, k=1.0), dict(period=20, k=2.0), dict(period=120, k=3.0)])
def test_bollinger_bands_are_the_oracles_full_array_values_bit_for_bit(params, tmp_path):
    from neural_trade.strategy.ta_rules import rolling_std, sma
    for seed in (4, 6):
        c = hard_close(seed)
        sub = tmp_path / str(seed)
        sub.mkdir()
        seen = probe_values(BollingerBreakout, params, c, sub)
        ts = sorted(seen)
        got = np.array([seen[t] for t in ts])
        ref = np.c_[c[ts], sma(c, params["period"])[ts], rolling_std(c, params["period"])[ts]]
        assert np.array_equal(got, ref, equal_nan=True)
        assert np.isfinite(got[:, 2]).sum() > 350


def test_rsi_exactly_at_the_exit_level_exits_both_sides_like_the_oracle(tmp_path):
    """period 2: 100 flat, 96 (RSI 0, BUY), 98 (avg gain 1 = avg loss 1: RSI exactly 50, the LONG exits), flat, 102
    (RSI above 70, SELL), 100 (RSI exactly 50 again, the SHORT exits): ``>=`` / ``<=`` as the oracle."""
    from neural_trade.strategy.ta_rules import wilder_rsi
    rng = np.random.default_rng(11)
    c = np.r_[np.full(10, 100.0), 96.0, 98.0, np.full(6, 98.0), 102.0, 100.0, np.full(6, 100.0),
              np.round(100 + np.cumsum(rng.normal(0, 1, 300)))]
    r = wilder_rsi(c, 2)
    assert r[11] == 50.0 and r[19] == 50.0 and r[10] < 30 and r[18] > 70
    run = compare("ta_rsi", RSIThreshold, dict(period=2), frame_from_close(c), tmp_path)
    assert [t[:3] for t in run.trades[:2]] == [(11, 12, "LONG"), (19, 20, "SHORT")], run.trades[:2]
    assert [t[4] for t in run.trades[:2]] == ["RSI_MID", "RSI_MID"], run.trades[:2]


def test_bollinger_close_exactly_on_a_band_does_not_enter(tmp_path):
    """period 5, k 2: four 100s and a 105 put the upper band at exactly 105 (mean 101, sd 2), four 100s and a 95 the
    lower band at exactly 95 (with a population sd the last of 5 values is at most 2 sd from their mean): the
    oracle's strict ``>`` / ``<`` enters on neither. With k 1.9 the same bars are real breakouts."""
    from neural_trade.strategy.ta_rules import rolling_std, sma
    c = np.r_[np.full(30, 100.0), 105.0, np.full(30, 100.0), 95.0, np.full(60, 100.0)]
    mid, sd = sma(c, 5), rolling_std(c, 5)
    assert mid[30] + 2.0 * sd[30] == 105.0 and mid[61] - 2.0 * sd[61] == 95.0
    df = frame_from_close(c)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert compare("ta_bollinger", BollingerBreakout, dict(period=5, k=2.0), df, tmp_path / "a").trades == []
    near = compare("ta_bollinger", BollingerBreakout, dict(period=5, k=1.9), df, tmp_path / "b")
    assert [(t[0], t[2]) for t in near.trades] == [(31, "LONG"), (62, "SHORT")]


class _CountingSource:
    """A HistoricalSource that records every ``history(n)`` asked for."""

    def __init__(self, store):
        self.inner, self.asked = HistoricalSource(store, SYM, 1), []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def history(self, n):
        self.asked.append(int(n))
        return self.inner.history(n)


@pytest.mark.parametrize("cls,params,window", [(MACross, dict(fast=5, slow=40), 40),
                                               (BollingerBreakout, dict(period=20, k=2.0), 20)])
def test_a_seed_cap_limits_what_a_live_strategy_reads_and_stays_close_to_full_history(cls, params, window,
                                                                                     synthetic_bars, tmp_path):
    """``max_seed_bars``: the first call reads at most the cap, later calls only the new bars; the decisions equal the
    full-history ones here (the values are equal up to rounding, ``test_online_kernels``), not bit for bit in general."""
    df = synthetic_bars.iloc[:900]
    cap = 3 * window
    start_ts = int(pd.Timestamp(df["datetime"].iloc[500]).value // 10 ** 6)
    sources = []

    def counting(store):
        sources.append(_CountingSource(store))
        return sources[-1]

    class Late(cls):          # a bot started at bar 500: its first on_bar is there
        def on_bar(self, ctx):
            if ctx.bar.ts_open_ms >= start_ts:
                super().on_bar(ctx)

    capped = drive(Late(**params), df, tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path,
                   source=counting, ctx_kwargs={"max_seed_bars": cap})
    full = drive(Late(**params), df, tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path,
                 source=counting)
    assert max(sources[0].asked) <= cap and sources[0].asked.count(cap) == 1
    assert max(sources[1].asked) > 500                 # without a cap the seed reads the whole history
    assert capped.trades == full.trades and capped.trades
    with pytest.raises(ValueError, match="max_seed_bars"):
        StrategyContext(execution=None, source=None, strategy_id="z", clock=SimClock(0), max_seed_bars=0)


def test_rsi_ignores_the_seed_cap_and_seeds_from_all_history(synthetic_bars, tmp_path):
    from neural_trade.strategy.ta_rules import wilder_rsi
    c = synthetic_bars["close"].to_numpy(float)[:400]
    seen = probe_values(RSIThreshold, dict(period=5), c, tmp_path, ctx_kwargs={"max_seed_bars": 10})
    ts = sorted(t for t, r in seen.items() if r is not None)
    assert len(ts) > 120 and np.array_equal([seen[t] for t in ts], wilder_rsi(c, 5)[ts])


@pytest.mark.parametrize("cls,params", [(RSIThreshold, dict(period=5)), (MACross, dict(fast=3, slow=30)),
                                        (BollingerBreakout, dict(period=10, k=1.0))])
def test_a_restarted_strategy_seeds_again_from_history(cls, params, tmp_path):
    df = frame_from_close(flat_series(5)[:600])
    first = drive(cls(**params), df, tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path)
    strat = cls(**params)
    second = drive(strat, df, tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path)
    again = drive(strat, df, tmp_path / "c" if (tmp_path / "c").mkdir() is None else tmp_path)   # same instance, on_start
    assert first.trades == second.trades == again.trades and first.trades


# ------------------------------------------------------------------ the baselines
def test_buy_and_hold_enters_once_and_holds_to_the_end(synthetic_bars, tmp_path):
    from neural_trade.strategy import BacktestConfig, Bars, SignalFrame, build_strategy, run_backtest
    from neural_trade.strategy.ta_rules import price_only_frame

    run = drive(BuyAndHold(), synthetic_bars, tmp_path)
    n = len(synthetic_bars)
    assert run.trades == [(1, n - 1, "LONG", "buy_and_hold", "EOW")]
    assert len([c for c in run.eng.orders]) == 2   # the entry and the EOW close; no TIME exit at the engine default of 30
    close = synthetic_bars["close"].to_numpy(float)
    ref = run_backtest(SignalFrame.build(price_only_frame(close), 1.0),
                       Bars(synthetic_bars["open"], synthetic_bars["high"], synthetic_bars["low"], close),
                       build_strategy("buy_and_hold", {}), BacktestConfig())
    assert (ref.trades[0].entry_bar, ref.trades[0].exit_bar) == (1, n - 1)
    np.testing.assert_allclose([e for _, e in run.pf.equity_curve], ref.equity[1:], rtol=0, atol=1e-6)


def test_always_flat_never_orders(synthetic_bars, tmp_path):
    run = drive(AlwaysFlat(), synthetic_bars, tmp_path)
    assert run.eng.orders == {} and run.trades == [] and len(run.calls) == len(synthetic_bars) - 1
    assert {e for _, e in run.pf.equity_curve} == {10_000.0}


def test_random_entries_are_seeded_reproducible_and_distinct_per_seed(synthetic_bars, tmp_path):
    df = synthetic_bars.iloc[:1200]
    runs = []
    for i, seed in enumerate((7, 7, 8)):
        sub = tmp_path / str(i)
        sub.mkdir()
        runs.append(drive(RandomEntries(trade_rate=0.05, hold_bars=10, seed=seed), df, sub))
    assert runs[0].trades == runs[1].trades and len(runs[0].trades) > 10
    assert runs[0].entries != runs[2].entries
    assert {s for _, s in runs[0].entries} == {"LONG", "SHORT"}
    assert all(t[1] - t[0] == 10 or t[4] == "EOW" for t in runs[0].trades)   # held hold_bars, then the TIME exit
    never, always = tmp_path / "never", tmp_path / "always"
    never.mkdir(), always.mkdir()
    assert drive(RandomEntries(trade_rate=0.0), df.iloc[:100], never).trades == []
    assert len(drive(RandomEntries(trade_rate=1.0, hold_bars=5), df.iloc[:100], always).trades) > 10
    long_only = tmp_path / "long"
    long_only.mkdir()
    r = drive(RandomEntries(trade_rate=0.2, hold_bars=5, seed=7, allow_short=False), df.iloc[:300], long_only)
    assert r.trades and {s for _, s in r.entries} == {"LONG"}


class FakeCtx:
    """Just what RandomEntries reads: a flat position, the current bar, and a recorder for the order."""

    def __init__(self, ts):
        self.bar = Bar(INS_ID, 1, ts, 1.0, 1.0, 1.0, 1.0, 1.0, True)
        self.position = Position(instrument_id=INS_ID, account_id="a")
        self.orders = []

    def submit_market(self, side, **kw):
        self.orders.append((side, kw["max_hold"]))


def test_random_entries_draw_depends_on_seed_and_bar_time_only():
    """The draw of a bar is a function of (seed, bar open time): not of the order the bars are visited in, nor of
    where the run started. Different seeds draw different bars."""
    def entered(seed, ts):
        c = FakeCtx(ts)
        RandomEntries(trade_rate=0.3, seed=seed).on_bar(c)
        return c.orders

    stamps = [1_700_000_000_000 + 60_000 * i for i in range(400)]
    fwd = [entered(5, ts) for ts in stamps]
    assert [entered(5, ts) for ts in reversed(stamps)][::-1] == fwd
    assert 60 < sum(bool(o) for o in fwd) < 180
    assert {o[0][0] for o in fwd if o} == {"BUY", "SELL"}
    assert [bool(entered(6, ts)) for ts in stamps] != [bool(o) for o in fwd]


# ------------------------------------------------------------------ the engine contract as the strategies see it
class Recorder(DeskStrategy):
    name = "recorder"
    warmup_bars = 5

    def __init__(self, **p):
        super().__init__(**p)
        self.frames = []

    def on_bar(self, ctx):
        self.frames.append((ctx.bar.ts_open_ms, ctx.bars(50), ctx.bars(1), ctx.bars(10 ** 9)))


def test_on_bar_is_called_only_when_the_contract_says(synthetic_bars, tmp_path):
    brk = np.zeros(60, dtype=bool)
    brk[[20, 21]] = True
    r = drive(Recorder(), synthetic_bars.iloc[:60], tmp_path, breaks=brk)
    ts = [c[0] for c in r.calls]
    assert ts[0] == 5 and 59 not in ts and 20 not in ts and 21 not in ts   # warm-up, last bar, break bars
    assert ts == [t for t in range(5, 59) if t not in (20, 21)]


@pytest.mark.parametrize("cls,params", [(MACross, {}), (RSIThreshold, {"lower": 40.0, "upper": 60.0}),
                                        (BollingerBreakout, {"k": 1.0}), (RandomEntries, {"trade_rate": 0.5})])
def test_builtins_never_enter_while_not_flat_and_make_no_decision_without_history(cls, params, synthetic_bars,
                                                                                  tmp_path):
    run = drive(cls(**params), synthetic_bars.iloc[:800], tmp_path, warmup=0)   # an engine that asks from bar 0
    assert run.trades and not run.violations
    first = run.trades[0][0]    # fill bar of the first entry = decision bar + 1
    need = {"ma_cross": 30, "rsi_threshold": 15, "bollinger_breakout": 20, "random_entries": 1}[cls.name]
    assert first - 1 >= need - 1, "an entry was decided before the indicator had its history"


def test_ctx_bars_has_the_contract_columns_and_never_shows_a_later_bar(synthetic_bars, tmp_path):
    df = synthetic_bars.iloc[:300].copy()
    df.loc[df.index[100], "volume"] = 1234567.891    # the trap: a unique value in bar 100

    class Ahead:   # a source whose history over-reads five bars: the context must still cut at the current bar
        def __init__(self, store):
            self.inner, self.frame = HistoricalSource(store, SYM, 1), store.read_bars(SYM, 1)

        def next_closed_bar(self):
            return self.inner.next_closed_bar()

        def history(self, n):
            if n == 1:   # the latest-row check of _advance reads exactly this; the over-read is for the wider windows
                return self.frame.iloc[self.inner.position - 1:self.inner.position].reset_index(drop=True)
            k = min(self.inner.position + 5, len(self.frame))
            return self.frame.iloc[max(k - n, 0):k].reset_index(drop=True)

    for i, source in enumerate((None, Ahead)):
        sub = tmp_path / str(i)
        sub.mkdir()
        rec = Recorder()
        drive(rec, df, sub, source=source, warmup=0)
        assert len(rec.frames) > 250
        marker_ts = to_ms(df, 100)
        for ts, f50, f1, fall in rec.frames:
            for f in (f50, f1, fall):
                assert list(f.columns) == list(BAR_COLUMNS) and f["ts"].iloc[-1] == ts   # ends at the current bar
                assert f["ts"].is_monotonic_increasing and f["ts"].is_unique
            assert len(f1) == 1 and len(f50) == min(50, fall.shape[0])
            assert (fall["volume"] == 1234567.891).any() == (ts >= marker_ts), ts   # the trap: never before bar 100


def to_ms(df, i):
    return int(pd.DatetimeIndex(df["datetime"]).asi8[i] // 10 ** 6)


def test_ctx_bars_returns_a_copy_and_the_current_bar_is_frozen(synthetic_bars, tmp_path):
    class Vandal(DeskStrategy):
        name = "vandal"

        def on_bar(self, ctx):
            f = ctx.bars(5)
            f.loc[:, "close"] = -1.0
            assert (ctx.bars(5)["close"] > 0).all()
            with pytest.raises(dataclasses.FrozenInstanceError):
                ctx.bar.close = 0.0

    drive(Vandal(), synthetic_bars.iloc[:20], tmp_path)


def test_a_strategy_changes_the_engine_only_through_the_order_helpers(synthetic_bars, tmp_path):
    seen = {}

    class Meddler(DeskStrategy):
        name = "meddler"

        def on_bar(self, ctx):
            pf = seen["pf"]
            if ctx.bar.ts_open_ms == to_ms(synthetic_bars, 3):
                ctx.submit_market("BUY", size_frac=0.5, tag="x")
                return
            pos = ctx.position
            if pos.side is PositionSide.FLAT:
                return
            before = (pf.cash, pf.positions[pos.position_id].qty, pf.positions[pos.position_id].side)
            pos.qty, pos.side, pos.avg_price = 999.0, PositionSide.SHORT, 1.0
            acct = ctx.account
            acct.cash, acct.equity = 0.0, 0.0
            assert (pf.cash, pf.positions[pos.position_id].qty, pf.positions[pos.position_id].side) == before
            assert ctx.account.cash == pf.cash and ctx.position.qty == before[1]
            assert not hasattr(ctx.clock, "set")
            with pytest.raises(ValueError, match="not an order of strategy"):
                ctx.cancel("someone-else:1")
            public = [getattr(ctx, n) for n in dir(ctx) if not n.startswith("_") and not callable(getattr(ctx, n))]
            for v in public:
                assert not isinstance(v, (ExecutionEngine, Portfolio, SimBroker, SimClock, RiskGuard, EventBus))
            seen["checked"] = seen.get("checked", 0) + 1

    drive(Meddler(), synthetic_bars.iloc[:20], tmp_path, setup=lambda run: seen.update(pf=run.pf))
    assert seen["checked"] > 3


def test_order_helpers_build_the_contract_orders(synthetic_bars, tmp_path):
    made = {}

    class Orders(DeskStrategy):
        name = "orders"

        def on_bar(self, ctx):
            if made:
                return
            made["m"] = ctx.submit_market("buy", qty=0.001, tp=1.0, sl=0.5, tp_is_offset=True, max_hold=7, tag="t1")
            made["l"] = ctx.submit_limit("SELL", 1e9, size_frac=0.1, tif="IOC", tag="t2")
            made["s"] = ctx.submit_stop("SELL", 1.0, qty=0.001, tag="t3")
            made["sl"] = ctx.submit_stop("SELL", 1.0, qty=0.001, limit=0.9, tag="t4")
            for kw in ({}, {"qty": 1.0, "size_frac": 0.5}):
                with pytest.raises(ValueError, match="exactly one of qty and size_frac"):
                    ctx.submit_market("BUY", **kw)
            ctx.cancel(made["s"])

    run = drive(Orders(), synthetic_bars.iloc[:12], tmp_path, warmup=0)
    o = run.eng.orders
    assert made["m"].startswith("orders:") and made["m"].endswith(":1") and o[made["m"]].bracket == {"take_profit": 1.0, "stop_loss": 0.5,
                                                               "tp_is_offset": True, "max_hold": 7}
    assert (o[made["m"]].side, o[made["m"]].tags) == (Side.BUY, {"tag": "t1", "reason": "t1"})
    assert o[made["l"]].type is OrderType.LIMIT and o[made["l"]].price == 1e9 and o[made["l"]].size_frac == 0.1
    assert o[made["sl"]].type is OrderType.STOP_LIMIT and (o[made["sl"]].trigger_price, o[made["sl"]].price) == (1.0, 0.9)
    assert o[made["s"]].type is OrderType.STOP_MARKET and o[made["s"]].status.value == "CANCELED"


# ------------------------------------------------------------------ (3) indicators, predictions, hooks
def test_ctx_indicator_raises_and_names_desk_7_1(synthetic_bars, tmp_path):
    caught = []

    class Ind(DeskStrategy):
        name = "ind"

        def on_bar(self, ctx):
            try:
                ctx.indicator("rsi", length=14)
            except IndicatorsUnavailable as e:
                caught.append(str(e))
            assert ctx.predictions() is None

    drive(Ind(), synthetic_bars.iloc[:5], tmp_path, warmup=0)
    assert caught and "DESK-7.1" in caught[0] and "rsi" in caught[0]


def test_hooks_run_in_order_and_on_bar_is_abstract(synthetic_bars, tmp_path):
    log = []

    class Hooks(DeskStrategy):
        name = "hooks"

        def on_start(self, ctx):
            log.append("start")

        def on_bar(self, ctx):
            log.append("bar")

        def on_stop(self, ctx):
            log.append("stop")

    drive(Hooks(), synthetic_bars.iloc[:4], tmp_path, warmup=0)
    assert log == ["start", "bar", "bar", "bar", "stop"]
    with pytest.raises(NotImplementedError):
        DeskStrategy().on_bar(None)


# ------------------------------------------------------------------ the registry
def test_registry_lists_the_builtins_with_parameters_for_the_ui():
    assert set(BUILTINS) <= set(DeskStrategies.list_names())
    schemas = DeskStrategies.schemas()
    assert set(BUILTINS) <= set(schemas) and all(isinstance(s, list) for s in schemas.values())
    assert [p["name"] for p in schemas["ma_cross"]] == ["fast", "slow", "allow_short", "size", "max_hold"]
    assert DeskStrategies.get("ma_cross") is MACross and DeskStrategies.build("random_entries", seed=3).params["seed"] == 3
    assert DeskStrategies.registry is not DeskStrategies.__mro__[1].registry   # its own registry, not the nine


def test_registry_rejects_unknown_names_duplicates_and_non_strategies():
    with pytest.raises(ComponentNotFoundError, match=r"'no_such' not found\. Available: \[.*ma_cross"):
        DeskStrategies.get("no_such")

    class Tmp(DeskStrategy):
        name = "tmp_nt227"

    DeskStrategies.register(name="tmp_nt227")(Tmp)
    try:
        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            DeskStrategies.register(name="tmp_nt227")(Tmp)
        with pytest.raises(DuplicateRegistrationError):
            DeskStrategies.register(name="ma_cross")(Tmp)
    finally:
        DeskStrategies.remove("tmp_nt227")
    with pytest.raises(ComponentValidationError):
        DeskStrategies.register(name="not_a_strategy")(dict)


# ------------------------------------------------------------------ session ids, stale contexts, exit reasons
def test_client_ids_carry_a_session_so_two_contexts_never_collide(synthetic_bars, tmp_path):
    ids = []

    class Once(DeskStrategy):
        name = "once"

        def on_bar(self, ctx):
            if not ids or ids[-1][0] is not ctx:
                ids.append((ctx, ctx.submit_market("BUY", qty=0.001)))

    r1 = drive(Once(), synthetic_bars.iloc[:5], tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path,
               warmup=0)
    r2 = drive(Once(), synthetic_bars.iloc[:5], tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path,
               warmup=0)
    a, b = (i[1] for i in ids)
    assert a != b and a.split(":")[0] == "once" and a.split(":")[2] == "1" and len(a.split(":")[1]) == 6
    assert r1.ctx.session != r2.ctx.session
    fixed = StrategyContext(execution=r1.eng, source=None, strategy_id="x", clock=SimClock(0), session="abc123")
    assert fixed.session == "abc123"


def test_a_stale_or_out_of_order_bar_fails_loudly(synthetic_bars, tmp_path):
    run = drive(AlwaysFlat(), synthetic_bars.iloc[:10], tmp_path)
    ctx = run.ctx
    store = to_store(synthetic_bars.iloc[:10], tmp_path / "s.sqlite")
    src = HistoricalSource(store, SYM, 1)
    fresh = StrategyContext(execution=run.eng, source=src, strategy_id="y", clock=SimClock(0))
    b0, b1 = src.next_closed_bar(), src.next_closed_bar()
    with pytest.raises(RuntimeError, match="stale context"):
        fresh._advance(b0)            # the source has delivered b1: the latest bar is not b0
    fresh._advance(b1)
    with pytest.raises(RuntimeError, match="stale context"):
        fresh._advance(b1)            # the same bar twice
    with pytest.raises(ValueError, match="closed"):
        ctx._advance(dataclasses.replace(b1, is_closed=False))


def test_close_position_orders_carry_the_exit_tag_as_reason(synthetic_bars, tmp_path):
    run = drive(MACross(), synthetic_bars.iloc[:600], tmp_path)
    closes = [o for o in run.eng.orders.values() if o.reduce_only and o.tags.get("tag") == "CROSS"]
    assert closes and all(o.tags["reason"] == "CROSS" for o in closes)
    assert not [o for o in run.eng.orders.values() if o.tags.get("reason") == "CLOSE"]

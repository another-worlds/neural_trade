"""The random null every backtest carries (NT-002): random entries at the strategy's trade rate,
holding time AND mean position size, one implementation for the engine and the notebooks.

After costs a return is mostly cost x size x trade count, so a null that trades at full size while
the strategy sizes down pays more costs and makes the strategy look skilled. The engine
(``neural-trade backtest``, scripts/backtest_gate.py) and the notebooks share one implementation, so
both report the same rank for the same run."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, Dict, Optional

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.strategy import (BacktestConfig, Bars, Order, RandomSignal, SignalFrame, Strategy, backtest,
                                   build_strategy, random_same_frequency, run_backtest, var_scale_from)

# the engine module (the package re-exports its ``backtest`` function under the same name)
engine = importlib.import_module("neural_trade.strategy.backtest")
SEEDS = 30


def _frame(n, seed, informative):
    rng = np.random.default_rng(seed)
    close = 100_000 + np.cumsum(rng.normal(0, 40, n + 20))
    lc = close[:n]
    y = np.stack([close[h:n + h] - lc for h in (10, 15, 20)], 1)
    noise = rng.normal(0, 60, (n, 3))
    delta = (y * 0.6 + noise) if informative else noise
    prob = 1 / (1 + np.exp(-delta / 40))
    var = rng.uniform(0.2, 1.5, (n, 3))
    f = PredictionFrame(y, lc, {h: delta[:, i] for i, h in enumerate(HORIZONS)},
                        {h: prob[:, i] for i, h in enumerate(HORIZONS)},
                        {h: var[:, i] for i, h in enumerate(HORIZONS)}, 250.0)
    return f, Bars.from_close(lc)


@pytest.fixture(scope="module")
def sized_run():
    """enhanced_multi_horizon (sizes 0.1-1.0 per order) on signals WITHOUT an edge: its rank among
    random entries of the same size must be unremarkable."""
    f, bars = _frame(800, seed=9, informative=False)
    signals = SignalFrame.build(f, var_scale_from(f))
    res = run_backtest(signals, bars, build_strategy("enhanced_multi_horizon"), BacktestConfig(random_seeds=SEEDS))
    sizes = np.array([d["size"] for d in res.decisions])
    assert res.summary["n_trades"] > 50 and sizes.max() < 1.0 and sizes.std() > 0     # a sized strategy
    return f, bars, signals, res


@dataclass
class Scripted(Strategy):
    """Places given orders at given bars (test helper)."""

    name: ClassVar[str] = "scripted"
    orders: Optional[Dict[int, Order]] = None

    def decide(self, s, t):
        return (self.orders or {}).get(t)


def test_random_signal_orders_carry_size_frac(sized_run):
    _, bars, signals, _ = sized_run
    full = run_backtest(signals, bars, RandomSignal(trade_rate=0.2, hold_bars=3, seed=1))
    sized = run_backtest(signals, bars, RandomSignal(trade_rate=0.2, hold_bars=3, seed=1, size_frac=0.25))
    assert RandomSignal().size_frac == 1.0 and {d["size"] for d in full.decisions} == {1.0}
    assert sized.trades and {d["size"] for d in sized.decisions} == {0.25}
    # the same seed draws the same entries: only the size differs
    assert [(d["bar"], d["side"]) for d in sized.decisions] == [(d["bar"], d["side"]) for d in full.decisions]
    for tr in sized.trades:        # equity[entry_bar] is the (flat) equity just before the fill
        assert tr.notional == pytest.approx(0.25 * sized.equity[tr.entry_bar], rel=1e-12)


def test_every_null_trade_has_the_strategy_mean_size(sized_run, monkeypatch):
    _, bars, signals, res = sized_run
    nulls = []
    real = engine.run_backtest

    def spy(*args, **kwargs):
        nulls.append(real(*args, **kwargs))
        return nulls[-1]

    monkeypatch.setattr(engine, "run_backtest", spy)
    out = random_same_frequency(signals, bars, res, seeds=5)
    mean_size = float(np.mean([d["size"] for d in res.decisions]))
    assert out["size_frac"] == pytest.approx(mean_size, rel=1e-12)
    assert len(nulls) == 5 and all(r.trades for r in nulls)
    for r in nulls:
        assert all(d["size"] == out["size_frac"] for d in r.decisions)
        for tr in r.trades:
            assert tr.notional == pytest.approx(mean_size * r.equity[tr.entry_bar], rel=1e-12)


def test_null_size_is_the_mean_fill_size_of_the_opened_positions():
    """The mean is over the orders that opened a position, each clipped to [0, 1] as the engine fills
    it: a size-0 order opens nothing and a size above 1 trades at 1."""
    f, bars = _frame(200, seed=2, informative=False)
    signals = SignalFrame.build(f, var_scale_from(f))
    orders = {10: Order("LONG", 0.2, max_hold=3), 30: Order("SHORT", 0.6, max_hold=3),
              50: Order("LONG", 1.5, max_hold=3), 70: Order("LONG", 0.0, max_hold=3)}
    res = run_backtest(signals, bars, Scripted(orders=orders))
    assert res.summary["n_trades"] == 3 and len(res.decisions) == 4
    out = random_same_frequency(signals, bars, res, seeds=3)
    assert out["size_frac"] == pytest.approx((0.2 + 0.6 + 1.0) / 3, rel=1e-12)


def test_engine_and_notebook_paths_give_the_same_rank(sized_run):
    from neural_trade.notebook import BacktestExplorer
    from neural_trade.notebook.backtest_ui import matched_random_null

    f, bars, signals, res = sized_run
    # the engine path: what `neural-trade backtest` and scripts/backtest_gate.py run
    eng = backtest(signals, bars, build_strategy("enhanced_multi_horizon"),
                   BacktestConfig(random_seeds=SEEDS)).baselines["random_same_freq"]
    nb = matched_random_null(signals, bars, res, seeds=SEEDS)
    assert eng["n_seeds"] == nb["n_seeds"] == SEEDS
    assert eng["percentile_total_return"] == nb["percentile_total_return"]
    assert eng == nb
    # without an edge the strategy is no better than random entries of its own size; the full-size
    # null ranked it above every seed (100th percentile) before NT-002
    assert eng["percentile_total_return"] < 95
    for k in ("size_frac", "random_p05_total_return", "random_p95_total_return", "random_mean_gross_return",
              "percentile_gross_return", "trade_rate", "hold_bars", "random_mean_total_return",
              "random_mean_sharpe_net", "percentile_sharpe_net"):
        assert np.isfinite(eng[k]), k
    assert eng["random_p05_total_return"] <= eng["random_mean_total_return"] <= eng["random_p95_total_return"]
    # the notebook explorer (notebook 02): run() and compare_strategies() report the same null
    ex = BacktestExplorer({"config": Config(), "test": f, "cal": f, "bars": bars,
                           "predictor": SimpleNamespace(bundle=SimpleNamespace(meta={"var_scale": signals.var_scale}))})
    ran = ex.run("enhanced_multi_horizon", costs={"random_seeds": SEEDS}).baselines["random_same_freq"]
    runs, _ = ex.compare_strategies(["enhanced_multi_horizon"], costs={"random_seeds": SEEDS})
    assert ran == eng and runs["enhanced_multi_horizon"].baselines["random_same_freq"] == eng


def test_no_trades_or_no_seeds_give_an_empty_null(sized_run):
    _, bars, signals, res = sized_run
    flat = run_backtest(signals, bars, build_strategy("always_flat"))
    for out in (random_same_frequency(signals, bars, flat), random_same_frequency(signals, bars, res, seeds=0)):
        assert out["n_seeds"] == 0 and np.isnan(out["percentile_total_return"])

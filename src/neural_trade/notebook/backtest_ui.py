"""BacktestExplorer: pick a strategy, edit its knobs and the costs, press Run.

Everything is replayed on the saved run's own TEST block (the purged split is rebuilt from the
run's config, so it is out of sample for that model). The calibration block's predictions give
the strategies' confidence scale (var_scale) and the calibrated_quantile thresholds.

    ex = BacktestExplorer.from_run("runs/<id>", csv_path="binance_btcusdt_1min_ccxt.csv")
    display(ex.widget())
    res = ex.run("liberal", {"min_agreement": 0.4}, {"fee_bps": 5})   # programmatic use
    runs, fig = ex.compare_strategies()     # every model strategy next to its matched random null
"""
from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from neural_trade.notebook._display import show

COST_FIELDS = ("fee_bps", "half_spread_bps", "slippage_bps", "max_hold", "tp_sl_on", "same_bar_tiebreak",
               "random_seeds")
_DERIVED = {"long_above", "short_below", "median"}   # set from the calibration block, not by hand
_BENCHMARKS = ("buy_and_hold", "always_flat")          # the baselines compare_strategies keeps by default
SUMMARY_COLS = ["n_trades", "total_return", "sharpe_net", "max_drawdown", "hit_rate", "hit_rate_gross", "profit_factor",
                "avg_win", "avg_loss", "expectancy", "n_long", "n_short", "gross_long", "gross_short", "net_long",
                "net_short", "exposure", "fees_paid", "costs_paid"]


def trade_stats(result) -> Dict[str, float]:
    """Per-trade figures the summary lacks: average win / loss and expectancy (net $ per trade), and
    the long / short split (count, gross $, net $). NaN where there is nothing to average."""
    net = np.array([t.net_pnl for t in result.trades], dtype=float)
    gross = np.array([t.gross_pnl for t in result.trades], dtype=float)
    side = np.array([t.side for t in result.trades], dtype=object)
    nan = float("nan")
    out = {"avg_win": float(net[net > 0].mean()) if (net > 0).any() else nan,
           "avg_loss": float(net[net <= 0].mean()) if (net <= 0).any() else nan,
           "expectancy": float(net.mean()) if len(net) else nan}
    for s in ("long", "short"):
        m = side == s.upper()
        out.update({f"n_{s}": int(m.sum()), f"gross_{s}": float(gross[m].sum()), f"net_{s}": float(net[m].sum())})
    return out


def _zero_delta_flag_notes(delta) -> Dict[str, str]:
    """Notes for the flags table when a served delta is 0 on every bar (the calibration's beta = 0): the
    strategies still see ``magnitude_coherent`` and ``direction_aligned``, but they are then decided by the 0
    (ties, P(up) alone), not measured on a prediction. {} when every served delta varies."""
    from neural_trade.evaluation.frame import HORIZONS

    d = np.asarray(delta, float)
    if d.ndim != 2 or not len(d):
        return {}
    zero = [h for h, z in zip(HORIZONS, ~np.any(d, axis=0)) if z]
    if not zero:
        return {}
    lead = f"served delta is 0 on {', '.join(zero)} (beta = 0)"
    if len(zero) == len(HORIZONS):
        return {"magnitude_coherent": f"{lead}: true on every bar by ties (|0| <= |0|), not a measured ordering",
                "direction_aligned": f"{lead}: the share of bars where all three P(up) <= 0.5, not a sign "
                                     "agreement"}
    return {"magnitude_coherent": f"{lead}: every comparison with it is fixed by the 0, not measured",
            "direction_aligned": f"{lead}: there it only asks P(up) <= 0.5"}


def _null_columns(null) -> Dict[str, float]:
    """The matched random null as table columns (empty when it was not run)."""
    if not null or not null.get("n_seeds"):
        return {}
    return {"random mean return": null.get("random_mean_total_return"),
            "random p05 (return)": null.get("random_p05_total_return"),
            "random p95 (return)": null.get("random_p95_total_return"),
            "random percentile (return)": null.get("percentile_total_return"),
            "random percentile (gross)": null.get("percentile_gross_return")}


class _Sized:
    """A strategy whose every order is resized to ``size`` (a random null that matches a strategy's
    position size: a null trading at full size pays more costs than a strategy that sizes down)."""

    def __init__(self, base, size: float):
        self.base, self.size = base, float(size)
        self.name = getattr(base, "name", type(base).__name__)
        self.max_hold = getattr(base, "max_hold", 30)

    def warmup(self) -> int:
        return self.base.warmup()

    def decide(self, s, t):
        order = self.base.decide(s, t)
        if order is not None:
            order.size_frac = self.size
        return order

    def exit_signal(self, *args):
        return self.base.exit_signal(*args)


def matched_random_null(signals, bars, result, *, seeds: Optional[int] = None, config=None) -> Dict[str, float]:
    """Random entries with ``result``'s trade rate, holding time AND mean position size, over
    ``seeds`` seeds (default: the config's ``random_seeds``, as the engine's own null): where the
    strategy ranks among them (after and before costs). Seeds 0..seeds-1, so the same call gives
    the same numbers.

    Same keys as ``strategy.random_same_frequency`` plus ``size_frac``, the 5th / 95th percentiles of
    the random total return, and the gross (before-cost) mean and percentile. After costs the
    return is mostly cost x trade count, so a null that trades at full size while the strategy
    sizes down would make the strategy look skilled when it is not.
    """
    from neural_trade.strategy import RandomSignal, run_backtest

    cfg = config or result.config
    k = int(cfg.random_seeds if seeds is None else seeds)
    n_tr = result.summary["n_trades"]
    if n_tr == 0 or k <= 0:
        return {"n_seeds": 0, "percentile_total_return": float("nan"), "percentile_sharpe_net": float("nan")}
    hold = max(1, int(round(result.summary["avg_hold_bars"])) or 1)
    flat_bars = max(1, len(bars) - int(result.summary["exposure"] * len(bars)))
    rate = min(1.0, n_tr / flat_bars)
    sizes = [float(d.get("size", 1.0)) for d in result.decisions]
    size = float(np.clip(np.mean(sizes), 0.0, 1.0)) if sizes else 1.0
    init = float(cfg.initial_equity)
    rets, sharpes, grosses = [], [], []
    for seed in range(k):
        r = run_backtest(signals, bars, _Sized(RandomSignal(trade_rate=rate, hold_bars=hold, seed=seed), size), cfg)
        rets.append(r.summary["total_return"])
        sharpes.append(r.summary["sharpe_net"])
        grosses.append(r.summary["gross_pnl"] / init)
    rets, sharpes, grosses = np.array(rets), np.array(sharpes), np.array(grosses)
    g0 = result.summary.get("gross_pnl", 0.0) / init
    return {
        "n_seeds": k, "trade_rate": rate, "hold_bars": hold, "size_frac": size,
        "random_mean_total_return": float(rets.mean()), "random_mean_sharpe_net": float(sharpes.mean()),
        "random_p05_total_return": float(np.percentile(rets, 5)), "random_p95_total_return": float(np.percentile(rets, 95)),
        "random_mean_gross_return": float(grosses.mean()),
        "percentile_total_return": float(100.0 * np.mean(rets < result.summary["total_return"])),
        "percentile_sharpe_net": float(100.0 * np.mean(sharpes < result.summary["sharpe_net"])),
        "percentile_gross_return": float(100.0 * np.mean(grosses < g0)),
    }


def load_run_blocks(run_dir, csv_path: Optional[str] = None) -> Dict[str, Any]:
    """Predictor, config, and the test/cal PredictionFrames + bars for a saved run (raw and served)."""
    from neural_trade.data.processor import split_arrays
    from neural_trade.serving.predictor import Predictor
    from neural_trade.strategy import Bars

    run_dir = Path(run_dir)
    predictor = Predictor.from_artifacts(run_dir / "artifacts")
    cfg = predictor.config.copy()
    if csv_path:
        cfg.override(CSV_PATH=str(csv_path))
    blocks = split_arrays(cfg)
    out = {"run_dir": run_dir, "predictor": predictor, "config": cfg, "blocks": blocks}
    for name in ("test", "cal"):
        b = blocks[name]
        for served in (True, False):
            batch = predictor.predict(b["X"], b["last_close"], calibrated=served)
            key = name if served else f"{name}_raw"
            out[key] = batch.to_prediction_frame(predictor.bundle.pred_scale, predictor.bundle.pred_mean,
                                                 y=b["y"], split=name)
            out[key].X_raw = b["X"]
        # the served frame carries its raw heads and betas, so evaluate() scores both without extra arguments
        out[name].meta["delta_raw"] = out[f"{name}_raw"].delta
        pipe = predictor.bundle.calibration_pipeline
        if pipe is not None:                     # served delta = beta x raw (beta = 1 without DELTA_SHRINKAGE)
            out[name].meta["delta_scale"] = dict(pipe.delta_scale)
    out["bars"] = Bars.from_frame(blocks["df"], blocks["test"]["anchor_bar"])
    out["times"] = _bar_times(blocks["df"], blocks["test"]["anchor_bar"])
    return out


def _bar_times(df, anchor_bars):
    """One timestamp per test bar (the decision bar), or None when the frame has no time column."""
    import numpy as np

    cols = [c for c in df.columns if str(c).lower() in ("datetime", "timestamp", "time", "date", "open_time")]
    if cols:
        values = df[cols[0]]
    elif isinstance(df.index, pd.DatetimeIndex):
        values = df.index.to_series()
    else:
        return None
    return pd.to_datetime(values).to_numpy()[np.asarray(anchor_bars, dtype=int)]


def _field_widget(field, value):
    import ipywidgets as w

    hint = field.type if not isinstance(field.type, str) else {"float": float, "int": int, "bool": bool,
                                                               "str": str}.get(field.type, str)
    style = {"description_width": "150px"}
    layout = w.Layout(width="300px")
    if hint is bool:
        return w.Checkbox(value=bool(value), description=field.name, style=style, layout=layout)
    if hint is int:
        return w.IntText(value=int(value), description=field.name, style=style, layout=layout)
    if hint is float:
        return w.FloatText(value=float(value), description=field.name, style=style, layout=layout)
    return None


class BacktestExplorer:
    def __init__(self, blocks: Dict[str, Any]):
        from neural_trade.strategy import SignalFrame

        self.blocks = blocks
        self.config = blocks["config"]
        var_scale = float(blocks["predictor"].bundle.meta.get("var_scale") or 1.0)
        self.signals = SignalFrame.build(blocks["test"], var_scale)
        self.cal_signals = SignalFrame.build(blocks["cal"], var_scale)
        self.bars = blocks["bars"]
        self.last = None
        self.last_strategy = None
        self._w = None

    @classmethod
    def from_run(cls, run_dir, csv_path: Optional[str] = None) -> "BacktestExplorer":
        return cls(load_run_blocks(run_dir, csv_path))

    # ------------------------------------------------------------------ programmatic
    def _backtest(self, strategy: str, params: Optional[Dict[str, Any]] = None,
                  costs: Optional[Dict[str, Any]] = None, baselines: bool = True):
        """(BacktestResult, Strategy) without touching ``last`` / ``last_strategy``.

        With ``baselines``: buy-and-hold and always-flat from the engine, and as the random null
        ``matched_random_null`` over ``random_seeds`` seeds (trade rate, hold AND size matched) - the
        same null, with the same seeds, that ``compare_strategies`` draws, so both report one rank.
        """
        from neural_trade.strategy import backtest, build_backtest_config, build_strategy

        strat = build_strategy(strategy, params, calibration=self.cal_signals)
        cfg = build_backtest_config(costs or {})
        # the engine's baselines without its own (full-size) random null; the matched one replaces it
        res = backtest(self.signals, self.bars, strat, dataclasses.replace(cfg, random_seeds=0), baselines=baselines)
        res.config = cfg
        if baselines:
            res.baselines["random_same_freq"] = matched_random_null(self.signals, self.bars, res,
                                                                    seeds=cfg.random_seeds, config=cfg)
        return res, strat

    def run(self, strategy: str, params: Optional[Dict[str, Any]] = None, costs: Optional[Dict[str, Any]] = None,
            baselines: bool = True):
        """Backtest ``strategy`` and make it the explorer's current result (``last``)."""
        self.last, self.last_strategy = self._backtest(strategy, params, costs, baselines)
        return self.last

    def dashboard(self, res=None, *, start=None, end=None):
        """Price / signals / confidence / sigma / equity / drawdown for the last run (or ``res``)."""
        from neural_trade.visualization.trading_dashboard import trading_dashboard_figure

        res = res or self.last
        strat = self.last_strategy if res is self.last else None
        return trading_dashboard_figure(res, self.bars, self.signals, strat, start=start, end=end,
                                        config=self.config, times=self.blocks.get("times"))

    def trade_analytics(self, res=None):
        """Per-trade view of the last run (or ``res``): P&L before / after costs, exit reasons,
        holding time, excursions, the entry signal vs the outcome (the predicted h1 move against
        the realised move over the same h1 bars), long vs short."""
        from neural_trade.visualization.trade_analytics import trade_analytics_figure

        steps = getattr(self.blocks.get("test"), "horizon_steps", None) or getattr(self.config, "HORIZON_STEPS", None)
        return trade_analytics_figure(res or self.last, self.bars, signals=self.signals, horizon_steps=steps)

    def compare_strategies(self, names=None, costs=None, *, null_seeds: Optional[int] = None):
        """Model strategies (or ``names``) with default knobs, plus buy-and-hold and always-flat:
        ({name: BacktestResult}, figure). Leaves ``last`` / ``last_strategy`` alone.

        Each model strategy that trades gets a matched random null (``null_seeds`` seeds, default the
        costs' ``random_seeds`` as in ``run``, with its trade rate, holding time and size) in
        ``result.baselines['random_same_freq']``, drawn on its bar: ``run(name)`` and this report
        the same rank. The standalone ``random_signal`` (5% entries, 10-bar hold, one seed) is not a
        matched null, so it is only run when named, and then labelled with its knobs.
        """
        from neural_trade.strategy import Strategies, build_backtest_config
        from neural_trade.visualization.trade_analytics import strategy_comparison_figure

        baseline = set(Strategies.filter_by_tag("baseline"))
        if names is None:
            names = [n for n in Strategies.list_names() if n not in baseline] + \
                    [n for n in _BENCHMARKS if Strategies.has(n)]
        k = build_backtest_config(dict(costs or {})).random_seeds if null_seeds is None else int(null_seeds)
        runs, labels = {}, {}
        for n in names:
            res, strat = self._backtest(n, costs=costs, baselines=False)
            if n not in baseline and res.summary["n_trades"] > 0 and k > 0:
                res.baselines["random_same_freq"] = matched_random_null(self.signals, self.bars, res, seeds=k,
                                                                        config=res.config)
            if n == "random_signal":
                labels[n] = (f"random_signal ({100 * strat.trade_rate:.0f}%/bar, {strat.hold_bars}-bar hold, "
                             f"seed {strat.seed})")
            runs[n] = res
        return runs, strategy_comparison_figure(runs, self.bars, labels=labels)

    def comparison_table(self, runs) -> pd.DataFrame:
        """One row per strategy of ``compare_strategies``: summary, win / loss and long / short
        figures, exit reasons, and the rank among its matched random null."""
        rows = {}
        for name, r in runs.items():
            row = {k: r.summary.get(k) for k in ("n_trades", "total_return", "sharpe_net", "max_drawdown", "hit_rate",
                                                  "hit_rate_gross", "profit_factor", "exposure", "gross_pnl", "costs_paid")}
            row.update(trade_stats(r))
            row.update(_null_columns(r.baselines.get("random_same_freq")))
            reasons = pd.Series([t.exit_reason for t in r.trades], dtype=object).value_counts()
            row["exits"] = ", ".join(f"{k} {v}" for k, v in reasons.items()) if len(reasons) else ""
            rows[name] = row
        return pd.DataFrame(rows).T

    def signal_summary(self) -> Dict[str, pd.DataFrame]:
        """What the strategies see on the test block: numeric features (count / mean / quantiles),
        the share of bars where each boolean flag is true, consensus and horizon-vote shares, and
        the confidence scale. (``DataFrame.describe`` silently drops boolean columns.) Where a served
        delta is 0 on every bar (beta = 0), the flags table gets a ``note`` column: magnitude_coherent
        and direction_aligned are then fixed by the 0, not measured."""
        s = self.signals
        feats = pd.DataFrame({"weighted_direction": s.weighted_direction, "weighted_move_$": s.weighted_move,
                              "strength": s.strength, "avg_confidence": s.avg_confidence, "agreement": s.agreement,
                              "volatility_$": s.volatility})
        flags = pd.DataFrame({"magnitude_coherent": s.magnitude_coherent, "direction_aligned": s.direction_aligned,
                              "var_spike": s.var_spike}).astype(float)
        shares = flags.mean().to_frame("share true").assign(bars=len(flags))
        notes = _zero_delta_flag_notes(s.delta)
        if notes:
            shares["note"] = [notes.get(k, "") for k in shares.index]
        p = np.asarray(s.p, float)
        votes = (p > 0.55).sum(1) + (p < 0.45).sum(1)
        return {
            "features": feats.describe().T,
            "flags (share of bars true)": shares,
            "consensus": pd.Series(s.consensus).map({1: "up", -1: "down", 0: "neutral"}).value_counts(normalize=True)
                           .reindex(["up", "down", "neutral"]).fillna(0.0).to_frame("share of bars"),
            "horizon votes (P > 0.55 or < 0.45)": pd.Series(np.minimum(votes, 2)).map({0: "0 votes", 1: "1 vote",
                                                                                      2: "2+ votes"})
                                                     .value_counts(normalize=True).to_frame("share of bars"),
            "confidence scale": pd.DataFrame({"var_scale": [s.var_scale]},
                                             index=["confidence = exp(-var / var_scale); var_scale = median "
                                                    "predicted variance on the CAL block"]),
        }

    def summary_frame(self, res=None) -> pd.DataFrame:
        """The run (default: ``last``) next to its baselines. The random null is a distribution over
        seeds, not one run: its row holds the random mean (return, Sharpe, gross return) and the
        5th / 95th percentiles of the random return; the strategy's row holds its rank among the
        seeds after and before costs (the same numbers ``compare_strategies`` shows)."""
        res = res or self.last
        init = float(res.config.initial_equity)
        rows = {res.strategy: {**{k: res.summary.get(k) for k in SUMMARY_COLS}, **trade_stats(res),
                               "gross_return": res.summary.get("gross_pnl", 0.0) / init}}
        for name, s in res.baselines.items():
            if name == "random_same_freq":
                if not s.get("n_seeds"):
                    rows["random same freq (not run: no trades or 0 seeds)"] = {}
                    continue
                label = f"random same freq (mean of {s['n_seeds']} seeds" + \
                        (f", size {s['size_frac']:.2f})" if "size_frac" in s else ")")
                rows[label] = {"total_return": s.get("random_mean_total_return"),
                               "sharpe_net": s.get("random_mean_sharpe_net"),
                               "gross_return": s.get("random_mean_gross_return"),
                               "random p05 (return)": s.get("random_p05_total_return"),
                               "random p95 (return)": s.get("random_p95_total_return")}
                rows[res.strategy].update({k: v for k, v in _null_columns(s).items() if "percentile" in k})
                continue
            rows[name] = {k: s.get(k) for k in SUMMARY_COLS if k in s}
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------ widgets
    def widget(self):
        import ipywidgets as w

        from neural_trade.strategy import BacktestConfig, Strategies

        if self._w is not None:
            return self._w["box"]
        strategy = w.Dropdown(options=Strategies.list_names(), value=Strategies.default, description="strategy")
        knobs = w.VBox()
        defaults = BacktestConfig()
        cost_w = {
            "fee_bps": w.FloatText(value=defaults.fee_bps, description="fee bps / side"),
            "half_spread_bps": w.FloatText(value=defaults.half_spread_bps, description="half-spread bps"),
            "slippage_bps": w.FloatText(value=defaults.slippage_bps, description="slippage bps"),
            "max_hold": w.IntText(value=defaults.max_hold, description="max hold (bars)"),
            "tp_sl_on": w.Dropdown(options=["high_low", "close"], value=defaults.tp_sl_on, description="TP/SL on"),
            "same_bar_tiebreak": w.Dropdown(options=["sl_first", "tp_first"], value=defaults.same_bar_tiebreak,
                                            description="same-bar"),
            # the same seed count compare_strategies uses by default, so both report one random rank
            "random_seeds": w.IntText(value=defaults.random_seeds, description="random seeds"),
        }
        for c in cost_w.values():
            c.style = {"description_width": "120px"}
        run = w.Button(description="Run backtest", icon="play", button_style="primary")
        status = w.HTML()
        table, trades, fig, per_trade = w.Output(), w.Output(), w.Output(), w.Output()

        def rebuild(_=None):
            cls = Strategies.get(strategy.value)
            inst = cls() if not hasattr(cls, "from_calibration") else cls.from_calibration(self.cal_signals)
            items = []
            for f in dataclasses.fields(cls):
                if f.name in _DERIVED or typing.get_origin(f.type) is typing.ClassVar or f.name == "name":
                    continue
                wid = _field_widget(f, getattr(inst, f.name))
                if wid is not None:
                    wid._field = f.name
                    items.append(wid)
            knobs.children = items

        def do_run(_=None):
            status.value = "<i>running...</i>"
            params = {c._field: c.value for c in knobs.children}
            costs = {k: c.value for k, c in cost_w.items()}
            try:
                res = self.run(strategy.value, params, costs)
            except Exception as exc:
                status.value = f"<b style='color:#b91c1c'>{exc}</b>"
                return
            show(table, self.summary_frame(res).round(4))
            show(fig, self.dashboard(res))
            show(per_trade, self.trade_analytics(res))
            tf_ = res.trades_frame()
            show(trades, tf_.tail(30) if len(tf_) else pd.DataFrame({"trades": []}))
            s = res.summary
            null = res.baselines.get("random_same_freq") or {}
            rank = (f", beats {null['percentile_total_return']:.0f}% of {null['n_seeds']} matched random runs"
                    if null.get("n_seeds") else "")
            status.value = (f"<b>{s['n_trades']}</b> trades, net <b>{100 * s['total_return']:+.2f}%</b>, "
                            f"gross {100 * s['gross_pnl'] / res.config.initial_equity:+.2f}%, "
                            f"costs {s['costs_paid']:.0f} on {res.config.initial_equity:,.0f} equity{rank}")

        strategy.observe(rebuild, names="value")
        run.on_click(do_run)
        rebuild()
        box = w.VBox([
            w.HBox([strategy, run, status]),
            w.HBox([w.VBox([w.HTML("<b>Strategy knobs</b>"), knobs]),
                    w.VBox([w.HTML("<b>Costs and execution</b>"), *cost_w.values()])]),
            table, fig, per_trade, w.Accordion(children=[trades], titles=("Last 30 trades",)),
        ])
        self._w = {"box": box, "strategy": strategy, "knobs": knobs, "costs": cost_w, "run": run, "status": status,
                   "do_run": do_run}
        return box

    def click_run(self):
        """Press Run programmatically (tests, or a notebook cell that should render the first result)."""
        self.widget()
        self._w["do_run"]()

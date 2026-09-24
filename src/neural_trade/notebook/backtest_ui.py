"""BacktestExplorer: pick a strategy, edit its knobs and the costs, press Run.

Everything is replayed on the saved run's own TEST block (the purged split is rebuilt from the
run's config, so it is out of sample for that model). The calibration block's predictions give
the strategies' confidence scale (var_scale) and the calibrated_quantile thresholds.

    ex = BacktestExplorer.from_run("runs/<id>", csv_path="binance_btcusdt_1min_ccxt.csv")
    display(ex.widget())
    res = ex.run("liberal", {"min_agreement": 0.4}, {"fee_bps": 5})   # programmatic use
"""
from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from neural_trade.notebook._display import show

COST_FIELDS = ("fee_bps", "half_spread_bps", "slippage_bps", "max_hold", "tp_sl_on", "same_bar_tiebreak",
               "random_seeds")
_DERIVED = {"long_above", "short_below", "median"}   # set from the calibration block, not by hand


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
    out["bars"] = Bars.from_frame(blocks["df"], blocks["test"]["anchor_bar"])
    return out


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
    def run(self, strategy: str, params: Optional[Dict[str, Any]] = None, costs: Optional[Dict[str, Any]] = None,
            baselines: bool = True):
        from neural_trade.strategy import backtest, build_backtest_config, build_strategy

        strat = build_strategy(strategy, params, calibration=self.cal_signals)
        self.last = backtest(self.signals, self.bars, strat, build_backtest_config(costs or {}), baselines=baselines)
        self.last_strategy = strat
        return self.last

    def dashboard(self, res=None, *, start=None, end=None):
        """Price / signals / confidence / sigma / equity / drawdown for the last run (or ``res``)."""
        from neural_trade.visualization.trading_dashboard import trading_dashboard_figure

        res = res or self.last
        strat = self.last_strategy if res is self.last else None
        return trading_dashboard_figure(res, self.bars, self.signals, strat, start=start, end=end)

    def trade_analytics(self, res=None):
        """Per-trade view of the last run (or ``res``): P&L, cost drag, exit reasons, excursions."""
        from neural_trade.visualization.trading_dashboard import trade_analytics_figure

        return trade_analytics_figure(res or self.last, self.bars)

    def compare_strategies(self, names=None, costs=None):
        """Every registered strategy (or ``names``) with default knobs: {name: BacktestResult} and a figure."""
        from neural_trade.strategy import Strategies
        from neural_trade.visualization.trading_dashboard import strategy_comparison_figure

        runs = {n: self.run(n, costs=dict(costs or {}, random_seeds=0), baselines=False)
                for n in (names or Strategies.list_names())}
        return runs, strategy_comparison_figure(runs, self.bars)

    def summary_frame(self, res=None) -> pd.DataFrame:
        res = res or self.last
        cols = ["n_trades", "total_return", "sharpe_net", "max_drawdown", "hit_rate", "hit_rate_gross", "profit_factor", "exposure",
                "fees_paid", "costs_paid"]
        rows = {res.strategy: {k: res.summary.get(k) for k in cols}}
        for name, s in res.baselines.items():
            if name == "random_same_freq":   # a distribution over seeds, not one run: show its mean and our rank
                rows[f"random same freq (mean of {s.get('n_seeds', 0)})"] = {
                    "total_return": s.get("random_mean_total_return"), "sharpe_net": s.get("random_mean_sharpe_net"),
                    "random percentile (return)": s.get("percentile_total_return")}
                rows[res.strategy]["random percentile (return)"] = s.get("percentile_total_return")
                continue
            rows[name] = {k: s.get(k) for k in cols if k in s}
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
            "random_seeds": w.IntText(value=20, description="random seeds"),
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
            status.value = (f"<b>{s['n_trades']}</b> trades, net <b>{100 * s['total_return']:+.2f}%</b>, "
                            f"gross {100 * s['gross_pnl'] / res.config.initial_equity:+.2f}%, "
                            f"costs {s['costs_paid']:.0f} on {res.config.initial_equity:,.0f} equity")

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

"""Visualizations registry (registry 6 of 9).

Components are ``f(data, config, **kwargs) -> figure | str | None``. Selected by
``Config.VISUALIZATION`` where a single default is needed (the training dashboard).
Trading and evaluation plots are added by neural_trade.strategy / neural_trade.evaluation.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, Tuple

from neural_trade.core.registry import BaseRegistry


class Visualizations(BaseRegistry):
    registry = {}
    strict = True
    default = "plotly_interactive"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.visualization.plotly_training",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        try:
            params = list(inspect.signature(component).parameters)
        except (TypeError, ValueError):
            return False
        return callable(component) and params[:2] == ["data", "config"]


from neural_trade.visualization import indicator_evolution as _ie  # noqa: E402
from neural_trade.visualization import matplotlib_splits as _ms  # noqa: E402
from neural_trade.visualization import plotly_training as _pt  # noqa: E402
from neural_trade.visualization import qbox_dashboard as _qb  # noqa: E402

Visualizations.register(name="plotly_interactive", tags=["plotly", "training", "default"],
                        dependencies=["plotly"])(_pt.plotly_interactive)
Visualizations.register(name="qbox_dashboard_html", tags=["html", "physics"])(_qb.qbox_dashboard_html)
Visualizations.register(name="matplotlib_splits", tags=["matplotlib", "data"],
                        dependencies=["matplotlib"])(_ms.matplotlib_splits)
Visualizations.register(name="indicator_evolution", tags=["plotly", "indicators"],
                        dependencies=["plotly"])(_ie.indicator_evolution)
from neural_trade.evaluation import plots as _ev  # noqa: E402

Visualizations.register(name="eval_report", tags=["plotly", "evaluation", "calibration"],
                        dependencies=["plotly"])(_ev.eval_report_figure)
from neural_trade.visualization import plotly_trading as _tr  # noqa: E402

Visualizations.register(name="plotly_trading", tags=["plotly", "backtest", "trading"],
                        dependencies=["plotly"])(_tr.plotly_trading)
from neural_trade.visualization import calibration_plots as _cp  # noqa: E402
from neural_trade.visualization import comparison as _cmp  # noqa: E402
from neural_trade.visualization import data_overview as _do  # noqa: E402

Visualizations.register(name="reliability", tags=["plotly", "calibration", "direction"],
                        dependencies=["plotly"])(_cp.reliability)
Visualizations.register(name="interval_coverage", tags=["plotly", "calibration", "intervals"],
                        dependencies=["plotly"])(_cp.interval_coverage)
Visualizations.register(name="runs_comparison", tags=["plotly", "experiments"], dependencies=["plotly"])(
    _cmp.runs_comparison)
Visualizations.register(name="ablation_deltas", tags=["plotly", "experiments", "ablation"],
                        dependencies=["plotly"])(_cmp.ablation_deltas)
Visualizations.register(name="split_overview", tags=["plotly", "data"], dependencies=["plotly"])(_do.split_overview)
from neural_trade.visualization import model_analytics as _ma  # noqa: E402
from neural_trade.visualization import trading_dashboard as _td  # noqa: E402
from neural_trade.visualization import training_dashboard as _trd  # noqa: E402

Visualizations.register(name="training_dashboard", tags=["plotly", "training"],
                        dependencies=["plotly"])(_trd.training_dashboard)
for _name, _fn, _tags in (("direction_analytics", _ma.direction_analytics, ["direction"]),
                          ("delta_analytics", _ma.delta_analytics, ["delta"]),
                          ("variance_analytics", _ma.variance_analytics, ["variance", "calibration"]),
                          ("confidence_analytics", _ma.confidence_analytics, ["direction", "confidence"]),
                          ("coherence_analytics", _ma.coherence_analytics, ["horizons"])):
    Visualizations.register(name=_name, tags=["plotly", "evaluation", *_tags], dependencies=["plotly"])(_fn)
for _name, _fn in (("trading_dashboard", _td.trading_dashboard), ("trade_analytics", _td.trade_analytics),
                   ("strategy_comparison", _td.strategy_comparison)):
    Visualizations.register(name=_name, tags=["plotly", "backtest", "trading"], dependencies=["plotly"])(_fn)
Visualizations._initialized = True

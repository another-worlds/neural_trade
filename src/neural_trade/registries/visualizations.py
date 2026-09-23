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
Visualizations._initialized = True

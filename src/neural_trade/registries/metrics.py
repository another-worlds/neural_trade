"""Metrics registry (registry 3 of 9), two tiers in one registry.

* **numpy tier** (evaluation): ``f(y_true, y_pred, *, mask=None, ...) -> float``, registered
  with ``Metrics.register``. ``Config.METRICS`` selects which ones the evaluation reports.
* **TF tier** (inside train_step/test_step): ``f(stats) -> tf.Tensor`` over epoch-accumulated
  direction statistics, registered with ``Metrics.register_tf``. ``Config.STEP_METRICS``
  selects them; ``CustomTrainModel`` resolves the list ONCE at construction, never inside a
  traced function. TF-tier functions must be graph-safe (no ``.numpy()``).

Sharpe, drawdown and other equity-curve statistics are not metrics of (y_true, y_pred): they
live in ``neural_trade.strategy.performance``.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, Tuple

from neural_trade.core.exceptions import ComponentNotFoundError, ComponentValidationError
from neural_trade.core.registry import BaseRegistry

TF_TAG = "tf"


def _param_names(fn):
    try:
        return list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return []


class Metrics(BaseRegistry):
    registry = {}
    strict = True
    default = "rmse"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.metrics.numpy_metrics",
                                                    "neural_trade.metrics.tf_direction")

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        """numpy tier: (y_true, y_pred, ...); TF tier: (stats)."""
        params = _param_names(component)
        return callable(component) and (params[:2] == ["y_true", "y_pred"] or params[:1] == ["stats"])

    @classmethod
    def register_tf(cls, name: Optional[str] = None, tags=None, **kwargs):
        """Register a graph-safe step metric ``f(stats) -> tensor``."""
        register = cls.register(name=name, tags=[TF_TAG, *(tags or [])], **kwargs)

        def decorator(fn: Callable) -> Callable:
            if _param_names(fn)[:1] != ["stats"]:
                raise ComponentValidationError(f"Metrics: TF-tier metric '{name}' must take `stats` first")
            return register(fn)

        return decorator

    @classmethod
    def numpy_names(cls):
        return sorted(n for n, e in cls.registry.items() if TF_TAG not in e.tags)

    @classmethod
    def tf_names(cls):
        return sorted(cls.filter_by_tag(TF_TAG))

    @classmethod
    def tf_functions(cls, names: Optional[Iterable[str]] = None) -> Dict[str, Callable]:
        """Ordered {name: fn} for the TF tier (all of them when ``names`` is None)."""
        names = list(names) if names else [n for n in cls.registry if TF_TAG in cls.registry[n].tags]
        out = {}
        for n in names:
            if TF_TAG not in cls.entry(n).tags:
                raise ComponentNotFoundError(f"Metrics: '{n}' is not a TF-tier step metric; "
                                             f"step metrics: {cls.tf_names()}")
            out[n] = cls.get(n)
        return out

    @classmethod
    def numpy_functions(cls, names: Optional[Iterable[str]] = None) -> Dict[str, Callable]:
        names = list(names) if names else cls.numpy_names()
        out = {}
        for n in names:
            if TF_TAG in cls.entry(n).tags:
                raise ComponentNotFoundError(f"Metrics: '{n}' is a TF-tier step metric, not an evaluation metric")
            out[n] = cls.get(n)
        return out


from neural_trade.metrics import numpy_metrics as _np  # noqa: E402
from neural_trade.metrics import tf_direction as _tfd  # noqa: E402

for _name, _tags in (
    ("mse", ["regression"]), ("rmse", ["regression", "default"]), ("mae", ["regression"]),
    ("explained_variance", ["regression"]), ("corr", ["regression"]), ("r2", ["regression"]),
    ("safe_mape", ["regression", "percentage"]), ("smape", ["regression", "percentage"]),
    ("wape", ["regression", "percentage"]),
    ("direction_accuracy", ["direction"]), ("direction_f1", ["direction"]), ("mcc", ["direction"]),
    ("brier", ["direction", "calibration"]), ("ece_pos", ["direction", "calibration"]),
    ("pit_ks", ["distribution", "calibration"]), ("coverage", ["distribution", "interval"]),
):
    Metrics.register(name=_name, tags=_tags)(getattr(_np, _name))

for _name, _fn in _tfd.STEP_METRIC_FUNCTIONS.items():
    Metrics.register_tf(name=_name, tags=["direction", "step"])(_fn)

REGRESSION_METRICS = _np.REGRESSION_METRICS
DIRECTION_METRICS = _np.DIRECTION_METRICS
DISTRIBUTION_METRICS = _np.DISTRIBUTION_METRICS
STEP_METRICS = tuple(_tfd.STEP_METRIC_FUNCTIONS)
Metrics._initialized = True

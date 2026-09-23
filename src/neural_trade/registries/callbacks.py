"""Callbacks registry (registry 4 of 9).

Components are ``builder(config, context: TrainContext) -> Callback | list[Callback]``.
``Config.CALLBACKS`` lists them in order; the default is exactly the list the trainer used to
hard-code (csv_logger, early_stopping, model_checkpoint, tqdm_progress, params_logger,
reduce_lr_on_plateau). ``build_callbacks`` flattens and validates.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, Iterable, List, Optional, Tuple

import tensorflow as tf

from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.core.registry import BaseRegistry


class Callbacks(BaseRegistry):
    registry = {}
    strict = True
    default = "early_stopping"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.training.callbacks",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        try:
            params = list(inspect.signature(component).parameters)
        except (TypeError, ValueError):
            return False
        return callable(component) and params[:2] == ["config", "context"]


def build_callbacks(config, context, names: Optional[Iterable[str]] = None) -> List[tf.keras.callbacks.Callback]:
    """Instantiate ``names`` (default Config.CALLBACKS) in order."""
    out: List[tf.keras.callbacks.Callback] = []
    for name in list(names if names is not None else getattr(config, "CALLBACKS", []) or []):
        made = Callbacks.build(name, config, context)
        for cb in (made if isinstance(made, (list, tuple)) else [made]):
            if not isinstance(cb, tf.keras.callbacks.Callback):
                raise ComponentValidationError(f"Callbacks: '{name}' produced {type(cb).__name__}")
            out.append(cb)
    return out


from neural_trade.training import callbacks as _cb  # noqa: E402

for _name, _tags, _deps in (
    ("csv_logger", ["logging", "default"], []),
    ("early_stopping", ["regularization", "default"], []),
    ("model_checkpoint", ["persistence", "default"], []),
    ("tqdm_progress", ["progress", "console", "default"], []),
    ("params_logger", ["logging", "indicators", "default"], []),
    ("reduce_lr_on_plateau", ["schedule", "default"], []),
    ("mcc_early_stopping", ["regularization", "direction"], []),
    ("jsonl_epoch_logger", ["logging", "telemetry"], []),
    ("lambda_schedule", ["schedule", "loss_weights"], []),
    ("metric_threshold", ["safety"], []),
    ("tensorboard", ["logging", "visualization"], ["tensorboard"]),
    ("interactive_plot", ["visualization", "notebook"], ["ipywidgets", "IPython", "plotly"]),
):
    Callbacks.register(name=_name, tags=_tags, dependencies=_deps)(getattr(_cb, f"build_{_name}"))
Callbacks._initialized = True

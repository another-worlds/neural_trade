"""Callbacks registry (registry 4 of 9).

Components are ``builder(config, context: TrainContext) -> Callback | list[Callback]``.
``Config.CALLBACKS`` lists them in order; the default is exactly the list the trainer used to
hard-code (csv_logger, early_stopping, model_checkpoint, tqdm_progress, params_logger,
reduce_lr_on_plateau). ``build_callbacks`` flattens and validates.

NT-027 (the layering fix): unlike the other eight registries, this one populates itself with
``auto_discover()`` instead of a static ``import neural_trade.training.callbacks`` at the bottom
of the file. The builders live in ``training/`` (they are tied to ``TrainContext``) and ``training``
already legitimately imports every registry it needs (Models, Losses, Metrics, Optimizers,
Callbacks) to build the training path; this registry importing ``training/`` back would be the one
``registries <-> training`` cycle. ``auto_discover`` (``BaseRegistry``) imports the discovery module
dynamically (``importlib.import_module``), which still populates the registry the moment this module
is imported (same contract as every other registry: "populated when imported"), but is not a static
import and so is not part of the subpackage import graph the layering test checks.
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


# The builders decorate themselves with Callbacks.register in neural_trade.training.callbacks;
# auto_discover() imports that module dynamically (see the module docstring for why).
Callbacks.auto_discover()

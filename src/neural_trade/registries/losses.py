"""Losses registry (registry 9 of 9): component losses and training objectives.

Two tiers in one registry:

* **component** losses take the model as their first parameter
  (``focal_loss(model, labels, logits, ...)``) and are registered with ``Losses.register``;
* **objectives** compute the whole training loss and return
  :class:`~neural_trade.core.outputs.LossComponents`. They are registered with
  ``Losses.register_objective`` and must have exactly the signature
  ``(model, x_window, y_true, y_pred, last_close, extended_trends, vacuum_overflow=None)``.
  ``CustomTrainModel`` resolves ``Config.LOSS_NAME`` to an objective once, at construction.

The implementations live in :mod:`neural_trade.losses.functions`, imported at the bottom
of this module so that importing the registry always yields a populated registry.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

from neural_trade.core.exceptions import ComponentNotFoundError, ComponentValidationError
from neural_trade.core.registry import BaseRegistry

OBJECTIVE_TAG = "objective"
OBJECTIVE_PARAMS = ("model", "x_window", "y_true", "y_pred", "last_close", "extended_trends")


def _params(fn) -> List[inspect.Parameter]:
    try:
        return list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return []


class Losses(BaseRegistry):
    """Registry of loss functions (component tier) and training objectives."""

    registry = {}
    strict = True
    default = "custom_loss"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.losses.functions",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        """Callable whose first parameter is ``model``."""
        params = _params(component)
        return callable(component) and bool(params) and params[0].name == "model"

    # ------------------------------------------------------------------ objective tier
    @classmethod
    def validate_objective(cls, component: Any) -> bool:
        params = _params(component)
        required = tuple(p.name for p in params
                         if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)
        overflow = [p for p in params if p.name == "vacuum_overflow"]
        return (callable(component) and required == OBJECTIVE_PARAMS
                and bool(overflow) and overflow[0].default is None)

    @classmethod
    def register_objective(cls, name: Optional[str] = None, tags: Optional[List[str]] = None, **kwargs):
        """Register a training objective (see module docstring for the required signature)."""
        register = cls.register(name=name, tags=[OBJECTIVE_TAG, *(tags or [])], **kwargs)

        def decorator(fn: Callable) -> Callable:
            if not cls.validate_objective(fn):
                raise ComponentValidationError(
                    f"Losses: objective '{name or fn.__name__}' must have the signature "
                    f"{OBJECTIVE_PARAMS} + keyword 'vacuum_overflow'")
            return register(fn)

        return decorator

    @classmethod
    def get_objective(cls, name: Optional[str] = None) -> Callable:
        """The objective registered as ``name`` (default: ``custom_loss``)."""
        resolved = cls.resolve(name)
        component = cls.get(resolved)
        if OBJECTIVE_TAG not in cls.entry(resolved).tags:
            raise ComponentNotFoundError(f"Losses: '{resolved}' is a component loss, not an objective. "
                                         f"Objectives: {cls.list_objectives()}")
        return component

    @classmethod
    def list_objectives(cls) -> List[str]:
        return sorted(cls.filter_by_tag(OBJECTIVE_TAG))

    @classmethod
    def list_components(cls) -> List[str]:
        return sorted(n for n in cls.registry if n not in cls.list_objectives())

    # ------------------------------------------------------------------ logging helpers (unchanged API)
    @classmethod
    def normalize_return(cls, out) -> Tuple[Any, Dict[str, Any]]:
        """Normalise common loss return shapes into ``(total, components)``."""
        import tensorflow as tf

        if isinstance(out, (tf.Tensor, float, int)):
            return out, {"loss": out}
        if isinstance(out, dict):
            comps = dict(out)
            if "loss" not in comps:
                loss_keys = [k for k in comps if k.endswith("_loss") or k == "loss"]
                if loss_keys:
                    total = None
                    for k in loss_keys:
                        v = comps[k]
                        v_mean = tf.reduce_mean(v) if isinstance(v, tf.Tensor) else v
                        total = v_mean if total is None else total + v_mean
                    comps["loss"] = total
                else:
                    for k, v in comps.items():
                        if isinstance(v, (float, int, tf.Tensor)):
                            comps["loss"] = tf.reduce_mean(v) if isinstance(v, tf.Tensor) else v
                            break
                    comps.setdefault("loss", 0.0)
            return comps["loss"], comps
        if isinstance(out, tuple):
            if len(out) == 2 and isinstance(out[1], dict):
                return out[0], out[1]
            return out[0], {}
        return out, {"loss": out}

    @classmethod
    def as_logging_dict(cls, out, total_key: str = "loss") -> Dict[str, Any]:
        total, comps = cls.normalize_return(out)
        result = dict(comps)
        result[total_key] = total
        return result


# Populate the registry (the functions module decorates itself with Losses.register*).
from neural_trade.losses import functions as _functions  # noqa: E402,F401

Losses._initialized = True

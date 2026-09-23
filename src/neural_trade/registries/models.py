"""Models registry (registry 1 of 9): architecture builders.

A component is ``builder(config) -> tf.keras.Model``. ``Models.build(name, config)`` checks
the result has the 10 outputs of :class:`~neural_trade.core.outputs.PredictiveOutputs`
(3 horizons x price/direction/variance + vacuum overflow), each of shape ``[None, 1]``, so a
new architecture cannot silently break the training objective or the serving path.

Deferred (named in REGISTRY_SPECIFICATIONS.md, no implementation exists yet):
``lstm_transformer``, ``conv1d_attention``.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, Tuple

import tensorflow as tf

from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.core.outputs import PredictiveOutputs
from neural_trade.core.registry import BaseRegistry


def ensure_predictive_outputs(model: tf.keras.Model, name: str = "model") -> tf.keras.Model:
    """Raise unless ``model`` returns the 10 PredictiveOutputs heads, each ``[None, 1]``."""
    outputs = list(model.outputs)
    expected = len(PredictiveOutputs._fields)
    if len(outputs) != expected:
        raise ComponentValidationError(
            f"{name}: expected {expected} outputs {PredictiveOutputs._fields}, got {len(outputs)}")
    bad = [(f, tuple(o.shape)) for f, o in zip(PredictiveOutputs._fields, outputs)
           if tuple(o.shape)[1:] != (1,)]
    if bad:
        raise ComponentValidationError(f"{name}: every output must be [None, 1]; got {bad}")
    return model


class Models(BaseRegistry):
    registry = {}
    strict = True
    default = "gru_attention"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.models.gru_attention",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        try:
            params = list(inspect.signature(component).parameters)
        except (TypeError, ValueError):
            return False
        return callable(component) and bool(params) and params[0] == "config"

    @classmethod
    def build(cls, name, config, /, *args, **kwargs) -> tf.keras.Model:
        resolved = cls.resolve(name)
        model = super().build(resolved, config, *args, **kwargs)
        if not isinstance(model, tf.keras.Model):
            raise ComponentValidationError(f"Models: '{resolved}' did not return a tf.keras.Model")
        return ensure_predictive_outputs(model, resolved)


from neural_trade.models.gru_attention import build_gru_attention  # noqa: E402

Models.register(name="gru_attention", version="3.0.0",
                tags=["rnn", "attention", "transformer", "multi_horizon", "default"],
                description="Learnable indicators + Bi-GRU + attention + energy-gated convs, 3 horizon towers")(
    build_gru_attention)
Models._initialized = True

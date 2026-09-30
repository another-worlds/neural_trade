"""The Models registry class (registry 1 of 9): architecture builders.

Moved here from ``neural_trade.registries.models`` (NT-027): ``training/trainer.py``,
``training/custom_model.py`` and ``training/artifacts.py`` dispatch through it
(``Models.build(...)``), so the registry now lives in the same package as its client and
``models/`` no longer needs to import ``neural_trade.registries``. ``neural_trade.registries
.models`` re-exports :class:`Models` and :func:`ensure_predictive_outputs` for their old callers.
(The ``PricePredictor`` facade that used to sit in front of ``Models.build`` was removed in
NT-028: it had no caller besides the two tests that checked it exist.)

A component is ``builder(config) -> tf.keras.Model``. ``Models.build(name, config)`` checks
the result has the 10 outputs of :class:`~neural_trade.core.outputs.PredictiveOutputs`
(3 horizons x price/direction/variance + vacuum overflow), each of shape ``[None, 1]``, so a
new architecture cannot silently break the training objective or the serving path.

Deferred (named in docs/archive/REGISTRY_SPECIFICATIONS.md, the pre-remediation registry design,
no implementation exists yet): ``lstm_transformer``, ``conv1d_attention``.
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
    discovery_modules: ClassVar[Tuple[str, ...]] = (
        "neural_trade.models.gru_attention",
        "neural_trade.models.gru_small",
        "neural_trade.models.linear_indicators",
    )

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
from neural_trade.models.gru_small import build_gru_small  # noqa: E402
from neural_trade.models.linear_indicators import build_linear_indicators  # noqa: E402

Models.register(name="gru_attention", version="3.0.0",
                tags=["rnn", "attention", "transformer", "multi_horizon", "default"],
                description="Learnable indicators + Bi-GRU + attention + energy-gated convs, 3 horizon towers")(
    build_gru_attention)
Models.register(name="gru_small", version="1.0.0",
                tags=["rnn", "multi_horizon", "capacity_study"],
                description="Learnable indicators + one GRU(32), same towers/heads as gru_attention (NT-104)")(
    build_gru_small)
Models.register(name="linear_indicators", version="1.0.0",
                tags=["linear", "multi_horizon", "capacity_study"],
                description="Learnable indicators, pooled, linear heads (NT-104)")(
    build_linear_indicators)
Models._initialized = True

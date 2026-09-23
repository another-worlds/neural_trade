"""Layers registry (registry 7 of 9): custom Keras layers by name.

Components are Layer classes; ``Layers.build(name, **kwargs)`` instantiates one and checks
the result is a ``tf.keras.layers.Layer``. ``Config.LAYERS`` maps architecture roles
(indicators, positional_encoding, vacuum_noise, energy_gate) to registered names, so an
architecture asks for "the indicators layer" and the configuration decides which.

Deferred (named in REGISTRY_SPECIFICATIONS.md, no implementation exists yet):
``wavenet_causal``, ``squeeze_excitation``.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, Dict, Tuple

import tensorflow as tf

from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.core.registry import BaseRegistry


class Layers(BaseRegistry):
    registry = {}
    strict = True
    default = "learnable_indicators"
    discovery_modules: ClassVar[Tuple[str, ...]] = ("neural_trade.models.layers",)

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        return inspect.isclass(component) and issubclass(component, tf.keras.layers.Layer)

    @classmethod
    def build(cls, name, *args, **kwargs) -> tf.keras.layers.Layer:
        layer = super().build(name, *args, **kwargs)
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ComponentValidationError(f"Layers: '{name}' did not produce a Keras Layer")
        return layer

    @classmethod
    def for_role(cls, config, role: str, *args, **kwargs) -> tf.keras.layers.Layer:
        """Instantiate the layer ``config.LAYERS[role]`` names."""
        return cls.build(dict(getattr(config, "LAYERS", {}) or {}).get(role, role), *args, **kwargs)

    @classmethod
    def as_custom_objects(cls) -> Dict[str, type]:
        """``custom_objects`` for tf.keras model loading: class name -> class."""
        return {e.component.__name__: e.component for e in cls.registry.values()}


from neural_trade.models.layers import (  # noqa: E402
    EnergyGate,
    LearnableIndicators,
    PositionalEncodingLayer,
    VacuumSaturationNoise,
)

Layers.register(name="learnable_indicators", tags=["indicators", "meta_learning", "default"],
                description="18 learnable EWMA periods -> 31 MA/MACD/RSI/Bollinger channels")(LearnableIndicators)
Layers.register(name="positional_encoding", tags=["attention", "transformer"],
                description="Sinusoidal positional encoding")(PositionalEncodingLayer)
Layers.register(name="vacuum_saturation_noise", tags=["noise", "t_perp", "regularisation"],
                description="Fills the T-perp subspace to E_max with calibrated noise (training only)")(VacuumSaturationNoise)
Layers.register(name="energy_gate", tags=["gating", "multi_scale", "volatility"],
                description="Volatility-adaptive blend of short/medium/long conv branches")(EnergyGate)
Layers._initialized = True

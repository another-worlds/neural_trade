"""Optimizers registry (registry 2 of 9).

Components are ``builder(config, learning_rate=None) -> tf.keras optimizer``; ``learning_rate``
defaults to ``Config.LR`` (the indicator optimizer passes ``LR * INDICATOR_LR_MULT``).
Selected by ``Config.OPTIMIZER_NAME`` and ``Config.INDICATOR_OPTIMIZER_NAME``. No clipnorm here:
clipping is done once, per parameter group, in the train step.
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

import tensorflow as tf

from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.core.registry import BaseRegistry


class Optimizers(BaseRegistry):
    registry = {}
    strict = True
    default = "adam"

    @classmethod
    def validate_component(cls, component: Any) -> bool:
        try:
            params = inspect.signature(component).parameters
        except (TypeError, ValueError):
            return False
        names = list(params)
        return callable(component) and names[:1] == ["config"] and "learning_rate" in params

    @classmethod
    def build(cls, name, config, /, *args, **kwargs):
        opt = super().build(name, config, *args, **kwargs)
        base = (tf.keras.optimizers.Optimizer, tf.keras.optimizers.experimental.Optimizer)
        if not isinstance(opt, base):
            raise ComponentValidationError(f"Optimizers: '{name}' did not return a Keras optimizer")
        return opt


def _lr(config, learning_rate):
    return float(config.LR if learning_rate is None else learning_rate)


@Optimizers.register(name="adam", tags=["adaptive", "default"])
def build_adam(config, learning_rate: Optional[float] = None):
    """Adam (Keras defaults unless ADAM_* are changed)."""
    return tf.keras.optimizers.Adam(learning_rate=_lr(config, learning_rate), beta_1=config.ADAM_BETA1,
                                    beta_2=config.ADAM_BETA2, epsilon=config.ADAM_EPSILON)


@Optimizers.register(name="adamw", tags=["adaptive", "weight_decay"])
def build_adamw(config, learning_rate: Optional[float] = None):
    """Adam with decoupled weight decay (WEIGHT_DECAY)."""
    return tf.keras.optimizers.experimental.AdamW(learning_rate=_lr(config, learning_rate),
                                                  weight_decay=config.WEIGHT_DECAY, beta_1=config.ADAM_BETA1,
                                                  beta_2=config.ADAM_BETA2, epsilon=config.ADAM_EPSILON)


@Optimizers.register(name="sgd_momentum", tags=["sgd", "momentum"])
def build_sgd_momentum(config, learning_rate: Optional[float] = None):
    """SGD with momentum (SGD_MOMENTUM) and optional Nesterov (SGD_NESTEROV)."""
    return tf.keras.optimizers.SGD(learning_rate=_lr(config, learning_rate), momentum=config.SGD_MOMENTUM,
                                   nesterov=bool(config.SGD_NESTEROV))


@Optimizers.register(name="rmsprop", tags=["adaptive"])
def build_rmsprop(config, learning_rate: Optional[float] = None):
    """RMSprop."""
    return tf.keras.optimizers.RMSprop(learning_rate=_lr(config, learning_rate))


@Optimizers.register(name="nadam", tags=["adaptive", "nesterov"])
def build_nadam(config, learning_rate: Optional[float] = None):
    """Adam with Nesterov momentum."""
    return tf.keras.optimizers.Nadam(learning_rate=_lr(config, learning_rate), beta_1=config.ADAM_BETA1,
                                     beta_2=config.ADAM_BETA2, epsilon=config.ADAM_EPSILON)


Optimizers._initialized = True

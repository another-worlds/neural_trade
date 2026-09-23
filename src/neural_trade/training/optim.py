"""The two optimizers of a training run (main network + indicator logits).

The indicator logits get their own optimizer at ``LR * INDICATOR_LR_MULT``: Adam normalises
gradient magnitudes, so scaling their gradients cannot give them a larger step - only a larger
learning rate can. Gradient clipping is NOT configured on the optimizers; it happens once, per
group, in CustomTrainModel.train_step.
"""
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class OptimizerPair:
    main: tf.keras.optimizers.Optimizer
    indicator: tf.keras.optimizers.Optimizer


def build_indicator_optimizer(config) -> tf.keras.optimizers.Optimizer:
    from neural_trade.registries.optimizers import Optimizers

    lr = float(config.LR) * float(getattr(config, "INDICATOR_LR_MULT", 10.0))
    return Optimizers.build(getattr(config, "INDICATOR_OPTIMIZER_NAME", None), config, learning_rate=lr)


def build_optimizers(config) -> OptimizerPair:
    from neural_trade.registries.optimizers import Optimizers

    return OptimizerPair(
        main=Optimizers.build(getattr(config, "OPTIMIZER_NAME", None), config),
        indicator=build_indicator_optimizer(config),
    )

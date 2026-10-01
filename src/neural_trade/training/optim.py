"""The two optimizers of a training run (main network + indicator logits).

The indicator logits get their own optimizer at ``LR * INDICATOR_LR_MULT``: Adam normalises
gradient magnitudes, so scaling their gradients cannot give them a larger step - only a larger
learning rate can (this is also why ``Config.INDICATOR_GRAD_MULT``, a straight-through gradient
scale applied inside ``LearnableIndicators``, changes nothing under Adam except how often the
indicator group hits ``GRAD_CLIP_NORM``; NT-097, ``B_model_indicators.md`` 2.2/7.9). Gradient
clipping is NOT configured on the optimizers; it happens once, per group, in
CustomTrainModel.train_step.

Left to itself, this indicator optimizer's rate never decays: ``training/callbacks.py``'s
``reduce_lr_on_plateau`` builder only ever touches the main optimizer by default. NT-097 point 10
adds ``Config.LR_SCHEDULE_BOTH_OPTIMIZERS`` (default False, unchanged behaviour): when True, the
same plateau schedule scales this optimizer's rate by the same factor
(``training.callbacks.ReduceLRBothOptimizers``), so the ratio between the two rates stops growing.
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

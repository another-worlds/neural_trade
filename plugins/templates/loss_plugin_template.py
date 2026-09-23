"""Template: a component loss (first parameter `model`) or a full training objective.

Component losses are called from an objective; a new objective replaces custom_loss when
selected with Config.LOSS_NAME and must return neural_trade.core.outputs.LossComponents.
"""
import tensorflow as tf

from neural_trade.registries.losses import Losses


@Losses.register(name="my_component_loss", tags=["custom"])
def my_component_loss(model, y_true_scaled, y_pred_scaled):
    """Mean absolute error in scaled units."""
    return tf.reduce_mean(tf.abs(tf.cast(y_true_scaled, tf.float32) - tf.cast(y_pred_scaled, tf.float32)))

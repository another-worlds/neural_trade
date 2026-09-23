"""Template: an architecture. Select it with Config.MODEL_NAME.

The builder must return a Keras model with the 10 outputs of
neural_trade.core.outputs.PredictiveOutputs, each [None, 1]; Models.build checks this.
"""
import tensorflow as tf
from tensorflow.keras import layers

from neural_trade.registries.models import Models


@Models.register(name="my_model", tags=["custom"])
def build_my_model(config):
    """A minimal dense baseline with the required ten heads."""
    inp = layers.Input(shape=(config.LOOKBACK,), name="close_sequence")
    h = layers.Dense(32, activation="gelu")(inp)
    outs = []
    for k in range(3):
        outs.append(layers.Dense(1, name=f"price_h{k}")(h))
        outs.append(layers.Dense(1, activation="sigmoid", name=f"direction_h{k}")(h))
        outs.append(layers.Dense(1, activation="softplus", name=f"variance_h{k}")(h))
    outs.append(layers.Lambda(lambda t: tf.zeros_like(t[:, :1]), name="vacuum_overflow")(h))
    return tf.keras.Model(inp, outs)

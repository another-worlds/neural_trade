"""Sinusoidal positional encoding (moved from model.py in Phase B6)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class PositionalEncodingLayer(layers.Layer):
    """Transformer sinusoidal encoding of shape [B, T, D] (parameter-free)."""

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        positions = tf.range(0, tf.cast(seq_len, tf.float32), dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [batch_size, 1])
        positions = tf.expand_dims(positions, axis=-1)
        div_term = tf.exp(2.0 * tf.range(0, tf.cast(d_model // 2, tf.float32), dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
        div_term = tf.expand_dims(div_term, axis=0)
        div_term = tf.expand_dims(div_term, axis=0)  # [1,1,d//2]
        angles = positions * div_term
        even = tf.sin(angles)
        odd = tf.cos(angles)
        pe = tf.concat([even, odd], axis=-1)
        return pe

"""EnergyGate: volatility-adaptive blend of short/medium/long convolution branches.

Extracted in Phase B6 from functional ops that were inline in the model builder; the
computation is unchanged. High local energy (window variance/range) favours the short
kernel, low energy the long one:

    gate = softmax(Dense([var(window), max(window)]))          # [B, n_branches]
    out  = sum_k gate[:, k] * branch_k                          # [B, L, C]
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class EnergyGate(layers.Layer):
    """Inputs ``[window [B, L], branch_0, ..., branch_{n-1}]`` (branches [B, L, C])."""

    def __init__(self, n_branches: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_branches = int(n_branches)
        self.gate = layers.Dense(self.n_branches, activation="softmax", name="energy_gate_dense")

    def call(self, inputs):
        window, branches = inputs[0], list(inputs[1:])
        if len(branches) != self.n_branches:
            raise ValueError(f"EnergyGate expects {self.n_branches} branches, got {len(branches)}")
        x = tf.expand_dims(tf.cast(window, tf.float32), -1)                  # [B, L, 1]
        local_mean = tf.reduce_mean(x, axis=1)                                # [B, 1]
        local_var = tf.reduce_mean(tf.square(x - tf.expand_dims(local_mean, 1)), axis=1)  # [B, 1]
        local_max = tf.reduce_max(x, axis=1)                                  # [B, 1]
        g = self.gate(tf.concat([local_var, local_max], axis=-1))             # [B, n]
        out = branches[0] * tf.expand_dims(tf.expand_dims(g[:, 0], 1), 2)
        for k in range(1, self.n_branches):
            out = out + branches[k] * tf.expand_dims(tf.expand_dims(g[:, k], 1), 2)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_branches": self.n_branches})
        return cfg

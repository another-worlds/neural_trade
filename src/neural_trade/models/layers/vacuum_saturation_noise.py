"""VacuumSaturationNoise layer (moved from model.py in Phase B6)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class VacuumSaturationNoise(layers.Layer):
    """Vacuum Saturation Noise layer for the T_⊥ perpendicular subspace.

    Maintains maximum vacuum kernel energy density at all times:
        natural_noise + artificial_noise = VACUUM_E_MAX (per dimension)

    During training:
    1. Measure actual per-dimension energy:  energy[d] = mean(h²[:, d])   [T_PERP_DIM]
    2. Compute deficit:                      deficit[d] = relu(E_max - energy[d])
    3. Inject calibrated Gaussian noise:     noise ~ N(0, stop_grad(sqrt(deficit + ε)))
    4. Return h + noise  (saturated subspace)

    The noise is stop-gradient w.r.t. the deficit measurement so the network learns to
    fill the vacuum with real signal rather than chasing the artificial noise level.

    During inference (training=False): pass-through (deterministic predictions).

    The per-sample mean energy above E_max (computed downstream as a Lambda layer)
    is the observable T_⊥ overflow intensity — proportional to the fraction of the
    prediction residual that lives in the hidden perpendicular subspace.
    """

    def __init__(self, e_max=1.0, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.e_max = float(e_max)
        self.eps   = float(eps)

    def call(self, h_perp, training=None):
        if not training:
            return h_perp

        # h_perp: [B, T_PERP_DIM], values in (-1, 1) due to upstream tanh
        h = tf.cast(h_perp, tf.float32)

        # Batch-level per-dimension energy measurement
        energy_per_dim = tf.reduce_mean(tf.square(h), axis=0)          # [T_PERP_DIM]

        # Deficit = how much energy is missing to reach E_max per dim
        deficit = tf.nn.relu(
            tf.constant(self.e_max, dtype=tf.float32) - energy_per_dim
        )                                                                # [T_PERP_DIM]

        # Stop-gradient: noise level is treated as a constant forcing signal,
        # not a target the network learns to game by minimising deficit.
        noise_std = tf.stop_gradient(
            tf.sqrt(deficit + tf.constant(self.eps, dtype=tf.float32))
        )                                                                # [T_PERP_DIM]

        noise = tf.random.normal(shape=tf.shape(h), dtype=tf.float32)  # [B, T_PERP_DIM]
        return h + noise * noise_std                                    # broadcast over B

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'e_max': self.e_max, 'eps': self.eps})
        return cfg

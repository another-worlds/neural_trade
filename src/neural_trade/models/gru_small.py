"""``gru_small``: a capacity-reduced twin of ``gru_attention`` (NT-104).

learnable indicators -> one GRU(32) -> the same towers/heads as ``gru_attention`` (price,
direction with the optional DIRECTION_SKIP linear logit, and variance conditioned on the same
T-perp / regime-gate side path), producing the same 10 named PredictiveOutputs heads. It drops the
Bi-GRU(64), the two multi-head attention blocks, the multi-scale convolutions, the energy gate and
the two transformer blocks: 296,591 parameters at the reference config, 44.6% of them in one 8x32
attention over a 128-wide sequence (B_model_indicators.md 1.1), against a network that is never
above (and significantly below at 1h) a 3-lag logistic regression (B_model_indicators.md 7 item 2).
This variant is one arm of the pre-registered capacity study (NT-104 acceptance (3), the
experimenter's later item), scored on dev-fold net Sharpe like every other model, never chosen by
inspection.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from neural_trade.models.head_kit import (build_input_and_indicators, build_side_path,
                                          build_towers_and_heads, direction_skip_features)

GRU_UNITS = 32


def build_gru_small(config) -> tf.keras.Model:
    """Build the ``gru_small`` architecture for ``config``: same input and indicators as
    ``gru_attention``, one unidirectional ``GRU(32)`` backbone, the same heads."""
    inp, close_seq, ind_seq = build_input_and_indicators(config)

    # DETERMINISTIC_GRU (NT-114): see the identical comment in models/gru_attention.py.
    _deterministic_gru = bool(getattr(config, 'DETERMINISTIC_GRU', False))
    memory = layers.GRU(GRU_UNITS, return_sequences=True, name='gru_small_backbone',
                        unroll=_deterministic_gru)(ind_seq)
    memory = layers.Dropout(0.1)(memory)

    context = layers.GlobalAveragePooling1D()(memory)
    perp_magnitude, vacuum_overflow, regime_gate = build_side_path(config, context, close_seq)

    seq_flat = layers.Flatten()(memory)
    shared_dense = layers.Dense(32, activation='gelu',
                                kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(seq_flat)
    shared_dense = layers.Concatenate()([shared_dense, context])

    skip_features = direction_skip_features(close_seq, config)
    heads = build_towers_and_heads(config, shared_dense, perp_magnitude, regime_gate, skip_features,
                                   tower_hidden_units=16)

    price_h0, direction_h0, variance_h0, price_h1, direction_h1, variance_h1, \
        price_h2, direction_h2, variance_h2 = heads

    return tf.keras.Model(
        inputs=inp,
        outputs=[
            price_h0, direction_h0, variance_h0,
            price_h1, direction_h1, variance_h1,
            price_h2, direction_h2, variance_h2,
            vacuum_overflow,
        ],
    )

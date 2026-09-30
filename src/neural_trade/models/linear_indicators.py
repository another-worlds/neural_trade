"""``linear_indicators``: a linear twin of ``gru_attention`` on pooled indicator features (NT-104).

learnable indicators -> average/max pooling over the window -> linear heads: price and direction
are plain ``Dense(1)`` projections of the pooled indicator vector (direction still adds the
optional DIRECTION_SKIP logit and its own sigmoid; variance is still ``softplus`` of a linear
combination, conditioned on the same T-perp / regime-gate side path as every other model, since a
variance head cannot itself be linear). No GRU, no attention, no hidden tower: everything between
the indicators and the heads is pooling. This is the low end of the pre-registered capacity study
(NT-104 acceptance (3)): if a linear read-out of the learned indicators matches the deep stack's
dev-fold net Sharpe, the attention and convolution blocks (72% of gru_attention's 296,591
parameters, B_model_indicators.md 1.1) are not earning their cost.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from neural_trade.models.head_kit import (build_input_and_indicators, build_side_path,
                                          build_towers_and_heads, direction_skip_features)


def build_linear_indicators(config) -> tf.keras.Model:
    """Build the ``linear_indicators`` architecture for ``config``: same input and indicators as
    ``gru_attention``, pooled (no recurrence, no attention), linear heads."""
    inp, close_seq, ind_seq = build_input_and_indicators(config)

    pooled = layers.Concatenate()([
        layers.GlobalAveragePooling1D()(ind_seq),
        layers.GlobalMaxPooling1D()(ind_seq),
    ])
    context = pooled
    perp_magnitude, vacuum_overflow, regime_gate = build_side_path(config, context, close_seq)

    skip_features = direction_skip_features(close_seq, config)
    heads = build_towers_and_heads(config, pooled, perp_magnitude, regime_gate, skip_features,
                                   tower_hidden_units=None)

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

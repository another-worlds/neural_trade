"""``linear_indicators``: a linear twin of ``gru_attention`` on pooled indicator features (NT-104).

learnable indicators -> pooling over the window -> linear heads: price and direction are plain
``Dense(1)`` projections of the pooled indicator vector (direction still adds the optional
DIRECTION_SKIP logit and its own sigmoid; variance is still ``softplus`` of a linear combination,
conditioned on the same T-perp / regime-gate side path as every other model, since a variance head
cannot itself be linear). No GRU, no attention, no hidden tower: everything between the indicators
and the heads is pooling plus a normalisation. This is the low end of the pre-registered capacity
study (NT-104 acceptance (3)): if a linear read-out of the learned indicators matches the deep
stack's dev-fold net Sharpe, the attention and convolution blocks (72% of gru_attention's 296,591
parameters, B_model_indicators.md 1.1) are not earning their cost.

Pooling design (repair round 1, QA of a68d72b): the 14 indicator families are on very different
natural scales (RSI 0-100, a window-relative close near 0, a soft extremum near +-1, ...), so a raw
average+max pool over the window has a per-channel std from 0.015 to 358 and an abs max of about
2.2e3 at the default (OHLCV, 14-family) config. Feeding that straight into a ``Dense(1)`` head with
glorot-uniform weights gave an initial price std in the thousands and a loss of 542,944 on one CPU
epoch (QA script ``nt104_linear_epoch.py``), with about half the validation window's P(up) saturated
at 0 or 1. Two changes fix this without losing the family-scale information a hidden layer would
otherwise have to learn to ignore:

1. ``LayerNormalization`` on the pooled vector (per-sample, trainable scale/bias) puts every
   channel's pooled statistic on a comparable footing before the linear heads read it, the same
   role ``window_relative`` normalisation plays for the raw OHLCV input.
2. The **last bar's** indicator values are concatenated alongside the window's average and max pool,
   not only the window-wide summary: a linear head otherwise has no way to see "where the window
   ends now", which the deep architectures get for free from the GRU/attention's last position and
   from the raw channel ``gru_attention`` appends to every input mode.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from neural_trade.models.head_kit import (build_input_and_indicators, build_side_path,
                                          build_towers_and_heads, direction_skip_features)


def build_linear_indicators(config) -> tf.keras.Model:
    """Build the ``linear_indicators`` architecture for ``config``: same input and indicators as
    ``gru_attention``, pooled and normalised (no recurrence, no attention), linear heads."""
    inp, close_seq, ind_seq = build_input_and_indicators(config)

    last_bar = layers.Lambda(lambda t: t[:, -1, :], name='indicators_last_bar')(ind_seq)
    pooled_raw = layers.Concatenate(name='indicators_pooled_raw')([
        layers.GlobalAveragePooling1D()(ind_seq),
        layers.GlobalMaxPooling1D()(ind_seq),
        last_bar,
    ])
    pooled = layers.LayerNormalization(name='indicators_pooled_norm')(pooled_raw)
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

"""Shared input/indicator wiring and output heads for the capacity variants (NT-104).

``build_gru_attention`` (gru_attention.py) keeps its own inline copy of this wiring untouched
(golden-run bit-for-bit, D-023): this module is a separate, parallel implementation used only by
the new capacity variants (``gru_small``, ``linear_indicators``), so it can change freely without
touching the default architecture. It reproduces the same input handling, the same
``LearnableIndicators`` call, the same T-perp / vacuum-overflow / regime-gate side path and the
same per-horizon price/direction/variance heads (same layer names, same math), so every variant
built from these helpers satisfies the Models registry's PredictiveOutputs contract exactly like
``gru_attention`` does, and the trainer, losses, calibration and Predictor do not need to know
which architecture produced the heads.
"""
from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from neural_trade.indicators import num_learnable_logits
from neural_trade.models.gru_attention import _direction_head, _trailing_return_features
from neural_trade.models.layers_registry import Layers

__all__ = ["build_input_and_indicators", "direction_skip_features", "build_side_path", "build_towers_and_heads"]


def build_input_and_indicators(config):
    """``(inp, close_seq, ind_seq)``: the model input, its close channel, and the LearnableIndicators
    output sequence ``[B, LOOKBACK, num_channels]`` - identical to the first part of
    ``build_gru_attention`` (close-only or OHLCV per ``Config.INPUT_SERIES``, NT-047)."""
    series = tuple(getattr(config, 'INPUT_SERIES', None) or ['close'])
    if len(series) > 1:
        inp = layers.Input(shape=(config.LOOKBACK, len(series)), name='input_window')
        close_seq = layers.Lambda(lambda t: t[:, :, series.index('close')], name='close_channel')(inp)
        meta_inp = layers.Concatenate()([
            layers.GlobalAveragePooling1D()(inp),
            layers.GlobalMaxPooling1D()(inp),
        ])
    else:
        inp = layers.Input(shape=(config.LOOKBACK,), name='close_sequence')
        close_seq = inp
        inp_resh = layers.Reshape((config.LOOKBACK, 1))(inp)
        meta_inp = layers.Concatenate()([
            layers.GlobalAveragePooling1D()(inp_resh),
            layers.GlobalMaxPooling1D()(inp_resh),
        ])
    num_logits = num_learnable_logits(config)
    meta_adjust = layers.Dense(num_logits, activation='tanh')(meta_inp)
    ind_seq = Layers.for_role(config, 'indicators', config, name='learnable_indicators')([inp, meta_adjust])
    return inp, close_seq, ind_seq


def direction_skip_features(close_seq, config):
    """``None`` when ``Config.DIRECTION_SKIP`` is off, else the trailing-return feature vector
    (identical to ``gru_attention``'s ``_trailing_return_features``, same skip lags)."""
    if not bool(getattr(config, 'DIRECTION_SKIP', False)):
        return None
    return layers.Lambda(_trailing_return_features, name='direction_skip_features')(close_seq)


def build_side_path(config, context, close_seq) -> Tuple[object, object, object]:
    """The T-perp / vacuum-saturation / regime-gate side path, identical to ``gru_attention``'s:
    returns ``(perp_magnitude, vacuum_overflow, regime_gate)`` from a global context vector and the
    close channel. ``vacuum_overflow`` is the required 10th PredictiveOutputs head."""
    t_perp_dim = int(getattr(config, 'T_PERP_DIM', 16))
    h_perp = layers.Dense(t_perp_dim, activation='tanh', name='t_perp_proj')(context)

    e_max = float(getattr(config, 'VACUUM_E_MAX', 1.0))
    h_perp_sat = Layers.for_role(config, 'vacuum_noise', e_max=e_max,
                                 seeded=bool(getattr(config, 'SEEDED_STOCHASTIC_LAYERS', False)),
                                 name='vacuum_saturation')(h_perp)

    vacuum_overflow = layers.Lambda(
        lambda h: tf.where(
            tf.math.is_finite(tf.reduce_mean(tf.square(h), axis=1, keepdims=True)),
            tf.nn.relu(tf.reduce_mean(tf.square(h), axis=1, keepdims=True) - tf.constant(e_max, dtype=tf.float32)),
            tf.zeros([tf.shape(h)[0], 1], dtype=tf.float32),
        ),
        name='vacuum_overflow',
    )(h_perp_sat)

    perp_magnitude = layers.Dense(1, activation='softplus', name='t_perp_magnitude')(h_perp_sat)

    inp_for_gate = layers.Reshape((config.LOOKBACK, 1))(close_seq)
    gate_vol = layers.GlobalAveragePooling1D()(
        layers.Lambda(lambda t: tf.abs(t - tf.reduce_mean(t, axis=1, keepdims=True)))(inp_for_gate)
    )
    regime_gate = layers.Dense(1, activation='sigmoid', name='regime_gate')(
        layers.Concatenate()([gate_vol, context]))
    return perp_magnitude, vacuum_overflow, regime_gate


def build_towers_and_heads(config, tower_input, perp_magnitude, regime_gate, skip_features,
                           tower_hidden_units: Optional[int] = 16,
                           horizon_names=('h0', 'h1', 'h2')):
    """The 9 price/direction/variance heads (PredictiveOutputs order), identical math and layer
    names to ``gru_attention``'s three towers.

    ``tower_input`` is the ``[B, F]`` feature vector every horizon's tower reads. When
    ``tower_hidden_units`` is an int, each horizon gets its own ``Dense(tower_hidden_units, gelu)``
    hidden tower (as ``gru_attention`` and ``gru_small`` do); ``None`` skips the hidden layer and
    every head reads ``tower_input`` directly (``linear_indicators``: a linear head on pooled
    indicator features, plus the T-perp/regime-gate conditioning of the variance head)."""
    var_bias_init = tf.keras.initializers.Constant(1.0)
    dir_bias_init = tf.keras.initializers.Zeros()
    outputs = []
    for name in horizon_names:
        if tower_hidden_units is not None:
            tower = layers.Dense(tower_hidden_units, activation='gelu',
                                 kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(tower_input)
        else:
            tower = tower_input

        price = layers.Dense(1, name=f'price_{name}')(tower)
        price = layers.Lambda(
            lambda t: tf.where(tf.math.is_finite(t), tf.clip_by_value(t, -100.0, 100.0), tf.zeros_like(t)),
            name=f'price_{name}_clip')(price)

        direction = _direction_head(config, tower, skip_features, f'direction_{name}', dir_bias_init)
        direction = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 1.0), name=f'direction_{name}_clip')(direction)

        var_input = layers.Concatenate()([tower, perp_magnitude, regime_gate])
        variance = layers.Dense(1, activation='softplus', name=f'variance_{name}',
                                bias_initializer=var_bias_init)(var_input)
        variance = layers.Lambda(lambda t: tf.where(tf.math.is_finite(t), t, tf.ones_like(t)),
                                 name=f'variance_{name}_clip')(variance)
        outputs.extend([price, direction, variance])
    return outputs

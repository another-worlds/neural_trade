"""The production architecture ``gru_attention`` (moved from PricePredictor.build_model in B7).

learnable indicators -> Bi-GRU -> temporal and cross-indicator attention -> multi-scale
convolutions blended by an energy gate -> positional encoding -> 2 transformer blocks ->
three horizon towers, each with a price (scaled delta), direction (P(up)) and variance head,
plus the T-perp / vacuum-overflow side outputs. Returns an uncompiled functional Keras model
with the 10 outputs of :class:`neural_trade.core.outputs.PredictiveOutputs`, in that order.

Custom layers are resolved by role through the Layers registry (``Config.LAYERS``).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from neural_trade.registries.layers import Layers


def build_gru_attention(config) -> tf.keras.Model:
    """Build the gru_attention architecture for ``config`` (LOOKBACK, indicators, T_PERP_DIM...)."""
    inp = layers.Input(shape=(config.LOOKBACK,), name='close_sequence')

    # Compute meta_adjust from raw input stats
    inp_resh = layers.Reshape((config.LOOKBACK, 1))(inp)  # [B, LOOKBACK, 1] for pooling
    meta_inp = layers.Concatenate()([
        layers.GlobalAveragePooling1D()(inp_resh),
        layers.GlobalMaxPooling1D()(inp_resh)
    ])
    num_logits = (len(config.MA_SPANS) +
                  len(config.MACD_SETTINGS) * 3 +
                  len(config.RSI_PERIODS) +
                  len(config.BB_PERIODS))
    meta_adjust = layers.Dense(num_logits, activation='tanh')(meta_inp)

    # Enhanced Learnable Indicators: Now takes [inp, meta_adjust], outputs sequences [B, LOOKBACK, num_ind]
    ind_seq = Layers.for_role(config, 'indicators', config, name='learnable_indicators')([inp, meta_adjust])

    # Memory-Supplemented Layers: Capture temporal interconnections
    memory = layers.Bidirectional(layers.GRU(64, return_sequences=True))(ind_seq)
    memory = layers.Dropout(0.1)(memory)

    # Interconnection Attention: Model relations between indicators
    att_key_dim = 32
    att = layers.MultiHeadAttention(num_heads=8, key_dim=att_key_dim)(memory, memory)
    x = layers.Add()([memory, att])
    x = layers.LayerNormalization()(x)

    # Graph-like view: Attend across indicators
    x_perm = layers.Permute((2, 1))(x)  # [B, num_ind, LOOKBACK]
    inter_att = layers.MultiHeadAttention(num_heads=4, key_dim=att_key_dim)(x_perm, x_perm)
    x_perm = layers.Add()([x_perm, inter_att])
    x = layers.Permute((2, 1))(layers.LayerNormalization()(x_perm))  # Back to [B, LOOKBACK, num_ind]

    # Multi-scale Conv feature extractor
    x_short = layers.Conv1D(16, 3, padding='same', activation='gelu')(x)
    x_med = layers.Conv1D(16, 7, padding='same', activation='gelu')(x)
    x_long = layers.Conv1D(16, 15, padding='same', activation='gelu')(x)

    # === ENERGY GATE - k(E) adaptive kernel weighting (neural_trade.models.layers.EnergyGate) ===
    # High local volatility (energy) -> short kernel dominates; low -> long kernel.
    x = Layers.for_role(config, 'energy_gate', n_branches=3, name='energy_gate')([inp, x_short, x_med, x_long])  # [B, LOOKBACK, 16]
    x = layers.LayerNormalization()(x)

    # Positional encoding
    x = layers.Add()([x, Layers.for_role(config, 'positional_encoding')(x)])

    # Transformer-style blocks (reduced to 2 for speed)
    for _ in range(2):
        att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = layers.Add()([x, att])
        x = layers.LayerNormalization()(x)
        ff = layers.Dense(32, activation='gelu')(x)
        ff = layers.Dropout(0.1)(ff)
        ff = layers.Dense(x.shape[-1])(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    # Global context vector
    context = layers.GlobalAveragePooling1D()(x)

    # === T_⊥ PERPENDICULAR PROJECTION ===
    # T_⊥ encodes the energy/information that escaped the observable projection.
    # In trading: unexplained residual variance = hidden order-flow / regime change.
    # perp_magnitude → conditions ALL variance heads: high T_⊥ → high predicted σ.
    # This prevents variance heads from driving uncertainty to zero when T_⊥ is large.
    _t_perp_dim = int(getattr(config, 'T_PERP_DIM', 16))
    h_perp = layers.Dense(_t_perp_dim, activation='tanh',
                           name='t_perp_proj')(context)               # [B, T_PERP_DIM]

    # === VACUUM SATURATION ===
    # Fill each kernel dimension to VACUUM_E_MAX with calibrated Gaussian noise.
    # natural_noise + artificial_noise = E_max at all times during training.
    # training=False: pure pass-through (deterministic inference).
    _e_max = float(getattr(config, 'VACUUM_E_MAX', 1.0))
    h_perp_sat = Layers.for_role(config, 'vacuum_noise',
        e_max=_e_max, name='vacuum_saturation')(h_perp)               # [B, T_PERP_DIM]

    # Overflow = per-sample mean energy above E_max in the saturated subspace.
    # This is the observable T_⊥ intensity: energy that could not be absorbed by
    # the vacuum kernels — proportional to unexplained prediction residual.
    vacuum_overflow = layers.Lambda(
        lambda h: tf.where(
            tf.math.is_finite(
                tf.reduce_mean(tf.square(h), axis=1, keepdims=True)
            ),
            tf.nn.relu(
                tf.reduce_mean(tf.square(h), axis=1, keepdims=True)
                - tf.constant(_e_max, dtype=tf.float32)
            ),
            tf.zeros( [tf.shape(h)[0], 1], dtype=tf.float32 )
        ),
        name='vacuum_overflow'
    )(h_perp_sat)                                                     # [B, 1]

    perp_magnitude = layers.Dense(1, activation='softplus',
                                  name='t_perp_magnitude')(h_perp_sat)  # [B, 1]

    # === REGIME GATE (White-hole / T_⊥^up detector) ===
    # Regime gate ≈ 1.0 when market is in a "white hole" state:
    #   new information is flowing IN from outside (regime breaks, flash crashes,
    #   macro news shocks), making the current visible projection insufficient.
    # Regime gate ≈ 0.0 = "black hole" state: coherent trend, info is observable.
    # Computed from local price std (volatility level) fused with global context.
    _inp_for_gate = layers.Reshape((config.LOOKBACK, 1))(inp)
    _gate_vol = layers.GlobalAveragePooling1D()(
        layers.Lambda(lambda t: tf.abs(t - tf.reduce_mean(t, axis=1, keepdims=True)))(
            _inp_for_gate)
    )                                                                    # [B, 1]
    regime_gate = layers.Dense(1, activation='sigmoid',
                               name='regime_gate')(
        layers.Concatenate()([_gate_vol, context]))                  # [B, 1]

    # Sequence summary for regression
    seq_flat = layers.Flatten()(x)

    # Shared dense layer for all output heads
    shared_dense = layers.Dense(32, activation='gelu',
                               kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(seq_flat)
    shared_dense = layers.Concatenate()([shared_dense, context])

    # === THREE INDEPENDENT OUTPUT TOWERS (h0, h1, h2) ===
    # Each horizon has its own price, direction, and confidence (variance) head

    # Variance bias initialization: softplus(x) ≈ x for x > 0
    # Initialize bias so initial variance ≈ 1.3 (higher than unit variance for calibration learning)
    # softplus(0) ≈ 0.693, softplus(0.5) ≈ 0.97, softplus(1.0) ≈ 1.31
    var_bias_init = tf.keras.initializers.Constant(1.0)  # Initial variance ≈ 1.31

    # Direction bias initialization: sigmoid(0) = 0.5 (unbiased)
    # Keep at 0 for balanced initial predictions
    dir_bias_init = tf.keras.initializers.Zeros()

    # ---- TOWER 0 (1-minute horizon) ----
    tower_h0 = layers.Dense(16, activation='gelu',
                           kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(shared_dense)
    price_h0 = layers.Dense(1, name='price_h0')(tower_h0)
    # Clip + sanitize price outputs (in *scaled* delta units) to prevent extreme values or NaN/Inf
    # from random init, high dropout (0.8), or early unstable indicator steps from producing inf/nan
    # that poisons losses (NLL err^2/var, logcosh, coherence signs, etc.) and drives weights to NaN.
    # NaN/Inf -> 0 (neutral delta); extremes clipped. Wide bound allows exploration.
    price_h0 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), tf.clip_by_value(t, -100.0, 100.0), tf.zeros_like(t)),
        name='price_h0_clip'
    )(price_h0)
    direction_h0 = layers.Dense(1, activation='sigmoid', name='direction_h0',
                               bias_initializer=dir_bias_init)(tower_h0)
    # Clip dir probs to [0,1]. (NaN/Inf protection is handled by loss guards + post-extraction sanitization
    # to avoid any appearance of hard-coded 0.5 in the architecture.)
    direction_h0 = layers.Lambda(
        lambda t: tf.clip_by_value(t, 0.0, 1.0),
        name='direction_h0_clip'
    )(direction_h0)
    # Variance head conditioned on T_⊥ and regime gate:
    #   high perp_magnitude → more energy in hidden dims → higher σ²
    #   high regime_gate → white-hole / regime-break → higher σ²
    tower_h0_var_input = layers.Concatenate()([tower_h0, perp_magnitude, regime_gate])
    variance_h0 = layers.Dense(1, activation='softplus', name='variance_h0',
                              bias_initializer=var_bias_init)(tower_h0_var_input)
    # Sanitize var (NaN/Inf -> 1.0); softplus already >=~0 but upstream nan can leak. Loss also clips.
    variance_h0 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), t, tf.ones_like(t)),
        name='variance_h0_clip'
    )(variance_h0)

    # ---- TOWER 1 (5-minute horizon - PRIMARY) ----
    tower_h1 = layers.Dense(16, activation='gelu',
                           kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(shared_dense)
    price_h1 = layers.Dense(1, name='price_h1')(tower_h1)
    price_h1 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), tf.clip_by_value(t, -100.0, 100.0), tf.zeros_like(t)),
        name='price_h1_clip'
    )(price_h1)
    direction_h1 = layers.Dense(1, activation='sigmoid', name='direction_h1',
                               bias_initializer=dir_bias_init)(tower_h1)
    direction_h1 = layers.Lambda(
        lambda t: tf.clip_by_value(t, 0.0, 1.0),
        name='direction_h1_clip'
    )(direction_h1)
    tower_h1_var_input = layers.Concatenate()([tower_h1, perp_magnitude, regime_gate])
    variance_h1 = layers.Dense(1, activation='softplus', name='variance_h1',
                              bias_initializer=var_bias_init)(tower_h1_var_input)
    variance_h1 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), t, tf.ones_like(t)),
        name='variance_h1_clip'
    )(variance_h1)

    # ---- TOWER 2 (15-minute horizon) ----
    tower_h2 = layers.Dense(16, activation='gelu',
                           kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(shared_dense)
    price_h2 = layers.Dense(1, name='price_h2')(tower_h2)
    price_h2 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), tf.clip_by_value(t, -100.0, 100.0), tf.zeros_like(t)),
        name='price_h2_clip'
    )(price_h2)
    direction_h2 = layers.Dense(1, activation='sigmoid', name='direction_h2',
                               bias_initializer=dir_bias_init)(tower_h2)
    direction_h2 = layers.Lambda(
        lambda t: tf.clip_by_value(t, 0.0, 1.0),
        name='direction_h2_clip'
    )(direction_h2)
    tower_h2_var_input = layers.Concatenate()([tower_h2, perp_magnitude, regime_gate])
    variance_h2 = layers.Dense(1, activation='softplus', name='variance_h2',
                              bias_initializer=var_bias_init)(tower_h2_var_input)
    variance_h2 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), t, tf.ones_like(t)),
        name='variance_h2_clip'
    )(variance_h2)

    # === FINAL MODEL: 10 outputs (3 horizons × 3 heads + vacuum_overflow) ===
    # Output index layout:
    #   0: price_h0    1: direction_h0    2: variance_h0
    #   3: price_h1    4: direction_h1    5: variance_h1
    #   6: price_h2    7: direction_h2    8: variance_h2
    #   9: vacuum_overflow  [B, 1]  (T_⊥ overflow intensity; 0 at inference)
    return models.Model(
        inputs=inp,
        outputs=[
            price_h0, direction_h0, variance_h0,
            price_h1, direction_h1, variance_h1,
            price_h2, direction_h2, variance_h2,
            vacuum_overflow,
        ]
    )

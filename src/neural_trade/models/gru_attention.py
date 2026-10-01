"""The production architecture ``gru_attention`` (moved from PricePredictor.build_model in B7).

learnable indicators -> Bi-GRU -> temporal and cross-indicator attention -> multi-scale
convolutions blended by an energy gate -> positional encoding -> 2 transformer blocks ->
three horizon towers, each with a price (scaled delta), direction (P(up)) and variance head,
plus the T-perp / vacuum-overflow side outputs. Returns an uncompiled functional Keras model
with the 10 outputs of :class:`neural_trade.core.outputs.PredictiveOutputs`, in that order.

Custom layers are resolved by role through the Layers registry (``Config.LAYERS``).

``Config.ATTENTION_MODE`` ('time' default, 'channels', 'none') and ``Config.HEAD_POOL``
('flatten' default, 'mean', 'attention') switch two blocks whose parameter count otherwise
depends on LOOKBACK (NT-105, B_model_indicators.md 1.1/1.4/7.1): the default path is unchanged.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from neural_trade.indicators import num_learnable_logits
from neural_trade.models.layers_registry import Layers


SKIP_LAGS = (1, 5, 10, 15, 20, 30)


def _trailing_return_features(x, lags=SKIP_LAGS):
    """[B, LOOKBACK] window-relative input -> [B, len(lags) + 2]: the (scaled) close change over each
    trailing lag, over the whole window, and the log of the 1-bar change volatility. The same
    information a logistic regression on trailing returns uses; all causal (inside the window)."""
    last = x[:, -1:]
    cols = [last - x[:, -1 - k:-k] for k in lags] + [last - x[:, :1]]
    vol = tf.math.reduce_std(x[:, 1:] - x[:, :-1], axis=1, keepdims=True)
    return tf.concat(cols + [tf.math.log(vol + 1e-6)], axis=1)


def _channel_attention_gate(ind_seq, key_dim=8, num_heads=2):
    """ATTENTION_MODE='channels' (NT-105, B_model_indicators.md 1.4): attention across the
    indicator channels (tokens = channels) instead of across the 128 GRU units with the window
    length as each unit's features. Runs on the LearnableIndicators output ``[B, L, C]``,
    before the GRU. Each channel is summarised by its mean and standard deviation over the
    window (2 numbers, independent of L), the channels attend over each other, and a sigmoid
    gate (0..1 per channel) re-weights ``ind_seq`` before it reaches the GRU. Every weight
    matrix here is sized by the fixed key_dim / head count, not by L, so (unlike the 'time'
    block it replaces) the parameter count does not depend on the window length."""
    mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name='channel_mean')(ind_seq)  # [B, C]
    std = layers.Lambda(lambda t: tf.math.reduce_std(t, axis=1), name='channel_std')(ind_seq)  # [B, C]
    stats = layers.Lambda(lambda ts: tf.stack(ts, axis=-1), name='channel_stats')([mean, std])  # [B, C, 2]
    tokens = layers.Dense(key_dim, name='channel_embed')(stats)  # [B, C, key_dim]
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                    name='channel_attention')(tokens, tokens)
    tokens = layers.Add(name='channel_attention_add')([tokens, att])
    tokens = layers.LayerNormalization(name='channel_attention_ln')(tokens)
    gate = layers.Dense(1, activation='sigmoid', name='channel_gate')(tokens)  # [B, C, 1]
    gate = layers.Lambda(lambda t: tf.transpose(t, [0, 2, 1]), name='channel_gate_transpose')(gate)  # [B, 1, C]
    return layers.Multiply(name='channel_gated')([ind_seq, gate])


def _attention_pool(x, key_dim=16):
    """HEAD_POOL='attention' (NT-105): a single learned query attends over the LOOKBACK
    positions of ``x`` ``[B, L, C]`` and returns the weighted sum ``[B, C]``; every weight here
    is sized by C and key_dim, not by L."""
    channels = x.shape[-1]
    query = layers.Dense(key_dim, use_bias=False, name='head_pool_query')(
        layers.Lambda(lambda t: tf.ones_like(t[:, :1, :1]), name='head_pool_query_seed')(x))  # [B, 1, key_dim]
    key = layers.Dense(key_dim, name='head_pool_key')(x)  # [B, L, key_dim]
    value = layers.Dense(channels, name='head_pool_value')(x)  # [B, L, C]
    scores = layers.Lambda(
        lambda qk: tf.matmul(qk[0], qk[1], transpose_b=True) / (float(key_dim) ** 0.5),
        name='head_pool_scores'
    )([query, key])  # [B, 1, L]
    weights = layers.Softmax(axis=-1, name='head_pool_softmax')(scores)
    pooled = layers.Lambda(lambda vw: tf.matmul(vw[0], vw[1]), name='head_pool_weighted_sum')(
        [weights, value])  # [B, 1, C]
    return layers.Reshape((channels,), name='head_pool_flatten')(pooled)  # [B, C]


def _direction_head(config, tower, skip_features, name, bias_init):
    """P(up) head. Without the skip this is exactly the original Dense(1, sigmoid) layer.

    ``Config.DIRECTION_DEEP_ZERO_INIT`` (NT-104, default off) zero-initialises the deep tower's
    kernel so the head starts exactly at the DIRECTION_SKIP linear logit (the bias is already 0);
    off keeps today's glorot-uniform kernel, bit-for-bit (golden run)."""
    if skip_features is None:
        return layers.Dense(1, activation='sigmoid', name=name, bias_initializer=bias_init)(tower)
    tower_kernel_init = 'zeros' if bool(getattr(config, 'DIRECTION_DEEP_ZERO_INIT', False)) else 'glorot_uniform'
    tower_logit = layers.Dense(1, name=f'{name}_logit', bias_initializer=bias_init,
                               kernel_initializer=tower_kernel_init)(tower)
    skip_logit = layers.Dense(1, name=f'{name}_skip', use_bias=False,
                              kernel_regularizer=regularizers.L2(float(config.DIRECTION_SKIP_L2)))(skip_features)
    return layers.Activation('sigmoid', name=name)(layers.Add()([tower_logit, skip_logit]))


def build_gru_attention(config) -> tf.keras.Model:
    """Build the gru_attention architecture for ``config`` (LOOKBACK, indicators, T_PERP_DIM...).

    The input is the close sequence ``[B, LOOKBACK]`` (``Config.INPUT_SERIES = ['close']``,
    the pre-NT-047 graph, kept bit-for-bit) or the multi-series window
    ``[B, LOOKBACK, len(INPUT_SERIES)]`` (NT-047). The close-derived paths (meta pooling in
    close mode, energy gate, regime gate, trailing-return skip) keep their semantics: in
    multi-series mode they read the close channel; only the meta_adjust pooling reads every
    channel (window statistics of the whole input)."""
    series = tuple(getattr(config, 'INPUT_SERIES', None) or ['close'])
    if len(series) > 1:
        inp = layers.Input(shape=(config.LOOKBACK, len(series)), name='input_window')
        _ci = series.index('close')
        close_seq = layers.Lambda(lambda t: t[:, :, _ci], name='close_channel')(inp)
        # meta_adjust reads avg/max pooled stats of EVERY input channel
        meta_inp = layers.Concatenate()([
            layers.GlobalAveragePooling1D()(inp),
            layers.GlobalMaxPooling1D()(inp)
        ])
    else:
        inp = layers.Input(shape=(config.LOOKBACK,), name='close_sequence')
        close_seq = inp

        # Compute meta_adjust from raw input stats
        inp_resh = layers.Reshape((config.LOOKBACK, 1))(inp)  # [B, LOOKBACK, 1] for pooling
        meta_inp = layers.Concatenate()([
            layers.GlobalAveragePooling1D()(inp_resh),
            layers.GlobalMaxPooling1D()(inp_resh)
        ])
    num_logits = num_learnable_logits(config)  # one meta-adjust column per learnable period
    # NT-097 (B_model_indicators.md 2.2, 7.6): alpha = sigmoid(logit + 0.5*tanh(Wz + c)) - the base
    # logit and this Dense's bias c are one unidentifiable direction, trained by two different
    # optimizers (indicator LR 0.005 vs. the main LR). META_ADJUST_BIAS keeps the bias on by
    # default (today's behaviour, golden-run bit-for-bit); the A/B of NT-097 point 7 decides
    # whether "no bias" becomes the new default. Named so evaluation code can recover this tensor
    # from the built model (neural_trade.evaluation.applied_periods, NT-097 point 5).
    meta_adjust = layers.Dense(num_logits, activation='tanh',
                               use_bias=bool(getattr(config, 'META_ADJUST_BIAS', True)),
                               name='meta_adjust')(meta_inp)

    # Enhanced Learnable Indicators: Now takes [inp, meta_adjust], outputs sequences [B, LOOKBACK, num_ind]
    ind_seq = Layers.for_role(config, 'indicators', config, name='learnable_indicators')([inp, meta_adjust])

    # ATTENTION_MODE (NT-105, B_model_indicators.md 1.4): 'channels' attends across the indicator
    # channels (tokens = channels) before the GRU, instead of the post-GRU 'time' block below.
    attention_mode = str(getattr(config, 'ATTENTION_MODE', 'time')).lower()
    if attention_mode == 'channels':
        ind_seq = _channel_attention_gate(ind_seq)
    elif attention_mode not in ('time', 'none'):
        raise ValueError(f"ATTENTION_MODE={attention_mode!r} is not one of time, channels, none")

    # Memory-Supplemented Layers: Capture temporal interconnections
    memory = layers.Bidirectional(layers.GRU(64, return_sequences=True))(ind_seq)
    memory = layers.Dropout(0.1)(memory)

    # Interconnection Attention: Model relations between indicators
    att_key_dim = 32
    att = layers.MultiHeadAttention(num_heads=8, key_dim=att_key_dim)(memory, memory)
    x = layers.Add()([memory, att])
    x = layers.LayerNormalization()(x)

    if attention_mode == 'time':
        # Graph-like view (today's default): attends over the 128 GRU units, with the LOOKBACK
        # time positions as each unit's features; ties the parameter count to LOOKBACK (D-045
        # math report 1.4: this is NOT attention across the 31 indicator channels, despite the
        # name - see ATTENTION_MODE='channels' above for that).
        x_perm = layers.Permute((2, 1))(x)  # [B, num_ind, LOOKBACK]
        inter_att = layers.MultiHeadAttention(num_heads=4, key_dim=att_key_dim)(x_perm, x_perm)
        x_perm = layers.Add()([x_perm, inter_att])
        x = layers.Permute((2, 1))(layers.LayerNormalization()(x_perm))  # Back to [B, LOOKBACK, num_ind]
    # 'channels': already mixed before the GRU, above. 'none': no cross-unit block here.

    # Multi-scale Conv feature extractor
    x_short = layers.Conv1D(16, 3, padding='same', activation='gelu')(x)
    x_med = layers.Conv1D(16, 7, padding='same', activation='gelu')(x)
    x_long = layers.Conv1D(16, 15, padding='same', activation='gelu')(x)

    # === ENERGY GATE - k(E) adaptive kernel weighting (neural_trade.models.layers.EnergyGate) ===
    # High local volatility (energy) -> short kernel dominates; low -> long kernel.
    x = Layers.for_role(config, 'energy_gate', n_branches=3, name='energy_gate')([close_seq, x_short, x_med, x_long])  # [B, LOOKBACK, 16]
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
    # seeded: Config.SEEDED_STOCHASTIC_LAYERS (NT-092), default False - see VacuumSaturationNoise.
    h_perp_sat = Layers.for_role(config, 'vacuum_noise', e_max=_e_max,
        seeded=bool(getattr(config, 'SEEDED_STOCHASTIC_LAYERS', False)),
        name='vacuum_saturation')(h_perp)                              # [B, T_PERP_DIM]

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
    _inp_for_gate = layers.Reshape((config.LOOKBACK, 1))(close_seq)
    _gate_vol = layers.GlobalAveragePooling1D()(
        layers.Lambda(lambda t: tf.abs(t - tf.reduce_mean(t, axis=1, keepdims=True)))(
            _inp_for_gate)
    )                                                                    # [B, 1]
    regime_gate = layers.Dense(1, activation='sigmoid',
                               name='regime_gate')(
        layers.Concatenate()([_gate_vol, context]))                  # [B, 1]

    # Sequence summary for regression. HEAD_POOL (NT-105, B_model_indicators.md 1.1/7.1):
    # 'flatten' (today) ties the Dense(32) below to LOOKBACK (512*L + 32 parameters); 'mean' and
    # 'attention' summarise x into a fixed-size vector first, independent of LOOKBACK.
    head_pool = str(getattr(config, 'HEAD_POOL', 'flatten')).lower()
    if head_pool == 'flatten':
        seq_summary = layers.Flatten()(x)
    elif head_pool == 'mean':
        seq_summary = layers.GlobalAveragePooling1D()(x)
    elif head_pool == 'attention':
        seq_summary = _attention_pool(x)
    else:
        raise ValueError(f"HEAD_POOL={head_pool!r} is not one of flatten, mean, attention")

    # Shared dense layer for all output heads
    shared_dense = layers.Dense(32, activation='gelu',
                               kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(seq_summary)
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

    # Optional linear path from trailing-return features straight to the direction logits
    # (Config.DIRECTION_SKIP): the heads can then represent at least the linear baseline, which
    # the deep path alone did not recover.
    skip_features = None
    if bool(getattr(config, 'DIRECTION_SKIP', False)):
        skip_features = layers.Lambda(_trailing_return_features, name='direction_skip_features')(close_seq)

    # ---- TOWER 0 (horizon 0: config.HORIZON_STEPS[0] bars) ----
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
    direction_h0 = _direction_head(config, tower_h0, skip_features, 'direction_h0', dir_bias_init)
    # Clip dir probs to [0,1]. Mathematically a no-op (the head's sigmoid activation already
    # guarantees [0,1], and clip_by_value does not sanitize NaN/Inf: clip(nan, 0, 1) == nan) -
    # NT-096 (D-029) looked at removing it, but the layer graph it sits in is load-bearing for
    # backward-compatible HDF5 weight loading: removing it shifts Keras's auto-numbering of the
    # unnamed Dense layers downstream (DIRECTION_SKIP's '*_skip'/'*_logit' layers), which broke
    # tests/test_legacy_bundle.py and tests/test_indicator_families.py (a real saved bundle failed
    # to load: "Weight count mismatch ... direction_h0_skip"). That is an effect, so D-029 does not
    # allow removing it; the layer stays. (NaN/Inf protection is the loss guards + post-extraction
    # sanitization, not this clip.)
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

    # ---- TOWER 1 (horizon 1: config.HORIZON_STEPS[1] bars) ----
    tower_h1 = layers.Dense(16, activation='gelu',
                           kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(shared_dense)
    price_h1 = layers.Dense(1, name='price_h1')(tower_h1)
    price_h1 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), tf.clip_by_value(t, -100.0, 100.0), tf.zeros_like(t)),
        name='price_h1_clip'
    )(price_h1)
    direction_h1 = _direction_head(config, tower_h1, skip_features, 'direction_h1', dir_bias_init)
    # Kept (NT-096, D-029): see the identical comment on direction_h0_clip above.
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

    # ---- TOWER 2 (horizon 2: config.HORIZON_STEPS[2] bars) ----
    tower_h2 = layers.Dense(16, activation='gelu',
                           kernel_regularizer=regularizers.L2(config.REG_MOMENTUM_L2))(shared_dense)
    price_h2 = layers.Dense(1, name='price_h2')(tower_h2)
    price_h2 = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), tf.clip_by_value(t, -100.0, 100.0), tf.zeros_like(t)),
        name='price_h2_clip'
    )(price_h2)
    direction_h2 = _direction_head(config, tower_h2, skip_features, 'direction_h2', dir_bias_init)
    # Kept (NT-096, D-029): see the identical comment on direction_h0_clip above.
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

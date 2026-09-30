"""LearnableIndicators: EWMA equivalence (matrix vs scan), output shape, gradient reach."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import neural_trade.utils.math as mh
from neural_trade.core.config import Config
from neural_trade.models.layers import LearnableIndicators

# the pre-NT-047 default this file pins: close-only input, the four families (18 logits)
OLD = dict(INPUT_SERIES=["close"], INDICATOR_FAMILIES={})

B, T = 16, 60


def _x(seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(np.cumsum(rng.normal(0, 1, (B, T)), axis=1).astype(np.float32))


@pytest.mark.parametrize("alpha_kind", ["scalar", "per_sample", "extremes"])
def test_ewma_matrix_equals_scan_values_and_gradients(alpha_kind):
    rng = np.random.default_rng(1)
    if alpha_kind == "scalar":
        a0 = np.float32(2.0 / 21.0)
    elif alpha_kind == "per_sample":
        a0 = rng.uniform(0.02, 0.9, B).astype(np.float32)
    else:  # period 2 (alpha 2/3) and period 60 (alpha 2/61), the clip bounds
        a0 = np.where(np.arange(B) % 2 == 0, 2.0 / 3.0, 2.0 / 61.0).astype(np.float32)
    x = tf.Variable(_x())
    a = tf.Variable(a0)

    with tf.GradientTape(persistent=True) as tape:
        ref = mh.ewma_sequence(x, a)
        got = mh.ewma_sequence_matrix(x, a)
        w = tf.constant(np.random.default_rng(2).normal(0, 1, (B, T)).astype(np.float32))
        l_ref = tf.reduce_sum(ref * w)
        l_got = tf.reduce_sum(got * w)
    np.testing.assert_allclose(got.numpy(), ref.numpy(), atol=1e-4, rtol=1e-5)
    for var in (x, a):
        g_ref = tape.gradient(l_ref, var).numpy()
        g_got = tape.gradient(l_got, var).numpy()
        np.testing.assert_allclose(g_got, g_ref, rtol=1e-4, atol=1e-3)


def test_ewma_matrix_is_finite_when_sigmoid_rounds_alpha_to_one():
    x = tf.Variable(_x())
    logit = tf.Variable(tf.fill([B], 40.0))  # float32 sigmoid(40) == 1.0 exactly
    with tf.GradientTape() as tape:
        y = mh.ewma_sequence_matrix(x, tf.sigmoid(logit))
        loss = tf.reduce_sum(y)
    assert float(tf.sigmoid(40.0)) == 1.0
    np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-3)  # alpha ~ 1 -> ema ~ x
    g = tape.gradient(loss, [x, logit])
    assert all(np.all(np.isfinite(t.numpy())) for t in g)


@pytest.mark.parametrize("impl", ["matrix", "scan"])
def test_layer_output_shape_and_gradient_reaches_all_18_logits(impl):
    cfg = Config(**OLD)
    cfg.EWMA_IMPL = impl
    layer = LearnableIndicators(cfg)
    x = _x()
    meta = tf.zeros([B, 18])
    with tf.GradientTape() as tape:
        out = layer([x, meta])
        loss = tf.reduce_mean(tf.square(out))
    assert out.shape == (B, cfg.LOOKBACK, 31)
    assert np.all(np.isfinite(out.numpy()))
    ind = layer.get_indicator_trainable_variables()
    assert len(ind) == 18
    grads = tape.gradient(loss, ind)
    assert all(g is not None and np.isfinite(float(g)) for g in grads)
    assert sum(abs(float(g)) > 0 for g in grads) == 18, "every learnable period must receive gradient"


def test_matrix_and_scan_layers_agree():
    outs = []
    for impl in ("scan", "matrix"):
        cfg = Config(**OLD)
        cfg.EWMA_IMPL = impl
        tf.keras.utils.set_random_seed(0)
        layer = LearnableIndicators(cfg)
        outs.append(layer([_x(3), tf.random.normal([B, 18], seed=4) * 0.1]).numpy())
    # RSI divides two EWMAs; compare with a tolerance scaled to each feature's magnitude
    scale = np.maximum(np.abs(outs[0]).max(axis=(0, 1), keepdims=True), 1.0)
    np.testing.assert_allclose(outs[1] / scale, outs[0] / scale, atol=2e-4)


def test_multi_series_ewma_equals_one_call_per_series():
    rng = np.random.default_rng(3)
    K = 5
    xs = tf.constant(np.cumsum(rng.normal(0, 1, (B, K, T)), axis=2).astype(np.float32))
    al = tf.constant(rng.uniform(0.02, 0.9, (B, K)).astype(np.float32))
    multi = mh.ewma_sequence_matrix_multi(xs, al).numpy()
    for k in range(K):
        one = mh.ewma_sequence_matrix(xs[:, k], al[:, k]).numpy()
        np.testing.assert_allclose(multi[:, k], one, rtol=1e-5, atol=1e-5)


def test_batched_layer_equals_per_indicator_layer_values_and_gradients():
    """The two-stage batched call (default) computes exactly the per-indicator features."""
    tf.keras.utils.set_random_seed(0)
    layer = LearnableIndicators(Config(**OLD))
    x = tf.Variable(_x(4) * 0.3)
    c = layer.config
    n_logits = len(c.MA_SPANS) + 3 * len(c.MACD_SETTINGS) + len(c.RSI_PERIODS) + len(c.BB_PERIODS)
    meta = tf.Variable(np.random.default_rng(5).uniform(-1, 1, (B, n_logits)).astype(np.float32))
    layer([x, meta])  # build
    with tf.GradientTape(persistent=True) as tape:
        batched = layer._call_batched(tf.convert_to_tensor(x), meta)
        reference = layer._call_per_indicator(tf.convert_to_tensor(x), meta)
        lb, lr = tf.reduce_sum(tf.sin(batched)), tf.reduce_sum(tf.sin(reference))
    np.testing.assert_allclose(batched.numpy(), reference.numpy(), rtol=1e-4, atol=1e-4)
    targets = [x, meta] + layer.get_indicator_trainable_variables()
    for gb, gr in zip(tape.gradient(lb, targets), tape.gradient(lr, targets)):
        np.testing.assert_allclose(gb.numpy(), gr.numpy(), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------- NT-097: applied-period bounds


def test_applied_period_unbounded_by_default_can_leave_the_clip_range():
    """INDICATOR_BOUND_APPLIED off (default) reproduces today's behaviour: an extreme per-window
    meta shift can push the applied period outside [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX]
    (B_model_indicators.md 4.2: up to 74.7 bars above a ceiling of 60 on real runs)."""
    cfg = Config(**OLD, MOMENTUM_CLIP_MIN=2.0, MOMENTUM_CLIP_MAX=60.0)
    assert cfg.INDICATOR_BOUND_APPLIED is False
    layer = LearnableIndicators(cfg)
    x = _x()
    meta_zero = tf.zeros([B, 18])
    layer([x, meta_zero])  # build
    n_logits = 18
    extreme = tf.ones([B, n_logits]) * 10.0  # tanh saturates at +1, the largest possible shift
    samples = layer.applied_period_samples(extreme)
    # macd_1_fast inits at period 5 (logit-space floor-adjacent); a maximal positive shift on the
    # base logit pushes the applied period below the floor of 2 (period is decreasing in logit).
    assert any(np.any(v < cfg.MOMENTUM_CLIP_MIN) or np.any(v > cfg.MOMENTUM_CLIP_MAX)
              for v in samples.values()), "an extreme meta shift should leave the bound when unclipped"


def test_applied_period_bounded_stays_inside_the_clip_range():
    """INDICATOR_BOUND_APPLIED on: no applied period, on any window, can leave
    [MOMENTUM_CLIP_MIN, MOMENTUM_CLIP_MAX], however extreme the per-window meta shift."""
    cfg = Config(**OLD, MOMENTUM_CLIP_MIN=2.0, MOMENTUM_CLIP_MAX=60.0, INDICATOR_BOUND_APPLIED=True)
    layer = LearnableIndicators(cfg)
    x = _x()
    meta_zero = tf.zeros([B, 18])
    layer([x, meta_zero])  # build
    for shift in (-10.0, 10.0):  # tanh saturates well before +-10
        extreme = tf.ones([B, 18]) * shift
        samples = layer.applied_period_samples(extreme)
        for name, v in samples.items():
            assert np.all(v >= cfg.MOMENTUM_CLIP_MIN - 1e-4), (name, v.min())
            assert np.all(v <= cfg.MOMENTUM_CLIP_MAX + 1e-4), (name, v.max())


def test_applied_period_samples_matches_forward_pass_alpha():
    """applied_period_samples must use exactly the alpha the forward pass computes (not a copy)."""
    cfg = Config(**OLD)
    layer = LearnableIndicators(cfg)
    x = _x()
    meta = tf.random.normal([B, 18], seed=7) * 0.3
    layer([x, meta])  # build
    samples = layer.applied_period_samples(meta)
    # Recompute the first family's first parameter's alpha directly, the same way _alpha does.
    family, _insts, varmaps = layer._families[0]
    first_param = family.params[0].name
    logit = varmaps[0][first_param]
    alpha = layer._alpha(logit, meta, 0)
    expected_period = (2.0 / (alpha.numpy() + layer.epsilon)) - 1.0
    key = family.learned_name(0, first_param)
    np.testing.assert_allclose(samples[key], expected_period, rtol=1e-6, atol=1e-6)

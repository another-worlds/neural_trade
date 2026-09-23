"""LearnableIndicators: EWMA equivalence (matrix vs scan), output shape, gradient reach."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import neural_trade.utils.math as mh
from neural_trade.core.config import Config
from neural_trade.models.layers import LearnableIndicators

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
    cfg = Config()
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
        cfg = Config()
        cfg.EWMA_IMPL = impl
        tf.keras.utils.set_random_seed(0)
        layer = LearnableIndicators(cfg)
        outs.append(layer([_x(3), tf.random.normal([B, 18], seed=4) * 0.1]).numpy())
    # RSI divides two EWMAs; compare with a tolerance scaled to each feature's magnitude
    scale = np.maximum(np.abs(outs[0]).max(axis=(0, 1), keepdims=True), 1.0)
    np.testing.assert_allclose(outs[1] / scale, outs[0] / scale, atol=2e-4)

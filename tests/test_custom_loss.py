"""Loss-system regression tests at REALISTIC scales.

The pre-existing smoke test (tests/test_losses.py) calls custom_loss with
pred_scale=1 and last_close=1 - the one parameterisation under which the
trend-loss unit bug was invisible. With pred_scale~261 and last_close~110,000
the old extended_trend_loss overflowed cosh, pinned itself at a constant and
produced a NaN gradient that poisoned every weight on the first step.
Every test here fails on the pre-fix code.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from losses import _logcosh_safe

B = 64
SCALES = [
    pytest.param(1.0, 0.0, 1.0, id="unit-scales-legacy-smoke"),
    pytest.param(261.0, 3.2, 110_000.0, id="realistic-btc"),
    pytest.param(0.01, 0.0, 5.0, id="tiny-scale"),
]


# CustomTrainModel is functional in production; see the make_loss_model fixture for why
# the bare CustomTrainModel(base_model=None, ...) form is order-dependent and not used here.


def _batch(rng, last_close):
    x_window = tf.constant(rng.normal(0.0, 1.0, size=(B, 60)).astype(np.float32))
    y_true = tf.constant(rng.normal(0.0, 1.0, size=(B, 3)).astype(np.float32))  # scaled deltas
    lc = tf.constant((last_close + rng.normal(0.0, 0.005 * last_close, size=(B, 1))).astype(np.float32))
    ext = tf.constant(rng.normal(0.0, 200.0, size=(B, 3)).astype(np.float32))  # raw dollar deltas
    return x_window, y_true, lc, ext


def _heads(rng):
    price = [tf.Variable(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    dirs = [tf.Variable(rng.uniform(0.05, 0.95, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    var = [tf.Variable(rng.uniform(0.5, 2.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    return price, dirs, var


def _y_pred(price, dirs, var):
    return (price[0], dirs[0], var[0], price[1], dirs[1], var[1], price[2], dirs[2], var[2])


def test_logcosh_safe_large_argument():
    x = tf.Variable([500.0, -500.0, 0.1, 0.0], dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = _logcosh_safe(x)
    g = tape.gradient(y, x).numpy()
    y = y.numpy()
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(g))
    np.testing.assert_allclose(g[:2], [1.0, -1.0], atol=1e-6)  # tanh(+-500)
    np.testing.assert_allclose(y[2], np.log(np.cosh(0.1)), atol=1e-6)  # exact where cosh is finite
    np.testing.assert_allclose(y[:2], 500.0 - np.log(2.0), atol=1e-3)  # |x| - log 2 asymptote
    assert abs(float(y[3])) < 1e-6


@pytest.mark.parametrize("pred_scale,pred_mean,last_close", SCALES)
def test_all_34_components_finite_with_finite_gradients(make_loss_model, pred_scale, pred_mean, last_close):
    rng = np.random.default_rng(0)
    m = make_loss_model(pred_scale, pred_mean)
    x, y, lc, ext = _batch(rng, last_close)
    price, dirs, var = _heads(rng)

    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
        total = out[0]

    vals = np.array([float(t) for t in out])
    assert vals.shape == (34,), "LossComponents contract is 34 fields"
    bad = [f for f, v in zip(out._fields, vals) if not np.isfinite(v)]
    assert not bad, f"non-finite components: {bad}"
    assert vals[0] > 0.0

    grads = tape.gradient(total, price + dirs + var)
    names = ["price"] * 3 + ["dir"] * 3 + ["var"] * 3
    for name, g in zip(names, grads):
        assert g is not None, f"no gradient reached the {name} head"
        assert bool(tf.reduce_all(tf.math.is_finite(g))), f"non-finite gradient on the {name} head"
    for g in grads[:3] + grads[6:]:  # price and variance heads must be supervised
        assert float(tf.reduce_max(tf.abs(g))) > 0.0


def test_extended_trend_is_not_the_clipped_constant(make_loss_model):
    """At realistic scales the old term was pinned at 10.0 (times lambda) in every epoch."""
    rng = np.random.default_rng(1)
    m = make_loss_model(261.0, 3.2)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
    lam = float(m.lambda_extended_trend)
    for name in ("extended_h0", "extended_h1", "extended_h2"):
        v = float(getattr(out, name))
        assert 0.0 < v < 5.0, f"{name}={v}"
        assert abs(v - 10.0 * lam) > 1e-3 and abs(v - 1.333295) > 1e-3, f"{name} still looks clipped: {v}"
    for name in ("local_h0", "global_h0", "local_h1", "global_h1", "local_h2", "global_h2"):
        assert float(getattr(out, name)) == 0.0  # retired terms are exactly zero


def test_extended_trend_matches_numpy_reference_and_responds_to_the_head(make_loss_model):
    """ext_k = lambda * mean(logcosh(price_k - scaled(ext_raw[:, k]))) in scaled-delta units."""
    rng = np.random.default_rng(2)
    pred_scale, pred_mean = 261.0, 3.2
    m = make_loss_model(pred_scale, pred_mean)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)

    lam = float(m.lambda_extended_trend)
    ext_np = ext.numpy()
    for k, name in enumerate(("extended_h0", "extended_h1", "extended_h2")):
        p = price[k].numpy().reshape(-1)
        ref = lam * np.mean(np.log(np.cosh(p - (ext_np[:, k] - pred_mean) / (pred_scale + 1e-8))))
        np.testing.assert_allclose(float(getattr(out, name)), ref, rtol=1e-4, atol=1e-5)

    price[0].assign_add(tf.fill([B, 1], 0.5))  # the term must depend on the head it supervises
    out2 = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
    assert abs(float(out2.extended_h0) - float(out.extended_h0)) > 1e-4
    assert abs(float(out2.extended_h1) - float(out.extended_h1)) < 1e-6

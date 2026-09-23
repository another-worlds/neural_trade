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

from neural_trade.losses import _logcosh_safe

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


# ---------------------------------------------------------------------------- the former stubs
# (tests/test_model_math_consistency.py::TestConfigDeepMathStubs, now real)

def _components(m, rng_seed=1, var_value=None):
    rng = np.random.default_rng(rng_seed)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    if var_value is not None:
        var = [tf.Variable(np.full((B, 1), var_value, np.float32)) for _ in range(3)]
    return m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext), (x, y, price, var)


@pytest.mark.parametrize("weight, fields, outer", [
    ("lambda_crps", ("crps_h0", "crps_h1", "crps_h2"), None),
    ("lambda_soft_ece", ("soft_ece_h0", "soft_ece_h1", "soft_ece_h2"), None),
    ("lambda_var", ("nll_h0", "nll_h1", "nll_h2"), "lambda_nll_outer"),
    ("lambda_dir", ("dir_h0", "dir_h1", "dir_h2"), "lambda_dir_outer"),
])
def test_total_is_the_weighted_sum_of_its_components(make_loss_model, weight, fields, outer):
    """Changing one loss weight changes the total by exactly weight x (its returned components)."""
    m = make_loss_model()
    m.set_lambda_values(**{weight: 0.0})
    base, _ = _components(m)
    m.set_lambda_values(**{weight: 2.5})
    out, _ = _components(m)
    comp = sum(float(getattr(out, f)) for f in fields)
    scale = float(getattr(m, outer)) if outer else 1.0
    assert float(out.total) - float(base.total) == pytest.approx(2.5 * scale * comp, rel=1e-4, abs=1e-5)


def test_already_weighted_physics_fields_scale_with_their_weight(make_loss_model):
    m = make_loss_model()
    m.set_lambda_values(lambda_t_perp=0.1, lambda_hd=0.1)
    a, _ = _components(m)
    m.set_lambda_values(lambda_t_perp=0.3, lambda_hd=0.3)
    b, _ = _components(m)
    assert float(b.t_perp_total) == pytest.approx(3 * float(a.t_perp_total), rel=1e-5)
    assert float(b.hd_val) == pytest.approx(3 * float(a.hd_val), rel=1e-5)


def test_nll_is_exact_at_the_variance_floor(make_loss_model):
    """Variance below VAR_FLOOR is floored (not capped above) before the Gaussian NLL."""
    m = make_loss_model()
    out, (_, y, price, _) = _components(m, var_value=1e-6)
    floor = m.config.VAR_FLOOR + 1e-8
    for h in range(3):
        err = y.numpy()[:, h] - price[h].numpy()[:, 0]
        ref = np.mean(0.5 * (np.log(2 * np.pi) + np.log(floor)) + 0.5 * err ** 2 / floor)
        assert float(getattr(out, f"nll_h{h}")) == pytest.approx(ref, rel=1e-4)


def test_vacuum_bandwidth_term_is_zero_unless_lambda_vac_is_set(make_loss_model):
    from neural_trade.core.config import Config

    off, _ = _components(make_loss_model(config=Config(LAMBDA_VAC=0.0)))
    assert float(off.vac_val) == 0.0
    on, (_, _, price, _) = _components(make_loss_model(config=Config(LAMBDA_VAC=0.1)))
    spread = np.std(np.stack([p.numpy()[:, 0] for p in price], 1), axis=1)
    assert float(on.vac_val) == pytest.approx(np.mean(np.maximum(spread - 0.1, 0.0)), rel=1e-4)

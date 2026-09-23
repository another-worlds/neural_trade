"""M2 regression tests: every physics-inspired term is bounded and pulls the right way,
lambdas are live tf.Variables, ECE uses the positive-class convention, and period
clipping never writes a saturated logit.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from losses import (
    casimir_interference_loss,
    hyper_decoherence_coupling_loss,
    information_flow_entropy_loss,
    t_perp_calibration_loss,
    vacuum_overflow_t_perp_loss,
)
from model import Config, LearnableIndicators

B = 128


@pytest.fixture
def m(make_loss_model):
    # Built functionally, like production; see make_loss_model in conftest.py for why the
    # bare CustomTrainModel(base_model=None, ...) form is test-order dependent.
    return make_loss_model(261.0, 3.2)


def _heads(rng):
    out = []
    for _ in range(3):
        out += [
            tf.constant(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)),
            tf.constant(rng.uniform(0.05, 0.95, size=(B, 1)).astype(np.float32)),
            tf.constant(rng.uniform(0.5, 2.0, size=(B, 1)).astype(np.float32)),
        ]
    return tuple(out)


def test_lambdas_are_live_variables_outside_the_weight_set(m):
    assert isinstance(m.lambda_hd, tf.Variable)
    m.lambda_hd = 0.25
    assert np.isclose(float(m.lambda_hd), 0.25)
    m.set_lambda_values(lambda_hd=0.0, lambda_casimir=0.5)
    vals = m.get_lambda_values()
    assert vals["lambda_hd"] == 0.0 and vals["lambda_casimir"] == 0.5
    with pytest.raises(KeyError):
        m.set_lambda_values(lambda_bogus=1.0)
    assert not any(v is m.lambda_hd for v in m.variables), "loss weights must not enter the Keras weight set"


def test_lambda_assign_changes_the_loss_without_retracing(m):
    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(0.0, 1.0, size=(B, 60)).astype(np.float32))
    y = tf.constant(rng.normal(0.0, 1.0, size=(B, 3)).astype(np.float32))
    lc = tf.constant((110_000.0 + rng.normal(0.0, 500.0, size=(B, 1))).astype(np.float32))
    ext = tf.constant(rng.normal(0.0, 200.0, size=(B, 3)).astype(np.float32))
    heads = _heads(rng)
    traced = tf.function(lambda: m.custom_loss(x, y, heads, lc, ext))

    m.lambda_hd = 0.3
    first = float(traced().hd_val)
    m.lambda_hd = 0.0
    second = float(traced().hd_val)
    assert first != 0.0 and second == 0.0
    assert traced.experimental_get_tracing_count() == 1, "changing a lambda must not retrace"


def test_hd_is_bounded_and_rewards_variance_ordering(m):
    rng = np.random.default_rng(1)
    scale = rng.uniform(0.2, 3.0, size=(B, 1)).astype(np.float32)
    x = tf.constant(rng.normal(0.0, 1.0, size=(B, 60)).astype(np.float32) * scale)
    vol = tf.math.reduce_std(x, axis=1)
    v_good = tf.reshape(vol ** 2, (B, 1))
    v_bad = tf.reshape(1.0 / (vol ** 2), (B, 1))
    v_const = tf.ones((B, 1))
    good = float(hyper_decoherence_coupling_loss(m, x, v_good, v_good, v_good))
    bad = float(hyper_decoherence_coupling_loss(m, x, v_bad, v_bad, v_bad))
    const = float(hyper_decoherence_coupling_loss(m, x, v_const, v_const, v_const))
    assert 0.0 <= good < 0.1
    assert 1.9 < bad <= 2.0 + 1e-6
    assert abs(const - 1.0) < 1e-3


def test_casimir_is_bounded_and_sends_no_gradient_into_price_heads(m):
    rng = np.random.default_rng(2)
    p = [tf.Variable(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    v = [tf.Variable(rng.uniform(1e-4, 5.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    with tf.GradientTape() as tape:
        val = casimir_interference_loss(m, *p, *v)
    assert 0.0 <= float(val) <= np.log(1.0 / 1e-4) + 1e-6
    grads = tape.gradient(val, p)
    assert all(g is None or float(tf.reduce_max(tf.abs(g))) == 0.0 for g in grads)
    big = [tf.ones((B, 1)) * 2.0] * 3
    assert float(casimir_interference_loss(m, *p, *big)) == 0.0  # no penalty once var >= v_ref


def test_t_perp_is_zero_for_correct_heteroscedastic_variance(m):
    rng = np.random.default_rng(3)
    sigma = rng.uniform(0.3, 3.0, size=(B, 1)).astype(np.float32)
    mu = np.zeros((B, 1), np.float32)
    y = tf.constant(mu + sigma)  # residual energy == sigma^2 exactly, per sample
    var = tf.constant(sigma ** 2)
    assert float(t_perp_calibration_loss(m, y, tf.constant(mu), var)) < 1e-6
    for c in (4.0, 0.25):  # minimised at the correct scale; log-ratio makes both sides symmetric
        np.testing.assert_allclose(
            float(t_perp_calibration_loss(m, y, tf.constant(mu), c * var)), np.log(4.0) ** 2, rtol=1e-3
        )
    mu_v = tf.Variable(mu)
    with tf.GradientTape() as tape:
        val = t_perp_calibration_loss(m, y, mu_v, var)
    assert tape.gradient(val, mu_v) is None, "the residual statistic is a target; no gradient into mu"


def test_ife_is_a_linear_hinge_with_a_tight_bound(m):
    rng = np.random.default_rng(4)
    p = tf.constant(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32))
    identical = float(information_flow_entropy_loss(m, p, p, p, rho_max=0.5))
    np.testing.assert_allclose(identical, 2 * (1.0 - 0.5), rtol=1e-4)
    q = tf.constant(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32))
    r = tf.constant(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32))
    assert 0.0 <= float(information_flow_entropy_loss(m, p, q, r, rho_max=0.95)) <= 2 * 0.05 + 1e-6


def test_vacuum_overflow_sends_no_gradient_into_price_heads(m):
    rng = np.random.default_rng(5)
    ov = tf.constant(rng.uniform(0.0, 1.0, size=(B, 1)).astype(np.float32))
    ys = [tf.constant(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    ps = [tf.Variable(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    with tf.GradientTape() as tape:
        val = vacuum_overflow_t_perp_loss(m, ov, ys[0], ps[0], ys[1], ps[1], ys[2], ps[2])
    assert float(val) >= 0.0
    assert all(g is None for g in tape.gradient(val, ps))


def test_ece_uses_the_positive_class_convention(m):
    n = 4000
    rng = np.random.default_rng(6)
    p = tf.fill([n], 0.05)
    t = tf.constant((rng.uniform(size=n) < 0.05).astype(np.float32))
    mask = tf.ones([n])
    out = m._compute_direction_metrics(t, t, t, p, p, p, mask, mask, mask, prefix="")
    assert float(out["dir_ece_h0"]) < 0.03  # the old top-label mix reported ~0.90 here


def test_pit_ks_is_small_when_calibrated_and_large_when_not(m):
    rng = np.random.default_rng(7)
    n = 5000
    sigma = rng.uniform(0.5, 2.0, size=n).astype(np.float32)
    y = tf.constant((sigma * rng.normal(0.0, 1.0, size=n)).astype(np.float32))
    mu = tf.zeros([n])
    assert float(m._pit_ks(y, mu, tf.constant(sigma ** 2))) < 0.03
    assert float(m._pit_ks(y, mu, tf.constant((sigma / 2.0) ** 2))) > 0.10


def test_clip_learned_periods_never_saturates_the_logit():
    cfg = Config()
    layer = LearnableIndicators(cfg)
    layer.build([(None, cfg.LOOKBACK), (None, 18)])
    for v in layer.get_indicator_trainable_variables():
        v.assign(30.0)  # far past the period floor; the old round trip wrote logit ~ +18.4 here
    layer.clip_learned_periods(cfg.MOMENTUM_CLIP_MIN, cfg.MOMENTUM_CLIP_MAX)
    for v in layer.get_indicator_trainable_variables():
        logit = float(v)
        s = 1.0 / (1.0 + np.exp(-logit))
        assert s * (1.0 - s) > 0.1, f"logit {logit:.3f} is saturated"
        period = float(layer._period_from_logit(v))
        assert abs(period - cfg.MOMENTUM_CLIP_MIN) < 1e-3

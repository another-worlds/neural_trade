"""Production loss functions vs independent numpy references (plan C1).

Replaces the first part of the old test_loss_functions_exhaustive.py, which re-implemented each
loss in numpy and tested the copy (no project import - it could not fail when the production
code changed). Here every reference is written from the mathematical definition and compared
with the registered TensorFlow function on 5 seeds at rtol 1e-5.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from scipy.stats import norm

from neural_trade.losses import _logcosh_safe
from neural_trade.registries.losses import Losses

SEEDS = range(5)
MODEL = SimpleNamespace(config=SimpleNamespace(FOCAL_ALPHA=0.4, FOCAL_GAMMA=2.0))


def _col(a):
    return tf.constant(np.asarray(a, np.float32).reshape(-1, 1))


def _probs(rng, n=512):
    return rng.uniform(0.02, 0.98, n).astype(np.float32), (rng.uniform(size=n) < 0.5).astype(np.float32)


# ------------------------------------------------------------------ references
def ref_logcosh(x):
    x = np.asarray(x, np.float64)
    return np.abs(x) + np.log1p(np.exp(-2 * np.abs(x))) - np.log(2.0)   # = log(cosh(x)), overflow-free


def ref_focal(y, p, alpha, gamma):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    p_t = np.where(y == 1, p, 1 - p)
    weight = np.where(y == 1, 1 - alpha, alpha)
    return np.mean(weight * (1 - p_t) ** gamma * -np.log(p_t))


def ref_dice(y, p, smooth=1.0):
    return np.mean(1 - (2 * y * p + smooth) / (y + p + smooth + 1e-8))


def ref_crps(y, mu, sigma):
    z = (y - mu) / sigma
    return np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def ref_soft_ece(y, p, m, n_bins=10):
    h2 = 2 * (1 / (2 * n_bins)) ** 2
    total = m.sum()
    ece = 0.0
    for i in range(n_bins):
        w = np.exp(-((p - (i + 0.5) / n_bins) ** 2) / h2) * m
        ece += w.sum() / total * abs((w * y).sum() / w.sum() - (w * p).sum() / w.sum())
    return ece


def ref_t_perp(y, mu, var):
    return (np.log(np.mean((y - mu) ** 2)) - np.log(np.mean(var))) ** 2


# ------------------------------------------------------------------ equality with production
@pytest.mark.parametrize("seed", SEEDS)
def test_logcosh_and_point_loss(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 3, 400).astype(np.float32)
    np.testing.assert_allclose(_logcosh_safe(tf.constant(x)).numpy(), ref_logcosh(x), rtol=1e-5, atol=1e-6)
    y, yp = rng.normal(0, 1, 256), rng.normal(0, 1, 256)
    got = Losses.get("point_huber")(MODEL, _col(y), _col(yp)).numpy()
    np.testing.assert_allclose(got, np.mean(ref_logcosh(y.astype(np.float32) - yp.astype(np.float32))), rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_focal_dice_and_combined(seed):
    p, y = _probs(np.random.default_rng(seed))
    focal = Losses.get("focal_loss")(MODEL, tf.constant(y), tf.constant(p)).numpy()
    np.testing.assert_allclose(focal, ref_focal(y, p, 0.4, 2.0), rtol=1e-5)
    dice = Losses.get("dice_loss")(MODEL, tf.constant(y), tf.constant(p)).numpy()
    np.testing.assert_allclose(dice, ref_dice(y, p), rtol=1e-5)
    comb = Losses.get("combined_direction_loss")(MODEL, tf.constant(y), tf.constant(p), focal_weight=0.3,
                                                   dice_weight=0.7).numpy()
    np.testing.assert_allclose(comb, 0.3 * ref_focal(y, p, 0.4, 2.0) + 0.7 * ref_dice(y, p), rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_crps_matches_the_closed_form(seed):
    rng = np.random.default_rng(seed)
    y, mu = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
    var = rng.uniform(0.2, 3.0, 300)
    got = Losses.get("crps_gaussian_loss")(MODEL, _col(y), _col(mu), _col(var)).numpy()
    np.testing.assert_allclose(got, ref_crps(y, mu, np.sqrt(var)), rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_ece_and_t_perp(seed):
    rng = np.random.default_rng(seed)
    p, y = _probs(rng)
    m = (rng.uniform(size=p.size) < 0.8).astype(np.float32)
    got = Losses.get("soft_ece_loss")(MODEL, tf.constant(y), tf.constant(p), tf.constant(m)).numpy()
    np.testing.assert_allclose(got, ref_soft_ece(y, p, m), rtol=1e-5)
    yt, mu = rng.normal(0, 1, 256), rng.normal(0, 0.5, 256)
    var = rng.uniform(0.1, 2.0, 256)
    got = Losses.get("t_perp_calibration_loss")(MODEL, _col(yt), _col(mu), _col(var)).numpy()
    np.testing.assert_allclose(got, ref_t_perp(yt, mu, var), rtol=1e-4)


# ------------------------------------------------------------------ properties of the production functions
def test_logcosh_is_quadratic_near_zero_linear_far_away_and_overflow_free():
    # (float32 cancellation in x + softplus(-2x) - log 2 limits ABSOLUTE precision to ~1e-7)
    small = _logcosh_safe(tf.constant([0.05], tf.float32)).numpy()[0]
    assert small == pytest.approx(0.05 ** 2 / 2, rel=2e-3)
    far = _logcosh_safe(tf.constant([500.0, -500.0], tf.float32)).numpy()
    np.testing.assert_allclose(far, 500 - np.log(2), rtol=1e-6)
    with tf.GradientTape() as tape:
        x = tf.constant([500.0])
        tape.watch(x)
        y = _logcosh_safe(x)
    assert float(tape.gradient(y, x)[0]) == pytest.approx(1.0)


def test_focal_with_gamma_zero_is_class_weighted_cross_entropy():
    p, y = _probs(np.random.default_rng(9))
    got = Losses.get("focal_loss")(MODEL, tf.constant(y), tf.constant(p), gamma=0.0).numpy()
    ce = -np.where(y == 1, 0.6 * np.log(p), 0.4 * np.log(1 - p))
    np.testing.assert_allclose(got, ce.mean(), rtol=1e-5)


def test_focal_down_weights_easy_examples():
    easy = Losses.get("focal_loss")(MODEL, tf.constant([1.0]), tf.constant([0.95])).numpy()
    hard = Losses.get("focal_loss")(MODEL, tf.constant([1.0]), tf.constant([0.30])).numpy()
    ce_ratio = np.log(0.30) / np.log(0.95)
    assert hard / easy > ce_ratio * 10   # focal separates them far more than plain CE


def test_dice_is_in_zero_one_and_zero_for_a_perfect_positive():
    p, y = _probs(np.random.default_rng(3))
    d = Losses.get("dice_loss")(MODEL, tf.constant(y), tf.constant(p), reduce=False).numpy()
    assert (d >= 0).all() and (d <= 1).all()
    assert Losses.get("dice_loss")(MODEL, tf.constant([1.0]), tf.constant([1.0])).numpy() == pytest.approx(0.0, abs=1e-6)


def test_crps_is_proper_the_true_distribution_scores_best():
    rng = np.random.default_rng(4)
    y = rng.normal(0.3, 1.2, 20_000)
    crps = Losses.get("crps_gaussian_loss")
    scores = {(mu, s): float(crps(MODEL, _col(y), _col(np.full_like(y, mu)), _col(np.full_like(y, s ** 2))))
              for mu in (0.0, 0.3, 0.6) for s in (0.8, 1.2, 1.6)}
    assert min(scores, key=scores.get) == (0.3, 1.2)


def test_soft_ece_is_near_zero_for_calibrated_probabilities():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 50_000).astype(np.float32)
    y = (rng.uniform(size=p.size) < p).astype(np.float32)
    ece = Losses.get("soft_ece_loss")(MODEL, tf.constant(y), tf.constant(p), tf.ones_like(tf.constant(p))).numpy()
    assert ece < 0.02
    # ...and a constant prediction at the base rate is (almost) free: the term cannot tell a
    # collapsed head from a skilful one.
    const = np.full_like(p, y.mean())
    ece_const = Losses.get("soft_ece_loss")(MODEL, tf.constant(y), tf.constant(const),
                                            tf.ones_like(tf.constant(p))).numpy()
    assert ece_const < 0.01


# ------------------------------------------------------------------ why the direction default is BCE
def _expected(loss_name, q, grid):
    fn = Losses.get(loss_name)
    ones, zeros = tf.ones(len(grid)), tf.zeros(len(grid))
    p = tf.constant(grid, tf.float32)
    return q * fn(MODEL, ones, p, reduce=False).numpy() + (1 - q) * fn(MODEL, zeros, p, reduce=False).numpy()


@pytest.mark.parametrize("q", [0.3, 0.5, 0.55, 0.7])
def test_bce_is_proper_its_expected_loss_is_minimised_at_the_true_rate(q):
    grid = np.linspace(0.01, 0.99, 981)
    assert grid[np.argmin(_expected("binary_cross_entropy_loss", q, grid))] == pytest.approx(q, abs=1e-3)


@pytest.mark.parametrize("q", [0.45, 0.5, 0.55])
def test_dice_is_improper_its_optimum_is_an_extreme(q):
    """The legacy focal+dice objective's dice half drives a weak-signal head to 0 or 1."""
    grid = np.linspace(0.0, 1.0, 1001)
    best = grid[np.argmin(_expected("dice_loss", q, grid))]
    assert best in (0.0, 1.0)


def test_direction_loss_option_switches_the_objective(make_loss_model):
    from neural_trade.core.config import Config

    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(size=(64, 60)).astype(np.float32))
    y = tf.constant(rng.normal(size=(64, 3)).astype(np.float32))
    lc = tf.constant(np.full((64, 1), 110_000.0, np.float32))
    ext = tf.constant(rng.normal(0, 200, (64, 3)).astype(np.float32))
    heads = []
    for _ in range(3):
        heads += [tf.constant(rng.normal(size=(64, 1)).astype(np.float32)),
                  tf.constant(rng.uniform(0.1, 0.9, (64, 1)).astype(np.float32)),
                  tf.constant(rng.uniform(0.5, 2, (64, 1)).astype(np.float32))]
    bce = make_loss_model(config=Config(DIRECTION_LOSS="bce")).custom_loss(x, y, tuple(heads), lc, ext)
    legacy = make_loss_model(config=Config(DIRECTION_LOSS="focal_dice")).custom_loss(x, y, tuple(heads), lc, ext)
    assert float(bce.dir_h1) != pytest.approx(float(legacy.dir_h1))
    assert 0.3 < float(bce.dir_h1) < 1.5   # ~log 2 for uninformative probabilities

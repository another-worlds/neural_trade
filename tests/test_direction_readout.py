"""The Gaussian direction readout P(up | |delta| > d) and its stable log-CDF.

Regression: the previous readout Phi((mu - d) / sigma) ignored that labels mask the
neutral band, so with DIR_DEADBAND_BPS = 5 at BTC ~110k every probability sat near
Phi(-0.21) ~ 0.42, nothing was ever predicted "up" and val_gauss_dir_mcc was exactly
0.0 whatever the model learned - which made one M3 gate unreachable.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from scipy.special import log_ndtr as sp_log_ndtr
from scipy.stats import norm

from losses import gaussian_up_prob_given_move, log_ndtr

PRED_SCALE, PRED_MEAN, LC = 261.0, 3.2, 110_000.0
D = 5e-4  # 5 bps


def _readout(mu_scaled, var_scaled, lc=LC, d=D, pred_mean=PRED_MEAN, pred_scale=PRED_SCALE):
    mu = tf.constant(np.asarray(mu_scaled, np.float32))
    var = tf.constant(np.asarray(var_scaled, np.float32))
    lc_t = tf.fill(tf.shape(mu), np.float32(lc))
    return gaussian_up_prob_given_move(mu, var, lc_t, d, pred_mean, pred_scale).numpy()


def test_log_ndtr_matches_scipy_across_both_branches():
    x = np.concatenate([np.linspace(-60.0, -8.5, 60), np.linspace(-8.0, 8.0, 161)]).astype(np.float32)
    got = log_ndtr(tf.constant(x)).numpy()
    ref = sp_log_ndtr(x.astype(np.float64))
    assert np.all(np.isfinite(got)), "log_ndtr must never return -inf or NaN"
    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-6)


def test_log_ndtr_gradient_is_finite_everywhere():
    x = tf.Variable(np.array([-200.0, -30.0, -8.0001, -7.9999, -1.0, 0.0, 3.0, 12.0], np.float32))
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(log_ndtr(x))
    g = tape.gradient(y, x).numpy()
    assert np.all(np.isfinite(g))
    # d/dx log Phi(x) = phi(x) / Phi(x) > 0 and ~ -x deep in the lower tail
    assert np.all(g > 0)
    np.testing.assert_allclose(g[1], 30.0, rtol=2e-3)


def test_readout_is_one_half_at_zero_mean_for_any_sigma_and_band():
    mu_zero_raw_scaled = -PRED_MEAN / PRED_SCALE  # raw mean exactly 0
    for var in (1e-4, 1e-2, 1.0, 50.0):
        p = _readout([mu_zero_raw_scaled], [var])
        np.testing.assert_allclose(p, 0.5, atol=1e-6)


def test_readout_reduces_to_phi_without_a_band():
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 1, 50).astype(np.float32)
    var = rng.uniform(0.2, 3.0, 50).astype(np.float32)
    got = _readout(mu, var, d=0.0)
    mu_raw = mu * PRED_SCALE + PRED_MEAN
    sigma_raw = np.sqrt(var) * PRED_SCALE
    np.testing.assert_allclose(got, norm.cdf(mu_raw / sigma_raw), atol=2e-5)


def test_readout_is_antisymmetric_and_monotone_in_mu():
    mu = np.linspace(-3, 3, 41).astype(np.float32) - PRED_MEAN / PRED_SCALE
    p = _readout(mu, np.full_like(mu, 0.7))
    np.testing.assert_allclose(p + p[::-1], 1.0, atol=1e-5)
    assert np.all(np.diff(p) > 0)


def test_readout_matches_monte_carlo_conditional_frequency():
    rng = np.random.default_rng(1)
    for mu_s, var_s in ((0.3, 0.5), (-0.1, 2.0), (0.05, 0.02)):
        mu_raw = mu_s * PRED_SCALE + PRED_MEAN
        sig_raw = np.sqrt(var_s) * PRED_SCALE
        draws = rng.normal(mu_raw, sig_raw, 2_000_000)
        d_raw = D * LC
        moved = np.abs(draws) > d_raw
        freq = np.mean(draws[moved] > d_raw)
        np.testing.assert_allclose(_readout([mu_s], [var_s])[0], freq, atol=2e-3)


def test_readout_stays_finite_when_sigma_collapses_to_the_floor():
    # sigma_raw = 0.01 * 261 = $2.6 against a $55 band: both tails ~1e-98 in float32
    mu = np.array([-1.0, -0.2, -PRED_MEAN / PRED_SCALE, 0.2, 1.0], np.float32)
    p = _readout(mu, np.full_like(mu, 1e-4))
    assert np.all(np.isfinite(p))
    assert p[0] < 1e-6 and p[-1] > 1 - 1e-6 and abs(p[2] - 0.5) < 1e-6


def test_weak_but_correctly_signed_predictions_give_positive_gaussian_mcc(make_loss_model):
    """The exact M3 regression, at the magnitudes of the real 2-epoch run.

    Early in training the price head is small (pred_std ~ $20 in the run) against a $55
    band and sigma ~ $261. The old readout put every sample near Phi(-0.21) < 0.5, so the
    predicted-up rate was 0 and the Gaussian MCC exactly 0.0 however good the signal was.
    """
    m = make_loss_model(PRED_SCALE, PRED_MEAN)
    rng = np.random.default_rng(2)
    n = 6000
    y_raw = rng.normal(0.0, PRED_SCALE, n)
    mu_raw = 0.08 * y_raw  # weak, correctly signed: std ~ $21, like the real run
    mu_s = ((mu_raw - PRED_MEAN) / PRED_SCALE).astype(np.float32)
    var_s = np.full(n, 1.0, np.float32)
    lc = np.full(n, LC, np.float32)
    ret = y_raw / LC
    mask = (np.abs(ret) > D).astype(np.float32)
    true_dir = (ret > D).astype(np.float32)

    def mcc(p):
        t = tf.constant(true_dir)
        pt = tf.constant(np.asarray(p, np.float32))
        mk = tf.constant(mask)
        return float(m._compute_direction_metrics(t, t, t, pt, pt, pt, mk, mk, mk, prefix="")["dir_mcc_h1"])

    old = norm.cdf((mu_s - D * LC / PRED_SCALE) / np.sqrt(var_s))
    assert np.mean(old[mask > 0] > 0.5) < 0.01, "precondition: the old readout never says up"
    assert mcc(old) < 0.1  # exactly 0.0 in the real run; a few synthetic tail samples cross 0.5 here

    new = gaussian_up_prob_given_move(tf.constant(mu_s), tf.constant(var_s), tf.constant(lc), D,
                                      PRED_MEAN, PRED_SCALE).numpy()
    up_rate = float(np.mean(new[mask > 0] > 0.5))
    assert 0.4 < up_rate < 0.6
    assert mcc(new) > 0.5  # the sign of mu is right for every sample; only the readout was broken

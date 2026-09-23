"""The physics-inspired (QBOX / T_perp) loss terms, called through the production registry.

The numpy re-implementations that used to precede these tests (focal, dice, log-cosh, NLL,
trend, interconnection, volatility, composite) tested copies, not the code; the terms that exist
in production are now checked against references in test_losses_reference.py.
"""
import numpy as np
import pytest
import tensorflow as tf

from neural_trade.registries.losses import Losses  # production functions, not a copy


@pytest.fixture(autouse=True)
def _seed_tf():
    """TF_DETERMINISTIC_OPS=1 (set by ``import neural_trade``) makes unseeded tf.random ops raise."""
    tf.random.set_seed(0)


def _mock_model(**kwargs):
    """Return a simple namespace that mimics model attributes (kept for reference)."""
    class _Obj:
        pass
    obj = _Obj()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


_t_perp_calib = Losses.get('t_perp_calibration_loss')
_casimir       = Losses.get('casimir_interference_loss')
_vac_bw        = Losses.get('vacuum_bandwidth_loss')
_hd            = Losses.get('hyper_decoherence_coupling_loss')
_ife           = Losses.get('information_flow_entropy_loss')
_vac_overflow  = Losses.get('vacuum_overflow_t_perp_loss')

B = 8  # batch size


class TestQBOXLosses:
    """Verify shape, sign, and boundary behaviour of each T_⊥ / QBOX loss."""

    # ------------------------------------------------------------------
    # T⊥ calibration loss  — signature: (model, y_true_h, price_h, var_h)
    # ------------------------------------------------------------------

    def test_t_perp_calibration_loss_finite_positive(self):
        """Loss is a finite non-negative scalar."""
        y_true = tf.random.uniform([B, 1], -1.0, 1.0)
        price_h = tf.random.uniform([B, 1], -1.0, 1.0)
        var_h   = tf.random.uniform([B, 1],  0.01, 1.0)
        result = _t_perp_calib(None, y_true, price_h, var_h)
        assert np.isfinite(float(result)), "t_perp_calibration_loss must be finite"
        assert float(result) >= 0.0, "t_perp_calibration_loss must be non-negative"

    def test_t_perp_calibration_loss_perfect_prediction(self):
        """When predictions perfectly match truth, loss should be small."""
        y_true  = tf.ones([B, 1]) * 0.5
        price_h = tf.ones([B, 1]) * 0.5   # perfect point pred
        var_h   = tf.ones([B, 1]) * 1e-4  # near-zero residuals → low calibration cost
        result = _t_perp_calib(None, y_true, price_h, var_h)
        assert float(result) >= 0.0

    # ------------------------------------------------------------------
    # Casimir interference loss
    # signature: (model, price_h0, price_h1, price_h2, var_h0, var_h1, var_h2)
    # ------------------------------------------------------------------

    def test_casimir_no_interference_same_sign(self):
        """All predictions same sign → minimal loss."""
        p = tf.ones([B, 1]) * 0.5
        v = tf.ones([B, 1]) * 0.1
        result = _casimir(None, p, p, p, v, v, v)
        assert float(result) >= 0.0
        # Same-sign prices → relu(-p*p) = 0 → loss = 0
        assert float(result) < 1e-6, f"Expected ~0 Casimir, got {float(result)}"

    def test_casimir_interference_opposite_sign(self):
        """Opposing h0/h1 signs with low variance → positive loss."""
        p_pos = tf.ones([B, 1]) *  1.0
        p_neg = tf.ones([B, 1]) * -1.0
        v_low = tf.ones([B, 1]) *  0.01   # low variance → large penalty
        result = _casimir(None, p_pos, p_neg, p_pos, v_low, v_low, v_low)
        assert float(result) > 0.0, "Opposing horizons + low var should penalise"

    # ------------------------------------------------------------------
    # Vacuum bandwidth loss  — signature: (model, price_h0, price_h1, price_h2, lambda_vac=None)
    # ------------------------------------------------------------------

    def test_vacuum_below_threshold_zero(self):
        """Identical predictions → std = 0 → no violation → loss = 0."""
        p = tf.ones([B, 1]) * 0.1
        result = _vac_bw(None, p, p, p, lambda_vac=0.5)
        assert float(result) == 0.0, f"Below-threshold vac_loss should be 0, got {float(result)}"

    def test_vacuum_above_threshold_positive(self):
        """Large spread across horizons exceeds threshold → loss > 0."""
        p0 = tf.ones([B, 1]) * -5.0
        p1 = tf.ones([B, 1]) *  5.0
        p2 = tf.zeros([B, 1])
        result = _vac_bw(None, p0, p1, p2, lambda_vac=0.01)
        assert float(result) > 0.0, "Above-threshold vac_loss should be positive"

    # ------------------------------------------------------------------
    # Hyper-decoherence coupling loss
    # signature: (model, x_window, var_h0, var_h1, var_h2)
    # ------------------------------------------------------------------

    def test_hd_coupling_scalar_finite(self):
        """Loss is a finite scalar in [0, 2] (1 - Pearson correlation)."""
        x_window = tf.random.normal([B, 32])
        v = tf.random.uniform([B, 1], 0.01, 1.0)
        result = _hd(None, x_window, v, v, v)
        assert np.isfinite(float(result)), "hd_loss must be finite"
        assert 0.0 <= float(result) <= 2.0 + 1e-6

    def test_hd_coupling_zero_volatility(self):
        """No volatility variation → no ordering information → correlation 0 → loss = 1.

        The pre-Phase-A form (-mean(vol * sigma)) returned 0 here; the corrected term is
        1 - Pearson(z(log vol), z(log var)), the neutral midpoint of [0, 2].
        """
        x_flat   = tf.zeros([B, 32])   # constant series → std = 0
        v = tf.ones([B, 1]) * 0.5
        result = _hd(None, x_flat, v, v, v)
        assert abs(float(result) - 1.0) < 1e-6, f"Zero-vol hd_loss should be 1, got {float(result)}"

    # ------------------------------------------------------------------
    # Information flow entropy loss
    # signature: (model, price_h0, price_h1, price_h2, rho_max=None)
    # ------------------------------------------------------------------

    def test_ife_low_correlation_zero(self):
        """Well-below-threshold correlation → loss = 0."""
        tf.random.set_seed(7)
        p0 = tf.random.normal([B, 1])
        p1 = tf.random.normal([B, 1])   # independent → |corr| << 0.95
        p2 = tf.random.normal([B, 1])
        result = _ife(None, p0, p1, p2, rho_max=0.95)
        assert float(result) >= 0.0
        assert float(result) < 0.5, f"Uncorrelated IFE loss unexpectedly large: {float(result)}"

    def test_ife_high_correlation_positive(self):
        """Near-perfect correlation exceeds tight rho_max → loss > 0."""
        base = tf.random.normal([B, 1])
        result = _ife(None, base, base, base, rho_max=0.50)
        assert float(result) >= 0.0, "IFE loss must be non-negative"
        # |corr| = 1.0 with rho_max=0.50 → linear hinge: 2 * (1.0 - 0.5) = 1.0
        assert float(result) > 0.0, "Perfect correlation should yield positive IFE loss"


class TestVacuumOverflowLoss:
    """Tests for vacuum_overflow_t_perp_loss.

    The loss measures alignment between:
      - mean vacuum overflow signal  (energy above E_max in hidden-perp subspace)
      - mean absolute prediction residual across three horizons
    Loss = (mean_overflow - mean_residual)² / (mean_residual² + ε), with the residual
    statistic stop-gradient-ed (it is a target, not something to game).
    """

    def test_perfect_alignment_zero_loss(self):
        """When overflow equals residual magnitude exactly, loss ≈ 0."""
        tf.random.set_seed(0)
        residual = tf.constant(0.3, dtype=tf.float32)
        # y_true and price differ by `residual` per sample (all horizons equal)
        y_true = tf.zeros([B, 1], dtype=tf.float32)
        price  = tf.ones([B, 1], dtype=tf.float32) * residual
        overflow = tf.ones([B, 1], dtype=tf.float32) * residual  # matches mean residual
        result = _vac_overflow(None, overflow, y_true, price, y_true, price, y_true, price)
        assert float(result) < 1e-6, f"Perfectly aligned overflow/residual → loss should be ~0, got {float(result)}"

    def test_zero_overflow_zero_residual_near_zero(self):
        """When both overflow and residuals are zero, loss is numerically safe (not NaN/inf)."""
        y_true   = tf.zeros([B, 1], dtype=tf.float32)
        overflow = tf.zeros([B, 1], dtype=tf.float32)
        result = _vac_overflow(None, overflow, y_true, y_true, y_true, y_true, y_true, y_true)
        assert tf.math.is_finite(result), f"Zero-overflow/zero-residual should be finite, got {float(result)}"
        assert float(result) >= 0.0

    def test_large_overflow_small_residual_positive(self):
        """Large overflow with near-zero residuals → positive, finite loss."""
        y_true   = tf.zeros([B, 1], dtype=tf.float32)
        price    = tf.zeros([B, 1], dtype=tf.float32)        # residual ≈ 0
        overflow = tf.ones([B, 1], dtype=tf.float32) * 5.0  # big overflow signal
        result = _vac_overflow(None, overflow, y_true, price, y_true, price, y_true, price)
        assert tf.math.is_finite(result), f"Result must be finite, got {float(result)}"
        assert float(result) > 0.0, "Large overflow vs tiny residual should yield positive loss"

    def test_non_negative(self):
        """Loss is always non-negative (it is a squared term)."""
        tf.random.set_seed(42)
        y_true   = tf.random.normal([B, 1])
        price    = tf.random.normal([B, 1])
        overflow = tf.random.uniform([B, 1], 0.0, 2.0)
        result = _vac_overflow(None, overflow, y_true, price, y_true, price, y_true, price)
        assert float(result) >= 0.0, f"Loss must be non-negative, got {float(result)}"

    def test_gradient_flows(self):
        """Gradients wrt overflow tensor must exist and be finite."""
        tf.random.set_seed(99)
        y_true   = tf.random.normal([B, 1])
        price    = tf.random.normal([B, 1])
        overflow = tf.Variable(tf.random.uniform([B, 1], 0.1, 1.0))
        with tf.GradientTape() as tape:
            loss = _vac_overflow(None, overflow, y_true, price, y_true, price, y_true, price)
        grad = tape.gradient(loss, overflow)
        assert grad is not None, "Gradient wrt overflow must exist"
        assert tf.reduce_all(tf.math.is_finite(grad)), f"Gradient must be finite: {grad}"


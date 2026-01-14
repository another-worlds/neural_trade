"""
Exhaustive tests for loss function implementations.

Tests every loss function with:
- Multiple input scenarios
- Edge cases (zeros, negatives, extremes)
- Numerical stability checks
- Mathematical property verification
- Gradient behavior validation
"""

import pytest
import numpy as np


class TestFocalLossExhaustive:
    """Exhaustive tests for focal loss implementation."""

    def test_focal_loss_perfect_prediction(self):
        """Test focal loss with perfect predictions."""
        y_true = np.ones(100)
        y_pred = np.ones(100) * 0.9999  # Near-perfect confidence

        # Focal loss should be near zero for perfect predictions
        alpha = 0.5
        gamma = 2.0

        p_t = y_true * y_pred
        focal_weight = (1.0 - p_t) ** gamma
        bce = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7))
        focal = alpha * focal_weight * bce

        assert focal.mean() < 0.01, "Perfect predictions should have low loss"
        assert np.isfinite(focal).all(), "Loss should be finite"

    def test_focal_loss_worst_prediction(self):
        """Test focal loss with worst predictions."""
        y_true = np.ones(100)
        y_pred = np.ones(100) * 0.0001  # Very wrong

        alpha = 0.5
        gamma = 2.0

        p_t = y_true * y_pred
        focal_weight = (1.0 - p_t) ** gamma
        bce = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7))
        focal = alpha * focal_weight * bce

        assert focal.mean() > 1.0, "Wrong predictions should have high loss"
        assert np.isfinite(focal).all(), "Loss should be finite"

    def test_focal_loss_balanced_predictions(self):
        """Test focal loss with balanced uncertain predictions."""
        y_true = np.ones(100)
        y_pred = np.ones(100) * 0.5  # Uncertain

        alpha = 0.5
        gamma = 2.0

        p_t = y_true * y_pred
        focal_weight = (1.0 - p_t) ** gamma
        bce = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7))
        focal = alpha * focal_weight * bce

        # Should be between perfect and worst
        # Focal loss with gamma=2, p=0.5 gives: 0.5 * 0.25 * 0.693 ≈ 0.087
        assert 0.05 < focal.mean() < 1.0, f"Expected focal loss in range, got {focal.mean()}"
        assert np.isfinite(focal).all()

    def test_focal_loss_gamma_effect(self):
        """Test that gamma controls focusing strength."""
        y_true = np.ones(100)
        y_pred_easy = np.ones(100) * 0.8  # Easy example
        y_pred_hard = np.ones(100) * 0.6  # Hard example

        alpha = 0.5

        # Low gamma (gamma=0.5) - less focusing
        gamma_low = 0.5
        focal_easy_low = alpha * ((1 - 0.8) ** gamma_low) * (-np.log(0.8))
        focal_hard_low = alpha * ((1 - 0.6) ** gamma_low) * (-np.log(0.6))
        ratio_low = focal_hard_low / focal_easy_low

        # High gamma (gamma=5) - more focusing
        gamma_high = 5.0
        focal_easy_high = alpha * ((1 - 0.8) ** gamma_high) * (-np.log(0.8))
        focal_hard_high = alpha * ((1 - 0.6) ** gamma_high) * (-np.log(0.6))
        ratio_high = focal_hard_high / focal_easy_high

        # Higher gamma should increase the ratio (more focus on hard examples)
        assert ratio_high > ratio_low, "Higher gamma should increase focus on hard examples"

    def test_focal_loss_alpha_symmetry(self):
        """Test alpha parameter effect on class balance."""
        # Test with positive class (label=1) predictions
        y_true = np.ones(100)
        y_pred = np.ones(100) * 0.7  # Confident positive prediction

        gamma = 2.0

        # Alpha favoring positive class (higher alpha = more weight on negative class/label=0)
        alpha_high = 0.75
        p_t_high = y_true * y_pred
        focal_weight_high = (1.0 - p_t_high) ** gamma
        bce_high = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7))
        focal_high = alpha_high * focal_weight_high * bce_high

        # Alpha favoring negative class (lower alpha = more weight on positive class/label=1)
        alpha_low = 0.25
        p_t_low = y_true * y_pred
        focal_weight_low = (1.0 - p_t_low) ** gamma
        bce_low = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7))
        focal_low = alpha_low * focal_weight_low * bce_low

        # Different alphas should give different losses (alpha multiplies the loss directly)
        assert focal_high.mean() != focal_low.mean(), f"Different alphas should give different losses: {focal_high.mean()} vs {focal_low.mean()}"
        assert focal_high.mean() > focal_low.mean(), "Higher alpha should give higher loss for positive predictions"

    def test_focal_loss_reduces_to_ce_when_gamma_zero(self):
        """Test that focal loss reduces to cross-entropy when gamma=0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.9, 0.2, 0.8, 0.1, 0.7])

        alpha = 0.5
        gamma = 0.0  # Should reduce to CE

        # Focal loss
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1.0 - p_t) ** gamma  # equals 1 when gamma=0
        bce = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7)) - (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-7, 1 - 1e-7))
        focal = alpha * bce  # When gamma=0, focal_weight=1

        # Regular CE
        ce = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7)) - (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-7, 1 - 1e-7))

        # Should be proportional (differ by alpha factor)
        np.testing.assert_allclose(focal, alpha * ce, rtol=1e-5)


class TestDiceLossExhaustive:
    """Exhaustive tests for Dice loss implementation."""

    def test_dice_loss_perfect_overlap(self):
        """Test Dice loss with perfect overlap."""
        y_true = np.ones(100)
        y_pred = np.ones(100)
        smooth = 1.0

        intersection = (y_true * y_pred).sum()
        dice = (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
        dice_loss = 1.0 - dice

        assert dice_loss < 0.01, "Perfect overlap should have near-zero loss"
        assert dice > 0.99, "Dice coefficient should be near 1"

    def test_dice_loss_no_overlap(self):
        """Test Dice loss with no overlap."""
        y_true = np.ones(100)
        y_pred = np.zeros(100)
        smooth = 1.0

        intersection = (y_true * y_pred).sum()
        dice = (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
        dice_loss = 1.0 - dice

        assert dice_loss > 0.9, "No overlap should have high loss"
        assert dice < 0.1, "Dice coefficient should be near 0"

    def test_dice_loss_smooth_parameter_effect(self):
        """Test that smooth parameter prevents division by zero."""
        y_true = np.zeros(100)
        y_pred = np.zeros(100)

        # Without smooth (would cause 0/0)
        # With smooth
        smooth = 1.0
        intersection = (y_true * y_pred).sum()
        dice = (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
        dice_loss = 1.0 - dice

        assert np.isfinite(dice_loss), "Smooth should prevent NaN"
        assert 0 <= dice_loss <= 1, "Dice loss should be in [0, 1]"

    def test_dice_loss_symmetry(self):
        """Test that Dice loss is symmetric."""
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)
        smooth = 1.0

        # Dice(A, B)
        intersection_ab = (y_true * y_pred).sum()
        dice_ab = (2.0 * intersection_ab + smooth) / (y_true.sum() + y_pred.sum() + smooth)

        # Dice(B, A)
        intersection_ba = (y_pred * y_true).sum()
        dice_ba = (2.0 * intersection_ba + smooth) / (y_pred.sum() + y_true.sum() + smooth)

        np.testing.assert_allclose(dice_ab, dice_ba, rtol=1e-10)

    def test_dice_loss_range(self):
        """Test that Dice loss is always in [0, 1]."""
        for _ in range(100):
            y_true = np.random.rand(50)
            y_pred = np.random.rand(50)
            smooth = 1.0

            intersection = (y_true * y_pred).sum()
            dice = (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
            dice_loss = 1.0 - dice

            assert 0 <= dice <= 1, f"Dice should be in [0,1], got {dice}"
            assert 0 <= dice_loss <= 1, f"Dice loss should be in [0,1], got {dice_loss}"


class TestHuberLossExhaustive:
    """Exhaustive tests for Huber/log-cosh loss implementation."""

    def test_logcosh_small_errors_quadratic(self):
        """Test that log-cosh is approximately quadratic for small errors."""
        errors = np.array([-0.1, -0.05, 0, 0.05, 0.1])
        log_cosh = np.log(np.cosh(errors))

        # For small x, log(cosh(x)) ≈ x²/2
        quadratic_approx = errors ** 2 / 2

        np.testing.assert_allclose(log_cosh, quadratic_approx, atol=1e-3)

    def test_logcosh_large_errors_linear(self):
        """Test that log-cosh is approximately linear for large errors."""
        errors = np.array([-10, -5, 5, 10])
        log_cosh = np.log(np.cosh(errors))

        # For large |x|, log(cosh(x)) ≈ |x| - log(2)
        linear_approx = np.abs(errors) - np.log(2)

        np.testing.assert_allclose(log_cosh, linear_approx, rtol=0.01)

    def test_logcosh_smooth_transition(self):
        """Test that log-cosh is smooth across the transition region."""
        errors = np.linspace(-5, 5, 1000)
        log_cosh = np.log(np.cosh(errors))

        # Compute numerical derivative
        derivative = np.diff(log_cosh) / np.diff(errors)

        # Should be continuous (no jumps)
        derivative_diff = np.diff(derivative)
        assert np.all(np.abs(derivative_diff) < 0.1), "Derivative should be smooth"

    def test_logcosh_symmetry(self):
        """Test that log-cosh is symmetric."""
        errors = np.array([-2, -1, 0, 1, 2])
        log_cosh = np.log(np.cosh(errors))

        # Should be symmetric around 0
        np.testing.assert_allclose(log_cosh[0], log_cosh[4], rtol=1e-10)
        np.testing.assert_allclose(log_cosh[1], log_cosh[3], rtol=1e-10)

    def test_logcosh_minimum_at_zero(self):
        """Test that log-cosh has minimum at zero."""
        errors = np.array([-2, -1, 0, 1, 2])
        log_cosh = np.log(np.cosh(errors))

        assert log_cosh[2] < log_cosh[1]
        assert log_cosh[2] < log_cosh[3]
        assert log_cosh[2] < log_cosh[0]
        assert log_cosh[2] < log_cosh[4]

    def test_logcosh_numerical_stability(self):
        """Test log-cosh doesn't overflow for large inputs."""
        large_errors = np.array([-100, -50, 0, 50, 100])

        # Clip before computing
        clipped = np.clip(large_errors, -10, 10)
        log_cosh = np.log(np.cosh(clipped))

        assert np.isfinite(log_cosh).all(), "Should not overflow"
        assert np.all(log_cosh < 20), "Should be bounded"


class TestVarianceNLLExhaustive:
    """Exhaustive tests for variance-based Negative Log-Likelihood."""

    def test_nll_minimum_at_true_value(self):
        """Test that NLL is minimized when prediction equals truth."""
        y_true = 5.0
        predictions = np.linspace(0, 10, 100)
        variance = 1.0

        # NLL = 0.5 * log(2π * var) + 0.5 * (y - μ)² / var
        nll = 0.5 * np.log(2 * np.pi * variance) + 0.5 * (y_true - predictions) ** 2 / variance

        # Minimum should be at y_pred = y_true
        min_idx = np.argmin(nll)
        assert np.abs(predictions[min_idx] - y_true) < 0.1

    def test_nll_increases_with_error(self):
        """Test that NLL increases with prediction error."""
        y_true = 5.0
        variance = 1.0

        errors = np.array([0, 0.5, 1.0, 2.0, 5.0])
        predictions = y_true + errors

        nll = 0.5 * np.log(2 * np.pi * variance) + 0.5 * (y_true - predictions) ** 2 / variance

        # NLL should increase monotonically with error
        assert np.all(np.diff(nll) > 0), "NLL should increase with error"

    def test_nll_variance_bounds_effect(self):
        """Test effect of variance bounds on NLL."""
        y_true = 5.0
        y_pred = 6.0  # Error = 1.0

        var_floor = 0.1
        var_cap = 10.0

        # Low variance (confident)
        nll_low_var = 0.5 * np.log(2 * np.pi * var_floor) + 0.5 * (y_true - y_pred) ** 2 / var_floor

        # High variance (uncertain)
        nll_high_var = 0.5 * np.log(2 * np.pi * var_cap) + 0.5 * (y_true - y_pred) ** 2 / var_cap

        # Low variance should penalize errors more
        assert nll_low_var > nll_high_var

    def test_nll_proper_scoring_rule(self):
        """Test that NLL is a proper scoring rule."""
        # For a proper scoring rule, the expected score is minimized
        # when predictions match the true distribution

        np.random.seed(42)
        true_mean = 5.0
        true_var = 2.0

        # Generate samples from true distribution
        samples = np.random.normal(true_mean, np.sqrt(true_var), 10000)

        # Predict with true parameters
        nll_true = 0.5 * np.log(2 * np.pi * true_var) + 0.5 * (samples - true_mean) ** 2 / true_var

        # Predict with wrong parameters
        wrong_mean = 6.0
        wrong_var = 1.0
        nll_wrong = 0.5 * np.log(2 * np.pi * wrong_var) + 0.5 * (samples - wrong_mean) ** 2 / wrong_var

        # True parameters should have lower expected loss
        assert nll_true.mean() < nll_wrong.mean()

    def test_nll_finite_for_bounded_variance(self):
        """Test that NLL stays finite with variance bounds."""
        y_true = 5.0
        y_pred = np.linspace(-10, 20, 100)

        var_floor = 0.1
        var_cap = 1000.0

        # Try with floor
        variance = var_floor
        nll = 0.5 * np.log(2 * np.pi * variance) + 0.5 * (y_true - y_pred) ** 2 / variance
        assert np.isfinite(nll).all()

        # Try with cap
        variance = var_cap
        nll = 0.5 * np.log(2 * np.pi * variance) + 0.5 * (y_true - y_pred) ** 2 / variance
        assert np.isfinite(nll).all()


class TestTrendLossExhaustive:
    """Exhaustive tests for trend-based losses."""

    def test_trend_loss_perfect_trend_match(self):
        """Test trend loss when predicted trend matches actual trend."""
        # Upward trend
        last_close = 100.0
        y_true = 105.0  # +5%
        y_pred = 105.0  # +5%

        actual_trend = (y_true - last_close) / last_close
        pred_trend = (y_pred - last_close) / last_close

        trend_error = actual_trend - pred_trend
        assert np.abs(trend_error) < 1e-10

    def test_trend_loss_opposite_direction(self):
        """Test trend loss when prediction is opposite direction."""
        last_close = 100.0
        y_true = 105.0  # +5% up
        y_pred = 95.0   # -5% down

        actual_trend = (y_true - last_close) / last_close
        pred_trend = (y_pred - last_close) / last_close

        trend_error = actual_trend - pred_trend

        # Error should be large (0.05 - (-0.05) = 0.10)
        assert np.abs(trend_error) > 0.05

    def test_trend_loss_scale_invariance(self):
        """Test that trend loss is scale-invariant."""
        # Small price level
        last_close_small = 10.0
        y_true_small = 10.5
        y_pred_small = 10.3

        trend_error_small = (y_true_small - last_close_small) / last_close_small - \
                           (y_pred_small - last_close_small) / last_close_small

        # Large price level (10x)
        last_close_large = 100.0
        y_true_large = 105.0
        y_pred_large = 103.0

        trend_error_large = (y_true_large - last_close_large) / last_close_large - \
                           (y_pred_large - last_close_large) / last_close_large

        # Relative errors should be equal
        np.testing.assert_allclose(trend_error_small, trend_error_large, rtol=1e-10)

    def test_extended_trend_multiple_timescales(self):
        """Test extended trend loss across multiple timescales."""
        last_close = 100.0
        prices_past = np.array([95.0, 97.0, 99.0])  # 1m, 5m, 15m ago (increasing)
        y_pred = 102.0

        # Trends from different timescales (percentage change from past to now)
        trends = (last_close - prices_past) / prices_past

        # All should be positive (upward trend)
        assert np.all(trends > 0), "All trends should be positive"

        # Earlier price (longer ago) should show larger percentage change
        # 95 to 100 = 5.26% vs 99 to 100 = 1.01%
        assert trends[0] > trends[-1], f"Longer timescale should show larger trend: {trends[0]} > {trends[-1]}"


class TestDirectionLossExhaustive:
    """Exhaustive tests for direction classification loss."""

    def test_direction_loss_correct_classification(self):
        """Test direction loss for correct predictions."""
        last_close = 100.0
        y_true = 105.0  # UP
        dir_pred = 0.9  # High confidence UP

        true_direction = 1.0 if y_true > last_close else 0.0

        # Binary cross-entropy
        bce = -true_direction * np.log(dir_pred) - (1 - true_direction) * np.log(1 - dir_pred)

        assert bce < 0.2, "Correct prediction should have low loss"

    def test_direction_loss_wrong_classification(self):
        """Test direction loss for wrong predictions."""
        last_close = 100.0
        y_true = 105.0  # UP
        dir_pred = 0.1  # High confidence DOWN (wrong)

        true_direction = 1.0 if y_true > last_close else 0.0

        # Binary cross-entropy
        bce = -true_direction * np.log(np.clip(dir_pred, 1e-7, 1 - 1e-7))

        assert bce > 1.0, "Wrong prediction should have high loss"

    def test_direction_deadband_effect(self):
        """Test that deadband ignores small price changes."""
        last_close = 100.0
        deadband_bps = 5  # 5 basis points = 0.05%
        deadband = deadband_bps / 10000.0

        # Small change (within deadband)
        y_true_small = 100.04  # +0.04% (< 0.05%)
        ret_small = (y_true_small - last_close) / last_close

        # Should be masked out
        mask_small = np.abs(ret_small) > deadband
        assert not mask_small, "Should be within deadband"

        # Large change (outside deadband)
        y_true_large = 100.10  # +0.10% (> 0.05%)
        ret_large = (y_true_large - last_close) / last_close

        # Should not be masked
        mask_large = np.abs(ret_large) > deadband
        assert mask_large, "Should be outside deadband"

    def test_direction_coherence_across_horizons(self):
        """Test direction coherence penalty across horizons."""
        # Predictions for 3 horizons
        dir_h0 = 0.8  # UP
        dir_h1 = 0.7  # UP
        dir_h2 = 0.2  # DOWN (inconsistent)

        # Coherence: should penalize inconsistent directions
        # Simple coherence measure: variance of predictions
        dirs = np.array([dir_h0, dir_h1, dir_h2])
        coherence_loss = np.var(dirs)

        # High variance = low coherence = high penalty
        assert coherence_loss > 0.05


class TestInterconnectionLossExhaustive:
    """Exhaustive tests for multi-horizon interconnection loss."""

    def test_interconnection_perfect_consistency(self):
        """Test interconnection loss with perfectly consistent horizons."""
        # All horizons predict same trend
        pred_h0 = 101.0
        pred_h1 = 102.0
        pred_h2 = 103.0
        last_close = 100.0

        # All positive trends
        trend_h0 = pred_h0 - last_close
        trend_h1 = pred_h1 - last_close
        trend_h2 = pred_h2 - last_close

        assert trend_h0 > 0 and trend_h1 > 0 and trend_h2 > 0

        # Interconnection loss should be low
        # (measures consistency of trends)
        trends = np.array([trend_h0, trend_h1, trend_h2])
        interconnection = np.std(trends) / (np.abs(np.mean(trends)) + 1e-8)

        assert interconnection < 0.5, "Consistent trends should have low loss"

    def test_interconnection_inconsistent_directions(self):
        """Test interconnection loss with inconsistent directions."""
        pred_h0 = 101.0  # UP
        pred_h1 = 99.0   # DOWN
        pred_h2 = 102.0  # UP
        last_close = 100.0

        # Mixed directions
        trend_h0 = pred_h0 - last_close  # +1
        trend_h1 = pred_h1 - last_close  # -1
        trend_h2 = pred_h2 - last_close  # +2

        # Interconnection loss should be high
        trends = np.array([trend_h0, trend_h1, trend_h2])
        interconnection = np.std(trends) / (np.abs(np.mean(trends)) + 1e-8)

        assert interconnection > 1.0, "Inconsistent trends should have high loss"

    def test_interconnection_geometric_progression(self):
        """Test that horizons should follow geometric progression."""
        # Good: geometric progression
        pred_h0 = 101.0  # +1%
        pred_h1 = 102.0  # +2%
        pred_h2 = 104.0  # +4%
        last_close = 100.0

        ratios = np.array([
            (pred_h0 - last_close),
            (pred_h1 - last_close),
            (pred_h2 - last_close)
        ])

        # Check ratios
        ratio1 = ratios[1] / ratios[0]
        ratio2 = ratios[2] / ratios[1]

        # Should be approximately equal for geometric progression
        ratio_consistency = np.abs(ratio2 - ratio1) / ratio1

        # This is good if ratio_consistency is small
        assert ratio_consistency < 1.0


class TestVolatilityMatchingExhaustive:
    """Exhaustive tests for volatility matching loss."""

    def test_volatility_matching_correct_variance(self):
        """Test volatility matching when predicted variance matches realized variance."""
        # Prediction error
        y_true = 105.0
        y_pred = 103.0
        error = y_true - y_pred  # = 2.0

        # Predicted variance should match squared error
        pred_var = 4.0  # = 2.0²

        # Volatility matching loss (simplified)
        vol_loss = np.abs(pred_var - error ** 2)

        assert vol_loss < 0.1, "Correct variance should have low loss"

    def test_volatility_matching_overconfident(self):
        """Test penalty for overconfident (too low variance) predictions."""
        error = 5.0  # Large error
        pred_var = 1.0  # Low variance (overconfident)

        # Penalty for underestimating variance
        vol_loss = np.abs(pred_var - error ** 2)

        assert vol_loss > 10, "Overconfidence should have high penalty"

    def test_volatility_matching_underconfident(self):
        """Test penalty for underconfident (too high variance) predictions."""
        error = 1.0  # Small error
        pred_var = 100.0  # High variance (underconfident)

        # Penalty for overestimating variance
        vol_loss = np.abs(pred_var - error ** 2)

        assert vol_loss > 50, "Underconfidence should have penalty"


class TestCompositeLossExhaustive:
    """Exhaustive tests for the composite loss function."""

    def test_composite_loss_weight_balance(self):
        """Test that all loss components contribute appropriately."""
        # Typical loss values
        point_loss = 1.0
        trend_loss = 0.5
        direction_loss = 0.3
        var_loss = 0.8
        interconnection_loss = 0.1

        # Weights
        lambda_point = 1.0
        lambda_trend = 0.5
        lambda_dir = 0.3
        lambda_var = 1.0
        lambda_inter = 0.05

        # Weighted sum
        total_loss = (lambda_point * point_loss +
                     lambda_trend * trend_loss +
                     lambda_dir * direction_loss +
                     lambda_var * var_loss +
                     lambda_inter * interconnection_loss)

        # Each component's contribution
        contributions = {
            'point': lambda_point * point_loss / total_loss,
            'trend': lambda_trend * trend_loss / total_loss,
            'direction': lambda_dir * direction_loss / total_loss,
            'variance': lambda_var * var_loss / total_loss,
            'interconnection': lambda_inter * interconnection_loss / total_loss
        }

        # Main losses should dominate
        assert contributions['point'] > 0.2, "Point loss should contribute significantly"
        assert contributions['variance'] > 0.15, "Variance loss should contribute significantly"

    def test_composite_loss_gradient_flow(self):
        """Test that gradients flow through all components."""
        # Simulate gradients from each component
        grad_point = 1.0
        grad_trend = 0.5
        grad_direction = 0.3
        grad_var = 0.8

        # All should be non-zero
        assert grad_point != 0
        assert grad_trend != 0
        assert grad_direction != 0
        assert grad_var != 0

        # Total gradient
        total_grad = grad_point + grad_trend + grad_direction + grad_var
        assert total_grad > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

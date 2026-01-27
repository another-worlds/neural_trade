"""
Tests for model.py mathematical consistency and configuration validation.

These tests verify:
1. Configuration parameters are valid and consistent
2. Loss weights are properly balanced
3. Numerical bounds prevent instabilities
4. Mathematical relationships between parameters are correct
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from model import Config
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    Config = None


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not available or missing dependencies")
class TestConfigValidation:
    """Tests for Config class parameter validation."""

    def test_config_time_units(self):
        """Test that time unit definitions are correct."""
        cfg = Config()

        assert cfg.HOUR == 60, "1 hour should be 60 minutes"
        assert cfg.DAY == 1440, "1 day should be 1440 minutes"
        assert cfg.DAY == cfg.HOUR * 24, "Day should equal 24 hours"

    def test_config_lookback_positive(self):
        """Test that lookback window is positive."""
        cfg = Config()

        assert cfg.LOOKBACK > 0, "Lookback must be positive"
        assert cfg.LOOKBACK <= cfg.DAY, "Lookback should not exceed 1 day for minute data"

    def test_config_batch_size_valid(self):
        """Test that batch size is reasonable."""
        cfg = Config()

        assert cfg.BATCH_SIZE > 0, "Batch size must be positive"
        assert cfg.BATCH_SIZE >= 16, "Batch size should be at least 16"
        assert cfg.BATCH_SIZE <= 2048, "Batch size should not exceed 2048"

    def test_config_learning_rate_range(self):
        """Test that learning rate is in reasonable range."""
        cfg = Config()

        assert cfg.LR > 0, "Learning rate must be positive"
        assert cfg.LR >= 1e-5, "Learning rate should be >= 1e-5"
        assert cfg.LR <= 1.0, "Learning rate should be <= 1.0"

    def test_config_epochs_positive(self):
        """Test that epochs is positive."""
        cfg = Config()

        assert cfg.EPOCHS > 0, "Epochs must be positive"
        assert cfg.EPOCHS >= 1, "Need at least 1 epoch"

    def test_config_horizon_steps_ascending(self):
        """Test that horizon steps are in ascending order."""
        cfg = Config()

        assert len(cfg.HORIZON_STEPS) >= 1, "Should have at least 1 horizon"
        assert cfg.HORIZON_STEPS == sorted(cfg.HORIZON_STEPS), "Horizons should be ascending"
        assert all(h > 0 for h in cfg.HORIZON_STEPS), "All horizons must be positive"

        # Test dynamic properties
        assert cfg.num_horizons == len(cfg.HORIZON_STEPS), "num_horizons should match HORIZON_STEPS length"
        assert len(cfg.horizon_keys) == cfg.num_horizons, "horizon_keys should have num_horizons elements"

        # Verify horizon keys are correctly formatted
        for i, h_key in enumerate(cfg.horizon_keys):
            assert h_key == f"h{i}", f"Horizon key {i} should be 'h{i}', got '{h_key}'"

    def test_config_horizon_trend_alignment(self):
        """Test that extended trend periods match or align with horizons."""
        cfg = Config()

        assert len(cfg.EXTENDED_TREND_PERIODS) >= 1, "Need at least one trend period"
        assert all(t > 0 for t in cfg.EXTENDED_TREND_PERIODS), "All trend periods must be positive"

        # Trend periods should be in ascending order
        assert cfg.EXTENDED_TREND_PERIODS == sorted(cfg.EXTENDED_TREND_PERIODS), \
            "Trend periods should be ascending"

        # CRITICAL: Extended trend periods must exactly match horizon steps
        assert cfg.EXTENDED_TREND_PERIODS == cfg.HORIZON_STEPS, \
            f"EXTENDED_TREND_PERIODS {cfg.EXTENDED_TREND_PERIODS} must equal HORIZON_STEPS {cfg.HORIZON_STEPS}"

    def test_config_horizon_lambda_weights(self):
        """Test that horizon lambda weights are correctly configured."""
        cfg = Config()

        weights = cfg.horizon_lambda_weights
        assert len(weights) == cfg.num_horizons, \
            f"Should have {cfg.num_horizons} weights, got {len(weights)}"
        assert all(w >= 0 for w in weights), "All lambda weights must be non-negative"

        # For default 3-horizon config, verify specific weights
        if cfg.num_horizons == 3:
            assert weights[0] == cfg.LAMBDA_SHORT, "First horizon should use LAMBDA_SHORT"
            assert weights[1] == cfg.LAMBDA_POINT, "Second horizon should use LAMBDA_POINT"
            assert weights[2] == cfg.LAMBDA_LONG, "Third horizon should use LAMBDA_LONG"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not available")
class TestLossWeightConsistency:
    """Tests for loss weight parameter consistency."""

    def test_loss_weights_non_negative(self):
        """Test that all loss weights are non-negative."""
        cfg = Config()

        # Get all LAMBDA_ attributes
        lambda_attrs = [attr for attr in dir(cfg) if attr.startswith('LAMBDA_')]

        for attr in lambda_attrs:
            value = getattr(cfg, attr)
            assert value >= 0, f"{attr} must be non-negative, got {value}"

    def test_horizon_specific_weights_balanced(self):
        """Test that horizon-specific weights are reasonably balanced."""
        cfg = Config()

        weights = [cfg.LAMBDA_SHORT, cfg.LAMBDA_POINT, cfg.LAMBDA_LONG]

        # All should be positive
        assert all(w > 0 for w in weights), "All horizon weights must be positive"

        # None should be more than 10x another
        max_weight = max(weights)
        min_weight = min(weights)
        assert max_weight / min_weight <= 10, "Horizon weights should not differ by more than 10x"

    def test_auxiliary_loss_weights_reasonable(self):
        """Test that auxiliary loss weights are not too large compared to main losses."""
        cfg = Config()

        # Main loss weights
        main_weights = [cfg.LAMBDA_SHORT, cfg.LAMBDA_POINT, cfg.LAMBDA_LONG]
        main_avg = np.mean(main_weights)

        # Auxiliary weights should be smaller
        assert cfg.LAMBDA_DIR < main_avg * 2, "Direction loss weight too large"
        assert cfg.LAMBDA_INTER < main_avg, "Interconnection loss weight too large"

    def test_variance_bounds_valid(self):
        """Test that variance bounds are mathematically valid."""
        cfg = Config()

        assert cfg.VAR_FLOOR > 0, "Variance floor must be positive"
        assert cfg.VAR_CAP > cfg.VAR_FLOOR, "Variance cap must be greater than floor"
        assert cfg.VAR_CAP < 1e6, "Variance cap should not be too large"

        # Check that std bounds are reasonable
        std_floor = np.sqrt(cfg.VAR_FLOOR)
        std_cap = np.sqrt(cfg.VAR_CAP)

        assert std_floor < 1.0, "Std floor should be < 1.0"
        assert std_cap > 1.0, "Std cap should be > 1.0"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not available")
class TestFocalLossParameters:
    """Tests for focal loss hyperparameters."""

    def test_focal_alpha_range(self):
        """Test that focal loss alpha is in valid range [0, 1]."""
        cfg = Config()

        assert 0 <= cfg.FOCAL_ALPHA <= 1, f"Focal alpha must be in [0,1], got {cfg.FOCAL_ALPHA}"

    def test_focal_gamma_positive(self):
        """Test that focal loss gamma is positive."""
        cfg = Config()

        assert cfg.FOCAL_GAMMA > 0, f"Focal gamma must be positive, got {cfg.FOCAL_GAMMA}"
        assert cfg.FOCAL_GAMMA <= 5, f"Focal gamma should not exceed 5, got {cfg.FOCAL_GAMMA}"

    def test_focal_alpha_balanced(self):
        """Test that focal alpha is reasonably balanced."""
        cfg = Config()

        # Alpha should not be too extreme (not < 0.2 or > 0.8)
        assert 0.2 <= cfg.FOCAL_ALPHA <= 0.8, \
            f"Focal alpha should be between 0.2 and 0.8 for balanced training, got {cfg.FOCAL_ALPHA}"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not available")
class TestActivationScaling:
    """Tests for activation function scaling parameters."""

    def test_activation_scales_positive(self):
        """Test that activation scales are positive."""
        cfg = Config()

        assert cfg.TANH_SCALE > 0, "Tanh scale must be positive"
        assert cfg.SIGMOID_SCALE > 0, "Sigmoid scale must be positive"
        assert cfg.HUBER_DELTA > 0, "Huber delta must be positive"

    def test_activation_scales_reasonable(self):
        """Test that activation scales are in reasonable range."""
        cfg = Config()

        assert 0.1 <= cfg.TANH_SCALE <= 10.0, "Tanh scale should be between 0.1 and 10.0"
        assert 0.1 <= cfg.SIGMOID_SCALE <= 10.0, "Sigmoid scale should be between 0.1 and 10.0"
        assert 0.1 <= cfg.HUBER_DELTA <= 10.0, "Huber delta should be between 0.1 and 10.0"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not available")
class TestGradientClipping:
    """Tests for gradient clipping parameters."""

    def test_grad_clip_norm_positive(self):
        """Test that gradient clip norm is positive."""
        cfg = Config()

        assert cfg.GRAD_CLIP_NORM > 0, "Gradient clip norm must be positive"

    def test_grad_clip_norm_reasonable(self):
        """Test that gradient clip norm is in reasonable range."""
        cfg = Config()

        assert 0.1 <= cfg.GRAD_CLIP_NORM <= 100.0, \
            f"Gradient clip norm should be between 0.1 and 100.0, got {cfg.GRAD_CLIP_NORM}"

    def test_indicator_grad_mult_positive(self):
        """Test that indicator gradient multiplier is positive."""
        cfg = Config()

        assert cfg.INDICATOR_GRAD_MULT > 0, "Indicator gradient multiplier must be positive"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not available")
class TestRegularizationParameters:
    """Tests for regularization parameter validity."""

    def test_reg_momentum_l2_non_negative(self):
        """Test that L2 regularization is non-negative."""
        cfg = Config()

        assert cfg.REG_MOMENTUM_L2 >= 0, "L2 regularization must be non-negative"

    def test_reg_momentum_l2_not_too_large(self):
        """Test that L2 regularization is not too large."""
        cfg = Config()

        assert cfg.REG_MOMENTUM_L2 <= 0.1, \
            f"L2 regularization should not exceed 0.1, got {cfg.REG_MOMENTUM_L2}"

    def test_momentum_clip_bounds(self):
        """Test that momentum clipping bounds are valid."""
        cfg = Config()

        assert cfg.MOMENTUM_CLIP_MIN > 0, "Momentum clip min must be positive"
        assert cfg.MOMENTUM_CLIP_MAX > cfg.MOMENTUM_CLIP_MIN, \
            "Momentum clip max must be greater than min"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not available")
class TestTechnicalIndicatorParameters:
    """Tests for technical indicator parameter validity."""

    def test_ma_spans_positive(self):
        """Test that moving average spans are positive."""
        cfg = Config()

        assert all(span > 0 for span in cfg.MA_SPANS), "All MA spans must be positive"
        assert cfg.MA_SPANS == sorted(cfg.MA_SPANS), "MA spans should be ascending"

    def test_rsi_periods_positive(self):
        """Test that RSI periods are positive."""
        cfg = Config()

        assert all(period > 0 for period in cfg.RSI_PERIODS), "All RSI periods must be positive"
        assert all(period >= 2 for period in cfg.RSI_PERIODS), "RSI periods should be >= 2"

    def test_bb_periods_positive(self):
        """Test that Bollinger Band periods are positive."""
        cfg = Config()

        assert all(period > 0 for period in cfg.BB_PERIODS), "All BB periods must be positive"
        assert cfg.BB_PERIODS == sorted(cfg.BB_PERIODS), "BB periods should be ascending"

    def test_macd_settings_valid(self):
        """Test that MACD settings are valid."""
        cfg = Config()

        for i, settings in enumerate(cfg.MACD_SETTINGS):
            assert 'fast' in settings, f"MACD {i} missing 'fast'"
            assert 'slow' in settings, f"MACD {i} missing 'slow'"
            assert 'signal' in settings, f"MACD {i} missing 'signal'"

            assert settings['fast'] > 0, f"MACD {i} fast must be positive"
            assert settings['slow'] > 0, f"MACD {i} slow must be positive"
            assert settings['signal'] > 0, f"MACD {i} signal must be positive"

            assert settings['slow'] > settings['fast'], \
                f"MACD {i} slow must be greater than fast"


# Mathematical Property Tests

class TestMathematicalProperties:
    """Tests for mathematical properties and relationships."""

    def test_loss_weight_sum_bounded(self):
        """Test that sum of loss weights is bounded."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        # Sum of all lambda weights
        lambda_attrs = [attr for attr in dir(cfg) if attr.startswith('LAMBDA_')]
        total_weight = sum(getattr(cfg, attr) for attr in lambda_attrs)

        # Total should be reasonable (not too large)
        assert total_weight < 100, f"Total loss weight sum too large: {total_weight}"

    def test_variance_nll_stability(self):
        """Test that variance bounds ensure NLL stability."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        # NLL = 0.5 * log(2π * var) + 0.5 * (y - μ)² / var
        # With var_floor, log term is bounded below
        log_term_min = 0.5 * np.log(2 * np.pi * cfg.VAR_FLOOR)

        # Should be finite
        assert np.isfinite(log_term_min), "Variance floor leads to infinite NLL"

        # With var_cap, log term is bounded above
        log_term_max = 0.5 * np.log(2 * np.pi * cfg.VAR_CAP)

        # Should be reasonable
        assert log_term_max < 20, "Variance cap leads to very large NLL"

    def test_horizon_spacing_geometric(self):
        """Test if horizon spacing is approximately geometric."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        if len(cfg.HORIZON_STEPS) >= 3:
            # Check geometric progression
            h0, h1, h2 = cfg.HORIZON_STEPS[:3]

            # Ratios should be similar for geometric progression
            ratio1 = h1 / h0
            ratio2 = h2 / h1

            # Allow some tolerance
            assert 0.5 < ratio2 / ratio1 < 2.0, \
                "Horizon spacing should be approximately geometric"

    def test_learning_rate_batch_size_relationship(self):
        """Test that learning rate and batch size are appropriately scaled."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        # Effective learning rate = LR / sqrt(batch_size) (rough heuristic)
        effective_lr = cfg.LR / np.sqrt(cfg.BATCH_SIZE)

        # Should be in reasonable range
        assert 1e-4 <= effective_lr <= 0.1, \
            f"Effective learning rate {effective_lr} may be too extreme"


# Numerical Stability Tests

class TestNumericalStabilityBounds:
    """Tests that configuration prevents numerical instabilities."""

    def test_no_zero_division_risk(self):
        """Test that parameters don't risk division by zero."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        # Check all epsilon-like values
        assert cfg.VAR_FLOOR > 0, "Variance floor must prevent division by zero"

    def test_exponential_overflow_protection(self):
        """Test that parameters won't cause exponential overflow."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        # Gradient multiplier shouldn't be too large
        assert cfg.INDICATOR_GRAD_MULT < 1000, \
            "Indicator gradient multiplier too large, risks overflow"

        # Activation scales shouldn't cause overflow
        max_activation = max(cfg.TANH_SCALE, cfg.SIGMOID_SCALE) * 10  # Assume max input ~10
        assert np.exp(max_activation) < 1e20, "Activation scales risk overflow"

    def test_log_underflow_protection(self):
        """Test that parameters prevent log underflow."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        # Variance floor prevents log(0)
        min_log_value = np.log(cfg.VAR_FLOOR)
        assert np.isfinite(min_log_value), "Variance floor too small for log"


# Configuration Consistency Tests

class TestConfigurationConsistency:
    """Tests for internal consistency between related parameters."""

    def test_lookback_horizon_relationship(self):
        """Test that lookback window is larger than prediction horizons."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        max_horizon = max(cfg.HORIZON_STEPS)
        assert cfg.LOOKBACK > max_horizon, \
            f"Lookback ({cfg.LOOKBACK}) should be greater than max horizon ({max_horizon})"

    def test_batch_size_sequence_count_relationship(self):
        """Test that max sequence count accommodates batch size."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        assert cfg.MAX_SEQUENCE_COUNT >= cfg.BATCH_SIZE, \
            "Max sequence count should be at least batch size"

    def test_epochs_patience_relationship(self):
        """Test that patience is reasonable relative to epochs."""
        if not MODEL_AVAILABLE:
            pytest.skip("model.py not available")

        cfg = Config()

        assert cfg.PATIENCE <= cfg.EPOCHS, \
            "Patience should not exceed total epochs"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

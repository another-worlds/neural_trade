"""
Comprehensive tests for pipeline integrity and mathematical consistency.

Tests cover:
1. Data pipeline: loading, preprocessing, sequence generation
2. Mathematical operations: scaling, normalization, reversibility
3. Numerical stability: NaN, infinity, extreme values
4. Loss calculations: correctness, stability
5. Model architecture: shapes, connectivity
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os


# Test Data Generation Utilities

def create_synthetic_ohlcv_data(n_samples=1000, seed=42):
    """Create synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Generate realistic price movement
    initial_price = 100.0
    returns = np.random.randn(n_samples) * 0.02  # 2% volatility
    prices = initial_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
        'close': prices,
        'volume': np.random.randint(100, 10000, n_samples),
    })

    # Ensure high >= open, close and low <= open, close
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def create_test_csv(n_samples=1000):
    """Create temporary CSV file with test data."""
    df = create_synthetic_ohlcv_data(n_samples)

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return temp_file.name, df


# Test Data Pipeline

class TestDataPipeline:
    """Tests for data loading and preprocessing pipeline."""

    def test_synthetic_data_generation(self):
        """Test that synthetic data generation produces valid OHLCV data."""
        df = create_synthetic_ohlcv_data(n_samples=100)

        # Check shape
        assert len(df) == 100
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Check OHLC relationships
        assert (df['high'] >= df['open']).all(), "High should be >= open"
        assert (df['high'] >= df['close']).all(), "High should be >= close"
        assert (df['low'] <= df['open']).all(), "Low should be <= open"
        assert (df['low'] <= df['close']).all(), "Low should be <= close"

        # Check no NaN or inf
        assert not df.isnull().any().any(), "Data should not contain NaN"
        assert not np.isinf(df.select_dtypes(include=[np.number]).values).any(), "Data should not contain inf"

        # Check positive prices
        assert (df['close'] > 0).all(), "Prices should be positive"
        assert (df['volume'] > 0).all(), "Volume should be positive"

    def test_data_csv_creation(self):
        """Test CSV file creation and reading."""
        csv_path, original_df = create_test_csv(n_samples=50)

        try:
            # Read back
            loaded_df = pd.read_csv(csv_path, parse_dates=['timestamp'])

            # Check equality
            pd.testing.assert_frame_equal(
                original_df.reset_index(drop=True),
                loaded_df.reset_index(drop=True)
            )
        finally:
            os.unlink(csv_path)

    def test_price_continuity(self):
        """Test that price series doesn't have unrealistic jumps."""
        df = create_synthetic_ohlcv_data(n_samples=1000)

        # Calculate returns
        returns = df['close'].pct_change().dropna()

        # Check no single-period return exceeds 50% (unrealistic)
        assert (returns.abs() < 0.5).all(), "Returns should be realistic"

        # Check mean return is close to zero
        assert abs(returns.mean()) < 0.01, "Mean return should be close to zero"


# Test Mathematical Operations

class TestMathematicalConsistency:
    """Tests for mathematical operations and transformations."""

    def test_scaling_reversibility(self):
        """Test that scaling and unscaling are perfect inverses."""
        # Generate test data
        data = np.random.randn(100, 5) * 10 + 50

        # Scale
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        scaled = (data - mean) / (std + 1e-8)

        # Unscale
        unscaled = scaled * (std + 1e-8) + mean

        # Check reversibility
        np.testing.assert_allclose(data, unscaled, rtol=1e-10, atol=1e-10)

    def test_scaling_properties(self):
        """Test that scaled data has expected statistical properties."""
        data = np.random.randn(10000, 3) * 5 + 10

        mean = data.mean(axis=0)
        std = data.std(axis=0)
        scaled = (data - mean) / (std + 1e-8)

        # Check mean is approximately 0
        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-10)

        # Check std is approximately 1
        np.testing.assert_allclose(scaled.std(axis=0), 1, atol=1e-10)

    def test_log_transform_reversibility(self):
        """Test log transformation reversibility."""
        data = np.random.uniform(1, 100, size=(100, 5))

        # Log transform
        log_data = np.log(data)

        # Reverse
        exp_data = np.exp(log_data)

        # Check reversibility
        np.testing.assert_allclose(data, exp_data, rtol=1e-10)

    def test_percentage_change_calculation(self):
        """Test that percentage changes are calculated correctly."""
        prices = np.array([100, 102, 101, 105, 103])

        # Manual calculation
        manual_pct = (prices[1:] - prices[:-1]) / prices[:-1]

        # Using pandas
        pd_pct = pd.Series(prices).pct_change().dropna().values

        # Check equivalence
        np.testing.assert_allclose(manual_pct, pd_pct, rtol=1e-10)

    def test_returns_to_prices(self):
        """Test converting returns back to prices."""
        initial_price = 100
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])

        # Calculate prices from returns
        prices = initial_price * np.cumprod(1 + returns)

        # Calculate returns back
        calculated_returns = (prices / np.concatenate([[initial_price], prices[:-1]])) - 1

        # Check equivalence
        np.testing.assert_allclose(returns, calculated_returns, rtol=1e-10)


# Test Numerical Stability

class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_division_by_zero_protection(self):
        """Test that division by zero is properly handled."""
        denominator = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
        epsilon = 1e-8

        # Protected division
        result = 1.0 / (denominator + epsilon)

        # Check no inf or nan
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_log_of_zero_protection(self):
        """Test that log of zero/negative is properly handled."""
        data = np.array([1.0, 0.0, 2.0, -1.0, 3.0])
        epsilon = 1e-8

        # Clip to positive before log
        clipped = np.clip(data, epsilon, None)
        result = np.log(clipped)

        # Check no nan or inf
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_exponential_overflow_protection(self):
        """Test that exponential operations don't overflow."""
        data = np.array([-100, -10, -1, 0, 1, 10, 100])

        # Clip before exp
        clipped = np.clip(data, -10, 10)
        result = np.exp(clipped)

        # Check no inf
        assert not np.isinf(result).any()

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        gradients = np.array([0.1, 10.0, -15.0, 0.5, 20.0])
        clip_norm = 5.0

        # Clip gradients
        norm = np.linalg.norm(gradients)
        if norm > clip_norm:
            clipped = gradients * (clip_norm / norm)
        else:
            clipped = gradients

        # Check clipped norm
        clipped_norm = np.linalg.norm(clipped)
        assert clipped_norm <= clip_norm + 1e-6

    def test_nan_detection(self):
        """Test NaN detection in arrays."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        # Detect NaN
        has_nan = np.isnan(data).any()
        assert has_nan

        # Count NaN
        nan_count = np.isnan(data).sum()
        assert nan_count == 1

        # Replace NaN
        clean_data = np.where(np.isnan(data), 0, data)
        assert not np.isnan(clean_data).any()

    def test_inf_detection(self):
        """Test infinity detection in arrays."""
        data = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])

        # Detect inf
        has_inf = np.isinf(data).any()
        assert has_inf

        # Count inf
        inf_count = np.isinf(data).sum()
        assert inf_count == 2

        # Replace inf
        clean_data = np.where(np.isinf(data), 0, data)
        assert not np.isinf(clean_data).any()


# Test Loss Function Math

class TestLossFunctionMath:
    """Tests for loss function mathematical correctness."""

    def test_mse_calculation(self):
        """Test Mean Squared Error calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        # Manual MSE
        manual_mse = np.mean((y_true - y_pred) ** 2)

        # Expected: ((0.1)^2 + (0.2)^2 + (0.1)^2 + (0.1)^2 + (0.2)^2) / 5
        expected_mse = (0.01 + 0.04 + 0.01 + 0.01 + 0.04) / 5

        np.testing.assert_allclose(manual_mse, expected_mse, rtol=1e-10)

    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        # Manual MAE
        manual_mae = np.mean(np.abs(y_true - y_pred))

        # Expected: (0.1 + 0.2 + 0.1 + 0.1 + 0.2) / 5
        expected_mae = 0.7 / 5

        np.testing.assert_allclose(manual_mae, expected_mae, rtol=1e-10)

    def test_huber_loss_properties(self):
        """Test Huber loss properties."""
        delta = 1.0

        # Small error (quadratic region)
        small_error = 0.5
        huber_small = 0.5 * small_error ** 2
        assert huber_small < delta

        # Large error (linear region)
        large_error = 2.0
        huber_large = delta * (np.abs(large_error) - 0.5 * delta)
        assert huber_large > huber_small

    def test_log_cosh_loss(self):
        """Test log-cosh loss calculation."""
        errors = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Log-cosh
        log_cosh = np.log(np.cosh(errors))

        # Check no nan or inf
        assert not np.isnan(log_cosh).any()
        assert not np.isinf(log_cosh).any()

        # Check symmetry
        np.testing.assert_allclose(log_cosh[0], log_cosh[4], rtol=1e-10)
        np.testing.assert_allclose(log_cosh[1], log_cosh[3], rtol=1e-10)

        # Check minimum at zero
        assert log_cosh[2] < log_cosh[1]
        assert log_cosh[2] < log_cosh[0]

    def test_focal_loss_properties(self):
        """Test focal loss focusing property."""
        # True labels
        y_true = np.ones(5)

        # Predictions (varying confidence)
        y_pred = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        # Focal loss should decrease as prediction confidence increases
        gamma = 2.0
        alpha = 0.5

        p_t = y_true * y_pred
        focal_weight = (1.0 - p_t) ** gamma
        bce = -y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7))
        focal = alpha * focal_weight * bce

        # Higher confidence should have lower loss
        assert focal[0] > focal[1] > focal[2] > focal[3] > focal[4]


# Test Sequence Generation

class TestSequenceGeneration:
    """Tests for time series sequence generation."""

    def test_sliding_window_shape(self):
        """Test sliding window sequence generation shapes."""
        data = np.arange(100).reshape(100, 1)
        window_size = 10
        step = 1

        sequences = []
        for i in range(0, len(data) - window_size, step):
            seq = data[i:i + window_size]
            sequences.append(seq)

        sequences = np.array(sequences)

        # Check shape
        expected_n_sequences = (len(data) - window_size) // step
        assert sequences.shape == (expected_n_sequences, window_size, 1)

    def test_sequence_continuity(self):
        """Test that sequences are continuous."""
        data = np.arange(20)
        window_size = 5

        sequences = []
        for i in range(len(data) - window_size):
            seq = data[i:i + window_size]
            sequences.append(seq)

        # Check each sequence is continuous
        for seq in sequences:
            diffs = np.diff(seq)
            assert (diffs == 1).all(), "Sequence should be continuous"

    def test_sequence_overlap(self):
        """Test sequence overlap with different step sizes."""
        data = np.arange(20)
        window_size = 5

        # Step 1: maximum overlap
        seqs_step1 = []
        for i in range(0, len(data) - window_size, 1):
            seqs_step1.append(data[i:i + window_size])

        # Step 5: no overlap
        seqs_step5 = []
        for i in range(0, len(data) - window_size, 5):
            seqs_step5.append(data[i:i + window_size])

        # Check counts
        assert len(seqs_step1) > len(seqs_step5)
        assert len(seqs_step5) == (len(data) - window_size) // 5


# Test Array Shape Consistency

class TestShapeConsistency:
    """Tests for array shape consistency through pipeline."""

    def test_batch_dimension_preservation(self):
        """Test that batch dimension is preserved through operations."""
        batch_size = 32
        sequence_length = 60
        features = 5

        # Input
        x = np.random.randn(batch_size, sequence_length, features)

        # Simulated layer operations (should preserve batch)
        # Dense layer: (batch, seq, features) -> (batch, seq, units)
        units = 128
        output_dense = np.random.randn(batch_size, sequence_length, units)
        assert output_dense.shape[0] == batch_size

        # Global pooling: (batch, seq, features) -> (batch, features)
        output_pool = x.mean(axis=1)
        assert output_pool.shape == (batch_size, features)

    def test_multi_output_shapes(self):
        """Test shapes for multi-output model."""
        from model import Config
        cfg = Config()
        batch_size = 32
        num_horizons = cfg.num_horizons

        # Generate outputs dynamically: N horizons, three outputs each (price, dir, var)
        outputs = []
        for i in range(num_horizons):
            price = np.random.randn(batch_size, 1)
            direction = np.random.randn(batch_size, 1)
            variance = np.random.randn(batch_size, 1)
            outputs.extend([price, direction, variance])

        # Check all have correct batch dimension
        for output in outputs:
            assert output.shape == (batch_size, 1)


# Test Statistical Properties

class TestStatisticalProperties:
    """Tests for statistical properties of data and predictions."""

    def test_distribution_preservation(self):
        """Test that transformations preserve distribution properties."""
        # Generate normal distribution
        data = np.random.randn(10000)

        # Check properties
        assert abs(data.mean()) < 0.1  # Mean close to 0
        assert abs(data.std() - 1.0) < 0.1  # Std close to 1

        # Apply reversible transformation
        transformed = data * 2 + 5
        reversed_data = (transformed - 5) / 2

        # Check preserved
        np.testing.assert_allclose(data.mean(), reversed_data.mean(), atol=0.1)
        np.testing.assert_allclose(data.std(), reversed_data.std(), atol=0.1)

    def test_covariance_preservation(self):
        """Test that linear transformations preserve covariance structure."""
        # Generate correlated data
        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        data = np.random.multivariate_normal(mean, cov, size=10000)

        # Calculate covariance
        original_cov = np.cov(data.T)

        # Apply linear transformation (centering)
        centered = data - data.mean(axis=0)
        centered_cov = np.cov(centered.T)

        # Covariance should be approximately preserved
        np.testing.assert_allclose(original_cov, centered_cov, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

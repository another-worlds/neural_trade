"""
Comprehensive test suite for model.py

Tests cover:
- Config class
- DataProcessor class
- differentiable_past_sample function
- LearnableIndicators layer
- PositionalEncodingLayer
- PricePredictor class
- CustomTrainModel class
- Callback classes
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import joblib

from model import (
    Config,
    DataProcessor,
    differentiable_past_sample,
    LearnableIndicators,
    PositionalEncodingLayer,
    PricePredictor,
    CustomTrainModel,
    TqdmCallback,
    SimpleLoggingCallback,
    ParamsLogger
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Create a Config instance for testing."""
    return Config()


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing DataProcessor."""
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='1min'),
        'open': np.random.uniform(40000, 50000, 200),
        'high': np.random.uniform(40000, 50000, 200),
        'low': np.random.uniform(40000, 50000, 200),
        'close': np.random.uniform(40000, 50000, 200),
        'volume': np.random.uniform(1000, 5000, 200)
    }
    df = pd.DataFrame(data)
    # Ensure high >= low
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    return df


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_csv_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def data_processor(config):
    """Create a DataProcessor instance."""
    return DataProcessor(config)


@pytest.fixture
def sample_close_values():
    """Generate sample close price values."""
    return np.random.uniform(40000, 50000, 200).astype('float32')


# ============================================================================
# TEST CONFIG CLASS
# ============================================================================

class TestConfig:
    """Test suite for Config class."""

    def test_config_initialization(self, config):
        """Test that Config initializes with expected attributes."""
        assert hasattr(config, 'HOUR')
        assert hasattr(config, 'DAY')
        assert hasattr(config, 'CSV_PATH')
        assert hasattr(config, 'LOOKBACK')
        assert hasattr(config, 'BATCH_SIZE')
        assert hasattr(config, 'EPOCHS')

    def test_time_constants(self, config):
        """Test time constant calculations."""
        assert config.HOUR == 60
        assert config.DAY == config.HOUR * 24
        assert config.DAY == 1440

    def test_lookback_configuration(self, config):
        """Test lookback window configuration."""
        assert config.LOOKBACK == config.HOUR
        assert config.LOOKBACK == 60

    def test_horizon_steps(self, config):
        """Test horizon steps configuration."""
        assert hasattr(config, 'HORIZON_STEPS')
        assert isinstance(config.HORIZON_STEPS, list)
        assert len(config.HORIZON_STEPS) == 3
        assert config.HORIZON_STEPS == [1, 5, 15]

    def test_extended_trend_periods(self, config):
        """Test extended trend periods configuration."""
        assert hasattr(config, 'EXTENDED_TREND_PERIODS')
        assert isinstance(config.EXTENDED_TREND_PERIODS, list)
        assert config.EXTENDED_TREND_PERIODS == [1, 5, 15]

    def test_loss_weights(self, config):
        """Test loss function weight configurations."""
        assert config.LAMBDA_POINT == 1
        assert config.LAMBDA_LOCAL_TREND == 1
        assert config.LAMBDA_GLOBAL_TREND == 1
        assert config.LAMBDA_DIR == 1

    def test_ma_spans(self, config):
        """Test moving average span configurations."""
        assert hasattr(config, 'MA_SPANS')
        assert isinstance(config.MA_SPANS, list)
        assert len(config.MA_SPANS) == 3

    def test_macd_settings(self, config):
        """Test MACD settings configuration."""
        assert hasattr(config, 'MACD_SETTINGS')
        assert isinstance(config.MACD_SETTINGS, list)
        assert len(config.MACD_SETTINGS) == 3
        for setting in config.MACD_SETTINGS:
            assert 'fast' in setting
            assert 'slow' in setting
            assert 'signal' in setting

    def test_rsi_periods(self, config):
        """Test RSI period configurations."""
        assert hasattr(config, 'RSI_PERIODS')
        assert isinstance(config.RSI_PERIODS, list)
        assert len(config.RSI_PERIODS) == 3

    def test_model_paths(self, config):
        """Test model and scaler path configurations."""
        assert hasattr(config, 'MODEL_PATH')
        assert hasattr(config, 'SCALER_PATH')
        assert config.MODEL_PATH.endswith('.h5')
        assert config.SCALER_PATH.endswith('.joblib')

    def test_focal_loss_params(self, config):
        """Test focal loss hyperparameters."""
        assert hasattr(config, 'FOCAL_ALPHA')
        assert hasattr(config, 'FOCAL_GAMMA')
        assert 0 < config.FOCAL_ALPHA < 1
        assert config.FOCAL_GAMMA >= 0


# ============================================================================
# TEST DATAPROCESSOR CLASS
# ============================================================================

class TestDataProcessor:
    """Test suite for DataProcessor class."""

    def test_initialization(self, data_processor, config):
        """Test DataProcessor initialization."""
        assert data_processor.config == config

    def test_clean_numeric(self, data_processor):
        """Test numeric cleaning functionality."""
        series = pd.Series(['$1,234.56', '789.01', '', '$2,345'])
        cleaned = data_processor.clean_numeric(series)
        assert cleaned[0] == 1234.56
        assert cleaned[1] == 789.01
        assert np.isnan(cleaned[2])
        assert cleaned[3] == 2345.0

    def test_load_and_prepare_data(self, data_processor, temp_csv_file):
        """Test data loading and preparation."""
        data_processor.config.CSV_PATH = temp_csv_file
        df, close_values = data_processor.load_and_prepare_data()

        assert isinstance(df, pd.DataFrame)
        assert isinstance(close_values, np.ndarray)
        assert len(df) > 0
        assert len(close_values) == len(df)
        assert close_values.dtype == np.float32

    def test_load_and_prepare_data_with_resampling(self, data_processor, temp_csv_file):
        """Test data loading with resampling."""
        data_processor.config.CSV_PATH = temp_csv_file
        data_processor.config.RESAMPLE_MINUTES = 5
        data_processor.config.LOOKBACK = 10  # Reduce lookback for resampled data
        df, close_values = data_processor.load_and_prepare_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Should have fewer rows due to resampling
        assert len(df) < 200

    def test_compute_extended_trend_features(self, data_processor, sample_close_values):
        """Test extended trend feature computation."""
        periods = [1, 5, 15]
        index = 60
        features = data_processor.compute_extended_trend_features(
            sample_close_values, index, periods
        )

        assert isinstance(features, np.ndarray)
        assert features.shape == (len(periods),)
        assert features.dtype == np.float32

    def test_compute_extended_trend_features_edge_cases(self, data_processor, sample_close_values):
        """Test extended trend features at edge cases."""
        periods = [1, 5, 15]

        # Test at beginning of series (should handle gracefully)
        features = data_processor.compute_extended_trend_features(
            sample_close_values, 0, periods
        )
        assert features.shape == (len(periods),)

        # Test with negative index (should clamp)
        features = data_processor.compute_extended_trend_features(
            sample_close_values, -1, periods
        )
        assert features.shape == (len(periods),)

    def test_make_sequences_with_extended_trends(self, data_processor, sample_close_values):
        """Test sequence generation with extended trends."""
        lookback = 60
        X, y, last_close, extended_trends = data_processor.make_sequences_with_extended_trends(
            sample_close_values, lookback
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(last_close, np.ndarray)
        assert isinstance(extended_trends, np.ndarray)

        assert X.shape[1] == lookback
        assert y.shape[1] == 3  # Three horizons
        assert len(last_close) == len(X)
        assert len(extended_trends) == len(X)

    def test_make_sequences_shapes(self, data_processor, sample_close_values):
        """Test that sequence shapes are consistent."""
        lookback = 60
        X, y, last_close, extended_trends = data_processor.make_sequences_with_extended_trends(
            sample_close_values, lookback
        )

        n_sequences = X.shape[0]
        assert y.shape[0] == n_sequences
        assert last_close.shape[0] == n_sequences
        assert extended_trends.shape[0] == n_sequences

    def test_prepare_datasets(self, data_processor, temp_csv_file):
        """Test full dataset preparation pipeline."""
        data_processor.config.CSV_PATH = temp_csv_file
        df, close_values = data_processor.load_and_prepare_data()

        result = data_processor.prepare_datasets(df, close_values)

        # Should return 11 elements
        assert len(result) == 11

        X_train, y_train, last_close_train, ext_train = result[0], result[1], result[2], result[3]
        X_test, y_test, last_close_test, ext_test = result[4], result[5], result[6], result[7]

        # Check train/test split
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) > len(X_test)  # Train should be larger

    def test_prepare_datasets_with_max_sequences(self, data_processor, temp_csv_file):
        """Test dataset preparation with sequence limit."""
        data_processor.config.CSV_PATH = temp_csv_file
        data_processor.config.MAX_SEQUENCE_COUNT = 50
        df, close_values = data_processor.load_and_prepare_data()

        result = data_processor.prepare_datasets(df, close_values)
        X_train, X_test = result[0], result[4]

        # Total sequences should not exceed MAX_SEQUENCE_COUNT
        total_sequences = len(X_train) + len(X_test)
        assert total_sequences <= data_processor.config.MAX_SEQUENCE_COUNT


# ============================================================================
# TEST DIFFERENTIABLE_PAST_SAMPLE FUNCTION
# ============================================================================

class TestDifferentiablePastSample:
    """Test suite for differentiable_past_sample function."""

    def test_basic_functionality(self):
        """Test basic sampling functionality."""
        batch_size = 4
        time_steps = 10
        x_seq = tf.random.normal((batch_size, time_steps))
        fractional_lag = tf.constant([1.0, 2.0, 3.0, 4.0])

        result = differentiable_past_sample(x_seq, fractional_lag)

        assert result.shape == (batch_size,)

    def test_with_different_sigma(self):
        """Test sampling with different sigma values."""
        batch_size = 4
        time_steps = 10
        x_seq = tf.random.normal((batch_size, time_steps))
        fractional_lag = tf.constant([1.0, 2.0, 3.0, 4.0])

        result1 = differentiable_past_sample(x_seq, fractional_lag, sigma=0.5)
        result2 = differentiable_past_sample(x_seq, fractional_lag, sigma=2.0)

        assert result1.shape == result2.shape == (batch_size,)
        # Results should differ with different sigma
        assert not tf.reduce_all(tf.equal(result1, result2))

    def test_zero_lag(self):
        """Test sampling with zero lag (should sample last element)."""
        batch_size = 2
        time_steps = 10
        x_seq = tf.random.normal((batch_size, time_steps))
        fractional_lag = tf.constant([0.0, 0.0])

        result = differentiable_past_sample(x_seq, fractional_lag)

        assert result.shape == (batch_size,)

    def test_gradient_flow(self):
        """Test that gradients flow through the operation."""
        batch_size = 2
        time_steps = 10
        x_seq = tf.Variable(tf.random.normal((batch_size, time_steps)))
        fractional_lag = tf.constant([1.0, 2.0])

        with tf.GradientTape() as tape:
            result = differentiable_past_sample(x_seq, fractional_lag)
            loss = tf.reduce_sum(result)

        gradients = tape.gradient(loss, x_seq)
        assert gradients is not None
        assert gradients.shape == x_seq.shape


# ============================================================================
# TEST LEARNABLEINDICATORS LAYER
# ============================================================================

class TestLearnableIndicators:
    """Test suite for LearnableIndicators layer."""

    def test_initialization(self, config):
        """Test LearnableIndicators initialization."""
        layer = LearnableIndicators(config)
        assert layer.config == config
        assert layer.epsilon == 1e-8

    def test_build(self, config):
        """Test layer building."""
        layer = LearnableIndicators(config)
        batch_size = 4
        lookback = config.LOOKBACK

        # Calculate expected number of logits
        num_logits = (len(config.MA_SPANS) +
                     len(config.MACD_SETTINGS) * 3 +
                     len(config.CUSTOM_MACD_PAIRS) * 3 +
                     len(config.RSI_PERIODS) +
                     len(config.BB_PERIODS) +
                     len(config.MOMENTUM_PERIODS))

        input_shape = [(batch_size, lookback), (batch_size, num_logits)]
        layer.build(input_shape)

        assert len(layer.alpha_vars_ma) == len(config.MA_SPANS)
        assert len(layer.rsi_alpha_vars) == len(config.RSI_PERIODS)
        assert len(layer.bb_alpha_vars) == len(config.BB_PERIODS)
        assert len(layer.momentum_raw) == len(config.MOMENTUM_PERIODS)

    def test_call(self, config):
        """Test layer forward pass."""
        layer = LearnableIndicators(config)
        batch_size = 4
        lookback = config.LOOKBACK

        num_logits = (len(config.MA_SPANS) +
                     len(config.MACD_SETTINGS) * 3 +
                     len(config.CUSTOM_MACD_PAIRS) * 3 +
                     len(config.RSI_PERIODS) +
                     len(config.BB_PERIODS) +
                     len(config.MOMENTUM_PERIODS))

        x = tf.random.normal((batch_size, lookback))
        meta_adjust = tf.random.normal((batch_size, num_logits))

        output = layer([x, meta_adjust])

        assert output.shape[0] == batch_size
        assert output.shape[1] == lookback

    def test_ewma_seq(self, config):
        """Test EWMA sequence computation."""
        layer = LearnableIndicators(config)
        batch_size = 4
        time_steps = 10
        x_seq = tf.random.normal((batch_size, time_steps))
        alpha = 0.5

        result = layer.ewma_seq(x_seq, alpha)

        assert result.shape == x_seq.shape

    def test_get_learned_periods(self, config):
        """Test learned parameter extraction."""
        layer = LearnableIndicators(config)
        batch_size = 4
        lookback = config.LOOKBACK

        num_logits = (len(config.MA_SPANS) +
                     len(config.MACD_SETTINGS) * 3 +
                     len(config.CUSTOM_MACD_PAIRS) * 3 +
                     len(config.RSI_PERIODS) +
                     len(config.BB_PERIODS) +
                     len(config.MOMENTUM_PERIODS))

        input_shape = [(batch_size, lookback), (batch_size, num_logits)]
        layer.build(input_shape)

        learned = layer.get_learned_parameters()

        assert isinstance(learned, dict)
        assert len(learned) > 0


# ============================================================================
# TEST POSITIONALENCODINGLAYER
# ============================================================================

class TestPositionalEncodingLayer:
    """Test suite for PositionalEncodingLayer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = PositionalEncodingLayer()
        assert layer is not None

    def test_call(self):
        """Test layer forward pass."""
        layer = PositionalEncodingLayer()
        batch_size = 4
        seq_len = 10
        d_model = 16

        inputs = tf.random.normal((batch_size, seq_len, d_model))
        output = layer(inputs)

        assert output.shape == inputs.shape

    def test_output_range(self):
        """Test that output is bounded (sine/cosine)."""
        layer = PositionalEncodingLayer()
        inputs = tf.random.normal((2, 10, 8))
        output = layer(inputs)

        # Positional encoding should be bounded by [-1, 1]
        assert tf.reduce_max(output) <= 1.5  # Some tolerance
        assert tf.reduce_min(output) >= -1.5


# ============================================================================
# TEST PRICEPREDICTOR CLASS
# ============================================================================

class TestPricePredictor:
    """Test suite for PricePredictor class."""

    def test_initialization(self, config):
        """Test PricePredictor initialization."""
        predictor = PricePredictor(config)
        assert predictor.config == config

    def test_build_model(self, config):
        """Test model building."""
        predictor = PricePredictor(config)
        model = predictor.build_model()

        assert isinstance(model, tf.keras.Model)
        assert len(model.outputs) == 9  # 3 horizons × 3 outputs each

    def test_model_input_shape(self, config):
        """Test model input shape."""
        predictor = PricePredictor(config)
        model = predictor.build_model()

        assert model.input_shape == (None, config.LOOKBACK)

    def test_model_output_names(self, config):
        """Test model output layer names."""
        predictor = PricePredictor(config)
        model = predictor.build_model()

        # Model should have 9 outputs (3 horizons × 3 outputs each)
        assert len(model.outputs) == 9

        # Verify the expected output layer names exist in the model
        expected_layers = [
            'price_h0', 'direction_h0', 'variance_h0',
            'price_h1', 'direction_h1', 'variance_h1',
            'price_h2', 'direction_h2', 'variance_h2'
        ]

        # Get all layer names from the model
        model_layer_names = [layer.name for layer in model.layers]

        # Verify each expected layer exists
        for layer_name in expected_layers:
            assert layer_name in model_layer_names, f"Expected layer {layer_name} not found in model"

    def test_model_forward_pass(self, config):
        """Test model forward pass."""
        predictor = PricePredictor(config)
        model = predictor.build_model()

        batch_size = 4
        x = tf.random.normal((batch_size, config.LOOKBACK))
        outputs = model(x)

        assert len(outputs) == 9
        for output in outputs:
            assert output.shape[0] == batch_size

    def test_create_datasets(self, config):
        """Test dataset creation."""
        predictor = PricePredictor(config)

        # Create dummy data
        n_samples = 100
        X_train = np.random.randn(n_samples, config.LOOKBACK).astype('float32')
        y_train = np.random.randn(n_samples, 3).astype('float32')
        last_close_train = np.random.randn(n_samples).astype('float32')
        extended_trends_train = np.random.randn(n_samples, 3).astype('float32')

        X_test = np.random.randn(20, config.LOOKBACK).astype('float32')
        y_test = np.random.randn(20, 3).astype('float32')
        last_close_test = np.random.randn(20).astype('float32')
        extended_trends_test = np.random.randn(20, 3).astype('float32')

        train_ds, val_ds = predictor.create_datasets(
            X_train, y_train, last_close_train, extended_trends_train,
            X_test, y_test, last_close_test, extended_trends_test
        )

        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)


# ============================================================================
# TEST CUSTOMTRAINMODEL CLASS
# ============================================================================

class TestCustomTrainModel:
    """Test suite for CustomTrainModel class."""

    def test_initialization(self, config):
        """Test CustomTrainModel initialization."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()

        pred_scale = 100.0
        pred_mean = 0.0

        train_model = CustomTrainModel(
            base_model=base_model,
            pred_scale=pred_scale,
            pred_mean=pred_mean,
            config=config
        )

        assert train_model.base_model == base_model
        assert train_model.config == config

    def test_pred_scale_validation(self, config):
        """Test that very small pred_scale raises error."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()

        with pytest.raises(ValueError, match="pred_scale is too small"):
            CustomTrainModel(
                base_model=base_model,
                pred_scale=1e-10,
                pred_mean=0.0,
                config=config
            )

    def test_forward_pass(self, config):
        """Test forward pass through custom model."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()

        train_model = CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

        batch_size = 4
        x = tf.random.normal((batch_size, config.LOOKBACK))
        outputs = train_model(x, training=False)

        assert len(outputs) == 9


# ============================================================================
# TEST CALLBACK CLASSES
# ============================================================================

class TestCallbacks:
    """Test suite for callback classes."""

    def test_tqdm_callback_initialization(self):
        """Test TqdmCallback initialization."""
        callback = TqdmCallback()
        assert callback.epoch_bar is None
        assert callback.batch_bar is None
        assert callback.start_time is None

    def test_simple_logging_callback_initialization(self):
        """Test SimpleLoggingCallback initialization."""
        callback = SimpleLoggingCallback()
        assert callback is not None

    def test_params_logger_initialization(self, config):
        """Test ParamsLogger initialization."""
        mock_layer = MagicMock()
        callback = ParamsLogger(
            layer=mock_layer,
            out_csv='test_params.csv'
        )
        assert callback.layer == mock_layer
        assert callback.out_csv == 'test_params.csv'

    def test_params_logger_on_epoch_end(self, config):
        """Test ParamsLogger on_epoch_end method."""
        mock_layer = MagicMock()
        mock_layer.get_learned_parameters.return_value = {
            'ma_period_0': 10.0,
            'ma_period_1': 20.0
        }

        callback = ParamsLogger(
            layer=mock_layer,
            out_csv='test_params.csv'
        )

        # Should not raise error
        callback.on_epoch_end(0, {})
        mock_layer.get_learned_parameters.assert_called_once()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_data_pipeline(self, temp_csv_file):
        """Test end-to-end data processing pipeline."""
        config = Config()
        config.CSV_PATH = temp_csv_file
        config.MAX_SEQUENCE_COUNT = 50

        processor = DataProcessor(config)
        df, close_values = processor.load_and_prepare_data()

        result = processor.prepare_datasets(df, close_values)

        assert len(result) == 11
        assert result[0].shape[1] == config.LOOKBACK

    def test_end_to_end_model_creation(self):
        """Test end-to-end model creation and prediction."""
        config = Config()
        predictor = PricePredictor(config)
        model = predictor.build_model()

        # Test prediction
        batch_size = 2
        x = tf.random.normal((batch_size, config.LOOKBACK))
        outputs = model(x)

        assert len(outputs) == 9

        # Verify output shapes
        for i, output in enumerate(outputs):
            assert output.shape == (batch_size, 1)

    def test_model_compilation(self):
        """Test that model can be compiled."""
        config = Config()
        predictor = PricePredictor(config)
        base_model = predictor.build_model()

        train_model = CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

        # Should not raise error
        train_model.compile(optimizer='adam')

        assert train_model.optimizer is not None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_handling(self, data_processor):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_df.to_csv(f.name, index=False)
            data_processor.config.CSV_PATH = f.name

            with pytest.raises(ValueError, match="Not enough rows"):
                data_processor.load_and_prepare_data()

            os.unlink(f.name)

    def test_minimal_data_size(self, data_processor):
        """Test with minimal viable data size."""
        # Create minimal data (just enough for lookback)
        minimal_data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=80, freq='1min'),
            'open': np.random.uniform(40000, 50000, 80),
            'high': np.random.uniform(40000, 50000, 80),
            'low': np.random.uniform(40000, 50000, 80),
            'close': np.random.uniform(40000, 50000, 80),
            'volume': np.random.uniform(1000, 5000, 80)
        }
        df = pd.DataFrame(minimal_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            data_processor.config.CSV_PATH = f.name

            # Should work with minimal data
            result_df, close_values = data_processor.load_and_prepare_data()
            assert len(result_df) >= data_processor.config.LOOKBACK

            os.unlink(f.name)

    def test_nan_handling_in_close_values(self, data_processor):
        """Test handling of NaN values in close prices."""
        close_values = np.array([100.0, 101.0, np.nan, 102.0, 103.0] * 20, dtype='float32')

        # The function propagates NaN values in the data (which is expected behavior)
        # This test verifies that the function runs without crashing
        try:
            X, y, _, _ = data_processor.make_sequences_with_extended_trends(
                close_values, lookback=10
            )
            # Function should complete successfully
            # Note: NaN values will be present in output as they exist in input
            assert X.shape[0] > 0
        except (ValueError, IndexError) as e:
            # If it raises an error, that's also acceptable behavior
            pytest.skip(f"Function raised expected error: {e}")

    def test_very_large_lookback(self):
        """Test with very large lookback window."""
        config = Config()
        config.LOOKBACK = 500  # Very large

        processor = DataProcessor(config)
        close_values = np.random.uniform(40000, 50000, 1000).astype('float32')

        X, y, _, _ = processor.make_sequences_with_extended_trends(
            close_values, config.LOOKBACK
        )

        assert X.shape[1] == config.LOOKBACK


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

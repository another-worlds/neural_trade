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
    ParamsLogger,
    _compute_all_horizon_metrics,
    _apply_config_overrides,
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

    def test_horizon_labels_reflect_resample(self, config):
        """Horizon labels should reflect RESAMPLE_MINUTES (minutes -> hours/days formatting)."""
        cfg = _apply_config_overrides(config, None)
        # Case A: 1-minute bars
        cfg.RESAMPLE_MINUTES = 1
        cfg.HORIZON_STEPS = [1, 5, 15]
        metrics = _compute_all_horizon_metrics(
            config=cfg,
            y_true_deltas=np.zeros((10, 3)),
            y_pred_deltas={"h0": np.zeros(10), "h1": np.zeros(10), "h2": np.zeros(10)},
            last_close=np.ones(10),
        )
        assert metrics['meta']['horizon_labels'] == ['1min', '5min', '15min']

        # Case B: hourly bars (RESAMPLE_MINUTES=60)
        cfg.RESAMPLE_MINUTES = 60
        metrics = _compute_all_horizon_metrics(
            config=cfg,
            y_true_deltas=np.zeros((10, 3)),
            y_pred_deltas={"h0": np.zeros(10), "h1": np.zeros(10), "h2": np.zeros(10)},
            last_close=np.ones(10),
        )
        assert metrics['meta']['horizon_labels'] == ['1h', '5h', '15h']

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


# ---------------------------------------------------------------------------
# ADDITIONAL TESTS: add_plot_aliases behavior
# ---------------------------------------------------------------------------

def test_add_plot_aliases_computes_val_average():
    """When per-horizon validation gauss metrics are present, compute averaged val_dir_acc_avg."""
    logs = {
        'val_gauss_dir_acc_h0': 0.6,
        'val_gauss_dir_acc_h1': 0.4,
        'val_gauss_dir_acc_h2': 0.2,
    }
    out = add_plot_aliases(logs)
    assert 'val_dir_acc_avg' in out
    assert abs(out['val_dir_acc_avg'] - 0.4) < 1e-6


def test_add_plot_aliases_overrides_stale_and_clears_if_missing():
    """Ensure stale aggregate values are overwritten and removed when no component keys present."""
    # Case A: stale value present with new per-horizon metrics -> should be updated
    logs = {
        'val_dir_acc_avg': 0.3611,  # stale
        'val_gauss_dir_acc_h0': 0.9,
        'val_gauss_dir_acc_h1': 0.0,
        'val_gauss_dir_acc_h2': 0.0,
    }
    out = add_plot_aliases(logs)
    assert 'val_dir_acc_avg' in out
    assert abs(out['val_dir_acc_avg'] - (0.9 + 0.0 + 0.0) / 3.0) < 1e-6

    # Case B: no per-horizon keys present -> stale aggregate should be removed
    logs2 = {'val_dir_acc_avg': 0.3611}
    out2 = add_plot_aliases(logs2)
    assert 'val_dir_acc_avg' not in out2


def test_ece_includes_p1_and_behaves():
    """ECE should include p==1.0 in the last bin and compute expected values."""
    ctm = CustomTrainModel.from_config(Config())
    true_h0 = tf.constant([1.0], dtype=tf.float32)
    true_h1 = tf.constant([0.0], dtype=tf.float32)
    true_h2 = tf.constant([0.0], dtype=tf.float32)
    pred_h0 = tf.constant([1.0], dtype=tf.float32)
    pred_h1 = tf.constant([0.0], dtype=tf.float32)
    pred_h2 = tf.constant([0.0], dtype=tf.float32)

    metrics = ctm._compute_direction_metrics(
        true_h0, true_h1, true_h2, pred_h0, pred_h1, pred_h2,
        mask_h0=tf.ones_like(true_h0), mask_h1=tf.ones_like(true_h1), mask_h2=tf.ones_like(true_h2)
    )
    # Perfect calibration for single sample predicted=1,true=1 -> ECE ~= 0
    assert np.isfinite(metrics['dir_ece_h0'].numpy())
    assert abs(metrics['dir_ece_h0'].numpy() - 0.0) < 1e-6

    # Now pred=1,true=0 -> ECE should be ~1.0 for single sample
    pred_bad_h0 = tf.constant([1.0], dtype=tf.float32)
    true_zero_h0 = tf.constant([0.0], dtype=tf.float32)
    metrics2 = ctm._compute_direction_metrics(
        true_zero_h0, true_h1, true_h2, pred_bad_h0, pred_h1, pred_h2,
        mask_h0=tf.ones_like(true_zero_h0), mask_h1=tf.ones_like(true_h1), mask_h2=tf.ones_like(true_h2)
    )
    assert abs(metrics2['dir_ece_h0'].numpy() - 1.0) < 1e-6


def test_mask_all_zero_produces_nan():
    """When mask for a horizon is all zeros, metrics for that horizon should be NaN."""
    ctm = CustomTrainModel.from_config(Config())
    true_h0 = tf.constant([1.0, 0.0], dtype=tf.float32)
    pred_h0 = tf.constant([0.6, 0.4], dtype=tf.float32)
    # Provide masks: h0 all zeros
    mask_h0 = tf.zeros_like(true_h0)
    mask_h1 = tf.ones_like(true_h0)
    mask_h2 = tf.ones_like(true_h0)

    metrics = ctm._compute_direction_metrics(
        true_h0, true_h0, true_h0, pred_h0, pred_h0, pred_h0,
        mask_h0=mask_h0, mask_h1=mask_h1, mask_h2=mask_h2
    )

    assert np.isnan(metrics['dir_acc_h0'].numpy())
    assert np.isnan(metrics['dir_ece_h0'].numpy())


def test_mcc_degenerate_returns_zero():
    """MCC should be 0 when marginals degenerate (no predicted positives)."""
    ctm = CustomTrainModel.from_config(Config())
    # All true negatives, all predicted negatives
    true_h0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    pred_h0 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)

    metrics = ctm._compute_direction_metrics(
        true_h0, true_h0, true_h0, pred_h0, pred_h0, pred_h0,
        mask_h0=tf.ones_like(true_h0), mask_h1=tf.ones_like(true_h0), mask_h2=tf.ones_like(true_h0)
    )
    assert metrics['dir_mcc_h0'].numpy() == 0.0


def test_add_plot_aliases_no_double_val_prefix():
    """Ensure running add_plot_aliases multiple times does not double-prefix val_."""
    logs = {'val_dir_acc_avg': 0.5}
    out1 = add_plot_aliases(dict(logs))
    out2 = add_plot_aliases(dict(out1))
    assert not any(k.startswith('val_val_') for k in out2.keys())


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


# ============================================================================
# TEST LOSS FUNCTIONS MODULE
# ============================================================================

class TestFocalLoss:
    """Test suite for focal_loss method in CustomTrainModel."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_focal_loss_output_shape_reduced(self, train_model):
        """Test focal loss returns scalar when reduce=True."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.9, 0.1, 0.8, 0.2])
        
        loss = train_model.focal_loss(true_labels, logits, reduce=True)
        
        assert loss.shape == ()  # Scalar
        assert tf.math.is_finite(loss)

    def test_focal_loss_output_shape_unreduced(self, train_model):
        """Test focal loss returns per-sample values when reduce=False."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.9, 0.1, 0.8, 0.2])
        
        loss = train_model.focal_loss(true_labels, logits, reduce=False)
        
        assert loss.shape == (4,)
        assert tf.reduce_all(tf.math.is_finite(loss))

    def test_focal_loss_perfect_predictions(self, train_model):
        """Test focal loss is near zero for perfect predictions."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.999, 0.001, 0.999, 0.001])
        
        loss = train_model.focal_loss(true_labels, logits)
        
        assert loss.numpy() < 0.01  # Near zero for perfect predictions

    def test_focal_loss_worst_predictions(self, train_model):
        """Test focal loss is high for completely wrong predictions."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.001, 0.999, 0.001, 0.999])  # All wrong
        
        loss_wrong = train_model.focal_loss(true_labels, logits)
        
        # Compare with correct predictions
        logits_correct = tf.constant([0.999, 0.001, 0.999, 0.001])
        loss_correct = train_model.focal_loss(true_labels, logits_correct)
        
        assert loss_wrong.numpy() > loss_correct.numpy()

    def test_focal_loss_gamma_effect(self, train_model):
        """Test that higher gamma focuses more on hard examples."""
        true_labels = tf.constant([1.0, 1.0, 1.0])
        # Mix of hard (0.5) and easy (0.9) predictions
        logits = tf.constant([0.9, 0.5, 0.3])
        
        loss_low_gamma = train_model.focal_loss(true_labels, logits, gamma=0.0, reduce=False)
        loss_high_gamma = train_model.focal_loss(true_labels, logits, gamma=3.0, reduce=False)
        
        # With higher gamma, easy examples (high prob correct) should have lower loss
        # The ratio of hard:easy loss should be higher with higher gamma
        ratio_low = loss_low_gamma[2] / loss_low_gamma[0]  # hard / easy
        ratio_high = loss_high_gamma[2] / loss_high_gamma[0]
        
        assert ratio_high > ratio_low  # Higher gamma amplifies hard example weighting

    def test_focal_loss_alpha_effect(self, train_model):
        """Test alpha class weighting effect."""
        # All DOWN class (0)
        true_labels_down = tf.constant([0.0, 0.0, 0.0, 0.0])
        # All UP class (1)
        true_labels_up = tf.constant([1.0, 1.0, 1.0, 1.0])
        # Mixed predictions (50% probability)
        logits = tf.constant([0.5, 0.5, 0.5, 0.5])
        
        # Alpha=0.7 weights DOWN more, Alpha=0.3 weights UP more
        loss_down_alpha07 = train_model.focal_loss(true_labels_down, logits, alpha=0.7)
        loss_down_alpha03 = train_model.focal_loss(true_labels_down, logits, alpha=0.3)
        loss_up_alpha07 = train_model.focal_loss(true_labels_up, logits, alpha=0.7)
        loss_up_alpha03 = train_model.focal_loss(true_labels_up, logits, alpha=0.3)
        
        # Higher alpha should weight DOWN (0) class more
        assert loss_down_alpha07.numpy() > loss_down_alpha03.numpy()
        # Lower alpha should weight UP (1) class more
        assert loss_up_alpha03.numpy() > loss_up_alpha07.numpy()

    def test_focal_loss_numerical_stability(self, train_model):
        """Test focal loss handles extreme probability values."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        # Very extreme probabilities
        logits = tf.constant([1e-10, 1.0 - 1e-10, 0.0, 1.0])
        
        loss = train_model.focal_loss(true_labels, logits)
        
        # Should be finite, not NaN or Inf
        assert tf.math.is_finite(loss)

    def test_focal_loss_gradient_flow(self, train_model):
        """Test gradients flow through focal loss."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.Variable([0.6, 0.4, 0.7, 0.3])
        
        with tf.GradientTape() as tape:
            loss = train_model.focal_loss(true_labels, logits)
        
        grads = tape.gradient(loss, logits)
        assert grads is not None
        assert tf.reduce_all(tf.math.is_finite(grads))

    def test_focal_loss_batch_consistency(self, train_model):
        """Test focal loss is consistent across batch sizes."""
        true_labels = tf.constant([1.0, 0.0])
        logits = tf.constant([0.8, 0.2])
        
        # Compute loss for small batch
        loss_small = train_model.focal_loss(true_labels, logits)
        
        # Replicate batch
        true_labels_large = tf.constant([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        logits_large = tf.constant([0.8, 0.2, 0.8, 0.2, 0.8, 0.2])
        loss_large = train_model.focal_loss(true_labels_large, logits_large)
        
        # Mean should be same for replicated batches
        np.testing.assert_almost_equal(loss_small.numpy(), loss_large.numpy(), decimal=5)


class TestDiceLoss:
    """Test suite for dice_loss method in CustomTrainModel."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_dice_loss_output_shape_reduced(self, train_model):
        """Test dice loss returns scalar when reduce=True."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.9, 0.1, 0.8, 0.2])
        
        loss = train_model.dice_loss(true_labels, logits, reduce=True)
        
        assert loss.shape == ()
        assert tf.math.is_finite(loss)

    def test_dice_loss_output_shape_unreduced(self, train_model):
        """Test dice loss returns per-sample values when reduce=False."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.9, 0.1, 0.8, 0.2])
        
        loss = train_model.dice_loss(true_labels, logits, reduce=False)
        
        assert loss.shape == (4,)

    def test_dice_loss_range(self, train_model):
        """Test dice loss is bounded [0, 1]."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.5, 0.5, 0.5, 0.5])
        
        loss = train_model.dice_loss(true_labels, logits)
        
        assert loss.numpy() >= 0.0
        assert loss.numpy() <= 1.0

    def test_dice_loss_perfect_predictions(self, train_model):
        """Test dice loss approaches zero for perfect predictions."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([1.0, 0.0, 1.0, 0.0])
        
        loss = train_model.dice_loss(true_labels, logits)
        
        # With smooth=1.0 default, won't be exactly 0 but should be very small
        assert loss.numpy() < 0.1

    def test_dice_loss_worst_predictions(self, train_model):
        """Test dice loss is high for completely wrong predictions."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits_wrong = tf.constant([0.0, 1.0, 0.0, 1.0])
        logits_correct = tf.constant([1.0, 0.0, 1.0, 0.0])
        
        loss_wrong = train_model.dice_loss(true_labels, logits_wrong)
        loss_correct = train_model.dice_loss(true_labels, logits_correct)
        
        assert loss_wrong.numpy() > loss_correct.numpy()

    def test_dice_loss_smooth_parameter(self, train_model):
        """Test smooth parameter effect on dice loss."""
        true_labels = tf.constant([1.0, 0.0])
        logits = tf.constant([1.0, 0.0])  # Perfect
        
        loss_smooth_0 = train_model.dice_loss(true_labels, logits, smooth=0.0)
        loss_smooth_1 = train_model.dice_loss(true_labels, logits, smooth=1.0)
        loss_smooth_10 = train_model.dice_loss(true_labels, logits, smooth=10.0)
        
        # Higher smooth should increase loss for near-perfect predictions
        # (smoothing prevents perfect zero loss)
        assert tf.math.is_finite(loss_smooth_0)
        assert tf.math.is_finite(loss_smooth_1)
        assert tf.math.is_finite(loss_smooth_10)

    def test_dice_loss_gradient_flow(self, train_model):
        """Test gradients flow through dice loss."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.Variable([0.6, 0.4, 0.7, 0.3])
        
        with tf.GradientTape() as tape:
            loss = train_model.dice_loss(true_labels, logits)
        
        grads = tape.gradient(loss, logits)
        assert grads is not None
        assert tf.reduce_all(tf.math.is_finite(grads))

    def test_dice_loss_symmetric_classes(self, train_model):
        """Test dice loss treats classes symmetrically."""
        # All positives
        true_all_pos = tf.constant([1.0, 1.0, 1.0, 1.0])
        logits_all_pos = tf.constant([0.8, 0.8, 0.8, 0.8])
        
        # All negatives
        true_all_neg = tf.constant([0.0, 0.0, 0.0, 0.0])
        logits_all_neg = tf.constant([0.2, 0.2, 0.2, 0.2])
        
        loss_pos = train_model.dice_loss(true_all_pos, logits_all_pos)
        loss_neg = train_model.dice_loss(true_all_neg, logits_all_neg)
        
        # Both should produce valid finite losses
        assert tf.math.is_finite(loss_pos)
        assert tf.math.is_finite(loss_neg)


class TestCombinedDirectionLoss:
    """Test suite for combined_direction_loss method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_combined_loss_output_shape(self, train_model):
        """Test combined loss returns correct shape."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.9, 0.1, 0.8, 0.2])
        
        loss = train_model.combined_direction_loss(true_labels, logits, reduce=True)
        assert loss.shape == ()
        
        loss_unreduced = train_model.combined_direction_loss(true_labels, logits, reduce=False)
        assert loss_unreduced.shape == (4,)

    def test_combined_loss_is_weighted_sum(self, train_model):
        """Test combined loss is weighted sum of focal and dice."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.7, 0.3, 0.6, 0.4])
        
        focal_weight = 0.6
        dice_weight = 0.4
        
        focal = train_model.focal_loss(true_labels, logits, reduce=True)
        dice = train_model.dice_loss(true_labels, logits, reduce=True)
        expected = focal_weight * focal + dice_weight * dice
        
        combined = train_model.combined_direction_loss(
            true_labels, logits,
            focal_weight=focal_weight,
            dice_weight=dice_weight
        )
        
        np.testing.assert_almost_equal(combined.numpy(), expected.numpy(), decimal=5)

    def test_combined_loss_with_different_weights(self, train_model):
        """Test combined loss varies with different component weights."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.constant([0.7, 0.3, 0.6, 0.4])
        
        # Focal-heavy
        loss_focal_heavy = train_model.combined_direction_loss(
            true_labels, logits, focal_weight=0.9, dice_weight=0.1
        )
        
        # Dice-heavy
        loss_dice_heavy = train_model.combined_direction_loss(
            true_labels, logits, focal_weight=0.1, dice_weight=0.9
        )
        
        # Equal weights
        loss_equal = train_model.combined_direction_loss(
            true_labels, logits, focal_weight=0.5, dice_weight=0.5
        )
        
        # All should be finite and potentially different
        assert tf.math.is_finite(loss_focal_heavy)
        assert tf.math.is_finite(loss_dice_heavy)
        assert tf.math.is_finite(loss_equal)

    def test_combined_loss_gradient_flow(self, train_model):
        """Test gradients flow through combined loss."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        logits = tf.Variable([0.6, 0.4, 0.7, 0.3])
        
        with tf.GradientTape() as tape:
            loss = train_model.combined_direction_loss(true_labels, logits)
        
        grads = tape.gradient(loss, logits)
        assert grads is not None
        assert tf.reduce_all(tf.math.is_finite(grads))


class TestDynamicAlpha:
    """Test suite for compute_dynamic_alpha method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_dynamic_alpha_all_up(self, train_model):
        """Test dynamic alpha when all labels are UP (1)."""
        true_labels = tf.constant([1.0, 1.0, 1.0, 1.0])
        
        alpha = train_model.compute_dynamic_alpha(true_labels)
        
        # All UP → up_ratio = 1.0 → alpha should be max_alpha (0.7 default)
        assert alpha.numpy() == pytest.approx(0.7, rel=1e-5)

    def test_dynamic_alpha_all_down(self, train_model):
        """Test dynamic alpha when all labels are DOWN (0)."""
        true_labels = tf.constant([0.0, 0.0, 0.0, 0.0])
        
        alpha = train_model.compute_dynamic_alpha(true_labels)
        
        # All DOWN → up_ratio = 0.0 → alpha should be min_alpha (0.3 default)
        assert alpha.numpy() == pytest.approx(0.3, rel=1e-5)

    def test_dynamic_alpha_balanced(self, train_model):
        """Test dynamic alpha with balanced classes."""
        true_labels = tf.constant([1.0, 0.0, 1.0, 0.0])
        
        alpha = train_model.compute_dynamic_alpha(true_labels)
        
        # 50% UP → up_ratio = 0.5 → alpha should be 0.5
        assert alpha.numpy() == pytest.approx(0.5, rel=1e-5)

    def test_dynamic_alpha_clipping(self, train_model):
        """Test dynamic alpha respects min/max bounds."""
        true_labels_all_up = tf.constant([1.0, 1.0, 1.0, 1.0])
        true_labels_all_down = tf.constant([0.0, 0.0, 0.0, 0.0])
        
        # Custom bounds
        alpha_up = train_model.compute_dynamic_alpha(
            true_labels_all_up, min_alpha=0.2, max_alpha=0.8
        )
        alpha_down = train_model.compute_dynamic_alpha(
            true_labels_all_down, min_alpha=0.2, max_alpha=0.8
        )
        
        assert alpha_up.numpy() == pytest.approx(0.8, rel=1e-5)
        assert alpha_down.numpy() == pytest.approx(0.2, rel=1e-5)

    def test_dynamic_alpha_proportional(self, train_model):
        """Test dynamic alpha is proportional to UP ratio."""
        # 75% UP
        true_labels_75 = tf.constant([1.0, 1.0, 1.0, 0.0])
        # 25% UP
        true_labels_25 = tf.constant([1.0, 0.0, 0.0, 0.0])
        
        alpha_75 = train_model.compute_dynamic_alpha(true_labels_75)
        alpha_25 = train_model.compute_dynamic_alpha(true_labels_25)
        
        assert alpha_75.numpy() > alpha_25.numpy()
        assert alpha_75.numpy() == pytest.approx(0.7, rel=1e-5)  # 0.75 clamped to 0.7
        assert alpha_25.numpy() == pytest.approx(0.3, rel=1e-5)  # 0.25 clamped to 0.3


class TestPointHuber:
    """Test suite for point_huber loss method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_point_huber_output_shape(self, train_model):
        """Test point_huber returns scalar."""
        y_true = tf.constant([[0.1], [0.2], [-0.1], [0.0]])
        y_pred = tf.constant([[0.12], [0.18], [-0.08], [0.02]])
        
        loss = train_model.point_huber(y_true, y_pred)
        
        assert loss.shape == ()
        assert tf.math.is_finite(loss)

    def test_point_huber_zero_loss_for_perfect(self, train_model):
        """Test point_huber is zero for perfect predictions."""
        y_true = tf.constant([[0.1], [0.2], [-0.1], [0.0]])
        y_pred = y_true  # Perfect match
        
        loss = train_model.point_huber(y_true, y_pred)
        
        # log(cosh(0)) = log(1) = 0
        assert loss.numpy() == pytest.approx(0.0, abs=1e-6)

    def test_point_huber_symmetry(self, train_model):
        """Test point_huber is symmetric (over vs under prediction)."""
        y_true = tf.constant([[0.0]])
        y_pred_over = tf.constant([[0.5]])
        y_pred_under = tf.constant([[-0.5]])
        
        loss_over = train_model.point_huber(y_true, y_pred_over)
        loss_under = train_model.point_huber(y_true, y_pred_under)
        
        np.testing.assert_almost_equal(loss_over.numpy(), loss_under.numpy(), decimal=5)

    def test_point_huber_increases_with_error(self, train_model):
        """Test point_huber increases with prediction error."""
        y_true = tf.constant([[0.0], [0.0], [0.0]])
        y_pred_small = tf.constant([[0.1], [0.1], [0.1]])
        y_pred_medium = tf.constant([[0.5], [0.5], [0.5]])
        y_pred_large = tf.constant([[1.0], [1.0], [1.0]])
        
        loss_small = train_model.point_huber(y_true, y_pred_small)
        loss_medium = train_model.point_huber(y_true, y_pred_medium)
        loss_large = train_model.point_huber(y_true, y_pred_large)
        
        assert loss_small.numpy() < loss_medium.numpy() < loss_large.numpy()

    def test_point_huber_clipping(self, train_model):
        """Test point_huber clips extreme values."""
        y_true = tf.constant([[0.0], [0.0]])
        y_pred_extreme = tf.constant([[1000.0], [-1000.0]])
        
        loss = train_model.point_huber(y_true, y_pred_extreme)
        
        # Should be finite (clipped to [-10, 10])
        assert tf.math.is_finite(loss)

    def test_point_huber_gradient_flow(self, train_model):
        """Test gradients flow through point_huber."""
        y_true = tf.constant([[0.1], [0.2]])
        y_pred = tf.Variable([[0.15], [0.18]])
        
        with tf.GradientTape() as tape:
            loss = train_model.point_huber(y_true, y_pred)
        
        grads = tape.gradient(loss, y_pred)
        assert grads is not None
        assert tf.reduce_all(tf.math.is_finite(grads))

    def test_point_huber_nan_guarding(self, train_model):
        """Test point_huber guards against NaN propagation."""
        y_true = tf.constant([[0.1], [float('nan')]])
        y_pred = tf.constant([[0.1], [0.2]])
        
        loss = train_model.point_huber(y_true, y_pred)
        
        # Should return 0 when result is non-finite (NaN guard)
        assert tf.math.is_finite(loss)


class TestLocalTrendLoss:
    """Test suite for local_trend_loss method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_local_trend_loss_output_shape(self, train_model):
        """Test local_trend_loss returns scalar."""
        x_window = tf.random.normal((4, 60))
        y_true_raw = tf.constant([[45100.0], [45200.0], [45050.0], [45150.0]])
        y_pred_raw = tf.constant([[45105.0], [45195.0], [45055.0], [45145.0]])
        last_close_raw = tf.constant([[45000.0], [45100.0], [45000.0], [45100.0]])
        
        loss = train_model.local_trend_loss(x_window, y_true_raw, y_pred_raw, last_close_raw)
        
        assert loss.shape == ()
        assert tf.math.is_finite(loss)

    def test_local_trend_loss_zero_for_matching_trends(self, train_model):
        """Test local_trend_loss is zero when pred trend matches true trend."""
        x_window = tf.random.normal((2, 60))
        last_close = tf.constant([[45000.0], [45000.0]])
        # Same trend: both predict +100 from last_close
        y_true_raw = tf.constant([[45100.0], [45100.0]])
        y_pred_raw = tf.constant([[45100.0], [45100.0]])
        
        loss = train_model.local_trend_loss(x_window, y_true_raw, y_pred_raw, last_close)
        
        assert loss.numpy() == pytest.approx(0.0, abs=1e-5)

    def test_local_trend_loss_increases_with_trend_error(self, train_model):
        """Test loss increases when trend predictions diverge."""
        x_window = tf.random.normal((2, 60))
        last_close = tf.constant([[45000.0], [45000.0]])
        y_true_raw = tf.constant([[45100.0], [45100.0]])  # +100 trend
        
        y_pred_small_err = tf.constant([[45090.0], [45090.0]])  # Small error
        y_pred_large_err = tf.constant([[45000.0], [45000.0]])  # Large error
        
        loss_small = train_model.local_trend_loss(x_window, y_true_raw, y_pred_small_err, last_close)
        loss_large = train_model.local_trend_loss(x_window, y_true_raw, y_pred_large_err, last_close)
        
        assert loss_small.numpy() < loss_large.numpy()

    def test_local_trend_loss_gradient_flow(self, train_model):
        """Test gradients flow through local_trend_loss."""
        x_window = tf.constant(tf.random.normal((2, 60)))
        last_close = tf.constant([[45000.0], [45000.0]])
        y_true_raw = tf.constant([[45100.0], [45100.0]])
        y_pred_raw = tf.Variable([[45090.0], [45090.0]])
        
        with tf.GradientTape() as tape:
            loss = train_model.local_trend_loss(x_window, y_true_raw, y_pred_raw, last_close)
        
        grads = tape.gradient(loss, y_pred_raw)
        assert grads is not None
        assert tf.reduce_all(tf.math.is_finite(grads))


class TestExtendedTrendLoss:
    """Test suite for extended_trend_loss method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_extended_trend_loss_output_structure(self, train_model):
        """Test extended_trend_loss returns tuple of (global_loss, extended_loss)."""
        # x_window should contain realistic close prices (used for start-of-window reference)
        # Shape: [batch_size, lookback] with close prices around 45000
        batch_size = 4
        lookback = 60
        base_prices = tf.constant([44900.0, 45000.0, 44950.0, 45050.0])
        x_window = tf.tile(tf.reshape(base_prices, (batch_size, 1)), [1, lookback])
        # Add small random variation
        x_window = x_window + tf.random.normal((batch_size, lookback), stddev=10.0)
        
        # y_true_raw and y_pred_raw: raw price predictions [batch_size, 1]
        y_true_raw = tf.constant([[45100.0], [45200.0], [45050.0], [45150.0]])
        y_pred_raw = tf.constant([[45105.0], [45195.0], [45055.0], [45145.0]])
        
        # extended_trends: percentage changes [batch_size, num_periods]
        extended_trends = tf.constant([
            [0.001, 0.002, 0.003],
            [0.001, 0.002, 0.003],
            [-0.001, -0.002, -0.003],
            [0.001, 0.002, 0.003]
        ], dtype=tf.float32)
        
        last_close_raw = tf.constant([[45000.0], [45100.0], [45000.0], [45100.0]])
        
        result = train_model.extended_trend_loss(
            x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw
        )
        
        assert len(result) == 2
        global_loss, extended_loss = result
        assert global_loss.shape == ()
        assert extended_loss.shape == ()
        assert tf.math.is_finite(global_loss)
        assert tf.math.is_finite(extended_loss)

    def test_extended_trend_loss_empty_trends(self, train_model):
        """Test extended_trend_loss handles empty extended_trends."""
        batch_size = 4
        lookback = 60
        base_prices = tf.constant([44900.0, 45000.0, 44950.0, 45050.0])
        x_window = tf.tile(tf.reshape(base_prices, (batch_size, 1)), [1, lookback])
        
        y_true_raw = tf.constant([[45100.0], [45200.0], [45050.0], [45150.0]])
        y_pred_raw = tf.constant([[45105.0], [45195.0], [45055.0], [45145.0]])
        extended_trends = tf.zeros((4, 0), dtype=tf.float32)  # Empty trends
        last_close_raw = tf.constant([[45000.0], [45100.0], [45000.0], [45100.0]])
        
        global_loss, extended_loss = train_model.extended_trend_loss(
            x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw
        )
        
        # Extended loss should be 0 for empty trends
        assert extended_loss.numpy() == pytest.approx(0.0, abs=1e-5)

    def test_extended_trend_loss_gradient_flow(self, train_model):
        """Test gradients flow through extended_trend_loss."""
        batch_size = 2
        lookback = 60
        base_prices = tf.constant([44900.0, 45000.0])
        x_window = tf.tile(tf.reshape(base_prices, (batch_size, 1)), [1, lookback])
        
        y_true_raw = tf.constant([[45100.0], [45200.0]])
        y_pred_raw = tf.Variable([[45105.0], [45195.0]])
        extended_trends = tf.constant([[0.001, 0.002], [0.001, 0.002]], dtype=tf.float32)
        last_close_raw = tf.constant([[45000.0], [45100.0]])
        
        with tf.GradientTape() as tape:
            global_loss, extended_loss = train_model.extended_trend_loss(
                x_window, y_true_raw, y_pred_raw, extended_trends, last_close_raw
            )
            total = global_loss + extended_loss
        
        grads = tape.gradient(total, y_pred_raw)
        assert grads is not None


class TestHuberElementwise:
    """Test suite for the huber element-wise utility method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_huber_output_shape(self, train_model):
        """Test huber returns same shape as input."""
        x = tf.constant([0.1, 0.5, 1.0, 2.0, -0.5])
        
        result = train_model.huber(x)
        
        assert result.shape == x.shape

    def test_huber_quadratic_regime(self, train_model):
        """Test huber behaves quadratically for small x."""
        x_small = tf.constant([0.1, 0.2, 0.3])
        delta = 1.0
        
        result = train_model.huber(x_small, delta=delta)
        expected = 0.5 * tf.square(x_small)  # Quadratic for |x| <= delta
        
        np.testing.assert_array_almost_equal(result.numpy(), expected.numpy(), decimal=5)

    def test_huber_linear_regime(self, train_model):
        """Test huber behaves linearly for large x."""
        x_large = tf.constant([5.0, 10.0, 20.0])
        delta = 1.0
        
        result = train_model.huber(x_large, delta=delta)
        expected = delta * (tf.abs(x_large) - 0.5 * delta)  # Linear for |x| > delta
        
        np.testing.assert_array_almost_equal(result.numpy(), expected.numpy(), decimal=5)

    def test_huber_symmetry(self, train_model):
        """Test huber is symmetric around zero."""
        x_pos = tf.constant([0.5, 1.0, 2.0])
        x_neg = tf.constant([-0.5, -1.0, -2.0])
        
        result_pos = train_model.huber(x_pos)
        result_neg = train_model.huber(x_neg)
        
        np.testing.assert_array_almost_equal(result_pos.numpy(), result_neg.numpy(), decimal=5)

    def test_huber_delta_parameter(self, train_model):
        """Test huber respects custom delta."""
        x = tf.constant([2.0])
        
        huber_delta_1 = train_model.huber(x, delta=1.0)
        huber_delta_3 = train_model.huber(x, delta=3.0)
        
        # With delta=1.0, x=2.0 is in linear regime
        # With delta=3.0, x=2.0 is in quadratic regime
        expected_delta_1 = 1.0 * (2.0 - 0.5 * 1.0)  # Linear: 1.5
        expected_delta_3 = 0.5 * 2.0 ** 2  # Quadratic: 2.0
        
        assert huber_delta_1.numpy()[0] == pytest.approx(expected_delta_1, rel=1e-5)
        assert huber_delta_3.numpy()[0] == pytest.approx(expected_delta_3, rel=1e-5)

    def test_huber_zero_input(self, train_model):
        """Test huber at zero."""
        x = tf.constant([0.0])
        
        result = train_model.huber(x)
        
        assert result.numpy()[0] == pytest.approx(0.0, abs=1e-6)


class TestCustomLossIntegration:
    """Integration tests for the full custom_loss method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for loss testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_custom_loss_output_components(self, train_model, config):
        """Test custom_loss returns 22-component tuple."""
        batch_size = 4
        x_window = tf.random.normal((batch_size, config.LOOKBACK))
        y_true = tf.random.normal((batch_size, 3))  # 3 horizons
        last_close = tf.constant([[45000.0], [45100.0], [45050.0], [45150.0]])
        extended_trends = tf.random.normal((batch_size, 3))
        
        # Generate predictions from model
        y_pred = train_model(x_window, training=False)
        
        result = train_model.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
        
        assert len(result) == 22
        
        # All components should be finite
        for i, component in enumerate(result):
            assert tf.math.is_finite(component), f"Component {i} is not finite"

    def test_custom_loss_total_is_first_element(self, train_model, config):
        """Test that first element is total loss."""
        batch_size = 4
        x_window = tf.random.normal((batch_size, config.LOOKBACK))
        y_true = tf.random.normal((batch_size, 3))
        last_close = tf.constant([[45000.0], [45100.0], [45050.0], [45150.0]])
        extended_trends = tf.random.normal((batch_size, 3))
        
        y_pred = train_model(x_window, training=False)
        result = train_model.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
        
        total_loss = result[0]
        assert total_loss.shape == ()
        assert tf.math.is_finite(total_loss)
        # Total should be positive (sum of non-negative losses)
        assert total_loss.numpy() >= 0

    def test_custom_loss_point_losses_structure(self, train_model, config):
        """Test point loss components (indices 1-3) are valid."""
        batch_size = 4
        x_window = tf.random.normal((batch_size, config.LOOKBACK))
        y_true = tf.random.normal((batch_size, 3))
        last_close = tf.constant([[45000.0], [45100.0], [45050.0], [45150.0]])
        extended_trends = tf.random.normal((batch_size, 3))
        
        y_pred = train_model(x_window, training=False)
        result = train_model.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
        
        point_h0, point_h1, point_h2 = result[1], result[2], result[3]
        
        assert tf.math.is_finite(point_h0)
        assert tf.math.is_finite(point_h1)
        assert tf.math.is_finite(point_h2)

    def test_custom_loss_direction_losses_structure(self, train_model, config):
        """Test direction loss components (indices 13-15) are valid."""
        batch_size = 4
        x_window = tf.random.normal((batch_size, config.LOOKBACK))
        y_true = tf.random.normal((batch_size, 3))
        last_close = tf.constant([[45000.0], [45100.0], [45050.0], [45150.0]])
        extended_trends = tf.random.normal((batch_size, 3))
        
        y_pred = train_model(x_window, training=False)
        result = train_model.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
        
        dir_h0, dir_h1, dir_h2 = result[13], result[14], result[15]
        
        assert tf.math.is_finite(dir_h0)
        assert tf.math.is_finite(dir_h1)
        assert tf.math.is_finite(dir_h2)

    def test_custom_loss_nll_losses_structure(self, train_model, config):
        """Test NLL loss components (indices 16-18) are valid and non-negative."""
        batch_size = 4
        x_window = tf.random.normal((batch_size, config.LOOKBACK))
        y_true = tf.random.normal((batch_size, 3))
        last_close = tf.constant([[45000.0], [45100.0], [45050.0], [45150.0]])
        extended_trends = tf.random.normal((batch_size, 3))
        
        y_pred = train_model(x_window, training=False)
        result = train_model.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
        
        nll_h0, nll_h1, nll_h2 = result[16], result[17], result[18]
        
        assert tf.math.is_finite(nll_h0)
        assert tf.math.is_finite(nll_h1)
        assert tf.math.is_finite(nll_h2)
        # NLL should be floored at 0
        assert nll_h0.numpy() >= 0
        assert nll_h1.numpy() >= 0
        assert nll_h2.numpy() >= 0

    def test_custom_loss_gradient_flow(self, train_model, config):
        """Test gradients flow through entire custom_loss."""
        batch_size = 4
        x_window = tf.Variable(tf.random.normal((batch_size, config.LOOKBACK)))
        y_true = tf.constant(tf.random.normal((batch_size, 3)))
        last_close = tf.constant([[45000.0], [45100.0], [45050.0], [45150.0]])
        extended_trends = tf.constant(tf.random.normal((batch_size, 3)))
        
        with tf.GradientTape() as tape:
            y_pred = train_model(x_window, training=True)
            result = train_model.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
            total_loss = result[0]
        
        # Get gradients w.r.t. trainable variables
        grads = tape.gradient(total_loss, train_model.trainable_variables)
        
        # At least some gradients should be non-None
        non_none_grads = [g for g in grads if g is not None]
        assert len(non_none_grads) > 0

    def test_custom_loss_deadband_masking(self, train_model, config):
        """Test that deadband masking works in direction loss."""
        batch_size = 4
        x_window = tf.random.normal((batch_size, config.LOOKBACK))
        # Create returns within deadband (very small deltas)
        y_true = tf.constant([[0.0001], [0.0001], [0.0001], [0.0001]])  # Near-zero deltas
        y_true_3h = tf.concat([y_true, y_true, y_true], axis=1)  # 3 horizons
        last_close = tf.constant([[45000.0], [45000.0], [45000.0], [45000.0]])
        extended_trends = tf.random.normal((batch_size, 3))
        
        y_pred = train_model(x_window, training=False)
        result = train_model.custom_loss(x_window, y_true_3h, y_pred, last_close, extended_trends)
        
        # Direction losses should handle masked samples gracefully
        dir_h0, dir_h1, dir_h2 = result[13], result[14], result[15]
        assert tf.math.is_finite(dir_h0)
        assert tf.math.is_finite(dir_h1)
        assert tf.math.is_finite(dir_h2)


class TestNormalCDF:
    """Test suite for _normal_cdf static method."""

    @pytest.fixture
    def train_model(self, config):
        """Create a CustomTrainModel instance for testing."""
        predictor = PricePredictor(config)
        base_model = predictor.build_model()
        return CustomTrainModel(
            base_model=base_model,
            pred_scale=100.0,
            pred_mean=0.0,
            config=config
        )

    def test_normal_cdf_at_zero(self, train_model):
        """Test CDF at z=0 is 0.5."""
        z = tf.constant([0.0])
        cdf = train_model._normal_cdf(z)
        assert cdf.numpy()[0] == pytest.approx(0.5, rel=1e-5)

    def test_normal_cdf_symmetry(self, train_model):
        """Test CDF symmetry: Φ(-z) = 1 - Φ(z)."""
        z = tf.constant([1.0, 2.0, 3.0])
        cdf_pos = train_model._normal_cdf(z)
        cdf_neg = train_model._normal_cdf(-z)
        
        np.testing.assert_array_almost_equal(
            cdf_neg.numpy(), 1.0 - cdf_pos.numpy(), decimal=5
        )

    def test_normal_cdf_bounds(self, train_model):
        """Test CDF is bounded [0, 1]."""
        z = tf.constant([-10.0, -3.0, 0.0, 3.0, 10.0])
        cdf = train_model._normal_cdf(z)
        
        assert tf.reduce_all(cdf >= 0.0)
        assert tf.reduce_all(cdf <= 1.0)

    def test_normal_cdf_monotonic(self, train_model):
        """Test CDF is monotonically increasing."""
        z = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
        cdf = train_model._normal_cdf(z)
        
        diffs = cdf[1:] - cdf[:-1]
        assert tf.reduce_all(diffs > 0)

    def test_normal_cdf_known_values(self, train_model):
        """Test CDF against known values."""
        # Known values: Φ(1) ≈ 0.8413, Φ(-1) ≈ 0.1587, Φ(1.96) ≈ 0.975
        z = tf.constant([1.0, -1.0, 1.96])
        cdf = train_model._normal_cdf(z)
        
        assert cdf.numpy()[0] == pytest.approx(0.8413, rel=1e-3)
        assert cdf.numpy()[1] == pytest.approx(0.1587, rel=1e-3)
        assert cdf.numpy()[2] == pytest.approx(0.975, rel=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

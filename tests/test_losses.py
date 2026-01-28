import tensorflow as tf
import numpy as np
from losses import Losses


def test_registry_has_registered_losses():
    # Ensure the important losses have been registered
    keys = set(Losses.list())
    expected = {
        'focal_loss', 'dice_loss', 'combined_direction_loss', 'point_huber',
        'local_trend_loss', 'extended_trend_loss', 'custom_loss'
    }
    assert expected.issubset(keys), f"Missing expected losses: {expected - keys}"


def test_as_logging_dict_with_dict_and_tensor():
    # Dict case: ensure 'loss' is created/summed if missing
    out = {'point_loss': tf.constant(1.0), 'dir_loss': tf.constant(0.5)}
    d = Losses.as_logging_dict(out)
    assert 'loss' in d
    assert hasattr(d['loss'], 'numpy')  # tensor-like

    # Tensor scalar case
    t = tf.constant(2.0)
    d2 = Losses.as_logging_dict(t)
    assert d2['loss'].numpy() == 2.0


def test_point_huber_executes_with_dummy_self():
    # Prepare realistic price predictions with actual errors
    # Simulate BTC price deltas: true changes vs predicted changes
    y_true = tf.constant([[0.5], [1.2], [-0.8], [2.0], [0.3]], dtype=tf.float32)
    y_pred = tf.constant([[0.3], [1.0], [-0.5], [1.5], [0.4]], dtype=tf.float32)

    # Create a dummy object exposing the required method
    class Dummy:
        def _reduce_mean(self, x):
            return tf.reduce_mean(tf.cast(x, tf.float32))

    dummy = Dummy()

    fn = Losses.get('point_huber')
    # Unbound function: call with dummy instance
    result = fn(dummy, y_true, y_pred)
    assert isinstance(result, tf.Tensor)
    assert float(result.numpy()) >= 0.0

    # Test mathematical property: Huber loss should be smooth (no discontinuities)
    # With these inputs, errors are: [0.2, 0.2, -0.3, 0.5, -0.1]
    # All within typical Huber threshold, so should use quadratic region
    loss_value = float(result.numpy())
    assert 0.05 < loss_value < 0.5, f"Huber loss should be in reasonable range, got {loss_value}"


def test_registry_functions_from_module():
    # Ensure the registered functions come from the centralized module
    f = Losses.get('point_huber')
    assert f.__module__ == 'losses'


def test_custom_loss_smoke_runs():
    from model import CustomTrainModel, Config
    # Instantiate a model with minimal settings
    cfg = Config()
    m = CustomTrainModel(base_model=None, pred_scale=1.0, pred_mean=0.0,
                         lambda_point=1.0, lambda_local_trend=1.0, lambda_global_trend=0.2,
                         lambda_extended_trend=0.16, lambda_dir=1.0, config=cfg)

    B = 4
    num_horizons = cfg.num_horizons
    horizon_keys = cfg.horizon_keys

    # Use realistic inputs (not all zeros!)
    # Simulate actual price sequence
    x_window = tf.constant([[[50000.0]], [[50100.0]], [[49900.0]], [[50200.0]]], dtype=tf.float32)

    # True price changes: mix of positive and negative
    y_true = tf.constant([[0.5, 1.0, 1.5],
                          [0.3, 0.8, 1.2],
                          [-0.2, -0.5, -1.0],
                          [0.1, 0.2, 0.3]], dtype=tf.float32)

    last_close = tf.constant([[50000.0], [50100.0], [49900.0], [50200.0]], dtype=tf.float32)

    # Extended trends with realistic values
    extended_trends = tf.constant([[0.01, 0.02, 0.03],
                                   [0.006, 0.016, 0.024],
                                   [-0.004, -0.010, -0.020],
                                   [0.002, 0.004, 0.006]], dtype=tf.float32)

    # Construct y_pred dynamically with realistic values (not perfect predictions!)
    y_pred = []
    for i in range(num_horizons):
        # Predictions with some error
        price = tf.constant([[0.4], [0.5], [-0.1], [0.2]], dtype=tf.float32)  # Differs from y_true
        direction = tf.constant([[0.7], [0.8], [0.3], [0.6]], dtype=tf.float32)  # Not all 0.5
        variance = tf.constant([[0.3], [0.4], [0.6], [0.5]], dtype=tf.float32)  # Varied
        y_pred.extend([price, direction, variance])

    y_pred = tuple(y_pred)

    out = m.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)

    # Now custom_loss returns a dict
    assert isinstance(out, dict)
    assert 'total' in out

    # Check that horizon-specific keys exist
    for h_key in horizon_keys:
        assert f'point_loss_{h_key}' in out
        assert f'dir_loss_{h_key}' in out
        assert f'nll_{h_key}' in out

    # Validate loss values are reasonable (not zero, not NaN, not Inf)
    total_loss = float(out['total'].numpy())
    assert not tf.math.is_nan(out['total']), "Total loss should not be NaN"
    assert not tf.math.is_inf(out['total']), "Total loss should not be Inf"
    assert total_loss > 0, f"Total loss should be positive with non-perfect predictions, got {total_loss}"

    # Individual loss components should be finite
    for h_key in horizon_keys:
        point_loss = float(out[f'point_loss_{h_key}'].numpy())
        dir_loss = float(out[f'dir_loss_{h_key}'].numpy())
        nll = float(out[f'nll_{h_key}'].numpy())

        assert not np.isnan(point_loss), f"Point loss for {h_key} should not be NaN"
        assert not np.isnan(dir_loss), f"Direction loss for {h_key} should not be NaN"
        assert not np.isnan(nll), f"NLL for {h_key} should not be NaN"
        assert point_loss >= 0, f"Point loss for {h_key} should be non-negative"

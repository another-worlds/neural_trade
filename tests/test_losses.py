import tensorflow as tf
from losses import Losses


def test_registry_has_registered_losses():
    # Ensure the important losses have been registered
    keys = set(Losses.list_names())
    expected = {
        'focal_loss', 'dice_loss', 'combined_direction_loss', 'point_huber',
        'local_trend_loss', 'extended_trend_loss', 'custom_loss',
        't_perp_calibration_loss', 'casimir_interference_loss',
        'vacuum_bandwidth_loss', 'hyper_decoherence_coupling_loss',
        'information_flow_entropy_loss', 'vacuum_overflow_t_perp_loss',
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
    # Prepare a small batch with shape [B, 1]
    y_true = tf.constant([[0.1], [0.2]], dtype=tf.float32)
    y_pred = tf.constant([[0.0], [0.1]], dtype=tf.float32)

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


def test_registry_functions_from_module():
    # Ensure the registered functions come from the centralized module
    f = Losses.get('point_huber')
    assert f.__module__ == 'losses'


def test_custom_loss_smoke_runs():
    from model import CustomTrainModel, Config
    # Instantiate a model with minimal settings
    m = CustomTrainModel(base_model=None, pred_scale=1.0, pred_mean=0.0,
                         lambda_point=1.0, lambda_local_trend=1.0, lambda_global_trend=0.2,
                         lambda_extended_trend=0.16, lambda_dir=1.0, config=Config())

    B = 2
    x_window = tf.zeros([B, 1, 1], dtype=tf.float32)
    y_true = tf.zeros([B, 3], dtype=tf.float32)
    last_close = tf.ones([B, 1], dtype=tf.float32)
    extended_trends = tf.zeros([B, 3], dtype=tf.float32)

    # Construct y_pred as tuple of nine tensors: price, dir, var per horizon
    price_h0 = tf.zeros([B, 1], dtype=tf.float32)
    dir_h0 = tf.fill([B, 1], 0.5)
    var_h0 = tf.fill([B, 1], 0.5)
    price_h1 = tf.zeros([B, 1], dtype=tf.float32)
    dir_h1 = tf.fill([B, 1], 0.5)
    var_h1 = tf.fill([B, 1], 0.5)
    price_h2 = tf.zeros([B, 1], dtype=tf.float32)
    dir_h2 = tf.fill([B, 1], 0.5)
    var_h2 = tf.fill([B, 1], 0.5)

    y_pred = (price_h0, dir_h0, var_h0, price_h1, dir_h1, var_h1, price_h2, dir_h2, var_h2)

    out = m.custom_loss(x_window, y_true, y_pred, last_close, extended_trends)
    assert isinstance(out, tuple)
    assert len(out) >= 4

"""The Losses registry: what is registered, the logging helper, and that it hands out production code.

Numerical checks of each term live in test_losses_reference.py / test_custom_loss.py /
test_qbox_losses.py.
"""
import tensorflow as tf
from neural_trade.registries.losses import Losses


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


def test_registry_functions_from_module():
    # The registry must hand out the production function itself - not a copy from a stale
    # module (registries/losses.py used to be one, and its tests checked dead code).
    import neural_trade.losses.functions as functions

    assert Losses.get('point_huber') is functions.point_huber
    assert Losses.get_objective() is functions.custom_loss

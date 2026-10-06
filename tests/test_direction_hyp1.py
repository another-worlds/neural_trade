"""Tactical hyp1: direction-head switches (default off = today's graph bit-for-bit; on = changes)."""
import numpy as np
import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.models.registry import Models

DIR = (1, 4, 7)


def _build(name="gru_attention", **kw):
    tf.keras.utils.set_random_seed(0)
    return Models.build(name, Config(LOOKBACK=32, **kw))



def _dirs(m, x, training=False):
    out = m(x, training=training)
    return [out[i].numpy() for i in DIR]


def _x_for(m):
    shape = (16,) + tuple(m.input_shape[1:])
    return np.random.default_rng(1).normal(size=shape).astype("float32")


def test_defaults_are_today():
    c = Config()
    assert (c.DIRECTION_HEAD_MODE, c.DIRECTION_DEEP_SHRINK, c.DIRECTION_DEEP_DROPOUT) == ("mixed", 0.0, 0.0)


@pytest.mark.parametrize("name", ["gru_attention", "gru_small"])
def test_explicit_off_is_bit_identical_to_default(name):
    a = _build(name)
    b = _build(name, DIRECTION_HEAD_MODE="mixed", DIRECTION_DEEP_SHRINK=0.0, DIRECTION_DEEP_DROPOUT=0.0)
    assert [tuple(w.shape) for w in a.weights] == [tuple(w.shape) for w in b.weights]
    x = _x_for(a)
    for u, v in zip(_dirs(a, x), _dirs(b, x)):
        np.testing.assert_array_equal(u, v)
    assert len(a.losses) == len(b.losses)


@pytest.mark.parametrize("name", ["gru_attention", "gru_small"])
def test_skip_only_deep_path_contributes_zero(name):
    m = _build(name, DIRECTION_HEAD_MODE="skip_only")
    x = _x_for(m)
    for h in ("h0", "h1", "h2"):
        layer = m.get_layer(f"direction_{h}_logit")
        assert not layer.trainable_weights
        np.testing.assert_array_equal(layer.get_weights()[0], 0.0)
    probe = tf.keras.Model(m.input, [m.get_layer(f"direction_h{i}_logit").output for i in range(3)]
                           + [m.get_layer(f"direction_h{i}_skip").output for i in range(3)])
    o = probe(x)
    for i in range(3):
        np.testing.assert_array_equal(o[i].numpy(), 0.0)
    skip_w = m.get_layer("direction_h0_skip").trainable_weights
    assert len(skip_w) == 2  # kernel + bias
    assert not np.array_equal(_dirs(m, x)[0], _dirs(_build(name), x)[0])


def test_skip_only_needs_skip():
    with pytest.raises(ValueError):
        Config(DIRECTION_HEAD_MODE="skip_only", DIRECTION_SKIP=False).validate()


def test_deep_shrink_adds_activity_penalty_only_when_on():
    off = _build()
    on = _build(DIRECTION_DEEP_SHRINK=1.0)
    x = _x_for(off)
    off(x)
    on(x)
    assert len(on.losses) == len(off.losses) + 3
    assert float(tf.add_n(on.losses)) != float(tf.add_n(off.losses))
    for u, v in zip(_dirs(off, x), _dirs(on, x)):  # same weights/seed: the forward pass is unchanged
        np.testing.assert_array_equal(u, v)


def test_deep_dropout_only_in_training():
    off = _build()
    on = _build(DIRECTION_DEEP_DROPOUT=0.5)
    x = _x_for(off)
    for u, v in zip(_dirs(off, x), _dirs(on, x)):  # inference identical
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(_dirs(on, x, True)[0], _dirs(on, x)[0])

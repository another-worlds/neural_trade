"""Tactical switch DIRECTION_INDICATOR_SKIP (exploratory; default off = today's graph bit-for-bit)."""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config  # import neural_trade before tensorflow (CUDA DLLs on PATH)
from neural_trade.core.exceptions import InvalidConfigurationError

from tests.test_direction_geometry import _fixed_x, _model
from tests.test_noprice_switches import _build, _var_names

DIR = (1, 4, 7)


def test_default_is_off_and_explicit_off_is_bit_identical(tf):
    assert Config().DIRECTION_INDICATOR_SKIP is False
    a = _model(tf)
    tf.keras.backend.clear_session()  # Keras auto-numbering must not differ between the two builds
    b = _model(tf, DIRECTION_INDICATOR_SKIP=False)
    assert a.count_params() == b.count_params() == 316751 and len(a.layers) == len(b.layers) == 87
    assert [ly.name for ly in a.layers] == [ly.name for ly in b.layers]
    x = _fixed_x()
    for u, v in zip(a(x, training=False), b(x, training=False)):
        np.testing.assert_array_equal(u.numpy(), v.numpy())


def test_skip_only_needs_some_skip():
    Config(DIRECTION_HEAD_MODE="skip_only", DIRECTION_SKIP=False, DIRECTION_INDICATOR_SKIP=True).validate()
    with pytest.raises(InvalidConfigurationError):
        Config(DIRECTION_HEAD_MODE="skip_only", DIRECTION_SKIP=False)


@pytest.mark.parametrize("over", [dict(), dict(DIRECTION_HEAD_MODE="skip_only"),
                                  dict(INDICATOR_GRAD_SOURCE="direction"), dict(DIRECTION_SKIP=False)])
def test_switch_builds_outputs_finite_direction_in_unit_interval(tf, over):
    base = _model(tf, **{k: v for k, v in over.items() if k != "INDICATOR_GRAD_SOURCE"})
    m = _model(tf, DIRECTION_INDICATOR_SKIP=True, **over)
    kernel = m.get_layer("direction_h0_skip").kernel
    n_ch = kernel.shape[0] - (0 if over.get("DIRECTION_SKIP") is False else base.get_layer("direction_h0_skip").kernel.shape[0])
    assert n_ch > 0
    out = m(_fixed_x(), training=False)
    assert len(out) == 10
    for o in out:
        assert bool(tf.reduce_all(tf.math.is_finite(o)))
    for i in DIR:
        assert float(tf.reduce_min(out[i])) >= 0.0 and float(tf.reduce_max(out[i])) <= 1.0


def test_parameter_count_change_is_the_new_skip_rows(tf):
    base, m = _model(tf), _model(tf, DIRECTION_INDICATOR_SKIP=True)
    rows = base.get_layer("direction_h0_skip").kernel.shape[0]
    n_ch = m.get_layer("direction_h0_skip").kernel.shape[0] - rows
    assert m.count_params() - base.count_params() == 3 * n_ch  # one kernel column per direction head
    assert m.count_params() > base.count_params()


def test_gradients_reach_new_skip_kernels_and_indicator_periods(tf):
    x = tf.constant(_fixed_x())

    def grads(**over):
        m = _model(tf, DIRECTION_HEAD_MODE="skip_only", **over)
        with tf.GradientTape() as tape:
            out = m(x, training=True)
            loss = tf.add_n([tf.reduce_sum(tf.math.log(out[i] + 1e-6)) for i in DIR])
        ind_vars = m.get_layer("learnable_indicators").trainable_variables
        assert ind_vars
        return m, tape.gradient(loss, [m.get_layer("direction_h1_skip").kernel] + ind_vars)

    _, g_off = grads()
    m, g_on = grads(DIRECTION_INDICATOR_SKIP=True)
    for g in g_on:
        assert g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))
    k_grad = g_on[0].numpy()
    n_old = _model(tf).get_layer("direction_h1_skip").kernel.shape[0]
    assert np.abs(k_grad[n_old:]).max() > 0.0  # the new rows learn
    ind_on = max(float(tf.reduce_max(tf.abs(g))) for g in g_on[1:])
    ind_off = max(float(tf.reduce_max(tf.abs(g))) for g in g_off[1:] if g is not None) if any(
        g is not None for g in g_off[1:]) else 0.0
    assert ind_on > 0.0 and ind_off == 0.0  # indicator periods get a gradient only through the new skip


def test_skip_only_direction_logit_is_exactly_linear_in_the_features(tf):
    m = _model(tf, DIRECTION_HEAD_MODE="skip_only", DIRECTION_INDICATOR_SKIP=True)
    x = _fixed_x()
    feats = tf.keras.Model(m.input, m.get_layer("direction_skip_all_features").output)(x).numpy()
    probe = tf.keras.Model(m.input, [m.get_layer(f"direction_h{i}_skip").output for i in range(3)]
                           + [m.get_layer(f"direction_h{i}_logit").output for i in range(3)])
    o = [t.numpy() for t in probe(x)]
    out = m(x, training=False)
    rng = np.random.default_rng(3)
    for i in range(3):
        layer = m.get_layer(f"direction_h{i}_skip")
        layer.set_weights([rng.normal(size=layer.kernel.shape).astype("float32"),
                           rng.normal(size=(1,)).astype("float32")])
    o = [t.numpy() for t in probe(x)]
    out = m(x, training=False)
    for i in range(3):
        w, b = m.get_layer(f"direction_h{i}_skip").get_weights()
        np.testing.assert_allclose(o[i], feats @ w + b, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(o[3 + i], 0.0)  # deep logit frozen at zero
        np.testing.assert_allclose(out[DIR[i]].numpy(), 1 / (1 + np.exp(-(feats @ w + b))), rtol=1e-4, atol=1e-5)


def test_features_are_window_zscores_of_the_last_bar(tf):
    m = _model(tf, DIRECTION_INDICATOR_SKIP=True)
    x = _fixed_x()
    seq = tf.keras.Model(m.input, m.get_layer("learnable_indicators").output)(x).numpy()
    f = tf.keras.Model(m.input, m.get_layer("direction_indicator_features").output)(x).numpy()
    ref = (seq[:, -1, :] - seq.mean(1)) / np.sqrt(seq.var(1) + 1e-8)
    np.testing.assert_allclose(f, ref, rtol=1e-3, atol=1e-3)


def test_trains_two_steps_with_price_head_none(tf, tiny_close_only_config, tmp_path, synthetic_bars):
    model, train_ds = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, PRICE_HEAD="none",
                             DIRECTION_INDICATOR_SKIP=True, DIRECTION_HEAD_MODE="skip_only",
                             INDICATOR_GRAD_SOURCE="direction")
    assert any(n.startswith("direction_h1_skip") for n in _var_names(model))
    hist = model.fit(train_ds, epochs=1, steps_per_epoch=2, verbose=0)
    assert np.isfinite(hist.history["loss"][-1]) and hist.history["nonfinite_grad_steps"][-1] == 0
    assert all(np.all(np.isfinite(w)) for w in model.get_weights())

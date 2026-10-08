"""Tactical switches DIRECTION_REGIME_GATE and INDICATOR_GEOMETRY (exploratory; both default off = today's graph)."""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config  # import neural_trade before tensorflow (CUDA DLLs on PATH)
from neural_trade.core.exceptions import InvalidConfigurationError

from tests.test_noprice_switches import _build, _var_names

# parameter count added by each switch on the default (OHLCV, 14 families, 82 channels, DIRECTION_SKIP) graph
EXPECTED_EXTRA = {"geom": 2050, "gate": 222, "both": 13342}


def _fixed_x():
    return np.random.default_rng(1).normal(0, 1, (8, 60, 5)).astype(np.float32).cumsum(1)


def _model(tf, **over):
    from neural_trade.models.gru_attention import build_gru_attention

    tf.keras.utils.set_random_seed(0)
    return build_gru_attention(Config(**over))


def test_config_defaults_and_validation():
    cfg = Config()
    assert cfg.DIRECTION_REGIME_GATE is False and cfg.INDICATOR_GEOMETRY is False and cfg.GEOM_SLOPE_BARS == [3, 10]
    for bad in ([], [0], [60], [True]):
        with pytest.raises(InvalidConfigurationError):
            Config(INDICATOR_GEOMETRY=True, GEOM_SLOPE_BARS=bad)
    with pytest.raises(InvalidConfigurationError):
        Config(DIRECTION_REGIME_GATE=True, DIRECTION_HEAD_MODE="skip_only")


def test_defaults_are_the_base_commits_graph_and_outputs(tf):
    """Recorded on the base commit (nt-tactical 092ee24): same parameters, layers and outputs (seed 0)."""
    m = _model(tf)
    assert m.count_params() == 316751 and len(m.layers) == 87
    out = m(_fixed_x(), training=False)
    sums = [11.64286, 2.939623, 10.72249, -3.54005, 3.562382, 8.0642, 2.825687, 3.948166, 12.404559, 0.0]
    first = [1.309363, 3e-06, 0.577187, -1.114922, 0.744113, 0.714312, 0.624923, 0.628457, 1.940374, 0.0]
    for o, s, f in zip(out, sums, first):
        assert float(np.sum(o)) == pytest.approx(s, rel=1e-5, abs=1e-5)
        assert float(o[0, 0]) == pytest.approx(f, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("over", [dict(INDICATOR_GEOMETRY=True), dict(DIRECTION_REGIME_GATE=True),
                                  dict(INDICATOR_GEOMETRY=True, DIRECTION_REGIME_GATE=True)])
def test_switch_builds_and_outputs_are_valid(tf, over):
    base = _model(tf).count_params()
    m = _model(tf, **over)
    assert m.count_params() > base
    out = m(_fixed_x(), training=False)
    assert len(out) == 10
    for o in out:
        assert bool(tf.reduce_all(tf.math.is_finite(o)))
    for i in (1, 4, 7):
        assert float(tf.reduce_min(out[i])) >= 0.0 and float(tf.reduce_max(out[i])) <= 1.0


def test_parameter_counts_are_recorded(tf):
    base = _model(tf).count_params()
    got = {k: _model(tf, **v).count_params() - base for k, v in {
        "geom": dict(INDICATOR_GEOMETRY=True), "gate": dict(DIRECTION_REGIME_GATE=True),
        "both": dict(INDICATOR_GEOMETRY=True, DIRECTION_REGIME_GATE=True)}.items()}
    assert got == EXPECTED_EXTRA


def test_gate_reads_but_does_not_train_the_variance_head(tf):
    """The direction output has zero gradient w.r.t. the variance head's weights (stop_gradient)."""
    m = _model(tf, DIRECTION_REGIME_GATE=True)
    var_layer = m.get_layer("variance_h1")
    with tf.GradientTape() as tape:
        out = m(_fixed_x(), training=False)
        loss = tf.reduce_sum(out[4])
    g = tape.gradient(loss, var_layer.trainable_variables)
    assert all(gi is None or float(tf.reduce_max(tf.abs(gi))) == 0.0 for gi in g)


def test_geometry_layer_is_causal_differentiable_and_has_the_documented_shape(tf):
    from neural_trade.registries.layers import Layers

    lay = Layers.build("indicator_geometry", slope_bars=[3, 10])
    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(0, 1, (4, 60, 6)).astype(np.float32).cumsum(1))
    f = lay(x)
    assert f.shape == (4, 6 * 5)
    x2 = x.numpy().copy()
    x2[:, -1, :] += 0.7  # perturb the window's last bar: the features move
    assert float(np.max(np.abs(lay(tf.constant(x2)).numpy() - f.numpy()))) > 1e-3
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.reduce_sum(lay(x) * tf.range(30, dtype=tf.float32))
    g = tape.gradient(y, x)
    assert bool(tf.reduce_all(tf.math.is_finite(g))) and float(tf.reduce_max(tf.abs(g))) > 0.0
    lay = Layers.build("indicator_geometry", slope_bars=[3, 10])
    const = tf.ones((2, 60, 3))  # a constant channel: finite output and gradient
    with tf.GradientTape() as tape:
        tape.watch(const)
        y = tf.reduce_sum(lay(const))
    assert bool(tf.reduce_all(tf.math.is_finite(tape.gradient(y, const))))


def test_geometry_features_follow_their_definitions(tf):
    from neural_trade.models.layers import IndicatorGeometry

    lay = IndicatorGeometry(slope_bars=[3])
    lay.norm = lambda t: t  # look at the raw features
    n = 40
    t = np.arange(n, dtype=np.float32)
    close = 100 + t + np.where(t % 2 == 0, 0.5, -0.5)  # a trend with 1-bar noise
    ma = close - 2.0                                    # a line 2 below the close
    x = tf.constant(np.stack([ma, close], axis=-1)[None].astype(np.float32))
    f = lay(x).numpy()[0]  # [slope_3 x2 channels, distance x2, cross x2, squeeze x2]
    s_c = np.sqrt(np.var(np.diff(close)) + 1e-8)
    assert f[0] == pytest.approx((ma[-1] - ma[-4]) / (3 * np.sqrt(np.var(np.diff(ma)) + 1e-8)), rel=1e-4)
    assert f[2] == pytest.approx(np.arcsinh(2.0 / s_c), rel=1e-4)  # distance of the close to the line
    assert f[3] == pytest.approx(0.0, abs=1e-5)                    # distance of the close to itself
    assert f[5] == pytest.approx(0.0, abs=1e-5)                    # the close never "crosses" itself
    assert f[4] == pytest.approx(0.0, abs=1e-3)                    # a line parallel below the close: no crossing
    sq = np.std(ma[-10:]) / np.std(ma)
    assert f[6] == pytest.approx(sq, rel=1e-3)


def _grads(tf, cfg, tmp_path, synthetic_bars, **over):
    model, train_ds = _build(tf, cfg, tmp_path, synthetic_bars, **over)
    x, y, lc, ext = next(iter(train_ds))
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        c = model.custom_loss(x, y, out[:9], lc, ext, vacuum_overflow=out[9])
    grads = tape.gradient(c.total, model.trainable_variables)
    return model, train_ds, dict(zip(_var_names(model), grads))


def test_gradients_are_finite_and_reach_the_new_layers_and_the_indicator_logits(
        tf, tiny_close_only_config, tmp_path, synthetic_bars):
    model, train_ds, by = _grads(tf, tiny_close_only_config, tmp_path, synthetic_bars,
                                 INDICATOR_GEOMETRY=True, DIRECTION_REGIME_GATE=True)
    for n, g in by.items():
        if g is not None:
            assert bool(tf.reduce_all(tf.math.is_finite(g))), n
    new = [n for n in by if ("direction_h" in n and any(k in n for k in ("_gate", "_A_", "_B_")))
           and "kernel" in n]
    assert new
    for n in new:
        assert by[n] is not None and float(tf.reduce_max(tf.abs(by[n]))) > 0.0, n
    # geometry alone reaches the indicator period logits
    x, *_ = next(iter(train_ds))
    logit_vars = [v for v in model.trainable_variables if v.name.startswith("learnable_indicators/")]
    assert logit_vars
    sub = tf.keras.Model(model.inputs, model.get_layer("indicator_geometry").output)
    with tf.GradientTape() as tape:
        out = sub(x, training=True)
        loss = tf.reduce_sum(out * tf.range(out.shape[-1], dtype=tf.float32))
    gs = tape.gradient(loss, logit_vars)
    assert any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0.0 for g in gs)
    assert all(g is None or bool(tf.reduce_all(tf.math.is_finite(g))) for g in gs)


def test_both_switches_with_price_none_and_active_horizons_train_two_steps(
        tf, tiny_close_only_config, tmp_path, synthetic_bars):
    model, train_ds = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, INDICATOR_GEOMETRY=True,
                             DIRECTION_REGIME_GATE=True, PRICE_HEAD="none", ACTIVE_HORIZONS=[0, 2])
    names = _var_names(model)
    assert not any(n.startswith("price_h") for n in names)
    assert not any(n.startswith("direction_h1") for n in names)
    assert any(n.startswith("direction_h2_gate") for n in names)
    hist = model.fit(train_ds, epochs=1, steps_per_epoch=2, verbose=0)
    assert np.isfinite(hist.history["loss"][-1]) and hist.history["nonfinite_grad_steps"][-1] == 0
    assert all(np.all(np.isfinite(w)) for w in model.get_weights())


def test_geometry_only_with_price_none_trains(tf, tiny_close_only_config, tmp_path, synthetic_bars):
    model, train_ds = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, INDICATOR_GEOMETRY=True,
                             PRICE_HEAD="none")
    hist = model.fit(train_ds, epochs=1, steps_per_epoch=2, verbose=0)
    assert np.isfinite(hist.history["loss"][-1]) and hist.history["nonfinite_grad_steps"][-1] == 0

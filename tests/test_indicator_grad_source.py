"""Tactical switch INDICATOR_GRAD_SOURCE ('total' default | 'direction').

With 'direction' the learned-indicator variables follow the gradient of the direction term only
(lambda_dir_outer * lambda_dir * sum of the per-horizon direction BCE); every other variable keeps
the total-loss gradient. Checked with plain SGD (update = -lr * grad) against a manual tape.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config  # import neural_trade before tensorflow
from neural_trade.core.exceptions import InvalidConfigurationError

LR = 1e-3


def test_default_and_validation():
    assert Config().INDICATOR_GRAD_SOURCE == "total"
    assert Config(INDICATOR_GRAD_SOURCE="direction").INDICATOR_GRAD_SOURCE == "direction"
    for bad in ("dir", "Total", "", "both"):
        with pytest.raises(InvalidConfigurationError):
            Config(INDICATOR_GRAD_SOURCE=bad)


def _batch(in_shape, n=16):
    import tensorflow as tf

    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(0, 1, size=(n,) + tuple(in_shape)).astype(np.float32))
    y = tf.constant(rng.normal(0, 1, size=(n, 3)).astype(np.float32))
    lc = tf.constant((110_000.0 + rng.normal(0, 500, size=(n, 1))).astype(np.float32))
    ext = tf.constant(rng.normal(0, 200, size=(n, 3)).astype(np.float32))
    return x, y, lc, ext


def _model(tf, **over):
    from neural_trade.registries.models import Models
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.utils.seeding import seed_everything

    seed_everything(7)
    tf.keras.backend.clear_session()
    cfg = Config(GRAD_CLIP_NORM=0.0, **over)
    base = Models.build(cfg.MODEL_NAME, cfg)
    m = CustomTrainModel(base_model=base, pred_scale=250.0, pred_mean=0.0, config=cfg,
                         inputs=base.inputs, outputs=base.outputs,
                         indicator_optimizer=tf.keras.optimizers.SGD(LR))
    m.compile(optimizer=tf.keras.optimizers.SGD(LR))
    return m, base.input_shape[1:]


def _manual_grads(tf, m, batch):
    """(total grads, direction-only grads) for every trainable variable, on the current weights."""
    x, y, lc, ext = batch
    from neural_trade.utils.seeding import seed_everything

    seed_everything(7)  # same stateful-RNG draw as the train step's forward pass
    with tf.GradientTape(persistent=True) as tape:
        out = m(x, training=True)
        c = m.custom_loss(x, y, out[:9], lc, ext, vacuum_overflow=out[9])
        d = m.lambda_dir_outer * m.lambda_dir * (c.dir_h0 + c.dir_h1 + c.dir_h2)
    vs = m.trainable_variables
    return tape.gradient(c.total, vs), tape.gradient(d, vs)


def _step_updates(tf, **over):
    m, in_shape = _model(tf, **over)
    batch = _batch(in_shape)
    vs = m.trainable_variables
    before = [v.numpy().copy() for v in vs]
    g_total, g_dir = _manual_grads(tf, m, batch)
    from neural_trade.utils.seeding import seed_everything

    seed_everything(7)
    m.train_step(batch)
    upd = [(b - v.numpy()) for b, v in zip(before, vs)]
    return m, vs, upd, g_total, g_dir


def _expected(g):
    import tensorflow as tf

    return LR * tf.convert_to_tensor(g).numpy()  # IndexedSlices (embeddings) densified


def test_total_is_the_total_gradient_update(tf):
    m, vs, upd, g_total, _ = _step_updates(tf)
    n_ind = 0
    for v, u, g in zip(vs, upd, g_total):
        if g is None:
            continue
        n_ind += id(v) in m._indicator_var_ids
        np.testing.assert_allclose(u, _expected(g), rtol=1e-4, atol=3e-7, err_msg=v.name)
    assert n_ind > 0


def test_direction_source_indicator_update_is_the_direction_gradient(tf):
    m, vs, upd, g_total, g_dir = _step_updates(tf, INDICATOR_GRAD_SOURCE="direction")
    checked_ind, differs = 0, False
    for v, u, gt, gd in zip(vs, upd, g_total, g_dir):
        if id(v) in m._indicator_var_ids:
            if gd is None:
                np.testing.assert_allclose(u, 0.0, atol=1e-9, err_msg=v.name)
                continue
            np.testing.assert_allclose(u, _expected(gd), rtol=1e-4, atol=3e-7, err_msg=v.name)
            checked_ind += 1
            if gt is not None and not np.allclose(_expected(gt), _expected(gd), rtol=1e-3, atol=1e-8):
                differs = True
        elif gt is not None:
            np.testing.assert_allclose(u, _expected(gt), rtol=1e-4, atol=3e-7, err_msg=v.name)
    assert checked_ind > 0
    assert differs, "direction-only gradient equals the total gradient; the test would pin nothing"


def test_direction_source_finite_with_price_head_none_and_active_horizons(tf):
    for over in (dict(PRICE_HEAD="none"), dict(PRICE_HEAD="none", ACTIVE_HORIZONS=[1])):
        m, in_shape = _model(tf, INDICATOR_GRAD_SOURCE="direction", **over)
        batch = _batch(in_shape)
        for _ in range(2):
            logs = m.train_step(batch)
        assert np.isfinite(float(logs["loss"]))
        assert float(logs["nonfinite_grad_steps"]) == 0.0
        assert all(np.all(np.isfinite(v.numpy())) for v in m.trainable_variables)


def test_direction_source_with_geometry_indicator_update_is_the_direction_gradient(tf):
    """The hc6_geomind combination (INDICATOR_GEOMETRY + direction source): the geometry features add a
    second path from the indicator sequence to the direction heads; the indicator variables must still
    move by exactly the direction-only gradient (that path included) and everything else by the total."""
    over = dict(PRICE_HEAD="none", INDICATOR_GEOMETRY=True)
    m, vs, upd, g_total, g_dir = _step_updates(tf, INDICATOR_GRAD_SOURCE="direction", **over)
    checked_ind = 0
    for v, u, gt, gd in zip(vs, upd, g_total, g_dir):
        if id(v) in m._indicator_var_ids:
            if gd is None:
                np.testing.assert_allclose(u, 0.0, atol=1e-9, err_msg=v.name)
                continue
            assert np.all(np.isfinite(u)), v.name
            np.testing.assert_allclose(u, _expected(gd), rtol=1e-4, atol=3e-7, err_msg=v.name)
            checked_ind += 1
        elif gt is not None:
            np.testing.assert_allclose(u, _expected(gt), rtol=1e-4, atol=3e-7, err_msg=v.name)
    assert checked_ind > 0

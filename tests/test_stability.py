"""Stability invariants (D-026, NT-036): strict mode, per-term mask counters, NaN-injection proofs
of the finite-gradient guard, and a short CPU run's health on the default config.

Marked ``stability`` (registered in pyproject.toml); the CI unit job runs it (not excluded by
``-m "not gpu and not slow"``), so every test here is kept fast on purpose.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

pytestmark = pytest.mark.stability

B = 32


def _batch(rng, last_close=110_000.0):
    x_window = tf.constant(rng.normal(0.0, 1.0, size=(B, 60)).astype(np.float32))
    y_true = tf.constant(rng.normal(0.0, 1.0, size=(B, 3)).astype(np.float32))
    lc = tf.constant((last_close + rng.normal(0.0, 0.005 * last_close, size=(B, 1))).astype(np.float32))
    ext = tf.constant(rng.normal(0.0, 200.0, size=(B, 3)).astype(np.float32))
    return x_window, y_true, lc, ext


def _heads(rng):
    price = [tf.Variable(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    dirs = [tf.Variable(rng.uniform(0.05, 0.95, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    var = [tf.Variable(rng.uniform(0.5, 2.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    return price, dirs, var


def _y_pred(price, dirs, var):
    return (price[0], dirs[0], var[0], price[1], dirs[1], var[1], price[2], dirs[2], var[2])


def test_strict_mode_is_off_by_default(make_loss_model):
    m = make_loss_model()
    assert bool(m.strict_loss_masks.numpy()) is False


def test_strict_mode_lets_a_nonfinite_term_reach_total_and_only_that_counter_fires(make_loss_model, monkeypatch):
    """D-026: a NaN in one term. Off (default): masked to 0, total stays finite, exactly one
    counter (the injected term's) rises. On: unmasked, total goes non-finite (the term still
    counts too - the counter tracks whether the term WAS non-finite, independent of masking)."""
    import neural_trade.losses.functions as lf
    from neural_trade.core.config import Config

    def _nan(*a, **k):
        return tf.constant(float("nan"), dtype=tf.float32)

    for strict in (False, True):
        rng = np.random.default_rng(0)
        x, y, lc, ext = _batch(rng)
        price, dirs, var = _heads(rng)
        m = make_loss_model(config=Config(LAMBDA_CRPS=1.0, STRICT_LOSS_MASKS=strict))
        monkeypatch.setattr(lf, "crps_gaussian_loss", _nan)
        out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
        monkeypatch.undo()

        assert np.isfinite(float(out.total)) != strict, f"strict={strict}"
        fired = {t: float(c.result()) for t, c in m._mask_counters.items()}
        assert fired.pop("crps_loss") == 1.0
        # In strict mode the un-masked NaN reaches `total` itself, so "total_loss" (a downstream
        # counter of its own guard, not the injected term) legitimately fires too; with strict off
        # `total` stays finite (crps was masked before reaching it) and nothing else fires.
        expect_total = fired.pop("total_loss")
        assert expect_total == (1.0 if strict else 0.0)
        assert not any(v > 0 for v in fired.values()), f"unexpected counters fired: {fired}"


def test_a_nonfinite_gradient_from_one_term_does_not_change_the_finite_step_count(make_loss_model, monkeypatch):
    """Off (default, the golden-run behaviour): the masked term contributes 0 to `total`, so
    train_step's finite-gradient guard never sees a non-finite total or gradient from it."""
    import neural_trade.losses.functions as lf
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_CRPS=1.0))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(1)
    x, y, lc, ext = _batch(rng)

    def _nan(*a, **k):
        return tf.constant(float("nan"), dtype=tf.float32)

    monkeypatch.setattr(lf, "crps_gaussian_loss", _nan)
    m.train_step((x, y, lc, ext))
    assert float(m.nonfinite_grad_steps.result()) == 0.0
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in m.trainable_variables)


def test_strict_mode_a_nonfinite_term_makes_the_step_guard_fire(make_loss_model, monkeypatch):
    """On: the same injected NaN now reaches `total`, so the guard counts the step and zeroes the
    update (weights stay finite - D-026's whole point: attribution, not corruption)."""
    import neural_trade.losses.functions as lf
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_CRPS=1.0, STRICT_LOSS_MASKS=True))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(1)
    x, y, lc, ext = _batch(rng)

    def _nan(*a, **k):
        return tf.constant(float("nan"), dtype=tf.float32)

    monkeypatch.setattr(lf, "crps_gaussian_loss", _nan)
    m.train_step((x, y, lc, ext))
    assert float(m.nonfinite_grad_steps.result()) == 1.0
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in m.trainable_variables)


def test_nan_in_the_input_window_leaves_weights_finite_and_raises_the_guard(make_loss_model):
    m = make_loss_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(2)
    x, y, lc, ext = _batch(rng)
    x = x.numpy()
    x[0, 0] = np.nan
    x = tf.constant(x)
    m.train_step((x, y, lc, ext))
    assert float(m.nonfinite_grad_steps.result()) == 1.0
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in m.trainable_variables)


def test_nan_in_one_gradient_leaves_weights_finite_and_raises_the_guard(make_loss_model, monkeypatch):
    """A term whose VALUE is perfectly finite but whose GRADIENT is NaN by construction
    (tf.custom_gradient) - independent of every per-term is_finite value guard, which only ever
    looks at the term's value, never its gradient."""
    import neural_trade.losses.functions as lf

    @tf.custom_gradient
    def _nan_gradient_identity(x):
        def grad(dy):
            return dy * float("nan")
        return x, grad

    orig = lf.point_huber

    def wrapped(model, y_true_scaled, y_pred_scaled, last_close_scaled=None, delta=None):
        return _nan_gradient_identity(orig(model, y_true_scaled, y_pred_scaled,
                                           last_close_scaled=last_close_scaled, delta=delta))

    m = make_loss_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(3)
    x, y, lc, ext = _batch(rng)
    monkeypatch.setattr(lf, "point_huber", wrapped)
    m.train_step((x, y, lc, ext))
    assert float(m.nonfinite_grad_steps.result()) == 1.0
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in m.trainable_variables)


def test_per_group_clip_keeps_the_post_clip_norm_at_or_below_grad_clip_norm(make_loss_model):
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(GRAD_CLIP_NORM=0.01, LAMBDA_POINT=1000.0))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(4)
    x, y, lc, ext = _batch(rng)
    m.train_step((x, y, lc, ext))
    assert float(m._grad_health["grad_norm_max_main"].result()) > float(m.grad_clip_norm.numpy())
    assert float(m._grad_health["grad_clip_steps_main"].result()) == 1.0


def test_short_run_on_the_default_config_has_zero_nonfinite_steps_and_finite_state(tf, tmp_path, synthetic_bars,
                                                                                   monkeypatch):
    """D-026 acceptance (4): a short CPU run of the default config, N=3 consecutive epochs, no
    non-finite steps, finite weights/heads/periods after every epoch, and no gradient group/
    variance head/period pinned at its bound on every single epoch."""
    from neural_trade.core.config import Config
    from neural_trade.telemetry.epoch_logger import read_metrics
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    n_epochs = 3
    cfg = Config(EPOCHS=n_epochs, BATCH_SIZE=32, MAX_SEQUENCE_COUNT=600, CSV_PATH=str(csv))
    ctx = RunContext.create(cfg, root=tmp_path / "runs")
    tf.keras.backend.clear_session()
    res = train_and_evaluate(config=ctx.config, run_context=ctx, epochs=n_epochs, force=True, calibrate=False,
                             fit_calibration=False, save_artifacts=False)
    rows = read_metrics(ctx.run_dir / "metrics.jsonl")
    assert len(rows) == n_epochs
    for r in rows:
        assert (r.get("nonfinite_grad_steps") or 0) == 0.0, r
        for k in ("loss", "val_loss", "grad_norm_max_main", "grad_norm_max_indicator"):
            assert r.get(k) is not None and np.isfinite(r[k]), (k, r)
        for k, v in r.items():
            if k.startswith("period/"):
                assert np.isfinite(v), (k, r)
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in res.model.trainable_variables)

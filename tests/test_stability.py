"""Stability invariants (D-026, NT-036): strict mode, per-term mask counters, NaN-injection proofs
of the finite-gradient guard, and a short CPU run's health on the default config.

Marked ``stability`` (registered in pyproject.toml); the CI unit job runs it (not excluded by
``-m "not gpu and not slow"``), so every test here is kept fast on purpose.
"""
from __future__ import annotations

from pathlib import Path

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
    assert m.strict_loss_masks is False


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
    """QA repair round 1, fix 6: the POST-clip norm of BOTH groups (not just whether the pre-clip
    norm exceeded the threshold) is at or below GRAD_CLIP_NORM."""
    from neural_trade.core.config import Config

    clip = 0.01
    m = make_loss_model(config=Config(GRAD_CLIP_NORM=clip, LAMBDA_POINT=1000.0))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(4)
    x, y, lc, ext = _batch(rng)

    # Independent recomputation (a fresh tape, not train_step's own) of exactly the clip train_step
    # applies: per-group tf.clip_by_global_norm on the pre-finite-guard gradients.
    with tf.GradientTape() as tape:
        y_pred = m(x, training=True)
        loss_components = m.custom_loss(x, y, y_pred[:9], lc, ext, vacuum_overflow=y_pred[9])
    grads = tape.gradient(loss_components.total, m.trainable_variables)
    nn_gs = [g for g, v in zip(grads, m.trainable_variables) if g is not None and id(v) not in m._indicator_var_ids]
    ind_gs = [g for g, v in zip(grads, m.trainable_variables) if g is not None and id(v) in m._indicator_var_ids]
    pre_nn = float(tf.linalg.global_norm(nn_gs))
    assert pre_nn > clip, "the test setup must actually need clipping (main group)"
    post_nn = float(tf.linalg.global_norm(tf.clip_by_global_norm(nn_gs, clip)[0]))
    post_ind = float(tf.linalg.global_norm(tf.clip_by_global_norm(ind_gs, clip)[0])) if ind_gs else 0.0
    assert post_nn <= clip * 1.0001
    assert post_ind <= clip * 1.0001

    m.train_step((x, y, lc, ext))
    assert float(m._grad_health["grad_norm_max_main"].result()) == pytest.approx(pre_nn, rel=1e-3)
    assert float(m._grad_health["grad_clip_steps_main"].result()) == 1.0


@pytest.mark.data
def test_short_run_on_the_default_config_has_zero_nonfinite_steps_and_finite_state(tf, monkeypatch, tmp_path):
    """D-026 acceptance (4), QA repair round 1 fix 6: a short CPU run of the default config **on
    the bundled CSV**, N=3 consecutive epochs: zero non-finite steps; finite weights, head outputs
    and learned periods after every epoch; and two of the three 'not stuck' conditions never hold
    (a variance head at VAR_FLOOR for every sample; a learned period sitting at its configured
    bound) - var-at-floor share < 1.0 (of ``dir_n_h0`` samples) and ``periods_at_bound``
    (health_block) empty, every epoch.

    The third condition (a gradient group clipped on every single step that epoch) is NOT
    asserted per epoch here: on NT-047's larger default model (14 indicator families, D-047), the
    pre-clip norm on this exact config (bundled CSV, 3 epochs, batch 64, 3,000 sequences) measures
    up to ~171 (main) / ~235 (indicator) against the default ``GRAD_CLIP_NORM: 20`` (QA repair
    round 2 measured the same numbers independently), so it legitimately clips every step of some
    early epochs (confirmed identical on origin/remediation/plan's own HEAD, unrelated to this
    item) - not a regression, but real, accepted behaviour of the new default (D-047: "the 1.6x
    step cost is accepted"). Only the weaker, still meaningful claim is checked: clipping is not
    stuck at 100% for the WHOLE run. The per-term mask counters (masked_head_dir_h*/
    masked_head_var_h*, NT-036) DO stay at 0 every epoch: this run never actually produces a
    non-finite head output, so the strict-mode/mask machinery is never exercised - that is
    covered on synthetic NaN injections elsewhere (test_strict_mode_makes_every_qa_listed_
    injection_site_nonfinite and friends), not here.
    """
    from neural_trade.core.config import Config
    from neural_trade.evaluation.report import health_block
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.telemetry.epoch_logger import read_metrics
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    n_epochs = 3
    csv = str((Path(__file__).resolve().parent.parent / "binance_btcusdt_1min_ccxt.csv"))
    cfg = Config(EPOCHS=n_epochs, BATCH_SIZE=64, MAX_SEQUENCE_COUNT=3000, CSV_PATH=csv)
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
        # Head outputs finite every epoch (QA repair round 2 fix 2): the head-sanitisation mask
        # counters (masked_head_price_h*/head_dir_h*/head_var_h*, NT-036) only rise when a raw
        # head output was actually non-finite that step; staying at 0 is direct evidence the
        # price/direction/variance heads themselves were finite throughout, not only the loss.
        for pre in ("price", "dir", "var"):
            for h in ("h0", "h1", "h2"):
                key = f"masked_head_{pre}_{h}"
                assert (r.get(key) or 0) == 0.0, f"epoch {r.get('epoch')} {key}: a head output went non-finite"
        n_steps = r.get("n_steps") or 0
        assert n_steps > 0, r
        for h in ("h0", "h1", "h2"):
            n_dir = r.get(f"dir_n_{h}")
            at_floor = r.get(f"var_at_floor_{h}") or 0
            if n_dir:  # n_dir counts the deadband-surviving samples, an upper bound on the batch size
                assert at_floor < n_dir, f"epoch {r.get('epoch')} {h}: every sample at VAR_FLOOR"
    h = health_block(rows, cfg)
    assert not h.get("periods_at_bound"), f"a learned period sat at its bound: {h['periods_at_bound']}"
    for grp in ("main", "indicator"):
        total_clipped = sum(r.get(f"grad_clip_steps_{grp}") or 0 for r in rows)
        total_steps = sum(r.get("n_steps") or 0 for r in rows)
        assert total_clipped < total_steps, f"{grp} group clipped on every single step of the whole run"
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in res.model.trainable_variables)


def test_probe_does_not_double_count_a_masked_step(make_loss_model, monkeypatch):
    """QA repair round 1 fix 5: the probe's own custom_loss call must not double-count a step
    train_step's own call already counted into _mask_counters."""
    import neural_trade.losses.functions as lf
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_CRPS=1.0, PROBE_GRADIENTS=True, PROBE_EVERY=1))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(20)
    x, y, lc, ext = _batch(rng)
    monkeypatch.setattr(lf, "crps_gaussian_loss", lambda *a, **k: tf.constant(float("nan"), dtype=tf.float32))
    m.train_step((x, y, lc, ext))
    assert float(m._mask_counters["crps_loss"].result()) == 1.0


def test_probe_on_or_off_reaches_the_same_weights_after_n_steps(tf):
    """QA repair round 1 fix 5: the probe forwards with training=False (no dropout/noise draw), so
    it must not perturb the legacy stateful-RNG stream the rest of training depends on - same
    seed, same weights after N steps, whether or not PROBE_GRADIENTS is on."""
    from neural_trade.core.config import Config
    from neural_trade.registries.models import Models
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.utils.seeding import seed_everything

    def run(probe):
        seed_everything(7)
        tf.keras.backend.clear_session()
        cfg = Config(PROBE_GRADIENTS=probe, PROBE_EVERY=1)
        base = Models.build(cfg.MODEL_NAME, cfg)
        m = CustomTrainModel(base_model=base, pred_scale=250.0, pred_mean=0.0, config=cfg,
                             inputs=base.inputs, outputs=base.outputs)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
        rng = np.random.default_rng(0)
        in_shape = base.input_shape[1:]  # (LOOKBACK,) or (LOOKBACK, n_channels) - NT-047 OHLCV
        for _ in range(3):
            x = tf.constant(rng.normal(0, 1, size=(16,) + tuple(in_shape)).astype(np.float32))
            y = tf.constant(rng.normal(0, 1, size=(16, 3)).astype(np.float32))
            lc = tf.constant((110_000.0 + rng.normal(0, 500, size=(16, 1))).astype(np.float32))
            ext = tf.constant(rng.normal(0, 200, size=(16, 3)).astype(np.float32))
            m.train_step((x, y, lc, ext))
        return [w.numpy().copy() for w in m.trainable_variables]

    w_off = run(False)
    w_on = run(True)
    assert len(w_off) == len(w_on)
    for a, b in zip(w_off, w_on):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_strict_mode_makes_every_qa_listed_injection_site_nonfinite(make_loss_model, monkeypatch):
    """QA repair round 1 fix 1 (D:/nt_qa/nt037-out/strict_gaps.py): every non-finite guard in
    losses/functions.py, not only the 15 that feed `total` directly - direction BCE, inside
    point_huber, a variance/price head output, and pnl_utility's own total - must make the total
    non-finite in strict mode."""
    import neural_trade.losses.functions as lf
    from neural_trade.core.config import Config

    rng = np.random.default_rng(0)
    x, y, lc, ext = _batch(rng)
    price, dirs, var = _heads(rng)
    y_pred = _y_pred(price, dirs, var)

    def _case(strict, patch=None, loss_name="custom_loss", y_pred_override=None, **cfgkw):
        m = make_loss_model(config=Config(STRICT_LOSS_MASKS=strict, **cfgkw))
        saved = {}
        for k, v in (patch or {}).items():
            saved[k] = getattr(lf, k)
            monkeypatch.setattr(lf, k, v)
        try:
            f = getattr(lf, loss_name)
            out = f(m, x, y, y_pred_override or y_pred, lc, ext)
        finally:
            for k, v in saved.items():
                monkeypatch.setattr(lf, k, v)
        return out, m

    def nan_bce(*a, **k):
        return tf.fill([B], float("nan"))

    def nan_scalar(*a, **k):
        return tf.constant(float("nan"))

    out, m = _case(True, {"binary_cross_entropy_loss": nan_bce})
    assert not np.isfinite(float(out.total))
    assert float(m._mask_counters["dir_loss"].result()) == 1.0

    out, m = _case(True, {"_logcosh_safe": lambda d: d * float("nan")})
    assert not np.isfinite(float(out.total))
    assert float(m._mask_counters["point_huber_raw"].result()) > 0

    vnan = [tf.constant(np.full((B, 1), np.nan, np.float32)), var[1], var[2]]
    out, m = _case(True, y_pred_override=_y_pred(price, dirs, vnan))
    assert not np.isfinite(float(out.total))
    assert float(m._mask_counters["head_var_h0"].result()) == 1.0

    pnan = [tf.constant(np.full((B, 1), np.nan, np.float32)), price[1], price[2]]
    out, m = _case(True, y_pred_override=_y_pred(pnan, dirs, var))
    assert not np.isfinite(float(out.total))
    assert float(m._mask_counters["head_price_h0"].result()) == 1.0

    out, m = _case(True, {"crps_gaussian_loss": nan_scalar}, loss_name="pnl_utility", LAMBDA_CRPS=1.0)
    assert not np.isfinite(float(out.total))
    assert float(m._mask_counters["crps_loss"].result()) == 1.0
    assert float(m._mask_counters["pnl_total"].result()) == 1.0

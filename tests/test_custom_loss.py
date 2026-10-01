"""Loss-system regression tests at REALISTIC scales.

The pre-existing smoke test (tests/test_losses.py) calls custom_loss with
pred_scale=1 and last_close=1 - the one parameterisation under which the
trend-loss unit bug was invisible. With pred_scale~261 and last_close~110,000
the old extended_trend_loss overflowed cosh, pinned itself at a constant and
produced a NaN gradient that poisoned every weight on the first step.
Every test here fails on the pre-fix code.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from neural_trade.losses import _logcosh_safe

B = 64
SCALES = [
    pytest.param(1.0, 0.0, 1.0, id="unit-scales-legacy-smoke"),
    pytest.param(261.0, 3.2, 110_000.0, id="realistic-btc"),
    pytest.param(0.01, 0.0, 5.0, id="tiny-scale"),
]


# CustomTrainModel is functional in production; see the make_loss_model fixture for why
# the bare CustomTrainModel(base_model=None, ...) form is order-dependent and not used here.


def _batch(rng, last_close):
    x_window = tf.constant(rng.normal(0.0, 1.0, size=(B, 60)).astype(np.float32))
    y_true = tf.constant(rng.normal(0.0, 1.0, size=(B, 3)).astype(np.float32))  # scaled deltas
    lc = tf.constant((last_close + rng.normal(0.0, 0.005 * last_close, size=(B, 1))).astype(np.float32))
    ext = tf.constant(rng.normal(0.0, 200.0, size=(B, 3)).astype(np.float32))  # raw dollar deltas
    return x_window, y_true, lc, ext


def _heads(rng):
    price = [tf.Variable(rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    dirs = [tf.Variable(rng.uniform(0.05, 0.95, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    var = [tf.Variable(rng.uniform(0.5, 2.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    return price, dirs, var


def _y_pred(price, dirs, var):
    return (price[0], dirs[0], var[0], price[1], dirs[1], var[1], price[2], dirs[2], var[2])


def test_logcosh_safe_large_argument():
    x = tf.Variable([500.0, -500.0, 0.1, 0.0], dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = _logcosh_safe(x)
    g = tape.gradient(y, x).numpy()
    y = y.numpy()
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(g))
    np.testing.assert_allclose(g[:2], [1.0, -1.0], atol=1e-6)  # tanh(+-500)
    np.testing.assert_allclose(y[2], np.log(np.cosh(0.1)), atol=1e-6)  # exact where cosh is finite
    np.testing.assert_allclose(y[:2], 500.0 - np.log(2.0), atol=1e-3)  # |x| - log 2 asymptote
    assert abs(float(y[3])) < 1e-6


@pytest.mark.parametrize("pred_scale,pred_mean,last_close", SCALES)
def test_all_34_components_finite_with_finite_gradients(make_loss_model, pred_scale, pred_mean, last_close):
    rng = np.random.default_rng(0)
    m = make_loss_model(pred_scale, pred_mean)
    x, y, lc, ext = _batch(rng, last_close)
    price, dirs, var = _heads(rng)

    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
        total = out[0]

    vals = np.array([float(t) for t in out])
    assert vals.shape == (37,), ("LossComponents contract is 37 fields (NT-087 added pnl_val; "
                                 "NT-037/D-045 added dir_align_val, coherence_penalty_val)")
    bad = [f for f, v in zip(out._fields, vals) if not np.isfinite(v)]
    assert not bad, f"non-finite components: {bad}"
    assert vals[0] > 0.0

    grads = tape.gradient(total, price + dirs + var)
    names = ["price"] * 3 + ["dir"] * 3 + ["var"] * 3
    for name, g in zip(names, grads):
        assert g is not None, f"no gradient reached the {name} head"
        assert bool(tf.reduce_all(tf.math.is_finite(g))), f"non-finite gradient on the {name} head"
    for g in grads[:3] + grads[6:]:  # price and variance heads must be supervised
        assert float(tf.reduce_max(tf.abs(g))) > 0.0


def test_extended_trend_is_not_the_clipped_constant(make_loss_model):
    """At realistic scales the old term was pinned at 10.0 (times lambda) in every epoch."""
    rng = np.random.default_rng(1)
    m = make_loss_model(261.0, 3.2)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
    lam = float(m.lambda_extended_trend)
    for name in ("extended_h0", "extended_h1", "extended_h2"):
        v = float(getattr(out, name))
        assert 0.0 < v < 5.0, f"{name}={v}"
        assert abs(v - 10.0 * lam) > 1e-3 and abs(v - 1.333295) > 1e-3, f"{name} still looks clipped: {v}"
    for name in ("local_h0", "global_h0", "local_h1", "global_h1", "local_h2", "global_h2"):
        assert float(getattr(out, name)) == 0.0  # retired terms are exactly zero


def test_extended_trend_matches_numpy_reference_and_responds_to_the_head(make_loss_model):
    """ext_k = lambda * mean(logcosh(price_k - scaled(ext_raw[:, k]))) in scaled-delta units."""
    rng = np.random.default_rng(2)
    pred_scale, pred_mean = 261.0, 3.2
    m = make_loss_model(pred_scale, pred_mean)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)

    lam = float(m.lambda_extended_trend)
    ext_np = ext.numpy()
    for k, name in enumerate(("extended_h0", "extended_h1", "extended_h2")):
        p = price[k].numpy().reshape(-1)
        ref = lam * np.mean(np.log(np.cosh(p - (ext_np[:, k] - pred_mean) / (pred_scale + 1e-8))))
        np.testing.assert_allclose(float(getattr(out, name)), ref, rtol=1e-4, atol=1e-5)

    price[0].assign_add(tf.fill([B, 1], 0.5))  # the term must depend on the head it supervises
    out2 = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
    assert abs(float(out2.extended_h0) - float(out.extended_h0)) > 1e-4
    assert abs(float(out2.extended_h1) - float(out.extended_h1)) < 1e-6


# ---------------------------------------------------------------------------- the former stubs
# (tests/test_model_math_consistency.py::TestConfigDeepMathStubs, now real)

def _components(m, rng_seed=1, var_value=None):
    rng = np.random.default_rng(rng_seed)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    if var_value is not None:
        var = [tf.Variable(np.full((B, 1), var_value, np.float32)) for _ in range(3)]
    return m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext), (x, y, price, var)


@pytest.mark.parametrize("weight, fields, outer", [
    ("lambda_crps", ("crps_h0", "crps_h1", "crps_h2"), None),
    ("lambda_soft_ece", ("soft_ece_h0", "soft_ece_h1", "soft_ece_h2"), None),
    ("lambda_var", ("nll_h0", "nll_h1", "nll_h2"), "lambda_nll_outer"),
    ("lambda_dir", ("dir_h0", "dir_h1", "dir_h2"), "lambda_dir_outer"),
])
def test_total_is_the_weighted_sum_of_its_components(make_loss_model, weight, fields, outer):
    """Changing one loss weight changes the total by exactly weight x (its returned components)."""
    m = make_loss_model()
    m.set_lambda_values(**{weight: 0.0})
    base, _ = _components(m)
    m.set_lambda_values(**{weight: 2.5})
    out, _ = _components(m)
    comp = sum(float(getattr(out, f)) for f in fields)
    scale = float(getattr(m, outer)) if outer else 1.0
    assert float(out.total) - float(base.total) == pytest.approx(2.5 * scale * comp, rel=1e-4, abs=1e-5)


def test_already_weighted_physics_fields_scale_with_their_weight(make_loss_model):
    m = make_loss_model()
    m.set_lambda_values(lambda_t_perp=0.1, lambda_hd=0.1)
    a, _ = _components(m)
    m.set_lambda_values(lambda_t_perp=0.3, lambda_hd=0.3)
    b, _ = _components(m)
    assert float(b.t_perp_total) == pytest.approx(3 * float(a.t_perp_total), rel=1e-5)
    assert float(b.hd_val) == pytest.approx(3 * float(a.hd_val), rel=1e-5)


def test_nll_is_exact_at_the_variance_floor(make_loss_model):
    """Variance below VAR_FLOOR is floored (not capped above) before the Gaussian NLL."""
    m = make_loss_model()
    out, (_, y, price, _) = _components(m, var_value=1e-6)
    floor = m.config.VAR_FLOOR + 1e-8
    for h in range(3):
        err = y.numpy()[:, h] - price[h].numpy()[:, 0]
        ref = np.mean(0.5 * (np.log(2 * np.pi) + np.log(floor)) + 0.5 * err ** 2 / floor)
        assert float(getattr(out, f"nll_h{h}")) == pytest.approx(ref, rel=1e-4)


def test_vacuum_bandwidth_term_is_zero_unless_lambda_vac_is_set(make_loss_model):
    from neural_trade.core.config import Config

    off, _ = _components(make_loss_model(config=Config(LAMBDA_VAC=0.0)))
    assert float(off.vac_val) == 0.0
    on, (_, _, price, _) = _components(make_loss_model(config=Config(LAMBDA_VAC=0.1)))
    spread = np.std(np.stack([p.numpy()[:, 0] for p in price], 1), axis=1)
    assert float(on.vac_val) == pytest.approx(np.mean(np.maximum(spread - 0.1, 0.0)), rel=1e-4)


# ---------------------------------------------------------------------------- NT-037 (D-026/D-045)
# Per-term contributions, mask counters and the dead-zone counters, all reached through
# test_step (no gradients needed): it runs the same _update_diagnostics path as train_step.

CONTRIB_TERM_KEYS = ('point', 'trend', 'dir', 'dir_align', 'reg', 'inter_reg', 'vol', 'coherence',
                     'nll', 'crps', 'soft_ece', 't_perp', 'casimir', 'vac', 'hd', 'ife',
                     'vac_overflow', 'pnl')


def test_contrib_terms_sum_to_the_total(make_loss_model):
    """Every named contrib_* (the addends of `total`, D-026/D-045) sums to `loss`/`val_loss`
    within 1e-4 relative - the acceptance criterion of NT-037 (3)."""
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_CRPS=0.3, LAMBDA_SOFT_ECE=0.2, LAMBDA_T_PERP=0.1,
                                      LAMBDA_CASIMIR=0.1, LAMBDA_HD=0.1, LAMBDA_IFE=0.1,
                                      LAMBDA_VAC_OVERFLOW=0.1, LAMBDA_VAC=0.1))
    rng = np.random.default_rng(7)
    x, y, lc, ext = _batch(rng, 110_000.0)
    logs = m.test_step((x, y, lc, ext))
    total = sum(float(logs[f'contrib_{k}']) for k in CONTRIB_TERM_KEYS)
    assert total == pytest.approx(float(logs['loss']), rel=1e-4)


def test_contrib_terms_sum_to_the_train_loss(make_loss_model):
    """QA repair round 2 fix 3: the TRAIN-side companion of test_contrib_terms_sum_to_the_total
    (round 1's contrib_* bug - accumulated only on TRAIN_METRICS_EVERY-sampled steps, so it did
    not sum to the train loss - had no test on the train path; this is that test). Multiple
    train_step calls: contrib_* and 'loss' are both running means over every step, and must
    average exactly the same steps."""
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_CRPS=0.3, LAMBDA_SOFT_ECE=0.2, LAMBDA_T_PERP=0.1,
                                      LAMBDA_CASIMIR=0.1, LAMBDA_HD=0.1, LAMBDA_IFE=0.1,
                                      LAMBDA_VAC_OVERFLOW=0.1, LAMBDA_VAC=0.1,
                                      TRAIN_METRICS_EVERY=10))  # the default: most train diagnostics
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))          # are sampled; contrib_* must not be
    rng = np.random.default_rng(13)
    for _ in range(4):
        x, y, lc, ext = _batch(rng, 110_000.0)
        m.train_step((x, y, lc, ext))
    logs = m.train_epoch_logs()
    total = sum(float(logs[f'contrib_{k}']) for k in CONTRIB_TERM_KEYS)
    assert total == pytest.approx(float(logs['loss']), rel=1e-4)


def test_mask_counters_count_only_the_injected_term(make_loss_model, monkeypatch):
    import neural_trade.losses.functions as lf
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_CRPS=1.0))
    rng = np.random.default_rng(8)
    x, y, lc, ext = _batch(rng, 110_000.0)

    monkeypatch.setattr(lf, "crps_gaussian_loss", lambda *a, **k: tf.constant(float("nan"), dtype=tf.float32))
    m.test_step((x, y, lc, ext))
    fired = {t: float(c.result()) for t, c in m._mask_counters.items()}
    assert fired.pop("crps_loss") == 1.0
    assert not any(v > 0 for v in fired.values()), f"unexpected: {fired}"


def test_dead_zone_counters_dir_n_and_var_at_floor(make_loss_model):
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(VAR_FLOOR=0.5))
    rng = np.random.default_rng(9)
    x, y, lc, ext = _batch(rng, 110_000.0)

    logs = m.test_step((x, y, lc, ext))
    for h in ("h0", "h1", "h2"):
        assert logs[f"dir_n_{h}"] is not None

    # Direct custom_loss/_update_diagnostics call with every variance head below the floor: every
    # sample of every horizon must be counted "at floor".
    price, dirs, _ = _heads(rng)
    var = [tf.Variable(np.full((B, 1), 0.1, np.float32)) for _ in range(3)]  # < VAR_FLOOR (0.5)
    y_pred = _y_pred(price, dirs, var)
    out = m.custom_loss(x, y, y_pred, lc, ext)
    m._update_diagnostics(out, tf.cast(tf.shape(y)[0], tf.float32), y, y_pred, lc, training=False)
    for h in range(3):
        assert float(m._var_floor_counters[f'var_at_floor_h{h}'].result()) >= B - 1e-6


def test_gradient_probe_off_by_default_writes_no_probe_key(make_loss_model):
    m = make_loss_model()  # PROBE_GRADIENTS defaults False
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(10)
    x, y, lc, ext = _batch(rng, 110_000.0)
    m.train_step((x, y, lc, ext))
    logs = m.train_epoch_logs()
    assert not any(k.startswith("probe_") for k in logs)


def test_gradient_probe_shares_sum_to_one_and_every_key_is_written(make_loss_model):
    """NT-037 acceptance (6), QA repair round 1 fix 5: with the flag on, a 1-epoch CPU smoke run
    has the probe keys (per group) and the value/gradient shares sum to 1 within 1e-4. The
    ``make_loss_model`` fixture's tiny functional model has no indicator layer and no head-named
    Dense layers, so every trainable variable falls into the 'trunk' group - the 'head' and
    'indicator' groups are legitimately empty and untested here (see the real-model probe check
    used in QA, D:/nt_qa/nt037-out/probe_check.py)."""
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(PROBE_GRADIENTS=True, PROBE_EVERY=1, LAMBDA_CRPS=0.2,
                                      LAMBDA_SOFT_ECE=0.2, LAMBDA_T_PERP=0.1, LAMBDA_CASIMIR=0.1,
                                      LAMBDA_HD=0.1, LAMBDA_IFE=0.1, LAMBDA_VAC_OVERFLOW=0.1))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    rng = np.random.default_rng(11)
    x, y, lc, ext = _batch(rng, 110_000.0)
    m.train_step((x, y, lc, ext))
    logs = m.train_epoch_logs()
    terms = m._probe_terms
    value_shares = [logs[f"probe_value_share_{t}"] for t in terms]
    grad_shares = [logs[f"probe_grad_share_{t}_trunk"] for t in terms]
    assert sum(value_shares) == pytest.approx(1.0, abs=1e-4)
    assert sum(grad_shares) == pytest.approx(1.0, abs=1e-4)
    assert -1.0 - 1e-6 <= logs["probe_conflict_mean_trunk"] <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= logs["probe_conflict_min_trunk"] <= 1.0 + 1e-6
    for t in terms:
        assert -1.0 - 1e-6 <= logs[f"probe_cos_{t}_trunk"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------- NT-096 (D-045)
# (1) sqrt(var + eps) inside every batch std of the loss (vol, HD, IFE, vacuum); (2) coherence
# keeps only the magnitude-ordering part.

def test_vol_loss_std_is_gradient_safe_at_an_exactly_constant_price_head(make_loss_model):
    """A1 (NT-096): `tf.math.reduce_std` differentiates `sqrt` at the computed variance, so a
    price_h1 head that is literally the same value for every example in the batch (variance
    exactly 0) gave an infinite - NaN after the division - gradient through vol_loss's
    `abs(pred_std - actual_std)` before the eps guard (confirmed directly:
    `tf.gradients(tf.abs(tf.math.reduce_std(const) - c), const)` is all-NaN on the pre-fix
    `tf.math.reduce_std`). The per-term `_finite_or_zero` guard hides it (vol_loss's own value
    stays finite - the NaN is in the gradient, which the guard does not see) so it was never
    actually observed in a real run, where a batch is never perfectly constant. The guard is
    opt-in (`Config.LOSS_SAFE_STD`, repair round 1: D-045 ships a value-moving change behind a
    switch whose default keeps today's graph); this test turns it on, see
    `test_loss_safe_std_defaults_off_and_vol_loss_still_nans_there` for the default path."""
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_VOL=1.0, LOSS_SAFE_STD=True))
    rng = np.random.default_rng(21)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price = [tf.Variable(tf.fill((B, 1), tf.constant(0.5, dtype=tf.float32))) for _ in range(3)]
    dirs = [tf.Variable(rng.uniform(0.3, 0.7, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    var = [tf.Variable(rng.uniform(0.5, 2.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]

    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
        total = out[0]
    assert np.isfinite(float(out.vol_loss))
    grads = tape.gradient(total, price)
    for g in grads:
        assert g is not None
        assert bool(tf.reduce_all(tf.math.is_finite(g))), "non-finite gradient from vol_loss's std"


def test_std_based_losses_finite_gradient_on_batch_constant_heads(make_loss_model):
    """NT-096 acceptance (1): feeding a batch-constant head (every price/variance head the same
    value for every example, which is also constant ACROSS the three horizons, the degenerate
    point of vacuum's per-example cross-horizon std) into vol, HD, IFE and vacuum together gives
    finite gradients on every head, with all four terms active and `Config.LOSS_SAFE_STD=True`."""
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(LAMBDA_VOL=1.0, LAMBDA_HD=0.1, LAMBDA_IFE=0.1, LAMBDA_VAC=0.05,
                                      LOSS_SAFE_STD=True))
    rng = np.random.default_rng(22)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price = [tf.Variable(tf.fill((B, 1), tf.constant(0.5, dtype=tf.float32))) for _ in range(3)]
    dirs = [tf.Variable(tf.fill((B, 1), tf.constant(0.6, dtype=tf.float32))) for _ in range(3)]
    var = [tf.Variable(tf.fill((B, 1), tf.constant(1.0, dtype=tf.float32))) for _ in range(3)]

    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
        total = out[0]
    assert np.isfinite(float(total))
    grads = tape.gradient(total, price + dirs + var)
    names = ["price0", "price1", "price2", "dir0", "dir1", "dir2", "var0", "var1", "var2"]
    for name, g in zip(names, grads):
        assert g is not None, f"no gradient reached {name}"
        assert bool(tf.reduce_all(tf.math.is_finite(g))), f"non-finite gradient on {name}"


def test_loss_safe_std_defaults_off_and_vol_loss_still_nans_there(make_loss_model):
    """NT-096 repair round 1 (D-045, same ruling as NT-037's tf.cond guards): `LOSS_SAFE_STD`
    defaults to False, which must be bit-for-bit today's graph (`scripts/golden_run.py verify`),
    so the known hazard this field documents - vol_loss's gradient is NaN at an exactly
    batch-constant price_h1 head - is still live at the default. This is the mirror image of
    `test_vol_loss_std_is_gradient_safe_at_an_exactly_constant_price_head` (which turns the
    switch on and shows the NaN is gone)."""
    from neural_trade.core.config import Config

    assert Config().LOSS_SAFE_STD is False
    m = make_loss_model(config=Config(LAMBDA_VOL=1.0))  # LOSS_SAFE_STD left at its default
    rng = np.random.default_rng(21)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price = [tf.Variable(tf.fill((B, 1), tf.constant(0.5, dtype=tf.float32))) for _ in range(3)]
    dirs = [tf.Variable(rng.uniform(0.3, 0.7, size=(B, 1)).astype(np.float32)) for _ in range(3)]
    var = [tf.Variable(rng.uniform(0.5, 2.0, size=(B, 1)).astype(np.float32)) for _ in range(3)]

    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)
        total = out[0]
    assert np.isfinite(float(out.vol_loss))  # the forward value is fine; the NaN is in the gradient
    grads = tape.gradient(total, price)
    nonfinite = [not bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads]
    assert any(nonfinite), ("expected the documented default-path hazard (a NaN gradient from "
                            "vol_loss's unguarded std) to still be present with LOSS_SAFE_STD=False")


def test_safe_std_matches_reduce_std_away_from_zero_variance(tf):
    """`_safe_std`'s 1e-12 guard is far below float32 precision at realistic variances: its
    VALUE matches `tf.math.reduce_std` bit-for-bit there (only the gradient at variance 0
    changes), so the eps guard does not perturb any normal run (`scripts/golden_run.py`)."""
    from neural_trade.losses.functions import _safe_std

    rng = np.random.default_rng(23)
    x = tf.constant(rng.normal(0.0, 1.0, size=(64,)).astype(np.float32))
    np.testing.assert_array_equal(_safe_std(x).numpy(), tf.math.reduce_std(x).numpy())
    x2 = tf.constant(rng.normal(0.0, 1.0, size=(64, 3)).astype(np.float32))
    np.testing.assert_array_equal(_safe_std(x2, axis=1).numpy(), tf.math.reduce_std(x2, axis=1).numpy())


def test_coherence_penalty_is_exactly_the_magnitude_ordering_term(make_loss_model):
    """NT-096 acceptance (2), repair round 2 (c): with `Config.COHERENCE_MAGNITUDE_ONLY=True`,
    coherence's logged contribution equals the magnitude-ordering term alone
    (`relu(|p0|-|p1|) + relu(|p1|-|p2|)`, batch mean, /3 - the old three-term average's weight
    on this term) - the only part of the old three-term average with a non-zero gradient. The
    default (False) keeps today's three-term value; see the switch comparison below."""
    from neural_trade.core.config import Config

    m = make_loss_model(config=Config(COHERENCE_MAGNITUDE_ONLY=True))
    rng = np.random.default_rng(24)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)

    abs0, abs1, abs2 = (tf.abs(tf.squeeze(p, axis=1)) for p in price)
    expected = tf.reduce_mean(tf.nn.relu(abs0 - abs1) + tf.nn.relu(abs1 - abs2)) / 3.0
    np.testing.assert_allclose(float(out.coherence_penalty_val), float(expected), rtol=1e-6)


def test_coherence_default_is_the_three_term_value_not_magnitude_only(make_loss_model):
    """NT-096 repair round 2: `Config.COHERENCE_MAGNITUDE_ONLY` defaults to False, which must
    stay today's exact three-term coherence_penalty (dir_disagree_loss + magnitude_loss +
    target_smoothness_loss) / 3 - NOT the magnitude-only form - so a real run's served epoch,
    EarlyStopping and ReduceLROnPlateau are untouched at the default (QA of repair round 1:
    dir_disagree_loss's value depends on the weights and is not a per-batch constant)."""
    from neural_trade.core.config import Config

    assert Config().COHERENCE_MAGNITUDE_ONLY is False
    m = make_loss_model()  # COHERENCE_MAGNITUDE_ONLY left at its default
    rng = np.random.default_rng(24)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price, dirs, var = _heads(rng)
    out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)

    p0, p1, p2 = (tf.squeeze(p, axis=1) for p in price)
    sign0, sign1, sign2 = tf.sign(p0), tf.sign(p1), tf.sign(p2)
    agree01 = tf.reduce_mean(tf.cast(tf.equal(sign0, sign1), tf.float32))
    agree12 = tf.reduce_mean(tf.cast(tf.equal(sign1, sign2), tf.float32))
    dir_disagree_loss = 1.0 - (agree01 + agree12) / 2.0

    y_true_raw = y * m.pred_scale + m.pred_mean  # custom_loss's own raw-units transform
    y_true_raw_h0, y_true_raw_h1, y_true_raw_h2 = y_true_raw[:, 0], y_true_raw[:, 1], y_true_raw[:, 2]
    sign_t0, sign_t1, sign_t2 = (tf.sign(t) for t in (y_true_raw_h0, y_true_raw_h1, y_true_raw_h2))
    target_smoothness_loss = tf.reduce_mean(tf.cast(
        tf.math.logical_xor(sign_t1 == sign_t0, sign_t1 == sign_t2), tf.float32))

    abs0, abs1, abs2 = tf.abs(p0), tf.abs(p1), tf.abs(p2)
    magnitude_loss = tf.reduce_mean(tf.nn.relu(abs0 - abs1) + tf.nn.relu(abs1 - abs2))

    expected = (dir_disagree_loss + magnitude_loss + target_smoothness_loss) / 3.0
    np.testing.assert_allclose(float(out.coherence_penalty_val), float(expected), rtol=1e-6)
    # sanity: this is NOT the magnitude-only value (dir_disagree/target_smoothness are non-zero here)
    assert abs(float(out.coherence_penalty_val) - float(magnitude_loss) / 3.0) > 1e-6


def test_coherence_dead_parts_removal_does_not_change_any_gradient(make_loss_model):
    """NT-096 acceptance (2), repair round 2 (b): dir_disagree_loss (`tf.sign`/`tf.equal`, both
    non-differentiable) and target_smoothness_loss (reads only the labels) are the two
    sub-terms `Config.COHERENCE_MAGNITUDE_ONLY=True` drops from coherence_penalty; both have
    ZERO gradient everywhere, so comparing the two switch settings through the actual model's
    `custom_loss`, on the SAME fixed batch and heads, shows the gradients on every price head
    are bit-for-bit identical - only the forward value (dir_disagree_loss's weight-dependent,
    non-constant contribution plus target_smoothness_loss's label-constant one) differs."""
    from neural_trade.core.config import Config

    rng = np.random.default_rng(25)
    x, y, lc, ext = _batch(rng, 110_000.0)
    price0, dirs, var = _heads(np.random.default_rng(26))
    price_false = [tf.Variable(p.numpy()) for p in price0]
    price_true = [tf.Variable(p.numpy()) for p in price0]  # identical starting values

    m_false = make_loss_model(config=Config(COHERENCE_MAGNITUDE_ONLY=False))
    m_true = make_loss_model(config=Config(COHERENCE_MAGNITUDE_ONLY=True))

    with tf.GradientTape() as tape_false:
        out_false = m_false.custom_loss(x, y, _y_pred(price_false, dirs, var), lc, ext)
        coherence_false = out_false.coherence_penalty_val
    with tf.GradientTape() as tape_true:
        out_true = m_true.custom_loss(x, y, _y_pred(price_true, dirs, var), lc, ext)
        coherence_true = out_true.coherence_penalty_val

    g_false = tape_false.gradient(coherence_false, price_false)
    g_true = tape_true.gradient(coherence_true, price_true)
    for gf, gt in zip(g_false, g_true):
        np.testing.assert_array_equal(gf.numpy(), gt.numpy())
    # the forward values differ: the two dropped terms contributed a non-zero offset
    assert abs(float(coherence_false) - float(coherence_true)) > 1e-6

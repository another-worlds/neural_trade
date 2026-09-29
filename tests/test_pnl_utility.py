"""Tests for the ``pnl_utility`` training objective (NT-087).

``pnl_utility`` wraps ``custom_loss`` with a mean-variance P&L utility, with a linear trading
cost, on the direction heads' implied positions ``a = 2*p - 1``
(``docs/research/2026-09-29-pnl-target/README.md`` section 3, "E2"). These tests exercise it in
isolation from the full GRU model: a toy shared-parameter position head
``p = sigmoid(w*feat + b)`` (only ``w, b`` trainable) stands in for "the network", with the price
and variance heads held at fixed constants so the only gradient path into ``w, b`` is the pnl
term itself (``LAMBDA_DIR=0`` also removes the BCE path).

Criterion numbers below refer to the NT-087 acceptance criteria.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from neural_trade.core.config import Config

from tests.test_custom_loss import SCALES, _batch, _heads, _y_pred

EDGE_BPS = 200.0        # 200 bps, well above the 26 bps default PNL_COST_BPS
LAST_CLOSE = 100.0
LOOKBACK = 60


def _walk_window(rng, B, last_close=LAST_CLOSE, pred_scale=1.0, lookback=LOOKBACK, step_vol=0.001):
    """A realistic i.i.d. random-walk price window, correctly NORMALISED the way the real data
    pipeline feeds ``x_window`` into the loss objectives: ``window_relative`` normalisation is
    ``x = (raw_close - last_close) / pred_scale`` (``data/scaling.py``), and the window's last bar
    is exactly ``last_close`` (``data/windowing.py``: ``window = close[i-lookback:i]``, whose last
    element is ``close[i-1] = last_close``). NT-087 repair round 1: an earlier version of this
    fixture returned an UN-normalised, arbitrarily-anchored walk (values near 1.0, unrelated to
    ``last_close``/``pred_scale``), which happened to mask the production bug this fixture is meant
    to exercise (``PNL_SIGMA_SOURCE='realized_vol'`` reconstructing the raw window from the
    normalised one)."""
    steps = rng.normal(0.0, step_vol, size=(B, lookback))
    path = np.cumprod(1.0 + steps, axis=1)                        # arbitrary starting level
    raw = path * (last_close / path[:, -1:])                      # anchor: raw[:, -1] == last_close
    normalized = (raw - last_close) / pred_scale
    return tf.constant(normalized.astype(np.float32))


def _planted_edge_batch(rng, B, last_close=LAST_CLOSE, edge_bps=EDGE_BPS):
    """``feat`` perfectly encodes the sign of a large planted forward return on every horizon."""
    sign = rng.choice([-1.0, 1.0], size=B).astype(np.float32)
    feat = (sign + rng.normal(0.0, 0.01, size=B).astype(np.float32)).reshape(-1, 1)
    y_h = (last_close * sign * (edge_bps / 10000.0)).astype(np.float32)
    y_true = np.stack([y_h, y_h, y_h], axis=1).astype(np.float32)
    lc = np.full((B, 1), last_close, dtype=np.float32)
    return tf.constant(feat), tf.constant(y_true), tf.constant(lc)


def _no_edge_batch(rng, B, last_close=LAST_CLOSE, edge_bps=EDGE_BPS):
    """``feat`` is independent of the (mean-zero) forward return: no exploitable signal."""
    feat = rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=B).astype(np.float32)
    y_h = (last_close * sign * (edge_bps / 10000.0)).astype(np.float32)
    y_true = np.stack([y_h, y_h, y_h], axis=1).astype(np.float32)
    lc = np.full((B, 1), last_close, dtype=np.float32)
    return tf.constant(feat), tf.constant(y_true), tf.constant(lc)


def _toy_position_model(make_loss_model, *, lambda_pnl=1.0):
    cfg = Config(LOSS_NAME="pnl_utility", LAMBDA_PNL=lambda_pnl, PNL_GAMMA=1.0,
                 PNL_COST_BPS=26.0, PNL_SIGMA_SOURCE="realized_vol")
    return make_loss_model(1.0, 0.0, config=cfg, lambda_dir=0.0)


def _run_toy(make_loss_model, batch_fn, seed, n_steps=200, B=256, lr=0.3):
    """Trains the toy position head for n_steps, returning per-step ``|2p-1|`` and ``pnl_val``
    (both logged from the forward pass BEFORE that step's gradient update is applied, so index 0
    reflects the untrained w=b=0 state exactly: a=0, pnl_val=0)."""
    rng = np.random.default_rng(seed)
    m = _toy_position_model(make_loss_model)
    w = tf.Variable(0.0, dtype=tf.float32)
    b = tf.Variable(0.0, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(lr)
    price0 = tf.zeros((B, 1), dtype=tf.float32)
    var1 = tf.ones((B, 1), dtype=tf.float32)
    ext = tf.zeros((B, 3), dtype=tf.float32)

    abs_a_hist, pnl_hist = [], []
    for _ in range(n_steps):
        feat, y_true, lc = batch_fn(rng, B)
        x_window = _walk_window(rng, B)
        with tf.GradientTape() as tape:
            p = tf.sigmoid(w * feat + b)
            y_pred = (price0, p, var1, price0, p, var1, price0, p, var1)
            out = m.custom_loss(x_window, y_true, y_pred, lc, ext)
        grads = tape.gradient(out.total, [w, b])
        opt.apply_gradients(zip(grads, [w, b]))
        abs_a_hist.append(float(tf.reduce_mean(tf.abs(2.0 * p - 1.0))))
        pnl_hist.append(float(out.pnl_val))
    return np.array(abs_a_hist), np.array(pnl_hist)


# ---------------------------------------------------------------------------- criterion 1

def test_planted_edge_above_cost_grows_the_position_and_improves_utility(make_loss_model):
    """A 200 bps planted edge (well above the 26 bps default cost), perfectly readable from
    ``feat``, must push the toy position head away from flat and improve utility (pnl_val, more
    negative = higher utility since pnl_val = -utility). Fixed seed=0: reproducibly converges to
    a stable, non-trivial position, verified over multiple independent re-runs during development
    (final |2p-1| ~ 0.19, final pnl_val ~ -1.07); thresholds below carry a wide margin around
    those observed numbers.
    """
    abs_a, pnl = _run_toy(make_loss_model, _planted_edge_batch, seed=0)

    assert abs_a[0] == 0.0 and pnl[0] == 0.0, "the untrained baseline (w=b=0) must be exactly flat"

    tail_a = float(np.mean(abs_a[-30:]))
    tail_pnl = float(np.mean(pnl[-30:]))
    assert tail_a > 0.10, f"position did not grow away from flat: mean|2p-1| (last 30) = {tail_a}"
    assert tail_pnl < -0.3, f"utility did not improve: mean pnl_val (last 30) = {tail_pnl}"


# ---------------------------------------------------------------------------- criterion 2

@pytest.mark.parametrize("seed", [0, 1])
def test_no_edge_stays_flat(make_loss_model, seed):
    """When ``feat`` carries no information about the (mean-zero) forward return, the cost term
    means any persistent |a| > 0 is a pure loss: the optimal solution is flat (a=0, p=0.5).
    Observed mean|2p-1| over the last 30 of 200 steps across 6 independent seeds during
    development ranged 0.017-0.040; 0.08 gives a wide margin while still being a "flat solution"
    threshold, checked here on seeds 0 and 1 (both must pass)."""
    abs_a, _ = _run_toy(make_loss_model, _no_edge_batch, seed=seed)
    tail_a = float(np.mean(abs_a[-30:]))
    assert tail_a < 0.08, f"position did not stay flat with no edge: mean|2p-1| (last 30) = {tail_a}"


# ---------------------------------------------------------------------------- criterion 3

def _pnl_window(rng, kind, last_close, B=64, lookback=LOOKBACK):
    if kind == "random":
        return tf.constant(rng.normal(0.0, 1.0, size=(B, lookback)).astype(np.float32))
    if kind == "constant":
        return tf.constant(np.full((B, lookback), last_close, dtype=np.float32))
    if kind == "jump":
        w = np.full((B, lookback), last_close, dtype=np.float32)
        w[:, lookback // 2] = last_close * 1000.0 + 1.0
        return tf.constant(w)
    raise ValueError(kind)


@pytest.mark.parametrize("pred_scale,pred_mean,last_close", SCALES)
@pytest.mark.parametrize("sigma_source", ["realized_vol", "model"])
@pytest.mark.parametrize("window_kind", ["random", "constant", "jump"])
def test_pnl_utility_gradients_are_finite(make_loss_model, pred_scale, pred_mean, last_close,
                                          sigma_source, window_kind):
    """D-026: every gradient reaching the price/direction/variance heads, and pnl_val itself,
    must stay finite across scale x sigma-source x window-kind combinations - including the
    stress cases 'constant' (step_sd exactly 0, exercising the tf.maximum(sigma, 1e-6) floor;
    this makes the cost term blow up, since cost/sigma -> cost/1e-6, but it must stay FINITE,
    not NaN/Inf) and 'jump' (one huge spike in an otherwise-constant window)."""
    cfg = Config(LOSS_NAME="pnl_utility", LAMBDA_PNL=1.0, PNL_SIGMA_SOURCE=sigma_source)
    m = make_loss_model(pred_scale, pred_mean, config=cfg)
    rng = np.random.default_rng(0)
    x, y, lc, ext = _batch(rng, last_close)
    x = _pnl_window(rng, window_kind, last_close)
    price, dirs, var = _heads(rng)
    y_pred = (price[0], dirs[0], var[0], price[1], dirs[1], var[1], price[2], dirs[2], var[2])

    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, y_pred, lc, ext)
        total = out.total

    vals = np.array([float(t) for t in out])
    bad = [f for f, v in zip(out._fields, vals) if not np.isfinite(v)]
    assert not bad, f"non-finite components: {bad}"
    assert np.isfinite(float(out.pnl_val))

    grads = tape.gradient(total, price + dirs + var)
    names = ["price"] * 3 + ["dir"] * 3 + ["var"] * 3
    for name, g in zip(names, grads):
        assert g is not None, f"no gradient reached the {name} head"
        assert bool(tf.reduce_all(tf.math.is_finite(g))), f"non-finite gradient on the {name} head"


# ---------------------------------------------------------------------------- also required

def test_pnl_utility_end_to_end_train_and_test_step(make_loss_model):
    """A full CustomTrainModel with LOSS_NAME='pnl_utility' must run train_step and test_step
    without exception and produce a finite loss (the smoke test the full loop needs beyond the
    hand-rolled toy-head tests above)."""
    cfg = Config(LOSS_NAME="pnl_utility", LAMBDA_PNL=0.5)
    m = make_loss_model(1.0, 0.0, config=cfg)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    rng = np.random.default_rng(0)
    B = 32
    x = tf.constant(rng.normal(100.0, 1.0, size=(B, LOOKBACK)).astype(np.float32))
    y = tf.constant(rng.normal(0.0, 1.0, size=(B, 3)).astype(np.float32))
    lc = tf.constant(np.full((B, 1), 100.0, dtype=np.float32))
    ext = tf.constant(rng.normal(0.0, 1.0, size=(B, 3)).astype(np.float32))

    train_logs = m.train_step((x, y, lc, ext))
    assert np.isfinite(float(train_logs["loss"]))

    test_logs = m.test_step((x, y, lc, ext))
    assert np.isfinite(float(test_logs["loss"]))
    assert np.isfinite(float(test_logs["pnl_val"]))


# ------------------------------------------------------------- repair round 1, P0 (real-data sigma scale)

@pytest.mark.data
def test_realized_vol_sigma_matches_raw_close_scale_on_the_bundled_csv():
    """Regression pin for NT-087 repair round 1, P0: an earlier version of
    ``PNL_SIGMA_SOURCE='realized_vol'`` computed step returns directly on the NORMALISED window
    ``x_window`` (``(raw_close - last_close) / pred_scale``, whose last column is exactly 0 by
    construction - ``data/scaling.py``'s ``window_relative``), instead of reconstructing the raw
    close window first. That made sigma ~3,500x too large on the bundled CSV (median step_sd 1.73
    on the normalised window vs 4.9e-4 on raw closes), so the utility's cost term (``c~ =
    cost/sigma``) and gamma term were ~1e-4 of their intended scale and effectively vanished.

    This test runs ``DataProcessor.prepare_datasets`` on the bundled CSV (the real production
    pipeline, default Config: ``WINDOW_NORMALIZER='window_relative'``) and checks that the sigma
    ``pnl_utility`` would compute from the NORMALISED windows it is actually fed (reconstructing
    the raw window via ``x * pred_scale + last_close``, exactly as ``losses.functions.pnl_utility``
    does) has the same scale, to within 10% at the median, as a sigma computed directly from the
    RAW close windows (an independent ground truth taken from the same fold, never normalised).
    """
    from neural_trade.core.config import Config
    from neural_trade.data.processor import DataProcessor
    from tests.conftest import BUNDLED_CSV

    cfg = Config(CSV_PATH=str(BUNDLED_CSV))
    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    (X_train, _y_tr_s, lc_train, _ext_tr, _X_te, _y_te_s, _lc_te, _ext_te,
     _y_tr, _y_te, target_scaler) = dp.prepare_datasets(df, close)

    pred_scale = float(target_scaler.scale_[0])
    assert dp.normalizer.kind == "window_relative"
    np.testing.assert_allclose(dp.normalizer.scale, pred_scale, rtol=1e-6)

    # Ground truth: RAW close windows for the same training indices, never normalised.
    X_raw_all, _y_all, _lc_all, _ext_all = dp.make_sequences_with_extended_trends(close, cfg.LOOKBACK)
    X_raw_train = X_raw_all[dp.fold.train]
    raw_step_ret = (X_raw_train[:, 1:] - X_raw_train[:, :-1]) / X_raw_train[:, :-1]
    sd_raw = np.std(raw_step_ret, axis=1)

    # What pnl_utility computes: reconstruct raw from the NORMALISED window it is actually fed.
    reconstructed = X_train.astype(np.float64) * pred_scale + lc_train.reshape(-1, 1)
    np.testing.assert_allclose(reconstructed, X_raw_train, rtol=1e-3, atol=1e-3)  # the reconstruction itself is exact
    recon_step_ret = (reconstructed[:, 1:] - reconstructed[:, :-1]) / reconstructed[:, :-1]
    sd_reconstructed = np.std(recon_step_ret, axis=1)

    med_raw, med_recon = float(np.median(sd_raw)), float(np.median(sd_reconstructed))
    assert med_raw > 0.0
    ratio = med_recon / med_raw
    assert 0.9 <= ratio <= 1.1, (
        f"reconstructed sigma scale is off by {ratio:.2f}x vs the raw-close ground truth "
        f"(median raw={med_raw:.3g}, median reconstructed={med_recon:.3g}) - this is exactly the "
        f"bug repair round 1 fixed (the old code was off by ~3,500x)"
    )
    # Sanity: a 1-minute BTC bar's realised step vol is a few bps, not the ~1.7 the old bug gave.
    assert med_raw < 0.01 and med_recon < 0.01


# ------------------------------------------------------------- repair round 1, P2 (sanitise before use)

def _extreme_window(kind, B, lookback, last_close):
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.02, size=(B, lookback)).astype(np.float64)  # a plausible normalised window
    if kind == "normalised_zeros":
        # Production case: many bars equal to last_close, so the NORMALISED value is exactly 0
        # (window_relative: x = (raw - last_close) / pred_scale) - the case that triggered the
        # original bug (the last column of every real window is always exactly 0 this way).
        w = base.copy()
        w[:, ::7] = 0.0
        return w
    if kind == "normalised_negeps":
        # x + eps == 0 exactly in the OLD (buggy) formula's denominator; must not resurface as a
        # problem once eps is added to the RECONSTRUCTED raw price instead.
        w = base.copy()
        w[:, 10] = -1e-8
        return w
    if kind == "window_with_inf":
        w = base.copy()
        w[:, 20] = np.inf
        return w
    raise ValueError(kind)


@pytest.mark.parametrize("window_kind", ["normalised_zeros", "normalised_negeps", "window_with_inf"])
@pytest.mark.parametrize("sigma_source", ["realized_vol", "model"])
def test_pnl_utility_sanitises_sigma_before_use_on_extreme_windows(make_loss_model, sigma_source, window_kind):
    """NT-087 repair round 1, P2: a non-finite (or wildly extreme) ``sigma`` must be caught BEFORE
    it is used (the fix sanitises the reconstructed raw window, the step returns and ``sigma``
    itself, not only the final per-horizon loss), so it can never turn into a NaN gradient on the
    direction heads. Regression pin for QA's ``extreme.py`` cases."""
    B, lookback, last_close = 64, LOOKBACK, 110_000.0
    # LAMBDA_HD=0: hyper_decoherence_coupling_loss reads the same x_window with an UNGUARDED
    # tf.math.reduce_std (losses/functions.py, hyper_decoherence_coupling_loss) and is not
    # finite-safe against a +inf bar - a PRE-EXISTING custom_loss limitation, confirmed present
    # under LOSS_NAME='custom_loss' too and therefore out of this repair round's scope (which is
    # pnl_utility's own sigma path); flagged separately ("found, not done"). Disabling it here
    # isolates the P2 fix actually under test.
    cfg = Config(LOSS_NAME="pnl_utility", LAMBDA_PNL=1.0, PNL_SIGMA_SOURCE=sigma_source, LAMBDA_HD=0.0)
    m = make_loss_model(257.5, 0.3, config=cfg)
    rng = np.random.default_rng(6)
    x = tf.constant(_extreme_window(window_kind, B, lookback, last_close).astype(np.float32))
    y = tf.constant(rng.normal(0.0, 1.0, size=(B, 3)).astype(np.float32))
    lc = tf.constant(np.full((B, 1), last_close, dtype=np.float32))
    ext = tf.constant(rng.normal(0.0, 200.0, size=(B, 3)).astype(np.float32))
    price, dirs, var = _heads(rng)

    with tf.GradientTape(persistent=True) as tape:
        out = m.custom_loss(x, y, _y_pred(price, dirs, var), lc, ext)

    for field, val in zip(out._fields, out):
        assert np.isfinite(float(val)), f"non-finite LossComponents.{field}"

    # The gradient of pnl_val into the DIRECTION heads is the exact thing this P2 fix protects (a
    # non-finite sigma feeds r_tilde/c_tilde, which multiply the position a=2p-1). Check it
    # directly, for every window kind. Variance and price heads are intentionally NOT connected to
    # pnl_val's gradient (sigma is read under tf.stop_gradient even for PNL_SIGMA_SOURCE='model',
    # by design - section 1.1 of the research note - and price heads are not read at all), so
    # tape.gradient legitimately returns None for them here; that is not a bug.
    pnl_dir_grads = tape.gradient(out.pnl_val, dirs)
    for name, g in zip(["dir_h0", "dir_h1", "dir_h2"], pnl_dir_grads):
        assert g is not None and bool(tf.reduce_all(tf.math.is_finite(g))), \
            f"non-finite/missing pnl_val gradient on {name} ({window_kind}, {sigma_source})"

    if window_kind != "window_with_inf":
        # A literal +inf window bar also poisons the UNRELATED, pre-existing
        # hyper_decoherence_coupling_loss term's gradient via var heads (confirmed present under
        # plain LOSS_NAME='custom_loss' too, independent of LAMBDA_HD's value - see the repair
        # round 1 report, "found, not done"): 0 * NaN reintroduces NaN through the chain rule even
        # though hyper_decoherence_coupling_loss's own forward value is correctly zeroed by its
        # tf.where guard. That is out of this repair round's scope (a different loss term, not
        # touched by NT-087); pnl_val's own gradient (checked above, unconditionally) is exactly
        # what P2 is about and stays clean in every case.
        grads = tape.gradient(out.total, dirs + var + price)
        for name, g in zip(["dir"] * 3 + ["var"] * 3 + ["price"] * 3, grads):
            assert g is not None and bool(tf.reduce_all(tf.math.is_finite(g))), \
                f"non-finite/missing total gradient on the {name} head ({window_kind}, {sigma_source})"

"""NT-101: CALIB_MODE='gradient' (GradNorm-style loss-weight calibration).

A_losses.md recommendation 2: the default ('value') calibration rescales weights by loss
*value*, which amplifies a small-valued but steep-gradient term (soft ECE: 1 -> 2.907). The
alternative measures each term's *gradient norm* on the shared trunk and equalises that instead.
CALIB_MODE='value' stays the default, byte-for-byte the pre-NT-101 pass (tests/test_calibration_pass.py).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from neural_trade.training.lambda_calibration import _term_values, _trunk_variables, rescale_weight


def test_calib_mode_defaults_to_value_and_validates_choices():
    from neural_trade.core.config import Config
    from neural_trade.core.exceptions import InvalidConfigurationError

    cfg = Config()
    assert cfg.CALIB_MODE == "value"
    cfg.CALIB_MODE = "gradient"
    cfg.validate()  # no error
    cfg.CALIB_MODE = "bogus"
    with pytest.raises(InvalidConfigurationError):
        cfg.validate()


def test_rescale_weight_equalizes_gradient_norms_on_a_synthetic_two_term_loss(tf):
    """The exact function the gradient branch of calibrate_loss_weights uses to turn a measured
    gradient norm into a new weight. Two terms with very different, but constant, gradient
    magnitudes on a shared trunk variable: after calibration (damping=1, the CALIB_DAMPING
    default), ``new_weight_i * ||grad_i||`` is the same for both terms."""
    w = tf.Variable([1.0, -2.0, 0.5], dtype=tf.float32)  # the "shared trunk"
    with tf.GradientTape(persistent=True) as tape:
        term_a = tf.reduce_sum(3.0 * w)      # d(term_a)/dw = 3 everywhere -> ||grad|| = 3*sqrt(3)
        term_b = tf.reduce_sum(0.1 * w * w)  # d(term_b)/dw = 0.2*w -> a different, smaller norm
    grad_a = tape.gradient(term_a, [w])
    grad_b = tape.gradient(term_b, [w])
    del tape
    norm_a = float(tf.linalg.global_norm(grad_a))
    norm_b = float(tf.linalg.global_norm(grad_b))
    assert norm_a > 0.0 and norm_b > 0.0 and not np.isclose(norm_a, norm_b), "the two terms must start unequal"

    ref = float(np.mean([norm_a, norm_b]))
    lam_min, lam_max = 0.1, 20.0
    new_a = rescale_weight(1.0, norm_a, damping=1.0, ref=ref, lam_min=lam_min, lam_max=lam_max)
    new_b = rescale_weight(1.0, norm_b, damping=1.0, ref=ref, lam_min=lam_min, lam_max=lam_max)

    assert lam_min <= new_a <= lam_max and lam_min <= new_b <= lam_max
    np.testing.assert_allclose(new_a * norm_a, new_b * norm_b, rtol=1e-6)
    np.testing.assert_allclose(new_a * norm_a, ref, rtol=1e-6)


def test_rescale_weight_leaves_an_inactive_term_unchanged():
    # measured <= eps: the component is inactive (or disconnected from the trunk) -> keep orig.
    assert rescale_weight(0.7, 0.0, damping=1.0, ref=5.0, lam_min=0.1, lam_max=20.0) == 0.7


def test_trunk_variables_excludes_price_direction_variance_heads_and_indicator_vars(tf):
    """The shared trunk for gradient-norm calibration is every main-group variable except the
    price_h*/direction_h*/variance_h* head Dense layers (by layer name) and the indicator
    variables (routed to the second optimizer)."""
    inp = tf.keras.Input(shape=(4,), name="input")
    trunk = tf.keras.layers.Dense(3, name="shared_dense")(inp)
    price_h0 = tf.keras.layers.Dense(1, name="price_h0")(trunk)
    direction_h0_logit = tf.keras.layers.Dense(1, name="direction_h0_logit")(trunk)
    variance_h0 = tf.keras.layers.Dense(1, name="variance_h0")(trunk)
    other = tf.keras.layers.Dense(2, name="not_a_head")(trunk)
    model = tf.keras.Model(inp, [price_h0, direction_h0_logit, variance_h0, other])

    shared_dense = model.get_layer("shared_dense")
    fake_indicator_var = model.get_layer("not_a_head").trainable_variables[0]
    fake_custom_model = SimpleNamespace(
        trainable_variables=model.trainable_variables,
        _indicator_var_ids={id(fake_indicator_var)},
    )

    trunk_vars = _trunk_variables(fake_custom_model)
    trunk_names = {v.name for v in trunk_vars}

    for v in shared_dense.trainable_variables:
        assert v.name in trunk_names, f"{v.name} (trunk) was wrongly excluded"
    for layer_name in ("price_h0", "direction_h0_logit", "variance_h0"):
        for v in model.get_layer(layer_name).trainable_variables:
            assert v.name not in trunk_names, f"{v.name} (a head) was wrongly kept"
    assert fake_indicator_var.name not in trunk_names, "an indicator variable was wrongly kept"
    # the rest of 'not_a_head' (its bias) is neither a head nor an indicator var: stays in the trunk
    other_names = {v.name for v in model.get_layer("not_a_head").trainable_variables}
    assert (other_names - {fake_indicator_var.name}) <= trunk_names


def test_term_values_reads_by_attribute_not_by_position():
    """NT-101: the gradient branch reads LossComponents by attribute so it keeps working once a
    later change (nt-037 already did this once, adding dir_align_val/coherence_penalty_val after
    pnl_val; LossComponents now has 37 fields) appends fields — unpacking by position would
    silently misalign as soon as the tuple grows. Simulated here with two more hypothetical
    fields appended after today's last one, so this test keeps testing forward-compatibility
    regardless of how many fields nt-037 itself ends up adding."""
    from collections import namedtuple

    from neural_trade.core.outputs import LossComponents

    Extended = namedtuple("Extended", LossComponents._fields + ("future_field_a", "future_field_b"))
    values = {name: float(i) for i, name in enumerate(Extended._fields)}
    lc = Extended(**values)

    terms = _term_values(lc)
    assert terms["short"] == values["point_h0"]
    assert terms["vol"] == values["vol_loss"]
    assert terms["ext"] == pytest.approx((values["extended_h0"] + values["extended_h1"] + values["extended_h2"]) / 3.0)
    assert terms["dir"] == pytest.approx((values["dir_h0"] + values["dir_h1"] + values["dir_h2"]) / 3.0)
    assert terms["crps"] == pytest.approx((values["crps_h0"] + values["crps_h1"] + values["crps_h2"]) / 3.0)
    assert terms["t_perp"] == values["t_perp_total"]
    assert terms["hd"] == values["hd_val"]
    assert terms["ife"] == values["ife_val"]
    assert terms["vac_overflow"] == values["vac_overflow_val"]
    # the appended fields (nt-037's real ones, and these two hypothetical ones) are simply not
    # read: no crash, no misalignment.


def test_gradient_mode_end_to_end_records_weights_and_gradient_shares(tiny_config, tmp_path, synthetic_bars,
                                                                      monkeypatch):
    """Acceptance (1)+(2)+(3, QA repair round 1): CALIB_MODE='gradient' runs through
    train_and_evaluate's calibration pass, changes the served lambdas, and meta.json's
    calibration_lambdas (TrainResult.calibration_lambdas) carries the chosen weights plus, for
    every measured (non-skipped) term, its pre- and post-calibration gradient norm and its share
    of the pre-calibration total."""
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.CALIB_MODE = "gradient"
    cfg.validate()

    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)

    cal = result.calibration_lambdas
    assert cal is not None, "the gradient-mode calibration pass produced nothing (did it fail silently?)"
    assert cal["calib_mode"] == "gradient"

    recorded = {k: v for k, v in cal.items() if k.startswith("lambda_")}
    assert recorded, "the pass recorded no lambdas"
    live = result.model.get_lambda_values()
    for name, value in recorded.items():
        assert np.isclose(live[name], value, rtol=1e-6), (name, live[name], value)

    assert "grad_norms_pre" in cal and "grad_norms_post" in cal and "grad_shares" in cal
    assert set(cal["grad_norms_pre"]) == set(cal["grad_norms_post"]) == set(cal["grad_shares"])
    assert set(cal["grad_norms_pre"]) <= set(recorded), "a gradient share was reported for an unrescaled lambda"
    # vac/vac_overflow never receive a weight; ext/t_perp/casimir/hd/ife default to damping 0 (a
    # no-op): none of these five are measured, so none should appear in the pre/post/share dicts.
    assert not ({"lambda_extended_trend", "lambda_t_perp", "lambda_casimir", "lambda_hd", "lambda_ife"}
               & set(cal["grad_norms_pre"])), "a damping-0 (no-op) term was measured; should have been skipped"
    shares = list(cal["grad_shares"].values())
    assert all(s >= 0.0 for s in shares)
    assert np.isclose(sum(shares), 1.0, atol=1e-6), "gradient shares must sum to 1 over the measured terms"

    # Post-calibration norms should all equal ref_loss (within the [0.1, 20] clamp): that is the
    # whole point of CALIB_MODE=gradient (A_losses.md recommendation 2).
    for name, post in cal["grad_norms_post"].items():
        if 0.1 + 1e-6 < cal[name] < 20.0 - 1e-6:  # not clamp-limited: equality should be exact
            assert np.isclose(post, cal["ref_loss"], rtol=1e-3), (name, post, cal["ref_loss"])

    for name, value in recorded.items():
        assert 0.1 - 1e-9 <= value <= 20.0 + 1e-9, f"{name}={value} outside the [0.1, 20] clamp"


def test_active_terms_built_from_multiple_horizons_keep_a_nonzero_trunk_gradient(
        tiny_config, tmp_path, synthetic_bars, monkeypatch):
    """Regression test for a real NT-101 defect: ``_term_values`` averages three horizons
    (``(h0+h1+h2)/3``) for 'ext', 'dir', 'var', 'crps' and 'ece'. The first version of the
    gradient branch built that average from ``tf.GradientTape``-recorded tensors but OUTSIDE the
    tape's ``with`` block, so the +/÷ ops were never recorded: ``tape.gradient()`` on the
    resulting (perfectly real, nonzero-valued) sum found no path into the tape's graph and
    returned None for every trunk variable, silently leaving those five terms' weights at their
    original value every run (the same fallback used for a genuinely inactive term). The nine
    single-field terms (short/point/long/vol/t_perp/casimir/hd/ife/vac_overflow) were unaffected,
    which is what made this look like batch noise in a subset of terms rather than a tape-scope
    bug in every multi-horizon one. ``DIR_DEADBAND_BPS=0`` guarantees a nonzero direction mask
    (every example is labelled up or down), so 'dir' is genuinely active on every sampled batch.
    """
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.CALIB_MODE = "gradient"
    cfg.DIR_DEADBAND_BPS = 0.0     # every example gets a direction label: 'dir' cannot be masked to 0
    cfg.CALIB_DAMPING_TREND = 1.0  # force 'ext' to be measured too (default 0 would skip it entirely)
    cfg.validate()

    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)
    cal = result.calibration_lambdas
    assert cal is not None
    grad_norms = cal["grad_norms_pre"]

    # These four are built from a multi-horizon sum (dir/var/crps) or a lambda_trend_outer-scaled
    # sum (ext), per _total_term_tensors, and are measured (CALIB_DAMPING_TREND=1 forces 'ext' on)
    # at the tiny_config defaults (LAMBDA_DIR/VAR/CRPS/EXTENDED_TREND all > 0): each must show a
    # real, nonzero trunk gradient. Before the fix every one of them was exactly 0.0 here.
    for lam in ("lambda_dir", "lambda_var", "lambda_crps", "lambda_extended_trend"):
        assert grad_norms[lam] > 0.0, (
            f"{lam}'s measured gradient norm is exactly 0 with DIR_DEADBAND_BPS=0 (a nonzero-value, "
            f"zero-gradient term almost always means the combined tensor was built outside the "
            f"GradientTape's `with` block, not that the term is genuinely inactive): {grad_norms}"
        )


def test_gradient_mode_restores_lambdas_on_failure(tiny_config, tmp_path, synthetic_bars, monkeypatch):
    """The gradient branch must restore-on-failure exactly like the value branch (test_calibration_pass.py)."""
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.CALIB_MODE = "gradient"
    cfg.LAMBDA_DIR = 0.83  # a non-default value distinguishable from a "reset to 1.0"

    original = CustomTrainModel.custom_loss
    calls = {"n": 0}

    def explode_once(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom inside the gradient-norm sampler")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CustomTrainModel, "custom_loss", explode_once)
    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)

    assert calls["n"] >= 1
    assert result.calibration_lambdas is None
    assert np.isclose(float(result.model.lambda_dir), 0.83)


def test_gradient_mode_cost_is_reported(tiny_config, tmp_path, synthetic_bars, monkeypatch):
    """Acceptance (3): the calibration pass's cost. Times both CALIB_MODE values on the same tiny
    setup; the gradient pass does O(#rescaled terms) extra backward passes per sampled batch, so
    it costs measurably more. This is a report of the measured cost, not a fixed number: the CI
    machine's absolute speed is unknown, only that both finish and gradient mode is not free."""
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)

    def _run(mode):
        cfg = tiny_config
        cfg.CSV_PATH = str(tmp_path / "bars.csv")
        cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
        cfg.MODEL_PATH = str(tmp_path / "weights.h5")
        cfg.CALIB_MODE = mode
        t0 = time.perf_counter()
        result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)
        dt = time.perf_counter() - t0
        assert result.calibration_lambdas is not None
        return dt

    t_value = _run("value")
    t_gradient = _run("gradient")
    print(f"\n[NT-101] calibration-pass wall time on tiny_config: value={t_value:.3f}s gradient={t_gradient:.3f}s "
          f"(x{t_gradient / t_value:.2f})")
    assert t_value > 0.0 and t_gradient > 0.0
    assert t_gradient < 120.0, "the gradient pass on a tiny model should not take minutes"


@pytest.mark.slow
def test_default_model_gradient_mode_equalises_terms_as_they_enter_total(tmp_path, synthetic_bars, monkeypatch):
    """Acceptance (1), QA repair round 1 on 119319f: on the REAL DEFAULT model (not tiny_config —
    the full OHLCV + 14 indicator-family input, D-047), after a CALIB_MODE='gradient' calibration
    pass, recompute each default-damped term's trunk gradient norm **independently** of
    ``_total_term_tensors`` (written from scratch here, mirroring losses/functions.py's `total =`
    composition by inspection, not by import) and check they land within a stated tolerance of
    each other. Equalising the /3-per-horizon-average the pre-repair code measured is a different,
    weaker claim than equalising what `total` actually sums (horizon SUMS times outer
    multipliers) — this test exercises the real claim end to end.
    """
    import tensorflow as tf

    from neural_trade.core.config import Config
    from neural_trade.data.datasets import create_datasets
    from neural_trade.data.processor import DataProcessor
    from neural_trade.registries.models import Models
    from neural_trade.training.custom_model import CustomTrainModel
    from neural_trade.training.lambda_calibration import calibrate_loss_weights
    from neural_trade.training.trainer import train_and_evaluate  # noqa: F401 (keeps import style consistent)
    from neural_trade.utils.seeding import seed_everything

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = Config()
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.CALIB_MODE = "gradient"
    cfg.MAX_SEQUENCE_COUNT = 1200  # bound the real default model's calibration pass for a test
    cfg.DIR_DEADBAND_BPS = 0.0     # every example gets a direction label: 'dir'/'ece' can't mask to 0
    cfg.validate()
    seed_everything(0)

    dp = DataProcessor(cfg)
    df, close_values = dp.load_and_prepare_data()
    (X_train, y_train, lc_train, ext_train, _X_test, _y_test, _lc_test, _ext_test,
    _y_train_raw, _y_test_raw, _target_scaler) = dp.prepare_datasets(df, close_values)
    vb = dp.val_block
    train_ds, _val_ds = create_datasets(
        cfg, X_train, y_train, lc_train, ext_train,
        vb["X"], vb["y_scaled"], vb["last_close"], vb["extended_trends"],
    )

    base = Models.build(cfg.MODEL_NAME, cfg)
    pred_scale = float(np.std(y_train)) or 1.0
    model = CustomTrainModel(base_model=base, pred_scale=pred_scale, pred_mean=float(np.mean(y_train)),
                             config=cfg, inputs=base.inputs, outputs=base.outputs)

    cal = calibrate_loss_weights(model, train_ds, cfg, X_train.shape[0])
    assert cal is not None, "the real default model's calibration pass failed"

    from neural_trade.training.lambda_calibration import _LAMBDA_NAME_OF, _trunk_variables
    trunk_vars = _trunk_variables(model)
    # Default-damped terms (CALIB_DAMPING default 1.0; CALIB_DAMPING_TREND/PHYSICS default 0, so
    # ext/t_perp/casimir/hd/ife are legitimately excluded — same set the implementation measures).
    default_damped = ("short", "point", "long", "dir", "var", "vol", "crps", "ece")
    assert {_LAMBDA_NAME_OF[name] for name in default_damped} == set(cal["grad_norms_pre"])

    def independent_terms(lc):
        """Built from scratch here, independently of _total_term_tensors, evaluated AFTER
        calibration with the model's own (now-calibrated) weights, exactly as `total` would
        combine them on the next real training step. point_h0/h1/h2 and vol_loss already carry
        their own lambda inside the returned LossComponents field (see losses/functions.py:
        point_loss_h0_val = model.lambda_short * point_huber(...), vol_loss = vol_diff *
        model.lambda_vol); dir_h0/nll_h0/crps_h0/soft_ece_h0 do not (lambda_dir/var/crps/soft_ece
        are only applied when `total` builds total_dir_loss/total_nll/total_crps/total_soft_ece),
        so those four need their inner lambda multiplied in here explicitly.
        """
        return {
            'short': lc.point_h0, 'point': lc.point_h1, 'long': lc.point_h2,
            'dir': model.lambda_dir_outer * model.lambda_dir * (lc.dir_h0 + lc.dir_h1 + lc.dir_h2),
            'var': model.lambda_nll_outer * model.lambda_var * (lc.nll_h0 + lc.nll_h1 + lc.nll_h2),
            'vol': 0.1 * lc.vol_loss,
            'crps': model.lambda_crps * (lc.crps_h0 + lc.crps_h1 + lc.crps_h2),
            'ece': model.lambda_soft_ece * (lc.soft_ece_h0 + lc.soft_ece_h1 + lc.soft_ece_h2),
        }

    norms = {name: [] for name in default_damped}
    for batch in train_ds.take(3):
        x_batch, y_batch, last_batch, ext_batch = batch
        with tf.GradientTape(persistent=True) as tape:
            raw = model(x_batch, training=True)
            (*yp, vac_of) = raw
            lc = model.custom_loss(x_batch, y_batch, yp, last_batch, ext_batch, vacuum_overflow=vac_of)
            terms = independent_terms(lc)
        for name in default_damped:
            g = tape.gradient(terms[name], trunk_vars)
            present = [gg for gg in g if gg is not None]
            norms[name].append(float(tf.linalg.global_norm(present)) if present else 0.0)
        del tape

    mean_norms = {name: float(np.mean(v)) for name, v in norms.items()}
    values = list(mean_norms.values())
    assert all(v > 0.0 for v in values), f"an independently-recomputed term has a zero trunk gradient: {mean_norms}"
    spread = (max(values) - min(values)) / (float(np.mean(values)) + 1e-8)
    # A generous tolerance: this re-samples fresh batches from train_ds (shuffled independently of
    # calibration's own sample), not the exact batches calibration measured, so some spread from
    # sampling noise is expected; the claim under test is "roughly equal", not "bit-identical".
    assert spread < 0.5, f"independently-recomputed gradient norms are not equalised (spread {spread:.3f}): {mean_norms}"

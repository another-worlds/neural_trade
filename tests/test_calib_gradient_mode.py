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
    later change (e.g. nt-037) appends fields after ``pnl_val`` — unpacking by position would
    silently misalign as soon as the tuple grows."""
    from collections import namedtuple

    from neural_trade.core.outputs import LossComponents

    Extended = namedtuple("Extended", LossComponents._fields + ("dir_align_val", "coherence_penalty_val"))
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
    # the two nt-037 fields are simply not read: no crash, no misalignment.


def test_gradient_mode_end_to_end_records_weights_and_gradient_shares(tiny_config, tmp_path, synthetic_bars,
                                                                      monkeypatch):
    """Acceptance (1)+(2): CALIB_MODE='gradient' runs through train_and_evaluate's calibration
    pass, changes the served lambdas, and meta.json's calibration_lambdas (TrainResult.calibration_lambdas)
    carries the chosen weights plus each rescaled term's gradient norm and share of the total."""
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

    assert "grad_norms" in cal and "grad_shares" in cal
    assert set(cal["grad_norms"]) == set(cal["grad_shares"])
    assert set(cal["grad_norms"]) <= set(recorded), "a gradient share was reported for an unrescaled lambda"
    shares = list(cal["grad_shares"].values())
    assert all(s >= 0.0 for s in shares)
    assert np.isclose(sum(shares), 1.0, atol=1e-6), "gradient shares must sum to 1 over the rescaled terms"

    for name, value in recorded.items():
        assert 0.1 - 1e-9 <= value <= 20.0 + 1e-9, f"{name}={value} outside the [0.1, 20] clamp"


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

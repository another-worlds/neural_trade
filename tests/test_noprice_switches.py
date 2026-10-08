"""Tactical switches PRICE_HEAD and ACTIVE_HORIZONS (exploratory; defaults = today's behaviour exactly).

PRICE_HEAD='none': no price Dense layers, constant-zero price outputs, every price-only loss term exactly 0.
ACTIVE_HORIZONS=[...]: only the listed towers are built/trained; the others are constants and their loss
terms are 0. The default graph and loss are pinned against numbers recorded on the base commit.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config  # import neural_trade before tensorflow (CUDA DLLs on PATH)
from neural_trade.core.exceptions import InvalidConfigurationError

B = 64
ALL_ON = dict(LAMBDA_CASIMIR=1.0, LAMBDA_IFE=1.0, LAMBDA_HD=1.0, LAMBDA_T_PERP=1.0, LAMBDA_VAC=1.0,
              LAMBDA_VAC_OVERFLOW=1.0, LAMBDA_CRPS=1.0, LAMBDA_SOFT_ECE=1.0, LAMBDA_VOL=1.0)
PRICE_ONLY_TERMS = ("point_h0", "point_h1", "point_h2", "extended_h0", "extended_h1", "extended_h2",
                    "coherence_penalty_val", "ife_val", "vol_loss", "casimir_val", "vac_val")


# ------------------------------------------------------------------------------ config
def test_defaults_and_validation():
    cfg = Config()
    assert cfg.PRICE_HEAD == "on" and cfg.ACTIVE_HORIZONS == [0, 1, 2]
    assert Config(ACTIVE_HORIZONS=[1]).ACTIVE_HORIZONS == [1]
    assert Config(ACTIVE_HORIZONS=(0, 2)).ACTIVE_HORIZONS == [0, 2]
    assert Config(PRICE_HEAD="none").PRICE_HEAD == "none"
    for bad in ([], [3], [-1], [1, 1], [2, 1], [0.5], ["a"], [True]):
        with pytest.raises(InvalidConfigurationError):
            Config(ACTIVE_HORIZONS=bad)
    with pytest.raises(InvalidConfigurationError):
        Config(PRICE_HEAD="off")


# ------------------------------------------------------------------------------ defaults unchanged
def _fixed_batch():
    import tensorflow as tf

    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(0, 1, (B, 60)).astype(np.float32))
    y = tf.constant(rng.normal(0, 1, (B, 3)).astype(np.float32))
    lc = tf.constant((110000 + rng.normal(0, 550, (B, 1))).astype(np.float32))
    ext = tf.constant(rng.normal(0, 200, (B, 3)).astype(np.float32))
    price = [tf.constant(rng.normal(0, 1, (B, 1)).astype(np.float32)) for _ in range(3)]
    dirs = [tf.constant(rng.uniform(0.05, 0.95, (B, 1)).astype(np.float32)) for _ in range(3)]
    var = [tf.constant(rng.uniform(0.5, 2.0, (B, 1)).astype(np.float32)) for _ in range(3)]
    ov = tf.constant(rng.uniform(0, 1, (B, 1)).astype(np.float32))
    heads = (price[0], dirs[0], var[0], price[1], dirs[1], var[1], price[2], dirs[2], var[2])
    return x, y, lc, ext, heads, ov, price, dirs, var


def test_default_graph_is_the_base_commits_graph(tf):
    """Recorded on the base commit (nt-tactical a4ba2f4): 316751 parameters, 87 layers, the same
    price/direction/variance layer names. The switches add no layer at their defaults."""
    from neural_trade.models.gru_attention import build_gru_attention

    m = build_gru_attention(Config())
    assert m.count_params() == 316751 and len(m.layers) == 87
    names = [layer.name for layer in m.layers]
    for i in range(3):
        for head in ("price", "direction", "variance"):
            assert f"{head}_h{i}" in names and f"{head}_h{i}_clip" in names
    assert build_gru_attention(Config(PRICE_HEAD="on", ACTIVE_HORIZONS=[0, 1, 2])).count_params() == 316751


def test_default_loss_equals_the_base_commits_loss(make_loss_model):
    """Components recorded on the base commit for this fixed batch (rtol 1e-5)."""
    x, y, lc, ext, heads, ov, *_ = _fixed_batch()
    m = make_loss_model(config=Config(**ALL_ON))
    out = m.custom_loss(x, y, heads, lc, ext, vacuum_overflow=ov)
    recorded = {'total': 17.628054, 'point_h0': 0.860887, 'point_h1': 0.594293, 'point_h2': 0.617704,
                'extended_h0': 0.092253, 'extended_h1': 0.084604, 'extended_h2': 0.095261,
                'dir_h0': 1.025881, 'dir_h1': 0.863248, 'dir_h2': 0.820238,
                'nll_h0': 2.320793, 'nll_h1': 1.741975, 'nll_h2': 1.930252, 'vol_loss': 0.089039,
                'crps_h0': 1.015702, 'crps_h1': 0.761252, 'crps_h2': 0.794592,
                'soft_ece_h0': 0.404306, 'soft_ece_h1': 0.364892, 'soft_ece_h2': 0.277787,
                't_perp_total': 0.905878, 'casimir_val': 0.017257, 'vac_val': 0.057838, 'hd_val': 1.109795,
                'ife_val': 0.0, 'vac_overflow_val': 0.326063, 'coherence_penalty_val': 0.536397}
    for name, v in recorded.items():
        np.testing.assert_allclose(float(getattr(out, name)), v, rtol=1e-5, atol=1e-6, err_msg=name)


# ------------------------------------------------------------------------------ PRICE_HEAD='none'
def test_price_none_builds_without_price_dense_layers_and_outputs_zero(tf):
    from neural_trade.models.gru_attention import build_gru_attention

    base = build_gru_attention(Config())
    m = build_gru_attention(Config(PRICE_HEAD="none"))
    names = [layer.name for layer in m.layers]
    assert not any(isinstance(layer, tf.keras.layers.Dense) and layer.name.startswith("price_h")
                   for layer in m.layers)
    assert not any(v.name.startswith("price_h") for v in m.weights)
    assert m.count_params() == base.count_params() - 3 * (16 + 1)  # three Dense(1) on 16 inputs
    assert len(m.outputs) == 10 and all(f"direction_h{i}" in names for i in range(3))
    out = m(np.random.default_rng(0).normal(size=(5, m.input_shape[1], *m.input_shape[2:])).astype("float32"))
    for i in (0, 3, 6):
        assert out[i].shape == (5, 1) and float(tf.reduce_max(tf.abs(out[i]))) == 0.0


def test_price_none_zeroes_the_price_only_terms_and_keeps_the_rest(make_loss_model, tf):
    x, y, lc, ext, _heads, ov, _price, dirs, var = _fixed_batch()
    zero = tf.zeros((B, 1))
    dirs = [tf.Variable(d) for d in dirs]
    var = [tf.Variable(v) for v in var]
    heads = (zero, dirs[0], var[0], zero, dirs[1], var[1], zero, dirs[2], var[2])
    m = make_loss_model(config=Config(PRICE_HEAD="none", **ALL_ON))
    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, heads, lc, ext, vacuum_overflow=ov)
    for name in PRICE_ONLY_TERMS:
        assert float(getattr(out, name)) == 0.0, name
    # kept: direction, NLL and CRPS with the centre at 0, T-perp, soft ECE, HD, vac_overflow (residual |y|)
    for name in ("dir_h0", "nll_h0", "nll_h2", "crps_h1", "t_perp_total", "soft_ece_h0", "hd_val",
                 "vac_overflow_val"):
        assert float(getattr(out, name)) > 0.0, name
    assert np.isfinite(float(out.total))
    g = tape.gradient(out.total, dirs + var)
    for gi in g:
        assert gi is not None and bool(tf.reduce_all(tf.math.is_finite(gi))) and float(tf.reduce_max(tf.abs(gi))) > 0
    # NLL with centre 0 is the Gaussian NLL of y itself
    y0 = y.numpy()[:, 0:1]
    v0 = np.maximum(var[0].numpy(), 1e-4)
    expect = np.mean(0.5 * (1.8378770664093453 + np.log(v0 + 1e-8)) + 0.5 * y0 ** 2 / (v0 + 1e-8))
    np.testing.assert_allclose(float(out.nll_h0), expect, rtol=1e-4)


# ------------------------------------------------------------------------------ ACTIVE_HORIZONS
def test_active_horizons_zeroes_the_inactive_horizons_terms(make_loss_model, tf):
    x, y, lc, ext, heads, ov, price, dirs, var = _fixed_batch()
    m = make_loss_model(config=Config(ACTIVE_HORIZONS=[1], **ALL_ON))
    out = m.custom_loss(x, y, heads, lc, ext, vacuum_overflow=ov)
    for name in ("point_h0", "point_h2", "extended_h0", "extended_h2", "dir_h0", "dir_h2", "nll_h0", "nll_h2",
                 "crps_h0", "crps_h2", "soft_ece_h0", "soft_ece_h2", "casimir_val", "vac_val", "ife_val",
                 "hd_val", "coherence_penalty_val"):
        assert float(getattr(out, name)) == 0.0, name
    for name in ("point_h1", "extended_h1", "dir_h1", "nll_h1", "crps_h1", "soft_ece_h1", "vol_loss",
                 "t_perp_total", "vac_overflow_val"):
        assert float(getattr(out, name)) > 0.0, name
    # the active horizon's terms equal the all-active run's
    full = make_loss_model(config=Config(**ALL_ON)).custom_loss(x, y, heads, lc, ext, vacuum_overflow=ov)
    for name in ("point_h1", "dir_h1", "nll_h1", "crps_h1", "soft_ece_h1"):
        np.testing.assert_allclose(float(getattr(out, name)), float(getattr(full, name)), rtol=1e-6, err_msg=name)


def test_active_horizons_inactive_heads_get_zero_gradient(make_loss_model, tf):
    x, y, lc, ext, _h, ov, price, dirs, var = _fixed_batch()
    price = [tf.Variable(p) for p in price]
    dirs = [tf.Variable(d) for d in dirs]
    var = [tf.Variable(v) for v in var]
    heads = (price[0], dirs[0], var[0], price[1], dirs[1], var[1], price[2], dirs[2], var[2])
    m = make_loss_model(config=Config(ACTIVE_HORIZONS=[1], **ALL_ON))
    with tf.GradientTape() as tape:
        out = m.custom_loss(x, y, heads, lc, ext, vacuum_overflow=ov)
    g = tape.gradient(out.total, price + dirs + var)
    inactive = [g[0], g[2], g[3 + 0], g[3 + 2], g[6 + 0], g[6 + 2]]
    for gi in inactive:
        assert gi is None or float(tf.reduce_max(tf.abs(gi))) == 0.0
    for gi in (g[1], g[3 + 1], g[6 + 1]):
        assert gi is not None and float(tf.reduce_max(tf.abs(gi))) > 0.0


# ------------------------------------------------------------------------------ built models
def _build(tf, cfg, tmp_path, synthetic_bars, **over):
    from neural_trade.data.datasets import create_datasets
    from neural_trade.data.processor import DataProcessor
    from neural_trade.models.registry import Models
    from neural_trade.training.custom_model import CustomTrainModel

    tf.keras.utils.set_random_seed(0)
    for k, v in over.items():
        setattr(cfg, k, v)
    csv_path = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv_path, index=False)
    cfg.CSV_PATH = str(csv_path)
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.validate()
    dp = DataProcessor(cfg)
    df, close = dp.load_and_prepare_data()
    (X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te, y_tr, _y_te, _scaler) = dp.prepare_datasets(df, close)
    base = Models.build(getattr(cfg, 'MODEL_NAME', None), cfg)
    std = float(np.std(y_tr))
    model = CustomTrainModel(
        base_model=base, pred_scale=std if std > 0 else 1.0, pred_mean=float(np.mean(y_tr)),
        lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
        lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND, lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND,
        lambda_dir=cfg.LAMBDA_DIR, config=cfg, inputs=base.inputs, outputs=base.outputs)
    train_ds, val_ds = create_datasets(cfg, X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LR))
    return model, train_ds


def _var_names(model):
    return [v.name for v in model.trainable_variables]


def test_active_horizons_builds_only_the_active_tower(tf, tiny_close_only_config, tmp_path, synthetic_bars):
    model, train_ds = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, ACTIVE_HORIZONS=[1])
    names = _var_names(model)
    for i in (0, 2):
        assert not any(n.startswith((f"price_h{i}/", f"direction_h{i}", f"variance_h{i}/")) for n in names), i
    assert any(n.startswith("variance_h1/") for n in names) and any(n.startswith("price_h1/") for n in names)
    x, y, lc, ext = next(iter(train_ds))
    out = model(x, training=False)
    assert len(out) == 10
    assert float(tf.reduce_max(tf.abs(out[1] - 0.5))) == 0.0 and float(tf.reduce_max(tf.abs(out[2] - 1.0))) == 0.0
    assert float(tf.reduce_max(tf.abs(out[0]))) == 0.0 and float(tf.reduce_max(tf.abs(out[8] - 1.0))) == 0.0
    # gradient of the real loss reaches h1's heads and not a tower that does not exist
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        c = model.custom_loss(x, y, out[:9], lc, ext, vacuum_overflow=out[9])
    grads = tape.gradient(c.total, model.trainable_variables)
    by_name = dict(zip(_var_names(model), grads))
    for n, g in by_name.items():
        if n.startswith(("variance_h1/", "direction_h1")):
            assert g is not None and float(tf.reduce_max(tf.abs(g))) > 0.0, n


def test_both_switches_together_train_one_step_with_the_probe_on(tf, tiny_close_only_config, tmp_path,
                                                                 synthetic_bars):
    model, train_ds = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, PRICE_HEAD="none",
                             ACTIVE_HORIZONS=[1], PROBE_GRADIENTS=True, PROBE_EVERY=1)
    assert not any(n.startswith("price_h") for n in _var_names(model))
    hist = model.fit(train_ds, epochs=1, steps_per_epoch=2, verbose=0)
    assert np.isfinite(hist.history["loss"][-1])
    assert hist.history["nonfinite_grad_steps"][-1] == 0
    assert np.isfinite(hist.history["grad_global_norm"][-1])
    assert all(np.all(np.isfinite(w)) for w in model.get_weights())
    probe = {k: v[-1] for k, v in hist.history.items() if k.startswith("probe")}
    assert probe and all(np.isfinite(v) for v in probe.values())


def test_price_none_trains_with_finite_gradients_for_direction_and_variance(tf, tiny_close_only_config, tmp_path,
                                                                           synthetic_bars):
    model, train_ds = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, PRICE_HEAD="none")
    x, y, lc, ext = next(iter(train_ds))
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        c = model.custom_loss(x, y, out[:9], lc, ext, vacuum_overflow=out[9])
    for name in PRICE_ONLY_TERMS:
        assert float(getattr(c, name)) == 0.0, name
    assert np.isfinite(float(c.total))
    grads = tape.gradient(c.total, model.trainable_variables)
    by_name = dict(zip(_var_names(model), grads))
    for n, g in by_name.items():
        if g is not None:
            assert bool(tf.reduce_all(tf.math.is_finite(g))), n
        if n.startswith(("variance_h", "direction_h")) and "skip" not in n:
            assert g is not None and float(tf.reduce_max(tf.abs(g))) > 0.0, n
    hist = model.fit(train_ds, epochs=1, steps_per_epoch=2, verbose=0)
    assert np.isfinite(hist.history["loss"][-1]) and hist.history["nonfinite_grad_steps"][-1] == 0


# ------------------------------------------------------------------------------ calibration
@pytest.mark.parametrize("mode", ["value", "gradient"])
def test_calibration_skips_the_zeroed_terms_under_price_none(tiny_config, tmp_path, synthetic_bars, monkeypatch,
                                                             mode):
    from neural_trade.training.trainer import train_and_evaluate

    monkeypatch.chdir(tmp_path)
    synthetic_bars.to_csv(tmp_path / "bars.csv", index=False)
    cfg = tiny_config
    cfg.CSV_PATH = str(tmp_path / "bars.csv")
    cfg.SCALER_PATH = str(tmp_path / "scaler.joblib")
    cfg.MODEL_PATH = str(tmp_path / "weights.h5")
    cfg.CALIB_MODE = mode
    cfg.PRICE_HEAD = "none"
    cfg.LAMBDA_CASIMIR = 0.5
    cfg.LAMBDA_IFE = 0.3
    cfg.validate()
    result = train_and_evaluate(config=cfg, epochs=0, force=True, calibrate=True, fit_calibration=False)
    cal = result.calibration_lambdas
    assert cal is not None and not cal.get("calib_failed")
    assert cal["lambda_vol"] == 0.0 and float(result.model.lambda_vol) == 0.0  # not lifted to CALIB_LAMBDA_MIN
    for name, configured in (("lambda_short", cfg.LAMBDA_SHORT), ("lambda_point", cfg.LAMBDA_POINT),
                             ("lambda_long", cfg.LAMBDA_LONG), ("lambda_extended_trend", cfg.LAMBDA_EXTENDED_TREND),
                             ("lambda_casimir", 0.5), ("lambda_ife", 0.3)):
        assert cal[name] == pytest.approx(configured), name
    assert cal["lambda_dir"] > 0.0 and cal["lambda_var"] > 0.0
    if mode == "gradient":
        for name in ("lambda_short", "lambda_point", "lambda_long", "lambda_extended_trend", "lambda_vol",
                     "lambda_casimir", "lambda_ife"):
            assert name not in cal["grad_norms_pre"], name


# ------------------------------------------------------------------------------ screen scoring
@pytest.mark.slow
def test_screen_scores_constant_outputs_and_saves_predictions(tmp_path, synthetic_bars):
    import json

    from neural_trade.experiments.screen import ScreenSpec, run_screen

    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    spec = ScreenSpec.from_dict({
        "schema_version": 1, "name": "tiny_noprice", "description": "test",
        "overrides": {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "N_FOLDS": 2, "VAL_FRACTION": 0.1,
                      "CAL_FRACTION": 0.1, "DATA_END_PROTECTED_DAYS": 0.001,
                      "PRICE_HEAD": "none", "ACTIVE_HORIZONS": [1]},
        "grid": {"axes": {}}, "slices": ["2025-10-13T04:19:00+00:00"], "seeds": [0],
        "run": {"calibrate": False, "epochs": 1, "save_predictions": True}, "rules": {"finite": True}})
    report = run_screen(spec, store=tmp_path)
    assert report.ran == 1
    row = json.loads((tmp_path / "screens" / spec.name / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["direction_auc"]["h0"]["auc"] is None and row["direction_auc"]["h2"]["auc"] is None
    assert row["head_metrics"]["h0"]["direction"]["auc"] is None
    assert row["head_metrics"]["h0"]["delta"]["corr"] is None
    assert row["direction_auc"]["h1"]["auc"] is not None
    npz = tmp_path / "screens" / spec.name / "preds" / f"{row['trial_key']}.npz"
    with np.load(npz) as z:
        assert np.ptp(z["delta_h1"]) == 0.0 and np.ptp(z["p_up_h1"]) > 0.0 and np.ptp(z["p_up_h0"]) == 0.0

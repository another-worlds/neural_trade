"""Tactical hypothesis E: Config.PATH_HEAD, LAMBDA_PATH, LAMBDA_PATH_IND (exploratory; default off).

The default graph, datasets and loss are pinned (see also tests/test_noprice_switches.py, which pins the
default graph at 316751 parameters); with the switch on the model gets an extra [B, P] output after the
10 contract outputs, the batches a fifth element (the scaled future path), and the loss two terms.
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config  # import neural_trade before tensorflow (CUDA DLLs on PATH)
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.data.windowing import make_path_targets, make_sequences_with_extended_trends


# ------------------------------------------------------------------------------ config
def test_defaults_and_validation():
    cfg = Config()
    assert cfg.PATH_HEAD is False and cfg.LAMBDA_PATH == 0.0 and cfg.LAMBDA_PATH_IND == 0.0
    assert cfg.PATH_IND_EMA == [5, 10] and cfg.PATH_IND_RSI == 10
    assert Config(PATH_IND_EMA=(3, 6)).PATH_IND_EMA == [3, 6]
    with pytest.raises(InvalidConfigurationError):
        Config(LAMBDA_PATH_IND=1.0)  # no predicted path without the head
    with pytest.raises(InvalidConfigurationError):
        Config(PATH_HEAD=True, PATH_IND_EMA=[])
    Config(PATH_HEAD=True, LAMBDA_PATH=0.5, LAMBDA_PATH_IND=1.0)


# ------------------------------------------------------------------------------ the target
def _cfg_small(**kw):
    return Config(HORIZON_STEPS=[2, 3, 5], EXTENDED_TREND_PERIODS=[2, 3, 5], LOOKBACK=8, MOMENTUM_CLIP_MAX=8, **kw)


def test_path_target_is_the_next_p_close_deltas_and_the_price_targets_are_its_entries():
    cfg = _cfg_small()
    close = np.cumsum(np.random.default_rng(1).normal(0, 1, 60)) + 100.0
    path = make_path_targets(cfg, close, cfg.LOOKBACK)
    X, y, lc, _ext = make_sequences_with_extended_trends(cfg, close, cfg.LOOKBACK)
    assert path.shape == (X.shape[0], 5) and path.dtype == np.float32
    start = max(cfg.LOOKBACK, max(cfg.EXTENDED_TREND_PERIODS))
    for k in (0, 1, X.shape[0] - 1):
        i = start + k  # the window is close[i-8:i]; its last close is close[i-1]
        np.testing.assert_allclose(path[k], close[i:i + 5] - close[i - 1], atol=1e-4)
        np.testing.assert_allclose(path[k][[1, 2, 4]], y[k], atol=1e-4)  # horizons 2, 3, 5 -> entries h-1
    # the existing arrays are the same function of the data with or without the path target
    assert X.shape[0] == 60 - 5 + 1 - start


def test_path_target_never_reads_bars_beyond_the_horizon():
    cfg = _cfg_small()
    close = np.cumsum(np.random.default_rng(2).normal(0, 1, 60)) + 100.0
    base = make_path_targets(cfg, close, cfg.LOOKBACK)
    start, p = 8, 5
    m = 40
    bumped = close.copy()
    bumped[m] += 1000.0
    after = make_path_targets(cfg, bumped, cfg.LOOKBACK)
    for k in range(base.shape[0]):
        i = start + k
        reads = range(i - 1, i + p)  # last close and the next P bars
        if m in reads:
            assert not np.allclose(base[k], after[k])
        else:
            np.testing.assert_array_equal(base[k], after[k])


def test_window_step_and_trim_keep_the_anchors_aligned():
    from neural_trade.data.processor import DataProcessor

    cfg = _cfg_small(WINDOW_STEP=3, MAX_SEQUENCE_COUNT=10, INPUT_SERIES=['close'])
    close = np.cumsum(np.random.default_rng(5).normal(0, 1, 200)) + 100.0
    dp = DataProcessor(cfg)
    X, y, lc, ext, _Xm = dp.build_windows(close)
    path = dp.build_path_targets(close)
    assert path.shape[0] == X.shape[0] == 10
    np.testing.assert_allclose(path[:, [1, 2, 4]], y, atol=1e-4)


# ------------------------------------------------------------------------------ the indicator loss
def test_features_have_the_documented_values(tf):
    from neural_trade.losses.path_loss import path_feature_names, path_features

    assert path_feature_names([5, 10], 10) == ["ema_slope_5", "ema_slope_10", "rsi_10", "efficiency"]
    up = np.tile(np.linspace(0.2, 4.0, 20, dtype="float32"), (3, 1))       # a straight rise
    flat = np.zeros((3, 20), dtype="float32")
    f_up = path_features(tf.constant(up)).numpy()
    f_flat = path_features(tf.constant(flat)).numpy()
    assert f_up.shape == (3, 4)
    assert np.all(f_up[:, :2] > 0) and f_up[0, 2] > 0.8 and f_up[0, 3] > 0.9   # rising, RSI high, efficient
    np.testing.assert_allclose(f_flat[:, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(f_flat[:, 2], 0.5, atol=1e-3)
    np.testing.assert_allclose(f_flat[:, 3], 0.0, atol=1e-6)
    f_down = path_features(tf.constant(-up)).numpy()
    assert np.all(f_down[:, :2] < 0) and f_down[0, 2] < 0.2 and f_down[0, 3] < -0.9


def test_constant_prediction_gets_a_positive_shape_loss_when_the_real_path_trends(tf):
    from neural_trade.losses.path_loss import path_indicator_loss

    rng = np.random.default_rng(3)
    slope = rng.normal(0, 0.3, (64, 1)).astype("float32")                   # trending real paths
    real = slope * np.arange(1, 21, dtype="float32")[None, :] + rng.normal(0, 0.05, (64, 20)).astype("float32")
    const = tf.zeros((64, 20))
    loss_const, per = path_indicator_loss(tf.constant(real), tf.constant(real))
    assert float(loss_const) == pytest.approx(0.0, abs=1e-9)
    loss_flat, per_flat = path_indicator_loss(const, tf.constant(real))
    assert float(loss_flat) > 0.3 and bool(tf.reduce_all(per_flat > 0.05))   # every feature punished
    # the loss is in real-feature standard deviations: a noisy copy is far better than a flat line
    noisy = real + rng.normal(0, 0.05, real.shape).astype("float32")
    assert float(path_indicator_loss(tf.constant(noisy), tf.constant(real))[0]) < 0.2 * float(loss_flat)


def test_loss_terms_are_skipped_at_weight_zero_and_finite_otherwise(tf):
    from neural_trade.losses.path_loss import path_loss_terms

    pred, real = tf.zeros((8, 20)), tf.constant(np.random.default_rng(4).normal(0, 1, (8, 20)).astype("float32"))
    a, b = path_loss_terms(pred, real, Config(PATH_HEAD=True))
    assert float(a) == 0.0 and float(b) == 0.0
    a, b = path_loss_terms(pred, real, Config(PATH_HEAD=True, LAMBDA_PATH=1.0, LAMBDA_PATH_IND=1.0))
    assert np.isfinite(float(a)) and float(a) > 0 and np.isfinite(float(b)) and float(b) > 0


# ------------------------------------------------------------------------------ the model
def test_default_model_has_ten_outputs_and_the_switch_adds_one_after_them(tf):
    from neural_trade.models.gru_attention import build_gru_attention

    base = build_gru_attention(Config())
    assert len(base.outputs) == 10 and base.count_params() == 316751 and len(base.layers) == 87
    on = build_gru_attention(Config(PATH_HEAD=True))
    assert len(on.outputs) == 11 and [o.shape[1:] for o in on.outputs[:10]] == [o.shape[1:] for o in base.outputs]
    p = max(Config().HORIZON_STEPS)
    assert tuple(on.outputs[10].shape) == (None, p)
    path_layer = on.get_layer("path_head")
    assert on.count_params() == base.count_params() + path_layer.count_params()
    out = on(np.random.default_rng(0).normal(size=(5, *on.input_shape[1:])).astype("float32"))
    assert out[10].shape == (5, p) and bool(tf.reduce_all(tf.math.is_finite(out[10])))


def test_registry_validation_and_postprocess_tolerate_the_extra_output(tf):
    from neural_trade.core.postprocess import heads_to_predictions
    from neural_trade.models.gru_attention import build_gru_attention
    from neural_trade.models.registry import ensure_predictive_outputs

    cfg = Config(PATH_HEAD=True)
    m = ensure_predictive_outputs(build_gru_attention(cfg), "gru")
    x = np.random.default_rng(0).normal(size=(4, *m.input_shape[1:])).astype("float32")
    heads = [np.asarray(h) for h in m(x)]
    preds = heads_to_predictions(heads, 4, 1.0, 0.0, cfg)
    assert set(preds["delta"]) == set(heads_to_predictions(heads[:10], 4, 1.0, 0.0, cfg)["delta"])


# ------------------------------------------------------------------------------ training
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
    train_ds, val_ds = create_datasets(cfg, X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te,
                                       path_train=dp.path_train, path_test=dp.path_test)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LR))
    return model, train_ds, val_ds, dp, y_tr_s


def test_default_batches_are_four_tuples_and_the_loss_hook_is_a_noop(tf, tiny_close_only_config, tmp_path,
                                                                    synthetic_bars):
    model, train_ds, _val, dp, _y = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars)
    batch = next(iter(train_ds))
    assert len(batch) == 4 and dp.path_train is None and "path_scaled" not in dp.val_block
    assert model._add_path_loss("sentinel", None, None) == "sentinel" and model._path_means == {}


def test_path_target_is_scaled_like_the_price_targets(tf, tiny_close_only_config, tmp_path, synthetic_bars):
    _m, train_ds, _v, dp, y_tr_s = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, PATH_HEAD=True)
    x, y, lc, ext, path = next(iter(train_ds))
    cfg = dp.config
    p = max(cfg.HORIZON_STEPS)
    assert path.shape[1] == p and path.shape[0] == x.shape[0]
    pos = [h - 1 for h in cfg.HORIZON_STEPS]
    np.testing.assert_allclose(path.numpy()[:, pos], y.numpy(), atol=1e-4)  # same units, same rows


def test_gradients_reach_the_path_head_and_the_trunk(tf, tiny_close_only_config, tmp_path, synthetic_bars):
    from neural_trade.losses.path_loss import path_loss_terms

    model, train_ds, _v, _dp, _y = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, PATH_HEAD=True,
                                          LAMBDA_PATH_IND=1.0, LAMBDA_PATH=0.5)
    x, y, lc, ext, path = next(iter(train_ds))
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        path_term, ind_term = path_loss_terms(out[10], path, model.config)
        total = path_term + ind_term
    assert np.isfinite(float(path_term)) and np.isfinite(float(ind_term)) and float(ind_term) > 0
    grads = dict(zip([v.name for v in model.trainable_variables], tape.gradient(total, model.trainable_variables)))
    head = [g for n, g in grads.items() if n.startswith("path_head")]
    trunk = [g for n, g in grads.items() if g is not None and not n.startswith(("path_head", "price_h", "direction_h",
                                                                               "variance_h"))]
    assert head and all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in head)
    assert any(float(tf.reduce_max(tf.abs(g))) > 0 for g in head)
    assert trunk and all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in trunk)
    assert any(float(tf.reduce_max(tf.abs(g))) > 0 for g in trunk)


def test_total_gains_exactly_the_weighted_terms(tf, tiny_close_only_config, tmp_path, synthetic_bars):
    from neural_trade.losses.path_loss import path_loss_terms

    model, train_ds, _v, _dp, _y = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars, PATH_HEAD=True,
                                          LAMBDA_PATH_IND=2.0, LAMBDA_PATH=0.5)
    x, y, lc, ext, path = next(iter(train_ds))
    out = model(x, training=False)
    c = model.custom_loss(x, y, out[:9], lc, ext, vacuum_overflow=None)
    c2 = model._add_path_loss(c, out, path)
    a, b = path_loss_terms(out[10], path, model.config)
    np.testing.assert_allclose(float(c2.total), float(c.total) + 0.5 * float(a) + 2.0 * float(b), rtol=1e-5)
    np.testing.assert_allclose(float(model._path_means["path_ind_loss"].result()), float(b), rtol=1e-5)


def test_price_none_plus_path_head_trains_two_steps(tf, tiny_close_only_config, tmp_path, synthetic_bars):
    model, train_ds, val_ds, _dp, _y = _build(tf, tiny_close_only_config, tmp_path, synthetic_bars,
                                              PRICE_HEAD="none", ACTIVE_HORIZONS=[1], PATH_HEAD=True,
                                              LAMBDA_PATH_IND=1.0, LAMBDA_PATH=0.5)
    assert not any(v.name.startswith("price_h") for v in model.trainable_variables)
    before = [w.copy() for w in model.get_weights()]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=1, steps_per_epoch=2, validation_steps=1, verbose=0)
    h = hist.history
    assert np.isfinite(h["loss"][-1]) and h["nonfinite_grad_steps"][-1] == 0
    assert np.isfinite(h["path_ind_loss"][-1]) and h["path_ind_loss"][-1] > 0 and np.isfinite(h["val_path_loss"][-1])
    assert all(np.all(np.isfinite(w)) for w in model.get_weights())
    head_w = [(n.name, a, b) for n, a, b in zip(model.weights, before, model.get_weights()) if "path_head" in n.name]
    assert head_w and any(not np.allclose(a, b) for _n, a, b in head_w)  # the path head actually moved

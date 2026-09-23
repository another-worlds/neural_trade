"""Logged step metrics are EPOCH aggregates, not the last batch.

Keras keeps only the dict returned by the final train/test step of an epoch. Before this
fix the steps returned per-batch tensors, so every train_*/val_* value - including the
val_loss that drives EarlyStopping, ModelCheckpoint and ReduceLROnPlateau - described one
batch (~50 of 2,866 validation samples) and the M1-M3 gates were read off noise.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

B, N = 64, 250  # 4 batches: 64, 64, 64, 58


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (N, 60)).astype(np.float32)
    y = rng.normal(0, 1, (N, 3)).astype(np.float32)
    lc = (110_000 + rng.normal(0, 300, (N, 1))).astype(np.float32)
    ext = rng.normal(0, 200, (N, 3)).astype(np.float32)
    return x, y, lc, ext


def _ds(arrays):
    return tf.data.Dataset.from_tensor_slices(arrays).batch(B)


@pytest.fixture
def model(make_loss_model):
    tf.keras.utils.set_random_seed(0)
    m = make_loss_model(261.0, 3.2)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return m


def test_val_loss_is_the_sample_weighted_mean_over_all_batches(model):
    arrays = _data()
    logs = model.evaluate(_ds(arrays), verbose=0, return_dict=True)
    per_batch, sizes = [], []
    for xb, yb, lcb, extb in _ds(arrays):
        preds = model(xb, training=False)
        comps = model.custom_loss(xb, yb, preds[:9], lcb, extb, vacuum_overflow=None)
        per_batch.append(float(comps.total))
        sizes.append(int(xb.shape[0]))
    expected = np.average(per_batch, weights=sizes)
    assert len(set(np.round(per_batch, 6))) > 1, "batches must differ for this test to mean anything"
    np.testing.assert_allclose(logs["loss"], expected, rtol=1e-5)
    assert abs(logs["loss"] - per_batch[-1]) > 1e-6, "must not be the last batch"


def test_direction_and_pit_metrics_match_the_full_arrays(model):
    x, y, lc, ext = _data(1)
    logs = model.evaluate(_ds((x, y, lc, ext)), verbose=0, return_dict=True)

    preds = model.predict(tf.data.Dataset.from_tensor_slices(x).batch(B), verbose=0)
    y_raw = y * 261.0 + 3.2
    ret = y_raw / (lc + 1e-8)
    d = model.config.DIR_DEADBAND_BPS / 1e4
    masks = [tf.constant((np.abs(ret[:, i]) > d).astype(np.float32)) for i in range(3)]
    trues = [tf.constant((ret[:, i] > d).astype(np.float32)) for i in range(3)]
    probs = [tf.constant(np.asarray(preds[k]).reshape(-1)) for k in (1, 4, 7)]
    full = model._compute_direction_metrics(*trues, *probs, *masks, prefix="")
    for key in ("dir_acc_h1", "dir_mcc_h1", "dir_f1_h0", "dir_brier_h2", "dir_ece_h1", "pred_up_rate_h1"):
        np.testing.assert_allclose(logs[key], float(full[key]), rtol=1e-4, atol=1e-6, err_msg=key)

    exact_ks = float(model._pit_ks(tf.constant(y[:, 1]), tf.constant(np.asarray(preds[3]).reshape(-1)),
                                   tf.constant(np.asarray(preds[5]).reshape(-1))))
    assert abs(logs["pit_ks_h1"] - exact_ks) <= 1.0 / 200 + 1e-4  # binned at 1/200 resolution


def test_accumulators_reset_between_evaluations(model):
    ds = _ds(_data(2))
    first = model.evaluate(ds, verbose=0, return_dict=True)
    second = model.evaluate(ds, verbose=0, return_dict=True)
    for key in ("loss", "dir_mcc_h1", "pit_ks_h1", "nll_h1"):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-6, err_msg=key)


def test_fit_logs_are_epoch_aggregates_for_train_and_val(model):
    ds = _ds(_data(3))
    hist = model.fit(ds, validation_data=_ds(_data(4)), epochs=2, verbose=0).history
    for key in ("loss", "val_loss", "train_dir_mcc_h1", "val_dir_mcc_h1", "val_pit_ks_h1", "grad_global_norm"):
        assert key in hist and len(hist[key]) == 2 and np.all(np.isfinite(hist[key])), key
    assert "val_grad_global_norm" not in hist and "val_nonfinite_grad_steps" not in hist

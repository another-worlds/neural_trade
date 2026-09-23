"""One direction-labelling rule everywhere: NumPy and TF twins agree, including at the
deadband edges, and the Gaussian readouts agree too."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from neural_trade.metrics.direction_labels import direction_labels_np, gaussian_up_prob_given_move_np
from neural_trade.metrics.tf_direction import direction_labels_tf, gaussian_up_prob_given_move

LC = 110_000.0


def _deltas():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 150.0, (2000, 3))
    band = 5e-4 * LC  # exactly on / just inside / just outside the 5 bps band
    y[:6, 1] = [band, -band, band * (1 + 1e-4), -band * (1 + 1e-4), band * (1 - 1e-4), 0.0]
    return y


def test_numpy_and_tf_labels_agree_including_band_edges():
    y = _deltas()
    lc = np.full(len(y), LC)
    np_out = direction_labels_np(y, lc, 5.0)
    tf_out = direction_labels_tf(tf.constant(y, tf.float32), tf.constant(lc, tf.float32), 5.0)
    for i, h in enumerate(("h0", "h1", "h2")):
        labels, mask = np_out[h]
        np.testing.assert_array_equal(mask.astype(np.float32), tf_out[i].numpy())
        np.testing.assert_array_equal(labels.astype(np.float32), tf_out[3 + i].numpy())
    labels, mask = np_out["h1"]
    assert list(mask[:6]) == [False, False, True, True, False, False]  # |ret| must EXCEED the band
    assert list(labels[:6]) == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_numpy_and_tf_gaussian_readouts_agree():
    rng = np.random.default_rng(1)
    pred_scale, pred_mean = 261.0, 3.2
    mu_raw = rng.normal(0, 80, 500)
    var_s = rng.uniform(1e-4, 3.0, 500)
    lc = np.full(500, LC)
    p_np = gaussian_up_prob_given_move_np(mu_raw, var_s, lc, 5.0, pred_scale)
    mu_s = (mu_raw - pred_mean) / pred_scale
    p_tf = gaussian_up_prob_given_move(tf.constant(mu_s, tf.float32), tf.constant(var_s, tf.float32),
                                       tf.constant(lc, tf.float32), 5e-4, pred_mean, pred_scale).numpy()
    np.testing.assert_allclose(p_tf, p_np, atol=2e-4)

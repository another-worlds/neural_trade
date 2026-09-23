"""Metrics registry: numpy evaluation tier and TF step tier, contracts and selection."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import brier_score_loss, explained_variance_score, matthews_corrcoef, r2_score

from neural_trade.core.config import Config
from neural_trade.core.exceptions import ComponentNotFoundError
from neural_trade.metrics import tf_direction
from neural_trade.metrics.evaluate import registry_metrics
from neural_trade.registries.metrics import STEP_METRICS, Metrics


def test_both_tiers_are_registered_and_separate():
    assert len(Metrics.tf_names()) == 11
    assert set(Config().METRICS) <= set(Metrics.numpy_names())  # plugins may add more
    assert len(Config().METRICS) == 16
    assert set(Config().STEP_METRICS) == set(Metrics.tf_names()) == set(STEP_METRICS)
    with pytest.raises(ComponentNotFoundError):
        Metrics.tf_functions(["rmse"])
    with pytest.raises(ComponentNotFoundError):
        Metrics.numpy_functions(["dir_mcc"])


def test_numpy_metrics_agree_with_sklearn():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    yhat = 0.6 * y + rng.normal(0, 0.8, 500)
    assert np.isclose(Metrics.get("r2")(y, yhat), r2_score(y, yhat))
    assert np.isclose(Metrics.get("explained_variance")(y, yhat), explained_variance_score(y, yhat))
    labels = (y > 0).astype(float)
    probs = 1 / (1 + np.exp(-2 * yhat))
    assert np.isclose(Metrics.get("mcc")(labels, probs), matthews_corrcoef(labels, probs > 0.5))
    assert np.isclose(Metrics.get("brier")(labels, probs), brier_score_loss(labels, probs))
    mask = np.abs(y) > 0.3
    assert np.isclose(Metrics.get("mcc")(labels, probs, mask=mask),
                      matthews_corrcoef(labels[mask], probs[mask] > 0.5))


def test_ece_pos_is_zero_for_calibrated_probabilities_and_large_for_overconfident():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200_000)
    labels = (rng.uniform(0, 1, p.size) < p).astype(float)
    assert Metrics.get("ece_pos")(labels, p) < 0.01
    assert Metrics.get("ece_pos")(labels, np.where(p > 0.5, 0.99, 0.01)) > 0.2


def test_pit_ks_and_coverage():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.5, 2, 5000)
    y = rng.normal(0, sigma)
    assert Metrics.get("pit_ks")(y, np.zeros_like(y), variance=sigma ** 2) < 0.03
    assert Metrics.get("pit_ks")(y, np.zeros_like(y), variance=(sigma / 3) ** 2) > 0.2
    lo, hi = -1.645 * sigma, 1.645 * sigma
    assert abs(Metrics.get("coverage")(y, np.stack([lo, hi], 1)) - 0.90) < 0.02


def test_tf_tier_is_graph_safe():
    for name in Metrics.tf_names():
        src = inspect.getsource(Metrics.get(name))
        assert ".numpy(" not in src, f"{name} is not graph-safe"
    stats = tf_direction.direction_stats(tf.constant([30., 40., 10., 20., 5., 50., 100.]),
                                         tf.ones(10) * 10, tf.ones(10) * 5, tf.ones(10) * 5)

    @tf.function
    def run():
        return {n: f(stats) for n, f in Metrics.tf_functions().items()}

    out = {k: float(v) for k, v in run().items()}
    assert np.isclose(out["dir_acc"], 0.7) and np.isclose(out["pred_up_rate"], 0.4)
    assert np.isclose(out["dir_mcc"], (30 * 40 - 10 * 20) / np.sqrt(40 * 50 * 50 * 60))


def test_step_metrics_selection_limits_what_the_model_logs(make_loss_model):
    cfg = Config(STEP_METRICS=["dir_mcc", "dir_acc"])
    m = make_loss_model(261.0, 3.2, config=cfg)
    t = tf.constant([1., 0., 1., 0.])
    p = tf.constant([.7, .2, .4, .6])
    out = m._compute_direction_metrics(t, t, t, p, p, p, prefix="")
    assert set(out) == {f"{n}_{h}" for n in ("dir_mcc", "dir_acc") for h in ("h0", "h1", "h2")}


def test_registry_metrics_per_horizon():
    rng = np.random.default_rng(3)
    n = 400
    y = rng.normal(0, 200, (n, 3))
    preds = {"delta": {h: y[:, i] * 0.5 for i, h in enumerate(("h0", "h1", "h2"))},
             "direction_prob": {h: 1 / (1 + np.exp(-y[:, i] / 100)) for i, h in enumerate(("h0", "h1", "h2"))},
             "variance": {h: np.ones(n) for h in ("h0", "h1", "h2")}}
    out = registry_metrics(Config(), y, preds, np.full(n, 110_000.0), pred_scale=200.0)
    assert set(out) == {"h0", "h1", "h2"}
    assert out["h1"]["corr"] > 0.99 and out["h1"]["mcc"] > 0.9 and "pit_ks" in out["h1"]

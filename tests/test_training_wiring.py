"""The training model resolves its objective and optimizers through the registries; ablation."""
from __future__ import annotations

import pytest

from neural_trade.core.config import Config
from neural_trade.registries.losses import Losses
from neural_trade.training.lambdas import ablate


def test_objective_comes_from_the_losses_registry(make_loss_model):
    m = make_loss_model(261.0, 3.2)
    assert m.objective is Losses.get_objective("custom_loss")
    calls = []

    def spy(model, x_window, y_true, y_pred, last_close, extended_trends, vacuum_overflow=None):
        calls.append(1)
        return Losses.get_objective()(model, x_window, y_true, y_pred, last_close, extended_trends,
                                      vacuum_overflow=vacuum_overflow)

    m2 = make_loss_model(261.0, 3.2, objective=spy)
    import tensorflow as tf
    x = tf.zeros([2, 60]); y = tf.zeros([2, 3]); lc = tf.ones([2, 1]) * 1e5; ext = tf.zeros([2, 3])
    heads = tuple(tf.fill([2, 1], 0.5) for _ in range(9))
    m2.custom_loss(x, y, heads, lc, ext)
    assert calls == [1]


def test_indicator_optimizer_uses_config_name(make_loss_model):
    import tensorflow as tf
    m = make_loss_model(261.0, 3.2, config=Config(INDICATOR_OPTIMIZER_NAME="rmsprop"))
    assert isinstance(m.indicator_optimizer, tf.keras.optimizers.RMSprop)


def test_ablate_zeroes_variables_and_the_vac_threshold(make_loss_model):
    m = make_loss_model(261.0, 3.2, config=Config(LAMBDA_VAC=0.5))
    applied = ablate(m, ["LAMBDA_HD", "LAMBDA_CASIMIR", "LAMBDA_VAC"])
    vals = m.get_lambda_values()
    assert applied == ["LAMBDA_HD", "LAMBDA_CASIMIR", "LAMBDA_VAC"]
    assert vals["lambda_hd"] == 0.0 and vals["lambda_casimir"] == 0.0 and vals["lambda_t_perp"] > 0
    assert m.config.LAMBDA_VAC == 0.0
    with pytest.raises(ValueError):
        ablate(m, ["LAMBDA_QUANTILE"])

"""Optimizers registry and the optimizer pair of a run."""
from __future__ import annotations

import pytest
import tensorflow as tf

from neural_trade.core.config import Config
from neural_trade.core.exceptions import ComponentValidationError
from neural_trade.registries.optimizers import Optimizers
from neural_trade.training.optim import build_optimizers


def test_five_optimizers_registered_and_buildable():
    assert set(Optimizers.list_names()) == {"adam", "adamw", "sgd_momentum", "rmsprop", "nadam"}
    cfg = Config()
    for name in Optimizers.list_names():
        opt = Optimizers.build(name, cfg)
        assert abs(float(tf.keras.backend.get_value(opt.learning_rate)) - cfg.LR) < 1e-9, name
    with pytest.raises(ComponentValidationError):
        Optimizers.register(name="bad")(lambda cfg: None)


def test_adam_matches_the_keras_default_the_monolith_used():
    opt = Optimizers.build("adam", Config())
    ref = tf.keras.optimizers.Adam(learning_rate=1e-3)
    assert type(opt) is type(ref)
    for key in ("beta_1", "beta_2", "epsilon"):
        assert opt.get_config()[key] == ref.get_config()[key]


def test_pair_uses_the_indicator_multiplier_and_config_names():
    cfg = Config(OPTIMIZER_NAME="rmsprop", INDICATOR_OPTIMIZER_NAME="adam", INDICATOR_LR_MULT=4.0)
    pair = build_optimizers(cfg)
    assert isinstance(pair.main, tf.keras.optimizers.RMSprop)
    assert isinstance(pair.indicator, tf.keras.optimizers.Adam)
    assert abs(float(tf.keras.backend.get_value(pair.indicator.learning_rate)) - 4e-3) < 1e-9

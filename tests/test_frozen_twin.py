"""NT-033 (1): the frozen-period twin, ``Config.FREEZE_INDICATOR_PERIODS``.

Every period logit stays at its configured value (a non-trainable weight: no gradient, no optimizer
update, no clip) and the per-window meta_adjust shift is off (the ADAPTIVE_INDICATORS switch NT-046
defined), so after training every applied period in every window equals the configured one. The rest
of the network is the learned network's (same layers, same parameter count).
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.models.registry import Models

OLD = dict(INPUT_SERIES=["close"], INDICATOR_FAMILIES={})


def _layer(model):
    for lyr in model.layers:
        if lyr.name.startswith("learnable_indicators"):
            return lyr
    raise AssertionError("no indicator layer")


def test_default_is_off_and_adaptive_stays_on():
    cfg = Config(**OLD)
    assert cfg.FREEZE_INDICATOR_PERIODS is False
    model = Models.build(cfg.MODEL_NAME, cfg)
    layer = _layer(model)
    assert layer.frozen is False and layer.adaptive is True
    assert len(layer.trainable_weights) == 18


def test_frozen_twin_has_the_same_parameters_but_none_trainable_in_the_indicator_layer():
    free = Models.build(Config(**OLD).MODEL_NAME, Config(**OLD))
    cfg = Config(**OLD, FREEZE_INDICATOR_PERIODS=True)
    twin = Models.build(cfg.MODEL_NAME, cfg)
    assert twin.count_params() == free.count_params()          # same architecture, same parameter count
    layer = _layer(twin)
    assert layer.frozen and not layer.adaptive
    assert layer.trainable_weights == [] and len(layer.non_trainable_weights) == 18
    assert (len(twin.trainable_weights) + 18 == len(free.trainable_weights))
    assert twin.get_layer("meta_adjust").trainable      # the meta layer stays in the network (zero gradient)


def test_frozen_logits_ignore_the_meta_shift_and_equal_the_configured_periods():
    cfg = Config(**OLD, FREEZE_INDICATOR_PERIODS=True)
    model = Models.build(cfg.MODEL_NAME, cfg)
    x = np.random.default_rng(0).normal(size=(16,) + tuple(model.input_shape[1:])).astype(np.float32)
    from neural_trade.evaluation.applied_periods import applied_period_samples

    applied = applied_period_samples(model, x)
    configured = {"ma_period_0": 5, "ma_period_1": 10, "ma_period_2": 30, "rsi_period_0": 9, "rsi_period_1": 14,
                  "rsi_period_2": 21, "bb_period_0": 10, "bb_period_1": 20, "bb_period_2": 25}
    for name, want in configured.items():
        np.testing.assert_allclose(applied[name], want, rtol=2e-6, err_msg=name)
    for name, v in applied.items():                              # one value per parameter, in every window
        assert np.ptp(v) <= 1e-4 * max(1.0, abs(float(v[0]))), name


def test_one_epoch_cpu_run_leaves_every_applied_period_at_its_configured_value(tf, tiny_close_only_config, tmp_path,
                                                                              synthetic_bars, monkeypatch, run_eagerly):
    import neural_trade.training.trainer as trainer_mod
    from neural_trade.evaluation.applied_periods import applied_period_samples
    from neural_trade.experiments.run_context import RunContext

    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    cfg = tiny_close_only_config
    cfg.CSV_PATH = str(csv)
    cfg.MAX_SEQUENCE_COUNT, cfg.BATCH_SIZE = 400, 32
    cfg.FREEZE_INDICATOR_PERIODS = True
    cfg.CALLBACKS = ["early_stopping"]
    cfg.validate()
    ctx = RunContext.create(cfg, root=tmp_path / "runs")
    configured = {"ma_period_0": 5, "rsi_period_0": 14, "bb_period_0": 20}

    result = trainer_mod.train_and_evaluate(config=ctx.config, run_context=ctx, epochs=1, force=True,
                                            calibrate=False, fit_calibration=False)
    model = result.model
    layer = model._indicator_layer
    assert layer.trainable_weights == []
    x = np.random.default_rng(1).normal(size=(32,) + tuple(model.base_model.input_shape[1:])).astype(np.float32)
    applied = applied_period_samples(model.base_model, x)
    for name, want in configured.items():
        np.testing.assert_allclose(applied[name], want, rtol=2e-6, err_msg=name)
    # the logits themselves: untouched by the optimizer and the clip (the period logit of the configured value)
    for name, v in layer.get_learned_parameters().items():
        if name in configured:
            assert v == pytest.approx(configured[name], rel=2e-6), name

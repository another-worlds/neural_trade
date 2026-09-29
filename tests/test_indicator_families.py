"""NT-046: the registry-driven LearnableIndicators reproduces the pre-NT-046 numbers,
legacy configs give the same instances, a toy family plugs in without editing the model,
and the adaptive per-window shift has an off switch (frozen twin)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.indicators import (
    ChannelSpec,
    FamilyContext,
    IndicatorFamily,
    Indicators,
    ParamSpec,
    indicator_instances,
    num_learnable_logits,
)

DATA = Path(__file__).parent / "data"
# The pre-NT-047 default: close-only input, the four families alone. The NT-046 fixtures
# were recorded there; NT-047's OHLCV default is pinned by tests/test_ohlcv_input.py.
OLD = dict(INPUT_SERIES=["close"], INDICATOR_FAMILIES={})
DEFAULT_INSTANCES = {
    "ma": [5, 10, 30],
    "macd": [{"fast": 12, "slow": 26, "signal": 9}, {"fast": 5, "slow": 35, "signal": 5},
             {"fast": 8, "slow": 17, "signal": 9}],
    "rsi": [9, 14, 21],
    "bb": [10, 20, 25],
}


# --------------------------------------------------------------------- criterion 5: numbers
@pytest.mark.parametrize("impl", ["matrix", "scan"])
def test_layer_reproduces_the_pre_nt046_output_fixture(tf, impl):
    """tests/data/nt046_layer_fixture.npz was recorded at the base commit f5aee70 (before the
    registry rewrite); on the recording machine the new layer matches it bit-for-bit. The
    tolerance here only allows for a different BLAS in CI."""
    from neural_trade.models.layers import LearnableIndicators

    fx = np.load(DATA / "nt046_layer_fixture.npz", allow_pickle=True)
    cfg = Config(**OLD)
    cfg.EWMA_IMPL = impl
    tf.keras.utils.set_random_seed(0)
    layer = LearnableIndicators(cfg)
    out = layer([tf.constant(fx["x"]), tf.constant(fx["meta"])]).numpy()
    ref = fx[f"layer_out_{impl}"]
    assert out.shape == ref.shape == (8, 60, 31)
    scale = np.maximum(np.abs(ref).max(axis=(0, 1), keepdims=True), 1.0)
    np.testing.assert_allclose(out / scale, ref / scale, atol=2e-5, rtol=0)


def test_variable_names_and_learned_keys_are_unchanged(tf):
    """Saved bundles load by these weight names; telemetry reports these keys."""
    from neural_trade.models.layers import LearnableIndicators

    fx = np.load(DATA / "nt046_layer_fixture.npz", allow_pickle=True)
    layer = LearnableIndicators(Config(**OLD))
    layer([tf.constant(fx["x"]), tf.constant(fx["meta"])])
    got = [v.name.split("/", 1)[1] for v in layer.get_indicator_trainable_variables()]
    want = [n.split("/", 1)[1] for n in fx["var_names"]]  # the layer prefix is a session uid
    assert got == want
    learned = layer.get_learned_parameters()
    assert sorted(learned) == list(fx["learned_keys"])
    np.testing.assert_allclose([learned[k] for k in sorted(learned)], fx["learned_vals"],
                               rtol=1e-6)


def test_model_loads_a_pre_nt046_weights_file_and_predicts_identically(tf):
    """tests/data/nt046_model.weights.h5 and its predictions were saved at the base commit:
    the rebuilt default model must load them (same weight structure) and reproduce the
    predictions (bit-for-bit on the recording machine; tolerance for CI's BLAS)."""
    from neural_trade.registries import load_all
    from neural_trade.registries.models import Models

    cfg = Config(**OLD)
    load_all(cfg)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)
    model = Models.build(cfg.MODEL_NAME, cfg)
    model.load_weights(str(DATA / "nt046_model.weights.h5"))
    fx = np.load(DATA / "nt046_model_fixture.npz")
    preds = model.predict(fx["x"], verbose=0)
    for i, p in enumerate(preds):
        np.testing.assert_allclose(p, fx[f"out_{i}"], rtol=1e-4, atol=1e-5, err_msg=f"out_{i}")


# --------------------------------------------------------------------- criterion 2: config
def test_default_config_lists_three_instances_per_family_with_todays_periods():
    # the four NT-046 families keep their instances inside the 14-family NT-047 default
    inst = indicator_instances(Config())
    assert {k: inst[k] for k in DEFAULT_INSTANCES} == DEFAULT_INSTANCES
    assert indicator_instances(Config(**OLD)) == DEFAULT_INSTANCES


def test_legacy_fields_still_configure_the_instances(tmp_path):
    cfg = Config(MA_SPANS=[3, 7], RSI_PERIODS=[5], BB_PERIODS=[12, 40],
                 MACD_SETTINGS=[{"fast": 6, "slow": 13, "signal": 4}], INDICATOR_FAMILIES={})
    got = indicator_instances(cfg)
    assert got == {"ma": [3, 7], "macd": [{"fast": 6, "slow": 13, "signal": 4}],
                   "rsi": [5], "bb": [12, 40]}
    # a config FILE that sets the legacy fields loads and gives the same instances
    yml = tmp_path / "legacy.yaml"
    yml.write_text("MA_SPANS: [3, 7]\nRSI_PERIODS: [5]\nBB_PERIODS: [12, 40]\n"
                   "MACD_SETTINGS:\n  - {fast: 6, slow: 13, signal: 4}\nINDICATOR_FAMILIES: {}\n",
                   encoding="utf-8")
    assert indicator_instances(Config.from_yaml(yml)) == got


def test_indicator_families_field_adds_and_overrides():
    cfg = Config(INDICATOR_FAMILIES={"ma": [4]})
    assert indicator_instances(cfg)["ma"] == [4]  # overrides MA_SPANS
    assert indicator_instances(cfg)["rsi"] == [9, 14, 21]


def test_unregistered_family_is_reported_by_config_validation():
    from neural_trade.registries import validate_config_components

    assert validate_config_components(Config(INDICATOR_FAMILIES={"nope": [5]})) == ["Indicators:nope"]


# --------------------------------------------------------------------- criterion 3: extension
class _ToyFamily(IndicatorFamily):
    """One EWMA of the close under a made-up name (registered only inside the test)."""

    name = "toy"
    inputs = ("close",)
    params = (ParamSpec("period", default=4.0, minimum=2.0),)
    channels = (ChannelSpec("toy_ema"),)
    draw = "panel"

    def stage1(self, ctx: FamilyContext, alphas, cache):
        return {"ema": (ctx.close, alphas["period"])}

    def outputs(self, ctx, alphas, s1, s2, cache):
        return [s1["ema"]]


@pytest.fixture
def toy_family():
    Indicators.register(name="toy", description="toy test family")(_ToyFamily())
    yield
    Indicators.remove("toy")


def test_a_family_is_added_by_one_registry_entry_and_one_config_line(tf, toy_family):
    """No edit to the model or the training pipeline: the layer and the architecture pick the
    new family up from the registry entry and the config line alone."""
    from neural_trade.models.gru_attention import build_gru_attention
    from neural_trade.models.layers import LearnableIndicators
    from neural_trade.registries import validate_config_components

    cfg = Config(INDICATOR_FAMILIES={"toy": [4, 8]})
    assert validate_config_components(cfg) == []
    assert num_learnable_logits(cfg) == 20
    layer = LearnableIndicators(cfg)
    rng = np.random.default_rng(0)
    x = tf.constant(np.cumsum(rng.normal(0, 1, (4, 60)), axis=1).astype(np.float32))
    out = layer([x, tf.zeros([4, 20])])
    assert out.shape == (4, 60, 33)  # 31 + one toy channel per instance
    assert len(layer.get_indicator_trainable_variables()) == 20
    assert {"toy_period_0", "toy_period_1"} <= set(layer.get_learned_parameters())
    model = build_gru_attention(cfg)  # the whole architecture builds unchanged
    assert len(model.outputs) == 10


# --------------------------------------------------------------------- criterion 4: switch
def test_adaptive_indicators_off_applies_the_global_period_in_every_window(tf):
    """With ADAPTIVE_INDICATORS = False the meta adjustment is ignored: the output is the
    same for any meta input and equals the zero-shift (global learned period) output, while
    the meta path keeps a defined, exactly-zero gradient."""
    from neural_trade.models.layers import LearnableIndicators

    rng = np.random.default_rng(1)
    x = tf.constant(np.cumsum(rng.normal(0, 1, (4, 60)), axis=1).astype(np.float32))
    metas = [tf.constant(rng.uniform(-1, 1, (4, 18)).astype(np.float32)) for _ in range(2)]

    frozen = LearnableIndicators(Config(ADAPTIVE_INDICATORS=False, **OLD))
    adaptive = LearnableIndicators(Config(**OLD))
    out_zero = adaptive([x, tf.zeros([4, 18])]).numpy()  # the global periods, no shift
    outs = []
    for meta in metas:
        meta_v = tf.Variable(meta)
        with tf.GradientTape() as tape:
            out = frozen([x, meta_v])
            loss = tf.reduce_mean(tf.square(out))
        outs.append(out.numpy())
        g_meta = tape.gradient(loss, meta_v)
        assert g_meta is not None and np.all(g_meta.numpy() == 0.0)
        assert adaptive([x, meta]).numpy().shape == out.numpy().shape
        assert not np.array_equal(adaptive([x, meta]).numpy(), out.numpy())  # the shift acts
    np.testing.assert_array_equal(outs[0], outs[1])
    np.testing.assert_array_equal(outs[0], out_zero)

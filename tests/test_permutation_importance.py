"""Grouped permutation importance (NT-048): signal ranks first, a noise group stays inside its band,
and the training package never reads the module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_trade.evaluation.permutation_importance import (
    grouped_importance,
    importance_from_model,
    indicator_channel_groups,
)
from neural_trade.visualization import permutation_importance as PI
from neural_trade.visualization import theme as T


def _scores(y, pred):
    err = (pred - y) ** 2
    hit = (np.sign(pred) == np.sign(y)).astype(float)
    return {"loss": float(err.mean()), "auc": {"h0": float(hit.mean())}, "loss_i": err, "hit_i": {"h0": hit}}


def _synthetic():
    rng = np.random.default_rng(0)
    n, lookback, channels = 400, 8, 6
    y = rng.normal(size=n)
    features = rng.normal(size=(n, lookback, channels))
    features[:, :, 0:2] = y[:, None, None]

    def score_fn(feats):
        return _scores(y, feats[:, :, 0].mean(axis=1))

    groups = [("signal", slice(0, 2)), ("noise", slice(2, 4)), ("other", slice(4, 6))]
    rows = grouped_importance(features, score_fn(features), score_fn, groups)
    return features, rows


def test_signal_family_ranks_first_and_noise_stays_inside_its_band():
    """The channels that equal the label move the loss; an unused family does not leave its band."""
    features, rows = _synthetic()
    by_name = {row.name: row for row in rows}
    assert by_name["signal"].loss > by_name["noise"].loss
    assert by_name["signal"].loss > by_name["other"].loss
    assert by_name["signal"].loss > 0
    noise = by_name["noise"]
    assert noise.loss_lo <= noise.loss <= noise.loss_hi
    untouched = features.copy()
    grouped_importance(features, _scores(np.zeros(len(features)), features[:, :, 0].mean(1)),
                       lambda feats: _scores(np.zeros(len(features)), feats[:, :, 0].mean(1)),
                       [("signal", slice(0, 2))])
    np.testing.assert_array_equal(features, untouched)


def test_channel_groups_follow_instances_and_leave_out_the_raw_close():
    from neural_trade.indicators import Indicators

    class Cfg:
        MA_SPANS = [10, 20]
        MACD_SETTINGS = [{"fast": 12, "slow": 26, "signal": 9}]
        RSI_PERIODS = []
        BB_PERIODS = []
        INDICATOR_FAMILIES = {}

    groups = indicator_channel_groups(Cfg())
    assert [name for name, _sl in groups] == ["ma #0", "ma #1", "macd #0"]
    ma_w = len(Indicators.get("ma").channels)
    macd_w = len(Indicators.get("macd").channels)
    assert groups[0][1] == slice(0, ma_w)
    assert groups[1][1] == slice(ma_w, 2 * ma_w)
    assert groups[2][1] == slice(2 * ma_w, 2 * ma_w + macd_w)
    assert all(not name.startswith("close") for name, _sl in groups)


def test_a_channel_selector_has_to_be_one_contiguous_slice():
    features = np.zeros((4, 3, 2))
    base = {"loss": 0.0, "auc": {}, "loss_i": np.zeros(4), "hit_i": {}}

    def score_fn(_feats):
        return base

    with pytest.raises(ValueError, match="slice"):
        grouped_importance(features, base, score_fn, [("bad", [0])])
    with pytest.raises(ValueError, match="no windows"):
        grouped_importance(np.zeros((0, 3, 2)), base, score_fn, [])
    with pytest.raises(ValueError, match="loss_i"):
        grouped_importance(features, {**base, "loss_i": np.zeros(3)}, score_fn, [("ok", slice(0, 1))])
    with pytest.raises(ValueError, match="no groups"):
        PI.permutation_importance([], None)


def test_the_importance_figure_is_one_panel_with_a_band_and_no_horizon_or_copy_colour():
    _features, rows = _synthetic()
    fig = PI.permutation_importance({"groups": rows}, None)
    assert T.empty_panels(fig) == []
    assert [name for name in fig.layout if str(name).startswith("yaxis")] == ["yaxis"]
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert list(bar.y) == ["signal", "noise", "other"]
    assert "h0" in "".join(bar.hovertext)
    lo = np.asarray(bar.x, float) - np.asarray(bar.error_x.arrayminus, float)
    hi = np.asarray(bar.x, float) + np.asarray(bar.error_x.array, float)
    noise_at = list(bar.y).index("noise")
    assert lo[noise_at] <= bar.x[noise_at] <= hi[noise_at]
    colours = set()
    for trace in fig.data:
        for colour in (getattr(getattr(trace, "marker", None), "color", None),
                       getattr(getattr(trace, "line", None), "color", None),
                       getattr(getattr(trace, "error_x", None), "color", None)):
            colours.update(colour if isinstance(colour, (list, tuple)) else [colour])
        if getattr(getattr(trace, "line", None), "dash", None) == "dot":
            raise AssertionError(trace.name)
    banned = set(T.HORIZON_COLORS.values()) | set(T.OTHER_SERIES[:3])
    assert not colours & banned
    from neural_trade.registries.visualizations import Visualizations

    assert Visualizations.get("permutation_importance") is PI.permutation_importance
    assert Visualizations.validate_component(PI.permutation_importance)
    assert T.empty_panels(Visualizations.build("permutation_importance", rows, None)) == []


def test_importance_from_model_scores_the_tail_after_the_indicator_layer():
    import tensorflow as tf

    rng = np.random.default_rng(1)
    windows = rng.normal(size=(40, 4, 3)).astype(np.float32)
    inp = tf.keras.Input(shape=(4, 3), name="window")
    ind = tf.keras.layers.Dense(3, name="learnable_indicators")(inp)
    out = tf.keras.layers.Dense(1, name="head")(tf.keras.layers.GlobalAveragePooling1D()(ind))
    model = tf.keras.Model(inp, out)

    def score_fn(pred):
        pred = np.asarray(pred, dtype=float).reshape(-1)
        err = pred ** 2
        hit = (pred > 0).astype(float)
        return {"loss": float(err.mean()), "auc": {"h0": float(hit.mean())}, "loss_i": err, "hit_i": {"h0": hit}}

    rows = importance_from_model(model, windows, score_fn, groups=[("mix", slice(0, 3))],
                                 n_boot=40, block=10, batch=16)
    assert [row.name for row in rows] == ["mix"]
    assert np.isfinite(rows[0].loss)
    assert rows[0].loss_lo <= rows[0].loss_hi


def test_importance_from_model_accepts_several_outputs_and_the_layer_config():
    import tensorflow as tf

    class Cfg:
        MA_SPANS = [10]
        MACD_SETTINGS = []
        RSI_PERIODS = []
        BB_PERIODS = []
        INDICATOR_FAMILIES = {}

    inp = tf.keras.Input(shape=(4, 2), name="window")
    ind = tf.keras.layers.Dense(2, name="learnable_indicators")(inp)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(ind)
    model = tf.keras.Model(inp, [tf.keras.layers.Dense(1, name="h0")(pooled),
                                 tf.keras.layers.Dense(1, name="h1")(pooled)])
    model.get_layer("learnable_indicators").config = Cfg()
    windows = np.random.default_rng(2).normal(size=(12, 4, 2)).astype(np.float32)

    def score_fn(pred):
        assert isinstance(pred, list) and len(pred) == 2
        assert pred[0].shape[0] == len(windows) and pred[1].shape == pred[0].shape
        value = np.asarray(pred[0], dtype=float).reshape(-1)
        err = value ** 2
        hit = (value > 0).astype(float)
        return {"loss": float(err.mean()), "auc": {"h0": float(hit.mean())}, "loss_i": err, "hit_i": {"h0": hit}}

    rows = importance_from_model(model, windows, score_fn, n_boot=20, block=4, batch=5)
    assert [row.name for row in rows] == ["ma #0"]
    assert np.isfinite(rows[0].loss)


def test_a_model_whose_graph_cannot_be_cut_is_refused():
    import tensorflow as tf

    class Box(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(2, name="learnable_indicators")

        def call(self, inputs):
            return self.dense(inputs)

    model = Box()
    model(tf.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="learnable_indicators"):
        importance_from_model(model, np.zeros((2, 3), np.float32), lambda pred: pred,
                              groups=[("a", slice(0, 2))], n_boot=4, block=2)


def test_training_does_not_mention_permutation_importance():
    root = Path(__file__).resolve().parents[1] / "src" / "neural_trade" / "training"
    offenders = [str(path.relative_to(root)) for path in sorted(root.rglob("*.py"))
                 if "permutation_importance" in path.read_text(encoding="utf-8")]
    assert offenders == []

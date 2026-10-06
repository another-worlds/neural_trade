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
    return {"loss": float(err.mean()), "loss_i": err, "hit_i": {"h0": hit},
            "labels": {"h0": (y > 0).astype(float)}, "scores": {"h0": np.asarray(pred, float)}}


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
    base = {"loss": 0.0, "loss_i": np.zeros(4), "hit_i": {}, "labels": {}, "scores": {}}

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


def test_auc_is_a_real_auc_drop_and_differs_from_the_hit_rate_drop():
    """The signal group's AUC falls from about 1 to about 0.5 when shuffled; the noise group's drop stays inside
    its block-bootstrap band; the band is of the AUC itself (not of the hit)."""
    from sklearn.metrics import roc_auc_score

    features, rows = _synthetic()
    by_name = {row.name: row for row in rows}
    sig, noise = by_name["signal"], by_name["noise"]
    y = features[:, :, 0].mean(axis=1)
    base = roc_auc_score((y > 0).astype(int), y)
    assert base > 0.99
    assert 0.35 < sig.auc["h0"] < 0.65                      # about base - 0.5
    assert sig.auc_lo["h0"] > 0.2 and sig.auc_lo["h0"] <= sig.auc["h0"] <= sig.auc_hi["h0"]
    assert noise.auc_lo["h0"] <= noise.auc["h0"] <= noise.auc_hi["h0"]
    assert noise.auc_lo["h0"] <= 0.0 <= noise.auc_hi["h0"]
    assert sig.auc["h0"] != pytest.approx(sig.hit_drop["h0"], abs=1e-9)
    assert sig.hit_lo["h0"] <= sig.hit_drop["h0"] <= sig.hit_hi["h0"]


def test_the_weighted_auc_of_a_resample_equals_the_auc_of_the_repeated_rows():
    from sklearn.metrics import roc_auc_score

    from neural_trade.evaluation.permutation_importance import _weighted_auc

    rng = np.random.default_rng(3)
    n = 60
    labels = (rng.random(n) > 0.4).astype(float)
    labels[::7] = np.nan                                     # unlabelled windows
    scores = np.round(rng.normal(size=n) + labels.clip(0, 1), 1)   # rounded: ties
    counts = rng.integers(0, 4, size=(5, n)).astype(float)
    got = _weighted_auc(labels, scores, counts)
    for b in range(5):
        idx = np.repeat(np.arange(n), counts[b].astype(int))
        idx = idx[np.isfinite(labels[idx])]
        assert got[b] == pytest.approx(roc_auc_score(labels[idx], scores[idx]), abs=1e-12)
    assert np.isnan(_weighted_auc(np.ones(4), np.arange(4.0), np.ones((1, 4)))).all()   # one class only


def test_the_bootstrap_block_has_to_cover_the_horizon():
    features = np.random.default_rng(0).normal(size=(30, 3, 2))
    score = _scores(np.ones(30), features[:, :, 0].mean(1))
    with pytest.raises(ValueError, match="cover the horizon"):
        grouped_importance(features, score, lambda f: score, [("a", slice(0, 1))], block=5, horizon_bars=20)
    rows = grouped_importance(features, score, lambda f: score, [("a", slice(0, 1))], block=20,
                              horizon_bars=20, n_boot=10)
    assert [r.name for r in rows] == ["a"]


def test_the_importance_figure_has_a_loss_panel_and_one_auc_panel_per_horizon():
    _features, rows = _synthetic()
    fig = PI.permutation_importance({"groups": rows}, None)
    assert T.empty_panels(fig) == []
    assert sorted([name for name in fig.layout if str(name).startswith("yaxis")]) == ["yaxis", "yaxis2"]
    assert len(fig.data) == 3                                   # loss, h0 AUC drop, h0 hit-rate drop
    bar, auc_bar, _hit = fig.data
    assert list(bar.y) == list(auc_bar.y) == ["signal", "noise", "other"]
    hover = "".join(bar.hovertext)
    assert "h0" in hover and "AUC drop" in hover and "hit-rate drop" in hover
    for trace in (bar, auc_bar):
        lo = np.asarray(trace.x, float) - np.asarray(trace.error_x.arrayminus, float)
        hi = np.asarray(trace.x, float) + np.asarray(trace.error_x.array, float)
        noise_at = list(trace.y).index("noise")
        assert lo[noise_at] <= trace.x[noise_at] <= hi[noise_at]
    assert bar.marker.color == T.NEUTRAL                       # an indicator never takes a horizon colour
    assert auc_bar.marker.color == T.HORIZON_COLORS["h0"]      # the AUC panel is a horizon's, in its colour
    for trace in fig.data[:2]:
        assert getattr(getattr(trace, "line", None), "dash", None) != "dot"
        assert getattr(trace.marker, "color", None) not in set(T.OTHER_SERIES[:3])
    from neural_trade.registries.visualizations import Visualizations

    assert Visualizations.get("permutation_importance") is PI.permutation_importance
    assert Visualizations.validate_component(PI.permutation_importance)
    assert T.empty_panels(Visualizations.build("permutation_importance", rows, None)) == []


def test_every_default_family_instance_is_a_group():
    from neural_trade.core.config import Config
    from neural_trade.indicators import Indicators, indicator_instances

    cfg = Config()
    fams = list(indicator_instances(cfg))
    assert len(fams) == 14 and all(len(indicator_instances(cfg)[f]) == 3 for f in fams)
    groups = indicator_channel_groups(cfg)
    assert {name for name, _sl in groups} == {f"{fam} #{i}" for fam in fams for i in range(3)}
    assert len(groups) == 42
    assert sum(sl.stop - sl.start for _n, sl in groups) == sum(len(Indicators.get(f).channels) * 3 for f in fams)


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
        return {"loss": float(err.mean()), "loss_i": err, "hit_i": {"h0": hit},
                "labels": {"h0": (np.arange(len(err)) % 2).astype(float)}, "scores": {"h0": pred}}

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
        return {"loss": float(err.mean()), "loss_i": err, "hit_i": {"h0": hit},
                "labels": {"h0": (np.arange(len(err)) % 2).astype(float)}, "scores": {"h0": value}}

    rows = importance_from_model(model, windows, score_fn, n_boot=20, block=4, batch=5)
    assert [row.name for row in rows] == ["ma #0"]
    assert np.isfinite(rows[0].loss)


def test_a_skip_around_the_indicator_layer_stays_on_the_original_window():
    """The production graph reads the raw window after the indicator layer (energy gate, direction
    skip). The tail keeps that window in its own order and still matches the full model."""
    import tensorflow as tf

    from neural_trade.evaluation.permutation_importance import _probe_and_tail

    inp = tf.keras.Input(shape=(6, 3), name="input_window")
    ind = tf.keras.layers.Dense(2, name="learnable_indicators")(inp)
    skip = tf.keras.layers.GlobalAveragePooling1D()(inp)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(ind)
    out = tf.keras.layers.Dense(1)(tf.keras.layers.Concatenate()([pooled, skip]))
    model = tf.keras.Model(inp, out)
    layer = model.get_layer("learnable_indicators")
    with pytest.raises(ValueError):
        tf.keras.Model(layer.output, model.output)

    x = np.random.default_rng(0).normal(size=(5, 6, 3)).astype(np.float32)
    probe, tail, takes_windows = _probe_and_tail(model, layer)
    assert takes_windows is True
    feat = probe(tf.constant(x), training=False)
    np.testing.assert_allclose(tail([tf.constant(x), feat], training=False).numpy(),
                               model(tf.constant(x), training=False).numpy(), atol=1e-6)

    def score_fn(pred):
        value = np.asarray(pred, dtype=float).reshape(-1)
        err = value ** 2
        hit = (value > 0).astype(float)
        return {"loss": float(err.mean()), "loss_i": err, "hit_i": {"h0": hit},
                "labels": {"h0": (np.arange(len(err)) % 2).astype(float)}, "scores": {"h0": value}}

    rows = importance_from_model(model, x, score_fn, groups=[("mix", slice(0, 2))], n_boot=8, block=2, batch=4)
    assert [row.name for row in rows] == ["mix"] and np.isfinite(rows[0].loss)


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


def test_the_panel_titles_give_each_horizons_baseline_auc_and_the_hit_drop_is_drawn():
    _features, rows = _synthetic()
    assert rows[0].auc_base["h0"] > 0.99 and all(r.auc_base == rows[0].auc_base for r in rows)
    fig = PI.permutation_importance(rows, None)
    titles = [a.text for a in fig.layout.annotations]
    assert any("baseline AUC %.3f" % rows[0].auc_base["h0"] in t for t in titles), titles
    diamonds = [t for t in fig.data if t.type == "scatter"]
    assert len(diamonds) == 1 and list(diamonds[0].x) == [r.hit_drop["h0"] for r in rows]
    assert T.empty_panels(fig) == []


def test_hit_drop_is_over_the_labelled_windows_only():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.normal(size=n)
    features = np.stack([y, y], axis=1)[:, None, :].repeat(3, axis=1)
    labelled = np.arange(n) % 2 == 0

    def score_fn(feats):
        pred = feats[:, 0, 0]
        hit = (np.sign(pred) == np.sign(y)).astype(float)
        hit[~labelled] = 0.0                                   # unlabelled windows carry a 0, as the report does
        lab = np.where(labelled, (y > 0).astype(float), np.nan)
        return {"loss": 0.0, "loss_i": (pred - y) ** 2, "hit_i": {"h0": hit}, "labels": {"h0": lab},
                "scores": {"h0": pred}}

    row = grouped_importance(features, score_fn(features), score_fn, [("g", slice(0, 2))], n_boot=20, block=5)[0]
    hits_before = score_fn(features)["hit_i"]["h0"][labelled]
    assert hits_before.mean() == 1.0
    assert 0.3 < row.hit_drop["h0"] < 0.7                        # about 0.5 on the labelled windows, not halved

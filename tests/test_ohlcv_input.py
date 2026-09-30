"""NT-047: OHLCV model input - sequence building, scaling, model, serving and gradients.

Criterion 1 (shapes and alignment on the bundled data), criterion 3 (finite gradients for
every learnable parameter of every family on bundled and extreme inputs), criterion 4
(all 14 families on by default with 3 instances each; more or fewer is a config change).
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.indicators import Indicators, indicator_instances, num_learnable_logits

SMALL = dict(MAX_SEQUENCE_COUNT=1500, EXTENDED_TREND_PERIODS=[10, 15, 20])


def _num_channels(cfg) -> int:
    return sum(len(Indicators.get(n).channels) * len(v)
               for n, v in indicator_instances(cfg).items()) + 1  # + raw close


# --------------------------------------------------------------------- criterion 4: defaults
def test_all_fourteen_families_are_on_by_default_with_three_instances_each():
    inst = indicator_instances(Config())
    assert list(inst) == ["ma", "macd", "rsi", "bb", "atr", "stoch", "willr", "keltner",
                          "obv", "vwap", "mfi", "adx", "cci", "donchian"]
    assert all(len(v) == 3 for v in inst.values())
    assert num_learnable_logits(Config()) == 54


def test_more_or_fewer_instances_is_a_config_change():
    cfg = Config(INDICATOR_FAMILIES={"atr": [5, 9], "donchian": [20]})
    inst = indicator_instances(cfg)
    assert inst["atr"] == [5, 9] and inst["donchian"] == [20]
    assert set(inst) == {"ma", "macd", "rsi", "bb", "atr", "donchian"}
    off = Config(INDICATOR_FAMILIES={})
    assert set(indicator_instances(off)) == {"ma", "macd", "rsi", "bb"}


def test_input_series_is_validated():
    assert Config(INPUT_SERIES=["close"]).input_series() == ("close",)
    assert Config().close_channel() == 3
    with pytest.raises(InvalidConfigurationError):
        Config(INPUT_SERIES=["open", "high", "low"])  # no close
    with pytest.raises(InvalidConfigurationError):
        Config(INPUT_SERIES=["close", "open"])  # not in canonical order
    with pytest.raises(InvalidConfigurationError):
        Config(INPUT_SERIES=["close", "close"])  # repeats


def test_a_volume_family_with_a_close_only_input_is_refused(tf):
    from neural_trade.models.layers import LearnableIndicators

    cfg = Config(INPUT_SERIES=["close"])  # default families include volume readers
    layer = LearnableIndicators(cfg)
    with pytest.raises(ValueError, match="INPUT_SERIES"):
        layer([tf.zeros([2, cfg.LOOKBACK]), tf.zeros([2, num_learnable_logits(cfg)])])


# --------------------------------------------------------------------- criterion 1: windows
@pytest.mark.data
def test_bundled_sequences_carry_ohlcv_and_end_at_the_decision_bar(real_slice):
    """[N, LOOKBACK, 5] on the bundled CSV; window k's channels are exactly the frame's
    open/high/low/close/volume rows ending at the anchor's decision bar, and the close
    channel equals the close-only windows bit-for-bit."""
    from neural_trade.data.processor import DataProcessor
    from neural_trade.data.windowing import (frame_series, make_multichannel_windows,
                                             make_sequences_with_extended_trends,
                                             sequence_anchor_bars)

    cfg = Config(**SMALL)
    dp = DataProcessor(cfg)
    df = dp.preprocess(real_slice.copy())
    close = df["Close"].to_numpy(dtype="float32")
    X = make_multichannel_windows(cfg, frame_series(cfg, df), cfg.LOOKBACK)
    Xc, y, lc, ext = make_sequences_with_extended_trends(cfg, close, cfg.LOOKBACK)
    assert X.shape == (Xc.shape[0], cfg.LOOKBACK, 5)
    np.testing.assert_array_equal(X[..., 3], Xc)  # close channel == close windows
    anchors = sequence_anchor_bars(Config(**dict(SMALL, MAX_SEQUENCE_COUNT=0)),
                                   len(close))
    cols = {0: "Open", 1: "High", 2: "Low", 3: "Close", 4: "Volume"}
    for k in (0, 7, len(X) - 1):
        bar = anchors[k]
        for j, col in cols.items():
            np.testing.assert_array_equal(
                X[k, :, j], df[col].to_numpy(dtype="float32")[bar - cfg.LOOKBACK + 1: bar + 1])


@pytest.mark.data
def test_processor_blocks_scale_ohlcv_and_keep_close_raw_windows(real_slice):
    """prepare_datasets: scaled model windows are [N, L, 5] (OHLC window-relative, volume
    by the train-mean scale, fit on TRAIN); the raw-window consumers (cal X_raw,
    test_windows_raw) still receive close windows [N, L]."""
    from neural_trade.data.processor import DataProcessor

    cfg = Config(**SMALL)
    dp = DataProcessor(cfg)
    df = dp.preprocess(real_slice.copy())
    out = dp.prepare_datasets(df, df["Close"].to_numpy(dtype="float32"))
    X_train = out[0]
    assert X_train.ndim == 3 and X_train.shape[2] == 5
    assert dp.normalizer.input_series == ("open", "high", "low", "close", "volume")
    assert dp.normalizer.vol_scale > 0
    # the close channel is window-relative: last bar == 0 (window ends at the decision bar)
    np.testing.assert_allclose(X_train[:, -1, 3], 0.0, atol=1e-6)
    # volume channel: raw volume / train-mean volume (non-negative)
    assert np.all(X_train[..., 4] >= 0)
    assert np.isclose(np.mean(X_train[..., 4]), 1.0, atol=0.35)
    # raw-window consumers get CLOSE windows
    assert dp.cal_block["X_raw"].ndim == 2
    assert dp.test_windows_raw.ndim == 2
    assert dp.cal_block["X"].shape[1:] == (cfg.LOOKBACK, 5)
    # round trip: normalizer dict round-trips the volume scale (serving bundles)
    from neural_trade.data.scaling import WindowNormalizer

    n2 = WindowNormalizer.from_dict(dp.normalizer.to_dict())
    assert n2.input_series == dp.normalizer.input_series
    assert n2.vol_scale == dp.normalizer.vol_scale


def test_model_input_shape_follows_input_series(tf):
    from neural_trade.models.gru_attention import build_gru_attention

    m = build_gru_attention(Config())
    assert tuple(m.inputs[0].shape) == (None, 60, 5)
    m_old = build_gru_attention(Config(INPUT_SERIES=["close"], INDICATOR_FAMILIES={}))
    assert tuple(m_old.inputs[0].shape) == (None, 60)


@pytest.mark.data
def test_predictor_round_trips_ohlcv_windows(real_slice, tmp_path, tf):
    """Serving reads OHLCV: an untrained bundle at the OHLCV default predicts from raw
    OHLCV windows and from a raw frame; the windows end at the frame's last bar."""
    from neural_trade.data.scaling import WindowNormalizer, fit_target_scaler
    from neural_trade.registries import load_all
    from neural_trade.registries.models import Models
    from neural_trade.serving.predictor import Predictor
    from neural_trade.training.artifacts import ArtifactBundle

    cfg = Config(**SMALL)
    load_all(cfg)
    model = Models.build(cfg.MODEL_NAME, cfg)
    y = np.random.default_rng(0).normal(0, 5.0, (64, 3))
    scaler = fit_target_scaler(y)
    series = ("open", "high", "low", "close", "volume")
    norm = WindowNormalizer("window_relative", float(scaler.scale_[0]), None, series, 3.0)
    bundle = ArtifactBundle(cfg, float(scaler.scale_[0]), float(scaler.mean_[0]), norm)
    bundle._model = model
    bundle.save(tmp_path / "artifacts")
    p = Predictor.from_artifacts(tmp_path / "artifacts")

    batch, df, anchors = p.predict_windows_frame(real_slice.iloc[:400].copy())
    assert anchors[-1] == len(df) - 1  # the last window ends at the newest bar
    assert np.isclose(batch.last_close[-1], df["Close"].iloc[-1])
    for h in ("h0", "h1", "h2"):
        assert np.all(np.isfinite(batch.delta[h]))
    # a close-only window batch is refused with a clear message
    with pytest.raises(ValueError, match="INPUT_SERIES"):
        p.predict(np.zeros((4, cfg.LOOKBACK), dtype="float32"))
    with pytest.raises(ValueError, match="INPUT_SERIES"):
        p.predict_last(df["Close"].to_numpy())


# --------------------------------------------------------------------- criterion 3: gradients
def _grad_check(tf, x, cfg=None):
    from neural_trade.models.layers import LearnableIndicators

    cfg = cfg or Config()
    layer = LearnableIndicators(cfg)
    meta = tf.zeros([x.shape[0], num_learnable_logits(cfg)])
    xv = tf.constant(np.asarray(x, dtype="float32"))
    with tf.GradientTape() as tape:
        out = layer([xv, meta])
        loss = tf.reduce_mean(tf.square(out))
    assert np.all(np.isfinite(out.numpy())), "non-finite channel values"
    grads = tape.gradient(loss, layer.get_indicator_trainable_variables())
    names = [v.name for v in layer.get_indicator_trainable_variables()]
    assert len(grads) == num_learnable_logits(cfg)
    bad = [n for n, g in zip(names, grads) if g is None or not np.all(np.isfinite(g.numpy()))]
    assert not bad, f"non-finite/None gradients: {bad}"


def _stack(rng, close, spread=0.2, vol=None):
    n = close.shape
    high = close + np.abs(rng.normal(0, spread, n))
    low = close - np.abs(rng.normal(0, spread, n))
    opn = np.roll(close, 1, axis=1)
    opn[:, 0] = close[:, 0]
    v = vol if vol is not None else np.abs(rng.lognormal(0, 0.6, n))
    return np.stack([opn, high, low, close, v], axis=-1)


@pytest.mark.data
def test_gradients_finite_for_every_family_on_bundled_windows(real_slice, tf):
    """Every learnable parameter of every default family has a finite gradient on real
    normalised OHLCV windows from the bundled CSV."""
    from neural_trade.data.processor import DataProcessor

    cfg = Config(**SMALL)
    dp = DataProcessor(cfg)
    df = dp.preprocess(real_slice.copy())
    out = dp.prepare_datasets(df, df["Close"].to_numpy(dtype="float32"))
    _grad_check(tf, out[0][:16], cfg)


def test_gradients_finite_on_extreme_inputs(tf):
    """Constant windows, zero volume, and price jumps (both raw-scale and huge)."""
    rng = np.random.default_rng(5)
    L = Config().LOOKBACK
    close = np.cumsum(rng.normal(0, 0.3, (4, L)), axis=1)

    _grad_check(tf, np.zeros((4, L, 5)))                       # all-zero constant window
    _grad_check(tf, np.full((4, L, 5), 7.0))                   # constant non-zero window
    _grad_check(tf, _stack(rng, close, vol=np.zeros((4, L))))  # zero volume throughout

    jump = close.copy()
    jump[:, L // 2:] += 50.0                                    # a 50-unit jump mid-window
    _grad_check(tf, _stack(rng, jump))

    huge = close * 1000.0                                       # far outside the exact band
    _grad_check(tf, _stack(rng, huge, spread=20.0))

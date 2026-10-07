"""NT-041 S1: the dataset spec, wall-clock window / horizons, and the data hygiene checks (a-e)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.core.dataset_spec import DatasetSpec, bar_label, minutes_to_bars
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.data.loaders import median_bar_minutes, validate_ohlcv_frame
from neural_trade.data.preprocessors import sort_dedupe, standardize_ohlcv


# ---------------------------------------------------------------- criterion 1 / 2: spec and wall-clock lengths
def test_reference_defaults_give_todays_bars():
    cfg = Config()
    spec = DatasetSpec.from_config(cfg)
    assert (spec.symbol, spec.quote_currency, spec.bar_minutes) == ("BTC/USDT", "USDT", 1.0)
    assert spec.window_bars == 60 and spec.horizon_bars == [10, 15, 20]
    assert spec.window_minutes == 60 and spec.horizon_minutes == [10, 15, 20]
    assert spec.data_file == cfg.CSV_PATH
    assert spec.title_tag == "BTC/USDT 1-minute"


def test_wall_clock_fields_convert_to_bars_by_the_bar_size():
    cfg = Config(RESAMPLE_MINUTES=5, WINDOW_MINUTES=120, HORIZON_MINUTES=[10, 15, 20],
                 EXTENDED_TREND_MINUTES=[10, 15, 20])
    assert cfg.LOOKBACK == 24 and cfg.HORIZON_STEPS == [2, 3, 4] and cfg.EXTENDED_TREND_PERIODS == [2, 3, 4]
    spec = DatasetSpec.from_config(cfg)
    assert spec.window_minutes == 120 and spec.horizon_minutes == [10, 15, 20]
    # the reference setup written in minutes is the reference setup in bars
    ref = Config(WINDOW_MINUTES=60, HORIZON_MINUTES=[10, 15, 20], EXTENDED_TREND_MINUTES=[10, 15, 20])
    assert (ref.LOOKBACK, ref.HORIZON_STEPS, ref.EXTENDED_TREND_PERIODS) == (60, [10, 15, 20], [10, 15, 20])


@pytest.mark.parametrize("bad", [dict(WINDOW_MINUTES=61), dict(HORIZON_MINUTES=[10, 15, 22]),
                                 dict(EXTENDED_TREND_MINUTES=[10, 16, 20]), dict(WINDOW_MINUTES=2.5)])
def test_a_length_that_does_not_divide_the_bar_size_is_refused(bad):
    with pytest.raises(InvalidConfigurationError, match="whole number"):
        Config(RESAMPLE_MINUTES=5, **bad)


def test_override_and_copy_re_resolve_and_refuse():
    cfg = Config(RESAMPLE_MINUTES=5, WINDOW_MINUTES=120)
    assert cfg.LOOKBACK == 24
    assert cfg.copy(RESAMPLE_MINUTES=10).LOOKBACK == 12      # follows the bar size on every path
    with pytest.raises(InvalidConfigurationError, match="WINDOW_MINUTES"):
        cfg.copy(RESAMPLE_MINUTES=7)                          # 120 / 7 is not whole
    again = Config.from_dict(cfg.to_dict())
    assert again.LOOKBACK == 24 and again.WINDOW_MINUTES == 120


def test_yaml_round_trip_keeps_the_wall_clock_fields(tmp_path):
    cfg = Config(RESAMPLE_MINUTES=5, WINDOW_MINUTES=120, HORIZON_MINUTES=[10, 15, 20])
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded.LOOKBACK == 24 and loaded.HORIZON_STEPS == [2, 3, 4] and loaded.HORIZON_MINUTES == [10, 15, 20]


def test_minutes_to_bars_and_labels():
    assert minutes_to_bars(15, 5) == 3 and minutes_to_bars(0.5, 0.5) == 1
    for minutes, bar in ((7, 5), (0, 1), (3, 0), (-5, 5)):
        with pytest.raises(InvalidConfigurationError):
            minutes_to_bars(minutes, bar)
    assert [bar_label(m) for m in (1, 5, 60, 240, 1440, 90)] == ["1-minute", "5-minute", "1-hour", "4-hour",
                                                                  "1-day", "90-minute"]


# ---------------------------------------------------------------- (a) the declared bar size against the data
def _frame(n=20, step_minutes=1.0, start="2025-01-01"):
    ts = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * step_minutes, unit="min")
    return pd.DataFrame({"timestamp": ts, "Close": np.linspace(100, 110, n)})


def test_declared_bar_size_must_match_the_measured_spacing():
    assert median_bar_minutes(_frame(step_minutes=5)["timestamp"]) == 5.0
    validate_ohlcv_frame(_frame(step_minutes=1), bar_minutes=1)
    validate_ohlcv_frame(_frame(step_minutes=5), bar_minutes=5)
    with pytest.raises(ValueError, match="median bar spacing is 5"):
        validate_ohlcv_frame(_frame(step_minutes=5), bar_minutes=1)
    with pytest.raises(ValueError, match="declared bar size is 5"):
        validate_ohlcv_frame(_frame(step_minutes=1), bar_minutes=5)


def test_a_mismatched_file_is_refused_by_the_data_processor(tmp_path):
    from neural_trade.data.processor import DataProcessor

    df = _frame(200, step_minutes=5)
    df.rename(columns={"Close": "close"}).assign(open=lambda d: d["close"], high=lambda d: d["close"],
                                                 low=lambda d: d["close"], volume=1.0).to_csv(
        tmp_path / "five.csv", index=False)
    with pytest.raises(ValueError, match="bar spacing"):
        DataProcessor(Config(CSV_PATH=str(tmp_path / "five.csv"), LOOKBACK=20)).load_and_prepare_data()


# ---------------------------------------------------------------- (c) a non-positive close
@pytest.mark.parametrize("value", [0.0, -1.5])
def test_a_zero_or_negative_close_is_refused(value):
    df = _frame()
    df.loc[7, "Close"] = value
    with pytest.raises(ValueError, match="zero or negative close"):
        validate_ohlcv_frame(df)


# ---------------------------------------------------------------- (d) a stable sort
def test_sort_dedupe_is_stable_for_rows_with_one_timestamp():
    ts = pd.Timestamp("2025-01-01")
    rows = [(ts + pd.Timedelta(minutes=m), i) for i, m in enumerate([2, 1, 1, 0, 1, 1, 1, 2, 0])]
    df = pd.DataFrame(rows, columns=["timestamp", "Close"])
    out = sort_dedupe(df, Config())
    # the LAST row of each timestamp in file order survives, whatever the sort algorithm does with ties
    assert out["Close"].tolist() == [8, 6, 7]


# ---------------------------------------------------------------- (e) epoch timestamps
@pytest.mark.parametrize("unit,scale", [("s", 1), ("ms", 1_000), ("us", 1_000_000), ("ns", 1_000_000_000)])
def test_epoch_timestamps_parse_in_their_own_unit(unit, scale):
    base = 1_735_689_600  # 2025-01-01T00:00:00Z
    raw = pd.DataFrame({"timestamp": [(base + 60 * k) * scale for k in range(5)], "close": [1.0] * 5})
    out = standardize_ohlcv(raw, Config())
    assert out["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 00:00:00")
    assert out["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_text_timestamps_parse_as_before():
    raw = pd.DataFrame({"datetime": ["2025-01-01 00:00:00+00:00", "2025-01-01 00:01:00+00:00"], "close": [1.0, 2.0]})
    out = standardize_ohlcv(raw, Config())
    assert str(out["timestamp"].dt.tz) == "UTC" and len(out) == 2


# ---------------------------------------------------------------- (f) the bootstrap block scales with the horizon
def test_bootstrap_block_scales_with_the_longest_horizon_and_keeps_the_reference_value():
    from neural_trade.metrics.statistics import BLOCK, bootstrap_block

    assert bootstrap_block([10, 15, 20]) == BLOCK == 80          # the reference setup: today's block
    assert bootstrap_block([5, 60, 240]) == 960                  # 80 bars was 44% too narrow at h = 240
    assert bootstrap_block(7) == 80


# ---------------------------------------------------------------- (g) the first anchor when a lag >= the window
def test_first_anchor_leaves_room_for_the_longest_lag_and_keeps_the_reference():
    from neural_trade.data.windowing import first_anchor

    assert first_anchor(60, [10, 15, 20]) == 60                  # reference: unchanged
    assert first_anchor(20, [20, 25, 30]) == 31                  # was 30: close[i - 1 - 30] fell before the data


def test_no_past_delta_is_zero_filled_when_a_lag_is_not_shorter_than_the_window():
    from neural_trade.data.windowing import make_sequences_with_extended_trends

    cfg = Config(LOOKBACK=20, EXTENDED_TREND_PERIODS=[20, 25, 30], HORIZON_STEPS=[2, 3, 4], N_FOLDS=2)
    close = (100 + np.cumsum(np.random.default_rng(0).normal(size=300))).astype("float32")
    X, _y, lc, ext = make_sequences_with_extended_trends(cfg, close, cfg.LOOKBACK)
    first = 31
    for k, p in enumerate(cfg.EXTENDED_TREND_PERIODS):
        assert ext[0, k] == pytest.approx(close[first - 1] - close[first - 1 - p])
    assert np.all(ext != 0.0)

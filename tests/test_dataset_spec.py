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


# ================================================================ S2: fingerprint in every run, cost profile
def _bars_csv(path, n=400, minutes=1):
    ts = pd.Timestamp("2025-03-01") + pd.to_timedelta(np.arange(n) * minutes, unit="min")
    close = 100 + 10 * np.sin(np.arange(n) / 50.0) + np.random.default_rng(1).normal(size=n)
    pd.DataFrame({"timestamp": ts, "open": close, "high": close + 1, "low": close - 1, "close": close,
                  "volume": 1.0}).to_csv(path, index=False)
    return path


def test_every_run_directory_records_the_dataset_fingerprint_and_the_setup(tmp_path):
    import hashlib
    import json

    from neural_trade.experiments.run_context import RunContext

    csv = _bars_csv(tmp_path / "bars.csv")
    ctx = RunContext.create(Config(CSV_PATH=str(csv), LOOKBACK=20), root=tmp_path / "runs")
    meta = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))
    ds = meta["dataset"]
    assert ds["sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert ds["n_bars"] == 400 and ds["first_timestamp"].startswith("2025-03-01T00:00") \
        and ds["last_timestamp"].startswith("2025-03-01T06:39")
    assert meta["setup"]["symbol"] == "BTC/USDT" and meta["setup"]["window_minutes"] == 20
    assert meta["setup"]["horizon_minutes"] == [10, 15, 20] and meta["setup"]["quote_currency"] == "USDT"


def test_a_run_whose_data_cannot_be_fingerprinted_says_so_instead_of_failing(tmp_path):
    import json

    from neural_trade.experiments.run_context import RunContext

    ctx = RunContext.create(Config(CSV_PATH=str(tmp_path / "missing.csv")), root=tmp_path / "runs")
    meta = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert "error" in meta["dataset"] and "sha256" not in meta["dataset"]


def test_the_fingerprint_follows_data_end(tmp_path):
    from neural_trade.experiments.dataset import dataset_fingerprint

    csv = _bars_csv(tmp_path / "long.csv", n=200_000 // 100 * 100, minutes=1)
    full = dataset_fingerprint(Config(CSV_PATH=str(csv), DATA_END_PROTECTED_DAYS=0.0))
    cut = dataset_fingerprint(Config(CSV_PATH=str(csv), DATA_END="2025-03-02", DATA_END_PROTECTED_DAYS=0.0))
    assert cut["n_bars"] < full["n_bars"] and cut["sha256"] == full["sha256"]
    assert cut["last_timestamp"].startswith("2025-03-02T00:00")


def test_the_cost_profile_defaults_equal_the_engines_zero_costs_and_reaches_the_backtest():
    from neural_trade.core.costs import COST_FIELDS, cost_profile_of
    from neural_trade.strategy import BacktestConfig, build_backtest_config

    cfg = Config()
    base = BacktestConfig()
    assert cost_profile_of(cfg) == {k: getattr(base, k) for k in COST_FIELDS} == {k: 0.0 for k in COST_FIELDS}
    assert DatasetSpec.from_config(cfg).cost_profile == cost_profile_of(cfg)

    paid = Config(FEE_BPS=10.0, HALF_SPREAD_BPS=1.0, SLIPPAGE_BPS=2.0)
    bcfg = build_backtest_config({"bar_minutes": 1.0}, cost_profile=cost_profile_of(paid))
    assert (bcfg.fee_bps, bcfg.half_spread_bps, bcfg.slippage_bps) == (10.0, 1.0, 2.0)
    # a backtest: entry of a scenario still overrides the instrument's profile
    over = build_backtest_config({"fee_bps": 0.0}, cost_profile=cost_profile_of(paid))
    assert (over.fee_bps, over.half_spread_bps) == (0.0, 1.0)
    assert build_backtest_config({"fee_bps": 3.0}).fee_bps == 3.0          # no profile: as before


def test_the_scorer_backtests_with_the_setups_costs():
    from neural_trade.core.costs import cost_profile_of
    from neural_trade.experiments import scorer

    seen = {}

    def fake_build(params, cost_profile=None):
        seen.update(params=dict(params), cost=cost_profile)
        raise RuntimeError("stop")

    import neural_trade.strategy as strategy_pkg

    real = strategy_pkg.build_backtest_config
    strategy_pkg.build_backtest_config = fake_build
    try:
        cfg = Config(FEE_BPS=7.0)
        with pytest.raises(RuntimeError, match="stop"):
            scorer.fit_and_backtest(type("S", (), {"cal": None})(), None, bar_minutes=1.0, strategy="buy_and_hold",
                                    cost_profile=cost_profile_of(cfg))
    finally:
        strategy_pkg.build_backtest_config = real
    assert seen["cost"]["fee_bps"] == 7.0


def test_a_leaderboard_row_carries_the_fingerprint_and_the_setup():
    import json

    from neural_trade.experiments.leaderboard import TABLE_HEADER, build_leaderboard, leaderboard_markdown, table_cells

    row = {"scenario": "s", "configuration": "c", "fold": -1, "role": "test", "seed": 0, "status": "done", "error": None,
           "dataset_sha256": "ab" * 32, "dataset_first": "2025-10-11T02:30:00+00:00",
           "dataset_last": "2025-11-10T07:29:00+00:00", "dataset_n_bars": 43500, "bar_minutes": 1.0,
           "symbol": "BTC/USDT", "window_minutes": 60.0, "horizon_minutes": json.dumps([10.0, 15.0, 20.0]),
           "horizon_steps": json.dumps([10, 15, 20]), "strategy": "calibrated_quantile", "sharpe_net": 1.0,
           "total_return": 0.01, "max_drawdown": 0.01, "n_trades": 10.0, "fee_bps": 0.0, "half_spread_bps": 0.0,
           "slippage_bps": 0.0}
    [r] = build_leaderboard([row])
    cells = dict(zip(TABLE_HEADER, table_cells(r)))
    assert cells["dataset fingerprint (sha256, first 12)"] == "ab" * 6
    assert cells["dataset bars (count, first .. last)"] == "43,500 bars, 2025-10-11T02:30 .. 2025-11-10T07:29"
    assert (cells["instrument"], cells["window (min)"], cells["horizons (min)"]) == ("BTC/USDT", "60", "10, 15, 20")
    assert "43,500 bars" in leaderboard_markdown([r])


def test_an_index_made_before_the_setup_columns_is_migrated(tmp_path):
    import sqlite3

    from neural_trade.experiments.store import RUN_COLUMNS, RunIndex

    old = [c for c in RUN_COLUMNS if c[0] not in ("symbol", "window_minutes", "horizon_minutes")]
    path = tmp_path / "index.sqlite"
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE runs ({', '.join(f'{c} {t}' for c, t in old)})")
    con.commit()
    con.close()
    RunIndex(path).ensure_schema()
    con = sqlite3.connect(path)
    have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    con.close()
    assert {"symbol", "window_minutes", "horizon_minutes"} <= have
    assert RunIndex(path).rows() == []

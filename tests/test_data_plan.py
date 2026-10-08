"""NT-041 S3b / S4: block lengths in time, walk-forward folds over a long file, holes in the bars, and the
dataset check of a notebook's rebuilt blocks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.data.gaps import flat_runs, gap_flags
from neural_trade.data.processor import split_arrays
from neural_trade.experiments.dataset import DatasetMismatch, data_layout, verify_run_dataset

DAY = 1440


def _file(path, days=100, holes=(), flat=(), minutes=1):
    """A synthetic OHLCV CSV: ``days`` of ``minutes``-minute bars from 2025-01-01, ``holes`` = (start bar, length)
    removed, ``flat`` = (start bar, length) forward-filled (flat, zero volume)."""
    n = days * DAY // minutes
    ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) * minutes, unit="min")
    close = 100 + 10 * np.sin(np.arange(n) / 700.0) + np.random.default_rng(0).normal(0, 0.05, n)
    vol = np.ones(n)
    for a, k in flat:
        close[a:a + k] = close[a]
        vol[a:a + k] = 0.0
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + (vol > 0), "low": close - (vol > 0),
                       "close": close, "volume": vol})
    open_ = df["open"].to_numpy().copy()
    df["open"] = np.where(vol > 0, open_, close)
    keep = np.ones(n, bool)
    for a, k in holes:
        keep[a:a + k] = False
    df[keep].to_csv(path, index=False)
    return path


def _timed(path, **kw):
    base = dict(CSV_PATH=str(path), FOLD_LAYOUT="timed", N_FOLDS=3, FOLD_SPACING_DAYS=30.0, MAX_SEQUENCE_COUNT=0)
    base.update(kw)
    return Config(**base)


@pytest.fixture(scope="module")
def long_file(tmp_path_factory):
    return _file(tmp_path_factory.mktemp("long") / "long.csv", days=100)


# ---------------------------------------------------------------- criterion 6: block lengths in time
def test_the_training_block_is_seven_days_by_default_and_the_other_blocks_have_their_own_lengths(long_file):
    cfg = _timed(long_file)
    assert (cfg.TRAIN_MINUTES, cfg.VAL_MINUTES, cfg.CAL_MINUTES, cfg.TEST_MINUTES) == (7 * DAY, 2 * DAY, 2 * DAY, 5 * DAY)
    gap = cfg.LOOKBACK + max(cfg.HORIZON_STEPS)
    for fold in data_layout(cfg).folds:
        b = fold["blocks"]
        assert [b[k]["n"] for k in ("train", "val", "cal", "test")] == [7 * DAY, 2 * DAY, 2 * DAY, 5 * DAY]
        # the purge gap stays (D-005): the next block starts gap bars after the last sequence before it
        for a, nxt in (("train", "val"), ("val", "cal"), ("cal", "test")):
            last = pd.Timestamp(b[a]["last_timestamp"])
            first = pd.Timestamp(b[nxt]["first_timestamp"])
            assert (first - last) == pd.Timedelta(minutes=gap), (a, nxt)
        assert fold["gap"] == gap


def test_block_lengths_follow_the_configuration_and_the_bar_size(tmp_path):
    path = _file(tmp_path / "h.csv", days=100, minutes=60)
    cfg = _timed(path, RESAMPLE_MINUTES=60, LOOKBACK=24, HORIZON_STEPS=[1, 2, 4], EXTENDED_TREND_PERIODS=[1, 2, 4],
                 TRAIN_MINUTES=20 * DAY, VAL_MINUTES=3 * DAY, CAL_MINUTES=3 * DAY, TEST_MINUTES=6 * DAY, N_FOLDS=2)
    for fold in data_layout(cfg).folds:
        assert [fold["blocks"][k]["n"] for k in ("train", "val", "cal", "test")] == [20 * 24, 3 * 24, 3 * 24, 6 * 24]


def test_a_block_length_that_is_not_a_whole_number_of_bars_is_refused(tmp_path):
    with pytest.raises(InvalidConfigurationError, match="TRAIN_MINUTES"):
        Config(FOLD_LAYOUT="timed", RESAMPLE_MINUTES=60, TRAIN_MINUTES=100)


# ---------------------------------------------------------------- criterion 7: folds over the long history
def test_folds_fall_in_different_months_and_no_blocks_overlap(long_file):
    layout = data_layout(_timed(long_file))
    assert layout.n_folds == 3
    starts = [pd.Timestamp(f["blocks"]["train"]["first_timestamp"]) for f in layout.folds]
    assert len({(s.year, s.month) for s in starts}) == 3 and starts == sorted(starts)
    prev_end = None
    for f in layout.folds:
        spans = [(pd.Timestamp(f["blocks"][k]["first_timestamp"]), pd.Timestamp(f["blocks"][k]["last_timestamp"]))
                 for k in ("train", "val", "cal", "test")]
        for (_, a_end), (b_start, _) in zip(spans, spans[1:]):
            assert a_end < b_start                                     # blocks of a fold never overlap
        lo, hi = spans[0][0], spans[-1][1]
        if prev_end is not None:
            assert lo > prev_end                                       # nor do the folds (spacing > fold length)
        prev_end = hi
        assert f["read_range"] == {"first_timestamp": lo.isoformat(), "last_timestamp": hi.isoformat()} or \
            (f["read_range"]["first_timestamp"][:16], f["read_range"]["last_timestamp"][:16]) == \
            (lo.isoformat()[:16], hi.isoformat()[:16])


def test_the_newest_fold_ends_at_the_last_bar_and_fold_index_picks_by_position(long_file):
    cfg = _timed(long_file)
    layout = data_layout(cfg)
    assert layout.fold(-1)["role"] == "test" and layout.fold(-2)["role"] == "dev"
    last = layout.fingerprint["last_timestamp"]
    newest_end = pd.Timestamp(layout.fold(-1)["blocks"]["test"]["last_timestamp"])
    assert pd.Timestamp(last) - newest_end == pd.Timedelta(minutes=max(cfg.HORIZON_STEPS))
    a = split_arrays(cfg.copy(FOLD_INDEX=-1))
    b = split_arrays(cfg.copy(FOLD_INDEX=0))
    assert a["test"]["anchor_bar"][0] > b["test"]["anchor_bar"][-1]


def test_folds_at_configured_dates_record_their_dates(long_file):
    starts = ["2025-01-05 00:00", "2025-02-20 12:00", "2025-03-10"]
    cfg = _timed(long_file, FOLD_STARTS=starts, N_FOLDS=3)
    layout = data_layout(cfg)
    got = [f["blocks"]["train"]["first_timestamp"][:16] for f in layout.folds]
    assert got[0] == "2025-01-05T00:00" and got[1] == "2025-02-20T12:00" and got[2] == "2025-03-10T00:00"
    assert [f["planned_start"][:10] for f in layout.folds] == ["2025-01-05", "2025-02-20", "2025-03-10"]


def test_a_fold_that_does_not_fit_the_data_is_refused_by_name(long_file):
    with pytest.raises(InvalidConfigurationError, match="no complete"):
        data_layout(_timed(long_file, FOLD_STARTS=["2025-04-05"]))
    with pytest.raises(InvalidConfigurationError, match="no timed fold fits"):
        data_layout(_timed(long_file, TRAIN_MINUTES=99 * DAY))


def test_fold_starts_are_validated():
    with pytest.raises(InvalidConfigurationError, match="ascending"):
        Config(FOLD_LAYOUT="timed", FOLD_STARTS=["2025-02-01", "2025-01-01"])
    with pytest.raises(InvalidConfigurationError, match="unreadable"):
        Config(FOLD_LAYOUT="timed", FOLD_STARTS=["not a date"])


def test_split_arrays_reads_only_the_chosen_folds_span_and_matches_the_layout(long_file):
    cfg = _timed(long_file, FOLD_INDEX=1)
    arrays = split_arrays(cfg)
    layout = data_layout(cfg)
    fold = layout.fold(1)
    df = arrays["df"]
    for name in ("train", "val", "cal", "test"):
        blk = arrays[name]
        assert len(blk["X"]) == fold["blocks"][name]["n"]
        assert str(df["timestamp"].iloc[int(blk["anchor_bar"][0])].isoformat())[:16] == \
            fold["blocks"][name]["first_timestamp"][:16]
    # nothing outside the fold's span was windowed
    total = sum(len(arrays[k]["X"]) for k in ("train", "val", "cal", "test"))
    assert total < 0.2 * len(df)


def test_the_default_layout_is_unchanged_and_ignores_the_timed_fields(long_file):
    base = Config(CSV_PATH=str(long_file), MAX_SEQUENCE_COUNT=20_000)
    other = Config(CSV_PATH=str(long_file), MAX_SEQUENCE_COUNT=20_000, TRAIN_MINUTES=500, FOLD_SPACING_DAYS=3.0)
    a, b = data_layout(base), data_layout(other)
    assert a.folds == b.folds and a.n_sequences == 20_000


# ---------------------------------------------------------------- criterion 8: holes in the bars
HOLES = ((30 * DAY + 5, 180), (80 * DAY + 100, 700))


@pytest.fixture(scope="module")
def holey_file(tmp_path_factory):
    return _file(tmp_path_factory.mktemp("holey") / "holey.csv", days=100, holes=HOLES)


def test_holes_are_detected(holey_file):
    df = pd.read_csv(holey_file, parse_dates=["timestamp"])
    flags = gap_flags(df, 1.0)
    assert flags.sum() == 2
    assert not gap_flags(pd.read_csv(holey_file, parse_dates=["timestamp"]).iloc[:1000], 1.0).any()


def test_no_window_or_target_spans_a_hole_and_the_dropped_count_is_recorded(holey_file):
    cfg = Config(CSV_PATH=str(holey_file), MAX_SEQUENCE_COUNT=0, N_FOLDS=2, VAL_FRACTION=0.05, CAL_FRACTION=0.05)
    arrays = split_arrays(cfg)
    ts = arrays["df"]["timestamp"].to_numpy("datetime64[ns]")
    back = max(cfg.LOOKBACK, max(cfg.EXTENDED_TREND_PERIODS) + 1)
    fwd = max(cfg.HORIZON_STEPS) - 1
    one = np.timedelta64(1, "m")
    n = 0
    for name in ("train", "val", "cal", "test"):
        for bar in arrays[name]["anchor_bar"]:                # anchor_bar = the last input bar
            i = int(bar) + 1                                   # the anchor: the first bar after the window
            span = ts[i - back:i + fwd + 1]
            assert (np.diff(span) == one).all()
            n += 1
    assert n > 0
    gaps = arrays["gaps"]
    # a hole of k bars removes every anchor whose span reaches across it: back + fwd positions each
    assert gaps["n_gaps"] == 2 and gaps["n_missing_bars"] == 880 and gaps["longest_gap_bars"] == 700
    assert gaps["n_windows_dropped"] == 2 * (back + fwd)
    assert gaps["policy"] == "drop"


def test_the_dropped_windows_are_recorded_in_the_fingerprint_of_every_run(holey_file, tmp_path):
    from neural_trade.experiments.run_context import RunContext

    cfg = Config(CSV_PATH=str(holey_file))
    ctx = RunContext.create(cfg, root=tmp_path / "runs")
    meta = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))
    g = meta["dataset"]["gaps"]
    expected = 2 * (max(cfg.LOOKBACK, max(cfg.EXTENDED_TREND_PERIODS) + 1) + max(cfg.HORIZON_STEPS) - 1)
    assert g["n_gaps"] == 2 and g["n_windows_dropped"] == expected
    assert data_layout(cfg.copy(MAX_SEQUENCE_COUNT=0)).fingerprint["gaps"] == g


def test_gap_policy_refuse_and_ignore(holey_file):
    with pytest.raises(ValueError, match="GAP_POLICY is 'refuse'"):
        split_arrays(Config(CSV_PATH=str(holey_file), GAP_POLICY="refuse", MAX_SEQUENCE_COUNT=0))
    ignore = split_arrays(Config(CSV_PATH=str(holey_file), GAP_POLICY="ignore", MAX_SEQUENCE_COUNT=0))
    drop = split_arrays(Config(CSV_PATH=str(holey_file), GAP_POLICY="drop", MAX_SEQUENCE_COUNT=0))
    n_ignore = sum(len(ignore[k]["X"]) for k in ("train", "val", "cal", "test"))
    n_drop = sum(len(drop[k]["X"]) for k in ("train", "val", "cal", "test"))
    assert n_ignore > n_drop and ignore["gaps"]["n_windows_dropped"] == 0


def test_a_file_without_holes_is_unchanged_by_the_policy(long_file):
    a = split_arrays(Config(CSV_PATH=str(long_file), GAP_POLICY="drop", MAX_SEQUENCE_COUNT=3000))
    b = split_arrays(Config(CSV_PATH=str(long_file), GAP_POLICY="ignore", MAX_SEQUENCE_COUNT=3000))
    assert a["gaps"]["n_windows_dropped"] == 0 and a["gaps"]["n_gaps"] == 0
    for k in ("train", "val", "cal", "test"):
        np.testing.assert_array_equal(a[k]["X"], b[k]["X"])
        np.testing.assert_array_equal(a[k]["anchor_bar"], b[k]["anchor_bar"])


def test_timed_folds_skip_the_sequences_a_hole_removes(tmp_path):
    path = _file(tmp_path / "h.csv", days=100, holes=((92 * DAY + 100, 700),))      # inside the newest fold's cal block
    layout = data_layout(_timed(path, N_FOLDS=3))
    vc = [f["blocks"]["val"]["n"] + f["blocks"]["cal"]["n"] for f in layout.folds]
    assert vc[:2] == [4 * DAY, 4 * DAY] and vc[2] < 4 * DAY      # only the fold over the hole is shorter


def test_flat_forward_filled_runs_are_counted_not_dropped(tmp_path):
    path = _file(tmp_path / "flat.csv", days=10, flat=((2000, 300), (6000, 30)))
    df = pd.read_csv(path)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    assert flat_runs(df) == {"n_runs": 1, "n_bars": 300, "longest_bars": 300}      # the 30-bar run is under an hour
    arrays = split_arrays(Config(CSV_PATH=str(path), MAX_SEQUENCE_COUNT=0, N_FOLDS=2))
    assert arrays["gaps"]["flat_runs"]["n_runs"] == 1 and arrays["gaps"]["n_windows_dropped"] == 0


# ---------------------------------------------------------------- (b) notebooks rebuild blocks from a CSV path
def test_rebuilding_a_run_from_another_dataset_is_refused(tmp_path):
    from neural_trade.experiments.run_context import RunContext

    mine = _file(tmp_path / "a.csv", days=5)
    other = _file(tmp_path / "b.csv", days=6)
    ctx = RunContext.create(Config(CSV_PATH=str(mine), LOOKBACK=20), root=tmp_path / "runs")
    assert verify_run_dataset(ctx.run_dir, Config(CSV_PATH=str(mine), LOOKBACK=20))["sha256"]
    with pytest.raises(DatasetMismatch, match="not the dataset run"):
        verify_run_dataset(ctx.run_dir, Config(CSV_PATH=str(other), LOOKBACK=20))


def test_a_run_without_a_recorded_fingerprint_is_not_checked(tmp_path, capsys):
    run = tmp_path / "old"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"run_id": "old"}), encoding="utf-8")
    assert verify_run_dataset(run, Config()) == {}
    out = capsys.readouterr()
    assert "no dataset fingerprint" in out.out + out.err


# ---------------------------------------------------------------- the engine's meta
def test_engine_meta_records_the_read_range_and_the_gap_record(holey_file):
    layout = data_layout(Config(CSV_PATH=str(holey_file), MAX_SEQUENCE_COUNT=0, N_FOLDS=2, VAL_FRACTION=0.05,
                                CAL_FRACTION=0.05))
    assert layout.fingerprint["gaps"]["n_gaps"] == 2
    for f in layout.folds:
        assert f["read_range"]["first_timestamp"] == f["blocks"]["train"]["first_timestamp"]
        assert f["read_range"]["last_timestamp"] == f["blocks"]["test"]["last_timestamp"]

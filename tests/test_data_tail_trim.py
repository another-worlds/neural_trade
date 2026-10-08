"""DATA_TAIL_TRIM (tactical): the trimmed path yields bit-identical windows, blocks and prepared arrays."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BUNDLED = REPO / "binance_btcusdt_1min_ccxt.csv"
TACTICAL_CSV = Path("D:/nt/neural_trade/Bitcoin_BTCUSDT_tactical.csv")
BLOCK_KEYS = ("X", "X_model", "y", "last_close", "extended_trends", "index")


def _cfg(csv, trim, **kw):
    from neural_trade.core.config import Config

    base = dict(CSV_PATH=str(csv), MAX_SEQUENCE_COUNT=5000, N_FOLDS=3, VAL_FRACTION=0.1, CAL_FRACTION=0.1,
                FOLD_INDEX=-2, DATA_TAIL_TRIM=trim)
    base.update(kw)
    return Config(**base)


def _assert_same(a, b):
    for f in ("train", "val", "cal", "test"):
        assert np.array_equal(getattr(a["fold"], f), getattr(b["fold"], f)), f
    assert (a["fold"].fold, a["fold"].gap) == (b["fold"].fold, b["fold"].gap)
    ts_a, ts_b = a["df"]["timestamp"].to_numpy(), b["df"]["timestamp"].to_numpy()
    for name in ("train", "val", "cal", "test"):
        for k in BLOCK_KEYS:
            assert np.array_equal(a[name][k], b[name][k]), (name, k)
        assert np.array_equal(ts_a[a[name]["anchor_bar"]], ts_b[b[name]["anchor_bar"]]), name


def test_default_is_off():
    from neural_trade.core.config import Config

    assert Config().DATA_TAIL_TRIM is False


@pytest.mark.parametrize("kw", [dict(), dict(WINDOW_STEP=3), dict(INPUT_SERIES=["close"], MAX_SEQUENCE_COUNT=1234),
                                dict(EXTENDED_TREND_PERIODS=[10, 90, 200], HORIZON_STEPS=[5, 30, 45])])
def test_trim_on_equals_off_on_bundled_csv(kw):
    from neural_trade.data.processor import split_arrays

    off = split_arrays(_cfg(BUNDLED, False, **kw))
    on = split_arrays(_cfg(BUNDLED, True, **kw))
    assert len(on["df"]) < len(off["df"])
    _assert_same(off, on)


def test_prepared_datasets_identical():
    from neural_trade.data.processor import DataProcessor

    outs = []
    for trim in (False, True):
        dp = DataProcessor(_cfg(BUNDLED, trim))
        df, close = dp.load_and_prepare_data()
        outs.append(dp.prepare_datasets(df, close))
    for a, b in zip(*outs):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)


def test_off_changes_nothing_and_none_cap_is_noop():
    from neural_trade.data.processor import DataProcessor

    n_off = len(DataProcessor(_cfg(BUNDLED, False)).load_and_prepare_data()[0])
    assert n_off == len(DataProcessor(_cfg(BUNDLED, True, MAX_SEQUENCE_COUNT=0)).load_and_prepare_data()[0])
    assert n_off == len(DataProcessor(_cfg(BUNDLED, False)).load_and_prepare_data()[0])


def test_cap_above_available_windows_is_noop():
    from neural_trade.data.processor import DataProcessor

    n = len(DataProcessor(_cfg(BUNDLED, False, MAX_SEQUENCE_COUNT=10**7)).load_and_prepare_data()[0])
    assert n == len(DataProcessor(_cfg(BUNDLED, True, MAX_SEQUENCE_COUNT=10**7)).load_and_prepare_data()[0])


def test_data_key_separates_trim():
    from neural_trade.experiments.dataset import data_key

    assert data_key(_cfg(BUNDLED, False)) != data_key(_cfg(BUNDLED, True))


def test_trimmed_row_count_formula():
    from neural_trade.data.processor import TAIL_TRIM_MARGIN_WINDOWS, DataProcessor

    cfg = _cfg(BUNDLED, True)
    n = len(DataProcessor(cfg).load_and_prepare_data()[0])
    start = max(cfg.LOOKBACK, max(cfg.EXTENDED_TREND_PERIODS))
    assert n == start + (cfg.MAX_SEQUENCE_COUNT + TAIL_TRIM_MARGIN_WINDOWS - 1) * 1 + max(cfg.HORIZON_STEPS)


def _run_check(trim):
    script = REPO / "runs" / "tactical" / "trim_check.py"
    r = subprocess.run([sys.executable, str(script), str(TACTICAL_CSV), str(int(trim)), "2020-03-30T04:00:00",
                        "126000"], capture_output=True, text=True, cwd=str(REPO),
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src"),
                            "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-2000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.slow
@pytest.mark.skipif(not TACTICAL_CSV.is_file(), reason="needs the tactical long-file excerpt")
def test_long_file_slice_identical_and_cheaper():
    off, on = _run_check(False), _run_check(True)
    meas = {k: (off.pop(k), on.pop(k)) for k in ("sec", "peak_mb", "n_bars")}
    print("TRIM_MEASURE (off, on):", meas)
    assert off == on
    assert meas["n_bars"][1] < meas["n_bars"][0]

"""DataLoaders and Preprocessors registries; the default pipeline reproduces the old loader."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import ComponentValidationError, DependencyError
from neural_trade.data.loaders import validate_ohlcv_frame
from neural_trade.registries.data_loaders import DataLoaders
from neural_trade.registries.preprocessors import Preprocessors, run_preprocessors


def _raw():
    ts = pd.date_range("2025-01-01", periods=8, freq="1min").astype(str).tolist()
    ts = [ts[3], ts[0], ts[1], ts[1], "not a date", ts[2], ts[4], ts[5]]  # unsorted, dup, garbage
    return pd.DataFrame({"datetime": ts, "open": 1.0, "high": 2.0, "low": 0.5,
                         "close": [4, 1, 2, 2.5, 9, 3, None, 6], "volume": 10})


def test_registered_components():
    assert set(DataLoaders.list_names()) == {"csv", "parquet", "dataframe"}
    assert {"standardize_ohlcv", "sort_dedupe", "resample_bars", "drop_missing_close",
            "strip_currency_symbols", "log_returns", "add_time_features"} <= set(Preprocessors.list_names())
    with pytest.raises(ComponentValidationError):
        Preprocessors.register(name="bad")(lambda frame: frame)


def test_default_pipeline_cleans_sorts_dedupes_and_resamples():
    df = run_preprocessors(DataLoaders.build("dataframe", Config(), _raw()), Config())
    validate_ohlcv_frame(df)
    assert list(df["Close"]) == [1.0, 2.5, 3.0, 4.0, 6.0]  # garbage and NaN close gone, dup kept last
    assert df["timestamp"].is_monotonic_increasing and "Date" in df


def test_optional_preprocessors():
    df = run_preprocessors(_raw(), Config(), ["standardize_ohlcv", "sort_dedupe", "resample_bars",
                                               "drop_missing_close", "log_returns", "add_time_features"])
    assert np.isclose(df["LogReturn"].iloc[1], np.log(2.5 / 1.0))
    assert {"minute_of_day", "day_of_week"} <= set(df.columns)
    money = pd.DataFrame({"timestamp": ["2025-01-01 00:00"], "Close": ["$1,234.5"]})
    assert run_preprocessors(money, Config(), ["strip_currency_symbols"])["Close"].iloc[0] == 1234.5


def test_csv_loader_matches_the_bundled_file(real_slice, tmp_path):
    path = tmp_path / "slice.csv"
    real_slice.to_csv(path, index=False)
    df = DataLoaders.build("csv", Config(), path=str(path))
    assert len(df) == len(real_slice)


def test_parquet_loader_declares_its_dependency():
    import importlib.util

    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(DependencyError):
            DataLoaders.build("parquet", Config())

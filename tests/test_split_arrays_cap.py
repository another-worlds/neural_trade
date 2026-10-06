"""NT-177: MAX_SEQUENCE_COUNT is applied BEFORE windowing (only the newest sequences are built),
with outputs identical to the old window-everything-then-cut path."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd
import pytest

from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor, _select_fold, split_arrays
from neural_trade.data.scaling import WindowNormalizer, fit_target_scaler
from neural_trade.data.splits import make_purged_splits
from neural_trade.data.windowing import (frame_series, make_multichannel_windows,
                                         make_sequences_with_extended_trends, sequence_anchor_bars)


def _frame(n, seed=3):
    rng = np.random.default_rng(seed)
    close = 20_000 + np.cumsum(rng.normal(0, 5, n))
    spread = rng.uniform(0.5, 4, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
        "Open": close + rng.normal(0, 1, n), "High": close + spread, "Low": close - spread,
        "Close": close, "Volume": rng.uniform(1, 50, n)}).astype("float64", errors="ignore")


def _cfg(**kw):
    base = dict(LOOKBACK=60, HORIZON_STEPS=[10, 15, 20], EXTENDED_TREND_PERIODS=[10, 15, 20],
                MAX_SEQUENCE_COUNT=1500, N_FOLDS=3, FOLD_INDEX=-1, INPUT_SERIES=["close"], WINDOW_STEP=1)
    base.update(kw)
    return Config(**base)


def _patch_loader(monkeypatch, df):
    close = df["Close"].values.astype("float32")
    monkeypatch.setattr(DataProcessor, "load_and_prepare_data", lambda self, read_csv_kwargs=None, **kw: (df, close))


def _old_split_arrays(config, df, close):
    """The pre-NT-177 body of split_arrays: window the whole file, then cut to the newest cap."""
    X, y, lc, ext = make_sequences_with_extended_trends(config, close, config.LOOKBACK)
    names = list(config.INPUT_SERIES or ["close"])
    Xm = X if names == ["close"] else make_multichannel_windows(config, frame_series(config, df), config.LOOKBACK)
    n_total = X.shape[0]
    cap = config.MAX_SEQUENCE_COUNT
    if cap and X.shape[0] > cap:
        X, y, lc, ext, Xm = X[-cap:], y[-cap:], lc[-cap:], ext[-cap:], Xm[-cap:]
    folds = make_purged_splits(X.shape[0], lookback=config.LOOKBACK, horizon_steps=config.HORIZON_STEPS,
                               window_step=int(max(1, config.WINDOW_STEP)), n_folds=config.N_FOLDS,
                               val_fraction=config.VAL_FRACTION, cal_fraction=config.CAL_FRACTION)
    fold = _select_fold(folds, config.FOLD_INDEX)
    out = {"fold": fold}
    for name in ("train", "val", "cal", "test"):
        idx = getattr(fold, name)
        out[name] = {"X": X[idx], "X_model": Xm[idx], "y": y[idx], "last_close": lc[idx],
                     "extended_trends": ext[idx], "index": idx,
                     "anchor_bar": sequence_anchor_bars(config, len(close), n_total, idx)}
    return out


CASES = [
    dict(MAX_SEQUENCE_COUNT=1500),
    dict(MAX_SEQUENCE_COUNT=1500, FOLD_INDEX=0),
    dict(MAX_SEQUENCE_COUNT=1200, FOLD_INDEX=1, INPUT_SERIES=["close", "volume"]),
    dict(MAX_SEQUENCE_COUNT=1300, LOOKBACK=30, HORIZON_STEPS=[5, 10, 25], EXTENDED_TREND_PERIODS=[5, 40, 90],
         INPUT_SERIES=["open", "high", "low", "close", "volume"]),
    dict(MAX_SEQUENCE_COUNT=700, WINDOW_STEP=3, N_FOLDS=2, EXTENDED_TREND_PERIODS=[10, 15, 120]),
    dict(MAX_SEQUENCE_COUNT=0),           # no cap: nothing is dropped
    dict(MAX_SEQUENCE_COUNT=10_000),      # cap above the sequence count: nothing is dropped
]


@pytest.mark.parametrize("kw", CASES, ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()))
def test_cap_before_windowing_matches_window_all_then_cut(monkeypatch, kw):
    df = _frame(3_000)
    cfg = _cfg(**kw)
    _patch_loader(monkeypatch, df)
    close = df["Close"].values.astype("float32")
    old = _old_split_arrays(cfg, df, close)
    new = split_arrays(cfg)
    for f in ("train", "val", "cal", "test"):
        assert getattr(old["fold"], f).tolist() == getattr(new["fold"], f).tolist()
    assert old["fold"].gap == new["fold"].gap
    for name in ("train", "val", "cal", "test"):
        for key, ref in old[name].items():
            np.testing.assert_array_equal(new[name][key], ref, err_msg=f"{name}.{key}")
            assert new[name][key].dtype == ref.dtype


@pytest.mark.parametrize("kw", CASES[:5], ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()))
def test_prepare_datasets_and_scaler_fits_are_unchanged(monkeypatch, kw):
    df = _frame(3_000)
    cfg = _cfg(**kw)
    close = df["Close"].values.astype("float32")
    # old: window all, cut, hand the trimmed windows to the split/scale step
    X, y, lc, ext = make_sequences_with_extended_trends(cfg, close, cfg.LOOKBACK)
    names = list(cfg.INPUT_SERIES)
    Xm = X if names == ["close"] else make_multichannel_windows(cfg, frame_series(cfg, df), cfg.LOOKBACK)
    cap = cfg.MAX_SEQUENCE_COUNT
    if cap and X.shape[0] > cap:
        X, y, lc, ext, Xm = X[-cap:], y[-cap:], lc[-cap:], ext[-cap:], Xm[-cap:]
    dp_old = DataProcessor(cfg)
    old = dp_old.prepare_datasets_from_windows(X, y, lc, ext, X_model=Xm)
    dp_new = DataProcessor(cfg)
    new = dp_new.prepare_datasets(df, close)
    for a, b in zip(old[:-1], new[:-1]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(old[-1].mean_, new[-1].mean_)
    np.testing.assert_array_equal(old[-1].scale_, new[-1].scale_)
    assert dp_old.normalizer.to_dict() == dp_new.normalizer.to_dict()
    for blk in ("val_block", "cal_block"):
        for k, v in getattr(dp_old, blk).items():
            np.testing.assert_array_equal(getattr(dp_new, blk)[k], v)
    np.testing.assert_array_equal(dp_old.test_windows_raw, dp_new.test_windows_raw)
    assert dp_new.fold.train.tolist() == dp_old.fold.train.tolist()
    # keep the imports honest: the fits above are the ones fit_target_scaler / WindowNormalizer make
    assert fit_target_scaler is not None and WindowNormalizer is not None


def test_peak_memory_follows_the_cap_not_the_file(monkeypatch):
    """500k bars, cap 2000: the old path windowed ~500k x 60 x 5 float32 (~600 MB) before cutting."""
    n = 500_000
    df = _frame(n)
    _patch_loader(monkeypatch, df)
    cfg = _cfg(MAX_SEQUENCE_COUNT=2_000, N_FOLDS=2, INPUT_SERIES=["open", "high", "low", "close", "volume"])
    whole_file_windows = n * cfg.LOOKBACK * 5 * 4
    tracemalloc.start()
    try:
        out = split_arrays(cfg)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert sum(len(out[b]["index"]) for b in ("train", "val", "cal", "test")) <= 2_000
    assert peak < 0.05 * whole_file_windows, (peak / 2 ** 20, whole_file_windows / 2 ** 20)
    # the newest sequences are the ones kept: the last test anchor is the file's last usable decision bar
    assert out["test"]["anchor_bar"][-1] == n - max(cfg.HORIZON_STEPS) - 1

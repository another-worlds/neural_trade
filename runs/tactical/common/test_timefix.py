"""Tests of timefix.py (T11 Part 1). Data tests skip when a file is absent. usage: python -m pytest -q runs/tactical/common/test_timefix.py"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import timefix as TF  # noqa: E402

TAC = os.path.dirname(HERE)
T09 = os.path.join(TAC, "tasks", "T09_regression_anatomy")
CUT = "D:/nt/nt_recut_data/Bitcoin_BTCUSDT_cut.csv"
BIN06 = "D:/nt/nt_fwd_data/BTCUSDT_1m_2025-06.csv"


def _t09_lib():
    sys.path.insert(0, T09)
    try:
        spec = importlib.util.spec_from_file_location("t09_lib_for_t11", os.path.join(T09, "lib.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.path.remove(T09)
    return m


def _grid():
    """Every 7th minute 2016-06..2026-01 plus a dense minute grid around each DST change and the splice."""
    g = [pd.date_range("2016-06-01", "2026-01-01", freq="7min")]
    for y in range(2016, 2026):
        for m, n in ((3, 2), (11, 1)):
            d = TF._nth_sunday(y, m, n); g.append(pd.date_range(d - pd.Timedelta(hours=3), d + pd.Timedelta(hours=27), freq="min"))
    g.append(pd.date_range(TF.SPLICE - pd.Timedelta(hours=3), TF.SPLICE + pd.Timedelta(hours=3), freq="min"))
    return g[0].union_many(g[1:]) if hasattr(g[0], "union_many") else pd.DatetimeIndex(np.unique(np.concatenate([x.values for x in g])))


def test_reproduces_t09_utc_hour_exactly():
    L = _t09_lib(); t = _grid(); tn = t.values
    assert np.array_equal(TF.utc_hour(t, "bitstamp"), L.utc_hour(tn, "cut"))
    assert np.array_equal(TF.utc_hour(t, "binance"), L.utc_hour(tn, "fwd"))
    assert np.array_equal(TF.to_utc(t, "binance"), t)


def test_splice_offset_and_known_pre_splice_offsets():
    # +4 in US summer, +5 in winter (H91: best lags 240 and 300 minutes); 0 from the splice on
    s = pd.DatetimeIndex(["2023-07-01 12:00", "2023-01-15 12:00", "2018-07-04 00:30", "2020-12-31 23:59"])
    assert list(TF.to_utc(s) ) == list(s - pd.to_timedelta([4, 5, 4, 5], unit="h"))
    assert list(TF.file_offset_hours(s)) == [4, 5, 4, 5]
    assert list(TF.file_offset_hours(pd.DatetimeIndex(["2024-09-13 07:59", "2024-09-13 08:00", "2024-09-13 08:01", "2025-01-01"]))) == [4, 0, 0, 0]
    assert TF.to_utc(pd.DatetimeIndex(["2024-09-13 08:00"]))[0] == pd.Timestamp("2024-09-13 08:00")
    # the daylight-saving dates, by date: 2023 = Mar 12 .. Nov 5
    d = pd.DatetimeIndex(["2023-03-11 23:59", "2023-03-12 00:00", "2023-11-04 23:59", "2023-11-05 00:00"])
    assert list(TF.file_offset_hours(d)) == [5, 4, 4, 5]


def test_input_types_and_source_validation():
    t = pd.DatetimeIndex(["2022-02-02 10:00"])
    assert TF.to_utc(t.values, "bitstamp")[0] == pd.Timestamp("2022-02-02 05:00")
    assert TF.to_utc(pd.Series(t), "cut")[0] == pd.Timestamp("2022-02-02 05:00")
    with pytest.raises(ValueError):
        TF.to_utc(t, "kraken")


def test_ny_time_matches_zoneinfo_and_rule():
    zi = pytest.importorskip("zoneinfo")
    try:
        tz = zi.ZoneInfo("America/New_York")
    except Exception:
        pytest.skip("tzdata missing")
    rng = np.random.default_rng(0)
    t = pd.DatetimeIndex(np.sort(pd.Timestamp("2016-01-01").value + rng.integers(0, int(11 * 365.25 * 86400e9), 4000)).astype("datetime64[ns]"))
    ref = t.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    assert np.array_equal(TF.ny_time(t).values, ref.values)
    # the transition instants themselves (2023: Mar 12 07:00 UTC and Nov 5 06:00 UTC)
    e = pd.DatetimeIndex(["2023-03-12 06:59", "2023-03-12 07:00", "2023-11-05 05:59", "2023-11-05 06:00"])
    assert list(TF.ny_time(e)) == list(pd.DatetimeIndex(["2023-03-12 01:59", "2023-03-12 03:00", "2023-11-05 01:59", "2023-11-05 01:00"]))


def test_seam_mask_marks_dst_days_and_the_splice():
    t = pd.DatetimeIndex(["2018-03-11 08:00", "2018-03-11 00:00", "2018-03-10 22:59", "2018-03-10 23:00", "2018-03-12 00:59", "2018-03-12 01:01",
                          "2024-09-13 08:01", "2024-09-13 05:00", "2024-09-13 05:59", "2024-09-13 11:00", "2024-09-13 11:02", "2021-05-05 12:00",
                          "2025-03-09 12:00"])
    assert list(TF.seam_mask(t, 0)) == [True, True, False, False, False, False, True, False, False, False, False, False, False]
    assert list(TF.seam_mask(t, 61))[:6] == [True, True, True, True, True, False]
    assert list(TF.seam_mask(t, 61))[6:11] == [True, True, True, True, False]   # splice region 06:00-10:00, padded 61 min: 04:59 .. 11:01
    assert not TF.seam_mask(t, 61, "binance").any()


def test_seam_mask_equals_t09_before_the_splice_day_and_differs_only_after():
    L = _t09_lib(); t = pd.date_range("2016-12-01", "2025-02-01", freq="13min")
    mine = TF.seam_mask(t, 61); ref = L.seam_mask(t.values, "cut", 61)
    diff = mine != ref
    assert not diff[np.asarray(t < pd.Timestamp("2024-09-14"))].any()
    # T09 also masks the 2024-11-03 UTC day (no seam there once the file is UTC); the helper does not
    assert diff.sum() > 0 and set(t[diff].normalize()) <= {pd.Timestamp("2024-11-02"), pd.Timestamp("2024-11-03"), pd.Timestamp("2024-11-04")}
    assert not (mine & ~ref).any()


def _read_window(path, lo, hi, cols=("timestamp", "close")):
    parts = []
    for ch in pd.read_csv(path, usecols=list(cols), chunksize=500_000):
        ts = pd.to_datetime(ch["timestamp"]); k = ((ts >= lo) & (ts <= hi)).to_numpy()
        if k.any():
            c = ch[k].copy(); c.index = ts[k]; parts.append(c["close"])
        if (ts > hi).any():
            break
    return pd.concat(parts)


@pytest.mark.skipif(not os.path.exists(CUT), reason="cut file absent")
def test_known_seams_in_the_data():
    d = _read_window(CUT, "2018-03-11 06:00", "2018-03-11 10:00"); r = np.log(d).diff().abs() * 1e4
    assert r.idxmax() == pd.Timestamp("2018-03-11 08:00") and 100 < r.max() < 120           # the 109 bps jump
    assert TF.seam_mask(pd.DatetimeIndex([r.idxmax()]), 0)[0]
    d = _read_window(CUT, "2024-09-13 06:00", "2024-09-13 10:00"); r = np.log(d).diff().abs() * 1e4
    assert r.idxmax() == pd.Timestamp("2024-09-13 08:01") and r.max() > 40
    assert TF.seam_mask(pd.DatetimeIndex([r.idxmax()]), 0)[0]


@pytest.mark.skipif(not (os.path.exists(CUT) and os.path.exists(BIN06)), reason="data absent")
def test_binance_2025_06_cross_check_lag_zero_after_the_splice():
    a = _read_window(CUT, "2025-06-01 00:00", "2025-06-30 23:59"); b = _read_window(BIN06, "2025-06-01 00:00", "2025-06-30 23:59")
    ra = np.log(a).diff().dropna(); rb = np.log(b).diff().dropna(); ra.index = TF.to_utc(ra.index, "bitstamp")
    j = ra.index.intersection(rb.index); assert len(j) > 40000
    cor = {lag: float(np.corrcoef(ra.reindex(j).values[10:-10], rb.shift(lag).reindex(j).values[10:-10])[0, 1]) for lag in range(-8, 9)}
    best = max(cor, key=cor.get)
    assert best == 0 and cor[0] > 0.7 and cor[0] > 2 * max(v for k, v in cor.items() if k != 0)


@pytest.mark.skipif(not os.path.exists(CUT), reason="cut file absent")
def test_pre_splice_volatility_peak_moves_to_the_us_open_in_utc():
    """File time smears the US-session volatility peak; in UTC it sits at 13-15 h (NY 09-10 h plus the macro releases at 12:30-14:30 UTC)."""
    d = _read_window(CUT, "2021-12-01 00:00", "2022-06-30 23:59"); r = (np.log(d).diff().abs()).dropna()
    by_utc = r.groupby(TF.to_utc(r.index, "bitstamp").hour).mean(); by_file = r.groupby(r.index.hour).mean()
    assert by_utc.idxmax() in (13, 14, 15) and by_file.idxmax() in (17, 18, 19, 20)       # file time = UTC + 5 (winter) or + 4 (summer)

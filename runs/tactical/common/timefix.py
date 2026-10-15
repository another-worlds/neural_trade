"""Correct clocks (T11, hypothesis 1.1): file time -> UTC -> New York, and the file-time seams.

The Bitstamp history (the cut file) is in FILE time before the 2024-09-13 splice: UTC+4 in US daylight saving, UTC+5 otherwise (H91; the best
cross-correlation lags against a UTC source are 240 and 300 minutes). From 2024-09-13 08:00 file time on, the file is UTC. The Binance forward
file is UTC throughout. Several tactical libraries read the hour from file time (t04_lib, volnet_lib); this module is the one place that
converts. Pure numpy / pandas; no tz database needed (zoneinfo is used only by the tests, as a cross-check).

  to_utc(ts, source)   file timestamps -> UTC (naive DatetimeIndex)
  ny_time(ts_utc)      UTC -> New York local time (naive DatetimeIndex)
  seam_mask(ts, pad_min, source)  True for timestamps within pad_min minutes of an artificial jump of the file
  utc_hour(ts, source) hour of day in UTC, identical to tasks/T09_regression_anatomy/lib.py:utc_hour
"""
import numpy as np
import pandas as pd

SPLICE = pd.Timestamp("2024-09-13 08:00")        # file time: from here on the cut file is UTC
SOURCES_FILE_TIME = ("bitstamp", "cut", "file")  # the cut Bitstamp history
SOURCES_UTC = ("binance", "utc")                 # already UTC


def _nth_sunday(y, m, n):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(6 - d.dayofweek) % 7) + pd.Timedelta(weeks=n - 1)


def _idx(ts):
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ts)).astype("datetime64[ns]")) if not isinstance(ts, pd.DatetimeIndex) else ts


def _years(t, pad=1):
    if len(t) == 0:
        return range(2016, 2017)
    return range(int(t.min().year) - pad, int(t.max().year) + pad + 1)


def _check_source(source):
    if source not in SOURCES_FILE_TIME + SOURCES_UTC:
        raise ValueError(f"unknown source {source!r}; use one of {SOURCES_FILE_TIME + SOURCES_UTC}")


def us_dst_by_date(t):
    """Boolean array: the DATE lies in US daylight saving (2nd Sunday of March 00:00 to 1st Sunday of November 00:00, calendar-day boundaries)."""
    t = _idx(t); out = np.zeros(len(t), bool)
    for y in _years(t):
        a, b = _nth_sunday(y, 3, 2), _nth_sunday(y, 11, 1)
        out |= np.asarray((t >= a) & (t < b))
    return out


def file_offset_hours(ts):
    """Hours to subtract from a Bitstamp file timestamp to get UTC: 4 in US daylight saving (by date), 5 otherwise, 0 from the splice on."""
    t = _idx(ts); off = np.where(us_dst_by_date(t), 4, 5)
    off[np.asarray(t >= SPLICE)] = 0
    return off


def to_utc(ts, source="bitstamp"):
    """File timestamps to UTC. Bitstamp (the cut file): minus 4 h in US daylight saving (2nd Sunday of March to 1st Sunday of November, by date),
    minus 5 h otherwise, 0 from 2024-09-13 08:00 file time on. Binance: unchanged."""
    _check_source(source); t = _idx(ts)
    if source in SOURCES_UTC:
        return t
    return t - pd.to_timedelta(file_offset_hours(t), unit="h")


def ny_time(ts_utc):
    """New York local time of UTC timestamps (naive result). Daylight time (UTC-4) runs from the 2nd Sunday of March 07:00 UTC (02:00 EST)
    to the 1st Sunday of November 06:00 UTC (02:00 EDT); UTC-5 otherwise. The US rule in force since 2007 (earlier years are not used)."""
    t = _idx(ts_utc); dst = np.zeros(len(t), bool)
    for y in _years(t):
        a = _nth_sunday(y, 3, 2) + pd.Timedelta(hours=7); b = _nth_sunday(y, 11, 1) + pd.Timedelta(hours=6)
        dst |= np.asarray((t >= a) & (t < b))
    return t - pd.to_timedelta(np.where(dst, 4, 5), unit="h")


def seam_regions(years=range(2016, 2026)):
    """File-time intervals of the cut file that hold an artificial jump: the whole day of each US daylight-saving change that falls before the
    splice (the seam sits at a varying file hour) and 2 hours around the 2024-09-13 splice."""
    reg = []
    for y in years:
        for m, n in ((3, 2), (11, 1)):
            d = _nth_sunday(y, m, n)
            if d < SPLICE:
                reg.append((d, d + pd.Timedelta(days=1)))
    reg.append((SPLICE - pd.Timedelta(hours=2), SPLICE + pd.Timedelta(hours=2)))
    return reg


def seam_mask(ts, pad_min=61, source="bitstamp"):
    """True for file timestamps within pad_min minutes of a seam (every DST-change day before the splice, +-2 h around the splice).
    pad_min 61 covers an anchor whose 60-bar window or 60-bar label touches a seam (T09). A UTC source has no seam."""
    _check_source(source); t = _idx(ts); out = np.zeros(len(t), bool)
    if source in SOURCES_UTC:
        return out
    pad = pd.Timedelta(minutes=pad_min)
    for a, b in seam_regions(_years(t)):
        out |= np.asarray((t >= a - pad) & (t < b + pad))
    return out


def utc_hour(ts, source="bitstamp"):
    """Hour of day in UTC (int array); identical to T09's utc_hour(t, which='cut' / forward)."""
    return np.asarray(to_utc(ts, source).hour.values, dtype=int)

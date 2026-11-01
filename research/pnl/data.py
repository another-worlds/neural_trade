"""Data layer for the PnL study: BTCUSDT USD-M perp 1-min bars and 8-hourly funding.

Development: 2020-01-01 .. 2025-10-31. SEALED: 2025-11-01 .. 2026-10-30 (scored once, after pre-registration).
Known data facts:
- one missing day, 2026-10-28 (not in the Binance archive): the bar after the gap is flagged (`gap_before`);
- funding for 2026-10 is rebuilt from premium-index klines with Binance's time-weighted formula
  (validated on 2026-08/09: mean abs error 0.02 bps per print, corr 0.993); `funding_estimated` flags those prints.
"""
import datetime as dt
import os

import numpy as np

DATA = os.environ.get("PNL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "data"))
SEAL_START = int(dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc).timestamp())
DEV_START = int(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp())
# windows studied before on spot data (inside the seal); the sealed result is also reported without them
SEEN_WINDOWS = [(int(dt.datetime(2025, 10, 11, tzinfo=dt.timezone.utc).timestamp()),
                 int(dt.datetime(2025, 11, 10, 7, 30, tzinfo=dt.timezone.utc).timestamp())),
                (int(dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc).timestamp()),
                 int(dt.datetime(2026, 10, 30, 23, 59, tzinfo=dt.timezone.utc).timestamp()))]


class Bars:
    """1-min (or resampled) bars as numpy arrays, plus funding aligned to bar indices."""

    def __init__(self, t, o, h, l, c, v, qv, n, tbb, minutes, funding_t, funding_r, funding_est):
        self.t, self.o, self.h, self.l, self.c = t, o, h, l, c
        self.v, self.qv, self.n, self.tbb = v, qv, n, tbb
        self.minutes = minutes
        self.lc = np.log(c)
        self.gap_before = np.zeros(len(t), bool)
        self.gap_before[1:] = np.diff(t) > 60 * minutes
        # funding paid at the funding timestamp by the position held over it: map each print to the first bar whose
        # open time is >= the print (a position held through the end of the previous bar pays it)
        idx = np.searchsorted(t, funding_t)
        ok = idx < len(t)
        self.funding = np.zeros(len(t))
        np.add.at(self.funding, idx[ok], funding_r[ok])
        self.funding_estimated = np.zeros(len(t), bool)
        self.funding_estimated[idx[ok & funding_est]] = True

    def __len__(self):
        return len(self.t)

    def slice(self, a, b):
        out = object.__new__(Bars)
        for k, v in self.__dict__.items():
            out.__dict__[k] = v[a:b] if isinstance(v, np.ndarray) else v
        return out

    def index_of(self, ts):
        return int(np.searchsorted(self.t, ts))


def load(minutes=1):
    d = np.load(os.path.join(DATA, "btcusdt_perp_1m.npz"))
    f = np.load(os.path.join(DATA, "funding_full.npz"))
    t, o, h, l, c = d["t"], d["open"], d["high"], d["low"], d["close"]
    v, qv, n, tbb = d["volume"], d["quote_volume"], d["trades"], d["taker_buy_base"]
    if minutes > 1:
        t, o, h, l, c, v, qv, n, tbb = resample(minutes, t, o, h, l, c, v, qv, n, tbb)
    return Bars(t, o, h, l, c, v, qv, n, tbb, minutes, f["t"], f["rate"], f["estimated"])


def resample(k, t, o, h, l, c, v, qv, n, tbb):
    """Aggregate 1-min bars into k-minute bars aligned to UTC multiples of k minutes (missing minutes skipped)."""
    key = t // (60 * k)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(t)]
    agg = lambda x, f: f.reduceat(x, starts) if f is not None else x
    return (key[starts] * 60 * k, o[starts], np.maximum.reduceat(h, starts), np.minimum.reduceat(l, starts),
            c[ends - 1], np.add.reduceat(v.astype(np.float64), starts).astype(np.float32),
            np.add.reduceat(qv.astype(np.float64), starts).astype(np.float32),
            np.add.reduceat(n.astype(np.float64), starts).astype(np.float32),
            np.add.reduceat(tbb.astype(np.float64), starts).astype(np.float32))


def utc(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)

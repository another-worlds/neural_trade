import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import regime_vars as R


def _close(n=6000, seed=0):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 1e-3, n)))


def test_no_future_bars():
    c = _close(); pos = np.array([1500, 3000, 4500]); a = R.raw_vars(c, pos)
    c2 = c.copy(); c2[4501:] *= np.exp(np.random.default_rng(1).normal(0, 0.5, len(c2) - 4501))   # wreck everything after the last decision bar
    b = R.raw_vars(c2, pos)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    c3 = c.copy(); c3[3001:] *= 3.0                                                               # and after the middle one
    assert np.array_equal(R.raw_vars(c3, pos[:2])["vol"], a["vol"][:2])


def test_short_history_has_no_regime():
    c = _close(); assert np.isnan(R.raw_vars(c, np.array([1439]))["vol"][0]) and np.isfinite(R.raw_vars(c, np.array([1440]))["vol"][0])


def test_values_match_definition():
    c = _close(); i = 3000; r = np.diff(np.log(c))
    v = R.raw_vars(c, np.array([i]))
    v24 = np.sqrt(np.mean(r[i - 1440:i] ** 2)); v1 = np.sqrt(np.mean(r[i - 60:i] ** 2))
    assert np.isclose(v["vol"][0], v24) and np.isclose(v["vtrend"][0], v1 / v24)
    assert np.isclose(v["trend"][0], abs(np.log(c[i] / c[i - 1440])) / (v24 * np.sqrt(1440)))


def test_cuts_from_training_only():
    c = _close(); tr = np.arange(1500, 3000, 7); ev = np.arange(3500, 5900, 7)
    t = pd.date_range("2020-01-01", periods=len(c), freq="min")
    _, cut1 = R.cell_codes(c, ev, t[ev] + pd.Timedelta("1min"), train_pos=tr)
    c2 = c.copy(); c2[3400:] *= np.exp(np.random.default_rng(2).normal(0, 0.1, len(c2) - 3400))   # alter the evaluated period
    _, cut2 = R.cell_codes(c2, ev, t[ev] + pd.Timedelta("1min"), train_pos=tr)
    for k in cut1:
        assert np.array_equal(cut1[k], cut2[k])
    codes, _ = R.cell_codes(c, ev, t[ev] + pd.Timedelta("1min"), cut_fit=cut1)
    assert set(np.unique(codes["vol"])) <= {0, 1, 2}


def test_calendar():
    t = pd.DatetimeIndex(["2024-07-06 07:59", "2024-07-06 08:00", "2024-07-08 23:59", "2024-07-08 16:00"])
    s, w = R.calendar(t)
    assert list(s) == [0, 1, 2, 2] and list(w) == [1, 1, 0, 0]

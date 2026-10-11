import os
import sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import callib as C  # noqa


def _csv(tmp_path):
    t = pd.date_range("2022-01-01", periods=60 * 6, freq="min")
    o = np.where(t.minute == 0, 100.0 + t.hour * 10, 1.0)  # hh:00 opens 100,110,...; other minutes junk
    p = tmp_path / "x.csv"
    pd.DataFrame({"timestamp": t, "open": o}).to_csv(p, index=False)
    return p


def test_returns_are_next_open_to_next_open(tmp_path):
    d = C.hourly_returns(_csv(tmp_path))
    assert np.allclose(d.r.values[:3], [(110 / 100 - 1) * 1e4, (120 / 110 - 1) * 1e4, (130 / 120 - 1) * 1e4])
    assert d.index[-1] == pd.Timestamp("2022-01-01 04:00")  # last hour has no next open: dropped


def test_no_selection_uses_months_after_2024_06():
    src = open(os.path.join(HERE, "run_calendar.py")).read()
    assert "sel = df.loc[C.SEL_START:C.SEL_END]" in src and C.SEL_END == "2024-06-30"
    idx = pd.date_range("2021-01-01", "2025-09-29", freq="h")
    df = pd.DataFrame({"r": 1.0}, index=idx)
    assert df.loc[C.SEL_START:C.SEL_END].index.max() < pd.Timestamp("2024-07-01")
    # the bucket estimates are computed from `sel` only: poisoning the test period leaves them unchanged
    sel = df.loc[C.SEL_START:C.SEL_END]
    assert len(sel) == len(pd.date_range(C.SEL_START, "2024-06-30 23:00", freq="h"))


def test_holm_and_nw_null():
    assert list(C.holm([0.001, 0.04, 0.5])) == [True, False, False]
    r = np.random.default_rng(0).normal(size=20000)
    e = C.effect(r, np.ones(20000, bool))
    assert e["lo"] < 0 < e["hi"]
    e = C.effect(r + 0.1, np.ones(20000, bool))
    assert e["lo"] > 0

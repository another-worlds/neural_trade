"""Tests for research/pnl/metrics.py on synthetic series with known answers."""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M  # noqa: E402


def _days(y, m, d, n):
    t0 = int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())
    return t0 + 86400 * np.arange(n)


def test_returns_known():
    r = np.full(365, 0.01)
    out = M.returns(r)
    assert out["total_additive_bps"] == pytest.approx(36500)
    assert out["ann_additive_bps"] == pytest.approx(36500)
    assert out["total_compound_bps"] == pytest.approx(1e4 * np.expm1(3.65))
    assert out["ann_compound_bps"] == pytest.approx(out["total_compound_bps"])
    half = M.returns(np.full(730, 0.001))                      # two years: annualised = half the total (additive)
    assert half["ann_additive_bps"] == pytest.approx(365 * 10) and half["total_additive_bps"] == pytest.approx(7300)


def test_sharpe_vol_sortino_known():
    a, b, T = 0.001, 0.02, 1000
    r = a + b * np.tile([1.0, -1.0], T // 2)
    sd = b * np.sqrt(T / (T - 1))
    assert M.vol(r) == pytest.approx(sd * np.sqrt(365))
    assert M.sharpe(r) == pytest.approx(a / sd * np.sqrt(365))
    # downside deviation over all days: sqrt(mean(min(r,0)^2)) = sqrt(0.5 * (a - b)^2)
    assert M.sortino(r) == pytest.approx(a / np.sqrt(0.5 * (a - b) ** 2) * np.sqrt(365))
    assert M.sharpe(np.zeros(10)) == 0 and M.sortino(np.zeros(10)) == 0
    assert M.sortino(np.full(10, 0.01)) == np.inf
    R = np.vstack([r, -r])
    assert np.allclose(M.sharpe(R, axis=1), [M.sharpe(r), -M.sharpe(r)])


def test_max_drawdown_known():
    # log equity 0, .1, 0, -.1, .2: peak at point 1, trough at 3 (DD 1 - e^-0.2), recovered at 4 -> 3 days
    out = M.drawdown(np.array([0.1, -0.1, -0.1, 0.3]))
    assert out["max_dd"] == pytest.approx(1 - np.exp(-0.2))
    assert (out["peak"], out["trough"], out["recovery"], out["duration_days"]) == (1, 3, 4, 3)
    assert out["recovered"] and out["longest_underwater_days"] == 3
    # not recovered: log equity 0, -.1, -.05, .05, -.25, -.26; max DD from .05 down to -.26, open for 2 days;
    # the earlier dip (points 1-2) recovered at point 3 and spans 3 days
    out = M.drawdown(np.array([-0.1, 0.05, 0.1, -0.3, -0.01]))
    assert out["max_dd"] == pytest.approx(1 - np.exp(-0.31))
    assert (out["peak"], out["trough"], out["recovery"], out["duration_days"]) == (3, 5, None, 2)
    assert not out["recovered"] and out["longest_underwater_days"] == 3
    # a flat stretch at the peak: the drawdown starts from the last point at the peak
    out = M.drawdown(np.array([0.1, 0.0, -0.05, 0.05]))
    assert (out["peak"], out["duration_days"]) == (2, 2)
    assert M.drawdown(np.full(5, 0.01))["max_dd"] == 0
    assert M.max_drawdown(np.array([0.1, -0.1, -0.1, 0.3])) == pytest.approx(1 - np.exp(-0.2))


def test_calmar_known():
    r = np.r_[np.full(10, -0.01), np.full(355, 0.002)]        # one year; DD 1 - e^-0.1
    assert M.calmar(r) == pytest.approx(np.expm1(r.sum()) / (1 - np.exp(-0.1)))
    assert M.calmar(np.full(20, 0.01)) == np.inf


def test_positive_months_and_years():
    days = _days(2021, 12, 1, 62)                              # December 2021 and January 2022
    r = np.r_[np.full(31, 0.001), np.full(31, -0.002)]
    pm = M.positive_months(r, days)
    assert pm == {"pct_positive_months": 0.5, "months": 2}
    yrs = M.pnl_by_year(r, days)
    assert list(yrs) == [2021, 2022]
    assert yrs[2021]["additive_bps"] == pytest.approx(310) and yrs[2022]["additive_bps"] == pytest.approx(-620)
    assert yrs[2022]["compound_bps"] == pytest.approx(1e4 * np.expm1(-0.062)) and yrs[2021]["days"] == 31
    # same answer from day numbers, datetime64 and tz-aware pandas timestamps
    assert M.positive_months(r, days // 86400, unit="D") == pm
    assert M.positive_months(r, days // 86400) == pm and M.pnl_by_year(r, days // 86400) == yrs   # sim day ids
    assert M.positive_months(r, days, unit="s") == pm
    assert M.positive_months(r, days.astype("datetime64[s]")) == pm
    assert M.pnl_by_year(r, pd.to_datetime(days, unit="s", utc=True)) == yrs


def test_exposure_and_turnover():
    x = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    assert M.exposure_stats(x) == pytest.approx({"mean_abs_x": 0.6, "mean_x": 0.2, "pct_in_market": 0.6})
    tv = M.turnover(x)                                         # 1 + 0 + 2 + 1 from flat
    assert tv["turnover"] == 4
    assert M.turnover(np.r_[x, np.zeros(360)], bar_minutes=1440)["turnover_per_year"] == pytest.approx(4)
    assert M.turnover(np.ones(525600))["turnover_per_year"] == pytest.approx(1)     # one entry in a 1-min year


def test_static_timing_split():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(5000) * 1e-3
    x = rng.choice([-1.0, 0.0, 1.0], 5000)
    out = M.static_timing_split(x, r)
    xh = np.r_[0.0, x[:-1]]
    assert out["total_bps"] == pytest.approx(1e4 * xh @ np.expm1(r))
    assert out["static_bps"] + out["timing_bps"] == pytest.approx(out["total_bps"])
    const = M.static_timing_split(np.full(100, 0.5), r[:100], held=True)
    assert const["timing_bps"] == pytest.approx(0, abs=1e-9)
    assert const["static_bps"] == pytest.approx(1e4 * 0.5 * np.expm1(r[:100]).sum())


def test_trade_stats_known():
    trades = [{"gross_bps": g, "net_bps": g - 8, "fees_bps": 7, "funding_bps": 1, "hold_minutes": h}
              for g, h in [(20, 15), (-4, 15), (10, 30), (2, 60)]]
    out = M.trade_stats(trades, years=2)
    net = np.array([12.0, -12.0, 2.0, -6.0])
    assert out["count"] == 4 and out["trades_per_year"] == 2
    assert out["hit_rate"] == 0.5
    assert out["mean_net_bps"] == pytest.approx(-1.0) and out["median_net_bps"] == pytest.approx(-2.0)
    assert out["sd_net_bps"] == pytest.approx(net.std(ddof=1))
    assert out["mean_gross_bps"] == 7 and out["mean_fees_bps"] == 7 and out["mean_funding_bps"] == 1
    assert out["mean_hold_minutes"] == 30 and out["median_hold_minutes"] == 22.5
    assert M.trade_stats(pd.DataFrame(trades))["mean_net_bps"] == pytest.approx(-1.0)
    empty = M.trade_stats([])
    assert empty["count"] == 0 and np.isnan(empty["mean_net_bps"])


def test_beta_alpha_known():
    rng = np.random.default_rng(1)
    n = 800
    b = rng.standard_normal(n) * 0.03
    e = rng.standard_normal(n) * 0.01
    X = np.column_stack([np.ones(n), b])
    e -= X @ np.linalg.lstsq(X, e, rcond=None)[0]              # residual orthogonal to [1, b]: OLS is exact
    r = 0.0005 + 0.4 * b + e
    out = M.beta_alpha(r, b)
    assert out["beta"] == pytest.approx(0.4) and out["alpha"] == pytest.approx(0.0005)
    assert out["alpha_ann_bps"] == pytest.approx(0.0005 * 365 * 1e4)
    lr = linregress(b, r)
    assert out["t_beta"] == pytest.approx(lr.slope / lr.stderr)
    assert out["t_alpha"] == pytest.approx(lr.intercept / lr.intercept_stderr)
    assert out["r2"] == pytest.approx(lr.rvalue ** 2)
    hac = M.beta_alpha(r, b, hac_lag=0)                        # White SEs: same coefficients, finite t
    assert hac["beta"] == pytest.approx(0.4) and np.isfinite(hac["t_alpha"])


def test_summary_json():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.02, 400)
    days = _days(2021, 1, 1, 400)
    x = rng.choice([0.0, 0.5, 1.0], 400 * 24)
    trades = [{"gross_bps": 5, "net_bps": -2, "fees_bps": 7, "funding_bps": 0, "hold_minutes": 60}] * 3
    out = M.summary(r, days=days, exposure=x, trades=trades, bench=rng.normal(0, 0.03, 400), bar_minutes=60)
    assert set(out["by_year"]) == {2021, 2022} and out["trades"]["count"] == 3
    assert out["sharpe"] == pytest.approx(M.sharpe(r)) and out["max_dd"] == pytest.approx(M.max_drawdown(r))
    json.dumps(out)

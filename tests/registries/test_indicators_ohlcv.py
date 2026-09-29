"""The ten OHLCV indicator families of NT-047 (D-031): contract, textbook tolerance,
causality and the M(eps) empirical offset-invariance bound of D-037.

Smoothing convention (stated per family below): every rolling average is the EWMA with
alpha = 2 / (period + 1) - the same convention NT-046's four families pinned - so the
numpy textbook references here use that EWMA where the classic description says SMA or
Wilder smoothing. The smooth substitutions under test are the soft sign / gate
(tanh / sigmoid at sharpness 10) and the smooth rolling extremum
(neural_trade.indicators.base.soft_rolling_extremum).
"""
from __future__ import annotations

import numpy as np
import pytest

from neural_trade.indicators import (
    ChannelSpec,
    Indicators,
    ParamSpec,
    compute_reference,
)

FAMILIES = ("atr", "stoch", "willr", "keltner", "obv", "vwap", "mfi", "adx", "cci", "donchian")
TEXTBOOK = {
    "atr": {"period": 14.0},
    "stoch": {"k_period": 14.0, "d_period": 3.0},
    "willr": {"period": 14.0},
    "keltner": {"period": 20.0, "atr_period": 10.0},
    "obv": {"period": 20.0},
    "vwap": {"period": 20.0},
    "mfi": {"period": 14.0},
    "adx": {"period": 14.0},
    "cci": {"period": 20.0},
    "donchian": {"period": 20.0},
}
EPS = 1e-3
SHIFT = -0.5  # the maximal slowing meta shift (META_SCALE * tanh(-1) -> -0.5)


# --------------------------------------------------------------------- synthetic OHLCV
def make_ohlcv(n=400, seed=3, step=0.15, hl=0.2, price_level=0.0):
    """A bounded-range random-walk OHLCV set in target-scaler-like units.

    ``step`` keeps the walk's range within the soft extremum's exact band
    (SOFT_EXTREMUM_CLIP / BETA of the series start), the regime the 60-bar production
    windows live in; ``price_level`` shifts every price series (MFI's tolerance test uses
    a realistic level-to-range ratio)."""
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0.0, step, n)).astype(np.float64)
    close -= close[0]
    high = close + np.abs(rng.normal(0.0, hl, n))
    low = close - np.abs(rng.normal(0.0, hl, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.abs(rng.lognormal(0.0, 0.6, n))
    return {name: (arr + (price_level if name != "volume" else 0.0)).astype(np.float32)
            for name, arr in
            (("open", open_), ("high", high), ("low", low), ("close", close),
             ("volume", volume))}


def make_trend_ohlcv(n=400, slope=0.6, up=21, down=13, spread=0.12):
    """A deterministic zig-zag with DECISIVE one-bar moves (|d close| = slope every bar),
    so the textbook sign branches of OBV / MFI / ADX are unambiguous: the hand-made
    series for the soft-gate tolerance tests (tanh/sigmoid at sharpness 10 are within
    2e-3 of the hard sign at |move| >= 0.6). The volume varies deterministically."""
    leg = np.concatenate([np.full(up, slope), np.full(down, -slope)])
    d_close = np.tile(leg, n // len(leg) + 1)[:n]
    close = np.cumsum(d_close)
    close -= close[0]
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = 1.0 + 0.8 * np.sin(np.arange(n) / 7.0) ** 2
    return {name: arr.astype(np.float32) for name, arr in
            (("open", open_), ("high", high), ("low", low), ("close", close),
             ("volume", volume))}


def make_windows(data, length=60, stride=40):
    """[B, length] windows of every series, each shifted to start at 0 in price (the
    window-relative regime the production layer sees; the soft extremum's exact band is
    around the window's first value)."""
    n = len(data["close"])
    starts = list(range(0, n - length + 1, stride))
    out = {}
    for key, arr in data.items():
        w = np.stack([arr[s:s + length] for s in starts])
        if key != "volume":
            w = w - np.stack([data["close"][s:s + length] for s in starts])[:, :1]
        out[key] = w.astype(np.float32)
    return out


def channels(name, data, periods=None, shift=0.0):
    fam = Indicators.get(name)
    series = {k: v for k, v in data.items() if k != "close"}
    return compute_reference(fam, data["close"], periods or TEXTBOOK[name], shift,
                             series=series)


# --------------------------------------------------------------------- numpy references
def np_ewma(x, period):
    a = 2.0 / (period + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def np_true_range(d):
    h, low, c = (np.asarray(d[k], np.float64) for k in ("high", "low", "close"))
    pc = np.concatenate([[c[0]], c[:-1]])
    return np.maximum(h - low, np.maximum(np.abs(h - pc), np.abs(low - pc)))


def np_roll(x, p, fn):
    return np.array([fn(x[max(0, i - p + 1):i + 1]) for i in range(len(x))])


def np_typical(d):
    return (np.asarray(d["high"], np.float64) + np.asarray(d["low"], np.float64)
            + np.asarray(d["close"], np.float64)) / 3.0


# --------------------------------------------------------------------- criterion 2: contract
@pytest.mark.parametrize("name", FAMILIES)
def test_each_family_entry_declares_the_full_contract(name):
    from neural_trade.indicators import BASE_SERIES, IndicatorFamily

    fam = Indicators.get(name)
    assert isinstance(fam, IndicatorFamily) and fam.name == name
    assert fam.inputs and all(inp in BASE_SERIES for inp in fam.inputs)
    assert len(fam.params) == len(TEXTBOOK[name])
    for p in fam.params:
        assert isinstance(p, ParamSpec)
        assert p.default == TEXTBOOK[name][p.name]  # textbook defaults
        assert p.minimum == 2.0 and p.maximum is None  # None -> the configured ceiling
    assert len(fam.channels) >= 1
    assert all(isinstance(c, ChannelSpec) and c.draw in ("price", "panel") for c in fam.channels)
    assert fam.draw in ("price", "panel")
    m = fam.m_eps(TEXTBOOK[name], EPS, SHIFT)
    assert isinstance(m, int) and m > 0


# --------------------------------------------------------------------- criterion 2: tolerance
# Each test states its textbook reference and its tolerance on the hand-made series above.

def test_atr_equals_the_ewma_of_the_textbook_true_range(tf):
    """ATR = EWMA(TR, 14); the true range itself is exact, so the tolerance is float32
    round-off (atol 1e-4 on values of order 1)."""
    d = make_ohlcv()
    (atr,) = channels("atr", d)
    ref = np_ewma(np_true_range(d), 14.0)
    np.testing.assert_allclose(atr[0], ref, atol=1e-4)


def test_keltner_matches_the_textbook_channel(tf):
    """Middle = EWMA(close, 20), bands +- 2 * EWMA-ATR(10); exact up to float32 round-off."""
    d = make_ohlcv()
    mid, up, lo = channels("keltner", d)
    m = np_ewma(np.asarray(d["close"], np.float64), 20.0)
    a = np_ewma(np_true_range(d), 10.0)
    np.testing.assert_allclose(mid[0], m, atol=1e-4)
    np.testing.assert_allclose(up[0], m + 2 * a, atol=2e-4)
    np.testing.assert_allclose(lo[0], m - 2 * a, atol=2e-4)


def test_cci_matches_the_ewma_textbook_form(tf):
    """CCI over EWMA mean and EWMA mean absolute deviation (period 20); the smooth form is
    exact, tolerance covers float32 round-off in the ratio (rtol 1e-3 after burn-in)."""
    d = make_ohlcv()
    (cci,) = channels("cci", d)
    tp = np_typical(d)
    m = np_ewma(tp, 20.0)
    md = np_ewma(np.abs(tp - m), 20.0)
    ref = (tp - m) / (0.015 * md + 1e-8)
    scale = np.maximum(np.abs(ref[30:]), 1.0)
    np.testing.assert_allclose(cci[0][30:] / scale, ref[30:] / scale, atol=2e-3)


def test_vwap_matches_the_ewma_ratio_form(tf):
    """Rolling VWAP as EWMA(tp * vol, 20) / EWMA(vol, 20); exact up to float32 round-off."""
    d = make_ohlcv()
    (vwap,) = channels("vwap", d)
    tp = np_typical(d)
    v = np.asarray(d["volume"], np.float64)
    ref = np_ewma(tp * v, 20.0) / (np_ewma(v, 20.0) + 1e-8)
    np.testing.assert_allclose(vwap[0], ref, atol=1e-3)


def test_obv_oscillator_matches_the_hard_sign_reference(tf):
    """OBV oscillator (cumsum(sign(dClose) * vol) minus its EWMA(20)) with the soft sign
    tanh(10 dC): on the decisive zig-zag (every move 0.6, |tanh| > 0.997) the oscillator
    stays within 5% of the hard-sign textbook oscillator's scale."""
    d = make_trend_ohlcv()
    (osc,) = channels("obv", d)
    c = np.asarray(d["close"], np.float64)
    v = np.asarray(d["volume"], np.float64)
    dc = np.concatenate([[0.0], np.diff(c)])
    flow = np.sign(dc) * v
    flow[0] = 0.0
    obv = np.cumsum(flow)
    ref = obv - np_ewma(obv, 20.0)
    tol = 0.05 * np.abs(ref[40:]).max()
    assert np.abs(osc[0][40:] - ref[40:]).max() <= tol


def test_mfi_stays_within_2_points_of_textbook_at_a_realistic_price_level(tf):
    """Textbook MFI (positive/negative money flow tp * vol split by sign(dTP), EWMA(14)
    smoothing) on the decisive zig-zag at price level 100 (level >> range, as on the
    reference setup): the volume-only flow and the sigmoid gate keep the smooth MFI
    within 2 points (of 100). On series with many near-zero moves the textbook's hard
    sign is discontinuous and the gap grows - that is the smoothing, not a bug."""
    d = make_trend_ohlcv()
    d = {k: (v + 100.0 if k != "volume" else v).astype(np.float32) for k, v in d.items()}
    (mfi,) = channels("mfi", d)
    tp = np_typical(d)
    v = np.asarray(d["volume"], np.float64)
    dtp = np.concatenate([[0.0], np.diff(tp)])
    mf = tp * v
    pos = np_ewma(np.where(dtp > 0, mf, 0.0), 14.0)
    neg = np_ewma(np.where(dtp < 0, mf, 0.0), 14.0)
    ref = 100.0 * pos / (pos + neg + 1e-8)
    assert np.abs(mfi[0][40:] - ref[40:]).max() <= 2.0


def test_adx_dmi_stays_within_2_points_of_the_textbook_reference(tf):
    """Textbook DMI with EWMA(14) smoothing (+DM/-DM keep the larger move, DIs over the
    smoothed TR, ADX = EWMA(DX)) on the decisive zig-zag: the sigmoid gate keeps
    +DI / -DI / ADX within 2 points (of 100)."""
    d = make_trend_ohlcv()
    pdi, ndi, adx = channels("adx", d)
    h, low = np.asarray(d["high"], np.float64), np.asarray(d["low"], np.float64)
    dh = np.concatenate([[0.0], np.diff(h)])
    dl = np.concatenate([[0.0], -np.diff(low)])
    udm, ddm = np.maximum(dh, 0.0), np.maximum(dl, 0.0)
    pdm = np.where(udm > ddm, udm, 0.0)
    ndm = np.where(ddm > udm, ddm, 0.0)
    tr = np_ewma(np_true_range(d), 14.0)
    rp = 100.0 * np_ewma(pdm, 14.0) / (tr + 1e-8)
    rn = 100.0 * np_ewma(ndm, 14.0) / (tr + 1e-8)
    dx = 100.0 * np.abs(rp - rn) / (rp + rn + 1e-8)
    ra = np_ewma(dx, 14.0)
    assert np.abs(pdi[0][40:] - rp[40:]).max() <= 2.0
    assert np.abs(ndi[0][40:] - rn[40:]).max() <= 2.0
    assert np.abs(adx[0][40:] - ra[40:]).max() <= 2.0


# The extremum tolerances are stated on 60-bar window-relative windows - the regime the
# production layer computes in (windows of Config.LOOKBACK bars, values relative to the
# window; the soft extremum's exact band surrounds each window's first value). The
# textbook rolling extremum is DISCONTINUOUS at the bars where a peak leaves its window,
# so any differentiable form deviates there by up to the size of that drop; the
# tolerances are therefore stated as mean, 95th percentile and a worst-bar cap, all
# measured with headroom over the observed values on this series (seed 3, step 0.2).

def _window_extrema(dw, period):
    hu = np.stack([np_roll(h.astype(np.float64), period, np.max) for h in dw["high"]])
    hl = np.stack([np_roll(lo.astype(np.float64), period, np.min) for lo in dw["low"]])
    return hu, hl


def _extremum_windows():
    return make_windows(make_ohlcv(n=1000, step=0.2))


def test_donchian_tolerance_of_textbook_on_60_bar_windows(tf):
    """Textbook Donchian (hard rolling max(high, 20) / min(low, 20)) per 60-bar window;
    the smooth rolling extremum stays within 20% of the mean channel width on average,
    25% at the 95th percentile and 30% at the worst bar."""
    dw = _extremum_windows()
    up, lo, mid = channels("donchian", dw)
    hu, hl = _window_extrema(dw, 20)
    width = float(np.mean(hu[:, 20:] - hl[:, 20:]))
    err = np.maximum(np.abs(up[:, 20:] - hu[:, 20:]), np.abs(lo[:, 20:] - hl[:, 20:])) / width
    assert err.mean() <= 0.20
    assert np.quantile(err, 0.95) <= 0.25
    assert err.max() <= 0.30
    np.testing.assert_allclose(mid, (up + lo) / 2.0, atol=1e-5)


def test_stochastic_tolerance_of_textbook_on_60_bar_windows(tf):
    """Textbook %K over hard rolling extrema (14) and %D = EWMA(%K, 3) per 60-bar window;
    the smooth extrema keep %K and %D within 12 points (of 100) on average, 27 at the
    95th percentile and 45 at the worst bar."""
    dw = _extremum_windows()
    k, dd = channels("stoch", dw)
    hu, hl = _window_extrema(dw, 14)
    c = dw["close"].astype(np.float64)
    kt = 100.0 * (c - hl) / (hu - hl + 1e-9)
    dt = np.stack([np_ewma(row, 3.0) for row in kt])
    for got, ref in ((k, kt), (dd, dt)):
        err = np.abs(got[:, 20:] - ref[:, 20:])
        assert err.mean() <= 12.0
        assert np.quantile(err, 0.95) <= 27.0
        assert err.max() <= 45.0


def test_williams_r_tolerance_of_textbook_on_60_bar_windows(tf):
    dw = _extremum_windows()
    (w,) = channels("willr", dw)
    hu, hl = _window_extrema(dw, 14)
    ref = -100.0 * (hu - dw["close"].astype(np.float64)) / (hu - hl + 1e-9)
    err = np.abs(w[:, 20:] - ref[:, 20:])
    assert err.mean() <= 12.0
    assert np.quantile(err, 0.95) <= 27.0
    assert err.max() <= 45.0


# --------------------------------------------------------------------- criterion 2: causality
@pytest.mark.parametrize("name", FAMILIES)
def test_changing_later_bars_leaves_channel_values_at_t_unchanged(tf, name):
    """Every channel at bar t depends only on bars up to t: perturbing every series after
    bar t changes nothing at or before t (bit-for-bit)."""
    t = 220
    d = make_ohlcv(n=320, seed=7)
    before = channels(name, d)
    rng = np.random.default_rng(9)
    tampered = {}
    for key, arr in d.items():
        arr2 = arr.copy()
        arr2[t + 1:] = arr2[t + 1:] + rng.normal(5.0, 3.0, len(arr2) - t - 1).astype(np.float32)
        if key == "volume":
            arr2[t + 1:] = np.abs(arr2[t + 1:])
        tampered[key] = arr2
    after = channels(name, tampered)
    for cb, ca in zip(before, after):
        np.testing.assert_array_equal(cb[0, :t + 1], ca[0, :t + 1])


# --------------------------------------------------------------------- criterion 2: M(eps)
def _empirical_offset_invariance(name, periods, n=1400, k=400):
    """Smallest m such that starting the series k bars later changes no channel by more
    than EPS (relative to the channel's own scale) from bar k+m on. The series' range
    stays inside the soft extremum's exact band (make_ohlcv step 0.02), the regime the
    production windows live in."""
    d = make_ohlcv(n=n, seed=11, step=0.02, hl=0.03)
    full = channels(name, d, periods, SHIFT)
    off = channels(name, {key: v[k:] for key, v in d.items()}, periods, SHIFT)
    worst = np.zeros(n - k)
    for cf, co in zip(full, off):
        scale = max(1.0, float(np.abs(cf).max()))
        worst = np.maximum(worst, np.abs(cf[0, k:] - co[0, :]) / scale)
    over = np.nonzero(worst > EPS)[0]
    return int(over[-1]) + 1 if len(over) else 0


@pytest.mark.parametrize("name, periods", [
    ("atr", {"period": 28.0}),                       # the slowest configured instance
    ("stoch", {"k_period": 21.0, "d_period": 5.0}),
    ("willr", {"period": 28.0}),
    ("keltner", {"period": 40.0, "atr_period": 20.0}),
    ("obv", {"period": 40.0}),
    ("vwap", {"period": 40.0}),
    ("mfi", {"period": 28.0}),
    ("adx", {"period": 28.0}),
    ("cci", {"period": 40.0}),
    ("donchian", {"period": 55.0}),
])
def test_m_eps_is_at_least_the_empirical_offset_invariance_value(tf, name, periods):
    fam = Indicators.get(name)
    declared = fam.m_eps(periods, EPS, SHIFT)
    empirical = _empirical_offset_invariance(name, periods)
    assert declared >= empirical, (f"{name}: declared M(eps)={declared} < empirical "
                                   f"offset-invariance {empirical}")

"""Variance-head analytics: the numbers, the noise bands, the baseline and the encoding."""
from __future__ import annotations

import re

import numpy as np
import pytest

from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")
STEPS = (10, 15, 20)


def _frame(y, delta, sigma, *, intervals=None, X_raw=None, split="test"):
    from neural_trade.evaluation.frame import PredictionFrame

    n = len(y)
    scale = 100.0
    return PredictionFrame(
        y=y, last_close=np.full(n, 100_000.0), delta={h: delta[:, i] for i, h in enumerate(H)},
        direction_prob={h: np.full(n, 0.5) for h in H},
        variance_scaled={h: (sigma[:, i] / scale) ** 2 for i, h in enumerate(H)}, pred_scale=scale,
        horizon_steps=STEPS, split=split, intervals=intervals, X_raw=X_raw)


@pytest.fixture(scope="module")
def market():
    """Consecutive 1-minute samples of a random walk with slowly varying volatility, so the targets
    overlap (y at bar t and t+1 share h - 1 bars), as in the real blocks. sigma is the true scale
    times a little noise; the intervals are the served delta +/- 1.645 sigma; X_raw are the 60-bar
    close windows ending at each sample."""
    rng = np.random.default_rng(7)
    n, lookback = 7236, 60
    total = n + lookback + max(STEPS)
    vol = 20.0 * np.exp(0.5 * np.sin(np.arange(total) / 900.0) + 0.3 * rng.normal(size=total).cumsum() / 60)
    close = 100_000 + np.cumsum(vol * rng.normal(size=total))
    t = np.arange(lookback - 1, lookback - 1 + n)          # sample i ends at bar t[i]
    y = np.column_stack([close[t + k] - close[t] for k in STEPS])
    sigma = np.column_stack([vol[t] * np.sqrt(k) * np.exp(0.15 * rng.normal(size=n)) for k in STEPS])
    delta = 0.02 * y + rng.normal(0, 5, (n, 3))
    iv = {h: (delta[:, i] - 1.645 * sigma[:, i], delta[:, i] + 1.645 * sigma[:, i]) for i, h in enumerate(H)}
    windows = np.stack([close[i - lookback + 1:i + 1] for i in t])
    return dict(y=y, delta=delta, sigma=sigma, intervals=iv, windows=windows,
                frame=_frame(y, delta, sigma, intervals=iv, X_raw=windows))


def _row_col(tr):
    xi = int((tr.xaxis or "x")[1:] or 1)
    return (xi - 1) // 3 + 1, (xi - 1) % 3 + 1


def _has_data(tr):
    """A drawn trace (not a legend-only key, whose x is [None])."""
    if tr.x is None:
        return getattr(tr, "x0", None) is not None
    return len(tr.x) > 0 and tr.x[0] is not None


def _colors(tr):
    line, marker = getattr(tr, "line", None), getattr(tr, "marker", None)
    return {getattr(line, "color", None), getattr(marker, "color", None)} - {None}


def _traces(fig, row, col=None, name=None):
    return [t for t in fig.data if _has_data(t) and _row_col(t)[0] == row
            and (col is None or _row_col(t)[1] == col) and (name is None or (t.name or "").startswith(name))]


def _title(fig, row, col):
    return fig.layout.annotations[(row - 1) * 3 + (col - 1)].text


def _fig(frame, **kw):
    from neural_trade.visualization.analytics_variance import variance_analytics_figure

    return variance_analytics_figure(frame, None, **kw)


def _s(v):
    """A signed number as the figure prints it (3 decimals, a real minus sign)."""
    return f"{v:+.3f}".replace("-", "−")


@pytest.fixture(scope="module")
def raw_heads(market):
    return {h: 5 * market["delta"][:, i] for i, h in enumerate(H)}


@pytest.fixture(scope="module")
def plain_fig(market):
    """The notebook call: frame only (windows from frame.X_raw), default 500-sample window."""
    return _fig(market["frame"])


@pytest.fixture(scope="module")
def raw_fig(market, raw_heads):
    return _fig(market["frame"], raw_delta=raw_heads)


# ------------------------------------------------------------------ statistics helpers
def test_lrv_factor_matches_the_overlap_of_a_moving_sum():
    from neural_trade.visualization.analytics_variance import lrv_factor

    rng = np.random.default_rng(0)
    iid = rng.normal(size=40_000)
    assert lrv_factor(iid, 30) == pytest.approx(1.0, abs=0.15)
    ma = np.convolve(rng.normal(size=40_010), np.ones(10), mode="valid")   # a 10-bar moving sum
    # the long-run variance of an h-bar moving sum is h times its variance (Bartlett, 3h lags: ~8.9)
    assert 7.5 < lrv_factor(ma, 30) < 10.5


def test_spearman_difference_interval_is_exact_when_every_resample_is_the_whole_series():
    from scipy.stats import spearmanr

    from neural_trade.visualization.analytics_variance import spearman_diff_ci

    rng = np.random.default_rng(1)
    t = rng.normal(size=500) ** 2
    a, b = t + rng.normal(0, 2, 500), t + rng.normal(0, 1, 500)
    exact = spearmanr(b, t).correlation - spearmanr(a, t).correlation
    lo, hi = spearman_diff_ci(a, b, t, block=500, reps=20)     # one block = a rotation of all samples
    assert lo == pytest.approx(exact, abs=1e-12) and hi == pytest.approx(exact, abs=1e-12)


def test_binned_rms_intervals_widen_with_overlapping_targets(market):
    """The per-bin RMS interval is a block bootstrap: with 20-bar overlapping targets it is well
    wider than the iid delta-method interval (which treats every sample as independent)."""
    from neural_trade.visualization.analytics_variance import binned_rms

    sig, e = market["sigma"][:, 2], market["y"][:, 2] - market["delta"][:, 2]
    t = binned_rms(sig, e, 12, block=60)
    assert np.all((t[:, 4] <= t[:, 3]) & (t[:, 3] <= t[:, 5]))
    order = np.array_split(np.argsort(sig, kind="mergesort"), 12)
    assert np.allclose(t[:, 3], [np.sqrt(np.mean(e[p] ** 2)) for p in order])
    iid = np.array([1.96 * (e[p] ** 2).std() / np.sqrt(len(p)) / (2 * np.sqrt(np.mean(e[p] ** 2))) for p in order])
    assert np.median((t[:, 5] - t[:, 4]) / 2 / iid) > 1.3
    assert np.isnan(binned_rms(sig, e, 12, reps=0)[:, 4:6]).all()


# ------------------------------------------------------------------ finding 40: heavy tails vs scale
def test_heavy_tails_are_reported_as_tails_not_as_a_too_wide_sigma():
    """Student-t(4) errors with the right overall scale: std(z) ~ 1, but |z| > 3 far above 0.27%."""
    rng = np.random.default_rng(3)
    n = 6000
    sigma = rng.uniform(80, 240, (n, 3))
    z = rng.standard_t(4, (n, 3)) / np.sqrt(2.0)          # unit variance, heavy tails
    fig = _fig(_frame(sigma * z, np.zeros((n, 3)), sigma))
    for j in (1, 2, 3):
        std = float(re.search(r"std z ([0-9.]+)", _title(fig, 2, j)).group(1))
        assert abs(std - 1) < 0.1
        # shape with the scale removed: well below the Gaussian 0.80 for a peaked, heavy-tailed error
        shape = float(re.search(r"mean \|z\| ÷ std z ([0-9.]+) \(Gaussian 0.80\)", _title(fig, 2, j)).group(1))
        assert shape < 0.78
        p3 = float(re.search(r"\|z\| > 3: ([0-9.]+)%", _title(fig, 3, j)).group(1))
        assert p3 > 0.27 * 2
        tail = _traces(fig, 3, j, "observed")[0]
        k3 = list(np.round(tail.x, 2)).index(3.0)
        assert tail.y[k3] > 2                                # the tail row shows the excess directly
    sub = fig.layout.title.text
    assert "hump in the middle = sigma too wide" not in sub
    assert "tails heavier than a Gaussian, not a too-wide" in sub


def test_summary_numbers_match_a_direct_computation(market):
    from scipy.stats import kurtosis

    from neural_trade.visualization.analytics_variance import variance_summary

    fr = market["frame"]
    s = variance_summary(fr)
    for i, h in enumerate(H):
        e = market["y"][:, i] - market["delta"][:, i]
        z = e / market["sigma"][:, i]
        lo, hi = market["intervals"][h]
        assert s[h]["std_z"] == pytest.approx(z.std())
        assert s[h]["mean_abs_z_over_std"] == pytest.approx(np.abs(z).mean() / z.std())
        assert s[h]["excess_kurtosis"] == pytest.approx(kurtosis(z))
        assert s[h]["p_z3"] == pytest.approx(np.mean(np.abs(z) > 3))
        assert s[h]["cov_conf"] == pytest.approx(np.mean((market["y"][:, i] >= lo) & (market["y"][:, i] <= hi)))
        assert s[h]["width_gauss"] == pytest.approx(np.mean(2 * 1.6448536 * market["sigma"][:, i]), rel=1e-6)


# ------------------------------------------------------------------ finding 41: realised-vol baseline
def test_realised_vol_baseline_is_drawn_and_scored_from_the_input_windows(market, plain_fig):
    from scipy.stats import spearmanr

    from neural_trade.calibration.conformal import interval_scale

    fr, fig = market["frame"], plain_fig                    # windows default to frame.X_raw
    u = interval_scale("realized_vol", windows=market["windows"], horizon_steps=STEPS)
    for j, h in enumerate(H, start=1):
        e = market["y"][:, j - 1] - market["delta"][:, j - 1]
        rho = spearmanr(u[h], e ** 2).correlation
        head = spearmanr(market["sigma"][:, j - 1], e ** 2).correlation
        title = _title(fig, 1, j)
        assert f"60-bar vol {_s(rho)}" in title
        assert f"head {_s(head)}" in title
        assert "95% CI" in title
        base = _traces(fig, 1, j, "baseline")
        assert len(base) == 1 and base[0].line.color == T.NEUTRAL
    # explicit windows win over X_raw; rv_bars keeps the last bars of each window only
    fig2 = _fig(fr, windows=market["windows"], rv_bars=20)
    u20 = interval_scale("realized_vol", windows=market["windows"][:, -20:], horizon_steps=STEPS)["h0"]
    e0 = market["y"][:, 0] - market["delta"][:, 0]
    assert f"20-bar vol {_s(spearmanr(u20, e0 ** 2).correlation)}" in _title(fig2, 1, 1)


def test_no_baseline_without_windows(viz_frame):
    fig = _fig(viz_frame)
    assert not any((t.name or "").startswith("baseline") for t in fig.data)
    assert "no realised-vol baseline" in _title(fig, 1, 1)


def test_windows_of_the_wrong_length_are_rejected(market):
    with pytest.raises(ValueError):
        _fig(market["frame"], windows=market["windows"][:100])


# ------------------------------------------------------------------ finding 42: noise bands
def test_coverage_band_accounts_for_overlapping_targets(market, plain_fig):
    from neural_trade.visualization.analytics_variance import lrv_factor

    fig = plain_fig
    rects = {s.yref: s for s in fig.layout.shapes if s.type == "rect"}
    for j, (h, k) in enumerate(zip(H, STEPS), start=1):
        band = rects[f"y{9 + j}"]                         # row 4
        lo, hi = market["intervals"][h]
        y = market["y"][:, j - 1]
        inside = (y >= lo) & (y <= hi)
        gauss = np.abs(y - market["delta"][:, j - 1]) <= 1.6448536269514722 * market["sigma"][:, j - 1]
        # one band for both lines: the wider of the two indicators' overlap factors
        vif = max(lrv_factor(inside, 3 * k), lrv_factor(gauss, 3 * k))
        half = 1.96 * np.sqrt(0.09 * vif / 500)
        assert band.y0 == pytest.approx(0.9 - half)
        assert half > 1.5 * 1.96 * np.sqrt(0.09 / 500)       # wider than for independent samples
        edges = _traces(fig, 2, j, "noise range")            # the PIT row has its band too (as edges)
        assert len(edges) == 1
        pts = _traces(fig, 1, j, "RMS error per")[0]
        assert pts.error_y.array is not None and np.all(np.asarray(pts.error_y.array) > 0)
    assert "trailing 500 samples (1-min bars, ~8 h)" in fig.layout.legend4.title.text


# ------------------------------------------------------------------ finding 107: which mean
def test_errors_are_labelled_as_served_and_raw_heads_are_an_extra_view(market, raw_heads, raw_fig, plain_fig):
    raw, fig = raw_heads, raw_fig
    names = {t.name for t in fig.data}
    assert not any("raw Gaussian" in (n or "") for n in names)
    assert "served delta" in fig.layout.xaxis4.title.text and "served delta" in fig.layout.yaxis.title.text
    assert any("around served delta" in (n or "") for n in names)
    sig, y = market["sigma"][:, 1], market["y"][:, 1]
    tr = _traces(fig, 1, 2, "RMS error around the raw")[0]
    order = np.array_split(np.argsort(sig, kind="mergesort"), 12)
    assert np.allclose(tr.y, [np.sqrt(np.mean((y[p] - raw["h1"][p]) ** 2)) for p in order])
    # the served-mean metrics do not move when raw_delta is given (they match eval_report)
    assert [a.text for a in fig.layout.annotations[:15]] == [a.text for a in plain_fig.layout.annotations[:15]]


# ------------------------------------------------------------------ finding 108: overall numbers + width
def test_overall_coverage_and_width_are_stated_and_width_is_drawn(market, plain_fig):
    fig = plain_fig
    for j, h in enumerate(H, start=1):
        lo, hi = market["intervals"][h]
        y = market["y"][:, j - 1]
        cov = np.mean((y >= lo) & (y <= hi))
        assert f"conformal {cov:.3f}" in _title(fig, 4, j)
        assert f"conformal ${np.mean(hi - lo):,.0f}" in _title(fig, 5, j)
        wtr = _traces(fig, 5, j, "conformal width")[0]
        roll = np.convolve(hi - lo, np.ones(500) / 500, mode="valid")
        step = wtr.dx
        assert wtr.x0 == 499 and np.allclose(wtr.y, roll[::step], rtol=1e-5)


# ------------------------------------------------------------------ findings 109 / 110: encoding
def test_no_misleading_mean_abs_error_line_and_dotted_only_for_references(raw_fig):
    fig = raw_fig
    assert not any((t.name or "").startswith("mean |error|") for t in fig.data)
    horizon_colors = set(T.HORIZON_COLORS.values())
    for t in fig.data:
        if getattr(getattr(t, "line", None), "dash", None) == "dot":
            assert t.line.color == T.NEUTRAL, t.name          # dotted = a grey reference, never a series
        if not _has_data(t):                                  # legend keys show the style in neutral ink
            assert not _colors(t) & horizon_colors, t.name
        else:                                                 # a horizon colour only in its own column
            assert _colors(t) & horizon_colors <= {T.HORIZON_COLORS[H[_row_col(t)[1] - 1]]}, t.name


# ------------------------------------------------------------------ finding 111: axes
def test_every_row_has_a_y_title_and_rows_2_to_4_share_a_scale(plain_fig):
    fig = plain_fig
    for r in range(1, 6):
        assert fig.layout[f"yaxis{'' if r == 1 else (r - 1) * 3 + 1}"].title.text
    for r in (2, 4):
        first = (r - 1) * 3 + 1
        for c in (2, 3):
            assert fig.layout[f"yaxis{first + c - 1}"].matches == f"y{first}"
        assert fig.layout[f"yaxis{first}"].range is None       # autorange (row 4 once broke: see below)
    # row 3: one fixed log range on all three axes, and no `matches` (plotly.js 3.5 switches off the
    # row-4 autorange when a matched group has a fixed range)
    rng = [tuple(fig.layout[f"yaxis{a}"].range) for a in (7, 8, 9)]
    assert rng[0] == rng[1] == rng[2]
    assert all(fig.layout[f"yaxis{a}"].matches is None for a in (7, 8, 9))
    assert all(fig.layout[f"yaxis{a}"].type == "log" for a in (7, 8, 9))


def test_tail_axis_has_a_tick_below_one_and_every_point_inside_its_range(market, plain_fig):
    """The sub-Gaussian rates at k ~ 1 (a peaked centre) must read off a tick below 1x."""
    fig = plain_fig
    ax = fig.layout.yaxis7
    lo, hi = (10 ** v for v in ax.range)
    ticks = list(ax.tickvals)
    assert 1 in ticks and min(ticks) < 1 and all(lo <= t <= hi for t in ticks)
    assert list(ax.ticktext) == [f"{t:g}×" for t in ticks]
    for j in (1, 2, 3):
        tail = _traces(fig, 3, j, "observed")[0]
        ok = np.isfinite(np.asarray(tail.y, float))
        top = np.asarray(tail.y, float)[ok] + np.asarray(tail.error_y.array)[ok]
        bot = np.asarray(tail.y, float)[ok] - np.asarray(tail.error_y.arrayminus)[ok]
        assert top.max() <= hi and bot[bot > 0].min() >= lo      # no clipped point or interval


def test_log_ticks_never_overprint_on_a_short_panel():
    from neural_trade.visualization.analytics_variance import _log_ticks

    tall, short = _log_ticks(0.43, 26, 220), _log_ticks(0.43, 26, 120)
    assert 0.75 in tall and 0.75 not in short and 0.5 in short and 1 in short
    for ticks, px in ((tall, 220), (short, 120)):
        per_decade = px / np.log10(26 / 0.43)
        assert np.all(np.diff(np.log10(ticks)) * per_decade >= 14 - 1e-9)


# ------------------------------------------------------------------ finding 113: size
def test_figure_stays_within_the_size_budget(raw_fig):
    import plotly.io as pio

    fig = raw_fig
    assert len(pio.to_json(fig)) < 600_000
    for t in fig.data:
        assert t.type != "histogram"                          # PIT pre-binned, not 7k raw values
        if t.y is not None:
            assert len(t.y) <= 1500


# ------------------------------------------------------------------ finding 114: which block
def test_subtitle_names_the_block_and_the_sample_count(market, plain_fig, viz_frame):
    import copy

    assert f"test block, {len(market['frame']):,} samples" in plain_fig.layout.title.text
    cal = copy.copy(viz_frame)
    cal.split = "cal"
    assert "by construction" in _fig(cal).layout.title.text


# ------------------------------------------------------------------ layout: height and width
@pytest.mark.parametrize("height", [None, 900, 1300, 1500, 2600])
def test_row_headings_keep_a_fixed_pixel_gap_whatever_the_height(market, height):
    """Each row's heading (legend title + keys) and the two-line panel titles sit in a gap fixed in
    pixels, so a caller's height can never put a heading on the previous row's x-axis title."""
    from neural_trade.visualization import analytics_variance as AV

    fig = _fig(market["frame"], height=height)
    H_px = fig.layout.height
    assert H_px == max(height or AV.DEFAULT_HEIGHT, AV.MIN_HEIGHT)
    plot_h = H_px - fig.layout.margin.t - fig.layout.margin.b
    doms = [fig.layout[f"yaxis{'' if r == 1 else (r - 1) * 3 + 1}"].domain for r in range(1, 6)]
    tops = [fig.layout[lid].y for lid in ("legend", "legend2", "legend3", "legend4")]
    tops.append(next(a.y for a in fig.layout.annotations if (a.text or "").startswith("<b>5 ·")))
    for r in range(1, 6):
        lift = (tops[r - 1] - doms[r - 1][1]) * plot_h            # heading bottom above the panel top
        assert lift == pytest.approx(AV._TITLES_PX + 2, abs=0.01)  # room for the two-line panel titles
        if r > 1:
            gap = (doms[r - 2][0] - doms[r - 1][1]) * plot_h        # previous panel bottom to this panel top
            assert gap == pytest.approx(AV._GAP_PX, abs=0.01)
            free = gap - lift - AV._HEAD_PX                         # left for ticks + x title below the row above
            assert free >= AV._AX_BELOW_PX
    for r in range(1, 6):                                          # no panel squeezed below 120 px
        assert (doms[r - 1][1] - doms[r - 1][0]) * plot_h > 120
    assert fig.layout.margin.t >= AV._HEAD_PX + AV._TITLES_PX + 16 * 4


def _plain(text):
    return re.sub(r"<[^>]+>", "", text)


def test_text_fits_a_1000_px_output(market, raw_heads):
    """Proxy for the rendered width (checked on screenshots at 1000 and 1500 px): the panel-title lines,
    subtitle lines and each row's keys stay within a character budget that fits 1000 px."""
    import copy

    fig = _fig(market["frame"], raw_delta=raw_heads)
    cal = copy.copy(market["frame"])
    cal.split = "cal"
    for f in (fig, _fig(cal, raw_delta=raw_heads)):
        for a in f.layout.annotations[:15]:
            first, second = (_plain(s) for s in a.text.split("<br>"))
            assert len(first) <= 50, first                 # 12 px, a 1000 px output leaves ~320 px a column
            assert len(second) <= 54, second               # 11 px
        for line in _plain(f.layout.title.text.replace("<br>", "\n")).split("\n")[1:]:
            assert len(line) <= 110, line
    # keys of one row: ~5.8 px a character at 11 px + ~40 px of swatch and gap each, within the
    # 912 px plot width of a 1000 px output
    for lid in ("legend", "legend2", "legend3", "legend4"):
        names = [t.name for t in fig.data if t.legend == lid and not _has_data(t)]
        assert sum(5.8 * len(nm) + 40 for nm in names) <= 912, (lid, names)
        assert fig.layout[lid].title.side == "top"          # heading on its own line: plotly wraps
                                                             # side-left headings wrongly and cuts keys


# ------------------------------------------------------------------ row 2: the band does not tint the bars
def test_pit_band_is_drawn_as_edges_and_its_key_matches(market, plain_fig):
    from scipy.stats import norm

    from neural_trade.visualization.analytics_variance import lrv_factor

    fig = plain_fig
    assert not any(s.layer == "above" for s in fig.layout.shapes)
    assert not any(s.yref in ("y4", "y5", "y6") for s in fig.layout.shapes)
    for j, k in enumerate(STEPS, start=1):
        e = market["y"][:, j - 1] - market["delta"][:, j - 1]
        pit = norm.cdf(e / market["sigma"][:, j - 1])
        edges = np.linspace(0, 1, 21)
        vif = np.mean([lrv_factor((pit >= a) & (pit < b), 3 * k) for a, b in zip(edges[:-1], edges[1:])])
        hw = 1.96 * np.sqrt(0.05 * 0.95 * vif / len(pit)) / 0.05
        tr = _traces(fig, 2, j, "noise range")[0]
        assert None in tr.y                                  # two separate edges, nothing filled
        assert [v for v in tr.y if v is not None] == pytest.approx([1 - hw, 1 - hw, 1 + hw, 1 + hw])
        assert tr.line.color == T.NEUTRAL and tr.line.width <= 1.25
        bars = [t for t in _traces(fig, 2, j) if t.type == "bar"]
        assert len(bars) == 1 and bars[0].marker.color == T.HORIZON_COLORS[H[j - 1]]
    key = next(t for t in fig.data if t.legendgroup == "pit-band" and not _has_data(t))
    drawn = _traces(fig, 2, 1, "noise range")[0]
    assert (key.line.dash, key.line.color, key.line.width) == (drawn.line.dash, drawn.line.color, drawn.line.width)
    # keys drawn at the traces' own widths (plotly's "constant" sizing makes every key line 5 px wide)
    assert all(fig.layout[lid].itemsizing == "trace" for lid in ("legend", "legend2", "legend3", "legend4"))


# ------------------------------------------------------------------ general readability
def test_no_empty_panel_with_and_without_intervals(plain_fig, viz_frame):
    import copy

    bare = copy.copy(viz_frame)
    bare.intervals = None
    for fig in (plain_fig, _fig(viz_frame), _fig(bare)):
        assert T.empty_panels(fig) == []
        assert len({_row_col(t) for t in fig.data if _has_data(t)}) == 15


def test_registry_passes_the_new_options_through(viz_frame):
    from neural_trade.registries.visualizations import Visualizations

    rng = np.random.default_rng(5)
    windows = 100_000 + np.cumsum(rng.normal(0, 20, (len(viz_frame), 60)), axis=1)
    fig = Visualizations.build("variance_analytics", viz_frame, None, raw_delta=viz_frame.delta, windows=windows)
    assert any((t.name or "").startswith("RMS error around the raw") for t in fig.data)
    assert any((t.name or "").startswith("baseline") for t in fig.data)

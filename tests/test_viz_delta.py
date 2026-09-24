"""Price-head (delta) analytics: numbers, noise bands and readability of delta_analytics_figure."""
from __future__ import annotations

import numpy as np
import pytest

H = ("h0", "h1", "h2")
STEPS = (10, 15, 20)


# ------------------------------------------------------------------ synthetic blocks
def _overlap_frame(n=4000, seed=0, beta=(0.2, 0.05, 0.3), edge=0.15):
    """A block like the real one: targets are sums of the next h one-bar moves (overlapping), the raw
    heads are persistent and weakly informative, and the served delta is beta x raw."""
    from neural_trade.evaluation.frame import PredictionFrame

    rng = np.random.default_rng(seed)
    inc = rng.normal(0, 30, n + max(STEPS) + 400)
    y = np.stack([np.convolve(inc, np.ones(s), "valid")[1:n + 1] for s in STEPS], 1)
    raw, served = {}, {}
    for i, h in enumerate(H):
        noise = np.convolve(rng.normal(0, 1, n + 400), np.ones(8) / 8, "valid")[:n] * 60 * np.sqrt(8)
        raw[h] = noise + edge * y[:, i]
        served[h] = beta[i] * raw[h]
    p = {h: 1 / (1 + np.exp(-raw[h] / 80)) for h in H}
    frame = PredictionFrame(y=y, last_close=np.full(n, 100_000.0), delta=served, direction_prob=p,
                            variance_scaled={h: np.full(n, 1.0) for h in H}, pred_scale=100.0,
                            horizon_steps=STEPS, direction_prob_calibrated=p)
    return frame, raw


@pytest.fixture(scope="module")
def block():
    return _overlap_frame()


@pytest.fixture(scope="module")
def fig_block(block, viz_config):
    from neural_trade.visualization.model_analytics import delta_analytics_figure

    frame, raw = block
    return delta_analytics_figure(frame, viz_config, raw_delta=raw)


def _axes(fig, row, col):
    sp = fig.get_subplot(row, col)
    return sp.xaxis.plotly_name.replace("axis", ""), sp.yaxis.plotly_name.replace("axis", "")


def _traces(fig, row, col):
    xa, ya = _axes(fig, row, col)
    return [t for t in fig.data if t.type != "table" and (t.xaxis or "x") == xa and (t.yaxis or "y") == ya]


def _named(fig, row, col, prefix):
    return [t for t in _traces(fig, row, col) if (t.name or "").startswith(prefix) and t.x is not None
            and len(t.x) and t.x[0] is not None]


def _titles(fig):
    return [a.text for a in fig.layout.annotations]


def _table(fig):
    (tab,) = [t for t in fig.data if t.type == "table"]
    return tab


def _row(fig, prefix):
    """(cells h0..h2, colours h0..h2) of the table row whose label starts with ``prefix``."""
    tab = _table(fig)
    labels = list(tab.cells.values[0])
    (i,) = [k for k, lab in enumerate(labels) if lab.startswith(prefix)]
    return [tab.cells.values[c][i] for c in (1, 2, 3)], [tab.cells.font.color[c][i] for c in (1, 2, 3)]


# ------------------------------------------------------------------ what is plotted
def test_delta_analytics_uses_the_raw_heads_when_given(viz_frame, viz_config):
    from neural_trade.visualization.model_analytics import delta_analytics_figure

    raw = {h: 2 * viz_frame.delta[h] for h in H}
    fig = delta_analytics_figure(viz_frame, viz_config, raw_delta=raw)
    (inside,) = _named(fig, 2, 1, "sample")
    beyond = _named(fig, 2, 1, "beyond")
    xs = np.r_[np.asarray(inside.x, float), np.asarray(beyond[0].customdata, float)[:, 0] if beyond else []]
    assert np.allclose(np.sort(xs), np.sort(raw["h0"]), rtol=1e-5, atol=1e-3)


def test_points_beyond_the_range_sit_in_a_shaded_margin_inside_the_axes(fig_block, block):
    """The clip level is marked (the inner edge of a shaded margin on each side) and a clipped point is
    drawn in the middle of that margin: never on the axis edge, where half its marker would be cut."""
    from neural_trade.visualization.analytics_delta import EDGE_X, EDGE_Y, scatter_box

    frame, raw = block
    shapes = list(fig_block.layout.shapes or ())
    for j, h in enumerate(H, start=1):
        d, y = raw[h], frame.y[:, j - 1]
        (xr, yr) = scatter_box(d, y)
        xa, ya = _axes(fig_block, 2, j)
        ax_x = fig_block.layout["xaxis" + xa[1:]].range
        ax_y = fig_block.layout["yaxis" + ya[1:]].range
        mx, my = EDGE_X * (xr[1] - xr[0]), EDGE_Y * (yr[1] - yr[0])
        assert ax_x == pytest.approx([xr[0] - mx, xr[1] + mx]) and ax_y[0] == pytest.approx(yr[0] - my)
        assert ax_y[1] > yr[1] + my                                  # the note's headroom stays above
        rects = [s for s in shapes if s.type == "rect" and s.xref == xa and s.yref == ya]
        assert len(rects) == 4 and all(s.fillcolor and s.layer == "below" for s in rects)
        edges = {round(v, 6) for s in rects for v in (s.x0, s.x1, s.y0, s.y1)}
        assert {round(v, 6) for v in (*xr, *yr)} <= edges           # the clip levels are marked
        (beyond,) = _named(fig_block, 2, j, "beyond")
        real = np.asarray(beyond.customdata, float)
        bx, by = np.asarray(beyond.x, float), np.asarray(beyond.y, float)
        for v, drawn, (lo, hi), m in ((real[:, 0], bx, xr, mx), (real[:, 1], by, yr, my)):
            assert np.allclose(drawn[v < lo], lo - m / 2, rtol=1e-5) and np.allclose(drawn[v > hi], hi + m / 2, rtol=1e-5)
            inside = (v >= lo) & (v <= hi)
            assert np.allclose(drawn[inside], v[inside], rtol=1e-5, atol=1e-2)
        assert ((real[:, 1] < yr[0]).any() or (real[:, 1] > yr[1]).any())   # this block clips in y too


def test_every_sample_is_drawn_once_and_inside_the_axes(fig_block, block):
    """Nothing is silently clipped: points beyond the plotted range sit at its edge, with their real
    values in the hover, and every sample appears exactly once."""
    frame, raw = block
    for j, h in enumerate(H, start=1):
        (inside,) = _named(fig_block, 2, j, "sample")
        (beyond,) = _named(fig_block, 2, j, "beyond")
        assert len(inside.x) + len(beyond.x) == len(frame)
        xa, ya = _axes(fig_block, 2, j)
        xr = fig_block.layout["xaxis" + xa[1:]].range
        yr = fig_block.layout["yaxis" + ya[1:]].range
        for t in (inside, beyond):
            assert xr[0] <= np.min(t.x) and np.max(t.x) <= xr[1]
            assert yr[0] <= np.min(t.y) and np.max(t.y) <= yr[1]
        real = np.asarray(beyond.customdata, float)
        assert np.allclose(np.sort(np.r_[inside.x, real[:, 0]]), np.sort(raw[h]), rtol=1e-5, atol=1e-2)


def test_scatter_x_axis_spans_the_predictions_not_the_outcomes(fig_block, block):
    """The old zero-line trace ran over +/- the realised range and squeezed the cloud into a sliver."""
    frame, raw = block
    for j, h in enumerate(H, start=1):
        xa, _ = _axes(fig_block, 2, j)
        lo, hi = fig_block.layout["xaxis" + xa[1:]].range
        p_lo, p_hi = np.percentile(raw[h], [0.5, 99.5])
        span = p_hi - p_lo
        assert p_lo - 0.1 * span <= lo <= min(p_lo, 0) and max(p_hi, 0) <= hi <= p_hi + 0.1 * span
        assert hi < 0.8 * np.percentile(np.abs(frame.y[:, j - 1]), 99.5)


def test_scatter_x_range_is_fenced_against_a_burst_of_extreme_predictions():
    from neural_trade.visualization.analytics_delta import scatter_x_range

    rng = np.random.default_rng(3)
    d = np.r_[rng.normal(0, 50, 2000), np.full(15, 3000.0)]      # one 15-sample burst = top 0.75%
    lo, hi = scatter_x_range(d)
    assert hi < 600 and lo > -600
    g = rng.normal(0, 50, 20000)                                   # the fence never binds on a Gaussian
    assert scatter_x_range(g) == pytest.approx(tuple(np.percentile(g, [0.5, 99.5])))


def test_served_beta_is_derived_and_drawn(block, fig_block):
    """served = beta x raw: beta is shown in the table and titles and drawn as y = beta x."""
    frame, raw = block
    beta = {h: float(np.dot(frame.delta[h], raw[h]) / np.dot(raw[h], raw[h])) for h in H}
    assert beta == pytest.approx({"h0": 0.2, "h1": 0.05, "h2": 0.3})
    assert _row(fig_block, "served β")[0] == [f"{beta[h]:.3f}" for h in H]
    for j, h in enumerate(H, start=1):
        (line,) = _named(fig_block, 3, j, "served β × raw")
        assert np.allclose(np.asarray(line.y) / np.asarray(line.x), beta[h], rtol=1e-5)
        assert any(f"served β {beta[h]:.3f}" in t for t in _titles(fig_block))


def test_no_served_rolling_correlation_duplicate(fig_block):
    """corr(beta x raw, y) == corr(raw, y): the old figure drew the served copy under the raw line."""
    assert not any((t.name or "").startswith("served (shrunk)") for t in fig_block.data)
    for row in (4, 5):
        for j in (1, 2, 3):
            ys = [np.asarray(t.y, float) for t in _traces(fig_block, row, j)
                  if t.y is not None and len(t.y) > 10]
            for a in range(len(ys)):
                for b in range(a + 1, len(ys)):
                    assert not (len(ys[a]) == len(ys[b]) and np.allclose(ys[a], ys[b], equal_nan=True))


def test_titles_name_the_slope_through_zero_and_give_rank_correlation(fig_block, block):
    from scipy.stats import spearmanr

    frame, raw = block
    text = " | ".join(_titles(fig_block))
    assert "LS slope" not in text
    for i, h in enumerate(H):
        y = frame.y[:, i]
        slope = np.dot(y, raw[h]) / np.dot(raw[h], raw[h])
        assert f"fit {slope:+.2f} · w/o top 0.5%" in text
        corr, rank = np.corrcoef(raw[h], y)[0, 1], spearmanr(raw[h], y).correlation
        assert f"corr {corr:+.3f}, rank {rank:+.3f} (noise ±" in text
    assert "slope through 0" in fig_block.layout.title.text
    assert any(t.name == "fit through 0" and t.showlegend for t in fig_block.data if t.type != "table")


def test_trimmed_fit_drops_the_top_half_percent(fig_block, block):
    frame, raw = block
    for j, h in enumerate(H, start=1):
        d, y = raw[h], frame.y[:, j - 1]
        keep = np.abs(d) <= np.percentile(np.abs(d), 99.5)
        (line,) = _named(fig_block, 3, j, "fit w/o top 0.5%")
        assert np.allclose(np.asarray(line.y) / np.asarray(line.x), np.dot(y[keep], d[keep]) / np.dot(d[keep], d[keep]),
                           rtol=1e-4)


# ------------------------------------------------------------------ noise
def test_binned_ci_is_overlap_adjusted():
    """Deciles of a slowly moving prediction are runs of neighbouring samples whose h-bar targets
    overlap: the interval must widen by about sqrt(h), not stay at the iid width."""
    from neural_trade.visualization.analytics_delta import binned_means

    rng = np.random.default_rng(1)
    s = 10
    inc = rng.normal(0, 1, 20_000 + s)
    y = np.convolve(inc, np.ones(s), "valid")[1:20_001]
    x = np.arange(len(y), dtype=float)                           # every decile is one contiguous run
    iid = binned_means(x, y, 10)
    adj = binned_means(x, y, 10, steps=s)
    ratio = (adj[:, 3] - adj[:, 2]) / (iid[:, 3] - iid[:, 2])
    assert 2.0 < np.median(ratio) < 3.4                          # sqrt(10) = 3.16 for a fully contiguous run
    assert np.all(adj[:, 5] < adj[:, 4] / 3)                      # n_eff reported per bin


def test_figure_uses_the_overlap_adjusted_ci(fig_block, block):
    from neural_trade.visualization.analytics_delta import binned_means

    frame, raw = block
    for j, h in enumerate(H, start=1):
        (tr,) = _named(fig_block, 3, j, "decile mean")
        t = binned_means(raw[h], frame.y[:, j - 1], 10, steps=STEPS[j - 1])
        iid = binned_means(raw[h], frame.y[:, j - 1], 10)
        assert np.allclose(tr.error_y.array, t[:, 3] - t[:, 1], rtol=1e-4)
        assert np.median(np.asarray(tr.error_y.array) / (iid[:, 3] - iid[:, 1])) > 1.4


def test_correlation_band_matches_a_circular_shift_null():
    """The Bartlett band (effective samples N / deff) holds the no-skill correlation 95% of the time."""
    from neural_trade.visualization.analytics_delta import corr_band, design_effect

    frame, raw = _overlap_frame(n=6000, seed=5, edge=0.0)          # no relationship at all
    rng = np.random.default_rng(0)
    for i, h in enumerate(H):
        d, y = raw[h], frame.y[:, i]
        band = corr_band(len(y), design_effect(d, y, STEPS[i]))
        null = [np.corrcoef(d, np.roll(y, k))[0, 1] for k in rng.integers(500, len(y) - 500, 300)]
        assert np.percentile(np.abs(null), 95) == pytest.approx(band, rel=0.25)
        assert band > 1.96 / np.sqrt(len(y)) * 1.5                 # far wider than the iid band


def test_served_skill_band_keeps_95_percent_coverage_across_volatility_regimes():
    from neural_trade.visualization.analytics_common import _rolling_mean
    from neural_trade.visualization.analytics_delta import design_effect, served_skill_band

    rng = np.random.default_rng(7)
    n, s, w = 8000, 10, 400
    vol = np.where(np.arange(n + s) < (n + s) // 2, 10.0, 40.0)   # a quiet half, then a volatile half
    y = np.convolve(rng.normal(0, 1, n + s) * vol, np.ones(s), "valid")[1:n + 1]
    served = 0.3 * np.convolve(rng.normal(0, 30, n + 7), np.ones(8) / 8, "valid")[:n]
    deff = design_effect(served, y, s)
    out = []
    for k in rng.integers(w, n - w, 60):
        yy = np.roll(y, k)
        lo, hi = served_skill_band(served, yy, w, deff)
        _, mse = _rolling_mean((yy - served) ** 2, w)
        _, my2 = _rolling_mean(yy ** 2, w)
        sk = 1 - mse / my2
        out.append(np.mean((sk < lo) | (sk > hi)))
    assert 0.02 < np.mean(out) < 0.09


def test_rolling_rows_carry_a_no_skill_band_and_the_block_value(fig_block, block):
    from neural_trade.visualization.analytics_delta import corr_band, design_effect

    frame, raw = block
    for j, h in enumerate(H, start=1):
        band = [t for t in _traces(fig_block, 4, j) if t.fill == "tonexty"]
        assert len(band) == 1
        d, y = raw[h], frame.y[:, j - 1]
        assert max(band[0].y) == pytest.approx(corr_band(500, design_effect(d, y, STEPS[j - 1])), rel=1e-5)
        (blk,) = [t for t in _traces(fig_block, 4, j) if t.name == "whole block" and t.x[0] is not None]
        assert blk.y[0] == pytest.approx(np.corrcoef(d, y)[0, 1], abs=1e-9)
        assert any(t.fill == "tonexty" for t in _traces(fig_block, 5, j))
        assert _named(fig_block, 5, j, "served delta")


# ------------------------------------------------------------------ the table
def _block_bootstrap_skill_sd(y, p, steps, reps=1000, seed=1):
    """Moving-block bootstrap (blocks of 4 x steps samples) of the skill vs predicting 0."""
    rng = np.random.default_rng(seed)
    n, blk = len(y), 4 * steps
    y2, loss = y * y, p * p - 2 * y * p
    starts = rng.integers(0, n - blk + 1, (reps, int(np.ceil(n / blk))))
    idx = (starts[:, :, None] + np.arange(blk)).reshape(reps, -1)[:, :n]
    return np.std(-loss[idx].mean(1) / y2[idx].mean(1))


def test_skill_interval_matches_a_block_bootstrap_on_heavy_tailed_predictions():
    """A head with fat tails and a few bursts of extreme predictions: E[pred^2] is noisy, and the
    interval must carry that noise (the old 2 E[y pred] term alone was about 3x too narrow)."""
    from neural_trade.visualization.analytics_delta import design_effect, skill_half_width

    rng = np.random.default_rng(0)
    n, s = 6000, 10
    y = np.convolve(rng.normal(0, 30, n + s), np.ones(s), "valid")[1:n + 1]
    p = np.convolve(rng.standard_t(2.5, n + 7), np.ones(8) / 8, "valid")[:n] * 40 + 0.1 * y
    for st in rng.integers(0, n - 20, 5):
        p[st:st + 15] += rng.choice([-1, 1]) * rng.uniform(300, 600)
    hw = skill_half_width(y, p, steps=s)
    assert hw == pytest.approx(1.96 * _block_bootstrap_skill_sd(y, p, s), rel=0.2)
    old = 2 * 1.96 * np.sqrt(np.mean(p * p) / np.mean(y * y)) * np.sqrt(design_effect(p, y, s) / n)
    assert hw > 2 * old
    # a light-tailed, weakly informative head: still the bootstrap width, and far wider than iid
    q = 0.3 * np.convolve(rng.normal(0, 30, n + 7), np.ones(8) / 8, "valid")[:n] + 0.02 * y
    assert skill_half_width(y, q, steps=s) == pytest.approx(1.96 * _block_bootstrap_skill_sd(y, q, s), rel=0.2)
    assert skill_half_width(y, q, steps=s) > 1.5 * skill_half_width(y, q, steps=1)


def test_skill_intervals_in_the_table_and_row_titles(fig_block, block):
    from neural_trade.visualization.analytics_delta import skill_half_width

    frame, raw = block
    raw_row, srv_row = _row(fig_block, "skill vs predicting 0: raw")[0], _row(fig_block, "skill vs predicting 0: served")[0]
    titles = " | ".join(_titles(fig_block))
    for i, h in enumerate(H):
        y = frame.y[:, i]
        hw_raw = skill_half_width(y, raw[h], steps=STEPS[i])
        hw_srv = skill_half_width(y, frame.delta[h], steps=STEPS[i])
        assert f"± {hw_raw:.4f}" in raw_row[i] and f"± {hw_srv:.4f}" in srv_row[i]
        skill = 1 - np.mean((y - frame.delta[h]) ** 2) / np.mean(y ** 2)
        assert f"block skill {skill:+.4f} ± {hw_srv:.4f}" in titles


def test_table_errors_skill_and_sign_agreement(fig_block, block):
    frame, raw = block
    rmse_row, mae_row = _row(fig_block, "RMSE")[0], _row(fig_block, "MAE")[0]
    skill_raw_row = _row(fig_block, "skill vs predicting 0: raw")[0]
    skill_srv_row = _row(fig_block, "skill vs predicting 0: served")[0]
    sign_row = _row(fig_block, "sign(Δ)")[0]
    for i, h in enumerate(H):
        y, d, s = frame.y[:, i], raw[h], frame.delta[h]
        rmse = [np.sqrt(np.mean((y - p) ** 2)) for p in (d, s, 0 * y)]
        assert rmse_row[i] == f"{rmse[0]:,.1f} / {rmse[1]:,.1f} / {rmse[2]:,.1f}"
        mae = [np.mean(np.abs(y - p)) for p in (d, s, 0 * y)]
        assert mae_row[i] == f"{mae[0]:,.1f} / {mae[1]:,.1f} / {mae[2]:,.1f}"
        skill_raw, skill_srv = (1 - np.mean((y - p) ** 2) / np.mean(y ** 2) for p in (d, s))
        assert skill_raw_row[i].startswith(f"{skill_raw:+.4f} ± ")
        assert skill_srv_row[i].startswith(f"{skill_srv:+.4f} ± ")
        agree = np.mean((d > 0) == (frame.prob(h, True) > 0.5))
        assert sign_row[i].startswith(f"{100 * agree:.1f}%")


def test_table_cells_avoid_the_greater_than_sign(fig_block):
    """A '>' in a plotly table cell (raw or &gt;) makes that row taller and misaligns its text."""
    tab = _table(fig_block)
    for col in tab.cells.values:
        assert not any(">" in v or "&gt;" in v for v in col)


def test_skill_cells_use_status_colours_only_with_a_symbol(fig_block):
    from neural_trade.visualization import theme as T

    for prefix in ("skill vs predicting 0: raw", "skill vs predicting 0: served"):
        for text, colour in zip(*_row(fig_block, prefix)):
            if colour == T.CRITICAL:
                assert text.endswith("▼")
            elif colour == T.GOOD:
                assert text.endswith("▲")
            else:
                assert text.endswith("≈0") or text == "n/a"


def test_magnitude_ordering_raw_vs_served_is_stated(fig_block, block):
    frame, raw = block

    def chain(D):
        A = np.abs(np.stack([D[h] for h in H], 1))
        return np.mean((A[:, 0] <= A[:, 1]) & (A[:, 1] <= A[:, 2]))

    text = " ".join(_titles(fig_block))
    assert f"full chain: raw {100 * chain(raw):.1f}% → served {100 * chain(frame.delta):.1f}%" in text


# ------------------------------------------------------------------ readability
def test_no_dotted_lines_dotted_means_training(fig_block):
    for t in fig_block.data:
        line = getattr(t, "line", None)
        assert line is None or line.dash != "dot", t.name
    for sh in fig_block.layout.shapes or ():
        assert sh.line.dash != "dot"


def test_reference_lines_have_legend_entries(fig_block):
    names = {t.name for t in fig_block.data if getattr(t, "showlegend", None)}
    assert "y = x (face value)" in names
    assert "block mean (no skill)" in names
    assert "95% band if no skill" in names
    assert not any("zero prediction baseline" in (t.name or "") for t in fig_block.data)


def test_diagonal_and_block_mean_swatches_look_different(fig_block):
    """Both are neutral grey: y = x needs a visible dash pattern at legend-swatch length (about 30 px)."""
    (diag,) = [t for t in fig_block.data if t.name == "y = x (face value)" and t.showlegend]
    (mean,) = [t for t in fig_block.data if t.name == "block mean (no skill)" and t.showlegend]
    assert mean.line.dash in (None, "solid")
    dash = diag.line.dash
    assert dash not in (None, "solid", "dot", "dash") and diag.line.width >= 1.5
    if "px" in dash:
        assert sum(float(v.strip().rstrip("px")) for v in dash.split(",")) <= 12   # several dashes per swatch


def test_every_row_has_its_own_legend_heading(fig_block):
    for k in ("legend2", "legend3", "legend4", "legend5"):
        assert fig_block.layout[k].title.text
    for t in fig_block.data:
        if getattr(t, "showlegend", None):
            assert t.name and t.legend in {"legend2", "legend3", "legend4", "legend5"}, t.name


def test_axis_titles_on_every_row(fig_block):
    for row in (2, 3, 4, 5):
        xa, ya = _axes(fig_block, row, 1)
        assert fig_block.layout["yaxis" + ya[1:]].title.text
        for j in (1, 2, 3):
            xa, _ = _axes(fig_block, row, j)
            assert fig_block.layout["xaxis" + xa[1:]].title.text


def test_rolling_x_axis_is_in_hours(fig_block, block):
    frame, _ = block
    (line,) = _named(fig_block, 4, 1, "raw price head")
    assert line.x[-1] == pytest.approx((len(frame) - 1) / 60, rel=1e-4)
    xa, _ = _axes(fig_block, 4, 1)
    assert "hours" in fig_block.layout["xaxis" + xa[1:]].title.text


def test_colours_by_role(fig_block):
    """Horizon colours only on horizon series in their own column; served in its own non-horizon,
    non-status colour; status colours never on a plotted series."""
    from neural_trade.visualization import theme as T
    from neural_trade.visualization.analytics_delta import SERVED_COLOR

    assert SERVED_COLOR not in T.HORIZON_COLORS.values()
    assert SERVED_COLOR not in (T.GOOD, T.WARNING, T.SERIOUS, T.CRITICAL)
    for j, h in enumerate(H, start=1):
        for row in (2, 3, 4, 5):
            for t in _traces(fig_block, row, j):
                cols = {str(getattr(getattr(t, "line", None), "color", None)),
                        str(getattr(getattr(t, "marker", None), "color", None))}
                others = {v for k, v in T.HORIZON_COLORS.items() if k != h}
                assert not any(o in c for o in others for c in cols), (row, j, t.name)
                assert not any(s in c for s in (T.GOOD, T.CRITICAL) for c in cols)
    served = [t for t in fig_block.data if (t.name or "").startswith("served")]
    assert served and all(t.line.color == SERVED_COLOR for t in served)


def test_no_empty_panel_and_size_budget_on_a_full_size_block(viz_config):
    from neural_trade.visualization import theme as T
    from neural_trade.visualization.model_analytics import delta_analytics_figure

    frame, raw = _overlap_frame(n=7236, seed=2)
    fig = delta_analytics_figure(frame, viz_config, raw_delta=raw)
    assert T.empty_panels(fig) == []
    assert len(fig.to_json()) < 600_000


def test_without_raw_heads_the_figure_says_it_shows_the_served_delta(block, viz_config):
    from neural_trade.visualization import theme as T
    from neural_trade.visualization.model_analytics import delta_analytics_figure

    frame, _ = block
    fig = delta_analytics_figure(frame, viz_config)
    assert T.empty_panels(fig) == []
    assert "served delta" in fig.layout.title.text
    footer = [a.text for a in fig.layout.annotations if "Magnitude ordering" in (a.text or "")]
    assert len(footer) == 1 and "reflects the per-horizon β" in footer[0] and "raw_delta=" in footer[0]
    assert not any((t.name or "").startswith("served β") for t in fig.data)
    assert _row(fig, "served β")[0] == ["n/a"] * 3
    (inside,) = _named(fig, 2, 1, "sample")
    beyond = _named(fig, 2, 1, "beyond")
    xs = np.r_[np.asarray(inside.x, float), np.asarray(beyond[0].customdata, float)[:, 0] if beyond else []]
    assert np.allclose(np.sort(xs), np.sort(frame.delta["h0"]), rtol=1e-5, atol=1e-3)


def test_short_block_and_constant_prediction_still_draw(viz_config):
    from neural_trade.visualization import theme as T
    from neural_trade.visualization.model_analytics import delta_analytics_figure

    frame, raw = _overlap_frame(n=300, seed=4)
    fig = delta_analytics_figure(frame, viz_config, raw_delta=raw)            # window 500 > n: shrunk
    assert T.empty_panels(fig) == []
    raw["h1"] = np.zeros(len(frame))
    frame.delta["h1"] = np.zeros(len(frame))
    fig = delta_analytics_figure(frame, viz_config, raw_delta=raw)
    assert "constant prediction or too few samples" in " ".join(_titles(fig))

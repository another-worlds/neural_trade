"""Confidence and cross-horizon coherence analytics: honest intervals, chance references, readable encodings."""
from __future__ import annotations

import math

import numpy as np
import pytest

from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")
LC = 100_000.0


def _frame(p, y, *, var=None, steps=(10, 15, 20), delta=None):
    """A PredictionFrame with the same P(up) / realised move / variance for every horizon (or [N, 3] arrays)."""
    from neural_trade.evaluation.frame import PredictionFrame

    p, y = np.asarray(p, float), np.asarray(y, float)
    P = p if p.ndim == 2 else np.column_stack([p] * 3)
    Y = y if y.ndim == 2 else np.column_stack([y] * 3)
    n = len(P)
    V = np.full((n, 3), 1.0) if var is None else (np.asarray(var, float) if np.ndim(var) == 2
                                                  else np.column_stack([np.asarray(var, float)] * 3))
    Dl = (P - 0.5) * 100 if delta is None else np.asarray(delta, float)
    return PredictionFrame(y=Y, last_close=np.full(n, LC), delta={h: Dl[:, i] for i, h in enumerate(H)},
                           direction_prob={h: P[:, i] for i, h in enumerate(H)},
                           variance_scaled={h: V[:, i] for i, h in enumerate(H)}, pred_scale=100.0,
                           horizon_steps=steps, direction_prob_calibrated={h: P[:, i] for i, h in enumerate(H)})


def _runs_frame(n=4000, run=40, seed=3):
    """Correctness constant within runs of ``run`` bars: neighbouring outcomes are strongly dependent."""
    rng = np.random.default_rng(seed)
    k = n // run
    p = np.repeat(rng.choice([0.35, 0.45, 0.55, 0.65], k), run)
    up = np.repeat(rng.random(k) < 0.5, run)
    y = np.where(up, 200.0, -200.0)
    return _frame(p, y)


def _conf(frame, config=None, **kw):
    from neural_trade.visualization.analytics_confidence import confidence_analytics_figure

    return confidence_analytics_figure(frame, config, **kw)


def _coh(frame, config=None, **kw):
    from neural_trade.visualization.analytics_confidence import coherence_analytics_figure

    return coherence_analytics_figure(frame, config, **kw)


def _one(fig, row, col, name):
    """The one data trace called ``name`` in panel (row, col) (legend-only keys have no data)."""
    tr = [t for t in fig.select_traces(row=row, col=col) if t.name == name and t.x is not None and t.x[0] is not None]
    assert len(tr) == 1, (row, col, name, [t.name for t in fig.select_traces(row=row, col=col)])
    return tr[0]


def _hlines(fig, row, col):
    yref = fig.get_subplot(row, col).yaxis.plotly_name.replace("axis", "")
    return [s for s in fig.layout.shapes if s.type == "line" and s.yref == yref and s.y0 == s.y1]


def _title(fig, row, col):
    """The subplot title annotation above panel (row, col)."""
    sp = fig.get_subplot(row, col)
    xd, yd = sp.xaxis.domain, sp.yaxis.domain
    for a in fig.layout.annotations:
        if a.xref == "paper" and abs(a.x - (xd[0] + xd[1]) / 2) < 1e-6 and abs(a.y - yd[1]) < 1e-6:
            return a.text
    raise AssertionError(f"no subplot title for {(row, col)}")


def _wilson_iid(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 2 * half


# ------------------------------------------------------------------ correctness: intervals
def test_intervals_widen_for_overlapping_outcomes():
    """Runs of dependent outcomes: the 95% interval must be much wider than an i.i.d. Wilson interval
    (the old figure drew i.i.d. Wilson intervals on every 1-minute sample)."""
    fr = _runs_frame()
    fig = _conf(fr)
    band = _one(fig, 2, 1, "95% CI")
    x, y = np.asarray(band.x, float), np.asarray(band.y, float)
    m = len(x) // 2
    share, hi, lo = x[:m], y[:m], y[m:][::-1]
    assert share[-1] == pytest.approx(1.0)
    line = _one(fig, 2, 1, "h0")
    acc, n = float(np.asarray(line.y)[-1]), len(fr)
    assert hi[-1] - lo[-1] > 2.0 * _wilson_iid(acc * n, n)
    # the decile whiskers too
    dots = _one(fig, 1, 1, "h0")
    width = np.asarray(dots.error_y.array) + np.asarray(dots.error_y.arrayminus)
    k, nd = np.asarray(dots.y, float), np.asarray(dots.customdata)[:, 0]
    iid = np.array([_wilson_iid(a * m_, m_) for a, m_ in zip(k, nd)])
    assert np.median(width / iid) > 1.5


def test_independent_outcomes_keep_intervals_close_to_wilson(viz_frame, viz_config):
    """The block bootstrap does not inflate intervals when the outcomes really are independent."""
    fig = _conf(viz_frame, viz_config)
    band = _one(fig, 2, 1, "95% CI")
    y = np.asarray(band.y, float)
    m = len(y) // 2
    line = _one(fig, 2, 1, "h0")
    n = int(np.asarray(line.customdata)[-1, 0])
    acc = float(np.asarray(line.y)[-1])
    assert 0.7 < (y[m - 1] - y[m]) / _wilson_iid(acc * n, n) < 1.5


def test_short_block_falls_back_to_effective_sample_wilson():
    from neural_trade.visualization import stats as S

    rng = np.random.default_rng(0)
    n = 120                                          # shorter than two 80-bar bootstrap blocks
    p = rng.uniform(0.3, 0.7, n)
    y = np.where(rng.random(n) < 0.5, 150.0, -150.0)
    fig = _conf(_frame(p, y))
    dots = _one(fig, 1, 1, "h0")
    k = np.asarray(dots.y, float) * np.asarray(dots.customdata)[:, 0]
    _, lo, hi = S.wilson(k, np.asarray(dots.customdata)[:, 0], steps=10)
    np.testing.assert_allclose(np.asarray(dots.error_y.array), hi - np.asarray(dots.y), atol=1e-6)
    np.testing.assert_allclose(np.asarray(dots.error_y.arrayminus), np.asarray(dots.y) - lo, atol=1e-6)


def _dense_weights(n, n_boot=1000, seed=0, block=80):
    """The [n_boot, n] resample counts of the report's moving-block draw, built the direct way."""
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float)


def test_prefix_sum_bootstrap_equals_the_dense_resample_weights():
    """The low-memory bootstrap gives exactly the rates of the [n_boot, N] weight matrix it replaces."""
    from neural_trade.visualization.analytics_confidence import _boot_plan, _resample_sums, _subset_rates

    rng = np.random.default_rng(5)
    n = 1234                                           # not a multiple of the block: the last block is cut
    x = (rng.random(n) < 0.55).astype(float)
    sub = rng.random((7, n)) < np.linspace(0.05, 0.9, 7)[:, None]
    sub[3] = False                                     # an empty subset
    plan = _boot_plan(n, n_boot=300, seed=11)
    W = _dense_weights(n, n_boot=300, seed=11)
    np.testing.assert_array_equal(_resample_sums(sub, x, plan), (W @ (sub * x).T).T)
    np.testing.assert_array_equal(_resample_sums(sub, None, plan, budget=10), (W @ sub.T).T)   # chunked
    rate, lo, hi, cnt = _subset_rates(x, sub, plan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (W @ (sub * x).T) / (W @ sub.T)
    for k in range(len(sub)):
        if cnt[k] == 0:
            assert np.isnan(rate[k]) and np.isnan(lo[k]) and np.isnan(hi[k])
            continue
        want_lo, want_hi = np.nanpercentile(r[:, k], [2.5, 97.5])
        assert lo[k] == pytest.approx(min(want_lo, rate[k])) and hi[k] == pytest.approx(max(want_hi, rate[k]))
    assert _boot_plan(159, n_boot=10) is None and _boot_plan(5000, n_boot=0) is None


@pytest.mark.parametrize("case", ["works", "inverted", "noise", "one_sided", "short"])
def test_confidence_gap_matches_the_eval_report(case):
    """Rows 1 and 3 print the report's confidence gap; the figure's low-memory copy returns the same row."""
    from neural_trade.evaluation.report import confidence_gap
    from neural_trade.visualization.analytics_confidence import _confidence_gap

    rng = np.random.default_rng(7)
    n = 100 if case == "short" else 3000
    conf = rng.random(n)
    edge = {"works": 0.25, "inverted": -0.25}.get(case, 0.0)
    correct = (rng.random(n) < 0.5 + edge * (conf - 0.5)).astype(float)
    thr = -1.0 if case == "one_sided" else float(np.median(conf))
    got, want = _confidence_gap(correct, conf, thr), confidence_gap(correct, conf, thr)
    assert got["verdict"] == want["verdict"]
    if case in ("works", "inverted", "noise"):
        assert got["verdict"] == case.upper()
    assert set(got) == set(want)
    for key in ("gap", "acc_high", "acc_low", "threshold", "n_high"):
        if key in want:
            assert got[key] == pytest.approx(want[key], nan_ok=True)
    np.testing.assert_allclose(got["ci"], want["ci"], rtol=0, atol=1e-12)


def test_peak_memory_stays_small_on_a_month_of_bars():
    """A month of 1-minute bars: the old [n_boot, N] resample matrices needed about 1.2 GB per figure."""
    import tracemalloc

    rng = np.random.default_rng(8)
    n = 43_200
    fr = _frame(rng.uniform(0.3, 0.7, (n, 3)), rng.normal(0, 200, (n, 3)), var=rng.uniform(0.5, 2, (n, 3)))
    for build in (lambda: _conf(fr, var_scale=1.0), lambda: _coh(fr, raw_delta={h: rng.normal(0, 50, n) for h in H})):
        tracemalloc.start()
        try:
            build()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 200 * 2 ** 20, peak / 2 ** 20


# ------------------------------------------------------------------ correctness: encodings and references
def test_accuracy_is_drawn_as_points_and_bars_start_at_zero(viz_frame, viz_config):
    """Accuracy on a cut-off axis must be a point with an interval, never a bar measured from 0."""
    conf = _conf(viz_frame, viz_config)
    assert not [t for t in conf.data if t.type == "bar"]
    coh = _coh(viz_frame, viz_config)
    for t in coh.data:
        if t.type != "bar":
            continue
        ax = coh.layout[t.yaxis.replace("y", "yaxis", 1)]
        assert t.base is None
        assert ax.range is None or ax.range[0] == 0, (t.name, ax.range)


def test_chance_reference_is_the_test_block_majority_and_the_call_mix():
    """70% of the moves go up: the reference line sits at 70%, not 50%, and each decile carries the no-skill
    level q*b + (1-q)*(1-b) of its own call mix."""
    rng = np.random.default_rng(1)
    n = 3000
    p = rng.uniform(0.3, 0.7, n)
    up = rng.random(n) < 0.7
    fig = _conf(_frame(p, np.where(up, 150.0, -150.0)))
    lines = _hlines(fig, 1, 1)
    assert len(lines) == 1 and lines[0].y0 == pytest.approx(up.mean())
    assert all(abs(s.y0 - 0.5) > 0.05 for s in _hlines(fig, 2, 1))
    ns = _one(fig, 1, 1, "no skill")
    conf = np.abs(p - 0.5)
    for k, idx in enumerate(np.array_split(np.argsort(conf, kind="mergesort"), 10)):
        q, b = (p[idx] > 0.5).mean(), up[idx].mean()
        assert ns.y[k] == pytest.approx(q * b + (1 - q) * (1 - b))


def test_expected_if_calibrated_marker_and_decile_edges(viz_frame, viz_config):
    from neural_trade.visualization.analytics_common import _labels

    fig = _conf(viz_frame, viz_config)
    lab, mask = _labels(viz_frame, viz_config)["h1"]
    p = viz_frame.prob("h1", True)[mask]
    conf = np.abs(p - 0.5)
    chunks = np.array_split(np.argsort(conf, kind="mergesort"), 10)
    ex = _one(fig, 1, 2, "expected if calibrated")
    np.testing.assert_allclose(ex.y, [0.5 + conf[c].mean() for c in chunks])
    ticks = fig.layout[fig.get_subplot(1, 2).xaxis.plotly_name].ticktext
    assert [t.split("<br>")[1] for t in ticks] == ["&#8805;" + f"{0.5 + conf[c].min():.3f}".replace("0.", ".", 1)
                                                   for c in chunks]


def test_every_accuracy_row_shares_one_y_range(viz_frame, viz_config):
    fig = _conf(viz_frame, viz_config)
    for row in (1, 2, 3):
        ranges = {tuple(fig.layout[fig.get_subplot(row, c).yaxis.plotly_name].range) for c in (1, 2, 3)}
        assert len(ranges) == 1, (row, ranges)
    r1 = fig.layout[fig.get_subplot(1, 1).yaxis.plotly_name].range
    r3 = fig.layout[fig.get_subplot(3, 1).yaxis.plotly_name].range
    assert tuple(r1) == tuple(r3)


def test_confusion_matrix_is_row_normalised_with_correct_calls_on_the_diagonal(viz_frame, viz_config):
    from neural_trade.metrics import numpy_metrics as npm
    from neural_trade.visualization.analytics_common import _labels

    fig = _conf(viz_frame, viz_config)
    for j, h in enumerate(H, start=1):
        hm = [t for t in fig.select_traces(row=4, col=j) if t.type == "heatmap"][0]
        z = np.asarray(hm.z, float)
        np.testing.assert_allclose(z.sum(1), 1.0)
        assert list(hm.y) == ["real down", "real up"]
        assert fig.layout[fig.get_subplot(4, j).yaxis.plotly_name].autorange == "reversed"
        lab, mask = _labels(viz_frame, viz_config)[h]
        t, pr = lab[mask] > 0.5, viz_frame.prob(h, True)[mask] > 0.5
        assert z[0, 0] == pytest.approx(np.mean(~pr[~t]))          # recall of the down class, top-left
        assert z[1, 1] == pytest.approx(np.mean(pr[t]))            # recall of the up class, bottom-right
        prec_up = np.mean(t[pr])
        assert f"precision {prec_up:.1%}" in hm.x[1]
        mcc = npm.mcc(lab[mask], viz_frame.prob(h, True)[mask])
        assert "always" in _title(fig, 4, j)
        xt = fig.layout[fig.get_subplot(4, j).xaxis.plotly_name].title.text
        assert f"MCC {mcc:+.3f}" in xt and "balanced accuracy" in xt


def test_report_confidence_gap_goes_into_the_titles(viz_frame, viz_config):
    gap = {"gap": 0.0123, "ci": [0.002, 0.03], "verdict": "WORKS"}
    report = {"n": len(viz_frame), "model": {"horizons": {h: {"confidence_gap": gap} for h in H}}}
    fig = _conf(viz_frame, viz_config, report=report)
    assert "gap +0.012 [+0.002, +0.030] WORKS" in _title(fig, 1, 2)
    # a report scored on another block is not shown as if it described this one
    other = {"n": len(viz_frame) + 1, "model": report["model"]}
    fig2 = _conf(viz_frame, viz_config, report=other)
    assert "+0.012 [+0.002, +0.030]" not in _title(fig2, 1, 2)
    assert "the report given scored" in fig2.layout.title.text


def test_report_accepts_an_eval_report_json_path(tmp_path, viz_frame, viz_config):
    import json

    gap = {"gap": -0.05, "ci": [-0.09, -0.02], "verdict": "INVERTED"}
    path = tmp_path / "eval_report_test.json"
    path.write_text(json.dumps({"n": len(viz_frame), "model": {"horizons": {h: {"confidence_gap": gap}
                                                                             for h in H}}}))
    fig = _conf(viz_frame, viz_config, report=path)
    assert "INVERTED" in _title(fig, 1, 1)
    missing = _conf(viz_frame, viz_config, report=tmp_path / "absent.json")   # a run without a report
    assert "no eval report at absent.json" in missing.layout.title.text


def test_variance_confidence_row_uses_the_strategies_confidence(viz_frame, viz_config):
    """Row 3: deciles of SignalFrame.confidence = exp(-var / var_scale), over all bars."""
    from neural_trade.strategy import SignalFrame

    vs = 1.7
    fig = _conf(viz_frame, viz_config, var_scale=vs)
    conf = SignalFrame.build(viz_frame, vs).confidence[:, 0]
    chunks = np.array_split(np.argsort(conf, kind="mergesort"), 10)
    ticks = fig.layout[fig.get_subplot(3, 1).xaxis.plotly_name].ticktext
    assert [t.split("<br>")[1] for t in ticks] == ["&#8805;" + f"{conf[c].min():.2f}".replace("0.", ".", 1)
                                                   for c in chunks]
    assert "&#961; vs |P&#8722;0.5|" in _title(fig, 3, 1)
    # without var_scale the same deciles are labelled by predicted sigma
    fig0 = _conf(viz_frame, viz_config)
    assert "$" in fig0.layout[fig0.get_subplot(3, 1).xaxis.plotly_name].ticktext[0]


def test_selective_curve_is_thinned_and_its_axis_shows_2_percent(viz_frame, viz_config):
    fig = _conf(viz_frame, viz_config)
    for j, h in enumerate(H, start=1):
        line = _one(fig, 2, j, h)
        assert len(line.x) <= 200
        assert line.x[-1] == pytest.approx(1.0)
        ax = fig.layout[fig.get_subplot(2, j).xaxis.plotly_name]
        assert ax.type == "log" and ax.range[0] == pytest.approx(math.log10(0.02)) and ax.range[1] == 0


def test_size_budget_on_a_long_block():
    rng = np.random.default_rng(2)
    n = 20_000
    p = rng.uniform(0.3, 0.7, (n, 3))
    y = rng.normal(0, 200, (n, 3))
    fr = _frame(p, y, var=rng.uniform(0.5, 2, (n, 3)))
    assert len(_conf(fr).to_json()) < 600_000
    assert len(_coh(fr, raw_delta={h: rng.normal(0, 50, n) for h in H}).to_json()) < 600_000


# ------------------------------------------------------------------ coherence
def test_vote_patterns_and_the_independence_reference(viz_frame, viz_config):
    fig = _coh(viz_frame, viz_config)
    bar = _one(fig, 1, 2, "share of bars")
    assert list(bar.x) == ["UUU", "UUD", "UDU", "UDD", "DUU", "DUD", "DDU", "DDD"]
    U = np.column_stack([viz_frame.prob(h, True) > 0.5 for h in H])
    for k, pat in enumerate(bar.x):
        want = np.all(U == np.array([c == "U" for c in pat]), axis=1).mean()
        assert bar.y[k] == pytest.approx(want)
    q = U.mean(0)
    ind = _one(fig, 1, 2, "if the horizons voted independently")
    assert ind.y[2] == pytest.approx(q[0] * (1 - q[1]) * q[2])      # UDU


def test_up_rate_by_votes_per_horizon(viz_frame, viz_config):
    from neural_trade.visualization.analytics_common import _labels

    fig = _coh(viz_frame, viz_config)
    votes = np.column_stack([viz_frame.prob(h, True) > 0.5 for h in H]).sum(1)
    for h in H:
        lab, mask = _labels(viz_frame, viz_config)[h]
        tr = _one(fig, 1, 3, h)
        assert tr.marker.color == T.HORIZON_COLORS[h]
        want = [lab[mask & (votes == v)].mean() for v in range(4)]
        np.testing.assert_allclose(tr.y, want)


def test_head_alignment_and_raw_magnitude_ordering(viz_frame, viz_config):
    from neural_trade.evaluation.report import coherence_block

    rng = np.random.default_rng(4)
    raw = {h: rng.normal(0, 10 * (i + 1), len(viz_frame)) for i, h in enumerate(H)}
    fig = _coh(viz_frame, viz_config, raw_delta=raw)
    cb = coherence_block(viz_frame)
    al = _one(fig, 2, 1, "share of bars")
    assert list(al.x) == ["h0", "h1", "h2", "all 3"]
    assert al.y[1] == pytest.approx(cb["coherence_primary"]) and al.y[3] == pytest.approx(cb["delta_dir_align_all"])
    a = np.abs(np.column_stack([raw[h] for h in H]))
    r = _one(fig, 2, 2, "raw price heads")
    assert r.y[2] == pytest.approx(np.mean((a[:, 0] <= a[:, 1]) & (a[:, 1] <= a[:, 2])))
    served = _one(fig, 2, 2, "served deltas (shrunk)")
    assert served.y[2] == pytest.approx(cb["mag_order_full"])
    # the raw PredictionFrame itself is accepted, and [N, 1] columns too
    from types import SimpleNamespace

    fig2 = _coh(viz_frame, viz_config, raw_delta=SimpleNamespace(delta={h: v[:, None] for h, v in raw.items()}))
    np.testing.assert_allclose(_one(fig2, 2, 2, "raw price heads").y, r.y)
    # without raw deltas the panel says what it shows
    fig3 = _coh(viz_frame, viz_config)
    assert _one(fig3, 2, 2, "deltas as given").y[2] == pytest.approx(cb["mag_order_full"])
    assert "raw_delta" in fig3.layout.legend4.title.text


def test_strategy_vote_agreement_comes_from_signalframe(viz_frame, viz_config):
    from neural_trade.strategy import SignalFrame

    fig = _coh(viz_frame, viz_config)
    agr = np.round(SignalFrame.build(viz_frame, 1.0).agreement * 3).astype(int)
    bar = _one(fig, 2, 3, "share of bars")
    np.testing.assert_allclose(bar.y, [(agr == a).mean() for a in (1, 2, 3)])


def test_coherence_colours_by_role(viz_frame, viz_config):
    """Horizon colours only mark horizons; the correlation heatmap uses a neutral ramp, no diagonal."""
    fig = _coh(viz_frame, viz_config)
    hc = set(T.HORIZON_COLORS.values())
    for t in fig.data:
        if t.type == "heatmap":
            assert not {c for _, c in t.colorscale} & hc
            assert all(np.isnan(np.asarray(t.z, float)[i, i]) for i in range(3))
            continue
        colors = getattr(t.marker, "color", None)
        colors = list(colors) if isinstance(colors, (list, tuple)) else [colors]
        for k, c in enumerate(colors):
            if c in hc:
                key = t.x[k] if len(colors) > 1 else t.name
                assert T.HORIZON_COLORS.get(key) == c, (t.name, key, c)


# ------------------------------------------------------------------ readability and compatibility
def test_categorical_axes_keep_every_category(viz_frame, viz_config):
    """Plotly turns '1', '2' into a numeric axis unless told otherwise, silently dropping text categories."""
    from neural_trade.visualization.model_analytics import coherence_analytics_figure, confidence_analytics_figure

    coh = coherence_analytics_figure(viz_frame, viz_config)
    assert coh.layout.xaxis2.type == "category" and len(coh.data[1].x) == 8
    conf = confidence_analytics_figure(viz_frame, viz_config)
    assert conf.layout.xaxis.type == "category" and list(conf.data[0].x) == [str(k) for k in range(1, 11)]
    assert conf.layout.xaxis7.type == "category"


def test_legends_are_row_and_panel_headings(viz_frame, viz_config):
    conf = _conf(viz_frame, viz_config, var_scale=1.0)
    for legend in ("legend", "legend2", "legend3"):
        items = [t.name for t in conf.data if (t.legend or "legend") == legend and t.showlegend is not False]
        assert items and conf.layout[legend].title.text
    assert any("no skill" in n for n in items)
    coh = _coh(viz_frame, viz_config)
    for legend in ("legend", "legend2", "legend3", "legend4", "legend5"):
        assert coh.layout[legend].title.text


def _shown_keys(fig):
    """(legend, name, colour, trace) for every legend entry the figure shows."""
    out = []
    for t in fig.data:
        if t.showlegend is False or t.type == "heatmap":
            continue
        line = getattr(t, "line", None)
        colour = (line.color if t.type == "scatter" and "lines" in (t.mode or "") else None) or t.marker.color
        out.append((t.legend or "legend", t.name, colour, t))
    return out


def test_every_legend_key_has_the_colour_of_what_it_names(viz_frame, viz_config):
    """A horizon colour appears on a key only when the key names that horizon; a bar trace whose bars have
    several colours never shows its own key (Plotly would draw its first bar's colour)."""
    hc = {c: h for h, c in T.HORIZON_COLORS.items()}
    for fig in (_conf(viz_frame, viz_config, var_scale=1.0), _coh(viz_frame, viz_config, raw_delta=viz_frame.delta)):
        for legend, name, colour, _ in _shown_keys(fig):
            assert not isinstance(colour, (list, tuple)), (legend, name)
            if colour in hc:
                assert name == hc[colour], (legend, name, colour)


def test_selective_accuracy_keys_are_the_horizon_lines(viz_frame, viz_config):
    """Row 2: the accuracy key is the horizon-coloured line itself; the no-skill key is a thinner grey line."""
    fig = _conf(viz_frame, viz_config)
    keys = {name: t for legend, name, _, t in _shown_keys(fig) if legend == "legend2"}
    for h in H:
        assert keys[h].line.color == T.HORIZON_COLORS[h] and keys[h].x[0] is not None      # the data line
    noskill = next(t for n, t in keys.items() if n.startswith("no skill"))
    assert noskill.line.color == T.NEUTRAL and noskill.line.width < keys["h0"].line.width
    assert not [n for n in keys if "accuracy" in n]                    # no separate grey "accuracy" key
    assert "most decided x%" in fig.layout.legend2.title.text and "95% CI" in fig.layout.legend2.title.text
    # the majority values sit in the row's titles
    assert "vs always" in _title(fig, 2, 1)


def test_coherence_bar_keys_and_wording(viz_frame, viz_config):
    fig = _coh(viz_frame, viz_config, raw_delta=viz_frame.delta)
    assert _one(fig, 1, 2, "share of bars").showlegend is False
    assert "light = unanimous UUU / DDD" in fig.layout.legend.title.text
    assert _one(fig, 2, 1, "share of bars").showlegend is False
    key = [t for t in fig.select_traces(row=2, col=1) if t.name == "same sign"]
    assert len(key) == 1 and key[0].marker.color == T.INK_2 and key[0].showlegend is not False
    assert "Does |delta| grow with the horizon?" in fig.layout.legend4.title.text
    sub = fig.layout.title.text
    assert "realised moves |y|" in sub and "unordered outputs" not in sub
    # one n_eff convention across both figures: N // steps, as the evaluation report counts it
    n, steps = len(viz_frame), max(viz_frame.horizon_steps)
    note = [a.text for a in fig.layout.annotations if "n_eff" in a.text]
    assert len(note) == 1 and f"n_eff = N / {steps} = {n // steps:,})" in note[0]
    assert f"{n // steps:,}" in _conf(viz_frame, viz_config).layout.title.text


def test_vote_lines_are_the_ones_signalframe_applies(viz_frame, viz_config, monkeypatch):
    from neural_trade.strategy import signals
    from neural_trade.visualization.analytics_confidence import _vote_lines

    up, dn = _vote_lines()
    assert 0.5 <= up < 1 and 0 < dn <= 0.5
    sf = signals.SignalFrame.build(viz_frame, 1.0)
    p = sf.p
    assert np.array_equal(sf.consensus, np.sign((p > up).sum(1) - (p < dn).sum(1)))
    fig = _coh(viz_frame, viz_config)
    assert f"P(up) &gt; {up:g} or &lt; {dn:g}" in fig.layout.legend5.title.text
    monkeypatch.setattr(signals, "VOTE_UP", 0.6, raising=False)
    monkeypatch.setattr(signals, "VOTE_DOWN", 0.4, raising=False)
    assert _vote_lines() == (0.6, 0.4)                                 # module constants win when defined


def test_registry_passes_the_new_options(viz_frame, viz_config):
    from neural_trade.registries.visualizations import Visualizations

    fig = Visualizations.build("confidence_analytics", viz_frame, viz_config, var_scale=1.2, n_boot=50, height=1500)
    assert T.empty_panels(fig) == [] and fig.layout.height == 1500
    fig = Visualizations.build("coherence_analytics", viz_frame, viz_config, raw_delta=viz_frame.delta, n_boot=50)
    assert T.empty_panels(fig) == []


def test_a_horizon_without_scored_samples_does_not_break_the_figure(viz_frame, viz_config):
    import copy

    fr = copy.deepcopy(viz_frame)
    fr.y[:, 2] = 0.0                                        # every h2 move inside the deadband
    fig = _conf(fr, viz_config)
    assert "fewer than 20 scored samples" in _title(fig, 1, 3)
    _coh(fr, viz_config)

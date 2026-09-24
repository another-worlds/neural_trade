"""Direction-head analytics: the numbers printed on the figure, their uncertainty, and readability."""
from __future__ import annotations

import re

import numpy as np
import pytest

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T

H = ("h0", "h1", "h2")


# ------------------------------------------------------------------ helpers
def _frame(p_cal, up, *, p_raw=None, n=None, steps=(10, 15, 20), delta=None):
    """A PredictionFrame whose realised moves are +-$200 (outside a 5 bps deadband at $100k):
    ``up`` [N] or [N, 3] gives the class, ``p_cal`` / ``p_raw`` [N] or {h: [N]} the probabilities."""
    from neural_trade.evaluation.frame import PredictionFrame

    up = np.asarray(up, bool)
    n = len(up) if n is None else n
    up = np.repeat(up[:, None], 3, 1) if up.ndim == 1 else up
    per = lambda v: v if isinstance(v, dict) else {h: np.asarray(v, float) for h in H}  # noqa: E731
    p_cal = per(p_cal)
    p_raw = per(p_raw if p_raw is not None else p_cal)
    y = np.where(up, 200.0, -200.0)
    return PredictionFrame(
        y=y, last_close=np.full(n, 100_000.0), delta=per(delta if delta is not None else y[:, 0] * 0.1),
        direction_prob=p_raw, variance_scaled={h: np.full(n, 1.0) for h in H}, pred_scale=100.0,
        horizon_steps=tuple(steps), direction_prob_calibrated=p_cal)


def _stretch(p, k=1.3):
    """A raw head whose monotone calibration is ``p`` (so the frame counts as calibrated)."""
    if isinstance(p, dict):
        return {h: _stretch(v, k) for h, v in p.items()}
    return np.clip(0.5 + k * (np.asarray(p, float) - 0.5), 0.001, 0.999)


def _row_titles(fig, prefix):
    return [a.text for a in fig.layout.annotations if a.text and a.text.startswith(prefix)]


def _plain(text):
    """Visible text of a plotly rich-text string."""
    return re.sub(r"<[^>]+>", "", text or "")


def _keys(fig, legend_id):
    """Names of the entries shown in one legend."""
    return [t.name for t in fig.data if t.type != "table" and t.showlegend is not False
            and (t.legend or "legend") == legend_id]


def _named(fig, row, col, name):
    """The data traces called ``name`` in one panel (legend-only keys, drawn at x=[None], excluded)."""
    return [t for t in fig.select_traces(row=row, col=col)
            if t.name == name and not (len(t.x) == 1 and t.x[0] is None)]


# ------------------------------------------------------------------ statistics
def test_roc_curve_matches_sklearn(viz_frame):
    from sklearn.metrics import roc_auc_score

    from neural_trade.visualization.model_analytics import roc_curve

    lab = (viz_frame.y[:, 1] > 0).astype(float)
    assert roc_curve(lab, viz_frame.direction_prob["h1"])[2] == pytest.approx(roc_auc_score(lab, viz_frame.direction_prob["h1"]),
                                                                          abs=1e-9)


def test_roc_points_auc_matches_sklearn_with_ties_and_thresholds_are_monotone():
    from sklearn.metrics import roc_auc_score

    from neural_trade.visualization.analytics_direction import roc_points

    rng = np.random.default_rng(3)
    lab = rng.random(5000) < 0.5
    score = np.round(0.5 + 0.1 * rng.normal(size=5000) + 0.03 * lab, 2)          # many ties
    fpr, tpr, thr, auc = roc_points(lab, score, max_points=300)
    assert auc == pytest.approx(roc_auc_score(lab, score), abs=1e-12)
    assert len(fpr) <= 300 and fpr[0] == 0 and tpr[-1] == 1
    assert np.all(np.diff(thr[1:]) <= 0)
    assert np.isnan(roc_points(np.ones(10), np.arange(10.0))[3])                   # one class: no AUC


def test_auc_difference_is_paired_and_scales_with_effective_samples():
    from sklearn.metrics import roc_auc_score

    from neural_trade.visualization.analytics_direction import auc_difference

    rng = np.random.default_rng(4)
    lab = rng.random(4000) < 0.5
    a = rng.normal(size=4000) + 0.3 * lab
    b = a + rng.normal(size=4000)
    d, lo, hi = auc_difference(lab, a, b)
    assert d == pytest.approx(roc_auc_score(lab, a) - roc_auc_score(lab, b), abs=1e-12)
    assert lo < d < hi
    d4, lo4, hi4 = auc_difference(lab, a, b, steps=4)
    assert (hi4 - lo4) == pytest.approx(2 * (hi - lo), rel=1e-9)                 # sqrt(4): n / steps samples
    assert auc_difference(lab, a, a) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_reliability_bars_are_clustered_for_overlapping_outcomes():
    """Labels that come in runs (overlapping targets) must widen the bars; independent labels must not."""
    from neural_trade.visualization.analytics_direction import reliability_rows

    rng = np.random.default_rng(5)
    n = 6000
    p = rng.uniform(0.3, 0.7, n)
    runs = np.repeat(rng.random(n // 40) < 0.5, 40).astype(float)              # 40-bar runs
    iid = (rng.random(n) < 0.5).astype(float)
    t = np.arange(n)
    for labels, low, high in ((runs, 1.6, 4.0), (iid, 0.99, 1.3)):
        rows = reliability_rows(labels, p, t, block=80)
        half = (rows[:, 4] - rows[:, 3]) / 2
        binom = S.Z95 * np.sqrt(rows[:, 1] * (1 - rows[:, 1]) / rows[:, 2])
        assert low <= np.mean(half / binom) <= high
    rows = reliability_rows(np.zeros(100), np.linspace(0, 1, 100), block=10, steps=10)
    assert np.all(np.isfinite(rows)) and np.all(rows[:, 4] > 0)                 # all-down bins: Wilson fallback


# ------------------------------------------------------------------ the figure: printed numbers
def test_auc_titles_carry_the_effective_sample_interval(viz_frame, viz_config):
    from sklearn.metrics import roc_auc_score

    from neural_trade.visualization.analytics_common import _labels
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    titles = _row_titles(fig, "AUC head")
    assert len(titles) == 3
    labels = _labels(viz_frame, viz_config)
    for i, (h, text) in enumerate(zip(H, titles)):
        m = re.match(r"AUC head (\d\.\d{3}) \[(\d\.\d\d), (\d\.\d\d)\] · Gaussian (\d\.\d{3}) \[", text)
        assert m, text
        lab, mask = labels[h]
        auc = roc_auc_score(lab[mask], viz_frame.prob(h)[mask])
        lo, hi = S.auc_ci(auc, int(lab[mask].sum()), int((1 - lab[mask]).sum()), steps=viz_frame.horizon_steps[i])
        assert float(m.group(1)) == pytest.approx(auc, abs=5e-4)
        assert (float(m.group(2)), float(m.group(3))) == pytest.approx((lo, hi), abs=5e-3)
        assert "chance" in text


def test_ece_title_describes_the_plotted_equal_count_bins():
    """All P(up) in one 0.1-wide bin but badly ordered: the fixed-width ECE is ~0, the plotted bins are not."""
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    rng = np.random.default_rng(6)
    n = 4000
    p = rng.uniform(0.505, 0.595, n)
    up = rng.random(n) < (0.55 - 3 * (p - 0.55))
    from neural_trade.metrics.numpy_metrics import ece_pos

    assert ece_pos(up.astype(float), p) < 0.02                                       # the fixed-width ECE hides it
    fig = direction_analytics_figure(_frame(p, up, p_raw=_stretch(p)), None)
    titles = _row_titles(fig, "ECE")
    assert len(titles) == 3
    for j, text in enumerate(titles, start=1):
        m = re.search(r"raw (\d\.\d\d) → calibrated (\d\.\d\d)", text)
        assert m, text
        (cal,) = _named(fig, 3, j, "calibrated P(up)")
        x, y, w = np.asarray(cal.x, float), np.asarray(cal.y, float), np.asarray(cal.customdata)[:, 1]
        drawn = float(np.average(np.abs(y - x), weights=w))
        assert drawn > 0.05
        assert float(m.group(2)) == pytest.approx(drawn, abs=0.0051)
        assert "equal-count" in text and "noise" in text


def test_scorecard_matches_the_evaluation_report(viz_frame, viz_config):
    from neural_trade.evaluation.report import score_frame
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    (table,) = [t for t in fig.data if t.type == "table"]
    cols = [list(c) for c in table.cells.values]
    rows = {label: i for i, label in enumerate(cols[0])}
    sc = score_frame(viz_frame, float(viz_config.DIR_DEADBAND_BPS))

    def cell(j, key):
        return cols[j][rows[next(k for k in rows if k.startswith(key))]]

    for j, h in enumerate(H, start=1):
        d, g = sc["horizons"][h]["direction"], sc["horizons"][h]["gauss_direction"]
        n, steps = d["n_masked"], viz_frame.horizon_steps[j - 1]
        assert cell(j, "labelled n") == f"{n:,} · {n // steps:,}"
        assert cell(j, "realised up-rate") == f"{d['true_up_rate']:.1%} · {d['pred_up_rate']:.1%}"
        assert cell(j, "accuracy").startswith(f"{d['acc']:.1%} [")
        assert cell(j, "accuracy").endswith(f"{d['bal_acc']:.1%}")
        assert cell(j, "MCC") == f"{d['mcc']:+.3f}"
        assert cell(j, "Brier").split(" · ") == [f"{d['brier']:.4f}", "0.2500", f"{g['brier']:.4f}"]
        assert f"→ {d['ece_pos']:.3f}" in cell(j, "ECE, report bins")
        assert cell(j, "ECE, report bins").endswith(f"{g['ece_pos']:.3f}")
        assert "calibrated head" in next(k for k in rows if k.startswith("Brier"))


def test_titles_state_labelled_and_effective_n(viz_frame, viz_config):
    from neural_trade.visualization.analytics_common import _labels
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    labels = _labels(viz_frame, viz_config)
    titles = [a.text for a in fig.layout.annotations if a.text and "labelled (≈" in a.text]
    assert len(titles) == 3
    for i, (h, text) in enumerate(zip(H, titles)):
        n = int(labels[h][1].sum())
        assert f"n {n:,} labelled (≈{n // viz_frame.horizon_steps[i]:,} effective)" in text


# ------------------------------------------------------------------ the figure: readability
def test_reliability_axes_are_shared_and_ignore_single_extreme_predictions(viz_frame, viz_config):
    import copy

    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fr = copy.deepcopy(viz_frame)
    for h in H:
        fr.direction_prob[h] = fr.direction_prob[h].copy()
        fr.direction_prob[h][:5], fr.direction_prob[h][5:10] = 0.995, 0.005          # a handful of outliers
    fig = direction_analytics_figure(fr, viz_config)
    ranges = [tuple(fig.get_subplot(3, j).xaxis.range) for j in (1, 2, 3)]
    assert ranges[0] == ranges[1] == ranges[2]
    assert tuple(fig.get_subplot(3, 1).yaxis.range) == ranges[0]
    lo, hi = ranges[0]
    assert hi < 0.9 and lo > 0.1 and lo + hi == pytest.approx(1.0)
    for j in (1, 2, 3):
        for name in ("raw P(up)", "calibrated P(up)"):
            (tr,) = _named(fig, 3, j, name)
            assert lo <= np.min(tr.x) and np.max(tr.x) <= hi
        (cal,) = _named(fig, 3, j, "calibrated P(up)")
        assert np.all(np.asarray(cal.y) + np.asarray(cal.error_y.array) <= hi + 1e-6)


def test_raw_and_calibrated_are_told_apart_by_marker_and_nothing_is_dotted(viz_frame, viz_config):
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    for j in (1, 2, 3):
        (raw,) = _named(fig, 3, j, "raw P(up)")
        (cal,) = _named(fig, 3, j, "calibrated P(up)")
        assert raw.marker.symbol == "circle-open" and raw.mode == "markers" and raw.error_y.array is None
        assert cal.marker.symbol == "circle" and cal.error_y.array is not None
        assert any(t.legendgroup == "shift" for t in fig.select_traces(row=3, col=j))   # raw -> calibrated moves
    assert not [t.name for t in fig.data if getattr(t, "line", None) is not None and t.line.dash == T.TRAIN_DASH]
    assert not [s for s in fig.layout.shapes if s.line.dash == T.TRAIN_DASH]


def test_reliability_panels_draw_the_realised_up_rate(viz_frame, viz_config):
    from neural_trade.visualization.analytics_common import _labels
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    labels = _labels(viz_frame, viz_config)
    for j, h in enumerate(H, start=1):
        lab, mask = labels[h]
        (line,) = [t for t in fig.select_traces(row=3, col=j) if (t.name or "").startswith("realised up-rate")]
        assert np.allclose(line.y, lab[mask].mean())


def test_histogram_row_folds_the_tails_into_the_end_bins_and_labels_its_axes():
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    rng = np.random.default_rng(7)
    n = 4000
    p = rng.uniform(0.4, 0.6, n)
    p[:12] = 0.97                                                                    # beyond the 99.5th percentile
    up = rng.random(n) < 0.5
    up[:12] = True
    fig = direction_analytics_figure(_frame(p, up, p_raw=_stretch(p)), None)
    (tr,) = _named(fig, 1, 1, "realised up")
    x, dens = np.asarray(tr.x, float), np.asarray(tr.y, float)
    lo, hi = fig.get_subplot(1, 1).xaxis.range
    assert x[0] == pytest.approx(lo) and x[-1] == pytest.approx(hi)              # end bins drawn full width
    width = (x[2] - x[1])
    assert dens[-2] * width * up.sum() >= 12                                       # the folded tail is counted
    assert tr.marker.symbol == "triangle-up" and tr.line.color == T.UP_COLOR
    assert "tail above the axis" in tr.customdata[-2] and "tail below the axis" in tr.customdata[1]
    assert "calibrated P(up)" in fig.get_subplot(1, 1).xaxis.title.text
    assert fig.get_subplot(1, 1).yaxis.title.text == "density"
    assert "beyond the axis, in end bins" in _row_titles(fig, "<span")[0]


def test_histogram_axis_is_shared_and_not_stretched_by_one_heavy_tail():
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    rng = np.random.default_rng(10)
    n = 4000
    p = {h: 0.5 + 0.03 * rng.normal(size=n) for h in H}
    p["h1"][:120] = 0.97                                                            # 3% heavy tail on one horizon
    fig = direction_analytics_figure(_frame(p, rng.random(n) < 0.5), None)
    ranges = [tuple(fig.get_subplot(1, j).xaxis.range) for j in (1, 2, 3)]
    assert ranges[0] == ranges[1] == ranges[2]
    assert ranges[0][1] < 0.75                                                       # the core keeps the width
    lo, hi = ranges[0]
    share = np.mean((p["h1"] < lo) | (p["h1"] > hi))
    assert share >= 0.03 and f"{share:.1%} beyond the axis" in _row_titles(fig, "<span")[1]


def test_legend_keys_use_role_colours_and_every_reference_is_explained(viz_frame, viz_config):
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    horizon = set(T.HORIZON_COLORS.values())
    shown = [t for t in fig.data if t.type != "table" and t.showlegend is not False]
    for t in shown:
        colours = {getattr(t.line, "color", None), getattr(t.marker, "color", None)} - {None}
        assert not colours & horizon, t.name                                      # a key never looks like h0 only
    names = {t.name for t in shown}
    assert {"realised up", "realised down", "direction head", "Gaussian readout (price head)", "raw",
            "perfect", "realised up-rate"} <= names
    assert any("no-skill" in n for n in names) and any(n.startswith("calibrated (served)") for n in names)
    for legend_id in ("legend", "legend2", "legend3"):                              # one heading per row
        assert fig.layout[legend_id].title.text
    for j, h in enumerate(H, start=1):                                              # data keep the horizon colour
        (head,) = _named(fig, 2, j, "direction head")
        assert head.line.color == T.HORIZON_COLORS[h]


def test_calibration_shift_key_does_not_look_like_the_perfect_line(viz_frame, viz_config):
    """The raw -> calibrated connector is a thick translucent bar, key included; 'perfect' is a thin line,
    and the reliability legend draws both at their own widths."""
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    shown = {t.name: t for t in fig.data if t.type != "table" and t.showlegend is not False}
    (shift_key,) = [t for n, t in shown.items() if "calibration shift" in n]
    assert shift_key.name.startswith("○→●")                                        # open (raw) to filled (calibrated)
    perfect = shown["perfect"]
    assert shift_key.line.width >= 3 * perfect.line.width
    assert fig.layout.legend3.itemsizing == "trace"                                # widths survive in the legend
    for j in (1, 2, 3):
        (seg,) = [t for t in fig.select_traces(row=3, col=j) if t.legendgroup == "shift" and len(t.x) > 1]
        assert seg.line.width == shift_key.line.width


def test_frame_without_calibration_is_labelled_as_raw(viz_frame, viz_config):
    """No fitted calibration (None, or the raw head served as 'calibrated'): nothing claims calibration,
    the reliability row draws one series and no shift."""
    import copy

    from neural_trade.visualization.analytics_direction import calibration_applied
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    assert calibration_applied(viz_frame)
    none = copy.copy(viz_frame)
    none.direction_prob_calibrated = None
    same = copy.copy(viz_frame)
    same.direction_prob_calibrated = {h: v.copy() for h, v in viz_frame.direction_prob.items()}
    for fr in (none, same):
        assert not calibration_applied(fr)
        fig = direction_analytics_figure(fr, viz_config)
        keys = _keys(fig, "legend3")
        assert not [k for k in keys if k == "raw" or "shift" in k or k.startswith("calibrated")]
        assert any(k.startswith("raw P(up) (served)") for k in keys)
        assert not fig.layout.legend.title.text.lower().startswith("<b>calibrated")
        assert "no calibration" in fig.layout.legend.title.text
        for j in (1, 2, 3):
            traces = list(fig.select_traces(row=3, col=j))
            assert not [t for t in traces if getattr(t.marker, "symbol", None) == "circle-open"]
            assert not [t for t in traces if t.legendgroup == "shift"]
            assert "raw P(up)" in fig.get_subplot(1, j).xaxis.title.text
        for text in _row_titles(fig, "ECE"):
            assert "no calibration applied" in text and "→" not in text
        (table,) = [t for t in fig.data if t.type == "table"]
        labels = list(table.cells.values[0])
        assert any(k.startswith("Brier: head (uncalibrated)") for k in labels)
        assert not [k for k in labels if "raw → cal" in k]
        assert "no calibration applied" in fig.layout.title.text


def test_layout_fits_a_1000px_notebook(viz_frame, viz_config):
    """Text budgets that keep every line inside a ~1000 px container (the renders at 1000 / 1150 / 1500 px
    were checked by eye): short column titles, legend rows and subtitle lines, one-line table cells, and a
    table height that grows with its rows and leaves room for a wrapped line."""
    from neural_trade.visualization.analytics_direction import _HEADER_PX, _ROW_PX
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    fig = direction_analytics_figure(viz_frame, viz_config)
    for a in fig.layout.annotations:
        if a.text and a.text.startswith(("<span", "AUC", "ECE")):                   # the per-column titles
            first, *rest = a.text.split("<br>")
            assert len(_plain(first)) <= 60, first
            assert all(len(_plain(r)) <= 66 for r in rest), rest
    for legend_id in ("legend", "legend2", "legend3"):                              # heading + keys on one row
        width = len(_plain(fig.layout[legend_id].title.text)) + sum(len(k) + 8 for k in _keys(fig, legend_id))
        assert width <= 155, (legend_id, width)
    for line in fig.layout.title.text.split("<br>"):
        assert len(_plain(line)) <= 140, line
    (table,) = [t for t in fig.data if t.type == "table"]
    labels, *values = [list(c) for c in table.cells.values]
    assert all(len(k) <= 62 for k in labels), max(labels, key=len)
    assert all(len(v) <= 30 for col in values for v in col)
    assert not [c for col in [labels, *values] for c in col if "<" in c or ">" in c]   # no rich-text rows
    d0, d1 = table.domain.y
    plot_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    assert (d1 - d0) * plot_px >= _HEADER_PX + len(labels) * _ROW_PX + 30         # room for a wrapped line


def test_no_empty_panel_size_budget_and_old_calls(viz_config):
    from neural_trade.visualization.model_analytics import direction_analytics, direction_analytics_figure

    rng = np.random.default_rng(8)
    n = 40_000
    p = rng.uniform(0.3, 0.7, n)
    fr = _frame(p, rng.random(n) < p)
    fig = direction_analytics_figure(fr, viz_config)
    assert T.empty_panels(fig) == []
    assert len(fig.to_json()) < 600_000
    assert direction_analytics(fr, viz_config, bins=40, height=1020).layout.height >= 1020   # old keywords
    assert direction_analytics_figure(fr).layout.title.text.startswith("<b>Direction heads")


def test_degenerate_horizon_still_draws_every_panel():
    from neural_trade.visualization.model_analytics import direction_analytics_figure

    rng = np.random.default_rng(9)
    n = 2000
    p = rng.uniform(0.3, 0.7, n)
    up = np.c_[rng.random(n) < 0.5, rng.random(n) < 0.5, np.ones(n, bool)]         # h2: every move up
    fig = direction_analytics_figure(_frame(p, up), None)
    assert T.empty_panels(fig) == []
    assert "n/a" in _row_titles(fig, "AUC head")[2]

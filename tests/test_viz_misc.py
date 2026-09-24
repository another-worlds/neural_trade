"""Calibration explorer figures, the split overview and the run comparison (review fixes: misc)."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T

SIZE_BUDGET = 600_000
STATUS = {T.GOOD, T.WARNING, T.SERIOUS, T.CRITICAL}


def _persistent_labels(n, run, seed=0):
    """0/1 labels that repeat in runs of ``run`` (like overlapping horizon outcomes), with P(up) that
    follows them loosely: consecutive samples are far from independent."""
    rng = np.random.default_rng(seed)
    lab = np.repeat(rng.uniform(size=n // run + 1) < 0.5, run)[:n].astype(float)
    p = np.clip(0.5 + 0.05 * (lab - 0.5) + np.repeat(rng.normal(0, 0.05, n // run + 1), run)[:n]
                + rng.normal(0, 0.01, n), 0.01, 0.99)
    return lab, p


def _names(fig):
    return [t.name for t in fig.data]


def _trace(fig, name):
    return next(t for t in fig.data if t.name == name)


# ------------------------------------------------------------------ reliability
def test_reliability_bands_account_for_overlapping_outcomes():
    from neural_trade.visualization.calibration_plots import reliability_table

    lab, p = _persistent_labels(6000, 15)
    naive = reliability_table(lab, p, 10)
    hac = reliability_table(lab, p, 10, lag=15)
    y, n = naive[:, 1], naive[:, 2]
    old = 1.96 * np.sqrt(np.maximum(y * (1 - y), 1e-12) / n)          # the independent-samples formula
    assert np.allclose(naive[:, 4] - naive[:, 1], old)                 # lag=0 is unchanged
    assert np.allclose(hac[:, :3], naive[:, :3])                       # same bins, same rates
    widen = (hac[:, 4] - hac[:, 1]) / (naive[:, 4] - naive[:, 1])
    assert np.all(widen >= 1.0 - 1e-9) and np.median(widen) > 1.5     # runs of 15 shared outcomes


def test_reliability_figure_zooms_to_the_bins_and_colours_by_horizon():
    from neural_trade.visualization.calibration_plots import reliability_figure

    rng = np.random.default_rng(1)
    p = np.clip(rng.normal(0.5, 0.04, 5000), 0.01, 0.99)
    lab = (rng.uniform(size=p.size) < p).astype(float)
    p[0], p[1] = 0.995, 0.02                                          # lone extremes must not set the range
    p_cal = 0.5 + 0.8 * (p - 0.5)                                      # monotone: same bins as raw
    fig = reliability_figure(lab, p, p_cal, horizon="h2", horizon_steps=20, ref_rate=0.44,
                             title="Direction reliability, h2")
    x0, x1 = fig.layout.xaxis.range
    assert 0.0 <= x0 and x1 <= 1.0 and x1 - x0 < 0.6 and tuple(fig.layout.yaxis.range) == (x0, x1)
    cal, raw = _trace(fig, "calibrated P(up)"), _trace(fig, "raw P(up)")
    assert cal.line.color == raw.line.color == T.HORIZON_COLORS["h2"]
    assert cal.line.dash == T.VAL_DASH and raw.line.dash not in (T.TRAIN_DASH, T.VAL_DASH)
    assert cal.error_y.visible and raw.error_y.visible is False        # shared bins: one set of bars
    assert _trace(fig, "perfect").line.color == T.NEUTRAL
    base = next(t for t in fig.data if (t.name or "").startswith("test up-rate"))
    assert np.allclose(base.y, lab.mean()) and base.line.color == T.NEUTRAL
    ref = next(t for t in fig.data if (t.name or "").startswith("cal-block up-rate 0.440"))
    # three grey keys that a legend swatch (about 30 px) must tell apart: solid, dashed, dash-dot.
    # longdashdot draws as a solid bar at that length; dotted means training
    dashes = [_trace(fig, "perfect").line.dash or "solid", base.line.dash, ref.line.dash]
    assert dashes[0] == "solid" and len(set(dashes)) == 3
    assert not set(dashes[1:]) & {"solid", "longdashdot", "longdash", T.TRAIN_DASH}
    text = fig.layout.title.text
    assert text.startswith("<b>Direction reliability, h2</b>") and "HAC" in text and "AUC" in text
    assert "ECE" in text and "outside" in text                         # extremes reported, not hidden
    assert fig.layout.yaxis2.title.text == "samples"                   # the prediction histogram row
    assert len(fig.to_json()) < SIZE_BUDGET


def test_reliability_without_horizon_uses_a_neutral_ink_not_h0():
    from neural_trade.visualization.calibration_plots import reliability_figure

    lab, p = _persistent_labels(2000, 1, seed=2)
    fig = reliability_figure(lab, p)
    assert _trace(fig, "raw P(up)").line.color not in T.HORIZON_COLORS.values()
    assert "binomial" in fig.layout.title.text


# ------------------------------------------------------------------ coverage
def test_coverage_inflation_is_one_for_independent_and_large_for_overlapping_outcomes():
    from neural_trade.visualization.calibration_plots import coverage_inflation

    rng = np.random.default_rng(3)
    iid = (rng.uniform(size=20_000) < 0.9).astype(float)
    assert coverage_inflation(iid, 15, window=500) < 1.2
    shocks = rng.normal(size=20_015)
    moving = np.convolve(shocks, np.ones(15), mode="valid")[:20_000]   # 15-bar overlapping sums
    inside = (np.abs(moving) < 1.645 * np.sqrt(15)).astype(float)
    assert coverage_inflation(inside, 15, window=500) > 4


def test_coverage_figure_band_target_and_horizon_colour():
    from neural_trade.visualization.calibration_plots import coverage_inflation, coverage_over_time_figure

    rng = np.random.default_rng(4)
    n = 20_000
    y = np.convolve(rng.normal(size=n + 14), np.ones(15), mode="valid")
    lo, hi = np.full(n, -4.0), np.full(n, 4.0)
    fig = coverage_over_time_figure(y, lo, hi, window=500, target=0.7, horizon="h2", horizon_steps=15)
    tgt = [t for t in fig.data if (t.name or "").startswith("target")]
    assert tgt and all(np.allclose(t.y, 0.7) for t in tgt)             # follows the refit's alpha
    inside = ((y >= lo) & (y <= hi)).astype(float)
    half = S.Z95 * np.sqrt(0.7 * 0.3 * coverage_inflation(inside, 15, window=500) / 500)
    band = next(t for t in fig.data if "chance" in (t.name or ""))
    assert np.isclose(min(band.y), 0.7 - half) and np.isclose(max(band.y), min(1.0, 0.7 + half))
    assert _trace(fig, "coverage").line.color == T.HORIZON_COLORS["h2"]
    assert "overall coverage" in fig.layout.title.text and "effective samples" in fig.layout.title.text
    assert fig.layout.xaxis3.title.text.startswith("test sample")
    assert len(fig.to_json()) < SIZE_BUDGET


def test_coverage_figure_draws_a_second_pipeline_when_given():
    from neural_trade.visualization.calibration_plots import coverage_over_time_figure

    rng = np.random.default_rng(5)
    y = rng.normal(size=3000)
    fig = coverage_over_time_figure(y, y * 0 - 1, y * 0 + 1, window=300, target=0.7, horizon="h1",
                                    saved=(y * 0 - 2, y * 0 + 2), saved_target=0.9)
    saved = [t for t in fig.data if t.name == "saved pipeline"]
    assert len(saved) == 2 and all(t.line.dash == T.ALT_DASH for t in saved)
    assert "built for 0.90" in fig.layout.title.text


# ------------------------------------------------------------------ calibration explorer (synthetic run)
def _frames(n, seed, pred_scale=100.0):
    from neural_trade.evaluation.frame import PredictionFrame

    rng = np.random.default_rng(seed)
    path = 100_000 + np.cumsum(rng.normal(0, 30.0, n + 90))
    idx = np.arange(n)
    windows = np.stack([path[i + 1:i + 61] for i in idx])
    last = windows[:, -1]
    y = np.stack([path[idx + 60 + s] - last for s in (10, 15, 20)], 1)
    delta = {h: 0.2 * y[:, j] + rng.normal(0, 60.0, n) for j, h in enumerate(("h0", "h1", "h2"))}
    prob = {h: 1 / (1 + np.exp(-(y[:, j] / 150.0 + rng.normal(0, 1.0, n)))) for j, h in enumerate(("h0", "h1", "h2"))}
    var = {h: np.full(n, (120.0 / pred_scale) ** 2) for h in ("h0", "h1", "h2")}
    f = PredictionFrame(y=y, last_close=last, delta=delta, direction_prob=prob, variance_scaled=var,
                        pred_scale=pred_scale, horizon_steps=(10, 15, 20))
    f.X_raw = windows
    return f


@pytest.fixture(scope="module")
def explorer():
    from neural_trade.calibration import CalibrationPipeline
    from neural_trade.core.config import Config
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.notebook.calibration_ui import CalibrationExplorer

    cfg = Config()
    cal_raw, test_raw = _frames(1500, 10), _frames(2000, 11)
    preds = CalibrationExplorer._preds
    saved = CalibrationPipeline(conformal_scale="realized_vol", shrink_delta=True).fit_from_arrays(
        preds(cal_raw), cal_raw.y, cal_raw.last_close, deadband_bps=float(cfg.DIR_DEADBAND_BPS),
        conformal_alpha=0.1, windows=cal_raw.X_raw, pred_scale=100.0, horizon_steps=tuple(cfg.HORIZON_STEPS))

    def served(raw):
        out = saved.apply(preds(raw), alpha=0.1, windows=raw.X_raw)
        f = PredictionFrame(y=raw.y, last_close=raw.last_close, delta=out["delta"], direction_prob=raw.direction_prob,
                            variance_scaled=raw.variance_scaled, pred_scale=100.0, horizon_steps=(10, 15, 20),
                            direction_prob_calibrated=out["direction_prob"], intervals=out["intervals"])
        f.X_raw = raw.X_raw
        return f

    blocks = {"config": cfg, "cal_raw": cal_raw, "test_raw": test_raw, "cal": served(cal_raw),
              "test": served(test_raw),
              "predictor": SimpleNamespace(bundle=SimpleNamespace(pred_scale=100.0, calibration_pipeline=saved))}
    return CalibrationExplorer(blocks)


def test_explorer_coverage_target_follows_the_miscoverage_slider(explorer):
    explorer.refit("realized_vol", True, alpha=0.3)
    rel, cov = explorer.figures("h1")
    tgt = [t for t in cov.data if (t.name or "").startswith("target")]
    assert tgt and all(np.allclose(t.y, 0.7) for t in tgt)
    assert "alpha 0.30" in cov.layout.title.text
    # alpha moves the intervals, not P(up): the saved pipeline is drawn on coverage only
    assert "saved pipeline" in _names(cov) and not any("saved" in (n or "") for n in _names(rel))
    assert _trace(cov, "coverage").line.color == T.HORIZON_COLORS["h1"]
    assert _trace(rel, "calibrated P(up)").line.color == T.HORIZON_COLORS["h1"]
    assert explorer.last_table.loc["h1", "target"] == pytest.approx(0.7)


def test_explorer_compares_with_the_saved_pipeline(explorer):
    t = explorer.refit("realized_vol", True, alpha=0.1)
    assert explorer.matches_saved()
    both = explorer.comparison_table()
    assert list(both.index[:2]) == [("h0", "refit"), ("h0", "saved")]
    assert np.allclose(both.xs("refit", level=1)["coverage"], both.xs("saved", level=1)["coverage"], atol=1e-3)
    assert {"coverage cal (in-sample)", "up-rate cal", "up-rate test"} <= set(t.columns)
    rel, cov = explorer.figures("h0")
    assert "saved pipeline" not in _names(cov)                         # identical to the refit: not drawn twice
    explorer.refit("none", False, alpha=0.1)
    assert not explorer.matches_saved()
    assert "saved pipeline" in _names(explorer.figures("h0")[1])


def test_explorer_widget_stacks_the_figures(explorer):
    import ipywidgets as w

    box = explorer.widget()

    def walk(x):
        yield x
        for c in getattr(x, "children", ()):
            yield from walk(c)

    for hb in (x for x in walk(box) if isinstance(x, w.HBox)):
        assert not any(isinstance(c, w.Output) for c in hb.children)   # unsized plots side by side overflow
    explorer.click_refit()
    assert explorer._w["rel"].outputs and explorer._w["cov"].outputs


# ------------------------------------------------------------------ split overview
def test_split_overview_reads_on_the_dark_theme(synthetic_bars, tmp_path):
    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.visualization.data_overview import split_overview_figure

    synthetic_bars.to_csv(tmp_path / "b.csv", index=False)
    cfg = Config(CSV_PATH=str(tmp_path / "b.csv"), MAX_SEQUENCE_COUNT=2000)
    fig = split_overview_figure(split_arrays(cfg), cfg)
    assert fig.data[0].line.color == T.INK_2
    shapes = fig.layout.shapes
    assert len(shapes) == 4 and all(s.layer == "below" for s in shapes)
    fills = {s.fillcolor for s in shapes}
    assert not fills & (set(T.HORIZON_COLORS.values()) | STATUS)
    names = [a.text for a in fig.layout.annotations]
    assert all(any(a.startswith(f"<b>{b}</b>") for a in names) for b in ("train", "val", "cal", "test"))
    assert all(a.yref == "paper" and a.y >= 1.0 for a in fig.layout.annotations)   # above the price line
    # the short val and cal blocks sit side by side: their labels are on different rows, so they never touch
    row = {a.text.split("</b>")[0][3:]: a.yshift or 0 for a in fig.layout.annotations}
    assert row["val"] != row["cal"] and row["cal"] != row["test"] and row["train"] != row["val"]
    assert fig.layout.title.text.startswith("<b>") and "purged" in fig.layout.title.text
    assert fig.layout.yaxis.title.text and fig.layout.xaxis.title.text
    assert len(fig.to_json()) < SIZE_BUDGET


# ------------------------------------------------------------------ run comparison
IDS = ["20260924T042208Z-b5d9b70-af67ee43", "20260924T043447Z-b5d9b70-dirty-af67ee43",
       "20260924T114819Z-e23fd9f-dirty-af67ee43"]


def test_run_labels_are_short_unique_and_keep_the_time():
    from neural_trade.visualization.comparison import run_labels

    assert run_labels(IDS) == ["09-24 04:22 b5d9b70", "09-24 04:34 b5d9b70*", "09-24 11:48 e23fd9f*"]
    arms = ["20260923T195537Z-6dec27a-53cbe6af-only-LAMBDA_VAC_OVERFLOW__s0__P1",
            "20260923T195537Z-6dec27a-63ef2e2c-without-LAMBDA_VAC_OVERFLOW__s0__P1"]
    lab = run_labels(arms)
    assert len(set(lab)) == 2 and all("LAMBDA_VAC_OVERFLOW" in x for x in lab)
    same_minute = ["20260924T043401Z-b5d9b70-dirty-af67ee43", "20260924T043459Z-b5d9b70-dirty-af67ee43"]
    assert run_labels(same_minute) == ["09-24 04:34:01 b5d9b70*", "09-24 04:34:59 b5d9b70*"]
    assert run_labels(["custom", "custom"]) == ["custom", "custom #2"]


def test_runs_comparison_gives_every_run_its_own_row(tmp_path, monkeypatch):
    from neural_trade.visualization.comparison import runs_comparison_figure

    monkeypatch.chdir(tmp_path)

    # the old labels (last 28 characters) were equal for these two ablation arms: plotly stacked their bars
    ids = ["20260923T195537Z-6dec27a-53cbe6af-only-LAMBDA_VAC_OVERFLOW__s0__P1",
           "20260923T200108Z-6dec27a-c41d9410-without-LAMBDA_VAC_OVERFLOW__s0__P1"]
    assert ids[0][-28:] == ids[1][-28:]
    df = pd.DataFrame({"h1/direction/auc": [0.4905, 0.5053]}, index=ids)
    fig = runs_comparison_figure(df)
    t = fig.data[0]
    assert len(set(np.round(t.y, 6))) == 2 and np.allclose(sorted(t.x), [0.4905, 0.5053])
    assert len(set(fig.layout.yaxis.ticktext)) == 2
    lo, hi = fig.layout.xaxis.range
    assert lo > 0.4 and hi < 0.6                                       # zoomed to the data, not from 0


def test_runs_comparison_marks_missing_values_and_empty_panels(tmp_path, monkeypatch):
    from neural_trade.visualization.comparison import runs_comparison_figure

    monkeypatch.chdir(tmp_path)                                         # no runs/ folder: no reports
    df = pd.DataFrame({"h1/direction/auc": [0.51, np.nan, 0.49], "backtest/sharpe_net": [None, None, None]},
                      index=IDS)
    fig = runs_comparison_figure(df, ["h1/direction/auc", "backtest/sharpe_net"])
    assert all(t.yaxis in (None, "y") for t in fig.data)               # no trace on the empty panel
    assert any("not in these runs' eval reports" in (a.text or "") for a in fig.layout.annotations)
    na = [t for t in fig.data if t.mode == "text" and any("n/a" in x for x in t.text)]
    assert na and [x.strip() for x in na[0].text] == ["n/a"] and np.allclose(na[0].x, 0.5)
    assert na[0].textposition == "middle right"                        # beside the dashed line, not on it
    assert fig.data[0].marker.color == T.HORIZON_COLORS["h1"]
    text = fig.layout.title.text
    assert "no evaluation reports found" in text and "whether the runs share a block is unknown" in text
    assert "* after the commit = run from a working tree with uncommitted changes" in text


def _write_run(root, rid, auc, mcc, ev, cov, beta_h1, *, up=0.5147, n_masked=5697, rmse=236.1, logreg=0.5228):
    d = root / rid
    (d / "artifacts" / "calibration").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"seed": 42, "tags": []}))
    model = {"horizons": {"h1": {"n": 7236, "n_eff": 482,
                                 "direction": {"auc": auc, "mcc": mcc, "n_masked": n_masked, "true_up_rate": up},
                                 "delta": {"ev": ev, "corr": 0.0},
                                 "variance": {"coverage90": cov, "crpss": 0.01}}}}
    baselines = {"logreg_lags": {"horizons": {"h1": {"direction": {"auc": logreg, "mcc": 0.0397}}}},
                 "class_prior": {"horizons": {"h1": {"direction": {"auc": 0.5, "mcc": 0.0}}}},
                 "zero_delta": {"horizons": {"h1": {"delta": {"rmse": rmse, "ev": 0.0}}}}}
    (d / "eval_report_test.json").write_text(json.dumps(
        {"run_id": rid, "split": "test", "n": 7236, "deadband_bps": 5.0, "model": model, "baselines": baselines,
         "backtest": None, "meta": {"horizon_steps": [10, 15, 20]}}))
    (d / "artifacts" / "calibration" / "pipeline_meta.json").write_text(
        json.dumps({"delta_scale": {"h0": 0.2, "h1": beta_h1, "h2": 0.25}}))


def test_runs_comparison_intervals_references_baselines_and_beta(tmp_path):
    from neural_trade.experiments.compare import compare_runs
    from neural_trade.visualization.comparison import runs_comparison_figure

    _write_run(tmp_path, IDS[0], 0.5172, 0.0088, 0.0, 0.9059, 0.0)
    _write_run(tmp_path, IDS[1], 0.4966, -0.0165, 0.00046, 0.9081, 0.023)
    metrics = ["h1/direction/auc", "h1/direction/mcc", "h1/delta/ev", "h1/variance/coverage90", "backtest/sharpe_net"]
    runs = compare_runs(str(tmp_path / "*"), metrics=metrics, skip_unscored=True)
    fig = runs_comparison_figure(runs, metrics, run_dirs=str(tmp_path / "*"))
    auc = next(t for t in fig.data if t.xaxis in (None, "x") and t.mode == "markers" and t.name.startswith("h1"))
    lo, hi = S.auc_ci(0.5172, round(5697 * 0.5147), 5697 - round(5697 * 0.5147), steps=15)
    assert np.isclose(auc.error_x.array[0], hi - 0.5172, atol=1e-5)
    assert np.isclose(auc.error_x.arrayminus[0], 0.5172 - lo, atol=1e-5)
    assert auc.marker.color == T.HORIZON_COLORS["h1"]
    refs = {round(s.x0, 3) for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1}
    assert {0.5, 0.0, 0.9} <= refs                                     # chance, no skill, coverage target
    base = [t for t in fig.data if (t.name or "").startswith("best baseline")]
    assert base and np.allclose(base[0].x, 0.5228)                     # logreg_lags, not class_prior
    # the baseline is a grey open diamond: not an error-bar cap, not in a horizon colour
    assert base[0].marker.symbol == "diamond-open" and base[0].marker.color == T.NEUTRAL
    ev = next(t for t in fig.data if t.mode == "text" and any(x.strip() == "0 (β=0)" for x in t.text))
    tags = [x.strip() for x in ev.text]
    assert tags[0] == "0 (β=0)" and tags[1] != "0 (β=0)"
    # written beside the dot, clear of it and of the dashed zero line, never above it (a neighbour's dot)
    assert list(ev.textposition)[0] == "middle right" and ev.x[0] == 0.0 and ev.text[0].startswith(" ")
    cov = next(t for t in fig.data if t.mode == "markers" and np.isclose(t.x[0], 0.9059, atol=1e-4))
    _, clo, chi = S.wilson(0.9059 * 7236, 7236, steps=15)
    assert np.isclose(cov.error_x.array[0], chi - 0.9059, atol=1e-4)
    text = fig.layout.title.text
    assert "95% interval" in text and "β=0" in text and "logreg_lags" in text
    assert "the runs share one test block" in text and "[A]" not in str(fig.layout.yaxis.ticktext)
    assert any("not in these runs' eval reports" in (a.text or "") for a in fig.layout.annotations)
    for t in fig.data:                                                  # no colour cycling into other horizons
        if getattr(t, "marker", None) is not None and t.marker.color is not None:
            assert t.marker.color in (T.HORIZON_COLORS["h1"], T.INK_2, T.NEUTRAL)
    lines = re.sub(r"</?span[^>]*>", "", text).split("<br>")[1:]
    assert len(lines) > 3 and all(len(x) <= 140 for x in lines)        # wrapped to fit a notebook cell


def test_runs_comparison_says_when_runs_come_from_different_blocks(tmp_path):
    from neural_trade.experiments.compare import compare_runs
    from neural_trade.visualization.comparison import block_key, runs_comparison_figure

    # two periods (the ablation's P1 / P2): equal n, different realised outcomes. In P1 the best
    # baseline is class_prior at chance (logreg_lags is below 0.5), so no diamond is drawn there
    p1 = ["20260923T195537Z-6dec27a-53cbe6af-only-X__s0__P1", "20260923T200108Z-6dec27a-c41d9410-without-X__s0__P1"]
    p2 = ["20260923T201204Z-6dec27a-f2f93af0-only-X__s0__P2", "20260923T201903Z-6dec27a-aa4aa01b-without-X__s0__P2"]
    for rid in p1:
        _write_run(tmp_path, rid, 0.49, -0.01, 0.001, 0.93, 0.1, up=0.462, n_masked=5381, rmse=270.05, logreg=0.492)
    for rid in p2:
        _write_run(tmp_path, rid, 0.51, 0.01, 0.001, 0.91, 0.1)
    reps = {rid: json.loads((tmp_path / rid / "eval_report_test.json").read_text()) for rid in p1 + p2}
    assert block_key(reps[p1[0]]) == block_key(reps[p1[1]]) != block_key(reps[p2[0]])
    metrics = ["h1/direction/auc", "h1/variance/coverage90"]
    runs = compare_runs(str(tmp_path / "*"), metrics=metrics, skip_unscored=True)
    fig = runs_comparison_figure(runs, metrics, run_dirs=str(tmp_path / "*"))
    text = fig.layout.title.text
    assert "2 different test blocks" in text and "share one" not in text
    assert "h1 up-rate 0.462" in text and "h1 up-rate 0.515" in text
    ticks = list(fig.layout.yaxis.ticktext)
    assert sum(t.endswith("[A]") for t in ticks) == 2 and sum(t.endswith("[B]") for t in ticks) == 2
    base = next(t for t in fig.data if (t.name or "").startswith("best baseline") and t.xaxis in (None, "x"))
    assert len(base.x) == 2                                            # P2 only: P1's best is at chance
    assert "none in the other 2 (best baseline on the reference)" in text


def test_runs_comparison_finds_the_reports_without_run_dirs(tmp_path, monkeypatch):
    from neural_trade.experiments.compare import compare_runs
    from neural_trade.visualization.comparison import find_run_dirs, runs_comparison_figure

    nested = tmp_path / "runs" / "ablations" / "abl" / "runs"
    _write_run(tmp_path / "runs", IDS[0], 0.5172, 0.0088, 0.0, 0.9059, 0.0)
    _write_run(nested, IDS[2], 0.4966, -0.0165, 0.00046, 0.9081, 0.02)
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")                          # as notebook 05 runs
    found, root = find_run_dirs([IDS[0], IDS[2], "missing"])
    assert root == tmp_path / "runs" and set(found) == {IDS[0], IDS[2]}
    metrics = ["h1/direction/auc", "h1/delta/ev"]
    runs = compare_runs([str(tmp_path / "runs" / IDS[0]), str(nested / IDS[2])], metrics=metrics)
    fig = runs_comparison_figure(runs, metrics)                        # no run_dirs=
    auc = next(t for t in fig.data if t.mode == "markers" and t.name.startswith("h1"))
    assert np.all(np.asarray(auc.error_x.array) > 0)                   # intervals from the found reports
    assert "looked up by run id" in fig.layout.title.text
    assert any(x.strip() == "0 (β=0)" for t in fig.data if t.mode == "text" for x in t.text)


def test_runs_comparison_groups_horizons_and_stays_small(tmp_path, monkeypatch):
    from neural_trade.visualization.comparison import runs_comparison_figure

    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(6)
    ids = [f"20260923T{2000 + i:04d}00Z-6dec27a-{i:08x}-arm{i}" for i in range(80)]
    cols = {f"{h}/{k}": rng.normal(0.5, 0.02, 80) for h in T.HORIZONS
            for k in ("direction/auc", "direction/mcc", "delta/ev", "variance/crpss", "variance/coverage90")}
    fig = runs_comparison_figure(pd.DataFrame(cols, index=ids))
    legend = {t.name for t in fig.data if t.showlegend}
    assert legend == {"h0", "h1", "h2"}
    assert not [t for t in fig.data if t.mode == "text"]               # 80 runs: values on hover only
    assert len([a for a in fig.layout.annotations if (a.text or "").startswith("<b>")]) == 5   # one panel per kind
    assert len(fig.to_json()) < SIZE_BUDGET


def test_text_beside_a_dot_clears_its_cap_and_the_baseline_diamond():
    from neural_trade.visualization.comparison import _DOT_GAP, _text_start

    assert _text_start(0.60, 0.575, True, False) == (0.60, _DOT_GAP)   # diamond past the cap: after it
    assert _text_start(0.52, 0.575, True, False) == (0.575, " ")        # diamond inside the interval
    assert _text_start(None, 0.0, True, True) == (0.0, _DOT_GAP)        # no interval: clear of the dot
    assert _text_start(-0.2, -0.1, False, False) == (-0.2, _DOT_GAP)    # left side, same rule


def test_runs_comparison_grouped_zero_tags_sit_beside_their_own_dot(tmp_path, monkeypatch):
    from neural_trade.visualization.comparison import runs_comparison_figure

    monkeypatch.chdir(tmp_path)
    cols = {f"{h}/delta/ev": [0.0, 0.0, 0.004 * (k - 1)] for k, h in enumerate(T.HORIZONS)}
    fig = runs_comparison_figure(pd.DataFrame(cols, index=IDS))
    dots = {t.name[:2]: t for t in fig.data if t.mode == "markers"}
    texts = [t for t in fig.data if t.mode == "text"]
    assert texts and all(p in ("middle right", "middle left") for t in texts for p in t.textposition)
    for t in texts:                                                     # same row offset as its horizon's dots
        h = next(h for h, d in dots.items() if t.textfont.color == d.marker.color)
        assert set(np.round(t.y, 6)) <= set(np.round(dots[h].y, 6))
        for x, p, s in zip(t.x, t.textposition, t.text):
            if s.strip().startswith("0 ("):                              # a zero sits on the dashed line
                assert x == 0.0 and p == "middle right" and s.startswith(" ")
    lo, hi = fig.layout.xaxis.range
    assert hi > 0.004 + 0.08 * 0.012                                   # room for the text right of the dots

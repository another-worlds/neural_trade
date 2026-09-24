"""Notebook front-ends (neural_trade.notebook) and their figures, driven headlessly."""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from neural_trade.visualization.calibration_plots import reliability_figure, reliability_table


def test_reliability_table_is_flat_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    t = reliability_table(y, p, 10)
    assert t.shape == (10, 5) and np.all(np.abs(t[:, 0] - t[:, 1]) < 0.03) and t[:, 2].sum() == p.size
    fig = reliability_figure(y, p, np.clip(p * 1.1, 0, 1))
    assert {"perfect", "raw P(up)", "calibrated P(up)"} <= {tr.name for tr in fig.data}
    assert all(0 <= v <= 1 for v in fig.layout.xaxis.range)          # zoomed to the bins, inside [0, 1]


def test_comparison_and_ablation_figures(tmp_path, monkeypatch):
    from neural_trade.visualization.comparison import ablation_deltas_figure, runs_comparison_figure

    monkeypatch.chdir(tmp_path)                                  # no runs/ folder to look the ids up in

    df = pd.DataFrame({"h1/direction/auc": [0.51, 0.53], "backtest/sharpe_net": [-3.0, -1.0], "seed": [0, 1]},
                      index=["run-a", "run-b"])
    fig = runs_comparison_figure(df)
    dots = [t for t in fig.data if t.mode == "markers"]
    assert len(dots) == 2 and [len(t.x) for t in dots] == [2, 2]              # one dot per run and panel
    assert list(fig.layout.yaxis.ticktext) == ["run-a", "run-b"]
    analysis = {"terms": {"LAMBDA_HD": {"verdict": "NEUTRAL", "modes": {"leave_one_in": {"verdict": "NEUTRAL", "metrics": [
        {"metric": "h1/variance/crpss", "mean_delta": 0.001, "sd_delta": 0.002, "verdict": "NEUTRAL"}]}}}},
                "family": {"verdict": "VALUE", "metrics": [{"metric": "h1/direction/mcc", "mean_delta": 0.02,
                                                            "sd_delta": 0.01, "verdict": "VALUE"}]}}
    fig = ablation_deltas_figure(analysis)
    assert {t.name for t in fig.data} == {"NEUTRAL", "VALUE"}


def test_split_table_and_overview(synthetic_bars, tmp_path):
    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.visualization.data_overview import split_overview_figure, split_table

    synthetic_bars.to_csv(tmp_path / "b.csv", index=False)
    cfg = Config(CSV_PATH=str(tmp_path / "b.csv"), MAX_SEQUENCE_COUNT=2000)
    blocks = split_arrays(cfg)
    t = split_table(blocks, cfg)
    assert list(t.index) == ["train", "val", "cal", "test"] and (t["from"] <= t["to"]).all()
    assert t.loc["val", "from"] > t.loc["train", "to"] and t.loc["test", "from"] > t.loc["cal", "to"]  # purged, ordered
    assert 0 < t.loc["test", "up-rate h1"] < 1
    assert len(split_overview_figure(blocks, cfg).layout.shapes) == 4


def test_session_pause_blocks_the_training_thread_until_resume():
    from neural_trade.core.config import Config
    from neural_trade.notebook import TrainingSession

    s = TrainingSession(Config(EPOCHS=1))
    s.status = "training"
    cb = s._callback()

    class _M:
        stop_training = False

    cb.model = _M()
    s.pause()
    assert s.status == "paused"
    t = threading.Thread(target=cb.on_train_batch_end, args=(1,))
    t.start()
    time.sleep(0.6)
    assert t.is_alive(), "a paused session must hold the training thread"
    s.resume()
    t.join(2)
    assert not t.is_alive() and not cb.model.stop_training
    s.stop()
    cb.on_train_batch_end(2)
    assert cb.model.stop_training


# ---------------------------------------------------------------------------- one real run, three front-ends
@pytest.fixture(scope="module")
def session_run(tmp_path_factory, synthetic_bars):
    from neural_trade.core.config import Config
    from neural_trade.experiments.run_context import RunContext
    from neural_trade.notebook import TrainingSession

    tmp = tmp_path_factory.mktemp("ui")
    csv = tmp / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    cfg = Config(EPOCHS=2, BATCH_SIZE=32, MAX_SEQUENCE_COUNT=1500, CSV_PATH=str(csv),
                 CALLBACKS=["early_stopping", "tqdm_progress"])
    ctx = RunContext.create(cfg, root=tmp / "runs")
    session = TrainingSession(cfg, run_context=ctx, epochs=2, calibrate=False)
    session.widget()
    session.start()
    result = session.wait(timeout=900)
    return session, result, ctx, csv


@pytest.mark.slow
def test_training_session_trains_in_a_thread_with_live_widgets(session_run):
    session, result, ctx, _ = session_run
    assert session.status == "finished" and result is not None and len(session.history) == 2
    assert {"loss", "val_loss", "seconds"} <= set(session.history[-1])
    assert session._w["epochs"].value == 2 and session._w["curves"].outputs  # curves were drawn
    assert any("Training on" in o.get("text", "") or "Epoch" in o.get("text", "") or o.get("text")
               for o in session._w["log"].outputs)               # the thread's log went to the panel
    assert (ctx.run_dir / "artifacts" / "weights.h5").exists()
    assert "tqdm_progress" not in ctx.config.CALLBACKS            # no console bars from a thread


@pytest.mark.slow
def test_stop_ends_training_early_but_still_evaluates(session_run, tmp_path):
    from neural_trade.notebook import TrainingSession

    session, _, ctx, csv = session_run
    s2 = TrainingSession(ctx.config.copy().override(MODEL_PATH=str(tmp_path / "w.h5"),
                                                    SCALER_PATH=str(tmp_path / "s.joblib")),
                         epochs=5, calibrate=False, save_artifacts=False)
    s2.start()
    s2.stop()
    res = s2.wait(timeout=900)
    assert res is not None and s2.status == "stopped early - evaluated" and len(s2.history) <= 1
    assert "delta" in res.predictions


@pytest.mark.slow
def test_backtest_explorer_runs_every_strategy_from_the_widget(session_run):
    from neural_trade.notebook import BacktestExplorer
    from neural_trade.strategy import Strategies

    _, _, ctx, csv = session_run
    ex = BacktestExplorer.from_run(ctx.run_dir, csv_path=str(csv))
    box = ex.widget()
    for name in Strategies.list_names():
        ex._w["strategy"].value = name                       # rebuilds the knob widgets
        assert all(hasattr(k, "_field") for k in ex._w["knobs"].children)
        ex._w["costs"]["random_seeds"].value = 2
        ex.click_run()
        assert ex.last is not None and ex.last.strategy == name, ex._w["status"].value
    table = ex.summary_frame()
    assert {"buy_and_hold", "always_flat"} <= set(table.index)
    assert any(i.startswith("random same freq") for i in table.index)
    res = ex.run("calibrated_quantile", {"entry_quantile": 0.8}, {"fee_bps": 0.0, "random_seeds": 0})
    assert res.config.fee_bps == 0.0 and box is ex.widget()


def test_backtest_explorer_compare_strategies_keeps_the_last_run(viz_frame, viz_backtest):
    """compare_strategies must not replace the explorer's current result: dashboard(), trade_analytics()
    and summary_frame() after it still show the strategy the user ran (not the last one compared)."""
    import copy
    from types import SimpleNamespace

    from neural_trade.core.config import Config
    from neural_trade.notebook import BacktestExplorer

    _, bars, _, _ = viz_backtest
    fr = copy.deepcopy(viz_frame)
    fr.last_close = bars.close
    ex = BacktestExplorer({"config": Config(), "test": fr, "cal": fr, "bars": bars,
                           "predictor": SimpleNamespace(bundle=SimpleNamespace(meta={"var_scale": 1.0}))})
    res = ex.run("calibrated_quantile", costs={"random_seeds": 2})
    strat = ex.last_strategy
    runs, _ = ex.compare_strategies(null_seeds=2)
    assert ex.last is res and ex.last_strategy is strat and ex.last.baselines
    assert ex.summary_frame().index[0] == "calibrated_quantile"
    assert runs["calibrated_quantile"] is not res


def test_backtest_explorer_compares_each_strategy_with_its_matched_random_null(viz_frame, viz_backtest):
    import copy
    import dataclasses
    from types import SimpleNamespace

    from neural_trade.core.config import Config
    from neural_trade.notebook import BacktestExplorer
    from neural_trade.notebook.backtest_ui import matched_random_null
    from neural_trade.visualization import theme as T

    _, bars, _, _ = viz_backtest
    fr = copy.deepcopy(viz_frame)
    fr.last_close = bars.close
    ex = BacktestExplorer({"config": Config(), "test": fr, "cal": fr, "bars": bars,
                           "predictor": SimpleNamespace(bundle=SimpleNamespace(meta={"var_scale": 1.0}))})
    runs, fig = ex.compare_strategies(null_seeds=3)
    # the unmatched random_signal (5% entries, 10-bar hold, one seed) is not the default null
    assert "random_signal" not in runs and {"buy_and_hold", "always_flat"} <= set(runs)
    traded = [n for n, r in runs.items() if n not in ("buy_and_hold", "always_flat") and r.summary["n_trades"]]
    assert traded
    for n in traded:
        null = runs[n].baselines["random_same_freq"]
        sizes = [d["size"] for d in runs[n].decisions]
        assert null["n_seeds"] == 3 and null["size_frac"] == pytest.approx(np.mean(sizes))
        assert null["random_p05_total_return"] <= null["random_mean_total_return"] <= null["random_p95_total_return"]
    assert sum("beats" in t for t in fig.layout.yaxis2.ticktext) == len(traded)
    assert T.empty_panels(fig) == []
    # asked for explicitly, random_signal is labelled with its knobs
    _, fig2 = ex.compare_strategies(["random_signal"], null_seeds=0)
    assert "random_signal (5%/bar, 10-bar hold, seed 0)" in fig2.layout.yaxis2.ticktext[0]
    table = ex.comparison_table(runs)
    assert {"avg_win", "avg_loss", "expectancy", "n_long", "n_short", "exits",
            "random percentile (return)"} <= set(table.columns)
    # a strategy that sizes down is compared with random entries of the same size (less cost drag)
    half = ex.run("calibrated_quantile", {"size": 0.5}, {"random_seeds": 4})
    null = half.baselines["random_same_freq"]
    assert null["size_frac"] == pytest.approx(0.5) and null["n_seeds"] == 4
    full = matched_random_null(ex.signals, ex.bars, dataclasses.replace(
        half, decisions=[dict(d, size=1.0) for d in half.decisions]), seeds=4)
    assert full["size_frac"] == 1.0 and null["random_mean_total_return"] > full["random_mean_total_return"]


def test_backtest_explorer_run_and_compare_report_the_same_random_rank(viz_frame, viz_backtest):
    """run() / summary_frame() and compare_strategies() answer the same question (where does the
    strategy rank among random entries?), so they use the same matched null with the same seeds:
    by default the costs' random_seeds in both."""
    import copy
    from types import SimpleNamespace

    from neural_trade.core.config import Config
    from neural_trade.notebook import BacktestExplorer

    _, bars, _, _ = viz_backtest
    fr = copy.deepcopy(viz_frame)
    fr.last_close = bars.close
    ex = BacktestExplorer({"config": Config(), "test": fr, "cal": fr, "bars": bars,
                           "predictor": SimpleNamespace(bundle=SimpleNamespace(meta={"var_scale": 1.0}))})
    costs = {"random_seeds": 6}
    res = ex.run("calibrated_quantile", costs=costs)
    runs, fig = ex.compare_strategies(["calibrated_quantile"], costs=costs)     # null_seeds defaults to 6
    a, b = res.baselines["random_same_freq"], runs["calibrated_quantile"].baselines["random_same_freq"]
    assert a["n_seeds"] == b["n_seeds"] == 6 and "size_frac" in a
    for k in ("percentile_total_return", "percentile_gross_return", "random_mean_total_return",
              "random_p05_total_return", "random_p95_total_return"):
        assert a[k] == pytest.approx(b[k]), k
    assert f"beats {a['percentile_total_return']:.0f}% of random" in fig.layout.yaxis2.ticktext[0]
    # summary_frame shows the rank before and after costs and the null's 5-95% band
    table = ex.summary_frame()
    row = table.loc["calibrated_quantile"]
    assert row["random percentile (return)"] == pytest.approx(a["percentile_total_return"])
    assert row["random percentile (gross)"] == pytest.approx(a["percentile_gross_return"])
    null_row = table.loc[[i for i in table.index if i.startswith("random same freq (mean of 6 seeds")][0]]
    assert null_row["random p05 (return)"] == pytest.approx(a["random_p05_total_return"])
    assert null_row["random p95 (return)"] == pytest.approx(a["random_p95_total_return"])
    assert null_row["total_return"] == pytest.approx(a["random_mean_total_return"])
    assert res.config.random_seeds == 6          # the result keeps the user's config
    # the comparison table carries the same columns
    ct = ex.comparison_table(runs).loc["calibrated_quantile"]
    assert ct["random p05 (return)"] == pytest.approx(a["random_p05_total_return"])
    # the widget's default seed count is the engine default, which compare_strategies also uses
    from neural_trade.strategy import BacktestConfig

    ex.widget()
    assert ex._w["costs"]["random_seeds"].value == BacktestConfig().random_seeds


def test_backtest_explorer_tables_and_trade_analytics(viz_frame, viz_backtest):
    import copy
    from types import SimpleNamespace

    from neural_trade.core.config import Config
    from neural_trade.notebook import BacktestExplorer
    from neural_trade.visualization import theme as T

    _, bars, _, _ = viz_backtest
    fr = copy.deepcopy(viz_frame)
    fr.last_close = bars.close
    ex = BacktestExplorer({"config": Config(), "test": fr, "cal": fr, "bars": bars,
                           "predictor": SimpleNamespace(bundle=SimpleNamespace(meta={"var_scale": 1.0}))})
    res = ex.run("calibrated_quantile", costs={"random_seeds": 0})
    row = ex.summary_frame().loc["calibrated_quantile"]
    net = np.array([t.net_pnl for t in res.trades])
    assert row["expectancy"] == pytest.approx(net.mean()) and row["avg_loss"] == pytest.approx(net[net <= 0].mean())
    assert row["n_long"] + row["n_short"] == len(res.trades)
    fig = ex.trade_analytics()
    assert T.empty_panels(fig) == []                            # the signal panels get the explorer's signals
    assert "same 15 bars" in fig.layout.yaxis9.title.text       # ... and the horizon steps (like for like)
    summ = ex.signal_summary()
    flags = summ["flags (share of bars true)"]
    assert set(flags.index) == {"magnitude_coherent", "direction_aligned", "var_spike"}   # describe() drops bools
    assert flags["share true"].between(0, 1).all()
    assert "weighted_move_$" in summ["features"].index
    assert summ["consensus"]["share of bars"].sum() == pytest.approx(1.0)
    assert summ["confidence scale"]["var_scale"].iloc[0] == 1.0


def _explorer(viz_frame, bars, test=None):
    import copy
    from types import SimpleNamespace

    from neural_trade.core.config import Config
    from neural_trade.notebook import BacktestExplorer

    fr = copy.deepcopy(viz_frame)
    fr.last_close = bars.close
    return BacktestExplorer({"config": Config(), "test": fr if test is None else test, "cal": fr, "bars": bars,
                             "predictor": SimpleNamespace(bundle=SimpleNamespace(meta={"var_scale": 1.0}))}), fr


def test_backtest_explorer_trade_analytics_scores_the_raw_head_when_beta_is_zero(viz_frame, viz_backtest):
    """load_run_blocks puts the raw heads and the shrink betas on the served test frame; with beta = 0
    the served delta is 0 on every bar, so trade_analytics must hand the raw heads to the figure
    (else its predicted-vs-realised panel is one vertical line at $0 with 'ρ +nan, sign right 0%')."""
    import copy

    _, bars, _, _ = viz_backtest
    _, fr = _explorer(viz_frame, bars)
    served = copy.deepcopy(fr)
    served.delta = {h: np.zeros_like(v) for h, v in fr.delta.items()}
    served.meta = {"delta_raw": {h: v.copy() for h, v in fr.delta.items()},
                   "delta_scale": {"h0": 0.0, "h1": 0.0, "h2": 0.0}}
    ex, _ = _explorer(viz_frame, bars, test=served)
    res = ex.run("calibrated_quantile", costs={"random_seeds": 0}, baselines=False)
    assert res.summary["n_trades"] > 5
    fig = ex.trade_analytics()
    title = fig.layout.annotations[8].text
    assert title.startswith("Raw h1 head vs realised move") and "β = 0" in title and "nan" not in title
    x = np.concatenate([np.asarray(t.x, float) for t in fig.data if t.xaxis == "x9" and t.mode == "markers"])
    sign = {"LONG": 1, "SHORT": -1}
    steps = int(ex.config.HORIZON_STEPS[1])                  # trades whose h1 window runs past the block are not scored
    want = [sign[s] * fr.delta["h1"][t.entry_bar - 1] for s in ("LONG", "SHORT") for t in res.trades
            if t.side == s and t.entry_bar - 1 + steps < len(fr)]
    np.testing.assert_allclose(x, want, rtol=1e-5, atol=1e-4)
    # a frame without raw heads or betas (an older run) still draws the served delta
    ex2, _ = _explorer(viz_frame, bars)
    ex2.run("calibrated_quantile", costs={"random_seeds": 0}, baselines=False)
    assert ex2.trade_analytics().layout.annotations[8].text.startswith("Predicted vs realised h1 move")


def test_backtest_explorer_tables_keep_numbers_numeric_and_show_every_metric(viz_frame, viz_backtest):
    """The comparison table's text 'exits' column must not turn every column into object dtype (round(4)
    then does nothing), the summary keeps its 'not run' null row, and styled=True shows every metric:
    one row per metric (the one-row-per-strategy frames have 24-25 columns, which pandas cuts with '...')."""
    from pandas.api.types import is_numeric_dtype
    from pandas.io.formats.style import Styler

    from neural_trade.notebook.backtest_ui import _METRIC_ROWS

    _, bars, _, _ = viz_backtest
    ex, _ = _explorer(viz_frame, bars)
    runs, _ = ex.compare_strategies(null_seeds=2)
    ct = ex.comparison_table(runs)
    assert [c for c in ct.columns if not is_numeric_dtype(ct[c])] == ["exits"]
    assert ct["n_trades"].dtype.kind == "i"
    r4 = ct.round(4)
    num = [c for c in ct.columns if c != "exits"]
    np.testing.assert_allclose(r4[num].to_numpy(float), np.round(ct[num].to_numpy(float), 4), equal_nan=True)
    sty = ex.comparison_table(runs, styled=True)
    assert isinstance(sty, Styler) and sty.data.shape == (ct.shape[1], ct.shape[0])
    assert list(sty.data.columns) == list(ct.index)
    labels = {k: label for k, label, _ in _METRIC_ROWS}
    html = sty.to_html()
    for c in ct.columns:                                  # every metric has its own, labelled row
        assert f">{labels[c]}<" in html.replace("&amp;", "&"), c
    name = next(n for n, r in runs.items() if r.summary["n_trades"])
    assert f"{runs[name].summary['total_return']:+.2%}" in html          # returns in %, 2 decimals
    assert f">{runs[name].summary['n_trades']:,}<" in html                  # counts as integers
    with pd.option_context("display.max_columns", 20, "display.max_rows", 60):
        assert "..." not in sty._repr_html_() and "…" not in sty._repr_html_()

    res = ex.run("calibrated_quantile", costs={"random_seeds": 0})
    plain = ex.summary_frame()
    assert all(is_numeric_dtype(plain[c]) for c in plain.columns)
    assert "random same freq (not run: no trades or 0 seeds)" in plain.index         # the empty row stays
    sty = ex.summary_frame(styled=True)
    html = sty.to_html().replace("&amp;", "&")
    for label in ("avg win ($)", "long trades", "short trades", "long P&L before costs ($)",
                  "short P&L after costs ($)", "exposure (share of bars in a position)"):
        assert f">{label}<" in html, label
    assert ">–<" in html and ">nan<" not in html.lower()                  # no trades / not measured: '–'
    net = np.array([t.net_pnl for t in res.trades])
    assert f"{net.mean():+,.2f}" in html                                           # expectancy, $ with 2 decimals
    # the widget's summary table is the styled one: every metric is in it (no '...' column)
    ex.widget()
    ex._w["costs"]["random_seeds"].value = 0
    ex.click_run()
    shown = ex._w["box"].children[2].outputs[0]["data"]["text/html"].replace("&amp;", "&")
    assert all(f">{label}<" in shown for label in ("long trades", "short trades", "costs paid ($)"))
    assert "..." not in shown


def test_metrics_view_prints_a_negative_zero_unsigned_and_an_infinite_profit_factor_as_infinity():
    """No sign on an exact 0, a negative zero included ('-0.00%' read as a loss), and a profit factor
    with no losing trade (gains / 0: buy-and-hold with one winning trade) prints as '∞', not 'inf'."""
    from neural_trade.notebook.backtest_ui import _cell_format, metrics_view

    assert _cell_format("{:+.2%}")(-0.0) == "0.00%"
    assert _cell_format("{:+,.0f}")(np.float64(-0.0)) == "0"
    assert _cell_format("{:.1%}")(-0.0) == "0.0%"
    assert _cell_format("{:+.2%}")(-0.0123) == "-1.23%" and _cell_format("{:+.2%}")(0.0123) == "+1.23%"
    assert _cell_format("{:.2f}")(float("inf")) == "∞" and _cell_format("{:+,.2f}")(-np.inf) == "-∞"
    frame = pd.DataFrame({"n_trades": [1, 0], "total_return": [-0.0, 0.0], "gross_pnl": [-0.0, 0.0],
                          "profit_factor": [np.inf, np.nan]}, index=["buy_and_hold", "always_flat"])
    html = metrics_view(frame).to_html()
    assert ">-0" not in html and ">+0" not in html
    assert html.count(">0.00%<") == 2 and html.count(">0<") == 3                   # 2 returns; 2 P&L + 1 count
    assert ">∞<" in html and ">inf<" not in html and ">–<" in html                  # always_flat: no trades


def test_backtest_tables_say_what_a_dash_and_an_infinity_stand_for(viz_frame, viz_backtest):
    """'–' also marks a strategy that traded with no winning (or no losing) trade: avg win / avg loss
    have nothing to average. '∞' is explained when a profit factor is infinite (no losing trade)."""
    import dataclasses

    from neural_trade.notebook.backtest_ui import _caption

    _, bars, _, _ = viz_backtest
    ex, _ = _explorer(viz_frame, bars)
    runs, _ = ex.compare_strategies(null_seeds=2)
    cap = ex.comparison_table(runs, styled=True).caption
    assert "no winning / losing trade (avg win / avg loss)" in cap and "no matched random null (not run)" in cap
    assert ("∞" in cap) == bool(np.isinf(ex.comparison_table(runs)["profit_factor"].to_numpy(float)).any())
    name = next(n for n, r in runs.items() if r.summary["n_trades"] > 1)
    wins = [t for t in runs[name].trades if t.net_pnl > 0]
    runs[name] = dataclasses.replace(runs[name], trades=wins, summary=dict(runs[name].summary, n_trades=len(wins),
                                                                           profit_factor=float("inf")))
    sty = ex.comparison_table(runs, styled=True)
    assert "∞: no losing trade (profit factor)" in sty.caption
    html = sty.to_html()
    assert ">∞<" in html and ">inf<" not in html
    assert _caption(pd.DataFrame({"n_trades": [3]}), "x").endswith(", or x")         # no profit factor column
    res = ex.run("calibrated_quantile", costs={"random_seeds": 0})
    cap = ex.summary_frame(styled=True).caption
    assert "no winning / losing trade (avg win / avg loss)" in cap and "the random null is a distribution" in cap
    assert "buy-and-hold and always-flat keep their run summary only" in cap        # their per-trade rows are '–'
    # a baseline's gross return is in its run summary: shown, not '–'
    plain = ex.summary_frame()
    for name in ("buy_and_hold", "always_flat"):
        want = res.baselines[name]["gross_pnl"] / res.config.initial_equity
        assert plain.loc[name, "gross_return"] == pytest.approx(want), name


@pytest.mark.slow
def test_calibration_explorer_refits_and_scores_on_test(session_run):
    from neural_trade.notebook import CalibrationExplorer

    _, _, ctx, csv = session_run
    ex = CalibrationExplorer.from_run(ctx.run_dir, csv_path=str(csv))
    for scale, shrink in (("realized_vol", True), ("none", False)):
        t = ex.refit(scale, shrink_delta=shrink, alpha=0.2)
        assert list(t.index) == ["h0", "h1", "h2"] and t["coverage"].between(0, 1).all()
        if not shrink:
            assert (t["delta beta"] == 1.0).all() and np.allclose(t["EV raw delta"], t["EV served delta"])
    rel, cov = ex.figures("h2")
    assert {"raw P(up)", "calibrated P(up)"} <= {t.name for t in rel.data}
    assert {"coverage", "target 0.80"} <= {t.name for t in cov.data}     # the target follows alpha=0.2
    ex.click_refit()
    assert list(ex.comparison_table().index.get_level_values(0).unique()) == ["h0", "h1", "h2"]


def test_pick_run_finds_the_newest_servable_run_or_explains(tmp_path):
    import os

    from neural_trade.notebook import pick_run, servable_runs

    with pytest.raises(FileNotFoundError, match="notebook 01_train_and_monitor"):
        pick_run(None, tmp_path)
    for i, name in enumerate(("gates/m6", "20260101-a", "20260102-b")):
        (tmp_path / name / "artifacts").mkdir(parents=True)
        (tmp_path / name / "artifacts" / "weights.h5").write_bytes(b"")
        os.utime(tmp_path / name, (1_000 + i, 1_000 + i))
    (tmp_path / "no_bundle").mkdir()
    assert [p.name for p in servable_runs(tmp_path)] == ["20260102-b", "20260101-a", "m6"]
    assert pick_run(None, tmp_path).name == "20260102-b"
    assert pick_run(tmp_path / "gates" / "m6", tmp_path).name == "m6"
    with pytest.raises(FileNotFoundError, match="no serving bundle"):
        pick_run(tmp_path / "no_bundle", tmp_path)


def test_show_puts_figures_and_tables_into_the_widget_itself():
    import ipywidgets as w
    import plotly.graph_objects as go

    from neural_trade.notebook._display import show

    out = w.Output()
    show(out, go.Figure(go.Scatter(y=[1, 2, 3]), layout_title_text="curves"))
    (o,) = out.outputs
    assert "application/vnd.plotly.v1+json" in o["data"]
    assert "text/html" not in o["data"]                      # the figure is stored once, not twice
    show(out, pd.DataFrame({"a": [1]}))                      # replaces, never appends
    (o,) = out.outputs
    assert "<table" in o["data"]["text/html"]

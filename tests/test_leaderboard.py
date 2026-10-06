"""NT-031: the leaderboard (ranked by dev-fold net Sharpe after costs, guard-rails, test columns
that never rank). Builds fake run-index rows (the shape of neural_trade.experiments.store.RunIndex
.rows()) instead of training, so these tests are fast and do not touch the GPU."""
from __future__ import annotations

import json
import math

import pytest

from neural_trade.experiments.leaderboard import (
    DEFAULT_GUARD_RAILS, GuardRailSpec, build_leaderboard, leaderboard_markdown, winner,
)
from neural_trade.experiments.store import RunStore

SCENARIO = "unit_test_scenario"


def _row(configuration="default", fold=-2, role="dev", seed=0, status="done", **overrides):
    row = {
        "scenario": SCENARIO, "configuration": configuration, "fold": fold, "role": role, "seed": seed,
        "status": status, "error": None, "dataset_sha256": "abc123def456abc123def456", "bar_minutes": 1.0,
        "horizon_steps": json.dumps([10, 15, 20]), "strategy": "calibrated_quantile",
        "sharpe_net": 1.0, "total_return": 0.05, "max_drawdown": 0.05, "n_trades": 20.0,
        "buy_and_hold_return": 0.01, "random_percentile_return": 80.0,
        "fee_bps": 0.0, "half_spread_bps": 0.0, "slippage_bps": 0.0,      # D-044: the board's profile
    }
    row.update(overrides)
    return row


def _good_configuration(name="good", dev_folds=(-3, -2), test_fold=-1, sharpe=1.0, n_trades=20.0):
    rows = []
    for f in dev_folds:
        for s in (0, 1):
            rows.append(_row(configuration=name, fold=f, role="dev", seed=s,
                             sharpe_net=sharpe, n_trades=n_trades))
    rows.append(_row(configuration=name, fold=test_fold, role="test", seed=0, sharpe_net=sharpe - 0.5))
    return rows


# ------------------------------------------------------------------ criterion 1
def test_one_row_per_configuration_with_dev_mean_spread_and_test_columns():
    # two folds x two seeds each at different sharpes within the same configuration -> real spread
    rows = _row(configuration="default", fold=-3, role="dev", seed=0, sharpe_net=1.0), \
        _row(configuration="default", fold=-3, role="dev", seed=1, sharpe_net=1.2), \
        _row(configuration="default", fold=-2, role="dev", seed=0, sharpe_net=2.0), \
        _row(configuration="default", fold=-2, role="dev", seed=1, sharpe_net=2.2), \
        _row(configuration="default", fold=-1, role="test", seed=0, sharpe_net=0.5)
    board = build_leaderboard(list(rows))
    assert len(board) == 1
    row = board[0]
    assert row.configuration == "default"
    # fold means: -3 -> 1.1, -2 -> 2.1; overall mean 1.6, n_folds=2, n_rows=4
    assert row.dev.values["sharpe_net"] == pytest.approx(1.6)
    assert row.dev.n_folds == 2
    assert row.dev.n_rows == 4
    assert row.dev.spread["sharpe_net"] is not None and row.dev.spread["sharpe_net"] > 0
    assert row.test.values["sharpe_net"] == pytest.approx(0.5)
    assert row.test.n_folds == 1
    assert row.test.n_rows == 1


# ------------------------------------------------------------------ criterion 2
def test_sort_order_follows_dev_sharpe_even_when_test_order_differs():
    rows = []
    # "alpha" has the worse dev Sharpe but the better test Sharpe; "beta" the opposite
    rows += _good_configuration("alpha", sharpe=0.5)
    rows[-1]["sharpe_net"] = 9.0          # alpha's test row: best test Sharpe
    rows += _good_configuration("beta", sharpe=3.0)
    rows[-1]["sharpe_net"] = -9.0         # beta's test row: worst test Sharpe
    board = build_leaderboard(rows)
    assert [r.configuration for r in board] == ["beta", "alpha"]     # dev order: beta (3.0) before alpha (0.5)
    assert board[0].test.values["sharpe_net"] == pytest.approx(-9.0)  # test order is the opposite
    assert board[1].test.values["sharpe_net"] == pytest.approx(9.0)


# ------------------------------------------------------------------ criterion 3
def test_guard_rail_breach_disqualifies_and_names_the_failing_guard_rail():
    # zero trades: the default min_trades guard-rail (NT-076 QA note) must disqualify it
    rows = _good_configuration("zero_trade", n_trades=0.0)
    board = build_leaderboard(rows)
    row = board[0]
    assert row.disqualified is True
    failing = [g.name for g in row.guard_rails if not g.passed]
    assert "min_trades" in failing
    assert winner(board) is None


def test_a_passing_configuration_is_eligible_as_winner():
    rows = _good_configuration("ok", sharpe=1.0, n_trades=20.0)
    # make it beat buy-and-hold and the random null explicitly
    for r in rows:
        r["total_return"], r["buy_and_hold_return"], r["random_percentile_return"] = 0.10, 0.01, 90.0
    board = build_leaderboard(rows)
    row = board[0]
    assert row.disqualified is False
    assert all(g.passed for g in row.guard_rails)
    assert winner(board) is row


def test_custom_guard_rail_spec_is_honoured():
    rows = _good_configuration("tight_dd")
    for r in rows:
        r["max_drawdown"] = 0.30
    strict = GuardRailSpec(max_drawdown_max=0.10, min_trades=None, require_beat_buy_and_hold=False,
                           require_beat_random_null=False)
    board = build_leaderboard(rows, guard_rails=strict)
    row = board[0]
    assert row.disqualified is True
    assert any(g.name == "max_drawdown" and not g.passed for g in row.guard_rails)


# ------------------------------------------------------------------ criterion 4
def test_every_row_shows_dataset_fingerprint_bar_size_horizons_and_strategy():
    rows = _good_configuration("default")
    board = build_leaderboard(rows)
    row = board[0]
    assert row.dataset_fingerprint == "abc123def456abc123def456"
    assert row.bar_minutes == 1.0
    assert row.horizon_steps == (10, 15, 20)
    assert row.strategy == "calibrated_quantile"


def test_missing_fingerprint_reports_n_a_not_an_edit_of_the_index():
    rows = _good_configuration("no_fingerprint")
    for r in rows:
        r["dataset_sha256"] = None
    board = build_leaderboard(rows)
    assert board[0].dataset_fingerprint is None
    assert "n/a" in leaderboard_markdown(board)


# ------------------------------------------------------------------ criterion 5
def test_markdown_labels_the_ranking_column_and_the_test_columns():
    rows = _good_configuration("default")
    board = build_leaderboard(rows)
    text = leaderboard_markdown(board)
    assert "ranking column" in text.lower() or "ranking: dev net sharpe" in text.lower()
    assert "not used for ranking" in text.lower()


# ------------------------------------------------------------------ criterion 6
def test_failed_run_appears_as_a_failed_row():
    rows = [_row(configuration="broken", fold=-2, role="dev", seed=0, status="failed",
                error="ValueError: boom", sharpe_net=None, total_return=None, max_drawdown=None,
                n_trades=None, buy_and_hold_return=None, random_percentile_return=None),
           _row(configuration="broken", fold=-2, role="dev", seed=1, status="failed",
                error="ValueError: boom", sharpe_net=None, total_return=None, max_drawdown=None,
                n_trades=None, buy_and_hold_return=None, random_percentile_return=None)]
    board = build_leaderboard(rows)
    assert len(board) == 1
    row = board[0]
    assert row.status == "failed"
    assert row.disqualified is True
    assert row.n_failed == 2
    assert row.errors == ("ValueError: boom",)
    assert row.dev.values["sharpe_net"] is None
    # a failed configuration sorts behind any configuration with a finite dev Sharpe
    good = _good_configuration("ok")
    board2 = build_leaderboard(rows + good)
    assert board2[-1].configuration == "broken"


def test_mixed_done_and_failed_cells_is_status_done_with_the_done_cells_aggregated():
    rows = _good_configuration("partial")
    rows.append(_row(configuration="partial", fold=-3, role="dev", seed=1, status="failed",
                     error="RuntimeError: nan loss", sharpe_net=None, total_return=None,
                     max_drawdown=None, n_trades=None, buy_and_hold_return=None,
                     random_percentile_return=None))
    board = build_leaderboard(rows)
    row = board[0]
    assert row.status == "done"
    assert row.n_failed == 1
    assert row.errors == ("RuntimeError: nan loss",)
    assert row.dev.values["sharpe_net"] is not None


# ------------------------------------------------------------------ misc
def test_empty_index_returns_empty_leaderboard():
    assert build_leaderboard([]) == []


def test_default_guard_rails_object_is_frozen_and_reusable():
    assert DEFAULT_GUARD_RAILS.min_trades == 1.0
    assert DEFAULT_GUARD_RAILS.require_beat_buy_and_hold is True


def test_dev_values_never_include_nan():
    rows = _good_configuration("default")
    board = build_leaderboard(rows)
    for v in board[0].dev.values.values():
        assert v is None or math.isfinite(v)


# ------------------------------------------------------------------ criterion 7 (CLI + figure)
def _write_run_dir(root, *, scenario, configuration, fold, role, seed, status="done", sharpe_net=1.0,
                   n_trades=10.0, error=None, cost=(0.0, 0.0, 0.0), meta_backtest=None, report=True):
    from pathlib import Path

    run_id = f"{scenario}-{configuration}-f{fold}-s{seed}"
    d = Path(root) / "scenarios" / scenario / run_id
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "engine": {"scenario": scenario, "cell_key": f"{configuration}__f{fold}__s{seed}",
                  "configuration": configuration, "variant": configuration, "params": {}, "fold": fold,
                  "fold_id": abs(fold), "role": role, "seed": seed, "config_hash": "x", "settings_hash": "y",
                  "spec_hash": "z", "commit": "deadbee", "strategy": {"name": "calibrated_quantile", "params": {}},
                  "backtest": dict(meta_backtest or {}), "run": {"calibrate": True, "save_artifacts": False}},
        "dataset": {"path": "bars.csv", "sha256": "f" * 20, "size_bytes": 1, "n_bars": 100,
                   "first_timestamp": "2026-01-01T00:00:00+00:00", "last_timestamp": "2026-01-02T00:00:00+00:00"},
        "setup": {"bar_minutes": 1, "LOOKBACK": 60, "HORIZON_STEPS": [10, 15, 20]},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    scores = ({} if status != "done" else {
        "backtest/sharpe_net": sharpe_net, "backtest/total_return": 0.02, "backtest/max_drawdown": 0.05,
        "backtest/n_trades": n_trades, "backtest/buy_and_hold/total_return": 0.01,
        "backtest/random_same_freq/percentile_total_return": 80.0})
    result = {"status": status, "error": {"type": "ValueError", "message": "boom"} if error else None,
             "scores": scores, "finished_utc": "2026-01-01T00:00:00+00:00"}
    if status == "done" and report:
        # the scorer's stored report: backtest.config holds the costs the stored net Sharpe used
        result["report"] = f"eval_report_{role}.json"
        cfg = {"fill": "next_open", "fee_bps": cost[0], "half_spread_bps": cost[1], "slippage_bps": cost[2]}
        (d / result["report"]).write_text(json.dumps({"backtest": {"config": cfg}}), encoding="utf-8")
    (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return d


def test_cli_leaderboard_prints_the_table(tmp_path, capsys):
    from neural_trade.cli import main

    _write_run_dir(tmp_path, scenario="cli_test", configuration="default", fold=-2, role="dev", seed=0)
    _write_run_dir(tmp_path, scenario="cli_test", configuration="default", fold=-1, role="test", seed=0)
    assert main(["leaderboard", "cli_test", "--store", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Leaderboard: `cli_test`" in out
    assert "default" in out
    assert "not used for ranking" in out.lower()


def test_cli_leaderboard_without_a_scenario_prints_every_scenario(tmp_path, capsys):
    from neural_trade.cli import main

    _write_run_dir(tmp_path, scenario="scenario_a", configuration="default", fold=-1, role="test", seed=0)
    _write_run_dir(tmp_path, scenario="scenario_b", configuration="default", fold=-1, role="test", seed=0)
    assert main(["leaderboard", "--store", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "scenario_a" in out and "scenario_b" in out


def test_leaderboard_registered_in_visualizations_and_no_empty_panel():
    from neural_trade.registries.visualizations import Visualizations
    from neural_trade.visualization import theme as T

    rows = _good_configuration("default")
    board = build_leaderboard(rows)
    fn = Visualizations.get("leaderboard")
    fig = fn(board, None)
    assert T.empty_panels(fig) == []
    assert len(fig.data) >= 2          # the dev bar + the test marker (+ the table)


def test_leaderboard_figure_from_a_real_store_scenario_has_no_empty_panel():
    from pathlib import Path

    from neural_trade.experiments.leaderboard import leaderboard_for_scenario
    from neural_trade.registries.visualizations import Visualizations
    from neural_trade.visualization import theme as T

    repo_runs = Path(__file__).resolve().parent.parent / "runs"
    store = RunStore(root=repo_runs)
    if not (repo_runs / "scenarios" / "reference_default").is_dir():
        pytest.skip("runs/scenarios/reference_default not present in this checkout")
    rows = leaderboard_for_scenario(store, "reference_default")
    fn = Visualizations.get("leaderboard")
    fig = fn(rows, None)
    assert T.empty_panels(fig) == []


# ------------------------------------------------------------------ review additions (NT-031 repair)
def test_zero_trade_row_tops_the_sort_but_never_wins():
    # a 0-trade strategy has net Sharpe 0, above every losing row; the activity guard-rail removes it
    loser = _good_configuration("loser", sharpe=-1.0, n_trades=30.0)
    for r in loser:
        r["total_return"], r["buy_and_hold_return"], r["random_percentile_return"] = 0.10, 0.01, 90.0
    flat = _good_configuration("flat", sharpe=0.0, n_trades=0.0)
    board = build_leaderboard(loser + flat)
    assert board[0].configuration == "flat" and board[0].disqualified
    assert winner(board).configuration == "loser"
    assert "min_trades FAIL" in leaderboard_markdown(board)


@pytest.mark.parametrize("column,bad,rail", [
    ("max_drawdown", float("nan"), "max_drawdown"), ("n_trades", float("inf"), "min_trades"),
    ("random_percentile_return", float("nan"), "beat_random_null"),
    ("buy_and_hold_return", float("nan"), "beat_buy_and_hold"), ("sharpe_net", float("nan"), "ranking_value")])
def test_non_finite_guard_rail_value_disqualifies_with_a_stated_reason(column, bad, rail):
    rows = _good_configuration("nan_row")
    for r in rows:
        r["total_return"], r["buy_and_hold_return"], r["random_percentile_return"] = 0.10, 0.01, 90.0
        if r["role"] == "dev":
            r[column] = bad
    spec = GuardRailSpec(max_drawdown_max=0.5)
    board = build_leaderboard(rows, guard_rails=spec)
    row = board[0]
    assert row.disqualified and winner(board) is None
    failed = {g.name: g.detail for g in row.guard_rails if not g.passed}
    assert rail in failed and "non-finite" in failed[rail]
    assert "non-finite" in leaderboard_markdown(board)


def test_missing_guard_rail_value_disqualifies_as_missing():
    rows = _good_configuration("gap")
    for r in rows:
        r["max_drawdown"] = None
    board = build_leaderboard(rows, guard_rails=GuardRailSpec(max_drawdown_max=0.5))
    assert any(g.name == "max_drawdown" and not g.passed and "missing" in g.detail for g in board[0].guard_rails)


def test_non_finite_rank_value_sorts_last_and_cannot_win():
    good = _good_configuration("good")
    for r in good:
        r["total_return"], r["buy_and_hold_return"], r["random_percentile_return"] = 0.10, 0.01, 90.0
    bad = _good_configuration("bad", sharpe=float("nan"))
    board = build_leaderboard(bad + good)
    assert [r.configuration for r in board] == ["good", "bad"]
    assert winner(board).configuration == "good"


def test_failed_row_names_status_as_the_failing_guard_rail_with_the_error():
    rows = [_row(configuration="broken", status="failed", error="ValueError: boom", sharpe_net=None,
                 total_return=None, max_drawdown=None, n_trades=None, buy_and_hold_return=None,
                 random_percentile_return=None)]
    row = build_leaderboard(rows)[0]
    assert any(g.name == "status" and not g.passed and "boom" in g.detail for g in row.guard_rails)
    text = leaderboard_markdown([row])
    assert "failed" in text and "boom" in text


def test_seed_reruns_report_counts_and_seed_spread():
    rows = []
    for f in (-3, -2):
        for s, v in enumerate((1.0, 2.0, 3.0)):
            rows.append(_row(configuration="rerun", fold=f, role="dev", seed=s, sharpe_net=v))
    row = build_leaderboard(rows)[0]
    assert row.dev.n_seeds == 3 and row.dev.seeds_per_fold == (3, 3)
    assert row.dev.values["sharpe_net"] == pytest.approx(2.0)
    assert row.dev.seed_spread["sharpe_net"] == pytest.approx(1.0)       # sd of (1, 2, 3)
    assert row.dev.spread["sharpe_net"] == pytest.approx(0.0)            # identical fold means
    assert "3 seeds" in leaderboard_markdown([row]) and "seed sd" in leaderboard_markdown([row])


def test_every_test_column_is_labelled_in_the_table_and_the_figure():
    from neural_trade.experiments.leaderboard import TABLE_HEADER
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    test_cols = [h for h in TABLE_HEADER if h.startswith("test ")]
    assert len(test_cols) == 4 and all("test, not used for ranking" in h for h in test_cols)
    assert not any("test" in h for h in TABLE_HEADER if not h.startswith("test "))
    fig = leaderboard_figure(build_leaderboard(_good_configuration("a")))
    table = next(t for t in fig.data if t.type == "table")
    assert [h.replace("<br>", " ") for h in table.header.values] == list(TABLE_HEADER)
    scatter = next(t for t in fig.data if t.type == "scatter" and t.name.startswith("test "))
    assert "not used for ranking" in scatter.name
    assert any("ranking column" in a.text for a in fig.layout.annotations)


def test_figure_draws_the_dev_spread_and_marks_the_disqualified_row():
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    rows = _good_configuration("ok", dev_folds=(-4, -3, -2), sharpe=1.0)
    for r in rows:
        if r["role"] == "dev":
            r["sharpe_net"] = 1.0 + 0.5 * (r["fold"] + 3)
    rows += _good_configuration("flat", sharpe=0.0, n_trades=0.0)
    fig = leaderboard_figure(build_leaderboard(rows))
    fold_sd = next(t for t in fig.data if t.type == "scatter" and t.name.startswith("fold sd"))
    assert any(e > 0 for e in fold_sd.error_x.array)
    assert any("DISQUALIFIED" in y for y in fig.layout.yaxis.ticktext)


def test_cli_guard_rail_flags_and_out(tmp_path, capsys):
    from neural_trade.cli import main

    _write_run_dir(tmp_path, scenario="gr", configuration="default", fold=-2, role="dev", seed=0, n_trades=0)
    out = tmp_path / "out"
    assert main(["leaderboard", "gr", "--store", str(tmp_path), "--out", str(out)]) == 0
    assert "min_trades FAIL" in capsys.readouterr().out
    assert (out / "gr" / "leaderboard.md").is_file() and (out / "gr" / "leaderboard.html").is_file()
    assert main(["leaderboard", "gr", "--store", str(tmp_path), "--min-trades", "0", "--max-drawdown", "0.01",
                 "--no-beat-buy-and-hold", "--no-beat-random-null"]) == 0
    text = capsys.readouterr().out
    assert "max_drawdown FAIL" in text and "min_trades" in text


# ------------------------------------------------------------------ criterion 3 from the scenario; per-fold activity
def _scenario_dict(**extra):
    d = {"schema_version": 1, "name": "lb_spec", "folds": [-2, -1], "seeds": [0]}
    d.update(extra)
    return d


def test_scenario_leaderboard_block_parses_and_keeps_identity_unchanged():
    from neural_trade.experiments.scenario import Scenario

    plain = Scenario.from_dict(_scenario_dict())
    block = {"max_drawdown": 0.2, "min_trades": 5, "random_null_percentile": 70, "beat_buy_and_hold": False,
             "beat_random_null": True}
    with_block = Scenario.from_dict(_scenario_dict(leaderboard=block))
    assert plain.leaderboard == {} and with_block.leaderboard == block
    # scoring-side: nothing that identifies or hashes a run changes
    assert with_block.spec_hash == plain.spec_hash and with_block.settings_hash == plain.settings_hash
    assert with_block.to_dict() == plain.to_dict() and "leaderboard" not in with_block.to_dict()
    assert [c.name for c in with_block.configurations()] == [c.name for c in plain.configurations()]


def test_every_committed_scenario_still_loads_with_default_guard_rails():
    from pathlib import Path

    from neural_trade.experiments.scenario import Scenario

    files = sorted((Path(__file__).resolve().parent.parent / "configs" / "scenarios").glob("*.yaml"))
    assert files
    for f in files:
        assert Scenario.from_yaml(f).leaderboard == {}


@pytest.mark.parametrize("block", [{"max_dd": 0.2}, {"min_trades": -1}, {"beat_buy_and_hold": 1},
                                   {"max_drawdown": "high"}, {"min_trades": float("nan")}, "x"])
def test_scenario_leaderboard_block_refuses_unknown_keys_and_bad_values(block):
    from neural_trade.experiments.scenario import Scenario, ScenarioError

    with pytest.raises(ScenarioError):
        Scenario.from_dict(_scenario_dict(leaderboard=block))


def test_thresholds_come_from_the_scenario_and_flags_override_with_the_source_shown():
    from neural_trade.experiments.leaderboard import describe_guard_rails, scenario_guard_rails
    from neural_trade.experiments.scenario import Scenario

    sc = Scenario.from_dict(_scenario_dict(leaderboard={"max_drawdown": 0.2, "min_trades": 5}))
    spec, src = scenario_guard_rails(sc)
    assert spec.max_drawdown_max == 0.2 and spec.min_trades == 5 and spec.require_beat_buy_and_hold
    assert "max_drawdown=0.2 (scenario)" in src
    spec2, src2 = scenario_guard_rails(sc, max_drawdown=0.5, beat_random_null=False)
    assert spec2.max_drawdown_max == 0.5 and spec2.require_beat_random_null is False and spec2.min_trades == 5
    assert "max_drawdown=0.5 (command-line override)" in src2 and "max_drawdown=0.2" not in src2
    assert "command-line override" in describe_guard_rails(spec2, src2)
    assert scenario_guard_rails(None)[0] == DEFAULT_GUARD_RAILS


def test_leaderboard_for_scenario_reads_the_thresholds_from_the_scenario_spec(tmp_path):
    from neural_trade.experiments.leaderboard import leaderboard_for_scenario
    from neural_trade.experiments.scenario import Scenario

    _write_run_dir(tmp_path, scenario="sp", configuration="c", fold=-2, role="dev", seed=0, n_trades=10)
    store = RunStore(tmp_path)
    assert any(g.name == "min_trades" and g.passed for g in leaderboard_for_scenario(store, "sp")[0].guard_rails)
    sc = Scenario.from_dict(_scenario_dict(name="sp", leaderboard={"min_trades": 50}))
    row = leaderboard_for_scenario(store, "sp", spec=sc)[0]
    assert row.disqualified and any(g.name == "min_trades" and not g.passed for g in row.guard_rails)


def test_an_idle_dev_fold_disqualifies_even_when_the_mean_passes_and_names_the_fold():
    rows = [_row(configuration="idle_fold", fold=-3, role="dev", n_trades=0.0),
            _row(configuration="idle_fold", fold=-2, role="dev", n_trades=40.0),
            _row(configuration="idle_fold", fold=-1, role="test", n_trades=10.0)]
    for r in rows:
        r["total_return"], r["buy_and_hold_return"], r["random_percentile_return"] = 0.10, 0.01, 90.0
    board = build_leaderboard(rows)
    row = board[0]
    assert row.dev.values["n_trades"] == pytest.approx(20.0)         # the mean alone would pass
    bad = next(g for g in row.guard_rails if g.name == "min_trades")
    assert not bad.passed and "fold -3: 0.0" in bad.detail and "fold -2" not in bad.detail.split("below")[-1]
    assert row.disqualified and winner(board) is None


def test_cli_header_shows_scenario_thresholds_and_the_override(tmp_path, capsys):
    import yaml

    from neural_trade.cli import main

    _write_run_dir(tmp_path, scenario="hdr", configuration="default", fold=-2, role="dev", seed=0)
    spec = tmp_path / "hdr.yaml"
    spec.write_text(yaml.safe_dump(_scenario_dict(name="hdr", leaderboard={"max_drawdown": 0.01})), encoding="utf-8")
    assert main(["leaderboard", "hdr", "--store", str(tmp_path), "--spec", str(spec)]) == 0
    out = capsys.readouterr().out
    assert "max drawdown <= 1%" in out and "max_drawdown=0.01 (scenario)" in out and "max_drawdown FAIL" in out
    assert main(["leaderboard", "hdr", "--store", str(tmp_path), "--spec", str(spec), "--max-drawdown", "0.9"]) == 0
    out = capsys.readouterr().out
    assert "max_drawdown=0.9 (command-line override)" in out and "max_drawdown FAIL" not in out


# ------------------------------------------------------------------ repair round 2: figure (D-014, theme.py)
def _passing(rows):
    for r in rows:
        r["total_return"], r["buy_and_hold_return"], r["random_percentile_return"] = 0.10, 0.01, 90.0
    return rows


def _three_class_board():
    rows = _passing(_good_configuration("top", sharpe=2.0))
    rows += _passing(_good_configuration("second", sharpe=1.0))
    flat = _good_configuration("flat_zero_trades", sharpe=0.5, n_trades=0.0)
    for r in flat:
        if r["role"] == "dev":
            r["sharpe_net"] = 0.5 + 0.2 * r["seed"] + 0.3 * (r["fold"] + 3)
    return build_leaderboard(rows + flat)


def test_bars_never_use_a_horizon_colour_and_the_legend_names_each_class_truthfully():
    from neural_trade.visualization import theme as T
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    board = _three_class_board()
    fig = leaderboard_figure(board)
    horizon = {c.lower() for c in T.SERIES[:3]} | {T.rgba(c, a) for c in T.SERIES[:3] for a in (0.45, 0.75, 0.9)}
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 3
    labels = list(fig.layout.yaxis.ticktext)
    for b in bars:
        assert str(b.marker.color).lower() not in horizon and str(b.marker.line.color).lower() not in horizon
        members = [labels[int(y)] for y in b.y]
        if "winner" in b.name:
            assert members and all("winner" in m for m in members)
        elif "eligible" in b.name:
            assert members and not any("winner" in m or "DISQUALIFIED" in m for m in members)
        else:
            assert "disqualified" in b.name and all("DISQUALIFIED" in m for m in members)
            assert b.marker.pattern.shape == "/"          # hatched, as the docstring says
    for t in fig.data:
        if t.type == "scatter" and t.marker.color is not None:
            assert str(t.marker.color).lower() not in horizon


def test_the_winner_is_marked_by_shape_and_text_in_the_figure_table_and_markdown():
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    board = _three_class_board()
    assert [r.is_winner for r in board] == [True, False, False] and winner(board) is board[0]
    fig = leaderboard_figure(board)
    star = next(t for t in fig.data if t.type == "scatter" and t.marker.symbol == "star")
    assert len(star.y) == 1 and "winner" in fig.layout.yaxis.ticktext[int(star.y[0])]
    table = next(t for t in fig.data if t.type == "table")
    assert table.cells.values[0][0] == "1 (winner)" and "winner" not in table.cells.values[0][1]
    assert "| 1 (winner) |" in leaderboard_markdown(board)


def test_no_table_text_is_cut_off_and_the_table_fits_the_figure():
    from neural_trade.experiments.leaderboard import table_cells
    from neural_trade.visualization.leaderboard_fig import CELL_PAD_PX, CHAR_PX, COLUMN_CHARS, leaderboard_figure

    rows = _good_configuration("a_very_long_configuration_name__lr-0.0003__hidden-128", n_trades=0.0)
    for r in rows:
        r["strategy"] = "enhanced_multi_horizon_with_a_long_name"
        r["fee_bps"] = 10.0
    rows += [_row(configuration="broken", status="failed", error="ResourceExhaustedError: OOM " * 6, sharpe_net=None)]
    board = build_leaderboard(rows, guard_rails=GuardRailSpec(max_drawdown_max=0.01))
    fig = leaderboard_figure(board)
    table = next(t for t in fig.data if t.type == "table")
    assert list(table.columnwidth) == [w * CHAR_PX + CELL_PAD_PX for w in COLUMN_CHARS]
    columns = [[h] for h in table.header.values]
    for c, col in enumerate(table.cells.values):
        columns[c] += list(col)
    for c, cells in enumerate(columns):
        for cell in cells:
            for line in cell.split("<br>"):
                text = line.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
                assert len(text) <= COLUMN_CHARS[c], (c, text)
                assert not text.endswith("...") and not text.endswith("…"), (c, text)
    # every original character is still there (wrapping only, nothing dropped)
    for r_i, r in enumerate(board):
        for c, full in enumerate(table_cells(r)):
            shown = table.cells.values[c][r_i].replace("<br>", "").replace("&lt;", "<").replace("&gt;", ">")
            assert shown.replace(" ", "") == full.replace(" ", "")
    # the table's domain is tall enough for header + rows (no scrolled-away row)
    plot_h = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    dom = table.domain.y
    needed = table.header.height + table.cells.height * len(board)
    assert plot_h * (dom[1] - dom[0]) >= needed
    assert fig.layout.width >= fig.layout.margin.l + sum(table.columnwidth)


def test_both_spreads_are_drawn_distinguished_and_labelled():
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    rows = []
    for f in (-3, -2):
        for s, v in enumerate((1.0, 2.0)):
            rows.append(_row(configuration="spread", fold=f, seed=s, sharpe_net=v + (f + 3)))
    fig = leaderboard_figure(build_leaderboard(_passing(rows)))
    fold = next(t for t in fig.data if t.type == "scatter" and t.name.startswith("fold sd"))
    seed = next(t for t in fig.data if t.type == "scatter" and t.name.startswith("seed sd"))
    assert fold.error_x.array[0] == pytest.approx(math.sqrt(0.5))           # sd of fold means 1.5, 2.5
    assert seed.error_x.array[0] == pytest.approx(math.sqrt(0.5))           # sd of (1, 2) and of (2, 3)
    assert fold.error_x.color != seed.error_x.color and fold.y[0] != seed.y[0]


def test_a_disqualified_bar_has_a_visible_whisker_of_another_colour():
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    fig = leaderboard_figure(_three_class_board())
    dq = next(t for t in fig.data if t.type == "bar" and "disqualified" in t.name)
    dq_y = {int(y) for y in dq.y}
    fold = next(t for t in fig.data if t.type == "scatter" and t.name.startswith("fold sd"))
    assert any(int(y) in dq_y and e > 0 for y, e in zip(fold.y, fold.error_x.array))
    assert fold.error_x.color not in (dq.marker.color, dq.marker.line.color)


def test_a_single_cell_row_says_no_spread_on_the_figure():
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    fig = leaderboard_figure(build_leaderboard(_passing([_row(configuration="one")])))
    texts = [x for t in fig.data if t.type == "scatter" and t.mode == "text" for x in t.text]
    assert any("no spread (1 cell)" in x for x in texts)
    assert not any(t.type == "scatter" and (t.name or "").startswith(("fold sd", "seed sd")) for t in fig.data)


# ------------------------------------------------------------------ repair round 2: cost profile (D-044)
def test_cell_cost_profile_reads_the_report_then_meta_else_unknown(tmp_path):
    from neural_trade.experiments.leaderboard import CostProfile, cell_cost_profile

    a = _write_run_dir(tmp_path, scenario="c", configuration="a", fold=-2, role="dev", seed=0, cost=(10.0, 1.0, 2.0))
    assert cell_cost_profile(a) == (CostProfile(10.0, 1.0, 2.0), "report")
    b = _write_run_dir(tmp_path, scenario="c", configuration="b", fold=-2, role="dev", seed=0, report=False,
                       meta_backtest={"fee_bps": 5, "half_spread_bps": 0, "slippage_bps": 0, "random_seeds": 20})
    assert cell_cost_profile(b) == (CostProfile(5.0, 0.0, 0.0), "meta.json engine.backtest")
    c = _write_run_dir(tmp_path, scenario="c", configuration="c", fold=-2, role="dev", seed=0, report=False,
                       meta_backtest={"fee_bps": 5})                      # partial: the rest was a commit default
    assert cell_cost_profile(c) == (None, "unknown")
    assert CostProfile(10.0, 1.0, 2.0).text() == "13 bps/side (fee 10 + half-spread 1 + slippage 2)"


def test_a_row_at_another_cost_profile_is_marked_not_comparable_and_cannot_win():
    zero = _passing(_good_configuration("zero_cost", sharpe=1.0))
    old = _passing(_good_configuration("thirteen_bps", sharpe=3.0))
    for r in old:
        r["fee_bps"], r["half_spread_bps"], r["slippage_bps"] = 10.0, 1.0, 2.0
    board = build_leaderboard(zero + old)
    assert [r.configuration for r in board] == ["thirteen_bps", "zero_cost"]     # the order is still dev Sharpe
    top = board[0]
    assert top.disqualified and not top.cost_comparable and not top.is_winner
    rail = next(g for g in top.guard_rails if g.name == "cost_profile")
    assert not rail.passed and "not comparable" in rail.detail and "13 bps/side" in rail.detail
    assert winner(board).configuration == "zero_cost"
    text = leaderboard_markdown(board)
    assert "Cost profile: the board ranks at 0 bps/side (D-044)" in text
    assert "different cost profile: 13 bps/side (fee 10 + half-spread 1 + slippage 2) (1 row)" in text
    assert "13 bps/side (fee 10 + half-spread 1 + slippage 2) (not comparable)" in text   # the table column


def test_mixed_and_unknown_cost_profiles_are_not_comparable():
    mixed = _passing(_good_configuration("mixed"))
    mixed[0]["fee_bps"] = 10.0
    unknown = _passing(_good_configuration("unknown"))
    for r in unknown:
        for k in ("fee_bps", "half_spread_bps", "slippage_bps"):
            r.pop(k)
    board = build_leaderboard(mixed + unknown)
    by = {r.configuration: r for r in board}
    assert by["mixed"].cost_text.startswith("mixed: ") and "10 bps/side on 1 cell" in by["mixed"].cost_text
    assert by["unknown"].cost_text == "unknown"
    assert all(r.disqualified and not r.cost_comparable for r in board) and winner(board) is None


def test_a_scenario_that_sets_costs_ranks_at_its_own_profile_and_the_header_says_so():
    from neural_trade.experiments.leaderboard import scenario_cost_profile

    rows = _passing(_good_configuration("costed"))
    for r in rows:
        r["fee_bps"], r["half_spread_bps"], r["slippage_bps"] = 10.0, 1.0, 2.0
    board_cost = scenario_cost_profile({"fee_bps": 10, "half_spread_bps": 1, "slippage_bps": 2, "random_seeds": 5})
    board = build_leaderboard(rows, board_cost=board_cost)
    assert board[0].cost_comparable and board[0].is_winner
    text = leaderboard_markdown(board)
    assert "not D-044's 0 bps/side" in text and "every scored row's stored net Sharpe was computed at it" in text
    assert scenario_cost_profile(None).per_side == 0.0      # D-044 default


def test_the_figure_shows_the_cost_profile_per_row_and_in_the_header():
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    rows = _passing(_good_configuration("old"))
    for r in rows:
        r["fee_bps"], r["half_spread_bps"], r["slippage_bps"] = 10.0, 1.0, 2.0
    fig = leaderboard_figure(build_leaderboard(rows))
    assert "13 bps/side" in fig.layout.yaxis.ticktext[0] and "not comparable" in fig.layout.yaxis.ticktext[0]
    assert "Cost profile" in fig.layout.title.text and "different cost profile" in fig.layout.title.text
    table = next(t for t in fig.data if t.type == "table")
    col = [h.replace("<br>", " ") for h in table.header.values].index("cost profile of the stored net Sharpe")
    assert "13 bps/side" in table.cells.values[col][0].replace("<br>", " ")


def test_cli_reads_each_cells_cost_from_its_stored_report(tmp_path, capsys):
    from neural_trade.cli import main

    _write_run_dir(tmp_path, scenario="cost", configuration="old", fold=-2, role="dev", seed=0, cost=(10.0, 1.0, 2.0))
    assert main(["leaderboard", "cost", "--store", str(tmp_path), "--specs-dir", str(tmp_path / "none")]) == 0
    out = capsys.readouterr().out
    assert "cost_profile FAIL (not comparable: 13 bps/side" in out and "different cost profile" in out


def test_the_real_store_reference_board_says_13_bps_per_side():
    from pathlib import Path

    from neural_trade.experiments.leaderboard import find_scenario_spec, leaderboard_for_scenario

    repo = Path(__file__).resolve().parent.parent
    if not (repo / "runs" / "scenarios" / "reference_default").is_dir():
        pytest.skip("runs/scenarios/reference_default not present in this checkout")
    spec, where = find_scenario_spec("reference_default", repo / "configs" / "scenarios")
    assert where == "reference.yaml"
    board = leaderboard_for_scenario(RunStore(root=repo / "runs"), "reference_default", spec=spec)
    assert board and all(r.cost_text == "13 bps/side (fee 10 + half-spread 1 + slippage 2)" for r in board)
    assert all(not r.cost_comparable and r.disqualified for r in board)


# ------------------------------------------------------------------ repair round 2: spec lookup, fold coverage, status
def test_find_scenario_spec_by_name_not_file_name_then_the_stored_spec(tmp_path):
    import yaml

    from neural_trade.experiments.leaderboard import find_scenario_spec

    specs = tmp_path / "specs_dir"
    specs.mkdir()
    (specs / "differently_named.yaml").write_text(yaml.safe_dump(_scenario_dict(name="wanted")), encoding="utf-8")
    (specs / "wanted.yaml").write_text(yaml.safe_dump(_scenario_dict(name="something_else")), encoding="utf-8")
    spec, where = find_scenario_spec("wanted", specs)
    assert where == "differently_named.yaml" and spec.name == "wanted"
    stored = tmp_path / "stored"
    stored.mkdir()
    (stored / "abc.json").write_text(json.dumps({"name": "other", "folds": [-3, -2, -1],
                                                 "backtest": {"fee_bps": 1.0}}), encoding="utf-8")
    spec, where = find_scenario_spec("other", specs, stored)
    assert where == "stored spec specs/abc.json" and spec["folds"] == [-3, -2, -1]
    assert find_scenario_spec("missing", specs, tmp_path / "nope") == (None, "")


def test_cli_finds_the_spec_by_its_name_key(tmp_path, capsys):
    import yaml

    from neural_trade.cli import main

    _write_run_dir(tmp_path, scenario="named_x", configuration="default", fold=-2, role="dev", seed=0)
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "file_name.yaml").write_text(
        yaml.safe_dump(_scenario_dict(name="named_x", leaderboard={"max_drawdown": 0.01})), encoding="utf-8")
    assert main(["leaderboard", "named_x", "--store", str(tmp_path), "--specs-dir", str(specs)]) == 0
    out = capsys.readouterr().out
    assert "file_name.yaml: max_drawdown=0.01 (scenario)" in out and "max_drawdown FAIL" in out


def test_a_configuration_missing_a_dev_fold_fails_fold_coverage_naming_each_fold():
    full = _passing(_good_configuration("full", dev_folds=(-3, -2), sharpe=1.0))
    partial = _passing(_good_configuration("partial", dev_folds=(-3,), sharpe=5.0))
    partial.append(_row(configuration="partial", fold=-2, seed=0, status="failed", error="OOM", sharpe_net=None))
    partial.append(_row(configuration="partial", fold=-2, seed=1, status="incomplete", sharpe_net=None))
    board = build_leaderboard(full + partial, spec_folds=[-4, -3, -2, -1])
    by = {r.configuration: r for r in board}
    p = by["partial"]
    cov = next(g for g in p.guard_rails if g.name == "fold_coverage")
    assert not cov.passed and "fold -2 (failed, incomplete)" in cov.detail and "fold -4 (no cell)" in cov.detail
    assert "fold -1" not in cov.detail                               # -1 is the test fold
    assert p.disqualified and board[0] is p                          # still ranked first by its number
    f = next(g for g in by["full"].guard_rails if g.name == "fold_coverage")
    assert not f.passed and "fold -4 (no cell)" in f.detail          # the spec's fold -4 never ran
    board2 = build_leaderboard(full + partial)                       # without the spec: observed dev folds
    assert winner(board2).configuration == "full"
    assert next(g for g in board2[1].guard_rails if g.name == "fold_coverage").passed


def test_status_text_counts_failed_and_incomplete_cells():
    rows = _passing(_good_configuration("mixed_status"))
    rows.append(_row(configuration="mixed_status", fold=-3, seed=2, status="failed", error="x", sharpe_net=None))
    rows.append(_row(configuration="mixed_status", fold=-2, seed=2, status="incomplete", sharpe_net=None))
    row = build_leaderboard(rows)[0]
    assert row.n_incomplete == 1 and row.n_failed == 1
    assert "done (1 of 7 cells failed, 1 of 7 cells incomplete)" in leaderboard_markdown([row])


# ------------------------------------------------------------------ repair round 2: aggregation edge cases, PNG
def test_unequal_seeds_per_fold_use_the_mean_of_fold_means_not_the_flat_mean():
    rows = [_row(configuration="u", fold=-3, seed=s, sharpe_net=1.0) for s in (0, 1, 2)]
    rows.append(_row(configuration="u", fold=-2, seed=0, sharpe_net=4.0))
    row = build_leaderboard(rows)[0]
    assert row.dev.values["sharpe_net"] == pytest.approx(2.5)       # (1.0 + 4.0) / 2; the flat mean is 1.75
    assert row.dev.seeds_per_fold == (3, 1)


def test_an_all_negative_board_sorts_a_non_finite_rank_value_last():
    rows = (_good_configuration("minus_two", sharpe=-2.0) + _good_configuration("nan", sharpe=float("nan"))
            + _good_configuration("minus_one", sharpe=-1.0) + _good_configuration("inf", sharpe=float("-inf")))
    board = build_leaderboard(rows)
    assert [r.configuration for r in board][:2] == ["minus_one", "minus_two"]
    assert {r.configuration for r in board[2:]} == {"nan", "inf"}


def test_cli_out_also_writes_the_png(tmp_path, capsys, monkeypatch):
    import neural_trade.visualization.leaderboard_fig as lf
    from neural_trade.cli import main

    written = []

    def fake_png(fig, png, **kw):
        written.append(png)
        png.write_bytes(b"\x89PNG")
        return True

    monkeypatch.setattr(lf, "write_png", fake_png)
    _write_run_dir(tmp_path, scenario="png", configuration="default", fold=-2, role="dev", seed=0)
    out = tmp_path / "out"
    assert main(["leaderboard", "png", "--store", str(tmp_path), "--out", str(out)]) == 0
    assert written == [out / "png" / "leaderboard.png"] and (out / "png" / "leaderboard.png").is_file()


def test_no_leaderboard_line_is_dotted():
    """D-014: dotted lines mean training; a leaderboard has none (QA of NT-031, 2026-10-06)."""
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    fig = leaderboard_figure(_three_class_board())
    dashes = [getattr(getattr(t, "line", None), "dash", None) for t in fig.data]
    dashes += [s.line.dash for s in fig.layout.shapes or ()]
    dashes += [getattr(getattr(t, "error_x", None), "dash", None) for t in fig.data]
    assert "dot" not in dashes

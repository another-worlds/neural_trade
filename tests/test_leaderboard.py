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
                   n_trades=10.0, error=None):
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
                  "backtest": {}, "run": {"calibrate": True, "save_artifacts": False}},
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

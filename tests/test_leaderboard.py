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
    assert list(table.header.values) == list(TABLE_HEADER)
    scatter = next(t for t in fig.data if t.type == "scatter")
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
    bar = next(t for t in fig.data if t.type == "bar")
    assert any(e > 0 for e in bar.error_x.array)
    assert any("DISQUALIFIED" in y for y in bar.y)


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

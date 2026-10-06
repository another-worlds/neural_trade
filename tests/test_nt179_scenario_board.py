"""NT-179: several scenarios on ONE leaderboard (the learned model, its frozen twin and the TA rules, NT-050's
comparison), the rule-only sweep's CPU budget and single re-run, the trial counts in the nt033 spec headers,
and the pin that a price-only rule is scored on the block's own closes. Fake run directories, no training."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neural_trade.experiments.leaderboard import GuardRailSpec, build_leaderboard, leaderboard_markdown, winner
from tests.test_leaderboard import _row, _write_run_dir

REPO = Path(__file__).resolve().parent.parent
NO_RAILS = {"beat_buy_and_hold": False, "beat_random_null": False}


def _cells(scenario, configuration, *, dev_folds=(-3, -2), sharpe=1.0, **kw):
    rows = [_row(scenario=scenario, configuration=configuration, fold=f, role="dev", sharpe_net=sharpe, **kw)
            for f in dev_folds]
    rows.append(_row(scenario=scenario, configuration=configuration, fold=-1, role="test", sharpe_net=0.1, **kw))
    return rows


def _rail(row, name):
    return next(g for g in row.guard_rails if g.name == name)


# ------------------------------------------------------------------ (1) the combined board
def test_a_scenario_that_ran_other_folds_fails_fold_coverage_naming_them_and_cannot_win():
    rows = (_cells("learned", "default", sharpe=1.0)
            + _cells("twin", "default", sharpe=2.0, dev_folds=(-2,))             # the best Sharpe, on fewer folds
            + _cells("rule", "p1", sharpe=0.5))
    board = build_leaderboard(rows)
    by = {(r.scenario, r.configuration): r for r in board}
    twin = by[("twin", "default")]
    cov = _rail(twin, "fold_coverage")
    assert not cov.passed and twin.disqualified and "fold -3" in cov.detail and "board" in cov.detail
    assert _rail(by[("learned", "default")], "fold_coverage").passed
    assert board[0].scenario == "twin"                                           # ranked first by Sharpe...
    w = winner(board)
    assert (w.scenario, w.configuration) == ("learned", "default")               # ...the winner is across scenarios


def test_the_expected_folds_of_one_scenario_alone_are_unchanged():
    board = build_leaderboard(_cells("twin", "default", dev_folds=(-2,)))
    assert _rail(board[0], "fold_coverage").passed and "board" not in _rail(board[0], "fold_coverage").description


def test_a_row_at_another_cost_profile_is_not_comparable_across_scenarios():
    rows = _cells("learned", "default") + _cells("rule", "p1", sharpe=5.0, fee_bps=10.0)
    board = build_leaderboard(rows)
    rule = next(r for r in board if r.scenario == "rule")
    assert not rule.cost_comparable and rule.disqualified and not _rail(rule, "cost_profile").passed
    assert winner(board).scenario == "learned"


def test_each_scenario_is_judged_by_its_own_guard_rail_thresholds():
    rows = _cells("a", "x", n_trades=20.0) + _cells("b", "x", n_trades=20.0)
    rails = {"a": GuardRailSpec(min_trades=5, require_beat_buy_and_hold=False, require_beat_random_null=False),
             "b": GuardRailSpec(min_trades=50, require_beat_buy_and_hold=False, require_beat_random_null=False)}
    board = {r.scenario: r for r in build_leaderboard(rows, guard_rails=rails)}
    assert not board["a"].disqualified and board["b"].disqualified
    assert not _rail(board["b"], "min_trades").passed
    text = leaderboard_markdown(list(board.values()), guard_rails=rails,
                                guard_rail_source={"a": "a.yaml: min_trades=5 (scenario)", "b": "flags"})
    assert "`a` / `x`" in text and "`b` / `x`" in text
    assert "trades >= 5 " in text and "trades >= 50 " in text
    assert "[thresholds: a.yaml: min_trades=5 (scenario)]" in text and "[thresholds: flags]" in text


def _store(tmp_path):
    for name, folds, cost in (("learned", (-3, -2), 0.0), ("twin", (-3, -2), 0.0), ("rule", (-2,), 0.0),
                              ("pricey", (-3, -2), 10.0)):
        for f in folds:
            _write_run_dir(tmp_path, scenario=name, configuration="default", fold=f, role="dev", seed=0,
                           sharpe_net={"learned": 1.0, "twin": 0.5, "rule": 3.0, "pricey": 4.0}[name],
                           n_trades=20.0, cost=(cost, 0.0, 0.0))
        _write_run_dir(tmp_path, scenario=name, configuration="default", fold=-1, role="test", seed=0,
                       cost=(cost, 0.0, 0.0))
    return tmp_path


@pytest.mark.parametrize("argv", [["--scenario", "learned,twin,rule,pricey"],
                                  ["learned", "twin", "rule", "pricey"],
                                  ["--scenario", "learned,twin", "--scenario", "rule", "pricey"]])
def test_cli_puts_several_scenarios_on_one_board_and_writes_it(tmp_path, capsys, argv):
    from neural_trade.cli import main

    store = _store(tmp_path / "s")
    out = tmp_path / "out"
    assert main(["leaderboard", *argv, "--store", str(store), "--specs-dir", str(tmp_path / "none"),
                 "--min-trades", "1", "--no-beat-buy-and-hold", "--no-beat-random-null", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.count("# Leaderboard:") == 1                                      # one board, not one per scenario
    for name in ("learned", "twin", "rule", "pricey"):
        assert f"`{name}` / `default`" in text
        assert f"- `{name}`:" in text                                             # the header's per-scenario thresholds
    assert "command-line override" in text
    rule_line = next(line for line in text.splitlines() if "`rule` / `default`" in line)
    assert "fold_coverage FAIL" in rule_line and "fold -3" in rule_line            # ran fold -2 only
    pricey_line = next(line for line in text.splitlines() if "`pricey` / `default`" in line)
    assert "not comparable" in pricey_line
    assert "**Winner:** `learned` / `default`" in text                             # best eligible row, across scenarios
    assert (out / "combined" / "leaderboard.md").read_text(encoding="utf-8").strip() == text.strip()
    assert (out / "combined" / "leaderboard.html").is_file()


def test_cli_one_scenario_still_prints_its_own_board_and_spec_needs_one_scenario(tmp_path, capsys):
    from neural_trade.cli import main

    store = _store(tmp_path / "s")
    assert main(["leaderboard", "learned", "--store", str(store), "--specs-dir", str(tmp_path / "none")]) == 0
    text = capsys.readouterr().out
    assert "Leaderboard: `learned`" in text and "/ `default`" not in text
    spec = tmp_path / "spec.yaml"
    spec.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="one scenario's spec"):
        main(["leaderboard", "learned", "twin", "--store", str(store), "--spec", str(spec)])


def test_the_combined_figure_names_scenarios_and_marks_the_winner(tmp_path):
    from neural_trade.visualization.leaderboard_fig import leaderboard_figure

    rows = _cells("learned", "default", sharpe=1.0) + _cells("rule", "p1", sharpe=0.5)
    board = build_leaderboard(rows, guard_rails=GuardRailSpec(require_beat_buy_and_hold=False,
                                                              require_beat_random_null=False))
    fig = leaderboard_figure(board)
    text = str(fig.to_dict())
    assert "learned / default" in text and "rule / p1" in text and "winner" in text
    assert "learned, rule" in fig.layout.title.text


# ------------------------------------------------------------------ (2) spec headers, rule-only budget
def test_the_nt033_spec_headers_name_the_trial_count_that_fits_12_hours():
    # 17 trials: 11.89 h upper bound at sec_per_step 0.1735 (the sweep's own estimate, --dry-run); 30 is 15.90 h, refused
    for name in ("learned", "frozen_twin", "ta_ma_cross", "ta_rsi", "ta_bollinger"):
        head = "\n".join(line for line in (REPO / "configs" / "scenarios" / f"nt033_{name}.yaml")
                         .read_text(encoding="utf-8").splitlines() if line.startswith("#"))
        assert set(re.findall(r"--n-trials (\d+)", head)) == {"17"}, name
    learned = (REPO / "configs" / "scenarios" / "nt033_learned.yaml").read_text(encoding="utf-8")
    assert "11.89 h" in learned and "15.90 h" in learned


def test_a_rule_only_sweep_reports_a_cpu_budget_and_reruns_the_rule_once(tmp_path, synthetic_bars):
    pytest.importorskip("optuna")
    from tests.test_nt033_baselines import _rule_sweep

    csv = tmp_path / "bars.csv"
    synthetic_bars.to_csv(csv, index=False)
    messages = []
    sw = _rule_sweep(tmp_path, csv, n_trials=3, top_k=2, rerun_seeds=3, dry_run=True)
    sw.announce = messages.append
    res = sw.run()
    b = res.budget
    assert b["budget_kind"] == "cpu" and b["gpu_hours"] == 0.0 and b["expected_gpu_hours"] == 0.0
    assert b["cpu_hours"] > 0 and b["rerun"]["seeds"] == 1 and b["rerun"]["dev_folds_after_first_seed"] == 0
    assert any("CPU budget" in m and "no GPU used" in m for m in messages)
    assert not any("GPU budget" in m for m in messages)
    # the re-run's test cells: top_k x 1 seed (not 3 identical copies)
    assert b["rerun_gpu_hours"] * 3600 == pytest.approx(2 * len(sw.test_folds) * 1.0)
    assert sw.rerun_seed_count() == 1


def test_a_trained_sweep_still_reports_a_gpu_budget_and_all_rerun_seeds(tmp_path, bars_csv_trained):
    pytest.importorskip("optuna")
    from neural_trade.experiments.runner import Runner  # noqa: F401 - import check only
    from neural_trade.experiments.scenario import Scenario
    from neural_trade.experiments.store import RunStore
    from neural_trade.experiments.sweep import Sweep, SweepOptions
    from tests.test_experiment_engine import FakeTrainer, spec

    s = spec(bars_csv_trained, name="trained", search={"LR": {"low": 0.0001, "high": 0.01, "log": True}})
    s.pop("variants", None)
    sw = Sweep(Scenario.from_dict(s), RunStore(tmp_path / "runs"),
               SweepOptions(parallel_record=str(tmp_path / "no.json"), sec_per_step=0.01, rerun_seeds=3, dry_run=True,
                            n_trials=2), trainer=FakeTrainer(), announce=lambda t: None)
    b = sw.run().budget
    assert b["budget_kind"] == "gpu" and b["gpu_hours"] > 0 and b["cpu_hours"] == 0.0
    assert b["rerun"]["seeds"] == 3


@pytest.fixture
def bars_csv_trained(tmp_path, synthetic_bars):
    path = tmp_path / "trained_bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ (3) score_strategy_only reads the block's closes
def _block(close, n, offset):
    anchors = np.arange(offset, offset + n)
    return {"last_close": np.asarray(close)[anchors], "y": np.zeros((n, 3)), "anchor_bar": anchors}


def _arrays(n=40):
    close = 100.0 + np.cumsum(np.sin(np.arange(n + 40) / 3.0))
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close})
    return {"df": df, "test": _block(close, n, 30), "cal": _block(close, 20, 5)}, close


def _config():
    return SimpleNamespace(HORIZON_STEPS=(10, 15, 20), RESAMPLE_MINUTES=1)


def test_score_strategy_only_feeds_the_blocks_own_closes(monkeypatch):
    from neural_trade.experiments import scorer

    arrays, close = _arrays()
    seen = {}
    real = scorer.fit_and_backtest

    def spy(signals, bars, **kw):
        seen["signal_close"], seen["bars_close"] = np.array(signals.oos.close), np.array(bars.close)
        return real(signals, bars, **kw)

    monkeypatch.setattr(scorer, "fit_and_backtest", spy)
    scorer.score_strategy_only(_config(), role="dev", strategy="ta_ma_cross", strategy_params={"fast": 3, "slow": 10},
                               arrays=arrays)
    want = close[30:70]
    np.testing.assert_array_equal(seen["signal_close"], want)       # exactly the block's closes (bars 30..69)
    np.testing.assert_array_equal(seen["bars_close"], want)


def test_score_strategy_only_refuses_bars_read_from_a_later_position(monkeypatch):
    """Mutation: bars one position LATER than the block's anchors (a future close) must be refused, not scored."""
    from neural_trade.experiments import scorer
    from neural_trade.strategy import Bars

    arrays, _ = _arrays()
    real = Bars.from_frame

    def shifted(df, anchors):
        return real(df, np.asarray(anchors) + 1)

    monkeypatch.setattr(Bars, "from_frame", staticmethod(shifted))
    with pytest.raises(scorer.ScoringError, match="do not line up"):
        scorer.score_strategy_only(_config(), role="dev", strategy="ta_ma_cross", arrays=arrays)
    # and a frame whose last_close is a later close than its bars' is refused the same way
    monkeypatch.undo()
    arrays["test"]["last_close"] = arrays["test"]["last_close"][1:].tolist() + [999.0]
    arrays["test"]["last_close"] = np.asarray(arrays["test"]["last_close"])
    with pytest.raises(scorer.ScoringError, match="do not line up"):
        scorer.score_strategy_only(_config(), role="dev", strategy="ta_ma_cross", arrays=arrays)

"""NT-033 (2)-(4): the manual-search baselines through the experiment engine. The classic TA rules
score on the same folds as the network with no network trained (``run.train: false``), the frozen
twin is an ordinary training scenario with FREEZE_INDICATOR_PERIODS, and all of them reach the
leaderboard with the same dev-fold net Sharpe column. A fake trainer stands in for training (as in
test_experiment_engine.py); the rule scenarios must not call it at all."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_trade.experiments.leaderboard import build_leaderboard, leaderboard_markdown
from neural_trade.experiments.runner import Runner
from neural_trade.experiments.scenario import Scenario, ScenarioError
from neural_trade.experiments.store import RunStore
from neural_trade.experiments.sweep import SearchSpace, Sweep, SweepError, SweepOptions
from tests.test_experiment_engine import FakeTrainer, spec

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = REPO / "configs" / "scenarios"
FILES = ("nt033_learned", "nt033_frozen_twin", "nt033_ta_ma_cross", "nt033_ta_rsi", "nt033_ta_bollinger")
# a leaderboard that does not disqualify on buy-and-hold / the random null: these tests are about plumbing
BOARD = {"beat_buy_and_hold": False, "beat_random_null": False, "min_trades": None}


class NoTrainer:
    """A trainer a rule-only scenario must never call."""

    def __call__(self, ctx, *, calibrate, save_artifacts):
        raise AssertionError("a rule-only scenario trained a network")


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("nt033_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


def _spec(csv, name, **changes):
    return spec(csv, name=name, leaderboard=BOARD, **changes)


def _rule_spec(csv, name="rule", strategy="ta_ma_cross", params=None, **changes):
    changes.setdefault("run", {"train": False})
    return _spec(csv, name, strategy={"name": strategy, "params": {"fast": 3, "slow": 10} if params is None else params},
                 seeds=[0], **changes)


# ------------------------------------------------------------------ the scenario spec files
def test_the_scenario_specs_validate_and_share_folds_data_and_costs():
    docs = {n: Scenario.from_yaml(SCENARIOS / f"{n}.yaml") for n in FILES}
    learned, twin = docs["nt033_learned"], docs["nt033_frozen_twin"]
    for sc in docs.values():
        assert sc.folds == [-3, -2, -1] and sc.base_config == "../default.yaml" and sc.backtest == {}
        sc.validate()
    # the twin differs from the learned model only in the switch
    assert twin.base().FREEZE_INDICATOR_PERIODS is True and learned.base().FREEZE_INDICATOR_PERIODS is False
    assert ({k: v for k, v in twin.base().to_dict().items() if k != "FREEZE_INDICATOR_PERIODS"}
            == {k: v for k, v in learned.base().to_dict().items() if k != "FREEZE_INDICATOR_PERIODS"})
    assert (learned.seeds, learned.strategy, learned.run, learned.search) == \
        (twin.seeds, twin.strategy, twin.run, twin.search)
    # the rule scenarios train nothing and are searched through `strategy.<param>`
    for n in FILES[2:]:
        sc = docs[n]
        assert sc.run.train is False and sc.seeds == [0]
        space = SearchSpace.from_scenario(sc)
        assert space.names == list(sc.search) and all(p.startswith("strategy.") for p in space.names)
    assert all(docs[n].run.train for n in FILES[:2])


def test_the_frozen_twin_scenario_builds_a_network_with_the_learned_networks_parameter_count():
    from neural_trade.models.registry import Models

    twin = Scenario.from_yaml(SCENARIOS / "nt033_frozen_twin.yaml").base()
    learned = Scenario.from_yaml(SCENARIOS / "nt033_learned.yaml").base()
    a, b = Models.build(learned.MODEL_NAME, learned), Models.build(twin.MODEL_NAME, twin)
    assert a.count_params() == b.count_params()
    # exactly the period logits (18 families x 3 instances) moved from trainable to non-trainable weights
    assert len(a.trainable_weights) - len(b.trainable_weights) == len(b.non_trainable_weights) == 54
    assert len(a.non_trainable_weights) == 0


def test_a_rule_only_spec_does_not_change_the_hash_of_a_training_spec(bars_csv):
    sc = Scenario.from_dict(_spec(bars_csv, "plain"))
    assert "train" not in sc.to_dict()["run"] and "train" not in sc.settings()
    rule = Scenario.from_dict(_rule_spec(bars_csv))
    assert rule.to_dict()["run"]["train"] is False and rule.settings()["train"] is False
    assert rule.settings_hash != Scenario.from_dict(_rule_spec(bars_csv, run={"train": True})).settings_hash


def test_a_rule_only_scenario_refuses_a_strategy_that_reads_model_heads(bars_csv):
    with pytest.raises(ScenarioError, match="reads model heads"):
        Scenario.from_dict(_rule_spec(bars_csv, strategy="calibrated_quantile", params={})).validate()
    with pytest.raises(ScenarioError, match="no model to save"):
        Scenario.from_dict(_rule_spec(bars_csv, run={"train": False, "save_artifacts": True})).validate()
    with pytest.raises(ScenarioError, match="unknown 'ta_ma_cross' parameter"):
        Scenario.from_dict(_rule_spec(bars_csv, params={"fastt": 3})).validate()


# ------------------------------------------------------------------ (3) a rule scenario through the engine
def test_a_rule_scenario_runs_through_the_engine_without_training(tmp_path, bars_csv):
    store = RunStore(tmp_path / "runs")
    report = Runner(Scenario.from_dict(_rule_spec(bars_csv)), store, trainer=NoTrainer()).run()
    assert len(report.ran) == 2 and not report.failed            # folds -2 (dev) and -1 (test), one seed
    rows = {r["cell_key"]: r for r in store.index.rows("rule")}
    assert set(rows) == {"default__f-2__s0", "default__f-1__s0"}
    assert rows["default__f-2__s0"]["role"] == "dev" and rows["default__f-1__s0"]["role"] == "test"
    for r in rows.values():
        assert r["status"] == "done" and r["strategy"] == "ta_ma_cross"
        assert np.isfinite(r["sharpe_net"]) and r["n_trades"] is not None and r["total_return"] is not None
        d = tmp_path / "runs" / r["run_dir"]
        assert not (d / "weights.h5").exists() and not (d / "metrics.jsonl").exists()      # no network
        doc = json.loads((d / f"strategy_report_{r['role']}.json").read_text(encoding="utf-8"))
        assert doc["trained_network"] is False and doc["backtest"]["config"]["fee_bps"] == 0.0     # costs 0 (D-044)
        assert doc["backtest"]["params"]["fast"] == 3
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))["engine"]
        assert meta["run"]["train"] is False
    # resuming trains and scores nothing again
    again = Runner(Scenario.from_dict(_rule_spec(bars_csv)), store, trainer=NoTrainer()).run()
    assert again.ran == [] and len(again.skipped) == 2


def test_a_rule_cell_scores_the_same_block_and_costs_as_a_trained_cell(tmp_path, bars_csv):
    """The out-of-sample block, its bars and the backtest settings are the trained cell's: buy-and-hold and
    always-flat on the block are identical numbers in both."""
    store = RunStore(tmp_path / "runs")
    Runner(Scenario.from_dict(_rule_spec(bars_csv)), store, trainer=NoTrainer()).run()
    Runner(Scenario.from_dict(_spec(bars_csv, "trained", seeds=[0])), store, trainer=FakeTrainer()).run()
    for fold in (-2, -1):
        rule = next(r for r in store.index.rows("rule") if r["fold"] == fold)
        net = next(r for r in store.index.rows("trained") if r["fold"] == fold)
        a, b = store.index.scores(rule["run_id"]), store.index.scores(net["run_id"])
        checked = 0
        for k in a:
            if k.startswith(("backtest/buy_and_hold/", "backtest/always_flat/")):
                assert a[k] == pytest.approx(b[k], rel=1e-12, nan_ok=True), k
                checked += 1
        assert checked >= 10


# ------------------------------------------------------------------ (1)+(4) the frozen twin and the leaderboard
def test_the_frozen_twin_scenario_trains_with_the_switch_and_the_baselines_reach_the_leaderboard(tmp_path, bars_csv):
    store = RunStore(tmp_path / "runs")
    seen = []
    inner = FakeTrainer()

    def watching(ctx, *, calibrate, save_artifacts):
        seen.append((json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))["engine"]["scenario"],
                     ctx.config.FREEZE_INDICATOR_PERIODS))
        return inner(ctx, calibrate=calibrate, save_artifacts=save_artifacts)

    Runner(Scenario.from_dict(_spec(bars_csv, "board_learned", seeds=[0])), store, trainer=watching).run()
    Runner(Scenario.from_dict(_spec(bars_csv, "board_frozen", seeds=[0],
                                    overrides={**spec(bars_csv)["overrides"], "FREEZE_INDICATOR_PERIODS": True})),
           store, trainer=watching).run()
    Runner(Scenario.from_dict(_rule_spec(bars_csv, "board_rule")), store, trainer=NoTrainer()).run()
    assert {s for s in seen if s[0] == "board_learned"} == {("board_learned", False)}
    assert {s for s in seen if s[0] == "board_frozen"} == {("board_frozen", True)}

    board = build_leaderboard(store.index.rows(), store_root=store.root, spec_folds=(-2, -1))
    by = {r.scenario: r for r in board}
    assert set(by) == {"board_learned", "board_frozen", "board_rule"}
    for name, row in by.items():
        assert row.status == "done" and row.dev.n_folds == 1 and row.test.n_folds == 1
        assert np.isfinite(row.dev.values["sharpe_net"]), name             # the same column for every row
        assert row.cost_comparable                                          # all at the board's 0-cost profile
    text = leaderboard_markdown(board)
    assert "net Sharpe" in text and "`board_rule` / `default`" in text and "`board_frozen` / `default`" in text


# ------------------------------------------------------------------ the same search tunes the rule
class _Monitor:
    def start(self):
        pass

    def stop(self):
        return {}


def _rule_sweep(tmp_path, csv, **opts):
    s = _rule_spec(csv, "rulesweep", params={},
                   search={"strategy.fast": {"low": 2, "high": 5}, "strategy.slow": {"low": 6, "high": 20}})
    s.pop("variants", None)
    options = SweepOptions(parallel_record=str(tmp_path / "no.json"), overhead_s=1.0, **opts)

    def no_gpu_needed():
        raise AssertionError("a rule-only trial asked for the GPU")

    return Sweep(Scenario.from_dict(s), RunStore(tmp_path / "runs"), options, trainer=NoTrainer(),
                 gpu_check=no_gpu_needed, sleep=lambda x: None, announce=lambda t: None, monitor_factory=_Monitor)


def test_the_strategy_search_space_reads_the_declared_ranges_and_refuses_the_rest(bars_csv):
    sc = Scenario.from_dict(_rule_spec(bars_csv, params={}, search={"strategy.fast": None, "strategy.slow": {"high": 90}}))
    fast, slow = SearchSpace.from_scenario(sc).params
    assert (fast.name, fast.kind, fast.low, fast.high) == ("strategy.fast", "int", 2.0, 30.0)
    assert (slow.kind, slow.low, slow.high) == ("int", 30.0, 90.0)
    with pytest.raises(SweepError, match="declares no searchable parameter 'size'"):
        SearchSpace.from_scenario(Scenario.from_dict(_rule_spec(bars_csv, params={}, search={"strategy.size": None})))
    with pytest.raises(SweepError, match="unknown key"):
        SearchSpace.from_scenario(Scenario.from_dict(_rule_spec(bars_csv, params={}, search={"strategy.fast": {"x": 1}})))


def test_a_rule_only_sweep_needs_strategy_keys(tmp_path, bars_csv):
    s = _rule_spec(bars_csv, "bad", params={}, search={"LR": {"low": 0.0001, "high": 0.01, "log": True}})
    sw = Sweep(Scenario.from_dict(s), RunStore(tmp_path / "runs"),
               SweepOptions(parallel_record=str(tmp_path / "no.json")), trainer=NoTrainer())
    with pytest.raises(SweepError, match=r"strategy\.<param>"):
        sw.run()


def test_a_rule_only_optuna_sweep_tunes_the_strategy_parameters_without_a_gpu_or_a_network(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    sw = _rule_sweep(tmp_path, bars_csv, n_trials=3, top_k=2, rerun_seeds=2)
    res = sw.run()
    assert res.state == "complete" and len(res.trials) == 3
    assert res.budget["sec_per_step"] == 0.0 and "no network" in res.budget["sec_per_step_source"]["source"]
    rows = sw.store.index.rows(sw.sweep_id)
    for t in res.trials:
        assert set(t["params"]) == {"strategy.fast", "strategy.slow"}
        assert 2 <= t["params"]["strategy.fast"] <= 5 and 6 <= t["params"]["strategy.slow"] <= 20
        mine = [r for r in rows if r["variant"] == t["variant"]]
        assert mine and all(r["status"] == "done" for r in mine)
        meta = json.loads((sw.store.root / mine[0]["run_dir"] / "meta.json").read_text(encoding="utf-8"))["engine"]
        assert meta["strategy"]["params"] == {"fast": t["params"]["strategy.fast"],
                                              "slow": t["params"]["strategy.slow"]}
        assert meta["params"] == {}                      # no Config field was searched
    # the ranking number is the leaderboard's dev net Sharpe, as for the network
    assert all(r["value"] is not None for r in res.ranking if r["rank"] is not None)

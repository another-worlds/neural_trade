"""NT-182: two cells of different scenarios that share a config hash and start in the same second keep
their own index rows; the index refuses one run id for two cells; older run ids keep resolving."""
import json
import shutil
from datetime import datetime, timezone

import pytest

from neural_trade.experiments import run_context
from neural_trade.experiments.runner import Runner
from neural_trade.experiments.scenario import Scenario
from neural_trade.experiments.store import RunIdCollision, RunStore
from tests.test_experiment_engine import FakeTrainer, spec


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("nt182_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


@pytest.fixture
def frozen_clock(monkeypatch):
    class Frozen:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(run_context, "datetime", Frozen)
    monkeypatch.setattr(run_context, "git_sha", lambda: "abc1234")


def _two_scenarios(csv):
    # identical configs (one config hash), different scenario names and strategies: the strategy is not
    # part of the config identity
    a = spec(csv, name="scen_a", folds=[-1], seeds=[0], strategy={"name": "calibrated_quantile", "params": {}})
    b = spec(csv, name="scen_b", folds=[-1], seeds=[0], strategy={"name": "buy_and_hold", "params": {}})
    return Scenario.from_dict(a), Scenario.from_dict(b)


def test_two_scenarios_with_one_config_hash_in_the_same_second_keep_both_rows(tmp_path, bars_csv, frozen_clock):
    store = RunStore(tmp_path / "runs")
    sa, sb = _two_scenarios(bars_csv)
    for sc in (sa, sb):
        Runner(sc, store, trainer=FakeTrainer()).run()
    rows = store.index.rows()
    assert sorted(r["scenario"] for r in rows) == ["scen_a", "scen_b"]
    assert len({r["run_id"] for r in rows}) == 2 and len({r["config_hash"] for r in rows}) == 1
    for r in rows:
        assert (store.root / r["run_dir"]).is_dir() and r["status"] == "done"
    # the index rebuilt from the directories equals the kept one
    kept = store.index.dump()
    assert store.rebuild_index().dump() == kept


def test_the_index_refuses_one_run_id_for_two_different_cells_and_names_both(tmp_path, bars_csv):
    store = RunStore(tmp_path / "runs")
    sa, _ = _two_scenarios(bars_csv)
    Runner(sa, store, trainer=FakeTrainer()).run()
    [row] = store.index.rows()
    src = store.root / row["run_dir"]
    # a copy of the directory under another scenario: same meta.json run_id, another cell
    other = store.scenario_dir("scen_c") / src.name
    shutil.copytree(src, other)
    meta = json.loads((other / "meta.json").read_text(encoding="utf-8"))
    meta["engine"]["scenario"] = "scen_c"
    (other / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    before = store.index.dump()
    with pytest.raises(RunIdCollision) as e:
        store.index.add_run(other, store.root)
    assert "scen_a" in str(e.value) and "scen_c" in str(e.value) and row["run_id"] in str(e.value)
    assert store.index.dump() == before                       # nothing was replaced


def test_an_identical_resync_of_the_same_directory_still_replaces_its_row(tmp_path, bars_csv):
    store = RunStore(tmp_path / "runs")
    sa, _ = _two_scenarios(bars_csv)
    Runner(sa, store, trainer=FakeTrainer()).run()
    before = store.index.dump()
    store.sync("scen_a")
    store.index.add_run(store.root / store.index.rows()[0]["run_dir"], store.root)
    assert store.index.dump() == before


def test_older_run_ids_without_the_suffix_keep_resolving(tmp_path, bars_csv, frozen_clock):
    """A stored cell named the pre-NT-182 way (<stamp>-<sha>-<hash8>-<cell key>) is indexed under its own id
    next to a new one."""
    store = RunStore(tmp_path / "runs")
    sa, sb = _two_scenarios(bars_csv)
    Runner(sa, store, trainer=FakeTrainer()).run()
    [new] = store.index.rows("scen_a")
    old_id = new["run_id"].rsplit("-", 1)[0]                   # strip the suffix
    assert old_id.endswith(new["cell_key"])
    src = store.root / new["run_dir"]
    old_dir = src.with_name(old_id)
    shutil.copytree(src, old_dir)
    meta = json.loads((old_dir / "meta.json").read_text(encoding="utf-8"))
    meta["run_id"] = old_id
    (old_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    store.sync("scen_a")
    assert {r["run_id"] for r in store.index.rows("scen_a")} == {old_id, new["run_id"]}
    assert store.rebuild_index().dump() == store.index.dump()

"""DESK-5.6: the model registry promotes only format-1 bundles, records config facts and dev scores (never test
numbers), activates one model per (symbol, tf), and caches one CPU Predictor per bundle (D-067, D-020)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_trade.core.config import Config
from neural_trade.desk.lab.models import GpuVisible, ModelError, ModelRegistry, PredictorCache
from neural_trade.desk.store.db import open_db

from .test_lab_runs import make_run


def make_bundle(run_dir: Path, *, version=1, config: Config | None = None, weights=True, config_yaml=True) -> Path:
    art = run_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    meta = {"pred_scale": 1.0}
    if version is not None:
        meta["format_version"] = version
    (art / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if config_yaml:
        (config or Config()).to_yaml(art / "config.yaml")
    if weights:
        (art / "weights.h5").write_bytes(b"w")
    return art


@pytest.fixture
def store(tmp_path):
    from neural_trade.experiments.store import RunStore

    root = tmp_path / "store"
    make_run(root, "r_dev", sharpe=1.5)
    make_run(root, "r_test", role="test", sharpe=9.0)
    RunStore(root).sync("lab_scen")
    return root


@pytest.fixture
def reg(tmp_path, store):
    with open_db(tmp_path / "desk.sqlite") as db:
        yield ModelRegistry(db, runs_root=store, home=tmp_path / "home", clock=lambda: 1_700_000_000.0)


def run_dir(store, name):
    return store / "scenarios" / "lab_scen" / name


# ---------------------------------------------------------------------------------------------- acceptance 1
def test_promote_refuses_a_dir_without_meta_json(reg, store):
    d = run_dir(store, "r_dev")
    with pytest.raises(ModelError, match="artifacts/meta.json"):
        reg.promote(d)
    assert reg.list() == []


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_promote_refuses_a_wrong_format_version(reg, store, version):
    make_bundle(run_dir(store, "r_dev"), version=version)
    with pytest.raises(ModelError, match="format_version"):
        reg.promote(run_dir(store, "r_dev"))
    assert reg.list() == []


def test_promote_refuses_a_bundle_without_weights_or_config(reg, store):
    make_bundle(run_dir(store, "r_dev"), weights=False)
    with pytest.raises(ModelError, match="weights.h5"):
        reg.promote(run_dir(store, "r_dev"))
    make_bundle(run_dir(store, "r_test"), config_yaml=False)
    with pytest.raises(ModelError, match="config.yaml"):
        reg.promote(run_dir(store, "r_test"))


def test_promote_refuses_an_unreadable_meta(reg, store):
    art = make_bundle(run_dir(store, "r_dev"))
    (art / "meta.json").write_text("{half", encoding="utf-8")
    with pytest.raises(ModelError, match="not valid JSON"):
        reg.promote(art)


# ---------------------------------------------------------------------------------------------- promote facts
def test_promote_records_config_facts_and_dev_scores(reg, store):
    cfg = Config(SYMBOL="ETH/USDT", RESAMPLE_MINUTES=5, LOOKBACK=40, HORIZON_STEPS=[2, 3, 4], EXTENDED_TREND_PERIODS=[2, 3, 4])
    make_bundle(run_dir(store, "r_dev"), config=cfg)
    row = reg.promote(run_dir(store, "r_dev"))
    assert (row["id"], row["symbol"], row["tf"], row["lookback"], row["horizons"]) == ("r_dev", "ETH/USDT", 5, 40, [2, 3, 4])
    assert row["promoted_ms"] == 1_700_000_000_000
    assert row["bundle_dir"].endswith("r_dev/artifacts") and row["run_dir"].endswith("r_dev")
    scores = row["dev_scores"]
    assert scores["split"] == "dev" and scores["rank_eligible"] is True and scores["sharpe_net"] == 1.5
    assert reg.get("r_dev") == row == reg.list()[0]
    # the artifacts directory itself promotes to the same id
    again = reg.promote(run_dir(store, "r_dev") / "artifacts")
    assert again["id"] == "r_dev" and len(reg.list()) == 1


def test_a_test_fold_run_is_promoted_without_numbers(reg, store):
    """D-020: a test-role run's numbers are never recorded as scores."""
    make_bundle(run_dir(store, "r_test"))
    scores = reg.promote(run_dir(store, "r_test"))["dev_scores"]
    assert scores["split"] == "test" and scores["rank_eligible"] is False
    assert "sharpe_net" not in scores and "index_scores" not in scores and "9.0" not in json.dumps(scores)


def test_a_bundle_outside_the_run_store_has_no_scores(reg, tmp_path):
    other = tmp_path / "elsewhere" / "run_x"
    make_bundle(other)
    scores = reg.promote(other)["dev_scores"]
    assert scores["split"] is None and scores["rank_eligible"] is False


def test_copy_to_home_makes_an_independent_bundle(reg, store, tmp_path):
    make_bundle(run_dir(store, "r_dev"))
    row = reg.promote(run_dir(store, "r_dev"), copy_to_home=True)
    assert Path(row["bundle_dir"]) == tmp_path / "home" / "models" / "r_dev"
    assert (Path(row["bundle_dir"]) / "weights.h5").read_bytes() == b"w"
    assert row["run_dir"].endswith("r_dev")
    with pytest.raises(ModelError, match="already exists"):
        reg.promote(run_dir(store, "r_dev"), copy_to_home=True)


# ---------------------------------------------------------------------------------------------- activate
def test_activate_keeps_one_model_per_symbol_tf(reg, store):
    make_bundle(run_dir(store, "r_dev"))
    make_bundle(run_dir(store, "r_test"))
    reg.promote(run_dir(store, "r_dev"))
    reg.promote(run_dir(store, "r_test"))
    assert reg.active("BTC/USDT", 1) is None
    reg.activate("r_dev")
    assert reg.active("BTC/USDT", 1)["id"] == "r_dev"
    reg.activate("r_test")
    assert reg.active("BTC/USDT", 1)["id"] == "r_test"
    assert reg.get("r_dev")["active_for"] == []
    # a repeated promote keeps the activation
    reg.promote(run_dir(store, "r_test"))
    assert reg.get("r_test")["active_for"] == [{"symbol": "BTC/USDT", "tf": 1}]


def test_activate_refuses_another_bar_size_unless_forced(reg, store):
    make_bundle(run_dir(store, "r_dev"))
    reg.promote(run_dir(store, "r_dev"))
    with pytest.raises(ModelError, match="1-minute"):
        reg.activate("r_dev", "BTC/USDT", 5)
    with pytest.raises(ModelError, match="no model"):
        reg.activate("nope")
    reg.activate("r_dev", "BTC/USDT", 5, force=True)
    assert reg.active("BTC/USDT", 5)["id"] == "r_dev"


# ---------------------------------------------------------------------------------------------- predictor cache
def test_one_predictor_per_bundle_loaded_lazily(reg, store):
    loads = []
    reg.cache = PredictorCache(loader=lambda b: loads.append(b) or object(), gpu_visible=lambda: False)
    make_bundle(run_dir(store, "r_dev"))
    reg.promote(run_dir(store, "r_dev"))
    assert loads == []
    assert reg.predictor("r_dev") is reg.predictor("r_dev")
    assert len(loads) == 1 and reg.cache.loaded() == 1


def test_a_new_weights_file_reloads_and_drops_the_old_predictor(reg, store):
    reg.cache = PredictorCache(loader=lambda b: object(), gpu_visible=lambda: False)
    art = make_bundle(run_dir(store, "r_dev"))
    reg.promote(run_dir(store, "r_dev"))
    first = reg.predictor("r_dev")
    (art / "weights.h5").write_bytes(b"new weights")
    assert reg.predictor("r_dev") is not first and reg.cache.loaded() == 1


def test_a_predictor_is_refused_while_tensorflow_sees_a_gpu(reg, store):
    loads = []
    reg.cache = PredictorCache(loader=lambda b: loads.append(b) or object(), gpu_visible=lambda: True)
    make_bundle(run_dir(store, "r_dev"))
    reg.promote(run_dir(store, "r_dev"))
    with pytest.raises(GpuVisible):
        reg.predictor("r_dev")
    assert loads == []
    assert reg.predictor("r_dev", allow_gpu=True) is not None and len(loads) == 1


def test_a_missing_bundle_directory_is_a_model_error(reg, store):
    make_bundle(run_dir(store, "r_dev"))
    reg.promote(run_dir(store, "r_dev"))
    import shutil

    shutil.rmtree(run_dir(store, "r_dev") / "artifacts")
    with pytest.raises(ModelError, match="gone"):
        reg.predictor("r_dev")

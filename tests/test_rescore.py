"""NT-076: every scored engine cell stores its predictions, and ``neural-trade scenario rescore``
backtests a strategy study on the stored cells without retraining.

The cells come from the engine tests' fake trainer (tests/test_experiment_engine.py): real blocks of
the synthetic bars, predictions = targets seen through seeded noise, so the scorer, the backtest and
the re-scorer run their real code without TensorFlow training.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from neural_trade.core.config import Config
from neural_trade.experiments.rescore import (RescoreError, StrategyStudy, StudyError, leaderboard, rescore,
                                              select_cells)
from neural_trade.experiments.runner import Runner
from neural_trade.experiments.scenario import Scenario
from neural_trade.experiments.scorer import (PREDICTION_FILES, BlockSignals, fit_and_backtest, load_block,
                                             save_predictions, score_result)
from neural_trade.experiments.store import RunStore
from tests.test_experiment_engine import FakeTrainer, fake_result, json_out, spec

HORIZONS = ("h0", "h1", "h2")
REPO = Path(__file__).resolve().parent.parent
STUDY = {"schema_version": 1, "name": "tiny_study", "description": "rescore test",
         "entries": [{"id": "own", "strategy": "calibrated_quantile"},
                     {"id": "cq", "strategy": "calibrated_quantile", "params": {"max_hold": 10},
                      "backtest": {"fee_bps": 5.0}, "grid": {"entry_quantile": [0.8, 0.95]}},
                     {"id": "enhanced", "strategy": "enhanced_multi_horizon"}]}


# ------------------------------------------------------------------ fixtures and helpers
@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("rescore_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def stored(tmp_path_factory, bars_csv):
    """The tiny scenario (folds -2 dev and -1 test x seeds 0, 1) run by the fake trainer: its store and spec."""
    root = tmp_path_factory.mktemp("rescore_store")
    spec_path = root / "tiny.yaml"
    spec_path.write_text(yaml.safe_dump(spec(bars_csv)), encoding="utf-8")
    store = RunStore(root / "runs")
    report = Runner(Scenario.from_yaml(spec_path), store, trainer=FakeTrainer()).run()
    assert len(report.ran) == 4 and not report.failed
    return SimpleNamespace(root=root, spec=spec_path, store=store)


def copy_of(stored, tmp_path):
    """A private copy of the stored scenario (tests that change files work on it)."""
    shutil.copytree(stored.root, tmp_path / "copy")
    return SimpleNamespace(root=tmp_path / "copy", spec=tmp_path / "copy" / "tiny.yaml",
                           store=RunStore(tmp_path / "copy" / "runs"))


def write_study(tmp_path, study=None, name="study.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(copy.deepcopy(study or STUDY)), encoding="utf-8")
    return path


def snapshot(root: Path):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith("index.sqlite")}


def read_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class FakePipeline:
    """A fitted CalibrationPipeline stand-in: delta shrinkage betas, temperature-like probabilities, intervals."""

    delta_scale = {"h0": 0.25, "h1": 0.5, "h2": 0.0}

    def apply(self, preds, windows=None):
        out = {"delta": {h: self.delta_scale[h] * np.asarray(preds["delta"][h], float) for h in HORIZONS},
               "variance": dict(preds["variance"]),
               "direction_prob": {h: 0.5 + 0.8 * (np.asarray(preds["direction_prob"][h], float) - 0.5)
                                  for h in HORIZONS}}
        out["intervals"] = {h: (out["delta"][h] - 3.0, out["delta"][h] + 3.0) for h in HORIZONS}
        return out


def calibrated_result(cfg):
    """fake_result with a calibration pipeline: served delta != raw delta, calibrated P(up), intervals."""
    result = fake_result(cfg)
    result.calibration_pipeline = FakePipeline()
    result.predictions_calibrated = result.calibration_pipeline.apply(result.predictions)
    return result


# ------------------------------------------------------------------ (1) stored predictions
def test_every_scored_cell_stores_its_cal_and_oos_predictions_and_result_json_names_them(stored):
    for d in stored.store.run_dirs("tiny"):
        for name in PREDICTION_FILES.values():
            assert (d / name).is_file(), (d, name)
        doc = json.loads((d / "result.json").read_text(encoding="utf-8"))
        assert doc["predictions"] == {"oos": "predictions_oos.npz", "cal": "predictions_cal.npz"}
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        oos, bars, extra = load_block(d / PREDICTION_FILES["oos"])
        cal, cal_bars, cal_extra = load_block(d / PREDICTION_FILES["cal"])
        assert len(oos) == len(bars) == meta["blocks"]["test"]["n"] and len(cal) == meta["blocks"]["cal"]["n"]
        assert str(extra["anchor_timestamp"][0]) == meta["blocks"]["test"]["first_timestamp"]
        assert str(extra["anchor_timestamp"][-1]) == meta["blocks"]["test"]["last_timestamp"]
        assert str(cal_extra["anchor_timestamp"][0]) == meta["blocks"]["cal"]["first_timestamp"]
        assert float(extra["bar_minutes"]) == 1.0 and str(extra["block"]) == "test" and str(cal_extra["block"]) == "cal"
        assert np.array_equal(bars.close, oos.last_close) and np.array_equal(cal_bars.close, cal.last_close)
        assert oos.split == meta["engine"]["role"] and cal.split == "cal"


def test_a_frame_and_bars_loaded_from_the_npz_reproduce_the_scorers_backtest_exactly(tmp_path, bars_csv):
    from neural_trade.data.processor import split_arrays
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.strategy import Bars

    cfg = Config().override(CSV_PATH=str(bars_csv), MAX_SEQUENCE_COUNT=1500, FOLD_INDEX=-1, SEED=0)
    result = calibrated_result(cfg)
    params = {"random_seeds": 7}
    scored = score_result(result, role="test", strategy="calibrated_quantile", backtest_params=params, out_dir=tmp_path)
    assert scored.paths["predictions_oos"].name == "predictions_oos.npz"

    oos, bars, extra = load_block(tmp_path / "predictions_oos.npz")
    cal, _, _ = load_block(tmp_path / "predictions_cal.npz")
    # everything the scorer's frames held (the input windows X_raw aside) comes back bit for bit
    for name, mem in (("oos", PredictionFrame.from_result(result, "test")), ("cal", PredictionFrame.from_result(result, "cal"))):
        disk = oos if name == "oos" else cal
        np.testing.assert_array_equal(disk.y, mem.y)
        np.testing.assert_array_equal(disk.last_close, mem.last_close)
        assert (disk.pred_scale, disk.pred_mean, disk.horizon_steps) == (mem.pred_scale, mem.pred_mean, mem.horizon_steps)
        for h in HORIZONS:
            for kind in ("delta", "direction_prob", "variance_scaled", "direction_prob_calibrated"):
                np.testing.assert_array_equal(getattr(disk, kind)[h], getattr(mem, kind)[h])
            np.testing.assert_array_equal(disk.meta["delta_raw"][h], mem.meta["delta_raw"][h])
            np.testing.assert_array_equal(disk.intervals[h][0], mem.intervals[h][0])
            np.testing.assert_array_equal(disk.intervals[h][1], mem.intervals[h][1])
        assert disk.meta["delta_scale"] == mem.meta["delta_scale"] == FakePipeline.delta_scale
    # the served delta is the shrunk one; the raw head delta is kept beside it
    np.testing.assert_array_equal(oos.delta["h1"], 0.5 * oos.meta["delta_raw"]["h1"])
    assert not np.any(oos.delta["h2"]) and np.any(oos.meta["delta_raw"]["h2"])

    arrays = split_arrays(cfg)
    ref = Bars.from_frame(arrays["df"], arrays["test"]["anchor_bar"])
    for k in ("open", "high", "low", "close"):
        np.testing.assert_array_equal(getattr(bars, k), getattr(ref, k))
    np.testing.assert_array_equal(extra["anchor_bar"], arrays["test"]["anchor_bar"])

    res, strat = fit_and_backtest(BlockSignals.build(cal, oos), bars, strategy="calibrated_quantile",
                                  backtest_params=params, bar_minutes=float(extra["bar_minutes"]))
    assert set(res.summary) == set(scored.backtest.summary)
    np.testing.assert_equal(res.summary, scored.backtest.summary)          # every key, exactly (NaN == NaN)
    np.testing.assert_equal(res.baselines, scored.backtest.baselines)
    assert strat == scored.strategy
    np.testing.assert_array_equal(res.equity, scored.backtest.equity)

    # the engine never overwrites a run's files
    with pytest.raises(FileExistsError):
        save_predictions(tmp_path, oos, cal, arrays, bar_minutes=1.0)


def test_save_npz_round_trips_a_frame_without_calibration_and_refuses_other_files(tmp_path, bars_csv):
    from neural_trade.evaluation.frame import PredictionFrame

    cfg = Config().override(CSV_PATH=str(bars_csv), MAX_SEQUENCE_COUNT=1500, FOLD_INDEX=-2, SEED=1)
    frame = PredictionFrame.from_result(fake_result(cfg), "test")
    frame.save_npz(tmp_path / "f.npz", extra={"note": "x"})
    back = PredictionFrame.load_npz(tmp_path / "f.npz")
    assert back.direction_prob_calibrated is None and back.intervals is None and "delta_scale" not in back.meta
    assert str(back.meta["extra"]["note"]) == "x" and back.X_raw is None
    for h in HORIZONS:
        np.testing.assert_array_equal(back.delta[h], frame.delta[h])
    with pytest.raises(FileExistsError):
        frame.save_npz(tmp_path / "f.npz")
    np.savez(tmp_path / "gate.npz", y=frame.y)                  # a scripts/gate_run.py-style file
    with pytest.raises(ValueError, match="format_version"):
        PredictionFrame.load_npz(tmp_path / "gate.npz")


# ------------------------------------------------------------------ (2) the study spec
def test_a_study_expands_its_grid_into_configurations_with_ids():
    study = StrategyStudy.from_dict(STUDY)
    confs = study.configurations()
    assert [c.id for c in confs] == ["own", "cq[entry_quantile=0.8]", "cq[entry_quantile=0.95]", "enhanced"]
    cq = confs[1]
    assert cq.params == {"max_hold": 10, "entry_quantile": 0.8} and cq.backtest == {"fee_bps": 5.0}
    assert study.to_dict()["entries"][0] == {"id": "own", "strategy": "calibrated_quantile", "params": {},
                                             "backtest": {}, "grid": {}}
    two = copy.deepcopy(STUDY)
    two["entries"][1]["grid"] = {"entry_quantile": [0.8, 0.9], "require_delta_agreement": [False, True]}
    ids = [c.id for c in StrategyStudy.from_dict(two).configurations()]
    assert "cq[entry_quantile=0.9,require_delta_agreement=true]" in ids and len(ids) == 6


def test_the_example_study_is_valid_and_names_registered_strategies():
    from neural_trade.strategy import Strategies

    study = StrategyStudy.from_yaml(REPO / "configs" / "strategy_studies" / "example.yaml")
    confs = study.configurations()
    assert study.name == "example" and len(confs) == 10
    assert {c.strategy for c in confs} == {"calibrated_quantile", "enhanced_multi_horizon", "liberal", "threshold_spike"}
    assert all(Strategies.has(c.strategy) for c in confs)


def _entry(**changes):
    def change(s):
        s["entries"][1].update(changes)
    return change


BAD_STUDIES = [
    ("top-level typo", lambda s: s.update(entrys=[]), "entrys"),
    ("schema version", lambda s: s.update(schema_version=2), "schema_version"),
    ("unsafe name", lambda s: s.update(name="a b"), "name"),
    ("no entries", lambda s: s.update(entries=[]), "entries"),
    ("unknown entry key", _entry(strat="liberal"), "strat"),
    ("no id", lambda s: s["entries"][0].pop("id"), "id"),
    ("unregistered strategy", _entry(strategy="no_such_strategy"), "no_such_strategy"),
    ("unknown strategy param", _entry(params={"entry_quantil": 0.8}), "entry_quantil"),
    ("unknown grid param", _entry(grid={"max_hld": [5, 10]}), "max_hld"),
    ("empty grid axis", _entry(grid={"entry_quantile": []}), "non-empty"),
    ("repeated grid value", _entry(grid={"entry_quantile": [0.8, 0.8]}), "repeats"),
    ("param and grid clash", _entry(grid={"max_hold": [5, 10]}), "both params and grid"),
    ("knob fitted on cal", _entry(params={"long_above": 0.6}), "long_above"),
    ("unknown backtest param", _entry(backtest={"fee_bsp": 5}), "fee_bsp"),
    ("engine-owned backtest field", _entry(backtest={"bar_minutes": 5}), "bar_minutes"),
    ("same-bar fills", _entry(backtest={"fill": "close"}), "next_open"),
    ("repeated id", lambda s: s["entries"].append({"id": "own", "strategy": "liberal"}), "repeat"),
    ("unknown notebook-strategy knob", lambda s: s["entries"].append({"id": "lib", "strategy": "liberal",
                                                                      "params": {"tp_mult": 1}}), "tp_mult"),
]


@pytest.mark.parametrize("what, change, match", BAD_STUDIES, ids=[b[0] for b in BAD_STUDIES])
def test_a_bad_study_is_refused_before_anything_runs(tmp_path, stored, what, change, match, capsys):
    from neural_trade.cli import main

    s = copy.deepcopy(STUDY)
    change(s)
    with pytest.raises(StudyError, match=match):
        StrategyStudy.from_dict(s)
    before = snapshot(stored.store.root)
    assert main(["scenario", "rescore", str(stored.spec), "--study", str(write_study(tmp_path, s)),
                 "--store", str(stored.store.root)]) == 2
    assert snapshot(stored.store.root) == before and not (stored.store.scenario_dir("tiny") / "rescore").exists()


# ------------------------------------------------------------------ (3) the command
def test_rescore_writes_cells_leaderboard_study_and_meta_into_a_new_directory(tmp_path, stored, capsys):
    from neural_trade.cli import main

    st = copy_of(stored, tmp_path)
    cells_before = {d.name: snapshot(d) for d in st.store.run_dirs("tiny")}
    study_path = write_study(tmp_path)
    assert main(["scenario", "rescore", str(st.spec), "--study", str(study_path), "--store", str(st.store.root)]) == 0
    out = json_out(capsys)
    d = Path(out["out_dir"])
    assert d.parent == st.store.scenario_dir("tiny") / "rescore" and d.name.startswith("tiny_study-")
    assert sorted(p.name for p in d.iterdir()) == ["cells.csv", "leaderboard.csv", "leaderboard.md", "meta.json",
                                                   "study.yaml"]
    assert {d.name: snapshot(d) for d in st.store.run_dirs("tiny")} == cells_before     # cells untouched
    assert out["n_configurations"] == 4 and out["n_cells"] == 4 and out["skipped"] == [] and out["missing"] == []

    rows = read_csv(d / "cells.csv")
    assert len(rows) == 4 * 4
    for key in ("config_id", "run_dir", "fold", "seed", "role", "sharpe_net", "total_return", "max_drawdown",
                "n_trades", "buy_and_hold/total_return", "always_flat/total_return",
                "random_same_freq/percentile_sharpe_net", "random_same_freq/n_seeds"):
        assert key in rows[0], key
    assert {(r["fold"], r["role"]) for r in rows} == {("-2", "dev"), ("-1", "test")}
    assert {r["random_same_freq/n_seeds"] for r in rows if r["n_trades"] != "0"} == {"5"}   # the scenario's setting

    board = read_csv(d / "leaderboard.csv")
    assert [r["rank"] for r in board] == ["1", "2", "3", "4"] and len({r["config_id"] for r in board}) == 4
    for r in board:
        mine = [float(x["sharpe_net"]) for x in rows if x["config_id"] == r["config_id"] and x["role"] == "dev"]
        assert float(r["dev_sharpe_net_mean"]) == pytest.approx(np.mean(mine))
        assert float(r["dev_sharpe_net_sd"]) == pytest.approx(np.std(mine, ddof=1))
        assert r["n_dev"] == "2" and r["n_test"] == "2"
        for col in ("dev_total_return_mean", "dev_max_drawdown_mean", "dev_n_trades_mean",
                    "dev_beats_buy_and_hold_frac", "test_sharpe_net_mean", "test_total_return_mean",
                    "test_max_drawdown_mean", "test_n_trades_mean", "test_beats_buy_and_hold_frac"):
            assert r[col] != "", col
    means = [float(r["dev_sharpe_net_mean"]) for r in board]
    assert means == sorted(means, reverse=True)
    md = (d / "leaderboard.md").read_text(encoding="utf-8")
    assert "Ranked by the mean net Sharpe over the dev cells" in md and "never used to rank or choose" in md

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    sc = Scenario.from_yaml(st.spec)
    assert meta["scenario_spec_hash"] == sc.spec_hash and meta["git_sha"] and meta["n_configurations"] == 4
    assert sorted(c["cell_key"] for c in meta["cells"]) == sorted(c.key for c in sc.cells())
    assert yaml.safe_load((d / "study.yaml").read_text(encoding="utf-8")) == StrategyStudy.from_dict(STUDY).to_dict()

    # a second re-score goes into another new directory
    assert main(["scenario", "rescore", str(st.spec), "--study", str(study_path), "--store", str(st.store.root),
                 "--random-seeds", "3"]) == 0
    other = Path(json_out(capsys)["out_dir"])
    assert other != d and (d / "cells.csv").exists()
    again = read_csv(other / "cells.csv")
    assert {r["random_same_freq/n_seeds"] for r in again if r["n_trades"] != "0"} == {"3"}   # --random-seeds


# ------------------------------------------------------------------ (5) the scenario's own strategy reproduces result.json
def test_rescoring_the_scenarios_own_strategy_reproduces_every_cells_result_json(tmp_path, stored):
    st = copy_of(stored, tmp_path)
    sc = Scenario.from_yaml(st.spec)
    own = {"schema_version": 1, "name": "own", "entries": [{"id": "own", "strategy": sc.strategy.name,
                                                             "params": sc.strategy.params}]}
    report = rescore(sc, StrategyStudy.from_dict(own), st.store)
    assert len(report.rows) == 4
    for row in report.rows:
        scores = json.loads((st.store.root / row["run_dir"] / "result.json").read_text(encoding="utf-8"))["scores"]
        for key in ("sharpe_net", "total_return", "n_trades"):
            assert row[key] == scores[f"backtest/{key}"], (row["cell_key"], key)       # exactly
        for key, value in scores.items():            # and every other backtest number the scorer indexed
            if key.startswith("backtest/") and value is not None:
                name = key[len("backtest/"):]
                assert row[name] == value, (row["cell_key"], key)


# ------------------------------------------------------------------ ranking never uses the test cells
def test_the_ranking_ignores_the_test_cells():
    study = StrategyStudy.from_dict(STUDY)
    confs = study.configurations()
    rng = np.random.default_rng(0)
    rows = []
    for i, conf in enumerate(confs):
        for fold, role in ((-3, "dev"), (-2, "dev"), (-1, "test")):
            for seed in (0, 1):
                rows.append({"config_id": conf.id, "role": role, "fold": fold, "seed": seed,
                             "sharpe_net": float(i) + rng.normal(0, 0.1), "total_return": 0.01 * i,
                             "max_drawdown": 0.05, "n_trades": 10, "beats_buy_and_hold": i % 2,
                             "random_same_freq/percentile_sharpe_net": 50.0})
    order = [r["config_id"] for r in leaderboard(rows, confs)]
    assert order == [c.id for c in reversed(confs)]
    flipped = copy.deepcopy(rows)
    for r in flipped:
        if r["role"] == "test":
            r["sharpe_net"] = -1000.0 * r["sharpe_net"] + rng.normal(0, 50)
            r["total_return"] = -r["total_return"]
    board = leaderboard(flipped, confs)
    assert [r["config_id"] for r in board] == order                            # test numbers changed, order did not
    assert [r["test_sharpe_net_mean"] for r in board] != [r["test_sharpe_net_mean"] for r in leaderboard(rows, confs)]
    moved = copy.deepcopy(rows)
    for r in moved:
        if r["role"] == "dev" and r["config_id"] == confs[0].id:
            r["sharpe_net"] += 100.0
    assert leaderboard(moved, confs)[0]["config_id"] == confs[0].id              # dev numbers do move it


# ------------------------------------------------------------------ (4) refusals
def test_a_cell_without_stored_predictions_is_skipped_and_listed(tmp_path, stored, capsys):
    from neural_trade.cli import main

    st = copy_of(stored, tmp_path)
    dirs = st.store.run_dirs("tiny")
    dev = [d for d in dirs if "__f-2__" in d.name]
    (dev[0] / "predictions_oos.npz").unlink()               # as if scored before NT-076
    assert main(["scenario", "rescore", str(st.spec), "--study", str(write_study(tmp_path)),
                 "--store", str(st.store.root)]) == 0
    out = json_out(capsys)
    assert out["n_cells"] == 3 and len(out["skipped"]) == 1
    [skip] = out["skipped"]
    assert skip["run_dir"].endswith(dev[0].name) and "no stored predictions" in skip["reason"]
    meta = json.loads((Path(out["out_dir"]) / "meta.json").read_text(encoding="utf-8"))
    assert meta["skipped"] == out["skipped"] and len(meta["cells"]) == 3
    assert out["missing"] == [json.loads((dev[0] / "meta.json").read_text(encoding="utf-8"))["engine"]["cell_key"]]


def test_a_scenario_without_a_dev_cell_with_predictions_exits_non_zero_and_writes_nothing(tmp_path, stored, capsys):
    from neural_trade.cli import main

    st = copy_of(stored, tmp_path)
    for d in st.store.run_dirs("tiny"):
        if "__f-2__" in d.name:
            (d / "predictions_cal.npz").unlink()
    before = snapshot(st.store.root)
    code = main(["scenario", "rescore", str(st.spec), "--study", str(write_study(tmp_path)),
                 "--store", str(st.store.root)])
    assert code == 1
    assert snapshot(st.store.root) == before and not (st.store.scenario_dir("tiny") / "rescore").exists()
    with pytest.raises(RescoreError, match="no dev cell has stored predictions"):
        rescore(Scenario.from_yaml(st.spec), StrategyStudy.from_dict(STUDY), st.store)


def test_only_finished_runs_of_the_specs_cells_are_used_and_the_newest_run_of_a_cell_wins(tmp_path, stored):
    st = copy_of(stored, tmp_path)
    sc = Scenario.from_yaml(st.spec)
    # the same cell again under a changed spec: another config hash, so not a cell of this spec
    changed = Scenario.from_dict(spec(Path(sc.overrides["CSV_PATH"]), folds=[-1], seeds=[0],
                                      overrides={**sc.overrides, "EPOCHS": 2}))
    Runner(changed, st.store, trainer=FakeTrainer()).run()
    first, second = st.store.run_dirs("tiny")[:2]
    # an interrupted run (no result.json) and a newer finished run of the same cell as ``first``
    half = shutil.copytree(first, first.parent / (first.name + "-half"))
    (half / "result.json").unlink()
    newer = shutil.copytree(first, first.parent / (first.name + "-newer"))
    meta = json.loads((newer / "meta.json").read_text(encoding="utf-8"))
    meta.update(run_id=meta["run_id"] + "-newer", created_utc="99991231T000000Z")
    (newer / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    used, skipped, missing = select_cells(sc, st.store)
    assert sorted(c.cell_key for c in used) == sorted(c.key for c in sc.cells()) and missing == []
    assert newer in [c.run_dir for c in used] and first not in [c.run_dir for c in used]
    reasons = {Path(s["run_dir"]).name: s["reason"] for s in skipped}
    assert reasons[half.name] == "status incomplete"
    assert reasons[first.name].startswith("an older run of the same cell")
    assert sum(r.startswith("config hash differs") for r in reasons.values()) == 1
    assert second in [c.run_dir for c in used] and len(skipped) == 3


def test_the_rescore_module_does_not_build_on_the_frozen_set():
    from tests.test_experiment_engine import FROZEN_MODULES, _imports

    names = {part for name in _imports(REPO / "src" / "neural_trade" / "experiments" / "rescore.py")
             for part in name.split(".")}
    assert not names & set(FROZEN_MODULES)

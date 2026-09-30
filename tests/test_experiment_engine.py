"""Experiment engine (NT-026): the scenario spec, the resumable runner, the run store and its sqlite
index, the one scorer, the CLI, pick_run's guard and the frozen set's header notes.

The fast tests use a fake trainer: a TrainResult stand-in built on the fold's real blocks
(data.processor.split_arrays), whose predictions are the targets seen through seeded noise, so
the scorer, the backtest and the index run their real code without TensorFlow training. The slow
test trains for real on CPU through ``neural-trade scenario run``.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from neural_trade.core.config import Config
from neural_trade.experiments.runner import Runner
from neural_trade.experiments.scenario import Scenario, ScenarioError, config_hash, config_hash_of_dir, config_identity
from neural_trade.experiments.store import RunStore

HORIZONS = ("h0", "h1", "h2")
REPO = Path(__file__).resolve().parent.parent
# columns of the runs table that differ between two runs of the same cell (ids, clocks)
VOLATILE = {"run_id", "run_dir", "created_utc", "finished_utc", "wall_s", "sec_per_step"}


# ------------------------------------------------------------------ helpers
def fake_result(cfg: Config):
    """A TrainResult stand-in for ``cfg``'s fold: real blocks, predictions = targets + seeded noise."""
    from sklearn.preprocessing import StandardScaler

    from neural_trade.data.processor import split_arrays

    arrays = split_arrays(cfg)
    rng = np.random.default_rng([int(cfg.SEED), int(cfg.FOLD_INDEX) % 97])
    scaler = StandardScaler().fit(arrays["train"]["y"].reshape(-1, 1))
    scale = float(scaler.scale_[0])

    def heads(block):
        y = np.asarray(block["y"], float)
        signal = y / scale + rng.normal(0.0, 3.0, y.shape)
        return {"delta": {h: 0.1 * scale * signal[:, i] for i, h in enumerate(HORIZONS)},
                "direction_prob": {h: 1.0 / (1.0 + np.exp(-0.5 * signal[:, i])) for i, h in enumerate(HORIZONS)},
                "variance": {h: (1.0 + 0.1 * i) * (1.0 + 0.2 * rng.random(len(y))) for i, h in enumerate(HORIZONS)}}

    return SimpleNamespace(
        config=cfg, target_scaler=scaler, predictions=heads(arrays["test"]), y_test=arrays["test"]["y"],
        last_close_test=arrays["test"]["last_close"], predictions_calibrated=None, windows_test=arrays["test"]["X"],
        predictions_cal=heads(arrays["cal"]), y_cal=arrays["cal"]["y"], last_close_cal=arrays["cal"]["last_close"],
        windows_cal=arrays["cal"]["X"], calibration_pipeline=None, history=None, weights_epoch=1,
        weights_val_loss=0.5)


class FakeTrainer:
    """Counts calls by cell key; ``fail`` raises ValueError for those cells, ``interrupt_at`` raises
    KeyboardInterrupt on that call (1-based)."""

    def __init__(self, fail=(), interrupt_at=None):
        self.calls = []
        self.fail = set(fail)
        self.interrupt_at = interrupt_at

    def __call__(self, ctx, *, calibrate, save_artifacts):
        key = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))["engine"]["cell_key"]
        self.calls.append(key)
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        if key in self.fail:
            raise ValueError(f"injected failure in {key}")
        return fake_result(ctx.config)


def spec(csv, **changes):
    """A tiny scenario: 2 folds x 2 seeds of one variant on the synthetic bars."""
    s = {"schema_version": 1, "name": "tiny", "description": "engine test",
         "overrides": {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 1, "BATCH_SIZE": 32},
         "variants": {"default": {}}, "folds": [-2, -1], "seeds": [0, 1],
         "strategy": {"name": "calibrated_quantile", "params": {}}, "backtest": {"random_seeds": 5},
         "run": {"calibrate": False, "save_artifacts": False}}
    s.update(copy.deepcopy(changes))
    return s


def comparable(store: RunStore, status="done"):
    """The index of ``store`` without ids and clocks: {cell key: row}, {cell key: scores}."""
    idx = store.index
    rows = idx.rows(status=status)
    return ({r["cell_key"]: {k: v for k, v in r.items() if k not in VOLATILE} for r in rows},
            {r["cell_key"]: idx.scores(r["run_id"]) for r in rows})


def snapshot(root: Path):
    """{relative path: sha256} of every file under ``root`` except the (derived) index."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith("index.sqlite")}


def json_out(capsys):
    out = capsys.readouterr().out
    return json.JSONDecoder().raw_decode(out[out.index("{"):])[0]


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("engine_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory, bars_csv):
    """The tiny scenario run in one go: the reference the resumed runs are compared with."""
    store = RunStore(tmp_path_factory.mktemp("engine_uninterrupted") / "runs")
    trainer = FakeTrainer()
    report = Runner(Scenario.from_dict(spec(bars_csv)), store, trainer=trainer).run()
    assert len(report.ran) == 4 and not report.failed and len(trainer.calls) == 4
    return store


# ------------------------------------------------------------------ (1) the spec
def test_a_spec_expands_variants_axes_folds_and_seeds_into_cells_with_stable_hashes(tmp_path, bars_csv):
    s = spec(bars_csv, variants={"base": {}, "slow": {"LR": 0.0005}},
             sweep={"mode": "grid", "axes": {"BATCH_SIZE": [32, 64]}})
    sc = Scenario.from_dict(s)
    cells = sc.cells()
    assert len(cells) == 2 * 2 * 2 * 2 and len({c.key for c in cells}) == 16
    assert {c.configuration.name for c in cells} == {"base__batch_size-32", "base__batch_size-64",
                                                    "slow__batch_size-32", "slow__batch_size-64"}
    one = next(c for c in cells if c.key == "slow__batch_size-64__f-1__s1")
    cfg = one.config()
    assert (cfg.LR, cfg.BATCH_SIZE, cfg.FOLD_INDEX, cfg.SEED, cfg.EPOCHS) == (0.0005, 64, -1, 1, 1)
    assert one.configuration.params == {"LR": 0.0005, "BATCH_SIZE": 64}
    assert sc.to_dict()["schema_version"] == 1

    assert Scenario.from_dict(copy.deepcopy(s)).spec_hash == sc.spec_hash
    longer = Scenario.from_dict(spec(bars_csv, overrides={**s["overrides"], "EPOCHS": 2}))
    assert longer.spec_hash != sc.spec_hash and longer.settings_hash == Scenario.from_dict(spec(bars_csv)).settings_hash
    assert config_hash(longer.cells()[0].config()) != config_hash(Scenario.from_dict(spec(bars_csv)).cells()[0].config())
    cheaper = Scenario.from_dict(spec(bars_csv, backtest={"random_seeds": 5, "fee_bps": 5.0}))
    assert cheaper.settings_hash != Scenario.from_dict(spec(bars_csv)).settings_hash

    # from YAML, with a base_config relative to the spec file; the default variant when none is given
    (tmp_path / "base.yaml").write_text("EPOCHS: 3\nLR: 0.002\n", encoding="utf-8")
    on_disk = {k: v for k, v in spec(bars_csv).items() if k != "variants"}
    on_disk.update(base_config="base.yaml", overrides={"CSV_PATH": str(bars_csv), "MAX_SEQUENCE_COUNT": 1500})
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(on_disk), encoding="utf-8")
    loaded = Scenario.from_yaml(tmp_path / "s.yaml")
    first = loaded.cells()[0]
    assert first.key == "default__f-2__s0" and (first.config().EPOCHS, first.config().LR) == (3, 0.002)


def test_config_identity_ignores_default_valued_fields_nt083(bars_csv):
    """NT-083: config_hash hashes only the fields that differ from Config()'s default, so a Config
    field added later that a cell (or the current spec) leaves at its default does not appear in
    the identity, while overriding it away from default does."""
    base = Config().override(CSV_PATH=str(bars_csv), MAX_SEQUENCE_COUNT=1500)
    at_default = base.copy()                    # LR untouched: at Config()'s default
    away_from_default = base.copy(LR=0.0005)     # LR set away from default

    assert "LR" not in config_identity(at_default)
    assert config_identity(away_from_default)["LR"] == pytest.approx(0.0005)
    assert config_hash(at_default) == config_hash(base)          # an explicit default changes nothing
    assert config_hash(away_from_default) != config_hash(at_default)

    # explicitly re-setting a field back to its own default is the same identity as never touching it
    back_to_default = away_from_default.copy(LR=Config().LR)
    assert config_hash(back_to_default) == config_hash(at_default)


def test_config_hash_of_dir_recomputes_from_config_yaml_not_the_recorded_meta_json_value(tmp_path, bars_csv):
    """NT-083: config_hash_of_dir is the same identity hash config_hash would compute for the Config
    that config.yaml describes; it does not read meta.json's (possibly stale) config_hash at all."""
    sc = Scenario.from_dict(spec(bars_csv, folds=[-1], seeds=[0]))
    store = RunStore(tmp_path / "runs")
    Runner(sc, store, trainer=FakeTrainer()).run()
    [run_dir] = store.run_dirs("tiny")
    cfg_from_disk = Config.from_yaml(run_dir / "config.yaml")
    assert config_hash_of_dir(run_dir) == config_hash(cfg_from_disk)

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    meta["engine"]["config_hash"] = "garbage"
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert config_hash_of_dir(run_dir) == config_hash(cfg_from_disk)     # unaffected by meta.json

    assert config_hash_of_dir(tmp_path / "no-such-run-dir") is None


def _clash(s):
    s["variants"] = {"a": {"LR": 0.002}}
    s["sweep"] = {"axes": {"LR": [0.001, 0.0005]}}


BAD_SPECS = [
    ("top-level typo", lambda s: s.update(seedz=[1]), "seedz"),
    ("unknown run option", lambda s: s["run"].update(epochs=3), "run.epochs"),
    ("unknown Config field", lambda s: s["overrides"].update(LAMDA_HD=0.1), "did you mean LAMBDA_HD"),
    ("invalid Config value", lambda s: s["variants"].update(small={"BATCH_SIZE": 4}), "BATCH_SIZE"),
    ("uninterpretable value", lambda s: s["variants"].update(odd={"EPOCHS": "many"}), "EPOCHS"),
    ("engine-owned field", lambda s: s["overrides"].update(SEED=3), "SEED"),
    ("engine-owned axis", lambda s: s.update(sweep={"axes": {"FOLD_INDEX": [-1, -2]}}), "FOLD_INDEX"),
    ("invalid axis value", lambda s: s.update(sweep={"axes": {"LR": [0.001, 5.0]}}), "LR"),
    ("axis also set by a variant", _clash, "also sweeps"),
    ("sweep mode not built yet", lambda s: s.update(sweep={"mode": "optuna"}), "optuna"),
    ("unknown strategy", lambda s: s.update(strategy={"name": "no_such_strategy"}), "no_such_strategy"),
    ("unknown strategy knob", lambda s: s["strategy"]["params"].update(entry_quantil=0.8), "entry_quantil"),
    ("knob fitted on cal", lambda s: s["strategy"]["params"].update(long_above=0.6), "long_above"),
    ("unknown cost field", lambda s: s["backtest"].update(fee_bsp=5), "fee_bsp"),
    ("engine-owned cost field", lambda s: s["backtest"].update(bar_minutes=5), "bar_minutes"),
    ("same-bar fills", lambda s: s["backtest"].update(fill="close"), "next_open"),
    ("schema version", lambda s: s.update(schema_version=2), "schema_version"),
    ("unsafe name", lambda s: s.update(name="a b"), "name"),
    ("no seeds", lambda s: s.update(seeds=[]), "seeds"),
    ("repeated fold", lambda s: s.update(folds=[-1, -1]), "folds"),
    ("fold outside Config's range", lambda s: s.update(folds=[-9]), "FOLD_INDEX"),
    ("fold the data does not have", lambda s: s.update(folds=[-5]), "usable folds"),
    ("the same fold twice", lambda s: s.update(folds=[-1, 3]), "same fold"),
    ("unregistered component", lambda s: s["overrides"].update(MODEL_NAME="no_such_model"), "no_such_model"),
    ("strided windows", lambda s: s["overrides"].update(WINDOW_STEP=2), "WINDOW_STEP"),
    ("missing data file", lambda s: s["overrides"].update(CSV_PATH="no/such/bars.csv"), "does not exist"),
]


@pytest.mark.parametrize("what, change, match", BAD_SPECS, ids=[b[0] for b in BAD_SPECS])
def test_a_bad_spec_is_refused_before_any_run_starts(tmp_path, bars_csv, what, change, match):
    s = spec(bars_csv)
    change(s)
    trainer = FakeTrainer()
    with pytest.raises(ScenarioError, match=match):
        Runner(Scenario.from_dict(s), tmp_path / "runs", trainer=trainer).run()
    assert trainer.calls == [] and not (tmp_path / "runs").exists()


@pytest.mark.data
def test_the_reference_example_spec_is_valid_on_the_bundled_data(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)                                 # CSV_PATH is relative to the repository root
    sc = Scenario.from_yaml(REPO / "configs" / "scenarios" / "reference.yaml")
    planned = Runner(sc, tmp_path / "runs", trainer=FakeTrainer()).plan()
    assert sc.name == "reference_default" and sc.strategy.name == "calibrated_quantile"
    assert len(planned) == 9
    assert {pc.cell.fold: pc.role for pc in planned} == {-3: "dev", -2: "dev", -1: "test"}
    sha = hashlib.sha256((REPO / "binance_btcusdt_1min_ccxt.csv").read_bytes()).hexdigest()
    assert {pc.dataset["sha256"] for pc in planned} == {sha}
    assert planned[0].setup == {"bar_minutes": 1, "LOOKBACK": 60, "HORIZON_STEPS": [10, 15, 20]}
    assert all(pc.state == "pending" for pc in planned) and not (tmp_path / "runs").exists()   # plan writes nothing


# ------------------------------------------------------------------ (2) resume
def test_a_stopped_scenario_resumes_without_rerunning_and_matches_an_uninterrupted_run(tmp_path, bars_csv,
                                                                                       uninterrupted):
    sc = Scenario.from_dict(spec(bars_csv))
    store = RunStore(tmp_path / "runs")
    trainer = FakeTrainer()
    first = Runner(sc, store, trainer=trainer).run(max_cells=1)
    assert [r["cell"] for r in first.ran] == ["default__f-2__s0"] and len(first.pending) == 3
    assert [r["status"] for r in store.index.rows()] == ["done"]
    second = Runner(sc, store, trainer=trainer).run()
    assert second.skipped == ["default__f-2__s0"] and len(second.ran) == 3 and not second.pending
    assert len(trainer.calls) == 4 and len(set(trainer.calls)) == 4           # no cell trained twice
    assert comparable(store) == comparable(uninterrupted)                     # the index of an uninterrupted run
    assert len(store.index.rows()) == len(uninterrupted.index.rows()) == 4
    third = Runner(sc, store, trainer=trainer).run()
    assert third.ran == [] and len(third.skipped) == 4 and len(trainer.calls) == 4


def test_a_config_field_that_predates_a_cell_does_not_retrain_it_unless_set_away_from_default(tmp_path, bars_csv):
    """NT-083: simulates a Config field added after a cell ran, by deleting it from that cell's
    stored config.yaml and replacing its meta.json config_hash with a value no current code would
    compute (a stand-in for a hash recorded under an earlier hash rule)."""
    sc = Scenario.from_dict(spec(bars_csv, folds=[-1], seeds=[0]))
    store = RunStore(tmp_path / "runs")
    Runner(sc, store, trainer=FakeTrainer()).run()
    [run_dir] = store.run_dirs("tiny")

    def drop_lr_and_stale_the_recorded_hash():
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        del cfg["LR"]                                                    # as if LR did not exist yet
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        meta["engine"]["config_hash"] = "stale-hash-from-an-earlier-engine-version"
        (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    drop_lr_and_stale_the_recorded_hash()
    # LR stays at its default in the current spec too: the cell is recognised as done, and the
    # stale, unrecognisable recorded hash is not trusted for that decision
    [pc] = Runner(sc, store, trainer=FakeTrainer()).plan()
    assert pc.state == "done" and pc.runs == [run_dir.name]

    # the current spec now sets LR away from its default: the field's absence no longer matches
    assert Config().LR != 0.0007
    away = Scenario.from_dict(spec(bars_csv, folds=[-1], seeds=[0], overrides={**sc.overrides, "LR": 0.0007}))
    [pc2] = Runner(away, store, trainer=FakeTrainer()).plan()
    assert pc2.state == "pending" and pc2.runs == []


def test_an_interrupted_cell_is_kept_indexed_incomplete_and_trained_again(tmp_path, bars_csv, uninterrupted):
    sc = Scenario.from_dict(spec(bars_csv))
    store = RunStore(tmp_path / "runs")
    trainer = FakeTrainer(interrupt_at=2)
    with pytest.raises(KeyboardInterrupt):
        Runner(sc, store, trainer=trainer).run()
    status = {r["cell_key"]: r["status"] for r in store.index.rows()}
    assert status == {"default__f-2__s0": "done", "default__f-2__s1": "incomplete"}
    [half] = store.index.rows(status="incomplete")
    half_dir = store.root / half["run_dir"]
    kept = snapshot(half_dir)
    assert "meta.json" in kept and "result.json" not in kept

    trainer.interrupt_at = None
    report = Runner(sc, store, trainer=trainer).run()
    assert report.skipped == ["default__f-2__s0"]
    assert [r["cell"] for r in report.ran] == ["default__f-2__s1", "default__f-1__s0", "default__f-1__s1"]
    assert snapshot(half_dir) == kept                                         # the partial run is kept as it was
    assert comparable(store) == comparable(uninterrupted)
    assert [r["cell_key"] for r in store.index.rows(status="incomplete")] == ["default__f-2__s1"]


def test_a_failed_cell_is_recorded_not_dropped_and_retried_only_when_asked(tmp_path, bars_csv):
    sc = Scenario.from_dict(spec(bars_csv, folds=[-1]))
    store = RunStore(tmp_path / "runs")
    trainer = FakeTrainer(fail={"default__f-1__s1"})
    report = Runner(sc, store, trainer=trainer).run()
    assert report.failed == ["default__f-1__s1"] and [r["status"] for r in report.ran] == ["done", "failed"]
    failed = store.index.rows(status="failed")[0]
    assert "ValueError: injected failure" in failed["error"] and failed["sharpe_net"] is None
    doc = json.loads((store.root / failed["run_dir"] / "result.json").read_text(encoding="utf-8"))
    assert doc["status"] == "failed" and doc["error"]["type"] == "ValueError" and "Traceback" in doc["error"]["traceback"]

    assert Runner(sc, store, trainer=trainer).run().ran == [] and len(trainer.calls) == 2
    trainer.fail.clear()
    again = Runner(sc, store, trainer=trainer).run(retry_failed=True)
    assert [(r["cell"], r["status"]) for r in again.ran] == [("default__f-1__s1", "done")]
    statuses = sorted((r["cell_key"], r["status"]) for r in store.index.rows())
    assert statuses == [("default__f-1__s0", "done"), ("default__f-1__s1", "done"), ("default__f-1__s1", "failed")]
    assert (store.root / failed["run_dir"] / "result.json").exists()           # the failed run stays


# ------------------------------------------------------------------ (3) never overwrite
def test_the_runner_never_writes_into_an_existing_directory_or_changes_a_finished_run(tmp_path, bars_csv,
                                                                                     monkeypatch):
    from neural_trade.experiments import run_context

    class Frozen:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(run_context, "datetime", Frozen)
    monkeypatch.setattr(run_context, "git_sha", lambda: "abc1234")
    sc = Scenario.from_dict(spec(bars_csv, folds=[-1], seeds=[0]))
    store = RunStore(tmp_path / "runs")
    runner = Runner(sc, store, trainer=FakeTrainer())
    [pc] = runner.plan()
    taken = store.scenario_dir(sc.name) / f"20260102T030405Z-abc1234-{run_context.config_hash(pc.config)}-{pc.key}"
    taken.mkdir(parents=True)
    (taken / "evidence.txt").write_text("an earlier run's file", encoding="utf-8")
    report = runner.run()
    [ran] = report.ran
    assert ran["status"] == "done" and Path(ran["run_dir"]).name == taken.name + "-2"
    assert [p.name for p in taken.iterdir()] == ["evidence.txt"]
    assert (taken / "evidence.txt").read_text(encoding="utf-8") == "an earlier run's file"

    # a changed spec trains every cell again into new directories; nothing that existed changes or goes
    monkeypatch.undo()
    before = snapshot(store.root)
    changed = Scenario.from_dict(spec(bars_csv, folds=[-1], seeds=[0],
                                      overrides={**spec(bars_csv)["overrides"], "EPOCHS": 2}))
    assert len(Runner(changed, store, trainer=FakeTrainer()).run().ran) == 1
    after = snapshot(store.root)
    assert {k: after.get(k) for k in before} == before and len(after) > len(before)


# ------------------------------------------------------------------ (4) the scorer
def test_every_run_is_scored_on_its_out_of_sample_block_as_dev_or_test_with_the_leaderboard_numbers(uninterrupted):
    rows = uninterrupted.index.rows()
    assert {(r["fold"], r["role"]) for r in rows} == {(-2, "dev"), (-1, "test")}
    for row in rows:
        d = uninterrupted.root / row["run_dir"]
        role = row["role"]
        assert sorted(p.name for p in d.glob("eval_report_*")) == [f"eval_report_{role}.json", f"eval_report_{role}.md"]
        report = json.loads((d / f"eval_report_{role}.json").read_text(encoding="utf-8"))
        assert report["split"] == role and report["meta"]["role"] == role and report["meta"]["ranks"] == (role == "dev")
        assert report["meta"]["blocks"]["test"]["n"] == report["n"] == 250
        assert np.isfinite(report["model"]["horizons"]["h1"]["direction"]["auc"])
        assert set(report["baselines"]) >= {"logreg_lags", "zero_delta", "const_var"}
        bt = report["backtest"]
        assert bt["strategy"] == "calibrated_quantile" and bt["fitted_on"] == "cal"
        assert {"sharpe_net", "max_drawdown", "n_trades", "total_return"} <= set(bt["summary"])
        null = bt["baselines"]["random_same_freq"]
        assert null["n_seeds"] == 5 and null["size_frac"] == 1.0          # calibrated_quantile trades at size 1
        for key in ("random_p05_total_return", "random_p95_total_return", "random_mean_gross_return",
                    "percentile_gross_return", "percentile_total_return"):
            assert key in null
        assert {"buy_and_hold", "always_flat"} <= set(bt["baselines"])
        scores = uninterrupted.index.scores(row["run_id"])
        assert row["sharpe_net"] == scores["backtest/sharpe_net"] == pytest.approx(bt["summary"]["sharpe_net"])
        assert row["n_trades"] == bt["summary"]["n_trades"] and row["max_drawdown"] == pytest.approx(
            bt["summary"]["max_drawdown"])
        assert row["buy_and_hold_return"] == pytest.approx(bt["baselines"]["buy_and_hold"]["total_return"])
        assert row["random_percentile_return"] == pytest.approx(null["percentile_total_return"])
        assert scores["baseline/logreg_lags/h1/direction/auc"] == pytest.approx(
            report["baselines"]["logreg_lags"]["horizons"]["h1"]["direction"]["auc"])
        md = (d / f"eval_report_{role}.md").read_text(encoding="utf-8")
        assert f"Role: **{role}**" in md and "random null" in md and f"# Evaluation report - {role} split" in md


def test_the_scorer_fits_the_strategy_on_the_cal_block_and_fills_at_the_next_open_with_the_default_costs(bars_csv):
    from neural_trade.data.processor import split_arrays
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.experiments.scorer import ScoringError, score_result
    from neural_trade.strategy import BacktestConfig, Bars, SignalFrame, var_scale_from

    cfg = Config().override(CSV_PATH=str(bars_csv), MAX_SEQUENCE_COUNT=1500, FOLD_INDEX=-1, SEED=0)
    result = fake_result(cfg)
    scored = score_result(result, role="test", backtest_params={"random_seeds": 5})
    cal = PredictionFrame.from_result(result, "cal")
    vs = var_scale_from(cal)
    w = SignalFrame.build(cal, vs).weighted_direction
    s = scored.strategy
    assert (s.long_above, s.short_below, s.median) == pytest.approx(
        (np.quantile(w, 0.9), np.quantile(w, 0.1), np.quantile(w, 0.5)))
    assert scored.report.backtest["var_scale"] == pytest.approx(vs)
    assert scored.report.backtest["params"]["long_above"] == pytest.approx(s.long_above)

    # other TEST predictions leave the fitted knobs alone: they come from the cal block only
    other = copy.copy(result)
    other.predictions = {k: {h: np.asarray(v) * (1.5 if k == "delta" else 1.0) for h, v in d.items()}
                         for k, d in result.predictions.items()}
    other.predictions["direction_prob"] = {h: 1.0 - p for h, p in result.predictions["direction_prob"].items()}
    moved = score_result(other, role="test", backtest_params={"random_seeds": 5})
    assert (moved.strategy.long_above, moved.strategy.short_below) == (s.long_above, s.short_below)
    assert moved.report.backtest["var_scale"] == scored.report.backtest["var_scale"]

    # next-open fills and the default cost profile (13 bps per side)
    bc = scored.backtest.config
    assert (bc.fill, bc.fee_bps, bc.half_spread_bps, bc.slippage_bps) == ("next_open", 10.0, 1.0, 2.0)
    assert bc.bar_minutes == 1.0 and bc.minutes_per_year == BacktestConfig().minutes_per_year
    arrays = split_arrays(cfg)
    bars = Bars.from_frame(arrays["df"], arrays["test"]["anchor_bar"])
    decided = {d["bar"] for d in scored.backtest.decisions}
    assert scored.backtest.trades
    for t in scored.backtest.trades:
        sign = 1 if t.side == "LONG" else -1
        assert t.entry_bar - 1 in decided
        assert t.entry_price == pytest.approx(bars.open[t.entry_bar] * (1 + sign * bc.slip_rate))

    no_cal = copy.copy(result)
    no_cal.predictions_cal = None
    with pytest.raises(ScoringError, match="calibration block"):
        score_result(no_cal, role="dev")
    with pytest.raises(ValueError, match="role"):
        score_result(result, role="val")


# ------------------------------------------------------------------ (5) meta.json
def test_meta_records_the_dataset_fingerprint_and_the_setup(uninterrupted, bars_csv, synthetic_bars, tmp_path):
    sha = hashlib.sha256(Path(bars_csv).read_bytes()).hexdigest()
    for row in uninterrupted.index.rows():
        meta = json.loads((uninterrupted.root / row["run_dir"] / "meta.json").read_text(encoding="utf-8"))
        ds = meta["dataset"]
        assert ds["sha256"] == sha == row["dataset_sha256"]
        assert ds["n_bars"] == len(synthetic_bars) == row["dataset_n_bars"] and ds["n_rows"] == len(synthetic_bars)
        assert ds["first_timestamp"] == "2025-10-11T02:30:00+00:00" and ds["last_timestamp"] == "2025-10-13T04:29:00+00:00"
        assert meta["setup"] == {"bar_minutes": 1, "LOOKBACK": 60, "HORIZON_STEPS": [10, 15, 20]}
        assert (row["bar_minutes"], row["lookback"], json.loads(row["horizon_steps"])) == (1, 60, [10, 15, 20])
        eng = meta["engine"]
        assert eng["cell_key"] == row["cell_key"] and eng["config_hash"] == row["config_hash"]
        assert eng["commit"] and eng["spec_hash"] == row["spec_hash"] and meta["blocks"]["test"]["n"] == 250
    # the bar size follows RESAMPLE_MINUTES
    two = Scenario.from_dict(spec(bars_csv, variants={"m2": {"RESAMPLE_MINUTES": 2}}, folds=[-1], seeds=[0]))
    [pc] = Runner(two, tmp_path / "runs", trainer=FakeTrainer()).plan()
    assert pc.setup["bar_minutes"] == 2 and pc.dataset["n_bars"] == len(synthetic_bars) // 2 and pc.dataset["sha256"] == sha


# ------------------------------------------------------------------ the index
def test_a_rebuilt_index_equals_the_live_one_for_done_failed_and_incomplete_runs(tmp_path, bars_csv):
    sc = Scenario.from_dict(spec(bars_csv, folds=[-1], seeds=[0, 1, 2]))
    store = RunStore(tmp_path / "runs")
    with pytest.raises(KeyboardInterrupt):
        Runner(sc, store, trainer=FakeTrainer(fail={"default__f-1__s1"}, interrupt_at=3)).run()
    live = store.index.dump()
    assert sorted(r[RUN_FIELD_STATUS] for r in live["runs"]) == ["done", "failed", "incomplete"]
    assert store.rebuild_index(tmp_path / "rebuilt.sqlite").dump() == live
    store.index_path.unlink()                                                 # the index is only a cache
    assert store.rebuild_index().dump() == live


RUN_FIELD_STATUS = __import__("neural_trade.experiments.store", fromlist=["RUN_FIELDS"]).RUN_FIELDS.index("status")


# ------------------------------------------------------------------ (7) the CLI
def test_the_cli_plans_runs_resumes_and_reindexes_a_scenario(tmp_path, bars_csv, monkeypatch, capsys):
    from neural_trade.cli import main
    from neural_trade.experiments import runner as runner_module

    trainer = FakeTrainer()
    monkeypatch.setattr(runner_module, "train_cell", trainer)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(spec(bars_csv)), encoding="utf-8")
    store = tmp_path / "runs"
    args = [str(path), "--store", str(store)]

    assert main(["scenario", "plan", *args]) == 0
    plan = json_out(capsys)
    assert plan["counts"] == {"done": 0, "failed": 0, "pending": 4} and not store.exists()
    assert main(["scenario", "run", *args, "--max-cells", "1"]) == 0
    first = json_out(capsys)
    assert len(first["ran"]) == 1 and len(first["pending"]) == 3 and not first["failed"]
    assert main(["scenario", "run", *args]) == 0
    second = json_out(capsys)
    assert second["skipped"] == [first["ran"][0]["cell"]] and len(second["ran"]) == 3
    assert len(trainer.calls) == 4 and len(set(trainer.calls)) == 4
    assert main(["scenario", "plan", *args]) == 0
    assert json_out(capsys)["counts"] == {"done": 4, "failed": 0, "pending": 0}

    live = RunStore(store).index.dump()
    assert main(["scenario", "reindex", "--store", str(store)]) == 0
    assert json_out(capsys)["runs"] == 4 and RunStore(store).index.dump() == live

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(spec(bars_csv, seedz=[0])), encoding="utf-8")
    assert main(["scenario", "run", str(bad), "--store", str(tmp_path / "other")]) == 2
    assert not (tmp_path / "other").exists() and len(trainer.calls) == 4


# ------------------------------------------------------------------ (9) pick_run
def test_pick_run_keeps_returning_the_notebook_run_after_a_scenario(tmp_path, bars_csv):
    from neural_trade.notebook import pick_run, servable_runs

    runs = tmp_path / "runs"
    notebook_run = runs / "20260101T000000Z-aaaaaaa-bbbbbbbb"
    (notebook_run / "artifacts").mkdir(parents=True)
    (notebook_run / "artifacts" / "weights.h5").write_bytes(b"")
    os.utime(notebook_run, (1_000, 1_000))
    Runner(Scenario.from_dict(spec(bars_csv, folds=[-1], seeds=[0, 1])), runs, trainer=FakeTrainer()).run()
    engine = RunStore(runs).run_dirs("tiny")
    assert len(engine) == 2
    for d in engine:                                   # as if the scenario had saved serving bundles
        (d / "artifacts").mkdir()
        (d / "artifacts" / "weights.h5").write_bytes(b"")
    assert pick_run(None, runs) == notebook_run
    assert servable_runs(runs) == [notebook_run]
    assert set(servable_runs(runs, include_engine_runs=True)) == {notebook_run, *engine}
    assert pick_run(engine[0], runs) == engine[0]      # an engine run can still be loaded by name
    # an engine run outside the scenarios subtree is recognised by its meta.json
    shutil.copytree(engine[0], runs / "copied" / engine[0].name)
    assert pick_run(None, runs) == notebook_run


# ------------------------------------------------------------------ (6) the frozen set
FROZEN = ("scripts/gate_run.py", "scripts/check_gates.py", "scripts/backtest_gate.py",
          "scripts/direction_experiments.py", "scripts/ablate.py", "src/neural_trade/experiments/ablation.py")
ENGINE = ("src/neural_trade/experiments/scenario.py", "src/neural_trade/experiments/runner.py",
          "src/neural_trade/experiments/store.py", "src/neural_trade/experiments/scorer.py",
          "src/neural_trade/experiments/dataset.py", "src/neural_trade/cli.py", "src/neural_trade/notebook/runs.py")
FROZEN_MODULES = ("ablation", "gate_run", "check_gates", "backtest_gate", "direction_experiments", "ablate")


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""
            yield from (f"{node.module}.{a.name}" for a in node.names)


def test_the_frozen_set_names_the_engine_and_no_engine_module_imports_it():
    for rel in FROZEN:
        doc = " ".join((ast.get_docstring(ast.parse((REPO / rel).read_text(encoding="utf-8"))) or "").split())
        assert "Frozen (D-023)" in doc and "`neural-trade scenario run`" in doc, rel
    for rel in ENGINE:
        names = {part for name in _imports(REPO / rel) for part in name.split(".")}
        assert not names & set(FROZEN_MODULES), f"{rel} imports the frozen set: {names & set(FROZEN_MODULES)}"
    import neural_trade.experiments.ablation as ablation       # still importable (notebook 05's analysis)

    assert callable(ablation.analyze)


# ------------------------------------------------------------------ slow: real training through the CLI
@pytest.mark.slow
def test_a_real_scenario_trains_scores_and_resumes_through_the_cli(tmp_path, bars_csv, monkeypatch, capsys):
    from neural_trade.cli import main
    from neural_trade.notebook import pick_run

    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    notebook_run = runs / "20260101T000000Z-aaaaaaa-bbbbbbbb"
    (notebook_run / "artifacts").mkdir(parents=True)
    (notebook_run / "artifacts" / "weights.h5").write_bytes(b"")
    os.utime(notebook_run, (1_000, 1_000))
    s = spec(bars_csv, name="tiny_real", folds=[-2, -1], seeds=[0], run={"calibrate": False, "save_artifacts": True})
    path = tmp_path / "tiny_real.yaml"
    path.write_text(yaml.safe_dump(s), encoding="utf-8")

    assert main(["scenario", "run", str(path), "--store", str(runs), "--max-cells", "1"]) == 0
    first = json_out(capsys)
    assert main(["scenario", "run", str(path), "--store", str(runs)]) == 0
    second = json_out(capsys)
    assert len(first["ran"]) == 1 and second["skipped"] == [first["ran"][0]["cell"]] and len(second["ran"]) == 1
    store = RunStore(runs)
    rows = store.index.rows()
    assert [r["status"] for r in rows] == ["done", "done"] and {r["role"] for r in rows} == {"dev", "test"}
    sha = hashlib.sha256(Path(bars_csv).read_bytes()).hexdigest()
    for row in rows:
        d = runs / row["run_dir"]
        for name in (f"eval_report_{row['role']}.json", f"eval_report_{row['role']}.md", "metrics.jsonl",
                     "status.json", "result.json", "artifacts/weights.h5", "weights.h5"):
            assert (d / name).exists(), name
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        assert meta["dataset"]["sha256"] == sha and meta["setup"]["HORIZON_STEPS"] == [10, 15, 20]
        assert row["sharpe_net"] is not None and np.isfinite(row["sharpe_net"]) and row["sec_per_step"] > 0
        report = json.loads((d / f"eval_report_{row['role']}.json").read_text(encoding="utf-8"))
        assert report["split"] == row["role"] and np.isfinite(report["model"]["horizons"]["h1"]["variance"]["crps"])
    assert pick_run(None, runs) == notebook_run         # the engine's bundles are not "the latest run"

    # an uninterrupted run of the same scenario reproduces every indexed number
    assert main(["scenario", "run", str(path), "--store", str(tmp_path / "runs_b")]) == 0
    assert comparable(RunStore(tmp_path / "runs_b")) == comparable(store)

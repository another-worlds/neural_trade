"""Sweeps (NT-030): the search space, quick sizing, the resumable Optuna study, the GPU budget, parallel
batches with the GPU-free check and the watch level, cell claims, failed trials, the top-5 x 3-seed
re-run, and the CLI. A fake trainer stands in for training (as in test_experiment_engine.py), so the
engine, scorer and index run their real code on CPU in seconds; no test trains a network."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from neural_trade.cli import main
from neural_trade.core.config import Config
from neural_trade.experiments.claims import CellClaims, pid_alive
from neural_trade.experiments.runner import Runner
from neural_trade.experiments.scenario import Scenario
from neural_trade.experiments.store import RunStore
from neural_trade.experiments.sweep import (
    GpuStatus, SearchSpace, Sweep, SweepError, SweepOptions, cell_seconds, code_source_dir, dev_net_sharpe,
    exceeds_watch_level, gpu_status_from_dmon, held_out_columns, latest_sec_per_step, load_parallel_record,
    nvidia_smi_gpu_check, parse_dmon, record_setup_warnings, run_health, size_quick)

HORIZONS = ("h0", "h1", "h2")
SEARCH = {"LR": {"low": 1e-4, "high": 1e-2, "log": True}, "LAMBDA_DIR": {"low": 0.1, "high": 2.0}}


# ------------------------------------------------------------------ helpers
def fake_result(cfg: Config, noise: float = 3.0):
    from sklearn.preprocessing import StandardScaler

    from neural_trade.data.processor import split_arrays

    arrays = split_arrays(cfg)
    rng = np.random.default_rng([int(cfg.SEED), int(cfg.FOLD_INDEX) % 97, int(cfg.LR * 1e6) % 9973])
    scaler = StandardScaler().fit(arrays["train"]["y"].reshape(-1, 1))
    scale = float(scaler.scale_[0])

    def heads(block):
        y = np.asarray(block["y"], float)
        signal = y / scale + rng.normal(0.0, noise, y.shape)
        return {"delta": {h: 0.1 * scale * signal[:, i] for i, h in enumerate(HORIZONS)},
                "direction_prob": {h: 1.0 / (1.0 + np.exp(-0.5 * signal[:, i])) for i, h in enumerate(HORIZONS)},
                "variance": {h: (1.0 + 0.1 * i) * (1.0 + 0.2 * rng.random(len(y))) for i, h in enumerate(HORIZONS)}}

    return SimpleNamespace(
        config=cfg, target_scaler=scaler, predictions=heads(arrays["test"]), y_test=arrays["test"]["y"],
        last_close_test=arrays["test"]["last_close"], predictions_calibrated=None, windows_test=arrays["test"]["X"],
        predictions_cal=heads(arrays["cal"]), y_cal=arrays["cal"]["y"], last_close_cal=arrays["cal"]["last_close"],
        windows_cal=arrays["cal"]["X"], calibration_pipeline=None, history=None, weights_epoch=1,
        weights_val_loss=0.5)


def write_real_telemetry(run_dir, *, epochs=((1.0, 0.9),), nonfinite=0, served=0.9):
    """metrics.jsonl and status.json written by the trainer's OWN writers, as a real cell writes them: the
    JsonlEpochLogger callback (``loss`` / ``val_loss`` / ``nonfinite_grad_steps`` from the Keras logs; NaN and inf
    stored as null) and the served-epoch record (``_finite_or_none``: a non-finite served loss stored as null)."""
    from neural_trade.telemetry.epoch_logger import JsonlEpochLogger
    from neural_trade.training.trainer import _finite_or_none, _record_served_epoch_in_status

    run_dir = Path(run_dir)
    logger_cb = JsonlEpochLogger(run_dir, None, run_dir.name)
    logger_cb.on_train_begin()
    for e, (loss, val) in enumerate(epochs):
        logger_cb.on_epoch_begin(e)
        logger_cb.on_train_batch_end(0)
        logger_cb.on_epoch_end(e, {"loss": np.float32(loss), "val_loss": np.float32(val),
                                   "nonfinite_grad_steps": np.float32(nonfinite)})
    logger_cb.on_train_end()
    _record_served_epoch_in_status(SimpleNamespace(run_dir=run_dir), len(epochs), _finite_or_none(served), "test")


class FakeTrainer:
    """Fake predictions; the cell's telemetry is written by the real writers (a finite, healthy run)."""

    def __init__(self, fail_variants=(), on_call=None):
        self.calls = []
        self.fail_variants = set(fail_variants)
        self.on_call = on_call

    def __call__(self, ctx, *, calibrate, save_artifacts):
        eng = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))["engine"]
        self.calls.append(eng["cell_key"])
        if self.on_call:
            self.on_call(eng)
        if eng["variant"] in self.fail_variants:
            raise ValueError(f"injected failure in {eng['cell_key']}")
        self.telemetry(ctx.run_dir, eng)
        return fake_result(ctx.config)

    def telemetry(self, run_dir, eng):
        write_real_telemetry(run_dir)


def scenario_dict(csv, **changes):
    s = {"schema_version": 1, "name": "sw", "description": "sweep test",
         "overrides": {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 2, "BATCH_SIZE": 32},
         "folds": [-2, -1], "seeds": [0], "strategy": {"name": "calibrated_quantile", "params": {}},
         "backtest": {"random_seeds": 5}, "run": {"calibrate": False, "save_artifacts": False}, "search": SEARCH,
         # the fake trainer's predictions are not built to beat buy-and-hold; the guard-rails under test set their own
         "leaderboard": {"beat_buy_and_hold": False, "beat_random_null": False}}
    s.update(copy.deepcopy(changes))
    return s


def free_gpu():
    return GpuStatus(True, {"note": "stub"})


class Monitor:
    """A stubbed utilisation monitor: returns the same reading for every batch."""

    def __init__(self, reading=None):
        self.reading = reading or {"mean_sm_pct": 10.0, "mean_fb_mb": 1000.0, "peak_fb_mb": 1200.0}
        self.batches = 0

    def __call__(self):
        return self

    def start(self):
        self.batches += 1

    def stop(self):
        return dict(self.reading)


def make_sweep(tmp_path, csv, trainer, *, mode="optuna", changes=None, announce=None, **opts):
    opts.setdefault("sec_per_step", 0.01)
    opts.setdefault("overhead_s", 1.0)
    options = SweepOptions(mode=mode, parallel_record=str(tmp_path / "no_such_record.json"), **opts)
    return Sweep(Scenario.from_dict(scenario_dict(csv, **(changes or {}))), RunStore(tmp_path / "runs"), options,
                 trainer=trainer, gpu_check=free_gpu, monitor_factory=Monitor(), sleep=lambda s: None,
                 announce=announce or (lambda t: None))


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory, synthetic_bars):
    path = tmp_path_factory.mktemp("sweep_data") / "bars.csv"
    synthetic_bars.to_csv(path, index=False)
    return path


def summary(sweep):
    return json.loads(sweep.summary_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the search space
def _sc(**search):
    return Scenario.from_dict(scenario_dict("x.csv", search=search))


def test_the_space_comes_from_the_config_metadata_and_only_tunable_fields(tmp_path):
    space = SearchSpace.from_scenario(_sc(BATCH_SIZE={"low": 32, "high": 256}, RHO_MAX=None))
    bs, rho = space.params
    assert (rho.kind, rho.low, rho.high, rho.log) == ("float", 0.0, 1.0, False)       # the metadata's own range
    assert (bs.kind, bs.low, bs.high, bs.step) == ("int", 32.0, 256.0, 1)
    for bad, why in [({"CSV_PATH": None}, "not tunable"), ({"NOPE": None}, "unknown Config field"),
                     ({"LAMBDA_DIR": None}, "no finite, inclusive range"), ({"LR": None}, "no finite, inclusive range"), ({"LR": {"low": 1e-3, "high": 1e-4}}, "below high"),
                     ({"LR": {"low": 1e-3, "high": 5.0}}, "LR"), ({"SEED": None}, "set by the engine")]:
        with pytest.raises(SweepError, match=why):
            SearchSpace.from_scenario(_sc(**bad))


def test_resample_minutes_is_refused_in_the_space_and_in_the_scenario(tmp_path, bars_csv):
    with pytest.raises(SweepError, match="1-minute bars until NT-040"):
        SearchSpace.from_scenario(_sc(RESAMPLE_MINUTES={"low": 1, "high": 5}))
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), changes={"overrides": {"CSV_PATH": str(bars_csv),
                                                                             "RESAMPLE_MINUTES": 5}})
    with pytest.raises(SweepError, match="RESAMPLE_MINUTES"):
        sw.run()


def test_patience_is_searchable_and_capped_at_early_while_early_is_not_tunable():
    specs = Config.field_specs()
    assert specs["PATIENCE"].tunable and not specs["EARLY"].tunable
    space = SearchSpace.from_scenario(_sc(PATIENCE={"low": 0, "high": 50}))
    assert space.params[0].high == float(Config().EARLY)
    with pytest.raises(SweepError, match="not tunable"):
        SearchSpace.from_scenario(_sc(EARLY={"low": 1, "high": 5}))


def test_sampling_stays_inside_the_bounds_and_snaps_integers():
    space = SearchSpace.from_scenario(_sc(LR={"low": 1e-4, "high": 1e-2, "log": True}, T_PERP_DIM={"low": 4, "high": 20}))
    rng = np.random.default_rng(0)
    pts = [space.sample(rng) for _ in range(200)]
    assert all(1e-4 <= p["LR"] <= 1e-2 and 4 <= p["T_PERP_DIM"] <= 20 and isinstance(p["T_PERP_DIM"], int) for p in pts)
    assert len({p["T_PERP_DIM"] for p in pts}) > 5


# ------------------------------------------------------------------ ranking is dev-only
def _row(fold, role, seed, sharpe, status="done", trades=10, **kw):
    return {"fold": fold, "role": role, "seed": seed, "sharpe_net": sharpe, "status": status, "n_trades": trades,
            "error": kw.get("error")}


def test_the_ranking_number_reads_dev_folds_only_and_averages_seeds_within_a_fold():
    rows = [_row(-3, "dev", 0, 1.0), _row(-3, "dev", 1, 3.0), _row(-2, "dev", 0, 0.0), _row(-1, "test", 0, 99.0)]
    s = dev_net_sharpe(rows, [-3, -2])
    assert s.per_fold == {-3: 2.0, -2: 0.0} and s.value == pytest.approx(1.0) and s.reason is None
    assert held_out_columns(rows)["test_sharpe_net"] == 99.0
    # the test row changes nothing about the dev value
    assert dev_net_sharpe(rows[:-1], [-3, -2]).value == s.value


def test_a_failed_or_non_finite_fold_makes_the_trial_failed_with_a_reason():
    s = dev_net_sharpe([_row(-3, "dev", 0, 1.0), _row(-2, "dev", 0, None, status="failed", error="ValueError: x")], [-3, -2])
    assert s.value is None and "fold -2" in s.reason and "ValueError" in s.reason
    s = dev_net_sharpe([_row(-2, "dev", 0, float("nan"))], [-2])
    assert s.value is None and "non-finite" in s.reason


# ------------------------------------------------------------------ (1) quick mode
def test_quick_sizing_stays_within_the_budget_and_prefers_enough_trials():
    plan = size_quick(dev_steps={-3: 100, -2: 100}, epochs_cap=20, sec_per_step=0.1, overhead_s=10, budget_s=300)
    assert plan.estimated_s <= 300 and plan.n_trials >= 4 and plan.epochs <= 3
    # a slow step forces fewer epochs, then one fold
    slow = size_quick(dev_steps={-3: 400, -2: 400}, epochs_cap=20, sec_per_step=0.1, overhead_s=10, budget_s=300)
    assert slow.estimated_s <= 300 and (slow.epochs < 3 or len(slow.folds) == 1)
    assert plan.n_trials * cell_seconds(100, plan.epochs, 0.1, 10) * len(plan.folds) == pytest.approx(plan.estimated_s)
    with pytest.raises(SweepError, match="even one trial"):
        size_quick(dev_steps={-2: 100000}, epochs_cap=1, sec_per_step=1.0, overhead_s=10, budget_s=300)


def test_a_quick_sweep_sizes_itself_prints_the_estimate_and_labels_results_quick(tmp_path, bars_csv):
    said = []
    trainer = FakeTrainer(on_call=lambda eng: said.append(len(said)))
    sw = make_sweep(tmp_path, bars_csv, trainer, mode="quick", overhead_s=70.0, sec_per_step=0.001, announce=said.append)
    res = sw.run()
    assert isinstance(said[0], str) and "estimated" in said[0] and "quick" in said[0]      # printed before any trial
    est = res.budget["estimated_minutes"]
    assert est <= 5.0 and res.budget["n_trials"] == 4 and res.budget["epochs"] <= 2
    assert res.label == "quick" and res.state == "quick_complete" and res.winner is None
    assert len(trainer.calls) == 4 and all(t["label"] == "quick" and t["state"] == "COMPLETE" for t in res.trials)
    doc = summary(sw)
    assert doc["label"] == "quick" and doc["leader"] is not None and "not a winner" in doc["note"]
    # the engine's cells ran at the quick epochs, on dev folds only: no test cell was trained
    rows = sw.store.index.rows(sw.sweep_id)
    assert {r["role"] for r in rows} == {"dev"}


def test_quick_dry_run_trains_nothing(tmp_path, bars_csv):
    trainer = FakeTrainer()
    res = make_sweep(tmp_path, bars_csv, trainer, mode="quick", dry_run=True).run()
    assert res.state == "dry_run" and trainer.calls == []


def test_quick_without_a_measured_sec_per_step_refuses(tmp_path, bars_csv):
    with pytest.raises(SweepError, match="no measured sec_per_step"):
        make_sweep(tmp_path, bars_csv, FakeTrainer(), mode="quick", sec_per_step=None).run()


def test_sec_per_step_comes_from_the_latest_run_of_the_same_setup_in_the_index(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    trainer = FakeTrainer()
    first = make_sweep(tmp_path, bars_csv, trainer, mode="quick", overhead_s=70.0, sec_per_step=0.001)
    first.run()
    # the fake runs' telemetry times a 1-step epoch, so inject the value the way a real run's result.json carries it
    for d in first.store.run_dirs(first.sweep_id):
        doc = json.loads((d / "result.json").read_text(encoding="utf-8"))
        doc["sec_per_step"] = 0.123
        (d / "result.json").write_text(json.dumps(doc), encoding="utf-8")
    first.store.sync(first.sweep_id)
    again = Sweep(Scenario.from_dict(scenario_dict(bars_csv)), first.store, SweepOptions(mode="optuna", dry_run=True,
                  parallel_record=None), gpu_check=free_gpu, announce=lambda t: None)
    res = again.run()
    assert res.budget["sec_per_step"] == pytest.approx(0.123) and "latest run" in res.budget["sec_per_step_source"]["source"]


# ------------------------------------------------------------------ (2) resumable optuna study
def test_an_optuna_study_stops_after_two_trials_and_resumes_to_four_without_repeating(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    trainer = FakeTrainer()
    a = make_sweep(tmp_path, bars_csv, trainer, n_trials=4, stop_after=2, top_k=2)
    r1 = a.run()
    assert r1.state == "stopped" and len(r1.trials) == 2 and len(trainer.calls) == 2 and r1.winner is None
    assert (a.directory / "study.db").is_file()
    first_calls = list(trainer.calls)
    with pytest.raises(SweepError, match="resume"):
        make_sweep(tmp_path, bars_csv, trainer, n_trials=4).run()          # an existing sweep is never overwritten
    b = make_sweep(tmp_path, bars_csv, trainer, n_trials=4, resume=True, top_k=2)
    r2 = b.run()
    import optuna

    study = optuna.load_study(study_name=b.sweep_id, storage=f"sqlite:///{(b.directory / 'study.db').as_posix()}")
    assert [t.state.name for t in study.trials] == ["COMPLETE"] * 4
    new_dev = [c for c in trainer.calls if c.endswith("__s0") and c not in first_calls]
    assert len(set(trainer.calls[:2])) == 2 and not set(first_calls) & set(trainer.calls[2:4])   # no finished trial again
    assert r2.state == "complete" and {t["number"] for t in r2.trials} == {0, 1, 2, 3} and len(new_dev) >= 2
    # a third call with nothing left trains nothing
    n = len(trainer.calls)
    make_sweep(tmp_path, bars_csv, trainer, n_trials=4, resume=True, top_k=2).run()
    assert len(trainer.calls) == n


def test_an_interrupted_trial_is_finished_not_dropped_on_resume(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    import optuna

    class Boom(FakeTrainer):
        def __call__(self, ctx, **kw):
            if len(self.calls) == 1:
                self.calls.append("boom")
                raise KeyboardInterrupt
            return super().__call__(ctx, **kw)

    trainer = Boom()
    a = make_sweep(tmp_path, bars_csv, trainer, n_trials=2, top_k=1)
    with pytest.raises(KeyboardInterrupt):
        a.run()
    study = optuna.load_study(study_name=a.sweep_id, storage=f"sqlite:///{(a.directory / 'study.db').as_posix()}")
    assert [t.state.name for t in study.trials] == ["COMPLETE", "RUNNING"]
    params = dict(study.trials[1].params)
    res = make_sweep(tmp_path, bars_csv, trainer, n_trials=2, resume=True, top_k=1).run()
    study = optuna.load_study(study_name=a.sweep_id, storage=f"sqlite:///{(a.directory / 'study.db').as_posix()}")
    assert [t.state.name for t in study.trials][:2] == ["COMPLETE", "COMPLETE"] and dict(study.trials[1].params) == params
    assert len(study.trials) == 2 and res.state == "complete"


# ------------------------------------------------------------------ (3) the budget
def test_the_gpu_budget_is_printed_and_recorded_before_the_first_trial(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    seen = {}
    said = []

    def at_first_call(eng):
        if "doc" not in seen:
            seen["doc"] = json.loads(sweep.summary_path.read_text(encoding="utf-8"))
            seen["said"] = list(said)

    trainer = FakeTrainer(on_call=at_first_call)
    sweep = make_sweep(tmp_path, bars_csv, trainer, n_trials=3, top_k=2, announce=said.append)
    res = sweep.run()
    budget = seen["doc"]["launches"][0]["budget"]
    assert "GPU budget" in seen["said"][0] and "expected" in seen["said"][0] and "fit" in seen["said"][0]
    assert budget["gpu_hours"] > 0 and budget["max_hours"] == 12.0 and budget["expected_gpu_hours"] <= budget["gpu_hours"]
    assert budget["gpu_hours"] == pytest.approx(budget["search_gpu_hours"] + budget["rerun_gpu_hours"])
    assert budget["trials_to_run"] == 3 and budget["rerun"]["top_k"] == 2 and budget["rerun"]["seeds"] == 3
    # trials x dev folds x steps x epochs x sec_per_step (+ overhead) by hand, at the space's lowest batch size
    steps = sum(budget["steps_per_epoch_upper_per_fold"][str(f)] for f in budget["dev_folds"])
    assert budget["search_gpu_hours"] * 3600 == pytest.approx(3 * (steps * budget["epochs_upper_bound"] * 0.01 + 1.0))
    # the re-run: the first seed's dev cells already exist, so only the later seeds on the dev fold and all seeds on test
    e, ov = budget["epochs_upper_bound"], 1.0
    cell = lambda f: budget["steps_per_epoch_upper_per_fold"][str(f)] * e * 0.01 + ov  # noqa: E731
    want = 2 * (sum(cell(f) * 2 for f in budget["dev_folds"]) + sum(cell(f) * 3 for f in budget["test_folds"]))
    assert budget["rerun_gpu_hours"] * 3600 == pytest.approx(want)
    assert res.budget["gpu_hours"] == budget["gpu_hours"]


def test_a_budget_over_max_hours_is_refused_before_anything_starts(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    trainer = FakeTrainer()
    sw = make_sweep(tmp_path, bars_csv, trainer, n_trials=3, max_hours=1e-6)
    with pytest.raises(SweepError, match="over --max-hours") as exc:
        sw.run()
    assert trainer.calls == [] and not sw.summary_path.exists()
    assert SweepOptions().max_hours == 12.0                                   # the default: one night
    assert not (sw.directory.parent).exists()                                 # a refused launch leaves nothing under sweeps/
    assert "trial(s) fit" in str(exc.value)


# ------------------------------------------------------------------ (4) parallel batches
def _record(tmp_path, allowed=3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "parallel_n.json"
    p.write_text(json.dumps({"allowed_n": allowed, "utilization": {
        "1": {"mean_sm_pct": 23.7, "peak_sm_pct": 100.0, "mean_fb_mb": 2575, "peak_fb_mb": 3731},
        "2": {"mean_sm_pct": 27.3, "peak_sm_pct": 100.0, "mean_fb_mb": 3743, "peak_fb_mb": 5567},
        "3": {"mean_sm_pct": 32.2, "peak_sm_pct": 99.0, "mean_fb_mb": 4795, "peak_fb_mb": 7821}}}), encoding="utf-8")
    return p


def test_parallel_above_one_needs_the_recorded_n(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    for record, match in [(None, "allows N = 1"), (_record(tmp_path, allowed=2), "allows N = 2")]:
        sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=3, parallel=3)
        sw.options.parallel_record = str(record) if record else str(tmp_path / "missing.json")
        with pytest.raises(SweepError, match=match):
            sw.run()
    assert load_parallel_record(tmp_path / "missing.json")["allowed_n"] == 1
    assert load_parallel_record(_record(tmp_path))["allowed_n"] == 3


class Launcher:
    """Stands in for the batch's N processes: runs each trial spec through the engine, like
    ``scenario run --claim-cells``, and marks the batch as running for the GPU-check assertion."""

    def __init__(self, trainer, state):
        self.trainer, self.state, self.batches = trainer, state, []

    def __call__(self, paths, store):
        self.batches.append([p.name for p in paths])
        self.state["running"] = True
        try:
            for p in paths:
                Runner(Scenario.from_yaml(p), store, trainer=self.trainer, claim_cells=True).run()
        finally:
            self.state["running"] = False
        return [0 for _ in paths]


def _parallel_sweep(tmp_path, csv, *, gpu_free=True, reading=None, n_trials=4, parallel=2):
    state = {"running": False, "checks_while_running": 0, "checks": 0}

    def check():
        state["checks"] += 1
        state["checks_while_running"] += int(state["running"])
        return GpuStatus(gpu_free, {"stub": True})

    trainer = FakeTrainer()
    sw = make_sweep(tmp_path, csv, trainer, n_trials=n_trials, parallel=parallel, top_k=1)
    sw.options.parallel_record = str(_record(tmp_path))
    sw.gpu_check = check
    sw.monitor_factory = Monitor(reading)
    sw.launcher = Launcher(trainer, state)
    return sw, trainer, state


def test_parallel_trials_launch_in_batches_with_the_gpu_check_between_them(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    sw, trainer, state = _parallel_sweep(tmp_path, bars_csv)
    res = sw.run()
    assert res.state == "complete" and len(res.trials) == 4
    trial_batches = [b for b in sw.launcher.batches if len(b) == 2]
    assert len(trial_batches) == 2 and all(len(b) <= 2 for b in sw.launcher.batches)
    assert state["checks"] >= 3 and state["checks_while_running"] == 0         # never checked while own trials ran
    assert len(trainer.calls) == len(set(trainer.calls))                      # no cell trained twice
    assert sw.monitor_factory.batches >= 2


def test_a_busy_gpu_stops_the_sweep_before_the_batch(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    sw, trainer, state = _parallel_sweep(tmp_path, bars_csv, gpu_free=False)
    res = sw.run()
    assert res.state == "stopped" and "GPU is busy" in res.stop_reason and trainer.calls == []
    # wait mode polls, then gives up after wait_max_s
    waits = []
    sw2, trainer2, _ = _parallel_sweep(tmp_path / "w", bars_csv, gpu_free=False)
    sw2.options.when_busy, sw2.options.wait_poll_s, sw2.options.wait_max_s = "wait", 5.0, 15.0
    sw2.sleep = waits.append
    assert sw2.run().state == "stopped" and waits == [5.0, 5.0, 5.0] and trainer2.calls == []


def test_a_gpu_reading_above_the_recorded_level_stops_launching_new_batches(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    high = {"mean_sm_pct": 95.0, "mean_fb_mb": 9000.0, "peak_fb_mb": 11000.0}
    sw, trainer, _ = _parallel_sweep(tmp_path, bars_csv, reading=high, n_trials=4, parallel=2)
    res = sw.run()
    assert res.state == "stopped" and "stopped launching" in res.stop_reason
    assert len([b for b in sw.launcher.batches if len(b) == 2]) == 1 and len(res.trials) == 2
    level = load_parallel_record(_record(tmp_path))["utilization"]["2"]
    assert exceeds_watch_level(high, level) and not exceeds_watch_level(
        {"mean_sm_pct": 30.0, "peak_fb_mb": 5600.0}, level) and exceeds_watch_level(
        {"mean_sm_pct": 90.0, "peak_fb_mb": 5000.0}, level) and exceeds_watch_level({"peak_fb_mb": None}, level) is None


def test_two_runners_never_train_the_same_cell(tmp_path, bars_csv):
    store = RunStore(tmp_path / "runs")
    sc = Scenario.from_dict(scenario_dict(bars_csv, search={}, folds=[-2, -1], seeds=[0, 1]))
    t_a, t_b = FakeTrainer(), FakeTrainer()
    other = Runner(sc, store, trainer=t_b, claim_cells=True)
    claims = other._claims()
    assert claims.claim("default__f-2__s0")                       # "another process" holds one cell
    rep_a = Runner(sc, store, trainer=t_a, claim_cells=True).run()
    assert "default__f-2__s0" not in t_a.calls and len(t_a.calls) == 3 and "default__f-2__s0" in rep_a.skipped
    claims.release("default__f-2__s0")
    Runner(sc, store, trainer=t_b, claim_cells=True).run()
    assert t_b.calls == ["default__f-2__s0"]                      # only the cell that was claimed
    assert not list((store.scenario_dir("sw") / "claims").glob("*.lock"))      # every claim released
    # two threads, one scenario: every cell trained exactly once
    import threading

    store2 = RunStore(tmp_path / "runs2")
    ts = [FakeTrainer(), FakeTrainer()]
    threads = [threading.Thread(target=lambda t=t: Runner(sc, store2, trainer=t, claim_cells=True).run()) for t in ts]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(ts[0].calls + ts[1].calls) == sorted({c.key for c in sc.cells()})


def test_a_claim_of_a_dead_process_is_taken_over_and_a_live_one_respected(tmp_path):
    claims = CellClaims(tmp_path / "claims")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert not pid_alive(dead.pid) and pid_alive(__import__("os").getpid())
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "c.lock").write_text(json.dumps({"pid": dead.pid}), encoding="utf-8")
    assert claims.claim("c") and claims.holder("c") is not None
    assert not CellClaims(tmp_path / "claims").claim("c")         # our own live pid holds it now
    claims.release("c")
    assert claims.claim("c")


# ------------------------------------------------------------------ (5) failed trials, (6) the re-run
def test_a_failed_trial_is_recorded_as_failed_and_never_dropped(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    import optuna

    trainer = FakeTrainer(fail_variants={"t0001"})
    sw = make_sweep(tmp_path, bars_csv, trainer, n_trials=3, top_k=2)
    res = sw.run()
    by = {t["number"]: t for t in res.trials}
    assert by[1]["state"] == "FAIL" and by[1]["value"] is None and "injected failure" in by[1]["reason"]
    assert by[0]["state"] == by[2]["state"] == "COMPLETE" and len(res.trials) == 3
    study = optuna.load_study(study_name=sw.sweep_id, storage=f"sqlite:///{(sw.directory / 'study.db').as_posix()}")
    assert [t.state.name for t in study.trials] == ["COMPLETE", "FAIL", "COMPLETE"]
    assert [r["number"] for r in res.ranking if r["rank"] is None] == [1]       # listed after the ranked ones
    failed = [r for r in sw.store.index.rows(sw.sweep_id) if r["status"] == "failed"]
    assert len(failed) == 1                                                       # the failed cell is in the index
    assert res.winner is not None and res.winner["number"] != 1


def test_the_top_trials_are_rerun_with_three_seeds_and_ranked_by_the_dev_seed_mean(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    trainer = FakeTrainer()
    sw = make_sweep(tmp_path, bars_csv, trainer, n_trials=3, top_k=2, rerun_seeds=3)
    res = sw.run()
    doc = summary(sw)
    table = doc["rerun"]
    assert len(table) == 2 and all(r["n_seeds"] == 3 and r["n_dev_cells"] == 3 for r in table)
    assert all(r["test_sharpe_net"] is not None and "never used to rank" in r["test_note"] for r in table)
    vals = [r["dev_seed_mean_net_sharpe"] for r in table]
    assert vals == sorted(vals, reverse=True) and res.winner["number"] == table[0]["number"]
    rows = sw.store.index.rows(sw.sweep_id)
    for r in table:
        mine = [x for x in rows if x["variant"] == r["variant"] and x["role"] == "dev"]
        assert r["dev_seed_mean_net_sharpe"] == pytest.approx(np.mean([x["sharpe_net"] for x in mine]))
    assert {x["role"] for x in rows if x["variant"] not in {t["variant"] for t in table}} == {"dev"}   # test: top-k only
    assert json.loads((sw.directory / "winner.json").read_text(encoding="utf-8"))["number"] == res.winner["number"]
    assert doc["ranking_column"].endswith("(test never ranks)")


def test_a_winner_must_trade(tmp_path, bars_csv, monkeypatch):
    pytest.importorskip("optuna")
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=2, top_k=2,
                    changes={"leaderboard": {"min_trades": 10 ** 9, "beat_buy_and_hold": False, "beat_random_null": False}})
    res = sw.run()
    assert res.state == "complete" and res.winner is None


# ------------------------------------------------------------------ (8) the CLI, (9) bar size
def test_the_cli_documents_both_modes_and_runs_a_dry_sweep(tmp_path, bars_csv, capsys):
    with pytest.raises(SystemExit):
        main(["sweep", "--help"])
    help_text = capsys.readouterr().out
    assert "quick" in help_text and "optuna" in help_text and "--max-hours" in help_text and "--parallel" in help_text
    spec = tmp_path / "sw.yaml"
    spec.write_text(yaml.safe_dump(scenario_dict(bars_csv)), encoding="utf-8")
    args = ["sweep", str(spec), "--store", str(tmp_path / "runs"), "--sec-per-step", "0.01", "--dry-run",
            "--parallel-record", str(tmp_path / "none.json")]
    assert main([*args, "--mode", "quick"]) == 0
    out = capsys.readouterr().out
    assert "quick sweep" in out and "estimated" in out and '"state": "dry_run"' in out
    pytest.importorskip("optuna")
    assert main([*args, "--mode", "optuna", "--n-trials", "3"]) == 0
    assert "GPU budget" in capsys.readouterr().out
    assert main([*args, "--mode", "optuna", "--max-hours", "1e-9"]) == 2


def test_the_cli_refuses_another_bar_size(tmp_path, bars_csv, capsys):
    spec = tmp_path / "sw.yaml"
    d = scenario_dict(bars_csv)
    d["overrides"]["RESAMPLE_MINUTES"] = 5
    spec.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert main(["sweep", str(spec), "--mode", "quick", "--store", str(tmp_path / "runs"), "--sec-per-step", "0.1"]) == 2
    assert "RESAMPLE_MINUTES" in capsys.readouterr().err and not (tmp_path / "runs" / "sweeps").exists()


def test_a_scenario_with_variants_or_axes_is_refused_by_a_sweep(tmp_path, bars_csv):
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), mode="quick", changes={"variants": {"a": {"LR": 0.01}}})
    with pytest.raises(SweepError, match="variants"):
        sw.run()


def test_the_search_block_does_not_change_an_existing_scenarios_hash(tmp_path, bars_csv):
    d = scenario_dict(bars_csv, search={})
    plain = Scenario.from_dict(d)
    assert "search" not in plain.to_dict()
    assert math.isfinite(len(Scenario.from_dict(scenario_dict(bars_csv)).to_dict()["search"]))


# ------------------------------------------------------------------ repair round 1
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_the_dmon_parser_reads_the_real_captured_header_by_name_and_the_memory_rule_fires():
    text = (FIXTURES / "nvidia_smi_dmon_um.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0].split()[1:] == ["gpu", "sm", "mem", "enc", "dec", "jpg", "ofa", "fb", "bar1", "ccpm"]
    rows = parse_dmon(text)
    assert len(rows) == 4 and all(r["fb"] > 9000 and r["enc"] == 0 and r["mem"] < 5 for r in rows)   # fb, not enc
    status = gpu_status_from_dmon(text)
    assert not status.free and status.detail["median_fb_mb"] > 2000 and status.detail["median_sm_pct"] <= 30
    # the same sample with a quiet desktop (fb ~ 1000 MB, sm low) is free
    quiet = text.replace("10001", " 1001").replace("10000", " 1000")
    assert gpu_status_from_dmon(quiet).free
    # through the public check, with the capture as nvidia-smi's output
    import os

    old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        assert not nvidia_smi_gpu_check(run_dmon=lambda n: text).free
        assert nvidia_smi_gpu_check(run_dmon=lambda n: quiet).free
    finally:
        if old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
    assert not gpu_status_from_dmon("# gpu sm\n").free                      # no samples: not free


def test_the_trial_processes_launch_with_the_callers_code_on_pythonpath(tmp_path, bars_csv, monkeypatch):
    import os

    import neural_trade
    from neural_trade.experiments import sweep as sweep_mod

    seen = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen.append((cmd, kw.get("env")))

        def wait(self):
            return 0

    monkeypatch.setattr(sweep_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=2, parallel=2)
    spec = tmp_path / "t.yaml"
    spec.write_text("x: 1", encoding="utf-8")
    assert sw._subprocess_launcher([spec], sw.store) == [0]
    cmd, env = seen[0]
    src = str(Path(neural_trade.__file__).resolve().parent.parent)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == src == str(code_source_dir())
    assert env["PYTHONPATH"].endswith("/somewhere/else") and "--claim-cells" in cmd
    # the code path and sha are in sweep.json
    pytest.importorskip("optuna")
    sw2 = make_sweep(tmp_path / "b", bars_csv, FakeTrainer(), n_trials=1, top_k=1)
    sw2.run()
    code = summary(sw2)["launches"][0]["code"]
    assert code["source_dir"] == src and code["git_sha"]


def test_run_health_names_non_finite_losses_and_non_finite_gradient_steps(tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()
    write_real_telemetry(ok, epochs=((1.0, 0.9), (0.8, 0.7)))
    assert run_health(ok) is None
    for name, kw, text in [("nan_val", {"epochs": ((1.0, 0.9), (1.0, float("nan")))}, "non-finite val_loss in epoch 1"),
                           ("inf_loss", {"epochs": ((float("inf"), 0.9),)}, "non-finite loss in epoch 0"),
                           ("nan_served", {"served": float("nan")}, "non-finite weights_val_loss"),
                           ("grads", {"nonfinite": 2}, "nonfinite_grad_steps 2 > 0")]:
        d = tmp_path / name
        d.mkdir()
        write_real_telemetry(d, **kw)
        assert text in (run_health(d) or ""), (name, run_health(d))
    assert run_health(tmp_path / "grads", max_nonfinite_grad_steps=5) is None        # the limit is a setting


def test_run_health_fails_the_null_that_the_real_writers_store_for_nan_and_a_missing_key(tmp_path):
    """P1-c (repair round 2): the writers never store NaN, they store null; a null loss is non-finite."""
    d = tmp_path / "nan"
    d.mkdir()
    write_real_telemetry(d, epochs=((1.0, 0.9), (1.0, float("nan"))), served=float("nan"))
    rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    assert rows[1]["val_loss"] is None and "val_loss" in rows[1]                # on disk: null, not NaN
    assert status["weights_val_loss"] is None and "weights_val_loss" in status
    why = run_health(d)
    assert why and "non-finite val_loss in epoch 1" in why and "non-finite weights_val_loss" in why
    # a key missing from an epoch row or from status.json is a failure with its own reason
    m = tmp_path / "missing"
    m.mkdir()
    write_real_telemetry(m)
    row = json.loads((m / "metrics.jsonl").read_text(encoding="utf-8"))
    del row["val_loss"]
    (m / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    st = json.loads((m / "status.json").read_text(encoding="utf-8"))
    del st["weights_val_loss"]
    (m / "status.json").write_text(json.dumps(st), encoding="utf-8")
    why = run_health(m) or ""
    assert "missing val_loss in epoch 0" in why and "missing weights_val_loss" in why
    # a cell that never finished an epoch: no metrics.jsonl, the served record says null
    never = tmp_path / "never"
    never.mkdir()
    from neural_trade.training.trainer import _record_served_epoch_in_status

    _record_served_epoch_in_status(SimpleNamespace(run_dir=never), None, None, "unknown (no validation history)")
    why = run_health(never) or ""
    assert "metrics.jsonl missing" in why and "non-finite weights_val_loss" in why
    assert "metrics.jsonl missing" in (run_health(tmp_path / "nothing_written") or "")   # no telemetry: unverifiable


class UnstableTrainer(FakeTrainer):
    """Telemetry by the real writers; the variants in ``bad`` (on the seeds in ``bad_seeds``, default all) get a
    NaN validation loss in epoch 1 and a NaN served loss, which the writers store as null."""

    def __init__(self, bad=(), bad_seeds=None, **kw):
        super().__init__(**kw)
        self.bad = set(bad)
        self.bad_seeds = None if bad_seeds is None else set(bad_seeds)

    def telemetry(self, run_dir, eng):
        nan = eng["variant"] in self.bad and (self.bad_seeds is None or int(eng["seed"]) in self.bad_seeds)
        write_real_telemetry(run_dir, epochs=((1.0, 0.95), (1.0, float("nan") if nan else 0.9)),
                             served=float("nan") if nan else 0.9)


def test_a_trial_with_a_non_finite_loss_is_failed_with_the_reason_and_told_to_optuna_as_fail(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    import optuna

    sw = make_sweep(tmp_path, bars_csv, UnstableTrainer(bad={"t0001"}), n_trials=3, top_k=5)
    res = sw.run()
    by = {t["number"]: t for t in res.trials}
    assert by[1]["state"] == "FAIL" and by[1]["value"] is None and "non-finite" in by[1]["reason"]
    assert "unstable training" in by[1]["reason"] and by[0]["state"] == by[2]["state"] == "COMPLETE"
    study = optuna.load_study(study_name=sw.sweep_id, storage=f"sqlite:///{(sw.directory / 'study.db').as_posix()}")
    assert [t.state.name for t in study.trials] == ["COMPLETE", "FAIL", "COMPLETE"]
    # P3: the re-run takes the successful trials only (fewer than top_k is fine)
    doc = summary(sw)
    assert sorted(r["number"] for r in doc["rerun"]) == [0, 2] and res.winner["number"] in (0, 2)


def _run_with(store_root, csv, *, overrides=None, name="other", sec=0.5):
    ov = {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 2, "BATCH_SIZE": 32}
    ov.update(overrides or {})
    sc = Scenario.from_dict(scenario_dict(csv, name=name, search={}, folds=[-2], seeds=[0], overrides=ov))
    Runner(sc, RunStore(store_root), trainer=FakeTrainer()).run()
    store = RunStore(store_root)
    for d in store.run_dirs(name):
        doc = json.loads((d / "result.json").read_text(encoding="utf-8"))
        doc["sec_per_step"] = sec
        (d / "result.json").write_text(json.dumps(doc), encoding="utf-8")
    store.sync(name)
    return store


def test_sec_per_step_only_comes_from_a_run_of_exactly_the_same_setup(tmp_path, bars_csv, synthetic_bars):
    root = tmp_path / "runs"
    same = _run_with(root, bars_csv, name="same", sec=0.5)
    base_cfg = Scenario.from_dict(scenario_dict(bars_csv)).base().copy(FOLD_INDEX=-2)
    sha = same.index.rows("same")[0]["dataset_sha256"]
    got, refused = latest_sec_per_step(same, base_cfg, sha, "cpu")
    assert got["sec_per_step"] == 0.5 and refused == []
    for name, ov in {"batch": {"BATCH_SIZE": 64}, "lookback": {"LOOKBACK": 30}, "series": {"INPUT_SERIES": ["close"]},
                     "maxseq": {"MAX_SEQUENCE_COUNT": 1400}}.items():
        _run_with(root, bars_csv, overrides=ov, name=name, sec=9.99)
    other_csv = tmp_path / "other.csv"
    synthetic_bars.iloc[:-50].to_csv(other_csv, index=False)               # another dataset fingerprint
    _run_with(root, other_csv, name="dataset", sec=9.99)
    store = RunStore(root)
    got, refused = latest_sec_per_step(store, base_cfg, sha, "cpu")
    assert got["sec_per_step"] == 0.5 and len(refused) == 5                # every other setup was refused, with a reason
    for word in ("BATCH_SIZE", "LOOKBACK", "INPUT_SERIES", "MAX_SEQUENCE_COUNT", "dataset"):
        assert any(word in r for r in refused), word
    # the device: a GPU run is not the CPU setup, and the CPU run is not the GPU setup
    assert latest_sec_per_step(store, base_cfg, sha, "gpu")[0] is None
    run_dir = store.root / store.index.rows("same")[0]["run_dir"]
    (run_dir / "env.json").write_text(json.dumps({"gpus": ["/physical_device:GPU:0"]}), encoding="utf-8")
    assert latest_sec_per_step(store, base_cfg, sha, "gpu")[0]["sec_per_step"] == 0.5
    assert latest_sec_per_step(store, base_cfg, sha, "cpu")[0] is None
    # a sweep whose setup has no run of its own refuses instead of falling back to another setup
    sw = Sweep(Scenario.from_dict(scenario_dict(bars_csv)), store, SweepOptions(mode="optuna", dry_run=True,
               parallel_record=None, device="gpu" if False else "cpu"), gpu_check=free_gpu, announce=lambda t: None)
    with pytest.raises(SweepError, match="THIS setup") as exc:
        sw.run()
    assert "run(s) of other setups were not used" in str(exc.value)


def test_the_budget_uses_each_trials_batch_and_does_not_count_the_first_seeds_dev_cells_twice(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=7, top_k=5, dry_run=True,
                    changes={"search": {"BATCH_SIZE": {"low": 16, "high": 128, "log": True}, "LR": SEARCH["LR"]}})
    res = sw.run()
    b = res.budget
    up, ex = b["steps_per_epoch_upper_per_fold"], b["steps_per_epoch_expected_per_fold"]
    assert all(ex[f] < up[f] for f in up)                                       # expected < the lowest-batch bound
    assert b["expected_gpu_hours"] < b["gpu_hours"]
    # the trials that fit: the largest n whose upper bound stays within max_hours, with the re-run of top_k
    per = b["search_gpu_hours"] / 7
    assert b["trials_that_fit"] == int((12 - b["rerun_gpu_hours"]) // per)
    # no double counting: the re-run is (seeds-1) dev cells + seeds test cells per trial, not seeds on every fold
    e = b["epochs_upper_bound"]

    def cell(f):
        return up[str(f)] * e * 0.01 + 1.0

    per_trial = sum(cell(f) * 2 for f in b["dev_folds"]) + sum(cell(f) * 3 for f in b["test_folds"])
    assert b["rerun_gpu_hours"] * 3600 == pytest.approx(5 * per_trial)
    from neural_trade.experiments.sweep import DEFAULT_SEARCH

    assert DEFAULT_SEARCH["BATCH_SIZE"]["low"] == 128                         # not 64: the bound is within 2x of the default


def test_a_refused_launch_leaves_nothing_under_sweeps(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=500, max_hours=1e-4)
    with pytest.raises(SweepError):
        sw.run()
    assert not (tmp_path / "runs" / "sweeps").exists()


def test_the_parallel_record_warns_when_its_measured_setup_differs(tmp_path, bars_csv, caplog):
    cfg = Config()
    rec = {"setup": "BTC/USDT 1-minute (x.csv), LOOKBACK 60, HORIZON_STEPS [10,15,20], BATCH_SIZE 256, FOLD_INDEX -3"}
    w = record_setup_warnings(rec, cfg)
    assert any("input layout" in x and "D-047" in x for x in w)                  # the pre-OHLCV record
    assert any("BATCH_SIZE 256" in x for x in record_setup_warnings(rec, cfg.copy(BATCH_SIZE=128)))
    assert any("HORIZON_STEPS" in x for x in record_setup_warnings(rec, cfg.copy(HORIZON_STEPS=[5, 10, 15])))
    assert record_setup_warnings({"setup": rec["setup"] + " OHLCV"}, cfg) == []
    assert record_setup_warnings({}, cfg) == ["the parallel record states no measured setup"]
    pytest.importorskip("optuna")
    p = _record(tmp_path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc.update(setup=rec["setup"], measured_utc="2026-09-29T04:00Z")
    p.write_text(json.dumps(doc), encoding="utf-8")
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=2, parallel=2, dry_run=True)
    sw.options.parallel_record = str(p)
    import logging

    with caplog.at_level(logging.WARNING):
        sw.run()
    assert any("input layout" in x for x in sw.record["warnings"])    # also logged, and in sweep.json


# ---- claims on Windows edge cases
def test_a_claim_file_being_written_is_held_not_stolen_and_a_stale_takeover_is_atomic(tmp_path):
    import os
    import threading

    d = tmp_path / "claims"
    d.mkdir()
    claims = CellClaims(d)
    (d / "half.lock").write_text("", encoding="utf-8")                      # the owner has created it, not written yet
    assert not claims.claim("half") and (d / "half.lock").exists()           # held: not deleted, not taken
    old = (d / "half.lock").stat().st_mtime - 60
    os.utime(d / "half.lock", (old, old))                                    # an empty file left long ago is stale
    assert claims.claim("half")
    # exactly one of many contenders takes a stale claim
    (d / "c.lock").write_text(json.dumps({"pid": 2 ** 22 + 12345}), encoding="utf-8")        # no such process
    won = []

    def take():
        if CellClaims(d).claim("c"):
            won.append(1)

    ts = [threading.Thread(target=take) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(won) == 1 and not list(d.glob("*.stale-*"))
    # a PermissionError opening the claim (Windows: being deleted by its owner) counts as held, no crash
    (d / "busy.lock").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    import builtins

    real_open = builtins.open

    def refuse(path, mode="r", *a, **k):
        if str(path).endswith("busy.lock") and "x" in mode:
            raise PermissionError(13, "denied")
        return real_open(path, mode, *a, **k)

    builtins.open = refuse
    try:
        assert not claims.claim("busy")
    finally:
        builtins.open = real_open


def test_a_reused_pid_is_told_from_the_owner_by_the_process_start_time(tmp_path):
    import os
    import time

    from neural_trade.experiments.claims import process_start_time

    me = process_start_time(os.getpid())
    assert me is not None and abs(me - time.time()) < 3600 * 24 * 365
    d = tmp_path / "claims"
    d.mkdir()
    claims = CellClaims(d)
    assert claims.claim("k") and claims.holder("k") == os.getpid()           # our claim records our start time
    doc = json.loads((d / "k.lock").read_text(encoding="utf-8"))
    assert doc["start"] == pytest.approx(me, abs=1.0)
    doc["start"] = me - 5000                                                  # the pid is alive, but not the same process
    (d / "k.lock").write_text(json.dumps(doc), encoding="utf-8")
    assert claims.holder("k") is None and claims.claim("k")


# ---- the winner and the re-run follow the dev seed mean, whatever the test column and one seed say
class ShapedTrainer(FakeTrainer):
    """Dev quality of a trial: its FIRST seed is uninformative (a trap for a single-seed ranking), the later seeds
    (hence the seed mean) favour a high LAMBDA_DIR; the test fold favours a LOW LAMBDA_DIR (anti-correlated)."""

    def __call__(self, ctx, *, calibrate, save_artifacts):
        eng = json.loads((ctx.run_dir / "meta.json").read_text(encoding="utf-8"))["engine"]
        self.calls.append(eng["cell_key"])
        cfg = ctx.config
        high, seed = float(cfg.LAMBDA_DIR) >= 1.0, int(cfg.SEED)       # LAMBDA_DIR in (0, 2]
        if int(cfg.FOLD_INDEX) == -1:
            noise = 20.0 if high else 0.5                              # the test fold prefers a LOW value
        elif seed == 0:
            noise = 3.0                                                # the search's own seed cannot tell them apart
        else:
            noise = 0.5 if high else 20.0                              # the later seeds (the mean) prefer HIGH
        write_real_telemetry(ctx.run_dir)
        return fake_result(cfg, noise=noise)


def _dev_mean_and_test(sw, variant):
    rows = [r for r in sw.store.index.rows(sw.sweep_id) if r["variant"] == variant and r["status"] == "done"]
    dev_by_fold = {}
    for r in rows:
        if r["role"] == "dev":
            dev_by_fold.setdefault(r["fold"], []).append(r["sharpe_net"])
    dev = float(np.mean([np.mean(v) for v in dev_by_fold.values()]))
    test = float(np.mean([r["sharpe_net"] for r in rows if r["role"] == "test"]))
    seed0 = float(np.mean([r["sharpe_net"] for r in rows if r["role"] == "dev" and r["seed"] == 0]))
    return dev, test, seed0


def test_the_winner_and_the_rerun_order_follow_the_dev_seed_mean_not_the_test_column_nor_one_seed(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    sw = make_sweep(tmp_path, bars_csv, ShapedTrainer(), n_trials=6, top_k=3, rerun_seeds=3, sampler_seed=4)
    res = sw.run()
    table = summary(sw)["rerun"]
    assert len(table) == 3 and all(r["n_seeds"] == 3 for r in table)
    # the re-run takes the BEST trials of the search (by the dev single-seed value), not any others
    values = {t["number"]: t["value"] for t in res.trials}
    assert {r["number"] for r in table} == {n for n, _ in sorted(values.items(), key=lambda kv: -kv[1])[:3]}
    assert [r["number"] for r in res.ranking[:3]] == sorted(values, key=lambda n: -values[n])[:3]
    stats = {r["variant"]: _dev_mean_and_test(sw, r["variant"]) for r in table}
    best_dev = max(stats, key=lambda v: stats[v][0])
    best_test = max(stats, key=lambda v: stats[v][1])
    best_seed0 = max(stats, key=lambda v: stats[v][2])
    assert best_test != best_dev and best_seed0 != best_dev, (stats, [(r["variant"], r["params"]) for r in table])        # the data really are a trap for both mutants
    assert res.winner["variant"] == best_dev and table[0]["variant"] == best_dev
    assert [r["variant"] for r in table] == sorted(stats, key=lambda v: -stats[v][0])
    for r in table:
        assert r["dev_seed_mean_net_sharpe"] == pytest.approx(stats[r["variant"]][0])
        assert r["test_sharpe_net"] == pytest.approx(stats[r["variant"]][1]) and "never used to rank" in r["test_note"]


# ------------------------------------------------------------------ repair round 2
OLD_REFERENCE = FIXTURES / "nt030_reference_default_82a848f"


def test_a_pre_d047_reference_run_is_not_the_same_setup_as_todays_default(tmp_path, bars_csv):
    """P1-d: the 9 reference_default runs of 82a848f are close-only (before D-047); their config.yaml has no
    INPUT_SERIES. Read through Config.from_yaml they get today's defaults and look 'same' (0.10 s/step instead of
    D-047's ~0.17); compared by their raw stored keys, the missing field is unknown and refuses them."""
    from neural_trade.experiments.sweep import SETUP_FIELDS, setup_mismatch

    assert {"MODEL_NAME", "ATTENTION_MODE", "DETERMINISTIC_GRU", "PROBE_GRADIENTS"} <= set(SETUP_FIELDS)
    today = Config()
    for cfg_file in sorted(OLD_REFERENCE.glob("config_*.yaml")):
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        assert "INPUT_SERIES" not in raw and "INDICATOR_FAMILIES" not in raw          # the stored file, unchanged
        # the trap: Config.from_yaml fills today's defaults, so every setup field looks equal
        filled = Config.from_yaml(cfg_file)
        assert all(getattr(filled, n) == getattr(today, n) for n in ("INPUT_SERIES", "BATCH_SIZE", "LOOKBACK"))
        run_dir = tmp_path / cfg_file.stem
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(cfg_file.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "env.json").write_text((OLD_REFERENCE / "env.json").read_text(encoding="utf-8"), encoding="utf-8")
        why = setup_mismatch({"dataset_sha256": "x"}, run_dir, today, "x", "gpu")
        assert why is not None and "INPUT_SERIES missing" in why, why
    # through the index: the old run is refused, and a sweep of today's setup refuses rather than use it
    store = _run_with(tmp_path / "runs", bars_csv, name="oldref", sec=0.1092)
    row = store.index.rows("oldref")[0]
    run_dir = store.root / row["run_dir"]
    (run_dir / "config.yaml").write_text((OLD_REFERENCE / "config_f-3__s0.yaml").read_text(encoding="utf-8"),
                                         encoding="utf-8")
    (run_dir / "env.json").write_text((OLD_REFERENCE / "env.json").read_text(encoding="utf-8"), encoding="utf-8")
    got, refused = latest_sec_per_step(store, today, row["dataset_sha256"], "gpu")
    assert got is None and len(refused) == 1 and "INPUT_SERIES missing" in refused[0]
    sw = Sweep(Scenario.from_dict(scenario_dict(bars_csv)), store, SweepOptions(mode="optuna", device="gpu",
               parallel_record=None), gpu_check=free_gpu, announce=lambda t: None)
    sw.base_config, sw.dataset_sha = today, row["dataset_sha256"]
    with pytest.raises(SweepError, match="no measured sec_per_step for THIS setup") as exc:
        sw._sec_per_step()
    assert "INPUT_SERIES missing" in str(exc.value)
    sw.options.sec_per_step = 0.1735                                              # or the caller states it
    assert sw._sec_per_step() == (0.1735, {"source": "given (--sec-per-step)"})


def test_each_new_setup_field_refuses_a_run_that_differs_or_lacks_it(tmp_path, bars_csv):
    from neural_trade.experiments.sweep import setup_mismatch

    store = _run_with(tmp_path / "runs", bars_csv, name="same", sec=0.5)
    row = store.index.rows("same")[0]
    run_dir = store.root / row["run_dir"]
    cfg = Config.from_yaml(run_dir / "config.yaml")
    assert setup_mismatch(row, run_dir, cfg, row["dataset_sha256"], "cpu") is None
    text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    for name, other in [("DETERMINISTIC_GRU", not cfg.DETERMINISTIC_GRU), ("PROBE_GRADIENTS", not cfg.PROBE_GRADIENTS),
                        ("ATTENTION_MODE", "none" if cfg.ATTENTION_MODE != "none" else "time")]:
        assert name in setup_mismatch(row, run_dir, cfg.copy(**{name: other}), row["dataset_sha256"], "cpu")
        lines = [ln for ln in text.splitlines() if not ln.startswith(f"{name}:")]
        (run_dir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert f"{name} missing" in setup_mismatch(row, run_dir, cfg, row["dataset_sha256"], "cpu")
        (run_dir / "config.yaml").write_text(text, encoding="utf-8")


def test_a_rerun_stopped_by_a_busy_gpu_is_stopped_with_a_reason_not_complete(tmp_path, bars_csv):
    """P2 (1): the re-run's GPU-free check fails: state 'stopped', a stop_reason, no winner; resume finishes it."""
    pytest.importorskip("optuna")
    checks = []

    def busy_from_third():
        checks.append(1)
        return GpuStatus(len(checks) <= 2, {"stub": len(checks)})

    trainer = FakeTrainer()
    sw = make_sweep(tmp_path, bars_csv, trainer, n_trials=2, top_k=1)
    sw.gpu_check = busy_from_third
    res = sw.run()
    assert res.state == "stopped" and res.winner is None and res.stop_reason and "re-run stopped" in res.stop_reason
    doc = summary(sw)
    assert doc["state"] == "stopped" and "re-run stopped" in doc["stop_reason"] and not (sw.directory / "winner.json").exists()
    n = len(trainer.calls)
    again = make_sweep(tmp_path, bars_csv, trainer, n_trials=2, top_k=1, resume=True).run()
    assert again.state == "complete" and again.winner is not None and len(trainer.calls) > n


def test_the_cli_exits_non_zero_when_the_rerun_is_stopped(tmp_path, bars_csv, monkeypatch, capsys):
    pytest.importorskip("optuna")
    from neural_trade.experiments import runner as runner_mod
    from neural_trade.experiments import sweep as sweep_mod

    checks = []

    def busy_from_third(*a, **k):
        checks.append(1)
        return GpuStatus(len(checks) <= 2, {"stub": len(checks)})

    monkeypatch.setattr(sweep_mod, "nvidia_smi_gpu_check", busy_from_third)
    monkeypatch.setattr(runner_mod, "train_cell", FakeTrainer())
    spec = tmp_path / "sw.yaml"
    spec.write_text(yaml.safe_dump(scenario_dict(bars_csv)), encoding="utf-8")
    code = main(["sweep", str(spec), "--mode", "optuna", "--store", str(tmp_path / "runs"), "--n-trials", "2",
                 "--top-k", "1", "--sec-per-step", "0.01", "--parallel-record", str(tmp_path / "none.json")])
    out = capsys.readouterr().out
    assert code == 1 and '"state": "stopped"' in out and "re-run stopped" in out


def test_a_rerun_batch_above_the_watch_level_stops_launching_the_next(tmp_path, bars_csv):
    """The re-run's batches are watched like the search's (N > 1)."""
    pytest.importorskip("optuna")
    sw, trainer, _ = _parallel_sweep(tmp_path, bars_csv, n_trials=4, parallel=2)
    sw.options.top_k = 4
    quiet = {"mean_sm_pct": 10.0, "mean_fb_mb": 1000.0, "peak_fb_mb": 1200.0}
    high = {"mean_sm_pct": 95.0, "mean_fb_mb": 9000.0, "peak_fb_mb": 11000.0}

    class Seq(Monitor):
        def stop(self):
            return dict(high if self.batches == 3 else quiet)       # the first re-run batch reads high

    sw.monitor_factory = Seq()
    res = sw.run()
    assert res.state == "stopped" and "re-run stopped launching" in res.stop_reason and res.winner is None
    assert sw.monitor_factory.batches == 3


class PatchingRunner(Runner):
    """Rewrites a finished cell's stored scores the way the scorer would have stored them (QA's probe): ``plan``
    maps a variant to (net Sharpe, trades, fee bps)."""

    patch = {}

    def run(self, **kw):
        rep = super().run(**kw)
        for item in rep.ran:
            d = Path(item["run_dir"])
            res = json.loads((d / "result.json").read_text(encoding="utf-8"))
            eng = json.loads((d / "meta.json").read_text(encoding="utf-8"))["engine"]
            if eng["variant"] not in self.patch or res.get("status") != "done":
                continue
            sharpe, trades, fee = self.patch[eng["variant"]]
            res["scores"].update({"backtest/sharpe_net": sharpe, "backtest/n_trades": trades})
            (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
            if fee is not None:
                rp = d / res["report"]
                rep_doc = json.loads(rp.read_text(encoding="utf-8"))
                rep_doc["backtest"]["config"]["fee_bps"] = fee
                rp.write_text(json.dumps(rep_doc), encoding="utf-8")
        return rep


def test_the_rerun_takes_only_trials_that_pass_the_search_time_guard_rails(tmp_path, bars_csv):
    """P2 (2), QA's probe: t0000 never trades (Sharpe 5000), t0001 was scored at 13 bps (Sharpe 4000), t0002 is
    eligible. Neither ineligible trial enters the re-run, t0002 wins, and Optuna is told INELIGIBLE_VALUE for both."""
    pytest.importorskip("optuna")
    import optuna

    from neural_trade.experiments.sweep import INELIGIBLE_VALUE

    trainer = FakeTrainer()

    class Probe(PatchingRunner):
        patch = {"t0000": (5000.0, 0, None), "t0001": (4000.0, 50, 13.0)}

    sw = make_sweep(tmp_path, bars_csv, trainer, n_trials=3, top_k=2)
    sw._runner_factory = lambda sc: Probe(sc, sw.store, trainer=trainer, claim_cells=True)
    res = sw.run()
    by = {t["number"]: t for t in res.trials}
    assert by[0]["value"] == 5000.0 and not by[0]["eligible"] and "min_trades" in by[0]["ineligible"]
    assert by[1]["value"] == 4000.0 and not by[1]["eligible"] and "cost_profile" in by[1]["ineligible"]
    assert by[2]["eligible"] and all(t["state"] == "COMPLETE" for t in res.trials)
    assert [r["number"] for r in summary(sw)["rerun"]] == [2] and res.winner["number"] == 2
    assert [r["number"] for r in res.ranking if r["rank"] is not None] == [2]
    study = optuna.load_study(study_name=sw.sweep_id, storage=f"sqlite:///{(sw.directory / 'study.db').as_posix()}")
    told = {t.number: t.value for t in study.trials}
    assert told[0] == told[1] == INELIGIBLE_VALUE and told[2] == by[2]["value"]
    # quick mode: the leader is an eligible trial too
    q = make_sweep(tmp_path / "q", bars_csv, trainer, mode="quick", overhead_s=70.0, sec_per_step=0.001)
    q._runner_factory = lambda sc: Probe(sc, q.store, trainer=trainer, claim_cells=True)
    q.run()
    leader = summary(q)["leader"]
    assert leader is None or leader["number"] not in (0, 1)


def test_at_n1_the_parallel_records_watch_level_is_not_applied(tmp_path, bars_csv):
    """P2 (3): a single D-047 process may peak above the pre-D-047 record's N=1 level; at N=1 the record is not
    used (no level, no warning), the GPU-free check before each batch still runs."""
    pytest.importorskip("optuna")
    p = _record(tmp_path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["setup"] = "LOOKBACK 60, HORIZON_STEPS [10,15,20], BATCH_SIZE 256"
    p.write_text(json.dumps(doc), encoding="utf-8")
    checks = []
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=3, top_k=1)
    sw.options.parallel_record = str(p)
    sw.monitor_factory = Monitor({"mean_sm_pct": 20.0, "mean_fb_mb": 4000.0, "peak_fb_mb": 5000.0})
    sw.gpu_check = lambda: checks.append(1) or GpuStatus(True, {})
    assert exceeds_watch_level({"mean_sm_pct": 20.0, "peak_fb_mb": 5000.0}, doc["utilization"]["1"])   # would stop
    res = sw.run()
    assert res.state == "complete" and len(res.trials) == 3 and res.winner is not None
    assert len(checks) >= 4 and "warnings" not in sw.record                       # 3 search batches + the re-run


def test_a_cell_finished_by_another_process_after_the_plan_is_not_trained_again(tmp_path, bars_csv):
    """M5: after claiming a cell, the runner re-checks whether another process finished it since the plan."""
    store = RunStore(tmp_path / "runs")
    sc = Scenario.from_dict(scenario_dict(bars_csv, search={}, folds=[-2, -1], seeds=[0, 1]))
    other = FakeTrainer()

    def run_the_other_process_once(eng):
        if not other.calls:
            Runner(sc, store, trainer=other, claim_cells=True).run()     # finishes every cell this one does not hold

    first = FakeTrainer(on_call=run_the_other_process_once)
    rep = Runner(sc, store, trainer=first, claim_cells=True).run()
    assert len(first.calls) == 1 and len(other.calls) == 3
    assert sorted(first.calls + other.calls) == sorted(c.key for c in sc.cells())
    assert len(rep.skipped) == 3


def test_a_trial_unstable_only_in_its_rerun_seeds_cannot_win(tmp_path, bars_csv):
    """M7: the re-run's later seeds of every top trial have a non-finite loss: no winner, the table says why."""
    pytest.importorskip("optuna")
    sw = make_sweep(tmp_path, bars_csv, UnstableTrainer(bad={"t0000", "t0001"}, bad_seeds={1, 2}), n_trials=2,
                    top_k=2, rerun_seeds=3)
    res = sw.run()
    assert all(t["state"] == "COMPLETE" for t in res.trials)                     # the search's seed 0 was fine
    table = summary(sw)["rerun"]
    assert len(table) == 2 and all(r["unstable"] and "non-finite" in r["unstable"] for r in table)
    assert res.state == "complete" and res.winner is None
    # one stable trial among them wins
    sw2 = make_sweep(tmp_path / "b", bars_csv, UnstableTrainer(bad={"t0000"}, bad_seeds={1, 2}), n_trials=2,
                     top_k=2, rerun_seeds=3)
    assert sw2.run().winner["number"] == 1


def test_the_budget_upper_bound_uses_the_spaces_lowest_batch_not_the_base_batch(tmp_path, bars_csv):
    """M14: base BATCH_SIZE 32, the space's lowest 16: the upper bound counts the steps of batch 16."""
    pytest.importorskip("optuna")
    from neural_trade.experiments.sweep import steps_per_epoch

    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=3, dry_run=True,
                    changes={"search": {"BATCH_SIZE": {"low": 16, "high": 128, "log": True}}})
    res = sw.run()
    up = res.budget["steps_per_epoch_upper_per_fold"]
    assert int(sw.base_config.BATCH_SIZE) == 32
    for f, n in sw.train_n.items():
        assert up[str(f)] == steps_per_epoch(n, 16) > steps_per_epoch(n, 32)


# ------------------------------------------------------------------ slow: real training, real processes
@pytest.mark.slow
def test_a_real_parallel_optuna_sweep_trains_on_cpu_in_two_processes_through_the_cli(tmp_path, bars_csv, monkeypatch, capsys):
    """Two real `scenario run --claim-cells` processes per batch (CPU, EPOCHS 1); the stubbed tests cover the rest."""
    pytest.importorskip("optuna")
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    d = scenario_dict(bars_csv)
    d["overrides"]["EPOCHS"] = 1
    d["name"] = "swreal"
    path = tmp_path / "swreal.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    rec = tmp_path / "parallel_n.json"
    rec.write_text(json.dumps({"allowed_n": 2, "utilization": {"2": {"mean_sm_pct": 100.0, "peak_sm_pct": 100.0,
                                                                     "mean_fb_mb": 1, "peak_fb_mb": 10 ** 9}}}),
                   encoding="utf-8")
    assert main(["sweep", str(path), "--mode", "optuna", "--store", str(runs), "--n-trials", "2", "--parallel", "2",
                 "--parallel-record", str(rec), "--top-k", "1", "--rerun-seeds", "1", "--sec-per-step", "0.05"]) == 0
    assert "GPU budget" in capsys.readouterr().out
    doc = json.loads((runs / "sweeps" / "swreal-optuna" / "sweep.json").read_text(encoding="utf-8"))
    assert doc["state"] == "complete" and len(doc["rerun"]) == 1 and doc["winner"] is not None
    rows = RunStore(runs).index.rows("swreal-optuna")
    done = [r for r in rows if r["status"] == "done"]
    assert len({r["cell_key"] for r in done}) == len(done) == 3          # 2 dev cells + the winner's test cell
    assert all(r["sharpe_net"] is not None and math.isfinite(r["sharpe_net"]) for r in done)

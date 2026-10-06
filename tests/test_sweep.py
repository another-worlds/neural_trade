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
    GpuStatus, SearchSpace, Sweep, SweepError, SweepOptions, cell_seconds, dev_net_sharpe, exceeds_watch_level,
    load_parallel_record, size_quick, held_out_columns)

HORIZONS = ("h0", "h1", "h2")
SEARCH = {"LR": {"low": 1e-4, "high": 1e-2, "log": True}, "LAMBDA_DIR": {"low": 0.1, "high": 2.0}}


# ------------------------------------------------------------------ helpers
def fake_result(cfg: Config):
    from sklearn.preprocessing import StandardScaler

    from neural_trade.data.processor import split_arrays

    arrays = split_arrays(cfg)
    rng = np.random.default_rng([int(cfg.SEED), int(cfg.FOLD_INDEX) % 97, int(cfg.LR * 1e6) % 9973])
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
        return fake_result(ctx.config)


def scenario_dict(csv, **changes):
    s = {"schema_version": 1, "name": "sw", "description": "sweep test",
         "overrides": {"CSV_PATH": str(csv), "MAX_SEQUENCE_COUNT": 1500, "EPOCHS": 2, "BATCH_SIZE": 32},
         "folds": [-2, -1], "seeds": [0], "strategy": {"name": "calibrated_quantile", "params": {}},
         "backtest": {"random_seeds": 5}, "run": {"calibrate": False, "save_artifacts": False}, "search": SEARCH}
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
    trainer = FakeTrainer()
    first = make_sweep(tmp_path, bars_csv, trainer, mode="quick", overhead_s=70.0, sec_per_step=0.001)
    first.run()
    # the fake runs record no status.json, so inject a value the way a real run's result.json carries it
    for d in first.store.run_dirs(first.sweep_id)[:1]:
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
    assert "GPU budget" in seen["said"][0] and budget["gpu_hours"] > 0 and budget["max_hours"] == 12.0
    assert budget["gpu_hours"] == pytest.approx(budget["search_gpu_hours"] + budget["rerun_gpu_hours"])
    assert budget["trials_to_run"] == 3 and budget["rerun"]["top_k"] == 2 and budget["rerun"]["seeds"] == 3
    # trials x dev folds x steps x sec_per_step (+ overhead) by hand
    steps = sum(budget["steps_per_epoch_per_dev_fold"].values())
    assert budget["search_gpu_hours"] * 3600 == pytest.approx(3 * (steps * budget["epochs_upper_bound"] * 0.01 + 1.0))
    assert res.budget["gpu_hours"] == budget["gpu_hours"]


def test_a_budget_over_max_hours_is_refused_before_anything_starts(tmp_path, bars_csv):
    pytest.importorskip("optuna")
    trainer = FakeTrainer()
    sw = make_sweep(tmp_path, bars_csv, trainer, n_trials=3, max_hours=1e-6)
    with pytest.raises(SweepError, match="over --max-hours"):
        sw.run()
    assert trainer.calls == [] and not sw.summary_path.exists()
    assert SweepOptions().max_hours == 12.0                                   # the default: one night


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
    sw = make_sweep(tmp_path, bars_csv, FakeTrainer(), n_trials=2, top_k=2, min_trades=10 ** 9)
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

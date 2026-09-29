"""NT-082: the long-run launcher and tracker (notebook/longrun.py), the long_360d scenario spec and the
SHUFFLE_BUFFER option of the training dataset. Fake run directories only: nothing trains, nothing
is launched (subprocess.Popen is mocked)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from neural_trade.core.config import Config
from neural_trade.core.exceptions import InvalidConfigurationError
from neural_trade.notebook import longrun as L

REPO = Path(__file__).resolve().parent.parent
LONG_SPEC = REPO / "configs" / "scenarios" / "long_360d.yaml"


# ------------------------------------------------------------------ SHUFFLE_BUFFER
def test_shuffle_buffer_field_metadata_and_range():
    spec = Config.field_specs()["SHUFFLE_BUFFER"]
    assert spec.default == 2048 and spec.group == "training" and spec.unit == "sequences"
    assert spec.minimum == 0 and not spec.tunable and not spec.deprecated
    assert Config(SHUFFLE_BUFFER=0).SHUFFLE_BUFFER == 0
    with pytest.raises(InvalidConfigurationError):
        Config(SHUFFLE_BUFFER=-1)


def test_shuffle_buffer_size_zero_means_the_whole_training_block():
    from neural_trade.data.datasets import shuffle_buffer_size

    assert shuffle_buffer_size(Config(), 518_432) == 2048
    assert shuffle_buffer_size(Config(SHUFFLE_BUFFER=0), 518_432) == 518_432
    assert shuffle_buffer_size(Config(SHUFFLE_BUFFER=100), 50) == 100
    assert shuffle_buffer_size(object(), 10) == 2048          # a config without the field: today's buffer


def _arrays(n):
    X = np.arange(n, dtype=np.float32)[:, None].repeat(4, axis=1)
    y = np.zeros((n, 3), np.float32)
    lc = np.arange(n, dtype=np.float32)
    ext = np.zeros((n, 3), np.float32)
    return X, y, lc, ext


def _order(ds):
    return np.concatenate([b[0][:, 0].numpy() for b in ds]).astype(int)


def test_default_training_pipeline_is_the_old_one_and_validation_stays_unshuffled():
    """With the default the training element order equals the pre-NT-082 pipeline's (a 2048 buffer,
    the same seed), in two consecutive epochs; validation keeps its order."""
    import tensorflow as tf

    from neural_trade.data.datasets import create_datasets

    cfg = Config(BATCH_SIZE=64, SEED=7)
    X, y, lc, ext = _arrays(5000)
    train, val = create_datasets(cfg, X, y, lc, ext, X[:300], y[:300], lc[:300], ext[:300])
    old = tf.data.Dataset.from_tensor_slices((X, y, lc.reshape(-1, 1), ext))
    old = old.shuffle(buffer_size=2048, seed=7, reshuffle_each_iteration=True).batch(64).prefetch(tf.data.AUTOTUNE)
    for _ in range(2):
        np.testing.assert_array_equal(_order(train), _order(old))
    np.testing.assert_array_equal(_order(val), np.arange(300))


def test_full_shuffle_mixes_the_whole_block_and_reshuffles_each_epoch():
    from neural_trade.data.datasets import create_datasets

    X, y, lc, ext = _arrays(5000)
    train, val = create_datasets(Config(BATCH_SIZE=256, SHUFFLE_BUFFER=0, SEED=3), X, y, lc, ext,
                                 X[:300], y[:300], lc[:300], ext[:300])
    e1, e2 = _order(train), _order(train)
    assert sorted(e1) == list(range(5000)) and sorted(e2) == list(range(5000))
    assert (e1 != e2).any()
    # a 2048 buffer can only emit elements < 2048 + k at position k; a full buffer draws from the whole block
    assert e1[:256].max() > 2048 + 256
    np.testing.assert_array_equal(_order(val), np.arange(300))
    again, _ = create_datasets(Config(BATCH_SIZE=256, SHUFFLE_BUFFER=0, SEED=3), X, y, lc, ext,
                               X[:300], y[:300], lc[:300], ext[:300])
    np.testing.assert_array_equal(_order(again), e1)                      # seeded


# ------------------------------------------------------------------ the scenario spec
def test_long_360d_spec_is_one_dev_cell_on_fold_minus_2():
    from neural_trade.experiments.scenario import Scenario

    sc = Scenario.from_yaml(LONG_SPEC)
    assert sc.name == "long_360d" and sc.strategy.name == "calibrated_quantile"
    assert sc.run.calibrate is True and sc.run.save_artifacts is True
    ((cell, cfg),) = sc.validate()
    assert (cell.fold, cell.seed) == (-2, 0)
    assert cfg.CSV_PATH == "Bitcoin_BTCUSDT.csv" and cfg.N_FOLDS == 14 and cfg.MAX_SEQUENCE_COUNT == 698160
    assert cfg.VAL_FRACTION == cfg.CAL_FRACTION == 0.061877 and cfg.SHUFFLE_BUFFER == 0
    assert cfg.BATCH_SIZE == 2048 and cfg.EPOCHS == 40 and cfg.LR == Config().LR


def test_long_360d_layout_gives_a_360_day_training_block():
    """The fold arithmetic of the spec without the 290 MB file: fold -2 trains on 518,432 windows
    (360.02 days of one-minute anchors) with 43,200-window val and cal blocks."""
    from neural_trade.data.splits import make_purged_splits

    folds = make_purged_splits(698160, lookback=60, horizon_steps=[10, 15, 20], n_folds=14,
                               val_fraction=0.061877, cal_fraction=0.061877)
    f = folds[-2]
    assert (len(f.train), len(f.val), len(f.cal), len(f.test)) == (518432, 43200, 43200, 46544)
    assert len(folds) == 13 and f.gap == 80


# ------------------------------------------------------------------ fake scenario and cells
@pytest.fixture
def spec(tmp_path):
    """A one-cell scenario spec in tmp_path (root) whose base config is the repo's default."""
    path = tmp_path / "spec.yaml"
    path.write_text(json.dumps({
        "schema_version": 1, "name": "fake_long", "base_config": str(REPO / "configs" / "default.yaml"),
        "overrides": {"EPOCHS": 20}, "folds": [-2], "seeds": [0]}), encoding="utf-8")
    return path


def _expected(spec_path):
    from neural_trade.experiments.scenario import Scenario, config_hash

    sc = Scenario.from_yaml(spec_path)
    ((cell, cfg),) = sc.validate()
    return sc, cell.key, config_hash(cfg)


def _stamp(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_cell(root, spec_path, *, created, epochs=0, epoch_seconds=None, partial=True, result=None, mtime=None,
              name=None, config_hash=None):
    sc, key, chash = _expected(spec_path)
    d = root / "runs" / "scenarios" / sc.name / (name or f"{_stamp(created)}-abc-{key}")
    d.mkdir(parents=True)
    meta = {"run_id": d.name, "created_utc": _stamp(created),
            "engine": {"scenario": sc.name, "cell_key": key, "config_hash": config_hash or chash,
                       "settings_hash": sc.settings_hash}, "blocks": {"val": {"n": 43200}}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "config.yaml").write_text(Config(EPOCHS=20).to_yaml(), encoding="utf-8")
    secs = epoch_seconds or [30.0] + [10.0] * max(0, epochs - 1)
    rows = []
    for e in range(epochs):
        rows.append({"epoch": e, "time": created + 60 + sum(secs[:e + 1]), "loss": 8 - e * 0.1,
                     "val_loss": 7 - e * 0.1 + (0.5 if e == 2 else 0), "lr": 1e-3, "lr_indicator": 5e-3,
                     "epoch_seconds": secs[e], "sec_per_step": secs[e] / 100,
                     **{f"val_crps_h{i}": 0.3 + i / 10 for i in range(3)}, **{f"crps_h{i}": 0.4 for i in range(3)},
                     **{f"val_dir_bal_acc_h{i}": 0.5 for i in range(3)}})
    if epochs:
        text = "".join(json.dumps(r) + "\n" for r in rows)
        if partial:
            text += '{"epoch": ' + str(epochs) + ', "loss": 7.'          # an epoch line being written
        (d / "metrics.jsonl").write_text(text, encoding="utf-8")
        cols = ["epoch", "loss", "val_loss"]
        csv = ",".join(cols) + "\n" + "".join(f"{r['epoch']},{r['loss']},{r['val_loss']}\n" for r in rows)
        (d / "training_log.csv").write_text(csv + (f"{epochs},7.1" if partial else ""), encoding="utf-8")
        (d / "status.json").write_text(json.dumps({"epochs_completed": epochs, "sec_per_step": 0.1}), encoding="utf-8")
    if result is not None:
        (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if mtime is not None:
        for f in d.iterdir():
            os.utime(f, (mtime, mtime))
    return d


def launch_record(root, spec_path, *, pid, started):
    sc, _, _ = _expected(spec_path)
    d = root / "runs" / "scenarios" / sc.name / "launch"
    d.mkdir(parents=True, exist_ok=True)
    log = d / f"{_stamp(started)}.log"
    log.write_text("".join(f"line {i}\n" for i in range(30)), encoding="utf-8")
    (d / f"{_stamp(started)}.pid.json").write_text(json.dumps({"pid": pid, "log": str(log), "started": started}),
                                                   encoding="utf-8")
    return log


@pytest.fixture
def alive(monkeypatch):
    """pid_alive answers from this set (no real process is inspected)."""
    pids = set()
    monkeypatch.setattr(L, "pid_alive", lambda pid: pid in pids)
    return pids


def prog(spec_path, root, **kw):
    return L.progress(spec_path, "runs", root=root, gpu=False, **kw)


# ------------------------------------------------------------------ readers
def test_readers_skip_a_partial_last_line(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"epoch": 0}\n{"epoch": 1}\n{"epoch": 2, "lo', encoding="utf-8")
    assert [r["epoch"] for r in L.read_jsonl(p)] == [0, 1]
    c = tmp_path / "t.csv"
    c.write_text("epoch,loss\n0,1.5\n1,1.4\n2,1.", encoding="utf-8")
    assert L.read_csv_rows(c) == [{"epoch": 0.0, "loss": 1.5}, {"epoch": 1.0, "loss": 1.4}]
    c.write_text("epoch,loss\n0,1.5\n1,1.4\n2\n", encoding="utf-8")
    assert len(L.read_csv_rows(c)) == 2
    assert L.read_jsonl(tmp_path / "missing") == [] and L.read_csv_rows(tmp_path / "missing") == []
    assert L.tail(tmp_path / "missing") == []


# ------------------------------------------------------------------ states
def test_not_started(tmp_path, spec, alive):
    p = prog(spec, tmp_path)
    assert p["state"] == "not started" and p["run_dir"] is None and p["epochs_done"] == 0
    assert L.progress_figure(p) is None and p["gpu"] is None


def test_running_with_eta_and_partial_lines(tmp_path, spec, alive):
    now = time.time()
    d = make_cell(tmp_path, spec, created=now - 600, epochs=5)
    p = prog(spec, tmp_path, now=now)
    assert p["state"] == "running" and p["run_dir"] == str(d)
    assert p["epochs_done"] == 5 and p["epochs_total"] == 20          # the partial 6th line is skipped
    assert p["median_epoch_s"] == 10.0
    assert p["eta_s"] == pytest.approx(15 * 10.0) and "estimate" in p["eta_note"]
    assert p["lr"] == pytest.approx(1e-3) and p["lr_indicator"] == pytest.approx(5e-3)
    assert p["best_val_loss"] == pytest.approx(6.6) and p["best_epoch"] == 5
    assert p["elapsed_s"] == pytest.approx(600, abs=2) and p["sec_per_step"] == pytest.approx(0.1)
    assert list(p["table"]["epoch"]) == [1, 2, 3, 4, 5] and "val CRPS h2" in p["table"].columns
    assert p["horizon_steps"] == [10, 15, 20]


def test_stalled_after_the_larger_of_three_epochs_and_30_minutes(tmp_path, spec, alive):
    now = time.time()
    make_cell(tmp_path, spec, created=now - 7200, epochs=3, mtime=now - 31 * 60)
    p = prog(spec, tmp_path, now=now)
    assert p["state"] == "stalled" and p["stall_after_s"] == 30 * 60
    assert L.stall_after(1000.0) == 3000.0 and L.stall_after(None) == 30 * 60
    # 29 minutes without an update is still running
    make_cell(tmp_path, spec, created=now - 7000, epochs=3, mtime=now - 29 * 60)
    assert prog(spec, tmp_path, now=now)["state"] == "running"


def test_done_and_failed_from_result_json(tmp_path, spec, alive):
    now = time.time()
    make_cell(tmp_path, spec, created=now - 5000, epochs=20, partial=False,
              result={"status": "done", "role": "dev", "wall_s": 1234.0, "scores": {}}, mtime=now - 4000)
    p = prog(spec, tmp_path, now=now)
    assert p["state"] == "done" and p["eta_s"] == 0.0 and p["elapsed_s"] == 1234.0
    make_cell(tmp_path, spec, created=now - 100, epochs=1,
              result={"status": "failed", "error": {"type": "ValueError", "message": "boom"}})
    p = prog(spec, tmp_path, now=now)
    assert p["state"] == "failed" and "boom" in p["reason"]


def test_a_dead_launched_process_without_result_is_failed(tmp_path, spec, alive):
    now = time.time()
    launch_record(tmp_path, spec, pid=4242, started=now - 700)
    make_cell(tmp_path, spec, created=now - 650, epochs=2)
    p = prog(spec, tmp_path, now=now)
    assert p["state"] == "failed" and "no result.json" in p["reason"]
    assert p["log_tail"] == [f"line {i}" for i in range(10, 30)]          # the last 20 lines
    alive.add(4242)
    assert prog(spec, tmp_path, now=now)["state"] == "running"


def test_launched_process_before_its_cell_exists(tmp_path, spec, alive):
    now = time.time()
    launch_record(tmp_path, spec, pid=77, started=now - 20)
    alive.add(77)
    p = prog(spec, tmp_path, now=now)
    assert p["state"] == "running" and p["run_dir"] is None and p["elapsed_s"] == pytest.approx(20, abs=2)
    alive.clear()
    assert prog(spec, tmp_path, now=now)["state"] == "failed"


def test_eta_arithmetic():
    assert L.eta_seconds(5, 20, 12.5) == 187.5
    assert L.eta_seconds(20, 20, 12.5) == 0 and L.eta_seconds(25, 20, 1.0) == 0
    assert L.eta_seconds(5, None, 1.0) is None and L.eta_seconds(5, 20, None) is None


# ------------------------------------------------------------------ figures
def test_progress_figure_has_no_empty_panel_and_uses_the_theme(tmp_path, spec, alive):
    from neural_trade.visualization import theme as T

    now = time.time()
    make_cell(tmp_path, spec, created=now - 600, epochs=4)
    p = prog(spec, tmp_path, now=now)
    fig = L.progress_figure(p)
    assert T.empty_panels(fig) == []
    crps = [t for t in fig.data if t.name and t.name.startswith("h")]
    assert {t.line.color for t in crps} == set(T.HORIZON_COLORS.values())
    assert crps[0].name == "h0 (10 bars)"
    dotted = [t for t in fig.data if getattr(t, "line", None) is not None and t.line.dash == T.TRAIN_DASH]
    assert {t.name for t in dotted} >= {"training loss", "h0 training"}
    assert any("ETA" in (t.name or "") for t in fig.data)
    dash, terms = L.training_figures(p)
    assert dash.layout.title.text.startswith("<b>Training dashboard")


def test_show_progress_prints_the_state_instead_of_an_empty_figure(tmp_path, spec, alive, capsys):
    pytest.importorskip("IPython")
    L.show_progress(prog(spec, tmp_path))
    out = capsys.readouterr().out
    assert "NOT STARTED" in out and "No epoch has finished yet" in out


# ------------------------------------------------------------------ GPU snapshot
def test_gpu_snapshot_parses_nvidia_smi_and_is_none_without_it(monkeypatch):
    class Done:
        stdout = "41, 5400, 12282, 180.52\n"

    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: Done())
    g = L.gpu_snapshot()
    assert (g["utilization_pct"], g["memory_used_mb"], g["memory_total_mb"], g["power_w"]) == (41, 5400, 12282, 180.52)

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(L.subprocess, "run", missing)
    assert L.gpu_snapshot() is None


def test_pid_alive_never_signals():
    assert L.pid_alive(os.getpid())
    assert not L.pid_alive(None) and not L.pid_alive(0)
    assert not L.pid_alive(2 ** 22 + 12345)


# ------------------------------------------------------------------ launch
class FakePopen:
    calls = []

    def __init__(self, cmd, **kwargs):
        FakePopen.calls.append((cmd, kwargs))
        self.pid = 31337


@pytest.fixture
def popen(monkeypatch):
    FakePopen.calls = []
    monkeypatch.setattr(L.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    return FakePopen


def test_launch_starts_a_detached_scenario_run(tmp_path, spec, alive, popen):
    (tmp_path / "src" / "neural_trade").mkdir(parents=True)
    r = L.launch(spec.name, "runs", root=tmp_path)
    assert r["launched"] and r["pid"] == 31337
    ((cmd, kw),) = popen.calls
    assert cmd == [sys.executable, "-m", "neural_trade.cli", "scenario", "run", str(spec.resolve()),
                   "--store", str((tmp_path / "runs").resolve())]
    assert r["command"] == cmd and kw["cwd"] == str(tmp_path.resolve())
    assert kw["stdin"] is subprocess.DEVNULL and kw["stderr"] is subprocess.STDOUT
    if sys.platform == "win32":
        assert kw["creationflags"] == 0x8 | 0x200            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        assert kw["start_new_session"] is True
    env = kw["env"]
    assert "CUDA_VISIBLE_DEVICES" not in env and env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str((tmp_path / "src").resolve())
    log = Path(r["log"])
    assert log.parent == tmp_path / "runs" / "scenarios" / "fake_long" / "launch" and log.suffix == ".log"
    rec = json.loads(Path(r["pid_file"]).read_text(encoding="utf-8"))
    assert rec["pid"] == 31337 and rec["log"] == str(log) and rec["command"] == cmd
    assert Path(r["pid_file"]).parent == log.parent


def test_launch_log_dir_and_retry_flag(tmp_path, spec, alive, popen):
    r = L.launch(spec, "runs", root=tmp_path, log_dir=tmp_path / "logs", retry_failed=True)
    assert Path(r["log"]).parent == tmp_path / "logs" and popen.calls[0][0][-1] == "--retry-failed"


def test_launch_refuses_while_a_cell_is_running(tmp_path, spec, alive, popen):
    now = time.time()
    make_cell(tmp_path, spec, created=now - 600, epochs=2)
    r = L.launch(spec, "runs", root=tmp_path)
    assert not r["launched"] and "running" in r["reason"] and not popen.calls


def test_launch_refuses_while_the_last_launch_is_alive(tmp_path, spec, alive, popen):
    launch_record(tmp_path, spec, pid=555, started=time.time() - 5)
    alive.add(555)
    r = L.launch(spec, "runs", root=tmp_path)
    assert not r["launched"] and "alive" in r["reason"] and not popen.calls


def test_launch_refuses_a_done_cell_but_not_one_of_another_config(tmp_path, spec, alive, popen):
    now = time.time()
    make_cell(tmp_path, spec, created=now - 9000, epochs=1, mtime=now - 8000, config_hash="other",
              result={"status": "done"})
    assert L.launch(spec, "runs", root=tmp_path)["launched"]                 # another config: trains anew
    popen.calls.clear()
    make_cell(tmp_path, spec, created=now - 5000, epochs=1, mtime=now - 4000, result={"status": "done"})
    r = L.launch(spec, "runs", root=tmp_path)
    assert not r["launched"] and r["reason"].startswith("already done") and not popen.calls


def test_launch_refuses_a_failed_cell_unless_retry(tmp_path, spec, alive, popen):
    now = time.time()
    make_cell(tmp_path, spec, created=now - 5000, epochs=1, mtime=now - 4000, result={"status": "failed"})
    r = L.launch(spec, "runs", root=tmp_path)
    assert not r["launched"] and "retry_failed" in r["reason"] and not popen.calls
    assert L.launch(spec, "runs", root=tmp_path, retry_failed=True)["launched"]


def test_an_old_interrupted_cell_does_not_block_a_launch(tmp_path, spec, alive, popen):
    now = time.time()
    make_cell(tmp_path, spec, created=now - 9000, epochs=2, mtime=now - 16 * 60)
    assert L.launch(spec, "runs", root=tmp_path)["launched"]


# ------------------------------------------------------------------ results
def test_results_of_a_scored_cell(tmp_path, spec, alive):
    assert L.results(spec, "runs", root=tmp_path)["available"] is False
    now = time.time()
    scores = {"h0/direction/auc": 0.51, "h1/direction/auc": 0.52, "h0/variance/crps": 100.0,
              "backtest/sharpe_net": -1.5, "backtest/buy_and_hold/sharpe_net": 2.0,
              "backtest/random_same_freq/percentile_sharpe_net": 40.0}
    d = make_cell(tmp_path, spec, created=now - 5000, epochs=1,
                  result={"status": "done", "role": "dev", "wall_s": 99.0, "scores": scores})
    (d / "eval_report_dev.md").write_text("# Evaluation report - dev split\n", encoding="utf-8")
    r = L.results(spec, "runs", root=tmp_path)
    assert r["available"] and r["role"] == "dev" and r["report_md"].startswith("# Evaluation report")
    assert list(r["horizons"].columns) == ["h0", "h1"] and r["horizons"].loc["direction AUC", "h1"] == 0.52
    assert r["backtest"].loc["net Sharpe", "strategy"] == -1.5
    assert r["backtest"].loc["net Sharpe", "buy and hold"] == 2.0
    assert r["random_null"] == {"percentile_sharpe_net": 40.0}

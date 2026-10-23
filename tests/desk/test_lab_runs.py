"""DESK-5.3: Lab runs are a faithful, tolerant, read-only view of the run store (D-067, D-020, D-065)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from neural_trade.desk.lab import runs as lab_runs
from neural_trade.desk.lab.runs import LabRuns, RunNotFound
from neural_trade.notebook import longrun

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "nt038_healthy"
NOW = 1_800_000_000.0
SCEN = "lab_scen"
EVAL_DEV = {"model": {"horizons": {"h0": {"direction": {"auc": 0.51}}}}, "backtest": {"summary": {"sharpe_net": 1.0}}}
EVAL_TEST = {"model": {"horizons": {"h0": {"direction": {"auc": 0.49}}}}}


def _touch_all(d: Path, t: float) -> None:
    for p in d.iterdir():
        os.utime(p, (t, t))


def make_run(store: Path, name: str, src: str = "reference_default_f3", *, role: str = "dev", result: bool = True,
             age_s: float = 10.0, sharpe: float | None = None, created: str = "20260101T000000Z",
             eval_reports: bool = True) -> Path:
    d = store / "scenarios" / SCEN / name
    shutil.copytree(FIX / src, d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta.update(run_id=name, created_utc=created)
    meta["engine"].update(scenario=SCEN, role=role, cell_key=name)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if result:
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        res.update(run_id=name, scenario=SCEN, cell_key=name, role=role)
        if sharpe is not None:
            res["scores"]["backtest/sharpe_net"] = sharpe
        (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    else:
        (d / "result.json").unlink()
    if eval_reports:
        (d / "eval_report_dev.json").write_text(json.dumps(EVAL_DEV), encoding="utf-8")
        (d / "eval_report_test.json").write_text(json.dumps(EVAL_TEST), encoding="utf-8")
    _touch_all(d, NOW - age_s)
    return d


@pytest.fixture
def store(tmp_path) -> Path:
    from neural_trade.experiments.store import RunStore

    root = tmp_path / "store"
    make_run(root, "r_a", sharpe=0.5, created="20260101T000000Z")
    make_run(root, "r_b", src="capacity_gru_f96", sharpe=2.0, created="20260102T000000Z")
    make_run(root, "r_t", role="test", sharpe=9.0, created="20260103T000000Z")
    RunStore(root).sync(SCEN)
    return root


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def lab(store) -> LabRuns:
    return LabRuns(store, clock=lambda: NOW)


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- criterion 1: detail equals the files
def test_detail_equals_files(store):
    d = store / "scenarios" / SCEN / "r_a"
    det = lab(store).detail("r_a")
    assert det["meta"] == _json(d / "meta.json")
    assert det["status"] == _json(d / "status.json")
    assert det["result"] == _json(d / "result.json")
    assert det["eval_report"] == {"dev": EVAL_DEV, "test": EVAL_TEST}
    assert set(det["eval_report"]["dev"]) == set(_json(d / "eval_report_dev.json"))
    rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert det["metrics_tail"] == rows[-longrun.LOG_TAIL_LINES:]
    assert det["metrics_tail"][-1]["epoch"] == rows[-1]["epoch"]
    assert det["state"] == "done" and det["scenario"] == SCEN
    assert det["scores"]["backtest/sharpe_net"] == 0.5


def test_missing_run_id_raises_clear_keyerror(store):
    for call in (lambda: lab(store).detail("nope"), lambda: lab(store).progress("nope")):
        with pytest.raises(KeyError, match="no run 'nope'"):
            call()
    assert issubclass(RunNotFound, KeyError)


# ---------------------------------------------------------------- criterion 2: progress states
def _state(store, name) -> dict:
    from neural_trade.experiments.store import RunStore

    RunStore(store).sync(SCEN)
    return lab(store).progress(name)


def test_state_done_and_failed(store):
    assert lab(store).progress("r_a")["state"] == "done"
    d = store / "scenarios" / SCEN / "r_a"
    res = _json(d / "result.json")
    res.update(status="failed", error={"type": "ValueError", "message": "boom"})
    (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    p = lab(store).progress("r_a")
    assert p["state"] == "failed" and "boom" in p["reason"]


def _meta_only(store, name, age_s):
    d = make_run(store, name, result=False, age_s=age_s)
    for n in ("status.json", "metrics.jsonl", "training_log.csv", "eval_report_dev.json", "eval_report_test.json"):
        (d / n).unlink(missing_ok=True)
    _touch_all(d, NOW - age_s)
    return d


def test_state_not_started_only_without_any_file(store):
    d = make_run(store, "r_new", result=False)
    for f in list(d.iterdir()):
        f.unlink()
    assert lab(store)._progress_of(d, {"run_id": "r_new"})["state"] == "not started"


def test_meta_only_cell_follows_longrun_rules(store, monkeypatch):
    stall = longrun.STALL_MIN_S
    alive = {"v": False}
    monkeypatch.setattr(longrun, "pid_alive", lambda pid: alive["v"])
    _meta_only(store, "m_fresh", 60)                      # fresh, no launch record: running
    assert _state(store, "m_fresh")["state"] == "running"
    _meta_only(store, "m_old", stall + 600)               # old, no launch record: stalled
    assert _state(store, "m_old")["state"] == "stalled"
    _launch(store, 4242, longrun._created(store / "scenarios" / SCEN / "m_old"))
    assert lab(store).progress("m_old")["state"] == "failed"     # old, launched, dead pid
    alive["v"] = True
    assert lab(store).progress("m_old")["state"] == "stalled"    # old, live pid
    _meta_only(store, "m_live", 30)
    _launch(store, 4242, longrun._created(store / "scenarios" / SCEN / "m_live"))
    assert _state(store, "m_live")["state"] == "running"         # fresh, live pid


def test_state_running_stalled_by_mtime(store):
    make_run(store, "r_run", result=False, age_s=60)
    assert _state(store, "r_run")["state"] == "running"
    stall = longrun.stall_after(longrun._median(r.get("epoch_seconds") for r in
                                                longrun._epoch_rows(store / "scenarios" / SCEN / "r_run")))
    make_run(store, "r_old", result=False, age_s=stall + 120)
    p = _state(store, "r_old")
    assert p["state"] == "stalled" and p["stall_after_s"] == stall
    # the threshold is longrun's: just inside it is still running
    make_run(store, "r_edge", result=False, age_s=stall - 120)
    assert _state(store, "r_edge")["state"] == "running"


def _launch(store: Path, pid: int, started: float) -> None:
    ld = store / "scenarios" / SCEN / longrun.LAUNCH_DIR
    ld.mkdir(exist_ok=True)
    (ld / "a.pid.json").write_text(json.dumps({"pid": pid, "started": started, "log": str(ld / "a.log")}),
                                   encoding="utf-8")


def test_state_with_pid_liveness(store, monkeypatch):
    make_run(store, "r_p", result=False, age_s=40 * 3600)
    created = longrun._created(store / "scenarios" / SCEN / "r_p")
    _launch(store, 4242, created)
    monkeypatch.setattr(longrun, "pid_alive", lambda pid: True)
    p = _state(store, "r_p")
    assert p["state"] == "stalled" and p["pid_alive"] and "alive" in p["reason"]
    monkeypatch.setattr(longrun, "pid_alive", lambda pid: False)
    p = _state(store, "r_p")
    assert p["state"] == "failed" and not p["pid_alive"]
    # a fresh run whose process is alive is running
    for f in (store / "scenarios" / SCEN / "r_p").iterdir():
        os.utime(f, (NOW - 5, NOW - 5))
    monkeypatch.setattr(longrun, "pid_alive", lambda pid: True)
    assert lab(store).progress("r_p")["state"] == "running"


# ---------------------------------------------------------------- list, D-020, pagination
def test_list_items_and_dev_test_separation(store):
    out = lab(store).list_runs(SCEN)
    by = {i["run_id"]: i for i in out["items"]}
    assert out["total"] == 3 and set(by) == {"r_a", "r_b", "r_t"}
    a, t = by["r_a"], by["r_t"]
    assert a["scenario"] == SCEN and a["state"] == "done" and a["created_utc"] == "20260101T000000Z"
    assert a["key_numbers"]["split"] == "dev" and a["key_numbers"]["rank_eligible"] is True
    assert a["dev_sharpe_net"] == 0.5 and a["test_sharpe_net"] is None
    assert t["key_numbers"]["split"] == "test" and t["key_numbers"]["rank_eligible"] is False
    assert t["test_sharpe_net"] == 9.0 and t["dev_sharpe_net"] is None and t["rank_eligible"] is False


@pytest.mark.parametrize("descending", [True, False])
def test_test_numbers_never_rank(store, descending):
    ids = [i["run_id"] for i in lab(store).list_runs(SCEN, sort="dev_sharpe_net", descending=descending)["items"]]
    assert ids[-1] == "r_t"                        # the test run (sharpe 9.0) sorts last, never first
    assert ids[:2] == (["r_b", "r_a"] if descending else ["r_a", "r_b"])
    for bad in ("test_sharpe_net", "sharpe_net", "total_return", "test_total_return"):
        with pytest.raises(ValueError, match="test columns never rank"):
            lab(store).list_runs(SCEN, sort=bad)


def test_pagination_and_sort_by_created(store):
    full = [i["run_id"] for i in lab(store).list_runs(SCEN)["items"]]
    assert full == ["r_t", "r_b", "r_a"]
    p1 = lab(store).list_runs(SCEN, limit=2)
    p2 = lab(store).list_runs(SCEN, limit=2, offset=2)
    assert [i["run_id"] for i in p1["items"] + p2["items"]] == full
    assert p1["total"] == p2["total"] == 3 and len(p2["items"]) == 1
    assert lab(store).list_runs("other")["total"] == 0
    assert lab(store).list_runs(SCEN, offset=99)["items"] == []


# ---------------------------------------------------------------- tolerance and read-only
def test_half_written_json_is_tolerated(store):
    d = store / "scenarios" / SCEN / "r_a"
    (d / "status.json").write_text('{"epochs_comp', encoding="utf-8")
    (d / "result.json").write_text('{"status": "do', encoding="utf-8")
    (d / "eval_report_dev.json").write_text("[1, 2", encoding="utf-8")
    with open(d / "metrics.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"epoch": 99, "val_lo')
    det = lab(store).detail("r_a")
    assert det["status"] is None and det["result"] is None
    assert "dev" not in det["eval_report"] and "test" in det["eval_report"]
    assert det["metrics_tail"][-1]["epoch"] == 19
    assert lab(store).list_runs(SCEN)["total"] == 3
    assert lab(store).progress("r_a")["state"] in lab_runs.STATES


@pytest.mark.parametrize("name", ["meta.json", "status.json", "result.json", "eval_report_dev.json"])
@pytest.mark.parametrize("indexed", [True, False])
def test_half_written_file_keeps_the_run_listed(store, name, indexed):
    if not indexed:
        (store / "index.sqlite").unlink()
    d = store / "scenarios" / SCEN / "r_b"
    (d / name).write_text('{"half', encoding="utf-8")
    lr = lab(store)
    assert "r_b" in {i["run_id"] for i in lr.list_runs(SCEN)["items"]}
    det = lr.detail("r_b")
    assert any(e.startswith(name) for e in det["errors"])
    if name == "meta.json":
        assert det["state"] == "unreadable" and det["meta"] is None
        assert det["status"] == _json(d / "status.json")        # the readable parts are still served
    else:
        assert det["state"] != "unreadable"


def test_limit_is_capped_and_descending_none(store):
    assert lab(store).list_runs(SCEN, limit=10_000)["limit"] == lab_runs.MAX_LIMIT == 500
    assert lab(store).list_runs(SCEN, limit=0)["limit"] == 1
    assert lab(store).list_runs(SCEN, descending=None) == lab(store).list_runs(SCEN, descending=True)


def test_unindexed_runs_are_listed_and_flagged(store):
    from neural_trade.experiments.store import RunStore

    make_run(store, "r_late", created="20260104T000000Z")          # never synced into the index
    out = {i["run_id"]: i for i in lab(store).list_runs(SCEN)["items"]}
    assert out["r_late"]["indexed"] is False and out["r_a"]["indexed"] is True and len(out) == 4
    assert lab(store).detail("r_late")["indexed"] is False
    only = lab(store).list_runs(SCEN, include_unindexed=False)
    assert only["total"] == 3 and "r_late" not in {i["run_id"] for i in only["items"]}
    RunStore(store).sync(SCEN)
    assert lab(store).list_runs(SCEN)["items"][0]["indexed"] is True


def test_missing_index_falls_back_to_files(store):
    (store / "index.sqlite").unlink()
    assert {i["run_id"] for i in lab(store).list_runs(SCEN)["items"]} == {"r_a", "r_b", "r_t"}
    assert lab(store).detail("r_b")["state"] == "done"


def test_no_file_is_written(store):
    before = tree_hash(store)
    lr = lab(store)
    lr.list_runs(SCEN)
    lr.list_runs(SCEN, sort="dev_sharpe_net", limit=1, offset=1)
    for rid in ("r_a", "r_b", "r_t"):
        lr.detail(rid)
        lr.progress(rid)
    with pytest.raises(KeyError):
        lr.detail("nope")
    assert tree_hash(store) == before

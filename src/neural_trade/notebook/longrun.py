"""Launch and track one long experiment-engine run from a notebook (NT-082, notebook 08, D-040).

    from neural_trade.notebook.longrun import launch, progress, show_progress, results
    launch("configs/scenarios/long_360d.yaml", "runs", root="..")     # detached; returns pid, log, command
    show_progress(progress("configs/scenarios/long_360d.yaml", "runs", root=".."))

* :func:`launch` starts ``python -m neural_trade.cli scenario run <spec> --store <store>`` as a
  detached background process that survives the notebook kernel and the editor closing, with its
  stdout and stderr in ``<store>/scenarios/<name>/launch/<UTC>.log`` and a pid file next to it. It
  refuses (and starts nothing) while a cell of the scenario is running or once the cell is done.
* :func:`progress` reads the newest cell directory of the scenario tolerantly (files being written:
  a partial last line is skipped) and returns its state, epochs, timing, an ETA estimate, the
  learning rate, the best validation loss, the per-epoch table, the launch log's tail and a GPU
  snapshot from ``nvidia-smi``.
* :func:`progress_figure` draws the run's progress (losses, learning rates, epoch time, step time,
  elapsed time with the ETA, validation CRPS per horizon); :func:`show_progress` shows it with the
  table, the per-epoch training dashboard of notebook 01 and the log tail.
* :func:`results` reads the scored cell (``result.json``, ``eval_report_<role>.md``) once it exists.

Paths: ``spec`` and ``store`` are relative to ``root`` (the repository root; the notebook passes
".." because it runs in ``notebooks/``), and the launched process runs in ``root``, where the
spec's relative ``CSV_PATH`` resolves. Nothing here imports TensorFlow.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

RUNNING_WINDOW_S = 15 * 60      # a cell without result.json updated this recently counts as running
STALL_MIN_S = 30 * 60           # stalled: no update for max(3 x median epoch, this)
STALL_EPOCHS = 3
LOG_TAIL_LINES = 20
LAUNCH_DIR = "launch"
PROGRESS_FILES = ("status.json", "metrics.jsonl", "training_log.csv", "meta.json")
NVIDIA_SMI = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
              "--format=csv,noheader,nounits"]
ETA_NOTE = ("an estimate: remaining epochs x the median epoch time; early stopping (EARLY) can end the run "
            "sooner, and scoring after the last epoch adds a few minutes")

# Windows process-creation flags (subprocess has the names on Windows only).
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

STATES = ("not started", "running", "stalled", "done", "failed")


# ------------------------------------------------------------------ paths and the scenario
def _resolve(path, root) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def _scenario(spec, root):
    from neural_trade.experiments.scenario import Scenario

    return Scenario.from_yaml(_resolve(spec, root))


def _scenario_path(store_path: Path, name: str) -> Path:
    from neural_trade.experiments.store import ENGINE_SUBTREE

    return Path(store_path) / ENGINE_SUBTREE / name


def scenario_dir(spec, store="runs", *, root=".") -> Path:
    """``<store>/scenarios/<scenario name>`` for the spec."""
    return _scenario_path(_resolve(store, root), _scenario(spec, root).name)


def cell_dirs(scen_dir) -> List[Path]:
    """The scenario's cell (run) directories, oldest first (run ids start with their UTC stamp)."""
    from neural_trade.experiments.store import is_engine_run_dir

    d = Path(scen_dir)
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_dir() and is_engine_run_dir(p)), key=lambda p: p.name)


# ------------------------------------------------------------------ tolerant readers
def _read_json(path) -> Optional[dict]:
    """A JSON file, or None when it is missing or half-written."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_jsonl(path) -> List[dict]:
    """The complete JSON lines of a file being appended to: a partial or malformed line is skipped."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def read_csv_rows(path) -> List[dict]:
    """The complete rows of a CSV being appended to (Keras CSVLogger): a row whose field count differs
    from the header's (a partial last line) is skipped; numbers are floats where they parse."""
    import csv

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if not lines:
        return []
    rows = list(csv.reader(lines))
    header, out = rows[0], []
    for i, row in enumerate(rows[1:], start=1):
        if len(row) != len(header):
            continue
        if i == len(rows) - 1 and not text.endswith(("\n", "\r")):
            continue            # the last line has no newline yet: it may still be growing
        rec = {}
        for k, v in zip(header, row):
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
        out.append(rec)
    return out


def tail(path, n: int = LOG_TAIL_LINES) -> List[str]:
    """The last ``n`` lines of a text file (empty when missing)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            data = fh.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def _epoch_rows(run_dir: Path) -> List[dict]:
    """One dict per finished epoch: metrics.jsonl, else training_log.csv (0-based ``epoch``)."""
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        rows = read_csv_rows(run_dir / "training_log.csv")
    by_epoch: Dict[int, dict] = {}
    for i, r in enumerate(rows):
        try:
            ep = int(r.get("epoch", i))
        except (TypeError, ValueError):
            ep = i
        by_epoch[ep] = {**r, "epoch": ep}
    return [by_epoch[k] for k in sorted(by_epoch)]


def _last_update(run_dir: Path) -> Optional[float]:
    """The newest modification time of the run's progress files (status.json first of all)."""
    times = []
    for name in PROGRESS_FILES:
        try:
            times.append((run_dir / name).stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def _created(run_dir: Path) -> Optional[float]:
    meta = _read_json(run_dir / "meta.json") or {}
    stamp = meta.get("created_utc")
    if isinstance(stamp, str):
        try:
            return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    try:
        return (run_dir / "meta.json").stat().st_mtime
    except OSError:
        return None


def _config_value(run_dir: Path, name: str):
    try:
        import yaml

        data = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, ImportError):
        return None
    except Exception:       # yaml.YAMLError on a half-written file
        return None
    return data.get(name) if isinstance(data, dict) else None


def _num(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ------------------------------------------------------------------ processes
def pid_alive(pid: Optional[int]) -> bool:
    """Whether process ``pid`` is running. Never signals it (``os.kill(pid, 0)`` terminates a process
    on Windows)."""
    if not pid:
        return False
    pid = int(pid)
    try:
        import psutil  # optional

        return bool(psutil.pid_exists(pid)) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259                 # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch_records(scen_dir) -> List[dict]:
    """The pid files under ``<scenario>/launch/``, oldest first, each with its ``path``."""
    d = Path(scen_dir) / LAUNCH_DIR
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.pid.json")):
        rec = _read_json(p)
        if rec is not None:
            out.append({**rec, "path": str(p)})
    return out


def _latest_launch(scen_dir) -> Optional[dict]:
    recs = launch_records(scen_dir)
    return recs[-1] if recs else None


def gpu_snapshot(*, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Utilisation, memory and power of each GPU from nvidia-smi; None when nvidia-smi is missing or fails."""
    try:
        out = subprocess.run(NVIDIA_SMI, capture_output=True, text=True, timeout=timeout, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        util, used, total, power = (_num(p) for p in parts)
        gpus.append({"utilization_pct": util, "memory_used_mb": used, "memory_total_mb": total, "power_w": power})
    if not gpus:
        return None
    return {**gpus[0], "gpus": gpus, "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}


# ------------------------------------------------------------------ state
def _finished(run_dir: Path) -> Optional[dict]:
    from neural_trade.experiments.store import RESULT_FILE

    return _read_json(run_dir / RESULT_FILE)


def _running_cells(dirs: List[Path], now: float) -> List[Path]:
    out = []
    for d in dirs:
        if _finished(d) is not None:
            continue
        t = _last_update(d)
        if t is not None and now - t <= RUNNING_WINDOW_S:
            out.append(d)
    return out


def _cells_of_spec(sc, dirs: List[Path]) -> Dict[str, List[Tuple[Path, dict]]]:
    """Finished cell directories of the spec as it is now (same config hash and settings hash, the
    runner's resume rule), grouped by status."""
    from neural_trade.experiments.scenario import config_hash
    from neural_trade.experiments.store import engine_meta

    expected = {cell.key: config_hash(cfg) for cell, cfg in sc.validate()}
    out: Dict[str, List[Tuple[Path, dict]]] = {"done": [], "failed": []}
    for d in dirs:
        eng = engine_meta(d) or {}
        res = _finished(d)
        if res is None or eng.get("cell_key") not in expected:
            continue
        if eng.get("config_hash") != expected[eng["cell_key"]] or eng.get("settings_hash") != sc.settings_hash:
            continue
        if res.get("status") in out:
            out[res["status"]].append((d, res))
    return out


def launch_command(spec_path: Path, store_path: Path, *, retry_failed: bool = False,
                   python: Optional[str] = None) -> List[str]:
    cmd = [python or sys.executable, "-m", "neural_trade.cli", "scenario", "run", str(spec_path),
           "--store", str(store_path)]
    if retry_failed:
        cmd.append("--retry-failed")
    return cmd


def launch_env(root: Path, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The launched process's environment: this one without CUDA_VISIBLE_DEVICES (it trains on the GPU),
    PYTHONIOENCODING=utf-8, and ``<root>/src`` first on PYTHONPATH when the root is a checkout."""
    env = dict(os.environ if base is None else base)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONIOENCODING"] = "utf-8"
    src = Path(root).resolve() / "src"
    if (src / "neural_trade").is_dir():
        env["PYTHONPATH"] = os.pathsep.join([str(src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def launch(spec, store="runs", *, root=".", log_dir=None, retry_failed: bool = False,
           python: Optional[str] = None) -> Dict[str, Any]:
    """Start ``scenario run`` for ``spec`` as a detached background process, unless it must not start.

    Refused (``launched`` False, ``reason`` says why, nothing starts) when a cell of the scenario is
    running (a cell directory without result.json whose progress files changed in the last 15
    minutes, or a live process in the newest pid file) or when every cell of the spec is already
    finished (done; failed too unless ``retry_failed``). Otherwise the process runs in ``root`` with
    this Python (``python`` overrides it), stdout and stderr go to ``<log_dir>/<UTC>.log`` (default
    ``<store>/scenarios/<name>/launch/``) and ``<UTC>.pid.json`` next to it records pid, command and
    log. Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, so it survives the kernel and VS Code
    closing; elsewhere a new session. Stop it with ``stop(pid)`` or ``taskkill /PID <pid> /T /F``.
    """
    root = Path(root)
    spec_path = _resolve(spec, root).resolve()
    store_path = _resolve(store, root).resolve()
    sc = _scenario(spec_path, root)
    scen_dir = _scenario_path(store_path, sc.name)
    now = time.time()
    dirs = cell_dirs(scen_dir)
    base = {"launched": False, "scenario": sc.name, "spec": str(spec_path), "store": str(store_path)}

    running = _running_cells(dirs, now)
    if running:
        return {**base, "reason": f"a cell is running: {running[-1].name} (updated in the last "
                                  f"{RUNNING_WINDOW_S // 60} minutes, no result.json)", "run_dir": str(running[-1])}
    last = _latest_launch(scen_dir)
    if last and pid_alive(last.get("pid")):
        return {**base, "reason": f"the process of the last launch is alive (pid {last.get('pid')}, "
                                  f"{last.get('path')})", "pid": last.get("pid"), "log": last.get("log")}
    finished = _cells_of_spec(sc, dirs)
    n_cells = len(sc.cells())
    done, failed = finished["done"], finished["failed"]
    done_keys = {engine_key(d) for d, _ in done}
    failed_keys = {engine_key(d) for d, _ in failed} - done_keys
    if len(done_keys) >= n_cells:
        return {**base, "reason": f"already done: {', '.join(d.name for d, _ in done)}",
                "run_dir": str(done[-1][0])}
    if not retry_failed and len(done_keys | failed_keys) >= n_cells:
        return {**base, "reason": f"the cell failed: {', '.join(d.name for d, _ in failed)} (see its result.json); "
                                  "launch(..., retry_failed=True) trains it again", "run_dir": str(failed[-1][0])}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(log_dir) if log_dir is not None else scen_dir / LAUNCH_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path, pid_path = log_dir / f"{stamp}.log", log_dir / f"{stamp}.pid.json"
    cmd = launch_command(spec_path, store_path, retry_failed=retry_failed, python=python)
    kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(root.resolve()), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=launch_env(root), close_fds=True, **kwargs)
    record = {"pid": proc.pid, "command": cmd, "cwd": str(root.resolve()), "log": str(log_path),
              "started_utc": stamp, "started": now}
    pid_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    _log.info("launched scenario %s: pid %s, log %s", sc.name, proc.pid, log_path)
    return {**base, "launched": True, "pid": proc.pid, "log": str(log_path), "pid_file": str(pid_path),
            "command": cmd}


def engine_key(run_dir: Path) -> Optional[str]:
    from neural_trade.experiments.store import engine_meta

    return (engine_meta(run_dir) or {}).get("cell_key")


def stop(pid: int) -> bool:
    """Stop a launched run and its children (Windows: taskkill /T /F). The cell stays without result.json
    (``incomplete``); launching again trains it anew."""
    if not pid_alive(pid):
        return False
    if sys.platform == "win32":
        return subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True).returncode == 0
    import signal

    os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
    return True


# ------------------------------------------------------------------ progress
TABLE_COLUMNS = (
    ("epoch", "epoch"), ("time_utc", "finished (UTC)"), ("epoch_seconds", "epoch s"), ("sec_per_step", "s / step"),
    ("loss", "train loss"), ("val_loss", "val loss"), ("lr", "LR"), ("lr_indicator", "indicator LR"),
    ("val_crps_h0", "val CRPS h0"), ("val_crps_h1", "val CRPS h1"), ("val_crps_h2", "val CRPS h2"),
    ("val_nll_h0", "val NLL h0"), ("val_nll_h1", "val NLL h1"), ("val_nll_h2", "val NLL h2"),
    ("val_dir_bal_acc_h0", "val bal. acc h0"), ("val_dir_bal_acc_h1", "val bal. acc h1"),
    ("val_dir_bal_acc_h2", "val bal. acc h2"),
    ("val_dir_mcc_h0", "val MCC h0"), ("val_dir_mcc_h1", "val MCC h1"), ("val_dir_mcc_h2", "val MCC h2"),
)


def epoch_table(rows: List[dict]):
    """The per-epoch table (1-based epochs): timing, losses, learning rates and the headline
    validation metrics per horizon; columns a run did not log are left out."""
    import pandas as pd

    recs = []
    for r in rows:
        rec = {}
        for key, label in TABLE_COLUMNS:
            if key == "epoch":
                rec[label] = int(r["epoch"]) + 1
            elif key == "time_utc":
                t = _num(r.get("time"))
                rec[label] = (datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M") if t else None)
            elif key in r:
                rec[label] = _num(r.get(key))
        recs.append(rec)
    df = pd.DataFrame(recs)
    return df.dropna(axis=1, how="all") if len(df) else df


def styled_epoch_table(df):
    """The epoch table for display: learning rates in scientific notation, times with one decimal, the
    rest with four; missing values as n/a."""
    fmt = {}
    for c in df.columns:
        if c in ("epoch", "finished (UTC)"):
            continue
        fmt[c] = "{:.2e}" if "LR" in c else ("{:.1f}" if c == "epoch s" else "{:.4f}")
    return df.style.format(fmt, na_rep="n/a").hide(axis="index")


def _median(values) -> Optional[float]:
    v = sorted(x for x in (_num(x) for x in values) if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def eta_seconds(epochs_done: int, epochs_total: Optional[int], median_epoch_s: Optional[float]) -> Optional[float]:
    """Remaining epochs x the median epoch time (an estimate: early stopping can end the run sooner)."""
    if epochs_total is None or median_epoch_s is None:
        return None
    return max(0, int(epochs_total) - int(epochs_done)) * float(median_epoch_s)


def stall_after(median_epoch_s: Optional[float]) -> float:
    """Seconds without an update after which a running cell counts as stalled."""
    return max(STALL_EPOCHS * median_epoch_s, STALL_MIN_S) if median_epoch_s else STALL_MIN_S


def progress(spec, store="runs", *, root=".", now: Optional[float] = None, gpu: bool = True) -> Dict[str, Any]:
    """The state of the scenario's newest cell (see the module docstring); every number is None when
    it is not known yet. ``state``: not started / running / stalled / done / failed."""
    root = Path(root)
    sc = _scenario(spec, root)
    scen_dir = _scenario_path(_resolve(store, root), sc.name)
    now = time.time() if now is None else float(now)
    dirs = cell_dirs(scen_dir)
    last = _latest_launch(scen_dir)
    alive = bool(last) and pid_alive(last.get("pid"))
    out: Dict[str, Any] = {
        "scenario": sc.name, "state": "not started", "reason": "", "run_dir": None, "epochs_done": 0,
        "epochs_total": _spec_epochs(sc), "elapsed_s": None, "median_epoch_s": None, "sec_per_step": None, "eta_s": None,
        "eta_note": ETA_NOTE, "last_update_s": None, "stall_after_s": None, "lr": None, "lr_indicator": None,
        "best_val_loss": None, "best_epoch": None, "rows": [], "table": epoch_table([]),
        "pid": last.get("pid") if last else None, "pid_alive": alive, "log": last.get("log") if last else None,
        "log_tail": tail(last["log"]) if last and last.get("log") else [], "result": None, "horizon_steps": None,
        "gpu": gpu_snapshot() if gpu else None,
    }
    if not dirs:
        if alive:
            out.update(state="running", reason="the launched process is starting (loading the data, planning); "
                                               "no cell directory yet")
            out["elapsed_s"] = now - float(last.get("started", now))
        elif last:
            out.update(state="failed", reason="the launched process ended without creating a cell directory: "
                                              "see the log tail")
        else:
            out["reason"] = "no cell directory and no launch record: run launch() or `neural-trade scenario run`"
        return out

    run_dir = dirs[-1]
    out["run_dir"] = str(run_dir)
    rows = _epoch_rows(run_dir)
    status = _read_json(run_dir / "status.json") or {}
    result = _finished(run_dir)
    total = _config_value(run_dir, "EPOCHS")
    steps = _config_value(run_dir, "HORIZON_STEPS")
    done = max(len(rows), int(status.get("epochs_completed") or 0))
    med = _median(r.get("epoch_seconds") for r in rows)
    created = _created(run_dir)
    updated = _last_update(run_dir)
    out.update(rows=rows, table=epoch_table(rows), epochs_done=done,
               epochs_total=int(total) if isinstance(total, (int, float)) else _spec_epochs(sc), median_epoch_s=med,
               sec_per_step=_num(status.get("sec_per_step")) if status.get("sec_per_step") is not None else
               (_num(rows[-1].get("sec_per_step")) if rows else None),
               last_update_s=(now - updated) if updated is not None else None, stall_after_s=stall_after(med),
               result=result, horizon_steps=list(steps) if isinstance(steps, list) else None)
    if rows:
        out["lr"], out["lr_indicator"] = _num(rows[-1].get("lr")), _num(rows[-1].get("lr_indicator"))
        vals = [(_num(r.get("val_loss")), int(r["epoch"])) for r in rows]
        vals = [v for v in vals if v[0] is not None]
        if vals:
            best = min(vals)
            out["best_val_loss"], out["best_epoch"] = best[0], best[1] + 1
    if result is not None:
        status_word = result.get("status")
        out["state"] = "done" if status_word == "done" else "failed"
        err = (result.get("error") or {})
        out["reason"] = ("scored: result.json is written" if out["state"] == "done" else
                         f"result.json says {status_word}: {err.get('type', '')} {err.get('message', '')}".strip())
        out["elapsed_s"] = _num(result.get("wall_s"))
        out["eta_s"] = 0.0
        return out

    out["elapsed_s"] = (now - created) if created is not None else _num(status.get("elapsed_seconds"))
    out["eta_s"] = eta_seconds(done, out["epochs_total"], med)
    launched_this = bool(last) and created is not None and float(last.get("started", 0)) <= created + 60
    if last and launched_this and not alive:
        out.update(state="failed", reason="the launched process has ended but the cell has no result.json "
                                          "(interrupted or crashed): see the log tail; launching again trains "
                                          "the cell anew")
    elif alive or (out["last_update_s"] is not None and out["last_update_s"] <= out["stall_after_s"]):
        out.update(state="running", reason=f"epoch {done + 1} in progress" if out["epochs_total"] is None or
                   done < out["epochs_total"] else "training finished; scoring the out-of-sample block")
        if alive and out["last_update_s"] is not None and out["last_update_s"] > out["stall_after_s"]:
            out.update(state="stalled", reason=f"the process is alive but nothing was written for "
                                               f"{out['last_update_s'] / 60:.0f} min (limit "
                                               f"{out['stall_after_s'] / 60:.0f} min)")
    else:
        out.update(state="stalled", reason=f"no update for {(out['last_update_s'] or 0) / 60:.0f} min (limit "
                                           f"{out['stall_after_s'] / 60:.0f} min: {STALL_EPOCHS} x the median "
                                           f"epoch or {STALL_MIN_S // 60} min) and no live process recorded")
    return out


def _spec_epochs(sc) -> Optional[int]:
    """EPOCHS of the spec's (last) cell, before its run directory exists."""
    try:
        cells = sc.validate()
    except Exception:
        return None
    return int(cells[-1][1].EPOCHS) if cells else None


def _hms(s: Optional[float]) -> str:
    if s is None:
        return "n/a"
    s = int(round(s))
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


def summary_lines(prog: Dict[str, Any]) -> List[str]:
    """The progress in a few printable lines."""
    total = prog.get("epochs_total")
    lines = [f"scenario {prog['scenario']}: {prog['state'].upper()} - {prog['reason']}",
             f"run directory: {prog.get('run_dir') or 'none yet'}",
             f"epochs: {prog['epochs_done']} / {total if total is not None else '?'}   elapsed: {_hms(prog['elapsed_s'])}"
             f"   median epoch: {_fmt(prog['median_epoch_s'], '.1f')} s   s/step: {_fmt(prog['sec_per_step'], '.4f')}",
             f"ETA: {_hms(prog['eta_s'])} ({prog['eta_note']})",
             f"LR: {_fmt(prog['lr'], '.3g')}   indicator LR: {_fmt(prog['lr_indicator'], '.3g')}   best val loss: "
             f"{_fmt(prog['best_val_loss'], '.4f')} at epoch {prog['best_epoch'] or 'n/a'}",
             f"last update: {_hms(prog['last_update_s'])} ago   pid: {prog.get('pid') or 'none'} "
             f"({'alive' if prog.get('pid_alive') else 'not running'})   log: {prog.get('log') or 'none'}"]
    g = prog.get("gpu")
    lines.append("GPU: nvidia-smi not available" if not g else
                 f"GPU ({g['time_utc']} UTC): {_fmt(g['utilization_pct'], '.0f')}% busy, memory "
                 f"{_fmt(g['memory_used_mb'], '.0f')} / {_fmt(g['memory_total_mb'], '.0f')} MB, "
                 f"{_fmt(g['power_w'], '.0f')} W")
    return lines


def _fmt(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


# ------------------------------------------------------------------ figure
def _horizon_keys(rows: List[dict], family: str) -> List[Tuple[int, str]]:
    import re

    found = set()
    pat = re.compile(rf"^val_{re.escape(family)}_h(\d+)$")
    for r in rows:
        for k in r:
            m = pat.match(k)
            if m:
                found.add(int(m.group(1)))
    return [(i, f"h{i}") for i in sorted(found)]


def _horizon_color(i: int) -> str:
    from neural_trade.visualization import theme as T

    return T.HORIZON_COLORS.get(f"h{i}", T.SERIES[i % len(T.SERIES)])


def progress_figure(prog: Dict[str, Any], *, height: Optional[int] = None):
    """Six panels of the run's progress: training (dotted) and validation (solid) loss with the best
    epoch, the learning rates, seconds per epoch, seconds per step, elapsed hours with the ETA
    projection, and validation CRPS per horizon (training dotted). None while no epoch has finished
    (:func:`show_progress` prints the state instead of an empty figure)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from neural_trade.visualization import theme as T

    rows = prog.get("rows") or []
    if not rows:
        return None
    x = [int(r["epoch"]) + 1 for r in rows]

    def col(key):
        return [_num(r.get(key)) for r in rows]

    titles = ("Total loss (validation solid, training dotted; star = best validation epoch)",
              "Learning rate in effect at each epoch's end (log)",
              "Seconds per epoch (bars) and their median (line)",
              "Seconds per training step",
              "Elapsed training time (h) and the ETA projection (dashed, an estimate)",
              "Validation CRPS per horizon (training dotted; lower is better)")
    fig = make_subplots(rows=3, cols=2, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.09)
    first = T.legend_once()
    c_loss, c_lr, c_lri, c_time = T.OTHER_SERIES[0], T.OTHER_SERIES[1], T.OTHER_SERIES[3], T.OTHER_SERIES[2]

    fig.add_trace(go.Scatter(x=x, y=col("val_loss"), name="validation loss", mode="lines+markers",
                             line=dict(color=c_loss, dash=T.VAL_DASH)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=col("loss"), name="training loss", mode="lines+markers",
                             line=dict(color=c_loss, dash=T.TRAIN_DASH), marker=dict(symbol="circle-open")),
                  row=1, col=1)
    if prog.get("best_epoch") is not None:
        fig.add_trace(go.Scatter(x=[prog["best_epoch"]], y=[prog["best_val_loss"]], name="best validation epoch",
                                 mode="markers", marker=dict(symbol="star", size=14, color=T.INK)), row=1, col=1)

    for key, name, color in (("lr", "main LR", c_lr), ("lr_indicator", "indicator LR", c_lri)):
        y = col(key)
        if any(v is not None for v in y):
            fig.add_trace(go.Scatter(x=x, y=T.positive(y), name=name, mode="lines+markers", line=dict(color=color)),
                          row=1, col=2)
    fig.update_yaxes(type="log", tickformat=".0e", row=1, col=2)

    secs = col("epoch_seconds")
    fig.add_trace(go.Bar(x=x, y=secs, name="epoch seconds", marker=dict(color=T.rgba(c_time, 0.75)),
                         hovertemplate="epoch %{x}: %{y:.1f} s<extra></extra>"), row=2, col=1)
    if prog.get("median_epoch_s") is not None:
        fig.add_trace(go.Scatter(x=[x[0] - 0.5, x[-1] + 0.5], y=[prog["median_epoch_s"]] * 2, mode="lines",
                                 name=f"median {prog['median_epoch_s']:.1f} s",
                                 line=dict(color=T.NEUTRAL, dash="dash", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=col("sec_per_step"), name="s / step", mode="lines+markers",
                             line=dict(color=c_time), hovertemplate="epoch %{x}: %{y:.4f} s/step<extra></extra>"),
                  row=2, col=2)

    cum, total_s = [], 0.0
    for s in secs:
        total_s += s or 0.0
        cum.append(total_s / 3600.0)
    fig.add_trace(go.Scatter(x=x, y=cum, name="elapsed (h)", mode="lines+markers", line=dict(color=c_time)),
                  row=3, col=1)
    total = prog.get("epochs_total")
    med = prog.get("median_epoch_s")
    if total and med and x[-1] < total and prog.get("state") != "done":
        fig.add_trace(go.Scatter(x=[x[-1], total], y=[cum[-1], cum[-1] + (total - x[-1]) * med / 3600.0],
                                 name=f"ETA to epoch {total} (estimate)", mode="lines+markers",
                                 line=dict(color=T.NEUTRAL, dash="dash"), marker=dict(symbol="diamond-open")),
                      row=3, col=1)

    for i, h in _horizon_keys(rows, "crps"):
        color = _horizon_color(i)
        show = first(h)
        steps = prog.get("horizon_steps") or []
        label = f"{h} ({steps[i]} bars)" if i < len(steps) else h
        fig.add_trace(go.Scatter(x=x, y=col(f"val_crps_{h}"), name=label, legendgroup=h, showlegend=show,
                                 mode="lines+markers", line=dict(color=color, dash=T.VAL_DASH)), row=3, col=2)
        tr = col(f"crps_{h}")
        if any(v is not None for v in tr):
            fig.add_trace(go.Scatter(x=x, y=tr, name=f"{h} training", legendgroup=h, showlegend=False,
                                     mode="lines", line=dict(color=color, dash=T.TRAIN_DASH)), row=3, col=2)

    for r in (1, 2, 3):
        for c in (1, 2):
            fig.update_xaxes(title_text="epoch", dtick=max(1, math.ceil(len(x) / 10)), row=r, col=c)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="learning rate", row=1, col=2)
    fig.update_yaxes(title_text="seconds", row=2, col=1)
    fig.update_yaxes(title_text="s / step", row=2, col=2)
    fig.update_yaxes(title_text="hours", row=3, col=1)
    fig.update_yaxes(title_text="CRPS (scaled units)", row=3, col=2)
    sub = (f"{prog['state']}: {prog['epochs_done']} / {total if total is not None else '?'} epochs, "
           f"elapsed {_hms(prog.get('elapsed_s'))}, ETA {_hms(prog.get('eta_s'))} (estimate; early stopping can "
           f"end sooner). Solid = validation, dotted = training.")
    T.apply(fig, title=f"Long run progress - {prog['scenario']}", subtitle=sub, height=height or 980)
    fig.update_layout(margin=dict(t=150), legend=dict(y=1.06))
    T.note_on_empty(fig)
    return fig


def _run_config(run_dir: Path):
    from neural_trade.core.config import Config

    try:
        return Config.from_yaml(run_dir / "config.yaml")
    except Exception:       # a half-written or older config: the dashboard works without it
        return None


def training_figures(prog: Dict[str, Any]) -> list:
    """Notebook 01's per-epoch training dashboard and loss terms for the cell (every logged validation
    metric per horizon with its chance range), read from the finished epochs; empty while none has."""
    rows = prog.get("rows") or []
    if not rows or not prog.get("run_dir"):
        return []
    from neural_trade.visualization.training_dashboard import loss_terms_figure, training_dashboard_figure

    run_dir = Path(prog["run_dir"])
    cfg = _run_config(run_dir)
    meta = _read_json(run_dir / "meta.json") or {}
    n_val = ((meta.get("blocks") or {}).get("val") or {}).get("n")
    return [training_dashboard_figure(rows, cfg, n_val=n_val, run_dir=run_dir),
            loss_terms_figure(rows, cfg, run_dir=run_dir)]


def _text(display, text: str) -> None:
    """Show plain text as a notebook output (text/plain; the package does not print)."""
    from IPython.display import Pretty

    display(Pretty(text))


def show_progress(prog: Dict[str, Any], *, dashboards: bool = True) -> None:
    """Display the progress in a notebook: summary lines, the progress figure (or the state when no
    epoch has finished), the per-epoch table, the training dashboards and the launch log's tail."""
    from IPython.display import display

    _text(display, "\n".join(summary_lines(prog)))
    fig = progress_figure(prog)
    if fig is None:
        _text(display, "No epoch has finished yet: the figures appear after the first epoch "
                       "(the data load and the loss-weight calibration pass come first).")
    else:
        fig.show()
        table = prog.get("table")
        if table is not None and len(table):
            display(styled_epoch_table(table))
        if dashboards:
            for f in training_figures(prog):
                f.show()
    _text(display, f"launch log, last {LOG_TAIL_LINES} lines ({prog.get('log') or 'no launch log'}):\n"
                   + "\n".join(prog.get("log_tail") or ["(empty)"]))


# ------------------------------------------------------------------ results
BACKTEST_ROWS = (
    ("total_return", "total return"), ("sharpe_net", "net Sharpe"), ("sharpe_gross", "gross Sharpe"),
    ("max_drawdown", "max drawdown"), ("n_trades", "trades"), ("hit_rate", "hit rate (net)"),
    ("exposure", "exposure"), ("costs_paid", "costs paid"), ("net_pnl", "net P&L"),
)
HORIZON_ROWS = (
    ("direction/auc", "direction AUC"), ("direction/bal_acc", "direction balanced accuracy"),
    ("direction/mcc", "direction MCC"), ("direction/brier", "direction Brier"),
    ("gauss_direction/auc", "Gaussian readout AUC"), ("delta/skill_vs_zero", "delta skill vs zero (served)"),
    ("delta_raw/corr", "delta corr (raw heads)"), ("variance/crps", "CRPS"), ("variance/crpss", "CRPSS vs constant"),
    ("variance/coverage90", "coverage at 0.90"), ("variance/width90", "90% width"), ("variance/nll", "NLL"),
    ("variance/pit_ks", "PIT-KS"), ("n_eff", "n_eff"),
)


def results(spec, store="runs", *, root=".") -> Dict[str, Any]:
    """The scored cell once its result.json exists: status, role, the key numbers per horizon, the
    backtest against buy-and-hold, always-flat and the random null, and the report markdown."""
    import pandas as pd

    root = Path(root)
    sc = _scenario(spec, root)
    dirs = [d for d in cell_dirs(_scenario_path(_resolve(store, root), sc.name)) if _finished(d) is not None]
    if not dirs:
        return {"available": False, "reason": "no scored cell yet (no result.json)"}
    run_dir = dirs[-1]
    res = _finished(run_dir) or {}
    role = res.get("role") or "dev"
    scores = res.get("scores") or {}
    out = {"available": True, "run_dir": str(run_dir), "status": res.get("status"), "role": role,
           "error": res.get("error"), "wall_s": _num(res.get("wall_s")), "sec_per_step": _num(res.get("sec_per_step"))}
    hs = sorted({k.split("/")[0] for k in scores if k[:1] == "h" and k.split("/")[0][1:].isdigit()})
    out["horizons"] = pd.DataFrame({h: [_num(scores.get(f"{h}/{k}")) for k, _ in HORIZON_ROWS] for h in hs},
                                   index=[label for _, label in HORIZON_ROWS]) if hs else None
    bt = {}
    for name, prefix in (("strategy", "backtest/"), ("buy and hold", "backtest/buy_and_hold/"),
                         ("always flat", "backtest/always_flat/")):
        bt[name] = [_num(scores.get(prefix + k)) for k, _ in BACKTEST_ROWS]
    out["backtest"] = pd.DataFrame(bt, index=[label for _, label in BACKTEST_ROWS]) if scores else None
    null = {k.rsplit("/", 1)[1]: _num(v) for k, v in scores.items() if k.startswith("backtest/random_same_freq/")}
    out["random_null"] = null or None
    md = run_dir / f"eval_report_{role}.md"
    out["report_md"] = md.read_text(encoding="utf-8") if md.is_file() else None
    out["report_path"] = str(md)
    return out


def show_results(res: Dict[str, Any]) -> None:
    """Display :func:`results` in a notebook."""
    from IPython.display import Markdown, display

    if not res.get("available"):
        _text(display, res.get("reason", "no result"))
        return
    _text(display, f"{res['run_dir']}: {res['status']} ({res['role']} block), wall time "
                   f"{_hms(res.get('wall_s'))}, s/step {_fmt(res.get('sec_per_step'), '.4f')}")
    if res.get("error"):
        _text(display, f"error: {res['error'].get('type')} {res['error'].get('message')}")
        return
    from neural_trade.visualization import analytics_tables as AT

    if res.get("horizons") is not None:
        display(AT.styled(res["horizons"], digits=4))
    if res.get("backtest") is not None:
        display(AT.styled(res["backtest"], digits=4))
    if res.get("random_null"):
        _text(display, "random entries at the same frequency: "
                       + ", ".join(f"{k} {_fmt(v, '.4g')}" for k, v in res["random_null"].items()))
    if res.get("report_md"):
        display(Markdown(res["report_md"]))


__all__ = ["STATES", "cell_dirs", "epoch_table", "eta_seconds", "gpu_snapshot", "launch", "launch_command",
           "launch_env", "launch_records", "pid_alive", "progress", "progress_figure", "read_csv_rows",
           "read_jsonl", "results", "scenario_dir", "show_progress", "show_results", "stall_after", "stop",
           "styled_epoch_table",
           "summary_lines", "tail", "training_figures"]

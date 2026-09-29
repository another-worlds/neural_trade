"""NT-035 Part A harness: launch N `neural-trade train` processes at once, one repeat at a time.

Uses the standing CLI unmodified (no src/ or tests/ edit): each process gets its own --runs-dir
subfolder and --seed, so there is no shared scenario-store index and therefore no cell-locking race
(the engine "has no cross-process cell locking yet"; this harness sidesteps it by giving every
concurrent process its own run directory instead of a shared store).

Run from the pinned worktree (PYTHONPATH set to its src/, cwd at its root, so CSV_PATH resolves):
    python harness_concurrent.py --n 4 --repeat 1 --seed-base 1000 \
        --out runs/experiments/gpu_measurements_v1/throughput
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def _dmon_watcher(stop_event: threading.Event, samples: list) -> None:
    # One-second nvidia-smi dmon samples for the duration of the batch; parsed into (sm, fb) pairs.
    proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1"], stdout=subprocess.PIPE,
                             text=True)
    try:
        for line in proc.stdout:
            if stop_event.is_set():
                break
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit():
                try:
                    sm = float(parts[1]); fb = float(parts[7])
                    samples.append((sm, fb))
                except ValueError:
                    pass
    finally:
        proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fold-index", type=int, default=-3)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True, help="runs-dir for the child processes")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    samples: list = []
    watcher = threading.Thread(target=_dmon_watcher, args=(stop_event, samples), daemon=True)
    watcher.start()

    procs = []
    py = sys.executable
    t0 = time.perf_counter()
    for i in range(args.n):
        name = f"n{args.n}_r{args.repeat}_p{i}"
        cmd = [py, "-m", "neural_trade.cli", "train", "--config", args.config,
               "--set", f"FOLD_INDEX={args.fold_index}", "--epochs", str(args.epochs),
               "--seed", str(args.seed_base + i), "--no-calibrate", "--no-baselines",
               "--runs-dir", args.out, "--name", name]
        procs.append((name, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                              text=True)))

    run_dirs = {}
    for name, p in procs:
        out, err = p.communicate()
        run_dirs[name] = {"returncode": p.returncode, "stdout_tail": out.strip().splitlines()[-1:],
                           "stderr_tail": err.strip().splitlines()[-5:]}
    wall = time.perf_counter() - t0
    stop_event.set()
    time.sleep(1.2)  # let the dmon thread exit its readline loop

    sm_vals = [s for s, _ in samples]
    fb_vals = [f for _, f in samples]
    summary = {
        "n": args.n, "repeat": args.repeat, "wall_s": wall,
        "mean_sm_pct": sum(sm_vals) / len(sm_vals) if sm_vals else None,
        "peak_sm_pct": max(sm_vals) if sm_vals else None,
        "mean_fb_mb": sum(fb_vals) / len(fb_vals) if fb_vals else None,
        "peak_fb_mb": max(fb_vals) if fb_vals else None,
        "processes": run_dirs,
    }
    out_path = Path(args.out) / f"summary_n{args.n}_r{args.repeat}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(v["returncode"] == 0 for v in run_dirs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

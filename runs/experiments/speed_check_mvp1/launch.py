"""NT-075 launcher: interleaved A B A B A B runs, GPU-free check before each, CPU load sampled per run.

A = 1aeff1c, B = 426de4f (SPEC.md). Usage: python launch.py [order, default ABABAB] [first_k]
"""
import os
import statistics
import subprocess
import sys
import threading
import time

import psutil

PY = "C:/Users/Step/miniforge3/envs/nt/python"
OUT = "D:/nt/neural_trade/runs/experiments/speed_check_mvp1"
SIDES = {"A": ("D:/nt/nt_exp_speed_1aeff1c", "1aeff1c"), "B": ("D:/nt/nt_exp_speed_426de4f", "426de4f")}
CSV = "D:/nt/neural_trade/binance_btcusdt_1min_ccxt.csv"


def gpu_free():
    out = subprocess.run(["nvidia-smi", "dmon", "-s", "um", "-c", "10"], capture_output=True, text=True).stdout
    rows = [r.split() for r in out.splitlines() if r.strip() and not r.startswith("#")]
    sm = [int(r[1]) for r in rows]
    fb = [int(r[6]) for r in rows]
    return statistics.median(sm) <= 30 and max(fb) <= 2000, out


def main():
    order = sys.argv[1] if len(sys.argv) > 1 else "ABABAB"
    k0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(OUT + "/runs", exist_ok=True)
    counts = {"A": k0 - 1, "B": k0 - 1}
    for slot, s in enumerate(order):
        counts[s] += 1
        k = counts[s]
        name = f"{s}_{SIDES[s][1]}_r{k}"
        for attempt in range(60):
            ok, dmon = gpu_free()
            if ok:
                break
            print(name, "GPU busy, waiting", flush=True)
            time.sleep(60)
        else:
            print("GPU busy for an hour, stopping", flush=True)
            return 2
        open(f"{OUT}/{name}.gpucheck.txt", "w").write(dmon)
        wt = SIDES[s][0]
        env = dict(os.environ, PYTHONPATH=wt + "/src", PYTHONIOENCODING="utf-8")
        samples = []
        stop = threading.Event()

        def sampler():
            psutil.cpu_percent(None)
            while not stop.wait(5):
                samples.append((time.time(), psutil.cpu_percent(None)))

        th = threading.Thread(target=sampler)
        th.start()
        t0 = time.time()
        with open(f"{OUT}/{name}.stdout", "w") as so, open(f"{OUT}/{name}.stderr", "w") as se:
            rc = subprocess.run([PY, "-m", "neural_trade.cli", "train", "--config", "configs/default.yaml", "--csv", CSV,
                                 "--runs-dir", OUT + "/runs", "--name", name, "--no-baselines"],
                                cwd=wt, env=env, stdout=so, stderr=se).returncode
        stop.set()
        th.join()
        with open(f"{OUT}/{name}.cpu.csv", "w") as f:
            f.write("unix_time,cpu_percent\n")
            for t, c in samples:
                f.write(f"{t:.0f},{c}\n")
        cs = [c for _, c in samples]
        line = (f"{name} slot={slot} rc={rc} wall_s={time.time()-t0:.0f} cpu_mean={statistics.mean(cs) if cs else float('nan'):.1f}"
                f" cpu_max={max(cs) if cs else float('nan'):.1f}")
        print(line, flush=True)
        open(f"{OUT}/progress.txt", "a").write(line + "\n")
    open(f"{OUT}/progress.txt", "a").write("ALLDONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

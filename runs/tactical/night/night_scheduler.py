"""Night scheduler (owner, 2026-10-08: "take the GPU and run every waiting experiment until ~6:00; nothing else runs at
night - ignore other processes; watch actively at max resources and prevent crashes").

One process replaces the bash guard chain. Every 10 s it:
  * reads runs/tactical/night/queue.txt ("spec shard cap_mb" per line; lines can be appended while it runs);
  * skips a task whose shard file already holds all its trials (resume makes re-runs cheap anyway);
  * treats tactical screen processes it did not start (left from the old guards) as running tasks;
  * starts tasks while fewer than MAX_PROCS run, the GPU has at least cap + HEADROOM MB free and RAM >= MIN_RAM_GB;
    each task gets NT_GPU_MEMORY_LIMIT_MB = its cap and TF memory growth;
  * when a task ends, counts its rows: complete -> done; incomplete after a GPU out-of-memory -> re-queued solo with
    cap + 1500 MB (max 9000); otherwise re-queued up to 3 attempts (screen resume keeps finished trials);
  * emergency: free GPU memory < 300 MB or free RAM < 3 GB -> the newest task is stopped and re-queued;
  * writes runs/tactical/night/status.json and log.txt, and a load line to runs/tactical/hc4/resources.csv (dashboard).
usage: python night_scheduler.py"""
import json, math, os, re, subprocess, sys, time
import psutil
import yaml

ROOT = r"D:\nt\nt_tactical"; os.chdir(ROOT)
NIGHT = "runs/tactical/night"; QUEUE = f"{NIGHT}/queue.txt"; LOG = f"{NIGHT}/log.txt"; STATUS = f"{NIGHT}/status.json"
PY = r"C:\Users\Step\miniforge3\envs\nt\python.exe"
MAX_PROCS, HEADROOM, MIN_RAM_GB, CHECK = 3, 700, 6.0, 10
os.makedirs(f"{NIGHT}/logs", exist_ok=True)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True); open(LOG, "a", encoding="utf-8").write(line + "\n")


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return int(o[0]), int(o[1]), int(o[2])
    except Exception:
        return 0, 0, 12282


def expected_rows(spec, shard):
    s = yaml.safe_load(open(f"configs/tactical/{spec}.yaml"))
    total = len(s["slices"]) * len(s["seeds"])
    i, n = map(int, shard.split("/"))
    return sum(1 for j in range(total) if j % n == i)


def rows(spec, shard):
    i, n = shard.split("/")
    p = f"runs/tactical/screens/{spec}/results.shard-{i}-of-{n}.jsonl"
    return sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0


def external():
    """tactical screen processes not started by this scheduler: {(spec, shard): pid}."""
    out = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            c = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        m = re.search(r"configs/tactical/(\w+)\.yaml --store runs/tactical --shard (\d+/\d+)", c)
        if m and "neural_trade.cli" in c:
            out[(m.group(1), m.group(2))] = p.info["pid"]
    return out


def read_queue():
    q = []
    if os.path.exists(QUEUE):
        for l in open(QUEUE, encoding="utf-8"):
            parts = l.split()
            if len(parts) >= 3 and not l.startswith("#"):
                q.append((parts[0], parts[1], int(parts[2])))
    return q


running = {}   # (spec, shard) -> dict(proc, cap, started, solo)
state = {"done": [], "attempts": {}, "caps": {}, "solo": [], "failed": []}
if os.path.exists(STATUS):
    try:
        old = json.load(open(STATUS)); state.update({k: old.get(k, state[k]) for k in state})
    except Exception:
        pass
log(f"--- scheduler start; max {MAX_PROCS} processes")
idle_since = None
while True:
    # finished tasks
    for key in list(running):
        r = running[key]
        if r["proc"].poll() is None:
            continue
        rc = r["proc"].returncode; del running[key]
        spec, shard = key; got, need = rows(spec, shard), expected_rows(spec, shard)
        k = f"{spec} {shard}"
        if got >= need:
            state["done"].append(k); log(f"done {k} ({got}/{need}, rc {rc}, {round((time.time()-r['started'])/60,1)} min)")
        else:
            tail = open(r["log"], encoding="utf-8", errors="ignore").read()[-20000:] if os.path.exists(r["log"]) else ""
            state["attempts"][k] = state["attempts"].get(k, 0) + 1
            if "ResourceExhausted" in tail or "OOM" in tail:
                state["caps"][k] = min(9000, r["cap"] + 1500)
                if k not in state["solo"]:
                    state["solo"].append(k)
                log(f"OOM {k} ({got}/{need}) -> retry solo at {state['caps'][k]} MB")
            elif state["attempts"][k] >= 3:
                state["failed"].append(k); log(f"FAILED {k} ({got}/{need}) after 3 attempts, rc {rc}")
            else:
                log(f"incomplete {k} ({got}/{need}, rc {rc}) -> retry")
    ext = {k: v for k, v in external().items() if k not in running}
    util, used, total = gpu(); vfree = total - used
    ram = psutil.virtual_memory().available / 2**30
    n_run = len(running) + len(ext)
    # emergency
    if (vfree < 300 or ram < 3) and running:
        key = max(running, key=lambda k: running[k]["started"])
        try:
            psutil.Process(running[key]["proc"].pid).kill()
        except Exception:
            pass
        log(f"EMERGENCY stop {key[0]} {key[1]} (GPU free {vfree} MB, RAM {ram:.1f} GB)")
        time.sleep(20); continue
    # start
    started = None
    queue = read_queue()
    for spec, shard, cap in queue:
        k = f"{spec} {shard}"
        if k in state["done"] or k in state["failed"] or (spec, shard) in running or (spec, shard) in ext:
            continue
        if rows(spec, shard) >= expected_rows(spec, shard):
            state["done"].append(k); continue
        cap = state["caps"].get(k, cap); solo = k in state["solo"]
        if n_run >= MAX_PROCS or (solo and n_run > 0) or any(r["solo"] for r in running.values()):
            break
        if vfree < cap + HEADROOM or ram < MIN_RAM_GB:
            break
        i, n = shard.split("/"); lg = f"{NIGHT}/logs/{spec}_{i}of{n}.log"
        env = dict(os.environ, PYTHONPATH=r"D:\nt\nt_tactical\src", PYTHONIOENCODING="utf-8", TF_FORCE_GPU_ALLOW_GROWTH="true",
                   NT_GPU_MEMORY_LIMIT_MB=str(cap))
        env.pop("CUDA_VISIBLE_DEVICES", None)
        p = subprocess.Popen([PY, "-m", "neural_trade.cli", "screen", f"configs/tactical/{spec}.yaml", "--store", "runs/tactical",
                              "--shard", shard], cwd=ROOT, env=env, stdout=open(lg, "a"), stderr=subprocess.STDOUT)
        running[(spec, shard)] = {"proc": p, "cap": cap, "started": time.time(), "solo": solo, "log": lg}
        log(f"start {k} cap {cap} MB (GPU free {vfree} MB, RAM {ram:.1f} GB){' SOLO' if solo else ''}")
        started = k
        break  # one start per check: let it allocate before the next decision
    pending = [f"{s} {sh}" for s, sh, _ in queue if f"{s} {sh}" not in state["done"] and f"{s} {sh}" not in state["failed"]]
    json.dump({**state, "running": [f"{a} {b}" for a, b in running] + [f"{a} {b} (external)" for a, b in ext],
               "pending": pending, "gpu_util": util, "gpu_free_mb": vfree, "ram_free_gb": round(ram, 1),
               "time": time.strftime("%H:%M:%S")}, open(STATUS, "w"), indent=1)
    open("runs/tactical/hc4/resources.csv", "a").write(f"{time.strftime('%H:%M:%S')},{ram:.1f},{int(psutil.cpu_percent())},{util},{used},{n_run},{'start' if started else ''}\n")
    if not running and not ext and not pending:
        idle_since = idle_since or time.time()
        if time.time() - idle_since > 120:
            log("queue empty: all done"); break
    else:
        idle_since = None
    time.sleep(40 if started else CHECK)

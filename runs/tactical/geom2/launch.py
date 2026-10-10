"""Run shell-free commands in parallel at below-normal priority, 4 threads each (night program).
usage: python launch.py <workers> <logdir> -- cmd1 args ;; cmd2 args ;; ..."""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
w = int(sys.argv[1]); logdir = sys.argv[2]; os.makedirs(logdir, exist_ok=True)
cmds, cur = [], []
for t in sys.argv[4:]:
    if t == ";;":
        cmds.append(cur); cur = []
    else:
        cur.append(t)
if cur:
    cmds.append(cur)
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
def run(ic):
    i, c = ic; t = time.time()
    with open(f"{logdir}/job{i}.log", "w", encoding="utf-8") as f:
        p = subprocess.run([sys.executable] + c, stdout=f, stderr=subprocess.STDOUT, env=env, creationflags=0x4000)
    print(f"job{i} rc {p.returncode} {time.time() - t:.0f}s: {' '.join(c)[:100]}", flush=True)
with ThreadPoolExecutor(w) as ex:
    list(ex.map(run, enumerate(cmds)))

"""The night's hourly program, exactly as SPEC.md: stage 1 geometry (25), stage 2 features x primary x target x meta on the best 3
geometries (72), then the held-out check of the top 3 overall (once). 6 processes at below-normal priority, 4 threads each.
usage: python run_night.py   (resumable: configurations already in results.jsonl / final.jsonl are skipped)"""
import itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

os.chdir(r"D:\nt\nt_tactical"); OUT = "runs/tactical/hourly"; PY = sys.executable
LOG = open(f"{OUT}/night.log", "a", encoding="utf-8")
BELOW = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS


def log(m):
    LOG.write(time.strftime("%H:%M:%S ") + m + "\n"); LOG.flush()


def tag(c):
    return "T{T}_tp{tp}_sl{sl}_{feat}_{primary}_{target}_{meta}".format(**c)


def done(path):
    return {json.loads(l)["tag"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()


def run(c, final=False):
    if tag(c) in done(f"{OUT}/{'final' if final else 'results'}.jsonl"):
        return
    args = [PY, f"{OUT}/hourly.py", "--T", str(c["T"]), "--tp", str(c["tp"]), "--sl", str(c["sl"]), "--feat", c["feat"],
            "--primary", c["primary"], "--target", c["target"], "--meta", c["meta"]] + (["--final"] if final else [])
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    t = time.time(); p = subprocess.run(args, capture_output=True, text=True, env=env, creationflags=BELOW)
    log(f"{'FINAL ' if final else ''}{tag(c)} rc {p.returncode} {time.time() - t:.0f}s {p.stdout.strip()[-200:]} {p.stderr.strip()[-300:] if p.returncode else ''}")


def results():
    return [json.loads(l) for l in open(f"{OUT}/results.jsonl", encoding="utf-8")]


def rank_key(r):
    s = r["summary"]; return (s.get("meta_excess", [-1e9])[0], s.get("meta_bps", [0, -1e9])[1])


base = dict(feat="tb7", primary="logreg", target="sign", meta="base")
stage1 = [dict(base, T=T, tp=tp, sl=sl) for T in (4, 8, 12, 24, 48) for tp, sl in ((1.0, 1.0), (1.5, 1.5), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0))]
log(f"--- start: stage 1, {len(stage1)} configurations")
with ThreadPoolExecutor(6) as ex:
    list(ex.map(run, stage1))
s1 = [r for r in results() if r["tag"] in {tag(c) for c in stage1}]
geoms = [r["config"] for r in sorted(s1, key=rank_key, reverse=True)[:3]]
log("stage 1 top 3 geometries: " + ", ".join(f"T{g['T']} tp{g['tp']} sl{g['sl']}" for g in geoms))
stage2 = [dict(T=g["T"], tp=g["tp"], sl=g["sl"], feat=f, primary=p, target=t, meta=m)
          for g in geoms for f, p, t, m in itertools.product(("tb7", "rich", "ctx"), ("logreg", "hgb"), ("sign", "barrier"), ("base", "mag"))]
log(f"--- stage 2, {len(stage2)} configurations")
with ThreadPoolExecutor(6) as ex:
    list(ex.map(run, stage2))
allr = [r for r in results() if r["tag"] in {tag(c) for c in stage1 + stage2}]
top = [r["config"] for r in sorted(allr, key=rank_key, reverse=True)[:3]]
json.dump({"ranking": [(r["tag"], r["summary"].get("meta_excess"), r["summary"].get("meta_bps")) for r in sorted(allr, key=rank_key, reverse=True)],
           "top3": [tag(c) for c in top]}, open(f"{OUT}/ranking.json", "w"), indent=1)
log("--- held-out check (once), top 3: " + ", ".join(tag(c) for c in top))
for c in top:
    run(c, final=True)
log("--- night program done")

"""Lab slices (24): all arms, direction + volatility. Resumable per slice (rows appended to results_lab[_<tag>].jsonl).
usage: python lowrun.py run_lab.py [--shard i n] [--k 12] [--arms ...] [--targets dir,vol] [--tag name]"""
import argparse, glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import synthlib as S  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--shard", nargs=2, type=int, default=[0, 1]); ap.add_argument("--k", type=int, default=S.K_PRIMARY)
ap.add_argument("--arms", default="tb7,c3,A,B,C1,C2a,C2b"); ap.add_argument("--targets", default="dir,vol"); ap.add_argument("--tag", default="")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); S.tf_setup(1)
out = os.path.join(HERE, f"results_lab{('_' + a.tag) if a.tag else ''}.jsonl")
done = {json.loads(l)["slice"] for l in open(out)} if os.path.exists(out) else set()
paths = sorted(glob.glob(S.LAB_CACHE + "/*.npz"))
if a.limit: paths = paths[:a.limit]
for n, p in enumerate(paths):
    if n % a.shard[1] != a.shard[0]:
        continue
    name = os.path.basename(p)[:-4]
    if name in done:
        continue
    t0 = time.time(); D = S.lab_slice(p)
    tr = {"W": D["Wtr"], "r": D["rtr"], "v": D["vtr"], "h": D["htr"]}; va = {"W": D["Wva"], "r": D["rva"], "v": D["vva"], "h": D["hva"]}
    print(f"{name}: train {len(tr['W'])} val {len(va['W'])}", flush=True)
    res = S.run_block(tr, va, seed=n, k=a.k, arms=tuple(a.arms.split(",")), targets=tuple(a.targets.split(",")), log=lambda s: print(s, flush=True))
    with open(out, "a") as f:
        f.write(json.dumps({"slice": name, "data_end": D["data_end"], "k": a.k, "res": res, "sec": round(time.time() - t0, 1)}) + "\n")
    print(f"{name} done in {time.time() - t0:.0f}s", flush=True)

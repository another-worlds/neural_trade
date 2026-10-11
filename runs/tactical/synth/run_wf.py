"""Monthly walk-forward on the selection months 2021-01 .. 2024-06 (SPEC protocol 2): 365-day training (anchors every 20 bars),
next-open direction labels, next-hour log RV labels; all arms. Resumable per month.
usage: python lowrun.py run_wf.py [--shard i n] [--first 2021-01] [--last 2024-06] [--tag name] [--arms ...]"""
import argparse, json, os, sys, time
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import synthlib as S  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--shard", nargs=2, type=int, default=[0, 1]); ap.add_argument("--first", default="2021-01")
ap.add_argument("--last", default="2024-06"); ap.add_argument("--tag", default=""); ap.add_argument("--arms", default="tb7,c3,A,B,C1,C2a,C2b")
ap.add_argument("--targets", default="dir,vol")
a = ap.parse_args(); S.tf_setup(1)
out = os.path.join(HERE, f"results_wf{('_' + a.tag) if a.tag else ''}.jsonl")
done = {json.loads(l)["month"] for l in open(out)} if os.path.exists(out) else set()
d = S.WF()
for n, m in enumerate(pd.period_range(a.first, a.last, freq="M")):
    if n % a.shard[1] != a.shard[0] or str(m) in done:
        continue
    t0 = time.time(); m0, m1 = m.start_time, (m + 1).start_time
    mtr, mte = d.month_masks(m0, m1)
    blk = lambda sel: {"W": d.windows(sel), "r": d.r[sel], "v": d.v[sel], "h": S.har_features(d.windows(sel), d.mod[sel])}
    tr, te = blk(mtr), blk(mte)
    print(f"{m}: train {len(tr['W'])} test {len(te['W'])}", flush=True)
    res = S.run_block(tr, te, seed=n, arms=tuple(a.arms.split(",")), targets=tuple(a.targets.split(",")), log=lambda s: print(s, flush=True))
    with open(out, "a") as f:
        f.write(json.dumps({"month": str(m), "n_train": len(tr["W"]), "n_test": len(te["W"]), "res": res, "sec": round(time.time() - t0, 1)}) + "\n")
    print(f"{m} done in {time.time() - t0:.0f}s", flush=True)

"""H56 look months 2024-07..2025-09 with the entry price as a switch: close (the recorded study) or the next bar's open.
Same cache (recent/ feature blocks), same walk-forward (recent/wf.walk_forward: monthly, 365-day train, purge, deadband mask,
standardise+clip, models), same picks (recent/selection.json) and baselines (tb7:logreg:T); ONLY the label return r changes
(train labels and test labels both), so AUC / top-10% numbers are re-measured on the new label.
usage: python wf_no.py close|nextopen [--workers 4]
Writes nextopen/h56_<entry>/look_<bar>.jsonl (+ selection.json copy); analyze with analyze_no.py"""
import json, os, shutil, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REC = os.path.join(os.path.dirname(HERE), "recent")
sys.path.insert(0, REC); sys.path.insert(0, HERE)
import wf  # noqa: E402
import nolib  # noqa: E402


def task(a):
    bar, entry, name, kind, T = a; wf.lowprio(); t0 = time.time()
    blocks, t, r_cache = wf.load_cell(bar)
    r = nolib.returns_for_cache(bar, t, entry)
    if entry == "close":
        assert np.allclose(r, r_cache, rtol=0, atol=1e-12), "recomputed close-entry return differs from the cache"
    X = np.concatenate([np.asarray(blocks[b], np.float32) for b in wf.SETS[name]], 1) if len(wf.SETS[name]) > 1 else blocks[wf.SETS[name][0]]
    f, l = wf.MONTHS["look"]; lo = pd.Period(f, "M").start_time - pd.Timedelta(days=T + 1); hi = (pd.Period(l, "M") + 1).start_time
    ti = pd.DatetimeIndex(t); sel = np.asarray((ti >= lo) & (ti < hi))
    rows = wf.walk_forward(np.asarray(X)[sel], ti[sel], r[sel], bar, kind, T, f, l)
    return {"bar": bar, "set": name, "model": kind, "T": T, "entry": entry, "n_features": int(X.shape[1]), "seconds": round(time.time() - t0, 1), "months": rows}


if __name__ == "__main__":
    import multiprocessing as mp
    entry = sys.argv[1]; assert entry in ("close", "nextopen")
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    out = os.path.join(HERE, f"h56_{entry}"); os.makedirs(out, exist_ok=True)
    sel = json.load(open(f"{REC}/selection.json")); shutil.copy(f"{REC}/selection.json", f"{out}/selection.json")
    todo = []
    for bar in ("1m", "1h"):
        picks = sel["picks"][bar]; only = list(dict.fromkeys(picks + [f"tb7:logreg:{p.split(':')[2]}" for p in picks]))
        for p in only:
            s, m, T = p.split(":"); todo.append((bar, entry, s, m, int(T)))
    todo.sort(key=lambda x: (x[0] != "1m", x[3] != "l1"))     # slow L1 minute cells first
    done = set()
    for bar in ("1m", "1h"):
        pth = f"{out}/look_{bar}.jsonl"
        if os.path.exists(pth):
            done |= {(bar, d["set"], d["model"], d["T"]) for d in map(json.loads, open(pth, encoding="utf-8"))}
    todo = [x for x in todo if (x[0], x[2], x[3], x[4]) not in done]
    print(entry, len(todo), "cells to run", flush=True)
    with mp.Pool(workers) as pool:
        for d in pool.imap_unordered(task, todo):
            open(f"{out}/look_{d['bar']}.jsonl", "a", encoding="utf-8").write(json.dumps(d) + "\n")
            print(d["bar"], d["set"], d["model"], d["T"], len(d["months"]), "months", d["seconds"], "s", flush=True)

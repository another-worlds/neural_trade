"""Resumable monthly runner for hypothesis #6. usage: python run_var.py sel|look [--workers 4]
The state kernel (p*, c*) is read from results/state.json (written by analyze_omori.py sel from the SPEC.md selection). look: refused unless
results/var_selected.json lists a selected primary cell and the one look has not run."""
import os, sys, json, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import omorilib as O, varlib as V

RES = os.path.join(HERE, "results")
PHASES = {"sel": ("2019-01", "2024-06"), "look": ("2024-07", "2025-07")}
_G = {}


def _init():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    _G["B"] = O.Bars(); assert _G["B"].df.index.max() <= O.CUTOFF
    st = json.load(open(f"{RES}/state.json")); _G["p"], _G["c"] = st["p_star"], st["c_star"]


def _job(arg):
    m, T = arg; t0 = time.time(); r = V.score_month_var(_G["B"], m, T, _G["p"], _G["c"]); return m, T, r, time.time() - t0


if __name__ == "__main__":
    phase = sys.argv[1]; w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    first, last = PHASES[phase]
    if phase == "look":
        sel = json.load(open(f"{RES}/var_selected.json"))
        if not sel.get("cells"):
            sys.exit("look refused: no primary cell was selected")
        if os.path.exists(f"{RES}/var_look.jsonl"):
            sys.exit("look refused: the one look already ran")
    path = f"{RES}/var_{phase}.jsonl"; done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            j = json.loads(l); done.add((j["month"], j["T"]))
    jobs = [(str(m), T) for m in pd.period_range(first, last, freq="M") for T in V.TV if (str(m), T) not in done]
    print(f"var {phase}: {len(jobs)} jobs, {len(done)} done", flush=True); t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(w, initializer=_init) as pool, open(path, "a", encoding="utf-8") as fh:
        for m, T, r, sec in pool.imap_unordered(_job, jobs):
            if r is None:
                print(f"  {m} T={T}: skipped", flush=True); continue
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(f"  {m} T={T} {sec:.0f}s", flush=True)
    print(f"var {phase} done in {time.time() - t0:.0f}s", flush=True)

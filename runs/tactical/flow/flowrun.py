"""Runner for SPEC.md. usage: python flowrun.py sel|look [--workers 4]   (build the cache first: python flowlib.py build 1m|5m)
sel  -> sel_<cell>.jsonl : tb7 / flow / tb7+flow x logreg / hgb, T=365, months 2021-01..2024-06; plus flow-alone logreg
        stability months 2017-03..2024-06 (T=60 and T=365)
look -> look_<cell>.jsonl: refuses unless selection.json exists; tb7 and tb7+flow for the tests that added, plus flow-alone
        stability months 2024-07..2025-09. Run ONCE."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TAC, "recent")); sys.path.insert(0, HERE)
import wf                                           # noqa: E402  (walk_forward, lowprio: imported, unchanged)
import flowlib as FL                                # noqa: E402

SEL = ("2021-01", "2024-06"); LOOK = ("2024-07", "2025-07")
STAB_SEL = ("2017-03", "2024-06"); STAB_LOOK = ("2024-07", "2025-07")
LAST_SEL_MONTH = pd.Period("2024-06", "M")
SETS = {"tb7": ("tb7",), "flow": ("flow",), "tb7+flow": ("tb7", "flow")}


def load(bar):
    c = FL.CACHE
    return {k: np.load(f"{c}/{bar}_{k}.npy") for k in ("tb7", "flow", "t", "r")}


def run_cell(d, bar, setname, kind, T, first, last, last_allowed=None):
    """one walk-forward; `last_allowed` (a Period) makes the selection phase refuse later months."""
    if last_allowed is not None:
        assert pd.Period(last, "M") <= last_allowed, f"months after {last_allowed} are not allowed here"
        assert pd.Period(first, "M") <= pd.Period(last, "M")
    cfg = FL.CELLS[bar]
    X = np.concatenate([d[k] for k in SETS[setname]], 1) if len(SETS[setname]) > 1 else d[SETS[setname][0]]
    t = pd.DatetimeIndex(np.asarray(d["t"], np.int64).astype("datetime64[ns]"))
    lo = pd.Period(first, "M").start_time - pd.Timedelta(days=T + 1); hi = (pd.Period(last, "M") + 1).start_time
    sel = np.asarray((t >= lo) & (t < hi))
    return wf.walk_forward(X[sel], t[sel], d["r"][sel], bar, kind, T, first, last, H=cfg["H"], bar_min=cfg["bar_min"], db=FL.DB)


def _task(a):
    bar, phase, setname, kind, T, first, last, tag = a; wf.lowprio(); t0 = time.time()
    d = load(bar)
    rows = run_cell(d, bar, setname, kind, T, first, last, LAST_SEL_MONTH if phase == "sel" else None)
    return {"bar": bar, "phase": phase, "tag": tag, "set": setname, "model": kind, "T": T, "n_features": int(
        sum(d[k].shape[1] for k in SETS[setname])), "seconds": round(time.time() - t0, 1), "months": rows}


def tasks(phase):
    out = []
    for bar in FL.CELLS:
        if phase == "sel":
            for kind in ("logreg", "hgb"):
                for s in SETS:
                    out.append((bar, "sel", s, kind, 365, *SEL, "main"))
            for T in (60, 365):
                out.append((bar, "sel", "flow", "logreg", T, *STAB_SEL, "stab"))
        else:
            sel = json.load(open(f"{HERE}/selection.json"))
            for key in sel["adds"]:
                b, kind = key.split(":")
                if b == bar:
                    for s in ("tb7", "tb7+flow"):
                        out.append((bar, "look", s, kind, 365, *LOOK, "main"))
            for T in (60, 365):
                out.append((bar, "look", "flow", "logreg", T, *STAB_LOOK, "stab"))
    return out


def run(phase, workers=4):
    import multiprocessing as mp
    if phase == "look" and not os.path.exists(f"{HERE}/selection.json"):
        sys.exit("look refused: selection.json does not exist")
    if phase == "look" and any(os.path.exists(f"{HERE}/look_{b}.jsonl") for b in FL.CELLS):
        sys.exit("look refused: it was already run (one look only)")
    todo = tasks(phase)
    todo.sort(key=lambda x: x[3] != "hgb")
    print(phase, len(todo), "configurations", flush=True)
    with mp.Pool(workers) as pool:
        for d in pool.imap_unordered(_task, todo):
            with open(f"{HERE}/{phase}_{d['bar']}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(d) + "\n")
            print(d["bar"], d["tag"], d["set"], d["model"], d["T"], len(d["months"]), "months", d["seconds"], "s",
                  f"auc {np.mean([m['auc'] for m in d['months']]):.4f}", flush=True)


if __name__ == "__main__":
    ph = sys.argv[1]; w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    run(ph, w)

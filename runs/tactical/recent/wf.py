"""Monthly walk-forward for the 'recent' study (SPEC.md). Pure function `walk_forward` (tested) + a resumable parallel runner.
usage: python wf.py run 1m|1h sel|look [--workers 6] [--only set:model:T,...]"""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CACHE = "D:/nt/nt_recent_cache"
SETS = {"tb7": ["tb7"], "rich": ["rich"], "ctx": ["ctx"], "gR": ["gR"], "novel": ["novel"], "har": ["har"],
        "all": ["tb7", "rich", "ctx", "gR", "novel", "har"]}
MODELS = ("logreg", "l1", "hgb")
TRAIN_DAYS = (60, 365)
MONTHS = {"sel": ("2021-01", "2024-06"), "look": ("2024-07", "2025-09")}
BAR_MIN = {"1m": 1, "1h": 60}; HOR = {"1m": 15, "1h": 1}
CS = (0.003, 0.01, 0.03, 0.1, 0.3); CLIP = 8.0
MIN_TRAIN, MIN_TEST = 500, 300


def deadband(bar):
    return 5e-4 * math.sqrt(HOR[bar] * BAR_MIN[bar] / 15.0)


def split_masks(t, m0, m1, H, bar_min, train_days):
    """train: the anchor's label bar CLOSES by the month start (t_open + (H+1) bars <= m0), at most train_days old;
    test: anchors opening in [m0, m1)."""
    t = pd.DatetimeIndex(t); lab_end = t + pd.Timedelta(minutes=(H + 1) * bar_min)
    tr = (t >= m0 - pd.Timedelta(days=train_days)) & (lab_end <= m0)
    te = (t >= m0) & (t < m1)
    return np.asarray(tr), np.asarray(te)


def _fit(kind, Xtr, ytr, gap_rows=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    info = {}
    if kind == "logreg":
        return LogisticRegression(C=0.1, max_iter=1000).fit(Xtr, ytr), info
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=0).fit(Xtr, ytr), info
    n = len(ytr); cut = int(0.8 * n); a = np.arange(0, max(cut - gap_rows, 1)); b = np.arange(cut, n)
    best, bc = -1.0, CS[0]
    if len(np.unique(ytr[a])) > 1 and len(np.unique(ytr[b])) > 1:
        for C in CS:
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=200).fit(Xtr[a], ytr[a])
            s = roc_auc_score(ytr[b], m.decision_function(Xtr[b]))
            if s > best:
                best, bc = s, C
    info["C"] = bc
    return LogisticRegression(penalty="l1", solver="liblinear", C=bc, max_iter=200).fit(Xtr, ytr), info


def walk_forward(X, t, r, bar, kind, train_days, first, last, H=None, bar_min=None, db=None):
    """X [N, F], t anchor open times (datetime64 / ns int), r forward returns. Rows are chronologically sorted. One dict per month."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    H = H or HOR[bar]; bar_min = bar_min or BAR_MIN[bar]; db = deadband(bar) if db is None else db
    t = pd.DatetimeIndex(t); mask = np.abs(r) > db; y = (r > 0).astype(int); out = []
    for m in pd.period_range(first, last, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = split_masks(t, m0, m1, H, bar_min, train_days)
        trm, tem = tr & mask, te & mask
        if trm.sum() < MIN_TRAIN or tem.sum() < MIN_TEST or len(np.unique(y[tem])) < 2 or len(np.unique(y[trm])) < 2:
            continue
        Xtr = np.asarray(X[trm], np.float32)
        sc = StandardScaler().fit(Xtr)
        f = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -CLIP, CLIP)
        mdl, info = _fit(kind, f(Xtr), y[trm], gap_rows=H + 1)
        p = mdl.predict_proba(f(X[te]))[:, 1]; pm = p[mask[te]]
        conf = np.abs(p - 0.5); k = conf >= np.quantile(conf, 0.9)
        mv = np.where(p[k] > 0.5, 1, -1) * r[te][k] * 1e4
        out.append({"month": str(m), "auc": float(roc_auc_score(y[tem], pm)), "n_train": int(trm.sum()), "n_test": int(tem.sum()),
                    "top10_hit": float(np.mean(mv > 0)), "top10_bps": float(np.mean(mv)), "n_top10": int(k.sum()), **info})
    return out


# ------------------------------------------------------------------ runner
def load_cell(bar):
    t = np.load(f"{CACHE}/{bar}_t.npy"); r = np.load(f"{CACHE}/{bar}_r.npy")
    blocks = {b: np.load(f"{CACHE}/{bar}_{b}.npy", mmap_mode="r") for b in SETS["all"]}
    return blocks, t, r


def lowprio():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _task(a):
    bar, phase, name, kind, T = a; lowprio(); t0 = time.time()
    blocks, t, r = load_cell(bar)
    X = np.concatenate([np.asarray(blocks[b], np.float32) for b in SETS[name]], 1) if len(SETS[name]) > 1 else blocks[SETS[name][0]]
    # only rows that can matter: from the first train start to the last month end
    f, l = MONTHS[phase]; lo = pd.Period(f, "M").start_time - pd.Timedelta(days=T + 1); hi = (pd.Period(l, "M") + 1).start_time
    ti = pd.DatetimeIndex(t); sel = np.asarray((ti >= lo) & (ti < hi))
    rows = walk_forward(np.asarray(X)[sel], ti[sel], r[sel], bar, kind, T, f, l)
    return {"bar": bar, "set": name, "model": kind, "T": T, "n_features": int(X.shape[1]), "seconds": round(time.time() - t0, 1), "months": rows}


def configs(bar, only=None):
    c = [(s, m, T) for T in TRAIN_DAYS for s in SETS for m in MODELS]
    if only:
        c = [x for x in c if f"{x[0]}:{x[1]}:{x[2]}" in only]
    return c


def run(bar, phase, workers=6, only=None):
    import multiprocessing as mp
    path = f"{HERE}/{phase}_{bar}.jsonl"
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            d = json.loads(l); done.add((d["set"], d["model"], d["T"]))
    todo = [(bar, phase, s, m, T) for (s, m, T) in configs(bar, only) if (s, m, T) not in done]
    print(f"{bar} {phase}: {len(todo)} configurations to run, {len(done)} done", flush=True)
    with mp.Pool(workers) as pool, open(path, "a", encoding="utf-8") as fh:
        for d in pool.imap_unordered(_task, todo):
            fh.write(json.dumps(d) + "\n"); fh.flush()
            print(f"  {d['set']}:{d['model']}:{d['T']} {len(d['months'])} months {d['seconds']}s "
                  f"auc {np.mean([m['auc'] for m in d['months']]):.4f}", flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "run":
        only = None
        if "--only" in a:
            only = set(a[a.index("--only") + 1].split(","))
        w = int(a[a.index("--workers") + 1]) if "--workers" in a else 6
        if a[3] == "look":
            picks = json.load(open(f"{HERE}/selection.json")).get("picks", {}).get(a[2])
            if not picks:
                sys.exit("look refused: selection.json has no picks")
            only = set(picks) | {f"tb7:logreg:{p.split(':')[2]}" for p in picks}      # the picks and their baselines, nothing else
        run(a[2], a[3], w, only)

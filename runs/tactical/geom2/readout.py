"""Night experiments 1 and 2: nonlinear readouts (logreg, HistGradientBoosting, small MLP) on tb7 / rich / geometry sets.
Same 24 lab slices and the same per-slice evaluation as lab.py (lab.evaluate_slice), so numbers compare with H45.
Sets: tb7, rich, g5 (the old 5-channel geometry), gR (22 channels, crossing windows 3/5/10/20), and the unions tb7+g5, rich+g5,
tb7+gR, rich+gR. One shard = slices i::k; rows go to runs/tactical/geom2/readout_rows.jsonl (one line per slice x set x model).
usage: python readout.py --shard i --of k [--models logreg,hgb,mlp] [--sets ...]"""
import argparse, glob, json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "lab"))
import lab  # noqa: E402  (chdirs to the worktree root)
import geomlib as G  # noqa: E402

FC = os.path.join(HERE, "featcache"); os.makedirs(FC, exist_ok=True)
BASE = {"tb7": lab.fs_tb7, "rich": lab.fs_rich, "g5": G.geom_base5, "gR": G.geom_rich,
        **{f"gc{w}": (lambda W, w=w: G.geom(W, None, (w,))) for w in (3, 5, 10, 20)}}      # one crossing window each (exp 2)
SETS = ["tb7", "rich", "g5", "gR", "tb7+g5", "rich+g5", "tb7+gR", "rich+gR"]


def feats(key, name, W):
    p = f"{FC}/{key}_{name}.npy"
    if os.path.exists(p):
        return np.load(p)
    F = np.nan_to_num(BASE[name](W)).astype(np.float32); np.save(p, F); return F


def get(key, D, setname):
    parts = setname.split("+")
    tr = np.column_stack([feats(key + "_tr", n, D["Wtr"]) for n in parts]); va = np.column_stack([feats(key + "_va", n, D["Wva"]) for n in parts])
    return tr, va


def model(kind):
    if kind == "logreg":
        return LogisticRegression(C=0.1, max_iter=1000)
    if kind in ("logreg_c01", "logreg_c001"):          # stronger L2: C 0.01 / 0.001
        return LogisticRegression(C={"logreg_c01": 0.01, "logreg_c001": 0.001}[kind], max_iter=1000)
    if kind == "hgb":
        return lab.model_of("hgb", 0)
    if kind == "hgb_reg":                              # a much more regularised booster: fewer, shallower trees, big leaves
        return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=1000, l2_regularization=10.0, random_state=0)
    if kind == "mlp_reg":                              # smaller, heavily weight-decayed
        return MLPClassifier((16,), alpha=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=8, max_iter=100, random_state=0)
    return MLPClassifier((64, 32), alpha=1e-2, early_stopping=True, validation_fraction=0.15, n_iter_no_change=8, max_iter=100, random_state=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--models", default="logreg,hgb,mlp"); ap.add_argument("--sets", default=",".join(SETS)); a = ap.parse_args()
    OUT = os.path.join(HERE, f"readout_rows_{a.shard}.jsonl")
    files = sorted(glob.glob(lab.CACHE))[a.shard::a.of]; rng = np.random.default_rng(0)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l); done.add((r["slice"], r["set"], r["model"]))
    for p in files:
        D = np.load(p); key = str(D["data_end"])[:13].replace(":", ""); db = float(D["deadband"]); t0 = time.time()
        for sname in a.sets.split(","):
            Ftr, Fva = get(key, D, sname); sc = StandardScaler().fit(Ftr); Xtr, Xva = sc.transform(Ftr), sc.transform(Fva)
            for kind in a.models.split(","):
                if (key, sname, kind) in done:
                    continue
                P = np.zeros((len(Xva), 3))
                for i in range(3):
                    r = D["rtr"][:, i]; m = np.abs(r) > db
                    P[:, i] = model(kind).fit(Xtr[m], (r[m] > 0).astype(int)).predict_proba(Xva)[:, 1]
                res = lab.evaluate_slice(D, P, rng)
                open(OUT, "a", encoding="utf-8").write(json.dumps({"slice": key, "set": sname, "model": kind, "n_feat": int(Xtr.shape[1]),
                                                                  **{k: float(v) for k, v in res.items()}}) + "\n")
        print(f"{key} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

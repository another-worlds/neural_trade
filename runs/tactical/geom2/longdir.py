"""Night experiment 2c (added after exp 4): the same direction readout as readout.py, but trained on the 1-YEAR span before each lab
slice's val block (the new network's span: volbench's train span, 1-minute windows strided by 5, about 105k rows) instead of the lab's
10-day train block. Meta-labelling (exp 4) trained on years of data gained from geometry; the 10-day lab may simply be too short for
164 extra features. Val blocks, labels and the per-slice evaluation are the lab's (lab.evaluate_slice).
Sets: tb7, rich, g5, gR, tb7+g5, tb7+gR, rich+gR. Models: logreg (C 0.1), logreg_c001, hgb_reg (a regularised booster).
usage: python longdir.py --shard i --of k [--final]   (--final: the 6 held-out FINAL slices, from lab/cache_final; only for a winner's one-time check)"""
import argparse, glob, json, os, sys, time
import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(TAC, "vol")); sys.path.insert(0, os.path.join(TAC, "lab"))
import lab, readout as R, volgeom as VG, volbench as V, geomlib as G  # noqa: E402

SETS = ["tb7", "rich", "g5", "gR", "tb7+g5", "tb7+gR", "rich+gR"]
MODELS = ["logreg", "logreg_c001", "hgb_reg"]
STRIDE = 5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--final", action="store_true"); ap.add_argument("--sets", default=",".join(SETS)); ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    close, tod = V.load_minute(); A = VG.ohlcv(); assert len(A) == len(close)
    src = os.path.join(TAC, "lab", "cache_final") if a.final else V.LAB_CACHE
    files = sorted(glob.glob(src + "/*.npz"))[a.shard::a.of]; rng = np.random.default_rng(0)
    out = os.path.join(HERE, f"longdir{'_final' if a.final else ''}_rows_{a.shard}.jsonl")
    done = {(json.loads(l)["slice"], json.loads(l)["set"], json.loads(l)["model"]) for l in open(out)} if os.path.exists(out) else set()
    for p in files:
        D0 = np.load(p); key = os.path.basename(p)[:-4]; db = float(D0["deadband"]); rva = D0["rva"]; nva = len(rva); t0 = time.time()
        vs = V.locate_val(close, D0["Wva"]); avail = vs - V.GAPB - (1440 - V.WIN + 1) + 1
        n = int(min(V.SPAN, avail)); s0 = vs - V.GAPB - (n - 1)
        t_tr = s0 + V.WIN - 1 + np.arange(n); t_va = vs + V.WIN - 1 + np.arange(nva); sub = t_tr[::STRIDE]
        rtr = np.stack([close[sub + h] / close[sub] - 1.0 for h in V.HZ], 1).astype(np.float32)
        assert np.allclose(np.stack([close[t_va + h] / close[t_va] - 1.0 for h in V.HZ], 1), rva, rtol=1e-4, atol=2e-6) and sub[-1] + max(V.HZ) < vs
        Wtr, Wva = VG.windows(A, sub), D0["Wva"]
        D = dict(deadband=db, rtr=rtr, rva=rva); cache = {}
        def feat(name, W, tag):
            k = (name, tag)
            if k not in cache:
                cache[k] = {"tb7": lab.fs_tb7, "rich": lab.fs_rich, "g5": G.geom_base5, "gR": G.geom_rich}[name](W)
                cache[k] = np.nan_to_num(cache[k]).astype(np.float32)
            return cache[k]
        for sname in a.sets.split(","):
            parts = sname.split("+"); Ftr = np.column_stack([feat(x, Wtr, "tr") for x in parts]); Fva = np.column_stack([feat(x, Wva, "va") for x in parts])
            sc = StandardScaler().fit(Ftr); Xtr, Xva = sc.transform(Ftr), sc.transform(Fva)
            for kind in a.models.split(","):
                if (key, sname, kind) in done:
                    continue
                P = np.zeros((len(Xva), 3))
                for i in range(3):
                    r = rtr[:, i]; m = np.abs(r) > db
                    P[:, i] = R.model(kind).fit(Xtr[m], (r[m] > 0).astype(int)).predict_proba(Xva)[:, 1]
                res = lab.evaluate_slice(D, P, rng)
                open(out, "a", encoding="utf-8").write(json.dumps({"slice": key, "set": sname, "model": kind, "n_feat": int(Xtr.shape[1]), "n_train": int(len(sub)),
                                                                  **{k: float(v) for k, v in res.items()}}) + "\n")
        print(f"{key} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

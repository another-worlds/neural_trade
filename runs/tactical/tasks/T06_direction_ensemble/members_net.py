"""T06 members from the new network (newnet/train.py, UNCHANGED code and configuration of the logged 'verdict_try' 1y runs):
  R1y  the sklearn regression fitted on the 1-year span's inner-train part (the `linear` arch)
  P1y  PatchTST-lite on top of it (arch patch, 1y, seed 0, frozen regression, warm 0)
Per slice: P_val [N,3] and P_tail [n_tail,3] (the last 15% of the lab's 11.5-day training block; the same windows the lab members' tail uses,
located in the series). Trained from the CUT series only. GPU (one process), resumable per slice.
usage: python members_net.py --names a,b,c [--out D:/nt/nt_t06_preds]"""
import os
os.environ.setdefault("SERIES_CACHE", "D:/nt/nt_t06_cache")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import argparse, json, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TAC, "newnet")); sys.path.insert(0, os.path.join(TAC, "lab")); sys.path.insert(0, HERE)
import numpy as np
import train as T  # noqa: E402  (newnet/train.py)
import data as dm  # noqa: E402
from ens_lib import tail_rows  # noqa: E402

OUT = "D:/nt/nt_t06_preds"; LAB_CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--names", required=True); ap.add_argument("--out", default=OUT)
    ap.add_argument("--cpu", action="store_true"); a = ap.parse_args()
    device = T.setup_device(a.cpu)
    import tensorflow as tf
    import model as mdl
    os.makedirs(a.out, exist_ok=True); series = dm.Series(); assert os.environ["SERIES_CACHE"] == dm.SERIES_DIR
    for name in a.names.split(","):
        p = f"{a.out}/{name}__net.npz"
        if os.path.exists(p):
            continue
        t0 = time.time(); sp = dm.Split(name, "1y", series)
        coef, b = T.fit_linear(sp)
        fb = f"{os.environ['SERIES_CACHE']}/final_blocks/{name}.npz"
        nn = len(np.load(fb if os.path.exists(fb) else f"{LAB_CACHE}/{name}.npz")["rtr"])      # the lab block's own length (11,080 for the first slice)
        tpos = tail_rows(nn); tstarts = sp.vs - 81 - (nn - 1) + tpos      # windows of the lab block's tail, located in the series
        assert tstarts[-1] + 59 + 20 < sp.vs, "the tail's labels must end before the val block"
        Rv = T.linear_proba(sp, coef, b, sp.va_starts); Rt = T.linear_proba(sp, coef, b, tstarts)
        dmod, vmod, info = T.train_networks(tf, mdl, sp, "patch", coef, b, 0, False, 0)
        Pv = T.sigmoid(T.predict(tf, dmod, sp, sp.va_starts, "patch")); Pt = T.sigmoid(T.predict(tf, dmod, sp, tstarts, "patch"))
        np.savez(p, R1y_val=Rv.astype(np.float32), R1y_tail=Rt.astype(np.float32), P1y_val=Pv.astype(np.float32), P1y_tail=Pt.astype(np.float32),
                 rva=sp.rva, deadband=sp.deadband, n_span=sp.n, vs=sp.vs, tstarts=tstarts)
        json.dump(dict(name=name, seconds=round(time.time() - t0, 1), device=device, n_span=int(sp.n), best_epoch_dir=info["best_epoch_dir"],
                       epochs_run=info["epochs_run"]), open(f"{a.out}/{name}__net.json", "w"))
        print(name, "span", sp.n, "best", info["best_epoch_dir"], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

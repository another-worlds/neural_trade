"""T06 members L0, L5, L6, L7 (rebuild/ladder.py, UNCHANGED logic) on the 24 lab slices and the 6 FINAL slices, CPU only.
Per slice, per member: P_val [N,3] of the final model (what ladder.py scores) and P_tail [n_tail,3] of the INNER model (fitted on the first
85% of the training block, a 20-row gap before the tail; the tail = the last 15% = ladder's `iva`), for the stacker and the thresholds.
The residual steps are ladder.residual_step with the L5 stage shared (it is deterministic) and the inner model's tail predictions kept.
usage: python members_lab.py --names a,b,c [--out D:/nt/nt_t06_preds]"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, glob, json, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TAC, "rebuild")); sys.path.insert(0, HERE)
import numpy as np
import ladder  # noqa: E402  (imports lab, chdirs to the repo root)
import tensorflow as tf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

LAB_CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache")
FINAL_BLOCKS = os.environ.get("SERIES_CACHE", "D:/nt/nt_t06_cache") + "/final_blocks"
OUT = "D:/nt/nt_t06_preds"


def slice_path(name):
    p = f"{FINAL_BLOCKS}/{name}.npz"
    return p if os.path.exists(p) else f"{LAB_CACHE}/{name}.npz"


def pred(model, X):
    return ladder.predict(model, X)


def inner_model_from(F, inner, n_feat):
    model, lin = ladder.make_model(n_feat); lin.set_weights(inner); return model


def l0(S):
    """L0 (step_L0 on the full train block) and its inner model (fitted on the inner-train rows only) for the tail."""
    Pv, Pt = np.zeros((len(S["Fva"]), 3)), np.zeros((len(S["iva"]), 3))
    for i in range(3):
        m = S["M"][:, i] > 0
        Pv[:, i] = LogisticRegression(C=ladder.C0, max_iter=1000).fit(S["Ftr"][m], S["Y"][m, i].astype(int)).predict_proba(S["Fva"])[:, 1]
        mi = m.copy(); mi[:] = False; mi[S["itr"]] = m[S["itr"]]
        Pt[:, i] = LogisticRegression(C=ladder.C0, max_iter=1000).fit(S["Ftr"][mi], S["Y"][mi, i].astype(int)).predict_proba(S["Ftr"][S["iva"]])[:, 1]
    return Pv, Pt


def residual(S, l5, mlp, seq):
    """ladder.residual_step with the L5 stage handed in; returns (P_val, P_tail, best_epoch)."""
    Cs, inner, full, l5model = l5; F, Y, M, itr, iva = S["Ftr"], S["Y"], S["M"], S["itr"], S["iva"]
    Xs = [F] + ([S["Qtr"]] if seq else []); tf.keras.utils.set_random_seed(0)
    model, lin = ladder.make_model(F.shape[1], mlp=mlp, seq=seq); lin.set_weights(inner)
    be, _ = ladder.fit(model, lin, [x[itr] for x in Xs], Y[itr], M[itr], ladder.lam_vec(Cs, M[itr]), lr=ladder.NL_LR,
                       epochs=ladder.NL_EPOCHS, batch=ladder.NL_BATCH, val=([x[iva] for x in Xs], Y[iva], M[iva]),
                       patience=ladder.NL_PATIENCE, decay=False)
    Pt = pred(model, [x[iva] for x in Xs])                 # the inner model (best epoch restored; epoch 0 = the L5 inner solution)
    Xv = [S["Fva"]] + ([S["Qva"]] if seq else [])
    if be == 0:
        return pred(l5model, [S["Fva"]]), Pt, be
    tf.keras.utils.set_random_seed(0)
    model, lin = ladder.make_model(F.shape[1], mlp=mlp, seq=seq); lin.set_weights(full)
    ladder.fit(model, lin, Xs, Y, M, ladder.lam_vec(Cs, M), lr=ladder.NL_LR, epochs=be, batch=ladder.NL_BATCH, decay=False)
    return pred(model, Xv), Pt, be


def run_slice(name, out):
    t0 = time.time(); path = slice_path(name); res = {}; info = {}
    tf.keras.utils.set_random_seed(0)
    S = ladder.load_slice(path, "tb7"); Pv, Pt = l0(S); res["L0"] = (Pv, Pt); n = len(S["Ftr"])
    tail = S["iva"]
    D = S["D"]
    tf.keras.utils.set_random_seed(0)
    S = ladder.load_slice(path, "rich"); ladder.add_seq(S)
    assert len(S["Ftr"]) == n and np.array_equal(S["iva"], tail)
    l5 = ladder.l5_stage(S); Cs, inner, full, l5model = l5
    inner_m = inner_model_from(S["Ftr"], inner, S["Ftr"].shape[1])
    res["L5"] = (pred(l5model, [S["Fva"]]), pred(inner_m, [S["Ftr"][S["iva"]]]))
    info["C"] = [float(c) for c in Cs]
    for nm, mlp, seq in (("L6", True, False), ("L7", False, True)):
        Pv, Pt, be = residual(S, l5, mlp, seq); res[nm] = (Pv, Pt); info[f"best_epoch_{nm}"] = int(be)
    np.savez(f"{out}/{name}__lab.npz", rva=D["rva"], rtr=D["rtr"], deadband=float(D["deadband"]), tail=tail,
             **{f"{m}_val": v[0].astype(np.float32) for m, v in res.items()}, **{f"{m}_tail": v[1].astype(np.float32) for m, v in res.items()})
    json.dump(dict(name=name, seconds=round(time.time() - t0, 1), **info), open(f"{out}/{name}__lab.json", "w"))
    print(name, {m: round(float(np.mean(v[0])), 4) for m, v in res.items()}, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--names", required=True); ap.add_argument("--out", default=OUT); a = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4); tf.config.threading.set_inter_op_parallelism_threads(2)
    os.makedirs(a.out, exist_ok=True)
    for nm in a.names.split(","):
        if not os.path.exists(f"{a.out}/{nm}__lab.npz"):
            run_slice(nm, a.out)

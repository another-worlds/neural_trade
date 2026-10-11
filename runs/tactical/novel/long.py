"""Stage 2 (1-year spans), as runs/tactical/rebuild/longtrain.py: the 24 slices keep their val block (asserted equal to the cache); the
training span grows backwards from 81 bars before it over 365 days (fewer where the history is shorter), at most CAP rows by a stride over the
most recent window starts. tb7 vs tb7+family, logistic (C 0.1) and boosting, mean3 AUC, paired over slices.
The SPEC's trigger (tb7+family boosting gains more than +0.003 over tb7 boosting at 11.5 days) was NOT met by any family, so this run is an
EXPLORATORY addendum, labelled as such in the report; it takes no part in the verdict.
usage: python long.py slices [--workers 4] | python long.py report"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, glob, json, math, sys, time
import numpy as np
import run as R
from run import F, lab, fit_matrix, group_cols, predict, paired, holm, tci, GROUPS, lowprio
from numpy.lib.stride_tricks import sliding_window_view

CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"; HERE = R.HERE; PARTS = f"{HERE}/parts_long"
SPAN, GAPB, CAP, CHUNK, HZ = 525600, 81, 130_000, 20_000, (10, 15, 20)


def load_series():
    import pandas as pd
    return pd.read_csv(CSV, usecols=["open", "high", "low", "close", "volume"]).to_numpy(np.float64)


def windows(A, s0, n):
    return sliding_window_view(A[s0:s0 + n + 59], 60, axis=0).transpose(0, 2, 1)


def locate(A, Wva):
    w = Wva[0].astype(np.float64)
    for i in np.where(np.abs(A[:, 3] - w[0, 3]) < 1e-2)[0]:
        if i + 60 <= len(A) and np.allclose(A[i:i + 60], w, rtol=1e-5, atol=1e-6):
            return int(i)
    raise RuntimeError("val block not found in the series")


def one_slice(path):
    lowprio(); t0 = time.time(); D = np.load(path); name = os.path.basename(path)[:-4]; A = load_series()
    Wva = D["Wva"]; vs = locate(A, Wva); nva = len(Wva); db = float(D["deadband"])
    last = vs + 59 + np.arange(nva)
    assert np.allclose(np.stack([A[last + h, 3] / A[last, 3] - 1 for h in HZ], 1), D["rva"], rtol=1e-4, atol=2e-6), "rva differs"
    n = min(SPAN, vs - 80); s0 = vs - GAPB - (n - 1)
    k = max(1, math.ceil(n / CAP)); rows = np.arange(n)[::-1][::k][::-1]            # the most recent windows, strided
    tb, fam = [], {g: [] for g in F.FAMILIES}
    for i in range(0, len(rows), CHUNK):
        st = s0 + rows[i:i + CHUNK]; Wf = A[st[:, None] + np.arange(60)].astype(np.float32)
        tb.append(np.nan_to_num(lab.fs_tb7(Wf)))
        for g, v in F.compute(Wf).items():
            fam[g].append(v)
    tb = np.concatenate(tb); fam = {g: np.concatenate(v) for g, v in fam.items()}
    lt = s0 + rows + 59; rtr = np.stack([A[lt + h, 3] / A[lt, 3] - 1 for h in HZ], 1).astype(np.float32)
    assert (lt + max(HZ)).max() < vs, "train labels reach the val block"
    tb_va = np.nan_to_num(lab.fs_tb7(Wva)); fva = F.compute(Wva)
    Dd = dict(deadband=D["deadband"], rtr=rtr, rva=D["rva"]); out = dict(name=name, n_windows=int(n), n_rows=int(len(rows)), stride=int(k),
                                                                         full=bool(n == SPAN), auc={}, t_feat=round(time.time() - t0, 1))
    cfgs = [("tb7", None)] + [(f"tb7+{g}", g) for g in GROUPS]
    for tag, g in cfgs:
        if g is None:
            Xtr, Xva = fit_matrix([tb], [tb_va], 10 ** 6)
        else:
            Xtr, Xva = fit_matrix([tb, group_cols(fam, g)], [tb_va, group_cols(fva, g)], tb.shape[1])
        for kind, short in (("logreg", "lr"), ("hgb", "hgb")):
            t1 = time.time(); P = predict(Xtr, Xva, Dd, kind)
            r = lab.evaluate_slice(Dd, P, np.random.default_rng(0))
            out["auc"][f"{tag}|{short}"] = {m: float(v) for m, v in r.items()} | {"sec": round(time.time() - t1, 1)}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(PARTS, exist_ok=True); json.dump(out, open(f"{PARTS}/{name}.json", "w"))
    print(f"{name}: n {n}, rows {len(rows)} (stride {k}), {out['seconds']} s", flush=True)


def cmd_slices(workers):
    from concurrent.futures import ProcessPoolExecutor
    paths = [p for p in sorted(glob.glob(lab.CACHE)) if not os.path.exists(f"{PARTS}/{os.path.basename(p)[:-4]}.json")]
    print(len(paths), "slices to do", flush=True); t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(one_slice, paths))
    print(f"all done in {time.time() - t0:.0f} s", flush=True)


def cmd_report():
    P = [json.load(open(p)) for p in sorted(glob.glob(f"{PARTS}/*.json"))]; lines = []
    pr = lines.append
    cfg = lambda k: np.array([p["auc"][k]["auc3"] for p in P])  # noqa: E731
    pr(f"EXPLORATORY (the SPEC trigger was not met): 1-year spans, {len(P)} slices ({sum(p['full'] for p in P)} with the full 365 days), "
       f"mean rows {np.mean([p['n_rows'] for p in P]):.0f}; mean3 AUC, paired per slice vs tb7 with the same readout")
    rows = []
    for short in ("lr", "hgb"):
        b = cfg(f"tb7|{short}"); m = tci(b); pr(f"tb7 {short}: {m[0]:.4f} [{m[1]:.4f},{m[2]:.4f}]")
        sel = []
        for g in GROUPS:
            a = cfg(f"tb7+{g}|{short}"); d = paired(a, b); sel.append((g, a, d))
        adj = holm(np.array([s[2][3] for s in sel]))
        for (g, a, d), ah in zip(sel, adj):
            pr(f"  tb7+{g:4s} {short:3s} {a.mean():.4f}  d {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] p {d[3]:.4f} holm(8) {ah:.4f}")
            rows.append(dict(stage="long365_exploratory", group=g, readout=short, auc=float(a.mean()), d=d[0], lo=d[1], hi=d[2], p=d[3], p_holm8=float(ah),
                             per_slice=[round(float(x), 5) for x in a], slices=[p["name"] for p in P]))
    pr("honest top10% (mean over slices): " + "; ".join(
        f"{k} hit {np.mean([p['auc'][k]['hon10_hit'] for p in P]):.4f} bps {np.mean([p['auc'][k]['hon10_bps'] for p in P]):+.2f} "
        f"(null95 {np.mean([p['auc'][k]['hon10_null95'] for p in P]):+.2f})" for k in ("tb7|lr", "tb7|hgb", "tb7+ALL|lr", "tb7+ALL|hgb")))
    txt = "\n".join(lines); print(txt); open(f"{HERE}/report_long.txt", "w", encoding="utf-8").write(txt + "\n")
    with open(f"{HERE}/results.jsonl", "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["slices", "report"]); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cmd_slices(a.workers) if a.cmd == "slices" else cmd_report()

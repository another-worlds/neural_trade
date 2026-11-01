"""Network pipeline checks on real data (SPEC 'network' UNIT TESTS; slow, so a script, not a pytest):

1. PLANTED SIGNAL: add a component with IC ~0.1 at h = 15 to a copy of the price path (driven by a slow, causal
   feature the net sees: imb_240), train T1 fold Y2023 h = 15, and require positive held-out (2023) engine-F PnL at
   C7 costs for LIN and MLP. Failing is a bug in the pipeline or its training budget.
2. SHUFFLED RETURNS: permute 1-min returns (features unchanged, so no feature can predict them); the trained net
   must stay near cash on validation (mean |p| < 0.05).

  python checks_net.py [epochs_override]  -> results/checks_net.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

import fast
import net as NET
import prep
import train as TR

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def planted_store(base, ic=0.1, h=15, col="imb_240"):
    st = prep.Store(bars=base.b, X=base.X, names=base.names, sigma=base.sigma)
    j = st.names.index(col)
    f = np.asarray(st.X[:, j], np.float64)
    g = (f - np.median(f)) / (np.subtract(*np.percentile(f, [75, 25])) + 1e-12)
    g = np.clip(g, -5, 5)
    Rh = st.lc[h:] - st.lc[:-h]
    c = ic * np.std(Rh) / (h * np.std(g))
    inc = np.r_[0.0, c * g[:-1]]                         # increment over bar s uses g at s-1 (causal)
    st.lc = st.lc + np.cumsum(inc)
    cc = np.exp(st.lc)
    st.R1 = np.r_[cc[1:] / cc[:-1] - 1.0, 0.0]
    Rp = st.lc[h:] - st.lc[:-h]
    realised_ic = float(np.corrcoef(Rp[::h], g[:-h][::h])[0, 1])
    return st, realised_ic


def shuffled_store(base, seed=0):
    st = prep.Store(bars=base.b, X=base.X, names=base.names, sigma=base.sigma)
    r = np.diff(st.lc)
    st.lc = np.r_[st.lc[0], st.lc[0] + np.cumsum(np.random.default_rng(seed).permutation(r))]
    cc = np.exp(st.lc)
    st.R1 = np.r_[cc[1:] / cc[:-1] - 1.0, 0.0]
    return st


def heldout_pnl(st, fd, model, h):
    s, e = TR.fold_bounds(fd.fold)
    s, e = st.idx(s), st.idx(e)
    rows = np.arange(s, e)[prep.decision_mask(st.b.t[s:e], h)]
    rows = rows[rows + h < st.n]
    tau, b, _ = NET.infer(model, torch.from_numpy(fd.rows_X(rows)))
    p = fast.band_chain(tau.astype(np.float64), b.astype(np.float64), 0.0, 0.05)
    R, F, _ = st.target(rows, h)
    pp = np.r_[0.0, p[:-1]]
    dx = np.abs(p - pp)
    cost = NET.C_IN * (np.abs(p) + dx - np.abs(pp)) / 2 + NET.C_OUT * (np.abs(pp) + dx - np.abs(p)) / 2
    net = p * (R - F) - cost
    return {"net_total": float(net.sum()), "gross_total": float((p * (R - F)).sum()), "cost_total": float(cost.sum()),
            "mean_abs_p": float(np.abs(p).mean()), "turnover": float(dx.sum()), "n": int(len(p))}


def main(max_epochs=None):
    torch.set_num_threads(1)
    base = prep.Store()
    out = {}
    t0 = time.time()
    st, ic = planted_store(base)
    fd = TR.FoldData(st, 1, "Y2023", 15)
    out["planted"] = {"realised_ic": ic}
    for arch in ("LIN", "MLP"):
        model, info = TR.train_one(fd, arch, 1.0, 0, max_epochs=max_epochs, log=print)
        out["planted"][arch] = {"heldout": heldout_pnl(st, fd, model, 15), "U_val": info["U_val"],
                                "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                                "n_batches_per_epoch": info["n_batches"]}
        print(arch, out["planted"][arch], f"{time.time() - t0:.0f}s", flush=True)
    st2 = shuffled_store(base)
    fd2 = TR.FoldData(st2, 1, "Y2023", 15)
    out["shuffled"] = {}
    for arch in ("LIN", "MLP"):
        model, info = TR.train_one(fd2, arch, 1.0, 0, max_epochs=max_epochs, log=print)
        Rv, Fv, sv, dv = fd2.val
        tau, b, _ = NET.infer(model, torch.from_numpy(fd2.rows_X(fd2.val_rows)))
        U, p, _ = NET.chain_utility(tau, b, Rv, Fv, dv, fd2.rbar_val * 15)
        out["shuffled"][arch] = {"val_mean_abs_p": float(np.abs(p).mean()), "U_val": U,
                                 "heldout": heldout_pnl(st2, fd2, model, 15)}
        print(arch, out["shuffled"][arch], flush=True)
    out["pass"] = {"planted": all(out["planted"][a]["heldout"]["net_total"] > 0 for a in ("LIN", "MLP")),
                   "shuffled": all(out["shuffled"][a]["val_mean_abs_p"] < 0.05 for a in ("LIN", "MLP"))}
    out["epochs_override"] = max_epochs
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "checks_net.json"), "w"), indent=1)
    print(json.dumps(out["pass"]))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

"""Training of the PnL band nets (SPEC 'network' A): one job per (track, fold, h) trains the 4 cells
{LIN, MLP} x kappa {1, 2} x 5 seeds, early-stops each on inner-validation U, forms seed ensembles, and saves
inner-validation utilities, test-year tau/b at phase-0 decisions (dev folds) or frozen weights (fold S).

  python train.py TRACK FOLD H [--quick]      (FOLD in Y2021..Y2025, S)
  python train.py all [N_WORKERS]             (every job, a process pool of single-thread workers)
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

import folds as FO
import net as NET
import prep

GEOM = {1: (240, 1440, 64), 5: (96, 576, 64), 15: (64, 480, 64), 60: (48, 336, 64), 240: (24, 180, 64),
        1440: (14, 60, 32)}
HS = {1: (1, 5, 15), 2: (60, 240, 1440)}
ARCHS = ("LIN", "MLP")
KAPPAS = (1.0, 2.0)
SEEDS = (0, 1, 2, 3, 4)
NOISE_SD = 0.05
GROUP_P = 0.1
PATIENCE = 5
OUT = os.path.join(prep.WORK, "nets")


def fold_bounds(name):
    for f in FO.FOLDS + [FO.FOLD_S]:
        if f[0] == name:
            return f[1], f[2]
    raise KeyError(name)


class FoldData:
    """Normalised features for the whole training window of one (track, fold, h), plus label helpers."""

    def __init__(self, st, track, fold, h):
        self.st, self.track, self.fold, self.h = st, track, fold, h
        test_s, test_e = fold_bounds(fold)
        a_ts, b_ts = FO.train_range(track, test_s, h)
        (ta, tb), (va, vb) = FO.inner_split(track, test_s, h)
        self.a, self.b = st.idx(a_ts), st.idx(b_ts)
        self.tr_end = st.idx(tb)                          # training rows < tr_end (labels end >= 1 day before val)
        self.va, self.vb = st.idx(va), st.idx(vb)
        self.test = (st.idx(test_s), st.idx(test_e)) if fold != "S" else None
        self.cols = st.cols(track)
        self.gid, self.gkeys = st.group_index(self.cols)
        rows = np.arange(self.a, self.b)
        self.scaler = prep.Scaler(st.X, np.arange(self.a, self.tr_end), self.cols)   # training rows only
        self.Xn = np.empty((self.b - self.a, len(self.cols)), np.float32)
        for i in range(0, len(rows), 250_000):
            self.Xn[i:i + 250_000] = self.scaler(st.X, rows[i:i + 250_000])
        # training-window constants
        tr = np.arange(self.a, self.tr_end)
        self.rbar = float(st.R1[self.a:self.tr_end - 1].mean())                      # mean 1-min simple return
        dl = np.bincount(st.day[tr] - st.day[tr[0]], weights=np.r_[st.lc[tr[1:]] - st.lc[tr[:-1]], 0.0])
        self.s_d = float(np.std(dl[1:-1]))
        vr = np.arange(self.va, self.vb)
        self.rbar_val = float(st.R1[self.va:self.vb - 1].mean())
        self.val_rows = vr[prep.decision_mask(st.b.t[self.va:self.vb], h)]
        Rv, Fv, sv = st.target(self.val_rows, h)
        self.val = (Rv, Fv, sv, st.day[self.val_rows])

    def sample(self, rng, B, L):
        h = self.h
        hi = self.tr_end - h * L - h - 1
        r0 = rng.integers(self.a, hi, size=B)
        idx = r0[:, None] + h * np.arange(L)[None, :]
        X = self.Xn[idx - self.a]
        R, Fd, sh = self.st.target(idx.ravel(), h)
        return X, R.reshape(B, L), Fd.reshape(B, L), (sh * sh).reshape(B, L)

    def rows_X(self, rows):
        if rows.min() >= self.a and rows.max() < self.b:
            return self.Xn[rows - self.a]
        out = np.empty((len(rows), len(self.cols)), np.float32)
        for i in range(0, len(rows), 250_000):
            out[i:i + 250_000] = self.scaler(self.st.X, rows[i:i + 250_000])
        return out


def train_one(fd, arch, kappa, seed, max_epochs=None, log=None, warm_min_steps=0):
    h = fd.h
    W, T, B = GEOM[h]
    L = W + T
    spd = 1440 // h
    torch.manual_seed(seed)
    rng = np.random.default_rng(10_000 * seed + 17 * h + int(kappa))
    model = NET.PnLNet(len(fd.cols), arch)
    lr = 3e-3 if arch == "LIN" else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    n_train = (fd.tr_end - fd.a) // h
    steps = min(500_000, n_train) if h <= 15 else 100_000
    n_batches = max(1, int(round(steps / (B * T))))
    E = max_epochs or (30 if h <= 15 else 40)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(e, E) / E)))
    rbar_h = fd.rbar * h
    Rv, Fv, sv, dv = fd.val
    Xv = torch.from_numpy(fd.rows_X(fd.val_rows))
    best = (-np.inf, -1, None)
    hist = []
    gmask_keys = len(fd.gkeys)
    steps_done = 0
    first_band_ep = None
    for ep in range(E):
        model.train()
        band_on = ep >= 2 and steps_done >= warm_min_steps
        lsum = 0.0
        for _ in range(n_batches):
            steps_done += 1
            X, R, Fd, s2h = fd.sample(rng, B, L)
            X = torch.from_numpy(X)
            X = X + NOISE_SD * torch.randn_like(X)
            drop = rng.random((B, gmask_keys)) < GROUP_P
            keep = torch.from_numpy((~drop[:, fd.gid]).astype(np.float32))[:, None, :]
            X = X * keep
            loss, U, pabs, ql = NET.batch_loss(model, X, torch.from_numpy(R).float(), torch.from_numpy(Fd).float(),
                                               torch.from_numpy(s2h).float(), W, spd, kappa, rbar_h, fd.s_d, band_on)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lsum += float(loss.detach())
        sched.step()
        tau, b, aux = NET.infer(model, Xv)
        Uv, p, _ = NET.chain_utility(tau, b, Rv, Fv, dv, fd.rbar_val * h, kappa=1.0)
        hist.append({"ep": ep, "loss": lsum / n_batches, "U_val": Uv, "mean_abs_p": float(np.abs(p).mean()),
                     "turn_per_day": float(np.abs(np.diff(np.r_[0.0, p])).sum() / max(len(np.unique(dv)), 1)),
                     "b_med": float(np.median(b))})
        if Uv > best[0]:
            best = (Uv, ep, {k: v.clone() for k, v in model.state_dict().items()})
        if band_on and first_band_ep is None:
            first_band_ep = ep
        if warm_min_steps == 0:                                    # SPEC procedure (unchanged)
            if ep + 1 >= 3 and ep - best[1] >= PATIENCE:
                break
        elif first_band_ep is not None and ep - max(best[1], first_band_ep) >= PATIENCE:
            break                                                  # variant: patience starts when the band is on
    model.load_state_dict(best[2])
    tau, b, aux = NET.infer(model, Xv)
    ql_model = float(np.mean(aux + Rv ** 2 / (np.exp(aux) * sv ** 2)))
    ql_ewma = float(np.mean(Rv ** 2 / sv ** 2))
    if log:
        log(f"    {arch} k={kappa:g} s={seed}: best ep {best[1]} U_val {best[0] * 1e4:.3f} bps/d  "
            f"epochs {len(hist)}  qlike {ql_model:.4f} vs ewma {ql_ewma:.4f}")
    return model, {"U_val": best[0], "best_epoch": best[1], "epochs_run": len(hist), "hist": hist,
                   "qlike_val": ql_model, "qlike_ewma": ql_ewma, "n_batches": n_batches}


def run_job(track, fold, h, quick=False, log=print, variant=None):
    torch.set_num_threads(1)
    st = prep.Store()
    t0 = time.time()
    fd = FoldData(st, track, fold, h)
    log(f"T{track} {fold} h={h}: train rows {(fd.tr_end - fd.a):,} val decisions {len(fd.val_rows):,} "
        f"({time.time() - t0:.1f}s)")
    out_dir = OUT if variant is None else OUT + "_" + variant
    archs, warm = (ARCHS, 0) if variant is None else (("MLP",), 60)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"T{track}_{fold}_h{h}"
    Rv, Fv, sv, dv = fd.val
    if fd.test is not None:
        s, e = fd.test
        test_rows = np.arange(s, e)[prep.decision_mask(st.b.t[s:e], h)]
        Xt = torch.from_numpy(fd.rows_X(test_rows))
    res = {"track": track, "fold": fold, "h": h, "cells": {}, "rbar": fd.rbar, "s_d": fd.s_d,
           "rbar_val": fd.rbar_val, "n_train_rows": int(fd.tr_end - fd.a), "n_val": int(len(fd.val_rows)),
           "scaler": fd.scaler.state()}
    arrays = {}
    seeds = SEEDS[:2] if quick else SEEDS
    for arch in archs:
        for kappa in KAPPAS:
            cell = f"{arch}_k{kappa:g}"
            taus_v, bs_v, info, taus_t, bs_t, states = [], [], [], [], [], []
            for seed in seeds:
                model, inf = train_one(fd, arch, kappa, seed, max_epochs=3 if quick else None, log=log,
                                       warm_min_steps=warm)
                tv, bv, _ = NET.infer(model, torch.from_numpy(fd.rows_X(fd.val_rows)))
                taus_v.append(tv)
                bs_v.append(bv)
                info.append(inf)
                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                states.append(buf.getvalue())
                if fd.test is not None:
                    tt, bt, _ = NET.infer(model, Xt)
                    taus_t.append(tt)
                    bs_t.append(bt)
            tau_e, b_e = np.mean(taus_v, 0), np.mean(bs_v, 0)
            U_ens, p, Pi = NET.chain_utility(tau_e, b_e, Rv, Fv, dv, fd.rbar_val * h, kappa=1.0)
            U_seed = [NET.chain_utility(a, c, Rv, Fv, dv, fd.rbar_val * h)[0] for a, c in zip(taus_v, bs_v)]
            res["cells"][cell] = {"U_ens": U_ens, "U_seed": U_seed, "seeds": info,
                "val_mean_abs_p": float(np.abs(p).mean()),
                "val_turn_per_day": float(np.abs(np.diff(np.r_[0.0, p])).sum() / max(len(Pi), 1))}
            if fd.test is not None:
                arrays[f"{cell}_tau"] = np.mean(taus_t, 0).astype(np.float32)
                arrays[f"{cell}_b"] = np.mean(bs_t, 0).astype(np.float32)
                arrays[f"{cell}_tau_seeds"] = np.array(taus_t, np.float16)
                arrays[f"{cell}_b_seeds"] = np.array(bs_t, np.float16)
            wpath = os.path.join(out_dir, f"{tag}_{cell}.pt")
            torch.save({"states": states, "arch": arch, "kappa": kappa, "cols": fd.cols, "scaler": fd.scaler.state(),
                        "h": h, "track": track, "fold": fold}, wpath)
            res["cells"][cell]["weights"] = wpath
            res["cells"][cell]["weights_sha256"] = hashlib.sha256(open(wpath, "rb").read()).hexdigest()
            log(f"  {tag} {cell}: U_ens {U_ens * 1e4:.3f} bps/d (seeds {np.round(np.array(U_seed) * 1e4, 3)})")
    if fd.test is not None:
        arrays["test_rows"] = test_rows
        np.savez(os.path.join(out_dir, f"{tag}_test.npz"), **arrays)
    res["seconds"] = time.time() - t0
    res["variant"] = variant
    json.dump(res, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1, default=float)
    log(f"done {tag} in {res['seconds']:.0f}s")
    return res


def _worker(args):
    track, fold, h, variant = args
    tag = f"T{track}_{fold}_h{h}"
    out_dir = OUT if variant is None else OUT + "_" + variant
    if os.path.exists(os.path.join(out_dir, f"{tag}.json")):
        return tag + " (cached)"
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, f"{tag}.log"), "w")

    def log(m):
        logf.write(m + "\n")
        logf.flush()

    run_job(track, fold, h, log=log, variant=variant)
    return tag


def run_all(n_workers=3, variant=None):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    jobs = [(tr, f[0], h, variant) for tr in (1, 2) for f in FO.FOLDS + [FO.FOLD_S] for h in HS[tr]]
    jobs.sort(key=lambda j: (-{1: 6, 5: 3, 15: 2, 60: 1, 240: 1, 1440: 1}[j[2]], j[1]))   # long jobs first
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, maxtasksperchild=1) as pool:
        for tag in pool.imap_unordered(_worker, jobs):
            print(time.strftime("%H:%M:%S"), tag, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        run_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3, variant="mlpw" if "--mlpw" in sys.argv else None)
    else:
        run_job(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), quick="--quick" in sys.argv)

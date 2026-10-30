"""Timestral-LIG trainer.

  python train.py cache  --set repo          zero-shot pass with every module attached; constants; T0 on Test
  python train.py t1     --set repo --rungs A1,A2 --seeds 0,1,2,3,4
  python train.py t2     --set repo --rung A3 --seed 0
  python train.py select --set repo [--final] V2 ladder and Occam candidate; --final hashes every checkpoint
  python train.py prereg                       freezes both sets' candidates and the pass rule (before any Test inference)
  python train.py export --set repo            Test npz for exactly the hashed checkpoints, plus seed ensembles
Work files (cache, checkpoints, logs) go to $LIG_WORK/<set>/; Test npz files go to the bench results dir.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from data import CTX, PRED_LEN as PRED, Data
from indicators import GROUPS, PERIODS, Bank
from losses import TERMS, W_MEAN, Objective, agg_vs_base, block_ci, pinball_s, to_logbps
from model import CFHS, LIG

WORK = os.environ.get("LIG_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
TAG = "lig-28M_ctx2048"
T1_RUNGS = {"A1": dict(sites=("G",)), "A2": dict(sites=("H", "G")),
            "A2-plc": dict(sites=("H", "G"), placebo=True), "A2-g0": dict(sites=("H", "G"), geom_off=True),
            "A2-close": dict(sites=("H", "G"), groups_off=(9, 10)), "C-FHS": dict(cfhs=True)}
T2_RUNGS = {"A3": dict(sites=("E", "D", "H", "G"), lora=False, parent="A2"),
            "A4": dict(sites=("E", "D", "H", "G"), lora=True, parent="A2"),
            "A4L": dict(sites=("G",), lora=True, parent="A1")}


def load_ts():
    from timestral import TimestralForecaster
    return TimestralForecaster.from_pretrained("mistralai/Timestral-28M", local_files_only=True).model.eval()


def wdir(ds, *p):
    d = os.path.join(WORK, ds, *p[:-1]) if len(p) > 1 else os.path.join(WORK, ds)
    os.makedirs(d, exist_ok=True)
    return os.path.join(WORK, ds, *p)


# ------------------------------------------------------------------------------------------------ cache
def build_cache(D):
    """One no-grad pass per origin with every module attached at zero init (E, D, H, G, LoRA)."""
    torch.manual_seed(0)
    lig = LIG(load_ts(), Bank(), sites=("E", "D", "H", "G"), lora=True).eval()
    org = np.unique(np.concatenate([D.train[::2], D.v1, D.v2, D.test]))
    n = len(org)
    out = {"origins": org, "raw": np.zeros((n, 2, 15, 21), np.float32), "a_loc": np.zeros((n, 2), np.float32),
           "a_scale": np.zeros((n, 2), np.float32), "bounds": np.zeros((n, 2, 2), np.float32),
           "q0": np.zeros((n, 15, 21), np.float64)}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, 32):
            o = org[i:i + 32]
            B = len(o)
            x = D.context(o)
            X = torch.cat([x, -x])
            f = lig.features(D.raw(o), "seq")
            f_dec = lig.mirror(lig.gated(f[:, -1]))
            f_enc = lig.pool(lig.mirror(lig.gated(f)))
            raw, a_loc, a_scale, bounds = lig.backbone(X, f_enc, f_dec)
            Q = lig.sym(lig.head(raw, a_loc, a_scale, X[:, -1], f_dec, bounds)).clamp_min(0)
            for key, v in (("raw", raw), ("a_loc", a_loc), ("a_scale", a_scale), ("bounds", bounds)):
                out[key][i:i + B] = v.view(2, B, *v.shape[1:]).transpose(0, 1).numpy()
            out["q0"][i:i + B] = to_logbps(Q, torch.tensor(D.close[o])).numpy()
            if i % 3200 == 0:
                print(f"cache {D.dataset} {i}/{n} {time.time() - t0:.0f}s", flush=True)
    out["elapsed"] = time.time() - t0
    return out


def constants(D, C):
    """sigma_ref, frame_ref, P0, ln v0, initial loss weights, FHS z: all from Train origins only."""
    pos_tr = np.searchsorted(C["origins"], D.train[::2])
    assert np.array_equal(C["origins"][pos_tr], D.train[::2])
    rs = np.random.default_rng(0)
    smp = np.sort(rs.choice(D.train, min(4096, len(D.train)), replace=False))
    with torch.no_grad():
        f = Bank()(D.raw(smp), "last").numpy()
    k = {}
    k["sigma_ref"] = float(np.exp(np.median(f[:, 41])))        # ln sqrt(EMA60(u^2)) with sigma_ref = 1
    k["frame_ref"] = float(np.exp(np.median(f[:, 50])))
    y, s = D.target(D.train[::2])
    q0 = torch.from_numpy(C["q0"][pos_tr])
    k["P0"] = pinball_s(q0, y, s).mean(0).tolist()
    m = q0 @ W_MEAN
    v = torch.clamp((q0 ** 2) @ W_MEAN - m ** 2, min=1e-6)
    k["lnv0"] = torch.log(v).mean(0).tolist()
    sub = rs.choice(len(pos_tr), min(512, len(pos_tr)), replace=False)
    obj = Objective(k["P0"], k["lnv0"], {j: 1.0 for j in TERMS})
    t = obj.terms(q0[sub], y[sub], s[sub])
    k["L0"] = {j: float(t[j]) for j in t}
    lam0 = {"QL": 0.25, "fan": 1.0, "tail": 0.2}
    for j in ("term", "mcurv", "mrw"):
        lam0[j] = 0.03 * float(t["pin"]) / max(float(t[j]), 1e-12)
    k["lam0"] = lam0
    ya, sa = D.target(D.train)
    z = (ya / (sa[:, None] * torch.arange(1, 16, dtype=torch.float64).sqrt())).numpy()
    k["fhs_z"] = np.quantile(z, QL_LV, axis=0).T.tolist()        # (15, 21)
    return k


QL_LV = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


class Ctx:
    """Everything a run needs for one data set."""

    def __init__(self, ds):
        self.D = Data(ds)
        C = np.load(wdir(ds, "cache.npz"))
        self.C = {k: C[k] for k in C.files}
        self.k = json.load(open(wdir(ds, "const.json")))
        self.pos = {int(o): i for i, o in enumerate(self.C["origins"])}

    def idx(self, org):
        return np.array([self.pos[int(o)] for o in org])

    def cache(self, org):
        i = self.idx(org)
        cat = lambda a: torch.from_numpy(np.concatenate([a[i, 0], a[i, 1]]))
        close = torch.from_numpy(self.D.close[org])
        return {"raw": cat(self.C["raw"]), "a_loc": cat(self.C["a_loc"]), "a_scale": cat(self.C["a_scale"]),
                "bounds": cat(self.C["bounds"]), "xT": torch.cat([close, -close]).float()}

    def q0(self, org):
        return self.C["q0"][self.idx(org)]

    def bank(self):
        return Bank(self.k["sigma_ref"], self.k["frame_ref"])


def placebo_origins(org, rs=None, pool=None):
    """Feature origins for the placebo control.

    Training (rs given): a random Train origin per row, never one whose context window could hold the row's
    label bars (fo < o - 15 or fo > o + CTX + 15). Evaluation (rs None): a fixed half-length roll, which puts
    feature origins >= 2,160 bars away on V2 and >= 4,341 bars away on Test."""
    if rs is None:
        return np.roll(org, len(org) // 2)
    ok = lambda f: (f < org - PRED) | (f > org + CTX + PRED)
    fo = rs.choice(pool, len(org))
    bad = ~ok(fo)
    while bad.any():
        fo[bad] = rs.choice(pool, int(bad.sum()))
        bad = ~ok(fo)
    return fo


def make_model(X, rung, seed, ts=None):
    torch.manual_seed(seed)
    cfg = {**T1_RUNGS, **T2_RUNGS}[rung]
    if cfg.get("cfhs"):
        return CFHS(X.bank(), np.array(X.k["fhs_z"]))
    return LIG(ts if ts is not None else load_ts(), X.bank(), sites=cfg["sites"], lora=cfg.get("lora", False),
               groups_off=cfg.get("groups_off", ()))


def predict(X, model, org, placebo=False, full=False, export=False, chunk=512):
    """Log-bps quantiles (N, 15, 21) for origins (T1 models from the cache, full models through the backbone)."""
    out = []
    feat_org = placebo_origins(org) if placebo else org
    with torch.no_grad():
        for i in range(0, len(org), chunk if not full else 32):
            o, fo = org[i:i + chunk if not full else i + 32], feat_org[i:i + chunk if not full else i + 32]
            close = torch.from_numpy(X.D.close[o])
            if isinstance(model, CFHS):
                _, s = X.D.target(o)
                out.append(model(s, X.D.raw(fo)).numpy())
                continue
            needs_f = bool(model.sites & {"E", "D", "H"})
            rf = X.D.raw(fo) if needs_f else None
            if full:
                Q = model.forward_full(X.D.context(o), rf, export=export)
            else:
                Q = model.forward_cached(X.cache(o), rf, export=export)
            out.append(to_logbps(Q, close).numpy())
    return np.concatenate(out)


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def param_groups(model, tier):
    # cond: 62 inputs feed each per-row offset, so its lr is ~1/30 of G's to keep offset steps comparable
    lr = {"T1": dict(recal=1e-2, H=3e-4, theta=3e-3, gate=3e-3, cond=3e-4),
          "T2": dict(recal=1e-3, H=1e-4, theta=1e-3, gate=1e-3, E=2e-4, D=2e-4, lora=1e-4)}[tier]
    groups = []
    add = lambda params, l, wd: groups.append({"params": [p for p in params if p.requires_grad], "lr": l,
                                               "weight_decay": wd})
    add(model.recal.parameters(), lr["recal"], 0.0)
    add([model.bank.theta], lr["theta"], 0.0)
    add([model.gate_logit], lr["gate"], 0.0)
    if isinstance(model, CFHS):
        add(model.cond.parameters(), lr["cond"], 0.0)
    else:
        if "H" in model.sites:
            add(model.mlp_H.parameters(), lr["H"], 0.01)
        if "E" in model.sites:
            add(model.mlp_E.parameters(), lr["E"], 0.01)
        if "D" in model.sites:
            add(model.mlp_D.parameters(), lr["D"], 0.01)
        if len(model.lora):
            add(model.lora.parameters(), lr["lora"], 0.01)
    groups = [g for g in groups if g["params"]]
    return groups


def freeze_unused(model):
    """Only parameters of active sites train (a T1 rung without H keeps its periods and gates fixed)."""
    if isinstance(model, CFHS):
        return
    uses_f = bool(model.sites & {"E", "D", "H"})
    model.bank.theta.requires_grad_(uses_f)
    model.gate_logit.requires_grad_(uses_f)
    for site, mod in (("E", model.mlp_E), ("D", model.mlp_D), ("H", model.mlp_H)):
        for p in mod.parameters():
            p.requires_grad_(site in model.sites)


def schedule(opt, total, warm=50):
    f = lambda t: min(1.0, (t + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def site_usage(model, X, org):
    """RMS of the H delta (median channel and increments), G parameters, gates, periods."""
    u = {"gates": dict(zip(GROUPS, [round(float(v), 4) for v in model.gates().detach()])),
         "periods": {f"{n}#{i}": round(float(p), 3) for i, ((n, _, _), p) in enumerate(zip(PERIODS, model.bank.periods()))}}
    if isinstance(model, CFHS):
        u["recal"] = model.recal.side.detach().numpy().round(4).tolist()
        u["cond_norm"] = float(model.cond.weight.norm())
        return u
    u["recal"] = {"side": model.recal.side.detach().numpy().round(4).tolist(),
                  "rho": model.recal.rho.detach().numpy().round(4).tolist()}
    if "H" in model.sites:
        with torch.no_grad():
            f = model.mirror(model.gated(model.features(X.D.raw(org[:512]), "last")))
            dh = (model.kappa * model.mlp_H(f).view(-1, 15, 21))
        u["H_rms_median"] = float(dh[..., 10].pow(2).mean().sqrt())
        u["H_rms_incr"] = float(torch.cat([dh[..., :10], dh[..., 11:]], -1).pow(2).mean().sqrt())
    if len(model.lora):
        u["lora_BA"] = [round(float((m.B @ m.A).norm()), 5) for m in model.lora]
    return u


def model_state(model):
    """Adapter state only: the frozen backbone (ts.*, and LoRA base layers) is never saved."""
    return {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("ts.") and ".base." not in k}


def load_state(model, st):
    missing = model.load_state_dict(st, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys


# ------------------------------------------------------------------------------------------------ tier T1
def run_t1(X, rung, seed, ts, max_epochs=20, patience=3, batch=256):
    cfg = T1_RUNGS[rung]
    D = X.D
    model = make_model(X, rung, seed, ts)
    freeze_unused(model)
    obj = Objective(X.k["P0"], X.k["lnv0"], X.k["lam0"], geom_off=cfg.get("geom_off", False))
    opt = torch.optim.AdamW(param_groups(model, "T1"), betas=(0.9, 0.999), eps=1e-8)
    tr = D.train[::2]
    steps_ep = int(np.ceil(len(tr) / batch))
    sched = schedule(opt, steps_ep * max_epochs)
    rs = np.random.default_rng(1000 + seed)
    params = trainable(model)
    y2, s2 = D.target(D.v2)
    q0v2 = X.q0(D.v2)
    yv2 = y2.numpy()

    def fwd(org, rsp=None):
        fo = placebo_origins(org, rsp, D.train) if cfg.get("placebo") else org
        y, s = D.target(org)
        if isinstance(model, CFHS):
            q = model(s, D.raw(fo))
        else:
            rf = D.raw(fo) if "H" in model.sites else None
            q = to_logbps(model.forward_cached(X.cache(org), rf), torch.from_numpy(D.close[org]))
        return obj.terms(q, y, s)

    def flat(g):
        return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1).double() for gi, p in zip(g, params)])

    def evaluate():
        q = predict(X, model, D.v2, placebo=cfg.get("placebo", False))
        S, d, cells = agg_vs_base(q, q0v2, yv2)
        return S, d, cells, q

    S, _, cells, _ = evaluate()
    best = {"S": S, "epoch": 0, "state": model_state(model), "obj": copy.deepcopy(obj.state()), "cells": cells}
    log = {"rung": rung, "seed": seed, "set": D.dataset, "epochs": [{"epoch": 0, "S_val": S, **cells}]}
    t_start, step, bad = time.time(), 0, 0
    for ep in range(1, max_epochs + 1):
        perm = rs.permutation(len(tr))
        for b in range(steps_ep):
            org = np.sort(tr[perm[b * batch:(b + 1) * batch]])
            t = fwd(org, rs)
            lam_g = 1e-3 * float(np.clip((step - 150) / 350, 0, 1))
            loss = obj.total(t) + lam_g * model.gates().sum() + 1e-3 * model.bank.prior()
            if step % 25 == 0 and obj.loglam:
                gj = {j: flat(torch.autograd.grad(t[j], params, retain_graph=True, allow_unused=True)) for j in obj.loglam}
                tv = fwd(np.sort(rs.choice(D.v1, min(256, len(D.v1)), replace=False)), rs)
                gv = flat(torch.autograd.grad(tv["pin"], params, allow_unused=True))
                cos = {j: float(gv @ g / (gv.norm() * g.norm())) if gv.norm() > 0 and g.norm() > 0 else 0.0
                       for j, g in gj.items()}
                obj.govern(cos)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
        S, _, cells, _ = evaluate()
        log["epochs"].append({"epoch": ep, "S_val": S, "loss": float(loss.detach()), "lam": {j: obj.lam(j) for j in obj.loglam},
                              "time": round(time.time() - t_start, 1), **cells})
        print(f"[{D.dataset} {rung} s{seed}] ep {ep} S_val {S:+.4f} best {best['S']:+.4f}@{best['epoch']} "
              f"{time.time() - t_start:.0f}s", flush=True)
        if S < best["S"]:
            best = {"S": S, "epoch": ep, "state": model_state(model), "obj": copy.deepcopy(obj.state()), "cells": cells}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    load_state(model, best["state"])
    obj.load(best["obj"])
    S, d, cells, q = evaluate()
    assert abs(S - best["S"]) < 1e-9, (S, best["S"])
    log.update(best_epoch=best["epoch"], S_val=S, cells=cells, usage=site_usage(model, X, D.v2),
               lam_history=obj.history, time=round(time.time() - t_start, 1))
    save_run(X, rung, seed, model, obj, log, q)
    return log


def save_run(X, rung, seed, model, obj, log, qv2):
    ds = X.D.dataset
    ck = wdir(ds, "ckpt", f"{rung}_s{seed}.pt")
    torch.save({"state": model_state(model), "obj": obj.state(), "rung": rung, "seed": seed}, ck)
    np.save(wdir(ds, "v2", f"{rung}_s{seed}.npy"), qv2.astype(np.float32))
    json.dump(log, open(wdir(ds, "logs", f"{rung}_s{seed}.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ tier T2
def run_t2(X, rung, seed, max_steps=1200, eval_every=150, patience=3, batch=8):
    cfg = T2_RUNGS[rung]
    D = X.D
    model = make_model(X, rung, seed)
    parent = torch.load(wdir(D.dataset, "ckpt", f"{cfg['parent']}_s{seed}.pt"))
    load_state(model, parent["state"])
    freeze_unused(model)
    obj = Objective(X.k["P0"], X.k["lnv0"], X.k["lam0"])
    obj.load(parent["obj"])                                     # loss weights frozen from the seed's T1 run
    opt = torch.optim.AdamW(param_groups(model, "T2"), betas=(0.9, 0.999), eps=1e-8)
    sched = schedule(opt, max_steps)
    params = trainable(model)
    rs = np.random.default_rng(2000 + seed)
    v2s = D.v2[::6]
    y2, _ = D.target(v2s)
    q0 = X.q0(v2s)
    needs_f = bool(model.sites & {"E", "D", "H"})

    def evaluate(org=v2s, y=y2.numpy(), q0=q0):
        q = predict(X, model, org, full=True)
        S, d, cells = agg_vs_base(q, q0, y)
        return S, d, cells, q

    S, _, cells, _ = evaluate()
    best = {"S": S, "step": 0, "state": model_state(model), "cells": cells}
    log = {"rung": rung, "seed": seed, "set": D.dataset, "evals": [{"step": 0, "S_val_s6": S, **cells}]}
    perm = rs.permutation(len(D.train))
    t_start, bad = time.time(), 0
    for step in range(max_steps):
        org = np.sort(D.train[perm[step * batch:(step + 1) * batch]])
        y, s = D.target(org)
        Q = model.forward_full(D.context(org), D.raw(org) if needs_f else None)
        t = obj.terms(to_logbps(Q, torch.from_numpy(D.close[org])), y, s)
        loss = obj.total(t) + 1e-3 * model.gates().sum() + 1e-3 * model.bank.prior()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0:
            S, _, cells, _ = evaluate()
            log["evals"].append({"step": step + 1, "S_val_s6": S, "time": round(time.time() - t_start, 1), **cells})
            print(f"[{D.dataset} {rung} s{seed}] step {step + 1} S_val(s6) {S:+.4f} best {best['S']:+.4f}@{best['step']} "
                  f"{time.time() - t_start:.0f}s", flush=True)
            if S < best["S"]:
                best = {"S": S, "step": step + 1, "state": model_state(model), "cells": cells}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    load_state(model, best["state"])
    q = predict(X, model, D.v2, full=True)                     # full V2 for the ladder
    S, _, cells = agg_vs_base(q, X.q0(D.v2), D.target(D.v2)[0].numpy())
    log.update(best_step=best["step"], S_val=S, cells=cells, usage=site_usage(model, X, D.v2),
               time=round(time.time() - t_start, 1))
    save_run(X, rung, seed, model, obj, log, q)
    print(f"[{D.dataset} {rung} s{seed}] done: best step {best['step']}, full-V2 S_val {S:+.4f}", flush=True)
    return log


# ------------------------------------------------------------------------------------------------ selection
def runs_of(ds, rung):
    d = wdir(ds, "v2", "x")[:-1]
    return sorted(int(f.split("_s")[1][:-4]) for f in os.listdir(d) if f.startswith(rung + "_s"))


def v2q(ds, rung, seeds):
    return np.mean([np.load(wdir(ds, "v2", f"{rung}_s{s}.npy")).astype(np.float64) for s in seeds], 0)


LADDER = ["A0", "A1", "A2", "A3", "A4"]
PARENT = {"A1": "A0", "A2": "A1", "A3": "A2", "A4": "A3"}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def select(ds, final=False, force=False):
    """V2 ladder. Rule (spec 'governance'): rung a beats rung b if its V2 S improves by >= max(0.1%, 2 x a's seed SD)
    and the paired 60-min block-bootstrap CI excludes 0 (a one-seed T2 rung borrows A2's seed SD). The candidate is
    the lowest ladder rung that the best rung does not beat (Occam). Without --final it only reports
    (candidate_provisional.json); with --final it hashes every checkpoint into candidate.json, before Test is touched."""
    X = Ctx(ds)
    y = X.D.target(X.D.v2)[0].numpy()
    q0 = X.q0(X.D.v2)
    rep = {"set": ds, "final": final, "rungs": {}}
    for rung in ["A1", "A2", "A2-plc", "A2-g0", "A2-close", "C-FHS", "A3", "A4", "A4L"]:
        seeds = runs_of(ds, rung)
        if not seeds:
            continue
        per = [agg_vs_base(np.load(wdir(ds, "v2", f"{rung}_s{s}.npy")).astype(np.float64), q0, y)[0] for s in seeds]
        S, d, cells = agg_vs_base(v2q(ds, rung, seeds), q0, y)
        rep["rungs"][rung] = {"seeds": seeds, "S_ens": S, "S_seeds": per, "sd": float(np.std(per, ddof=1)) if len(per) > 1 else None,
                              "ci": block_ci(d, 60), "cells": cells}
    seeds = {r: v["seeds"] for r, v in rep["rungs"].items()}
    q_of = lambda r: q0 if r == "A0" else v2q(ds, r, seeds[r])

    def sd_of(r):
        if r == "A0":
            return 0.0
        v = rep["rungs"][r]["sd"]
        return v if v is not None else (rep["rungs"].get("A2", {}).get("sd") or 0.0)

    def beats(a, b):
        S, d, _ = agg_vs_base(q_of(a), q_of(b), y)
        lo, hi = block_ci(d, 60)
        thr = max(0.1, 2 * sd_of(a))
        return {"a": a, "b": b, "S_diff": S, "ci": [lo, hi], "threshold": thr, "beats": bool(-S >= thr and hi < 0)}

    present = [r for r in LADDER if r == "A0" or r in rep["rungs"]]
    rep["chain"] = [beats(r, PARENT[r]) for r in present if r != "A0" and PARENT[r] in present]
    S_of = {r: 0.0 if r == "A0" else rep["rungs"][r]["S_ens"] for r in present}
    best = min(present, key=lambda r: S_of[r])
    occam, cand = [], best
    for r in present:
        if r == best:
            break
        c = beats(best, r)
        occam.append(c)
        if not c["beats"]:
            cand = r
            break
    rep.update(best=best, occam=occam, candidate=cand)
    ck = wdir(ds, "ckpt", "x")[:-1]
    rep["checkpoints"] = {f[:-3]: sha256(os.path.join(ck, f)) for f in sorted(os.listdir(ck))}
    out = "candidate.json" if final else "candidate_provisional.json"
    if final:
        done = [f for f in os.listdir(X.D.results) if f.startswith(TAG)] if os.path.isdir(X.D.results) else []
        assert force or not done, f"Test npz already exist in {X.D.results}: refusing to re-register"
    json.dump(rep, open(wdir(ds, out), "w"), indent=1)
    print(f"{ds}: best {best}, candidate {cand} -> {out}")
    for c in rep["chain"] + occam:
        print(f"  {c['a']} vs {c['b']}: S_diff {c['S_diff']:+.3f} ci [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] thr {c['threshold']:.3f} beats={c['beats']}")
    for r, v in rep["rungs"].items():
        print(f"  {r:9s} S_ens {v['S_ens']:+.3f} seeds {np.round(v['S_seeds'], 3).tolist()} ci {np.round(v['ci'], 3).tolist()}")
    return rep


PASS_RULE = ("Candidate (seed ensemble) vs A0 on Test, aggregate = mean over h = 1, 5, 15, both sets required. "
             "P1 QLIKE agg <= -0.5% and CI upper < 0, no h cell significantly worse. P2 pinball21 agg CI upper <= +0.10%, "
             "no h cell point > +0.30%. P3 overlap-robust tail fails <= A0's. P4 every seed: QLIKE agg < 0 and P2. "
             "P5 candidate vs A1 QLIKE agg < 0 on both sets, CI < 0 on one; and A2 beats A2-plc on V2 (CI < 0). "
             "P6 candidate vs C-FHS QLIKE agg CI upper < 0 on both sets. P7 median MAE skill vs RW not lower than A0's by "
             "> 0.2 pp at any h. PASS = P1-P7; PASS-licence-free = P1-P5 and not P6; PASS-recal = A1 meets P1-P4 and "
             "P5 fails (common candidate A1 or higher); else FAIL. Common candidate = the lower of the two sets' candidates.")


def prereg(force=False):
    """Freeze the decision for both sets before any Test inference."""
    path = os.path.join(WORK, "preregistration.json")
    assert force or not os.path.exists(path), "already pre-registered"
    c = {ds: json.load(open(wdir(ds, "candidate.json"))) for ds in ("repo", "post")}
    assert all(v["final"] for v in c.values())
    common = min((v["candidate"] for v in c.values()), key=LADDER.index)
    rec = {"candidates": {ds: v["candidate"] for ds, v in c.items()}, "common_candidate": common,
           "candidate_json_sha256": {ds: sha256(wdir(ds, "candidate.json")) for ds in c}, "pass_rule": PASS_RULE}
    json.dump(rec, open(path, "w"), indent=1)
    print(json.dumps(rec, indent=1))


# ------------------------------------------------------------------------------------------------ export
def write_npz(X, tag, qlb, org, elapsed=0.0):
    """Log-bps quantiles (N, 15, 21) -> bench npz with price levels at h = 1, 5, 15."""
    price = np.exp(qlb / 1e4 + X.D.lc[org][:, None, None])
    np.savez(os.path.join(X.D.results, f"{tag}.npz"), q=price[:, [0, 4, 14]].astype(np.float32),
             med_path=price[:, :, 10].astype(np.float32), origins=org, framing="price", elapsed=elapsed,
             n=len(org), threads=torch.get_num_threads())


def export(ds):
    """Test inference, once, for exactly the checkpoints hashed in candidate.json (verified)."""
    X = Ctx(ds)
    pr = json.load(open(os.path.join(WORK, "preregistration.json")))
    assert sha256(wdir(ds, "candidate.json")) == pr["candidate_json_sha256"][ds], "candidate.json changed after prereg"
    cj = json.load(open(wdir(ds, "candidate.json")))
    os.makedirs(X.D.results, exist_ok=True)
    org = X.D.test
    by_rung = {}
    ts = load_ts()
    on_disk = sorted(f[:-3] for f in os.listdir(wdir(ds, "ckpt", "x")[:-1]))
    extra = sorted(set(on_disk) - set(cj["checkpoints"]))
    if extra:
        print(f"export {ds}: ignoring checkpoints not in candidate.json: {extra}")
    for name, h in cj["checkpoints"].items():
        path = wdir(ds, "ckpt", f"{name}.pt")
        assert sha256(path) == h, f"{name}: checkpoint changed after pre-registration"
        rung, seed = name.rsplit("_s", 1)
        full = rung in T2_RUNGS
        model = make_model(X, rung, int(seed), None if full else ts)
        load_state(model, torch.load(path)["state"])
        t = time.time()
        q = predict(X, model, org, placebo=T1_RUNGS.get(rung, {}).get("placebo", False), full=full, export=True)
        write_npz(X, f"{TAG}_{rung}_s{seed}", q, org, time.time() - t)
        by_rung.setdefault(rung, []).append(q)
        print(f"export {ds} {rung} s{seed}", flush=True)
    for rung, qs in by_rung.items():
        if len(qs) > 1:
            write_npz(X, f"{TAG}_{rung}_ens", np.mean(qs, 0), org)
    cand = pr["common_candidate"]
    if cand != "A0":
        src = f"{TAG}_{cand}_ens" if len(by_rung[cand]) > 1 else f"{TAG}_{cand}_s0"
        z = dict(np.load(os.path.join(X.D.results, f"{src}.npz"), allow_pickle=True))
        np.savez(os.path.join(X.D.results, f"{TAG}_candidate.npz"), **z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--set", default="repo")
    ap.add_argument("--rungs", default="A1,A2")
    ap.add_argument("--rung", default="A3")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    if a.cmd == "cache":
        D = Data(a.set)
        if os.path.exists(wdir(a.set, "cache.npz")):
            C = dict(np.load(wdir(a.set, "cache.npz")))
        else:
            C = build_cache(D)
            np.savez(wdir(a.set, "cache.npz"), **C)
        k = constants(D, C)
        json.dump(k, open(wdir(a.set, "const.json"), "w"), indent=1)
        z = np.load(os.path.join(D.bench_results, "timestral-28M_ctx2048_price.npz"), allow_pickle=True)
        k = np.searchsorted(z["origins"], D.test)
        assert np.array_equal(z["origins"][k], D.test)
        i = np.searchsorted(C["origins"], D.test)
        assert np.array_equal(C["origins"][i], D.test)
        ref = (np.log(z["q"][k].astype(np.float64)) - D.lc[D.test][:, None, None]) * 1e4
        dmax = float(np.abs(C["q0"][i][:, [0, 4, 14]] - ref).max())
        print(f"T0 {a.set}: max |diff| vs stored zero-shot npz on all {len(i)} Test origins = {dmax:.6f} bps")
        json.dump({"T0_max_diff_bps": dmax, "n": int(len(i)), "cache_s": float(C["elapsed"])}, open(wdir(a.set, "T0.json"), "w"))
        assert dmax < 1e-3, "T0 failed: nothing trains"
    elif a.cmd == "t1":
        X = Ctx(a.set)
        ts = load_ts()
        for rung in a.rungs.split(","):
            for s in [int(v) for v in a.seeds.split(",")]:
                if os.path.exists(wdir(a.set, "logs", f"{rung}_s{s}.json")):
                    continue
                run_t1(X, rung, s, ts)
    elif a.cmd == "t2":
        X = Ctx(a.set)
        if not os.path.exists(wdir(a.set, "logs", f"{a.rung}_s{a.seed}.json")):
            run_t2(X, a.rung, a.seed)
    elif a.cmd == "select":
        select(a.set, a.final, a.force)
    elif a.cmd == "prereg":
        prereg(a.force)
    elif a.cmd == "export":
        export(a.set)


if __name__ == "__main__":
    main()

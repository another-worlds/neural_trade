"""New network, training and scoring (PLAN 10).
usage: python train.py --slices 6|24 --span 11.5d|90d|1y|3y --arch patch|tcn|linear [--seeds N] [--tag name] [--cpu]
Per slice: the training span ends 81 bars before the lab cache's val block; the regression (sklearn logistic, C 0.1, one per horizon,
fitted on the inner-train part of the span) is the `linear` arch and the initial solution of the others; patch / tcn add a zero-init
residual branch (frozen regression by default: `--train-lin` fine-tunes it) trained with decoupled weight decay, dropout 0.2 and
early stopping on the inner split's direction BCE (the last 15% of the span, 80 windows apart; epoch 0 = the regression, so a branch
that never helps leaves the regression untouched). The volatility tower (log|r|, Laplace NLL) trains alongside on its own weights.
Every non-linear run also scores the regression on the same slice (suffix _lin; the paired difference is `d_auc3`), and the
tail filtered by the SAME predicted-|move| for both. One JSON line per (arch, span, seed) is appended to results.jsonl; with
--seeds N > 1 an extra line `seed: "ens"` scores the mean P of the seeds (the ensemble).
GPU: 3000 MB hard limit, below-normal priority, CPU fallback."""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lab"))
import data as dm  # noqa: E402

OUT = os.path.join(HERE, "results.jsonl")
BS, EPOCH_ROWS, MAX_EPOCHS, PATIENCE = 4096, 200_000, 40, 5
LR, VOL_LR, WD = 1e-3, 2e-3, 0.05
LIN_CAP, LIN_C = 300_000, 0.1
EVAL_CAP = 80_000


def setup_device(cpu=False, mem_mb=3000):
    """Below-normal priority; GPU with a hard memory limit unless --cpu or unavailable. Returns 'gpu' | 'cpu'."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        try:
            import neural_trade  # noqa: F401  (puts the CUDA DLLs on PATH; before tensorflow)
        except Exception:
            pass
    import tensorflow as tf
    if cpu:
        return "cpu"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return "cpu"
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mem_mb)])
        tf.config.list_logical_devices("GPU")
        return "gpu"
    except Exception as e:  # noqa: BLE001
        print("GPU unavailable, using the CPU:", e, flush=True)
        return "cpu"


# ------------------------------------------------------------------ the regression
def fit_linear(sp, cap=LIN_CAP):
    """sklearn logistic regression per horizon on the inner-train windows (a recent-aligned stride caps the rows).
    Returns coef [3, n_vec], intercept [3]; cached next to the split statistics."""
    p = f"{dm.CACHE}/{sp.name}_{sp.span}_lin.npz"
    if os.path.exists(p):
        z = np.load(p)
        if int(z["n"]) == sp.n:
            return z["coef"], z["intercept"]
    from sklearn.linear_model import LogisticRegression
    k = max(1, -(-len(sp.itr) // cap)); pos = sp.itr[::-1][::k][::-1]; st = sp.s0 + pos
    X = sp.vec(st); _, Y, M = sp.labels(st); coef = np.zeros((3, X.shape[1]), np.float32); b = np.zeros(3, np.float32)
    for h in range(3):
        m = M[:, h] > 0
        lr = LogisticRegression(C=LIN_C, max_iter=1000).fit(X[m], Y[m, h].astype(int))
        coef[h], b[h] = lr.coef_[0], lr.intercept_[0]
    os.makedirs(dm.CACHE, exist_ok=True); np.savez(p, coef=coef, intercept=b, n=sp.n)
    return coef, b


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def linear_proba(sp, coef, b, starts):
    return sigmoid(sp.vec(starts).astype(np.float64) @ coef.T.astype(np.float64) + b)


# ------------------------------------------------------------------ the networks
def inputs_of(sp, starts, arch):
    return [sp.vec(starts)] if arch == "linear" else [sp.vec(starts), sp.seq(starts)]


def predict(tf, model, sp, starts, arch, chunk=8192):
    out = []
    for i in range(0, len(starts), chunk):
        xs = [tf.constant(x) for x in inputs_of(sp, starts[i:i + chunk], arch)]
        out.append(model(xs if len(xs) > 1 else xs[0], training=False).numpy())
    return np.concatenate(out)


def masked_bce(tf, logit, y, m):
    l = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)
    return tf.reduce_sum(tf.reduce_sum(l * m, 0) / tf.maximum(tf.reduce_sum(m, 0), 1.0))


def train_networks(tf, mdl, sp, arch, coef, b, seed, train_lin=False, log=print):
    """Trains the direction model (arch != linear) and the volatility tower. Returns (dir_model, vol_model, log_b, info)."""
    tf.keras.utils.set_random_seed(seed)
    dmod = mdl.build_direction(arch, dm.N_VEC); mdl.set_linear(dmod, coef, b)
    vmod = mdl.build_volatility(dm.N_VEC, sp.mu_vol); log_b = tf.Variable(np.log(np.full(3, 0.7, np.float32)), name="log_b")
    nd, nv = mdl.n_params(dmod), mdl.n_params(vmod) + 3
    dvars = [v for v in dmod.trainable_variables if train_lin or not v.name.startswith("lin/")]
    decay = [v for v in dvars if "kernel" in v.name and not v.name.startswith("lin/")]
    vvars = vmod.trainable_variables + [log_b]
    dopt, vopt = tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(VOL_LR)

    @tf.function
    def dstep(xs, y, m):
        with tf.GradientTape() as t:
            loss = masked_bce(tf, dmod(xs if len(xs) > 1 else xs[0], training=True), y, m)
        dopt.apply_gradients(zip(t.gradient(loss, dvars), dvars))
        for v in decay:
            v.assign(v * (1.0 - LR * WD))
        return loss

    @tf.function
    def vstep(x, y):
        with tf.GradientTape() as t:
            loss = tf.reduce_sum(mdl.vol_nll(y, vmod(x, training=True), log_b))
        vopt.apply_gradients(zip(t.gradient(loss, vvars), vvars))
        return loss

    rs = np.random.RandomState(seed)
    k = max(1, -(-len(sp.iva) // EVAL_CAP)); epos = sp.iva[::k]; est = sp.s0 + epos
    (_, ev_y, ev_m) = sp.labels(est); ev_v = sp.vec(est)
    ev_logr = np.log(np.abs(sp.series.labels(est)) + 1e-5).astype(np.float32)

    def val_dir():
        return float(masked_bce(tf, tf.constant(predict(tf, dmod, sp, est, arch)), tf.constant(ev_y), tf.constant(ev_m)))

    def val_vol():
        mu = vmod(tf.constant(ev_v), training=False)
        return float(tf.reduce_sum(mdl.vol_nll(tf.constant(ev_logr), mu, log_b)))

    best_d = (val_dir() if arch != "linear" else 0.0, dmod.get_weights(), 0); best_v = (val_vol(), vmod.get_weights() + [log_b.numpy()], 0)
    d0 = best_d[0]; bad_d = bad_v = 0; ep = 0
    for ep in range(1, MAX_EPOCHS + 1):
        sel = rs.choice(len(sp.itr), min(EPOCH_ROWS, len(sp.itr)), replace=False)
        for i in range(0, len(sel) - BS + 1, BS):
            st = sp.s0 + sp.itr[sel[i:i + BS]]
            if arch != "linear" and bad_d < PATIENCE:
                _, y, m = sp.labels(st)
                dstep([tf.constant(x) for x in inputs_of(sp, st, arch)], tf.constant(y), tf.constant(m))
            if bad_v < PATIENCE:
                vstep(tf.constant(sp.vec(st)), tf.constant(np.log(np.abs(sp.series.labels(st)) + 1e-5)))
        if arch != "linear" and bad_d < PATIENCE:
            s = val_dir()
            if s < best_d[0] - 1e-6:
                best_d = (s, dmod.get_weights(), ep); bad_d = 0
            else:
                bad_d += 1
        if bad_v < PATIENCE:
            s = val_vol()
            if s < best_v[0] - 1e-6:
                best_v = (s, vmod.get_weights() + [log_b.numpy()], ep); bad_v = 0
            else:
                bad_v += 1
        if (arch == "linear" or bad_d >= PATIENCE) and bad_v >= PATIENCE:
            break
    dmod.set_weights(best_d[1]); w = best_v[1]; vmod.set_weights(w[:-1]); log_b.assign(w[-1])
    info = dict(params_dir=nd, params_vol=nv, epochs_run=ep, best_epoch_dir=best_d[2], best_epoch_vol=best_v[2],
                inner_bce_start=d0, inner_bce_best=best_d[0] if arch != "linear" else None, inner_vol_nll_best=best_v[0])
    return dmod, vmod, info


# ------------------------------------------------------------------ scoring
def spearman(a, b):
    from scipy import stats
    r = stats.spearmanr(a, b)[0]
    return float(r) if np.isfinite(r) else 0.0


def mag_tail(rva, db, P, mag, rng, pre):
    """Honest tail restricted to windows whose predicted |move| (h1) is above the median of the first half of val: thresholds from
    the first half, applied to the second half after the lab's 20-bar gap (the lab's method)."""
    n = len(P); A = np.zeros(n, bool); A[:n // 2] = True; B = np.zeros(n, bool); B[n // 2 + 20:] = True
    s = P.mean(1); conf = np.abs(s - 0.5); ret = rva[:, 1]; m = np.abs(ret) > db
    hi = mag >= np.median(mag[A]); out = {}
    for cov in (10, 5):
        thr = np.quantile(conf[A & hi], 1 - cov / 100); k = B & hi & (conf >= thr) & m
        mv = np.sign(s[k] - 0.5) * ret[k] * 1e4
        out[f"{pre}{cov}_hit"] = float(np.mean(mv > 0)) if k.sum() else float("nan")
        out[f"{pre}{cov}_bps"] = float(np.mean(mv)) if k.sum() else float("nan")
        out[f"{pre}{cov}_n"] = int(k.sum())
        if cov == 10:
            null = [np.mean(np.sign(s[k] - 0.5) * rng.choice([-1, 1], k.sum()) * ret[k] * 1e4) for _ in range(300)] if k.sum() else [float("nan")]
            out[f"{pre}10_null95"] = float(np.quantile(null, 0.95))
    return out


def score_slice(lab, sp, P, Plin, vol_pred, base_vol, rng):
    """All per-slice numbers of one (slice, arch, seed). P, Plin [N,3] P(up); vol_pred [N,3] predicted log|r|; base_vol [N]."""
    k = max(1, -(-sp.n // 100000)); rtr = sp.series.labels(sp.tr_starts[::k])
    D = dict(deadband=sp.deadband, rtr=rtr, rva=sp.rva)
    r = {kk: float(v) for kk, v in lab.evaluate_slice(D, P, rng).items()}
    rl = {kk + "_lin": float(v) for kk, v in lab.evaluate_slice(D, Plin, rng).items()}
    r.update(rl); r["d_auc3"] = r["auc3"] - r["auc3_lin"]
    r["d_ll_h1"] = r["ll_h1"] - r["ll_h1_lin"]
    absr = np.abs(sp.rva)
    for h in range(3):
        r[f"vol_rho_h{h}"] = spearman(vol_pred[:, h], absr[:, h]); r[f"vol_rho_base_h{h}"] = spearman(base_vol, absr[:, h])
    r["vol_rho"] = float(np.mean([r[f"vol_rho_h{h}"] for h in range(3)])); r["vol_rho_base"] = float(np.mean([r[f"vol_rho_base_h{h}"] for h in range(3)]))
    r["d_vol_rho"] = r["vol_rho"] - r["vol_rho_base"]
    r.update(mag_tail(sp.rva, sp.deadband, P, vol_pred[:, 1], rng, "mon"))
    r.update(mag_tail(sp.rva, sp.deadband, Plin, vol_pred[:, 1], rng, "mon_lin"))
    r.update(mag_tail(sp.rva, sp.deadband, P, base_vol, rng, "bon"))
    return r


def tci(v):
    from scipy import stats
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float(v.mean()) if len(v) else float("nan")] * 3
    m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return [round(float(m), 5), round(float(m - h), 5), round(float(m + h), 5)]


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def make_record(args, seed, names, per, extra, params, seconds, device):
    summ = {k: tci(v) for k, v in per.items() if k in (
        "auc3", "auc3_lin", "d_auc3", "ll_h1", "ll_const_h1", "d_ll_h1", "hon10_hit", "hon10_bps", "hon10_null95", "mon10_hit",
        "mon10_bps", "mon10_null95", "mon_lin10_bps", "bon10_bps", "vol_rho", "vol_rho_base", "d_vol_rho")}
    summ["ll_gap_vs_const"] = tci(np.array(per["ll_h1"]) - np.array(per["ll_const_h1"]))
    return dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), sha=git_sha(), arch=args.arch, span=args.span, seed=seed, n_slices=len(names),
                slices=names, device=device, seconds=round(seconds, 1), params=params, train_lin=args.train_lin, summary=summ,
                per_slice={k: [round(float(x), 5) for x in v] for k, v in per.items()}, slice_info=extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=6, choices=[6, 24]); ap.add_argument("--span", default="90d", choices=list(dm.SPANS))
    ap.add_argument("--arch", default="patch", choices=["patch", "tcn", "linear"]); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default=""); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--train-lin", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = setup_device(args.cpu)
    import tensorflow as tf
    import lab
    import model as mdl
    series = dm.Series(); names = dm.slice_names(args.slices); rng = np.random.default_rng(0)
    t0 = time.time(); per = {s: {} for s in range(args.seeds)}; ens_per = {}; extra = {s: [] for s in range(args.seeds)}; params = None
    times = {s: 0.0 for s in range(args.seeds)}
    for name in names:
        sp = dm.Split(name, args.span, series); coef, b = fit_linear(sp)
        Plin = linear_proba(sp, coef, b, sp.va_starts); base_vol = series.raw_vec(sp.va_starts)[:, dm.IDX_LOGVOL60]
        Ps, Vs = [], []
        for seed in range(args.seeds):
            t1 = time.time()
            dmod, vmod, info = train_networks(tf, mdl, sp, args.arch, coef, b, seed, args.train_lin)
            vol_pred = vmod(tf.constant(sp.vec(sp.va_starts)), training=False).numpy()
            P = Plin if args.arch == "linear" else sigmoid(predict(tf, dmod, sp, sp.va_starts, args.arch))
            res = score_slice(lab, sp, P, Plin, vol_pred, base_vol, rng); dt = time.time() - t1; times[seed] += dt
            for k, v in res.items():
                per[seed].setdefault(k, []).append(v)
            extra[seed].append(dict(slice=name, n_windows=sp.n, days=round(sp.n / 1440, 1), full_span=sp.full, fit_s=round(dt, 1), **info))
            params = dict(direction=info["params_dir"], volatility=info["params_vol"], total=info["params_dir"] + info["params_vol"])
            Ps.append(P); Vs.append(vol_pred)
            print(f"{args.arch} {args.span} {name} seed {seed}: auc3 {res['auc3']:.4f} lin {res['auc3_lin']:.4f} d {res['d_auc3']:+.4f} "
                  f"ll-const {res['ll_h1'] - res['ll_const_h1']:+.4f} volrho {res['vol_rho']:.3f}/{res['vol_rho_base']:.3f} "
                  f"epochs {info['epochs_run']} best {info['best_epoch_dir']} n {sp.n} {dt:.0f}s", flush=True)
        if args.seeds > 1:
            res = score_slice(lab, sp, np.mean(Ps, 0), Plin, np.mean(Vs, 0), base_vol, rng)
            for k, v in res.items():
                ens_per.setdefault(k, []).append(v)
    with open(args.out, "a", encoding="utf-8") as f:
        for seed in range(args.seeds):
            rec = make_record(args, seed, names, per[seed], extra[seed], params, times[seed], device); rec["tag"] = args.tag
            f.write(json.dumps(rec) + "\n")
            s = rec["summary"]
            print(f"== {args.arch} {args.span} seed {seed}: mean3 AUC {s['auc3'][0]:.4f} [{s['auc3'][1]:.4f},{s['auc3'][2]:.4f}]  "
                  f"d vs linear {s['d_auc3'][0]:+.4f} [{s['d_auc3'][1]:+.4f},{s['d_auc3'][2]:+.4f}]  params {params}  {rec['seconds']} s", flush=True)
        if args.seeds > 1:
            rec = make_record(args, "ens", names, ens_per, [], params, time.time() - t0, device); rec["tag"] = args.tag
            f.write(json.dumps(rec) + "\n"); s = rec["summary"]
            print(f"== ensemble of {args.seeds}: mean3 AUC {s['auc3'][0]:.4f}  d vs linear {s['d_auc3'][0]:+.4f} [{s['d_auc3'][1]:+.4f},{s['d_auc3'][2]:+.4f}]", flush=True)
    print(f"total {time.time() - t0:.0f} s on {device}")


if __name__ == "__main__":
    main()

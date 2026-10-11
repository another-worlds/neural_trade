"""Resumable chunked trainer of the arrow study (SPEC.md): one task = (model, seed, month) -> results/<model>_s<seed>_<month>.npz.
A process works until its time budget (default 100 s, wall clock incl. start-up) is nearly used, then exits: code 3 = tasks remain, 0 = done.
usage: python run.py [--part I --of N] [--budget 100] [--models full,abs,order,sign] [--seeds 0,1] [--limit K]"""
import os, sys, json, time
T_START = time.time()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arrowlib as AL

RES = f"{HERE}/results"
FIRST, LAST = "2020-07", "2025-06"            # evaluation months (calendar months wholly before the cutoff)
TRAIN_DAYS = 365
MIN_TRAIN = 8000
MAX_EPOCHS, PATIENCE, BATCH, LR = 12, 2, 256, 2e-3
GAP = pd.Timedelta(minutes=AL.L)
ORDER = ("full", "abs", "order", "sign", "full_ng")


def months():
    return [str(m) for m in pd.period_range(FIRST, LAST, freq="M")]


def month_split(t_ns, month):
    """train/val/test row masks for one evaluation month. Train windows end >= 60 min before the month starts (no shared bar with
    a test window); validation is the last 10% (by time) of the training rows, separated from the rest by 60 min."""
    t = pd.DatetimeIndex(t_ns.astype("datetime64[ns]")); m = pd.Period(month, "M"); m0, m1 = m.start_time, (m + 1).start_time
    pool = np.asarray((t >= m0 - pd.Timedelta(days=TRAIN_DAYS)) & (t <= m0 - GAP))
    te = np.asarray((t >= m0) & (t < m1))
    ids = np.nonzero(pool)[0]
    if len(ids) == 0:
        return None
    cut = ids[int(0.9 * len(ids))]
    va = pool & (np.arange(len(t)) >= cut)
    tr = pool & (t <= t[cut] - GAP)
    return np.asarray(tr), np.asarray(va), te


def tasks(models, seeds):
    return [(m, s, mo) for m in models for s in seeds for mo in months()]


def make_inputs(model_name, X):
    """forward and transformed model inputs for windows X [N, L, 5]."""
    fwd, rev, _ = AL.ALL_MODELS[model_name]
    return fwd(X), rev(X)


def fit_task(tf, model_name, seed, Xtr, Xva, Xte):
    n_ch = AL.ALL_MODELS[model_name][2]
    tf.keras.backend.clear_session(); tf.random.set_seed(seed); rs = np.random.RandomState(seed)
    mdl = AL.build_model(tf, n_ch); opt = tf.keras.optimizers.Adam(LR)
    ftr, rtr = (tf.constant(a, tf.float32) for a in make_inputs(model_name, Xtr))
    fva, rva = (tf.constant(a, tf.float32) for a in make_inputs(model_name, Xva))
    n = int(ftr.shape[0]); steps = n // BATCH

    @tf.function
    def step(idx):
        x = tf.concat([tf.gather(ftr, idx), tf.gather(rtr, idx)], 0)
        y = tf.concat([tf.ones_like(idx, tf.float32), tf.zeros_like(idx, tf.float32)], 0)
        with tf.GradientTape() as tape:
            z = tf.squeeze(mdl(x, training=True), -1)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z))
        opt.apply_gradients(zip(tape.gradient(loss, mdl.trainable_variables), mdl.trainable_variables))
        return loss

    def evaluate(f_, r_):
        zf = mdl.predict(f_, batch_size=4096, verbose=0)[:, 0]; zr = mdl.predict(r_, batch_size=4096, verbose=0)[:, 0]
        return zf, zr, float(np.mean(0.5 * (AL.bce_logits(zf, 1.0) + AL.bce_logits(zr, 0.0))))

    best, best_w, bad, ep_run = np.inf, None, 0, 0
    for ep in range(MAX_EPOCHS):
        perm = rs.permutation(n)
        for k in range(steps):
            step(tf.constant(perm[k * BATCH:(k + 1) * BATCH], tf.int32))
        ep_run = ep + 1
        _, _, vl = evaluate(fva, rva)
        if vl < best - 1e-5:
            best, best_w, bad = vl, mdl.get_weights(), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    mdl.set_weights(best_w)
    fte, rte = (tf.constant(a, tf.float32) for a in make_inputs(model_name, Xte))
    zf, zr, tl = evaluate(fte, rte)
    return zf, zr, tl, best, ep_run


def main():
    a = sys.argv
    arg = lambda k, d: a[a.index(k) + 1] if k in a else d
    part, of = int(arg("--part", 0)), int(arg("--of", 1)); budget = float(arg("--budget", 100)); limit = int(arg("--limit", 10 ** 9))
    models = arg("--models", ",".join(ORDER)).split(","); seeds = [int(x) for x in arg("--seeds", "0,1").split(",")]
    os.makedirs(RES, exist_ok=True)
    todo = [t for t in tasks(models, seeds) if not os.path.exists(f"{RES}/{t[0]}_s{t[1]}_{t[2]}.npz")][part::of]
    if not todo:
        print("all tasks done", flush=True); return 0
    tf = AL.setup_gpu(3000)
    X = np.load(f"{AL.CACHE}/X.npy", mmap_mode="r"); t = np.load(f"{AL.CACHE}/t.npy")
    assert pd.Timestamp(t.max()) <= AL.LAST_BAR
    est, done = 15.0, 0
    for (mn, seed, mo) in todo[:limit]:
        if time.time() - T_START + est * 1.3 > budget:
            break
        t0 = time.time(); sp = month_split(t, mo)
        if sp is None or sp[0].sum() < MIN_TRAIN or sp[2].sum() < 500:
            np.savez(f"{RES}/{mn}_s{seed}_{mo}.npz", skipped=np.array(1)); continue
        tr, va, te = sp
        zf, zr, tl, vl, ep = fit_task(tf, mn, seed, np.asarray(X[tr]), np.asarray(X[va]), np.asarray(X[te]))
        sec = time.time() - t0
        np.savez(f"{RES}/{mn}_s{seed}_{mo}.npz", zf=zf.astype(np.float32), zr=zr.astype(np.float32), rows=np.nonzero(te)[0].astype(np.int64),
                 test_loss=tl, val_loss=vl, epochs=ep, n_train=int(tr.sum()), n_val=int(va.sum()), seconds=sec)
        with open(f"{HERE}/run_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"model": mn, "seed": seed, "month": mo, "test_loss": tl, "ep_gain": float(np.log(2) - tl), "val_loss": vl,
                                 "epochs": ep, "n_train": int(tr.sum()), "n_test": int(te.sum()), "seconds": round(sec, 1)}) + "\n")
        est = max(est * 0.7, sec) if done else sec; done += 1
        print(f"{mn} s{seed} {mo}: ep_gain {np.log(2) - tl:+.5f} epochs {ep} {sec:.1f}s", flush=True)
    left = len([t_ for t_ in todo if not os.path.exists(f"{RES}/{t_[0]}_s{t_[1]}_{t_[2]}.npz")])
    print(f"chunk done: {done} tasks, {left} left in this part, {time.time() - T_START:.0f}s", flush=True)
    return 3 if left else 0


if __name__ == "__main__":
    sys.exit(main())

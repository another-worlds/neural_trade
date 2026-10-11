"""Monthly walk-forward for the hourly long-context study (exploratory, dev only). Uses runs/tactical/recent/wf.split_masks and its
deadband (1-bar label, 1e-3). Months 2019-01..2024-06; the hourly period from 2024-07 on is spent (H38, H56) and is dropped at load.
usage: python hwf.py run --T 365|1095 [--archs patch,tcn] [--lens 168,336] [--first 2019-01] [--last 2024-06] [--cpu]
GPU: 4000 MB cap, below-normal priority, tf.random.set_seed per month."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "recent"))
import wf  # noqa: E402
import hfeats as F  # noqa: E402

SPENT_FROM = pd.Timestamp("2024-07-01")
BAR_MIN, H = 60, 1
BS, EPOCH_ROWS, MAX_EPOCHS, PATIENCE, LR, WD = 256, 8192, 30, 5, 1e-3, 0.05
INNER_FRAC = 0.15
CLIPX = 8.0


def load():
    d = np.load(f"{F.CACHE}/h.npz")
    t = pd.DatetimeIndex(d["t"].astype("datetime64[ns]"))
    keep = np.asarray(t < SPENT_FROM - pd.Timedelta(minutes=(H + 1) * BAR_MIN))   # anchors whose label bar closes before the spent period
    out = {k: d[k][keep] for k in ("r", "anch", "vec", "tb7")}
    out["t"] = t[keep]; out["seq"] = d["seq"]
    assert out["t"].max() < SPENT_FROM
    return out


def windows(seq, anch, L):
    """[len(anch), L, C]: bars anch-L .. anch-1 (all strictly before the label bar)."""
    return seq[anch[:, None] + np.arange(-L, 0)]


def inner_split(n, L):
    """chronological inner split of n training rows: train [0, a), validation [b, n) with a gap of L rows (label/context overlap)."""
    cut = int(round((1 - INNER_FRAC) * n)); return np.arange(0, max(cut - L, 1)), np.arange(cut, n)


def setup_device(cpu=False, mem_mb=4000):
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        try:
            import neural_trade  # noqa: F401
        except Exception:
            pass
    import tensorflow as tf
    if cpu:
        return "cpu"
    try:
        g = tf.config.list_physical_devices("GPU")
        if not g:
            return "cpu"
        tf.config.set_logical_device_configuration(g[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mem_mb)])
        return "gpu"
    except Exception as e:  # noqa: BLE001
        print("GPU unavailable:", e, flush=True); return "cpu"


def fit_net(arch, L, Vtr, Str, ytr, Vte, Ste, seed):
    """Residual branch trained on the inner-train part, early stopping on the inner validation BCE (epoch 0 = the logistic start).
    Returns (test logits at the best epoch, test logits at the start, info)."""
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    import hmodel
    a, b = inner_split(len(ytr), L)
    lr0 = LogisticRegression(C=0.1, max_iter=1000).fit(Vtr[a], ytr[a])
    tf.keras.backend.clear_session(); tf.random.set_seed(seed); rng = np.random.default_rng(seed)
    m = hmodel.build(arch, Vtr.shape[1], Str.shape[2], L); hmodel.set_linear(m, lr0.coef_, lr0.intercept_)
    train_vars = m.trainable_weights; opt = tf.keras.optimizers.Adam(LR)
    decay = [v for v in train_vars if len(v.shape) >= 2]

    @tf.function(reduce_retracing=True)
    def step(v, s, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=m([v, s], training=True)[:, 0]))
        opt.apply_gradients(zip(tape.gradient(loss, train_vars), train_vars))
        for w in decay:
            w.assign(w * (1.0 - LR * WD))
        return loss

    def predict(V, S):
        return np.concatenate([m([V[i:i + 2048], S[i:i + 2048]], training=False).numpy()[:, 0] for i in range(0, len(V), 2048)])

    yb = ytr[b]

    def val_bce():
        z = predict(Vtr[b], Str[b])
        return float(np.mean(np.maximum(z, 0) - z * yb + np.log1p(np.exp(-np.abs(z)))))

    start_te = predict(Vte, Ste)
    best = v0 = val_bce(); best_w, best_ep, bad = m.get_weights(), 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        idx = rng.permutation(len(a))[:EPOCH_ROWS]
        for j in range(0, len(idx) - BS + 1, BS):
            q = a[idx[j:j + BS]]
            step(Vtr[q], Str[q], ytr[q].astype(np.float32))
        v = val_bce()
        if v < best - 1e-5:
            best, best_w, best_ep, bad = v, m.get_weights(), ep, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.set_weights(best_w)
    return predict(Vte, Ste), start_te, {"best_epoch": best_ep, "epochs_run": ep, "val_bce0": v0, "val_bce_best": best,
                                        "n_params": hmodel.n_params(m)}


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def walk_forward(D, T, first, last, nets=(), seed0=0, log=print, fit_net_fn=None):
    """One dict per month: AUC of tb7 logistic, ctx logistic, each net (and its logistic start), the ensembles.
    `nets` = [(arch, L), ...]. Training rows: split_masks (label bar closed by the month start, at most T days old) and the deadband."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    fit_net_fn = fit_net_fn or fit_net
    t, r = D["t"], D["r"]; db = wf.deadband("1h"); mask = np.abs(r) > db; y = (r > 0).astype(int); out = []
    for k, m in enumerate(pd.period_range(first, last, freq="M")):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = wf.split_masks(t, m0, m1, H, BAR_MIN, T)
        trm = tr & mask
        if trm.sum() < wf.MIN_TRAIN or (te & mask).sum() < wf.MIN_TEST or len(np.unique(y[trm])) < 2:
            continue
        itr, ite = np.flatnonzero(trm), np.flatnonzero(te); tem = mask[ite]; yt = y[ite][tem]
        if len(np.unique(yt)) < 2:
            continue
        row = {"month": str(m), "n_train": int(len(itr)), "n_test": int(tem.sum())}; P = {}
        for name, X in (("tb7", D["tb7"]), ("ctx", D["vec"])):
            sc = StandardScaler().fit(X[itr])
            f = lambda Z, sc=sc: np.clip(sc.transform(Z), -CLIPX, CLIPX).astype(np.float32)
            P[name] = LogisticRegression(C=0.1, max_iter=1000).fit(f(X[itr]), y[itr]).predict_proba(f(X[ite]))[:, 1]
            if name == "ctx":
                fv = f
        for (arch, L) in nets:
            t0 = time.time()
            Vtr, Vte = fv(D["vec"][itr]), fv(D["vec"][ite])
            Str = np.clip(windows(D["seq"], D["anch"][itr], L), -CLIPX, CLIPX); Ste = np.clip(windows(D["seq"], D["anch"][ite], L), -CLIPX, CLIPX)
            z, z0, info = fit_net_fn(arch, L, Vtr, Str, y[itr], Vte, Ste, seed0 + k)
            P[f"{arch}{L}"] = sig(z); P[f"{arch}{L}_start"] = sig(z0); row[f"{arch}{L}_info"] = {**info, "seconds": round(time.time() - t0, 1)}
        for L in sorted({L for _, L in nets}):
            ks = [f"{a}{L}" for a, LL in nets if LL == L]
            if len(ks) > 1:
                P[f"ens{L}"] = np.mean([P[x] for x in ks], 0)
        allk = [f"{a}{L}" for a, L in nets]
        if len(allk) > 1:
            P["ens_all"] = np.mean([P[x] for x in allk], 0)
        for name, p in P.items():
            row[f"auc_{name}"] = float(roc_auc_score(yt, p[tem]))
        out.append(row); log(json.dumps(row))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--archs", default="patch,tcn"); ap.add_argument("--lens", default="168,336"); ap.add_argument("--first", default=None)
    ap.add_argument("--last", default="2024-06"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    first = a.first or ("2019-01" if a.T <= 365 else "2020-03")           # a 3-year window needs 3 years of data (it starts 2017-02)
    assert pd.Period(a.last, "M") <= pd.Period("2024-06", "M")
    print("device", setup_device(a.cpu), flush=True)
    D = load(); nets = [(x, int(L)) for L in a.lens.split(",") for x in a.archs.split(",")] if a.archs else []
    tag = a.tag or f"T{a.T}"; path = f"{HERE}/walk_{tag}.jsonl"
    done = {json.loads(l)["month"] for l in open(path)} if os.path.exists(path) else set()
    fh = open(path, "a", encoding="utf-8")
    for m in pd.period_range(first, a.last, freq="M"):
        if str(m) in done:
            continue
        rows = walk_forward(D, a.T, str(m), str(m), nets, seed0=int(m.year * 12 + m.month), log=lambda s: print(s, flush=True))
        for r_ in rows:
            fh.write(json.dumps(r_) + "\n"); fh.flush()

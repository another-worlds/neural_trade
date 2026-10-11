"""Experiment 5: fine-tune a pretrained JEPA encoder end-to-end for direction and for volatility, per cached slice.
Direction model: logit[h] = Dense(3)(tb7 features, standardised)  +  Dense(3, zero-init)(dropout(embedding))
  the linear part starts at sklearn's logistic solution (C 0.1, deadband-masked), so at init the model IS the tb7 regression.
Volatility model: pred log(|r|+1e-5) = Dense(3)(tb7) + Dense(3, zero-init)(dropout(embedding)); the linear part starts at sklearn's Ridge(1).
Training on the slice's train block: the first 85% fit, a gap of 80 windows, the last 15% is the inner split for early stopping (best epoch
including epoch 0 = the initial regression; patience 3; 15 epochs; batch 256). Learning rates fixed before any run: encoder 3e-4, residual head 1e-3,
linear part 1e-4. Scoring on the slice's val block (never used for stopping): direction mean3 AUC (lab.evaluate_slice) against the tb7 logreg fitted on
the whole train block (the probe's baseline), paired over slices with a 95% t-CI; volatility Spearman of the predicted log|r| against |r| per horizon
(mean) against HAR-RV and HAR-RV+tod fitted on the SAME train block (volbench.minute_forecasts), tb7 ridge, and the frozen-embedding probe is in probe4.
usage: python finetune4.py --ckpt e1 --tag ft_e1 [--ref_cut ...]   (CPU)"""
import argparse, glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "v2"), os.path.join(HERE, "..", "..", "lab"), os.path.join(HERE, "..", "..", "vol")):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
GAP, EPOCHS, PATIENCE, BS = 80, 15, 3, 256
LR_ENC, LR_RES, LR_LIN = 3e-4, 1e-3, 1e-4


def build(enc_factory, n_feat, lin_w, lin_b):
    """returns (encoder, lin, res, forward). lin_w [n_feat, 3], lin_b [3]."""
    import tensorflow as tf
    enc = enc_factory()
    lin = tf.keras.layers.Dense(3); lin.build((None, n_feat)); lin.set_weights([np.asarray(lin_w, np.float32), np.asarray(lin_b, np.float32)])
    res = tf.keras.layers.Dense(3, kernel_initializer="zeros", bias_initializer="zeros"); res.build((None, enc.d))
    drop = tf.keras.layers.Dropout(0.2)

    def forward(x_tb, x_ctx, training=False):
        return lin(x_tb) + res(drop(enc(x_ctx), training=training))
    return enc, lin, res, forward


def fit_head(enc_factory, loss_fn, Xtb, Xc, Y, M, lin_w, lin_b, seed=0):
    """fine-tune; returns forward (best epoch restored) and the inner-val curve. Y [n,3], M [n,3] weights (deadband mask for direction)."""
    import tensorflow as tf
    tf.random.set_seed(seed)
    n = len(Xtb); cut = int(round(0.85 * n)); tr = np.arange(0, cut - GAP); va = np.arange(cut, n)
    enc, lin, res, fwd = build(enc_factory, Xtb.shape[1], lin_w, lin_b)
    groups = [(enc.trainable_variables, tf.keras.optimizers.Adam(LR_ENC, clipnorm=5.0)), (res.trainable_variables, tf.keras.optimizers.Adam(LR_RES)),
              (lin.trainable_variables, tf.keras.optimizers.Adam(LR_LIN))]

    @tf.function
    def step(xt, xc, y, m):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, fwd(xt, xc, True), m)
        gs = tape.gradient(loss, [v for vs, _ in groups for v in vs]); k = 0
        for vs, opt in groups:
            opt.apply_gradients(zip(gs[k:k + len(vs)], vs)); k += len(vs)
        return loss

    def val_loss():
        tot = 0.0
        for s in range(0, len(va), 2048):
            j = va[s:s + 2048]; tot += float(loss_fn(Y[j], fwd(Xtb[j], Xc[j]), M[j])) * len(j)
        return tot / len(va)
    best = (val_loss(), 0, [v.numpy() for v in enc.variables + lin.variables + res.variables]); curve = [best[0]]; rng = np.random.default_rng(seed)
    allv = lambda: enc.variables + lin.variables + res.variables
    for ep in range(1, EPOCHS + 1):
        perm = rng.permutation(tr)
        for s in range(0, len(perm) - BS + 1, BS):
            j = perm[s:s + BS]; step(Xtb[j], Xc[j], Y[j], M[j])
        v = val_loss(); curve.append(v)
        if v < best[0] - 1e-5: best = (v, ep, [x.numpy() for x in allv()])
        elif ep - best[1] >= PATIENCE: break
    for x, w in zip(allv(), best[2]): x.assign(w)
    return fwd, dict(best_epoch=best[1], curve=[round(c, 5) for c in curve])


def predict(fwd, Xtb, Xc):
    out = [fwd(Xtb[s:s + 2048], Xc[s:s + 2048]).numpy() for s in range(0, len(Xtb), 2048)]
    return np.concatenate(out)


def bce(y, z, m):
    import tensorflow as tf
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z) * m) / tf.maximum(tf.reduce_sum(m), 1.0)


def huber_loss(y, p, m):
    import tensorflow as tf
    return tf.reduce_mean(tf.keras.losses.huber(y[..., None], p[..., None], delta=1.0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--cutoff", default=None); ap.add_argument("--max_slices", type=int, default=99); ap.add_argument("--results", default=HERE + "/results_ft.jsonl")
    a = ap.parse_args()
    import neural_trade  # noqa: F401
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4); tf.config.threading.set_inter_op_parallelism_threads(2)
    import lab, probe as P1, volbench as VB
    import probe4 as PR
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from channels import context_channels
    import pandas as pd
    ec, std, meta, ctx = PR.load_enc(a.ckpt)
    import model4 as M4
    arch = meta.get("arch", dict(d=32, heads=4, ffn=128, layers=2, max_tok=12)); w0 = ec.get_weights()

    def enc_factory():
        e = M4.Encoder4(5, **arch); e(np.zeros((2, ctx, 5), np.float32)); e.set_weights(w0); return e
    cutoff = np.datetime64(a.cutoff) if a.cutoff else np.datetime64((pd.Timestamp(meta["span"][1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00"))
    ts, bars = PR.file_ohlcv(); close = bars[:, 3]
    tod = ((pd.to_datetime(ts).hour * 60 + pd.to_datetime(ts).minute).to_numpy(np.float64))
    lr_ = np.concatenate([[0.0], np.diff(np.log(np.maximum(close, 1e-9)))]); cs = VB.cum_sq(lr_)
    rng = np.random.default_rng(0)
    rows = []; t0 = time.time()
    for path in sorted(glob.glob(P1.CACHE)):
        D = np.load(path); i = P1.locate(close, D["Wtr"]); iv = P1.locate(close, D["Wva"])
        if i is None or iv is None or ts[i] < cutoff or i < 1500 or iv < 1500 or len(rows) >= a.max_slices: continue
        key = str(D["data_end"])[:13]; db = float(D["deadband"]); rtr, rva = D["rtr"].astype(np.float64), D["rva"].astype(np.float64)
        ntr, nva = len(rtr), len(rva)
        T_tb, V_tb = np.nan_to_num(lab.fs_tb7(D["Wtr"])), np.nan_to_num(lab.fs_tb7(D["Wva"]))
        sc = StandardScaler().fit(T_tb); Xt, Xv = sc.transform(T_tb).astype(np.float32), sc.transform(V_tb).astype(np.float32)

        def ctx_in(i0, n):
            out = []
            for s in range(0, n, 4096):
                j = np.arange(s, min(s + 4096, n)); W = bars[(i0 + j + 60)[:, None] + np.arange(-ctx, 0)]
                out.append(std(context_channels(W)[0]))
            return np.concatenate(out)
        Ct, Cv = ctx_in(i, ntr), ctx_in(iv, nva)
        row = dict(slice=key, train_start=str(ts[i])[:16], n_train=ntr, n_val=nva)
        # ---- direction. baseline = the probe's tb7 logreg on the whole train block
        P_base = P1.logit_probs(*P1.standardise(T_tb, V_tb), rtr, db)
        auc_base = lab.evaluate_slice(D, P_base, rng)["auc3"]
        cut = int(round(0.85 * ntr)); itr = np.arange(0, cut - GAP)
        coef, icp = np.zeros((Xt.shape[1], 3)), np.zeros(3)
        for h in range(3):
            m = np.abs(rtr[itr, h]) > db
            lg = LogisticRegression(C=0.1, max_iter=1000).fit(Xt[itr][m], (rtr[itr][m, h] > 0).astype(int)); coef[:, h], icp[h] = lg.coef_[0], lg.intercept_[0]
        Y = (rtr > 0).astype(np.float32); Mk = (np.abs(rtr) > db).astype(np.float32)
        fwd, info = fit_head(enc_factory, bce, Xt, Ct, Y, Mk, coef, icp)
        P_ft = 1 / (1 + np.exp(-predict(fwd, Xv, Cv)))
        auc_ft = lab.evaluate_slice(D, P_ft, rng)["auc3"]
        row.update(auc_base=float(auc_base), auc_ft=float(auc_ft), dir_best_epoch=info["best_epoch"], dir_curve=info["curve"])
        # ---- volatility. targets log(|r|+1e-5); HAR-RV on the same train block
        Yv = np.log(np.abs(rtr) + 1e-5).astype(np.float32); Mv = np.ones_like(Yv)
        rg = Ridge(alpha=1.0).fit(Xt[itr], Yv[itr])
        fwd2, info2 = fit_head(enc_factory, huber_loss, Xt, Ct, Yv, Mv, rg.coef_.T, rg.intercept_)
        pv = predict(fwd2, Xv, Cv); pr = rg.predict(Xv)
        # variant B: the linear part also reads the four HAR log-variances (15/60/240/1440 bars): the model starts at "ridge on tb7 + HAR"
        tt, tv = i + 59 + np.arange(ntr), iv + 59 + np.arange(nva)
        hl = lambda t: np.stack([np.log(VB.trailing_var(cs, k)[t] + 1e-12) for k in VB.HAR_LAGS], 1)
        sh = StandardScaler().fit(hl(tt)); XtB = np.hstack([Xt, sh.transform(hl(tt))]).astype(np.float32); XvB = np.hstack([Xv, sh.transform(hl(tv))]).astype(np.float32)
        rgB = Ridge(alpha=1.0).fit(XtB[itr], Yv[itr])
        fwd3, info3 = fit_head(enc_factory, huber_loss, XtB, Ct, Yv, Mv, rgB.coef_.T, rgB.intercept_)
        pvB = predict(fwd3, XvB, Cv); prB = rgB.predict(XvB)
        out_va, _, _ = VB.minute_forecasts(close, tod, lr_, cs, iv, nva, i, ntr, rtr, rva, None)
        sp = lambda S: float(np.mean([VB.spearman(S[:, h], np.abs(rva[:, h])) for h in range(3)]))
        row.update(vol_ft=sp(pv), vol_ridge_tb7=sp(pr), vol_har=sp(out_va["har"][0]), vol_har_tod=sp(out_va["har_tod"][0]),
                   vol_best_epoch=info2["best_epoch"], vol_curve=info2["curve"],
                   volB_ft=sp(pvB), volB_ridge=sp(prB), volB_best_epoch=info3["best_epoch"], volB_curve=info3["curve"])
        rows.append(row)
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k.startswith(("auc", "vol", "dir_b")) and "curve" not in k}, f"{time.time() - t0:.0f}s", flush=True)
    ci = P1.ci; col = lambda k: np.array([r[k] for r in rows])
    res = dict(tag=a.tag, ckpt=a.ckpt, pretrain=meta, n_slices=len(rows), cutoff=str(cutoff), rows=rows,
               dir=dict(base=ci(col("auc_base")), ft=ci(col("auc_ft")), d_ft_vs_base=ci(col("auc_ft") - col("auc_base"))),
               vol=dict(ft=ci(col("vol_ft")), ridge_tb7=ci(col("vol_ridge_tb7")), har=ci(col("vol_har")), har_tod=ci(col("vol_har_tod")),
                        d_ft_vs_har_tod=ci(col("vol_ft") - col("vol_har_tod")), d_ft_vs_har=ci(col("vol_ft") - col("vol_har")),
                        d_ft_vs_ridge=ci(col("vol_ft") - col("vol_ridge_tb7")), d_ridge_vs_har_tod=ci(col("vol_ridge_tb7") - col("vol_har_tod")),
                        B_ft=ci(col("volB_ft")), B_ridge=ci(col("volB_ridge")), d_Bft_vs_har_tod=ci(col("volB_ft") - col("vol_har_tod")),
                        d_Bft_vs_Bridge=ci(col("volB_ft") - col("volB_ridge")), d_Bridge_vs_har_tod=ci(col("volB_ridge") - col("vol_har_tod"))))
    open(a.results, "a", encoding="utf-8").write(json.dumps(res) + "\n")
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
    print(f"\n{a.tag}: {len(rows)} slices\nDIRECTION mean3 AUC: tb7 {f(res['dir']['base'])} | fine-tuned {f(res['dir']['ft'])} | d {f(res['dir']['d_ft_vs_base'])}")
    v = res["vol"]
    print(f"VOL Spearman: ft {f(v['ft'])} | tb7 ridge {f(v['ridge_tb7'])} | HAR {f(v['har'])} | HAR+tod {f(v['har_tod'])}\n  d ft-HARtod {f(v['d_ft_vs_har_tod'])} | d ft-HAR {f(v['d_ft_vs_har'])} | d ft-ridge {f(v['d_ft_vs_ridge'])} | d ridge-HARtod {f(v['d_ridge_vs_har_tod'])}")
    print(f"VOL B (linear part = tb7 + HAR log-variances): ft {f(v['B_ft'])} | ridge {f(v['B_ridge'])} | d ft-HARtod {f(v['d_Bft_vs_har_tod'])} | d ft-ridgeB {f(v['d_Bft_vs_Bridge'])} | d ridgeB-HARtod {f(v['d_Bridge_vs_har_tod'])}")


if __name__ == "__main__":
    main()

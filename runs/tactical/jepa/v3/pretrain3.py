"""JEPA v3 (long) pretraining: v2 pretrain2.py plus --save_at intermediate checkpoints; JEPA v2 pretraining on 2017-01-01..2019-12-31 only, no labels.
usage: python pretrain2.py --variant {ctl,a,b,c} --steps 40000 [--bs 256]
  ctl: v1's task (next 15 bars, one segment) in this pipeline - the control
  a:   next 60 bars as 3 target segments of 20, predictor conditioned on a learned segment token
  b:   hide 4-6 of 12 context patches (block masks); predict the target encoder's embeddings of the hidden patches from the
       visible ones; the target encoder sees the full context (I-JEPA style)
  c:   a + b (sum of both losses)
Every variant also has VICReg (variance 5, covariance 1) on the full-context pooled embedding, EMA momentum 0.996.
GPU: tf.random.set_seed is required (TF_DETERMINISTIC_OPS=1 refuses unseeded random ops, the v1 failure); memory capped (--mem, MB;
2 processes x 2000 = the 4000 MB total); below-normal process priority. CPU fallback: CUDA_VISIBLE_DEVICES=-1."""
import argparse, ctypes, json, os, queue, sys, threading, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.join(HERE, "..", "v2")
sys.path.insert(0, V2); sys.path.insert(0, os.path.join(V2, ".."))
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
SPAN = ("2017-01-01", "2019-12-31 23:59")
SPEC = {"ctl": dict(seg=15, n_seg=1, fut=False, mask=False), "a": dict(seg=20, n_seg=3, fut=True, mask=False),
        "b": dict(seg=20, n_seg=3, fut=False, mask=True), "c": dict(seg=20, n_seg=3, fut=True, mask=True)}
SPEC["ctl"]["fut"] = True


def load_span():
    d = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    d = d[(d.timestamp >= SPAN[0]) & (d.timestamp <= SPAN[1])]
    return d.timestamp.values, d[["open", "high", "low", "close", "volume"]].values.astype(np.float64)


def valid_anchors(bars, horizon):
    n = len(bars); a = np.arange(60, n - horizon); c = bars[:, 3]
    return a[np.abs(np.log(c[a - 1] / c[a - 60])) > 0]


def make_batch(bars, idx, std, sp, mask_pool, rng):
    from channels2 import context_channels, context_channels_masked, segment_channels
    ci = idx[:, None] + np.arange(-60, 0)
    W = bars[ci]
    full, vol, vm = context_channels(W)
    out = {"ctx": std(full)}
    if sp["fut"]:
        n = sp["seg"] * sp["n_seg"]
        F = bars[idx[:, None] + np.arange(n)]
        out["seg"] = std(segment_channels(F, bars[idx - 1, 3], vol, vm, sp["seg"]))
    if sp["mask"]:
        hid = mask_pool[rng.integers(0, len(mask_pool), len(idx))]
        out["mctx"] = std(context_channels_masked(W, hid)); out["vis"] = ~hid
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True, choices=list(SPEC))
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--mem", type=int, default=2000); ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_at", default="30000,60000,120000")
    a = ap.parse_args(); sp = SPEC[a.variant]; out = a.out or f"{HERE}/ckpt/{a.variant}"
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except Exception: pass
    import neural_trade  # noqa: F401 (puts the CUDA DLLs on PATH)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU"); dev = "cpu"
    if gpus and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=a.mem)]); dev = "gpu"
    tf.random.set_seed(a.seed)
    from channels2 import N_CH, Standardiser, block_masks, context_channels
    import model2 as M
    ts, bars = load_span(); H = sp["seg"] * sp["n_seg"]; anchors = valid_anchors(bars, H)
    rng = np.random.default_rng(a.seed)
    print(f"variant {a.variant} device {dev}: {len(bars)} bars, {len(anchors)} valid anchors", flush=True)
    std = Standardiser.fit(context_channels(bars[rng.choice(anchors, 20000, replace=False)[:, None] + np.arange(-60, 0)])[0])
    mask_pool = block_masks(np.random.default_rng(1), 8192) if sp["mask"] else None
    enc, tgt = M.Encoder2(N_CH), M.Encoder2(N_CH)
    x0 = tf.zeros((2, 60, N_CH)); enc(x0); tgt(x0); tgt(tf.zeros((2, sp["seg"], N_CH)))
    tgt.set_weights(enc.get_weights())
    mp = M.MaskPredictor() if sp["mask"] else None
    sg = M.SegPredictor(sp["n_seg"]) if sp["fut"] else None
    if mp: mp(tf.zeros((2, 12, M.D)), tf.ones((2, 12), tf.bool))
    if sg: sg(tf.zeros((2, M.D)))
    train_vars = enc.trainable_variables + (mp.trainable_variables if mp else []) + (sg.trainable_variables if sg else [])
    n_par = int(sum(np.prod(w.shape) for w in enc.trainable_weights))
    n_all = int(sum(np.prod(w.shape) for w in train_vars))
    print(f"encoder params {n_par}, trained params {n_all}", flush=True)
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, a.steps, alpha=0.05)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=5.0); MOM = 0.996
    huber = lambda t, p: tf.keras.losses.huber(tf.stop_gradient(t), p, delta=1.0)   # mean over D, [..] per position

    @tf.function
    def step(ctx, seg, mctx, vis):
        zero = tf.constant(0.0)
        with tf.GradientTape() as tape:
            z = enc(ctx, training=True)
            lf = lm = zero
            if sp["fut"]:
                t = tgt(tf.reshape(seg, (-1, sp["seg"], N_CH))); t = tf.reshape(t, (-1, sp["n_seg"], M.D))
                lf = tf.reduce_mean(huber(t, sg(z)))
            if sp["mask"]:
                tt = tgt.tokens_out(ctx)                                       # target encoder sees the full context
                pr = mp(enc.tokens_out(mctx, vis), vis)
                hid = tf.cast(~vis, tf.float32)
                lm = tf.reduce_sum(huber(tt, pr) * hid) / tf.maximum(tf.reduce_sum(hid), 1.0)
            var, cov = M.vicreg(z)
            loss = lf + lm + 5.0 * var + 1.0 * cov
        opt.apply_gradients(zip(tape.gradient(loss, train_vars), train_vars))
        M.ema_update(tgt, enc, MOM)
        return tf.stack([loss, lf, lm, var, cov])

    @tf.function
    def diag(ctx):
        z = enc(ctx, training=False); s = tf.math.reduce_std(z, 0)
        return tf.stack([tf.reduce_mean(s), tf.reduce_min(s), M.effective_rank(z)])

    zc, zs, zm, zv = (tf.zeros((a.bs, 60, N_CH)), tf.zeros((a.bs, sp["n_seg"], sp["seg"], N_CH)), tf.zeros((a.bs, 60, N_CH)),
                      tf.ones((a.bs, 12), tf.bool))
    q = queue.Queue(maxsize=16); stop = threading.Event()

    def producer():
        r = np.random.default_rng(a.seed + 7)
        while not stop.is_set():
            b = make_batch(bars, r.choice(anchors, a.bs), std, sp, mask_pool, r)
            q.put({k: tf.constant(v) for k, v in b.items()})
    threading.Thread(target=producer, daemon=True).start()

    os.makedirs(out, exist_ok=True); save_at = {int(x) for x in a.save_at.split(",") if x}
    log = open(f"{out}/pretrain_log.jsonl", "w"); t0 = time.time(); acc = []; EVERY = 500; wait = 0.0
    for s in range(1, a.steps + 1):
        tw = time.time(); b = q.get(); wait += time.time() - tw
        acc.append(step(b["ctx"], b.get("seg", zs), b.get("mctx", zm), b.get("vis", zv)))
        if s % EVERY == 0:
            m = np.mean([x.numpy() for x in acc], 0); acc = []; d = diag(b["ctx"]).numpy()
            r = dict(step=s, sec=round(time.time() - t0), loss=m[0], loss_fut=m[1], loss_mask=m[2], var=m[3], cov=m[4],
                     std_mean=d[0], std_min=d[1], eff_rank=d[2], wait_s=wait); wait = 0.0
            r = {k: (round(float(v), 5) if k not in ("step", "sec") else v) for k, v in r.items()}
            log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        if s in save_at and s != a.steps:
            os.makedirs(f"{out}/step{s}", exist_ok=True); enc.save_weights(f"{out}/step{s}/enc.h5")
            json.dump({"variant": a.variant, "std": std.state(), "steps": s, "of_total_steps": a.steps, "seed": a.seed, "span": SPAN, "bs": a.bs,
                       "sec": round(time.time() - t0), "note": "intermediate checkpoint: LR not annealed"}, open(f"{out}/step{s}/meta.json", "w"))
    stop.set()
    enc.save_weights(out + "/enc.h5")
    json.dump({"variant": a.variant, "std": std.state(), "steps": a.steps, "params": n_par, "trained_params": n_all, "span": SPAN,
               "device": dev, "sec": round(time.time() - t0), "bs": a.bs, "seed": a.seed}, open(out + "/meta.json", "w"))
    print("saved", a.steps, "steps", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()

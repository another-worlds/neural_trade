"""JEPA v4 pretraining (v2 variants b/c, generalised): span end, context length, encoder size, mask ratio are arguments.
usage: python pretrain4.py --name e1 --variant c --span-end "2021-12-31 23:59" --steps 60000
       [--ctx 60 --d 32 --layers 2 --heads 4 --ffn 128 --mask-ratio 0.5]
Same losses as v2 (huber on future-segment and masked-patch target embeddings, VICReg 5/1 on the pooled embedding, EMA 0.996,
Adam 1e-3 cosine). GPU: memory capped (--mem MB), below-normal priority, tf.random.set_seed before building.
A partial checkpoint is saved every 5000 steps."""
import argparse, ctypes, json, os, queue, sys, threading, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "v2"))
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
SPAN_START = "2017-01-01"


def load_span(end):
    d = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    d = d[(d.timestamp >= SPAN_START) & (d.timestamp <= end)]
    return d.timestamp.values, d[["open", "high", "low", "close", "volume"]].values.astype(np.float64)


def valid_anchors(bars, ctx, horizon):
    n = len(bars); a = np.arange(ctx, n - horizon); c = bars[:, 3]
    return a[np.abs(np.log(c[a - 1] / c[a - ctx])) > 0]


def make_batch(bars, idx, std, ctx, mask_pool, rng, with_mask=True):
    from channels4 import context_window, future_window, context_channels, context_channels_masked, segment_channels, SEG
    W = context_window(bars, idx, ctx)
    full, vol, vm = context_channels(W)
    out = {"ctx": std(full), "seg": std(segment_channels(future_window(bars, idx), bars[idx - 1, 3], vol, vm, SEG))}
    if with_mask:
        hid = mask_pool[rng.integers(0, len(mask_pool), len(idx))]
        out["mctx"] = std(context_channels_masked(W, hid)); out["vis"] = ~hid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--variant", default="c", choices=["b", "c"])
    ap.add_argument("--span-end", default="2019-12-31 23:59"); ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--mem", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=60); ap.add_argument("--d", type=int, default=32); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ffn", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=None)
    a = ap.parse_args(); fut = a.variant == "c"; n_patch = a.ctx // 5; out = f"{HERE}/ckpt/{a.name}"
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception: pass
    import neural_trade  # noqa: F401
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU"); dev = "cpu"
    if gpus and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=a.mem)]); dev = "gpu"
    tf.random.set_seed(a.seed)
    from channels4 import N_CH, Standardiser, context_channels, block_masks_ratio, SEG, N_SEG
    from channels2 import block_masks
    import model4 as M
    ts, bars = load_span(a.span_end); H = SEG * N_SEG; anchors = valid_anchors(bars, a.ctx, H)
    rng = np.random.default_rng(a.seed)
    print(f"{a.name} device {dev}: {len(bars)} bars, {len(anchors)} anchors, ctx {a.ctx}", flush=True)
    std = Standardiser.fit(context_channels(bars[rng.choice(anchors, 20000, replace=False)[:, None] + np.arange(-a.ctx, 0)])[0])
    mrng = np.random.default_rng(1)
    mask_pool = (block_masks(mrng, 8192, 4, 6, n_patch) if a.mask_ratio is None else block_masks_ratio(mrng, 8192, a.mask_ratio, n_patch))
    print(f"mean hidden fraction {mask_pool.mean():.3f}", flush=True)
    arch = dict(d=a.d, heads=a.heads, ffn=a.ffn, layers=a.layers, max_tok=n_patch)
    enc, tgt = M.Encoder4(N_CH, **arch), M.Encoder4(N_CH, **arch)
    x0 = tf.zeros((2, a.ctx, N_CH)); enc(x0); tgt(x0); tgt(tf.zeros((2, SEG, N_CH)))
    tgt.set_weights(enc.get_weights())
    mp = M.MaskPredictor4(a.d, a.heads, a.ffn, n_patch); sg = M.SegPredictor4(N_SEG, a.d) if fut else None
    mp(tf.zeros((2, n_patch, a.d)), tf.ones((2, n_patch), tf.bool))
    if sg: sg(tf.zeros((2, a.d)))
    train_vars = enc.trainable_variables + mp.trainable_variables + (sg.trainable_variables if sg else [])
    n_par = int(sum(np.prod(w.shape) for w in enc.trainable_weights))
    print(f"encoder params {n_par}", flush=True)
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, a.steps, alpha=0.05)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=5.0); MOM = 0.996
    huber = lambda t, p: tf.keras.losses.huber(tf.stop_gradient(t), p, delta=1.0)

    @tf.function
    def step(ctx, seg, mctx, vis):
        zero = tf.constant(0.0)
        with tf.GradientTape() as tape:
            z = enc(ctx, training=True); lf = zero
            if fut:
                t = tgt(tf.reshape(seg, (-1, SEG, N_CH))); t = tf.reshape(t, (-1, N_SEG, a.d))
                lf = tf.reduce_mean(huber(t, sg(z)))
            tt = tgt.tokens_out(ctx)
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

    q = queue.Queue(maxsize=16); stop = threading.Event()

    def producer():
        r = np.random.default_rng(a.seed + 7)
        while not stop.is_set():
            b = make_batch(bars, r.choice(anchors, a.bs), std, a.ctx, mask_pool, r)
            q.put({k: tf.constant(v) for k, v in b.items()})
    threading.Thread(target=producer, daemon=True).start()

    os.makedirs(out, exist_ok=True)
    meta = lambda s, sec: {"name": a.name, "variant": a.variant, "std": std.state(), "steps": s, "planned_steps": a.steps, "params": n_par,
                           "span": [SPAN_START, a.span_end], "device": dev, "sec": sec, "bs": a.bs, "seed": a.seed, "ctx": a.ctx,
                           "arch": arch, "mask_ratio": a.mask_ratio}
    log = open(f"{out}/pretrain_log.jsonl", "w"); t0 = time.time(); acc = []; EVERY = 500
    for s in range(1, a.steps + 1):
        b = q.get()
        acc.append(step(b["ctx"], b["seg"], b["mctx"], b["vis"]))
        if s % EVERY == 0:
            m = np.mean([x.numpy() for x in acc], 0); acc = []; d = diag(b["ctx"]).numpy()
            r = dict(step=s, sec=round(time.time() - t0), loss=m[0], loss_fut=m[1], loss_mask=m[2], var=m[3], cov=m[4],
                     std_mean=d[0], std_min=d[1], eff_rank=d[2])
            r = {k: (round(float(v), 5) if k not in ("step", "sec") else v) for k, v in r.items()}
            log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
        if s % 5000 == 0 and s < a.steps:
            enc.save_weights(out + "/enc_partial.h5"); json.dump(meta(s, round(time.time() - t0)), open(out + "/meta_partial.json", "w"))
    stop.set()
    enc.save_weights(out + "/enc.h5")
    json.dump(meta(a.steps, round(time.time() - t0)), open(out + "/meta.json", "w"))
    print("saved", a.steps, "steps", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()

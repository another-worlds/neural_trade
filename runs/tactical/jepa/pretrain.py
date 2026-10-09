"""JEPA pretraining on 2017-01-01..2019-12-31 only, no labels. usage: python pretrain.py [--minutes 20] [--bs 256]
Context 60 bars -> f_theta; next 15 bars -> f_xi (EMA of f_theta); predictor g; loss smooth-L1(g(f_theta(ctx)), sg(f_xi(fut)))
+ VICReg variance and covariance on the context embeddings. Window sampling is on the fly (no stored windows)."""
import argparse, ctypes, json, os, time, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
SPAN = ("2017-01-01", "2019-12-31 23:59")


def load_span():
    d = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    d = d[(d.timestamp >= SPAN[0]) & (d.timestamp <= SPAN[1])]
    return d.timestamp.values, d[["open", "high", "low", "close", "volume"]].values.astype(np.float64)


def sample_batch(bars, anchors, std=None):
    """anchors = index of the FIRST bar after the context. Returns standardised ctx [B,60,C], fut [B,15,C]."""
    from channels import CTX, FUT, context_channels, future_channels
    ci = anchors[:, None] + np.arange(-CTX, 0)
    fi = anchors[:, None] + np.arange(FUT)
    ctx_ch, vol, vm = context_channels(bars[ci])
    fut_ch = future_channels(bars[fi], bars[anchors - 1, 3], vol, vm)
    return (ctx_ch, fut_ch) if std is None else (std(ctx_ch), std(fut_ch))


def valid_anchors(bars, lo=60, stride=1):
    """anchors whose context has price movement (drops the dead 2017 flat stretches) and whose future fits."""
    from channels import CTX, FUT
    n = len(bars); a = np.arange(CTX, n - FUT, stride)
    c = bars[:, 3]
    return a[np.abs(np.log(c[a - 1] / c[a - CTX])) > 0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=20); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=10**9); ap.add_argument("--out", default=HERE + "/ckpt")
    a = ap.parse_args()
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except Exception: pass
    import neural_trade  # noqa: F401 (puts the CUDA DLLs on PATH)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3000)])
    tf.random.set_seed(0)
    from channels import N_CH, Standardiser, CTX, FUT
    import model as M
    ts, bars = load_span(); anchors = valid_anchors(bars)
    print(f"span {ts[0]}..{ts[-1]}: {len(bars)} bars, {len(anchors)} valid anchors", flush=True)
    rng = np.random.default_rng(0)
    c0, _ = sample_batch(bars, rng.choice(anchors, 20000, replace=False))
    std = Standardiser.fit(c0)
    enc, tgt = M.Encoder(N_CH), M.Encoder(N_CH); pred = M.make_predictor()
    enc(tf.zeros((2, CTX, N_CH))); tgt(tf.zeros((2, FUT, N_CH))); pred(tf.zeros((2, M.D)))
    tgt.set_weights(enc.get_weights())
    n_par = int(sum(np.prod(w.shape) for w in enc.trainable_weights)); n_pred = int(sum(np.prod(w.shape) for w in pred.trainable_weights))
    print(f"encoder params {n_par}, predictor params {n_pred}", flush=True)
    steps_est = 22000
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, steps_est, alpha=0.05)
    opt = tf.keras.optimizers.Adam(lr, clipnorm=5.0)
    MOM = 0.996

    @tf.function
    def step(ctx, fut):
        t = tgt(fut)
        with tf.GradientTape() as tape:
            z = enc(ctx, training=True); p = pred(z)
            sl1 = tf.reduce_mean(tf.keras.losses.huber(tf.stop_gradient(t), p, delta=1.0)) 
            var, cov = M.vicreg(z)
            loss = sl1 + 5.0 * var + 1.0 * cov
        vars_ = enc.trainable_variables + pred.trainable_variables
        opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
        M.ema_update(tgt, enc, MOM)
        zc = z - tf.reduce_mean(z, 0)
        return loss, sl1, var, cov, tf.reduce_mean(tf.math.reduce_std(z, 0)), tf.reduce_min(tf.math.reduce_std(z, 0)), M.effective_rank(z), tf.reduce_mean(tf.math.reduce_std(t, 0))

    log = open(HERE + "/pretrain_log.jsonl", "w"); t0 = time.time(); s = 0; acc = []; EVERY = 500
    while time.time() - t0 < a.minutes * 60 and s < a.max_steps:
        idx = rng.choice(anchors, a.bs)
        ctx, fut = sample_batch(bars, idx, std)
        acc.append([float(x) for x in step(tf.constant(ctx), tf.constant(fut))]); s += 1
        if s % EVERY == 0:
            m = np.mean(acc, 0); acc = []
            r = dict(step=s, sec=round(time.time() - t0), loss=m[0], smooth_l1=m[1], var=m[2], cov=m[3], std_mean=m[4], std_min=m[5],
                     eff_rank=m[6], target_std=m[7])
            r = {k: (round(float(v), 5) if k not in ("step", "sec") else v) for k, v in r.items()}
            log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
    os.makedirs(a.out, exist_ok=True)
    enc.save_weights(a.out + "/enc.h5"); json.dump({"std": std.state(), "steps": s, "params": n_par, "span": SPAN}, open(a.out + "/meta.json", "w"))
    print("saved", s, "steps", flush=True)


if __name__ == "__main__":
    main()

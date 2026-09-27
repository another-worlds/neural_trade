"""CPU profile of today's gru_attention train step (read-only on the repo).

Measures on CPU (CUDA_VISIBLE_DEVICES=-1): the real CustomTrainModel.train_step at batch 64 and 256,
forward+backward of each architectural block at batch 256, and the op count of the traced train step
(a proxy for GPU kernel launches). Prints analytical forward FLOPs per block.
"""
import os
import sys
import time
import json

SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = "C:/Users/Step/Documents/neural_trade"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import neural_trade  # noqa: F401  (before tensorflow)
import tensorflow as tf
from tensorflow.keras import layers

from neural_trade.core.config import Config
from neural_trade.data.processor import DataProcessor
from neural_trade.registries.models import Models
from neural_trade.training.custom_model import CustomTrainModel
from neural_trade.training.optim import build_optimizers
from neural_trade.registries.layers import Layers

tf.keras.utils.set_random_seed(0)
cfg = Config()
cfg.CSV_PATH = os.path.join(REPO, "binance_btcusdt_1min_ccxt.csv")
cfg.SCALER_PATH = os.path.join(SCRATCH, "scaler_tmp.joblib")
cfg.MODEL_PATH = os.path.join(SCRATCH, "weights_tmp.h5")
cfg.validate()

out = {"cpu_threads_intra": tf.config.threading.get_intra_op_parallelism_threads()}

# ---------------------------------------------------------------- data sizes (the real split)
dp = DataProcessor(cfg)
df, close = dp.load_and_prepare_data()
(X_tr, y_tr_s, lc_tr, ext_tr, X_te, y_te_s, lc_te, ext_te, y_tr, y_te, scaler) = dp.prepare_datasets(df, close)
fold = dp.fold
sizes = {k: int(len(getattr(fold, k))) for k in ("train", "val", "cal", "test")}
sizes["gap"] = int(fold.gap)
sizes["n_bars"] = int(len(close))
sizes["steps_per_epoch_train"] = int(np.ceil(sizes["train"] / cfg.BATCH_SIZE))
sizes["val_batches"] = int(np.ceil(sizes["val"] / cfg.BATCH_SIZE))
out["sizes"] = sizes
print("SIZES", json.dumps(sizes), flush=True)

# ---------------------------------------------------------------- the real train step
base = Models.build(getattr(cfg, "MODEL_NAME", None), cfg)
out["params_total"] = int(base.count_params())
pair = build_optimizers(cfg)
std = float(np.std(y_tr))
model = CustomTrainModel(
    base_model=base, pred_scale=std if std > 0 else 1.0, pred_mean=float(np.mean(y_tr)),
    lambda_point=cfg.LAMBDA_POINT, lambda_local_trend=cfg.LAMBDA_LOCAL_TREND,
    lambda_global_trend=cfg.LAMBDA_GLOBAL_TREND, lambda_extended_trend=cfg.LAMBDA_EXTENDED_TREND,
    lambda_dir=cfg.LAMBDA_DIR, config=cfg, indicator_optimizer=pair.indicator,
    inputs=base.inputs, outputs=base.outputs,
)
model.compile(optimizer=pair.main)


def batch(n, off=0):
    return (tf.constant(X_tr[off:off + n]), tf.constant(y_tr_s[off:off + n].astype("float32")),
            tf.constant(lc_tr[off:off + n].reshape(-1, 1)), tf.constant(ext_tr[off:off + n]))


def time_fn(fn, args, reps=5, warm=2):
    for _ in range(warm):
        r = fn(*args)
    _ = [np.asarray(t) for t in tf.nest.flatten(r)] if r is not None else None
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        _ = [np.asarray(t) for t in tf.nest.flatten(r)] if r is not None else None
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


step_fn = tf.function(lambda d: model.train_step(d))
res = {}
for B in (64, 256):
    d = batch(B)
    med, lo, hi = time_fn(lambda dd: step_fn(dd), (d,), reps=7, warm=3)
    res[B] = (med, lo, hi)
    print(f"CPU train_step B={B}: median {med*1000:.1f} ms (min {lo*1000:.1f}, max {hi*1000:.1f})", flush=True)
out["cpu_train_step_ms"] = {str(k): [round(v * 1000, 1) for v in vals] for k, vals in res.items()}

# op count of the traced train step (proxy for GPU kernel launches)
cf = step_fn.get_concrete_function(batch(256))
gd = cf.graph.as_graph_def()
host_like = {"Const", "NoOp", "Identity", "ReadVariableOp", "Placeholder", "Shape", "StridedSlice",
             "Pack", "Unpack", "Reshape", "ExpandDims", "Squeeze", "Cast", "Fill", "Range",
             "ZerosLike", "OnesLike", "BroadcastArgs", "BroadcastGradientArgs", "ConcatV2", "Size",
             "Rank", "StopGradient", "IdentityN", "_Arg", "_Retval", "AssignVariableOp", "VarHandleOp",
             "ShapeN", "DynamicStitch", "Transpose", "Tile"}
top = [n.op for n in gd.node]
lib = [n.op for f in gd.library.function for n in f.node_def]
from collections import Counter
ct = Counter(top)
cl = Counter(lib)
compute_top = sum(v for k, v in ct.items() if k not in host_like)
compute_lib = sum(v for k, v in cl.items() if k not in host_like)
out["train_step_graph_ops"] = {"top_level_total": len(top), "top_level_compute_like": compute_top,
                               "library_functions": len(gd.library.function),
                               "library_total": len(lib), "library_compute_like": compute_lib,
                               "top_level_most_common": ct.most_common(15),
                               "library_fn_names": [f.signature.name[:60] for f in gd.library.function][:30]}
print("GRAPH OPS", json.dumps(out["train_step_graph_ops"])[:1500], flush=True)

# ---------------------------------------------------------------- per-block forward+backward at B=256
L = cfg.LOOKBACK
B = 256
x_win = tf.constant(X_tr[:B])
num_logits = len(cfg.MA_SPANS) + 3 * len(cfg.MACD_SETTINGS) + len(cfg.RSI_PERIODS) + len(cfg.BB_PERIODS)


def block_indicators():
    inp = layers.Input(shape=(L,))
    r = layers.Reshape((L, 1))(inp)
    meta = layers.Concatenate()([layers.GlobalAveragePooling1D()(r), layers.GlobalMaxPooling1D()(r)])
    meta = layers.Dense(num_logits, activation="tanh")(meta)
    ind = Layers.for_role(cfg, "indicators", cfg, name="li")([inp, meta])
    return tf.keras.Model(inp, ind), x_win


def block_bigru():
    inp = layers.Input(shape=(L, 31))
    m = layers.Bidirectional(layers.GRU(64, return_sequences=True))(inp)
    return tf.keras.Model(inp, m), tf.random.normal((B, L, 31))


def block_temporal_att():
    inp = layers.Input(shape=(L, 128))
    a = layers.MultiHeadAttention(num_heads=8, key_dim=32)(inp, inp)
    x = layers.LayerNormalization()(layers.Add()([inp, a]))
    return tf.keras.Model(inp, x), tf.random.normal((B, L, 128))


def block_cross_att():
    inp = layers.Input(shape=(L, 128))
    xp = layers.Permute((2, 1))(inp)
    a = layers.MultiHeadAttention(num_heads=4, key_dim=32)(xp, xp)
    x = layers.Permute((2, 1))(layers.LayerNormalization()(layers.Add()([xp, a])))
    return tf.keras.Model(inp, x), tf.random.normal((B, L, 128))


def block_conv_gate():
    inp = layers.Input(shape=(L, 128))
    w = layers.Input(shape=(L,))
    xs = layers.Conv1D(16, 3, padding="same", activation="gelu")(inp)
    xm = layers.Conv1D(16, 7, padding="same", activation="gelu")(inp)
    xl = layers.Conv1D(16, 15, padding="same", activation="gelu")(inp)
    g = Layers.for_role(cfg, "energy_gate", n_branches=3, name="eg")([w, xs, xm, xl])
    g = layers.LayerNormalization()(g)
    return tf.keras.Model([inp, w], g), [tf.random.normal((B, L, 128)), x_win]


def block_transformers():
    inp = layers.Input(shape=(L, 16))
    x = layers.Add()([inp, Layers.for_role(cfg, "positional_encoding")(inp)])
    for _ in range(2):
        a = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = layers.LayerNormalization()(layers.Add()([x, a]))
        ff = layers.Dense(16)(layers.Dropout(0.1)(layers.Dense(32, activation="gelu")(x)))
        x = layers.LayerNormalization()(layers.Add()([x, ff]))
    return tf.keras.Model(inp, x), tf.random.normal((B, L, 16))


def block_full_model():
    return base, x_win


blocks = {"indicators(LearnableIndicators+meta)": block_indicators, "BiGRU(64)": block_bigru,
          "temporal MHA(8x32)+LN": block_temporal_att, "cross-indicator MHA(4x32)+LN": block_cross_att,
          "3xConv1D+EnergyGate+LN": block_conv_gate, "PE+2 transformer blocks": block_transformers,
          "FULL base model (sum-of-outputs loss)": block_full_model}
block_times = {}
for name, mk in blocks.items():
    m, xin = mk()

    @tf.function
    def fb(xx, m=m):
        with tf.GradientTape() as tape:
            if isinstance(xx, (list, tuple)):
                for t in xx:
                    tape.watch(t)
            else:
                tape.watch(xx)
            y = m(xx, training=True)
            loss = tf.add_n([tf.reduce_sum(tf.cast(t, tf.float32)) for t in tf.nest.flatten(y)])
        g = tape.gradient(loss, m.trainable_variables)
        return loss, [gg for gg in g if gg is not None]

    med, lo, hi = time_fn(fb, (xin,), reps=7, warm=2)
    cff = fb.get_concrete_function(xin)
    gdd = cff.graph.as_graph_def()
    n_ops = len(gdd.node) + sum(len(f.node_def) for f in gdd.library.function)
    block_times[name] = {"cpu_fwd_bwd_ms_median": round(med * 1000, 2), "min": round(lo * 1000, 2),
                         "max": round(hi * 1000, 2), "graph_ops_incl_library": n_ops}
    print(f"CPU fwd+bwd B=256 {name:45s} median {med*1000:8.2f} ms  ops {n_ops}", flush=True)
out["cpu_block_fwd_bwd_B256"] = block_times

# ---------------------------------------------------------------- analytical forward FLOPs per sample
T, C = L, 31
f = {}
f["indicators (24 EWMA 60x60 matmuls)"] = 24 * T * T * 2
f["indicators weight build (~6 elementwise ops on [24,60,60])"] = 6 * 24 * T * T
f["BiGRU(64)"] = 2 * T * 3 * (C * 64 + 64 * 64 + 2 * 64) * 2
f["temporal MHA 8x32 on [60,128]"] = (3 * T * 128 * 256 + T * 256 * 128) * 2 + 2 * 8 * T * T * 32 * 2
f["cross MHA 4x32 on [128,60]"] = (3 * 128 * 60 * 128 + 128 * 128 * 60) * 2 + 2 * 4 * 128 * 128 * 32 * 2
f["3xConv1D(16; k=3,7,15) on 128ch"] = T * 128 * 16 * (3 + 7 + 15) * 2
f["2 transformer blocks d=16"] = 2 * ((3 * T * 16 * 64 + T * 64 * 16) * 2 + 2 * 4 * T * T * 16 * 2 + T * 16 * 32 * 2 * 2)
f["flatten->Dense(32) + towers"] = T * 16 * 32 * 2 + 3 * (48 * 16 * 2 + 3 * 18 * 2)
tot = sum(f.values())
out["analytic_fwd_flops_per_sample"] = {k: int(v) for k, v in f.items()}
out["analytic_fwd_flops_total_per_sample"] = int(tot)
for k, v in f.items():
    print(f"FLOPs/sample fwd {k:55s} {v/1e6:7.2f} M ({100*v/tot:5.1f}%)")
print(f"TOTAL fwd {tot/1e6:.1f} MFLOP/sample; fwd+bwd ~3x = {3*tot/1e6:.0f} MFLOP; x256 = {3*tot*256/1e9:.1f} GFLOP/step")
json.dump(out, open(os.path.join(SCRATCH, "profile_model.json"), "w"), indent=2, default=str)
print("saved", os.path.join(SCRATCH, "profile_model.json"))

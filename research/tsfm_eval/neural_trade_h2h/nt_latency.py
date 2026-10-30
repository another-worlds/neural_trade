"""Single-origin (batch 1) latency of neural_trade (model.py as-is) with the seed-0 repo-dataset weights.

Usage: nice -n 19 venv/bin/python nt_latency.py --threads 1 [--calls 200] [--warmup 3]

Times the full per-origin inference path the way model.py does it: input_scaler.transform on the 60-bar
window -> keras Model.predict -> target_scaler.inverse_transform of the 3 price heads.
Also times a tf.function-compiled call of the same Keras model with a fixed [1, 60] signature
("compiled_call": same outputs, no Model.predict() overhead; a lower bound for a tuned live path).
Before timing, checks that the loaded weights reproduce the saved seed-0 npz at 5 origins.
Prints one JSON line.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
SCR_WORK = _os.environ.get("TSFM_EVAL_WORK", _os.path.join(_HERE, "..", "work"))
REPO_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--calls", type=int, default=200)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--run", default="repo_seed0")
args = ap.parse_args()

T = str(args.threads)
for k in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS"]:
    os.environ[k] = T
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["MPLBACKEND"] = "Agg"
sys.dont_write_bytecode = True

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(args.threads)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)

H2H = _os.path.join(SCR_WORK, "nt_h2h")
sys.path.insert(0, REPO_ROOT)
import model as M  # noqa: E402  (unmodified)

run_dir = f"{H2H}/runs/{args.run}"
cfg = M.Config()
tsc = joblib.load(f"{run_dir}/scaler_v3.joblib")
isc = joblib.load(f"{run_dir}/scaler_v3_input.joblib")
base = M.PricePredictor(cfg).build_model()
base.load_weights(f"{run_dir}/final_base.weights.h5")

import pandas as pd  # noqa: E402

close = pd.read_csv(f"{H2H}/data/repo_full.csv", usecols=["close"])["close"].to_numpy(np.float32)
rng = np.random.default_rng(0)
ts = rng.integers(cfg.LOOKBACK, len(close), size=args.warmup + args.calls)


def one_predict(t):
    x = isc.transform(close[t - cfg.LOOKBACK + 1:t + 1][None, :]).astype(np.float32)
    y = base.predict(x, verbose=0)
    return [tsc.inverse_transform(y[3 * k][:, :1])[0, 0] for k in range(3)], y[1], y[2]


_fn = tf.function(lambda x: base(x, training=False),
                  input_signature=[tf.TensorSpec([1, cfg.LOOKBACK], tf.float32)])


def one_call(t):
    x = isc.transform(close[t - cfg.LOOKBACK + 1:t + 1][None, :]).astype(np.float32)
    y = _fn(tf.constant(x))
    return [tsc.inverse_transform(y[3 * k].numpy()[:, :1])[0, 0] for k in range(3)]


# correctness check against the saved npz (seed-0 repo run)
_z = np.load(f"{H2H}/out/repo/neural_trade-asis-seed0_ctx60_price.npz")
_chk = []
for j in [0, 1, 100, 1000, len(_z["origins"]) - 1]:
    t = int(_z["origins"][j])
    a = np.array(one_predict(t)[0])
    b = np.array(one_call(t))
    _chk.append(max(np.abs(a - _z["delta_usd"][j]).max(), np.abs(b - _z["delta_usd"][j]).max()))


def bench(fn):
    for t in ts[:args.warmup]:
        fn(t)
    out = []
    for t in ts[args.warmup:]:
        s = time.perf_counter()
        fn(t)
        out.append(1000 * (time.perf_counter() - s))
    p = np.percentile(out, [50, 95, 99])
    return {"p50_ms": round(float(p[0]), 2), "p95_ms": round(float(p[1]), 2), "p99_ms": round(float(p[2]), 2),
            "mean_ms": round(float(np.mean(out)), 2)}


res = {"model": "neural_trade-asis", "weights": args.run, "ctx": cfg.LOOKBACK, "batch": 1,
       "threads": args.threads, "calls": args.calls, "warmup": args.warmup,
       "tf": tf.__version__, "max_abs_diff_vs_npz_usd": round(float(max(_chk)), 4),
       "predict_path": bench(one_predict), "compiled_call": bench(one_call)}
print(json.dumps(res), flush=True)

"""Rolling head-to-head: retrain neural_trade (model.py, imported unmodified) every RETRAIN minutes and predict only
the origins of the next block with the newest model. Same outputs as run_nt.py, saved as
neural_trade-rolling<R>m-seed<K>_ctx60_price.npz.

Usage: nice -n 19 venv/bin/python run_nt_rolling.py --data repo|post --seed K [--retrain-min 1440]

Each block runs in its own child process (TensorFlow does not free memory between trainings in one process),
saves runs_rolling/<data>_seed<K>_r<R>/blockNN/pred.npz, and is skipped on a rerun if that file exists.

Block k covers origins t in [t0 + k*R, t0 + (k+1)*R). Its model is trained on rows [0, t0 + k*R] (inclusive), the
same convention as run_nt.py, so no training target reaches past the first origin of its block. Only IO and the seed
are redirected, exactly as in run_nt.py.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
SCR_WORK = _os.environ.get("TSFM_EVAL_WORK", _os.path.join(_HERE, "..", "work"))
REPO_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

for k in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["MPLBACKEND"] = "Agg"
sys.dont_write_bytecode = True

ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=["repo", "post"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--retrain-min", type=int, default=1440)
ap.add_argument("--block", type=int, default=-1, help="internal: train and predict one block, then exit")
args = ap.parse_args()
R = args.retrain_min

import numpy as np  # noqa: E402

SCR = SCR_WORK
H2H = f"{SCR}/nt_h2h"
REPO = REPO_ROOT
os.environ["BENCH_DATA"] = args.data
sys.path.insert(0, _os.path.join(_HERE, "..", "bench"))
import common  # noqa: E402

df_c, info = common.load()
ORG = common.origins(len(df_c))
close64 = df_c["close"].to_numpy(np.float64)
t0 = int(ORG[0])
tag = f"neural_trade-rolling{R}m-seed{args.seed}_ctx60_price"
base_dir = f"{H2H}/runs_rolling/{args.data}_seed{args.seed}_r{R}"
os.makedirs(base_dir, exist_ok=True)
nq = len(common.QLEVELS)
n_blocks = int(np.ceil((ORG[-1] + 1 - t0) / R))

if args.block < 0:
    # ---- driver: one child process per block, then merge (no TensorFlow in this process) ----
    for k in range(n_blocks):
        if os.path.exists(f"{base_dir}/block{k:02d}/pred.npz"):
            continue
        subprocess.run([sys.executable, os.path.abspath(__file__), "--data", args.data, "--seed", str(args.seed),
                        "--retrain-min", str(R), "--block", str(k)], check=True)
    q_all = np.full((len(ORG), 3, nq), np.nan)
    dir_all = np.full((len(ORG), 3), np.nan)
    delta_all = np.full((len(ORG), 3), np.nan)
    blocks, elapsed_pred = [], 0.0
    for k in range(n_blocks):
        z = np.load(f"{base_dir}/block{k:02d}/pred.npz")
        sel = z["sel"]
        q_all[sel], dir_all[sel], delta_all[sel] = z["q"], z["dir_prob"], z["delta_usd"]
        blocks.append(json.loads(str(z["info"])))
        elapsed_pred += blocks[-1]["predict_wall_s"]
    assert not np.isnan(q_all).any()
    med = q_all[:, :, 10]
    med_path = np.stack([np.interp(np.arange(1, 16), common.HORIZONS, m) for m in med])
    out_dir = f"{H2H}/out/{args.data}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{tag}.npz"
    np.savez(path, q=q_all.astype(np.float32), med_path=med_path.astype(np.float32), origins=ORG.astype(np.int64),
             framing="price", elapsed=float(elapsed_pred), n=int(len(ORG)), threads=1,
             dir_prob=dir_all.astype(np.float32), delta_usd=delta_all.astype(np.float32))
    real = np.stack([close64[ORG + h] - close64[ORG] for h in common.HORIZONS], axis=1)
    summ = {"data": args.data, "seed": args.seed, "retrain_min": R, "n": int(len(ORG)), "blocks": blocks,
            "frac_dir_gt_0.5": [round(float((dir_all[:, j] > 0.5).mean()), 4) for j in range(3)],
            "mean_abs_delta_usd": [round(float(np.abs(delta_all[:, j]).mean()), 3) for j in range(3)],
            "mean_abs_real_delta_usd": [round(float(np.abs(real[:, j]).mean()), 3) for j in range(3)],
            "corr_delta_real": [round(float(np.corrcoef(delta_all[:, j], real[:, j])[0, 1]), 4) for j in range(3)],
            "npz": path}
    json.dump(summ, open(f"{base_dir}/summary.json", "w"), indent=1)
    print("[rolling] SUMMARY " + json.dumps({k: v for k, v in summ.items() if k != "blocks"}))
    sys.exit(0)

# ---- child: train and predict one block ----
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
sys.path.insert(0, REPO)
import model as M  # noqa: E402  (unmodified <repo>/model.py)

assert os.path.abspath(M.__file__) == f"{REPO}/model.py"
full_csv = f"{H2H}/data/{args.data}_full.csv"
raw = pd.read_csv(full_csv, dtype=str)
assert len(raw) == len(close64)
cfg_full = M.Config()
cfg_full.CSV_PATH = full_csv
_, close32 = M.DataProcessor(cfg_full).load_and_prepare_data()
assert np.array_equal(close32, close64.astype(np.float32))
_orig_set_seed = tf.keras.utils.set_random_seed


def run_block(k):
    """Train on rows [0, t0 + k*R] and predict the origins of block k; write blockNN/pred.npz."""
    tk = t0 + k * R
    sel = np.where((ORG >= tk) & (ORG < tk + R))[0]
    run_dir = f"{base_dir}/block{k:02d}"
    shutil.rmtree(run_dir, ignore_errors=True)       # model.py would reuse stale weights otherwise
    os.makedirs(run_dir)
    os.chdir(run_dir)
    train_csv = f"{H2H}/data/{args.data}_train_upto_{tk}.csv"
    if not os.path.exists(train_csv):
        raw.iloc[: tk + 1].to_csv(train_csv, index=False)
    seed_calls = []

    def _set_seed_override(s):
        seed_calls.append(s)
        _orig_set_seed(args.seed)

    tf.keras.utils.set_random_seed = _set_seed_override
    _orig_set_seed(args.seed)
    cfg = M.Config()
    overrides = {"CSV_PATH": train_csv, "MODEL_PATH": f"{run_dir}/nn_learnable_indicators_v3.weights.h5",
                 "SCALER_PATH": f"{run_dir}/scaler_v3.joblib"}
    ts = time.time()
    res = M.train_and_evaluate(config=cfg, config_overrides=overrides, force=False, calibrate=True)
    train_s = time.time() - ts
    tf.keras.utils.set_random_seed = _orig_set_seed
    assert seed_calls == [42], seed_calls
    model, tsc, isc = res.model, res.target_scaler, res.input_scaler
    L = cfg.LOOKBACK
    org = ORG[sel]
    idx = org[:, None] + np.arange(-L + 1, 1)[None, :]
    tp = time.time()
    X = isc.transform(close32[idx]).reshape(len(org), L).astype(np.float32)
    ya = model.predict(tf.data.Dataset.from_tensor_slices(X).batch(cfg.BATCH_SIZE), verbose=0)
    pred_s = time.time() - tp
    delta_usd = np.stack([tsc.inverse_transform(ya[3 * j][:, :1]).ravel() for j in range(3)], axis=1)
    dir_prob = np.stack([np.asarray(ya[3 * j + 1]).reshape(-1) for j in range(3)], axis=1)
    var_raw = np.stack([np.asarray(ya[3 * j + 2]).reshape(-1) for j in range(3)], axis=1)
    scale_usd = np.sqrt(np.clip(var_raw, cfg.VAR_FLOOR, cfg.VAR_CAP)) * float(tsc.scale_[0])
    z = student_t.ppf(np.array(common.QLEVELS), df=cfg.DF_VAR)
    q = close64[org][:, None, None] + delta_usd[:, :, None] + z[None, None, :] * scale_usd[:, :, None]
    zin = (close32[org] - isc.mean_[-1]) / isc.scale_[-1]
    info_k = {"block": k, "train_upto": int(tk), "n_origins": int(len(sel)), "train_wall_s": round(train_s, 1),
              "predict_wall_s": round(pred_s, 3),
              "epochs": len(res.history.history["loss"]) if res.history is not None else None,
              "lambdas": {a: float(getattr(model, a)) for a in ["lambda_point", "lambda_dir", "lambda_var"]},
              "frac_dir_gt_0.5": [round(float((dir_prob[:, j] > 0.5).mean()), 4) for j in range(3)],
              "frac_input_absz_gt3": round(float((np.abs(zin) > 3).mean()), 4)}
    np.savez(f"{run_dir}/pred.npz", sel=sel, q=q, dir_prob=dir_prob, delta_usd=delta_usd, info=json.dumps(info_k))
    print(f"[rolling] {args.data} seed {args.seed} block {k + 1}/{n_blocks}: {len(sel)} origins, train {train_s:.0f}s",
          flush=True)


run_block(args.block)
